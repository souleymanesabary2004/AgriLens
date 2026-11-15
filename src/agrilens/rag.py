"""
rag.py - E4 : recherche documentaire sur le corpus de fiches (retrieval) avec garde-fou et citations.

Principe (pas de generation de texte libre : on renvoie des passages SOURCES, jamais du texte invente) :
  1. Les fiches (07_rag_genai/corpus/*.md) sont decoupees en morceaux (chunks) : un morceau = une section.
  2. Chaque morceau et chaque question sont transformes en vecteurs TF-IDF (mots ponderes par leur rarete).
  3. On classe les morceaux par similarite cosinus avec la question et on garde les top_k.
  4. Garde-fou : si le meilleur score est sous un seuil, on REFUSE ("pas d'information sourcee").
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

REFUSAL = ("Je n'ai pas d'information sourcee sur ce point dans mon corpus de fiches. "
           "Pour une reponse fiable, consulte un conseiller agricole ou le service agricole local.")
DOSE_REFUSAL = ("Je ne donne ni produit ni dosage : cela depend du pays, de la reglementation et de la parcelle. "
                "Contacte le service agricole local ou un conseiller.")
HEALTHY_MESSAGE = ("Aucun symptome detecte par le modele. Cela ne garantit pas que la plante est saine : "
                   "en cas de doute, consulte un conseiller agricole. (Message fixe, sans source.)")
DISCLAIMER = ("Prototype pedagogique base sur des fiches reformulees de services d'extension agricole "
              "(surtout americains) ; ne remplace pas l'avis d'un agronome.")

_STOP = set("""a au aux avec ce ces cet cette comment dans de des du elle en est et il ils je la le les leur lui ma mais me
mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle quels quelles qui sa se ses son sur ta te tes toi ton
tu un une vos votre vous y c d j l m n s t qu est sont peut peux faire fait ont etre avoir si plus tres aussi donc alors
sans sous ainsi apres avant entre chez vers dont ou
donne donner donnes explique expliquer dis dire parle parler montre montrer peut-on peuton veux voudrais svp merci""".split())
# Cultures hors perimetre (le corpus ne couvre que tomate, mais, cacaoyer) : un score TF-IDF ne sait pas faire cette
# distinction (le mot "mildiou" existe aussi pour la vigne), donc garde-fou explicite. Liste non exhaustive : limite assumee.
_OUT_CROPS = re.compile(r"\b(vigne|raisin|pommes? de terre|patates?|fraises?|bles?|riz|soja|poivrons?|concombres?|"
                        r"courgettes?|pommiers?|poiriers?|cafe|cafeiers?|bananiers?|manioc|orge|colza|tournesol|aubergines?)\b")
OUT_OF_SCOPE_CROP = ("Mon corpus ne couvre que la tomate, le mais et le cacaoyer : je n'ai pas d'information sourcee "
                     "sur cette culture. Consulte un conseiller agricole.")
_DOSE = re.compile(r"\b(dose|doses|dosage|posologie|combien de (?:produit|litres?|grammes?|ml)|quelle quantite)\b")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _stem(w: str) -> str:
    # lemmatisation tres simple : pluriels francais (taches -> tache, feuilles -> feuille)
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    if len(w) > 4 and w.endswith("x"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    text = _strip_accents(text.lower())
    toks = re.findall(r"[a-z0-9]{2,}", text)
    return [_stem(t) for t in toks if t not in _STOP]


@dataclass
class Chunk:
    doc_id: str
    titre: str
    section: str
    texte: str
    sources: list[tuple[str, str]]


def _parse_fiche(path: Path) -> tuple[dict, list[Chunk]]:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    parts = re.split(r"^## ", body, flags=re.M)[1:]
    sections = {}
    for p in parts:
        head, _, content = p.partition("\n")
        sections[head.strip()] = content.strip()
    sources = []
    for key, content in sections.items():
        if key.lower().startswith("sources"):
            for line in content.splitlines():
                mm = re.match(r"- (.*?) : (https?://\S+)", line.strip())
                if mm:
                    sources.append((mm.group(1), mm.group(2)))
    chunks = []
    for key, content in sections.items():
        if key.lower().startswith("sources") or not content:
            continue
        chunks.append(Chunk(meta.get("id", path.stem), meta.get("titre", path.stem), key, content, sources))
    return meta, chunks


class Retriever:
    def __init__(self, corpus_dir: str | Path, threshold: float = 0.15):
        self.threshold = threshold
        self.chunks: list[Chunk] = []
        self.docs: dict[str, dict] = {}          # doc_id -> metadonnees + sections + sources (pour /fiche)
        for p in sorted(Path(corpus_dir).glob("*.md")):
            meta, chunks = _parse_fiche(p)
            self.chunks.extend(chunks)
            if chunks:
                self.docs[chunks[0].doc_id] = {
                    "titre": chunks[0].titre, "classe_tokens": _label_tokens(meta.get("classe_modele", "")),
                    "sections": {c.section: c.texte for c in chunks}, "sources": chunks[0].sources}
        if not self.chunks:
            raise FileNotFoundError(f"Aucune fiche .md dans {corpus_dir}")
        # le titre de la fiche est ajoute au texte indexe : une question qui nomme la maladie retrouve ses morceaux
        docs = [f"{c.titre} {c.titre} {c.section} {c.texte}" for c in self.chunks]
        self.vec = TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.matrix = self.vec.fit_transform(docs)

    def scores(self, question: str):
        q = self.vec.transform([question])
        return linear_kernel(q, self.matrix).ravel()

    def search(self, question: str, top_k: int = 3) -> dict:
        question = (question or "").strip()
        if not question:
            return {"answered": False, "reason": "question_vide", "message": REFUSAL, "results": [], "best_score": 0.0}
        if _DOSE.search(_strip_accents(question.lower())):
            return {"answered": False, "reason": "dosage_ou_produit", "message": DOSE_REFUSAL, "results": [],
                    "best_score": 0.0}
        if _OUT_CROPS.search(_strip_accents(question.lower())):
            return {"answered": False, "reason": "culture_hors_perimetre", "message": OUT_OF_SCOPE_CROP,
                    "results": [], "best_score": 0.0}
        s = self.scores(question)
        order = s.argsort()[::-1][:top_k]
        best = float(s[order[0]])
        results = [{"doc_id": self.chunks[i].doc_id, "titre": self.chunks[i].titre, "section": self.chunks[i].section,
                    "score": round(float(s[i]), 4), "texte": self.chunks[i].texte,
                    "sources": [{"titre": t, "url": u} for t, u in self.chunks[i].sources]} for i in order]
        if best < self.threshold:
            return {"answered": False, "reason": "score_sous_le_seuil", "message": REFUSAL,
                    "results": [], "best_score": round(best, 4)}
        return {"answered": True, "reason": "ok", "message": DISCLAIMER, "results": results,
                "best_score": round(best, 4)}


def _label_tokens(label: str) -> set[str]:
    """Mots d'un nom de classe : 'Corn_(maize)___Common_rust_' -> {corn, maize, common, rust}."""
    return set(re.findall(r"[a-z0-9]+", _strip_accents(label.lower())))


def _fiche_for_label(self, label: str) -> dict:
    """Relie une classe predite par le modele a sa fiche (comparaison de mots, mesure de Jaccard).

    - classe 'healthy' : message fixe, sans source ;
    - sinon : la fiche dont les mots recouvrent le mieux ceux de la classe (seuil 0,5, ecart net avec la 2e) ;
    - sinon : aucune fiche (on ne devine pas).
    """
    lt = _label_tokens(label)
    if "healthy" in lt or "sain" in lt or "saine" in lt:
        return {"kind": "healthy", "doc_id": None, "titre": None, "sections": {}, "sources": [],
                "message": HEALTHY_MESSAGE}
    scored = []
    for doc_id, d in self.docs.items():
        ft = d["classe_tokens"]
        if ft:
            scored.append((len(ft & lt) / len(ft | lt), doc_id))
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.5 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.1):
        d = self.docs[scored[0][1]]
        return {"kind": "fiche", "doc_id": scored[0][1], "titre": d["titre"], "sections": d["sections"],
                "sources": [{"titre": t, "url": u} for t, u in d["sources"]], "message": DISCLAIMER}
    return {"kind": "none", "doc_id": None, "titre": None, "sections": {}, "sources": [],
            "message": "Pas de fiche associee a cette classe dans le corpus."}


Retriever.fiche_for_label = _fiche_for_label


def load_threshold(path: str | Path, default: float = 0.15) -> float:
    p = Path(path)
    return float(json.loads(p.read_text())["threshold"]) if p.exists() else default
