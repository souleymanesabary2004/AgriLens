"""Genere un document de relecture du jeu de validation v2 (lecture seule).

Pour chaque fiche, le document affiche le texte des sections attendues puis les
questions rattachees, afin de verifier a la main que la reponse existe dans la
fiche. Pour les questions hors perimetre, il signale les mots-cles qui
apparaissent dans le corpus. Le fichier JSON n'est jamais modifie.

Usage : python scripts/review_validation_v2.py
Sortie : 07_rag_genai/eval/revue_validation_v2.md (a ne pas committer)
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "07_rag_genai" / "corpus"
VALIDATION = ROOT / "07_rag_genai" / "eval" / "validation_v2.json"
SORTIE = ROOT / "07_rag_genai" / "eval" / "revue_validation_v2.md"

MOTS_CLES_HORS_PERIMETRE = (
    "vigne",
    "pomme de terre",
    "pommes de terre",
    "caféier",
    "café",
    "blé",
    "engrais",
    "npk",
    "semer",
    "semis",
    "fraise",
    "pulvérisateur",
    "prix",
    "marché",
)


def _normaliser(texte: str) -> str:
    """Normalise la typographie pour comparer des titres de section."""
    texte = unicodedata.normalize("NFC", texte)
    texte = texte.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", texte).strip().lower()


def _lire_sections(chemin: Path) -> dict[str, tuple[str, str]]:
    """Retourne {titre normalise: (titre d'origine, contenu)} sans la section Sources."""
    texte = chemin.read_text(encoding="utf-8").replace("\r\n", "\n")
    morceaux = re.split(r"^## ", texte, flags=re.MULTILINE)[1:]
    sections: dict[str, tuple[str, str]] = {}
    for morceau in morceaux:
        titre, _, contenu = morceau.partition("\n")
        if _normaliser(titre) == "sources":
            continue
        sections[_normaliser(titre)] = (titre.strip(), contenu.strip())
    return sections


def _mentions(corpus: dict[str, str], mot: str) -> list[str]:
    """Liste les fiches dont le texte contient le mot (insensible a la casse)."""
    cible = mot.lower()
    return [fiche for fiche, texte in corpus.items() if cible in texte.lower()]


def main() -> None:
    jeu = json.loads(VALIDATION.read_text(encoding="utf-8"))
    questions = jeu["questions"]
    sections = {p.stem: _lire_sections(p) for p in sorted(CORPUS.glob("*.md"))}
    textes = {
        p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.md"))
    }

    par_fiche: dict[str, list[dict]] = defaultdict(list)
    hors_perimetre: list[dict] = []
    for q in questions:
        if q["answerable"]:
            par_fiche[q["fiches_attendues"][0]].append(q)
        else:
            hors_perimetre.append(q)

    lignes: list[str] = ["# Relecture du jeu de validation v2", ""]
    consigne = (
        "Pour chaque question : la reponse est-elle presente dans le texte "
        "des sections attendues ? Note : OK, a corriger, ou a supprimer."
    )
    lignes += [consigne, ""]

    for fiche in sorted(par_fiche):
        lignes += [f"## Fiche : {fiche}", ""]
        demandees: list[str] = []
        for q in par_fiche[fiche]:
            for s in q["sections_attendues"]:
                if _normaliser(s) not in demandees:
                    demandees.append(_normaliser(s))
        for cle in demandees:
            if cle in sections[fiche]:
                titre, contenu = sections[fiche][cle]
                lignes += [f"### Section : {titre}", "", contenu, ""]
        for q in par_fiche[fiche]:
            attendues = ", ".join(q["sections_attendues"])
            fiches_att = ", ".join(q["fiches_attendues"])
            lignes += [
                f"**{q['id']}** ({q['type']}, {q['langue']}) : {q['question']}",
                f"Sections attendues : {attendues} | Fiches attendues : {fiches_att}",
                "Note : [ ] OK   [ ] a corriger   [ ] a supprimer",
                "",
            ]
            for autre in q["fiches_attendues"][1:]:
                for s in q["sections_attendues"]:
                    cle = _normaliser(s)
                    if cle in sections[autre]:
                        titre, contenu = sections[autre][cle]
                        lignes += [
                            f"Autre fiche attendue ({autre}), section {titre} :",
                            "",
                            contenu,
                            "",
                        ]
            if q.get("notes"):
                lignes += [f"Remarque : {q['notes']}", ""]

    lignes += ["## Questions hors perimetre", ""]
    for q in hors_perimetre:
        lignes += [f"**{q['id']}** ({q['type']}) : {q['question']}"]
        trouves = []
        for mot in MOTS_CLES_HORS_PERIMETRE:
            if mot in _normaliser(q["question"]):
                fiches = _mentions(textes, mot)
                trouves.append(
                    f"'{mot}' : {', '.join(fiches) if fiches else 'absent du corpus'}"
                )
        if trouves:
            lignes += ["Mots-cles dans le corpus : " + " ; ".join(trouves)]
        lignes += ["Note : [ ] bien hors perimetre   [ ] traite par le corpus", ""]

    SORTIE.write_text("\n".join(lignes), encoding="utf-8")
    print(f"Document ecrit : {SORTIE}")
    print(
        f"{len(par_fiche)} fiches, {sum(len(v) for v in par_fiche.values())} questions, "
        f"{len(hors_perimetre)} hors perimetre"
    )


if __name__ == "__main__":
    main()
