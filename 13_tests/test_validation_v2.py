"""Tests de coherence du jeu de validation v2 du RAG AgriLens.

Ces tests ne mesurent pas la qualite du RAG : ils verifient que le fichier de
questions est bien forme et coherent avec le corpus (fiches et sections
attendues existent), et qu'aucune question du split test de la v1 n'a ete
reprise. La relecture humaine (champ ``relu``) reste indispensable.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "07_rag_genai" / "corpus"
VALIDATION = ROOT / "07_rag_genai" / "eval" / "validation_v2.json"
V1_QUESTIONS = ROOT / "07_rag_genai" / "eval_questions.json"

TYPES_AUTORISES = {
    "symptomes",
    "conditions",
    "prevention",
    "confusion",
    "plante_saine",
    "hors_perimetre_culture",
    "dosage",
    "hors_perimetre_general",
    "sans_rapport",
    "hors_corpus",
}
MIN_PAR_FICHE = 3
SECTION_SOURCES = "sources"


def _normaliser(texte: str) -> str:
    """Normalise un texte pour comparer sans etre gene par la typographie."""
    texte = unicodedata.normalize("NFC", texte)
    texte = texte.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", texte).strip().lower()


def _sections_de_fiche(chemin: Path) -> set[str]:
    """Retourne l'ensemble des titres de section ('## ...') d'une fiche."""
    contenu = chemin.read_text(encoding="utf-8").replace("\r\n", "\n")
    titres = re.findall(r"^## (.+)$", contenu, flags=re.MULTILINE)
    return {_normaliser(t) for t in titres}


@pytest.fixture(scope="module")
def jeu() -> dict:
    """Charge le fichier de validation v2."""
    assert VALIDATION.exists(), f"Fichier introuvable : {VALIDATION}"
    return json.loads(VALIDATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def questions(jeu: dict) -> list[dict]:
    return jeu["questions"]


@pytest.fixture(scope="module")
def fiches() -> dict[str, set[str]]:
    """Fiche (id = nom du fichier) -> ensemble de ses sections normalisees."""
    resultat = {p.stem: _sections_de_fiche(p) for p in sorted(CORPUS.glob("*.md"))}
    assert resultat, "Aucune fiche trouvee dans le corpus"
    return resultat


@pytest.fixture(scope="module")
def v1() -> list[dict]:
    return json.loads(V1_QUESTIONS.read_text(encoding="utf-8"))


def test_metadonnees(jeu: dict) -> None:
    assert jeu["version"] == "validation_v2"
    assert jeu["date"]
    assert jeu["description"]


def test_nombre_de_questions(questions: list[dict]) -> None:
    repondables = [q for q in questions if q["answerable"]]
    assert len(questions) == 66
    assert len(repondables) == 51
    assert len(questions) - len(repondables) == 15


def test_champs_obligatoires(questions: list[dict]) -> None:
    attendus = {
        "id",
        "question",
        "type",
        "answerable",
        "fiches_attendues",
        "sections_attendues",
        "origine",
        "langue",
        "relu",
    }
    for q in questions:
        manquants = attendus - set(q)
        assert not manquants, f"{q.get('id')} : champs manquants {manquants}"
        assert isinstance(q["relu"], bool), f"{q['id']} : relu doit etre booleen"
        assert q["langue"] in {"fr", "en"}, f"{q['id']} : langue inconnue"
        assert q["origine"] in {"v1-dev", "nouvelle"}, f"{q['id']} : origine inconnue"
        assert q["type"] in TYPES_AUTORISES, f"{q['id']} : type inconnu"


def test_identifiants_et_textes_uniques(questions: list[dict]) -> None:
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "identifiants dupliques"
    textes = [_normaliser(q["question"]) for q in questions]
    assert len(textes) == len(set(textes)), "questions dupliquees"


def test_coherence_repondable(questions: list[dict]) -> None:
    for q in questions:
        if q["answerable"]:
            assert q["fiches_attendues"], f"{q['id']} : fiche attendue manquante"
            assert q["sections_attendues"], f"{q['id']} : section attendue manquante"
        else:
            assert not q["fiches_attendues"], f"{q['id']} : fiche attendue interdite"
            assert not q["sections_attendues"], (
                f"{q['id']} : section attendue interdite"
            )


def test_fiches_attendues_existent(
    questions: list[dict], fiches: dict[str, set[str]]
) -> None:
    for q in questions:
        for fiche in q["fiches_attendues"]:
            assert fiche in fiches, f"{q['id']} : fiche inconnue {fiche}"


def test_sections_attendues_existent(
    questions: list[dict], fiches: dict[str, set[str]]
) -> None:
    """Chaque section attendue doit exister dans au moins une fiche attendue."""
    for q in questions:
        for section in q["sections_attendues"]:
            cible = _normaliser(section)
            assert cible != SECTION_SOURCES, (
                f"{q['id']} : la section Sources n'est pas indexee"
            )
            trouvee = any(cible in fiches[f] for f in q["fiches_attendues"])
            assert trouvee, (
                f"{q['id']} : section '{section}' absente des fiches attendues"
            )


def test_toutes_les_fiches_couvertes(
    questions: list[dict], fiches: dict[str, set[str]]
) -> None:
    """Chaque fiche (17 classes) est la fiche principale d'au moins 3 questions."""
    principales = Counter(
        q["fiches_attendues"][0] for q in questions if q["answerable"]
    )
    manquantes = {
        f: principales.get(f, 0)
        for f in fiches
        if principales.get(f, 0) < MIN_PAR_FICHE
    }
    assert not manquantes, f"fiches insuffisamment couvertes : {manquantes}"


def test_questions_en_anglais(questions: list[dict]) -> None:
    anglais = [q for q in questions if q["langue"] == "en"]
    assert len(anglais) == 4
    assert all(q["answerable"] for q in anglais)


def test_dosage_hors_perimetre(questions: list[dict]) -> None:
    dosage = [q for q in questions if q["type"] == "dosage"]
    assert len(dosage) == 4
    assert all(not q["answerable"] for q in dosage)


def test_origine_v1_dev_verifiee(questions: list[dict], v1: list[dict]) -> None:
    """Les questions marquees v1-dev existent bien dans le split dev de la v1."""
    dev = {_normaliser(x["q"]) for x in v1 if x["split"] == "dev"}
    for q in questions:
        if q["origine"] == "v1-dev":
            assert _normaliser(q["question"]) in dev, f"{q['id']} : absente du dev v1"


def test_aucune_question_du_test_v1(questions: list[dict], v1: list[dict]) -> None:
    """Aucune question de validation v2 ne reprend une question du test v1."""
    test = {_normaliser(x["q"]) for x in v1 if x["split"] == "test"}
    communes = [q["id"] for q in questions if _normaliser(q["question"]) in test]
    assert not communes, f"questions identiques au test v1 : {communes}"
