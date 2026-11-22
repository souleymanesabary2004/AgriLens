"""Controle qualite du corpus RAG (P1.1) : en-tetes, sections, sources, absence de dosage."""
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "07_rag_genai" / "corpus"
MANIFEST = ROOT / "01_data" / "metadata" / "split_manifest_v2.csv"

DISEASE_SECTIONS = [
    "Symptômes", "Conditions favorables", "Confusions possibles",
    "Prévention et pratiques générales", "À savoir", "Sources",
]
HEALTHY_SECTIONS = [
    "Aspect d'une plante saine", "Ce qui peut ressembler à une maladie",
    "Surveillance et pratiques générales", "À savoir", "Sources",
]
HEADER_FIELDS = ["id", "titre", "culture", "agent", "classe_modele", "label", "date_consultation"]
# dosage = nombre suivi d'une unite de quantite de produit (pas les degres ni les pourcentages d'humidite)
DOSAGE = re.compile(r"\d\s?(?:g|kg|ml|l|cl|ppm)\s?(?:/|par\s)\s?(?:l|ha|hl|m2|m²|litre|hectare)\b", re.I)
DOSAGE_WORDS = re.compile(r"\b(?:dose|dosage|posologie)\b", re.I)

FILES = sorted(CORPUS.glob("*.md"))


def parse(path):
    text = path.read_text(encoding="utf-8")
    parts = text.split("---")
    header = dict(
        ln.split(":", 1) for ln in parts[1].strip().splitlines() if ":" in ln
    )
    header = {k.strip(): v.strip() for k, v in header.items()}
    body = "---".join(parts[2:])
    sections = re.findall(r"^## (.+)$", body, re.M)
    return text, header, body, [s.split(" (")[0].strip() for s in sections]


def test_corpus_has_17_fiches():
    assert len(FILES) == 17


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_header_complete_and_id_matches_filename(path):
    _, header, _, _ = parse(path)
    for field in HEADER_FIELDS:
        assert header.get(field), f"{path.name}: champ {field} manquant"
    assert header["id"] == path.stem


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_label_in_manifest(path):
    _, header, _, _ = parse(path)
    known = set(pd.read_csv(MANIFEST)["label"].unique())
    assert header["label"] in known


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_sections(path):
    _, header, _, sections = parse(path)
    expected = HEALTHY_SECTIONS if header["label"].endswith("__healthy") else DISEASE_SECTIONS
    assert sections == expected


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_sources_have_url(path):
    _, _, body, _ = parse(path)
    sources = body.split("## Sources")[1]
    assert re.search(r"https?://\S+", sources), f"{path.name}: aucune URL dans Sources"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_disclaimer_present(path):
    text, _, _, _ = parse(path)
    assert "prototype pédagogique" in text


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_no_dosage_and_clean_encoding(path):
    text, _, _, _ = parse(path)
    assert not DOSAGE.search(text), f"{path.name}: dosage detecte"
    assert not DOSAGE_WORDS.search(text), f"{path.name}: mot de dosage detecte"
    assert "�" not in text


def test_each_manifest_label_has_one_fiche():
    labels = sorted(parse(p)[1]["label"] for p in FILES)
    assert labels == sorted(set(labels))
    assert set(labels) == set(pd.read_csv(MANIFEST)["label"].unique())
