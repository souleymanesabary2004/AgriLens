"""Tests du decoupage du corpus (P1.2)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from agrilens.genai.chunking import (
    ChunkingConfig,
    chunk_corpus,
    chunk_fiche,
    parse_fiche,
    split_section,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "07_rag_genai" / "corpus"
MANIFEST = ROOT / "01_data" / "metadata" / "split_manifest_v2.csv"

ENTETE = """---
id: test_fiche
titre: Fiche de test
culture: Tomate
agent: Aucun
classe_modele: Test___classe
label: tomato__test
date_consultation: 22 novembre 2026
---
# Fiche de test
"""


def _ecrire(tmp_path: Path, corps: str, entete: str = ENTETE) -> Path:
    path = tmp_path / "test_fiche.md"
    path.write_text(entete + corps, encoding="utf-8", newline="\n")
    return path


def _puces(n: int, taille: int) -> str:
    return "\n".join(f"- puce {i:02d} " + "x" * taille for i in range(n))


# --- tests unitaires sur des fiches synthetiques -------------------------


def test_section_courte_reste_entiere(tmp_path: Path) -> None:
    path = _ecrire(tmp_path, "\n## Symptômes\n- une puce courte\n")
    chunks = chunk_fiche(path)
    assert len(chunks) == 1
    assert chunks[0].texte == "Fiche de test : Symptômes\n- une puce courte"
    assert chunks[0].n_parts == 1


def test_section_sources_jamais_indexee_mais_en_metadonnee(tmp_path: Path) -> None:
    corps = (
        "\n## Symptômes\n- texte\n\n"
        "## Sources (consultées le 22 novembre 2026)\n"
        "- Organisme A : https://exemple.org/a\n"
        "- Organisme B : https://exemple.org/b\n"
    )
    chunks = chunk_fiche(_ecrire(tmp_path, corps))
    assert [c.section for c in chunks] == ["Symptômes"]
    assert all("http" not in c.texte for c in chunks)
    assert chunks[0].sources == (
        "Organisme A : https://exemple.org/a",
        "Organisme B : https://exemple.org/b",
    )


def test_section_longue_coupee_entre_les_puces_avec_reprise(tmp_path: Path) -> None:
    config = ChunkingConfig(max_chars=300)
    contenu = _puces(8, 80)
    parties = split_section(contenu, config)
    assert len(parties) > 1
    assert all(len(p) <= 300 for p in parties)
    lignes_initiales = set(contenu.splitlines())
    for partie in parties:
        assert set(partie.splitlines()) <= lignes_initiales  # jamais de puce coupee
    for avant, apres in pairwise(parties):
        assert avant.splitlines()[-1] == apres.splitlines()[0]  # reprise
    assert {ln for p in parties for ln in p.splitlines()} == lignes_initiales


def test_sans_reprise(tmp_path: Path) -> None:
    config = ChunkingConfig(max_chars=300, overlap=False)
    parties = split_section(_puces(8, 80), config)
    vus = [ln for p in parties for ln in p.splitlines()]
    assert len(vus) == len(set(vus)) == 8


def test_bloc_unique_trop_long_est_coupe_par_phrases() -> None:
    phrase = "Ceci est une phrase assez longue pour le test. "
    contenu = "- " + phrase * 30
    parties = split_section(contenu, ChunkingConfig(max_chars=300))
    assert len(parties) > 1
    assert all(len(p) <= 300 for p in parties)


def test_mot_unique_plus_long_que_la_limite_est_coupe() -> None:
    parties = split_section("x" * 500, ChunkingConfig(max_chars=200))
    assert all(len(p) <= 200 for p in parties)
    assert "".join(parties) == "x" * 500


def test_identifiants_stables_et_sans_accents(tmp_path: Path) -> None:
    path = _ecrire(tmp_path, "\n## À savoir\n- texte\n")
    chunk = chunk_fiche(path)[0]
    assert chunk.chunk_id == "test_fiche::a_savoir::1"


def test_fins_de_ligne_windows_acceptees(tmp_path: Path) -> None:
    path = tmp_path / "crlf.md"
    texte = (ENTETE + "\n## Symptômes\n- une puce\n").replace("\n", "\r\n")
    path.write_bytes(texte.encode("utf-8"))
    assert chunk_fiche(path)[0].contenu == "- une puce"


def test_champ_obligatoire_manquant(tmp_path: Path) -> None:
    entete = ENTETE.replace("label: tomato__test\n", "")
    path = _ecrire(tmp_path, "\n## Symptômes\n- texte\n", entete)
    with pytest.raises(ValueError, match="label"):
        parse_fiche(path)


def test_configuration_invalide() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars=10)


# --- tests sur le vrai corpus ---------------------------------------------


@pytest.fixture(scope="module")
def chunks() -> list:
    return chunk_corpus(CORPUS)


def test_corpus_toutes_les_fiches_representees(chunks: list) -> None:
    assert len({c.doc_id for c in chunks}) == 17


def test_corpus_labels_couvrent_le_manifeste(chunks: list) -> None:
    attendus = set(pd.read_csv(MANIFEST)["label"].unique())
    assert {c.label for c in chunks} == attendus


def test_corpus_taille_maximale_respectee(chunks: list) -> None:
    assert max(len(c.contenu) for c in chunks) <= 800


def test_corpus_aucun_chunk_ne_contient_les_sources(chunks: list) -> None:
    assert all("http" not in c.texte for c in chunks)
    assert all(c.sources for c in chunks)


def test_corpus_chaque_section_de_contenu_est_presente(chunks: list) -> None:
    for path in sorted(CORPUS.glob("*.md")):
        attendues = {titre for titre, _ in parse_fiche(path).sections}
        obtenues = {c.section for c in chunks if c.doc_id == path.stem}
        assert attendues == obtenues, path.name


def test_corpus_decoupage_deterministe() -> None:
    assert chunk_corpus(CORPUS) == chunk_corpus(CORPUS)
