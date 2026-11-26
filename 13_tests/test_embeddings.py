"""Tests du mode d'emploi des modeles d'embeddings (sans charger aucun modele)."""

import re

import pytest

from agrilens.genai.embeddings import (
    INSTRUCTION_TACHE,
    MODELS,
    format_documents,
    format_queries,
)

CLES = {
    "minilm",
    "bge-m3",
    "arctic-l-v2",
    "qwen3-0.6b",
    "e5-large-instruct",
    "gte-base",
}


def test_six_modeles_attendus() -> None:
    assert set(MODELS) == CLES


@pytest.mark.parametrize("cle", sorted(CLES))
def test_revision_est_un_sha_complet(cle: str) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", MODELS[cle].revision)


@pytest.mark.parametrize("cle", sorted(CLES))
def test_aucun_code_distant_autorise(cle: str) -> None:
    assert MODELS[cle].trust_remote_code is False


@pytest.mark.parametrize("cle", sorted(CLES))
def test_aucun_prefixe_sur_les_passages(cle: str) -> None:
    assert MODELS[cle].doc_prefix == ""


@pytest.mark.parametrize("cle", ["minilm", "bge-m3", "gte-base"])
def test_modeles_sans_prefixe_de_question(cle: str) -> None:
    assert MODELS[cle].query_prefix == ""


def test_prefixe_arctic() -> None:
    assert MODELS["arctic-l-v2"].query_prefix == "query: "


def test_instruction_qwen3() -> None:
    prefixe = MODELS["qwen3-0.6b"].query_prefix
    assert prefixe.startswith("Instruct: ")
    assert INSTRUCTION_TACHE in prefixe
    assert prefixe.endswith("\nQuery:")


def test_instruction_e5_instruct() -> None:
    prefixe = MODELS["e5-large-instruct"].query_prefix
    assert prefixe.startswith("Instruct: ")
    assert INSTRUCTION_TACHE in prefixe
    assert prefixe.endswith("\nQuery: ")


def test_qwen3_et_e5_ont_la_meme_instruction() -> None:
    qwen = MODELS["qwen3-0.6b"].query_prefix
    e5 = MODELS["e5-large-instruct"].query_prefix
    assert qwen.split("\nQuery:")[0] == e5.split("\nQuery: ")[0]


def test_format_queries_colle_le_prefixe() -> None:
    texte = format_queries(MODELS["arctic-l-v2"], ["une question"])
    assert texte == ["query: une question"]


def test_format_documents_ne_change_rien() -> None:
    assert format_documents(MODELS["bge-m3"], ["un passage"]) == ["un passage"]
