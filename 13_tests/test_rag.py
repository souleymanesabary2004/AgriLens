"""Tests E4 : recherche documentaire, garde-fous, endpoint /ask."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from main import create_app

from agrilens.rag import REFUSAL, Retriever, tokenize

CORPUS = Path(__file__).resolve().parents[1] / "07_rag_genai/corpus"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus absent")


@pytest.fixture(scope="module")
def r():
    return Retriever(CORPUS, threshold=0.12)


def test_tokenize_enleve_accents_pluriels_et_mots_vides():
    assert tokenize("Les taches brunes de la tomate") == ["tache", "brune", "tomate"]


def test_corpus_charge_14_fiches(r):
    assert len({c.doc_id for c in r.chunks}) == 14


def test_question_dans_le_corpus_trouve_la_bonne_fiche(r):
    res = r.search("Pustules couleur rouille sur les deux faces des feuilles de maïs")
    assert res["answered"] and res["results"][0]["doc_id"] == "mais_rouille_commune"


def test_chaque_reponse_a_des_sources_avec_url(r):
    res = r.search("Duvet vert olive sous les feuilles de tomates en serre humide")
    assert res["answered"]
    for p in res["results"]:
        assert p["sources"] and all(s["url"].startswith("http") for s in p["sources"])


def test_hors_sujet_est_refuse(r):
    res = r.search("Quelle est la capitale de la France ?")
    assert not res["answered"] and res["message"] == REFUSAL and res["results"] == []


def test_dosage_est_refuse(r):
    res = r.search("Quelle dose de produit pour traiter mes tomates ?")
    assert not res["answered"] and res["reason"] == "dosage_ou_produit"


def test_culture_hors_perimetre_est_refusee(r):
    assert r.search("Comment traiter le mildiou de la vigne ?")["reason"] == "culture_hors_perimetre"


def test_question_vide(r):
    assert not r.search("   ")["answered"]


@pytest.fixture(scope="module")
def client(r):
    from test_api import FakePredictor
    with TestClient(create_app(predictor=FakePredictor(), retriever=r)) as c:
        yield c


def test_ask_ok(client):
    out = client.post("/ask", json={"question": "Larves dans les cabosses et fèves collées entre elles"})
    assert out.status_code == 200 and out.json()["answered"] is True


def test_ask_refus(client):
    out = client.post("/ask", json={"question": "Explique la photosynthèse"})
    assert out.status_code == 200 and out.json()["answered"] is False


def test_ask_question_trop_longue(client):
    assert client.post("/ask", json={"question": "a" * 501}).status_code == 422


def test_ask_sans_corpus_503():
    from test_api import FakePredictor
    with TestClient(create_app(predictor=FakePredictor(), retriever=None)) as c:
        assert c.post("/ask", json={"question": "tomate"}).status_code == 503


# ---------- lien classe predite -> fiche ----------
@pytest.mark.parametrize("label,doc", [
    ("Tomato___Late_blight", "tomate_mildiou"),
    ("Tomato___Early_blight", "tomate_alternariose"),
    ("Tomato___Leaf_Mold", "tomate_cladosporiose"),
    ("Tomato___Septoria_leaf_spot", "tomate_septoriose"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "tomate_acariens"),
    ("Tomato___Target_Spot", "tomate_corynesporiose"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomate_tylcv"),
    ("Tomato___Tomato_mosaic_virus", "tomate_mosaique"),
    ("Tomato___Bacterial_spot", "tomate_tache_bacterienne"),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "mais_cercosporiose"),
    ("Corn_(maize)___Common_rust_", "mais_rouille_commune"),
    ("Corn_(maize)___Northern_Leaf_Blight", "mais_helminthosporiose"),
    ("black_pod_rot", "cacao_pourriture_brune"),
    ("pod_borer", "cacao_foreur_cabosses"),
])
def test_label_vers_fiche(r, label, doc):
    f = r.fiche_for_label(label)
    assert f["kind"] == "fiche" and f["doc_id"] == doc and f["sources"]


@pytest.mark.parametrize("label", ["Tomato___healthy", "Corn_(maize)___healthy", "healthy"])
def test_classe_saine_message_fixe(r, label):
    f = r.fiche_for_label(label)
    assert f["kind"] == "healthy" and f["sources"] == []


def test_classe_inconnue_pas_de_fiche(r):
    assert r.fiche_for_label("classe_totalement_inconnue")["kind"] == "none"


def test_endpoint_fiche(client):
    out = client.get("/fiche", params={"label": "Corn_(maize)___Common_rust_"})
    assert out.status_code == 200 and out.json()["doc_id"] == "mais_rouille_commune"
