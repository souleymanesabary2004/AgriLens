"""Tests de l'API avec un FAUX modele (instantane) : on teste le service, pas les predictions."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


class FakePredictor:
    threshold = 0.9

    def __init__(self, confidence=0.97):
        self.confidence = confidence

    def predict(self, img):
        abst = self.confidence < self.threshold
        return {"class_id": 3, "label": "tomato__late_blight", "display_fr": "Tomate - Mildiou",
                "confidence": self.confidence, "abstained": abst,
                "decision": "a_verifier_par_un_humain" if abst else "accepte",
                "threshold": self.threshold,
                "top": [{"label": "tomato__late_blight", "probability": self.confidence}]}


def png_bytes(size=(60, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 10)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(main.create_app(FakePredictor())) as c:
        yield c


def post(client, name, content, ctype):
    return client.post("/predict", files={"file": (name, content, ctype)})


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "threshold": 0.9}


def test_predict_image_valide(client):
    r = post(client, "a.png", png_bytes(), "image/png")
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "accepte" and body["abstained"] is False
    assert set(body) == {"class_id", "label", "display_fr", "confidence", "abstained",
                         "decision", "threshold", "top"}


def test_predict_abstention_transmise():
    with TestClient(main.create_app(FakePredictor(confidence=0.5))) as c:
        body = post(c, "a.png", png_bytes(), "image/png").json()
    assert body["abstained"] is True and body["decision"] == "a_verifier_par_un_humain"


def test_predict_type_non_supporte(client):
    assert post(client, "a.txt", b"hello", "text/plain").status_code == 415


def test_predict_contenu_corrompu(client):
    assert post(client, "a.png", b"ceci n'est pas une image", "image/png").status_code == 400


def test_predict_fichier_trop_gros(client):
    big = b"0" * (main.MAX_BYTES + 10)
    assert post(client, "a.png", big, "image/png").status_code == 413


def test_predict_sans_fichier(client):
    assert client.post("/predict").status_code == 422


def test_sans_modele_charge():
    c = TestClient(main.create_app())   # pas de 'with' : le demarrage n'est pas execute, aucun modele
    assert c.get("/health").json()["model_loaded"] is False
    assert post(c, "a.png", png_bytes(), "image/png").status_code == 503
