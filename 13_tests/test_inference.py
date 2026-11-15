"""Tests du module d'inference avec un modele JOUET (poids aleatoires, sans ImageNet) :
on verifie le comportement du code, pas la qualite des predictions."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from agrilens.inference import Predictor

K, D = 17, 1280


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    torch.manual_seed(0)
    head = torch.nn.Linear(D, K)
    torch.save({"state_dict": head.state_dict(), "mu": np.zeros(D, np.float32), "sd": np.ones(D, np.float32),
                "id2label": {i: f"classe_{i}" for i in range(K)}}, d / "head.pt")
    json.dump({"threshold": 0.2}, open(d / "thr.json", "w"))
    return Predictor(model_path=d / "head.pt", threshold_path=d / "thr.json", manifest_path=None, pretrained=False)


def image(seed=0, size=(70, 50), mode="RGB"):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (size[1], size[0], len(mode)), dtype=np.uint8), mode)


def test_forme_de_la_sortie(predictor):
    out = predictor.predict(image())
    assert set(out) == {"class_id", "label", "display_fr", "confidence", "abstained",
                        "decision", "threshold", "top"}
    assert 0 <= out["class_id"] < K and 0 < out["confidence"] <= 1
    assert len(out["top"]) == 3
    probs = [t["probability"] for t in out["top"]]
    assert probs == sorted(probs, reverse=True)


def test_deterministe(predictor):
    assert predictor.predict(image(1)) == predictor.predict(image(1))


def test_decision_suit_le_seuil(predictor):
    out = predictor.predict(image(2))
    assert out["abstained"] == (out["confidence"] < out["threshold"])
    assert out["decision"] == ("a_verifier_par_un_humain" if out["abstained"] else "accepte")


def test_embedding_taille(predictor):
    assert tuple(predictor.embedding(image()).shape) == (D,)


def test_accepte_les_images_non_rgb(predictor):
    assert predictor.predict(image(3, mode="RGBA"))["label"].startswith("classe_")


def test_taille_d_image_indifferente(predictor):
    assert predictor.embedding(image(4, size=(33, 90))).shape == predictor.embedding(image(4, size=(500, 120))).shape


@pytest.mark.parametrize("bad", [123, None, 3.5])
def test_entree_invalide_refusee(predictor, bad):
    with pytest.raises(ValueError):
        predictor.predict(bad)
