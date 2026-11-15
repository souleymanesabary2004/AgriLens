"""
inference.py - E3 : utiliser le modele AgriLens sur UNE photo.

Chaine : photo -> RGB -> redimensionnee 224x224 (sans recadrage) -> normalisation ImageNet
         -> MobileNetV2 gele (1280 nombres) -> standardisation (mu/sd du TRAIN) -> couche lineaire
         -> 17 probabilites -> classe + confiance -> abstention si confiance < seuil.

Ce pretraitement doit rester IDENTIQUE a scripts/extract_embeddings.py, sinon les scores changent
sans erreur visible. scripts/check_inference.py le verifie sur de vraies images.

Test rapide :  python src\\agrilens\\inference.py chemin\\vers\\photo.jpg
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(pretrained=True):
    """MobileNetV2 sans sa derniere couche : 1280 nombres par image, gele, mode eval."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    bb = torch.nn.Sequential(net.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    bb.eval()
    for p in bb.parameters():
        p.requires_grad = False
    return bb


class Predictor:
    def __init__(self, model_path="models/head_mobilenetv2_v1.pt",
                 threshold_path="05_deep_learning/results/abstention_threshold.json",
                 manifest_path="01_data/metadata/split_manifest_v2.csv",
                 pretrained=True, size=224):
        ckpt = torch.load(model_path, weights_only=False)
        self.id2label = {int(k): v for k, v in ckpt["id2label"].items()}
        self.mu = torch.from_numpy(np.asarray(ckpt["mu"], dtype=np.float32))
        self.sd = torch.from_numpy(np.asarray(ckpt["sd"], dtype=np.float32))
        self.head = torch.nn.Linear(len(self.mu), len(self.id2label))
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        self.backbone = build_backbone(pretrained)
        self.threshold = float(json.load(open(threshold_path))["threshold"])
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.display = {}
        if manifest_path and os.path.exists(manifest_path):
            m = pd.read_csv(manifest_path, usecols=["class_id", "display_fr"]).drop_duplicates("class_id")
            self.display = dict(zip(m["class_id"].astype(int), m["display_fr"]))

    @staticmethod
    def _open(image):
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise ValueError("image attendue : chemin ou PIL.Image")
        return image.convert("RGB")

    def embedding(self, image):
        """1280 nombres (non standardises) - meme resultat que extract_embeddings.py."""
        x = self.tf(self._open(image)).unsqueeze(0)
        with torch.inference_mode():
            return self.backbone(x)[0]

    def predict(self, image, top_k=3):
        emb = self.embedding(image)
        with torch.inference_mode():
            proba = torch.softmax(self.head((emb - self.mu) / self.sd), dim=0).numpy()
        order = np.argsort(proba)[::-1]
        best = int(order[0])
        conf = float(proba[best])
        return {
            "class_id": best,
            "label": self.id2label[best],
            "display_fr": self.display.get(best, self.id2label[best]),
            "confidence": round(conf, 4),
            "abstained": conf < self.threshold,
            "decision": "a_verifier_par_un_humain" if conf < self.threshold else "accepte",
            "threshold": self.threshold,
            "top": [{"label": self.id2label[int(i)], "probability": round(float(proba[i]), 4)}
                    for i in order[:top_k]],
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage : python src\\agrilens\\inference.py chemin\\vers\\photo.jpg")
    print(json.dumps(Predictor().predict(sys.argv[1]), indent=2, ensure_ascii=False))
