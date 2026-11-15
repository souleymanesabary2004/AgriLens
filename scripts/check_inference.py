"""
check_inference.py - E3 : prouve que inference.py donne les MEMES resultats que l'evaluation.

On prend N images du split test, on les passe une par une dans Predictor (photo -> prediction)
et on compare avec la chaine d'evaluation (embeddings deja calcules -> couche lineaire).
Attendu : memes classes, ecart de confiance minuscule.

Lancer depuis la racine :  python scripts\\check_inference.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from agrilens.inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="01_data/metadata/split_manifest_v2.csv")
    ap.add_argument("--embeddings", default="01_data/processed/embeddings_mobilenetv2_v1.parquet")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--no-pretrained", action="store_true", help="tests uniquement")
    a = ap.parse_args()

    pred = Predictor(pretrained=not a.no_pretrained)
    man = pd.read_csv(a.manifest)
    emb = pd.read_parquet(a.embeddings)
    df = man[man["split"] == "test"].merge(emb, on="filepath_rel", validate="one_to_one")
    df = df.sample(n=min(a.n, len(df)), random_state=2026).reset_index(drop=True)
    cols = [c for c in emb.columns if c != "filepath_rel"]

    same_class, gaps, emb_gaps = 0, [], []
    for _, r in df.iterrows():
        out = pred.predict(r["filepath_rel"])
        e_ref = torch.tensor(r[cols].values.astype(np.float32))
        with torch.no_grad():
            p_ref = torch.softmax(pred.head((e_ref - pred.mu) / pred.sd), 0).numpy()
        same_class += int(out["class_id"] == int(p_ref.argmax()))
        gaps.append(abs(out["confidence"] - float(p_ref.max())))
        emb_gaps.append(float((pred.embedding(r["filepath_rel"]) - e_ref).abs().max()))

    print(f"{len(df)} images du test")
    print(f"meme classe predite : {same_class}/{len(df)}")
    print(f"ecart maximal de confiance : {max(gaps):.2e}")
    print(f"ecart maximal sur les 1280 nombres : {max(emb_gaps):.2e}")
    ok = same_class == len(df) and max(gaps) < 1e-3
    print("RESULTAT :", "OK - le pretraitement de l'API est identique a celui de l'evaluation"
          if ok else "ANORMAL - le pretraitement differe de l'entrainement, ne pas continuer")


if __name__ == "__main__":
    main()
