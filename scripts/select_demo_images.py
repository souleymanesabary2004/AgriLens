"""
select_demo_images.py - choisit et copie quelques images du split TEST pour la demonstration en entretien.

Ce script NE re-mesure PAS le test et ne change rien : il lit le manifeste, les embeddings deja calcules et
le modele deja entraine, puis COPIE 4 images dans le dossier demo_images/. Il ne modifie aucun fichier du projet.
(Les images servent a illustrer, pas a evaluer ; le test a deja ete mesure une fois.)

Lancer depuis la racine du projet :  python scripts\\select_demo_images.py
Autre tirage :  python scripts\\select_demo_images.py --seed 7
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="01_data/metadata/split_manifest_v2.csv")
    ap.add_argument("--embeddings", default="01_data/processed/embeddings_mobilenetv2_v1.parquet")
    ap.add_argument("--model", default="models/head_mobilenetv2_v1.pt")
    ap.add_argument("--threshold", default="05_deep_learning/results/abstention_threshold.json")
    ap.add_argument("--out", default="demo_images")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    ckpt = torch.load(a.model, weights_only=False)
    id2label = {int(k): v for k, v in ckpt["id2label"].items()}
    mu = torch.from_numpy(np.asarray(ckpt["mu"], dtype=np.float32))
    sd = torch.from_numpy(np.asarray(ckpt["sd"], dtype=np.float32))
    head = torch.nn.Linear(len(mu), len(id2label))
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    thr = float(json.load(open(a.threshold))["threshold"])

    man = pd.read_csv(a.manifest)
    emb = pd.read_parquet(a.embeddings)
    df = man[man["split"] == a.split].merge(emb, on="filepath_rel", validate="one_to_one").reset_index(drop=True)
    cols = [c for c in emb.columns if c != "filepath_rel"]
    X = (torch.from_numpy(df[cols].values.astype(np.float32)) - mu) / sd
    with torch.no_grad():
        proba = torch.softmax(head(X), dim=1).numpy()
    df["pred_id"] = proba.argmax(1)
    df["conf"] = proba.max(1)
    df["true_label"] = df["class_id"].map(id2label)
    df["pred_label"] = df["pred_id"].map(id2label)
    df["correct"] = df["pred_id"] == df["class_id"]
    df["abstained"] = df["conf"] < thr
    cacao = {"black_pod_rot", "pod_borer"}
    df["is_cacao_disease"] = df["true_label"].isin(cacao)
    rng = np.random.RandomState(a.seed)

    def pick(mask, sort_by=None, ascending=False):
        sub = df[mask]
        if len(sub) == 0:
            return None
        if sort_by:
            sub = sub.sort_values(sort_by, ascending=ascending).head(30)
        return sub.iloc[rng.randint(len(sub))]

    choices = [
        ("1_reconnue", pick(df["correct"] & ~df["abstained"] & ~df["is_cacao_disease"] & (df["conf"] > 0.99)),
         "Cas simple : bonne classe, confiance elevee, decision acceptee."),
        ("2_cacao", pick(df["correct"] & ~df["abstained"] & df["is_cacao_disease"], "conf", False),
         "Cacao : classe plus difficile ; montre que le systeme couvre aussi le cacao."),
        ("4_renvoyee_bis", pick(df["abstained"] & df["correct"], "conf", False),
         "Cas prudent : bonne classe mais confiance sous le seuil ; a expliquer (le systeme prefere s'abstenir)."),
        ("3_renvoyee", pick(df["abstained"] & ~df["correct"], "conf", False),
         "Abstention utile : le modele se trompe, mais sa confiance est sous le seuil, donc renvoi a un humain."),
    ]
    os.makedirs(a.out, exist_ok=True)
    lines = [f"Seuil d'abstention : {thr}", f"Images tirees du split : {a.split}", ""]
    for name, row, why in choices:
        if row is None:
            lines.append(f"{name} : aucune image trouvee avec ces criteres.")
            continue
        src = row["filepath_rel"]
        ext = os.path.splitext(src)[1] or ".jpg"
        dst = os.path.join(a.out, name + ext)
        shutil.copyfile(src, dst)
        lines += [f"{os.path.basename(dst)}",
                  f"  vraie classe  : {row['true_label']}",
                  f"  predite       : {row['pred_label']}",
                  f"  confiance     : {row['conf']:.4f}",
                  f"  decision      : {'renvoyee a un humain' if row['abstained'] else 'acceptee'}",
                  f"  pourquoi      : {why}",
                  f"  source        : {src}", ""]
    open(os.path.join(a.out, "LISEZ_MOI.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"Images copiees dans : {a.out}")


if __name__ == "__main__":
    main()
