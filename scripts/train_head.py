"""
train_head.py - E1 : entraine la DERNIERE COUCHE (1280 -> 17 classes) sur les embeddings MobileNetV2
calcules par extract_embeddings.py, puis compare avec la regression logistique et le Random Forest.

Le backbone est gele : seuls 1280 x 17 + 17 = 21 777 parametres sont appris.

Protocole (identique a train_baselines.py) :
  - le split TEST n'est JAMAIS charge ;
  - les embeddings sont standardises avec la moyenne/ecart-type du TRAIN uniquement ;
  - metrique principale : macro-F1 ; le meilleur epoch est choisi sur la VALIDATION ;
  - 3 tests de bon sens avant l'entrainement (voir plus bas).

Lancer depuis la racine du projet :
    python scripts\\train_head.py
"""
import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

SEED = 2026


def seed_all(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ok_name(txt):
    """MLflow n'accepte que lettres, chiffres, _ - . espace / dans les noms de metriques."""
    import re
    return re.sub(r"[^A-Za-z0-9_\-\. /]", "", str(txt))


def metrics(y, pred):
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred)}


def by_group(frame, pred, col):
    out = {}
    frame = frame.reset_index(drop=True)
    for g, idx in frame.groupby(col).groups.items():
        idx = list(idx)
        yt = frame["class_id"].values[idx]
        out[g] = round(f1_score(yt, pred[idx], labels=np.unique(yt), average="macro", zero_division=0), 4)
    return out


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).argmax(1).numpy()


def train_epoch(model, opt, crit, X, y, batch, gen):
    model.train()
    perm = torch.randperm(len(X), generator=gen)
    for i in range(0, len(X), batch):
        idx = perm[i:i + batch]
        opt.zero_grad()
        loss = crit(model(X[idx]), y[idx])
        loss.backward()
        opt.step()


def make_model(d, k):
    m = torch.nn.Linear(d, k)
    torch.nn.init.zeros_(m.weight)    # poids a zero : le modele part de "toutes les classes sont egales"
    torch.nn.init.zeros_(m.bias)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="01_data/metadata/split_manifest_v2.csv")
    ap.add_argument("--embeddings", default="01_data/processed/embeddings_mobilenetv2_v1.parquet")
    ap.add_argument("--baselines", default="04_ml_baselines/results/comparison_baselines_val.csv")
    ap.add_argument("--out", default="05_deep_learning/results")
    ap.add_argument("--model-out", default="models/head_mobilenetv2_v1.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12, help="arret si pas d'amelioration pendant N epochs")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--mlflow-db", default="sqlite:///10_mlops/mlflow.db")
    ap.add_argument("--experiment", default="agrilens-s2-final")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed_all()

    # 1. donnees : train + validation seulement 
    man = pd.read_csv(a.manifest)
    emb = pd.read_parquet(a.embeddings)
    df = man.merge(emb, on="filepath_rel", how="inner", validate="one_to_one")
    assert len(df) == len(man), f"embeddings manquants : {len(man) - len(df)} images du manifeste sans embedding"
    val_name = "validation" if "validation" in set(df["split"]) else "val"
    cols = [c for c in emb.columns if c != "filepath_rel"]
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == val_name].reset_index(drop=True)      # le split 'test' n'est jamais utilise ici
    print(f"train={len(train)}  val={len(val)}  dimensions={len(cols)}  (split test ignore)")

    id2label = df.drop_duplicates("class_id").set_index("class_id")["label"].to_dict()
    K = len(id2label)
    Xtr_raw = train[cols].values.astype(np.float32)
    Xva_raw = val[cols].values.astype(np.float32)
    mu = Xtr_raw.mean(0)
    sd = np.maximum(Xtr_raw.std(0), 1e-6)       # calcules sur le TRAIN seulement
    Xtr = torch.from_numpy((Xtr_raw - mu) / sd)
    Xva = torch.from_numpy((Xva_raw - mu) / sd)
    ytr = torch.from_numpy(train["class_id"].values.astype(np.int64))
    yva = val["class_id"].values

    counts = np.bincount(ytr.numpy(), minlength=K).astype(np.float32)
    w = None if a.no_class_weight else torch.tensor(len(ytr) / (K * np.maximum(counts, 1)), dtype=torch.float32)
    crit = torch.nn.CrossEntropyLoss(weight=w)

    #  2. tests de bon sens 
    print("\n--- Tests de bon sens ---")
    m0 = make_model(len(cols), K)
    with torch.no_grad():
        l0 = crit(m0(Xtr[:2048]), ytr[:2048]).item()
    print(f"1) perte initiale = {l0:.4f} (attendu ln({K}) = {math.log(K):.4f})  "
          f"{'OK' if abs(l0 - math.log(K)) < 0.01 else 'ANORMAL'}")

    seed_all()
    sub = torch.from_numpy(np.random.RandomState(SEED).choice(len(Xtr), 32, replace=False))
    ms = make_model(len(cols), K)
    os_ = torch.optim.AdamW(ms.parameters(), lr=0.05, weight_decay=0.0)
    for _ in range(300):
        os_.zero_grad()
        ls = crit(ms(Xtr[sub]), ytr[sub])
        ls.backward()
        os_.step()
    acc_s = (ms(Xtr[sub]).argmax(1) == ytr[sub]).float().mean().item()
    print(f"2) sur-apprentissage d'un mini-lot de 32 images : perte finale = {ls.item():.4f}, "
          f"accuracy = {acc_s:.2f}  {'OK' if ls.item() < 0.05 and acc_s == 1.0 else 'ANORMAL'}")

    ws = []
    for _ in range(2):
        seed_all()
        mm = make_model(len(cols), K)
        oo = torch.optim.AdamW(mm.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        g = torch.Generator().manual_seed(SEED)
        for _e in range(2):
            train_epoch(mm, oo, crit, Xtr, ytr, a.batch, g)
        ws.append(mm.weight.detach().clone())
    diff = (ws[0] - ws[1]).abs().max().item()
    print(f"3) determinisme : ecart maximal entre deux entrainements identiques = {diff:.2e}  "
          f"{'OK' if diff < 1e-5 else 'ANORMAL'}")

    # 3. entrainement 
    print("\n--- Entrainement de la derniere couche ---")
    seed_all()
    model = make_model(len(cols), K)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(SEED)
    best = {"f1": -1, "epoch": 0, "state": None}
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        train_epoch(model, opt, crit, Xtr, ytr, a.batch, gen)
        pred = predict(model, Xva)
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": ep, "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep == 1 or ep % 5 == 0:
            print(f"  epoch {ep:3d} | macro-F1 validation = {f1:.4f} | meilleur = {best['f1']:.4f} (epoch {best['epoch']})")
        if ep - best["epoch"] >= a.patience:
            print(f"  arret anticipe a l'epoch {ep} : pas d'amelioration depuis {a.patience} epochs")
            break
    fit_s = time.time() - t0
    model.load_state_dict(best["state"])
    pred = predict(model, Xva)
    m = metrics(yva, pred)
    grp_source = by_group(val, pred, "source")
    grp_culture = by_group(val, pred, "culture")
    name = "mobilenetv2_frozen_head"
    print(f"\n== {name} ==  macro-F1={m['macro_f1']:.4f}  bal.acc={m['balanced_accuracy']:.4f}  "
          f"acc={m['accuracy']:.4f}  (epoch {best['epoch']}, {fit_s:.0f}s)")
    print("  macro-F1 par source :", grp_source, "| par culture :", grp_culture)

    #4. fichiers de sortie
    rep = classification_report(yva, pred, labels=sorted(id2label), target_names=[id2label[i] for i in sorted(id2label)],
                                zero_division=0, output_dict=True)
    rep_path = os.path.join(a.out, f"report_{name}.csv")
    pd.DataFrame(rep).T.to_csv(rep_path)
    cm = confusion_matrix(yva, pred, labels=sorted(id2label), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([id2label[i] for i in sorted(id2label)], rotation=90, fontsize=6)
    ax.set_yticklabels([id2label[i] for i in sorted(id2label)], fontsize=6)
    ax.set_xlabel("classe predite"); ax.set_ylabel("classe reelle")
    ax.set_title(f"{name} - validation (normalisee par classe)")
    fig.colorbar(im); fig.tight_layout()
    cm_path = os.path.join(a.out, f"confusion_{name}.png")
    fig.savefig(cm_path, dpi=110); plt.close(fig)

    os.makedirs(os.path.dirname(a.model_out) or ".", exist_ok=True)
    torch.save({"state_dict": best["state"], "mu": mu, "sd": sd, "id2label": id2label, "seed": SEED,
                "best_epoch": best["epoch"], "embeddings": os.path.basename(a.embeddings)}, a.model_out)

    manifest_sha = hashlib.sha256(open(a.manifest, "rb").read()).hexdigest()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = "inconnu"

    if not a.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(a.mlflow_db)
        mlflow.set_experiment(a.experiment)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": "MobileNetV2 gele + couche lineaire", "backbone_weights": "ImageNet",
                               "lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch,
                               "class_weight": not a.no_class_weight, "best_epoch": best["epoch"], "seed": SEED})
            mlflow.log_metrics(m)
            mlflow.log_metrics({f"f1_source_{ok_name(k)}": v for k, v in grp_source.items()})
            mlflow.log_metrics({f"f1_culture_{ok_name(k)}": v for k, v in grp_culture.items()})
            mlflow.set_tags({"dataset": "AgriLens", "data_version": "split_manifest_v2",
                             "manifest_sha256": manifest_sha, "git_commit": git_commit, "split_evalue": val_name})
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(rep_path)

    # ---------- 5. comparaison avec les baselines ----------
    rows = []
    if os.path.exists(a.baselines):
        base = pd.read_csv(a.baselines)
        for _, r in base.iterrows():
            src = ast.literal_eval(r["f1_par_source"]) if isinstance(r.get("f1_par_source"), str) else {}
            rows.append({"modele": r["modele"], "macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"],
                         "accuracy": r["accuracy"], "f1_PlantVillage": src.get("PlantVillage"),
                         "f1_cacao": src.get("cacao_diseases"), "note": "features a la main (75)"})
    else:
        print(f"\n(Pas de fichier {a.baselines} : tableau limite au nouveau modele.)")
    rows.append({"modele": name, "macro_f1": round(m["macro_f1"], 4), "balanced_accuracy": round(m["balanced_accuracy"], 4),
                 "accuracy": round(m["accuracy"], 4), "f1_PlantVillage": grp_source.get("PlantVillage"),
                 "f1_cacao": grp_source.get("cacao_diseases"), "note": "embeddings MobileNetV2 (1280)"})
    tab = pd.DataFrame(rows)
    tab_path = os.path.join(a.out, "comparison_3_models_val.csv")
    tab.to_csv(tab_path, index=False)
    print("\n=== COMPARAISON (validation) ===")
    print(tab[["modele", "macro_f1", "balanced_accuracy", "accuracy", "f1_PlantVillage", "f1_cacao"]].to_string(index=False))

    print("\nPar classe (F1) - classes du cacao :")
    for lab in [v for v in id2label.values() if v.startswith("cacao")]:
        print(f"  {lab:28s} F1={rep[lab]['f1-score']:.3f}  precision={rep[lab]['precision']:.3f}  "
              f"rappel={rep[lab]['recall']:.3f}  (n={int(rep[lab]['support'])})")
    json.dump({"metrics": m, "by_source": grp_source, "by_culture": grp_culture, "best_epoch": best["epoch"],
               "manifest_sha256": manifest_sha, "git_commit": git_commit, "split_evalue": val_name},
              open(os.path.join(a.out, "mobilenetv2_head_summary.json"), "w"), indent=2, default=str)
    print(f"\nResultats ecrits dans {a.out} ; poids dans {a.model_out}")


if __name__ == "__main__":
    main()
