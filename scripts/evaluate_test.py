"""
evaluate_test.py - E2 : seuil d'abstention (choisi sur la VALIDATION) puis mesure UNIQUE sur le TEST.

Deux modes, volontairement separes :

  1) python scripts\\evaluate_test.py
       Utilise seulement la validation. Calcule la courbe couverture / precision, choisit le seuil
       de confiance et l'ecrit dans 05_deep_learning/results/abstention_threshold.json.
       Le split test n'est PAS lu.

  2) python scripts\\evaluate_test.py --final-test
       Charge le seuil deja fige, mesure une seule fois sur le test, ecrit les resultats et pose un
       verrou (TEST_EVALUATED.lock). Un second lancement est refuse (sauf --force, a eviter :
       re-mesurer le test pour ajuster quelque chose, c'est le contaminer).

Lancer depuis la racine du projet.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_split(a, ckpt, split_wanted):
    """Charge manifeste + embeddings, standardise avec mu/sd du TRAIN (stockes dans le checkpoint)."""
    man = pd.read_csv(a.manifest)
    emb = pd.read_parquet(a.embeddings)
    df = man.merge(emb, on="filepath_rel", how="inner", validate="one_to_one")
    assert len(df) == len(man), f"embeddings manquants : {len(man) - len(df)} images sans embedding"
    names = set(df["split"])
    name = {"validation": "validation" if "validation" in names else "val"}.get(split_wanted, split_wanted)
    sub = df[df["split"] == name].reset_index(drop=True)
    cols = [c for c in emb.columns if c != "filepath_rel"]
    X = (sub[cols].values.astype(np.float32) - ckpt["mu"]) / ckpt["sd"]
    return sub, torch.from_numpy(X.astype(np.float32)), sub["class_id"].values


def probabilities(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).numpy()


def curve(y, proba, grid):
    """Pour chaque seuil : part des images traitees (couverture), precision sur celles-ci, erreurs evitees."""
    pred, conf = proba.argmax(1), proba.max(1)
    wrong_total = int((pred != y).sum())
    rows = []
    for t in grid:
        keep = conf >= t
        n = int(keep.sum())
        acc = float((pred[keep] == y[keep]).mean()) if n else float("nan")
        f1 = float(f1_score(y[keep], pred[keep], labels=np.unique(y[keep]), average="macro",
                            zero_division=0)) if n else float("nan")
        caught = wrong_total - int((pred[keep] != y[keep]).sum())
        rows.append({"seuil": round(float(t), 2), "couverture": n / len(y), "accuracy_acceptees": acc,
                     "macro_f1_acceptees": f1, "images_ecartees": len(y) - n,
                     "erreurs_evitees": caught, "part_erreurs_evitees": caught / wrong_total if wrong_total else 0.0})
    return pd.DataFrame(rows)


def choose_threshold(tab, target_acc, min_cov):
    ok = tab[(tab["accuracy_acceptees"] >= target_acc) & (tab["couverture"] >= min_cov)]
    if len(ok):
        return float(ok.iloc[0]["seuil"]), True          # le plus petit seuil qui atteint l'objectif
    cand = tab[tab["couverture"] >= min_cov].dropna(subset=["accuracy_acceptees"])
    return float(cand.sort_values("accuracy_acceptees").iloc[-1]["seuil"]), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="01_data/metadata/split_manifest_v2.csv")
    ap.add_argument("--embeddings", default="01_data/processed/embeddings_mobilenetv2_v1.parquet")
    ap.add_argument("--model", default="models/head_mobilenetv2_v1.pt")
    ap.add_argument("--out", default="05_deep_learning/results")
    ap.add_argument("--target-acc", type=float, default=0.98,
                    help="precision visee sur les images traitees (choix du seuil, validation)")
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="part minimale d'images que le modele doit traiter seul")
    ap.add_argument("--final-test", action="store_true", help="mesure unique sur le test")
    ap.add_argument("--force", action="store_true", help="ignore le verrou (a eviter)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    thr_path = os.path.join(a.out, "abstention_threshold.json")
    lock_path = os.path.join(a.out, "TEST_EVALUATED.lock")

    ckpt = torch.load(a.model, weights_only=False)
    id2label = {int(k): v for k, v in ckpt["id2label"].items()}
    K, d = len(id2label), len(ckpt["mu"])
    model = torch.nn.Linear(d, K)
    model.load_state_dict(ckpt["state_dict"])
    labels = sorted(id2label)
    names = [id2label[i] for i in labels]
    grid = np.round(np.arange(0.0, 1.0, 0.01), 2)

    # ===================== MODE 1 : choix du seuil sur la validation =====================
    if not a.final_test:
        val, Xv, yv = load_split(a, ckpt, "validation")
        proba = probabilities(model, Xv)
        tab = curve(yv, proba, grid)
        tab.to_csv(os.path.join(a.out, "abstention_curve_val.csv"), index=False)
        thr, reached = choose_threshold(tab, a.target_acc, a.min_coverage)
        row = tab[tab["seuil"] == thr].iloc[0]
        base = tab.iloc[0]
        print(f"Validation : {len(val)} images (le test n'est pas lu)")
        print(f"Sans abstention : accuracy = {base['accuracy_acceptees']:.4f}, macro-F1 = {base['macro_f1_acceptees']:.4f}")
        print(f"\nCompromis (extraits) :")
        print(tab[tab["seuil"].isin([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])][
            ["seuil", "couverture", "accuracy_acceptees", "images_ecartees", "part_erreurs_evitees"]
        ].round(4).to_string(index=False))
        print(f"\nObjectif : precision >= {a.target_acc:.0%} avec couverture >= {a.min_coverage:.0%} "
              f"-> {'ATTEINT' if reached else 'NON ATTEINT (meilleur compromis retenu)'}")
        print(f"SEUIL CHOISI = {thr:.2f} : couverture {row['couverture']:.1%}, precision sur traitees "
              f"{row['accuracy_acceptees']:.4f}, {int(row['images_ecartees'])} images envoyees a un humain, "
              f"{row['part_erreurs_evitees']:.0%} des erreurs evitees")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(tab["couverture"], tab["accuracy_acceptees"], lw=2)
        ax.scatter([row["couverture"]], [row["accuracy_acceptees"]], c="red", zorder=3, label=f"seuil {thr:.2f}")
        ax.set_xlabel("couverture (part d'images traitees par le modele)")
        ax.set_ylabel("precision sur les images traitees")
        ax.set_title("Validation : seuil d'abstention"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "abstention_curve_val.png"), dpi=110); plt.close(fig)
        json.dump({"threshold": thr, "target_accuracy": a.target_acc, "min_coverage": a.min_coverage,
                   "target_reached": reached, "chosen_on": "validation",
                   "validation_coverage": float(row["couverture"]),
                   "validation_accuracy_accepted": float(row["accuracy_acceptees"]),
                   "model_sha256": sha256(a.model), "manifest_sha256": sha256(a.manifest)},
                  open(thr_path, "w"), indent=2)
        print(f"\nSeuil ecrit dans {thr_path}.")
        print("Verifie-le, puis (quand tu es PRET et que tu ne changeras plus rien) : "
              "python scripts\\evaluate_test.py --final-test")
        return

    # ===================== MODE 2 : mesure unique sur le test =====================
    if not os.path.exists(thr_path):
        sys.exit("Seuil absent : lance d'abord le mode 1 (sans --final-test).")
    if os.path.exists(lock_path) and not a.force:
        sys.exit("REFUSE : le test a deja ete mesure (verrou present). Re-mesurer pour ajuster = contaminer le test.")
    thr_info = json.load(open(thr_path))
    if thr_info["model_sha256"] != sha256(a.model):
        sys.exit("REFUSE : le modele a change depuis le choix du seuil. Refais le mode 1.")
    thr = thr_info["threshold"]
    print("=" * 60 + f"\nMESURE UNIQUE SUR LE TEST - modele fige, seuil = {thr:.2f}\n" + "=" * 60)

    test, Xt, yt = load_split(a, ckpt, "test")
    proba = probabilities(model, Xt)
    pred, conf = proba.argmax(1), proba.max(1)
    keep = conf >= thr
    full = {"macro_f1": f1_score(yt, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yt, pred), "accuracy": accuracy_score(yt, pred)}
    acc_kept = float((pred[keep] == yt[keep]).mean())
    f1_kept = float(f1_score(yt[keep], pred[keep], labels=np.unique(yt[keep]), average="macro", zero_division=0))
    wrong = int((pred != yt).sum())
    caught = wrong - int((pred[keep] != yt[keep]).sum())
    by_source = {}
    for g, idx in test.groupby("source").groups.items():
        idx = list(idx)
        by_source[g] = round(f1_score(yt[idx], pred[idx], labels=np.unique(yt[idx]), average="macro",
                                      zero_division=0), 4)

    print(f"Test : {len(test)} images")
    print(f"SANS abstention : macro-F1 = {full['macro_f1']:.4f} | bal.acc = {full['balanced_accuracy']:.4f} "
          f"| accuracy = {full['accuracy']:.4f}")
    print(f"AVEC abstention (seuil {thr:.2f}) : couverture = {keep.mean():.1%} | precision sur traitees = {acc_kept:.4f} "
          f"| macro-F1 traitees = {f1_kept:.4f}")
    print(f"  {int((~keep).sum())} images envoyees a un humain ; {caught}/{wrong} erreurs evitees")
    print(f"macro-F1 par source : {by_source}")
    print(f"Rappel validation : precision visee {thr_info['validation_accuracy_accepted']:.4f}, "
          f"couverture {thr_info['validation_coverage']:.1%}")

    rep = classification_report(yt, pred, labels=labels, target_names=names, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(rep).T
    ab = pd.Series(~keep).groupby(yt).sum()
    rep_df["ecartees"] = [int(ab.get(i, 0)) for i in labels] + [np.nan] * (len(rep_df) - K)
    rep_df.to_csv(os.path.join(a.out, "report_test_mobilenetv2.csv"))
    cm = confusion_matrix(yt, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("classe predite"); ax.set_ylabel("classe reelle")
    ax.set_title("MobileNetV2 gele - TEST (normalisee par classe)")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "confusion_test_mobilenetv2.png"), dpi=110); plt.close(fig)

    json.dump({"split": "test", "n": int(len(test)), "threshold": thr, "metrics_no_abstention": full,
               "coverage": float(keep.mean()), "accuracy_accepted": acc_kept, "macro_f1_accepted": f1_kept,
               "images_sent_to_human": int((~keep).sum()), "errors_total": wrong, "errors_avoided": caught,
               "macro_f1_by_source": by_source, "model_sha256": thr_info["model_sha256"],
               "manifest_sha256": thr_info["manifest_sha256"], "date": time.strftime("%Y-%m-%d %H:%M")},
              open(os.path.join(a.out, "test_summary.json"), "w"), indent=2, default=float)
    open(lock_path, "w").write(f"Test mesure le {time.strftime('%Y-%m-%d %H:%M')} - ne pas relancer.\n")
    print(f"\nResultats ecrits dans {a.out}. Verrou pose : le test ne doit plus etre re-mesure.")


if __name__ == "__main__":
    main()
