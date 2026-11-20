"""
scripts/train_baselines.py - S2.4 : baselines classiques sur les features manuelles (Régression logistique + Random Forest)
et test du « raccourci source ».

Règles du protocole S2 respectées :
  - le split TEST n'est JAMAIS chargé (le script refuse de le lire) ;
  - le scaler est appris sur train uniquement (Pipeline) ;
  - métrique principale : macro-F1 ; en plus balanced accuracy, accuracy, résultats par source et par culture ;
  - chaque run est tracé dans MLflow avec data_version, manifest_sha256 et git_commit.

Lancer depuis la racine du projet :
    python scripts/train_baselines.py
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2026
p = argparse.ArgumentParser()
p.add_argument("--manifest", default="01_data/metadata/split_manifest_v2.csv")
p.add_argument("--features", default="01_data/processed/features_v1.parquet")
p.add_argument("--out", default="04_ml_baselines/results")
p.add_argument("--mlflow-db", default="sqlite:///10_mlops/mlflow.db")
p.add_argument("--experiment", default="agrilens-s2-final")
p.add_argument("--no-mlflow", action="store_true")
p.add_argument("--fast", action="store_true", help="test rapide : moins d'arbres")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------- 1. données : train + validation seulement ----------
man = pd.read_csv(a.manifest)
feat = pd.read_parquet(a.features)
df = man.merge(feat, on="filepath_rel", how="inner", validate="one_to_one")
assert len(df) == len(man), f"features manquantes : {len(man) - len(df)} images du manifest sans features"
val_name = "validation" if "validation" in set(df["split"]) else "val"
FEATURES = [c for c in feat.columns if c != "filepath_rel"]
BANNED = {"width", "height", "file_size", "largeur", "hauteur", "taille"}
assert not (BANNED & {c.lower() for c in FEATURES}), "feature technique interdite (raccourci source)"
train = df[df["split"] == "train"]
val = df[df["split"] == val_name]          # le split 'test' n'est jamais utilisé ici
print(f"train={len(train)}  val={len(val)}  features={len(FEATURES)}  (split test ignoré)")
Xtr, Xva = train[FEATURES].values, val[FEATURES].values
ytr, yva = train["class_id"].values, val["class_id"].values
id2label = df.drop_duplicates("class_id").set_index("class_id")["label"].to_dict()

manifest_sha = hashlib.sha256(open(a.manifest, "rb").read()).hexdigest()
try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_commit = "inconnu"

use_mlflow = not a.no_mlflow
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri(a.mlflow_db)
    mlflow.set_experiment(a.experiment)

def metrics(y, pred):
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred)}

def by_group(frame, pred, col):
    out = {}
    for g, idx in frame.reset_index(drop=True).groupby(col).groups.items():
        idx = list(idx)
        yt = frame["class_id"].values[idx]
        # labels = classes reellement presentes dans le groupe (evite les classes fantomes predites a tort)
        out[g] = round(f1_score(yt, pred[idx], labels=np.unique(yt), average="macro", zero_division=0), 4)
    return out

def plot_cm(y, pred, path, title):
    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title); ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def _ok(txt):
    """MLflow n'accepte que lettres, chiffres, _ - . espace / dans les noms de metriques."""
    import re
    return re.sub(r"[^A-Za-z0-9_\-\. /]", "", str(txt))


def run(name, model, params):
    t = time.time(); model.fit(Xtr, ytr); fit_s = time.time() - t
    pred = model.predict(Xva)
    m = metrics(yva, pred)
    grp_source = by_group(val, pred, "source")
    grp_culture = by_group(val, pred, "culture")
    cm_path = os.path.join(a.out, f"confusion_{name}.png")
    plot_cm(yva, pred, cm_path, f"{name} - validation (normalisée par classe)")
    rep = classification_report(yva, pred, target_names=[id2label[i] for i in sorted(id2label)],
                                zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(os.path.join(a.out, f"report_{name}.csv"))
    print(f"\n== {name} ==  macro-F1={m['macro_f1']:.4f}  bal.acc={m['balanced_accuracy']:.4f}  acc={m['accuracy']:.4f}  ({fit_s:.0f}s)")
    print("  macro-F1 par source :", grp_source, "| par culture :", grp_culture)
    if use_mlflow:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**params, "seed": SEED, "n_features": len(FEATURES)})
            mlflow.log_metrics(m)
            mlflow.log_metrics({f"f1_source_{_ok(k)}": v for k, v in grp_source.items()})
            mlflow.log_metrics({f"f1_culture_{_ok(k)}": v for k, v in grp_culture.items()})
            mlflow.set_tags({"dataset": "AgriLens", "data_version": "split_manifest_v2",
                             "manifest_sha256": manifest_sha, "git_commit": git_commit, "split_evalue": val_name})
            mlflow.log_artifact(cm_path); mlflow.log_artifact(os.path.join(a.out, f"report_{name}.csv"))
    return {"modele": name, **{k: round(v, 4) for k, v in m.items()}, "f1_par_source": grp_source,
            "f1_par_culture": grp_culture, "temps_s": round(fit_s, 1)}, pred

# ---------- 2. modèles ----------
results = []
lr = Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
r, _ = run("logreg_features", lr, {"model": "LogisticRegression", "class_weight": "balanced", "scaler": "StandardScaler"})
results.append(r)

n_trees = 60 if a.fast else 300
best = None
for md, msl in [(None, 1), (20, 2), (None, 5)][:2 if a.fast else 3]:
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=md, min_samples_leaf=msl,
                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    r, pred = run(f"rf_depth{md}_leaf{msl}", rf, {"model": "RandomForest", "n_estimators": n_trees,
                                                  "max_depth": md, "min_samples_leaf": msl,
                                                  "class_weight": "balanced_subsample"})
    results.append(r)
    if best is None or r["macro_f1"] > best[0]["macro_f1"]:
        best = (r, rf)

# importances des 15 meilleures features du meilleur RF : vérifier qu'aucune n'est un artefact de source
imp = pd.Series(best[1].feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.head(15).to_csv(os.path.join(a.out, "top15_importances_rf.csv"))
print("\nTop 15 features (meilleur RF) :\n", imp.head(15).round(4).to_string())

# ---------- 3. test du raccourci source ----------
# Un modèle peut-il deviner la SOURCE (PlantVillage / cacao) à partir des features ? Si oui (~100 %), l'accuracy globale est gonflée.
ysrc_tr = (train["source"] == train["source"].unique()[0]).astype(int).values
ysrc_va = (val["source"] == train["source"].unique()[0]).astype(int).values
rf_src = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1).fit(Xtr, ysrc_tr)
src_acc = accuracy_score(ysrc_va, rf_src.predict(Xva))
print(f"\nRACCOURCI SOURCE : accuracy de prédiction de la source = {src_acc:.4f}")
if use_mlflow:
    with mlflow.start_run(run_name="source_shortcut_test"):
        mlflow.log_metric("source_accuracy", src_acc)
        mlflow.set_tags({"dataset": "AgriLens", "manifest_sha256": manifest_sha, "git_commit": git_commit})

# ---------- 4. tableau comparatif ----------
tab = pd.DataFrame(results)
tab.to_csv(os.path.join(a.out, "comparison_baselines_val.csv"), index=False)
json.dump({"source_shortcut_accuracy": src_acc, "manifest_sha256": manifest_sha, "git_commit": git_commit,
           "split_evalue": val_name, "resultats": results}, open(os.path.join(a.out, "baselines_summary.json"), "w"),
          indent=2, ensure_ascii=False, default=str)
print("\nRésumé écrit dans", a.out)
