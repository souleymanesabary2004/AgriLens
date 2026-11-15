"""
scripts/build_features.py - S2.3 : calcule les 75 features de toutes les images et les sauvegarde en Parquet.
Lancer depuis la racine du projet :  python scripts/build_features.py
(test rapide sur 200 images :        python scripts/build_features.py --n 200)
"""
import argparse, os, time
import pandas as pd
from joblib import Parallel, delayed
from agrilens.features import extract_features

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=0, help="limiter à n images (test)")
a = p.parse_args()

df = pd.read_csv("01_data/metadata/split_manifest_v2.csv")
if a.n:
    df = df.sample(a.n, random_state=2026)

def one(rel):
    return extract_features(rel)

t = time.time()
rows = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1))(delayed(one)(r) for r in df["filepath_rel"])
feat = pd.DataFrame(rows)
feat.insert(0, "filepath_rel", df["filepath_rel"].values)
print(f"{len(feat)} images, {feat.shape[1]-1} features, {time.time()-t:.0f} s, NaN = {int(feat.isna().sum().sum())}")

out = "01_data/processed/features_v1_test.parquet" if a.n else "01_data/processed/features_v1.parquet"
os.makedirs("01_data/processed", exist_ok=True)
feat.to_parquet(out, index=False)
print("écrit :", out)