import shutil
from pathlib import Path
import pandas as pd

M = Path("01_data/metadata/split_manifest_v2.csv")
backup = M.with_name("split_manifest_v2_backup_avant_filepath_rel.csv")
if not backup.exists():
    shutil.copy(M, backup)          # sauvegarde de l'original, une seule fois

df = pd.read_csv(M)
rel = (df["filepath"].str.replace("\\", "/", regex=False)
                     .str.replace(r"^(\.\./)+", "", regex=True))
df["filepath_rel"] = rel

missing = [p for p in rel if not Path(p).exists()]
print("Images introuvables :", len(missing))
print("Exemple de chemin   :", rel.iloc[0])
print(df["split"].value_counts())

if missing:
    print(missing[:3])
    raise SystemExit("Rien sauvegardé : des chemins sont faux.")
df.to_csv(M, index=False)
print("OK : colonne filepath_rel ajoutée.")