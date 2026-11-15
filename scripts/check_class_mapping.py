"""
check_class_mapping.py - E4 : verifie, pour chaque classe du manifest, quelle fiche sera affichee.

Lancer depuis la racine du projet :  python scripts/check_class_mapping.py
Attendu : 14 classes -> une fiche chacune (toutes differentes) ; 3 classes saines -> message fixe.
Si une ligne dit "none" ou si deux classes pointent vers la meme fiche, il faut corriger le champ
classe_modele de la fiche concernee (ligne 'classe_modele:' en tete du fichier .md).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agrilens.rag import Retriever  # noqa: E402

r = Retriever(ROOT / "07_rag_genai/corpus")
labels = sorted(pd.read_csv(ROOT / "01_data/metadata/split_manifest_v2.csv")["label"].unique())
seen, problems = {}, 0
for lab in labels:
    f = r.fiche_for_label(lab)
    print(f"{lab:55s} -> {f['kind']:8s} {f['doc_id'] or ''}")
    if f["kind"] == "none":
        problems += 1
    if f["doc_id"]:
        if f["doc_id"] in seen:
            print(f"   ATTENTION : meme fiche que {seen[f['doc_id']]}")
            problems += 1
        seen[f["doc_id"]] = lab
n_f = sum(1 for lab in labels if r.fiche_for_label(lab)["kind"] == "fiche")
n_h = sum(1 for lab in labels if r.fiche_for_label(lab)["kind"] == "healthy")
print(f"\n{len(labels)} classes : {n_f} avec fiche, {n_h} saines, {problems} probleme(s)")
print("Fiches jamais utilisees :", sorted(set(r.docs) - set(seen)) or "aucune")
sys.exit(1 if problems or n_f != 14 or n_h != 3 else 0)
