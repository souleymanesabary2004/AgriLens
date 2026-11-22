"""Ajoute le champ `label` (nom de classe du manifeste) a l'en-tete des fiches du corpus.

Usage (depuis la racine du projet) :
    python scripts/migrate_corpus_labels.py          # simulation : n'ecrit rien
    python scripts/migrate_corpus_labels.py --apply  # ecrit les fichiers

Le script est idempotent : une fiche qui a deja un champ `label` est ignoree.
Le champ `classe_modele` n'est pas touche (la v1 du RAG le lit).
"""

import argparse
from pathlib import Path

import pandas as pd

CORPUS = Path("07_rag_genai/corpus")
MANIFEST = Path("01_data/metadata/split_manifest_v2.csv")

# identifiant de la fiche -> classe exacte du manifeste de split
LABELS = {
    "cacao_foreur_cabosses": "cacao__pod_borer",
    "cacao_pourriture_brune": "cacao__black_pod_rot",
    "mais_cercosporiose": "corn__cercospora_leaf_spot_gray_leaf_spot",
    "mais_helminthosporiose": "corn__northern_leaf_blight",
    "mais_rouille_commune": "corn__common_rust",
    "tomate_acariens": "tomato__spider_mites_two_spotted_spider_mite",
    "tomate_alternariose": "tomato__early_blight",
    "tomate_cladosporiose": "tomato__leaf_mold",
    "tomate_corynesporiose": "tomato__target_spot",
    "tomate_mildiou": "tomato__late_blight",
    "tomate_mosaique": "tomato__tomato_mosaic_virus",
    "tomate_septoriose": "tomato__septoria_leaf_spot",
    "tomate_tache_bacterienne": "tomato__bacterial_spot",
    "tomate_tylcv": "tomato__tomato_yellow_leaf_curl_virus",
}


def add_label(text, label):
    """Retourne (nouveau_texte, modifie). Conserve le type de saut de ligne du fichier."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    if not lines or lines[0].strip() != "---":
        raise ValueError("en-tete absent")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("fin d'en-tete introuvable")
    header = lines[1:end]
    if any(ln.startswith("label:") for ln in header):
        return text, False
    idx = next((i for i, ln in enumerate(header) if ln.startswith("classe_modele:")), None)
    if idx is None:
        raise ValueError("champ classe_modele introuvable")
    lines.insert(1 + idx + 1, f"label: {label}")
    return newline.join(lines), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="ecrit reellement les fichiers")
    args = ap.parse_args()

    known = set(pd.read_csv(MANIFEST)["label"].unique())
    unknown = {k: v for k, v in LABELS.items() if v not in known}
    if unknown:
        raise SystemExit(f"Labels absents du manifeste : {unknown}")

    files = sorted(CORPUS.glob("*.md"))
    undefined = [f.stem for f in files if f.stem not in LABELS]
    if undefined:
        raise SystemExit(f"Fiches sans label defini : {undefined}")

    to_change, already = 0, 0
    for f in files:
        with f.open(encoding="utf-8", newline="") as fh:
            text = fh.read()
        new_text, changed = add_label(text, LABELS[f.stem])
        if not changed:
            already += 1
            print(f"deja present : {f.stem}")
            continue
        to_change += 1
        mode = "ECRIT" if args.apply else "SIMULATION"
        print(f"[{mode}] {f.stem} -> label: {LABELS[f.stem]}")
        if args.apply:
            with f.open("w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)

    print(f"{to_change} fiche(s) a modifier, {already} deja a jour.")
    if not args.apply:
        print("Simulation : rien n'a ete ecrit.")


if __name__ == "__main__":
    main()
