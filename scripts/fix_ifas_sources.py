"""Corrige 4 lignes de sources UF/IFAS dans le corpus (titres officiels + domaine edis).
Usage : python scripts/fix_ifas_sources.py          (simulation)
        python scripts/fix_ifas_sources.py --apply  (ecrit les fichiers)
Idempotent : relancer ne change rien."""
import sys
from pathlib import Path

CORPUS = Path("07_rag_genai/corpus")
FIXES = {
    "tomate_corynesporiose.md": (
        "- UF/IFAS, PP351, Target Spot of Tomato : https://ask.ifas.ufl.edu/publication/PP351/pdf",
        "- UF/IFAS, PP351, Target Spot of Tomato in Florida : https://edis.ifas.ufl.edu/publication/PP351/pdf",
    ),
    "tomate_mosaique.md": (
        "- UF/IFAS, PP360, Tobamoviruses of tomato : https://ask.ifas.ufl.edu/publication/PP360/pdf",
        "- UF/IFAS, PP360, Tomato Brown Rugose Fruit Virus (ToBRFV): A Potential Threat for Tomato Production in Florida : https://edis.ifas.ufl.edu/publication/PP360/pdf",
    ),
    "tomate_tache_bacterienne.md": (
        "- UF/IFAS, PP353, Bacterial Spot of Tomato : https://ask.ifas.ufl.edu/publication/PP353/pdf",
        "- UF/IFAS, PP353, Integrated Management of Bacterial Spot on Tomato in Florida : https://edis.ifas.ufl.edu/publication/PP353/pdf",
    ),
    "tomate_tylcv.md": (
        "- UF/IFAS, IN1430, Tomato yellow leaf curl virus : https://ask.ifas.ufl.edu/publication/IN1430",
        "- UF/IFAS, ENY2111 (ancien IN1430), Managing Bemisia tabaci MEAM1 and Tomato yellow leaf curl virus in Florida Tomato : https://ask.ifas.ufl.edu/publication/IN1430",
    ),
}


def main():
    apply = "--apply" in sys.argv
    for name, (old, new) in FIXES.items():
        path = CORPUS / name
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        if new in text:
            print(f"[DEJA FAIT] {name}")
            continue
        if old not in text:
            print(f"[ERREUR] ligne attendue introuvable dans {name}")
            sys.exit(1)
        print(f"[{'APPLIQUE' if apply else 'SIMULATION'}] {name}")
        if apply:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text.replace(old, new))


if __name__ == "__main__":
    main()
