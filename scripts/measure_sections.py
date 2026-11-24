"""P1.2a : mesure la longueur (en caracteres) de chaque section des fiches. Ne modifie rien."""
import re
from pathlib import Path

CORPUS = Path("07_rag_genai/corpus")
LIMITE = 800


def sections(path):
    texte = path.read_text(encoding="utf-8")
    corps = "---".join(texte.split("---")[2:])
    morceaux = re.split(r"^## ", corps, flags=re.M)[1:]
    for m in morceaux:
        titre, _, contenu = m.partition("\n")
        yield titre.strip(), len(contenu.strip())


def main():
    lignes = []
    for f in sorted(CORPUS.glob("*.md")):
        for titre, n in sections(f):
            lignes.append((f.stem, titre, n))
    contenu = [x for x in lignes if not x[1].startswith("Sources")]
    longs = [x for x in contenu if x[2] > LIMITE]
    tailles = [x[2] for x in contenu]
    print(f"Sections de contenu (hors Sources) : {len(contenu)}")
    print(f"Longueur min / moyenne / max : {min(tailles)} / {sum(tailles) // len(tailles)} / {max(tailles)} caracteres")
    print(f"Sections de plus de {LIMITE} caracteres : {len(longs)}")
    for fiche, titre, n in sorted(longs, key=lambda x: -x[2]):
        print(f"  {n:5d}  {fiche} : {titre}")


if __name__ == "__main__":
    main()
