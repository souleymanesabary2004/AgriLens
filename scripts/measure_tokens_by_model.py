"""Compte, pour chaque modèle, combien de passages dépassent sa longueur max.

Télécharge uniquement les fichiers du tokenizer (quelques Mo par modèle),
jamais les poids. Lecture seule : rien n'est écrit dans le projet.
"""

import json
import statistics
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agrilens.genai.chunking import chunk_corpus  # noqa: E402

CORPUS = ROOT / "07_rag_genai" / "corpus"

MODELES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-multilingual-base",
]


def lire_limite(depot: str) -> int | None:
    """Lit la longueur max (en tokens) déclarée par le modèle, si elle existe."""
    try:
        chemin = hf_hub_download(repo_id=depot, filename="sentence_bert_config.json")
    except Exception:  # noqa: BLE001
        return None
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier).get("max_seq_length")


def mesurer(depot: str, textes: list[str]) -> None:
    """Affiche la longueur des passages en tokens pour le modèle `depot`."""
    print(f"\n=== {depot}")
    limite = lire_limite(depot)
    try:
        tokenizer = AutoTokenizer.from_pretrained(depot)
    except Exception as erreur:  # noqa: BLE001
        print(f"  tokenizer non chargé ({type(erreur).__name__})")
        return
    longueurs = [len(ids) for ids in tokenizer(textes)["input_ids"]]
    print(f"  limite declaree : {limite if limite else 'inconnue'} tokens")
    print(
        f"  tokens par passage : min {min(longueurs)}, "
        f"moyenne {statistics.mean(longueurs):.0f}, max {max(longueurs)}"
    )
    if limite:
        depassent = sum(1 for n in longueurs if n > limite)
        print(f"  passages tronques : {depassent} sur {len(textes)}")


if __name__ == "__main__":
    chunks = chunk_corpus(CORPUS)
    textes = [chunk.texte for chunk in chunks]
    print(f"{len(chunks)} passages (champ texte), max {max(map(len, textes))} caracteres")
    for modele in MODELES:
        mesurer(modele, textes)
