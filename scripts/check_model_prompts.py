"""Lit les prompts (préfixes) déclarés par chaque modèle d'embeddings.

Ne télécharge que le petit fichier de configuration de chaque modèle,
jamais les poids. Lecture seule : rien n'est écrit dans le projet.
"""

import json

from huggingface_hub import hf_hub_download

MODELES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-multilingual-base",
]

FICHIER = "config_sentence_transformers.json"


def lire_prompts(depot: str) -> None:
    """Affiche les prompts déclarés par le modèle `depot`, s'il en déclare."""
    print(f"\n=== {depot}")
    try:
        chemin = hf_hub_download(repo_id=depot, filename=FICHIER)
    except Exception as erreur:  # noqa: BLE001
        print(f"  pas de fichier {FICHIER} ({type(erreur).__name__})")
        return
    with open(chemin, encoding="utf-8") as fichier:
        config = json.load(fichier)
    prompts = json.dumps(config.get("prompts"), ensure_ascii=True)
    print(f"  prompts : {prompts}")
    print(f"  prompt par défaut : {config.get('default_prompt_name')}")


if __name__ == "__main__":
    for modele in MODELES:
        lire_prompts(modele)
