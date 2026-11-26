"""Lit la révision exacte, la licence et le code distant de chaque modèle.

Interroge uniquement les métadonnées sur Hugging Face : aucun fichier de
modèle n'est téléchargé et rien n'est écrit dans le projet.
"""

from huggingface_hub import HfApi

MODELES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-multilingual-base",
]


def decrire(api: HfApi, depot: str) -> None:
    """Affiche révision, date, licence et code distant du modèle `depot`."""
    print(f"\n=== {depot}")
    try:
        info = api.model_info(depot)
    except Exception as erreur:  # noqa: BLE001
        print(f"  metadonnees non lues ({type(erreur).__name__})")
        return
    licence = info.card_data.license if info.card_data else None
    code_distant = "custom_code" in (info.tags or [])
    print(f"  revision (sha) : {info.sha}")
    print(f"  derniere modification : {info.last_modified}")
    print(f"  licence declaree : {licence}")
    print(f"  code distant (custom_code) : {code_distant}")


if __name__ == "__main__":
    api = HfApi()
    for modele in MODELES:
        decrire(api, modele)
