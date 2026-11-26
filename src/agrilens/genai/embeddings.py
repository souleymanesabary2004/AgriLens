"""Mode d'emploi de chaque modele d'embeddings et fonctions d'encodage.

Un modele d'embeddings a son propre mode d'emploi : prefixe pour les questions,
prefixe pour les passages, revision exacte, besoin de code distant. Ce fichier
est le seul endroit qui les connait. Le reste du code demande seulement
"encode ces questions" ou "encode ces passages".
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Instruction de tache, identique pour Qwen3 et e5-instruct (comparaison equitable).
INSTRUCTION_TACHE = (
    "Given a question about a plant disease or plant health, "
    "retrieve the passages that answer the question"
)


@dataclass(frozen=True)
class ModelSpec:
    """Mode d'emploi d'un modele d'embeddings."""

    name: str
    repo: str
    revision: str
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="minilm",
            repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        ),
        ModelSpec(
            name="bge-m3",
            repo="BAAI/bge-m3",
            revision="5617a9f61b028005a4858fdac845db406aefb181",
        ),
        ModelSpec(
            name="arctic-l-v2",
            repo="Snowflake/snowflake-arctic-embed-l-v2.0",
            revision="ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
            query_prefix="query: ",
        ),
        ModelSpec(
            name="qwen3-0.6b",
            repo="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            query_prefix=f"Instruct: {INSTRUCTION_TACHE}\nQuery:",
        ),
        ModelSpec(
            name="e5-large-instruct",
            repo="intfloat/multilingual-e5-large-instruct",
            revision="274baa43b0e13e37fafa6428dbc7938e62e5c439",
            query_prefix=f"Instruct: {INSTRUCTION_TACHE}\nQuery: ",
        ),
        ModelSpec(
            name="gte-base",
            repo="Alibaba-NLP/gte-multilingual-base",
            revision="6dde5e2ebcac02a5459a3cf310346901fa1dc8ff",
        ),
    )
}


def format_queries(spec: ModelSpec, queries: list[str]) -> list[str]:
    """Colle le prefixe des questions devant chaque question."""
    return [spec.query_prefix + query for query in queries]


def format_documents(spec: ModelSpec, documents: list[str]) -> list[str]:
    """Colle le prefixe des passages devant chaque passage."""
    return [spec.doc_prefix + document for document in documents]


def load_model(spec: ModelSpec) -> "SentenceTransformer":
    """Charge le modele sur le processeur, a la revision epinglee."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        spec.repo,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        device="cpu",
    )


def _encode(model: "SentenceTransformer", textes: list[str], batch_size: int) -> np.ndarray:
    """Encode les textes et normalise toujours les vecteurs (longueur 1)."""
    vecteurs = model.encode(
        textes,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vecteurs, dtype=np.float32)


def encode_queries(
    model: "SentenceTransformer",
    spec: ModelSpec,
    queries: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """Encode des questions avec le prefixe du modele."""
    return _encode(model, format_queries(spec, queries), batch_size)


def encode_documents(
    model: "SentenceTransformer",
    spec: ModelSpec,
    documents: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """Encode des passages avec le prefixe du modele."""
    return _encode(model, format_documents(spec, documents), batch_size)
