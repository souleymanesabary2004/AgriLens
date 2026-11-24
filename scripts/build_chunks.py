"""Construit les chunks du corpus, les ecrit en JSONL et affiche des statistiques.

Usage : python scripts/build_chunks.py [--max-chars 800]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from agrilens.genai.chunking import ChunkingConfig, chunk_corpus

CORPUS = Path("07_rag_genai/corpus")
SORTIE = Path("07_rag_genai/index/chunks.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=800)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    chunks = chunk_corpus(CORPUS, ChunkingConfig(max_chars=args.max_chars))
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with SORTIE.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tailles = sorted(len(c.contenu) for c in chunks)
    print(f"Chunks : {len(chunks)} (fiches : {len({c.doc_id for c in chunks})})")
    print(
        f"Longueur du contenu min / mediane / max : "
        f"{tailles[0]} / {tailles[len(tailles) // 2]} / {tailles[-1]} caracteres"
    )
    multi = [c for c in chunks if c.n_parts > 1]
    print(f"Chunks issus d'une section coupee : {len(multi)}")
    for c in multi:
        print(f"  {c.chunk_id} ({len(c.contenu)} caracteres)")
    par_label = Counter(c.label for c in chunks)
    print(f"Labels couverts : {len(par_label)} sur 17")
    print(f"Ecrit : {SORTIE}")


if __name__ == "__main__":
    main()
