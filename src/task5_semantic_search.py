"""
Task 5 - Semantic search over the local vector index from Task 4.

The Task 4 index is stored at data/index/vector_index.jsonl. Each line contains
one chunk, its metadata, and a normalized local-hashing embedding. This module
embeds the query with the same function and ranks chunks by cosine similarity.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

from src.task4_chunking_indexing import (
    VECTOR_INDEX_PATH,
    _hash_embedding,
    run_pipeline,
)


@lru_cache(maxsize=1)
def _load_vector_index() -> tuple[dict, ...]:
    """Load the local JSONL vector index, building it first if needed."""
    if not VECTOR_INDEX_PATH.exists() or VECTOR_INDEX_PATH.stat().st_size == 0:
        run_pipeline()

    records: list[dict] = []
    with VECTOR_INDEX_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return tuple(records)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity; works even if vectors are not pre-normalized."""
    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks using vector similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0:
        return []

    query = (query or "").strip()
    if not query:
        return []

    query_embedding = _hash_embedding(query)
    results: list[dict] = []

    for record in _load_vector_index():
        score = _cosine_similarity(query_embedding, record.get("embedding", []))
        results.append(
            {
                "content": record.get("content", ""),
                "score": float(score),
                "metadata": record.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
