"""
Task 8 - PageIndex vectorless RAG.

If PAGEINDEX_API_KEY and the SDK are available, this module can be extended to
call the real PageIndex service. For the classroom repo, `pageindex_search`
provides a local vectorless fallback: it scans chunk text structurally and ranks
results by keyword coverage. Returned rows are marked with source="pageindex"
so Task 9 can use it as the fallback path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.task4_chunking_indexing import CHUNKS_PATH, STANDARDIZED_DIR, run_pipeline

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists() or CHUNKS_PATH.stat().st_size == 0:
        run_pipeline()

    chunks: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def upload_documents() -> list[dict]:
    """
    Prepare metadata for documents that would be uploaded to PageIndex.

    The real upload step requires a PageIndex account/API key. Returning local
    metadata keeps the notebook/demo flow inspectable without external state.
    """
    uploaded: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return uploaded

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        uploaded.append(
            {
                "filename": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
                "type": md_file.parent.name,
                "size": md_file.stat().st_size,
            }
        )
    return uploaded


def _vectorless_score(query_tokens: set[str], content: str) -> float:
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    content_set = set(content_tokens)
    coverage = len(query_tokens & content_set) / len(query_tokens)
    phrase_bonus = 0.1 if " ".join(query_tokens) in content.lower() else 0.0
    density = sum(1 for token in content_tokens if token in query_tokens) / len(content_tokens)
    return coverage + min(density * 2, 0.2) + phrase_bonus


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using a local PageIndex-like fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'pageindex'}.
    """
    if top_k <= 0:
        return []

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    results: list[dict] = []
    for chunk in _load_chunks():
        content = chunk.get("content", "")
        score = _vectorless_score(query_tokens, content)
        if score <= 0:
            continue

        results.append(
            {
                "content": content,
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
