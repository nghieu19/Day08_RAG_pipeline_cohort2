"""
Task 9 - Complete retrieval pipeline.

Pipeline:
1. Run semantic search and lexical BM25 search.
2. Merge both ranked lists with Reciprocal Rank Fusion.
3. Rerank merged candidates.
4. Fall back to PageIndex-style vectorless retrieval when hybrid confidence is
   below the threshold.
"""

from __future__ import annotations

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _mark_source(results: list[dict], source: str) -> list[dict]:
    marked: list[dict] = []
    for result in results:
        item = result.copy()
        item["metadata"] = dict(result.get("metadata", {}))
        item["source"] = source
        marked.append(item)
    return marked


def _is_useful_content(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False

    tokens = [token for token in text.replace("/", " ").replace("-", " ").split() if token.strip()]
    if len(tokens) < 8:
        return False

    lower = text.lower()
    url_count = lower.count("http://") + lower.count("https://")
    if url_count and len(tokens) < 35:
        return False

    table_marks = text.count("|") + text.count("---")
    if table_marks > len(tokens) / 2:
        return False

    alphabetic_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    return len(alphabetic_tokens) >= 6


def _filter_useful(results: list[dict]) -> list[dict]:
    return [result for result in results if _is_useful_content(result.get("content", ""))]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieve relevant chunks with hybrid search and PageIndex fallback.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'hybrid'|'pageindex'}.
    """
    if top_k <= 0:
        return []

    query = (query or "").strip()
    if not query:
        return []

    candidate_k = max(top_k * 3, 10)
    dense_results = _filter_useful(semantic_search(query, top_k=candidate_k * 2))
    sparse_results = _filter_useful(lexical_search(query, top_k=candidate_k * 2))

    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    merged = _filter_useful(merged)
    merged = _mark_source(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _mark_source(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        for index, result in enumerate(retrieve(query, top_k=3), 1):
            print(f"{index}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
