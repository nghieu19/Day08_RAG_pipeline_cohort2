"""
Task 6 - Lexical search with BM25.

This module uses the chunks created by Task 4 (`data/index/chunks.jsonl`) as
the search corpus. BM25 is implemented locally so the task works even when the
optional `rank-bm25` package is not installed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache

from src.task4_chunking_indexing import CHUNKS_PATH, run_pipeline


def _tokenize(text: str) -> list[str]:
    """Simple Unicode tokenization suitable enough for Vietnamese whitespace text."""
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


class SimpleBM25:
    """
    Okapi BM25 scorer.

    BM25 combines term frequency, inverse document frequency, and document
    length normalization. k1 controls term-frequency saturation; b controls how
    strongly document length is normalized.
    """

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.idf = self._compute_idf(tokenized_corpus)

    @staticmethod
    def _compute_idf(tokenized_corpus: list[list[str]]) -> dict[str, float]:
        doc_count = len(tokenized_corpus)
        doc_freq: Counter[str] = Counter()
        for doc in tokenized_corpus:
            doc_freq.update(set(doc))

        return {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for index, freqs in enumerate(self.term_freqs):
            doc_score = 0.0
            doc_len = self.doc_len[index]
            length_norm = self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))

            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue

                idf = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + length_norm
                doc_score += idf * numerator / denominator

            scores.append(doc_score)

        return scores


@lru_cache(maxsize=1)
def load_corpus() -> tuple[dict, ...]:
    """Load chunk corpus from Task 4, building it first if missing."""
    if not CHUNKS_PATH.exists() or CHUNKS_PATH.stat().st_size == 0:
        run_pipeline()

    corpus: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            corpus.append(
                {
                    "content": record.get("content", ""),
                    "metadata": record.get("metadata", {}),
                }
            )

    return tuple(corpus)


# Public corpus variable kept for notebooks/demo code that import it directly.
CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]) -> SimpleBM25:
    """
    Build a BM25 index from a corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    return SimpleBM25(tokenized_corpus)


@lru_cache(maxsize=1)
def _cached_index() -> tuple[tuple[dict, ...], SimpleBM25]:
    corpus = load_corpus()
    return corpus, build_bm25_index(list(corpus))


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by keyword relevance using BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    corpus, bm25 = _cached_index()
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    results: list[dict] = []
    for index in ranked_indices[:top_k]:
        score = float(scores[index])
        if score <= 0:
            continue

        doc = corpus[index]
        results.append(
            {
                "content": doc["content"],
                "score": score,
                "metadata": doc["metadata"],
            }
        )

    return results


if __name__ == "__main__":
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
