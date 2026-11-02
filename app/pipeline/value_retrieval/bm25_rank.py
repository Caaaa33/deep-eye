"""
Lightweight BM25 keyword-search retrieval for database value lookup.

Complements the embedding-based value retrieval pipeline (ChromaDB / local
index) with a sparse keyword-matching channel. Used as a fallback in the empty-
result deterministic repair operators when the embedding-based value examples
are insufficient to match a literal.

This module requires no external dependencies — it implements BM25 with
standard Python collections.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


class BM25Ranker:
    """In-memory BM25 scorer for a corpus of string documents."""

    k1: float
    b: float
    epsilon: float
    _corpus: List[str]
    _doc_lengths: List[int]
    _avgdl: float
    _df: Dict[str, int]
    _idf: Dict[str, float]
    _N: int

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._corpus = []
        self._doc_lengths = []
        self._avgdl = 0.0
        self._df = {}
        self._idf = {}
        self._N = 0

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphanumeric tokenization."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def index(self, documents: List[str]) -> None:
        """Build the BM25 index from a list of string documents."""
        self._corpus = list(documents)
        self._N = len(self._corpus)
        if self._N == 0:
            self._doc_lengths = []
            self._avgdl = 0.0
            self._df = {}
            self._idf = {}
            return

        self._doc_lengths = [len(self._tokenize(d)) for d in self._corpus]
        self._avgdl = sum(self._doc_lengths) / self._N
        self._df = {}
        for doc in self._corpus:
            seen = set()
            for token in self._tokenize(doc):
                if token not in seen:
                    self._df[token] = self._df.get(token, 0) + 1
                    seen.add(token)
        self._idf = {
            token: math.log(1 + (self._N - df + 0.5) / (df + 0.5)) + self.epsilon
            for token, df in self._df.items()
        }

    def score(self, query: str) -> List[Tuple[int, float]]:
        """Score all documents against a query string.
        Returns (doc_index, bm25_score) sorted descending.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self._N == 0:
            return []
        scores: List[float] = [0.0] * self._N
        for token in query_tokens:
            if token not in self._df:
                continue
            token_idf = self._idf.get(token, 0.0)
            for idx in range(self._N):
                doc_tokens = self._tokenize(self._corpus[idx])
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                doc_len = self._doc_lengths[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                scores[idx] += token_idf * numerator / denominator
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked if s > 0.0]

    def top_k(self, query: str, k: int = 5) -> List[str]:
        """Return top-k document strings for a query."""
        ranked = self.score(query)
        return [self._corpus[i] for i, _ in ranked[:k]]


def build_bm25_from_value_examples(
    value_examples: Dict[str, List[str]],
) -> Tuple[BM25Ranker, List[Tuple[str, str]]]:
    """Build a BM25 index from value examples keyed by column.

    Returns (ranker, mapping_list) where mapping_list entries are
    (column_key, document_text).
    """
    mapping: List[Tuple[str, str]] = []
    docs: List[str] = []
    for col_key, examples in value_examples.items():
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                docs.append(ex.strip())
                mapping.append((col_key, ex.strip()))
    ranker = BM25Ranker()
    ranker.index(docs)
    return ranker, mapping


def bm25_literal_match(
    literal: str,
    value_examples: Dict[str, List[str]],
    top_k: int = 5,
) -> List[str]:
    """Given a literal and value examples per column, return the top BM25
    matches (best-effort nearest values) as a flat list of strings.

    This is used as a deterministic fallback in empty-repair literal rebind:
    when embedding-retrieved value examples are insufficient to resolve a
    missing literal, BM25 provides a pure keyword-matching signal.
    """
    if not literal or not literal.strip():
        return []
    ranker, mapping = build_bm25_from_value_examples(value_examples)
    if ranker._N == 0:
        return []
    return ranker.top_k(literal.strip(), k=top_k)


def bm25_fuzzy_lookup(
    literal: str,
    candidates: List[str],
    k: int = 3,
) -> List[str]:
    """Score a flat candidate list against a literal and return top-k.

    Useful when the value examples are already known (e.g. from the schema
    service) and we just need the closest matches by keyword overlap.
    """
    ranker = BM25Ranker()
    ranker.index(candidates)
    return ranker.top_k(literal, k=k)