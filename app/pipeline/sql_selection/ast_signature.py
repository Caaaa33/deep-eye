"""
SIRIUS-style deterministic SQL structure-signature extraction
(Stage 2 structural support reranker).

Generates a normalized "structure signature" from SQL text using only regex
transformations — no external SQL parser dependency. The signature strips
literals, aliases, and noise while preserving the query skeleton (tables,
JOIN types, aggregate/HAVING structures, subquery depth, etc.).

Used by the selection stage to break ties when pairwise LLM comparison
cannot confidently separate top candidates.
"""

import re
from typing import List, Tuple


def extract_structure_signature(sql: str) -> str:
    """Extract a normalized query-structure signature from raw SQL.

    The signature preserves:
      - Top-level clause structure (SELECT / FROM / WHERE / GROUP BY / HAVING / ORDER BY)
      - Table identifiers (stripped of aliases)
      - JOIN type keywords (INNER / LEFT / RIGHT / CROSS)
      - Subquery boundaries (marked as @@)
      - Aggregate function names
    It removes:
      - Literals (string, numeric)
      - Column aliases (AS name)
      - Table aliases (after table name in FROM / JOIN)
      - Comments
      - Whitespace noise
    """
    cleaned = sql

    # 1) Remove single-line and block comments
    cleaned = re.sub(r'--[^\n]*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

    # 2) Remove string literals
    cleaned = re.sub(r"'[^']*'", "'...'", cleaned)

    # 3) Remove numeric literals (integers and decimals)
    cleaned = re.sub(r'\b\d+\.?\d*\b', '0', cleaned)

    # 4) Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 5) Mark subqueries with @@, preserving brackets
    # Replace innermost (SELECT ...) with @@
    # Iteratively replace innermost subqueries
    while True:
        cleaned, subs = _collapse_innermost_subquery(cleaned)
        if subs == 0:
            break

    # 6) Remove column aliases: "column AS alias" → "column"
    cleaned = re.sub(r'\bAS\s+\w+', '', cleaned, flags=re.IGNORECASE)

    # 7) Remove table aliases in FROM / JOIN:
    #    "FROM table t" → "FROM table"
    #    "JOIN table t ON" → "JOIN table ON"
    #    Pattern: keyword + word + single_word + (ON|WHERE|GROUP|ORDER|LIMIT|$|JOIN)
    cleaned = re.sub(
        r'(\b(?:FROM|JOIN)\s+)(\w+)\s+(\w+)\s+(ON\b|WHERE\b|GROUP\b|ORDER\b|LIMIT\b|JOIN\b|$)',
        r'\1\2 \4',
        cleaned,
        flags=re.IGNORECASE,
    )

    # 8) Collapse whitespace again
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 9) Normalize keyword casing to uppercase
    for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
               'LIMIT', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
               'CROSS JOIN', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
               'DISTINCT', 'BETWEEN', 'IN', 'EXISTS', 'NOT', 'AND', 'OR']:
        cleaned = re.sub(r'\b' + kw + r'\b', kw, cleaned, flags=re.IGNORECASE)

    return cleaned


def _collapse_innermost_subquery(sql: str) -> Tuple[str, int]:
    """Replace innermost (SELECT ...) subquery with @@ marker.
    Returns (transformed_sql, replacement_count).
    """
    # Match "( SELECT ... )" where the content has no internal sub-select
    pattern = re.compile(
        r'\(\s*(SELECT\s+(?:(?!SELECT\s).)*?)\s*\)',
        re.IGNORECASE | re.DOTALL,
    )
    count = 0
    result = sql
    while True:
        match = pattern.search(result)
        if not match:
            break
        result = result[:match.start()] + '@@' + result[match.end():]
        count += 1
    return result, count


def compute_cross_source_support(
    sqls: List[str],
    sources: List[str],
) -> Tuple[List[int], List[str]]:
    """SIRIUS cross-source support (`srcgrp`) for each candidate.

    A candidate's support is the number of *distinct generation sources*
    (dc / skeleton / icl / recovery ...) that produced candidates sharing
    its structure signature. Structure consensus backed by independent
    generators is stronger evidence than consensus from a single route.

    Returns (supports, signatures).
    """
    signatures = [extract_structure_signature(s) for s in sqls]
    grouped_sources: dict = {}
    for idx, sig in enumerate(signatures):
        bucket = grouped_sources.setdefault(sig, set())
        label = sources[idx] if idx < len(sources) and sources[idx] else "unknown"
        bucket.add(label)
    supports = [len(grouped_sources[sig]) for sig in signatures]
    return supports, signatures


def structural_support_select(
    sqls: List[str],
    merged_scores: List[float],
    sources: List[str],
    default_idx: int,
    tie_tol: float = 1e-9,
) -> Tuple[int, dict]:
    """Stage 2: break an exact top-score tie using cross-source support.

    Only candidates tied with the best merged score are comparable; among
    them the one with strictly more distinct-source structural support wins.
    A tie on support keeps ``default_idx`` (SIRIUS: override only when
    strictly stronger).

    Returns (selected_idx, detail).
    """
    detail = {
        "triggered": False,
        "default_idx": int(default_idx),
        "selected_idx": int(default_idx),
        "overridden": False,
        "tied_indices": [],
        "supports": [],
    }
    if len(sqls) <= 1 or not merged_scores:
        return int(default_idx), detail

    detail["triggered"] = True
    best_score = max(merged_scores)
    tied_indices = [
        idx for idx, score in enumerate(merged_scores)
        if abs(score - best_score) <= tie_tol
    ]
    supports, signatures = compute_cross_source_support(sqls, sources)
    detail["tied_indices"] = tied_indices
    detail["supports"] = supports
    detail["signatures"] = signatures

    selected_idx = int(default_idx)
    best_support = supports[int(default_idx)]
    for idx in tied_indices:
        if idx == int(default_idx):
            continue
        # strictly stronger only -> the first challenger that beats the
        # incumbent wins; later equal-support challengers do not override.
        if supports[idx] > best_support:
            selected_idx = idx
            best_support = supports[idx]

    detail["selected_idx"] = selected_idx
    detail["overridden"] = bool(selected_idx != int(default_idx))
    return selected_idx, detail