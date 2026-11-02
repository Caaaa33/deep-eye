"""Deterministic projection normalization (SIRIUS-style noise reduction).

Drops clearly redundant numeric-suffix duplicate columns (e.g. Email2 / Email3)
when the base column (Email) is already projected. This is a conservative,
text-only rewrite intended to reduce consensus noise for the downstream
selection stage; the caller re-executes and only keeps the rewrite when it still
returns a clean (non-empty) result.
"""

import re
from typing import List, Optional


def _split_select_items(select_body: str) -> List[str]:
    """Split a SELECT list on top-level commas (ignoring commas inside parens)."""
    items: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in select_body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        items.append("".join(current).strip())
    return items


def _extract_column_name(item: str) -> Optional[str]:
    """Return the bare column name of a simple projection item, else None.

    Only simple column references (``col`` or ``table.col``, optionally with an
    ``AS alias``) are recognized; function calls and expressions return None so
    they are never touched.
    """
    item = item.strip()
    alias_match = re.search(r"\s+AS\s+(\w+)\s*$", item, re.IGNORECASE)
    if alias_match:
        item = item[: alias_match.start()].strip()

    match = re.fullmatch(r"(?:`?\w+`?\.)?`?(\w+)`?", item)
    if not match:
        return None
    return match.group(1)


def normalize_projection(sql: str) -> str:
    """Drop numeric-suffix duplicate columns when the base column is present."""
    match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return sql

    select_body = match.group(1)
    items = _split_select_items(select_body)
    if len(items) <= 1:
        return sql

    names = [_extract_column_name(item) for item in items]
    base_names = {name for name in names if name}

    numeric_variants = set()
    for name in names:
        if not name:
            continue
        stripped = re.sub(r"\d+$", "", name)
        if stripped != name and stripped in base_names:
            numeric_variants.add(name)

    if not numeric_variants:
        return sql

    kept = [item for item, name in zip(items, names) if name not in numeric_variants]
    return sql[: match.start(1)] + " " + ", ".join(kept) + " " + sql[match.end(1):]


# ── Column-bundle ordering (SIRIUS-style) ──────────────────────────
# Sort SELECT items into a human-friendly canonical order to reduce
# result-hash variance across candidates that differ only in column order.

_COLUMN_BUNDLE_PRIORITY: dict = {
    # Highest: identifiers / keys
    "id": 0, "key": 1, "code": 1, "uuid": 1, "guid": 1,
    # Names / labels
    "name": 10, "label": 10, "title": 10, "type": 11,
    # Dates
    "date": 20, "year": 21, "month": 22, "day": 23,
    "time": 24, "timestamp": 24,
    # Measures / amounts
    "amount": 30, "price": 30, "cost": 30, "value": 30,
    "revenue": 31, "sales": 31, "salary": 31,
    "count": 32, "total": 32, "sum_": 32, "avg": 32,
    "score": 33, "rate": 33, "ratio": 33, "percentage": 33,
    "number": 34, "quantity": 34, "qty": 34,
    # Descriptions / text
    "description": 40, "desc": 40, "comment": 40, "note": 40,
    "details": 41, "info": 41, "text": 41,
    # Address / location
    "address": 50, "city": 51, "state": 52, "province": 52,
    "country": 53, "zip": 54, "postal": 54,
    # Phone / email
    "phone": 60, "email": 61, "url": 62, "website": 62,
    # Status / flags
    "status": 70, "flag": 71, "active": 71, "enabled": 72,
    "deleted": 73, "removed": 73, "is_": 71,
}
_DEFAULT_PRIORITY = 100


def _column_bundle_key(item: str) -> int:
    """Return a deterministic sort key for a SELECT projection item.

    Items are ranked by semantic category: id/keys < names < dates < measures
    < descriptions < location < contact < status < unknown.
    """
    name = _extract_column_name(item)
    if not name:
        return _DEFAULT_PRIORITY

    name_lower = name.lower()
    # Exact match
    if name_lower in _COLUMN_BUNDLE_PRIORITY:
        return _COLUMN_BUNDLE_PRIORITY[name_lower]
    # Prefix match (e.g. "date_created" → date, "sum_sales" → sum_)
    for prefix, prio in _COLUMN_BUNDLE_PRIORITY.items():
        if name_lower.startswith(prefix) and not prefix.endswith("_"):
            return prio
    return _DEFAULT_PRIORITY


def reorder_projection(sql: str) -> str:
    """Reorder SELECT items into a canonical column-bundle order.

    Only touches the projection when it is a simple comma-separated list of
    column references; expressions and sub-queries are left intact.
    """
    match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return sql

    select_body = match.group(1)
    items = _split_select_items(select_body)
    if len(items) <= 1:
        return sql

    # Only reorder if every item is a recognisable simple column reference
    if any(_extract_column_name(it) is None for it in items):
        return sql

    sorted_items = sorted(items, key=_column_bundle_key)
    # If already sorted, don't change (minimise churn)
    if sorted_items == items:
        return sql

    return sql[: match.start(1)] + " " + ", ".join(sorted_items) + " " + sql[match.end(1):]
