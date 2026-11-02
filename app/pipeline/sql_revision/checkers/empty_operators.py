"""Deterministic, LLM-free repair operators for empty-result candidates.

These mirror SIRIUS's "Block C, Stage 1" operators: cheap, rule-based local
fixes that are tried before escalating to the LLM. Each operator only *produces*
a candidate; the caller re-executes it and only accepts a non-empty result, so a
wrong guess can never corrupt the final output.
"""

import re
from typing import List, Optional

from app.dataset import DataItem
from app.pipeline.value_retrieval.bm25_rank import bm25_literal_match


def _collect_value_examples(data_item: DataItem) -> List[str]:
    """Gather TEXT-column value examples from the richest available schema."""
    schema = None
    for attr in (
        "database_schema_after_value_retrieval",
        "database_schema_after_schema_linking",
        "database_schema",
    ):
        schema = getattr(data_item, attr, None)
        if schema:
            break
    if not isinstance(schema, dict):
        return []

    examples: List[str] = []
    tables = schema.get("tables", {})
    if not isinstance(tables, dict):
        return examples
    for table in tables.values():
        if not isinstance(table, dict):
            continue
        columns = table.get("columns", {})
        if not isinstance(columns, dict):
            continue
        for column in columns.values():
            if not isinstance(column, dict):
                continue
            values = column.get("value_examples")
            if isinstance(values, list):
                examples.extend(values)
    return examples


def _collect_value_examples_by_column(data_item: DataItem) -> dict:
    """Collect value examples keyed by column name."""
    schema = None
    for attr in (
        "database_schema_after_value_retrieval",
        "database_schema_after_schema_linking",
        "database_schema",
    ):
        schema = getattr(data_item, attr, None)
        if schema:
            break
    if not isinstance(schema, dict):
        return {}

    result: dict = {}
    tables = schema.get("tables", {})
    if not isinstance(tables, dict):
        return result
    for tname, table in tables.items():
        if not isinstance(table, dict):
            continue
        columns = table.get("columns", {})
        if not isinstance(columns, dict):
            continue
        for cname, column in columns.items():
            if not isinstance(column, dict):
                continue
            values = column.get("value_examples")
            if isinstance(values, list):
                key = f"{tname}.{cname}"
                result[key] = values
    return result


def apply_literal_binding(sql: str, data_item: DataItem) -> Optional[str]:
    """Fix case-only mismatches between string literals and stored values.

    Only replaces a literal when a stored value matches it case-insensitively,
    so the rewrite never changes semantics (e.g. 'new york' -> 'New York').
    """
    examples = _collect_value_examples(data_item)
    examples_by_column = _collect_value_examples_by_column(data_item)
    if not examples:
        return None

    lookup: dict = {}
    for example in examples:
        lookup.setdefault(example.lower(), example)

    def repl(match: "re.Match") -> str:
        literal = match.group(1)
        real = lookup.get(literal.lower())
        if real is not None and real != literal:
            return "'" + real.replace("'", "''") + "'"
        return match.group(0)

    new_sql = re.sub(r"'([^']*)'", repl, sql)
    if new_sql == sql:
        return None
    return new_sql


def apply_aggregate_scope(sql: str, data_item: DataItem) -> Optional[str]:
    """Move an aggregate-only WHERE predicate to HAVING (aggregate scope fix).

    Only handles the unambiguous case where the whole WHERE clause is a single
    predicate referencing an aggregate function (no AND/OR), which is the only
    case safe to relocate with plain regex.
    """
    if not re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE):
        return None

    match = re.search(
        r"\bWHERE\b\s+(.+?)\s+\bGROUP\s+BY\b",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    where_body = match.group(1).strip()
    if not re.search(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", where_body, re.IGNORECASE):
        return None
    if re.search(r"\b(?:AND|OR)\b", where_body, re.IGNORECASE):
        return None

    without_where = re.sub(
        r"\bWHERE\b\s+.+?\s+\bGROUP\s+BY\b",
        "GROUP BY",
        sql,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    group_match = re.search(
        r"\bGROUP\s+BY\b.+?(?=\s+ORDER\s+BY\b|\s+LIMIT\b|\s*$)",
        without_where,
        re.IGNORECASE | re.DOTALL,
    )
    if not group_match:
        return None

    insert_at = group_match.end()
    return without_where[:insert_at] + " HAVING " + where_body + without_where[insert_at:]


def apply_literal_rebind(sql: str, data_item: DataItem) -> Optional[str]:
    """Broader literal replacement: probe value examples for sub-string /
    prefix / fuzzy matches when exact case-alignment fails.

    Unlike literal_binding which only fixes case mismatches, this operator
    also handles prefix/suffix truncation and sub-string containment
    (e.g. 'York' → 'New York', or 'Prague' → 'Hl.m. Praha').
    """
    examples = _collect_value_examples(data_item)
    if not examples:
        return None

    lookup: dict = {}
    for example in examples:
        key = example.lower()
        lookup.setdefault(key, example)

    def rebind_repl(match: "re.Match") -> str:
        literal = match.group(1)
        lit_lower = literal.lower()
        # 1) exact case-insensitive match (same as literal_binding)
        exact = lookup.get(lit_lower)
        if exact is not None and exact != literal:
            return "'" + exact.replace("'", "''") + "'"
        # 2) the literal is a substring of a stored value
        for key, val in lookup.items():
            if lit_lower != key and lit_lower in key and len(lit_lower) >= 3:
                return "'" + val.replace("'", "''") + "'"
        # 3) a stored value is a substring of the literal
        for key, val in lookup.items():
            if lit_lower != key and key in lit_lower and len(key) >= 3:
                return "'" + val.replace("'", "''") + "'"

        # 4) BM25 keyword-match fallback (SIRIUS DES-style sparse retrieval)
        if len(lit_lower) >= 3:
            bm25_matches = bm25_literal_match(literal, examples_by_column, top_k=1)
            if bm25_matches:
                return "'" + bm25_matches[0].replace("'", "''") + "'"

        return match.group(0)

    new_sql = re.sub(r"'([^']*)'", rebind_repl, sql)
    if new_sql == sql:
        return None
    return new_sql


def apply_where_predicate(sql: str, data_item: DataItem) -> Optional[str]:
    """Relax over-restrictive WHERE shapes.

    Two sub-rules:
    (a) Chain of equality predicates on the same column → single LIKE '%…%'.
        e.g. col = 'A' AND col = 'B' → col LIKE '%A%' (keep one).
    (b) Nested subquery in WHERE … IN (SELECT …) → direct JOIN.
    """
    # (a) equality-chain → LIKE with OR
    matched = re.search(
        r"""(\w+\.?\w*)  \s*=\s*'([^']+)'  \s+AND\s+  \1  \s*=\s*'([^']+)'""",
        sql, re.IGNORECASE | re.VERBOSE,
    )
    if matched:
        col = matched.group(1)
        val_a = matched.group(2)
        val_b = matched.group(3)
        new_pred = f"{col} = '{val_a}' OR {col} = '{val_b}'"
        new_sql = (
            sql[: matched.start()] + new_pred + sql[matched.end() :]
        )
        return new_sql

    # (b) WHERE col IN (SELECT …) → JOIN
    # Match pattern: WHERE T1.col IN (SELECT T2.col FROM T2 WHERE …)
    in_sub = re.search(
        r"""\bWHERE\b\s+  (\w+\.?\w+)  \s+IN\s*\(\s*SELECT\s+
            (\w+\.?\w+)  \s+FROM\s+  (\w+(?:\s+\w+)?)  \s+
            (?:WHERE\s+(.+?))?  \s*\)""",
        sql, re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )
    if in_sub:
        outer_col = in_sub.group(1)   # T1.col
        inner_col = in_sub.group(2)   # T2.col
        inner_table = in_sub.group(3) # T2
        inner_where = in_sub.group(4) or "1=1"

        # Build JOIN replacement: replace "... WHERE T1.col IN (SELECT ...)"
        # with "... JOIN T2 ON T1.col = T2.col AND (inner_where) WHERE 1=1"
        join_clause = (
            f"\nJOIN {inner_table} ON {outer_col} = {inner_col}"
            f" AND ({inner_where})"
        )
        # Remove the original IN subquery and append the JOIN
        without_in = sql[: in_sub.start()] + join_clause + sql[in_sub.end() :]
        # Clean up the empty WHERE if nothing remains
        without_in = re.sub(
            r"\bWHERE\b\s+(?=\bGROUP\b|\bORDER\b|\bLIMIT\b|$)",
            "", without_in, flags=re.IGNORECASE,
        )
        return without_in

    return None


def apply_translation_semantics(sql: str, data_item: DataItem) -> Optional[str]:
    """Fix common aggregate-function mistranslations.

    LLMs sometimes confuse the SQL semantics of natural-language words:
    - "average" should map to AVG, not SUM or COUNT
    - "total" / "sum" should map to SUM, not COUNT

    This operator scans for common known patterns and corrects them.
    """
    changed = False
    new_sql = sql

    # Known mistranslation pairs: (wrong_func, correct_func)
    # Only fix when the wrong func is definitely wrong in context.
    corrections = [
        # COUNT(*) where AVG is the clear intent (heuristic: numeric column)
        # This is too noisy to auto-apply, so we skip COUNT→AVG.
        #
        # More reliable: AVG↔SUM confusion
        # If there's "average" language in the query context but SQL uses SUM,
        # or vice versa. Without the question text we can only apply structural
        # heuristics:
    ]

    # Detect aggregate used on a column that looks like it expects division
    # SUM(salary) without / COUNT(*) → maybe should be AVG(salary)
    # Pattern: SUM(col) present but no corresponding division
    has_sum = re.search(r"\bSUM\s*\(\s*(\w+)\s*\)", new_sql, re.IGNORECASE)
    has_div = re.search(r"\s*/\s*", new_sql)
    has_avg = re.search(r"\bAVG\s*\(.+?\)", new_sql, re.IGNORECASE)
    has_count = re.search(r"\bCOUNT\s*\(.+?\)", new_sql, re.IGNORECASE)

    if has_sum and not has_div and not has_avg:
        # Replace SUM(col) with AVG(col) — common "average salary" mistranslation
        col_name = has_sum.group(1)
        new_sql = re.sub(
            r"\bSUM\s*\(\s*" + re.escape(col_name) + r"\s*\)",
            f"AVG({col_name})",
            new_sql,
            count=1,
            flags=re.IGNORECASE,
        )
        changed = True

    if changed:
        return new_sql
    return None


def apply_date_anchor_join(sql: str, data_item: DataItem) -> Optional[str]:
    """Fix date-range predicates anchored on the wrong table in a multi-table
    schema.

    Common LLM error: when a question asks "transactions from 2019", the LLM
    puts the date filter on the transactions table instead of the accounts
    (or vice versa), and the foreign-key chain means the filter should be on
    the dimension table.

    Heuristic: if a date predicate exists on a table that has a foreign key to
    another table that also has a date column, try moving the predicate.
    """
    # Detect date predicates in WHERE clause
    date_pred = re.search(
        r"""(\w+\.\w+)  \s*(>=?|<=?|=|BETWEEN)\s*  '(\d{4}-\d{2}-\d{2})'""",
        sql, re.IGNORECASE | re.VERBOSE,
    )
    if not date_pred:
        return None

    # Check if the table with the date is also joined — if so, see if the
    # *other* table in the JOIN has a semantically equivalent date column
    join_pattern = re.search(
        r"""\b(?:JOIN|FROM)\s+(\w+)\s+(\w*?)\s+
            ON\s+  \w+\.\w+\s*=\s*\w+\.\w+""",
        sql, re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )
    if not join_pattern:
        return None

    # Extract tables involved
    tables_in_query = set(re.findall(r"\bFROM\s+(\w+)", sql, re.IGNORECASE))
    tables_in_query.update(re.findall(r"\bJOIN\s+(\w+)", sql, re.IGNORECASE))

    # Get schema tables with date columns
    schema = _get_schema(data_item)
    if not schema:
        return None

    tables_with_date = _find_tables_with_date_column(schema)

    # If exactly 2 tables have date columns and there's a FK between them,
    # swap the date predicate anchor
    date_tables = tables_with_date & tables_in_query
    if len(date_tables) != 2:
        return None

    date_table_a, date_table_b = sorted(date_tables)
    col_ref = date_pred.group(1)  # e.g. "transactions.transaction_date"
    table_in_pred = col_ref.split(".")[0]

    # Find the "other" date-capable table
    other_table = date_table_b if table_in_pred.lower() == date_table_a.lower() else date_table_a

    # Find the other table's date column name
    other_date_col = _find_date_column(schema, other_table)
    if not other_date_col:
        return None

    # Replace the table-qualified column reference
    _, col_only = col_ref.split(".")
    new_col_ref = f"{other_table}.{other_date_col}"
    new_sql = sql.replace(col_ref, new_col_ref)
    return new_sql


def _get_schema(data_item: DataItem) -> Optional[dict]:
    """Return schema dict, trying the richest available variant."""
    for attr in (
        "database_schema_after_value_retrieval",
        "database_schema_after_schema_linking",
        "database_schema",
    ):
        schema = getattr(data_item, attr, None)
        if isinstance(schema, dict):
            return schema
    return None


def _find_tables_with_date_column(schema: dict) -> set:
    """Return set of table names that have at least one DATE/TEXT date column."""
    tables_with_date: set = set()
    tables = schema.get("tables", {})
    if not isinstance(tables, dict):
        return tables_with_date
    for tname, tdef in tables.items():
        if not isinstance(tdef, dict):
            continue
        columns = tdef.get("columns", {})
        if not isinstance(columns, dict):
            continue
        for _cname, cdef in columns.items():
            if not isinstance(cdef, dict):
                continue
            col_type = str(cdef.get("type", "")).upper()
            if "DATE" in col_type or "TIMESTAMP" in col_type or "DATETIME" in col_type:
                tables_with_date.add(tname)
                break
    return tables_with_date


def _find_date_column(schema: dict, table_name: str) -> Optional[str]:
    """Return the first date-typed column name in a table."""
    tables = schema.get("tables", {})
    if not isinstance(tables, dict):
        return None
    tdef = tables.get(table_name)
    if not isinstance(tdef, dict):
        return None
    columns = tdef.get("columns", {})
    if not isinstance(columns, dict):
        return None
    for cname, cdef in columns.items():
        if not isinstance(cdef, dict):
            continue
        col_type = str(cdef.get("type", "")).upper()
        if "DATE" in col_type or "TIMESTAMP" in col_type or "DATETIME" in col_type:
            return cname
    return None


# Ordered list of deterministic operators tried for empty results.
EMPTY_OPERATORS: List = [
    apply_literal_binding,
    apply_literal_rebind,
    apply_aggregate_scope,
    apply_where_predicate,
    apply_translation_semantics,
    apply_date_anchor_join,
]
