import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from app.services import get_execution_service
from app.tracing import trace_event

from .checkers.base import BaseChecker
from .checkers.empty_operators import EMPTY_OPERATORS
from .projection_normalize import normalize_projection, reorder_projection


# ---------------------------------------------------------------------------
# Directed repair prompts (execution-grounded, SIRIUS-style).
#
# Unlike the legacy generic prompt, each prompt is specialized for one failure
# class and feeds the model the exact diagnostic that distinguishes that class
# (engine error text / execution time / empty-result signal).
# ---------------------------------------------------------------------------

_RUNTIME_REPAIR_PROMPT = """
# Task:
You are an SQL database expert tasked with correcting a SQL query that failed at runtime with an execution error from the database engine. Your role is to diagnose the specific error and provide a corrected SQL query.

# Instructions:
1. Review Database Schema:
    - Examine the database schema to understand the database structure.
2. Analyze the Failure:
    - Original Question: Consider what information the query is supposed to retrieve.
    - Hint: Use the provided hints to understand the relationships and conditions relevant to the query.
    - Executed SQL Query: Review the SQL query that failed.
    - Engine Error: Read the raw error message from the database engine carefully. It pinpoints the exact cause (e.g., no such column/table, syntax error, type mismatch, ambiguous reference).
3. Correct the Query:
    - Fix what the engine error points to, keeping the rest of the query logic unchanged unless strictly necessary.
    - Ensure the corrected query is valid for the database schema and answers the question.

[IMPORTANT]
For key phrases mentioned in the question, we have provided the most similar values within the columns (TEXT-TYPE columns) denoted by "Value Examples". This is a critical hint to identify the tables/columns that will be used in the SQL query.

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    Your detailed reasoning for the SQL query revision, including analysis of the engine error and the database schema.
</reasoning>
<result>
    The final revised SQL query that answers the question and can be executed by the database engine directly. Ensure there is no SQL comment and no other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Engine Error:
{DIAGNOSTIC}

Based on the question, table schemas, the previous query, and the engine error, analyze the failure and fix the query. Only output the XML code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""

_TIMEOUT_REPAIR_PROMPT = """
# Task:
You are an SQL database expert tasked with correcting a SQL query that timed out because it was too slow to execute. Your role is to diagnose the likely performance problem and provide a corrected, faster SQL query that still returns the intended result.

# Instructions:
1. Review Database Schema:
    - Examine the database schema to understand the database structure.
2. Analyze the Failure:
    - Original Question: Consider what information the query is supposed to retrieve.
    - Hint: Use the provided hints to understand the relationships and conditions relevant to the query.
    - Executed SQL Query: Review the SQL query that timed out.
    - Timing Detail: Note how long the query ran before it was aborted.
3. Correct the Query:
    - The most common cause is a missing JOIN condition that produces a huge cross product, or an overly expensive subquery/aggregation.
    - Add the missing join conditions or restructure the query to avoid the cross product while preserving the exact intended semantics.

[IMPORTANT]
For key phrases mentioned in the question, we have provided the most similar values within the columns (TEXT-TYPE columns) denoted by "Value Examples". This is a critical hint to identify the tables/columns that will be used in the SQL query.

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    Your detailed reasoning for the SQL query revision, including analysis of the likely performance bottleneck and how the fix resolves it.
</reasoning>
<result>
    The final revised SQL query that answers the question and can be executed by the database engine directly. Ensure there is no SQL comment and no other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Timing Detail:
{DIAGNOSTIC}

Based on the question, table schemas, the previous query, and the timing detail, analyze the bottleneck and fix the query. Only output the XML code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""

_EMPTY_REPAIR_PROMPT = """
# Task:
You are an SQL database expert tasked with correcting a SQL query that executed successfully but returned an EMPTY result (or a result where all values are NULL). Your role is to diagnose why the query produced no useful rows and provide a corrected SQL query that returns the intended non-empty result.

# Instructions:
1. Review Database Schema:
    - Examine the database schema to understand the database structure.
2. Analyze the Failure:
    - Original Question: Consider what information the query is supposed to retrieve.
    - Hint: Use the provided hints to understand the relationships and conditions relevant to the query.
    - Executed SQL Query: Review the SQL query that returned empty.
    - Result Signal: Note that the query returned no rows (or all-null values).
3. Correct the Query:
    - Common causes: an over-restrictive WHERE condition, a wrong JOIN type (e.g. INNER JOIN dropping rows), filtering on the wrong column, a literal that does not match any value, or an aggregate scope error.
    - Re-read the question intent and adjust the conditions so the query returns the intended non-empty result.

[IMPORTANT]
For key phrases mentioned in the question, we have provided the most similar values within the columns (TEXT-TYPE columns) denoted by "Value Examples". This is a critical hint to identify the tables/columns that will be used in the SQL query.

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    Your detailed reasoning for the SQL query revision, including analysis of why the query returned empty and how the fix recovers the intended rows.
</reasoning>
<result>
    The final revised SQL query that answers the question and can be executed by the database engine directly. Ensure there is no SQL comment and no other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Result Signal:
{DIAGNOSTIC}

Based on the question, table schemas, the previous query, and the result signal, analyze why the query returned empty and fix it. Only output the XML code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""


_PROMPT_BY_CLASS = {
    "runtime": _RUNTIME_REPAIR_PROMPT,
    "timeout": _TIMEOUT_REPAIR_PROMPT,
    "empty": _EMPTY_REPAIR_PROMPT,
}


_RECOVERY_PROMPT = """
# Task:
You are an SQL database expert. A pool of candidate SQL queries for a question has been exhausted: every candidate either failed at runtime, timed out, or returned an empty result. Your role is to regenerate ONE correct SQL query from the condensed failure feedback, avoiding the mistakes that caused every candidate to fail.

# Instructions:
1. Review Database Schema:
    - Examine the database schema to understand the database structure.
2. Analyze the Failure Feedback:
    - Read every candidate's failure class and diagnostic to understand what went wrong.
3. Write a New Query:
    - Produce a fresh SQL query that answers the question correctly and avoids the reported mistakes.

[IMPORTANT]
For key phrases mentioned in the question, we have provided the most similar values within the columns (TEXT-TYPE columns) denoted by "Value Examples". This is a critical hint to identify the tables/columns that will be used in the SQL query.

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    Your detailed reasoning for the regenerated SQL query, referencing the failure feedback.
</reasoning>
<result>
    The final SQL query that answers the question and can be executed by the database engine directly. Ensure there is no SQL comment and no other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Failure Feedback:
{FEEDBACK}

Based on the question, table schemas, and the failure feedback, regenerate a correct query. Only output the XML code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""


def classify_execution_result(execution_result) -> str:
    """Map a SQLExecutionResult to a SIRIUS-style failure class.

    Returns one of:
        - "clean":   the query returned a non-empty, non-null result.
        - "runtime": the query raised an execution error.
        - "timeout": the query timed out.
        - "empty":   the query returned an empty or all-null result.
    """
    result_type = execution_result.result_type
    if result_type == "success":
        return "clean"
    if result_type == "timeout":
        return "timeout"
    if result_type in ("empty_result", "all_null_result"):
        return "empty"
    return "runtime"


def _build_diagnostic(failure_class: str, execution_result) -> str:
    if failure_class == "runtime":
        return execution_result.error_message or "Unknown execution error."
    if failure_class == "timeout":
        elapsed = execution_result.execution_time
        elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "unknown"
        message = execution_result.error_message or ""
        return f"The query ran for {elapsed_str} before it was aborted. {message}".strip()
    if failure_class == "empty":
        if execution_result.result_type == "all_null_result":
            return "The SQL query executed successfully but returned a result table where all values are NULL (all-null result)."
        return "The SQL query executed successfully but returned an EMPTY result table (zero rows)."
    return ""


class TypedRepairer(BaseChecker):
    """Execution-grounded repair: execute -> classify -> directed repair.

    This replaces the legacy sequential checkers (which largely regex-match SQL
    text without consulting execution) with a single SIRIUS-style pass:

        1. Execute the candidate SQL.
        2. Classify the result as clean / runtime / timeout / empty.
        3. For non-clean results, send a failure-class-specific prompt that feeds
           the exact diagnostic (engine error / timing / empty signal).
        4. Re-execute every returned candidate and only accept a candidate whose
           result is clean (non-empty, non-null); otherwise keep the original.
    """

    def check_and_revise(
        self,
        sql: str,
        data_item: DataItem,
        llm: LLM,
        sampling_budget: int = 1,
    ) -> Tuple[str, Dict[str, int]]:
        zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        item_id = data_item.get_item_id()

        # Block A: deterministic pre-screening (no execution, no LLM).
        pre_screen_passed = self._pre_screen(sql)
        trace_event(
            "repair_pre_screen",
            item_id=item_id,
            sql=sql,
            passed=pre_screen_passed,
            action="keep_original" if not pre_screen_passed else "continue",
        )
        if not pre_screen_passed:
            return sql, zero_tokens

        # Block B: execute and classify.
        execution_result = get_execution_service().execute(data_item, sql)
        failure_class = classify_execution_result(execution_result)
        trace_event(
            "repair_classify",
            item_id=item_id,
            sql=sql,
            result_type=execution_result.result_type,
            failure_class=failure_class,
            diagnostic=_build_diagnostic(failure_class, execution_result),
            error_message=execution_result.error_message,
            execution_time=execution_result.execution_time,
            row_count=len(execution_result.result_rows) if execution_result.result_rows is not None else None,
        )

        if failure_class == "clean":
            final_sql, tokens = sql, zero_tokens
            trace_event(
                "repair_decision",
                item_id=item_id,
                action="skip_repair_clean",
                final_sql=final_sql,
            )
        else:
            # Block C: failure-aware repair (empty first tries deterministic ops).
            final_sql, tokens = self._repair(
                sql, data_item, llm, sampling_budget, failure_class, execution_result
            )

        # Projection normalization (noise reduction) guarded by re-execution.
        final_sql = self._apply_projection_normalization(final_sql, data_item)

        trace_event(
            "repair_candidate_result",
            item_id=item_id,
            input_sql=sql,
            final_sql=final_sql,
            changed=final_sql != sql,
            failure_class=failure_class,
            token_usage=tokens,
        )
        return final_sql, tokens

    def _pre_screen(self, sql: str) -> bool:
        """Reject obviously malformed SQL before executing it (zero LLM cost)."""
        s = sql.strip()
        leading = re.sub(r"^\s*(--[^\n]*\n|#[^\n]*\n)+", "", s).lstrip()
        if not re.match(
            r"^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP)\b",
            leading,
            re.IGNORECASE,
        ):
            return False
        if leading.upper().startswith("SELECT") and "FROM" not in leading.upper():
            return False
        if leading.count("(") != leading.count(")"):
            return False
        if leading.count("'") % 2 != 0:
            return False
        return True

    def _repair(
        self,
        sql: str,
        data_item: DataItem,
        llm: LLM,
        sampling_budget: int,
        failure_class: str,
        execution_result,
    ) -> Tuple[str, Dict[str, int]]:
        zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        item_id = data_item.get_item_id()

        # empty: try deterministic operators before escalating to the LLM.
        if failure_class == "empty":
            for operator_idx, operator in enumerate(EMPTY_OPERATORS):
                operator_name = getattr(operator, "__name__", repr(operator))
                try:
                    candidate = operator(sql, data_item)
                except Exception as exc:
                    trace_event(
                        "repair_empty_operator",
                        item_id=item_id,
                        operator_index=operator_idx,
                        operator=operator_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        accepted=False,
                    )
                    continue
                if candidate and candidate != sql:
                    result = get_execution_service().execute(data_item, candidate)
                    accepted = result.result_type == "success"
                    trace_event(
                        "repair_empty_operator",
                        item_id=item_id,
                        operator_index=operator_idx,
                        operator=operator_name,
                        candidate_sql=candidate,
                        result_type=result.result_type,
                        error_message=result.error_message,
                        row_count=len(result.result_rows) if result.result_rows is not None else None,
                        accepted=accepted,
                    )
                    if accepted:
                        return candidate, zero_tokens
                else:
                    trace_event(
                        "repair_empty_operator",
                        item_id=item_id,
                        operator_index=operator_idx,
                        operator=operator_name,
                        candidate_sql=candidate,
                        skipped_reason="empty_or_unchanged",
                        accepted=False,
                    )

        diagnostic = _build_diagnostic(failure_class, execution_result)
        prompt_template = _PROMPT_BY_CLASS[failure_class]

        def prompt_format_func(schema_profile: str) -> str:
            return prompt_template.format(
                DATABASE_SCHEMA=schema_profile,
                QUESTION=data_item.question,
                HINT=data_item.evidence,
                QUERY=sql,
                DIAGNOSTIC=diagnostic,
            )

        final_prompt, _ = self._check_and_revise_with_progressive_stripping(
            data_item, llm, prompt_format_func
        )

        trace_event(
            "repair_prompt_built",
            item_id=item_id,
            failure_class=failure_class,
            diagnostic=diagnostic,
            prompt_available=final_prompt is not None,
            prompt_length=len(final_prompt) if final_prompt else 0,
        )

        if final_prompt is None:
            logger.error(
                f"CRITICAL: Even minimal typed-repair prompt for item {data_item.question_id} "
                f"({failure_class}) exceeds token limit. Returning original SQL."
            )
            return sql, zero_tokens

        extractor = self._get_extractor()
        all_sql_candidates, total_token_usage = extractor.extract_with_retry(
            llm=llm,
            messages=[{"role": "user", "content": final_prompt}],
            rule_parser=self._parse_llm_response,
            fix_end_token=llm.llm_config.fix_end_token,
            end_token="</result>",
            n=sampling_budget,
            trace_label=f"sql_revision.typed_repair.{failure_class}",
            trace_extra={"input_sql": sql, "failure_class": failure_class, "diagnostic": diagnostic},
        )

        trace_event(
            "repair_candidates",
            item_id=item_id,
            failure_class=failure_class,
            candidate_count=len(all_sql_candidates),
            candidates=all_sql_candidates,
        )

        selected_sql = self._select_clean_candidate(all_sql_candidates, data_item)
        trace_event(
            "repair_select",
            item_id=item_id,
            failure_class=failure_class,
            candidate_count=len(all_sql_candidates),
            selected_sql=selected_sql,
            fallback_to_original=selected_sql is None,
        )
        if selected_sql:
            return selected_sql, total_token_usage

        logger.warning(
            f"Typed repair ({failure_class}) produced no clean candidate for item "
            f"{data_item.question_id}; keeping original SQL."
        )
        return sql, total_token_usage

    def _apply_projection_normalization(self, sql: str, data_item: DataItem) -> str:
        """Apply projection normalization, keeping it only when still clean."""
        if not sql:
            return sql
        item_id = data_item.get_item_id()
        # 1) Drop numeric-suffix duplicate columns (Email2/Email3)
        normalized = normalize_projection(sql)
        # 2) Reorder columns into canonical bundles (SIRIUS column-bundle ordering)
        normalized = reorder_projection(normalized)
        if normalized == sql:
            trace_event(
                "repair_projection_normalize",
                item_id=item_id,
                input_sql=sql,
                changed=False,
                adopted=False,
            )
            return sql
        result = get_execution_service().execute(data_item, normalized)
        adopted = result.result_type == "success"
        trace_event(
            "repair_projection_normalize",
            item_id=item_id,
            input_sql=sql,
            normalized_sql=normalized,
            changed=True,
            adopted=adopted,
            result_type=result.result_type,
            error_message=result.error_message,
            final_sql=normalized if adopted else sql,
        )
        if adopted:
            return normalized
        return sql

    def recover_if_needed(
        self,
        data_item: DataItem,
        sql_candidates: List[str],
        llm: LLM,
        sampling_budget: int = 1,
    ) -> Tuple[Optional[str], Dict[str, int]]:
        """Regenerate one candidate when no candidate in the pool is clean.

        Returns (recovery_sql, tokens), or (None, zero_tokens) when the pool
        already contains a clean candidate or recovery produced nothing clean.
        """
        zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not sql_candidates:
            return None, zero_tokens

        item_id = data_item.get_item_id()
        execution_service = get_execution_service()
        unresolved: List[Tuple[str, str, str]] = []
        for sql in sql_candidates:
            if not sql:
                continue
            result = execution_service.execute(data_item, sql)
            failure_class = classify_execution_result(result)
            diagnostic = _build_diagnostic(failure_class, result)
            trace_event(
                "recovery_pool_check",
                item_id=item_id,
                sql=sql,
                result_type=result.result_type,
                failure_class=failure_class,
                diagnostic=diagnostic,
            )
            if failure_class == "clean":
                trace_event(
                    "recovery_skipped",
                    item_id=item_id,
                    reason="clean_candidate_in_pool",
                    clean_sql=sql,
                )
                return None, zero_tokens
            unresolved.append((sql, failure_class, diagnostic))

        if not unresolved:
            return None, zero_tokens

        return self._recover(data_item, unresolved, llm, sampling_budget)

    def _recover(
        self,
        data_item: DataItem,
        unresolved: List[Tuple[str, str, str]],
        llm: LLM,
        sampling_budget: int,
    ) -> Tuple[Optional[str], Dict[str, int]]:
        zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        item_id = data_item.get_item_id()
        feedback = self._format_failure_feedback(unresolved)

        trace_event(
            "recovery_start",
            item_id=item_id,
            unresolved_count=len(unresolved),
            unresolved=[{"sql": sql, "failure_class": cls, "diagnostic": diag} for sql, cls, diag in unresolved],
            feedback=feedback,
        )

        def prompt_format_func(schema_profile: str) -> str:
            return _RECOVERY_PROMPT.format(
                DATABASE_SCHEMA=schema_profile,
                QUESTION=data_item.question,
                HINT=data_item.evidence,
                FEEDBACK=feedback,
            )

        final_prompt, _ = self._check_and_revise_with_progressive_stripping(
            data_item, llm, prompt_format_func
        )
        trace_event(
            "recovery_prompt_built",
            item_id=item_id,
            prompt_available=final_prompt is not None,
            prompt_length=len(final_prompt) if final_prompt else 0,
        )
        if final_prompt is None:
            logger.error(
                f"CRITICAL: Even minimal pool-recovery prompt for item "
                f"{data_item.question_id} exceeds token limit. Skipping recovery."
            )
            return None, zero_tokens

        extractor = self._get_extractor()
        all_sql_candidates, total_token_usage = extractor.extract_with_retry(
            llm=llm,
            messages=[{"role": "user", "content": final_prompt}],
            rule_parser=self._parse_llm_response,
            fix_end_token=llm.llm_config.fix_end_token,
            end_token="</result>",
            n=sampling_budget,
            trace_label="sql_revision.pool_recovery",
            trace_extra={"unresolved_count": len(unresolved), "feedback": feedback},
        )

        trace_event(
            "recovery_candidates",
            item_id=item_id,
            candidate_count=len(all_sql_candidates),
            candidates=all_sql_candidates,
        )

        selected_sql = self._select_clean_candidate(all_sql_candidates, data_item)
        trace_event(
            "recovery_select",
            item_id=item_id,
            candidate_count=len(all_sql_candidates),
            selected_sql=selected_sql,
            success=selected_sql is not None,
        )
        if selected_sql:
            return selected_sql, total_token_usage

        logger.warning(
            f"Pool recovery produced no clean candidate for item {data_item.question_id}."
        )
        return None, total_token_usage

    @staticmethod
    def _format_failure_feedback(unresolved: List[Tuple[str, str, str]], limit: int = 5) -> str:
        lines = []
        for sql, failure_class, diagnostic in unresolved[:limit]:
            sql_brief = " ".join(sql.split())[:300]
            lines.append(f"- [{failure_class}] {sql_brief}\n  diagnostic: {diagnostic}")
        return "\n".join(lines)

    def _select_clean_candidate(self, all_sql_candidates: List[str], data_item: DataItem) -> Optional[str]:
        """Pick the majority clean candidate (result_type == 'success'), else None."""
        execution_service = get_execution_service()
        item_id = data_item.get_item_id()
        valid_candidates = []
        for sql_candidate in all_sql_candidates:
            execution_result = execution_service.execute(data_item, sql_candidate)
            is_clean = execution_result.result_type == "success"
            result_hash = None
            if is_clean:
                result_hash = execution_service.hash_result(data_item, execution_result.result_rows)
                valid_candidates.append((sql_candidate, result_hash))
            trace_event(
                "repair_execute_candidate",
                item_id=item_id,
                sql=sql_candidate,
                result_type=execution_result.result_type,
                error_message=execution_result.error_message,
                row_count=len(execution_result.result_rows) if execution_result.result_rows is not None else None,
                result_hash=result_hash,
                counted_as_clean=is_clean,
            )

        if len(valid_candidates) == 0:
            return None

        counter = Counter(result_hash for _, result_hash in valid_candidates)
        selected_sql = max(valid_candidates, key=lambda x: counter[x[1]])[0]
        trace_event(
            "repair_candidate_votes",
            item_id=item_id,
            candidate_count=len(all_sql_candidates),
            clean_count=len(valid_candidates),
            result_hash_votes={str(hash_value): count for hash_value, count in counter.items()},
            selected_sql=selected_sql,
        )
        return selected_sql
