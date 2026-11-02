from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import DatasetConfig
from app.config.config import PreliminarySQLConfig
from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from app.pipeline.sql_generation.generators import DCGenerator, SkeletonGenerator
from app.services import (
    configure_execution_service,
    configure_schema_service,
    get_execution_service,
    reset_execution_service,
    reset_schema_service,
)
from app.tracing import submit_with_trace, trace_event


TOKEN_USAGE_ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@dataclass
class PreliminarySQLResult:
    sql: Optional[str]
    candidates: List[str]
    executable_candidates: int
    non_empty_candidates: int
    consistency_score: float
    token_usage: Dict[str, int]


class PreliminarySQLGenerator:
    def __init__(
        self,
        config: PreliminarySQLConfig,
        dataset_config: DatasetConfig,
        *,
        extractor_max_retry: int,
        parallelism: int,
    ) -> None:
        if config.llm is None:
            raise ValueError("[few_shot_index.preliminary_sql.llm] is required")

        self._config = config
        self._dataset_config = dataset_config
        configure_schema_service(max_value_example_length=dataset_config.max_value_example_length)
        configure_execution_service(
            default_timeout=dataset_config.sql_execution_timeout,
            bigquery_credential_path=dataset_config.bigquery_credential_path,
            snowflake_credential_path=dataset_config.snowflake_credential_path,
        )
        self._llm = LLM(config.llm)
        self._parallelism = max(1, parallelism)
        self._executor = ThreadPoolExecutor(max_workers=self._parallelism)
        logger.info(f"Preliminary SQL parallelism: {self._parallelism}")
        self._execution_service = get_execution_service()
        self._dc_generator = DCGenerator(extractor_max_retry=extractor_max_retry)
        self._skeleton_generator = SkeletonGenerator(extractor_max_retry=extractor_max_retry)

    def generate(self, data_item: DataItem) -> PreliminarySQLResult:
        item_id = data_item.get_item_id()
        schema = data_item.database_schema if isinstance(data_item.database_schema, dict) else {}
        schema_tables = schema.get("tables", {}) if isinstance(schema.get("tables", {}), dict) else {}
        schema_column_count = sum(
            len(table.get("columns", {}) or {})
            for table in schema_tables.values()
            if isinstance(table, dict)
        )
        trace_event(
            "preliminary_start",
            item_id=item_id,
            dc_sampling_budget=self._config.dc_sampling_budget,
            skeleton_sampling_budget=self._config.skeleton_sampling_budget,
            schema_table_count=len(schema_tables),
            schema_column_count=schema_column_count,
        )

        generation_item = data_item.model_copy(deep=True)
        generation_item.database_schema_after_schema_linking = data_item.database_schema

        generation_tasks = {
            "dc": submit_with_trace(
                self._executor,
                self._dc_generator.generate,
                generation_item,
                self._llm,
                self._config.dc_sampling_budget,
            ),
            "skeleton": submit_with_trace(
                self._executor,
                self._skeleton_generator.generate,
                generation_item,
                self._llm,
                self._config.skeleton_sampling_budget,
            ),
        }

        candidates: List[str] = []
        total_token_usage = dict(TOKEN_USAGE_ZERO)
        for name, future in generation_tasks.items():
            error = None
            try:
                generated_candidates, token_usage = future.result()
            except Exception as exc:
                logger.error(f"Error in preliminary {name} generation for item {data_item.get_item_id()}: {exc}")
                traceback.print_exc()
                error = str(exc)
                generated_candidates, token_usage = [], TOKEN_USAGE_ZERO

            trace_event(
                "preliminary_generator_result",
                item_id=item_id,
                name=name,
                candidate_count=len(generated_candidates or []),
                candidates=generated_candidates or [],
                token_usage=token_usage,
                error=error,
            )
            for key in total_token_usage:
                total_token_usage[key] += token_usage.get(key, 0)
            candidates.extend(generated_candidates or [])

        raw_candidate_count = len(candidates)
        candidates = _deduplicate_sql_candidates(candidates)
        trace_event(
            "preliminary_dedup",
            item_id=item_id,
            before=raw_candidate_count,
            after=len(candidates),
            candidates=candidates,
        )
        selected_sql, executable_count, non_empty_count, consistency_score = select_preliminary_sql_by_consistency(
            data_item,
            candidates,
            execution_service=self._execution_service,
        )
        trace_event(
            "preliminary_select",
            item_id=item_id,
            sql=selected_sql,
            candidate_count=len(candidates),
            executable_candidates=executable_count,
            non_empty_candidates=non_empty_count,
            consistency_score=consistency_score,
            token_usage=total_token_usage,
        )
        return PreliminarySQLResult(
            sql=selected_sql,
            candidates=candidates,
            executable_candidates=executable_count,
            non_empty_candidates=non_empty_count,
            consistency_score=consistency_score,
            token_usage=total_token_usage,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        reset_execution_service()
        reset_schema_service()
        self._llm = None
        self._execution_service = None


def select_preliminary_sql_by_consistency(
    data_item: DataItem,
    sql_candidates: List[str],
    *,
    execution_service: Any = None,
) -> Tuple[Optional[str], int, int, float]:
    if not sql_candidates:
        trace_event(
            "preliminary_consistency",
            item_id=data_item.get_item_id(),
            executable_candidates=0,
            non_empty_candidates=0,
            votes={},
            selected_sql=None,
            selected_votes=0,
            consistency_score=0.0,
        )
        return None, 0, 0, 0.0

    execution_service = execution_service or get_execution_service()
    valid_candidates: List[Tuple[str, Any]] = []
    fallback_candidates: List[Tuple[str, Any]] = []
    result_by_sql = {}

    for index, sql_candidate in enumerate(sql_candidates):
        execution_result = execution_service.execute(data_item, sql_candidate)
        result_by_sql[sql_candidate] = execution_result
        result_rows = execution_result.result_rows
        trace_event(
            "preliminary_candidate_exec",
            item_id=data_item.get_item_id(),
            index=index,
            sql=sql_candidate,
            result_type=getattr(execution_result, "result_type", None),
            row_count=len(result_rows) if result_rows is not None else None,
            execution_time=getattr(execution_result, "execution_time", None),
            error=getattr(execution_result, "error_message", None),
        )
        if execution_result.result_rows is None:
            continue
        result_hash = execution_service.hash_result(data_item, execution_result.result_rows)
        fallback_candidates.append((sql_candidate, result_hash))
        if len(execution_result.result_rows) > 0:
            valid_candidates.append((sql_candidate, result_hash))

    executable_count = len(fallback_candidates)
    non_empty_count = len(valid_candidates)
    if not valid_candidates and fallback_candidates:
        valid_candidates = fallback_candidates

    if not valid_candidates:
        trace_event(
            "preliminary_consistency",
            item_id=data_item.get_item_id(),
            executable_candidates=executable_count,
            non_empty_candidates=non_empty_count,
            votes={},
            selected_sql=None,
            selected_votes=0,
            consistency_score=0.0,
        )
        return None, executable_count, non_empty_count, 0.0

    consistency_counter = Counter(result_hash for _, result_hash in valid_candidates)
    denominator = len(valid_candidates)
    deduplicated_candidates = []
    seen_result_hashes = set()
    for sql_candidate, result_hash in valid_candidates:
        if result_hash in seen_result_hashes:
            continue
        execution_result = result_by_sql[sql_candidate]
        execution_time = execution_result.execution_time if execution_result.execution_time is not None else np.inf
        consistency_score = consistency_counter[result_hash] / denominator
        deduplicated_candidates.append((sql_candidate, result_hash, consistency_score, execution_time))
        seen_result_hashes.add(result_hash)

    selected = sorted(
        deduplicated_candidates,
        key=lambda item: (item[2], -item[3]),
        reverse=True,
    )[0]
    trace_event(
        "preliminary_consistency",
        item_id=data_item.get_item_id(),
        executable_candidates=executable_count,
        non_empty_candidates=non_empty_count,
        votes=dict(consistency_counter),
        selected_sql=selected[0],
        selected_result_hash=selected[1],
        selected_votes=consistency_counter[selected[1]],
        consistency_score=selected[2],
        denominator=denominator,
        deduplicated_candidate_count=len(deduplicated_candidates),
    )
    return selected[0], executable_count, non_empty_count, selected[2]


def _deduplicate_sql_candidates(sql_candidates: List[str]) -> List[str]:
    deduplicated_candidates = []
    seen = set()
    for sql in sql_candidates:
        if not sql:
            continue
        normalized_sql = sql.strip()
        if not normalized_sql or normalized_sql in seen:
            continue
        deduplicated_candidates.append(normalized_sql)
        seen.add(normalized_sql)
    return deduplicated_candidates
