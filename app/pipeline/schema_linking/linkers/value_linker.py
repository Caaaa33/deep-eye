from .base import BaseSchemaLinker
from app.dataset import DataItem
from app.llm import LLM
from app.db_utils import map_lower_table_name_to_original_table_name, map_lower_column_name_to_original_column_name
from app.tracing import trace_event
from typing import Dict, List
from collections import defaultdict


class ValueLinker(BaseSchemaLinker):

    def __init__(self, value_distance_threshold: float = 0.05, extractor_max_retry: int | None = None):
        super().__init__(extractor_max_retry=extractor_max_retry)
        self._value_distance_threshold = value_distance_threshold
    
    def link(self, data_item: DataItem, llm: LLM, sampling_budget: int = 1) -> tuple[Dict[str, List[str]], Dict[str, int]]:
        linked_tables_and_columns = defaultdict(list)
        # 记录每列的最小距离，便于事后分析命中阈值的情况
        matched_min_distances = {}
        for table_name, columns in data_item.retrieved_values.items():
            table_name = map_lower_table_name_to_original_table_name(table_name, data_item.database_schema)
            if table_name is None:
                continue
            for column_name, values in columns.items():
                min_distance = min((value["distance"] for value in values), default=None)
                matched_min_distances.setdefault(table_name, {})[column_name] = min_distance
                if any(value["distance"] < self._value_distance_threshold for value in values):
                    column_name = map_lower_column_name_to_original_column_name(table_name, column_name, data_item.database_schema)
                    if column_name is None:
                        continue
                    linked_tables_and_columns[table_name].append(column_name)
        trace_event(
            "schema_link_result",
            linker="value",
            source="linker",
            value_distance_threshold=self._value_distance_threshold,
            tables=dict(linked_tables_and_columns),
            table_count=len(linked_tables_and_columns),
            column_count=sum(len(columns) for columns in linked_tables_and_columns.values()),
            candidate_min_distances=matched_min_distances,
        )
        return dict(linked_tables_and_columns), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
