from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.llm import LLM
from app.db_utils import filter_used_database_schema
from .linkers import DirectLinker, ReversedLinker, ValueLinker
from .utils import merge_schema_linking_results
from app.pipeline.validation import validate_pipeline_step
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from app.logger import logger
from app.progress import log_progress, should_checkpoint
import time
import traceback
from pathlib import Path
from app.services import ArtifactStore, STAGE_ARTIFACT_FIELDS, configure_schema_service, load_stage_dataset, reset_schema_service
from app.tracing import configure_tracing, submit_with_trace, trace_event, trace_scope

class SchemaLinkingRunner:
    
    _llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    _inner_thread_pool_executor: ThreadPoolExecutor = None
    
    _direct_linker: DirectLinker = None
    _reversed_linker: ReversedLinker = None
    _value_linker: ValueLinker = None
    _artifact_store: ArtifactStore = None
    _extractor_max_retry: int = 3
    _stage_config = None
    _input_save_path: str = ""
    _few_shot_examples_path: str | None = None
    _dataset_config = None
    _parallelism: int = 16
    _progress_log_interval: int = 50
    _checkpoint_interval: int = 20
    
    def __init__(
        self,
        stage_config,
        dataset_config,
        input_save_path: str,
        few_shot_examples_path: str | None,
        extractor_max_retry: int,
        parallelism: int,
        progress_log_interval: int,
        checkpoint_interval: int,
    ):
        self._stage_config = stage_config
        self._dataset_config = dataset_config
        self._input_save_path = input_save_path
        self._few_shot_examples_path = few_shot_examples_path
        self._extractor_max_retry = extractor_max_retry
        self._parallelism = max(1, parallelism)
        self._progress_log_interval = max(1, progress_log_interval)
        self._checkpoint_interval = max(1, checkpoint_interval)
        self._artifact_store = ArtifactStore(
            self._stage_config.save_path,
            "schema_linking",
            STAGE_ARTIFACT_FIELDS["schema_linking"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=self._input_save_path,
            artifact_store=self._artifact_store,
            stage_name="schema_linking",
        )
        logger.info(f"Initialized schema linking dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        self._llm = LLM(self._stage_config.llm)
        logger.info(f"Schema linking parallelism: {self._parallelism}")
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._direct_linker = DirectLinker(extractor_max_retry=self._extractor_max_retry)
        self._reversed_linker = ReversedLinker(
            few_shot_examples_path=self._few_shot_examples_path,
            extractor_max_retry=self._extractor_max_retry,
        )
        self._value_linker = ValueLinker(
            value_distance_threshold=self._stage_config.value_distance_threshold,
            extractor_max_retry=self._extractor_max_retry,
        )

        # 初始化执行追踪，事件落到 snapshot 同级的 trace 目录
        configure_tracing(Path(self._stage_config.save_path).parent / "trace", stage="schema_linking")

    @classmethod
    def from_config(cls, app_config=None) -> "SchemaLinkingRunner":
        if app_config is None:
            from app.config import get_config

            app_config = get_config()
        input_save_path = app_config.few_shot_index_config.prepared_save_path
        if not Path(input_save_path).exists():
            input_save_path = app_config.value_retrieval_config.save_path
        return cls(
            stage_config=app_config.schema_linking_config,
            dataset_config=app_config.dataset_config,
            input_save_path=input_save_path,
            few_shot_examples_path=app_config.sql_generation_config.icl_few_shot_examples_path,
            extractor_max_retry=app_config.llm_extractor_config.max_retry,
            parallelism=app_config.run_config.parallelism,
            progress_log_interval=app_config.run_config.progress_log_interval,
            checkpoint_interval=app_config.run_config.checkpoint_interval,
        )
    
    def _link_tables_and_columns(self, data_item: DataItem) -> None:
        start_time = time.time()
        
        # Track token usage for this specific data item
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 记录输入 schema 规模与 linker 配置
        input_schema = data_item.database_schema_after_value_retrieval or {}
        input_tables = input_schema.get("tables", {})
        trace_event(
            "schema_link_start",
            question_id=data_item.question_id,
            question=data_item.question,
            evidence=data_item.evidence,
            db_id=data_item.database_id,
            input_table_count=len(input_tables),
            input_column_count=sum(len(table.get("columns", {})) for table in input_tables.values()),
            input_tables=list(input_tables.keys()),
            linkers=["direct", "reversed", "value"],
            direct_sampling_budget=self._stage_config.direct_linking_sampling_budget,
            reversed_sampling_budget=self._stage_config.reversed_linking_sampling_budget,
            value_distance_threshold=self._stage_config.value_distance_threshold,
        )
        
        # Parallelize different linking methods within a single data item
        linker_start_time = time.time()
        linker_tasks = {
            "direct": submit_with_trace(self._inner_thread_pool_executor, self._direct_linker.link, data_item, self._llm, self._stage_config.direct_linking_sampling_budget),
            "reversed": submit_with_trace(self._inner_thread_pool_executor, self._reversed_linker.link, data_item, self._llm, self._stage_config.reversed_linking_sampling_budget),
            "value": submit_with_trace(self._inner_thread_pool_executor, self._value_linker.link, data_item, self._llm)
        }
        
        results = {}
        for name, future in linker_tasks.items():
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Error in {name} linking for item {data_item.question_id}: {e}")
                traceback.print_exc()
                # Set to None instead of empty dict to indicate failure
                results[name] = (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            # 逐 linker 记录选中结果
            linked_result, linker_tokens = results[name]
            trace_event(
                "schema_link_result",
                linker=name,
                source="runner",
                tables=linked_result,
                table_count=len(linked_result) if linked_result else 0,
                column_count=sum(len(columns) for columns in linked_result.values()) if linked_result else 0,
                token_usage=linker_tokens,
                failed=linked_result is None,
                elapsed_s=round(time.time() - linker_start_time, 3),
            )

        direct_linked_tables_and_columns, direct_tokens = results["direct"]
        reversed_linked_tables_and_columns, reversed_tokens = results["reversed"]
        value_linked_tables_and_columns, value_tokens = results["value"]
            
        # Accumulate token usage
        for tokens in [direct_tokens, reversed_tokens, value_tokens]:
            total_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            total_token_usage["completion_tokens"] += tokens["completion_tokens"]
            total_token_usage["total_tokens"] += tokens["total_tokens"]
        
        # Check if any linker failed (returned None) before merging
        if direct_linked_tables_and_columns is None or reversed_linked_tables_and_columns is None or value_linked_tables_and_columns is None:
            failed_linkers = []
            if direct_linked_tables_and_columns is None:
                failed_linkers.append("direct")
            if reversed_linked_tables_and_columns is None:
                failed_linkers.append("reversed")
            if value_linked_tables_and_columns is None:
                failed_linkers.append("value")
            logger.error(f"Linker(s) {failed_linkers} failed for item {data_item.question_id}, setting all linking results to None")
            data_item.direct_linked_tables_and_columns = direct_linked_tables_and_columns
            data_item.reversed_linked_tables_and_columns = reversed_linked_tables_and_columns
            data_item.value_linked_tables_and_columns = value_linked_tables_and_columns
            data_item.final_linked_tables_and_columns = None
            data_item.database_schema_after_schema_linking = None
            trace_event(
                "schema_link_merge",
                skipped=True,
                reason="linker_failed",
                failed_linkers=failed_linkers,
            )
        else:
            merged_linked_tables_and_columns = merge_schema_linking_results([
                direct_linked_tables_and_columns, 
                reversed_linked_tables_and_columns, 
                value_linked_tables_and_columns
            ])
            trace_event(
                "schema_link_merge",
                tables=merged_linked_tables_and_columns,
                table_count=len(merged_linked_tables_and_columns),
                column_count=sum(len(columns) for columns in merged_linked_tables_and_columns.values()),
            )
            data_item.direct_linked_tables_and_columns = direct_linked_tables_and_columns
            data_item.reversed_linked_tables_and_columns = reversed_linked_tables_and_columns
            data_item.value_linked_tables_and_columns = value_linked_tables_and_columns
            data_item.final_linked_tables_and_columns = merged_linked_tables_and_columns
            filtered_schema = filter_used_database_schema(data_item.database_schema_after_value_retrieval, merged_linked_tables_and_columns)
            data_item.database_schema_after_schema_linking = filtered_schema
            # 记录过滤后保留/丢弃的表与列
            filtered_tables = filtered_schema.get("tables", {})
            kept_columns = {
                table_name: list(table_dict.get("columns", {}).keys())
                for table_name, table_dict in filtered_tables.items()
            }
            dropped_columns = {
                table_name: [
                    column_name for column_name in columns
                    if column_name not in filtered_tables[table_name].get("columns", {})
                ]
                for table_name, columns in merged_linked_tables_and_columns.items()
                if table_name in filtered_tables
            }
            dropped_columns = {table_name: columns for table_name, columns in dropped_columns.items() if columns}
            trace_event(
                "schema_link_filter",
                force_include_pks_and_fks=True,
                merged_table_count=len(merged_linked_tables_and_columns),
                merged_column_count=sum(len(columns) for columns in merged_linked_tables_and_columns.values()),
                kept_table_count=len(filtered_tables),
                kept_column_count=sum(len(columns) for columns in kept_columns.values()),
                kept_columns=kept_columns,
                dropped_tables=[
                    table_name for table_name in merged_linked_tables_and_columns
                    if table_name not in filtered_tables
                ],
                dropped_columns=dropped_columns,
            )
        
        end_time = time.time()
        data_item.schema_linking_time = end_time - start_time
        data_item.schema_linking_llm_cost = total_token_usage
        data_item.total_time += data_item.schema_linking_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.schema_linking_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.schema_linking_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.schema_linking_llm_cost["total_tokens"],
        }
        self._eval_schema_linking_recall(data_item)

        # 记录最终 schema 规模
        final_schema = data_item.database_schema_after_schema_linking or {}
        final_tables = final_schema.get("tables", {})
        trace_event(
            "schema_link_end",
            total_time=data_item.schema_linking_time,
            final_table_count=len(final_tables),
            final_column_count=sum(len(table.get("columns", {})) for table in final_tables.values()),
            final_tables=list(final_tables.keys()),
            final_linked_tables_and_columns=data_item.final_linked_tables_and_columns,
            linkers_failed=data_item.final_linked_tables_and_columns is None,
            token_usage=data_item.schema_linking_llm_cost,
            total_llm_cost=data_item.total_llm_cost,
            final_linking_recall=data_item.final_linking_recall,
        )
        
    def _is_linking_complete(self, data_item: DataItem) -> bool:
        """Check if schema linking step completed successfully."""
        return data_item.is_stage_complete("schema_linking")
    
    def _process_item(self, data_item: DataItem):
        """单条 item 的处理入口：设置追踪上下文。"""
        with trace_scope(item_id=data_item.get_item_id()):
            self._link_tables_and_columns(data_item)

    def run(self):
        future_to_item = {}
        for data_item in self._dataset:
            if self._is_linking_complete(data_item):
                # If already linked but recall is missing, evaluate it now
                if data_item.get_stage_artifact("schema_linking").final_linking_recall is None:
                    with trace_scope(item_id=data_item.get_item_id()):
                        self._eval_schema_linking_recall(data_item)
                logger.info(f"Skipping data item {data_item.question_id} because it has already been linked")
                continue
            future = submit_with_trace(self._thread_pool_executor, self._process_item, data_item)
            future_to_item[future] = data_item
        for idx, future in tqdm(enumerate(as_completed(future_to_item), start=1), total=len(future_to_item), desc="Linking tables and columns"):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            log_progress("Linking tables and columns", idx, len(future_to_item), self._progress_log_interval, previous_completed=idx - 1)
            if should_checkpoint(idx, self._checkpoint_interval):
                self.save_result()
        logger.info("Linking tables and columns completed")
        
        # Validate that all required fields are filled
        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "schema_linking")
        self.save_result(materialize_snapshot=True)
        
        self._clean_up()
        
    
    def save_result(self, materialize_snapshot: bool = False):
        self._artifact_store.flush()
        if materialize_snapshot:
            save_dataset(self._dataset, self._stage_config.save_path)
            self._artifact_store.cleanup()
        
    def _eval_schema_linking_recall(self, data_item: DataItem):
        # Skip recall calculation if gold_sql is missing or empty (typical for Spider2 inference)
        if not hasattr(data_item, "gold_sql") or not data_item.gold_sql or not data_item.gold_sql.strip():
            # Initialize with default zero recall or None
            default_recall = {"table_recall": 0.0, "column_recall": 0.0}
            data_item.direct_linking_recall = default_recall
            data_item.reversed_linking_recall = default_recall
            data_item.value_linking_recall = default_recall
            data_item.final_linking_recall = default_recall
            # 无 gold SQL 时仍逐方法记录一条跳过事件
            for method in ("direct", "reversed", "value", "final"):
                trace_event(
                    "schema_link_recall",
                    method=method,
                    skipped=True,
                    reason="no_gold_sql",
                    table_recall=0.0,
                    column_recall=0.0,
                )
            return

        gold_tables_and_columns = self._reversed_linker._extract_tables_and_columns(data_item.gold_sql, data_item.database_schema_after_value_retrieval)
        
        def _calc_recall(linked_tables_and_columns):
            """Calculate table and column recall for a linking result."""
            if linked_tables_and_columns is None:
                return 0, 0
            table_recall = 0
            column_recall = 0
            for table_name, columns in linked_tables_and_columns.items():
                if table_name in gold_tables_and_columns:
                    table_recall += 1
                    for column_name in columns:
                        if column_name in gold_tables_and_columns[table_name]:
                            column_recall += 1
            table_recall /= len(gold_tables_and_columns.keys()) if len(gold_tables_and_columns.keys()) > 0 else 1
            column_recall /= sum(len(columns) for columns in gold_tables_and_columns.values()) if sum(len(columns) for columns in gold_tables_and_columns.values()) > 0 else 1
            return table_recall, column_recall
        
        direct_table_recall, direct_column_recall = _calc_recall(data_item.direct_linked_tables_and_columns)
        reversed_table_recall, reversed_column_recall = _calc_recall(data_item.reversed_linked_tables_and_columns)
        value_table_recall, value_column_recall = _calc_recall(data_item.value_linked_tables_and_columns)
        final_table_recall, final_column_recall = _calc_recall(data_item.final_linked_tables_and_columns)
        
        data_item.direct_linking_recall = {"table_recall": direct_table_recall, "column_recall": direct_column_recall}
        data_item.reversed_linking_recall = {"table_recall": reversed_table_recall, "column_recall": reversed_column_recall}
        data_item.value_linking_recall = {"table_recall": value_table_recall, "column_recall": value_column_recall}
        data_item.final_linking_recall = {"table_recall": final_table_recall, "column_recall": final_column_recall}

        # 记录逐方法的召回明细与命中的表/列
        def _recall_detail(linked_tables_and_columns):
            hit_tables = []
            hit_columns = {}
            for table_name, columns in (linked_tables_and_columns or {}).items():
                if table_name in gold_tables_and_columns:
                    hit_tables.append(table_name)
                    matched_columns = [column_name for column_name in columns if column_name in gold_tables_and_columns[table_name]]
                    if matched_columns:
                        hit_columns[table_name] = matched_columns
            return hit_tables, hit_columns

        gold_table_count = len(gold_tables_and_columns.keys())
        gold_column_count = sum(len(columns) for columns in gold_tables_and_columns.values())
        recall_entries = {
            "direct": (data_item.direct_linking_recall, data_item.direct_linked_tables_and_columns),
            "reversed": (data_item.reversed_linking_recall, data_item.reversed_linked_tables_and_columns),
            "value": (data_item.value_linking_recall, data_item.value_linked_tables_and_columns),
            "final": (data_item.final_linking_recall, data_item.final_linked_tables_and_columns),
        }
        for method, (recall, linked) in recall_entries.items():
            hit_tables, hit_columns = _recall_detail(linked)
            trace_event(
                "schema_link_recall",
                method=method,
                table_recall=recall["table_recall"],
                column_recall=recall["column_recall"],
                gold_table_count=gold_table_count,
                gold_column_count=gold_column_count,
                gold_tables=list(gold_tables_and_columns.keys()),
                gold=gold_tables_and_columns,
                linked=linked,
                predicted_table_count=len(linked) if linked else 0,
                predicted_column_count=sum(len(columns) for columns in linked.values()) if linked else 0,
                hit_table_count=len(hit_tables),
                hit_column_count=sum(len(columns) for columns in hit_columns.values()),
                hit_tables=hit_tables,
                hit_columns=hit_columns,
            )

    
    def _clean_up(self):
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        if self._inner_thread_pool_executor is not None:
            self._inner_thread_pool_executor.shutdown(wait=True)
            self._inner_thread_pool_executor = None
        self._llm = None
        self._dataset = None
        self._direct_linker = None
        self._reversed_linker = None
        self._value_linker = None
        if self._artifact_store is not None:
            self._artifact_store.close()
        reset_schema_service()
