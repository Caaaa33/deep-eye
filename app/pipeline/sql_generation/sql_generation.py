from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from .generators import DCGenerator, SkeletonGenerator, ICLGenerator
from app.pipeline.validation import validate_pipeline_step
import time
from pathlib import Path
from app.logger import logger
from app.progress import log_progress, should_checkpoint
from tqdm import tqdm
import traceback
from app.services import ArtifactStore, STAGE_ARTIFACT_FIELDS, configure_schema_service, load_stage_dataset, reset_schema_service
from app.tracing import configure_tracing, trace_event, trace_scope, submit_with_trace


class SQLGenerationRunner:
    
    _llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    _inner_thread_pool_executor: ThreadPoolExecutor = None
    
    _dc_generator: DCGenerator = None
    _skeleton_generator: SkeletonGenerator = None
    _icl_generator: ICLGenerator = None
    _artifact_store: ArtifactStore = None
    _extractor_max_retry: int = 3
    _stage_config = None
    _input_save_path: str = ""
    _dataset_config = None
    _parallelism: int = 16
    _progress_log_interval: int = 50
    _checkpoint_interval: int = 20
    
    def __init__(
        self,
        stage_config,
        dataset_config,
        input_save_path: str,
        extractor_max_retry: int,
        parallelism: int,
        progress_log_interval: int,
        checkpoint_interval: int,
    ):
        self._stage_config = stage_config
        self._dataset_config = dataset_config
        self._input_save_path = input_save_path
        self._extractor_max_retry = extractor_max_retry
        self._parallelism = max(1, parallelism)
        self._progress_log_interval = max(1, progress_log_interval)
        self._checkpoint_interval = max(1, checkpoint_interval)
        self._artifact_store = ArtifactStore(
            self._stage_config.save_path,
            "sql_generation",
            STAGE_ARTIFACT_FIELDS["sql_generation"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=self._input_save_path,
            artifact_store=self._artifact_store,
            stage_name="sql_generation",
        )
        logger.info(f"Initialized SQL generation dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        self._llm = LLM(self._stage_config.llm)
        logger.info(f"SQL generation parallelism: {self._parallelism}")
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._dc_generator = DCGenerator(extractor_max_retry=self._extractor_max_retry)
        self._skeleton_generator = SkeletonGenerator(extractor_max_retry=self._extractor_max_retry)
        self._icl_generator = ICLGenerator(
            few_shot_examples_path=self._stage_config.icl_few_shot_examples_path,
            extractor_max_retry=self._extractor_max_retry,
        )
        # 追踪落盘到 snapshot 同级目录下的 trace/
        configure_tracing(Path(self._stage_config.save_path).parent / "trace", stage="sql_generation")

    @classmethod
    def from_config(cls, app_config=None) -> "SQLGenerationRunner":
        if app_config is None:
            from app.config import get_config

            app_config = get_config()
        return cls(
            stage_config=app_config.sql_generation_config,
            dataset_config=app_config.dataset_config,
            input_save_path=app_config.schema_linking_config.save_path,
            extractor_max_retry=app_config.llm_extractor_config.max_retry,
            parallelism=app_config.run_config.parallelism,
            progress_log_interval=app_config.run_config.progress_log_interval,
            checkpoint_interval=app_config.run_config.checkpoint_interval,
        )
        
    def _generate_sql_in_scope(self, data_item: DataItem) -> None:
        # 每条 item 的处理都挂上 trace 上下文（item_id）
        with trace_scope(item_id=data_item.get_item_id()):
            self._generate_sql(data_item)

    def _generate_sql(self, data_item: DataItem) -> None:
        start_time = time.time()
        
        # Track token usage for this specific data item
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 记录阶段起始信息（schema 规模 / 采样预算 / few-shot 数量）
        linked_schema = getattr(data_item, "database_schema_after_schema_linking", None)
        schema_to_use = linked_schema or data_item.database_schema
        schema_tables = (schema_to_use or {}).get("tables") or {}
        trace_event(
            "sql_generation_start",
            db_id=data_item.database_id,
            db_type=getattr(data_item, "db_type", None),
            question=data_item.question,
            evidence=data_item.evidence,
            schema_source="database_schema_after_schema_linking" if linked_schema else "database_schema",
            schema_table_count=len(schema_tables),
            schema_column_count=sum(len((table or {}).get("columns") or {}) for table in schema_tables.values()),
            dc_sampling_budget=self._stage_config.dc_sampling_budget,
            skeleton_sampling_budget=self._stage_config.skeleton_sampling_budget,
            icl_sampling_budget=self._stage_config.icl_sampling_budget,
            few_shot_example_count=len(data_item.few_shot_examples or []),
            few_shot_source="data_item" if data_item.few_shot_examples else None,
        )
        
        # Parallelize different generation methods within a single data item
        generation_tasks = {
            "dc": submit_with_trace(self._inner_thread_pool_executor, self._dc_generator.generate, data_item, self._llm, self._stage_config.dc_sampling_budget),
            "skeleton": submit_with_trace(self._inner_thread_pool_executor, self._skeleton_generator.generate, data_item, self._llm, self._stage_config.skeleton_sampling_budget),
            "icl": submit_with_trace(self._inner_thread_pool_executor, self._icl_generator.generate, data_item, self._llm, self._stage_config.icl_sampling_budget)
        }
        
        results = {}
        for name, future in generation_tasks.items():
            try:
                results[name] = future.result()
                candidates, tokens = results[name]
                trace_event(
                    "generator_result",
                    name=name,
                    candidate_count=len(candidates or []),
                    candidates=candidates,
                    token_usage=tokens,
                    error=None,
                )
            except Exception as e:
                logger.error(f"Error in {name} generation for item {data_item.question_id}: {e}")
                traceback.print_exc()
                # Set to None instead of empty list to indicate failure
                results[name] = (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                trace_event(
                    "generator_result",
                    name=name,
                    candidate_count=0,
                    candidates=None,
                    token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    error=traceback.format_exc(),
                    note="生成器异常，候选置空（None）",
                )

        dc_sql_candidates, dc_tokens = results["dc"]
        skeleton_sql_candidates, skeleton_tokens = results["skeleton"]
        icl_sql_candidates, icl_tokens = results["icl"]
        
        # Accumulate token usage
        for tokens in [dc_tokens, skeleton_tokens, icl_tokens]:
            total_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            total_token_usage["completion_tokens"] += tokens["completion_tokens"]
            total_token_usage["total_tokens"] += tokens["total_tokens"]
        
        # Check if any generator failed (returned None)
        if dc_sql_candidates is None or skeleton_sql_candidates is None or icl_sql_candidates is None:
            failed_generators = []
            if dc_sql_candidates is None:
                failed_generators.append("dc")
            if skeleton_sql_candidates is None:
                failed_generators.append("skeleton")
            if icl_sql_candidates is None:
                failed_generators.append("icl")
            logger.error(f"Generator(s) {failed_generators} failed for item {data_item.question_id}, setting sql_candidates to None")
            data_item.sql_candidates = None
            data_item.sql_candidate_sources = None
        else:
            data_item.sql_candidates = dc_sql_candidates + skeleton_sql_candidates + icl_sql_candidates
            # 记录每条候选的生成来源，供选择阶段 Stage 2 统计“跨来源结构支持度”
            data_item.sql_candidate_sources = (
                ["dc"] * len(dc_sql_candidates)
                + ["skeleton"] * len(skeleton_sql_candidates)
                + ["icl"] * len(icl_sql_candidates)
            )

        # 按最终 sql_candidates 顺序编号，记录全部初始 SQL
        numbered_candidates = []
        if data_item.sql_candidates is not None:
            for source, source_candidates in (
                ("dc", dc_sql_candidates),
                ("skeleton", skeleton_sql_candidates),
                ("icl", icl_sql_candidates),
            ):
                for sql in source_candidates:
                    numbered_candidates.append({
                        "index": len(numbered_candidates) + 1,
                        "source": source,
                        "sql": sql,
                    })
        trace_event(
            "sql_generation_candidates",
            total=len(numbered_candidates),
            candidates=numbered_candidates if data_item.sql_candidates is not None else None,
            sql_candidates_is_none=data_item.sql_candidates is None,
            candidate_count_by_source={
                "dc": len(dc_sql_candidates or []),
                "skeleton": len(skeleton_sql_candidates or []),
                "icl": len(icl_sql_candidates or []),
            },
        )
        
        end_time = time.time()
        data_item.sql_generation_time = end_time - start_time
        data_item.sql_generation_llm_cost = total_token_usage
        data_item.total_time += data_item.sql_generation_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_generation_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_generation_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_generation_llm_cost["total_tokens"],
        }

        # 阶段结束汇总
        trace_event(
            "sql_generation_end",
            total_candidates=len(numbered_candidates),
            candidate_count_by_source={
                "dc": len(dc_sql_candidates or []),
                "skeleton": len(skeleton_sql_candidates or []),
                "icl": len(icl_sql_candidates or []),
            },
            sql_candidates_is_none=data_item.sql_candidates is None,
            sql_generation_time=data_item.sql_generation_time,
            token_usage=total_token_usage,
        )
        
    def run(self):
        future_to_item = {}
        for data_item in self._dataset:
            if data_item.is_stage_complete("sql_generation"):
                logger.info(f"Skipping data item {data_item.question_id} because it has already been generated")
                continue
            future = submit_with_trace(self._thread_pool_executor, self._generate_sql_in_scope, data_item)
            future_to_item[future] = data_item
        for idx, future in tqdm(enumerate(as_completed(future_to_item), start=1), total=len(future_to_item), desc="Generating SQL"):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            log_progress("Generating SQL", idx, len(future_to_item), self._progress_log_interval, previous_completed=idx - 1)
            if should_checkpoint(idx, self._checkpoint_interval):
                self.save_result()
        logger.info("Generating SQL completed")
        
        # Validate that all required fields are filled
        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "sql_generation")
        self.save_result(materialize_snapshot=True)
        
        self._clean_up()
        
    def save_result(self, materialize_snapshot: bool = False):
        self._artifact_store.flush()
        if materialize_snapshot:
            save_dataset(self._dataset, self._stage_config.save_path)
            self._artifact_store.cleanup()
        
    def _clean_up(self):
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        if self._inner_thread_pool_executor is not None:
            self._inner_thread_pool_executor.shutdown(wait=True)
            self._inner_thread_pool_executor = None
        self._llm = None
        self._dataset = None
        self._dc_generator = None
        self._skeleton_generator = None
        self._icl_generator = None
        if self._artifact_store is not None:
            self._artifact_store.close()
        reset_schema_service()
