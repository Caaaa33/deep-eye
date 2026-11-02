from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.llm import LLM
from app.logger import logger
from app.prompt import PromptFactory
from app.llm_extractor import LLMExtractor
from app.pipeline.validation import validate_pipeline_step
from app.progress import log_progress, should_checkpoint
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from collections import Counter
from tqdm import tqdm
import time
from app.services import ArtifactStore, STAGE_ARTIFACT_FIELDS, configure_execution_service, configure_schema_service, get_execution_service, get_schema_service, load_stage_dataset, reset_execution_service, reset_schema_service
from app.llm_extractor import LLMExtractor
from app.pipeline.sql_selection.ast_signature import structural_support_select
from app.tracing import configure_tracing, submit_with_trace, trace_event, trace_scope
from pathlib import Path
from tabulate import tabulate


_JUDGE_RESULT_MAX_DISTINCT_ROWS = 30


def _build_judge_result_view(execution_result) -> str:
    """Render an execution result for the pairwise judge.

    `SQLExecutionResult.result_table_str` shows only the first 5 rows and no row
    count, and it is shared with the L3 checkers, so it cannot be changed here.
    For the judge that view collapses distinct candidates into identical prompts:
    two queries that differ past the fifth row, or only in row multiplicity, look
    the same. This view states the row counts and deduplicates the rows, matching
    the set-of-rows comparison the judge is asked to make.
    """
    if execution_result.result_cols is None or execution_result.result_rows is None:
        return execution_result.error_message or ""
    rows = execution_result.result_rows
    distinct_rows = list(dict.fromkeys(rows))
    shown_rows = distinct_rows[:_JUDGE_RESULT_MAX_DISTINCT_ROWS]
    if len(shown_rows) < len(distinct_rows):
        header = (
            f"Total rows returned: {len(rows)}    Distinct rows: {len(distinct_rows)}"
            f"    (showing the first {len(shown_rows)} distinct rows)"
        )
    else:
        header = (
            f"Total rows returned: {len(rows)}    Distinct rows: {len(distinct_rows)}"
            "    (all distinct rows shown below; duplicate rows are omitted)"
        )
    table_rows = []
    for row in shown_rows:
        table_row = []
        for val in row:
            if isinstance(val, str) and len(val) > 100:
                table_row.append(f"'{val[:100]}...'")
            else:
                table_row.append(val)
        table_rows.append(table_row)
    table_str = tabulate(
        tabular_data=table_rows,
        headers=execution_result.result_cols,
        tablefmt="psql",
    )
    return f"{header}\n{table_str}"


class SQLSelectionRunner:
    
    _llm: LLM = None
    _judge_llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    _inner_thread_pool_executor: ThreadPoolExecutor = None
    _artifact_store: ArtifactStore = None
    _execution_service = None
    _extractor_max_retry: int = 3
    _stage_config = None
    _input_save_path: str = ""
    _dataset_config = None
    _timing_refine_repeat: int = 2
    _extractor: LLMExtractor = None
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
            "sql_selection",
            STAGE_ARTIFACT_FIELDS["sql_selection"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=self._input_save_path,
            artifact_store=self._artifact_store,
            stage_name="sql_selection",
        )
        logger.info(f"Initialized SQL selection dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        configure_execution_service(
            default_timeout=self._dataset_config.sql_execution_timeout,
            bigquery_credential_path=self._dataset_config.bigquery_credential_path,
            snowflake_credential_path=self._dataset_config.snowflake_credential_path,
        )
        self._llm = LLM(self._stage_config.llm)
        # SIRIUS uses a pairwise reward model as judge; when a dedicated judge
        # model is configured we use it for A/B comparison so the selecting
        # model is no longer the same one that generated the candidates.
        judge_llm_config = getattr(self._stage_config, "judge_llm", None)
        self._judge_llm = LLM(judge_llm_config) if judge_llm_config is not None else self._llm
        if judge_llm_config is not None:
            logger.info(
                f"Using dedicated judge LLM for pairwise comparison: model={judge_llm_config.model}, "
                f"temperature={judge_llm_config.temperature}"
            )
        else:
            logger.info(f"Pairwise judge falls back to the selection LLM: model={self._judge_llm.llm_config.model}")
        logger.info(f"SQL selection parallelism: {self._parallelism}")
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._execution_service = get_execution_service()
        self._extractor = LLMExtractor(max_retry=self._extractor_max_retry)
        configure_tracing(Path(self._stage_config.save_path).parent / "trace", stage="sql_selection")

    @classmethod
    def from_config(cls, app_config=None) -> "SQLSelectionRunner":
        if app_config is None:
            from app.config import get_config

            app_config = get_config()
        return cls(
            stage_config=app_config.sql_selection_config,
            dataset_config=app_config.dataset_config,
            input_save_path=app_config.sql_revision_config.save_path,
            extractor_max_retry=app_config.llm_extractor_config.max_retry,
            parallelism=app_config.run_config.parallelism,
            progress_log_interval=app_config.run_config.progress_log_interval,
            checkpoint_interval=app_config.run_config.checkpoint_interval,
        )
    
    @staticmethod
    def _normalize_judge_label(raw: str) -> Optional[str]:
        """将裁判给出的标签归一化为 C（Correct）/ I（Incorrect）。"""
        text = (raw or "").strip().upper()
        # INCORRECT 包含 CORRECT 子串，必须先判断
        if text.startswith("INCORRECT"):
            return "I"
        if text.startswith("CORRECT"):
            return "C"
        return None

    def _parse_llm_response(self, response: str) -> Optional[str]:
        """
        Parse the llm response into an A/B/TIE vote.

        Follows the SIRIUS pairwise reward-model format: the judge labels the two
        presented queries independently (sql1_judge / sql2_judge), and the vote is
        derived purely from the two labels, so equal labels (both correct or both
        incorrect) naturally become TIE instead of a forced A/B choice.
        """
        try:
            result_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            block = result_match.group(1) if result_match else response
            labels = []
            for tag in ("sql1_judge", "sql2_judge"):
                tag_match = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
                labels.append(self._normalize_judge_label(tag_match.group(1)) if tag_match else None)

            if labels[0] is not None and labels[1] is not None:
                if labels[0] == labels[1]:
                    return "TIE"
                return "A" if labels[0] == "C" else "B"

            # 兼容旧式单字母输出（模型偶尔忽略标签格式）
            legacy = (result_match.group(1) if result_match else response).strip().upper()
            if legacy in ("A", "B", "TIE"):
                return legacy
            logger.error("No <sql1_judge>/<sql2_judge> labels found in LLM response")
            logger.debug(f"Response content: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response content: {response}")
            return None

    def _get_top_k_sql_candidates(self, data_item: DataItem) -> Tuple[List[Tuple[str, str, float, float]], Dict[str, Any]]:
        item_id = data_item.get_item_id()
        trace_event(
            "selection_input",
            item_id=item_id,
            question=data_item.question,
            evidence=data_item.evidence,
            candidate_count=len(data_item.sql_candidates_after_revision),
            candidates=list(data_item.sql_candidates_after_revision),
        )
        valid_sql_candidates = []
        fallback_sql_candidates = []
        sql_map_to_execution_result = {}
        sql_map_to_result_hash = {}
        for candidate_idx, sql_candidate in enumerate(data_item.sql_candidates_after_revision):
            execution_result = sql_map_to_execution_result.get(sql_candidate)
            if execution_result is None:
                execution_result = self._execution_service.execute(data_item, sql_candidate)
                sql_map_to_execution_result[sql_candidate] = execution_result
                trace_event(
                    "selection_execution_summary",
                    item_id=item_id,
                    candidate_index=candidate_idx,
                    sql=sql_candidate,
                    result_type=execution_result.result_type,
                    error_message=execution_result.error_message,
                    execution_time=execution_result.execution_time,
                    row_count=len(execution_result.result_rows) if execution_result.result_rows is not None else None,
                    result_table_str=execution_result.result_table_str,
                )
            if execution_result.result_rows is None:
                continue
            result_hash = sql_map_to_result_hash.get(sql_candidate)
            if result_hash is None:
                result_hash = self._execution_service.hash_result(data_item, execution_result.result_rows)
                sql_map_to_result_hash[sql_candidate] = result_hash
            fallback_sql_candidates.append((sql_candidate, result_hash))
            if len(execution_result.result_rows) > 0:
                valid_sql_candidates.append((sql_candidate, result_hash))

        if len(valid_sql_candidates) == 0 and len(fallback_sql_candidates) > 0:
            logger.warning("No successful SQL candidates, backing to SQL candidates with not none result_rows")
            valid_sql_candidates = fallback_sql_candidates

        if len(valid_sql_candidates) == 0:
            trace_event(
                "selection_consistency",
                item_id=item_id,
                valid_count=0,
                fallback_count=len(fallback_sql_candidates),
                outcome="no_valid_candidate",
            )
            return [], {
                "valid_count": 0,
                "unique_result_count": 0,
                "max_class_size": 0,
                "fallback_count": len(fallback_sql_candidates),
            }

        counter = Counter(execution_result for _, execution_result in valid_sql_candidates)

        deduplicated_valid_sql_candidates = []
        seen_result_set = set()
        for sql_candidate, result_hash in valid_sql_candidates:
            if result_hash not in seen_result_set:
                execution_result = sql_map_to_execution_result[sql_candidate]
                execution_time = execution_result.execution_time if execution_result.execution_time is not None else np.inf
                deduplicated_valid_sql_candidates.append(
                    (sql_candidate, result_hash, counter[result_hash] / len(valid_sql_candidates), execution_time)
                )
                seen_result_set.add(result_hash)

        trace_event(
            "selection_consistency",
            item_id=item_id,
            valid_count=len(valid_sql_candidates),
            fallback_count=len(fallback_sql_candidates),
            unique_result_count=len(deduplicated_valid_sql_candidates),
            max_class_size=max(counter.values()) if counter else 0,
            result_hash_votes={str(hash_value): count for hash_value, count in counter.items()},
            deduplicated_candidates=[
                {
                    "sql": sql_candidate,
                    "result_hash": str(result_hash),
                    "consistency": consistency_score,
                    "execution_time": execution_time,
                }
                for sql_candidate, result_hash, consistency_score, execution_time in deduplicated_valid_sql_candidates
            ],
        )

        top_k_sql_candidates = sorted(
            deduplicated_valid_sql_candidates,
            key=self._get_sql_candidate_sort_key,
            reverse=True,
        )[:self._stage_config.filter_top_k_sql]
        top_k_sql_candidates = self._refine_relevant_tied_candidate_timings(
            data_item,
            deduplicated_valid_sql_candidates,
            top_k_sql_candidates,
        )

        trace_event(
            "selection_top_k",
            item_id=item_id,
            top_k=self._stage_config.filter_top_k_sql,
            top_k_candidates=[
                {
                    "index": idx,
                    "sql": sql_candidate,
                    "consistency": consistency_score,
                    "execution_time": execution_time,
                }
                for idx, (sql_candidate, _, consistency_score, execution_time) in enumerate(top_k_sql_candidates)
            ],
        )

        candidates = [
            (
                sql_candidate,
                _build_judge_result_view(sql_map_to_execution_result[sql_candidate]),
                consistency_score,
                execution_time,
            )
            for sql_candidate, _, consistency_score, execution_time in top_k_sql_candidates
        ]
        meta = {
            "valid_count": len(valid_sql_candidates),
            "unique_result_count": len(deduplicated_valid_sql_candidates),
            "max_class_size": max(counter.values()) if counter else 0,
            "fallback_count": len(fallback_sql_candidates),
        }
        return candidates, meta

    @staticmethod
    def _get_sql_candidate_sort_key(sql_candidate: Tuple[str, Any, float, float]) -> Tuple[float, float]:
        return (sql_candidate[2], -sql_candidate[3])

    def _refine_relevant_tied_candidate_timings(
        self,
        data_item: DataItem,
        all_sql_candidates: List[Tuple[str, Any, float, float]],
        provisional_top_k_sql_candidates: List[Tuple[str, Any, float, float]],
    ) -> List[Tuple[str, Any, float, float]]:
        if len(provisional_top_k_sql_candidates) <= 1:
            return provisional_top_k_sql_candidates

        relevant_consistency_scores = {sql_candidate[2] for sql_candidate in provisional_top_k_sql_candidates}
        consistency_score_counts = Counter(sql_candidate[2] for sql_candidate in all_sql_candidates)

        refined_sql_candidates = []
        did_refine = False
        for sql_candidate, result_hash, consistency_score, execution_time in all_sql_candidates:
            if consistency_score in relevant_consistency_scores and consistency_score_counts[consistency_score] > 1:
                original_execution_time = execution_time
                execution_time = self._execution_service.measure_time(
                    data_item,
                    sql_candidate,
                    repeat=self._timing_refine_repeat,
                )
                did_refine = True
                trace_event(
                    "selection_timing_refine",
                    item_id=data_item.get_item_id(),
                    sql=sql_candidate,
                    consistency=consistency_score,
                    repeat=self._timing_refine_repeat,
                    original_execution_time=original_execution_time,
                    refined_execution_time=execution_time,
                )
            refined_sql_candidates.append(
                (sql_candidate, result_hash, consistency_score, execution_time)
            )

        if not did_refine:
            return provisional_top_k_sql_candidates

        return sorted(
            refined_sql_candidates,
            key=self._get_sql_candidate_sort_key,
            reverse=True,
        )[:self._stage_config.filter_top_k_sql]
    
    def _get_pair_sqls_to_eval(self, top_k_sql_candidates: List[Tuple[str, str, float, float]]) -> List[Tuple[Tuple[str, str, float, float], Tuple[str, str, float, float], bool]]:
        """
        Get the pair of sqls to eval.

        SIRIUS Eq.25: every pair is judged in BOTH presentation orders, never a
        random single order. A position prior is then cancelled exactly by the
        paired difference instead of only statistically. Cost is 2x per pair.
        Returns (candidate_a, candidate_b, swapped).
        """
        pair_sqls_to_eval = []
        for first_idx, first_sql in enumerate(top_k_sql_candidates):
            for second_idx, second_sql in enumerate(top_k_sql_candidates):
                if first_idx < second_idx:
                    pair_sqls_to_eval.append((first_sql, second_sql, False))
                    pair_sqls_to_eval.append((first_sql, second_sql, True))
        return pair_sqls_to_eval
    
    def _compare_sqls(
        self,
        sql_a: str,
        execution_result_table_str_a: str,
        sql_b: str,
        execution_result_table_str_b: str,
        *,
        question: str,
        evidence: str,
        db_type: str | None,
        database_schema_profile: str,
        swapped: bool = False,
        a_idx: Optional[int] = None,
        b_idx: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, int], bool]:
        """
        Compare the two sqls. Returns (votes, token_usage, swapped).
        When swapped=True, the A/B arguments are reversed before feeding to the LLM
        to counteract anchor bias.
        """
        # Deterministically swap A/B positions: every pair is judged in both orders
        if swapped:
            (sql_a, execution_result_table_str_a,
             sql_b, execution_result_table_str_b) = (
             sql_b, execution_result_table_str_b,
             sql_a, execution_result_table_str_a,
            )
        prompt = PromptFactory.format_br_pair_selection_prompt(
            database_schema_profile,
            question,
            evidence,
            sql_a,
            execution_result_table_str_a,
            sql_b,
            execution_result_table_str_b,
            db_type=db_type,
        )
        trace_event(
            "selection_compare_prompt",
            a_idx=a_idx,
            b_idx=b_idx,
            swapped=swapped,
            judge_model=self._judge_llm.llm_config.model,
            judge_is_stage_llm=self._judge_llm is self._llm,
            presented_sql_a=sql_a,
            presented_sql_b=sql_b,
            presented_table_a=execution_result_table_str_a,
            presented_table_b=execution_result_table_str_b,
            prompt_length=len(prompt),
        )
        
        votes, total_token_usage = self._extractor.extract_with_retry(
            llm=self._judge_llm,
            messages=[{"role": "user", "content": prompt}],
            rule_parser=self._parse_llm_response,
            fix_end_token=self._judge_llm.llm_config.fix_end_token,
            end_token="</result>",
            n=self._stage_config.evaluator_sampling_budget,
            trace_label="sql_selection.pairwise",
            trace_extra={"a_idx": a_idx, "b_idx": b_idx, "swapped": swapped},
        )
        trace_event(
            "selection_compare",
            a_idx=a_idx,
            b_idx=b_idx,
            swapped=swapped,
            votes=votes,
            token_usage=total_token_usage,
        )
        
        return votes, total_token_usage, swapped
    
    def _update_win_matrix(self, sql_a_idx: int, sql_b_idx: int, votes: List[str], win_matrix: np.ndarray, swapped: bool = False, vote_offset: int = 0) -> None:
        """
        Update the win matrix.
        When swapped=True, the LLM saw A/B in reversed order, so we invert votes
        back to the original candidate indices.
        vote_offset writes this order's votes into its own voter slice, so the
        forward and reversed runs of the same pair never overwrite each other.
        """
        for voter_idx, vote in enumerate(votes):
            # When swapped, the LLM's "A" actually refers to the candidate at sql_b_idx,
            # and "B" refers to sql_a_idx. Invert the mapping.
            if swapped:
                if vote == "A":
                    vote = "B"
                elif vote == "B":
                    vote = "A"
            slot = vote_offset + voter_idx
            if vote == "A":
                win_matrix[sql_a_idx, sql_b_idx, slot] = 1
                win_matrix[sql_b_idx, sql_a_idx, slot] = 0
            elif vote == "B":
                win_matrix[sql_b_idx, sql_a_idx, slot] = 1
                win_matrix[sql_a_idx, sql_b_idx, slot] = 0
            elif vote == "TIE":
                win_matrix[sql_a_idx, sql_b_idx, slot] = 0.5
                win_matrix[sql_b_idx, sql_a_idx, slot] = 0.5
            else:
                logger.error(f"Invalid vote: {vote}")
                raise ValueError(f"Invalid vote: {vote}")
    
    def _compute_robust_win_matrix(self, win_matrix: np.ndarray) -> float:
        """
        Calculate the robust vote.
        """
        sql_count, _, _ = win_matrix.shape
        robust_win_matrix = np.zeros((sql_count, sql_count))
        for sql_a_idx in range(sql_count):
            for sql_b_idx in range(sql_count):
                if sql_a_idx != sql_b_idx:
                    # 只对真正产出的投票求均值；解析失败的空槽位保持 NaN，不参与统计。
                    votes = win_matrix[sql_a_idx, sql_b_idx, :]
                    valid_votes = votes[~np.isnan(votes)]
                    win_prob = np.mean(valid_votes) if valid_votes.size > 0 else 0.5
                    robust_win_matrix[sql_a_idx, sql_b_idx] = win_prob
        return robust_win_matrix
    
    def _compute_elo_reward_scores(
        self,
        robust_win_matrix: np.ndarray,
        k: float = 0.5,
        iterations: int = 200,
        scale: float = 400.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SIRIUS reward-model score `S_rm` via Elo iteration.

        The robust win matrix only records how each *observed* pair played out,
        so a plain row mean cannot propagate transitive evidence (A beats B and
        B beats C should lift A against C). Iteratively updating Elo ratings
        over every pair and then taking each candidate's expected score against
        the rest of the pool yields a transitivity-aware score that stays on the
        same [0, 1] scale as the former mean win probability.

        Returns (s_rm_scores, elo_ratings).
        """
        n = robust_win_matrix.shape[0]
        if n <= 1:
            return np.zeros(n), np.zeros(n)

        ratings = np.zeros(n)
        for _ in range(iterations):
            deltas = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    expected = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / scale))
                    deltas[i] += k * (robust_win_matrix[i, j] - expected)
            if np.max(np.abs(deltas)) < 1e-6:
                ratings += deltas
                break
            ratings += deltas

        # Expected score against the field: BT-consistent, hence in (0, 1)
        scores = np.zeros(n)
        for i in range(n):
            expected_scores = []
            for j in range(n):
                if i == j:
                    continue
                expected_scores.append(1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / scale)))
            scores[i] = float(np.mean(expected_scores))
        return scores, ratings

    def _select_best_sql(self, data_item: DataItem) -> None:
        """
        Select the best sql based on the eval scores.
        """
        start_time = time.time()
        item_id = data_item.get_item_id()
        
        # Track token usage for this specific data item
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
        top_k_sql_candidates, selection_meta = self._get_top_k_sql_candidates(data_item)
        
        if len(top_k_sql_candidates) == 0:
            trace_event(
                "selection_decision",
                item_id=item_id,
                branch="no_valid_candidate",
                action="fallback_to_first_candidate",
                final_selected_sql=data_item.sql_candidates_after_revision[0] if data_item.sql_candidates_after_revision else "Error",
            )
            logger.warning("No valid SQL candidates, backing to top-1 SQL")
            data_item.final_selected_sql = data_item.sql_candidates_after_revision[0] if data_item.sql_candidates_after_revision else "Error"
            data_item.sql_selection_time = time.time() - start_time
            data_item.sql_selection_llm_cost = total_token_usage
            data_item.total_time += data_item.sql_selection_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
            }
            trace_event(
                "selection_end",
                item_id=item_id,
                status="completed",
                branch="no_valid_candidate",
                final_selected_sql=data_item.final_selected_sql,
                token_usage=total_token_usage,
                duration_ms=data_item.sql_selection_time * 1000,
            )
            return
        
        if len(top_k_sql_candidates) == 1:
            trace_event(
                "selection_decision",
                item_id=item_id,
                branch="single_candidate",
                action="direct_select",
                final_selected_sql=top_k_sql_candidates[0][0],
            )
            logger.info("Only one valid SQL candidate, directly select it")
            data_item.final_selected_sql = top_k_sql_candidates[0][0]
            data_item.sql_selection_time = time.time() - start_time
            data_item.sql_selection_llm_cost = total_token_usage
            data_item.total_time += data_item.sql_selection_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
            }
            trace_event(
                "selection_end",
                item_id=item_id,
                status="completed",
                branch="single_candidate",
                final_selected_sql=data_item.final_selected_sql,
                token_usage=total_token_usage,
                duration_ms=data_item.sql_selection_time * 1000,
            )
            return
        
        # SIRIUS §3.5: pairwise reward-model comparison is mandatory whenever
        # >1 distinct-result candidate survives; the confidence gate only
        # decides whether Stage 2 (cross-source structural support) runs.
        # using pair-wise comparison to select the best sql
        database_schema_profile = get_schema_service().build_schema_profile(
            data_item.database_schema_after_schema_linking,
            include_value_statistics=True,
            include_value_examples=True,
        )
        db_type = getattr(data_item, "db_type", None)
        # 每对候选跑两种展示顺序（SIRIUS Eq.25），故票维是 2 * sampling budget；
        # 未产出投票的槽位保持 NaN，聚合时被忽略。
        sampling_budget = self._stage_config.evaluator_sampling_budget
        win_matrix = np.full(
            (len(top_k_sql_candidates), len(top_k_sql_candidates), sampling_budget * 2),
            np.nan,
        )
        sql_to_idx = {sql[0]: idx for idx, sql in enumerate(top_k_sql_candidates)}
        pair_sqls_to_eval = self._get_pair_sqls_to_eval(top_k_sql_candidates)

        trace_event(
            "selection_decision",
            item_id=item_id,
            branch="pairwise",
            action="start_pairwise_comparison",
            consistency_gap=top_k_sql_candidates[0][2] - top_k_sql_candidates[1][2],
            top_k_count=len(top_k_sql_candidates),
            valid_count=selection_meta.get("valid_count"),
            unique_result_count=selection_meta.get("unique_result_count"),
            max_class_size=selection_meta.get("max_class_size"),
        )
        trace_event(
            "selection_pair_plan",
            item_id=item_id,
            pair_count=len(pair_sqls_to_eval),
            evaluator_sampling_budget=self._stage_config.evaluator_sampling_budget,
            pairs=[
                {
                    "a_idx": sql_to_idx[sql_a[0]],
                    "b_idx": sql_to_idx[sql_b[0]],
                    "swapped": swapped,
                    "sql_a": sql_a[0],
                    "sql_b": sql_b[0],
                }
                for sql_a, sql_b, swapped in pair_sqls_to_eval
            ],
        )
        
        # Parallelize the pairwise comparisons
        has_failure = False
        future_to_pair = {
            submit_with_trace(
                self._inner_thread_pool_executor,
                self._compare_sqls,
                sql_a[0],
                sql_a[1],
                sql_b[0],
                sql_b[1],
                question=data_item.question,
                evidence=data_item.evidence,
                db_type=db_type,
                database_schema_profile=database_schema_profile,
                swapped=swapped,
                a_idx=sql_to_idx[sql_a[0]],
                b_idx=sql_to_idx[sql_b[0]],
            ): (sql_a, sql_b, swapped)
            for sql_a, sql_b, swapped in pair_sqls_to_eval
        }
        
        for future in as_completed(future_to_pair):
            sql_a, sql_b, swapped = future_to_pair[future]
            try:
                votes, token_usage, _ = future.result()
                self._update_win_matrix(
                    sql_to_idx[sql_a[0]],
                    sql_to_idx[sql_b[0]],
                    votes,
                    win_matrix,
                    swapped=swapped,
                    vote_offset=sampling_budget if swapped else 0,
                )
                total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
                total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
                total_token_usage["total_tokens"] += token_usage["total_tokens"]
            except Exception as e:
                logger.error(f"Error comparing SQLs {sql_a[0]} and {sql_b[0]}: {e}")
                trace_event(
                    "selection_compare_error",
                    item_id=item_id,
                    a_idx=sql_to_idx[sql_a[0]],
                    b_idx=sql_to_idx[sql_b[0]],
                    swapped=swapped,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                has_failure = True
        
        # If any comparison failed, set result to None
        if has_failure:
            logger.error(f"Some SQL comparisons failed for item {data_item.question_id}, setting final_selected_sql to None")
            data_item.final_selected_sql = None
            trace_event(
                "selection_end",
                item_id=item_id,
                status="failed",
                reason="pairwise_comparison_failure",
                final_selected_sql=None,
                token_usage=total_token_usage,
                duration_ms=(time.time() - start_time) * 1000,
            )
        else:
            # Candidate provenance (dc / skeleton / icl / recovery) aligned with
            # the revised pool; needed for cross-source structural support.
            revised_sqls = list(data_item.sql_candidates_after_revision or [])
            revised_sources = list(data_item.sql_candidate_sources or [])
            sql_to_source: Dict[str, str] = {}
            for src_idx, revised_sql in enumerate(revised_sqls):
                if revised_sql in sql_to_source:
                    continue
                label = revised_sources[src_idx] if src_idx < len(revised_sources) and revised_sources[src_idx] else "unknown"
                sql_to_source[revised_sql] = label
            top_k_sources = [sql_to_source.get(sql[0], "unknown") for sql in top_k_sql_candidates]

            robust_win_matrix = self._compute_robust_win_matrix(win_matrix)
            # S_rm from Elo iteration (transitivity-aware) instead of a plain
            # row mean; the row mean is kept for trace-side comparison only.
            s_rm, elo_ratings = self._compute_elo_reward_scores(robust_win_matrix)
            mean_win_scores = np.mean(robust_win_matrix, axis=1)
            s_maj = np.array([sql[2] for sql in top_k_sql_candidates], dtype=float)
            # SIRIUS Eq.4/Eq.23: S_coarse = S_maj + S_rm (additive fusion)
            merged_scores = s_maj + s_rm
            ranking = np.argsort(-merged_scores)

            trace_event(
                "selection_win_matrix",
                item_id=item_id,
                # NaN 槽位（未产出投票）序列化为 null，避免 events.jsonl 出现非法 JSON 字面量
                win_matrix=[
                    [[None if np.isnan(v) else float(v) for v in slot] for slot in row]
                    for row in win_matrix
                ],
                robust_win_matrix=robust_win_matrix.tolist(),
            )
            trace_event(
                "selection_ranking",
                item_id=item_id,
                rm_scores=s_rm.tolist(),
                mean_win_scores=mean_win_scores.tolist(),
                elo_ratings=elo_ratings.tolist(),
                consistency_scores=s_maj.tolist(),
                merged_scores=merged_scores.tolist(),
                ranking=ranking.tolist(),
                sources=top_k_sources,
            )

            # SIRIUS adaptive confidence gating: Stage 2 is only attempted when
            # the merged score and the reward-model score both leave the top-2
            # exactly tied, the candidate count is within 1<K<=6, and no
            # result-equivalence class exceeds Tmax=8.
            tie_tol = 1e-9
            merged_top2_gap = float(np.sort(merged_scores)[-1] - np.sort(merged_scores)[-2]) if len(merged_scores) > 1 else None
            rm_top2_gap = float(np.sort(s_rm)[-1] - np.sort(s_rm)[-2]) if len(s_rm) > 1 else None
            top_k_count = len(top_k_sql_candidates)
            max_class_size = selection_meta.get("max_class_size") or 0
            gate_conditions = {
                "merged_top2_gap_zero": merged_top2_gap is not None and merged_top2_gap <= tie_tol,
                "rm_top2_gap_zero": rm_top2_gap is not None and rm_top2_gap <= tie_tol,
                "k_within_range": 1 < top_k_count <= 6,
                "max_class_size_ok": max_class_size <= 8,
            }
            gate_triggered = all(gate_conditions.values())
            trace_event(
                "selection_gate",
                item_id=item_id,
                triggered=gate_triggered,
                conditions=gate_conditions,
                merged_top2_gap=merged_top2_gap,
                rm_top2_gap=rm_top2_gap,
                top_k_count=top_k_count,
                max_class_size=max_class_size,
            )

            best_idx = int(ranking[0])
            if gate_triggered:
                # Stage 2 (SIRIUS §3.5): break the exact tie by cross-source
                # structural support; override only when strictly stronger.
                sql_candidates = [sql[0] for sql in top_k_sql_candidates]
                best_idx, stage2_detail = structural_support_select(
                    sql_candidates,
                    merged_scores.tolist(),
                    top_k_sources,
                    default_idx=int(ranking[0]),
                    tie_tol=tie_tol,
                )
                if best_idx != ranking[0]:
                    logger.info(
                        f"Stage 2 structural override for item {data_item.question_id}: "
                        f"idx={ranking[0]} -> idx={best_idx} "
                        f"(support {stage2_detail['supports'][ranking[0]]} -> {stage2_detail['supports'][best_idx]})"
                    )
                trace_event(
                    "selection_structural_override",
                    item_id=item_id,
                    initial_best_idx=int(ranking[0]),
                    final_best_idx=int(best_idx),
                    **stage2_detail,
                )
            data_item.final_selected_sql = top_k_sql_candidates[best_idx][0]
        
        data_item.sql_selection_time = time.time() - start_time
        data_item.sql_selection_llm_cost = total_token_usage
        data_item.total_time += data_item.sql_selection_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
        }
        trace_event(
            "selection_end",
            item_id=item_id,
            status="completed",
            final_selected_sql=data_item.final_selected_sql,
            token_usage=total_token_usage,
            duration_ms=data_item.sql_selection_time * 1000,
        )
    
    def _select_best_sql_scoped(self, data_item: DataItem) -> None:
        with trace_scope(stage="sql_selection", item_id=data_item.get_item_id()):
            self._select_best_sql(data_item)

    def run(self):
        future_to_item = {}
        for data_item in self._dataset:
            if data_item.is_stage_complete("sql_selection"):
                logger.info(f"Skipping data item {data_item.question_id} because it has already been selected")
                continue
            future = submit_with_trace(self._thread_pool_executor, self._select_best_sql_scoped, data_item)
            future_to_item[future] = data_item
        for idx, future in tqdm(enumerate(as_completed(future_to_item), start=1), total=len(future_to_item), desc="Selecting Best SQL"):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            log_progress("Selecting Best SQL", idx, len(future_to_item), self._progress_log_interval, previous_completed=idx - 1)
            if should_checkpoint(idx, self._checkpoint_interval):
                self.save_result()
        logger.info("Selecting Best SQL completed")
        
        # Validate that all required fields are filled
        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "sql_selection")
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
        if self._artifact_store is not None:
            self._artifact_store.close()
        reset_execution_service()
        reset_schema_service()
        self._llm = None
        self._judge_llm = None
        self._extractor = None
        self._dataset = None
