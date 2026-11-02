"""
Unified evaluation script for all datasets (Spider, BIRD, Spider2).
Automatically detects dataset type and uses appropriate evaluation method.
"""

import sys
sys.path.append(".")

import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np

from app.db_utils.defaults import DEFAULT_SQL_EXECUTION_TIMEOUT
from app.logger import configure_logger, logger


def _resolve_snapshot_path(snapshot_path: Optional[str], default_snapshot_path: Optional[str] = None) -> str:
    resolved_path = snapshot_path or default_snapshot_path
    if resolved_path is None:
        raise ValueError("snapshot_path is required when no default snapshot path is provided")
    return resolved_path


# 逐题对比时最多保留多少行明细，避免 trace 文件被大结果集撑爆
_EVAL_ROW_PREVIEW = 20


def _exec_summary(result) -> dict:
    """把一个 SQLExecutionResult 压成可序列化的摘要（含前若干行）。"""
    rows = result.result_rows
    return {
        "result_type": result.result_type,
        "error_message": result.error_message,
        "execution_time": result.execution_time,
        "result_cols": list(result.result_cols) if result.result_cols else None,
        "row_count": None if rows is None else len(rows),
        "rows_preview": [list(r) for r in rows[:_EVAL_ROW_PREVIEW]] if rows else None,
        "rows_preview_truncated": bool(rows) and len(rows) > _EVAL_ROW_PREVIEW,
        "result_table_str": result.result_table_str,
    }


def _eval_ex_after_selection(pred_sql: str, gold_sql: str, db_path: str, timeout: Optional[int] = None) -> dict:
    """
    Evaluate execution accuracy by comparing query results.
    Used for Spider and BIRD datasets.

    返回逐题明细字典（而不是单纯的 0/1），供上层写入 trace：
    pred / gold 的执行摘要、行数、是否命中、以及差在哪几行。
    """
    from app.db_utils import execute_sql

    pred_result = execute_sql(db_path, pred_sql, timeout=timeout)
    gold_result = execute_sql(db_path, gold_sql, timeout=timeout)

    detail = {
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "db_path": db_path,
        "pred": _exec_summary(pred_result),
        "gold": _exec_summary(gold_result),
        "correct": None,
        "score": None,
        "reason": None,
        "missing_rows": None,
        "missing_count": None,
        "extra_rows": None,
        "extra_count": None,
    }

    if gold_result.result_rows is None:
        detail["reason"] = "gold_execution_failed"
        return detail

    if pred_result.result_rows is None:
        detail["reason"] = f"pred_execution_{pred_result.result_type}"
        detail["correct"] = 0
        detail["score"] = 0
        return detail

    pred_set = set(pred_result.result_rows)
    gold_set = set(gold_result.result_rows)
    if pred_set == gold_set:
        detail["correct"] = 1
        detail["score"] = 1
        detail["reason"] = "row_set_match"
        return detail

    missing = [r for r in gold_result.result_rows if r not in pred_set]
    extra = [r for r in pred_result.result_rows if r not in gold_set]
    detail["correct"] = 0
    detail["score"] = 0
    detail["reason"] = "row_set_mismatch"
    detail["missing_rows"] = [list(r) for r in missing[:_EVAL_ROW_PREVIEW]]
    detail["missing_count"] = len(missing)
    detail["extra_rows"] = [list(r) for r in extra[:_EVAL_ROW_PREVIEW]]
    detail["extra_count"] = len(extra)
    return detail


def evaluate_spider_bird(
    snapshot_path: str,
    max_workers: int = 32,
    dataset_type: Optional[str] = None,
    timeout: Optional[int] = None,
) -> float:
    """
    Evaluate Spider or BIRD dataset using direct SQL execution comparison.
    
    Args:
        snapshot_path: Path to the dataset snapshot with results.
        max_workers: Number of parallel workers.
        dataset_type: Dataset type, only used for trace metadata.
        timeout: SQL execution timeout (seconds).
        
    Returns:
        Execution accuracy as a float (0.0 to 1.0).
    """
    from app.dataset import load_dataset
    from app.tracing import configure_tracing, trace_event, trace_scope

    # 追踪与前面各阶段共用同一个 run 目录下的 trace/events.jsonl
    configure_tracing(Path(snapshot_path).parent / "trace", stage="evaluation")
    
    logger.info(f"Loading dataset snapshot from: {snapshot_path}")
    dataset = load_dataset(snapshot_path)
    
    trace_event(
        "evaluation_start",
        snapshot=str(snapshot_path),
        dataset_type=dataset_type,
        item_count=len(dataset),
        max_workers=max_workers,
        timeout=timeout,
        scoring="set(pred.result_rows) == set(gold.result_rows) (Execution Accuracy)",
        row_preview_limit=_EVAL_ROW_PREVIEW,
    )

    logger.info(f"Evaluating {len(dataset)} queries with {max_workers} workers...")
    executor = ProcessPoolExecutor(max_workers=max_workers)
    # 记录 future -> data_item 映射，便于回填题目信息
    future_to_item = {
        executor.submit(
            _eval_ex_after_selection,
            data_item.final_selected_sql,
            data_item.gold_sql,
            data_item.database_path,
            timeout,
        ): data_item
        for data_item in dataset
    }
    
    selected_results = []
    mismatch_ids = []
    gold_failed_ids = []
    crashed_ids = []
    for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Evaluating SQL"):
        data_item = future_to_item[future]
        item_id = str(data_item.get_item_id())
        try:
            detail = future.result()
        except Exception as exc:
            crashed_ids.append(item_id)
            trace_event(
                "evaluation_item_error",
                item_id=item_id,
                question_id=str(getattr(data_item, "question_id", "") or ""),
                db_id=getattr(data_item, "database_id", None),
                final_selected_sql=getattr(data_item, "final_selected_sql", None),
                gold_sql=getattr(data_item, "gold_sql", None),
                error=f"{type(exc).__name__}: {exc}",
            )
            logger.error(f"[evaluation] item {item_id} failed: {type(exc).__name__}: {exc}")
            continue

        with trace_scope(item_id=item_id):
            trace_event(
                "evaluation_item",
                question_id=str(getattr(data_item, "question_id", "") or ""),
                db_id=getattr(data_item, "database_id", None),
                question=getattr(data_item, "question", None),
                evidence=getattr(data_item, "evidence", None),
                **detail,
            )

        if detail.get("reason") == "gold_execution_failed":
            gold_failed_ids.append(item_id)
            logger.warning(f"Gold SQL execution failed for one query (item {item_id})")
            continue

        if detail.get("score") is not None:
            selected_results.append(detail["score"])
            if detail["score"] != 1:
                mismatch_ids.append(item_id)
        
        # Show progress
        if len(selected_results) > 0 and len(selected_results) % 10 == 0:
            current_acc = np.mean(selected_results) * 100
            logger.info(f"[Progress] {len(selected_results)}/{len(dataset)} queries - Current EX: {current_acc:.2f}%")
    
    executor.shutdown()
    
    if len(selected_results) == 0:
        logger.error("No valid results to evaluate!")
        trace_event(
            "evaluation_end",
            status="no_valid_result",
            item_count=len(dataset),
            gold_failed_count=len(gold_failed_ids),
            crashed_count=len(crashed_ids),
            gold_failed_ids=gold_failed_ids,
            crashed_ids=crashed_ids,
        )
        return 0.0

    accuracy = np.mean(selected_results)

    trace_event(
        "evaluation_end",
        status="completed",
        item_count=len(dataset),
        evaluated_count=len(selected_results),
        correct_count=int(sum(1 for s in selected_results if s == 1)),
        incorrect_count=len(mismatch_ids),
        accuracy=float(accuracy),
        accuracy_percent=float(accuracy * 100),
        mismatch_ids=mismatch_ids,
        gold_failed_count=len(gold_failed_ids),
        gold_failed_ids=gold_failed_ids,
        crashed_count=len(crashed_ids),
        crashed_ids=crashed_ids,
    )
    return accuracy


def evaluate_spider2(
    snapshot_path: str,
    dataset_split: str = "lite",
    sql_output_dir: str = None,
    max_workers: int = 8,
    timeout: int = DEFAULT_SQL_EXECUTION_TIMEOUT,
    skip_conversion: bool = False,
) -> Optional[int]:
    """
    Evaluate Spider2 dataset using official evaluation script.
    
    Args:
        snapshot_path: Path to the dataset snapshot with results.
        dataset_split: "lite" or "snow".
        sql_output_dir: Directory for SQL output files.
        max_workers: Number of parallel workers for evaluation.
        timeout: SQL execution timeout in seconds.
        skip_conversion: Skip snapshot-to-SQL conversion (use existing SQL files).
        
    Returns:
        Return code from evaluation script (0 = success, None = error).
    """
    snapshot_path = Path(snapshot_path)
    
    if sql_output_dir is None:
        sql_output_dir = snapshot_path.parent / "sql_output"
    else:
        sql_output_dir = Path(sql_output_dir)
    
    # Ensure absolute path for sql_output_dir as we'll change CWD
    sql_output_dir = sql_output_dir.resolve()
    
    # Step 1: Convert dataset snapshot to SQL files (if not skipping)
    if not skip_conversion:
        logger.info(f"Step 1: Converting dataset snapshot {snapshot_path} to SQL files...")
        from runner.convert_snapshot_to_sql import convert_to_sql_files
        convert_to_sql_files(
            snapshot_path=str(snapshot_path),
            output_dir=str(sql_output_dir)
        )
    else:
        logger.info("Step 1: Skipping snapshot conversion (using existing SQL files)")
    
    # Step 2: Run official evaluation script
    logger.info(f"Step 2: Running official Spider2-{dataset_split} evaluation...")
    
    # Determine paths based on dataset split
    project_root = Path(__file__).resolve().parent.parent
    
    # Support both old path format and new unified format
    if dataset_split == "lite":
        eval_script = project_root / "data" / "spider2-lite" / "evaluation_suite" / "evaluate.py"
        gold_dir = project_root / "data" / "spider2-lite" / "evaluation_suite" / "gold"
    elif dataset_split == "snow":
        eval_script = project_root / "data" / "spider2-snow" / "evaluation_suite" / "evaluate.py"
        gold_dir = project_root / "data" / "spider2-snow" / "evaluation_suite" / "gold"
    else:
        raise ValueError(f"Unknown dataset split: {dataset_split}. Expected 'lite' or 'snow'")
    
    if not eval_script.exists():
        logger.error(f"Evaluation script not found: {eval_script}")
        logger.error("Please ensure the Spider2 evaluation suite is properly set up.")
        return None
    
    if not gold_dir.exists():
        logger.error(f"Gold directory not found: {gold_dir}")
        logger.error("Please ensure the Spider2 gold data is properly set up.")
        return None
    
    # Build command
    cmd = [
        sys.executable,
        str(eval_script),
        "--mode", "sql",
        "--result_dir", str(sql_output_dir),
        "--gold_dir", str(gold_dir),
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Change to the evaluation suite directory (required for relative paths in eval script)
    eval_cwd = eval_script.parent
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(eval_cwd),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Evaluation failed with return code: {result.returncode}")
        else:
            logger.info("Evaluation completed successfully!")
            
        return result.returncode
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return None


def run_evaluation(
    snapshot_path: str = None,
    dataset_type: str = None,
    dataset_split: str = None,
    max_workers: int = None,
    timeout: int = None,
    sql_output_dir: str = None,
    skip_conversion: bool = False,
    default_snapshot_path: Optional[str] = None,
    default_dataset_type: Optional[str] = None,
    default_dataset_split: Optional[str] = None,
    default_timeout: Optional[int] = None,
):
    """
    Unified evaluation entry point. Auto-detects dataset type and uses appropriate method.
    
    Args:
        snapshot_path: Path to the dataset snapshot with results.
        dataset_type: Override dataset type (auto-detected from config if None).
        dataset_split: Dataset split (required for Spider2).
        max_workers: Number of parallel workers.
        timeout: SQL execution timeout (Spider2 only).
        sql_output_dir: SQL output directory (Spider2 only).
        skip_conversion: Skip SQL conversion (Spider2 only).
    """
    snapshot_path = _resolve_snapshot_path(snapshot_path, default_snapshot_path)
    
    if dataset_type is None:
        if default_dataset_type is None:
            raise ValueError("dataset_type is required when no default dataset type is provided")
        dataset_type = default_dataset_type
        logger.info(f"Auto-detected dataset type: {dataset_type}")
    
    if dataset_split is None and dataset_type == "spider2":
        if default_dataset_split is None:
            raise ValueError("dataset_split is required for spider2 when no default split is provided")
        dataset_split = default_dataset_split
        logger.info(f"Auto-detected dataset split: {dataset_split}")

    if timeout is None:
        timeout = default_timeout if default_timeout is not None else DEFAULT_SQL_EXECUTION_TIMEOUT
    
    # Set default max_workers
    if max_workers is None:
        max_workers = 32 if dataset_type in ["spider", "bird"] else 8
    
    # Route to appropriate evaluation method
    if dataset_type in ["spider", "bird"]:
        logger.info(f"=== Evaluating {dataset_type.upper()} Dataset ===")
        accuracy = evaluate_spider_bird(
            snapshot_path=snapshot_path,
            max_workers=max_workers,
            dataset_type=dataset_type,
            timeout=timeout,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"  Overall Execution Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"{'='*60}\n")
        return accuracy
        
    elif dataset_type == "spider2":
        from app.tracing import configure_tracing, trace_event

        configure_tracing(Path(snapshot_path).parent / "trace", stage="evaluation")
        logger.info(f"=== Evaluating Spider2-{dataset_split.upper()} Dataset ===")
        trace_event(
            "evaluation_start",
            snapshot=str(snapshot_path),
            dataset_type=dataset_type,
            dataset_split=dataset_split,
            max_workers=max_workers,
            timeout=timeout,
            sql_output_dir=sql_output_dir,
            skip_conversion=skip_conversion,
            scoring="spider2 official evaluate.py",
        )
        result = evaluate_spider2(
            snapshot_path=snapshot_path,
            dataset_split=dataset_split,
            sql_output_dir=sql_output_dir,
            max_workers=max_workers,
            timeout=timeout,
            skip_conversion=skip_conversion
        )
        trace_event(
            "evaluation_end",
            status="completed" if result == 0 else "failed",
            returncode=result,
            dataset_split=dataset_split,
        )
        if result == 0:
            logger.info("\n" + "="*60)
            logger.info("  Spider2 Evaluation Completed Successfully!")
            logger.info("="*60 + "\n")
        return result
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Expected 'spider', 'bird', or 'spider2'")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for Spider, BIRD, and Spider2 dataset snapshots"
    )
    parser.add_argument(
        "--snapshot_path",
        type=str,
        default=None,
        help="Path to the dataset snapshot. Default: use config"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["spider", "bird", "spider2"],
        default=None,
        help="Dataset type (default: auto-detect from config)"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        choices=["dev", "test", "lite", "snow"],
        default=None,
        help="Dataset split (auto-detected from config if not specified)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 32 for Spider/BIRD, 8 for Spider2)"
    )
    
    # Spider2-specific arguments
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="SQL execution timeout in seconds (Spider2 only, default: use config)"
    )
    parser.add_argument(
        "--sql_output_dir",
        type=str,
        default=None,
        help="Directory for SQL output files (Spider2 only)"
    )
    parser.add_argument(
        "--skip_conversion",
        action="store_true",
        help="Skip snapshot-to-SQL conversion for Spider2 (use existing SQL files)"
    )
    
    args = parser.parse_args()
    from app.config import get_config

    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)

    run_evaluation(
        snapshot_path=args.snapshot_path,
        dataset_type=args.dataset_type,
        dataset_split=args.dataset_split,
        max_workers=args.max_workers,
        timeout=args.timeout,
        sql_output_dir=args.sql_output_dir,
        skip_conversion=args.skip_conversion,
        default_snapshot_path=app_config.sql_selection_config.save_path,
        default_dataset_type=app_config.dataset_config.type,
        default_dataset_split=app_config.dataset_config.split,
        default_timeout=app_config.dataset_config.sql_execution_timeout,
    )


if __name__ == "__main__":
    main()
