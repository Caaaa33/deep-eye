"""Export a human-readable demo report from the final pipeline snapshot.

Reads the structured snapshot produced by the pipeline (``*.snapshot.data/items.jsonl``)
and renders each item's full journey -- question, gold SQL, few-shot examples,
schema linking, SQL generation candidates, revision, final selection, and the
execution comparison -- into a plain-text file suitable for a demo/meeting.

Usage:
    python runner/export_demo_report.py
    python runner/export_demo_report.py --snapshot workspace/runs/bird-dev/sql_selection.snapshot.data/items.jsonl --output demo_report.txt
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(".")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cost_str(cost: Optional[Dict[str, Any]]) -> str:
    if not cost:
        return "N/A"
    return (
        f"prompt={cost.get('prompt_tokens', 0):,}  "
        f"completion={cost.get('completion_tokens', 0):,}  "
        f"total={cost.get('total_tokens', 0):,}"
    )


def _time_str(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.1f}s"


def _linked_str(linked: Optional[Dict[str, List[str]]]) -> str:
    if not linked:
        return "(无)"
    parts = []
    for table, cols in linked.items():
        cols = cols or []
        parts.append(f"{table}({', '.join(cols)})")
    return "; ".join(parts)


def _recall_str(recall: Optional[Dict[str, float]]) -> str:
    if not recall:
        return "N/A"
    return f"table={recall.get('table_recall', 0):.2f}  column={recall.get('column_recall', 0):.2f}"


def _run_sql(db_path: str, sql: Optional[str]):
    if not sql:
        return None
    try:
        from app.db_utils import execute_sql

        return execute_sql(db_path, sql)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _result_str(result) -> str:
    if result is None:
        return "(SQL 为空，未执行)"
    if isinstance(result, dict) and "error" in result:
        return f"执行异常: {result['error']}"
    # SQLExecutionResult
    if result.result_rows is None:
        return f"执行失败 [{result.result_type}]: {result.error_message or ''}"
    table = result.result_table_str
    if table:
        return "\n" + table
    return f"(空结果) type={result.result_type}"


def _json_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def render_item(rec: Dict[str, Any]) -> List[str]:
    inp = rec.get("input", {})
    art = rec.get("pipeline_artifacts", {})

    qid = inp.get("question_id", "?")
    question = inp.get("question", "")
    evidence = inp.get("evidence", "")
    difficulty = inp.get("difficulty", "")
    db_id = inp.get("database_id", "")
    gold_sql = inp.get("gold_sql", "")

    db_path = inp.get("database_path")
    if db_path and not Path(db_path).is_absolute():
        db_path = str(PROJECT_ROOT / db_path)

    sl = art.get("schema_linking", {})
    sg = art.get("sql_generation", {})
    sr = art.get("sql_revision", {})
    ss = art.get("sql_selection", {})
    metrics = art.get("metrics", {})

    final_sql = ss.get("final_selected_sql")
    candidates = sg.get("sql_candidates") or []
    revised = sr.get("sql_candidates_after_revision") or []

    lines: List[str] = []
    bar = "=" * 78
    lines.append(bar)
    lines.append(f"题目 #{qid}   数据库: {db_id}   难度: {difficulty}")
    lines.append(bar)
    lines.append("")
    lines.append("【问题 Question】")
    lines.append(question)
    lines.append("")
    if evidence:
        lines.append("【Evidence（题目给的提示）】")
        lines.append(evidence)
        lines.append("")
    lines.append("【真实答案 Gold SQL】")
    lines.append(gold_sql)
    lines.append("")

    # Few-shot preparation (stored on input)
    lines.append("【Step 1 · Few-shot 准备】")
    fs_meta = inp.get("few_shot_preparation_metadata") or {}
    fs_examples = inp.get("few_shot_examples") or []
    prelim_sql = inp.get("few_shot_preliminary_sql")
    lines.append(f"  检索到的 few-shot 示例数: {len(fs_examples)}")
    if isinstance(fs_meta, dict):
        mask_src = fs_meta.get("target_mask_source")
        if mask_src:
            lines.append(f"  目标掩码来源: {mask_src}")
        prelim_meta = fs_meta.get("preliminary_sql")
        if isinstance(prelim_meta, dict):
            lines.append(
                f"  初步 SQL 元信息: 候选={prelim_meta.get('candidate_count', '?')}, "
                f"可执行={prelim_meta.get('executable_candidate_count', '?')}, "
                f"一致性={prelim_meta.get('consistency_score', '?')}"
            )
    if prelim_sql:
        lines.append(f"  初步 SQL (preliminary): {prelim_sql}")
    for i, ex in enumerate(fs_examples, start=1):
        if not isinstance(ex, dict):
            continue
        lines.append(f"    - 示例{i}: Q: {ex.get('question', '')}")
        lines.append(f"             SQL: {ex.get('sql', '')}")
    lines.append("")

    # Schema linking
    lines.append("【Step 2 · Schema Linking（表和列链接）】")
    lines.append(f"  Direct 链接:  {_linked_str(sl.get('direct_linked_tables_and_columns'))}")
    lines.append(f"  Reversed 链接: {_linked_str(sl.get('reversed_linked_tables_and_columns'))}")
    lines.append(f"  Value 链接:    {_linked_str(sl.get('value_linked_tables_and_columns'))}")
    lines.append(f"  最终合并链接:  {_linked_str(sl.get('final_linked_tables_and_columns'))}")
    lines.append(f"  最终链接召回率: {_recall_str(sl.get('final_linking_recall'))}")
    lines.append(f"  耗时: {_time_str(sl.get('schema_linking_time'))}  Token: {_cost_str(sl.get('schema_linking_llm_cost'))}")
    lines.append("")

    # SQL generation
    lines.append(f"【Step 3 · SQL 生成（{len(candidates)} 个候选）】")
    if not candidates:
        lines.append("  (无候选)")
    for i, c in enumerate(candidates, start=1):
        lines.append(f"  候选{i}: {c}")
    lines.append(f"  耗时: {_time_str(sg.get('sql_generation_time'))}  Token: {_cost_str(sg.get('sql_generation_llm_cost'))}")
    lines.append("")

    # SQL revision
    lines.append(f"【Step 4 · SQL 修正（执行反馈后，{len(revised)} 个）】")
    if not revised:
        lines.append("  (无修正候选)")
    for i, c in enumerate(revised, start=1):
        lines.append(f"  修正候选{i}: {c}")
    lines.append(f"  耗时: {_time_str(sr.get('sql_revision_time'))}  Token: {_cost_str(sr.get('sql_revision_llm_cost'))}")
    lines.append("")

    # SQL selection
    lines.append("【Step 5 · SQL 选择（最终答案）】")
    lines.append(final_sql or "(无)")
    lines.append(f"  耗时: {_time_str(ss.get('sql_selection_time'))}  Token: {_cost_str(ss.get('sql_selection_llm_cost'))}")
    lines.append("")

    # Execution comparison
    lines.append("【执行结果对比】")
    gold_result = _run_sql(db_path, gold_sql)
    pred_result = _run_sql(db_path, final_sql)
    lines.append("  -- gold SQL 执行结果 --")
    lines.append(_result_str(gold_result))
    lines.append("  -- 模型 SQL 执行结果 --")
    lines.append(_result_str(pred_result))

    ex_mark = "N/A"
    if gold_result is not None and pred_result is not None and not isinstance(gold_result, dict) and not isinstance(pred_result, dict):
        if gold_result.result_rows is None:
            ex_mark = "gold 执行失败，无法判定"
        elif pred_result.result_rows is None:
            ex_mark = "❌ 模型 SQL 执行失败"
        else:
            ex_mark = "✅ 一致（AC）" if set(pred_result.result_rows) == set(gold_result.result_rows) else "❌ 不一致"
    lines.append(f"  EX 判定: {ex_mark}")
    lines.append("")

    # Token / time totals
    lines.append("【总 Token 消耗 / 总耗时】")
    lines.append(f"  Token: {_cost_str(metrics.get('total_llm_cost'))}")
    lines.append(f"  总耗时: {_time_str(metrics.get('total_time'))}")
    lines.append("")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a demo report from the final pipeline snapshot.")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=str(PROJECT_ROOT / "workspace" / "runs" / "bird-dev" / "sql_selection.snapshot.data" / "items.jsonl"),
        help="Path to the final snapshot items.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "workspace" / "runs" / "bird-dev" / "demo_report.txt"),
        help="Output txt path",
    )
    args = parser.parse_args()

    snap_path = Path(args.snapshot)
    if not snap_path.exists():
        raise SystemExit(f"Snapshot not found: {snap_path}")

    lines: List[str] = []
    count = 0
    with open(snap_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lines.extend(render_item(rec))
            count += 1

    report = "\n".join(lines)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print("\n" + "=" * 78)
    print(f"共 {count} 题，报告已写入: {out_path}")


if __name__ == "__main__":
    main()
