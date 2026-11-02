"""Randomly pick N items from a BIRD dev.json for a small demo run.

Supports:
  - filtering by difficulty (--difficulty "moderate,challenging")
  - excluding already-run ids (--exclude "51,228,1309")
  - spreading picks across distinct databases (--diverse)

Prints the picked items and a comma-separated list of question ids that can be
passed to run_few_shot_preparation.py via --item_ids.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly pick N BIRD items for a demo run.")
    parser.add_argument("--n", type=int, default=3, help="Number of items to pick (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dev_json", type=str, default="data/bird/dev/dev.json", help="Path to the BIRD dev.json")
    parser.add_argument("--difficulty", type=str, default=None, help="Comma-separated difficulties to keep, e.g. 'moderate,challenging' (default: no filter)")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated question ids to exclude, e.g. '51,228,1309'")
    parser.add_argument("--diverse", action="store_true", help="Spread picks across as many distinct db_id as possible")
    args = parser.parse_args()

    data = json.loads(Path(args.dev_json).read_text(encoding="utf-8"))

    # filter by difficulty
    if args.difficulty:
        wanted = {d.strip() for d in args.difficulty.split(",") if d.strip()}
        data = [item for item in data if item.get("difficulty") in wanted]
        print(f"按难度过滤后剩 {len(data)} 题（difficulty in {sorted(wanted)}）")

    # exclude already-run ids
    if args.exclude:
        excluded = {e.strip() for e in args.exclude.split(",") if e.strip()}
        data = [item for item in data if str(item.get("question_id")) not in excluded]
        print(f"排除已跑题 {sorted(excluded, key=int)} 后剩 {len(data)} 题")

    if args.n > len(data):
        raise SystemExit(f"Requested {args.n} items but only {len(data)} remain after filtering")

    rng = random.Random(args.seed)

    if args.diverse:
        # group by db_id, shuffle within each db, then round-robin across dbs
        by_db = defaultdict(list)
        for item in data:
            by_db[item.get("db_id")].append(item)
        for items in by_db.values():
            rng.shuffle(items)
        dbs = list(by_db.keys())
        rng.shuffle(dbs)
        picked = []
        idx = 0
        while len(picked) < args.n:
            progressed = False
            for db in dbs:
                if idx < len(by_db[db]):
                    picked.append(by_db[db][idx])
                    progressed = True
                    if len(picked) >= args.n:
                        break
            if not progressed:
                break
            idx += 1
    else:
        picked = rng.sample(data, args.n)

    print(f"随机抽取 {len(picked)} 题（seed={args.seed}）：")
    print("=" * 72)
    ids = []
    for item in picked:
        qid = item.get("question_id")
        ids.append(str(qid))
        print(f"[{qid}] db={item.get('db_id')}  难度={item.get('difficulty')}")
        print(f"    题目：{item.get('question')}")
        print("-" * 72)

    print()
    print("要传给 pipeline 的 --item_ids（逗号分隔，直接复制）：")
    print(",".join(ids))


if __name__ == "__main__":
    main()
