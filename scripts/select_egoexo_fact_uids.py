#!/usr/bin/env python3
"""Select a tiny set of EgoExo4D takes suitable for FACT tokenizer debugging."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


def load_takes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_fact_candidate(take: dict) -> bool:
    videos = take.get("frame_aligned_videos") or {}
    best_exo = take.get("best_exo")
    return bool(
        not take.get("is_dropped")
        and take.get("validated", True)
        and videos.get("aria01", {}).get("rgb", {}).get("relative_path")
        and best_exo
        and videos.get(best_exo, {}).get("0", {}).get("relative_path")
    )


def diverse_round_robin(takes: list[dict], max_takes: int) -> list[dict]:
    """Pick takes by cycling parent tasks, universities, and task names."""
    parent_buckets: dict[str, dict[tuple[str, str], deque[dict]]] = defaultdict(lambda: defaultdict(deque))
    for take in takes:
        parent = take.get("parent_task_name") or "unknown_parent"
        subkey = (take.get("university_name") or "unknown_university", take.get("task_name") or "unknown_task")
        parent_buckets[parent][subkey].append(take)

    selected = []
    parent_state = {
        parent: deque(sorted(key for key, bucket in sub_buckets.items() if bucket))
        for parent, sub_buckets in parent_buckets.items()
    }
    parents = deque(sorted(parent_buckets))
    while parents and len(selected) < max_takes:
        parent = parents.popleft()
        sub_buckets = parent_buckets[parent]
        subkeys = parent_state[parent]
        while subkeys:
            key = subkeys.popleft()
            bucket = sub_buckets[key]
            if bucket:
                selected.append(bucket.popleft())
                if bucket:
                    subkeys.append(key)
                break
        if any(bucket for bucket in sub_buckets.values()):
            parents.append(parent)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoexo-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--output-uids", type=Path, default=Path("data/egoexo4d/fact_debug/uids.txt"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/egoexo4d/fact_debug/selected_takes.jsonl"))
    parser.add_argument("--max-takes", type=int, default=3)
    parser.add_argument("--university", default=None)
    parser.add_argument("--parent-task", default=None)
    parser.add_argument("--min-duration-sec", type=float, default=0.0)
    parser.add_argument("--diverse", action="store_true")
    args = parser.parse_args()

    takes = load_takes(args.egoexo_root / "takes.json")
    candidates = []
    for take in takes:
        if not is_fact_candidate(take):
            continue
        if args.university and take.get("university_name") != args.university:
            continue
        if args.parent_task and take.get("parent_task_name") != args.parent_task:
            continue
        if args.min_duration_sec and float(take.get("duration_sec") or 0.0) < args.min_duration_sec:
            continue
        candidates.append(take)

    selected = diverse_round_robin(candidates, args.max_takes) if args.diverse else candidates[: args.max_takes]

    if not selected:
        raise RuntimeError("No FACT-ready EgoExo4D takes matched the filters.")

    args.output_uids.parent.mkdir(parents=True, exist_ok=True)
    args.output_uids.write_text("\n".join(take["take_uid"] for take in selected) + "\n", encoding="utf-8")
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for take in selected:
            videos = take["frame_aligned_videos"]
            ego = videos["aria01"]["rgb"]
            exo = videos[take["best_exo"]]["0"]
            row = {
                "take_uid": take["take_uid"],
                "take_name": take["take_name"],
                "root_dir": take["root_dir"],
                "duration_sec": take.get("duration_sec"),
                "task_start_sec": take.get("task_start_sec", 0.0),
                "task_end_sec": take.get("task_end_sec", take.get("duration_sec")),
                "parent_task_name": take.get("parent_task_name"),
                "task_name": take.get("task_name"),
                "university_name": take.get("university_name"),
                "ego_camera": "aria01",
                "ego_stream": "rgb",
                "ego_relative_path": ego["relative_path"],
                "exo_camera": take["best_exo"],
                "exo_relative_path": exo["relative_path"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Selected {len(selected)} takes")
    print(f"Wrote {args.output_uids}")
    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
