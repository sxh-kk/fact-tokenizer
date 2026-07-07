#!/usr/bin/env python3
"""Export selected take JSONL rows from a filtered split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-split", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--include-splits", nargs="+", default=["train", "heldout"])
    parser.add_argument("--include-buckets", nargs="+", default=["fact_main"])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_labels(paths: list[Path]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            take_uid = str(row.get("take_uid") or row.get("take_id") or "")
            if take_uid:
                labels[take_uid] = row
    return labels


def main() -> None:
    args = parse_args()
    payload = json.loads(args.filtered_split.read_text(encoding="utf-8"))
    labels = load_labels(args.labels_jsonl)
    rows = []
    for split in args.include_splits:
        split_payload = payload["splits"].get(split, {})
        for bucket in args.include_buckets:
            for item in split_payload.get(bucket, []):
                take_uid = str(item["take_uid"])
                row = dict(labels.get(take_uid, {}))
                row.update(
                    {
                        "take_uid": take_uid,
                        "filtering_version": payload.get("version", ""),
                        "filtering_split": split,
                        "filtering_bucket": bucket,
                        "filtering_scores": {
                            key: item.get(key)
                            for key in [
                                "prob_tokenizer_main",
                                "prob_loco_aux",
                                "prob_discard",
                                "prob_diagnostic_candidate",
                                "hand_object_contact_score",
                                "phase_diversity_score_v2",
                                "exo_body_visibility_score",
                                "auto_confidence",
                            ]
                            if key in item
                        },
                    }
                )
                rows.append(row)
    rows.sort(key=lambda row: (str(row.get("filtering_split", "")), str(row.get("filtering_bucket", "")), str(row.get("take_uid", ""))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} selected takes to {args.out}")


if __name__ == "__main__":
    main()
