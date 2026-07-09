#!/usr/bin/env python3
"""Export the filtering_v2 active-review CSV for focused human calibration."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import read_csv, write_csv


ANNOTATION_COLUMNS = [
    "take_relevance",
    "ego_hand_visibility",
    "exo_body_visibility",
    "object_interaction",
    "phase_diversity",
    "scene_only_risk",
    "ego_exo_sync_quality",
    "usable_for",
    "confidence",
    "reason",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-review", type=int, default=150)
    parser.add_argument("--random-audit-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.scores)
    rng = random.Random(args.seed)
    review_rows = [row for row in rows if str(row.get("needs_human_review", "0")) == "1"]
    review_rows.sort(key=lambda row: float(row.get("auto_review_priority") or 0.0), reverse=True)
    selected = review_rows[: args.max_review]
    selected_ids = {row["take_uid"] for row in selected}

    audit_count = max(0, int(round(args.max_review * args.random_audit_fraction)))
    if audit_count:
        remaining = [row for row in rows if row.get("take_uid") not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[:audit_count])

    output = []
    for index, row in enumerate(selected, 1):
        output.append(
            {
                "review_id": f"{index:04d}",
                "take_uid": row.get("take_uid", ""),
                "contact_sheet_path": row.get("contact_sheet_path", ""),
                "auto_take_relevance": row.get("auto_take_relevance", ""),
                "auto_usable_for": row.get("auto_usable_for", ""),
                "auto_confidence": row.get("auto_confidence", ""),
                "auto_disagreement_score": row.get("auto_disagreement_score", ""),
                "auto_review_priority": row.get("auto_review_priority", ""),
                "review_reasons": row.get("review_reasons", ""),
                "prob_tokenizer_main": row.get("prob_tokenizer_main", ""),
                "prob_loco_aux": row.get("prob_loco_aux", ""),
                "prob_discard": row.get("prob_discard", ""),
                "prob_diagnostic_candidate": row.get("prob_diagnostic_candidate", ""),
                "hand_object_contact_score": row.get("hand_object_contact_score", ""),
                "exo_body_visibility_score": row.get("exo_body_visibility_score", ""),
                "phase_diversity_score_v2": row.get("phase_diversity_score_v2", ""),
                "scene_only_score": row.get("scene_only_score", ""),
                "ego_hand_visibility_prob": row.get("ego_hand_visibility_prob", ""),
                "object_presence_score": row.get("object_presence_score", ""),
                "object_motion_score": row.get("object_motion_score", ""),
                "body_phase_diversity_score": row.get("body_phase_diversity_score", ""),
                "vlm_usable_for": row.get("vlm_usable_for", ""),
                "vlm_confidence": row.get("vlm_confidence", ""),
                "vlm_reason": row.get("vlm_reason", ""),
                "parent_task_name": row.get("parent_task_name", ""),
                "task_name": row.get("task_name", ""),
                **{column: "" for column in ANNOTATION_COLUMNS},
            }
        )
    write_csv(args.out, output)
    print(f"Saved {len(output)} active-review rows to {args.out}")


if __name__ == "__main__":
    main()
