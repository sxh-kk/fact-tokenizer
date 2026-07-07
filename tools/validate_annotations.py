#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


VALID_VALUES = {
    "take_relevance": {
        "A_interaction_rich",
        "B_loco_body",
        "C_active_view_only",
        "D_scene_only",
        "E_fine_dexterous",
        "F_bad_or_unclear",
        "F_uncertain",
    },
    "ego_hand_visibility": {"0", "1", "2"},
    "exo_body_visibility": {"0", "1", "2"},
    "object_interaction": {"0", "1", "2"},
    "phase_diversity": {"0", "1", "2"},
    "usable_for": {"tokenizer_main", "loco_aux", "discard", "diagnostic_candidate"},
}

REQUIRED_COLUMNS = ["take_uid", *VALID_VALUES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manually filled take annotation CSV files.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--allow-empty", action="store_true", help="Only validate rows with at least one annotation value.")
    return parser.parse_args()


def validate_row(row: dict[str, str], row_number: int, allow_empty: bool) -> list[str]:
    errors: list[str] = []
    has_any_annotation = any(str(row.get(column, "")).strip() for column in VALID_VALUES)
    if allow_empty and not has_any_annotation:
        return errors
    for column, allowed in VALID_VALUES.items():
        value = str(row.get(column, "")).strip()
        if not value:
            errors.append(f"row {row_number}: missing {column}")
        elif value not in allowed:
            errors.append(f"row {row_number}: invalid {column}={value!r}")
    relevance = str(row.get("take_relevance", "")).strip()
    usable_for = str(row.get("usable_for", "")).strip()
    ego_hand = str(row.get("ego_hand_visibility", "")).strip()
    exo_body = str(row.get("exo_body_visibility", "")).strip()
    object_interaction = str(row.get("object_interaction", "")).strip()
    phase_diversity = str(row.get("phase_diversity", "")).strip()
    if usable_for == "tokenizer_main" and relevance != "A_interaction_rich":
        errors.append(f"row {row_number}: tokenizer_main conflicts with take_relevance={relevance!r}")
    if usable_for == "tokenizer_main" and ego_hand == "0" and object_interaction == "0":
        errors.append(f"row {row_number}: tokenizer_main requires hand or object interaction signal")
    if usable_for == "loco_aux" and exo_body == "0" and phase_diversity == "0":
        errors.append(f"row {row_number}: loco_aux requires exo body visibility or phase diversity")
    if relevance == "D_scene_only" and usable_for in {"tokenizer_main", "loco_aux"}:
        errors.append(f"row {row_number}: scene-only row cannot be main training data")
    return errors


def main() -> None:
    args = parse_args()
    with args.annotations.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns: {', '.join(missing)}")
        rows = list(reader)
    take_counts = Counter(str(row.get("take_uid", "")).strip() for row in rows)
    errors = [f"duplicate take_uid={take!r}" for take, count in take_counts.items() if take and count > 1]
    errors.extend("missing take_uid" for take, count in take_counts.items() if not take for _ in range(count))
    for index, row in enumerate(rows, start=2):
        errors.extend(validate_row(row, index, args.allow_empty))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(rows)} annotation rows from {args.annotations}")


if __name__ == "__main__":
    main()
