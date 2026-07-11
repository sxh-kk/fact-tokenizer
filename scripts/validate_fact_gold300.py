#!/usr/bin/env python3
"""Validate completed gold labels and gate on dual-annotator Cohen kappa."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.gold_annotations import cohens_kappa, validate_gold_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-a",
        type=Path,
        action="append",
        required=True,
        help="Annotator-A canonical CSV. Repeat to validate physically separated non-locked and locked files together.",
    )
    parser.add_argument(
        "--annotations-b",
        type=Path,
        action="append",
        default=[],
        help="Annotator-B dual-only canonical CSV. Repeat to aggregate physically separated packs.",
    )
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--minimum-kappa", type=float, default=0.70)
    parser.add_argument("--expected-a-count", type=int)
    parser.add_argument("--expected-dual-count", type=int, default=60)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_many(paths: list[Path]) -> list[dict[str, str]]:
    return [row for path in paths for row in load_csv(path)]


def normalized_frozen(field: str, value: object) -> object:
    if field == "timestamp":
        return float(value)
    if field in {"dual_annotation", "representation_training_valid"}:
        return str(value).strip().lower() in {"true", "1", "yes"}
    return str(value)


def main() -> None:
    args = parse_args()
    rows_a = load_many(args.annotations_a)
    errors = validate_gold_rows(rows_a, require_complete=not args.allow_incomplete)
    report: dict = {
        "rows": len(rows_a),
        "annotation_files_a": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.annotations_a
        ],
        "validation_errors": errors,
        "passed": not errors,
    }
    if args.expected_a_count is not None and len(rows_a) != args.expected_a_count:
        errors.append(f"annotator A must contain exactly {args.expected_a_count} rows")
    if args.annotations_b:
        rows_b = load_many(args.annotations_b)
        errors.extend(validate_gold_rows(rows_b, require_complete=not args.allow_incomplete))
        by_id_b = {row["sample_id"]: row for row in rows_b}
        dual_a = [row for row in rows_a if str(row.get("dual_annotation", "")).lower() in {"1", "true", "yes"}]
        missing = [row["sample_id"] for row in dual_a if row["sample_id"] not in by_id_b]
        dual_ids = {row["sample_id"] for row in dual_a}
        unexpected = sorted(set(by_id_b) - dual_ids)
        if len(dual_a) != args.expected_dual_count:
            errors.append(
                f"annotator A must contain exactly {args.expected_dual_count} dual rows; got {len(dual_a)}"
            )
        if len(rows_b) != args.expected_dual_count:
            errors.append(
                f"annotator B must contain exactly {args.expected_dual_count} rows; got {len(rows_b)}"
            )
        if missing:
            errors.append(f"second annotation file is missing {len(missing)} dual samples")
        if unexpected:
            errors.append(f"second annotation file contains {len(unexpected)} non-dual samples")
        frozen_fields = (
            "sample_id",
            "take_uid",
            "gold_split",
            "timestamp",
            "source_dataset",
            "dual_annotation",
            "representation_training_valid",
        )
        if not missing and not unexpected:
            for row in dual_a:
                other = by_id_b[row["sample_id"]]
                for field in frozen_fields:
                    if normalized_frozen(field, row.get(field, "")) != normalized_frozen(
                        field, other.get(field, "")
                    ):
                        errors.append(
                            f"annotator B changed frozen field {field} for sample_id={row['sample_id']}"
                        )
        annotator_ids_a = {str(row.get("annotator_id", "")).strip() for row in rows_a}
        annotator_ids_b = {str(row.get("annotator_id", "")).strip() for row in rows_b}
        if not args.allow_incomplete and len(annotator_ids_a) == 1 and annotator_ids_a == annotator_ids_b:
            errors.append("annotator A and B must use different annotator_id values")
        report["annotation_files_b"] = [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.annotations_b
        ]
        if not errors and not args.allow_incomplete:
            effect_kappa = cohens_kappa(
                [row["effect_label"] for row in dual_a],
                [by_id_b[row["sample_id"]]["effect_label"] for row in dual_a],
            )
            contact_kappa = cohens_kappa(
                [row["contact_label"] for row in dual_a],
                [by_id_b[row["sample_id"]]["contact_label"] for row in dual_a],
            )
            report.update(
                {
                    "dual_rows": len(dual_a),
                    "effect_cohens_kappa": effect_kappa,
                    "contact_cohens_kappa": contact_kappa,
                    "minimum_kappa": args.minimum_kappa,
                }
            )
            if min(effect_kappa, contact_kappa) < args.minimum_kappa:
                errors.append("Cohen kappa is below the guide-revision gate")
    report["validation_errors"] = errors
    report["passed"] = not errors
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
