#!/usr/bin/env python3
"""Validate completed gold labels and gate on dual-annotator Cohen kappa."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.gold_annotations import cohens_kappa, validate_gold_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-a", type=Path, required=True)
    parser.add_argument("--annotations-b", type=Path, help="Second file; only dual_annotation rows are compared.")
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--minimum-kappa", type=float, default=0.70)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    rows_a = load_csv(args.annotations_a)
    errors = validate_gold_rows(rows_a, require_complete=not args.allow_incomplete)
    report: dict = {"rows": len(rows_a), "validation_errors": errors, "passed": not errors}
    if args.annotations_b:
        rows_b = load_csv(args.annotations_b)
        errors.extend(validate_gold_rows(rows_b, require_complete=not args.allow_incomplete))
        by_id_b = {row["sample_id"]: row for row in rows_b}
        dual_a = [row for row in rows_a if str(row.get("dual_annotation", "")).lower() in {"1", "true", "yes"}]
        missing = [row["sample_id"] for row in dual_a if row["sample_id"] not in by_id_b]
        if missing:
            errors.append(f"second annotation file is missing {len(missing)} dual samples")
        if not missing and not args.allow_incomplete:
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
