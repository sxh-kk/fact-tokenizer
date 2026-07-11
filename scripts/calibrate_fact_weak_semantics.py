#!/usr/bin/env python3
"""Calibrate the frozen deterministic weak semantic map on the 60-sample dev set."""

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

from fact_tokenizer.effect_targets import (  # noqa: E402
    WEAK_VERB_MAP_VERSION,
    deterministic_weak_semantic_labels,
)
from fact_tokenizer.gold_annotations import validate_gold_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-index-jsonl", type=Path, required=True)
    parser.add_argument("--gold60-csv", type=Path, required=True)
    parser.add_argument("--gold-freeze", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.80)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_precision <= 1.0:
        raise ValueError("minimum precision must be between zero and one")
    weak: dict[str, dict] = {}
    with args.weak_index_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in weak:
                raise ValueError(f"invalid/duplicate weak sample at line {line_number}")
            weak[sample_id] = row
    with args.gold60_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        gold = list(csv.DictReader(handle))
    errors = validate_gold_rows(gold, require_complete=True)
    if errors:
        raise ValueError(f"gold60 validation failed: {errors[:3]}")
    if len(gold) != 60 or {str(row.get("gold_split")) for row in gold} != {"calibration_dev"}:
        raise ValueError("weak calibration requires exactly the frozen 60 calibration_dev rows")
    if len({row["sample_id"] for row in gold}) != 60:
        raise ValueError("gold60 contains duplicate sample IDs")
    freeze = json.loads(args.gold_freeze.read_text(encoding="utf-8"))
    frozen_manifest = args.gold_freeze.parent / "gold300_frozen.jsonl"
    if sha256_file(frozen_manifest) != freeze.get("manifest_sha256"):
        raise ValueError("gold frozen manifest hash differs from the gold freeze")
    frozen_dev_ids = {
        str(row["sample_id"])
        for row in (
            json.loads(line)
            for line in frozen_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("gold_split") == "calibration_dev"
    }
    if frozen_dev_ids != {row["sample_id"] for row in gold}:
        raise ValueError("gold60 IDs differ from the frozen calibration-dev split")
    missing = sorted({row["sample_id"] for row in gold} - set(weak))
    if missing:
        raise ValueError(f"weak index is missing {len(missing)} dev samples, including {missing[:3]}")

    predictions = []
    effect_correct = contact_correct = mapped = 0
    for gold_row in gold:
        source = weak[gold_row["sample_id"]]
        prediction = deterministic_weak_semantic_labels(
            source.get("atomic_descriptions", []),
            source.get("phase_segments", []),
        )
        is_mapped = bool(prediction["phase_valid"] and prediction["contact_valid"])
        mapped += int(is_mapped)
        effect_match = is_mapped and prediction["phase_name"] == gold_row["effect_label"]
        contact_match = is_mapped and prediction["contact_name"] == gold_row["contact_label"]
        effect_correct += int(effect_match)
        contact_correct += int(contact_match)
        predictions.append(
            {
                "sample_id": gold_row["sample_id"],
                "mapped": is_mapped,
                "effect_prediction": prediction["phase_name"],
                "contact_prediction": prediction["contact_name"],
                "effect_match": effect_match,
                "contact_match": contact_match,
                "matched_phrase": prediction["matched_phrase"],
            }
        )
    effect_precision = effect_correct / 60.0
    contact_precision = contact_correct / 60.0
    minimum_measured = min(effect_precision, contact_precision)
    sample_list_text = "\n".join(sorted(row["sample_id"] for row in gold)) + "\n"
    report = {
        "schema": "fact-weak-semantic-calibration-v1",
        "mapping_version": WEAK_VERB_MAP_VERSION,
        "dev_sample_count": 60,
        "minimum_precision": args.minimum_precision,
        "effect_precision_all_dev": effect_precision,
        "contact_precision_all_dev": contact_precision,
        "minimum_measured_precision": minimum_measured,
        "mapped_count": mapped,
        "coverage": mapped / 60.0,
        "gate_passed": minimum_measured >= args.minimum_precision,
        "weak_index_sha256": sha256_file(args.weak_index_jsonl),
        "gold60_sha256": sha256_file(args.gold60_csv),
        "gold_freeze_sha256": sha256_file(args.gold_freeze),
        "gold_manifest_sha256": sha256_file(frozen_manifest),
        "dev_sample_ids_sha256": hashlib.sha256(sample_list_text.encode("utf-8")).hexdigest(),
        "predictions": predictions,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.output_report)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
