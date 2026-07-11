#!/usr/bin/env python3
"""Join the 176 human take-quality labels and optional derived sampling weights."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import (  # noqa: E402
    EffectCapability,
    read_manifest_jsonl,
    write_manifest_jsonl,
)


HUMAN_BUCKETS = {"tokenizer_main", "diagnostic_candidate", "loco_aux", "discard"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--human-quality-csv", type=Path, required=True)
    parser.add_argument("--derived-weight-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-noncanonical-human-count", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_unique_csv(path: Path, key: str = "take_uid") -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        value = str(row.get(key, "")).strip()
        if not value or value in result:
            raise ValueError(f"missing/duplicate {key} at {path}:{row_number}")
        result[value] = row
    return result


def main() -> None:
    args = parse_args()
    records = read_manifest_jsonl(args.manifest)
    human = read_unique_csv(args.human_quality_csv)
    if not args.allow_noncanonical_human_count and len(human) != 176:
        raise ValueError(f"canonical human quality asset must contain 176 takes, got {len(human)}")
    invalid_buckets = sorted(
        {
            str(row.get("usable_for", "")).strip()
            for row in human.values()
            if str(row.get("usable_for", "")).strip() not in HUMAN_BUCKETS
        }
    )
    if invalid_buckets:
        raise ValueError(f"unexpected human usable_for buckets: {invalid_buckets}")
    derived = read_unique_csv(args.derived_weight_csv) if args.derived_weight_csv else {}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    human_sha = sha256_file(args.human_quality_csv)
    derived_sha = sha256_file(args.derived_weight_csv) if args.derived_weight_csv else None
    updated = []
    matched_human: set[str] = set()
    matched_derived: set[str] = set()
    for record in records:
        human_row = human.get(record.take_uid)
        weight_row = derived.get(record.take_uid)
        capabilities = dict(record.capability_validity)
        capabilities[EffectCapability.TAKE_QUALITY] = human_row is not None
        refs = dict(record.annotation_refs)
        provenance = dict(record.provenance)
        bucket = "unlabeled"
        if human_row is not None:
            matched_human.add(record.take_uid)
            bucket = str(human_row["usable_for"]).strip()
            refs["human_take_quality"] = {
                "take_uid": record.take_uid,
                "usable_for": bucket,
                "take_relevance": human_row.get("take_relevance"),
                "ego_hand_visibility": human_row.get("ego_hand_visibility"),
                "object_interaction": human_row.get("object_interaction"),
                "exo_body_visibility": human_row.get("exo_body_visibility"),
                "phase_diversity": human_row.get("phase_diversity"),
                "scene_only_risk": human_row.get("scene_only_risk"),
                "confidence": human_row.get("confidence"),
                "source_sha256": human_sha,
                "effect_label": False,
                "sampling_stratum_only": True,
            }
        quality_weight = 1.0
        if weight_row is not None:
            matched_derived.add(record.take_uid)
            quality_weight = float(weight_row["sample_weight"])
            if quality_weight < 0:
                raise ValueError(f"negative sample weight for take {record.take_uid}")
            refs["derived_sampling_weight"] = {
                "bucket": weight_row.get("bucket"),
                "sample_weight": quality_weight,
                "source_sha256": derived_sha,
                "human_label": False,
                "effect_target": False,
            }
        else:
            provenance["derived_quality_weight_valid"] = False
        updated.append(
            replace(
                record,
                capability_validity=capabilities,
                quality_bucket=bucket,
                quality_weight=quality_weight,
                annotation_refs=refs,
                provenance=provenance,
            )
        )
    manifest_path = output / "effect_manifest_take_quality_indexed.jsonl"
    write_manifest_jsonl(manifest_path, updated)
    manifest_takes = {record.take_uid for record in records}
    report = {
        "schema": "fact-take-quality-index-v1",
        "samples": len(records),
        "takes": len(manifest_takes),
        "human_asset_takes": len(human),
        "human_matched_takes": len(matched_human),
        "human_unmatched_takes": sorted(set(human) - manifest_takes),
        "derived_asset_takes": len(derived),
        "derived_matched_takes": len(matched_derived),
        "human_quality_sha256": human_sha,
        "derived_weight_sha256": derived_sha,
        "manifest": str(manifest_path),
        "quality_is_effect_target": False,
        "quality_usage": ["stratification", "sampling"],
    }
    (output / "take_quality_index_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
