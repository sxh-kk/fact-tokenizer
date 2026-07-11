#!/usr/bin/env python3
"""Freeze a non-formal, non-locked FACT effect/contact annotation pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid


FROZEN_FIELDS = (
    "sample_id",
    "take_uid",
    "gold_split",
    "timestamp",
    "source_dataset",
    "dual_annotation",
    "representation_training_valid",
)
EDITABLE_FIELDS = (
    "effect_label",
    "contact_label",
    "ambiguous_reason",
    "annotator_id",
    "notes",
)
ANNOTATION_FIELDS = (*FROZEN_FIELDS, *EDITABLE_FIELDS)
SELECTION_ALGORITHM = (
    "exclude every gold sample_id and take_uid; group eligible candidates by take_uid; "
    "rank takes by sha256('fact-gold300-pilot-v1:{seed}:{{take_uid}}'); select one "
    "lower-median transition per take after sorting by (timestamp,row_index,sample_id)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Held-out base JSONL manifest.")
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--annotation-guide", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path, *, label: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {label} at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} row at {path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _identity(row: dict, *, label: str, row_number: int) -> tuple[str, str]:
    sample_id = str(row.get("sample_id", "")).strip()
    take_uid = str(row.get("take_uid", "")).strip()
    if not sample_id or not take_uid:
        raise ValueError(f"{label} row {row_number} needs non-empty sample_id and take_uid")
    return sample_id, take_uid


def validate_gold(rows: list[dict]) -> tuple[set[str], set[str]]:
    sample_ids: list[str] = []
    take_uids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        sample_id, take_uid = _identity(row, label="gold manifest", row_number=row_number)
        sample_ids.append(sample_id)
        take_uids.add(take_uid)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("gold manifest sample_id values must be unique")
    return set(sample_ids), take_uids


def validate_candidates(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    sample_ids: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        sample_id, take_uid = _identity(row, label="candidate manifest", row_number=row_number)
        try:
            timestamp = float(row["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate row {row_number} needs a numeric timestamp") from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"candidate row {row_number} timestamp must be finite")
        try:
            row_index = int(row.get("row_index", row_number - 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate row {row_number} row_index must be an integer") from exc
        if row_index < 0:
            raise ValueError(f"candidate row {row_number} row_index must be non-negative")
        source_dataset = str(row.get("source_dataset", "egoexo")).strip()
        if not source_dataset:
            raise ValueError(f"candidate row {row_number} source_dataset must be non-empty")
        split = str(row.get("split", "")).strip()
        if split not in {"heldout", "calibration_dev"}:
            raise ValueError(
                f"candidate row {row_number} must come from heldout/non-locked data, got split={split!r}"
            )
        sample_ids.append(sample_id)
        normalized.append(
            {
                **row,
                "sample_id": sample_id,
                "take_uid": take_uid,
                "timestamp": timestamp,
                "row_index": row_index,
                "source_dataset": source_dataset,
                "split": split,
            }
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("candidate manifest sample_id values must be unique")
    return normalized


def select_pilot_rows(
    candidate_rows: list[dict],
    gold_rows: list[dict],
    *,
    count: int = 24,
    seed: int = 20260711,
) -> tuple[list[dict], dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    candidates = validate_candidates(candidate_rows)
    gold_sample_ids, gold_take_uids = validate_gold(gold_rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    excluded_sample_count = 0
    excluded_take_count = 0
    for row in candidates:
        if row["sample_id"] in gold_sample_ids:
            excluded_sample_count += 1
            continue
        if row["take_uid"] in gold_take_uids:
            excluded_take_count += 1
            continue
        grouped[row["take_uid"]].append(row)
    if len(grouped) < count:
        raise ValueError(f"need {count} eligible non-gold takes, found {len(grouped)}")

    def take_rank(take_uid: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"fact-gold300-pilot-v1:{seed}:{take_uid}".encode("utf-8")).hexdigest()
        return digest, take_uid

    selected: list[dict] = []
    for take_uid in sorted(grouped, key=take_rank)[:count]:
        take_rows = sorted(
            grouped[take_uid],
            key=lambda row: (row["timestamp"], row["row_index"], row["sample_id"]),
        )
        source = take_rows[(len(take_rows) - 1) // 2]
        selected.append(
            {
                "sample_id": source["sample_id"],
                "take_uid": source["take_uid"],
                "gold_split": "calibration_dev",
                "timestamp": source["timestamp"],
                "source_dataset": source["source_dataset"],
                "dual_annotation": True,
                "representation_training_valid": False,
            }
        )
    selected.sort(key=lambda row: (take_rank(str(row["take_uid"])), str(row["sample_id"])))
    report = {
        "candidate_sample_count": len(candidates),
        "candidate_take_count": len({row["take_uid"] for row in candidates}),
        "excluded_by_gold_sample_id_count": excluded_sample_count,
        "excluded_by_gold_take_uid_count": excluded_take_count,
        "eligible_sample_count": sum(len(rows) for rows in grouped.values()),
        "eligible_take_count": len(grouped),
        "selected_sample_count": len(selected),
        "selected_take_count": len({row["take_uid"] for row in selected}),
    }
    return selected, report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_annotation_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{field: "" for field in EDITABLE_FIELDS}})


def prepare_pilot(
    candidates_path: Path,
    gold_manifest_path: Path,
    annotation_guide_path: Path,
    output_dir: Path,
    *,
    count: int = 24,
    seed: int = 20260711,
) -> dict:
    candidates_path = Path(candidates_path)
    gold_manifest_path = Path(gold_manifest_path)
    annotation_guide_path = Path(annotation_guide_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen pilot assets: {output_dir}")
    if not annotation_guide_path.is_file():
        raise FileNotFoundError(f"missing annotation guide: {annotation_guide_path}")
    candidate_rows = load_jsonl(candidates_path, label="candidate manifest")
    gold_rows = load_jsonl(gold_manifest_path, label="gold manifest")
    selected, counts = select_pilot_rows(candidate_rows, gold_rows, count=count, seed=seed)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False)
    try:
        manifest_path = staging / "pilot_frozen.jsonl"
        annotations_path = staging / "pilot_annotations.csv"
        _write_jsonl(manifest_path, selected)
        _write_annotation_csv(annotations_path, selected)
        manifest_sha256 = sha256_file(manifest_path)
        annotations_sha256 = sha256_file(annotations_path)
        freeze = {
            "schema": "fact-effect-gold-v1",
            "pilot_schema": "fact-effect-gold-pilot-v1",
            "pilot": True,
            "formal_gold": False,
            "seed": int(seed),
            "selection_algorithm": SELECTION_ALGORITHM.format(seed=seed),
            "manifest_sha256": manifest_sha256,
            "sample_counts": {"calibration_dev": count},
            "take_counts": {"calibration_dev": count},
            "dual_annotation_count": count,
            "representation_training_valid": False,
            "usage_policy": {
                "formal_evaluation_valid": False,
                "representation_training_valid": False,
                "linear_probe_valid": False,
                "model_selection_valid": False,
                "locked_test_valid": False,
            },
            "annotation_guide": {
                "path": annotation_guide_path.name,
                "sha256": sha256_file(annotation_guide_path),
            },
            "annotation_templates": {
                "calibration_dev": {
                    "path": annotations_path.name,
                    "rows": count,
                    "template_sha256": annotations_sha256,
                }
            },
            "inputs": {
                "candidates": {
                    "path": str(candidates_path.resolve()),
                    "sha256": sha256_file(candidates_path),
                },
                "gold_manifest": {
                    "path": str(gold_manifest_path.resolve()),
                    "sha256": sha256_file(gold_manifest_path),
                },
                "annotation_guide": {
                    "path": str(annotation_guide_path.resolve()),
                    "sha256": sha256_file(annotation_guide_path),
                },
            },
            "outputs": {
                "pilot_frozen.jsonl": manifest_sha256,
                "pilot_annotations.csv": annotations_sha256,
            },
            "counts": counts,
        }
        (staging / "pilot_freeze.json").write_text(
            json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite frozen pilot assets: {output_dir}")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return freeze


def main() -> None:
    args = parse_args()
    freeze = prepare_pilot(
        args.candidates,
        args.gold_manifest,
        args.annotation_guide,
        args.output_dir,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
