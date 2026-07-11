#!/usr/bin/env python3
"""Freeze the common P0/P2/P3/P4 allowlist and valid P3 donor index."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_manifest import read_manifest_jsonl  # noqa: E402
from fact_tokenizer.paired_eligibility import (  # noqa: E402
    build_paired_control_eligibility,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Aligned effect manifest JSONL.")
    parser.add_argument(
        "--phase-index",
        type=Path,
        required=True,
        help="Aligned NPY/NPZ or sample_id-keyed CSV/JSON/JSONL phase labels.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-label-field", default="phase_label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--excluded-takes",
        type=Path,
        required=True,
        help="Frozen 47-take gold representation-training exclusion file.",
    )
    parser.add_argument("--gold-freeze", type=Path, required=True)
    return parser.parse_args()


def _phase_from_row(row: Mapping[str, Any], preferred: str) -> Any:
    for key in (preferred, "phase_label", "phase", "label", "step"):
        if row.get(key) not in (None, ""):
            return row[key]
    segments = row.get("phase_segments")
    if isinstance(segments, list) and segments:
        first = segments[0]
        if isinstance(first, Mapping):
            for key in ("step", "label", "step_name", "step_description"):
                if first.get(key) not in (None, ""):
                    return first[key]
        elif first not in (None, ""):
            return first
    return ""


def _table_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"phase index line {line_number} is not an object")
                rows.append(row)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("samples", payload.get("rows", payload)))
            if isinstance(payload, dict):
                payload = [
                    (
                        {**value, "sample_id": key}
                        if isinstance(value, dict)
                        else {"sample_id": key, "phase_label": value}
                    )
                    for key, value in payload.items()
                ]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("phase JSON must contain a row list or sample_id-keyed object")
        return [dict(row) for row in payload]
    raise ValueError(f"unsupported phase table format: {path}")


def load_phase_labels(path: Path, sample_ids: list[str], preferred_field: str) -> tuple[list[Any], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 1 or len(values) != len(sample_ids):
            raise ValueError("aligned phase NPY must be a vector with one value per manifest row")
        return values.tolist(), {"format": "aligned_npy", "provided_rows": len(values), "extra_rows": 0}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            label_key = preferred_field if preferred_field in payload.files else "phase_label"
            if label_key not in payload.files:
                raise ValueError(f"phase NPZ lacks {preferred_field!r} or 'phase_label'")
            labels = np.asarray(payload[label_key])
            if "sample_id" in payload.files:
                table_ids = np.asarray(payload["sample_id"]).astype(str)
                if labels.ndim != 1 or table_ids.ndim != 1 or len(labels) != len(table_ids):
                    raise ValueError("phase NPZ sample_id and phase labels must be aligned vectors")
                if len(set(table_ids.tolist())) != len(table_ids):
                    raise ValueError("phase NPZ contains duplicate sample IDs")
                mapping = {sample_id: label for sample_id, label in zip(table_ids, labels)}
                return [mapping.get(sample_id, "") for sample_id in sample_ids], {
                    "format": "sample_id_keyed_npz",
                    "provided_rows": len(labels),
                    "extra_rows": len(set(mapping) - set(sample_ids)),
                }
            if labels.ndim != 1 or len(labels) != len(sample_ids):
                raise ValueError("aligned phase NPZ must have one phase label per manifest row")
            return labels.tolist(), {"format": "aligned_npz", "provided_rows": len(labels), "extra_rows": 0}

    rows = _table_rows(path)
    mapping: dict[str, Any] = {}
    for line_number, row in enumerate(rows, start=2 if suffix == ".csv" else 1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"phase index row {line_number} is missing sample_id")
        if sample_id in mapping:
            raise ValueError(f"duplicate phase sample_id={sample_id!r}")
        mapping[sample_id] = _phase_from_row(row, preferred_field)
    return [mapping.get(sample_id, "") for sample_id in sample_ids], {
        "format": f"sample_id_keyed_{suffix.removeprefix('.')}",
        "provided_rows": len(mapping),
        "extra_rows": len(set(mapping) - set(sample_ids)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen eligibility: {args.output_dir}")
    records = read_manifest_jsonl(args.manifest)
    excluded_takes = {
        line.strip()
        for line in args.excluded_takes.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if len(excluded_takes) != 47:
        raise ValueError(f"paired eligibility requires exactly 47 excluded gold takes, got {len(excluded_takes)}")
    gold_freeze = json.loads(args.gold_freeze.read_text(encoding="utf-8"))
    exclusion_contract = gold_freeze.get("representation_train_excluded_takes", {})
    if (
        exclusion_contract.get("takes") != 47
        or exclusion_contract.get("sha256") != sha256_file(args.excluded_takes)
        or exclusion_contract.get("path") != args.excluded_takes.name
    ):
        raise ValueError("excluded-takes file differs from the gold freeze contract")
    gold_manifest_path = args.gold_freeze.parent / "gold300_frozen.jsonl"
    if sha256_file(gold_manifest_path) != gold_freeze.get("manifest_sha256"):
        raise ValueError("gold manifest differs from gold freeze")
    gold_rows = [
        json.loads(line)
        for line in gold_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    probe_rows = [row for row in gold_rows if row.get("gold_split") == "probe_train"]
    if len(probe_rows) != 140:
        raise ValueError(f"gold manifest must contain exactly 140 probe-train rows, got {len(probe_rows)}")
    probe_takes = {str(row["take_uid"]) for row in probe_rows}
    if probe_takes != excluded_takes:
        raise ValueError("excluded takes are not exactly the frozen probe-train takes")
    missing_excluded = sorted(excluded_takes - {record.take_uid for record in records})
    if missing_excluded:
        raise ValueError(f"manifest is missing excluded gold takes: {missing_excluded[:3]}")
    records = [
        replace(record, training_valid=False)
        if record.take_uid in excluded_takes
        else record
        for record in records
    ]
    sample_ids = [record.sample_id for record in records]
    phase_labels, phase_report = load_phase_labels(
        args.phase_index,
        sample_ids,
        args.phase_label_field,
    )
    arrays, report = build_paired_control_eligibility(records, phase_labels, seed=args.seed)

    staging = args.output_dir.with_name(f".{args.output_dir.name}.staging-{uuid.uuid4().hex[:12]}")
    staging.mkdir(parents=True)
    try:
        npz_path = staging / "paired_control_eligibility.npz"
        np.savez_compressed(npz_path, **arrays)
        np.save(staging / "eligible_sample_id.npy", arrays["sample_id"])
        np.save(staging / "eligible_manifest_index.npy", arrays["manifest_index"])
        np.save(staging / "p3_donor_index.npy", arrays["donor_index"])
        (staging / "eligible_sample_ids.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in arrays["sample_id"].astype(str)),
            encoding="utf-8",
        )
        artifact_names = (
            "paired_control_eligibility.npz",
            "eligible_sample_id.npy",
            "eligible_manifest_index.npy",
            "p3_donor_index.npy",
            "eligible_sample_ids.txt",
        )
        report.update(
            {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256_file(args.manifest),
                "phase_index": str(args.phase_index.resolve()),
                "phase_index_sha256": sha256_file(args.phase_index),
                "phase_index_summary": phase_report,
                "excluded_takes": str(args.excluded_takes.resolve()),
                "excluded_takes_sha256": sha256_file(args.excluded_takes),
                "excluded_take_count": len(excluded_takes),
                "gold_freeze_sha256": sha256_file(args.gold_freeze),
                "gold_manifest_sha256": sha256_file(gold_manifest_path),
                "artifacts_sha256": {
                    name: sha256_file(staging / name)
                    for name in artifact_names
                },
                "usage": {
                    "all_controls_allowlist": "eligible_sample_id.npy",
                    "all_controls_source_rows": "eligible_manifest_index.npy",
                    "p3_donor_index": "p3_donor_index.npy",
                    "p3_donor_index_basis": "eligible_sample_id.npy order",
                },
            }
        )
        (staging / "eligibility_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
