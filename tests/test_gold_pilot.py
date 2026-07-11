from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.materialize_fact_gold300_review import FROZEN_FIELDS as MATERIALIZER_FROZEN_FIELDS
from scripts.prepare_fact_gold300_pilot import (
    ANNOTATION_FIELDS,
    FROZEN_FIELDS,
    prepare_pilot,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _candidates() -> list[dict]:
    return [
        {
            "sample_id": f"sample-{take}-{transition}",
            "take_uid": f"take-{take}",
            "split": "heldout",
            "row_index": take * 4 + transition,
            "timestamp": float(transition),
            "source_dataset": "egoexo",
        }
        for take in range(35)
        for transition in range(4)
    ]


def _gold() -> list[dict]:
    return [
        {"sample_id": "sample-0-0", "take_uid": "take-0"},
        {"sample_id": "other-sample", "take_uid": "take-1"},
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pilot_exact_unique_nonoverlapping_deterministic_and_hash_bound(tmp_path: Path) -> None:
    candidates = _write_jsonl(tmp_path / "candidates.jsonl", _candidates())
    gold = _write_jsonl(tmp_path / "gold.jsonl", _gold())
    guide = tmp_path / "ANNOTATION_GUIDE.zh-CN.md"
    guide.write_text("# guide\n", encoding="utf-8")

    first = tmp_path / "pilot-a"
    second = tmp_path / "pilot-b"
    freeze = prepare_pilot(candidates, gold, guide, first)
    prepare_pilot(candidates, gold, guide, second)

    manifest_path = first / "pilot_frozen.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24
    assert len({row["sample_id"] for row in rows}) == 24
    assert len({row["take_uid"] for row in rows}) == 24
    assert not ({row["sample_id"] for row in rows} & {"sample-0-0", "other-sample"})
    assert not ({row["take_uid"] for row in rows} & {"take-0", "take-1"})
    assert {row["gold_split"] for row in rows} == {"calibration_dev"}
    assert all(row["dual_annotation"] is True for row in rows)
    assert all(row["representation_training_valid"] is False for row in rows)
    # Four transitions use the lower median (timestamp 1.0) within every take.
    assert {row["timestamp"] for row in rows} == {1.0}

    assert manifest_path.read_bytes() == (second / "pilot_frozen.jsonl").read_bytes()
    assert (first / "pilot_annotations.csv").read_bytes() == (second / "pilot_annotations.csv").read_bytes()
    assert freeze["manifest_sha256"] == _sha256(manifest_path)
    assert freeze["outputs"]["pilot_annotations.csv"] == _sha256(first / "pilot_annotations.csv")
    assert freeze["inputs"]["candidates"]["sha256"] == _sha256(candidates)
    assert freeze["inputs"]["gold_manifest"]["sha256"] == _sha256(gold)
    assert freeze["annotation_guide"]["sha256"] == _sha256(guide)
    assert freeze["sample_counts"] == {"calibration_dev": 24}
    assert freeze["usage_policy"] == {
        "formal_evaluation_valid": False,
        "representation_training_valid": False,
        "linear_probe_valid": False,
        "model_selection_valid": False,
        "locked_test_valid": False,
    }

    with (first / "pilot_annotations.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        annotation_rows = list(reader)
    assert tuple(reader.fieldnames or ()) == ANNOTATION_FIELDS
    assert FROZEN_FIELDS == MATERIALIZER_FROZEN_FIELDS
    assert len(annotation_rows) == 24
    assert all(
        not row[field]
        for row in annotation_rows
        for field in ("effect_label", "contact_label", "ambiguous_reason", "annotator_id", "notes")
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_pilot(candidates, gold, guide, first)


def test_pilot_rejects_train_or_locked_candidates(tmp_path: Path) -> None:
    candidates_rows = _candidates()
    candidates_rows[0]["split"] = "train"
    candidates = _write_jsonl(tmp_path / "candidates.jsonl", candidates_rows)
    gold = _write_jsonl(tmp_path / "gold.jsonl", _gold())
    guide = tmp_path / "ANNOTATION_GUIDE.zh-CN.md"
    guide.write_text("# guide\n", encoding="utf-8")
    with pytest.raises(ValueError, match="heldout/non-locked"):
        prepare_pilot(candidates, gold, guide, tmp_path / "pilot")


def test_pilot_cli_help_lists_required_inputs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/prepare_fact_gold300_pilot.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--candidates" in result.stdout
    assert "--gold-manifest" in result.stdout
    assert "--annotation-guide" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--count" in result.stdout
    assert "--seed" in result.stdout
