from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


FIELDS = (
    "sample_id",
    "take_uid",
    "gold_split",
    "timestamp",
    "source_dataset",
    "dual_annotation",
    "representation_training_valid",
    "effect_label",
    "contact_label",
    "ambiguous_reason",
    "annotator_id",
    "notes",
)


def make_rows(start: int, count: int, split: str, annotator_id: str) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample_{index}",
            "take_uid": f"take_{index // 2}",
            "gold_split": split,
            "timestamp": float(index),
            "source_dataset": "egoexo",
            "dual_annotation": True,
            "representation_training_valid": False,
            "effect_label": "no_effect" if index % 2 == 0 else "acquire_control",
            "contact_label": "none" if index % 2 == 0 else "onset",
            "ambiguous_reason": "",
            "annotator_id": annotator_id,
            "notes": "",
        }
        for index in range(start, start + count)
    ]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_kappa_gate_aggregates_physically_separated_dual_files(tmp_path: Path) -> None:
    a_nonlocked = tmp_path / "a_nonlocked.csv"
    a_locked = tmp_path / "a_locked.csv"
    b_nonlocked = tmp_path / "b_nonlocked.csv"
    b_locked = tmp_path / "b_locked.csv"
    write_rows(a_nonlocked, make_rows(0, 40, "calibration_dev", "ann_a01"))
    write_rows(a_locked, make_rows(40, 20, "locked_test", "ann_a01"))
    write_rows(b_nonlocked, make_rows(0, 40, "calibration_dev", "ann_b01"))
    write_rows(b_locked, make_rows(40, 20, "locked_test", "ann_b01"))
    report_path = tmp_path / "kappa_report.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "validate_fact_gold300.py"),
            "--annotations-a",
            str(a_nonlocked),
            "--annotations-a",
            str(a_locked),
            "--annotations-b",
            str(b_nonlocked),
            "--annotations-b",
            str(b_locked),
            "--expected-a-count",
            "60",
            "--expected-dual-count",
            "60",
            "--output-report",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["dual_rows"] == 60
    assert report["effect_cohens_kappa"] == 1.0
    assert report["contact_cohens_kappa"] == 1.0
    assert len(report["annotation_files_a"]) == 2
    assert len(report["annotation_files_b"]) == 2
