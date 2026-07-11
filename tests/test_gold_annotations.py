from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fact_tokenizer.gold_annotations import (
    build_gold_pack,
    cohens_kappa,
    validate_gold_rows,
    write_gold_pack,
)


def records(prefix: str, take_count: int, per_take: int) -> list[dict]:
    return [
        {
            "sample_id": f"{prefix}_{take}:{sample}.000",
            "take_uid": f"{prefix}_{take}",
            "timestamp": float(sample),
        }
        for take in range(take_count)
        for sample in range(per_take)
    ]


def test_build_gold300_exact_counts_excludes_diagnostics_and_freezes(tmp_path: Path) -> None:
    diagnostic = "train_0:0.000"
    required_locked = [f"locked_{index}" for index in range(12)]
    pack = build_gold_pack(
        records("train", 50, 4),
        records("dev", 25, 4),
        records("locked", 73, 8),
        [diagnostic],
        required_locked_takes=required_locked,
    )
    assert len(pack) == 300
    assert diagnostic not in {row["sample_id"] for row in pack}
    assert Counter(row["gold_split"] for row in pack) == {
        "probe_train": 140,
        "calibration_dev": 60,
        "locked_test": 100,
    }
    assert sum(row["dual_annotation"] for row in pack) == 60
    assert all(not row["representation_training_valid"] for row in pack)
    locked_takes = {row["take_uid"] for row in pack if row["gold_split"] == "locked_test"}
    assert set(required_locked).issubset(locked_takes)
    metadata = write_gold_pack(tmp_path, pack, 20260711)
    assert metadata["sample_counts"]["locked_test"] == 100
    assert len(metadata["manifest_sha256"]) == 64
    assert not (tmp_path / "gold300_annotations.csv").exists()
    split_files = {
        "probe_train": ("gold140_probe_train_annotations.csv", 140),
        "calibration_dev": ("gold60_calibration_dev_annotations.csv", 60),
        "locked_test": ("gold100_locked_test_annotations.csv", 100),
    }
    for split, (filename, expected) in split_files.items():
        with (tmp_path / filename).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected
        assert {row["gold_split"] for row in rows} == {split}
        assert metadata["annotation_templates"][split]["rows"] == expected


def test_gold_validation_and_kappa_gate() -> None:
    valid = {
        "sample_id": "s",
        "representation_training_valid": "false",
        "effect_label": "acquire_control",
        "contact_label": "onset",
        "ambiguous_reason": "",
    }
    assert validate_gold_rows([valid]) == []
    invalid = {**valid, "representation_training_valid": "true", "effect_label": "invented"}
    errors = validate_gold_rows([invalid])
    assert any("representation_training_valid=false" in error for error in errors)
    assert any("invalid effect_label" in error for error in errors)
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) == pytest.approx(-1.0)


def _complete_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for index, row in enumerate(rows):
        row["effect_label"] = "no_effect" if index % 2 == 0 else "acquire_control"
        row["contact_label"] = "none" if index % 2 == 0 else "onset"
        row["annotator_id"] = "test"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_features(directory: Path, rows: list[dict[str, str]], seed: int, checkpoint_hash: str) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((len(rows), 4)).astype(np.float32)
    ids = np.asarray([row["sample_id"] for row in rows])
    features_path = directory / "features.npy"
    ids_path = directory / "sample_id.npy"
    metadata_path = directory / "feature_metadata.json"
    np.save(features_path, features)
    np.save(ids_path, ids)
    metadata_path.write_text(
        json.dumps(
            {
                "encoder_frozen": True,
                "encoder_checkpoint_sha256": checkpoint_hash,
                "samples": len(rows),
                "feature_dim": 4,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return features_path, ids_path, metadata_path


def test_probe_dev_freezes_artifacts_and_single_locked_mode_is_disabled(tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    pack = build_gold_pack(
        records("train", 50, 4),
        records("dev", 25, 4),
        records("locked", 73, 8),
        [],
    )
    write_gold_pack(gold_root, pack, 20260711)
    train_csv = gold_root / "gold140_probe_train_annotations.csv"
    dev_csv = gold_root / "gold60_calibration_dev_annotations.csv"
    locked_csv = gold_root / "gold100_locked_test_annotations.csv"
    train_rows = _complete_labels(train_csv)
    dev_rows = _complete_labels(dev_csv)
    locked_rows = _complete_labels(locked_csv)
    checkpoint_hash = "a" * 64
    train_features = _write_features(tmp_path / "train_features", train_rows, 1, checkpoint_hash)
    dev_features = _write_features(tmp_path / "dev_features", dev_rows, 2, checkpoint_hash)
    locked_features = _write_features(tmp_path / "locked_features", locked_rows, 3, checkpoint_hash)
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "probe_fact_effect_gold.py"

    def command(mode: str, evaluation: tuple[Path, Path, Path], eval_csv: Path, output: Path) -> list[str]:
        result = [
            sys.executable,
            str(script),
            "--train-features-npy",
            str(train_features[0]),
            "--train-sample-id-npy",
            str(train_features[1]),
            "--train-feature-metadata",
            str(train_features[2]),
            "--train-gold-csv",
            str(train_csv),
            "--eval-features-npy",
            str(evaluation[0]),
            "--eval-sample-id-npy",
            str(evaluation[1]),
            "--eval-feature-metadata",
            str(evaluation[2]),
            "--eval-gold-csv",
            str(eval_csv),
            "--gold-frozen-manifest",
            str(gold_root / "gold300_frozen.jsonl"),
            "--mode",
            mode,
            "--output-dir",
            str(output),
            "--bootstrap-iterations",
            "10",
        ]
        if mode == "locked":
            result.extend(
                [
                    "--frozen-probe-config",
                    str(tmp_path / "dev_run" / "probe_config_frozen.json"),
                    "--locked-asset-root",
                    str(gold_root),
                ]
            )
        return result

    subprocess.run(
        command("dev", dev_features, dev_csv, tmp_path / "dev_run"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen_config = json.loads((tmp_path / "dev_run" / "probe_config_frozen.json").read_text(encoding="utf-8"))
    assert set(frozen_config["probe_artifacts"]) == {"effect", "contact"}
    locked_attempt = subprocess.run(
        command("locked", locked_features, locked_csv, tmp_path / "locked_run"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert locked_attempt.returncode != 0
    assert "single-model locked evaluation is disabled" in locked_attempt.stderr
    assert not (gold_root / "LOCKED_EVALUATION_CONSUMED.json").exists()
