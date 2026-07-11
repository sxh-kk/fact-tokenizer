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
from scripts.materialize_fact_gold300_review import (
    aligned_endpoints,
    load_sources,
    validate_freeze,
    validate_gold,
    validate_source_contracts,
    validate_templates,
)


@pytest.mark.parametrize(
    "script",
    [
        "prepare_fact_gold300.py",
        "validate_fact_gold300.py",
        "materialize_fact_gold300_review.py",
        "freeze_fact_npy_source_contract.py",
    ],
)
def test_gold_cli_bootstraps_repo_imports(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in completed.stdout


def test_materialize_gold_review_pack_joins_sources_and_renders_blind_images(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    gold_rows = []
    source_arguments = []
    for source_index, split in enumerate(("probe_train", "calibration_dev", "probe_train")):
        source = tmp_path / f"source_{source_index}"
        source.mkdir()
        sample_id = f"sample_{source_index}"
        frames = np.zeros((1, 2, 20, 24, 3), dtype=np.uint8)
        frames[:, 1] = 40 + source_index
        np.save(source / "ego.npy", frames)
        np.save(source / "exo.npy", frames[:, :, :, ::-1])
        np.save(source / "sample_id.npy", np.asarray([sample_id]))
        np.save(source / "take_uid.npy", np.asarray([f"take_{source_index}"]))
        np.save(source / "timestamp.npy", np.asarray([float(source_index)], dtype=np.float32))
        gold_rows.append(
            {
                "sample_id": sample_id,
                "take_uid": f"take_{source_index}",
                "gold_split": split,
                "timestamp": float(source_index),
                "source_dataset": "test",
                "dual_annotation": source_index == 1,
                "representation_training_valid": False,
            }
        )
        source_arguments.extend(["--source", f"source_{source_index}={source}"])
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "".join(json.dumps(row) + "\n" for row in gold_rows),
        encoding="utf-8",
    )
    template_path = tmp_path / "blank_annotations.csv"
    template_fields = [*gold_rows[0], "effect_label", "contact_label", "annotator_id"]
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=template_fields)
        writer.writeheader()
        writer.writerows(gold_rows)
    guide_path = tmp_path / "guide.md"
    guide_path.write_text("blind guide\n", encoding="utf-8")
    admin_output = tmp_path / "review_admin"
    annotator_a_output = tmp_path / "review_annotator_a"
    annotator_b_output = tmp_path / "review_annotator_b"
    subprocess.run(
        [
            sys.executable,
            "scripts/materialize_fact_gold300_review.py",
            "--gold-manifest",
            str(gold_path),
            *source_arguments,
            "--annotation-template",
            str(template_path),
            "--annotation-guide",
            str(guide_path),
            "--admin-output-dir",
            str(admin_output),
            "--annotator-a-output-dir",
            str(annotator_a_output),
            "--annotator-b-output-dir",
            str(annotator_b_output),
            "--allow-noncanonical-count",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((admin_output / "review_pack_admin.json").read_text(encoding="utf-8"))
    assert report["samples"] == 3
    assert report["labels_in_review_images"] is False
    assert report["sample_identity_in_review_images"] is False
    assert (admin_output / "sealed" / "image_inventory.jsonl").is_file()
    assert (admin_output / "sealed" / "blank_templates" / template_path.name).is_file()
    assert not (annotator_a_output / "sealed").exists()
    assert not (annotator_b_output / "sealed").exists()
    assert len(list((annotator_a_output / "images").rglob("*.png"))) == 3
    assert len(list((annotator_b_output / "images").rglob("*.png"))) == 1
    with (annotator_a_output / "task.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (annotator_b_output / "task.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_gold_review_pack_rejects_unfrozen_nonblind_and_misaligned_inputs(tmp_path: Path) -> None:
    row = {
        "sample_id": "sample",
        "take_uid": "take",
        "gold_split": "probe_train",
        "timestamp": 1.0,
        "source_dataset": "test",
        "dual_annotation": False,
        "representation_training_valid": False,
    }
    malicious = {**row, "gold_split": "../../outside"}
    with pytest.raises(ValueError, match="invalid gold_split"):
        validate_gold([malicious], canonical=False)

    manifest = tmp_path / "gold.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema": "fact-effect-gold-v1",
                "manifest_sha256": "0" * 64,
                "sample_counts": {"probe_train": 1},
                "representation_training_valid": False,
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not bind"):
        validate_freeze(freeze, manifest, [row], None, canonical=False)

    template = tmp_path / "nonblind.csv"
    fields = [*row, "effect_label", "contact_label", "annotator_id"]
    with template.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**row, "effect_label": "no_effect", "contact_label": "none"})
    with pytest.raises(ValueError, match="not blind"):
        validate_templates(
            [template],
            [row],
            canonical=False,
            expected_splits={"probe_train"},
        )

    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "ego.npy", np.zeros((1, 2, 8, 8, 3), dtype=np.uint8))
    np.save(source / "exo.npy", np.zeros((1, 2, 8, 8, 3), dtype=np.uint8))
    np.save(source / "sample_id.npy", np.asarray(["sample"]))
    np.save(source / "take_uid.npy", np.asarray(["wrong_take"]))
    np.save(source / "timestamp.npy", np.asarray([1.0], dtype=np.float32))
    _, sources = load_sources([("source", source)])
    with pytest.raises(ValueError, match="take_uid mismatch"):
        aligned_endpoints(sources["source"], 0, row)


def test_freeze_npy_source_contract_binds_rgb_transition_and_arrays(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "ego.npy", np.zeros((2, 2, 8, 8, 3), dtype=np.uint8))
    np.save(source / "exo.npy", np.ones((2, 2, 8, 8, 3), dtype=np.uint8))
    np.save(source / "sample_id.npy", np.asarray(["a", "b"]))
    np.save(source / "take_uid.npy", np.asarray(["ta", "tb"]))
    np.save(source / "timestamp.npy", np.asarray([1.0, 2.0], dtype=np.float32))
    producer = tmp_path / "producer.py"
    producer.write_text(
        "# cv2.COLOR_BGR2RGB\n# parser.add_argument('--transition-sec', default=0.5)\n",
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_fact_npy_source_contract.py",
            "--input-dir",
            str(source),
            "--output-json",
            str(contract),
            "--producer-code",
            str(producer),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    _, sources = load_sources([("source", source)])
    validated = validate_source_contracts([("source", contract)], sources, canonical=True)
    assert validated["source"]["sha256"]
    payload = json.loads(contract.read_text(encoding="utf-8"))
    assert payload["color_space"] == "RGB"
    assert payload["endpoint_semantics"] == ["t", "t+0.5s"]


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
    guide = (tmp_path / "ANNOTATION_GUIDE.zh-CN.md").read_text(encoding="utf-8")
    assert "FACT effect/contact 标注指南" in guide
    assert "只依据 Ego 与 Exo 图像" in guide


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
