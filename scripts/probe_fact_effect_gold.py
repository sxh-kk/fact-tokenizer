#!/usr/bin/env python3
"""Fit fixed linear probes on 140 gold samples and evaluate dev or locked gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_eval import take_bootstrap  # noqa: E402
from fact_tokenizer.gold_annotations import CONTACT_LABELS, EFFECT_LABELS, validate_gold_rows  # noqa: E402


PROBE_HYPERPARAMETERS = {
    "scaler": "StandardScaler",
    "classifier": "LogisticRegression",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features-npy", type=Path, required=True)
    parser.add_argument("--train-sample-id-npy", type=Path, required=True)
    parser.add_argument("--train-feature-metadata", type=Path, required=True)
    parser.add_argument("--train-gold-csv", type=Path, required=True)
    parser.add_argument("--eval-features-npy", type=Path, required=True)
    parser.add_argument("--eval-sample-id-npy", type=Path, required=True)
    parser.add_argument("--eval-feature-metadata", type=Path, required=True)
    parser.add_argument("--eval-gold-csv", type=Path, required=True)
    parser.add_argument("--gold-frozen-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["dev", "locked"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-probe-config", type=Path)
    parser.add_argument(
        "--locked-asset-root",
        type=Path,
        help="Canonical gold asset directory. Required in locked mode; owns the global consumed marker.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gold(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    errors = validate_gold_rows(rows, require_complete=True)
    if errors:
        raise ValueError(f"gold validation failed: {errors[:5]}")
    return rows


def load_frozen_manifest(path: Path) -> dict[str, dict[str, Any]]:
    frozen: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in frozen:
                raise ValueError(f"invalid/duplicate frozen sample_id at line {line_number}")
            frozen[sample_id] = row
    return frozen


def validate_gold_subset(
    rows: list[dict[str, str]],
    *,
    split: str,
    expected_count: int,
    frozen: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{split} gold must contain exactly {expected_count} rows, found {len(rows)}")
    wrong_split = [row["sample_id"] for row in rows if row.get("gold_split") != split]
    if wrong_split:
        raise ValueError(f"{split} gold contains rows from another split: {wrong_split[:3]}")
    frozen_ids = {sample_id for sample_id, row in frozen.items() if row.get("gold_split") == split}
    row_ids = {row["sample_id"] for row in rows}
    if row_ids != frozen_ids:
        raise ValueError(
            f"{split} gold IDs differ from frozen manifest: "
            f"missing={sorted(frozen_ids - row_ids)[:3]}, extra={sorted(row_ids - frozen_ids)[:3]}"
        )
    for row in rows:
        frozen_row = frozen[row["sample_id"]]
        if str(row.get("take_uid", "")) != str(frozen_row.get("take_uid", "")):
            raise ValueError(f"take_uid differs from frozen manifest for {row['sample_id']}")


def load_feature_set(
    features_path: Path,
    sample_ids_path: Path,
    metadata_path: Path,
    expected_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    sample_ids = np.load(sample_ids_path, mmap_mode="r", allow_pickle=False).astype(str)
    if features.ndim != 2 or len(features) != len(sample_ids):
        raise ValueError("features must be NxD and align with sample IDs")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("feature sample IDs are not unique")
    actual_ids = set(sample_ids.tolist())
    if actual_ids != expected_ids:
        raise ValueError(
            "feature IDs must match the requested gold subset exactly; "
            f"missing={sorted(expected_ids - actual_ids)[:3]}, extra={sorted(actual_ids - expected_ids)[:3]}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("encoder_frozen", False):
        raise ValueError("gold probes require pre-extracted frozen encoder features")
    checkpoint_hash = str(metadata.get("encoder_checkpoint_sha256", ""))
    if len(checkpoint_hash) != 64:
        raise ValueError("feature metadata needs a 64-character encoder_checkpoint_sha256")
    return features, sample_ids, metadata


def subset_indices(sample_ids: np.ndarray, rows: list[dict[str, str]]) -> np.ndarray:
    by_id = {sample_id: index for index, sample_id in enumerate(sample_ids.tolist())}
    return np.asarray([by_id[row["sample_id"]] for row in rows], dtype=np.int64)


def contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def claim_locked_evaluation(root: Path, contract: Mapping[str, Any], report_path: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "LOCKED_EVALUATION_CONSUMED.json"
    payload = {
        "schema": "fact-v7-locked-consumption-v1",
        "status": "claimed",
        "probe_contract_sha256": contract_sha256(contract),
        "intended_report": str(report_path.resolve()),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(marker, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"locked test has already been consumed: {marker}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return marker


def complete_locked_evaluation(marker: Path, contract: Mapping[str, Any], report_path: Path) -> None:
    payload = {
        "schema": "fact-v7-locked-consumption-v1",
        "status": "complete",
        "probe_contract_sha256": contract_sha256(contract),
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
    }
    temp = marker.with_suffix(marker.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(marker)


def make_probe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=0,
                    multi_class="auto",
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    if args.mode == "locked":
        raise ValueError(
            "single-model locked evaluation is disabled; use "
            "probe_fact_effect_locked_campaign.py to claim once and evaluate all 12 frozen runs"
        )
    evaluation_split = "calibration_dev" if args.mode == "dev" else "locked_test"
    evaluation_count = 60 if args.mode == "dev" else 100
    frozen = load_frozen_manifest(args.gold_frozen_manifest)
    train_rows = load_gold(args.train_gold_csv)
    validate_gold_subset(
        train_rows,
        split="probe_train",
        expected_count=140,
        frozen=frozen,
    )
    train_features, train_sample_ids, train_metadata = load_feature_set(
        args.train_features_npy,
        args.train_sample_id_npy,
        args.train_feature_metadata,
        {row["sample_id"] for row in train_rows},
    )
    base_probe_config = {
        "schema": "fact-v7-fixed-linear-probe-v2",
        "encoder_checkpoint_sha256": train_metadata["encoder_checkpoint_sha256"],
        "train_feature_metadata_sha256": sha256_file(args.train_feature_metadata),
        "train_features_sha256": sha256_file(args.train_features_npy),
        "train_sample_ids_sha256": sha256_file(args.train_sample_id_npy),
        "train_gold_csv_sha256": sha256_file(args.train_gold_csv),
        "gold_frozen_manifest_sha256": sha256_file(args.gold_frozen_manifest),
        "train_split": "probe_train",
        "train_samples": 140,
        "train_feature_dim": int(train_features.shape[1]),
        "hyperparameters": PROBE_HYPERPARAMETERS,
        "thresholds": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.mode}_probe_report.json"
    consumed: Path | None = None
    frozen_config = dict(base_probe_config)
    frozen_probe_paths: dict[str, Path] = {}
    if args.mode == "locked":
        if args.frozen_probe_config is None:
            raise ValueError("locked evaluation requires --frozen-probe-config from the dev run")
        expected = json.loads(args.frozen_probe_config.read_text(encoding="utf-8"))
        expected_base = {key: value for key, value in expected.items() if key != "probe_artifacts"}
        if expected_base != base_probe_config:
            raise ValueError("locked train/checkpoint/gold contract differs from the frozen dev config")
        artifacts = expected.get("probe_artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {"effect", "contact"}:
            raise ValueError("frozen dev config is missing effect/contact probe artifacts")
        for target, artifact in artifacts.items():
            path = args.frozen_probe_config.parent / artifact["filename"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"frozen {target} probe artifact hash mismatch")
            frozen_probe_paths[target] = path
        frozen_config = expected
        if args.locked_asset_root is None:
            raise ValueError("locked evaluation requires --locked-asset-root")
        locked_root = args.locked_asset_root.resolve()
        if args.eval_gold_csv.resolve().parent != locked_root:
            raise ValueError("locked eval gold CSV must live directly in --locked-asset-root")
        if args.gold_frozen_manifest.resolve().parent != locked_root:
            raise ValueError("gold frozen manifest must live directly in --locked-asset-root")
        for path in (
            args.eval_gold_csv,
            args.eval_features_npy,
            args.eval_sample_id_npy,
            args.eval_feature_metadata,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        # Claim before reading locked labels or features. A failed attempt remains
        # consumed so an error cannot become an implicit extra evaluation pass.
        consumed = claim_locked_evaluation(locked_root, frozen_config, report_path)
    else:
        if args.locked_asset_root is not None:
            raise ValueError("dev mode must not receive --locked-asset-root")

    test_rows = load_gold(args.eval_gold_csv)
    validate_gold_subset(
        test_rows,
        split=evaluation_split,
        expected_count=evaluation_count,
        frozen=frozen,
    )
    eval_features, eval_sample_ids, eval_metadata = load_feature_set(
        args.eval_features_npy,
        args.eval_sample_id_npy,
        args.eval_feature_metadata,
        {row["sample_id"] for row in test_rows},
    )
    if eval_metadata["encoder_checkpoint_sha256"] != frozen_config["encoder_checkpoint_sha256"]:
        raise ValueError("evaluation features use a different encoder checkpoint")
    if eval_features.shape[1] != train_features.shape[1]:
        raise ValueError("train and evaluation feature dimensions differ")
    train_indices = subset_indices(train_sample_ids, train_rows)
    test_indices = subset_indices(eval_sample_ids, test_rows)

    results = {}
    predictions_by_target: dict[str, list[str]] = {}
    for target, labels in (("effect", EFFECT_LABELS), ("contact", CONTACT_LABELS)):
        label_column = f"{target}_label"
        y_train = [row[label_column] for row in train_rows]
        y_test = [row[label_column] for row in test_rows]
        if len(set(y_train)) < 2:
            raise ValueError(f"{target} probe training labels contain fewer than two classes")
        if args.mode == "locked":
            probe = joblib.load(frozen_probe_paths[target])
        else:
            probe = make_probe()
            probe.fit(np.asarray(train_features[train_indices]), y_train)
        predictions = probe.predict(np.asarray(eval_features[test_indices])).tolist()
        predictions_by_target[target] = predictions
        results[target] = take_bootstrap(
            y_test,
            predictions,
            [row["take_uid"] for row in test_rows],
            labels=labels,
            iterations=args.bootstrap_iterations,
        )
        if args.mode == "dev":
            joblib.dump(probe, args.output_dir / f"{target}_linear_probe.joblib")
    if args.mode == "dev":
        frozen_config["probe_artifacts"] = {
            target: {
                "filename": f"{target}_linear_probe.joblib",
                "sha256": sha256_file(args.output_dir / f"{target}_linear_probe.joblib"),
            }
            for target in ("effect", "contact")
        }
        (args.output_dir / "probe_config_frozen.json").write_text(
            json.dumps(frozen_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    predictions_path = args.output_dir / f"{args.mode}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "take_uid",
            "gold_split",
            "effect_label",
            "effect_prediction",
            "contact_label",
            "contact_prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(test_rows):
            writer.writerow(
                {
                    **{key: row[key] for key in ("sample_id", "take_uid", "gold_split", "effect_label", "contact_label")},
                    "effect_prediction": predictions_by_target["effect"][index],
                    "contact_prediction": predictions_by_target["contact"][index],
                }
            )
    report = {
        "mode": args.mode,
        "evaluation_split": evaluation_split,
        "train_samples": len(train_rows),
        "evaluation_samples": len(test_rows),
        "encoder_updated": False,
        "results": results,
        "frozen_config": frozen_config,
        "evaluation_evidence": {
            "feature_metadata_sha256": sha256_file(args.eval_feature_metadata),
            "features_sha256": sha256_file(args.eval_features_npy),
            "sample_ids_sha256": sha256_file(args.eval_sample_id_npy),
            "gold_csv_sha256": sha256_file(args.eval_gold_csv),
            "predictions_csv_sha256": sha256_file(predictions_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if consumed is not None:
        complete_locked_evaluation(consumed, frozen_config, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
