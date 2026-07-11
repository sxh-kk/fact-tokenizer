#!/usr/bin/env python3
"""Claim short73 once, then evaluate all 12 frozen paired-control probes in one campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_eval import take_bootstrap  # noqa: E402
from fact_tokenizer.gold_annotations import CONTACT_LABELS, EFFECT_LABELS  # noqa: E402
from scripts.probe_fact_effect_gold import (  # noqa: E402
    claim_locked_evaluation,
    complete_locked_evaluation,
    load_feature_set,
    load_frozen_manifest,
    load_gold,
    sha256_file,
    subset_indices,
    validate_gold_subset,
)


EXPECTED_RUNS = {(experiment, seed) for experiment in ("P0", "P2", "P3", "P4") for seed in (42, 43, 44)}


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--locked-asset-root", type=Path, required=True)
    parser.add_argument("--short73-freeze", type=Path, required=True)
    parser.add_argument("--short73-materialization-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    spec = json.loads(args.campaign_spec.read_text(encoding="utf-8"))
    if spec.get("schema") != "fact-v7-locked-probe-campaign-v1":
        raise ValueError("unsupported locked campaign spec schema")
    runs = spec.get("runs")
    if not isinstance(runs, list) or {
        (str(row.get("experiment")), int(row.get("seed", -1))) for row in runs
    } != EXPECTED_RUNS:
        raise ValueError("locked campaign must contain exactly P0/P2/P3/P4 seeds 42/43/44")
    base = args.campaign_spec.parent
    locked_root = args.locked_asset_root.resolve()
    gold_manifest_path = locked_root / "gold300_frozen.jsonl"
    train_gold_path = locked_root / "gold140_probe_train_annotations.csv"
    locked_gold_path = locked_root / "gold100_locked_test_annotations.csv"
    for path in (gold_manifest_path, train_gold_path, locked_gold_path):
        if not path.is_file() or path.resolve().parent != locked_root:
            raise ValueError(f"locked campaign asset must live in canonical root: {path}")

    short_freeze = json.loads(args.short73_freeze.read_text(encoding="utf-8"))
    materialization = json.loads(args.short73_materialization_report.read_text(encoding="utf-8"))
    if short_freeze.get("freeze_stage") != "final" or short_freeze.get("evaluation_allowed") is not True:
        raise ValueError("locked campaign requires the final short73 freeze")
    if materialization.get("freeze_stage") != "final" or materialization.get("evaluation_allowed") is not True:
        raise ValueError("locked campaign requires final short73 materialization")
    if materialization.get("source_freeze_sha256") != sha256_file(args.short73_freeze):
        raise ValueError("short73 materialization is not bound to the supplied final freeze")
    locked_manifest_sha = materialization.get("effect_manifest_sha256")
    if len(str(locked_manifest_sha or "")) != 64:
        raise ValueError("short73 materialization lacks its locked effect-manifest hash")

    frozen_gold = load_frozen_manifest(gold_manifest_path)
    train_rows = load_gold(train_gold_path)
    validate_gold_subset(train_rows, split="probe_train", expected_count=140, frozen=frozen_gold)
    prepared: list[dict[str, Any]] = []
    for row in sorted(runs, key=lambda value: (value["experiment"], int(value["seed"]))):
        experiment, seed = str(row["experiment"]), int(row["seed"])
        paths = {
            name: resolve(base, str(row[name]))
            for name in (
                "checkpoint",
                "run_config",
                "frozen_probe_config",
                "train_features_npy",
                "train_sample_id_npy",
                "train_feature_metadata",
                "eval_features_npy",
                "eval_sample_id_npy",
                "eval_feature_metadata",
            )
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(next(path for path in paths.values() if not path.is_file()))
        run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
        if (
            run_config.get("experiment") != experiment
            or int(run_config.get("seed", -1)) != seed
            or run_config.get("stage") != "final"
            or int(run_config.get("steps", -1)) != 20_000
        ):
            raise ValueError(f"{experiment}:{seed} is not a frozen preregistered final run")
        checkpoint_sha = sha256_file(paths["checkpoint"])
        checkpoint_payload = torch.load(paths["checkpoint"], map_location="cpu")
        if (
            checkpoint_payload.get("run_fingerprint") != run_config
            or int(checkpoint_payload.get("step", -1)) != 20_000
            or checkpoint_payload.get("step_unit") != "optimizer_update"
        ):
            raise ValueError(f"checkpoint payload differs from final run config for {experiment}:{seed}")
        del checkpoint_payload
        probe_config = json.loads(paths["frozen_probe_config"].read_text(encoding="utf-8"))
        if probe_config.get("encoder_checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"probe config checkpoint mismatch for {experiment}:{seed}")
        artifacts = probe_config.get("probe_artifacts", {})
        probe_paths = {}
        for target in ("effect", "contact"):
            artifact = artifacts.get(target, {})
            probe_path = paths["frozen_probe_config"].parent / str(artifact.get("filename", ""))
            if not probe_path.is_file() or sha256_file(probe_path) != artifact.get("sha256"):
                raise ValueError(f"frozen {target} probe mismatch for {experiment}:{seed}")
            probe_paths[target] = probe_path
        train_features, train_ids, train_metadata = load_feature_set(
            paths["train_features_npy"],
            paths["train_sample_id_npy"],
            paths["train_feature_metadata"],
            {gold["sample_id"] for gold in train_rows},
        )
        if train_metadata["encoder_checkpoint_sha256"] != checkpoint_sha:
            raise ValueError(f"train features checkpoint mismatch for {experiment}:{seed}")
        eval_metadata = json.loads(paths["eval_feature_metadata"].read_text(encoding="utf-8"))
        if (
            eval_metadata.get("encoder_checkpoint_sha256") != checkpoint_sha
            or eval_metadata.get("manifest_sha256") != locked_manifest_sha
            or not eval_metadata.get("encoder_frozen", False)
        ):
            raise ValueError(f"locked features lack final short73/checkpoint binding for {experiment}:{seed}")
        prepared.append(
            {
                "identity": f"{experiment}:{seed}",
                "experiment": experiment,
                "seed": seed,
                "paths": paths,
                "checkpoint_sha256": checkpoint_sha,
                "probe_config": probe_config,
                "probe_paths": probe_paths,
                "train_features": train_features,
                "train_ids": train_ids,
            }
        )

    campaign_contract = {
        "schema": "fact-v7-locked-probe-campaign-v1",
        "campaign_spec_sha256": sha256_file(args.campaign_spec),
        "gold_manifest_sha256": sha256_file(gold_manifest_path),
        "short73_freeze_sha256": sha256_file(args.short73_freeze),
        "short73_materialization_report_sha256": sha256_file(args.short73_materialization_report),
        "run_checkpoints": {row["identity"]: row["checkpoint_sha256"] for row in prepared},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign_report_path = args.output_dir / "locked_campaign_report.json"
    # This is the sole claim, immediately before any locked labels/features are read.
    marker = claim_locked_evaluation(locked_root, campaign_contract, campaign_report_path)
    locked_rows = load_gold(locked_gold_path)
    validate_gold_subset(locked_rows, split="locked_test", expected_count=100, frozen=frozen_gold)
    reports = {}
    for run in prepared:
        paths = run["paths"]
        eval_features, eval_ids, _ = load_feature_set(
            paths["eval_features_npy"],
            paths["eval_sample_id_npy"],
            paths["eval_feature_metadata"],
            {row["sample_id"] for row in locked_rows},
        )
        if eval_features.shape[1] != run["train_features"].shape[1]:
            raise ValueError(f"feature dimension mismatch for {run['identity']}")
        test_indices = subset_indices(eval_ids, locked_rows)
        predictions_by_target = {}
        results = {}
        for target, labels in (("effect", EFFECT_LABELS), ("contact", CONTACT_LABELS)):
            probe = joblib.load(run["probe_paths"][target])
            predictions = probe.predict(np.asarray(eval_features[test_indices])).tolist()
            predictions_by_target[target] = predictions
            results[target] = take_bootstrap(
                [row[f"{target}_label"] for row in locked_rows],
                predictions,
                [row["take_uid"] for row in locked_rows],
                labels=labels,
                iterations=args.bootstrap_iterations,
            )
        run_dir = args.output_dir / run["identity"].replace(":", "_")
        run_dir.mkdir()
        predictions_path = run_dir / "locked_predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "take_uid",
                    "gold_split",
                    "effect_label",
                    "effect_prediction",
                    "contact_label",
                    "contact_prediction",
                ),
            )
            writer.writeheader()
            for index, row in enumerate(locked_rows):
                writer.writerow(
                    {
                        **{
                            key: row[key]
                            for key in ("sample_id", "take_uid", "gold_split", "effect_label", "contact_label")
                        },
                        "effect_prediction": predictions_by_target["effect"][index],
                        "contact_prediction": predictions_by_target["contact"][index],
                    }
                )
        report = {
            "schema": "fact-v7-locked-campaign-run-v1",
            "mode": "locked",
            "identity": run["identity"],
            "frozen_config": run["probe_config"],
            "results": results,
            "evaluation_evidence": {
                "predictions_csv_sha256": sha256_file(predictions_path),
                "features_sha256": sha256_file(paths["eval_features_npy"]),
                "sample_ids_sha256": sha256_file(paths["eval_sample_id_npy"]),
                "feature_metadata_sha256": sha256_file(paths["eval_feature_metadata"]),
                "gold_csv_sha256": sha256_file(locked_gold_path),
                "short73_freeze_sha256": sha256_file(args.short73_freeze),
            },
        }
        report_path = run_dir / "locked_probe_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reports[run["identity"]] = {
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
        }
    campaign_report = {
        **campaign_contract,
        "locked_gold_sha256": sha256_file(locked_gold_path),
        "runs": reports,
        "run_count": len(reports),
        "locked_read_count": 1,
    }
    campaign_report_path.write_text(
        json.dumps(campaign_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    complete_locked_evaluation(marker, campaign_contract, campaign_report_path)
    print(json.dumps(campaign_report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
