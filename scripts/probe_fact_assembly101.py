#!/usr/bin/env python3
"""Run the one-shot fixed 8/2 Assembly101 action probe on frozen FACT features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_eval import take_bootstrap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--assembly-dir", type=Path, required=True)
    parser.add_argument("--frozen-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--assembly-asset-root",
        type=Path,
        required=True,
        help="Canonical frozen Assembly probe asset root; owns the global one-shot marker.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-run-config", type=Path, required=True)
    parser.add_argument("--paired-campaign-gate", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset_root = args.assembly_asset_root.resolve()
    if args.frozen_split.resolve().parent != asset_root:
        raise ValueError("frozen Assembly split must live directly in --assembly-asset-root")
    prepare_report_path = asset_root / "assembly101_prepare_report.json"
    prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
    split_digest = hashlib.sha256(args.frozen_split.read_bytes()).hexdigest()
    if split_digest != prepare_report.get("split_file_sha256"):
        raise ValueError("frozen Assembly split hash differs from its prepare report")
    metadata = json.loads((args.feature_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("encoder_frozen") or metadata.get("vq_enabled", False):
        raise ValueError("Assembly probe requires frozen continuous encoder features")
    features = np.load(args.feature_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    feature_ids = np.load(args.feature_dir / "sample_id.npy", mmap_mode="r", allow_pickle=False).astype(str)
    assembly_ids = np.load(args.assembly_dir / "sample_id.npy", mmap_mode="r", allow_pickle=False).astype(str)
    take_uid = np.load(args.assembly_dir / "take_uid.npy", mmap_mode="r", allow_pickle=False).astype(str)
    action_label = np.load(args.assembly_dir / "action_label.npy", mmap_mode="r", allow_pickle=False).astype(str)
    for name, expected_hash in prepare_report.get("input_array_sha256", {}).items():
        path = args.assembly_dir / f"{name}.npy"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Assembly input array {name!r} differs from the frozen prepare report")
    if metadata.get("manifest_sha256") != prepare_report.get("probe_manifest_sha256"):
        raise ValueError("Assembly feature extraction did not use the frozen probe manifest")
    if feature_ids.tolist() != assembly_ids.tolist():
        raise ValueError("feature rows do not align exactly with Assembly sample IDs")
    split = json.loads(args.frozen_split.read_text(encoding="utf-8"))
    train_mask = np.isin(take_uid, split["train_takes"])
    test_mask = np.isin(take_uid, split["test_takes"])
    if train_mask.sum() + test_mask.sum() != len(take_uid):
        raise ValueError("frozen Assembly split does not cover every sample")
    if len(set(action_label[train_mask].tolist())) < 2:
        raise ValueError("Assembly train split has fewer than two action classes")
    run_config = json.loads(args.selected_run_config.read_text(encoding="utf-8"))
    if (
        run_config.get("experiment") != "P2"
        or run_config.get("stage") != "final"
        or int(run_config.get("steps", -1)) != 20_000
        or run_config.get("vq_enabled") is not False
    ):
        raise ValueError("Assembly probe requires a selected final 20k continuous P2 run")
    checkpoint_sha = hashlib.sha256(args.selected_checkpoint.read_bytes()).hexdigest()
    if metadata.get("encoder_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Assembly features do not match the selected checkpoint")
    checkpoint = torch.load(args.selected_checkpoint, map_location="cpu")
    if checkpoint.get("run_fingerprint") != run_config or int(checkpoint.get("step", -1)) != 20_000:
        raise ValueError("selected checkpoint payload differs from its final run config")
    del checkpoint
    campaign_gate = json.loads(args.paired_campaign_gate.read_text(encoding="utf-8"))
    if campaign_gate.get("schema") != "fact-v7-paired-campaign-gate-v1" or campaign_gate.get("go") is not True:
        raise ValueError("Assembly probe may run only after the paired campaign GO gate")

    # All hashes, rows, labels and selected-model contracts are preflighted
    # before the irreversible one-shot claim.
    consumed = asset_root / "ASSEMBLY101_PROBE_CONSUMED.json"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(consumed, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError("Assembly101 probe has already been consumed globally") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "status": "claimed",
                    "split_sha256": split_digest,
                    "checkpoint_sha256": checkpoint_sha,
                    "campaign_gate_sha256": hashlib.sha256(args.paired_campaign_gate.read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
    probe = Pipeline(
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
    probe.fit(np.asarray(features[train_mask]), action_label[train_mask])
    prediction = probe.predict(np.asarray(features[test_mask])).tolist()
    truth = action_label[test_mask].tolist()
    result = take_bootstrap(
        truth,
        prediction,
        take_uid[test_mask].tolist(),
        labels=sorted(set(action_label.tolist())),
        iterations=args.bootstrap_iterations,
    )
    report = {
        "schema": "fact-assembly101-probe-v1",
        "train_takes": split["train_takes"],
        "test_takes": split["test_takes"],
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "encoder_checkpoint_sha256": metadata["encoder_checkpoint_sha256"],
        "run_config_sha256": hashlib.sha256(args.selected_run_config.read_bytes()).hexdigest(),
        "campaign_gate_sha256": hashlib.sha256(args.paired_campaign_gate.read_bytes()).hexdigest(),
        "split_sha256": split_digest,
        "model_selection_allowed": False,
        "threshold_selection_allowed": False,
        "result": result,
    }
    report_path = args.output_dir / "assembly101_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp = consumed.with_suffix(f".tmp-{os.getpid()}")
    temp.write_text(
        json.dumps(
            {
                "status": "complete",
                "report": str(report_path.resolve()),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "result": result,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(consumed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
