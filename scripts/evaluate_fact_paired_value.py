#!/usr/bin/env python3
"""Apply frozen P2-vs-P0/P3/P4 paired-value GO thresholds with take bootstrap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_eval import macro_f1, paired_value_go_decision  # noqa: E402
from fact_tokenizer.gold_annotations import EFFECT_LABELS  # noqa: E402


def prediction_arg(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("predictions must use EXPERIMENT:SEED=CSV")
    identity, path = value.split("=", 1)
    experiment, seed = identity.split(":", 1)
    return experiment, int(seed), Path(path)


def run_artifact_arg(value: str) -> tuple[str, int, Path]:
    return prediction_arg(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", type=prediction_arg, required=True)
    parser.add_argument("--prediction-report", action="append", type=run_artifact_arg, required=True)
    parser.add_argument(
        "--leakage-features",
        action="append",
        type=run_artifact_arg,
        required=True,
        help="EXP:SEED=NPZ with features, sample_id, take_uid and view arrays.",
    )
    parser.add_argument("--run-config", action="append", type=run_artifact_arg, required=True)
    parser.add_argument("--dataset-name", choices=["filtered", "unfiltered", "fresh_short73"], required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "take_uid", "effect_label", "effect_prediction"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks required prediction columns")
    return sorted(rows, key=lambda row: row["sample_id"])


def aggregate_bootstrap(
    truth: list[str],
    takes: list[str],
    first_by_seed: list[list[str]],
    second_by_seed: list[list[str]],
    iterations: int,
) -> dict:
    unique_takes = sorted(set(takes))
    by_take = {take: np.flatnonzero(np.asarray(takes) == take) for take in unique_takes}
    rng = np.random.default_rng(20260711)
    deltas = np.empty(iterations)
    truth_array = np.asarray(truth)
    for iteration in range(iterations):
        sampled = rng.choice(unique_takes, len(unique_takes), replace=True)
        indices = np.concatenate([by_take[take] for take in sampled])
        deltas[iteration] = np.mean(
            [
                macro_f1(truth_array[indices].tolist(), np.asarray(first)[indices].tolist(), EFFECT_LABELS)
                - macro_f1(truth_array[indices].tolist(), np.asarray(second)[indices].tolist(), EFFECT_LABELS)
                for first, second in zip(first_by_seed, second_by_seed)
            ]
        )
    points = [
        macro_f1(truth, first, EFFECT_LABELS) - macro_f1(truth, second, EFFECT_LABELS)
        for first, second in zip(first_by_seed, second_by_seed)
    ]
    return {
        "macro_f1_delta": float(np.mean(points)),
        "seed_deltas": points,
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "take",
    }


def leakage_nmi(path: Path, seed: int) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "features",
            "sample_id",
            "take_uid",
            "view",
            "encoder_checkpoint_sha256",
            "run_fingerprint_sha256",
            "manifest_sha256",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"leakage feature NPZ is missing arrays: {sorted(missing)}")
        features = np.asarray(payload["features"], dtype=np.float32)
        sample_id = np.asarray(payload["sample_id"]).astype(str)
        take_uid = np.asarray(payload["take_uid"]).astype(str)
        view = np.asarray(payload["view"]).astype(str)
        checkpoint_sha256 = str(np.asarray(payload["encoder_checkpoint_sha256"]).item())
        run_fingerprint_sha256 = str(np.asarray(payload["run_fingerprint_sha256"]).item())
        manifest_sha256 = str(np.asarray(payload["manifest_sha256"]).item())
    if features.ndim != 2 or not (len(features) == len(sample_id) == len(take_uid) == len(view)):
        raise ValueError("leakage features and metadata must be aligned NxD/vectors")
    if len(features) < 4:
        raise ValueError("leakage NMI requires at least four feature rows")
    clusters = min(64, max(2, len(set(take_uid.tolist()))), len(features) - 1)
    assignments = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=10,
        batch_size=min(1024, len(features)),
    ).fit_predict(features)
    return {
        "samples": len(features),
        "clusters": clusters,
        "take_nmi": float(normalized_mutual_info_score(take_uid, assignments)),
        "view_nmi": float(normalized_mutual_info_score(view, assignments)),
        "feature_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "run_fingerprint_sha256": run_fingerprint_sha256,
        "manifest_sha256": manifest_sha256,
    }


def validate_run_configs(entries: list[tuple[str, int, Path]]) -> dict[str, dict]:
    expected = {(experiment, seed) for experiment in ("P0", "P2", "P3", "P4") for seed in (42, 43, 44)}
    if {(experiment, seed) for experiment, seed, _ in entries} != expected:
        raise ValueError("run configs must cover exactly P0/P2/P3/P4 seeds 42/43/44")
    configs = {
        f"{experiment}:{seed}": json.loads(path.read_text(encoding="utf-8"))
        for experiment, seed, path in entries
    }
    invariant_fields = (
        "stage",
        "steps",
        "global_batch",
        "world_size",
        "gradient_accumulation",
        "step_unit",
        "manifest_sha256",
        "target_manifest_sha256",
        "target_identity_sha256",
        "gold_take_exclusion",
        "paired_eligibility",
        "quality_sampling",
        "learning_rate",
        "weight_decay",
        "objective",
        "weak_calibration",
        "shared_control_contract",
        "shared_control_contract_sha256",
    )
    reference = configs["P0:42"]
    for identity, config in configs.items():
        if config.get("experiment") != identity.split(":")[0] or int(config.get("seed", -1)) != int(identity.split(":")[1]):
            raise ValueError(f"run config identity mismatch for {identity}")
        if config.get("stage") != "final" or config.get("steps") != 20_000:
            raise ValueError(f"{identity} is not a preregistered 20k final run")
        for field in invariant_fields:
            if config.get(field) != reference.get(field):
                raise ValueError(f"run contract field {field!r} differs for {identity}")
    return configs


def main() -> None:
    args = parse_args()
    expected = {(experiment, seed) for experiment in ("P0", "P2", "P3", "P4") for seed in (42, 43, 44)}
    provided = {(experiment, seed) for experiment, seed, _ in args.predictions}
    if provided != expected:
        raise ValueError(f"final paired-value evaluation requires exactly {sorted(expected)}, got {sorted(provided)}")
    if {(experiment, seed) for experiment, seed, _ in args.prediction_report} != expected:
        raise ValueError("prediction reports must cover exactly the 12 final paired-control runs")
    if {(experiment, seed) for experiment, seed, _ in args.leakage_features} != expected:
        raise ValueError("leakage features must cover exactly the 12 final paired-control runs")
    run_configs = validate_run_configs(args.run_config)
    loaded = {(experiment, seed): load_predictions(path) for experiment, seed, path in args.predictions}
    prediction_paths = {(experiment, seed): path for experiment, seed, path in args.predictions}
    prediction_reports = {
        (experiment, seed): json.loads(path.read_text(encoding="utf-8"))
        for experiment, seed, path in args.prediction_report
    }
    for identity, report in prediction_reports.items():
        if report.get("evaluation_evidence", {}).get("predictions_csv_sha256") != hashlib.sha256(
            prediction_paths[identity].read_bytes()
        ).hexdigest():
            raise ValueError(f"prediction CSV hash differs from its probe report for {identity}")
    reference = loaded[("P2", 42)]
    sample_ids = [row["sample_id"] for row in reference]
    truth = [row["effect_label"] for row in reference]
    takes = [row["take_uid"] for row in reference]
    for identity, rows in loaded.items():
        if [row["sample_id"] for row in rows] != sample_ids:
            raise ValueError(f"sample IDs differ for {identity}")
        if [row["effect_label"] for row in rows] != truth or [row["take_uid"] for row in rows] != takes:
            raise ValueError(f"truth/take metadata differs for {identity}")
    comparisons = {}
    seed_deltas = {}
    for baseline in ("P0", "P3", "P4"):
        report = aggregate_bootstrap(
            truth,
            takes,
            [[row["effect_prediction"] for row in loaded[("P2", seed)]] for seed in (42, 43, 44)],
            [[row["effect_prediction"] for row in loaded[(baseline, seed)]] for seed in (42, 43, 44)],
            args.bootstrap_iterations,
        )
        comparisons[baseline] = report
        seed_deltas[baseline] = report["seed_deltas"]
    leakage = {
        (experiment, seed): leakage_nmi(path, seed)
        for experiment, seed, path in args.leakage_features
    }
    for (experiment, seed), evidence in leakage.items():
        identity = f"{experiment}:{seed}"
        config = run_configs[identity]
        config_digest = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if evidence["run_fingerprint_sha256"] != config_digest:
            raise ValueError(f"leakage features are not bound to run config {identity}")
        if evidence["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError(f"leakage feature manifest differs for {identity}")
        report_checkpoint = prediction_reports[(experiment, seed)].get("frozen_config", {}).get(
            "encoder_checkpoint_sha256"
        )
        if report_checkpoint != evidence["encoder_checkpoint_sha256"]:
            raise ValueError(f"prediction and leakage features use different checkpoints for {identity}")
    p2_take = np.mean([leakage[("P2", seed)]["take_nmi"] for seed in (42, 43, 44)])
    p0_take = np.mean([leakage[("P0", seed)]["take_nmi"] for seed in (42, 43, 44)])
    p2_view = np.mean([leakage[("P2", seed)]["view_nmi"] for seed in (42, 43, 44)])
    p3_view = np.mean([leakage[("P3", seed)]["view_nmi"] for seed in (42, 43, 44)])
    leakage_nmi_change = float(max(p2_take - p0_take, p2_view - p3_view))
    decision = paired_value_go_decision(comparisons, seed_deltas, leakage_nmi_change)
    sample_ids_sha256 = hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest()
    truth_sha256 = hashlib.sha256(("\n".join(truth) + "\n").encode("utf-8")).hexdigest()
    evidence_contract = {
        "sample_ids_sha256": sample_ids_sha256,
        "truth_sha256": truth_sha256,
        "training_manifest_sha256": run_configs["P0:42"]["manifest_sha256"],
        "prediction_csv_sha256": {
            f"{experiment}:{seed}": hashlib.sha256(path.read_bytes()).hexdigest()
            for experiment, seed, path in args.predictions
        },
        "leakage_feature_sha256": {
            f"{experiment}:{seed}": value["feature_npz_sha256"]
            for (experiment, seed), value in leakage.items()
        },
    }
    domain_evidence_sha256 = hashlib.sha256(
        json.dumps(evidence_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": "fact-v7-paired-value-gate-v1",
        "dataset_name": args.dataset_name,
        "samples": len(sample_ids),
        "takes": len(set(takes)),
        "sample_ids_sha256": sample_ids_sha256,
        "truth_sha256": truth_sha256,
        "training_manifest_sha256": evidence_contract["training_manifest_sha256"],
        "domain_evidence_sha256": domain_evidence_sha256,
        "comparisons": comparisons,
        "leakage": {
            "by_run": {f"{experiment}:{seed}": value for (experiment, seed), value in leakage.items()},
            "P2_minus_P0_take_nmi": float(p2_take - p0_take),
            "P2_minus_P3_view_nmi": float(p2_view - p3_view),
            "maximum_worsening": leakage_nmi_change,
        },
        "run_config_sha256": {
            identity: hashlib.sha256(
                next(path for experiment, seed, path in args.run_config if f"{experiment}:{seed}" == identity).read_bytes()
            ).hexdigest()
            for identity in run_configs
        },
        "decision": decision,
        "reporting_requirement": "filtered, unfiltered and fresh_short73 must be reported separately",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
