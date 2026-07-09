#!/usr/bin/env python3
"""Build take-level quality weights for soft filtering experiments.

The v1 preset is intentionally simple and reproduces the first weighted-filter
experiment.  The v2 preset is a more continuous policy: it keeps full-data
diversity, but separates high-confidence hand/object interaction takes from
loco/diagnostic takes more strongly and applies task/parent diversity
correction before sampling.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "heldout", "all"], default="train")
    parser.add_argument("--preset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--main-weight", type=float, default=None)
    parser.add_argument("--loco-weight", type=float, default=None)
    parser.add_argument("--diagnostic-weight", type=float, default=None)
    parser.add_argument("--discard-weight", type=float, default=None)
    parser.add_argument("--min-keep-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--no-normalize-max", action="store_true")
    parser.add_argument("--task-balance-strength", type=float, default=0.35)
    parser.add_argument("--parent-balance-strength", type=float, default=0.20)
    parser.add_argument("--diversity-min-multiplier", type=float, default=0.65)
    parser.add_argument("--diversity-max-multiplier", type=float, default=1.35)
    parser.add_argument("--parent-mass-balance-strength", type=float, default=0.25)
    parser.add_argument("--task-mass-balance-strength", type=float, default=0.10)
    parser.add_argument("--mass-balance-min-multiplier", type=float, default=0.70)
    parser.add_argument("--mass-balance-max-multiplier", type=float, default=1.45)
    parser.add_argument("--temperature", type=float, default=1.15)
    return parser.parse_args()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bucket_for_row(row: dict[str, str]) -> str:
    for name in ("ranker_bucket", "auto_usable_for", "usable_for"):
        value = str(row.get(name, "")).strip()
        if value in {"tokenizer_main", "fact_main"}:
            return "tokenizer_main"
        if value == "loco_aux":
            return "loco_aux"
        if value == "diagnostic_candidate":
            return "diagnostic_candidate"
        if value == "discard":
            return "discard"
    probs = {
        "tokenizer_main": to_float(row.get("ranker_prob_tokenizer_main", row.get("prob_tokenizer_main"))),
        "loco_aux": to_float(row.get("ranker_prob_loco_aux", row.get("prob_loco_aux"))),
        "diagnostic_candidate": to_float(row.get("ranker_prob_diagnostic_candidate", row.get("prob_diagnostic_candidate"))),
        "discard": to_float(row.get("ranker_prob_discard", row.get("prob_discard"))),
    }
    return max(probs, key=probs.get)


def resolved_base_weights(args: argparse.Namespace) -> dict[str, float]:
    if args.preset == "v1":
        defaults = {
            "tokenizer_main": 1.0,
            "loco_aux": 0.40,
            "diagnostic_candidate": 0.18,
            "discard": 0.0,
        }
    else:
        defaults = {
            "tokenizer_main": 1.0,
            "loco_aux": 0.46,
            "diagnostic_candidate": 0.13,
            "discard": 0.0,
        }
    overrides = {
        "tokenizer_main": args.main_weight,
        "loco_aux": args.loco_weight,
        "diagnostic_candidate": args.diagnostic_weight,
        "discard": args.discard_weight,
    }
    return {name: float(defaults[name] if value is None else value) for name, value in overrides.items()}


def v1_sample_weight(row: dict[str, str], bucket: str, args: argparse.Namespace, base_weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    base = {
        "tokenizer_main": base_weights["tokenizer_main"],
        "loco_aux": base_weights["loco_aux"],
        "diagnostic_candidate": base_weights["diagnostic_candidate"],
        "discard": base_weights["discard"],
    }[bucket]
    main_prob = max(to_float(row.get("ranker_prob_tokenizer_main")), to_float(row.get("prob_tokenizer_main")))
    confidence = max(to_float(row.get("ranker_confidence")), to_float(row.get("auto_confidence")), main_prob)
    scene = to_float(row.get("scene_only_score"))
    discard_prob = max(to_float(row.get("ranker_prob_discard")), to_float(row.get("prob_discard")))
    quality = 0.65 + 0.35 * max(0.0, min(1.0, confidence))
    penalty = max(0.25, 1.0 - 0.50 * max(0.0, scene) - 0.35 * max(0.0, discard_prob))
    weight = base * quality * penalty
    if weight > 0.0:
        weight = max(args.min_keep_weight, weight)
    return float(weight), {
        "prob_mix": base,
        "feature_quality": quality,
        "risk_penalty": penalty,
        "diversity_multiplier": 1.0,
        "raw_weight": weight,
    }


def diversity_multiplier(row: dict[str, str], stats: dict[str, object], args: argparse.Namespace) -> float:
    split = str(row.get("split", "")).strip() or "train"
    task = str(row.get("task_name", "")).strip() or "<unknown_task>"
    parent = str(row.get("parent_task_name", "")).strip() or "<unknown_parent>"

    split_task_counts: Counter[str] = stats["task_counts_by_split"].get(split, Counter())  # type: ignore[assignment]
    split_parent_counts: Counter[str] = stats["parent_counts_by_split"].get(split, Counter())  # type: ignore[assignment]
    mean_task = float(stats["mean_task_count_by_split"].get(split, 1.0))  # type: ignore[index]
    mean_parent = float(stats["mean_parent_count_by_split"].get(split, 1.0))  # type: ignore[index]
    task_count = max(1.0, float(split_task_counts.get(task, 1)))
    parent_count = max(1.0, float(split_parent_counts.get(parent, 1)))

    task_factor = (mean_task / task_count) ** max(0.0, args.task_balance_strength)
    parent_factor = (mean_parent / parent_count) ** max(0.0, args.parent_balance_strength)
    return clamp(task_factor * parent_factor, args.diversity_min_multiplier, args.diversity_max_multiplier)


def v2_sample_weight(
    row: dict[str, str],
    bucket: str,
    args: argparse.Namespace,
    base_weights: dict[str, float],
    stats: dict[str, object],
) -> tuple[float, dict[str, float]]:
    p_main = max(to_float(row.get("ranker_prob_tokenizer_main")), to_float(row.get("prob_tokenizer_main")))
    p_loco = max(to_float(row.get("ranker_prob_loco_aux")), to_float(row.get("prob_loco_aux")))
    p_diag = max(to_float(row.get("ranker_prob_diagnostic_candidate")), to_float(row.get("prob_diagnostic_candidate")))
    p_discard = max(to_float(row.get("ranker_prob_discard")), to_float(row.get("prob_discard")))

    interaction = clamp(to_float(row.get("interaction_score")))
    object_motion = clamp(to_float(row.get("object_motion_proxy")))
    phase = clamp(to_float(row.get("phase_diversity_score_v2")))
    motion_change = clamp(to_float(row.get("motion_state_change_score"), to_float(row.get("temporal_diversity_score"))))
    temporal = clamp(to_float(row.get("temporal_diversity_score")))
    fine = clamp(to_float(row.get("fine_dexterous_score")))
    metadata_interaction = clamp(to_float(row.get("metadata_interaction_prior")))
    loco_score = clamp(to_float(row.get("loco_score")))
    exo_body = clamp(to_float(row.get("exo_body_motion_score")))
    scene = clamp(to_float(row.get("scene_only_score")))
    disagreement = clamp(max(to_float(row.get("ranker_disagreement_score")), to_float(row.get("auto_disagreement_score"))))

    feature_quality = clamp(
        0.28 * interaction
        + 0.18 * object_motion
        + 0.14 * phase
        + 0.14 * motion_change
        + 0.10 * temporal
        + 0.08 * metadata_interaction
        + 0.08 * fine
    )
    loco_quality = clamp(0.40 * loco_score + 0.24 * p_loco + 0.20 * exo_body + 0.16 * phase)
    diagnostic_quality = clamp(0.45 * p_diag + 0.30 * disagreement + 0.25 * temporal)

    if bucket == "discard":
        prob_mix = base_weights["discard"] * (0.35 + 0.65 * (1.0 - p_discard))
    elif bucket == "loco_aux":
        prob_mix = (
            0.42 * base_weights["loco_aux"]
            + 0.32 * base_weights["loco_aux"] * loco_quality
            + 0.18 * base_weights["tokenizer_main"] * p_main
            + 0.08 * base_weights["diagnostic_candidate"] * diagnostic_quality
        )
    elif bucket == "diagnostic_candidate":
        prob_mix = (
            0.45 * base_weights["diagnostic_candidate"]
            + 0.25 * base_weights["diagnostic_candidate"] * diagnostic_quality
            + 0.25 * base_weights["tokenizer_main"] * p_main
            + 0.05 * base_weights["loco_aux"] * p_loco
        )
    else:
        prob_mix = (
            0.50 * base_weights["tokenizer_main"]
            + 0.35 * base_weights["tokenizer_main"] * p_main
            + 0.10 * base_weights["loco_aux"] * p_loco
            + 0.05 * base_weights["diagnostic_candidate"] * p_diag
        )

    feature_multiplier = 0.45 + 0.75 * feature_quality
    risk_penalty = clamp(1.0 - 0.78 * p_discard - 0.55 * scene - 0.18 * p_diag - 0.10 * disagreement, 0.04, 1.0)
    auto_take_relevance = str(row.get("auto_take_relevance", "")).strip()
    auto_bucket = str(row.get("auto_bucket", "")).strip()
    if auto_take_relevance == "D_scene_only":
        risk_penalty *= 0.35
    elif auto_take_relevance == "C_active_view_only":
        risk_penalty *= 0.74
    if auto_bucket in {"D_scene_only", "discard"}:
        risk_penalty *= 0.50

    diversity = diversity_multiplier(row, stats, args) if bucket != "discard" else 1.0
    raw = prob_mix * feature_multiplier * risk_penalty * diversity
    if args.temperature > 0.0:
        raw = raw ** args.temperature
    if raw > 0.0:
        raw = max(args.min_keep_weight, raw)
    return float(raw), {
        "prob_mix": float(prob_mix),
        "feature_quality": float(feature_quality),
        "risk_penalty": float(risk_penalty),
        "diversity_multiplier": float(diversity),
        "raw_weight": float(raw),
    }


def build_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    task_counts_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    parent_counts_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split = str(row.get("split", "")).strip() or "train"
        task = str(row.get("task_name", "")).strip() or "<unknown_task>"
        parent = str(row.get("parent_task_name", "")).strip() or "<unknown_parent>"
        task_counts_by_split[split][task] += 1
        parent_counts_by_split[split][parent] += 1
    mean_task_count_by_split = {
        split: (sum(counts.values()) / max(1, len(counts))) for split, counts in task_counts_by_split.items()
    }
    mean_parent_count_by_split = {
        split: (sum(counts.values()) / max(1, len(counts))) for split, counts in parent_counts_by_split.items()
    }
    return {
        "task_counts_by_split": task_counts_by_split,
        "parent_counts_by_split": parent_counts_by_split,
        "mean_task_count_by_split": mean_task_count_by_split,
        "mean_parent_count_by_split": mean_parent_count_by_split,
    }


def effective_sample_size(weights: list[float]) -> float:
    total = sum(weights)
    denom = sum(weight * weight for weight in weights)
    if total <= 0.0 or denom <= 0.0:
        return 0.0
    return (total * total) / denom


def apply_mass_balance(output_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    parent_mass: Counter[str] = Counter()
    task_mass: Counter[str] = Counter()
    for row in output_rows:
        weight = to_float(row["sample_weight"])
        if weight <= 0.0:
            row["mass_balance_multiplier"] = f"{1.0:.9f}"
            continue
        parent_mass[row.get("parent_task_name", "")] += weight
        task_mass[row.get("task_name", "")] += weight

    positive_parent_mass = [mass for mass in parent_mass.values() if mass > 0.0]
    positive_task_mass = [mass for mass in task_mass.values() if mass > 0.0]
    if not positive_parent_mass or not positive_task_mass:
        return
    mean_parent_mass = sum(positive_parent_mass) / len(positive_parent_mass)
    mean_task_mass = sum(positive_task_mass) / len(positive_task_mass)

    for row in output_rows:
        weight = to_float(row["sample_weight"])
        if weight <= 0.0:
            row["mass_balance_multiplier"] = f"{1.0:.9f}"
            continue
        parent = row.get("parent_task_name", "")
        task = row.get("task_name", "")
        parent_factor = (mean_parent_mass / max(parent_mass[parent], 1e-12)) ** max(0.0, args.parent_mass_balance_strength)
        task_factor = (mean_task_mass / max(task_mass[task], 1e-12)) ** max(0.0, args.task_mass_balance_strength)
        multiplier = clamp(
            parent_factor * task_factor,
            args.mass_balance_min_multiplier,
            args.mass_balance_max_multiplier,
        )
        row["mass_balance_multiplier"] = f"{multiplier:.9f}"
        row["sample_weight"] = f"{weight * multiplier:.9f}"


def print_audit(output_rows: list[dict[str, str]]) -> None:
    weights = [to_float(row["sample_weight"]) for row in output_rows]
    positives = [weight for weight in weights if weight > 0.0]
    by_bucket: defaultdict[str, list[float]] = defaultdict(list)
    by_parent_mass: Counter[str] = Counter()
    by_task_mass: Counter[str] = Counter()
    for row, weight in zip(output_rows, weights):
        by_bucket[row["bucket"]].append(weight)
        by_parent_mass[row.get("parent_task_name", "")] += weight
        by_task_mass[row.get("task_name", "")] += weight
    print(f"rows={len(output_rows)} positive_weight_takes={len(positives)}")
    if positives:
        print(
            "weight_stats "
            f"min={min(positives):.6f} "
            f"mean={sum(positives) / len(positives):.6f} "
            f"max={max(positives):.6f} "
            f"effective_takes={effective_sample_size(positives):.2f}"
        )
    for bucket, bucket_weights in sorted(by_bucket.items()):
        bucket_pos = [weight for weight in bucket_weights if weight > 0.0]
        mean = sum(bucket_pos) / len(bucket_pos) if bucket_pos else 0.0
        print(f"bucket={bucket} count={len(bucket_weights)} positive={len(bucket_pos)} mean_positive={mean:.6f}")
    print("top_parent_weight_mass")
    for parent, mass in by_parent_mass.most_common(10):
        print(f"  {parent}: {mass:.6f}")
    print("top_task_weight_mass")
    for task, mass in by_task_mass.most_common(10):
        print(f"  {task}: {mass:.6f}")


def main() -> None:
    args = parse_args()
    with args.ranked.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{args.ranked} is empty")
        rows = list(reader)

    stats = build_stats(rows)
    base_weights = resolved_base_weights(args)
    output_rows = []
    for row in rows:
        split = str(row.get("split", "")).strip() or "train"
        if args.split != "all" and split != args.split:
            continue
        take_uid = str(row.get("take_uid", "")).strip()
        if not take_uid:
            continue
        bucket = bucket_for_row(row)
        if args.preset == "v1":
            weight, components = v1_sample_weight(row, bucket, args, base_weights)
        else:
            weight, components = v2_sample_weight(row, bucket, args, base_weights, stats)
        output_rows.append(
            {
                "take_uid": take_uid,
                "split": split,
                "bucket": bucket,
                "sample_weight": f"{weight:.9f}",
                "task_name": row.get("task_name", ""),
                "parent_task_name": row.get("parent_task_name", ""),
                "ranker_confidence": row.get("ranker_confidence", ""),
                "ranker_disagreement_score": row.get("ranker_disagreement_score", ""),
                "ranker_prob_tokenizer_main": row.get("ranker_prob_tokenizer_main", ""),
                "ranker_prob_loco_aux": row.get("ranker_prob_loco_aux", ""),
                "ranker_prob_diagnostic_candidate": row.get("ranker_prob_diagnostic_candidate", ""),
                "ranker_prob_discard": row.get("ranker_prob_discard", ""),
                "prob_tokenizer_main": row.get("prob_tokenizer_main", ""),
                "prob_loco_aux": row.get("prob_loco_aux", ""),
                "prob_diagnostic_candidate": row.get("prob_diagnostic_candidate", ""),
                "prob_discard": row.get("prob_discard", ""),
                "interaction_score": row.get("interaction_score", ""),
                "loco_score": row.get("loco_score", ""),
                "object_motion_proxy": row.get("object_motion_proxy", ""),
                "phase_diversity_score_v2": row.get("phase_diversity_score_v2", ""),
                "motion_state_change_score": row.get("motion_state_change_score", ""),
                "scene_only_score": row.get("scene_only_score", ""),
                "prob_mix": f"{components['prob_mix']:.9f}",
                "feature_quality": f"{components['feature_quality']:.9f}",
                "risk_penalty": f"{components['risk_penalty']:.9f}",
                "diversity_multiplier": f"{components['diversity_multiplier']:.9f}",
                "mass_balance_multiplier": f"{1.0:.9f}",
                "raw_weight": f"{components['raw_weight']:.9f}",
            }
        )

    if args.preset == "v2":
        apply_mass_balance(output_rows, args)

    if args.preset == "v2" and output_rows and not args.no_normalize_max:
        max_seen = max(to_float(row["sample_weight"]) for row in output_rows)
        if max_seen > 0.0:
            scale = float(args.max_weight) / max_seen
            for row in output_rows:
                raw = to_float(row["sample_weight"])
                row["sample_weight"] = f"{raw * scale:.9f}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "take_uid",
        "split",
        "bucket",
        "sample_weight",
        "task_name",
        "parent_task_name",
        "ranker_confidence",
        "ranker_disagreement_score",
        "ranker_prob_tokenizer_main",
        "ranker_prob_loco_aux",
        "ranker_prob_diagnostic_candidate",
        "ranker_prob_discard",
        "prob_tokenizer_main",
        "prob_loco_aux",
        "prob_diagnostic_candidate",
        "prob_discard",
        "interaction_score",
        "loco_score",
        "object_motion_proxy",
        "phase_diversity_score_v2",
        "motion_state_change_score",
        "scene_only_score",
        "prob_mix",
        "feature_quality",
        "risk_penalty",
        "diversity_multiplier",
        "mass_balance_multiplier",
        "raw_weight",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved take weights to {args.out}")
    print_audit(output_rows)


if __name__ == "__main__":
    main()
