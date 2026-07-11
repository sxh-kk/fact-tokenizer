"""Take-aware metrics and frozen linear-probe utilities for FACT v7."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("macro F1 needs equal non-empty label sequences")
    classes = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    scores = []
    for label in classes:
        true_positive = sum(truth == label and pred == label for truth, pred in zip(y_true, y_pred))
        false_positive = sum(truth != label and pred == label for truth, pred in zip(y_true, y_pred))
        false_negative = sum(truth == label and pred != label for truth, pred in zip(y_true, y_pred))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def take_bootstrap(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    take_uids: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
    iterations: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    if not (len(y_true) == len(y_pred) == len(take_uids)):
        raise ValueError("bootstrap arrays must have equal length")
    takes = sorted(set(take_uids))
    if len(takes) < 2:
        raise ValueError("take bootstrap needs at least two takes")
    by_take = {take: np.flatnonzero(np.asarray(take_uids) == take) for take in takes}
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    for iteration in range(iterations):
        sampled = rng.choice(takes, size=len(takes), replace=True)
        indices = np.concatenate([by_take[take] for take in sampled])
        values[iteration] = macro_f1(truth[indices].tolist(), prediction[indices].tolist(), labels)
    point = macro_f1(y_true, y_pred, labels)
    return {
        "macro_f1": point,
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "take",
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }

def paired_take_bootstrap_delta(
    y_true: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    take_uids: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
    iterations: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    if not (len(y_true) == len(predictions_a) == len(predictions_b) == len(take_uids)):
        raise ValueError("paired delta inputs must have equal length")
    takes = sorted(set(take_uids))
    by_take = {take: np.flatnonzero(np.asarray(take_uids) == take) for take in takes}
    truth = np.asarray(y_true)
    first = np.asarray(predictions_a)
    second = np.asarray(predictions_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(takes, size=len(takes), replace=True)
        indices = np.concatenate([by_take[take] for take in sampled])
        deltas[iteration] = macro_f1(truth[indices].tolist(), first[indices].tolist(), labels) - macro_f1(
            truth[indices].tolist(), second[indices].tolist(), labels
        )
    point = macro_f1(y_true, predictions_a, labels) - macro_f1(y_true, predictions_b, labels)
    return {
        "macro_f1_delta": point,
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "take",
    }


def normalized_mutual_information(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("NMI needs equal non-empty label sequences")
    count = len(labels_a)
    joint = Counter(zip(labels_a, labels_b))
    marginal_a = Counter(labels_a)
    marginal_b = Counter(labels_b)
    mutual_information = 0.0
    for (a, b), joint_count in joint.items():
        probability = joint_count / count
        mutual_information += probability * np.log(probability / ((marginal_a[a] / count) * (marginal_b[b] / count)))
    entropy_a = -sum((value / count) * np.log(value / count) for value in marginal_a.values())
    entropy_b = -sum((value / count) * np.log(value / count) for value in marginal_b.values())
    denominator = np.sqrt(entropy_a * entropy_b)
    return 0.0 if denominator == 0 else float(mutual_information / denominator)


def paired_value_go_decision(
    comparisons: Mapping[str, Mapping[str, float | Sequence[float]]],
    seed_deltas: Mapping[str, Sequence[float]],
    leakage_nmi_change: float,
) -> dict[str, Any]:
    """Apply the frozen P2-versus-control GO thresholds without moving them."""

    reasons: list[str] = []
    thresholds = {"P0": 0.05, "P3": 0.03, "P4": 0.03}
    for baseline, threshold in thresholds.items():
        comparison = comparisons.get(baseline)
        if comparison is None:
            reasons.append(f"missing P2 vs {baseline} comparison")
            continue
        delta = float(comparison["macro_f1_delta"])
        ci = list(comparison["ci95"])
        if delta < threshold:
            reasons.append(f"P2 vs {baseline} delta {delta:.4f} < {threshold:.4f}")
        if float(ci[0]) <= 0:
            reasons.append(f"P2 vs {baseline} CI lower bound is not > 0")
        values = list(seed_deltas.get(baseline, []))
        if len(values) != 3 or any(value <= 0 for value in values):
            reasons.append(f"P2 vs {baseline} does not improve in all three seeds")
    if leakage_nmi_change > 0.02:
        reasons.append(f"take/view leakage NMI worsened by {leakage_nmi_change:.4f} > 0.0200")
    return {
        "go": not reasons,
        "decision": "GO" if not reasons else "NO_GO",
        "reasons": reasons,
        "thresholds": {
            "P2_vs_P0_macro_f1_delta": 0.05,
            "P2_vs_P3_P4_macro_f1_delta": 0.03,
            "ci95_lower_bound_strictly_positive": True,
            "three_seed_direction_consistency": True,
            "maximum_leakage_nmi_worsening": 0.02,
        },
    }
