from __future__ import annotations

import pytest

from fact_tokenizer.effect_eval import (
    macro_f1,
    normalized_mutual_information,
    paired_take_bootstrap_delta,
    paired_value_go_decision,
    take_bootstrap,
)


def test_macro_f1_and_take_bootstrap_are_deterministic() -> None:
    truth = ["a", "b", "a", "b"]
    prediction = ["a", "b", "b", "b"]
    assert macro_f1(truth, prediction, ["a", "b"]) == pytest.approx((2 / 3 + 0.8) / 2)
    first = take_bootstrap(truth, prediction, ["t0", "t0", "t1", "t1"], labels=["a", "b"], iterations=100, seed=3)
    second = take_bootstrap(truth, prediction, ["t0", "t0", "t1", "t1"], labels=["a", "b"], iterations=100, seed=3)
    assert first == second


def test_paired_delta_uses_same_take_resamples() -> None:
    truth = ["a", "b", "a", "b"]
    strong = truth
    weak = ["b", "a", "b", "a"]
    report = paired_take_bootstrap_delta(
        truth,
        strong,
        weak,
        ["t0", "t0", "t1", "t1"],
        labels=["a", "b"],
        iterations=100,
    )
    assert report["macro_f1_delta"] == pytest.approx(1.0)
    assert report["ci95"][0] > 0


def test_nmi_and_frozen_go_decision() -> None:
    assert normalized_mutual_information(["a", "a", "b", "b"], ["x", "x", "y", "y"]) == pytest.approx(1.0)
    comparisons = {
        "P0": {"macro_f1_delta": 0.06, "ci95": [0.01, 0.10]},
        "P3": {"macro_f1_delta": 0.04, "ci95": [0.01, 0.08]},
        "P4": {"macro_f1_delta": 0.04, "ci95": [0.005, 0.07]},
    }
    seed_deltas = {baseline: [0.01, 0.02, 0.03] for baseline in comparisons}
    assert paired_value_go_decision(comparisons, seed_deltas, 0.01)["go"] is True
    assert paired_value_go_decision(comparisons, seed_deltas, 0.021)["go"] is False
