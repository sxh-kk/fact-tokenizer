#!/usr/bin/env python3
"""Evaluate whether a FACT tokenizer run passes the Stage-1 action-token gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-ego-random-take-delta", type=float, default=0.015)
    parser.add_argument("--min-ego-random-code-delta", type=float, default=0.025)
    parser.add_argument("--min-ego-same-take-delta", type=float, default=0.010)
    parser.add_argument("--min-exo-random-take-delta", type=float, default=0.002)
    parser.add_argument("--min-exo-zero-delta", type=float, default=0.006)
    parser.add_argument("--max-tuple-view-nmi", type=float, default=0.10)
    parser.add_argument("--max-tuple-take-nmi", type=float, default=0.35)
    parser.add_argument("--min-ego-used-codes", type=int, default=45)
    parser.add_argument("--max-ego-used-codes", type=int, default=55)
    parser.add_argument("--min-ego-action-without-private-saving", type=float, default=0.018)
    parser.add_argument("--max-private-only-action-gap", type=float, default=0.012)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dig(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def metric(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value = dig(payload, *keys)
    return float(default if value is None else value)


def add_gate(gates: list[dict[str, Any]], name: str, value: float, threshold: float, direction: str) -> None:
    if direction == ">=":
        passed = value >= threshold
    elif direction == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported gate direction: {direction}")
    gates.append(
        {
            "name": name,
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "pass": bool(passed),
        }
    )


def first_take_nmi(semantic: dict[str, Any], view_name: str) -> float:
    return metric(semantic, "automatic_controls", "take_leakage", view_name, "tuple", "nmi_sqrt")


def main() -> int:
    args = parse_args()
    probe_dir = args.probe_dir
    summary = load_json(probe_dir / "probe_summary.json")
    semantic = load_json(probe_dir / "semantic_probe.json")

    gates: list[dict[str, Any]] = []
    add_gate(
        gates,
        "ego_swap_random_take_delta",
        metric(summary, "causality", "ego_swap", "negative_controls", "random_take", "delta_vs_correct_mean"),
        args.min_ego_random_take_delta,
        ">=",
    )
    add_gate(
        gates,
        "ego_swap_random_code_delta",
        metric(summary, "causality", "ego_swap", "negative_controls", "random_code", "delta_vs_correct_mean"),
        args.min_ego_random_code_delta,
        ">=",
    )
    add_gate(
        gates,
        "ego_swap_same_take_delta",
        metric(summary, "causality", "ego_swap", "negative_controls", "same_take_shuffle", "delta_vs_correct_mean"),
        args.min_ego_same_take_delta,
        ">=",
    )
    add_gate(
        gates,
        "exo_swap_random_take_delta",
        metric(summary, "causality", "exo_swap", "negative_controls", "random_take", "delta_vs_correct_mean"),
        args.min_exo_random_take_delta,
        ">=",
    )
    add_gate(
        gates,
        "exo_swap_zero_delta",
        metric(summary, "causality", "exo_swap", "negative_controls", "zero", "delta_vs_correct_mean"),
        args.min_exo_zero_delta,
        ">=",
    )
    add_gate(
        gates,
        "ego_action_without_private_saving",
        metric(
            summary,
            "private_leakage",
            "ego_swap",
            "key_readouts",
            "mse_saved_by_correct_action_without_private_vs_zero_action",
        ),
        args.min_ego_action_without_private_saving,
        ">=",
    )
    add_gate(
        gates,
        "ego_private_only_gap",
        metric(summary, "private_leakage", "ego_swap", "key_readouts", "private_only_without_action", "delta_vs_correct_mean"),
        args.max_private_only_action_gap,
        "<=",
    )

    view_nmi = metric(semantic, "automatic_controls", "view_invariance", "tuple", "nmi_sqrt")
    take_nmi = max(first_take_nmi(semantic, "ego"), first_take_nmi(semantic, "exo"))
    ego_usage = dig(semantic, "views", "ego", "usage") or {}
    ego_used_codes = int(ego_usage.get("used_codes", 0))
    add_gate(gates, "tuple_view_nmi", view_nmi, args.max_tuple_view_nmi, "<=")
    add_gate(gates, "tuple_take_nmi", take_nmi, args.max_tuple_take_nmi, "<=")
    add_gate(gates, "ego_used_codes_min", float(ego_used_codes), float(args.min_ego_used_codes), ">=")
    add_gate(gates, "ego_used_codes_max", float(ego_used_codes), float(args.max_ego_used_codes), "<=")

    report = {
        "probe_dir": str(probe_dir),
        "passed": all(gate["pass"] for gate in gates),
        "gates": gates,
        "metrics": {
            "tuple_view_nmi": view_nmi,
            "tuple_take_nmi": take_nmi,
            "ego_used_codes": ego_used_codes,
            "ego_usage_fraction": ego_usage.get("usage_fraction"),
        },
    }
    output_json = args.output_json or probe_dir / "stage1_gate.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
