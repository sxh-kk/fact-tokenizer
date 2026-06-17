#!/usr/bin/env python3
"""Summarize FACT action-token probes and apply freeze/readiness gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--probe-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-used-codes", type=int, default=35)
    parser.add_argument("--min-effective-codes", type=float, default=16.0)
    parser.add_argument("--max-view-nmi", type=float, default=0.12)
    parser.add_argument("--max-take-nmi", type=float, default=0.50)
    parser.add_argument("--min-ego-shuffle-delta", type=float, default=0.010)
    parser.add_argument("--min-ego-random-delta", type=float, default=0.015)
    parser.add_argument("--min-exo-random-delta", type=float, default=0.002)
    parser.add_argument("--min-exo-zero-delta", type=float, default=0.004)
    parser.add_argument("--max-private-zero-delta", type=float, default=0.006)
    parser.add_argument("--min-action-private-ratio", type=float, default=3.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_probe_dir(path: Path) -> Path:
    if (path / "action_token_probe_report.json").exists():
        return path
    candidate = path / "action_token_probe"
    if (candidate / "action_token_probe_report.json").exists():
        return candidate
    raise FileNotFoundError(f"Could not find action_token_probe_report.json under {path}")


def effective_codes(histogram: dict[str, int]) -> float:
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        if count:
            prob = int(count) / total
            entropy -= prob * math.log(prob)
    return float(math.exp(entropy))


def max_code_fraction(histogram: dict[str, int]) -> float:
    total = sum(int(value) for value in histogram.values())
    return float(max(histogram.values()) / total) if total else 0.0


def index_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row.get(key) for key in keys): row for row in rows}


def row_float(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return float("nan")
    value = row.get(key)
    return float(value) if value is not None else float("nan")


def metric_from_semantic(semantic: dict[str, Any], *path: str) -> float:
    node: Any = semantic
    for key in path:
        node = node[key]
    return float(node)


def pass_gate(value: float, threshold: float, direction: str) -> bool:
    if math.isnan(value):
        return False
    if direction == ">=":
        return value >= threshold
    if direction == "<=":
        return value <= threshold
    raise ValueError(direction)


def summarize_probe(probe_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(probe_dir / "action_token_probe_report.json")
    semantic = load_json(probe_dir / "semantic_probe.json")
    causality_rows = load_json(probe_dir / "causality_ablation.json").get("rows", [])
    leakage_rows = load_json(probe_dir / "private_leakage_ablation.json").get("rows", [])
    causality = index_rows(causality_rows, "path", "action_mode")
    leakage = index_rows(leakage_rows, "path", "action_mode", "private_mode", "private_dropout")

    out: dict[str, Any] = {
        "probe_dir": str(probe_dir),
        "run_dir": str(probe_dir.parent),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_step": report.get("checkpoint_step"),
        "input_npz": report.get("input_npz"),
        "num_samples": report.get("num_samples"),
    }

    for view in ("ego", "exo"):
        usage = report["token_usage"][view]["usage"]
        histogram = {key: int(value) for key, value in usage["histogram"].items()}
        out[f"{view}_used_codes"] = int(usage["used_codes"])
        out[f"{view}_effective_codes"] = effective_codes(histogram)
        out[f"{view}_max_code_fraction"] = max_code_fraction(histogram)
        out[f"{view}_confidence_mean"] = float(report["token_usage"][view].get("confidence_mean", float("nan")))
        out[f"{view}_task_nmi"] = metric_from_semantic(
            semantic,
            "views",
            view,
            "tokens",
            "tuple",
            "task_name",
            "metrics",
            "nmi_sqrt",
        )
        out[f"{view}_take_nmi"] = metric_from_semantic(
            semantic,
            "automatic_controls",
            "take_leakage",
            view,
            "tuple",
            "nmi_sqrt",
        )

    out["view_nmi"] = metric_from_semantic(
        semantic,
        "automatic_controls",
        "view_invariance",
        "tuple",
        "nmi_sqrt",
    )

    for path in ("ego_self", "ego_swap", "exo_self", "exo_swap"):
        correct = causality.get((path, "correct"))
        global_shuffle = causality.get((path, "global_shuffle"))
        random_code = causality.get((path, "random_code"))
        zero = causality.get((path, "zero"))
        out[f"{path}_correct_mean"] = row_float(correct, "mean")
        out[f"{path}_global_delta"] = row_float(global_shuffle, "delta_vs_correct_mean")
        out[f"{path}_random_delta"] = row_float(random_code, "delta_vs_correct_mean")
        out[f"{path}_zero_delta"] = row_float(zero, "delta_vs_correct_mean")

        private_shuffle = leakage.get((path, "correct", "global_shuffle", None))
        private_zero = leakage.get((path, "correct", "zero", None))
        out[f"{path}_private_shuffle_delta"] = row_float(private_shuffle, "delta_vs_correct_mean")
        out[f"{path}_private_zero_delta"] = row_float(private_zero, "delta_vs_correct_mean")

    ego_action_delta = max(out["ego_self_random_delta"], out["ego_swap_random_delta"])
    ego_private_delta = max(out["ego_self_private_zero_delta"], out["ego_swap_private_zero_delta"])
    action_private_ratio = ego_action_delta / max(ego_private_delta, 1e-8)
    out["ego_action_private_ratio"] = action_private_ratio

    gates = {
        "confidence": min(out["ego_confidence_mean"], out["exo_confidence_mean"]) >= args.min_confidence,
        "used_codes": min(out["ego_used_codes"], out["exo_used_codes"]) >= args.min_used_codes,
        "effective_codes": min(out["ego_effective_codes"], out["exo_effective_codes"]) >= args.min_effective_codes,
        "ego_shuffle_causality": max(out["ego_self_global_delta"], out["ego_swap_global_delta"])
        >= args.min_ego_shuffle_delta,
        "ego_random_causality": max(out["ego_self_random_delta"], out["ego_swap_random_delta"])
        >= args.min_ego_random_delta,
        "exo_random_causality": max(out["exo_self_random_delta"], out["exo_swap_random_delta"])
        >= args.min_exo_random_delta,
        "exo_zero_causality": max(out["exo_self_zero_delta"], out["exo_swap_zero_delta"]) >= args.min_exo_zero_delta,
        "private_separation": ego_private_delta <= args.max_private_zero_delta
        and action_private_ratio >= args.min_action_private_ratio,
        "view_leakage": out["view_nmi"] <= args.max_view_nmi,
        "take_leakage": max(out["ego_take_nmi"], out["exo_take_nmi"]) <= args.max_take_nmi,
    }
    out["gates"] = gates
    out["passed"] = all(gates.values())
    return out


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key != "gates"}
    for gate, passed in row["gates"].items():
        flat[f"gate_{gate}"] = bool(passed)
    return flat


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "probe_dir",
        "passed",
        "ego_confidence_mean",
        "exo_confidence_mean",
        "ego_used_codes",
        "exo_used_codes",
        "ego_effective_codes",
        "exo_effective_codes",
        "ego_swap_random_delta",
        "exo_swap_random_delta",
        "ego_action_private_ratio",
        "view_nmi",
        "ego_take_nmi",
        "exo_take_nmi",
    ]
    print("\t".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print("\t".join(values))


def main() -> None:
    args = parse_args()
    probe_dirs = [resolve_probe_dir(path) for path in args.run_dir] + [resolve_probe_dir(path) for path in args.probe_dir]
    if not probe_dirs:
        raise SystemExit("Provide at least one --run-dir or --probe-dir")
    rows = [summarize_probe(path, args) for path in probe_dirs]
    print_table(rows)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump({"rows": rows}, handle, indent=2)
    if args.output_csv:
        flat_rows = [flatten(row) for row in rows]
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)


if __name__ == "__main__":
    main()
