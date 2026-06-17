#!/usr/bin/env python3
"""Create diagnostic plots for a FACT tokenizer run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_json(path: Path):
    if not path.exists():
        return None
    return load_json(path)


def row_value(row: dict, key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    return float(value) if value is not None else float(default)


def series(history: list[dict], key: str, default: float = np.nan) -> list[float]:
    return [row_value(row, key, default=default) for row in history]


def plot_training(history: list[dict], out_dir: Path) -> None:
    steps = [row["step"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, series(history, "loss"), label="total")
    axes[0, 0].plot(steps, series(history, "self_loss"), label="self")
    axes[0, 0].plot(steps, series(history, "swap_loss"), label="swap")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    for key in ["ego_self/feature_mse", "exo_self/feature_mse", "ego_swap/feature_mse", "exo_swap/feature_mse"]:
        axes[0, 1].plot(steps, series(history, key), label=key.replace("/feature_mse", ""))
    axes[0, 1].set_title("DINO Feature Reconstruction MSE")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, series(history, "vq_loss"), label="vq")
    axes[1, 0].plot(steps, series(history, "kl_loss"), label="exo->ego KL")
    axes[1, 0].plot(steps, series(history, "balance_loss"), label="balance")
    if "hard_usage_balance_loss" in history[-1]:
        axes[1, 0].plot(steps, series(history, "hard_usage_balance_loss"), label="hard usage")
    axes[1, 0].set_title("Tokenizer Auxiliary Losses")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(steps, series(history, "weight_self"), label="self weight")
    axes[1, 1].plot(steps, series(history, "weight_swap"), label="swap weight")
    axes[1, 1].plot(steps, series(history, "weight_kl"), label="kl weight")
    if "private_dropout" in history[-1]:
        axes[1, 1].plot(steps, series(history, "private_dropout"), label="private dropout")
    if "action_slot_dropout" in history[-1]:
        axes[1, 1].plot(steps, series(history, "action_slot_dropout"), label="action slot dropout")
    axes[1, 1].set_title("Loss Schedule")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_action_diagnostics(history: list[dict], out_dir: Path) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for key in ["action_top1_agreement", "assignment_entropy_mean"]:
        if key in history[-1]:
            axes[0, 0].plot(steps, series(history, key), label=key)
    axes[0, 0].set_title("Shared Action Assignment")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    for key in ["action_contrast_loss", "same_take_contrast_loss", "no_private_contrast_loss"]:
        if key in history[-1]:
            axes[0, 1].plot(steps, series(history, key), label=key)
    axes[0, 1].set_title("Action Contrast Losses")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    for key in ["take_uniformity_loss", "take_slot_uniformity_loss", "take_pair_uniformity_loss"]:
        if key in history[-1]:
            axes[1, 0].plot(steps, series(history, key), label=key)
    axes[1, 0].set_title("Take Leakage Regularizers")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    for key in ["action_only_loss", "motion_focus_loss", "delta_focus_loss", "private_reg"]:
        if key in history[-1]:
            axes[1, 1].plot(steps, series(history, key), label=key)
    axes[1, 1].set_title("Private Separation / Motion Focus")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.savefig(out_dir / "action_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_code_usage_summary(code_usage: dict, out_dir: Path) -> None:
    histogram = code_usage["histogram"]
    code_ids = [int(key) for key in histogram.keys()]
    counts = np.asarray([int(histogram[str(key)]) for key in code_ids])
    used = counts > 0
    nonzero = counts[used]
    total = max(int(counts.sum()), 1)
    probs = counts / total
    effective_codes = float(np.exp(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    colors = np.where(used, "#3b82f6", "#cbd5e1")
    axes[0].bar(code_ids, counts, color=colors)
    axes[0].axhline(total / len(code_ids), color="#ef4444", linestyle="--", linewidth=1, label="uniform target")
    axes[0].set_title(
        f"Hard Code Usage: {code_usage['used_codes']}/{code_usage['num_latents']} used "
        f"(effective {effective_codes:.1f})"
    )
    axes[0].set_xlabel("code id")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    if nonzero.size:
        axes[1].hist(nonzero, bins=min(20, max(5, nonzero.size)), color="#14b8a6")
    axes[1].set_title("Nonzero Code Count Distribution")
    axes[1].set_xlabel("tokens assigned to a used code")
    axes[1].set_ylabel("num codes")
    axes[1].grid(axis="y", alpha=0.3)

    fig.savefig(out_dir / "code_usage.png", dpi=180)
    plt.close(fig)


def plot_tokens(tokens_npz: Path, validation: dict, out_dir: Path) -> None:
    data = np.load(tokens_npz)
    indices = data["indices"]
    soft_probs = data["soft_probs"]
    confidence = data["confidence"]
    entropy = -(soft_probs * np.log(np.clip(soft_probs, 1e-8, 1.0))).sum(axis=-1)

    histogram = validation["usage"]["histogram"]
    code_ids = [int(key) for key in histogram.keys()]
    counts = [int(histogram[str(key)]) for key in code_ids]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(code_ids, counts)
    axes[0, 0].set_title("Hard Code Usage")
    axes[0, 0].set_xlabel("code id")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_xticks(code_ids)
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].hist(confidence.reshape(-1), bins=20)
    axes[0, 1].set_title("Assignment Confidence")
    axes[0, 1].set_xlabel("confidence")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].hist(entropy.reshape(-1), bins=20)
    axes[1, 0].set_title("Soft Assignment Entropy")
    axes[1, 0].set_xlabel("entropy")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(axis="y", alpha=0.3)

    token_grid = indices.reshape(indices.shape[0], -1)
    image = axes[1, 1].imshow(token_grid.T, aspect="auto", interpolation="nearest", cmap="tab20")
    axes[1, 1].set_title("Ego Token Timeline")
    axes[1, 1].set_xlabel("sample")
    axes[1, 1].set_ylabel("transition x slot")
    fig.colorbar(image, ax=axes[1, 1], label="code id")

    fig.savefig(out_dir / "token_diagnostics.png", dpi=180)
    plt.close(fig)


def write_summary(run_dir: Path, validation: dict | None, code_usage: dict | None, history: list[dict], out_dir: Path) -> None:
    first = history[0]
    last = history[-1]
    summary = {
        "run_dir": str(run_dir),
        "steps": len(history),
        "loss_start": first["loss"],
        "loss_end": last["loss"],
        "ego_self_mse_start": first["ego_self/feature_mse"],
        "ego_self_mse_end": last["ego_self/feature_mse"],
        "ego_swap_mse_start": first["ego_swap/feature_mse"],
        "ego_swap_mse_end": last["ego_swap/feature_mse"],
        "action_top1_agreement_end": last.get("action_top1_agreement"),
    }
    if code_usage is not None:
        counts = np.asarray([int(value) for value in code_usage["histogram"].values()])
        probs = counts / max(int(counts.sum()), 1)
        summary.update(
            {
                "used_codes": code_usage["used_codes"],
                "usage_fraction": code_usage["usage_fraction"],
                "effective_codes": float(np.exp(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())),
                "max_code_fraction": float(counts.max() / max(int(counts.sum()), 1)),
            }
        )
    if validation is not None:
        summary.update(
            {
                "validation_used_codes": validation["usage"]["used_codes"],
                "validation_usage_fraction": validation["usage"]["usage_fraction"],
                "confidence_mean": validation["confidence_mean"],
                "entropy_mean": validation["entropy_mean"],
                "tokens_in_range": validation["tokens_in_range"],
                "private_residual_exported": validation["private_residual_exported"],
            }
        )
    with (out_dir / "visual_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokens-npz", type=Path, default=None)
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_json(run_dir / "train_history.json")
    tokens_npz = args.tokens_npz or run_dir / "extracted" / "ego_tokens.npz"
    validation_json = args.validation_json or run_dir / "extracted" / "validation_report.json"
    validation = maybe_load_json(validation_json)
    code_usage = maybe_load_json(run_dir / "code_usage.json")

    plot_training(history, output_dir)
    plot_action_diagnostics(history, output_dir)
    if code_usage is not None:
        plot_code_usage_summary(code_usage, output_dir)
    if tokens_npz.exists() and validation is not None:
        plot_tokens(tokens_npz, validation, output_dir)
    write_summary(run_dir, validation, code_usage, history, output_dir)
    print(f"Wrote FACT visualizations to {output_dir}")


if __name__ == "__main__":
    main()
