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


def plot_training(history: list[dict], out_dir: Path) -> None:
    steps = [row["step"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [row["loss"] for row in history], label="total")
    axes[0, 0].plot(steps, [row["self_loss"] for row in history], label="self")
    axes[0, 0].plot(steps, [row["swap_loss"] for row in history], label="swap")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    for key in ["ego_self/feature_mse", "exo_self/feature_mse", "ego_swap/feature_mse", "exo_swap/feature_mse"]:
        axes[0, 1].plot(steps, [row[key] for row in history], label=key.replace("/feature_mse", ""))
    axes[0, 1].set_title("DINO Feature Reconstruction MSE")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, [row["vq_loss"] for row in history], label="vq")
    axes[1, 0].plot(steps, [row["kl_loss"] for row in history], label="exo->ego KL")
    axes[1, 0].plot(steps, [row["balance_loss"] for row in history], label="balance")
    axes[1, 0].set_title("Tokenizer Auxiliary Losses")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(steps, [row["weight_self"] for row in history], label="self weight")
    axes[1, 1].plot(steps, [row["weight_swap"] for row in history], label="swap weight")
    axes[1, 1].plot(steps, [row["weight_kl"] for row in history], label="kl weight")
    axes[1, 1].set_title("Loss Schedule")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.savefig(out_dir / "training_curves.png", dpi=180)
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


def write_summary(run_dir: Path, validation: dict, history: list[dict], out_dir: Path) -> None:
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
        "used_codes": validation["usage"]["used_codes"],
        "usage_fraction": validation["usage"]["usage_fraction"],
        "confidence_mean": validation["confidence_mean"],
        "entropy_mean": validation["entropy_mean"],
        "tokens_in_range": validation["tokens_in_range"],
        "private_residual_exported": validation["private_residual_exported"],
    }
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
    validation = load_json(validation_json)

    plot_training(history, output_dir)
    plot_tokens(tokens_npz, validation, output_dir)
    write_summary(run_dir, validation, history, output_dir)
    print(f"Wrote FACT visualizations to {output_dir}")


if __name__ == "__main__":
    main()
