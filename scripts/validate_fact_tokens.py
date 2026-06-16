#!/usr/bin/env python3
"""Mechanism-level validation report for exported FACT tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fact_tokenizer.utils import code_usage, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-npz", type=Path, required=True)
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.tokens_npz)
    indices = data["indices"]
    soft_probs = data["soft_probs"]
    confidence = data["confidence"]

    entropy = -(soft_probs * np.log(np.clip(soft_probs, 1e-8, 1.0))).sum(axis=-1)
    report = {
        "indices_shape": list(indices.shape),
        "soft_probs_shape": list(soft_probs.shape),
        "confidence_shape": list(confidence.shape),
        "token_min": int(indices.min()) if indices.size else None,
        "token_max": int(indices.max()) if indices.size else None,
        "confidence_mean": float(confidence.mean()) if confidence.size else None,
        "confidence_min": float(confidence.min()) if confidence.size else None,
        "confidence_max": float(confidence.max()) if confidence.size else None,
        "entropy_mean": float(entropy.mean()) if entropy.size else None,
        "usage": code_usage(torch.from_numpy(indices), args.num_latents),
        "private_residual_exported": False,
    }
    if report["token_min"] is not None:
        report["tokens_in_range"] = report["token_min"] >= 0 and report["token_max"] < args.num_latents
    else:
        report["tokens_in_range"] = False

    output_json = args.output_json or args.tokens_npz.with_name("validation_report.json")
    save_json(output_json, report)
    print(f"Saved FACT token validation report to {output_json}")


if __name__ == "__main__":
    main()
