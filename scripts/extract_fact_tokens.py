#!/usr/bin/env python3
"""Export ego shared-action FACT tokens from a trained FACT tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fact_tokenizer import FACTPairedNPZDataset, FACTTokenizer
from fact_tokenizer.losses import reconstruction_losses
from fact_tokenizer.utils import code_usage, concat_batches, load_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, default=ROOT / "outputs" / "lam_tokenizer" / "dummy_multiview.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fact_tokenizer" / "extracted_tokens")
    parser.add_argument("--source-view-keys", nargs="*", default=None)
    parser.add_argument("--view-names", nargs=2, default=None)
    parser.add_argument("--videos-layout", default="VBTCHW")
    parser.add_argument("--frame-pair", choices=["first-last", "first-next"], default="first-last")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-recon-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = dict(checkpoint["model_config"])
    if args.view_names:
        model_config["view_names"] = tuple(args.view_names)
    view_names = list(model_config.get("view_names", ("ego", "exo")))
    dataset = FACTPairedNPZDataset(
        input_npz=args.input_npz,
        source_view_keys=args.source_view_keys,
        output_view_names=view_names,
        videos_layout=args.videos_layout,
        frame_pair=args.frame_pair,
        start_index=args.start_index,
        resize=args.resize,
    )
    model = FACTTokenizer(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    indices = []
    soft_probs = []
    confidence = []
    recon_metrics = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            encoded = model.encode_shared_action(batch, view_name=view_names[0])
            indices.append(encoded["indices"].cpu())
            soft_probs.append(encoded["soft_probs"].cpu())
            confidence.append(encoded["confidence"].cpu())
            if not args.skip_recon_metrics:
                outputs = model(batch)
                for key, value in reconstruction_losses(outputs).items():
                    recon_metrics[f"batch_{batch_index}/{key}/feature_mse"] = float(value.cpu())

    indices_tensor = concat_batches(indices)
    soft_probs_tensor = concat_batches(soft_probs)
    confidence_tensor = concat_batches(confidence)
    np.savez_compressed(
        args.output_dir / "ego_tokens.npz",
        indices=indices_tensor.numpy(),
        soft_probs=soft_probs_tensor.numpy(),
        confidence=confidence_tensor.numpy(),
    )
    save_json(args.output_dir / "code_usage.json", code_usage(indices_tensor, model.num_latents))
    save_json(
        args.output_dir / "metadata.json",
        {
            "view_names": view_names,
            "exported_view": view_names[0],
            "shape_semantics": "batch x transitions x num_action_slots",
            "indices_shape": list(indices_tensor.shape),
            "soft_probs_shape": list(soft_probs_tensor.shape),
            "confidence_shape": list(confidence_tensor.shape),
            "checkpoint": str(args.checkpoint),
            "input_npz": str(args.input_npz),
        },
    )
    if recon_metrics:
        save_json(args.output_dir / "recon_metrics.json", recon_metrics)
    print(f"Saved ego FACT tokens to {args.output_dir / 'ego_tokens.npz'}")


if __name__ == "__main__":
    main()
