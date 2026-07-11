#!/usr/bin/env python3
"""Extract per-view frozen effect features for take/view leakage NMI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_tokenizer.effect_data import EffectClipSpec, FACTEffectNPYDataset  # noqa: E402
from fact_tokenizer.effect_experiments import BRIDGE_EXPERIMENTS, PAIRED_CONTROLS, make_experiment_model  # noqa: E402
from fact_tokenizer.model import DINOv2PatchFeatureExtractor, MockPatchFeatureExtractor  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-home")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    fingerprint = checkpoint["run_fingerprint"]
    if fingerprint.get("vq_enabled"):
        raise ValueError("leakage extraction requires a continuous VQ-off checkpoint")
    model_config = fingerprint["model"]
    if model_config["backbone"] == "dinov2":
        backbone = DINOv2PatchFeatureExtractor(model_config["dino_model"], args.torch_home)
    else:
        backbone = MockPatchFeatureExtractor(3, int(model_config["backbone_dim"]), 14)
    experiment = fingerprint["experiment"]
    spec = BRIDGE_EXPERIMENTS.get(experiment) or PAIRED_CONTROLS[experiment]
    model = make_experiment_model(
        spec,
        backbone=backbone,
        backbone_dim=int(model_config["backbone_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        semantic_dim=int(model_config["semantic_dim"]),
        private_dim=int(model_config["private_dim"]),
        camera_context_dim=int(model_config["camera_context_dim"]),
        current_index=0,
        vq_enabled=False,
        geometry_mode="image2d",
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    dataset = FACTEffectNPYDataset(
        args.input_dir,
        EffectClipSpec(history_frames=1, future_frames=1, source_current_index=0, resize=224),
        view_keys=spec.view_names,
        manifest=args.manifest,
        role="probe",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    features: list[np.ndarray] = []
    sample_ids: list[str] = []
    take_uids: list[str] = []
    views: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            for view in spec.view_names:
                videos = batch["views"][view].to(device)
                patch_features = model._backbone_features(videos)
                effect = model.effect_encoders[view](patch_features)
                value = effect["z_sem_cont"].cpu().numpy()
                features.append(value)
                sample_ids.extend(batch["sample_id"])
                take_uids.extend(batch["take_uid"])
                views.extend([view] * len(value))
    fingerprint_payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        features=np.concatenate(features),
        sample_id=np.asarray(sample_ids),
        take_uid=np.asarray(take_uids),
        view=np.asarray(views),
        encoder_checkpoint_sha256=np.asarray(sha256_file(args.checkpoint)),
        run_fingerprint_sha256=np.asarray(hashlib.sha256(fingerprint_payload).hexdigest()),
        manifest_sha256=np.asarray(sha256_file(args.manifest)),
    )
    print(
        json.dumps(
            {
                "samples": len(sample_ids),
                "views": list(spec.view_names),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "output_sha256": sha256_file(args.output_npz),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
