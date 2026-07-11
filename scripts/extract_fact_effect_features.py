#!/usr/bin/env python3
"""Extract frozen Ego ``z_sem_cont`` features without invoking cross/private paths."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-home")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    fingerprint = checkpoint["run_fingerprint"]
    model_config = fingerprint["model"]
    if fingerprint.get("vq_enabled") or any(
        "vq" in key.lower() or "codebook" in key.lower() for key in checkpoint["model"]
    ):
        raise ValueError("feature extraction only accepts continuous VQ-off checkpoints")
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
    model.eval().to(device)
    dataset = FACTEffectNPYDataset(
        args.input_dir,
        EffectClipSpec(history_frames=1, future_frames=1, source_current_index=0, resize=224),
        view_keys=("ego",),
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
    features = []
    sample_ids: list[str] = []
    take_uids: list[str] = []
    timestamps = []
    with torch.inference_mode():
        for batch in loader:
            videos = batch["views"]["ego"].to(device)
            patch_features = model._backbone_features(videos)
            effect = model.effect_encoders["ego"](patch_features)
            features.append(effect["z_sem_cont"].cpu().numpy())
            sample_ids.extend(batch["sample_id"])
            take_uids.extend(batch["take_uid"])
            timestamps.extend(batch["timestamp"].numpy().tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", np.concatenate(features))
    np.save(args.output_dir / "sample_id.npy", np.asarray(sample_ids))
    np.save(args.output_dir / "take_uid.npy", np.asarray(take_uids))
    np.save(args.output_dir / "timestamp.npy", np.asarray(timestamps, dtype=np.float32))
    metadata = {
        "schema": "fact-v7-effect-features-v1",
        "encoder_frozen": True,
        "encoder_checkpoint": str(args.checkpoint),
        "encoder_checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "samples": len(sample_ids),
        "feature_dim": int(features[0].shape[1]),
        "feature_name": "ego.z_sem_cont",
        "vq_enabled": False,
    }
    (args.output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
