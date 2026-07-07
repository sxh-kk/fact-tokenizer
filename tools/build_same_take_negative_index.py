#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common import group_indices_by_take, load_npz_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-take hard negative index aligned to a filtered NPZ.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-gap", type=float, default=3.0)
    parser.add_argument("--max-gap", type=float, default=24.0)
    parser.add_argument("--view-keys", nargs=2, default=["ego", "exo"])
    return parser.parse_args()


def transition_features(video: np.ndarray) -> np.ndarray:
    values = video.astype(np.float32)
    if values.max(initial=0.0) > 1.5:
        values = values / 255.0
    if values.shape[-1] in (1, 3):
        current = values[:, 0].reshape(values.shape[0], -1)
        target = values[:, -1].reshape(values.shape[0], -1)
    else:
        current = values[:, 0].reshape(values.shape[0], -1)
        target = values[:, -1].reshape(values.shape[0], -1)
    delta = target - current
    delta = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)
    context = current / np.maximum(np.linalg.norm(current, axis=1, keepdims=True), 1e-6)
    return np.concatenate([0.7 * delta, 0.3 * context], axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    metadata = load_npz_metadata(args.npz)
    grouped = group_indices_by_take(metadata["take_uid"])
    with np.load(args.npz, allow_pickle=False) as data:
        features = [transition_features(np.asarray(data[view_key])) for view_key in args.view_keys]
    feature = np.concatenate(features, axis=1)
    feature = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-6)
    n = feature.shape[0]
    donors = np.full((n, args.top_k), -1, dtype=np.int64)
    scores = np.zeros((n, args.top_k), dtype=np.float32)
    raw_scores = np.full((n, args.top_k), -np.inf, dtype=np.float32)
    for take_uid, indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) <= 1:
            continue
        local_feature = feature[idx]
        distance = 1.0 - np.matmul(local_feature, local_feature.T)
        local_time = metadata["timestamp"][idx]
        gap = np.abs(local_time[:, None] - local_time[None, :])
        valid = (gap >= args.min_gap) & (gap <= args.max_gap) & (~np.eye(len(idx), dtype=bool))
        masked = np.where(valid, distance + 0.1 * np.minimum(gap / 12.0, 1.0), -np.inf)
        order = np.argsort(-masked, axis=1)[:, : args.top_k]
        values = np.take_along_axis(masked, order, axis=1)
        for row, anchor in enumerate(idx):
            finite = np.isfinite(values[row])
            count = int(finite.sum())
            if count == 0:
                continue
            donors[anchor, :count] = idx[order[row, :count]]
            raw_scores[anchor, :count] = values[row, :count]
            row_values = values[row, :count]
            if count == 1:
                scores[anchor, 0] = 1.0
            else:
                span = max(float(row_values.max() - row_values.min()), 1e-6)
                scores[anchor, :count] = 0.75 + 0.5 * (row_values - row_values.min()) / span
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        donor_index_topk=donors,
        donor_score_topk=scores,
        donor_raw_score_topk=raw_scores,
        take_uid=metadata["take_uid"].astype(str),
        timestamp=metadata["timestamp"].astype(np.float32),
        sample_id=metadata["sample_id"].astype(str),
    )
    valid = donors >= 0
    print(f"Saved same-take negative index to {args.out}; valid_pair_fraction={valid.mean():.4f}")


if __name__ == "__main__":
    main()
