"""Shared utility helpers for FACT tokenizer scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


def code_usage(indices: torch.Tensor | np.ndarray, num_latents: int) -> dict:
    tensor = torch.as_tensor(indices).reshape(-1).cpu()
    counts = torch.bincount(tensor.long(), minlength=num_latents).tolist()
    used = sum(1 for count in counts if count)
    return {
        "num_latents": int(num_latents),
        "total_tokens": int(tensor.numel()),
        "used_codes": int(used),
        "usage_fraction": float(used / num_latents),
        "histogram": {str(index): int(count) for index, count in enumerate(counts)},
    }


def save_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unsupported FACT checkpoint format: {path}")
    return checkpoint


def concat_batches(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    collected = list(tensors)
    if not collected:
        raise RuntimeError("No tensors were collected")
    return torch.cat(collected, dim=0)
