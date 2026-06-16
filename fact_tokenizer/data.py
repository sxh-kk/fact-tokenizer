"""Data loading utilities for paired ego/exo FACT tokenizer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


VIDEO_LAYOUTS = {"VBTCHW", "BVTCHW", "BTVCHW", "VTBHWC", "BVTHWC", "BTVHWC"}


def _normalize_videos_array(array: np.ndarray, layout: str) -> np.ndarray:
    if array.ndim != 6:
        raise ValueError(f"Expected a 6D videos array, got shape {array.shape}")
    if layout == "VBTCHW":
        return array
    if layout == "BVTCHW":
        return np.transpose(array, (1, 0, 2, 3, 4, 5))
    if layout == "BTVCHW":
        return np.transpose(array, (2, 0, 1, 3, 4, 5))
    if layout == "VTBHWC":
        return np.transpose(array, (0, 2, 1, 5, 3, 4))
    if layout == "BVTHWC":
        return np.transpose(array, (1, 0, 2, 5, 3, 4))
    if layout == "BTVHWC":
        return np.transpose(array, (2, 0, 1, 5, 3, 4))
    raise ValueError(f"Unsupported videos layout: {layout}")


def _to_float_chw(array: np.ndarray) -> torch.Tensor:
    if array.ndim != 5:
        raise ValueError(f"Expected a 5D per-view array, got shape {array.shape}")
    if array.shape[2] in (1, 3):
        video = torch.from_numpy(array)
    elif array.shape[-1] in (1, 3):
        video = torch.from_numpy(array).permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(
            "Per-view video must be (B,T,C,H,W) or (B,T,H,W,C); "
            f"got shape {array.shape}"
        )
    video = video.float()
    if video.numel() and video.max() > 1.5:
        video = video / 255.0
    return video.contiguous()


def _select_transition(video: torch.Tensor, frame_pair: str, start_index: int) -> torch.Tensor:
    if video.shape[1] < 2:
        raise ValueError(f"FACT tokenizer needs at least 2 frames, got {video.shape[1]}")
    start_index = min(max(start_index, 0), video.shape[1] - 2)
    if frame_pair == "first-next":
        return video[:, start_index : start_index + 2]
    if frame_pair == "first-last":
        return torch.stack([video[:, start_index], video[:, -1]], dim=1)
    raise ValueError(f"Unsupported frame_pair: {frame_pair}")


def _resize_video(video: torch.Tensor, resize: Optional[int]) -> torch.Tensor:
    if not resize or tuple(video.shape[-2:]) == (resize, resize):
        return video
    batch, frames, channels, height, width = video.shape
    resized = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        size=(resize, resize),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, channels, resize, resize).contiguous()


class FACTPairedNPZDataset(Dataset):
    """In-memory paired transition dataset for FACT tokenizer smoke training.

    The dataset always exposes canonical output view names, by default
    ``ego`` and ``exo``. Source NPZ keys can be arbitrary and are mapped by
    order through ``source_view_keys``.
    """

    def __init__(
        self,
        input_npz: Path | str,
        source_view_keys: Optional[Sequence[str]] = None,
        output_view_names: Sequence[str] = ("ego", "exo"),
        videos_layout: str = "VBTCHW",
        frame_pair: str = "first-last",
        start_index: int = 0,
        resize: Optional[int] = 224,
    ) -> None:
        self.input_npz = Path(input_npz)
        self.output_view_names = list(output_view_names)
        if len(self.output_view_names) != 2:
            raise ValueError("FACT v0.1 expects exactly two output views: ego and exo")
        if videos_layout not in VIDEO_LAYOUTS:
            raise ValueError(f"videos_layout must be one of {sorted(VIDEO_LAYOUTS)}")

        with np.load(self.input_npz, allow_pickle=False) as data:
            arrays = self._load_arrays(data, source_view_keys, videos_layout)

        if len(arrays) < 2:
            raise ValueError("FACTPairedNPZDataset needs two synchronized views")

        videos: Dict[str, torch.Tensor] = {}
        for output_name, source_name in zip(self.output_view_names, list(arrays.keys())[:2]):
            video = _to_float_chw(arrays[source_name])
            video = _select_transition(video, frame_pair, start_index)
            video = _resize_video(video, resize)
            videos[output_name] = video

        batch_sizes = {value.shape[0] for value in videos.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"All views must share batch size, got {sorted(batch_sizes)}")

        self.videos = videos
        self.source_view_keys = list(arrays.keys())[:2]
        self.frame_pair = frame_pair
        self.resize = resize

    def _load_arrays(
        self,
        data: Mapping[str, np.ndarray],
        source_view_keys: Optional[Sequence[str]],
        videos_layout: str,
    ) -> Dict[str, np.ndarray]:
        if "videos" in data:
            videos = _normalize_videos_array(data["videos"], videos_layout)
            keys = list(source_view_keys) if source_view_keys else [f"view_{idx}" for idx in range(videos.shape[0])]
            if len(keys) > videos.shape[0]:
                raise ValueError(f"Requested {len(keys)} views but videos only has {videos.shape[0]}")
            return {key: videos[index] for index, key in enumerate(keys)}

        if source_view_keys:
            missing = [key for key in source_view_keys if key not in data]
            if missing:
                raise KeyError(f"Missing NPZ view keys: {missing}")
            return {key: data[key] for key in source_view_keys}

        preferred = [key for key in ("ego", "exo", "primary", "wrist") if key in data and data[key].ndim == 5]
        remaining = [key for key in data.keys() if key not in preferred and data[key].ndim == 5]
        keys = (preferred + remaining)[:2]
        return {key: data[key] for key in keys}

    def __len__(self) -> int:
        first_view = self.output_view_names[0]
        return int(self.videos[first_view].shape[0])

    def __getitem__(self, index: int) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            view_name: {
                "videos": self.videos[view_name][index],
                "sample_id": torch.tensor(index, dtype=torch.long),
            }
            for view_name in self.output_view_names
        }
