"""Data loading utilities for paired ego/exo FACT tokenizer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

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


def _prepare_video_sample(array: np.ndarray, frame_pair: str, start_index: int, resize: Optional[int]) -> torch.Tensor:
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D per-sample array, got shape {array.shape}")
    video = _to_float_chw(np.asarray(array).copy()[None])
    video = _select_transition(video, frame_pair, start_index)
    video = _resize_video(video, resize)
    return video[0].contiguous()


class FACTPairedNPYDataset(Dataset):
    """Lazy paired transition dataset backed by a directory of ``.npy`` arrays."""

    def __init__(
        self,
        input_dir: Path | str,
        source_view_keys: Optional[Sequence[str]] = None,
        output_view_names: Sequence[str] = ("ego", "exo"),
        frame_pair: str = "first-last",
        start_index: int = 0,
        resize: Optional[int] = 224,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_view_names = list(output_view_names)
        if len(self.output_view_names) != 2:
            raise ValueError("FACT v0.1 expects exactly two output views: ego and exo")

        keys = list(source_view_keys) if source_view_keys else self._infer_view_keys()
        if len(keys) < 2:
            raise ValueError("FACTPairedNPYDataset needs two synchronized views")
        self.source_view_keys = keys[:2]
        self.frame_pair = frame_pair
        self.start_index = start_index
        self.resize = resize
        self._videos = {
            output_name: np.load(self.input_dir / f"{source_name}.npy", mmap_mode="r")
            for output_name, source_name in zip(self.output_view_names, self.source_view_keys)
        }
        batch_sizes = {int(value.shape[0]) for value in self._videos.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"All views must share batch size, got {sorted(batch_sizes)}")
        self._num_samples = next(iter(batch_sizes))

        take_uid_path = self.input_dir / "take_uid.npy"
        timestamp_path = self.input_dir / "timestamp.npy"
        if take_uid_path.exists():
            take_uid = np.load(take_uid_path, mmap_mode="r").astype(str)
            if len(take_uid) != self._num_samples:
                raise ValueError(f"take_uid has length {len(take_uid)}, expected {self._num_samples}")
            unique_takes = {uid: idx for idx, uid in enumerate(sorted(set(take_uid.tolist())))}
            self.take_uids = take_uid.tolist()
            self.take_indices = torch.tensor([unique_takes[uid] for uid in self.take_uids], dtype=torch.long)
        else:
            self.take_uids = [str(index) for index in range(self._num_samples)]
            self.take_indices = torch.arange(self._num_samples, dtype=torch.long)

        if timestamp_path.exists():
            timestamp = np.load(timestamp_path, mmap_mode="r").astype(np.float32)
            if len(timestamp) != self._num_samples:
                raise ValueError(f"timestamp has length {len(timestamp)}, expected {self._num_samples}")
            self.timestamps = torch.from_numpy(np.asarray(timestamp, dtype=np.float32))
        else:
            self.timestamps = torch.arange(self._num_samples, dtype=torch.float32)

    def _infer_view_keys(self) -> list[str]:
        preferred = [key for key in ("ego", "exo", "primary", "wrist") if (self.input_dir / f"{key}.npy").exists()]
        remaining = [
            path.stem
            for path in sorted(self.input_dir.glob("*.npy"))
            if path.stem not in preferred and np.load(path, mmap_mode="r").ndim == 5
        ]
        return (preferred + remaining)[:2]

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            view_name: {
                "videos": _prepare_video_sample(
                    self._videos[view_name][index],
                    self.frame_pair,
                    self.start_index,
                    self.resize,
                ),
                "sample_id": torch.tensor(index, dtype=torch.long),
                "take_index": self.take_indices[index],
                "timestamp": self.timestamps[index],
            }
            for view_name in self.output_view_names
        }


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
        if self.input_npz.is_dir():
            self._delegate = FACTPairedNPYDataset(
                input_dir=self.input_npz,
                source_view_keys=source_view_keys,
                output_view_names=output_view_names,
                frame_pair=frame_pair,
                start_index=start_index,
                resize=resize,
            )
            self.output_view_names = self._delegate.output_view_names
            self.source_view_keys = self._delegate.source_view_keys
            self.frame_pair = self._delegate.frame_pair
            self.resize = self._delegate.resize
            self.take_uids = self._delegate.take_uids
            self.take_indices = self._delegate.take_indices
            self.timestamps = self._delegate.timestamps
            return
        self._delegate = None
        self.output_view_names = list(output_view_names)
        if len(self.output_view_names) != 2:
            raise ValueError("FACT v0.1 expects exactly two output views: ego and exo")
        if videos_layout not in VIDEO_LAYOUTS:
            raise ValueError(f"videos_layout must be one of {sorted(VIDEO_LAYOUTS)}")

        with np.load(self.input_npz, allow_pickle=False) as data:
            arrays = self._load_arrays(data, source_view_keys, videos_layout)
            take_uid = np.asarray(data["take_uid"]).astype(str) if "take_uid" in data else None
            timestamp = np.asarray(data["timestamp"], dtype=np.float32) if "timestamp" in data else None

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
        num_samples = len(self)
        if take_uid is not None:
            if len(take_uid) != num_samples:
                raise ValueError(f"take_uid has length {len(take_uid)}, expected {num_samples}")
            unique_takes = {uid: idx for idx, uid in enumerate(sorted(set(take_uid.tolist())))}
            self.take_uids = take_uid.tolist()
            self.take_indices = torch.tensor([unique_takes[uid] for uid in self.take_uids], dtype=torch.long)
        else:
            self.take_uids = [str(index) for index in range(num_samples)]
            self.take_indices = torch.arange(num_samples, dtype=torch.long)
        if timestamp is not None:
            if len(timestamp) != num_samples:
                raise ValueError(f"timestamp has length {len(timestamp)}, expected {num_samples}")
            self.timestamps = torch.from_numpy(timestamp.astype(np.float32))
        else:
            self.timestamps = torch.arange(num_samples, dtype=torch.float32)

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
        if self._delegate is not None:
            return len(self._delegate)
        first_view = self.output_view_names[0]
        return int(self.videos[first_view].shape[0])

    def __getitem__(self, index: int) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._delegate is not None:
            return self._delegate[index]
        return {
            view_name: {
                "videos": self.videos[view_name][index],
                "sample_id": torch.tensor(index, dtype=torch.long),
                "take_index": self.take_indices[index],
                "timestamp": self.timestamps[index],
            }
            for view_name in self.output_view_names
        }
