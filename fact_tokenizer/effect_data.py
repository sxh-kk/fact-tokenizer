"""Memory-mapped clip loading for FACT 5.0 effect learning.

The loader intentionally keeps the storage contract small: every dense array is
an individual ``.npy`` file with the sample dimension first.  Arrays stay
memory-mapped for the lifetime of the dataset; only the sample requested by
``__getitem__`` is copied before it is converted to a tensor.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_PREFERRED_VIEW_KEYS = ("ego", "exo", "primary", "wrist")
_METADATA_KEYS = ("sample_id", "take_uid", "take_index", "timestamp", "current_index", "role")
_ROLE_COLUMNS = (
    "role",
    "data_role",
    "dataset_role",
    "sample_role",
    "split_role",
    "split",
    "usage_role",
    "usable_for",
    "auto_usable_for",
    "bucket",
)
_DEFAULT_CAPABILITY_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "world": ("world_mask", "world", "can_world", "has_world", "c_world"),
    "token": ("token_mask", "token", "can_token", "has_token", "c_token"),
    "geometry": (
        "geometry_mask",
        "geo_mask",
        "geometry",
        "can_geometry",
        "has_geometry",
    ),
    "contact": ("contact_mask", "contact", "can_contact", "has_contact"),
}


@dataclass(frozen=True)
class EffectClipSpec:
    """Describe a causal-history/future clip around one current frame.

    ``history_frames`` includes the current frame.  Consequently, the current
    frame in the returned clip is always at ``history_frames - 1``.  A dataset
    may provide ``current_index.npy`` to choose a different source anchor for
    every sample; otherwise ``source_current_index`` is used, falling back to
    ``history_frames - 1``.
    """

    history_frames: int = 1
    future_frames: int = 1
    stride: int = 1
    source_current_index: Optional[int] = None
    resize: Optional[int | tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.history_frames < 1:
            raise ValueError("history_frames must be at least 1")
        if self.future_frames < 0:
            raise ValueError("future_frames cannot be negative")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.source_current_index is not None and self.source_current_index < 0:
            raise ValueError("source_current_index cannot be negative")
        if isinstance(self.resize, int) and self.resize < 1:
            raise ValueError("resize must be positive")
        if isinstance(self.resize, tuple):
            if len(self.resize) != 2 or any(size < 1 for size in self.resize):
                raise ValueError("resize tuple must contain two positive integers")

    @property
    def current_index(self) -> int:
        """Index of the current frame in a returned clip."""

        return self.history_frames - 1

    @property
    def frame_count(self) -> int:
        return self.history_frames + self.future_frames

    def source_indices(self, available_frames: int, current_index: Optional[int] = None) -> tuple[int, ...]:
        """Return validated source-frame indices for one sample."""

        anchor = current_index
        if anchor is None:
            anchor = self.source_current_index
        if anchor is None:
            anchor = self.current_index
        anchor = int(anchor)
        first = anchor - (self.history_frames - 1) * self.stride
        last = anchor + self.future_frames * self.stride
        if first < 0 or last >= available_frames:
            raise IndexError(
                "Effect clip is outside the source sample: "
                f"history={self.history_frames}, future={self.future_frames}, "
                f"stride={self.stride}, current={anchor}, frames={available_frames}"
            )
        return tuple(range(first, last + 1, self.stride))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Manifest line {line_number} is not an object")
                rows.append(dict(value))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("samples", "records", "rows"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("JSON manifest must be a list of objects or contain samples/records/rows")
        return [dict(row) for row in payload]
    raise ValueError(f"Unsupported manifest format: {path.suffix or '<none>'}")


def _text_scalar(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _bool_scalar(value: Any, *, field: str) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value == "":
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    try:
        return bool(float(normalized))
    except ValueError as exc:
        raise ValueError(f"Capability field {field!r} is not boolean-like: {value!r}") from exc


def _is_forbidden_train_role(value: Any) -> bool:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return (
        "probe" in normalized
        or "diagnostic" in normalized
        or "locked" in normalized
        or normalized
        in {
            "test",
            "heldout",
            "holdout",
            "val",
            "valid",
            "validation",
            "dev",
            "calibration_dev",
            "locked_test",
            "assembly101",
        }
    )


def _to_video_tensor(sample: np.ndarray, resize: Optional[int | tuple[int, int]]) -> torch.Tensor:
    if sample.ndim != 4:
        raise ValueError(f"Expected a per-sample TCHW or THWC array, got {sample.shape}")
    if sample.shape[-1] in (1, 3):
        tensor = torch.from_numpy(sample).permute(0, 3, 1, 2)
    elif sample.shape[1] in (1, 3):
        tensor = torch.from_numpy(sample)
    else:
        raise ValueError(f"Cannot infer channel axis for per-sample array {sample.shape}")
    tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    if resize is not None:
        size = (resize, resize) if isinstance(resize, int) else resize
        if tuple(tensor.shape[-2:]) != tuple(size):
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return tensor.contiguous()


class FACTEffectNPYDataset(Dataset):
    """Lazy one- or two-view effect clips backed exclusively by mmap NPY arrays.

    Parameters
    ----------
    input_dir:
        Directory containing one NPY file per view and optional metadata NPYs.
    view_keys:
        A sequence uses source names as output names.  A mapping is interpreted
        as ``{output_view_name: source_npy_stem}``.  If omitted, conventional
        ego/exo/primary/wrist names are inferred.
    manifest:
        Optional CSV, JSONL, or JSON metadata joined by sample_id or take_uid.
    role:
        Intended consumer role.  ``train`` refuses any joined row marked as a
        probe or diagnostic sample instead of silently training on it.
    capability_columns:
        Optional ``{output_mask_name: manifest_or_npy_column}`` mapping.  The
        default masks are world, token, geometry, and contact.
    """

    def __init__(
        self,
        input_dir: Path | str,
        clip_spec: Optional[EffectClipSpec] = None,
        view_keys: Optional[Sequence[str] | Mapping[str, str]] = None,
        *,
        manifest: Optional[Path | str] = None,
        manifest_join_key: Optional[str] = None,
        role: str = "train",
        capability_columns: Optional[Mapping[str, str] | Sequence[str]] = None,
        camera_context_keys: Optional[Mapping[str, str]] = None,
        require_manifest_match: bool = True,
        drop_nontraining: bool = False,
        excluded_take_uids: Optional[Iterable[str]] = None,
        included_sample_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Effect NPY directory does not exist: {self.input_dir}")
        self.clip_spec = clip_spec or EffectClipSpec()
        self.role = str(role).strip().lower()
        self.require_manifest_match = bool(require_manifest_match)
        self.excluded_take_uids = frozenset(
            str(take_uid).strip() for take_uid in (excluded_take_uids or ()) if str(take_uid).strip()
        )
        self.included_sample_ids = (
            None
            if included_sample_ids is None
            else tuple(str(sample_id).strip() for sample_id in included_sample_ids)
        )
        if self.included_sample_ids is not None:
            if any(not value for value in self.included_sample_ids):
                raise ValueError("included_sample_ids contains an empty sample ID")
            if len(set(self.included_sample_ids)) != len(self.included_sample_ids):
                raise ValueError("included_sample_ids contains duplicates")
        self._array_cache: dict[Path, np.ndarray] = {}

        view_mapping = self._resolve_view_mapping(view_keys)
        if not 1 <= len(view_mapping) <= 2:
            raise ValueError(f"FACTEffectNPYDataset supports one or two views, got {len(view_mapping)}")
        if len(set(view_mapping)) != len(view_mapping):
            raise ValueError("Output view names must be unique")
        if len(set(view_mapping.values())) != len(view_mapping):
            raise ValueError("Source view keys must be unique")
        self.view_keys = tuple(view_mapping)
        self.source_view_keys = tuple(view_mapping.values())
        self._view_arrays = {
            output_name: self._load_npy(self.input_dir / f"{source_name}.npy")
            for output_name, source_name in view_mapping.items()
        }
        for name, array in self._view_arrays.items():
            if array.ndim != 5:
                raise ValueError(f"View {name!r} must have shape (N,T,C,H,W) or (N,T,H,W,C), got {array.shape}")
        sample_counts = {int(array.shape[0]) for array in self._view_arrays.values()}
        frame_counts = {int(array.shape[1]) for array in self._view_arrays.values()}
        if len(sample_counts) != 1:
            raise ValueError(f"All views must share sample count, got {sorted(sample_counts)}")
        if len(frame_counts) != 1:
            raise ValueError(f"All views must share frame count, got {sorted(frame_counts)}")
        self._num_samples = next(iter(sample_counts))
        self._source_frames = next(iter(frame_counts))
        if camera_context_keys is None:
            camera_context_keys = {
                view: f"{view}_camera_context"
                for view in self.view_keys
                if (self.input_dir / f"{view}_camera_context.npy").is_file()
            }
        unknown_context_views = set(camera_context_keys) - set(self.view_keys)
        if unknown_context_views:
            raise ValueError(f"Camera context supplied for unknown views: {sorted(unknown_context_views)}")
        self._camera_context_arrays = {
            str(view): self._load_npy(
                Path(stem)
                if Path(stem).is_absolute()
                else (
                    self.input_dir / Path(stem)
                    if Path(stem).suffix == ".npy"
                    else self.input_dir / f"{stem}.npy"
                )
            )
            for view, stem in camera_context_keys.items()
        }
        for view, array in self._camera_context_arrays.items():
            if array.ndim < 2 or len(array) != self._num_samples:
                raise ValueError(
                    f"Camera context {view!r} must start with sample dimension {self._num_samples}, got {array.shape}"
                )

        self._metadata_arrays = {
            key: self._load_optional_vector(key)
            for key in _METADATA_KEYS
        }

        self._manifest_rows: dict[str, dict[str, Any]] = {}
        self._manifest_join_key: Optional[str] = None
        self._nested_capability_names: tuple[str, ...] = ()
        if manifest is not None:
            manifest_path = Path(manifest)
            rows = _read_manifest(manifest_path)
            self._manifest_join_key = self._choose_join_key(rows, manifest_join_key)
            for row_number, row in enumerate(rows, start=2 if manifest_path.suffix.lower() == ".csv" else 1):
                if self._manifest_join_key not in row or str(row[self._manifest_join_key]).strip() == "":
                    raise ValueError(
                        f"Manifest row {row_number} has no {self._manifest_join_key!r} join value"
                    )
                key = _text_scalar(row[self._manifest_join_key])
                if key in self._manifest_rows:
                    raise ValueError(f"Duplicate manifest join key {key!r}")
                self._manifest_rows[key] = row
            nested_names = {
                str(name)
                for row in self._manifest_rows.values()
                if isinstance(row.get("capability_validity"), Mapping)
                for name in row["capability_validity"]
            }
            self._nested_capability_names = tuple(sorted(nested_names))
            if self.require_manifest_match:
                missing = [
                    self._join_value(index)
                    for index in range(self._num_samples)
                    if self._join_value(index) not in self._manifest_rows
                ]
                if missing:
                    preview = ", ".join(repr(value) for value in missing[:3])
                    raise ValueError(f"Manifest is missing {len(missing)} dataset rows, including {preview}")

        self._capability_sources = self._resolve_capability_sources(capability_columns)
        self._capability_arrays: dict[str, Optional[np.ndarray]] = {}
        for name, columns in self._capability_sources.items():
            array = None
            for column in columns:
                path = self.input_dir / f"{column}.npy"
                if path.exists():
                    array = self._load_vector_path(path, column)
                    break
            self._capability_arrays[name] = array

        rejected_indices = self._validate_training_roles(raise_on_rejected=not drop_nontraining)
        rejected_set = set(rejected_indices)
        self.excluded_sample_count = sum(
            self._take_uid(index) in self.excluded_take_uids
            for index in range(self._num_samples)
        )
        included_set = None if self.included_sample_ids is None else set(self.included_sample_ids)
        if included_set is not None:
            available = {self._sample_id(index) for index in range(self._num_samples)}
            missing_included = sorted(included_set - available)
            if missing_included:
                raise ValueError(
                    f"dataset is missing {len(missing_included)} included sample IDs, including {missing_included[:3]}"
                )
        self._active_indices = np.asarray(
            [
                index
                for index in range(self._num_samples)
                if index not in rejected_set
                and self._take_uid(index) not in self.excluded_take_uids
                and (included_set is None or self._sample_id(index) in included_set)
            ],
            dtype=np.int64,
        )
        self._take_to_index: dict[str, int] = {}
        if self._metadata_arrays["take_index"] is None:
            take_uids = {self._take_uid(index) for index in range(self._num_samples)}
            self._take_to_index = {uid: index for index, uid in enumerate(sorted(take_uids))}

    def _load_npy(self, path: Path) -> np.ndarray:
        if path not in self._array_cache:
            if not path.is_file():
                raise FileNotFoundError(f"Missing NPY array: {path}")
            self._array_cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._array_cache[path]

    def _load_vector_path(self, path: Path, name: str) -> np.ndarray:
        array = self._load_npy(path)
        if array.ndim != 1 or len(array) != self._num_samples:
            raise ValueError(
                f"{name} must be a vector of length {self._num_samples}, got shape {array.shape}"
            )
        return array

    def _load_optional_vector(self, name: str) -> Optional[np.ndarray]:
        path = self.input_dir / f"{name}.npy"
        return self._load_vector_path(path, name) if path.exists() else None

    def _resolve_view_mapping(
        self, view_keys: Optional[Sequence[str] | Mapping[str, str]]
    ) -> dict[str, str]:
        if isinstance(view_keys, Mapping):
            return {str(output): str(source) for output, source in view_keys.items()}
        if view_keys is not None:
            return {str(key): str(key) for key in view_keys}
        preferred = [key for key in _PREFERRED_VIEW_KEYS if (self.input_dir / f"{key}.npy").is_file()]
        if preferred:
            return {key: key for key in preferred[:2]}
        candidates = []
        for path in sorted(self.input_dir.glob("*.npy")):
            if path.stem in _METADATA_KEYS:
                continue
            array = self._load_npy(path)
            if array.ndim == 5:
                candidates.append(path.stem)
        if not candidates:
            raise ValueError(f"Could not infer a view NPY in {self.input_dir}")
        if len(candidates) > 2:
            raise ValueError(f"Ambiguous view inference; pass view_keys explicitly: {candidates}")
        return {key: key for key in candidates}

    def _choose_join_key(self, rows: list[dict[str, Any]], requested: Optional[str]) -> str:
        if requested is not None:
            if requested not in {"sample_id", "take_uid"}:
                raise ValueError("manifest_join_key must be sample_id or take_uid")
            return requested
        if rows and all("sample_id" in row for row in rows) and self._metadata_arrays["sample_id"] is not None:
            return "sample_id"
        if rows and all("take_uid" in row for row in rows) and self._metadata_arrays["take_uid"] is not None:
            return "take_uid"
        if not rows:
            raise ValueError("Manifest is empty")
        raise ValueError("Cannot infer manifest join key; provide matching sample_id.npy or take_uid.npy")

    def _resolve_capability_sources(
        self, columns: Optional[Mapping[str, str] | Sequence[str]]
    ) -> dict[str, tuple[str, ...]]:
        if isinstance(columns, Mapping):
            return {str(name): (str(column),) for name, column in columns.items()}
        if columns is not None:
            return {str(name): (str(name),) for name in columns}
        if self._nested_capability_names:
            return {name: (name,) for name in self._nested_capability_names}
        return {name: tuple(aliases) for name, aliases in _DEFAULT_CAPABILITY_COLUMNS.items()}

    def _array_text(self, name: str, index: int) -> Optional[str]:
        array = self._metadata_arrays[name]
        return _text_scalar(array[index]) if array is not None else None

    def _sample_id(self, index: int) -> str:
        value = self._array_text("sample_id", index)
        if value is not None:
            return value
        row = self._manifest_row(index)
        if row is not None and str(row.get("sample_id", "")).strip():
            return _text_scalar(row["sample_id"])
        return str(index)

    def _take_uid(self, index: int) -> str:
        value = self._array_text("take_uid", index)
        if value is not None:
            return value
        row = self._manifest_row(index)
        if row is not None and str(row.get("take_uid", "")).strip():
            return _text_scalar(row["take_uid"])
        return self._sample_id(index)

    def _join_value(self, index: int) -> str:
        if self._manifest_join_key == "sample_id":
            return self._sample_id(index)
        if self._manifest_join_key == "take_uid":
            value = self._array_text("take_uid", index)
            if value is None:
                raise ValueError("take_uid manifest join requires take_uid.npy")
            return value
        raise RuntimeError("Manifest join key has not been configured")

    def _manifest_row(self, index: int) -> Optional[dict[str, Any]]:
        if self._manifest_join_key is None:
            return None
        return self._manifest_rows.get(self._join_value(index))

    def _validate_training_roles(self, *, raise_on_rejected: bool = True) -> list[int]:
        if self.role != "train":
            return []
        rejected = []
        role_array = self._metadata_arrays["role"]
        for index in range(self._num_samples):
            values: list[tuple[str, Any]] = []
            if role_array is not None:
                values.append(("role.npy", role_array[index]))
            row = self._manifest_row(index)
            if row is not None:
                values.extend((column, row[column]) for column in _ROLE_COLUMNS if column in row)
                if "source_dataset" in row:
                    values.append(("source_dataset", row["source_dataset"]))
                if "training_valid" in row and not _bool_scalar(
                    row["training_valid"], field="training_valid"
                ):
                    values.append(("training_valid", "diagnostic/non_training"))
                diagnostic = row.get("diagnostic_codelabel")
                if diagnostic not in (None, "", {}):
                    values.append(("diagnostic_codelabel", "diagnostic"))
                for column in ("is_probe", "is_diagnostic"):
                    if column in row and _bool_scalar(row[column], field=column):
                        values.append((column, column.removeprefix("is_")))
            for column, value in values:
                if _is_forbidden_train_role(value):
                    rejected.append((index, self._sample_id(index), column, _text_scalar(value)))
                    break
        if rejected and raise_on_rejected:
            _, sample_id, column, value = rejected[0]
            raise ValueError(
                f"train role refuses {len(rejected)} probe/diagnostic rows; "
                f"first sample {sample_id!r} has {column}={value!r}"
            )
        return [index for index, _, _, _ in rejected]

    def _capability_masks(self, index: int) -> Dict[str, torch.Tensor]:
        row = self._manifest_row(index)
        masks: Dict[str, torch.Tensor] = {}
        for name, columns in self._capability_sources.items():
            value: Any = None
            source_name = columns[0]
            if row is not None:
                nested = row.get("capability_validity")
                if isinstance(nested, Mapping) and name in nested:
                    value = nested[name]
                    source_name = f"capability_validity.{name}"
                for column in columns:
                    if value is None and column in row and row[column] not in (None, ""):
                        value = row[column]
                        source_name = column
                        break
            if value is None and self._capability_arrays[name] is not None:
                value = self._capability_arrays[name][index]
            masks[name] = torch.tensor(_bool_scalar(value, field=source_name), dtype=torch.bool)
        return masks

    def _take_index(self, index: int) -> int:
        array = self._metadata_arrays["take_index"]
        if array is not None:
            return int(array[index])
        row = self._manifest_row(index)
        if row is not None and str(row.get("take_index", "")).strip():
            return int(row["take_index"])
        return self._take_to_index[self._take_uid(index)]

    def _timestamp(self, index: int) -> float:
        array = self._metadata_arrays["timestamp"]
        if array is not None:
            return float(array[index])
        row = self._manifest_row(index)
        if row is not None and str(row.get("timestamp", "")).strip():
            return float(row["timestamp"])
        return float(index)

    def _source_current_index(self, index: int) -> Optional[int]:
        array = self._metadata_arrays["current_index"]
        return int(array[index]) if array is not None else None

    def _quality_bucket(self, index: int) -> str:
        row = self._manifest_row(index)
        value = row.get("quality_bucket") if row is not None else None
        return str(value).strip() if value not in (None, "") else "unlabeled"

    def _quality_weight(self, index: int) -> float:
        row = self._manifest_row(index)
        value = row.get("quality_weight") if row is not None else None
        weight = 1.0 if value in (None, "") else float(value)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid quality_weight for sample {self._sample_id(index)}: {weight}")
        return weight

    def __len__(self) -> int:
        return len(self._active_indices)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Stable active sample order without materializing any RGB frames."""

        return tuple(self._sample_id(int(index)) for index in self._active_indices)

    @property
    def take_uids(self) -> tuple[str, ...]:
        """Stable active take order without materializing any RGB frames."""

        return tuple(self._take_uid(int(index)) for index in self._active_indices)

    @property
    def quality_weights(self) -> np.ndarray:
        return np.asarray(
            [self._quality_weight(int(index)) for index in self._active_indices],
            dtype=np.float64,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self._active_indices)
        if not 0 <= index < len(self._active_indices):
            raise IndexError(index)
        index = int(self._active_indices[index])
        source_indices = self.clip_spec.source_indices(
            self._source_frames,
            self._source_current_index(index),
        )
        views = {}
        for name, array in self._view_arrays.items():
            # Index the sample first, then copy only its selected frames.  This
            # is the sole dense-array copy in the data path.
            sample = np.asarray(array[index][list(source_indices)]).copy()
            views[name] = _to_video_tensor(sample, self.clip_spec.resize)
        camera_contexts = {
            name: torch.from_numpy(np.asarray(array[index]).copy()).float().flatten()
            for name, array in self._camera_context_arrays.items()
        }
        return {
            "views": views,
            "camera_contexts": camera_contexts,
            "current_index": torch.tensor(self.clip_spec.current_index, dtype=torch.long),
            "sample_id": self._sample_id(index),
            "take_uid": self._take_uid(index),
            "take_index": torch.tensor(self._take_index(index), dtype=torch.long),
            "timestamp": torch.tensor(self._timestamp(index), dtype=torch.float32),
            "quality_bucket": self._quality_bucket(index),
            "quality_weight": torch.tensor(self._quality_weight(index), dtype=torch.float32),
            "capability_masks": self._capability_masks(index),
        }
