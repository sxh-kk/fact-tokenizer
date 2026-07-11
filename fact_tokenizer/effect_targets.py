"""Target construction utilities for FACT v7 effect representation learning.

The geometry produced here is deliberately two dimensional.  Camera rotation
is removed with a calibrated homography and a robust background homography is
then removed from the remaining optical flow.  Camera translation cannot be
metric without depth, so this module never exposes a ``flow_3d`` target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


TARGET_CACHE_VERSION = "fact-effect-target-v1"
WEAK_VERB_MAP_VERSION = "fact-weak-verb-map-v1"

WEAK_EFFECT_LABELS = (
    "no_effect",
    "approach_align",
    "acquire_control",
    "state_change_or_manipulate",
    "transport_reposition",
    "release_complete",
    "recover_abort",
    "ambiguous",
)
WEAK_CONTACT_LABELS = ("none", "onset", "stable", "release", "unknown")


@dataclass(frozen=True)
class EffectTargetConfig:
    """Versioned parameters used to build continuous FACT effect targets."""

    transition_seconds: float = 0.5
    pose_fps: float = 30.0
    max_pose_skew_frames: float = 1.0
    output_height: int = 224
    output_width: int = 224
    flow_fb_threshold_px: float = 1.5
    background_ransac_threshold_px: float = 2.0
    atomic_midpoint_tolerance_seconds: float = 0.75
    target_cache_version: str = TARGET_CACHE_VERSION

    def __post_init__(self) -> None:
        if self.transition_seconds <= 0:
            raise ValueError("transition_seconds must be positive")
        if self.pose_fps <= 0:
            raise ValueError("pose_fps must be positive")
        if self.max_pose_skew_frames < 0:
            raise ValueError("max_pose_skew_frames cannot be negative")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output dimensions must be positive")

    @property
    def max_pose_skew_seconds(self) -> float:
        return self.max_pose_skew_frames / self.pose_fps

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_supported_target(target_name: str) -> None:
    """Fail early for targets that cannot be supported by the current assets."""

    normalized = target_name.strip().lower().replace("-", "_")
    forbidden = any(
        token in normalized
        for token in ("flow_3d", "dense_3d_flow", "scene_flow", "depth_map")
    ) or normalized == "depth" or normalized.endswith("_depth")
    if forbidden and not normalized.endswith("_valid"):
        raise ValueError(
            f"Target {target_name!r} requires depth/metric 3D geometry; "
            "FACT v7 only supports rotation_compensated_flow_2d."
        )


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale a 3x3 camera matrix between ``(height, width)`` image sizes."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    source_h, source_w = source_size
    target_h, target_w = target_size
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("image dimensions must be positive")
    sx = target_w / source_w
    sy = target_h / source_h
    scaled = matrix.copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    scaled[2, :] = matrix[2, :]
    return scaled


def nearest_timestamp_index(
    timestamps: Sequence[float] | np.ndarray,
    query_seconds: float,
    max_skew_seconds: float,
) -> Optional[int]:
    """Return the deterministic nearest timestamp, or ``None`` outside tolerance."""

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        return None
    distances = np.abs(values - float(query_seconds))
    index = int(np.argmin(distances))
    if not np.isfinite(distances[index]) or distances[index] > max_skew_seconds + 1e-9:
        return None
    return index


def align_pose_pair(
    timestamps: Sequence[float] | np.ndarray,
    poses: Sequence[np.ndarray] | np.ndarray,
    start_seconds: float,
    end_seconds: float,
    max_skew_seconds: float,
) -> Optional[tuple[np.ndarray, np.ndarray, float, float]]:
    """Align two requested times to calibrated poses with independent validity."""

    i0 = nearest_timestamp_index(timestamps, start_seconds, max_skew_seconds)
    i1 = nearest_timestamp_index(timestamps, end_seconds, max_skew_seconds)
    if i0 is None or i1 is None:
        return None
    values = np.asarray(timestamps, dtype=np.float64)
    pose_values = np.asarray(poses, dtype=np.float64)
    if pose_values.shape[0] != values.shape[0]:
        raise ValueError("pose and timestamp lengths differ")
    return pose_values[i0], pose_values[i1], float(values[i0]), float(values[i1])


def relative_rotation_homography(
    intrinsics_t0: np.ndarray,
    intrinsics_t1: np.ndarray,
    world_to_camera_t0: np.ndarray,
    world_to_camera_t1: np.ndarray,
) -> np.ndarray:
    """Map t0 pixels to t1 pixels using camera rotation only.

    Extrinsics use the explicit world-to-camera convention.  Both 3x3
    rotations and 4x4 rigid transforms are accepted; translations are ignored.
    """

    k0 = np.asarray(intrinsics_t0, dtype=np.float64)
    k1 = np.asarray(intrinsics_t1, dtype=np.float64)
    e0 = np.asarray(world_to_camera_t0, dtype=np.float64)
    e1 = np.asarray(world_to_camera_t1, dtype=np.float64)
    if k0.shape != (3, 3) or k1.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    if e0.shape not in {(3, 3), (4, 4)} or e1.shape not in {(3, 3), (4, 4)}:
        raise ValueError("extrinsics must be 3x3 rotations or 4x4 transforms")
    r0 = e0[:3, :3]
    r1 = e1[:3, :3]
    relative = r1 @ r0.T
    homography = k1 @ relative @ np.linalg.inv(k0)
    if abs(homography[2, 2]) > 1e-12:
        homography /= homography[2, 2]
    return homography


def invert_extrinsic(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform while validating the input shape."""

    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {value.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return result


def pixel_grid(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([x, y], axis=-1).astype(np.float32)


def transform_points_homography(points_xy: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    if points.shape[-1] != 2 or matrix.shape != (3, 3):
        raise ValueError("points must end in xy and homography must be 3x3")
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    projected = homogeneous @ matrix.T
    denominator = projected[..., 2]
    valid = np.isfinite(projected).all(axis=-1) & (np.abs(denominator) > 1e-8)
    result = np.full(points.shape, np.nan, dtype=np.float64)
    result[valid] = projected[valid, :2] / denominator[valid, None]
    return result.astype(np.float32), valid


def homography_flow(homography: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid = pixel_grid(height, width)
    endpoint, valid = transform_points_homography(grid, homography)
    valid &= (
        (endpoint[..., 0] >= 0)
        & (endpoint[..., 0] <= width - 1)
        & (endpoint[..., 1] >= 0)
        & (endpoint[..., 1] <= height - 1)
    )
    return endpoint - grid, valid


def _bilinear_sample(field: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(field)
    coords = np.asarray(xy, dtype=np.float64)
    height, width = value.shape[:2]
    x = coords[..., 0]
    y = coords[..., 1]
    valid = np.isfinite(coords).all(axis=-1) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x_safe = np.clip(np.nan_to_num(x), 0, width - 1)
    y_safe = np.clip(np.nan_to_num(y), 0, height - 1)
    x0 = np.floor(x_safe).astype(np.int64)
    y0 = np.floor(y_safe).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x_safe - x0)[..., None]
    wy = (y_safe - y0)[..., None]
    if value.ndim == 2:
        value = value[..., None]
    sampled = (
        value[y0, x0] * (1 - wx) * (1 - wy)
        + value[y0, x1] * wx * (1 - wy)
        + value[y1, x0] * (1 - wx) * wy
        + value[y1, x1] * wx * wy
    )
    return sampled, valid


def forward_backward_consistency(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    threshold_px: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RAFT forward/backward consistency error and validity mask."""

    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[-1] != 2:
        raise ValueError("forward and backward flow must have matching HxWx2 shapes")
    endpoints = pixel_grid(*forward.shape[:2]) + forward
    sampled_backward, in_bounds = _bilinear_sample(backward, endpoints)
    error = np.linalg.norm(forward + sampled_backward, axis=-1)
    valid = in_bounds & np.isfinite(error) & (error <= threshold_px)
    return error.astype(np.float32), valid


def fit_background_homography(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    valid_mask: np.ndarray,
    ransac_threshold_px: float = 2.0,
    prefer_opencv: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a planar background mapping with RANSAC.

    Returns identity and an all-false inlier mask if fewer than four points are
    usable.  OpenCV is imported lazily so manifest-only environments stay light.
    """

    source = np.asarray(source_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if source.shape != target.shape or source.shape[-1] != 2 or source.shape[:-1] != valid.shape:
        raise ValueError("source, target and valid_mask shapes are inconsistent")
    finite = valid & np.isfinite(source).all(axis=-1) & np.isfinite(target).all(axis=-1)
    points0 = source[finite]
    points1 = target[finite]
    if len(points0) < 4:
        return np.eye(3, dtype=np.float64), np.zeros(valid.shape, dtype=bool)
    try:
        if not prefer_opencv:
            raise ImportError
        import cv2
    except ImportError:  # Lightweight manifest/test environments use a translation-only fallback.
        displacement = points1 - points0
        median = np.median(displacement, axis=0)
        matrix = np.asarray([[1.0, 0.0, median[0]], [0.0, 1.0, median[1]], [0.0, 0.0, 1.0]])
        errors = np.linalg.norm(displacement - median[None], axis=1)
        inliers = (errors <= ransac_threshold_px)[:, None].astype(np.uint8)
    else:
        matrix, inliers = cv2.findHomography(points0, points1, cv2.RANSAC, ransac_threshold_px)
    if matrix is None or inliers is None:
        return np.eye(3, dtype=np.float64), np.zeros(valid.shape, dtype=bool)
    inlier_mask = np.zeros(valid.shape, dtype=bool)
    inlier_mask[finite] = inliers.reshape(-1).astype(bool)
    matrix = matrix.astype(np.float64)
    if abs(matrix[2, 2]) > 1e-12:
        matrix /= matrix[2, 2]
    return matrix, inlier_mask


def _fit_homography_lstsq(source: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """Fit a homography with h22 fixed to one (small deterministic fallback)."""

    if len(source) < 4:
        return None
    x, y = source[:, 0], source[:, 1]
    u, v = target[:, 0], target[:, 1]
    rows = np.zeros((2 * len(source), 8), dtype=np.float64)
    rhs = np.zeros(2 * len(source), dtype=np.float64)
    rows[0::2, 0:3] = np.stack([x, y, np.ones_like(x)], axis=1)
    rows[0::2, 6:8] = -np.stack([u * x, u * y], axis=1)
    rhs[0::2] = u
    rows[1::2, 3:6] = np.stack([x, y, np.ones_like(x)], axis=1)
    rows[1::2, 6:8] = -np.stack([v * x, v * y], axis=1)
    rhs[1::2] = v
    try:
        solution, _, rank, _ = np.linalg.lstsq(rows, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 8 or not np.isfinite(solution).all():
        return None
    return np.asarray(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
            [solution[6], solution[7], 1.0],
        ],
        dtype=np.float64,
    )


def _numpy_find_homography_ransac(
    source: np.ndarray,
    target: np.ndarray,
    threshold_px: float,
    trials: int = 256,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Minimal seeded RANSAC used only when OpenCV is unavailable."""

    rng = np.random.default_rng(0)
    best_matrix: Optional[np.ndarray] = None
    best_inliers: Optional[np.ndarray] = None
    best_score = (-1, -np.inf)
    for _ in range(min(trials, max(32, len(source) * 2))):
        sample = rng.choice(len(source), size=4, replace=False)
        matrix = _fit_homography_lstsq(source[sample], target[sample])
        if matrix is None:
            continue
        projected, valid = transform_points_homography(source, matrix)
        error = np.linalg.norm(projected - target, axis=-1)
        inliers = valid & np.isfinite(error) & (error <= threshold_px)
        count = int(inliers.sum())
        median_score = -float(np.median(error[inliers])) if count else -np.inf
        if (count, median_score) > best_score:
            best_score = (count, median_score)
            best_matrix = matrix
            best_inliers = inliers
    if best_inliers is None or int(best_inliers.sum()) < 4:
        return None, None
    refined = _fit_homography_lstsq(source[best_inliers], target[best_inliers])
    return (refined if refined is not None else best_matrix), best_inliers[:, None].astype(np.uint8)


def rotation_compensated_flow_2d(
    forward_flow: np.ndarray,
    rotational_homography: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    background_mask: Optional[np.ndarray] = None,
    ransac_threshold_px: float = 2.0,
    prefer_opencv: bool = True,
) -> dict[str, np.ndarray]:
    """Remove camera rotation and robust planar background motion from 2D flow.

    The returned residual is an observed image-plane transition target.  It is
    not dense 3D flow and should not be interpreted as metric object motion.
    """

    flow = np.asarray(forward_flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("forward_flow must be HxWx2")
    height, width = flow.shape[:2]
    grid = pixel_grid(height, width)
    endpoints = grid + flow
    inverse_rotation = np.linalg.inv(np.asarray(rotational_homography, dtype=np.float64))
    derotated_endpoints, rotation_valid = transform_points_homography(endpoints, inverse_rotation)
    rotation_valid &= (
        (derotated_endpoints[..., 0] >= 0)
        & (derotated_endpoints[..., 0] <= width - 1)
        & (derotated_endpoints[..., 1] >= 0)
        & (derotated_endpoints[..., 1] <= height - 1)
    )
    valid = rotation_valid & np.isfinite(flow).all(axis=-1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    fit_valid = valid.copy()
    if background_mask is not None:
        fit_valid &= np.asarray(background_mask, dtype=bool)
    background_h, inliers = fit_background_homography(
        grid,
        derotated_endpoints,
        fit_valid,
        ransac_threshold_px=ransac_threshold_px,
        prefer_opencv=prefer_opencv,
    )
    background_endpoint, background_valid = transform_points_homography(grid, background_h)
    background_fit_valid = int(np.asarray(inliers, dtype=bool).sum()) >= 4
    final_valid = valid & background_valid & background_fit_valid
    residual = derotated_endpoints - background_endpoint
    residual[~final_valid] = 0.0
    return {
        "rotation_compensated_flow_2d": residual.astype(np.float32),
        "valid": final_valid,
        "background_homography": background_h,
        "background_inliers": inliers,
        "background_fit_valid": np.asarray(background_fit_valid),
        "derotated_flow_2d": (derotated_endpoints - grid).astype(np.float32),
    }


def propagate_mask_forward(
    mask: np.ndarray,
    forward_flow: np.ndarray,
    consistency_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat a boolean target mask using optical flow."""

    source_mask = np.asarray(mask, dtype=bool)
    flow = np.asarray(forward_flow, dtype=np.float32)
    if flow.shape != (*source_mask.shape, 2):
        raise ValueError("flow shape must be mask.shape + (2,)")
    valid = source_mask & np.isfinite(flow).all(axis=-1)
    if consistency_mask is not None:
        valid &= np.asarray(consistency_mask, dtype=bool)
    endpoints = np.rint(pixel_grid(*source_mask.shape) + flow).astype(np.int64)
    in_bounds = (
        (endpoints[..., 0] >= 0)
        & (endpoints[..., 0] < source_mask.shape[1])
        & (endpoints[..., 1] >= 0)
        & (endpoints[..., 1] < source_mask.shape[0])
    )
    keep = valid & in_bounds
    propagated = np.zeros_like(source_mask)
    propagated[endpoints[..., 1][keep], endpoints[..., 0][keep]] = True
    return propagated, keep


def propagate_relation_mask(
    mask_t0: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    threshold_px: float = 1.5,
) -> dict[str, np.ndarray | bool]:
    """Propagate a relation mask and invalidate inconsistent/occluded pixels."""

    error, consistent = forward_backward_consistency(forward_flow, backward_flow, threshold_px)
    mask_t1, source_valid = propagate_mask_forward(mask_t0, forward_flow, consistent)
    valid_fraction = float(source_valid.sum() / max(1, np.asarray(mask_t0, dtype=bool).sum()))
    return {
        "mask_t0": np.asarray(mask_t0, dtype=bool),
        "mask_t1": mask_t1,
        "source_valid": source_valid,
        "fb_error": error,
        "valid": bool(np.asarray(mask_t0, dtype=bool).any() and valid_fraction >= 0.5),
        "valid_fraction": np.float32(valid_fraction),
    }


def decode_relation_mask(encoded_mask: Any, frame_size: tuple[int, int]) -> np.ndarray:
    """Decode an Ego4D Relations mask with the official decoder."""

    try:
        from ego4d.research.util.masks import decode_mask
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Ego4D relation mask decoding requires ego4d and pycocotools in the fact_tokenizer environment"
        ) from exc
    height, width = frame_size
    payload = encoded_mask
    if not isinstance(payload, Mapping):
        payload = {"width": width, "height": height, "encodedMask": encoded_mask}
    decoded = decode_mask(payload)
    mask = np.asarray(decoded, dtype=bool)
    if mask.shape != (height, width):
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask = F.interpolate(tensor, size=(height, width), mode="nearest")[0, 0].bool().numpy()
    return mask


def dino_delta(features_t0: torch.Tensor, features_t1: torch.Tensor) -> torch.Tensor:
    if features_t0.shape != features_t1.shape:
        raise ValueError("DINO endpoint features must have the same shape")
    return features_t1 - features_t0


def roi_feature_delta(
    patch_delta: torch.Tensor,
    roi_mask: torch.Tensor,
    patch_grid: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool patch deltas over a target-only object ROI with explicit validity."""

    if patch_delta.ndim != 3:
        raise ValueError("patch_delta must be BxPxD")
    batch, patches, _ = patch_delta.shape
    if patch_grid is None:
        side = int(round(patches**0.5))
        if side * side != patches:
            raise ValueError("patch_grid is required for non-square patch layouts")
        patch_grid = (side, side)
    if patch_grid[0] * patch_grid[1] != patches:
        raise ValueError("patch_grid does not match patch count")
    mask = roi_mask.float()
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[0] != batch:
        raise ValueError("roi_mask must be BxHxW or Bx1xHxW")
    weights = F.interpolate(mask, size=patch_grid, mode="area").flatten(1)
    validity = weights.sum(dim=1) > 1e-6
    pooled = (patch_delta * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = pooled * validity[:, None]
    return pooled, validity


def _annotation_midpoint(annotation: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "timestamp_sec", "time", "frame_timestamp"):
        if annotation.get(key) is not None:
            return float(annotation[key])
    starts = [annotation.get(key) for key in ("start_sec", "start_time", "start")]
    ends = [annotation.get(key) for key in ("end_sec", "end_time", "end")]
    start = next((float(v) for v in starts if v is not None), None)
    end = next((float(v) for v in ends if v is not None), None)
    if start is not None and end is not None:
        return 0.5 * (start + end)
    return start


def align_atomic_descriptions(
    annotations: Iterable[Mapping[str, Any]],
    transition_start_seconds: float,
    transition_end_seconds: float,
    tolerance_seconds: float = 0.75,
) -> list[dict[str, Any]]:
    """Keep raw atomic text close to a transition midpoint; no gold semantics are inferred."""

    midpoint = 0.5 * (transition_start_seconds + transition_end_seconds)
    aligned: list[dict[str, Any]] = []
    for annotation in annotations:
        annotation_time = _annotation_midpoint(annotation)
        if annotation_time is None or abs(annotation_time - midpoint) > tolerance_seconds:
            continue
        aligned.append(
            {
                "timestamp": annotation_time,
                "offset_seconds": annotation_time - midpoint,
                "text": annotation.get("text") or annotation.get("description") or annotation.get("label"),
                "source": annotation.get("source", "atomic_description"),
                "raw": dict(annotation),
            }
        )
    return sorted(aligned, key=lambda row: (abs(row["offset_seconds"]), row["timestamp"]))


def align_phase_segments(
    segments: Iterable[Mapping[str, Any]],
    transition_start_seconds: float,
    transition_end_seconds: float,
) -> list[dict[str, Any]]:
    """Align raw keystep/procedure intervals by temporal overlap."""

    aligned: list[dict[str, Any]] = []
    for segment in segments:
        start = segment.get("start_sec", segment.get("start_time", segment.get("start")))
        end = segment.get("end_sec", segment.get("end_time", segment.get("end")))
        if start is None or end is None:
            continue
        start_value, end_value = float(start), float(end)
        overlap = max(0.0, min(end_value, transition_end_seconds) - max(start_value, transition_start_seconds))
        if overlap <= 0:
            continue
        aligned.append(
            {
                "start_sec": start_value,
                "end_sec": end_value,
                "overlap_seconds": overlap,
                "step": segment.get("step") or segment.get("label") or segment.get("name"),
                "source": segment.get("source", "keystep_or_procedure"),
                "raw": dict(segment),
            }
        )
    return sorted(aligned, key=lambda row: (-row["overlap_seconds"], row["start_sec"]))


def deterministic_weak_semantic_labels(
    atomic_annotations: Iterable[Mapping[str, Any]],
    phase_segments: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Map human text to the frozen 8-way effect and 5-way contact taxonomies.

    This is deliberately a small deterministic lexicon, not a learned labeler.
    Unmatched text is retained but marked invalid, so it cannot silently become
    an ``ambiguous`` training target.
    """

    def extract_text(rows: Iterable[Mapping[str, Any]]) -> list[str]:
        values: list[str] = []
        for row in rows:
            raw = row.get("raw") if isinstance(row.get("raw"), Mapping) else row
            for key in (
                "text",
                "description",
                "label",
                "step",
                "step_name",
                "step_description",
                "name",
            ):
                value = raw.get(key) if isinstance(raw, Mapping) else None
                if value:
                    values.append(str(value))
        return values

    atomic_text = extract_text(atomic_annotations)
    phase_text = extract_text(phase_segments)
    texts = atomic_text + phase_text

    def normalize(values: Sequence[str]) -> str:
        value = " ".join(values).lower().replace("_", "-")
        value = re.sub(r"[^a-z0-9\- ]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    normalized_atomic = normalize(atomic_text)
    normalized_phase = normalize(phase_text)
    normalized = normalize(texts)

    lexicon: tuple[tuple[int, int, tuple[str, ...]], ...] = (
        (5, 3, ("release", "let go", "drop", "set down", "put down")),
        (6, 4, ("abort", "recover", "retry", "fail", "miss", "undo")),
        (2, 1, ("grasp", "grab", "pick up", "take hold", "acquire", "hold")),
        (
            3,
            2,
            (
                "open",
                "close",
                "cut",
                "slice",
                "pour",
                "stir",
                "press",
                "turn",
                "rotate",
                "insert",
                "remove",
                "attach",
                "detach",
                "tighten",
                "loosen",
                "fold",
            ),
        ),
        (4, 2, ("carry", "transport", "reposition", "transfer", "place", "move to", "put in")),
        (1, 0, ("approach", "reach", "align", "move toward", "position near")),
        (0, 0, ("wait", "look", "inspect", "observe", "no action", "idle")),
    )
    effect_id = 7
    contact_id = 4
    matched_phrase = None
    for text_source in (normalized_atomic, normalized_phase):
        for candidate_effect, candidate_contact, phrases in lexicon:
            match = next((phrase for phrase in phrases if phrase in text_source), None)
            if match is not None:
                effect_id = candidate_effect
                contact_id = candidate_contact
                matched_phrase = match
                break
        if matched_phrase is not None:
            break
    valid = matched_phrase is not None
    return {
        "mapping_version": WEAK_VERB_MAP_VERSION,
        "raw_text": texts,
        "normalized_text": normalized,
        "matched_phrase": matched_phrase,
        "phase": effect_id,
        "phase_name": WEAK_EFFECT_LABELS[effect_id],
        "phase_valid": valid,
        "contact": contact_id,
        "contact_name": WEAK_CONTACT_LABELS[contact_id],
        "contact_valid": valid,
    }


class RAFTFlowEstimator:
    """Lazy Torchvision RAFT wrapper; weights are never downloaded at import time."""

    def __init__(self, weights_path: Path | str, device: str | torch.device = "cuda") -> None:
        from torchvision.models.optical_flow import raft_large

        self.device = torch.device(device)
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)
        model = raft_large(weights=None, progress=False)
        state = torch.load(self.weights_path, map_location="cpu")
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, image_t0: torch.Tensor, image_t1: torch.Tensor) -> torch.Tensor:
        if image_t0.shape != image_t1.shape or image_t0.ndim != 4:
            raise ValueError("RAFT inputs must be matching BxCxHxW tensors")
        # Torchvision RAFT expects [-1, 1] input and dimensions divisible by 8.
        first = image_t0.to(self.device).float() * 2.0 - 1.0
        second = image_t1.to(self.device).float() * 2.0 - 1.0
        result = self.model(first, second)[-1]
        return result.permute(0, 2, 3, 1).cpu()


class EffectTargetCacheWriter:
    """Atomic per-sample NPZ cache with a hash-verifiable JSONL index."""

    def __init__(
        self,
        output_dir: Path | str,
        config: EffectTargetConfig,
        resume: bool = False,
        identity: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.identity = json.loads(json.dumps(dict(identity or {}), sort_keys=True))
        identity_payload = {
            "config_sha256": config.fingerprint(),
            "sources": self.identity,
        }
        self.identity_sha256 = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.sample_dir = self.output_dir / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        manifest = self.output_dir / "target_manifest.jsonl"
        metadata_path = self.output_dir / "target_config.json"
        if resume and manifest.is_file():
            if not metadata_path.is_file():
                raise ValueError("cannot resume target cache without target_config.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config_sha256") != config.fingerprint():
                raise ValueError("target cache config changed; refusing an unsafe resume")
            if metadata.get("identity_sha256") != self.identity_sha256:
                raise ValueError("target cache source/weight identity changed; refusing an unsafe resume")
            with manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self._records[str(record["sample_id"])] = record

    @staticmethod
    def _filename(sample_id: str) -> str:
        return hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npz"

    def write(
        self,
        sample_id: str,
        targets: Mapping[str, np.ndarray | bool | float | int],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        for target_name, value in targets.items():
            require_supported_target(target_name)
            normalized = target_name.strip().lower().replace("-", "_")
            if normalized.endswith("depth_valid") or normalized.endswith("flow_3d_valid"):
                if bool(np.asarray(value).item()):
                    raise ValueError(
                        f"{target_name} must be false: FACT v7 has no depth or dense 3D flow"
                    )
        final_path = self.sample_dir / self._filename(sample_id)
        temp_path = final_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp_path, **{key: np.asarray(value) for key, value in targets.items()})
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        temp_path.replace(final_path)
        self._records[sample_id] = {
            "sample_id": sample_id,
            "path": str(final_path.relative_to(self.output_dir)),
            "sha256": digest,
            "target_keys": sorted(targets),
            "config_sha256": self.config.fingerprint(),
            "identity_sha256": self.identity_sha256,
            "metadata": dict(metadata or {}),
        }
        return final_path

    def contains(self, sample_id: str) -> bool:
        record = self._records.get(sample_id)
        if record is None:
            return False
        path = self.output_dir / record["path"]
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    def finalize(self) -> Path:
        manifest = self.output_dir / "target_manifest.jsonl"
        temp = manifest.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(self._records.values(), key=lambda row: row["sample_id"]):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temp.replace(manifest)
        metadata = {
            "target_cache_version": self.config.target_cache_version,
            "config": asdict(self.config),
            "config_sha256": self.config.fingerprint(),
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "samples": len(self._records),
        }
        (self.output_dir / "target_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest


def verify_target_cache(output_dir: Path | str) -> list[str]:
    """Return cache integrity errors without mutating the cache."""

    root = Path(output_dir)
    manifest = root / "target_manifest.jsonl"
    config_path = root / "target_config.json"
    errors: list[str] = []
    if not config_path.is_file():
        return [f"missing {config_path}"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = config.get("config_sha256")
    expected_identity = config.get("identity_sha256")
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in seen:
                errors.append(f"line {line_number}: duplicate sample_id={sample_id}")
            seen.add(sample_id)
            if record.get("config_sha256") != expected_config:
                errors.append(f"line {line_number}: config identity mismatch for {sample_id}")
            if record.get("identity_sha256") != expected_identity:
                errors.append(f"line {line_number}: source identity mismatch for {sample_id}")
            path = root / record["path"]
            if not path.is_file():
                errors.append(f"line {line_number}: missing {path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                errors.append(f"line {line_number}: sha256 mismatch for {sample_id}")
    return errors


def build_bidirectional_flow(
    estimator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    frames: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an injected RAFT-compatible estimator on Bx2xCxHxW transitions."""

    if frames.ndim != 5 or frames.shape[1] != 2:
        raise ValueError("frames must be Bx2xCxHxW")
    forward = estimator(frames[:, 0], frames[:, 1])
    backward = estimator(frames[:, 1], frames[:, 0])
    return np.asarray(forward), np.asarray(backward)
