"""Per-target masked losses for the continuous Stage 1A effect model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGE_2D_MODES = {"2d", "image2d", "image_2d"}
_TARGET_NAMES = ("dino_delta", "g_image2d", "contact", "phase")


@dataclass
class EffectLossConfig:
    dino_delta_weight: float = 1.0
    image_geometry_weight: float = 1.0
    contact_weight: float = 1.0
    phase_weight: float = 1.0
    regression_loss: str = "smooth_l1"
    geometry_mode: str = "image2d"
    use_3d: bool = False

    def __post_init__(self) -> None:
        _validate_config(self)


def _validate_config(config: EffectLossConfig) -> None:
    if config.use_3d or config.geometry_mode.lower() not in _IMAGE_2D_MODES:
        raise NotImplementedError(
            "3D effect losses are not implemented; use geometry_mode='image2d'"
        )
    if config.regression_loss not in {"l1", "mse", "smooth_l1"}:
        raise ValueError("regression_loss must be one of: l1, mse, smooth_l1")
    for name in (
        "dino_delta_weight",
        "image_geometry_weight",
        "contact_weight",
        "phase_weight",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")


def _reference_tensor(predictions: Mapping[str, torch.Tensor]) -> torch.Tensor:
    for prediction in predictions.values():
        if isinstance(prediction, torch.Tensor):
            return prediction
    raise ValueError("predictions must contain at least one tensor")


def _zero_from(prediction: torch.Tensor) -> torch.Tensor:
    """Graph-connected zero with exactly zero gradient."""

    return prediction.sum() * 0.0


def _mask_for_target(
    name: str,
    targets: Mapping[str, torch.Tensor],
    validity_masks: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    candidates = (name, f"{name}_valid")
    if validity_masks is not None:
        for key in candidates:
            mask = validity_masks.get(key)
            if mask is not None:
                return mask
    mask = targets.get(f"{name}_valid")
    return mask if isinstance(mask, torch.Tensor) else None


def _validated_weights(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=reference.device)
    if weights.dtype == torch.bool:
        weights = weights.to(dtype=reference.dtype)
    elif not torch.is_floating_point(weights):
        weights = (weights != 0).to(dtype=reference.dtype)
    else:
        weights = weights.to(dtype=reference.dtype)
    if not torch.isfinite(weights).all():
        raise ValueError("validity masks must be finite")
    if (weights < 0).any():
        raise ValueError("validity masks must be non-negative")
    return weights


def _expand_weights(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    try:
        return torch.broadcast_to(weights, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"validity mask shape {tuple(weights.shape)} cannot broadcast to loss shape {tuple(values.shape)}"
        ) from exc


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded = _expand_weights(weights, values)
    denominator = expanded.sum()
    # The branch avoids division by zero while keeping the result connected to
    # predictions. Multiplication by an all-zero mask guarantees zero gradient.
    return (values * expanded).sum() / denominator.clamp_min(1.0)


def masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    loss: str = "smooth_l1",
) -> torch.Tensor:
    """Regression loss whose absent/all-zero mask produces zero gradient."""

    if valid_mask is None:
        return _zero_from(prediction)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    weights = _validated_weights(valid_mask, prediction)
    expanded_weights = _expand_weights(weights, prediction)
    target = target.to(prediction)
    # Invalid cache entries commonly contain NaN placeholders. Replacing them
    # before computing the pointwise loss prevents NaN * 0 from poisoning the
    # batch while preserving exact zero gradient at invalid positions.
    safe_target = torch.where(expanded_weights > 0, target, prediction.detach())
    if loss == "l1":
        elementwise = F.l1_loss(prediction, safe_target, reduction="none")
    elif loss == "mse":
        elementwise = F.mse_loss(prediction, safe_target, reduction="none")
    elif loss == "smooth_l1":
        elementwise = F.smooth_l1_loss(prediction, safe_target, reduction="none")
    else:
        raise ValueError("loss must be one of: l1, mse, smooth_l1")
    return _weighted_mean(elementwise, weights)


def masked_classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Cross entropy over valid positions only; class dimension is last."""

    if valid_mask is None:
        return _zero_from(logits)
    if logits.ndim < 2:
        raise ValueError("classification logits need a final class dimension")
    expected_target_shape = logits.shape[:-1]
    if target.shape != expected_target_shape:
        raise ValueError(
            f"classification target must have shape {expected_target_shape}, got {tuple(target.shape)}"
        )
    weights = _validated_weights(valid_mask, logits)
    expanded_weights = _expand_weights(weights, logits[..., 0])
    target = target.to(device=logits.device, dtype=torch.long)
    # Invalid class labels are often encoded as -1. They must be replaced
    # before cross entropy, not merely multiplied by zero afterwards.
    safe_target = torch.where(expanded_weights > 0, target, torch.zeros_like(target))
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape(expected_target_shape)
    return _weighted_mean(flat_loss, expanded_weights)


def compute_effect_losses(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    validity_masks: Mapping[str, torch.Tensor] | None = None,
    config: EffectLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute Stage 1A losses using an independent mask for every target.

    A target is supervised only when its own mask is present. A missing mask is
    never interpreted as "all valid". If a mask is present and contains valid
    entries, a missing prediction or target is treated as a data-contract error.
    """

    config = config or EffectLossConfig()
    _validate_config(config)
    reference = _reference_tensor(predictions)
    losses: Dict[str, torch.Tensor] = {}

    for name in _TARGET_NAMES:
        prediction = predictions.get(name)
        mask = _mask_for_target(name, targets, validity_masks)
        if prediction is None:
            if mask is None or not _validated_weights(mask, reference).bool().any():
                losses[name] = _zero_from(reference)
                continue
            raise KeyError(f"Missing prediction for valid target '{name}'")
        if mask is None:
            losses[name] = _zero_from(prediction)
            continue
        weights = _validated_weights(mask, prediction)
        if not weights.bool().any():
            losses[name] = _zero_from(prediction)
            continue
        target = targets.get(name)
        if target is None:
            raise KeyError(f"Missing target '{name}' for a non-empty validity mask")
        if name in {"contact", "phase"}:
            losses[name] = masked_classification_loss(prediction, target, mask)
        else:
            losses[name] = masked_regression_loss(
                prediction,
                target,
                mask,
                loss=config.regression_loss,
            )

    weighted = {
        "dino_delta": losses["dino_delta"] * config.dino_delta_weight,
        "g_image2d": losses["g_image2d"] * config.image_geometry_weight,
        "contact": losses["contact"] * config.contact_weight,
        "phase": losses["phase"] * config.phase_weight,
    }
    losses["total"] = torch.stack(tuple(weighted.values())).sum()
    return losses


def compute_effect_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    validity_masks: Mapping[str, torch.Tensor] | None = None,
    config: EffectLossConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Repository-style scalar loss plus detached metric dictionary."""

    losses = compute_effect_losses(predictions, targets, validity_masks, config)
    metrics = {name: float(value.detach()) for name, value in losses.items()}
    return losses["total"], metrics


class ContinuousEffectLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`compute_effect_losses`."""

    def __init__(self, config: EffectLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or EffectLossConfig()

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        validity_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        return compute_effect_losses(predictions, targets, validity_masks, self.config)


@dataclass
class FACTV7ObjectiveConfig:
    """Frozen Stage-1A weights from the v7 paired-value preregistration."""

    full_dino_delta_weight: float = 1.00
    object_roi_dino_delta_weight: float = 0.50
    rotation_compensated_flow_2d_weight: float = 0.25
    cross_view_semantic_consistency_weight: float = 0.10
    pose_camera_nuisance_consistency_weight: float = 0.10
    weak_semantic_weight: float = 0.10
    weak_semantics_enabled: bool = False
    weak_semantic_dev_precision: Optional[float] = None
    weak_semantic_minimum_dev_precision: float = 0.80

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name.endswith("_weight") and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.weak_semantics_enabled:
            if self.weak_semantic_dev_precision is None:
                raise ValueError("weak semantic loss requires a measured 60-sample dev precision")
            if self.weak_semantic_dev_precision < self.weak_semantic_minimum_dev_precision:
                raise ValueError(
                    "weak semantic loss is gated until dev precision reaches "
                    f"{self.weak_semantic_minimum_dev_precision:.2f}"
                )


def _mean_or_zero(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else _zero_from(reference)


def _target_value(targets: Mapping[str, object], view: str, name: str) -> torch.Tensor | None:
    nested = targets.get(view)
    if isinstance(nested, Mapping):
        value = nested.get(name)
    else:
        value = targets.get(f"{view}_{name}")
    return value if isinstance(value, torch.Tensor) else None


def _validity_value(targets: Mapping[str, object], view: str, name: str) -> torch.Tensor | None:
    return _target_value(targets, view, f"{name}_valid")


def _patch_grid(patches: int) -> tuple[int, int]:
    side = int(round(patches**0.5))
    if side * side != patches:
        raise ValueError("v7 objective needs an explicit square patch grid")
    return side, side


def _flow_target_to_patches(
    target: torch.Tensor,
    valid: torch.Tensor,
    patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 4 or target.shape[-1] != 2:
        raise ValueError("rotation_compensated_flow_2d target must be BxHxWx2")
    if valid.ndim == 4 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    if valid.ndim != 3:
        raise ValueError("flow validity must be BxHxW")
    grid = _patch_grid(patches)
    weights = valid.float()[:, None]
    numerator = F.adaptive_avg_pool2d(target.permute(0, 3, 1, 2) * weights, grid)
    denominator = F.adaptive_avg_pool2d(weights, grid)
    pooled = numerator / denominator.clamp_min(1e-6)
    pooled = pooled.permute(0, 2, 3, 1).flatten(1, 2)
    patch_valid = denominator[:, 0].flatten(1) > 1e-6
    return pooled, patch_valid


def _roi_pool_patch_values(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError("patch values must be BxPxD")
    if mask.ndim == 3:
        mask = mask[:, None]
    grid = _patch_grid(values.shape[1])
    weights = F.adaptive_avg_pool2d(mask.float(), grid).flatten(1)
    valid = weights.sum(dim=1) > 1e-6
    pooled = (values * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return pooled * valid[:, None], valid


def compute_fact_v7_objective(
    model_output: Mapping[str, object],
    targets: Mapping[str, object],
    *,
    nuisance_output: Mapping[str, object] | None = None,
    config: FACTV7ObjectiveConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute the preregistered continuous v7 objective with independent masks.

    Gold effect/contact labels are intentionally not accepted by this function;
    they belong to fixed linear probes and never update the effect encoder.
    """

    config = config or FACTV7ObjectiveConfig()
    views = model_output.get("views")
    if not isinstance(views, Mapping) or not views:
        raise ValueError("model_output must contain non-empty views")
    first_view = next(iter(views.values()))
    if not isinstance(first_view, Mapping) or not isinstance(first_view.get("z_sem_cont"), torch.Tensor):
        raise ValueError("model view output lacks z_sem_cont")
    reference = first_view["z_sem_cont"]
    dino_losses: list[torch.Tensor] = []
    roi_losses: list[torch.Tensor] = []
    flow_losses: list[torch.Tensor] = []
    weak_losses: list[torch.Tensor] = []

    prediction_sets: list[tuple[str, Mapping[str, torch.Tensor]]] = []
    for view, prediction in views.items():
        if isinstance(prediction, Mapping):
            prediction_sets.append((str(view), prediction))
    cross = model_output.get("cross_predictions", {})
    if isinstance(cross, Mapping):
        for name, prediction in cross.items():
            if isinstance(prediction, Mapping) and "_from_" in str(name):
                prediction_sets.append((str(name).split("_from_", 1)[0], prediction))

    for view, prediction in prediction_sets:
        predicted_delta = prediction.get("dino_delta")
        target_delta = _target_value(targets, view, "full_dino_delta")
        target_valid = _validity_value(targets, view, "full_dino_delta")
        if isinstance(predicted_delta, torch.Tensor) and target_delta is not None and target_valid is not None:
            dino_losses.append(masked_regression_loss(predicted_delta, target_delta, target_valid))

            roi_target = _target_value(targets, view, "object_roi_dino_delta")
            roi_valid = _validity_value(targets, view, "object_roi_dino_delta")
            roi_mask = _target_value(targets, view, "relations_mask_t0")
            propagation_valid = _validity_value(targets, view, "relations_mask")
            if (
                roi_target is not None
                and roi_mask is not None
                and roi_valid is not None
                and propagation_valid is not None
            ):
                roi_prediction, predicted_roi_valid = _roi_pool_patch_values(predicted_delta, roi_mask)
                combined_valid = (
                    predicted_roi_valid
                    & roi_valid.bool().reshape(-1)
                    & propagation_valid.bool().reshape(-1)
                )
                roi_losses.append(masked_regression_loss(roi_prediction, roi_target, combined_valid))

        predicted_flow = prediction.get("g_image2d")
        target_flow = _target_value(targets, view, "rotation_compensated_flow_2d")
        flow_valid = _validity_value(targets, view, "rotation_compensated_flow_2d")
        if isinstance(predicted_flow, torch.Tensor) and target_flow is not None and flow_valid is not None:
            patch_target, patch_valid = _flow_target_to_patches(
                target_flow,
                flow_valid,
                predicted_flow.shape[1],
            )
            flow_losses.append(masked_regression_loss(predicted_flow, patch_target, patch_valid))

    semantic_losses: list[torch.Tensor] = []
    view_names = list(views)
    if len(view_names) == 2:
        first = views[view_names[0]]["z_sem_cont"]
        second = views[view_names[1]]["z_sem_cont"]
        semantic_losses.append((1.0 - F.cosine_similarity(first, second, dim=-1)).mean())

    nuisance_losses: list[torch.Tensor] = []
    nuisance_views = nuisance_output.get("views") if isinstance(nuisance_output, Mapping) else None
    if isinstance(nuisance_views, Mapping):
        for view in view_names:
            if view in nuisance_views:
                nuisance_losses.append(
                    (
                        1.0
                        - F.cosine_similarity(
                            views[view]["z_sem_cont"], nuisance_views[view]["z_sem_cont"], dim=-1
                        )
                    ).mean()
                )

    if config.weak_semantics_enabled:
        for view in view_names:
            prediction = views[view]
            local: list[torch.Tensor] = []
            for name, prediction_name in (("contact", "contact_logits"), ("phase", "phase_logits")):
                label = _target_value(targets, view, name)
                valid = _validity_value(targets, view, name)
                logits = prediction.get(prediction_name)
                if isinstance(logits, torch.Tensor) and label is not None and valid is not None:
                    local.append(masked_classification_loss(logits, label, valid))
            if local:
                weak_losses.append(torch.stack(local).mean())

    components = {
        "full_dino_delta": _mean_or_zero(dino_losses, reference),
        "object_roi_dino_delta": _mean_or_zero(roi_losses, reference),
        "rotation_compensated_flow_2d": _mean_or_zero(flow_losses, reference),
        "cross_view_semantic_consistency": _mean_or_zero(semantic_losses, reference),
        "pose_camera_nuisance_consistency": _mean_or_zero(nuisance_losses, reference),
        "weak_semantics": _mean_or_zero(weak_losses, reference),
    }
    components["total"] = (
        config.full_dino_delta_weight * components["full_dino_delta"]
        + config.object_roi_dino_delta_weight * components["object_roi_dino_delta"]
        + config.rotation_compensated_flow_2d_weight * components["rotation_compensated_flow_2d"]
        + config.cross_view_semantic_consistency_weight * components["cross_view_semantic_consistency"]
        + config.pose_camera_nuisance_consistency_weight * components["pose_camera_nuisance_consistency"]
        + config.weak_semantic_weight * components["weak_semantics"]
    )
    return components


__all__ = [
    "ContinuousEffectLoss",
    "EffectLossConfig",
    "FACTV7ObjectiveConfig",
    "compute_effect_loss",
    "compute_effect_losses",
    "compute_fact_v7_objective",
    "masked_classification_loss",
    "masked_regression_loss",
]
