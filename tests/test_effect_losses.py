from __future__ import annotations

import pytest
import torch

from fact_tokenizer.effect_losses import (
    EffectLossConfig,
    FACTV7ObjectiveConfig,
    compute_effect_loss,
    compute_effect_losses,
    compute_fact_v7_objective,
    masked_classification_loss,
    masked_regression_loss,
)


def make_predictions() -> dict[str, torch.Tensor]:
    return {
        "dino_delta": torch.ones(2, 3, 4, requires_grad=True),
        "g_image2d": torch.ones(2, 3, 2, requires_grad=True),
        "contact": torch.randn(2, 3, requires_grad=True),
        "phase": torch.randn(2, 4, requires_grad=True),
    }


def make_targets() -> dict[str, torch.Tensor]:
    return {
        "dino_delta": torch.zeros(2, 3, 4),
        "g_image2d": torch.zeros(2, 3, 2),
        "contact": torch.tensor([0, 2]),
        "phase": torch.tensor([1, 3]),
    }


def test_missing_masks_mean_no_supervision_and_zero_gradient() -> None:
    predictions = make_predictions()
    losses = compute_effect_losses(predictions, make_targets(), validity_masks={})
    assert losses["total"].item() == 0.0

    losses["total"].backward()
    for prediction in predictions.values():
        assert prediction.grad is not None
        assert torch.count_nonzero(prediction.grad) == 0


def test_each_target_uses_only_its_own_validity_mask() -> None:
    predictions = make_predictions()
    targets = make_targets()
    masks = {
        "dino_delta_valid": torch.tensor([True, False]),
        "g_image2d_valid": torch.zeros(2, 3, dtype=torch.bool),
        "contact_valid": torch.zeros(2, dtype=torch.bool),
        # phase mask deliberately absent
    }
    losses = compute_effect_losses(predictions, targets, masks)

    assert losses["dino_delta"] > 0
    assert losses["g_image2d"].item() == 0.0
    assert losses["contact"].item() == 0.0
    assert losses["phase"].item() == 0.0
    losses["total"].backward()

    assert predictions["dino_delta"].grad is not None
    assert predictions["dino_delta"].grad[0].abs().sum() > 0
    assert torch.count_nonzero(predictions["dino_delta"].grad[1]) == 0
    for name in ("g_image2d", "contact", "phase"):
        assert predictions[name].grad is not None
        assert torch.count_nonzero(predictions[name].grad) == 0


def test_masks_can_live_in_target_mapping() -> None:
    prediction = torch.tensor([[2.0], [3.0]], requires_grad=True)
    predictions = {"dino_delta": prediction}
    targets = {
        "dino_delta": torch.zeros_like(prediction),
        "dino_delta_valid": torch.tensor([1.0, 0.0]),
    }
    losses = compute_effect_losses(predictions, targets)
    assert losses["dino_delta"] > 0
    losses["total"].backward()
    assert prediction.grad is not None
    assert prediction.grad[0].abs().sum() > 0
    assert prediction.grad[1].item() == 0.0


def test_all_zero_mask_allows_missing_target_but_positive_mask_does_not() -> None:
    prediction = {"g_image2d": torch.randn(2, 3, 2, requires_grad=True)}
    zero = compute_effect_losses(
        prediction,
        {},
        {"g_image2d_valid": torch.zeros(2, 3, dtype=torch.bool)},
    )
    assert zero["total"].item() == 0.0

    with pytest.raises(KeyError, match="Missing target"):
        compute_effect_losses(
            prediction,
            {},
            {"g_image2d_valid": torch.ones(2, 3, dtype=torch.bool)},
        )


def test_masked_classification_has_no_invalid_sample_gradient() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    target = torch.tensor([0, -1, 2])
    loss = masked_classification_loss(logits, target, torch.tensor([True, False, True]))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert logits.grad[2].abs().sum() > 0


def test_invalid_regression_nan_placeholder_is_safely_masked() -> None:
    prediction = torch.ones(2, 2, requires_grad=True)
    target = torch.tensor([[0.0, 0.0], [float("nan"), float("nan")]])
    loss = masked_regression_loss(prediction, target, torch.tensor([True, False]))
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0].abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[1]) == 0


def test_missing_regression_mask_is_graph_connected_zero() -> None:
    prediction = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    loss = masked_regression_loss(prediction, target, None)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_3d_loss_configuration_fails_fast() -> None:
    with pytest.raises(NotImplementedError, match="3D"):
        EffectLossConfig(use_3d=True)
    with pytest.raises(NotImplementedError, match="3D"):
        EffectLossConfig(geometry_mode="3d")


def test_repository_style_loss_wrapper_returns_scalar_and_metrics() -> None:
    predictions = make_predictions()
    targets = make_targets()
    masks = {f"{name}_valid": torch.ones(2, dtype=torch.bool) for name in ("dino_delta", "g_image2d", "contact", "phase")}
    total, metrics = compute_effect_loss(predictions, targets, masks)
    assert total.ndim == 0
    assert set(metrics) == {"dino_delta", "g_image2d", "contact", "phase", "total"}
    assert metrics["total"] > 0


def _v7_output() -> dict:
    ego_z = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    exo_z = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    def view(z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "z_sem_cont": z,
            "dino_delta": torch.ones(2, 4, 3, requires_grad=True),
            "g_image2d": torch.ones(2, 4, 2, requires_grad=True),
            "contact_logits": torch.randn(2, 5, requires_grad=True),
            "phase_logits": torch.randn(2, 8, requires_grad=True),
        }

    return {"views": {"ego": view(ego_z), "exo": view(exo_z)}, "cross_predictions": {}}


def _v7_targets() -> dict:
    values = {}
    for view in ("ego", "exo"):
        mask = torch.zeros(2, 8, 8)
        mask[:, :4, :4] = 1
        values[view] = {
            "full_dino_delta": torch.zeros(2, 4, 3),
            "full_dino_delta_valid": torch.ones(2, dtype=torch.bool),
            "object_roi_dino_delta": torch.zeros(2, 3),
            "object_roi_dino_delta_valid": torch.ones(2, dtype=torch.bool),
            "relations_mask_t0": mask,
            "relations_mask_valid": torch.ones(2, dtype=torch.bool),
            "rotation_compensated_flow_2d": torch.zeros(2, 8, 8, 2),
            "rotation_compensated_flow_2d_valid": torch.ones(2, 8, 8, dtype=torch.bool),
            "contact": torch.tensor([0, 1]),
            "contact_valid": torch.ones(2, dtype=torch.bool),
            "phase": torch.tensor([2, 3]),
            "phase_valid": torch.ones(2, dtype=torch.bool),
        }
    return values


def test_v7_objective_uses_frozen_weights_and_independent_targets() -> None:
    output = _v7_output()
    losses = compute_fact_v7_objective(output, _v7_targets())
    expected = (
        losses["full_dino_delta"]
        + 0.5 * losses["object_roi_dino_delta"]
        + 0.25 * losses["rotation_compensated_flow_2d"]
        + 0.1 * losses["cross_view_semantic_consistency"]
    )
    torch.testing.assert_close(losses["total"], expected)
    assert losses["weak_semantics"].item() == 0.0
    losses["total"].backward()
    assert output["views"]["ego"]["dino_delta"].grad.abs().sum() > 0
    assert output["views"]["ego"]["g_image2d"].grad.abs().sum() > 0


def test_v7_roi_requires_explicit_target_and_propagation_validity() -> None:
    for missing in ("object_roi_dino_delta_valid", "relations_mask_valid"):
        output = _v7_output()
        targets = _v7_targets()
        for view in targets.values():
            view.pop(missing)
        losses = compute_fact_v7_objective(output, targets)
        assert losses["object_roi_dino_delta"].item() == 0.0


def test_weak_semantics_cannot_enable_before_dev_precision_gate() -> None:
    with pytest.raises(ValueError, match="measured 60-sample"):
        FACTV7ObjectiveConfig(weak_semantics_enabled=True)
    with pytest.raises(ValueError, match="gated"):
        FACTV7ObjectiveConfig(weak_semantics_enabled=True, weak_semantic_dev_precision=0.79)
    config = FACTV7ObjectiveConfig(weak_semantics_enabled=True, weak_semantic_dev_precision=0.80)
    losses = compute_fact_v7_objective(_v7_output(), _v7_targets(), config=config)
    assert losses["weak_semantics"].item() > 0
