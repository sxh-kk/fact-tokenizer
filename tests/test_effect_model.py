from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from fact_tokenizer.effect_model import (
    BridgeContinuousEffectModel,
    ContinuousEffectModel,
    CrossEffectDecoder,
    EffectEncoder,
    PrivateHistoryEncoder,
)


class TemporalMixingBackbone(nn.Module):
    """Backbone that would leak future context if given the full transition."""

    def __init__(self, feature_dim: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.5, -0.25, 0.75],
                    ]
                )
            )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        frame_mean = videos.mean(dim=(-1, -2))
        features = self.projection(frame_mean)
        # Every output time step contains the mean of all input time steps.
        # The model must therefore truncate raw video before this call for the
        # private/current path, not merely slice backbone output afterwards.
        features = features + features.mean(dim=1, keepdim=True)
        return features.unsqueeze(2)


def make_model(**kwargs: object) -> ContinuousEffectModel:
    torch.manual_seed(7)
    return ContinuousEffectModel(
        backbone=TemporalMixingBackbone(feature_dim=4),
        backbone_dim=4,
        hidden_dim=8,
        semantic_dim=5,
        private_dim=3,
        image_geometry_dim=2,
        num_contact_classes=3,
        num_phase_classes=4,
        current_index=1,
        **kwargs,
    )


def make_batch(requires_grad: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    torch.manual_seed(11)
    ego = torch.randn(2, 4, 3, 6, 6, requires_grad=requires_grad)
    exo = torch.randn(2, 4, 3, 6, 6, requires_grad=requires_grad)
    return {"ego": {"videos": ego}, "exo": {"videos": exo}}


def test_continuous_model_emits_required_view_outputs_without_vq() -> None:
    model = make_model().eval()
    outputs = model(make_batch())

    assert outputs["mode"] == "continuous"
    assert outputs["vq_enabled"] is False
    assert set(outputs["views"]) == {"ego", "exo"}
    for view in outputs["views"].values():
        assert {"z_sem_cont", "g_image2d", "contact_logits", "phase_logits", "r_priv"}.issubset(view)
        assert view["z_sem"].shape == (2, 5)
        assert view["g_image2d"].shape == (2, 1, 2)
        assert view["contact"].shape == (2, 3)
        assert view["phase"].shape == (2, 4)
        assert not {"z_q", "indices", "soft_probs", "codebook"}.intersection(view)

    assert set(outputs["cross_predictions"]) == {"ego_from_exo", "exo_from_ego"}
    for path in outputs["cross_predictions"].values():
        assert {"dino_delta", "g_image2d", "contact_logits", "phase_logits"}.issubset(path)
        assert path["dino_delta"].shape == (2, 1, 4)
        assert path["g_image2d"].shape == (2, 1, 2)


def test_effect_path_reads_future_but_private_path_is_future_blind() -> None:
    model = make_model().eval()
    batch = make_batch()
    changed = {
        view: {"videos": values["videos"].clone()}
        for view, values in batch.items()
    }
    changed["ego"]["videos"][:, 2:] += 20.0

    original = model(batch)["views"]["ego"]
    modified = model(changed)["views"]["ego"]

    assert not torch.allclose(original["z_sem"], modified["z_sem"])
    assert torch.equal(original["r_priv"], modified["r_priv"])


def test_private_gradient_is_exactly_zero_for_future_raw_frames() -> None:
    model = make_model().eval()
    batch = make_batch(requires_grad=True)
    private = model(batch)["views"]["ego"]["r_priv"]
    private.sum().backward()

    gradient = batch["ego"]["videos"].grad
    assert gradient is not None
    assert gradient[:, :2].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 2:]) == 0


def test_private_history_encoder_also_enforces_its_own_cutoff() -> None:
    torch.manual_seed(13)
    encoder = PrivateHistoryEncoder(feature_dim=4, hidden_dim=6, private_dim=3).eval()
    features = torch.randn(2, 5, 2, 4)
    changed = features.clone()
    changed[:, 3:] += 100.0

    first = encoder(features, current_index=2)
    second = encoder(changed, current_index=2)
    assert torch.equal(first, second)


def test_effect_encoder_requires_and_uses_a_full_transition() -> None:
    torch.manual_seed(17)
    encoder = EffectEncoder(
        feature_dim=4,
        hidden_dim=8,
        semantic_dim=5,
        image_geometry_dim=2,
        num_contact_classes=2,
        num_phase_classes=3,
    ).eval()
    with pytest.raises(ValueError, match="at least one future"):
        encoder(torch.randn(2, 1, 3, 4))

    transition = torch.randn(2, 3, 3, 4)
    changed = transition.clone()
    changed[:, -1, :, 0] += 5.0
    assert not torch.allclose(encoder(transition)["z_sem"], encoder(changed)["z_sem"])


def test_cross_decoder_api_has_no_private_or_source_geometry_argument() -> None:
    parameters = list(inspect.signature(CrossEffectDecoder.forward).parameters)
    assert parameters == ["self", "target_current", "source_z_sem", "target_camera_context"]

    decoder = CrossEffectDecoder(
        target_feature_dim=4,
        semantic_dim=5,
        hidden_dim=8,
        image_geometry_dim=2,
        num_contact_classes=2,
        num_phase_classes=3,
    )
    current = torch.randn(2, 3, 4)
    semantic = torch.randn(2, 5)
    with pytest.raises(TypeError):
        decoder(current, semantic, private=torch.randn(2, 3, 2))
    with pytest.raises(TypeError):
        decoder(current, semantic, source_g_image2d=torch.randn(2, 3, 2))


def test_cross_decoder_conditions_on_target_camera_context() -> None:
    decoder = CrossEffectDecoder(
        target_feature_dim=4,
        semantic_dim=5,
        hidden_dim=8,
        image_geometry_dim=2,
        num_contact_classes=2,
        num_phase_classes=3,
        camera_context_dim=3,
    ).eval()
    current = torch.randn(2, 3, 4)
    semantic = torch.randn(2, 5)
    first = decoder(current, semantic, torch.zeros(2, 3))["dino_delta"]
    second = decoder(current, semantic, torch.ones(2, 3))["dino_delta"]
    assert not torch.allclose(first, second)


def test_stage1a_rejects_vq_and_3d_configuration() -> None:
    with pytest.raises(NotImplementedError, match="VQ"):
        make_model(vq_enabled=True)
    with pytest.raises(NotImplementedError, match="3D"):
        make_model(use_3d=True)
    with pytest.raises(NotImplementedError, match="3D"):
        make_model(geometry_mode="3d")


def test_state_dict_has_no_vq_or_codebook_and_private_does_not_change_core_cross() -> None:
    model = make_model().eval()
    assert not any("vq" in key.lower() or "codebook" in key.lower() for key in model.state_dict())
    batch = make_batch()
    before = model(batch)["cross_predictions"]["ego_from_exo"]["dino_delta"].detach().clone()
    with torch.no_grad():
        for parameter in model.private_encoders.parameters():
            parameter.add_(torch.randn_like(parameter) * 10)
    after = model(batch)["cross_predictions"]["ego_from_exo"]["dino_delta"].detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_ego_only_continuous_model_has_no_cross_path() -> None:
    model = ContinuousEffectModel(
        backbone=TemporalMixingBackbone(feature_dim=4),
        backbone_dim=4,
        hidden_dim=8,
        semantic_dim=5,
        private_dim=3,
        view_names=("ego",),
    ).eval()
    output = model({"ego": {"videos": torch.randn(2, 3, 3, 6, 6)}})
    assert set(output["views"]) == {"ego"}
    assert output["cross_predictions"] == {}


def test_private_uses_current_camera_but_future_camera_gradient_is_exact_zero() -> None:
    model = ContinuousEffectModel(
        backbone=TemporalMixingBackbone(feature_dim=4),
        backbone_dim=4,
        hidden_dim=8,
        semantic_dim=5,
        private_dim=3,
        view_names=("ego",),
        camera_context_dim=3,
    ).eval()
    current = torch.randn(2, 3, requires_grad=True)
    future = torch.randn(2, 3, requires_grad=True)
    output = model(
        {
            "ego": {
                "videos": torch.randn(2, 3, 3, 6, 6),
                "camera_context": current,
                "future_camera_context": future,
            }
        }
    )["views"]["ego"]["r_priv"]
    output.sum().backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert future.grad is not None
    torch.testing.assert_close(future.grad, torch.zeros_like(future.grad), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("private_future_visible", "expect_future_gradient"),
    [(True, True), (False, False)],
)
def test_bridge_c0_c1_isolates_private_future_shortcut(
    private_future_visible: bool,
    expect_future_gradient: bool,
) -> None:
    model = BridgeContinuousEffectModel(
        backbone=TemporalMixingBackbone(feature_dim=4),
        backbone_dim=4,
        hidden_dim=8,
        semantic_dim=5,
        private_dim=3,
        view_names=("ego", "exo"),
        private_future_visible=private_future_visible,
        cross_uses_private=True,
    ).eval()
    batch = make_batch(requires_grad=True)
    private = model(batch)["views"]["ego"]["r_priv"]
    private.sum().backward()
    future_gradient = batch["ego"]["videos"].grad[:, 1:].abs().sum().item()
    assert (future_gradient > 0) is expect_future_gradient
