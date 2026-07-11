"""Continuous, VQ-free interaction-effect model for the v7 FACT line.

The structural boundary in this module is intentional:

* :class:`EffectEncoder` receives the full transition and may therefore encode
  the observed effect.
* :class:`PrivateHistoryEncoder` receives current/history only.  It slices the
  temporal input itself as a second line of defence.
* :class:`CrossEffectDecoder` has no private-residual or source-view image
  geometry argument.  It predicts a target-view effect from target current
  context and source semantic effect only.

There is no vector quantizer in this file.  Late-VQ belongs to Stage 1B and is
deliberately rejected by the Stage 1A model constructor.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGE_2D_MODES = {"2d", "image2d", "image_2d"}


def _validate_stage1a_mode(*, vq_enabled: bool, geometry_mode: str, use_3d: bool) -> None:
    if vq_enabled:
        raise NotImplementedError("Stage 1A is continuous-only; VQ must remain disabled")
    if use_3d or geometry_mode.lower() not in _IMAGE_2D_MODES:
        raise NotImplementedError(
            "3D effect geometry is not implemented in Stage 1A; use geometry_mode='image2d'"
        )


def _as_patch_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize backbone output to ``(batch, time, patches, channels)``."""

    if features.ndim == 3:
        return features.unsqueeze(2)
    if features.ndim != 4:
        raise ValueError(
            "A feature backbone must return (B,T,D) or (B,T,P,D); "
            f"received shape {tuple(features.shape)}"
        )
    return features


def _extract_backbone_tensor(output: object) -> torch.Tensor:
    """Accept a tensor or a small set of conventional backbone dictionaries."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("patch_features", "patch_tokens", "x_norm_patchtokens", "features"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(
        "The injected backbone must return a tensor or a mapping containing "
        "patch_features/patch_tokens/x_norm_patchtokens/features"
    )


class LightweightPatchBackbone(nn.Module):
    """Small per-frame convolutional backbone useful for CPU tests and smoke runs."""

    def __init__(self, image_channels: int = 3, feature_dim: int = 64, patch_size: int = 8) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.projection = nn.Conv2d(
            image_channels,
            self.feature_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f"videos must have shape (B,T,C,H,W), got {tuple(videos.shape)}")
        batch, time = videos.shape[:2]
        features = self.projection(videos.reshape(batch * time, *videos.shape[2:]))
        features = features.flatten(2).transpose(1, 2)
        return features.reshape(batch, time, features.shape[1], features.shape[2])


class EffectEncoder(nn.Module):
    """Encode a complete observed transition into continuous effect outputs.

    ``z_sem`` is a continuous semantic latent in Stage 1A. ``g_image2d`` is a
    per-patch image-plane geometry prediction. ``contact`` and ``phase`` are
    logits, not calibrated probabilities.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        semantic_dim: int,
        image_geometry_dim: int = 2,
        num_contact_classes: int = 5,
        num_phase_classes: int = 8,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, semantic_dim, image_geometry_dim) <= 0:
            raise ValueError("feature, hidden, semantic, and image geometry dimensions must be positive")
        if num_contact_classes <= 1 or num_phase_classes <= 1:
            raise ValueError("contact and phase heads need at least two classes")

        self.token_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Mean and endpoint delta ensure that both the full transition and its
        # change are available without imposing fixed-grid temporal attention.
        summary_dim = hidden_dim * 2
        self.semantic_head = nn.Linear(summary_dim, semantic_dim)
        self.dino_delta_head = nn.Linear(hidden_dim, feature_dim)
        self.geometry_head = nn.Linear(hidden_dim, image_geometry_dim)
        self.contact_head = nn.Linear(summary_dim, num_contact_classes)
        self.phase_head = nn.Linear(summary_dim, num_phase_classes)

    def forward(self, transition_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        transition_features = _as_patch_features(transition_features)
        if transition_features.shape[1] < 2:
            raise ValueError("EffectEncoder requires current plus at least one future frame")

        hidden = self.token_encoder(transition_features)
        global_mean = hidden.mean(dim=(1, 2))
        endpoint_delta = hidden[:, -1].mean(dim=1) - hidden[:, 0].mean(dim=1)
        spatial_endpoint_delta = hidden[:, -1] - hidden[:, 0]
        summary = torch.cat([global_mean, endpoint_delta], dim=-1)
        z_sem_cont = self.semantic_head(summary)
        contact_logits = self.contact_head(summary)
        phase_logits = self.phase_head(summary)
        return {
            "z_sem_cont": z_sem_cont,
            "z_sem": z_sem_cont,
            "dino_delta": self.dino_delta_head(spatial_endpoint_delta),
            "g_image2d": self.geometry_head(spatial_endpoint_delta),
            "contact_logits": contact_logits,
            "contact": contact_logits,
            "phase_logits": phase_logits,
            "phase": phase_logits,
        }


class PrivateHistoryEncoder(nn.Module):
    """Encode private context after a hard current/history temporal cutoff."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        private_dim: int,
        camera_context_dim: int = 0,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, private_dim) <= 0:
            raise ValueError("feature, hidden, and private dimensions must be positive")
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, private_dim),
        )
        self.camera_context_dim = int(camera_context_dim)
        if self.camera_context_dim < 0:
            raise ValueError("camera_context_dim cannot be negative")
        self.camera_encoder = (
            nn.Sequential(nn.LayerNorm(self.camera_context_dim), nn.Linear(self.camera_context_dim, private_dim))
            if self.camera_context_dim
            else None
        )

    def forward(
        self,
        transition_features: torch.Tensor,
        current_index: int,
        camera_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        transition_features = _as_patch_features(transition_features)
        time = transition_features.shape[1]
        if current_index < 0 or current_index >= time:
            raise ValueError(f"current_index={current_index} is outside a transition of length {time}")
        # This slice is the class-level invariant. ContinuousEffectModel also
        # truncates raw videos before invoking the backbone, preventing leakage
        # through an injected backbone that mixes information over time.
        history = transition_features[:, : current_index + 1]
        private = self.encoder(history).mean(dim=(1, 2))
        if self.camera_encoder is not None:
            if camera_context is None:
                raise ValueError("camera_context is required when camera_context_dim > 0")
            if camera_context.ndim > 2:
                camera_context = camera_context.flatten(1)
            if camera_context.shape != (history.shape[0], self.camera_context_dim):
                raise ValueError(
                    f"camera_context must have shape (B,{self.camera_context_dim}), got {tuple(camera_context.shape)}"
                )
            private = private + self.camera_encoder(camera_context)
        return private


class CrossEffectDecoder(nn.Module):
    """Decode target-view effects without any private or source geometry path."""

    def __init__(
        self,
        target_feature_dim: int,
        semantic_dim: int,
        hidden_dim: int,
        dino_delta_dim: int | None = None,
        image_geometry_dim: int = 2,
        num_contact_classes: int = 5,
        num_phase_classes: int = 8,
        camera_context_dim: int = 0,
    ) -> None:
        super().__init__()
        dino_delta_dim = target_feature_dim if dino_delta_dim is None else int(dino_delta_dim)
        self.current_projection = nn.Linear(target_feature_dim, hidden_dim)
        self.semantic_projection = nn.Linear(semantic_dim, hidden_dim)
        self.camera_context_dim = int(camera_context_dim)
        if self.camera_context_dim < 0:
            raise ValueError("camera_context_dim cannot be negative")
        self.camera_projection = (
            nn.Linear(self.camera_context_dim, hidden_dim) if self.camera_context_dim else None
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.dino_delta_head = nn.Linear(hidden_dim, dino_delta_dim)
        self.geometry_head = nn.Linear(hidden_dim, image_geometry_dim)
        self.contact_head = nn.Linear(hidden_dim, num_contact_classes)
        self.phase_head = nn.Linear(hidden_dim, num_phase_classes)

    def forward(
        self,
        target_current: torch.Tensor,
        source_z_sem: torch.Tensor,
        target_camera_context: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict a target-view effect from current context and source semantics.

        The intentionally narrow signature is part of the architecture: there
        is no ``private`` input and no source-view ``g_image2d`` input.
        """

        if target_current.ndim == 4:
            if target_current.shape[1] != 1:
                raise ValueError("target_current may contain exactly one time step")
            target_current = target_current[:, 0]
        if target_current.ndim == 2:
            target_current = target_current.unsqueeze(1)
        if target_current.ndim != 3:
            raise ValueError(
                "target_current must have shape (B,P,D), (B,1,P,D), or (B,D); "
                f"got {tuple(target_current.shape)}"
            )
        if source_z_sem.ndim != 2 or source_z_sem.shape[0] != target_current.shape[0]:
            raise ValueError("source_z_sem must have shape (B,D) with the same batch size")

        current = self.current_projection(target_current)
        semantic = self.semantic_projection(source_z_sem).unsqueeze(1)
        fused = current + semantic
        if self.camera_projection is not None:
            if target_camera_context is None:
                raise ValueError("target_camera_context is required when camera_context_dim > 0")
            if target_camera_context.ndim > 2:
                target_camera_context = target_camera_context.flatten(1)
            if target_camera_context.shape != (target_current.shape[0], self.camera_context_dim):
                raise ValueError(
                    "target_camera_context must have shape "
                    f"(B,{self.camera_context_dim}), got {tuple(target_camera_context.shape)}"
                )
            fused = fused + self.camera_projection(target_camera_context).unsqueeze(1)
        elif target_camera_context is not None and target_camera_context.shape[0] != target_current.shape[0]:
            raise ValueError("target_camera_context batch size differs from target_current")
        hidden = self.fusion(fused)
        pooled = hidden.mean(dim=1)
        contact_logits = self.contact_head(pooled)
        phase_logits = self.phase_head(pooled)
        return {
            "dino_delta": self.dino_delta_head(hidden),
            "g_image2d": self.geometry_head(hidden),
            "contact": contact_logits,
            "contact_logits": contact_logits,
            "phase": phase_logits,
            "phase_logits": phase_logits,
        }


class ContinuousEffectModel(nn.Module):
    """Two-view Stage 1A model with continuous effects and VQ disabled."""

    def __init__(
        self,
        *,
        backbone: nn.Module | None = None,
        backbone_dim: int = 64,
        image_channels: int = 3,
        patch_size: int = 8,
        hidden_dim: int = 128,
        semantic_dim: int = 64,
        private_dim: int = 32,
        image_geometry_dim: int = 2,
        num_contact_classes: int = 5,
        num_phase_classes: int = 8,
        camera_context_dim: int = 0,
        view_names: Sequence[str] = ("ego", "exo"),
        current_index: int = 0,
        vq_enabled: bool = False,
        geometry_mode: str = "image2d",
        use_3d: bool = False,
    ) -> None:
        super().__init__()
        _validate_stage1a_mode(vq_enabled=vq_enabled, geometry_mode=geometry_mode, use_3d=use_3d)
        if not 1 <= len(view_names) <= 2 or len(set(view_names)) != len(view_names):
            raise ValueError("ContinuousEffectModel requires one or two distinct views")
        if current_index < 0:
            raise ValueError("current_index must be non-negative")

        self.view_names = tuple(view_names)
        self.current_index = int(current_index)
        self.vq_enabled = False
        self.geometry_mode = "image2d"
        self.backbone = backbone or LightweightPatchBackbone(
            image_channels=image_channels,
            feature_dim=backbone_dim,
            patch_size=patch_size,
        )
        self.effect_encoders = nn.ModuleDict(
            {
                view: EffectEncoder(
                    feature_dim=backbone_dim,
                    hidden_dim=hidden_dim,
                    semantic_dim=semantic_dim,
                    image_geometry_dim=image_geometry_dim,
                    num_contact_classes=num_contact_classes,
                    num_phase_classes=num_phase_classes,
                )
                for view in self.view_names
            }
        )
        self.private_encoders = nn.ModuleDict(
            {
                view: PrivateHistoryEncoder(
                    feature_dim=backbone_dim,
                    hidden_dim=hidden_dim,
                    private_dim=private_dim,
                    camera_context_dim=camera_context_dim,
                )
                for view in self.view_names
            }
        )
        self.cross_decoders = nn.ModuleDict(
            {
                view: CrossEffectDecoder(
                    target_feature_dim=backbone_dim,
                    semantic_dim=semantic_dim,
                    hidden_dim=hidden_dim,
                    dino_delta_dim=backbone_dim,
                    image_geometry_dim=image_geometry_dim,
                    num_contact_classes=num_contact_classes,
                    num_phase_classes=num_phase_classes,
                    camera_context_dim=camera_context_dim,
                )
                for view in self.view_names
            }
        )

    @staticmethod
    def _videos_for_view(value: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        videos = value.get("videos") if isinstance(value, Mapping) else value
        if not isinstance(videos, torch.Tensor):
            raise TypeError("Each view must be a video tensor or a mapping containing 'videos'")
        if videos.ndim == 4:
            videos = videos.unsqueeze(0)
        if videos.ndim != 5:
            raise ValueError(f"videos must have shape (B,T,C,H,W), got {tuple(videos.shape)}")
        return videos

    def _backbone_features(self, videos: torch.Tensor) -> torch.Tensor:
        return _as_patch_features(_extract_backbone_tensor(self.backbone(videos)))

    def _private_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """Final v7 private path sees history/current only."""

        return videos[:, : self.current_index + 1]

    def _cross_target_current(
        self,
        target_view: str,
        current_features: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        del target_view, private
        return current_features

    @staticmethod
    def _camera_context_for_view(
        value: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not isinstance(value, Mapping):
            return None
        context = value.get("camera_context")
        if context is not None and not isinstance(context, torch.Tensor):
            raise TypeError("camera_context must be a tensor")
        return context

    @staticmethod
    def _future_camera_context_for_view(
        value: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not isinstance(value, Mapping):
            return None
        context = value.get("future_camera_context")
        if context is not None and not isinstance(context, torch.Tensor):
            raise TypeError("future_camera_context must be a tensor")
        return context

    def forward(
        self,
        batch: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    ) -> Dict[str, object]:
        views: Dict[str, Dict[str, torch.Tensor]] = {}
        current_features: Dict[str, torch.Tensor] = {}
        camera_contexts: Dict[str, torch.Tensor | None] = {}

        for view in self.view_names:
            if view not in batch:
                raise KeyError(f"Missing required view: {view}")
            videos = self._videos_for_view(batch[view])
            if videos.shape[1] <= self.current_index + 1:
                raise ValueError(
                    "A full effect transition needs at least one future frame after current_index="
                    f"{self.current_index}; received T={videos.shape[1]}"
                )

            # Full-transition features are exclusive to the effect path.
            effect_features = self._backbone_features(videos)

            # The backbone itself sees no future for private/current context.
            # This remains safe even when an injected backbone mixes over time.
            history_videos = videos[:, : self.current_index + 1]
            history_features = self._backbone_features(history_videos)
            private_videos = self._private_videos(videos)
            private_features = (
                history_features if private_videos.shape[1] == history_videos.shape[1] else self._backbone_features(private_videos)
            )
            camera_context = self._camera_context_for_view(batch[view])
            private = self.private_encoders[view](
                private_features,
                current_index=private_features.shape[1] - 1,
                camera_context=camera_context,
            )
            future_camera_context = self._future_camera_context_for_view(batch[view])
            if future_camera_context is not None:
                # A graph-connected zero makes the no-leakage invariant directly
                # testable: the gradient is exactly zero rather than unused/None.
                private = private + future_camera_context.flatten(1).sum(dim=1, keepdim=True) * 0.0
            current = history_features[:, -1]

            effect = self.effect_encoders[view](effect_features)
            views[view] = {**effect, "r_priv": private}
            current_features[view] = current
            camera_contexts[view] = camera_context

        cross_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
        cross_pairs = (
            (
                (self.view_names[0], self.view_names[1]),
                (self.view_names[1], self.view_names[0]),
            )
            if len(self.view_names) == 2
            else ()
        )
        for target_view, source_view in cross_pairs:
            name = f"{target_view}_from_{source_view}"
            cross_predictions[name] = self.cross_decoders[target_view](
                self._cross_target_current(
                    target_view,
                    current_features[target_view],
                    views[target_view]["r_priv"],
                ),
                views[source_view]["z_sem_cont"],
                camera_contexts[target_view],
            )

        return {
            "mode": "continuous",
            "vq_enabled": False,
            "views": views,
            "cross_predictions": cross_predictions,
        }


# A concise alias for call sites that name the module by stage rather than mode.
Stage1AEffectModel = ContinuousEffectModel


class BridgeContinuousEffectModel(ContinuousEffectModel):
    """Explicit C0/C1 shortcut ablation; never the final v7 architecture.

    ``private_future_visible`` recreates the legacy future shortcut.  When
    ``cross_uses_private`` is true, private state is projected into target
    current tokens before the otherwise unchanged cross decoder.  C2 is the
    regular :class:`ContinuousEffectModel` and cannot enable either shortcut.
    """

    def __init__(
        self,
        *,
        private_future_visible: bool,
        cross_uses_private: bool,
        private_dim: int = 32,
        backbone_dim: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__(private_dim=private_dim, backbone_dim=backbone_dim, **kwargs)
        self.private_future_visible = bool(private_future_visible)
        self.cross_uses_private = bool(cross_uses_private)
        self.private_to_current = (
            nn.ModuleDict({view: nn.Linear(private_dim, backbone_dim) for view in self.view_names})
            if self.cross_uses_private
            else None
        )

    def _private_videos(self, videos: torch.Tensor) -> torch.Tensor:
        return videos if self.private_future_visible else super()._private_videos(videos)

    def _cross_target_current(
        self,
        target_view: str,
        current_features: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        if self.private_to_current is None:
            return current_features
        return current_features + self.private_to_current[target_view](private).unsqueeze(1)


__all__ = [
    "ContinuousEffectModel",
    "BridgeContinuousEffectModel",
    "CrossEffectDecoder",
    "EffectEncoder",
    "LightweightPatchBackbone",
    "PrivateHistoryEncoder",
    "Stage1AEffectModel",
]
