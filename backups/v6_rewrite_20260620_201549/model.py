"""FACT tokenizer model for factorized ego/exo interaction tokens."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class LearnedTokenPosition(nn.Module):
    def __init__(self, model_dim: int, max_time: int = 8, max_tokens: int = 1024) -> None:
        super().__init__()
        self.time_embed = nn.Parameter(torch.zeros(max_time, model_dim))
        self.token_embed = nn.Parameter(torch.zeros(max_tokens, model_dim))
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.token_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, S, D)
        time, tokens = x.shape[1:3]
        if time > self.time_embed.shape[0] or tokens > self.token_embed.shape[0]:
            raise ValueError(
                f"Position table too small for T={time}, S={tokens}; "
                f"configured {self.time_embed.shape[0]}, {self.token_embed.shape[0]}"
            )
        return x + self.time_embed[:time][None, :, None, :] + self.token_embed[:tokens][None, None, :, :]


class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_time: int = 8,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim))
        self.position = LearnedTokenPosition(model_dim, max_time=max_time, max_tokens=max_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, tokens = x.shape[:3]
        x = self.in_proj(x)
        x = self.position(x)
        x = x.reshape(batch, time * tokens, x.shape[-1])
        x = self.blocks(x)
        return x.reshape(batch, time, tokens, -1)


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_time: int = 8,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.position = LearnedTokenPosition(model_dim, max_time=max_time, max_tokens=max_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: torch.Tensor, num_patch_tokens: int) -> torch.Tensor:
        # x: (B, T, S, D). Decode per transition, with spatial mixing inside each T.
        batch, time, tokens = x.shape[:3]
        x = self.position(x)
        x = x.reshape(batch * time, tokens, x.shape[-1])
        x = self.blocks(x)
        x = self.out(x).reshape(batch, time, tokens, -1)
        return x[:, :, -num_patch_tokens:]


class MockPatchFeatureExtractor(nn.Module):
    """Fast deterministic patch feature extractor for CPU smoke tests."""

    def __init__(self, image_channels: int, dino_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(image_channels, dino_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.requires_grad_(False)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch, time = videos.shape[:2]
        x = videos.reshape(batch * time, *videos.shape[2:])
        features = self.proj(x)
        features = rearrange(features, "bt d h w -> bt (h w) d")
        return features.reshape(batch, time, features.shape[1], features.shape[2])


class DINOv2PatchFeatureExtractor(nn.Module):
    def __init__(self, dino_model: str = "dinov2_vitb14_reg", torch_home: str | None = None) -> None:
        super().__init__()
        if torch_home:
            os.environ.setdefault("TORCH_HOME", torch_home)
        hub_root = Path(os.environ.get("TORCH_HOME", "")).expanduser() / "hub"
        local_repo = hub_root / "facebookresearch_dinov2_main"
        local_weights = hub_root / "checkpoints" / "dinov2_vitb14_reg4_pretrain.pth"
        if local_repo.exists() and local_weights.exists() and dino_model == "dinov2_vitb14_reg":
            self.encoder = torch.hub.load(
                str(local_repo),
                dino_model,
                source="local",
                weights=str(local_weights),
            )
        else:
            self.encoder = torch.hub.load("facebookresearch/dinov2", dino_model)
        self.encoder.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_DEFAULT_STD).view(1, 1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch, time = videos.shape[:2]
        x = (videos - self.mean) / self.std
        x = x.reshape(batch * time, *x.shape[2:])
        features = self.encoder.forward_features(x)["x_norm_patchtokens"]
        return features.reshape(batch, time, features.shape[1], features.shape[2])


class SoftVectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int, temperature: float = 0.1) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)
        self.num_latents = num_latents
        self.temperature = temperature
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)

    def reset_usage(self) -> None:
        self.usage.zero_()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        distances = torch.cdist(x, self.codebook.weight)
        indices = torch.argmin(distances, dim=-1)
        z = self.codebook(indices)
        z_q = x + (z - x).detach()
        soft_probs = torch.softmax(-distances / self.temperature, dim=-1)
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(self.num_latents)
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(indices.reshape(-1), minlength=self.num_latents).to(self.usage)
                self.usage.add_(counts)
        return {
            "z_q": z_q,
            "z": z,
            "emb": x,
            "indices": indices,
            "soft_probs": soft_probs,
            "confidence": confidence.clamp(0.0, 1.0),
            "distances": distances,
        }


class ViewFactorizedEncoder(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        model_dim: int,
        latent_dim: int,
        private_dim: int,
        num_action_slots: int,
        num_private_slots: int,
        enc_blocks: int,
        num_heads: int,
        dropout: float,
        max_time: int,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.num_action_slots = num_action_slots
        self.num_private_slots = num_private_slots
        self.action_latent = nn.Parameter(torch.empty(1, 1, num_action_slots, dino_dim))
        self.private_latent = nn.Parameter(torch.empty(1, 1, num_private_slots, dino_dim))
        nn.init.uniform_(self.action_latent, a=-1.0, b=1.0)
        nn.init.uniform_(self.private_latent, a=-1.0, b=1.0)
        self.encoder = SpatioTemporalEncoder(
            in_dim=dino_dim,
            model_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            max_time=max_time,
            max_tokens=max_tokens,
        )
        self.to_action = nn.Linear(model_dim, latent_dim)
        self.to_private = nn.Linear(model_dim, private_dim)

    def forward(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, time = patch_features.shape[:2]
        action_tokens = self.action_latent.expand(batch, time, -1, -1)
        private_tokens = self.private_latent.expand(batch, time, -1, -1)
        padded = torch.cat([action_tokens, private_tokens, patch_features], dim=2)
        encoded = self.encoder(padded)
        future = encoded[:, 1:]
        action_hidden = future[:, :, : self.num_action_slots]
        private_start = self.num_action_slots
        private_end = private_start + self.num_private_slots
        private_hidden = future[:, :, private_start:private_end]
        return {
            "z_act": self.to_action(action_hidden),
            "r_priv": self.to_private(private_hidden),
        }


class FACTTokenizer(nn.Module):
    """Factorized action tokenizer with shared action VQ and private residuals."""

    def __init__(
        self,
        image_channels: int = 3,
        model_dim: int = 768,
        dino_dim: int = 768,
        latent_dim: int = 128,
        private_dim: int = 64,
        num_latents: int = 16,
        num_action_slots: int = 4,
        num_private_slots: int = 2,
        patch_size: int = 14,
        enc_blocks: int = 4,
        dec_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        vq_temperature: float = 0.1,
        backbone: str = "dino",
        dino_model: str = "dinov2_vitb14_reg",
        torch_home: str | None = None,
        view_names: Tuple[str, str] = ("ego", "exo"),
        max_time: int = 8,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.view_names = tuple(view_names)
        self.num_action_slots = num_action_slots
        self.num_private_slots = num_private_slots
        self.latent_dim = latent_dim
        self.private_dim = private_dim
        self.num_latents = num_latents
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        if backbone == "mock":
            self.feature_extractor = MockPatchFeatureExtractor(image_channels, dino_dim, patch_size)
        elif backbone == "dino":
            self.feature_extractor = DINOv2PatchFeatureExtractor(dino_model=dino_model, torch_home=torch_home)
        else:
            raise ValueError("backbone must be 'dino' or 'mock'")

        encoder_kwargs = dict(
            dino_dim=dino_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            private_dim=private_dim,
            num_action_slots=num_action_slots,
            num_private_slots=num_private_slots,
            enc_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            max_time=max_time,
            max_tokens=max_tokens,
        )
        self.ego_encoder = ViewFactorizedEncoder(**encoder_kwargs)
        self.exo_encoder = ViewFactorizedEncoder(**encoder_kwargs)
        self.action_vq = SoftVectorQuantizer(num_latents, latent_dim, temperature=vq_temperature)

        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.private_up = nn.Linear(private_dim, model_dim)
        self.decoder = SpatialDecoder(
            model_dim=model_dim,
            out_dim=dino_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            max_time=max_time,
            max_tokens=max_tokens,
        )

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def encode_views(self, batch: Mapping[str, Mapping[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        encoded: Dict[str, Dict[str, torch.Tensor]] = {}
        for view_name in self.view_names:
            videos = batch[view_name]["videos"].to(self.device, non_blocking=True)
            if videos.ndim == 4:
                videos = videos.unsqueeze(0)
            patch_features = self.feature_extractor(videos)
            encoder = self.ego_encoder if view_name == self.view_names[0] else self.exo_encoder
            view_out = encoder(patch_features)
            quantized = self.action_vq(view_out["z_act"])
            encoded[view_name] = {
                "patches": patch_features,
                "current_patches": patch_features[:, :-1],
                "target_patches": patch_features[:, 1:],
                "r_priv": view_out["r_priv"],
                **quantized,
            }
        return encoded

    def _decode_path(
        self,
        obs_view: Dict[str, torch.Tensor],
        act_view: Dict[str, torch.Tensor],
        private_dropout: float = 0.0,
        zero_private: bool = False,
        zero_action: bool = False,
        action_slot_dropout: float = 0.0,
    ) -> torch.Tensor:
        current_tokens = self.patch_up(obs_view["current_patches"])
        action = act_view["z_q"]
        if zero_action:
            action = torch.zeros_like(action)
        elif self.training and action_slot_dropout > 0.0:
            keep_prob = max(1.0 - float(action_slot_dropout), 0.0)
            if keep_prob <= 0.0:
                action = torch.zeros_like(action)
            else:
                mask = torch.rand((*action.shape[:-1], 1), device=action.device, dtype=action.dtype) < keep_prob
                if action.shape[-2] > 1:
                    empty = mask.sum(dim=-2, keepdim=True) == 0
                    fallback = torch.zeros_like(mask)
                    fallback[..., 0, :] = 1
                    mask = torch.where(empty, fallback, mask)
                action = action * mask
        action_tokens = self.action_up(action)
        private = obs_view["r_priv"]
        if zero_private:
            private = torch.zeros_like(private)
        elif self.training and private_dropout > 0.0:
            keep_prob = max(1.0 - float(private_dropout), 0.0)
            if keep_prob <= 0.0:
                private = torch.zeros_like(private)
            else:
                mask = torch.rand((*private.shape[:-1], 1), device=private.device, dtype=private.dtype) < keep_prob
                private = private * mask
        private_tokens = self.private_up(private)
        decoder_input = torch.cat([action_tokens, private_tokens, current_tokens], dim=2)
        return self.decoder(decoder_input, num_patch_tokens=current_tokens.shape[2])

    def _make_shuffled_action_view(
        self,
        view: Dict[str, torch.Tensor],
        take_index: torch.Tensor | None = None,
        same_take: bool = False,
    ) -> Dict[str, torch.Tensor]:
        shuffled = dict(view)
        batch = view["z_q"].shape[0]
        if batch <= 1:
            shuffled["z_q"] = view["z_q"]
            return shuffled
        if same_take and take_index is not None:
            take_index = take_index.to(view["z_q"].device)
            permutation = torch.arange(batch, device=view["z_q"].device)
            for take in torch.unique(take_index):
                members = torch.nonzero(take_index == take, as_tuple=False).flatten()
                if members.numel() <= 1:
                    continue
                local = torch.randperm(members.numel(), device=view["z_q"].device)
                fixed = local == torch.arange(members.numel(), device=view["z_q"].device)
                if fixed.any():
                    local = local.roll(1)
                permutation[members] = members[local]
        else:
            permutation = torch.randperm(batch, device=view["z_q"].device)
            fixed = permutation == torch.arange(batch, device=view["z_q"].device)
            if fixed.any():
                permutation = permutation.roll(1)
        shuffled["z_q"] = view["z_q"].index_select(0, permutation)
        return shuffled

    def _make_temporal_offset_action_view(
        self,
        view: Dict[str, torch.Tensor],
        take_index: torch.Tensor | None,
        timestamp: torch.Tensor | None,
        offset: int = 4,
    ) -> Dict[str, torch.Tensor]:
        shifted = dict(view)
        batch = view["z_q"].shape[0]
        if batch <= 1 or take_index is None:
            shifted["z_q"] = view["z_q"]
            return shifted
        take_index = take_index.to(view["z_q"].device)
        if timestamp is None:
            timestamp = torch.arange(batch, device=view["z_q"].device, dtype=torch.float32)
        else:
            timestamp = timestamp.to(view["z_q"].device)
        permutation = torch.arange(batch, device=view["z_q"].device)
        shift = max(int(offset), 1)
        for take in torch.unique(take_index):
            members = torch.nonzero(take_index == take, as_tuple=False).flatten()
            if members.numel() <= 1:
                continue
            order = torch.argsort(timestamp.index_select(0, members))
            ordered_members = members.index_select(0, order)
            local_shift = shift % int(ordered_members.numel())
            if local_shift == 0:
                local_shift = 1
            permutation[ordered_members] = ordered_members.roll(local_shift)
        shifted["z_q"] = view["z_q"].index_select(0, permutation)
        return shifted

    def _make_action_aware_action_view(
        self,
        obs_view: Dict[str, torch.Tensor],
        act_view: Dict[str, torch.Tensor],
        take_index: torch.Tensor | None,
        context_weight: float = 0.35,
    ) -> Dict[str, torch.Tensor]:
        hardened = dict(act_view)
        batch = act_view["z_q"].shape[0]
        if batch <= 1 or take_index is None:
            hardened["z_q"] = act_view["z_q"]
            hardened["contrast_weight"] = act_view["z_q"].new_zeros(batch)
            return hardened

        device = act_view["z_q"].device
        take_index = take_index.to(device)
        current = obs_view["current_patches"].detach().reshape(batch, -1)
        delta = (obs_view["target_patches"] - obs_view["current_patches"]).detach().reshape(batch, -1)
        current = F.normalize(current, dim=1)
        delta = F.normalize(delta, dim=1)
        context_distance = torch.cdist(current, current)
        action_distance = torch.cdist(delta, delta)
        score = action_distance - float(context_weight) * context_distance

        same_take = take_index[:, None] == take_index[None, :]
        not_self = ~torch.eye(batch, dtype=torch.bool, device=device)
        valid = same_take & not_self
        masked_score = score.masked_fill(~valid, -torch.inf)
        best_score, permutation = masked_score.max(dim=1)
        has_negative = torch.isfinite(best_score)
        permutation = torch.where(has_negative, permutation, torch.arange(batch, device=device))

        source_confidence = act_view["confidence"].detach().mean(dim=tuple(range(1, act_view["confidence"].ndim)))
        pair_confidence = torch.minimum(source_confidence, source_confidence.index_select(0, permutation))
        distance_scale = action_distance.gather(1, permutation[:, None]).squeeze(1)
        distance_scale = distance_scale / distance_scale[has_negative].mean().clamp_min(1e-6) if has_negative.any() else distance_scale
        contrast_weight = has_negative.to(act_view["z_q"].dtype) * pair_confidence * distance_scale.clamp(0.25, 2.0)

        hardened["z_q"] = act_view["z_q"].index_select(0, permutation)
        hardened["contrast_weight"] = contrast_weight.detach()
        return hardened

    def _make_random_code_action_view(self, view: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        randomized = dict(view)
        indices = view["indices"]
        if self.num_latents <= 1:
            randomized["z_q"] = view["z_q"]
            return randomized
        offsets = torch.randint(
            low=1,
            high=self.num_latents,
            size=indices.shape,
            device=indices.device,
            dtype=indices.dtype,
        )
        random_indices = (indices + offsets) % self.num_latents
        randomized["z_q"] = self.action_vq.codebook(random_indices)
        return randomized

    def _add_reconstruction(
        self,
        reconstructions: Dict[str, Dict[str, torch.Tensor | str]],
        name: str,
        obs_view_name: str,
        act_view_name: str,
        views: Dict[str, Dict[str, torch.Tensor]],
        private_dropout: float,
        zero_private: bool = False,
        zero_action: bool = False,
        shuffled_action: bool = False,
        same_take_action: bool = False,
        temporal_offset_action: bool = False,
        temporal_offset: int = 4,
        action_aware_action: bool = False,
        action_aware_context_weight: float = 0.35,
        random_code_action: bool = False,
        action_slot_dropout: float = 0.0,
        loss_role: str = "base",
        take_index: torch.Tensor | None = None,
        timestamp: torch.Tensor | None = None,
    ) -> None:
        if random_code_action:
            act_view = self._make_random_code_action_view(views[act_view_name])
        elif action_aware_action:
            act_view = self._make_action_aware_action_view(
                views[obs_view_name],
                views[act_view_name],
                take_index=take_index,
                context_weight=action_aware_context_weight,
            )
        elif temporal_offset_action:
            act_view = self._make_temporal_offset_action_view(
                views[act_view_name],
                take_index=take_index,
                timestamp=timestamp,
                offset=temporal_offset,
            )
        elif shuffled_action:
            act_view = self._make_shuffled_action_view(
                views[act_view_name],
                take_index=take_index,
                same_take=same_take_action,
            )
        else:
            act_view = views[act_view_name]
        reconstructions[name] = {
            "recon": self._decode_path(
                views[obs_view_name],
                act_view,
                private_dropout=0.0 if zero_private else private_dropout,
                zero_private=zero_private,
                zero_action=zero_action,
                action_slot_dropout=action_slot_dropout,
            ),
            "target": views[obs_view_name]["target_patches"],
            "current": views[obs_view_name]["current_patches"],
            "obs_view": obs_view_name,
            "act_view": act_view_name,
            "loss_role": loss_role,
        }
        if "contrast_weight" in act_view:
            reconstructions[name]["contrast_weight"] = act_view["contrast_weight"]

    def forward(
        self,
        batch: Mapping[str, Mapping[str, torch.Tensor]],
        private_dropout: float = 0.0,
        action_slot_dropout: float = 0.0,
        include_action_only: bool = False,
        include_action_shuffle: bool = False,
        include_same_take_action_shuffle: bool = False,
        include_temporal_offset_action: bool = False,
        temporal_offset: int = 4,
        include_action_aware_action: bool = False,
        action_aware_context_weight: float = 0.35,
        include_zero_action: bool = False,
        include_random_code_action: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        views = self.encode_views(batch)
        ego_name, exo_name = self.view_names
        take_index = batch[ego_name].get("take_index")
        timestamp = batch[ego_name].get("timestamp")
        reconstructions: Dict[str, Dict[str, torch.Tensor | str]] = {}
        base_specs = {
            "ego_self": (ego_name, ego_name),
            "exo_self": (exo_name, exo_name),
            "ego_swap": (ego_name, exo_name),
            "exo_swap": (exo_name, ego_name),
        }
        for name, (obs_view_name, act_view_name) in base_specs.items():
            self._add_reconstruction(
                reconstructions,
                name,
                obs_view_name,
                act_view_name,
                views,
                private_dropout=private_dropout,
                action_slot_dropout=action_slot_dropout,
                loss_role="base",
            )
            if include_action_only:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_no_private",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=0.0,
                    zero_private=True,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="action_only",
                )
            if include_action_shuffle:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_action_shuffle",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=private_dropout,
                    shuffled_action=True,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="negative",
                )
                if include_action_only:
                    self._add_reconstruction(
                        reconstructions,
                        f"{name}_no_private_action_shuffle",
                        obs_view_name,
                        act_view_name,
                        views,
                        private_dropout=0.0,
                        zero_private=True,
                        shuffled_action=True,
                        action_slot_dropout=action_slot_dropout,
                        loss_role="negative_no_private",
                    )
            if include_random_code_action:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_random_code_action",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=private_dropout,
                    random_code_action=True,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="random_code_negative",
                )
                if include_action_only:
                    self._add_reconstruction(
                        reconstructions,
                        f"{name}_no_private_random_code_action",
                        obs_view_name,
                        act_view_name,
                        views,
                        private_dropout=0.0,
                        zero_private=True,
                        random_code_action=True,
                        action_slot_dropout=action_slot_dropout,
                        loss_role="random_code_negative_no_private",
                    )
            if include_zero_action:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_zero_action",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=private_dropout,
                    zero_action=True,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="zero_action_negative",
                )
                if include_action_only:
                    self._add_reconstruction(
                        reconstructions,
                        f"{name}_no_private_zero_action",
                        obs_view_name,
                        act_view_name,
                        views,
                        private_dropout=0.0,
                        zero_private=True,
                        zero_action=True,
                        action_slot_dropout=action_slot_dropout,
                        loss_role="zero_action_negative_no_private",
                    )
            if include_same_take_action_shuffle:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_same_take_action_shuffle",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=private_dropout,
                    shuffled_action=True,
                    same_take_action=True,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="same_take_negative",
                    take_index=take_index,
                )
                if include_action_only:
                    self._add_reconstruction(
                        reconstructions,
                        f"{name}_no_private_same_take_action_shuffle",
                        obs_view_name,
                        act_view_name,
                        views,
                        private_dropout=0.0,
                        zero_private=True,
                        shuffled_action=True,
                        same_take_action=True,
                        action_slot_dropout=action_slot_dropout,
                        loss_role="same_take_negative_no_private",
                        take_index=take_index,
                    )
            if include_temporal_offset_action:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_temporal_offset_action",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=private_dropout,
                    temporal_offset_action=True,
                    temporal_offset=temporal_offset,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="temporal_offset_negative",
                    take_index=take_index,
                    timestamp=timestamp,
                )
                if include_action_only:
                    self._add_reconstruction(
                        reconstructions,
                        f"{name}_no_private_temporal_offset_action",
                        obs_view_name,
                        act_view_name,
                        views,
                        private_dropout=0.0,
                        zero_private=True,
                        temporal_offset_action=True,
                        temporal_offset=temporal_offset,
                        action_slot_dropout=action_slot_dropout,
                        loss_role="temporal_offset_negative_no_private",
                        take_index=take_index,
                        timestamp=timestamp,
                    )
            if include_action_aware_action:
                self._add_reconstruction(
                    reconstructions,
                    f"{name}_action_aware_action",
                    obs_view_name,
                    act_view_name,
                    views,
                    private_dropout=private_dropout,
                    action_aware_action=True,
                    action_aware_context_weight=action_aware_context_weight,
                    action_slot_dropout=action_slot_dropout,
                    loss_role="action_aware_negative",
                    take_index=take_index,
                )
                if include_action_only:
                    self._add_reconstruction(
                        reconstructions,
                        f"{name}_no_private_action_aware_action",
                        obs_view_name,
                        act_view_name,
                        views,
                        private_dropout=0.0,
                        zero_private=True,
                        action_aware_action=True,
                        action_aware_context_weight=action_aware_context_weight,
                        action_slot_dropout=action_slot_dropout,
                        loss_role="action_aware_negative_no_private",
                        take_index=take_index,
                    )
        return {
            "views": views,
            "reconstructions": reconstructions,
            "take_index": take_index.to(self.device, non_blocking=True) if take_index is not None else None,
        }

    @torch.inference_mode()
    def encode_shared_action(
        self,
        batch: Mapping[str, Mapping[str, torch.Tensor]],
        view_name: str = "ego",
    ) -> Dict[str, torch.Tensor]:
        views = self.encode_views(batch)
        view = views[view_name]
        return {
            "indices": view["indices"],
            "soft_probs": view["soft_probs"],
            "confidence": view["confidence"],
            "z_q": view["z_q"],
        }

    def config_dict(self) -> dict:
        return {
            "image_channels": 3,
            "model_dim": self.patch_up.out_features,
            "dino_dim": self.patch_up.in_features,
            "latent_dim": self.latent_dim,
            "private_dim": self.private_dim,
            "num_latents": self.num_latents,
            "num_action_slots": self.num_action_slots,
            "num_private_slots": self.num_private_slots,
            "view_names": self.view_names,
        }
