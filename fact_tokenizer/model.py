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
            videos = batch[view_name]["videos"].to(self.device)
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
    ) -> torch.Tensor:
        current_tokens = self.patch_up(obs_view["current_patches"])
        action_tokens = self.action_up(act_view["z_q"])
        private_tokens = self.private_up(obs_view["r_priv"])
        decoder_input = torch.cat([action_tokens, private_tokens, current_tokens], dim=2)
        return self.decoder(decoder_input, num_patch_tokens=current_tokens.shape[2])

    def forward(self, batch: Mapping[str, Mapping[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        views = self.encode_views(batch)
        ego_name, exo_name = self.view_names
        reconstructions = {
            "ego_self": {
                "recon": self._decode_path(views[ego_name], views[ego_name]),
                "target": views[ego_name]["target_patches"],
                "obs_view": ego_name,
                "act_view": ego_name,
            },
            "exo_self": {
                "recon": self._decode_path(views[exo_name], views[exo_name]),
                "target": views[exo_name]["target_patches"],
                "obs_view": exo_name,
                "act_view": exo_name,
            },
            "ego_swap": {
                "recon": self._decode_path(views[ego_name], views[exo_name]),
                "target": views[ego_name]["target_patches"],
                "obs_view": ego_name,
                "act_view": exo_name,
            },
            "exo_swap": {
                "recon": self._decode_path(views[exo_name], views[ego_name]),
                "target": views[exo_name]["target_patches"],
                "obs_view": exo_name,
                "act_view": ego_name,
            },
        }
        return {"views": views, "reconstructions": reconstructions}

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
