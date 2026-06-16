"""Losses and schedules for FACT tokenizer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FACTLossConfig:
    vq_beta: float = 0.25
    kl_weight: float = 0.1
    balance_weight: float = 0.01
    private_reg_weight: float = 0.001


def scheduled_loss_weights(step: int, total_steps: int) -> Dict[str, float]:
    total_steps = max(int(total_steps), 1)
    progress = min(max(step / total_steps, 0.0), 1.0)
    if progress < 0.2:
        return {
            "self": 1.0,
            "swap": 0.5 * (progress / 0.2),
            "kl": 0.0 if progress < 0.1 else 0.1 * ((progress - 0.1) / 0.1),
        }
    return {"self": 0.2, "swap": 1.0, "kl": 0.1}


def _mean_square(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).mean(dim=1).mean()


def reconstruction_losses(outputs: dict) -> Dict[str, torch.Tensor]:
    losses = {}
    for name, path in outputs["reconstructions"].items():
        losses[name] = _mean_square((path["recon"] - path["target"]) ** 2)
    return losses


def vq_loss_for_view(view: dict, beta: float) -> torch.Tensor:
    codebook_loss = _mean_square((view["emb"].detach() - view["z"]) ** 2)
    commitment_loss = _mean_square((view["emb"] - view["z"].detach()) ** 2)
    return codebook_loss + beta * commitment_loss


def confidence_gated_kl(ego_view: dict, exo_view: dict) -> torch.Tensor:
    teacher = exo_view["soft_probs"].detach().clamp_min(1e-8)
    student = ego_view["soft_probs"].clamp_min(1e-8)
    kl = (teacher * (teacher.log() - student.log())).sum(dim=-1)
    return (kl * exo_view["confidence"].detach()).mean()


def code_usage_balance_loss(views: Dict[str, dict]) -> torch.Tensor:
    probs = torch.cat([view["soft_probs"].reshape(-1, view["soft_probs"].shape[-1]) for view in views.values()], dim=0)
    avg_probs = probs.mean(dim=0)
    uniform = torch.full_like(avg_probs, 1.0 / avg_probs.numel())
    return F.kl_div(avg_probs.clamp_min(1e-8).log(), uniform, reduction="sum")


def private_regularization(views: Dict[str, dict]) -> torch.Tensor:
    return torch.stack([view["r_priv"].pow(2).mean() for view in views.values()]).mean()


def compute_fact_loss(
    outputs: dict,
    step: int,
    total_steps: int,
    config: FACTLossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = scheduled_loss_weights(step, total_steps)
    recon = reconstruction_losses(outputs)
    views = outputs["views"]
    view_names = list(views.keys())
    if len(view_names) != 2:
        raise ValueError(f"FACT v0.1 loss expects two views, got {view_names}")
    ego_view = views[view_names[0]]
    exo_view = views[view_names[1]]

    self_loss = recon["ego_self"] + recon["exo_self"]
    swap_loss = recon["ego_swap"] + recon["exo_swap"]
    vq_loss = vq_loss_for_view(ego_view, config.vq_beta) + vq_loss_for_view(exo_view, config.vq_beta)
    kl_loss = confidence_gated_kl(ego_view, exo_view)
    balance_loss = code_usage_balance_loss(views)
    private_loss = private_regularization(views)

    loss = (
        weights["self"] * self_loss
        + weights["swap"] * swap_loss
        + vq_loss
        + weights["kl"] * config.kl_weight * kl_loss
        + config.balance_weight * balance_loss
        + config.private_reg_weight * private_loss
    )

    logs = {
        "loss": float(loss.detach().cpu()),
        "self_loss": float(self_loss.detach().cpu()),
        "swap_loss": float(swap_loss.detach().cpu()),
        "vq_loss": float(vq_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "balance_loss": float(balance_loss.detach().cpu()),
        "private_reg": float(private_loss.detach().cpu()),
        "weight_self": weights["self"],
        "weight_swap": weights["swap"],
        "weight_kl": weights["kl"] * config.kl_weight,
    }
    for name, value in recon.items():
        logs[f"{name}/feature_mse"] = float(value.detach().cpu())
    return loss, logs
