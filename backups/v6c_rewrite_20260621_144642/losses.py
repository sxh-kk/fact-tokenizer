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
    action_only_weight: float = 0.0
    action_contrast_weight: float = 0.0
    no_private_contrast_weight: float = 0.0
    random_code_contrast_weight: float = 0.0
    no_private_random_code_contrast_weight: float = 0.0
    zero_action_contrast_weight: float = 0.0
    no_private_zero_action_contrast_weight: float = 0.0
    no_private_same_take_contrast_weight: float = 0.0
    temporal_offset_contrast_weight: float = 0.0
    no_private_temporal_offset_contrast_weight: float = 0.0
    action_aware_contrast_weight: float = 0.0
    no_private_action_aware_contrast_weight: float = 0.0
    action_aware_context_weight: float = 0.35
    action_contrast_margin: float = 0.01
    action_aux_start_fraction: float = 0.2
    action_aux_ramp_fraction: float = 0.2
    action_consistency_weight: float = 0.0
    assignment_entropy_weight: float = 0.0
    assignment_entropy_target: float = 0.0
    slot_balance_weight: float = 0.0
    hard_usage_balance_weight: float = 0.0
    usage_capacity_weight: float = 0.0
    usage_capacity_max_fraction: float = 0.07
    slot_diversity_weight: float = 0.0
    motion_focus_weight: float = 0.0
    action_only_motion_focus_weight: float = 0.0
    motion_contrast_weight: float = 0.0
    delta_focus_weight: float = 0.0
    action_only_delta_focus_weight: float = 0.0
    delta_contrast_weight: float = 0.0
    no_private_delta_contrast_weight: float = 0.0
    delta_direction_magnitude_weight: float = 0.25
    motion_gated_usage_weight: float = 0.0
    motion_gated_usage_gamma: float = 2.0
    motion_focus_gamma: float = 2.0
    motion_focus_max_weight: float = 6.0
    exo_aux_multiplier: float = 1.0
    teacher_ego_uncertainty_weight: float = 0.0
    teacher_disagreement_weight: float = 0.0
    teacher_base_bias: float = 0.0
    same_take_contrast_weight: float = 0.0
    take_uniform_weight: float = 0.0
    take_slot_uniform_weight: float = 0.0
    take_pair_uniform_weight: float = 0.0


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


def scheduled_aux_weight(step: int, total_steps: int, base_weight: float, start: float, ramp: float) -> float:
    if base_weight <= 0.0:
        return 0.0
    total_steps = max(int(total_steps), 1)
    progress = min(max(step / total_steps, 0.0), 1.0)
    if progress <= start:
        return 0.0
    if ramp <= 0.0:
        return float(base_weight)
    return float(base_weight) * min((progress - start) / ramp, 1.0)


def _mean_square(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).mean(dim=1).mean()


def _per_sample_mean_square(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).mean(dim=1)


def _path_aux_weight(name: str, exo_aux_multiplier: float) -> float:
    if name.startswith("exo_"):
        return float(exo_aux_multiplier)
    return 1.0


def reconstruction_losses(outputs: dict) -> Dict[str, torch.Tensor]:
    losses = {}
    for name, path in outputs["reconstructions"].items():
        losses[name] = _mean_square((path["recon"] - path["target"]) ** 2)
    return losses


def summed_reconstruction_loss(recon: Dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    available = [recon[name] for name in names if name in recon]
    if not available:
        reference = next(iter(recon.values()))
        return reference.new_zeros(())
    return torch.stack(available).sum()


def weighted_summed_reconstruction_loss(
    recon: Dict[str, torch.Tensor],
    names: list[str],
    exo_aux_multiplier: float,
) -> torch.Tensor:
    available = [recon[name] * _path_aux_weight(name, exo_aux_multiplier) for name in names if name in recon]
    if not available:
        reference = next(iter(recon.values()))
        return reference.new_zeros(())
    return torch.stack(available).sum()


def vq_loss_for_view(view: dict, beta: float) -> torch.Tensor:
    codebook_loss = _mean_square((view["emb"].detach() - view["z"]) ** 2)
    commitment_loss = _mean_square((view["emb"] - view["z"].detach()) ** 2)
    return codebook_loss + beta * commitment_loss


def confidence_gated_kl(
    ego_view: dict,
    exo_view: dict,
    ego_uncertainty_weight: float = 0.0,
    disagreement_weight: float = 0.0,
    base_bias: float = 0.0,
) -> torch.Tensor:
    teacher = exo_view["soft_probs"].detach().clamp_min(1e-8)
    student = ego_view["soft_probs"].clamp_min(1e-8)
    kl = (teacher * (teacher.log() - student.log())).sum(dim=-1)
    teacher_weight = exo_view["confidence"].detach()
    if ego_uncertainty_weight or disagreement_weight or base_bias:
        ego_confidence = ego_view["confidence"].detach()
        ego_uncertainty = 1.0 - ego_confidence
        student_detached = student.detach()
        midpoint = (0.5 * (student_detached + teacher)).clamp_min(1e-8)
        ego_kl = (student_detached * (student_detached.log() - midpoint.log())).sum(dim=-1)
        exo_kl = (teacher * (teacher.log() - midpoint.log())).sum(dim=-1)
        disagreement = (0.5 * (ego_kl + exo_kl) / torch.log(torch.tensor(float(teacher.shape[-1]), device=teacher.device))).sqrt()
        corrective_gate = torch.sigmoid(
            float(base_bias)
            + float(ego_uncertainty_weight) * ego_uncertainty
            + float(disagreement_weight) * disagreement
        )
        teacher_weight = teacher_weight * corrective_gate
    return (kl * teacher_weight).mean()


def symmetric_action_consistency(ego_view: dict, exo_view: dict) -> torch.Tensor:
    ego_probs = ego_view["soft_probs"].clamp_min(1e-8)
    exo_probs = exo_view["soft_probs"].clamp_min(1e-8)
    midpoint = (0.5 * (ego_probs + exo_probs)).clamp_min(1e-8)
    ego_kl = (ego_probs * (ego_probs.log() - midpoint.log())).sum(dim=-1)
    exo_kl = (exo_probs * (exo_probs.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (ego_kl + exo_kl).mean()


def action_top1_agreement(ego_view: dict, exo_view: dict) -> torch.Tensor:
    return (ego_view["indices"] == exo_view["indices"]).float().mean()


def code_usage_balance_loss(views: Dict[str, dict]) -> torch.Tensor:
    probs = torch.cat([view["soft_probs"].reshape(-1, view["soft_probs"].shape[-1]) for view in views.values()], dim=0)
    avg_probs = probs.mean(dim=0)
    uniform = torch.full_like(avg_probs, 1.0 / avg_probs.numel())
    return F.kl_div(avg_probs.clamp_min(1e-8).log(), uniform, reduction="sum")


def assignment_entropy_loss(views: Dict[str, dict], target: float = 0.0) -> torch.Tensor:
    entropies = []
    for view in views.values():
        probs = view["soft_probs"].clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        normalized = entropy / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device))
        if target > 0.0:
            normalized = F.relu(normalized - float(target))
        entropies.append(normalized)
    return torch.stack([entropy.mean() for entropy in entropies]).mean()


def assignment_entropy_mean(views: Dict[str, dict]) -> torch.Tensor:
    entropies = []
    for view in views.values():
        probs = view["soft_probs"].clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        entropies.append(entropy / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device)))
    return torch.stack([entropy.mean() for entropy in entropies]).mean()


def slot_code_usage_balance_loss(views: Dict[str, dict]) -> torch.Tensor:
    penalties = []
    for view in views.values():
        probs = view["soft_probs"]
        uniform = torch.full((probs.shape[-1],), 1.0 / probs.shape[-1], device=probs.device, dtype=probs.dtype)
        for slot in range(probs.shape[-2]):
            slot_probs = probs[..., slot, :].reshape(-1, probs.shape[-1]).mean(dim=0)
            penalties.append(F.kl_div(slot_probs.clamp_min(1e-8).log(), uniform, reduction="sum"))
    if not penalties:
        reference = next(iter(views.values()))["soft_probs"]
        return reference.new_zeros(())
    return torch.stack(penalties).mean()


def _straight_through_one_hot(view: dict) -> torch.Tensor:
    probs = view["soft_probs"]
    hard = F.one_hot(view["indices"], num_classes=probs.shape[-1]).to(dtype=probs.dtype, device=probs.device)
    return hard - probs.detach() + probs


def hard_code_usage_balance_loss(views: Dict[str, dict]) -> torch.Tensor:
    assignments = torch.cat([_straight_through_one_hot(view).reshape(-1, view["soft_probs"].shape[-1]) for view in views.values()], dim=0)
    avg_probs = assignments.mean(dim=0)
    uniform = torch.full_like(avg_probs, 1.0 / avg_probs.numel())
    return F.kl_div(avg_probs.clamp_min(1e-8).log(), uniform, reduction="sum")


def code_usage_capacity_loss(views: Dict[str, dict], max_fraction: float) -> torch.Tensor:
    assignments = torch.cat(
        [_straight_through_one_hot(view).reshape(-1, view["soft_probs"].shape[-1]) for view in views.values()],
        dim=0,
    )
    avg_probs = assignments.mean(dim=0)
    excess = F.relu(avg_probs - float(max_fraction))
    return excess.pow(2).sum() * avg_probs.numel()


def slot_diversity_loss(views: Dict[str, dict]) -> torch.Tensor:
    penalties = []
    for view in views.values():
        assignments = _straight_through_one_hot(view)
        num_slots = assignments.shape[-2]
        if num_slots < 2:
            continue
        similarity = torch.matmul(assignments, assignments.transpose(-1, -2))
        off_diagonal = similarity - torch.eye(num_slots, device=similarity.device, dtype=similarity.dtype)
        penalties.append(off_diagonal.sum(dim=(-2, -1)) / (num_slots * (num_slots - 1)))
    if not penalties:
        reference = next(iter(views.values()))["soft_probs"]
        return reference.new_zeros(())
    return torch.stack([penalty.mean() for penalty in penalties]).mean()


def take_uniformity_loss(views: Dict[str, dict], take_index: torch.Tensor | None) -> torch.Tensor:
    reference = next(iter(views.values()))["soft_probs"]
    if take_index is None or take_index.numel() <= 1:
        return reference.new_zeros(())
    take_index = take_index.to(reference.device)
    unique_takes = torch.unique(take_index)
    penalties = []
    for view in views.values():
        probs = view["soft_probs"].reshape(view["soft_probs"].shape[0], -1, view["soft_probs"].shape[-1])
        batch_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0).detach().clamp_min(1e-8)
        for take in unique_takes:
            mask = take_index == take
            if int(mask.sum().item()) <= 1:
                continue
            take_probs = probs[mask].reshape(-1, probs.shape[-1]).mean(dim=0)
            penalties.append(F.kl_div(take_probs.clamp_min(1e-8).log(), batch_probs, reduction="sum"))
    if not penalties:
        return reference.new_zeros(())
    return torch.stack(penalties).mean()


def take_slot_uniformity_loss(views: Dict[str, dict], take_index: torch.Tensor | None) -> torch.Tensor:
    reference = next(iter(views.values()))["soft_probs"]
    if take_index is None or take_index.numel() <= 1:
        return reference.new_zeros(())
    take_index = take_index.to(reference.device)
    unique_takes = torch.unique(take_index)
    penalties = []
    for view in views.values():
        probs = view["soft_probs"]
        batch_probs = probs.mean(dim=0).detach().clamp_min(1e-8)
        for take in unique_takes:
            mask = take_index == take
            if int(mask.sum().item()) <= 1:
                continue
            take_probs = probs[mask].mean(dim=0)
            kl = F.kl_div(take_probs.clamp_min(1e-8).log(), batch_probs, reduction="none").sum(dim=-1)
            penalties.append(kl.mean())
    if not penalties:
        return reference.new_zeros(())
    return torch.stack(penalties).mean()


def take_pair_uniformity_loss(views: Dict[str, dict], take_index: torch.Tensor | None) -> torch.Tensor:
    reference = next(iter(views.values()))["soft_probs"]
    if take_index is None or take_index.numel() <= 1:
        return reference.new_zeros(())
    take_index = take_index.to(reference.device)
    unique_takes = torch.unique(take_index)
    penalties = []
    for view in views.values():
        probs = view["soft_probs"].reshape(view["soft_probs"].shape[0], -1, view["soft_probs"].shape[-1])
        num_positions = probs.shape[1]
        if num_positions < 2:
            continue
        for left in range(num_positions):
            left_probs = probs[:, left]
            for right in range(left + 1, num_positions):
                right_probs = probs[:, right]
                batch_pair = torch.einsum("bk,bl->bkl", left_probs, right_probs).mean(dim=0).detach().clamp_min(1e-8)
                for take in unique_takes:
                    mask = take_index == take
                    if int(mask.sum().item()) <= 1:
                        continue
                    take_pair = torch.einsum("bk,bl->bkl", left_probs[mask], right_probs[mask]).mean(dim=0)
                    penalties.append(F.kl_div(take_pair.clamp_min(1e-8).log(), batch_pair, reduction="sum"))
    if not penalties:
        return reference.new_zeros(())
    return torch.stack(penalties).mean()


def private_regularization(views: Dict[str, dict]) -> torch.Tensor:
    return torch.stack([view["r_priv"].pow(2).mean() for view in views.values()]).mean()


def _motion_weights(path: dict, gamma: float, max_weight: float) -> torch.Tensor | None:
    current = path.get("current")
    if current is None or gamma <= 0.0:
        return None
    motion = ((path["target"] - current) ** 2).mean(dim=-1, keepdim=True)
    flat = motion.reshape(motion.shape[0], -1)
    mean = flat.mean(dim=1).reshape(-1, *([1] * (motion.ndim - 1))).clamp_min(1e-8)
    weights = 1.0 + float(gamma) * (motion / mean)
    if max_weight > 0.0:
        weights = weights.clamp(max=float(max_weight))
    weights = weights / weights.reshape(weights.shape[0], -1).mean(dim=1).reshape(
        -1,
        *([1] * (weights.ndim - 1)),
    ).clamp_min(1e-8)
    return weights.detach()


def _per_sample_motion_focused_square(path: dict, gamma: float, max_weight: float) -> torch.Tensor:
    error = (path["recon"] - path["target"]) ** 2
    weights = _motion_weights(path, gamma=gamma, max_weight=max_weight)
    if weights is not None:
        error = error * weights
    return _per_sample_mean_square(error)


def motion_focused_reconstruction_losses(outputs: dict, gamma: float, max_weight: float) -> Dict[str, torch.Tensor]:
    losses = {}
    for name, path in outputs["reconstructions"].items():
        losses[name] = _per_sample_motion_focused_square(path, gamma=gamma, max_weight=max_weight).mean()
    return losses


def _motion_score_from_path(path: dict) -> torch.Tensor | None:
    current = path.get("current")
    if current is None:
        return None
    return (path["target"] - current).pow(2).reshape(path["target"].shape[0], -1).mean(dim=1)


def _per_sample_delta_square(path: dict, magnitude_weight: float = 0.25) -> torch.Tensor:
    """Motion-relative transition loss.

    A plain MSE on (recon-current)-(target-current) is algebraically identical
    to future-feature MSE. This loss instead compares transition direction and
    relative magnitude, with high-motion patches carrying more weight.
    """
    current = path.get("current")
    if current is None:
        return _per_sample_mean_square((path["recon"] - path["target"]) ** 2)
    eps = 1e-6
    recon_delta = path["recon"] - current
    target_delta = path["target"] - current
    target_norm = target_delta.norm(dim=-1).clamp_min(eps)
    recon_norm = recon_delta.norm(dim=-1).clamp_min(eps)
    cosine_loss = 1.0 - F.cosine_similarity(recon_delta, target_delta, dim=-1, eps=eps)
    motion_weight = target_norm.detach()
    motion_weight = motion_weight / motion_weight.reshape(motion_weight.shape[0], -1).mean(dim=1).reshape(
        -1,
        *([1] * (motion_weight.ndim - 1)),
    ).clamp_min(eps)
    magnitude_loss = ((recon_norm - target_norm) / target_norm.detach().clamp_min(0.05)).pow(2)
    loss = cosine_loss.clamp(0.0, 2.0) + float(magnitude_weight) * magnitude_loss
    return (loss * motion_weight).reshape(loss.shape[0], -1).mean(dim=1)


def delta_reconstruction_losses(outputs: dict) -> Dict[str, torch.Tensor]:
    losses = {}
    for name, path in outputs["reconstructions"].items():
        losses[name] = _per_sample_delta_square(
            path,
            magnitude_weight=outputs.get("delta_direction_magnitude_weight", 0.25),
        ).mean()
    return losses


def motion_gated_hard_usage_balance_loss(outputs: dict, gamma: float = 2.0) -> torch.Tensor:
    views = outputs["views"]
    base_paths = [
        outputs["reconstructions"][name]
        for name in ("ego_self", "exo_self")
        if name in outputs["reconstructions"]
    ]
    scores = [_motion_score_from_path(path) for path in base_paths]
    scores = [score for score in scores if score is not None]
    if not scores:
        reference = next(iter(views.values()))["soft_probs"]
        return reference.new_zeros(())
    motion = torch.stack(scores).mean(dim=0)
    motion = motion / motion.mean().clamp_min(1e-8)
    base_sample_weight = (1.0 + float(gamma) * motion).detach()
    base_sample_weight = base_sample_weight / base_sample_weight.mean().clamp_min(1e-8)
    weighted_assignments = []
    for view in views.values():
        assignments = _straight_through_one_hot(view)
        sample_weight = base_sample_weight
        while sample_weight.ndim < assignments.ndim:
            sample_weight = sample_weight.unsqueeze(-1)
        weighted_assignments.append((assignments * sample_weight).reshape(-1, assignments.shape[-1]))
    avg_probs = torch.cat(weighted_assignments, dim=0).mean(dim=0)
    avg_probs = avg_probs / avg_probs.sum().clamp_min(1e-8)
    uniform = torch.full_like(avg_probs, 1.0 / avg_probs.numel())
    return F.kl_div(avg_probs.clamp_min(1e-8).log(), uniform, reduction="sum")


def action_contrast_loss(
    outputs: dict,
    positive_names: list[str],
    suffix: str,
    margin: float,
    exo_aux_multiplier: float = 1.0,
    motion_focused: bool = False,
    delta_focused: bool = False,
    motion_focus_gamma: float = 2.0,
    motion_focus_max_weight: float = 6.0,
) -> torch.Tensor:
    penalties = []
    for name in positive_names:
        if name not in outputs["reconstructions"]:
            continue
        negative_name = f"{name}{suffix}"
        if negative_name not in outputs["reconstructions"]:
            continue
        path = outputs["reconstructions"][name]
        negative = outputs["reconstructions"][negative_name]
        if delta_focused:
            pos_mse = _per_sample_delta_square(path)
            neg_mse = _per_sample_delta_square(negative)
        elif motion_focused:
            pos_mse = _per_sample_motion_focused_square(path, motion_focus_gamma, motion_focus_max_weight)
            neg_mse = _per_sample_motion_focused_square(negative, motion_focus_gamma, motion_focus_max_weight)
        else:
            pos_mse = _per_sample_mean_square((path["recon"] - path["target"]) ** 2)
            neg_mse = _per_sample_mean_square((negative["recon"] - negative["target"]) ** 2)
        path_weight = _path_aux_weight(name, exo_aux_multiplier)
        penalty = F.relu(margin + pos_mse - neg_mse)
        contrast_weight = negative.get("contrast_weight")
        if contrast_weight is not None:
            contrast_weight = contrast_weight.to(device=penalty.device, dtype=penalty.dtype)
            normalizer = contrast_weight.mean().clamp_min(1e-6)
            penalty = penalty * (contrast_weight / normalizer)
        penalties.append(path_weight * penalty.mean())
    if not penalties:
        reference = next(iter(outputs["views"].values()))["r_priv"]
        return reference.new_zeros(())
    return torch.stack(penalties).mean()


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
    outputs["delta_direction_magnitude_weight"] = config.delta_direction_magnitude_weight

    self_loss = recon["ego_self"] + recon["exo_self"]
    swap_loss = recon["ego_swap"] + recon["exo_swap"]
    vq_loss = vq_loss_for_view(ego_view, config.vq_beta) + vq_loss_for_view(exo_view, config.vq_beta)
    kl_loss = confidence_gated_kl(
        ego_view,
        exo_view,
        ego_uncertainty_weight=config.teacher_ego_uncertainty_weight,
        disagreement_weight=config.teacher_disagreement_weight,
        base_bias=config.teacher_base_bias,
    )
    balance_loss = code_usage_balance_loss(views)
    entropy_loss = assignment_entropy_loss(views, target=config.assignment_entropy_target)
    entropy_mean = assignment_entropy_mean(views)
    slot_balance_loss = slot_code_usage_balance_loss(views)
    hard_balance_loss = hard_code_usage_balance_loss(views)
    capacity_loss = code_usage_capacity_loss(views, config.usage_capacity_max_fraction)
    motion_gated_usage_loss = motion_gated_hard_usage_balance_loss(
        outputs,
        gamma=config.motion_gated_usage_gamma,
    )
    slot_div_loss = slot_diversity_loss(views)
    take_uniform_loss = take_uniformity_loss(views, outputs.get("take_index"))
    take_slot_uniform_loss = take_slot_uniformity_loss(views, outputs.get("take_index"))
    take_pair_uniform_loss = take_pair_uniformity_loss(views, outputs.get("take_index"))
    consistency_loss = symmetric_action_consistency(ego_view, exo_view)
    top1_agreement = action_top1_agreement(ego_view, exo_view)
    private_loss = private_regularization(views)
    focus_recon = motion_focused_reconstruction_losses(
        outputs,
        gamma=config.motion_focus_gamma,
        max_weight=config.motion_focus_max_weight,
    )
    delta_recon = delta_reconstruction_losses(outputs)
    action_only_self_loss = weighted_summed_reconstruction_loss(
        recon,
        ["ego_self_no_private", "exo_self_no_private"],
        config.exo_aux_multiplier,
    )
    action_only_swap_loss = weighted_summed_reconstruction_loss(
        recon,
        ["ego_swap_no_private", "exo_swap_no_private"],
        config.exo_aux_multiplier,
    )
    action_only_loss = action_only_self_loss + action_only_swap_loss
    base_names = ["ego_self", "exo_self", "ego_swap", "exo_swap"]
    no_private_names = [
        "ego_self_no_private",
        "exo_self_no_private",
        "ego_swap_no_private",
        "exo_swap_no_private",
    ]
    contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    no_private_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    random_code_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_random_code_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    no_private_random_code_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_random_code_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    zero_action_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_zero_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    no_private_zero_action_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_zero_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    motion_focus_loss = weighted_summed_reconstruction_loss(
        focus_recon,
        base_names,
        config.exo_aux_multiplier,
    )
    action_only_motion_focus_loss = weighted_summed_reconstruction_loss(
        focus_recon,
        no_private_names,
        config.exo_aux_multiplier,
    )
    motion_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
        motion_focused=True,
        motion_focus_gamma=config.motion_focus_gamma,
        motion_focus_max_weight=config.motion_focus_max_weight,
    )
    delta_focus_loss = weighted_summed_reconstruction_loss(
        delta_recon,
        base_names,
        config.exo_aux_multiplier,
    )
    action_only_delta_focus_loss = weighted_summed_reconstruction_loss(
        delta_recon,
        no_private_names,
        config.exo_aux_multiplier,
    )
    delta_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
        delta_focused=True,
    )
    no_private_delta_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
        delta_focused=True,
    )
    same_take_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_same_take_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    no_private_same_take_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_same_take_action_shuffle",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    temporal_offset_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_temporal_offset_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    no_private_temporal_offset_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_temporal_offset_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
    )
    action_aware_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=base_names,
        suffix="_action_aware_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
        delta_focused=True,
    )
    no_private_action_aware_contrast_loss = action_contrast_loss(
        outputs,
        positive_names=no_private_names,
        suffix="_action_aware_action",
        margin=config.action_contrast_margin,
        exo_aux_multiplier=config.exo_aux_multiplier,
        delta_focused=True,
    )

    action_only_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.action_only_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.action_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    random_code_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.random_code_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_random_code_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_random_code_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    zero_action_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.zero_action_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_zero_action_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_zero_action_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    consistency_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.action_consistency_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    entropy_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.assignment_entropy_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    slot_balance_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.slot_balance_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    hard_balance_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.hard_usage_balance_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    capacity_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.usage_capacity_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    slot_diversity_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.slot_diversity_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    motion_focus_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.motion_focus_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    action_only_motion_focus_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.action_only_motion_focus_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    motion_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.motion_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    delta_focus_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.delta_focus_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    action_only_delta_focus_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.action_only_delta_focus_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    delta_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.delta_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_delta_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_delta_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    same_take_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.same_take_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_same_take_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_same_take_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    temporal_offset_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.temporal_offset_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_temporal_offset_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_temporal_offset_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    action_aware_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.action_aware_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    no_private_action_aware_contrast_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.no_private_action_aware_contrast_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    take_uniform_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.take_uniform_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    take_slot_uniform_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.take_slot_uniform_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    take_pair_uniform_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.take_pair_uniform_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )
    motion_gated_usage_weight = scheduled_aux_weight(
        step,
        total_steps,
        config.motion_gated_usage_weight,
        config.action_aux_start_fraction,
        config.action_aux_ramp_fraction,
    )

    loss = (
        weights["self"] * self_loss
        + weights["swap"] * swap_loss
        + action_only_weight * action_only_loss
        + contrast_weight * contrast_loss
        + no_private_contrast_weight * no_private_contrast_loss
        + random_code_contrast_weight * random_code_contrast_loss
        + no_private_random_code_contrast_weight * no_private_random_code_contrast_loss
        + zero_action_contrast_weight * zero_action_contrast_loss
        + no_private_zero_action_contrast_weight * no_private_zero_action_contrast_loss
        + consistency_weight * consistency_loss
        + entropy_weight * entropy_loss
        + slot_balance_weight * slot_balance_loss
        + hard_balance_weight * hard_balance_loss
        + capacity_weight * capacity_loss
        + slot_diversity_weight * slot_div_loss
        + motion_focus_weight * motion_focus_loss
        + action_only_motion_focus_weight * action_only_motion_focus_loss
        + motion_contrast_weight * motion_contrast_loss
        + delta_focus_weight * delta_focus_loss
        + action_only_delta_focus_weight * action_only_delta_focus_loss
        + delta_contrast_weight * delta_contrast_loss
        + no_private_delta_contrast_weight * no_private_delta_contrast_loss
        + same_take_contrast_weight * same_take_contrast_loss
        + no_private_same_take_contrast_weight * no_private_same_take_contrast_loss
        + temporal_offset_contrast_weight * temporal_offset_contrast_loss
        + no_private_temporal_offset_contrast_weight * no_private_temporal_offset_contrast_loss
        + action_aware_contrast_weight * action_aware_contrast_loss
        + no_private_action_aware_contrast_weight * no_private_action_aware_contrast_loss
        + take_uniform_weight * take_uniform_loss
        + take_slot_uniform_weight * take_slot_uniform_loss
        + take_pair_uniform_weight * take_pair_uniform_loss
        + motion_gated_usage_weight * motion_gated_usage_loss
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
        "slot_balance_loss": float(slot_balance_loss.detach().cpu()),
        "hard_usage_balance_loss": float(hard_balance_loss.detach().cpu()),
        "usage_capacity_loss": float(capacity_loss.detach().cpu()),
        "motion_gated_usage_loss": float(motion_gated_usage_loss.detach().cpu()),
        "slot_diversity_loss": float(slot_div_loss.detach().cpu()),
        "take_uniformity_loss": float(take_uniform_loss.detach().cpu()),
        "take_slot_uniformity_loss": float(take_slot_uniform_loss.detach().cpu()),
        "take_pair_uniformity_loss": float(take_pair_uniform_loss.detach().cpu()),
        "assignment_entropy_loss": float(entropy_loss.detach().cpu()),
        "assignment_entropy_mean": float(entropy_mean.detach().cpu()),
        "action_consistency_loss": float(consistency_loss.detach().cpu()),
        "action_top1_agreement": float(top1_agreement.detach().cpu()),
        "private_reg": float(private_loss.detach().cpu()),
        "action_only_loss": float(action_only_loss.detach().cpu()),
        "action_only_self_loss": float(action_only_self_loss.detach().cpu()),
        "action_only_swap_loss": float(action_only_swap_loss.detach().cpu()),
        "action_contrast_loss": float(contrast_loss.detach().cpu()),
        "no_private_contrast_loss": float(no_private_contrast_loss.detach().cpu()),
        "random_code_contrast_loss": float(random_code_contrast_loss.detach().cpu()),
        "no_private_random_code_contrast_loss": float(no_private_random_code_contrast_loss.detach().cpu()),
        "zero_action_contrast_loss": float(zero_action_contrast_loss.detach().cpu()),
        "no_private_zero_action_contrast_loss": float(no_private_zero_action_contrast_loss.detach().cpu()),
        "motion_focus_loss": float(motion_focus_loss.detach().cpu()),
        "action_only_motion_focus_loss": float(action_only_motion_focus_loss.detach().cpu()),
        "motion_contrast_loss": float(motion_contrast_loss.detach().cpu()),
        "delta_focus_loss": float(delta_focus_loss.detach().cpu()),
        "action_only_delta_focus_loss": float(action_only_delta_focus_loss.detach().cpu()),
        "delta_contrast_loss": float(delta_contrast_loss.detach().cpu()),
        "no_private_delta_contrast_loss": float(no_private_delta_contrast_loss.detach().cpu()),
        "same_take_contrast_loss": float(same_take_contrast_loss.detach().cpu()),
        "no_private_same_take_contrast_loss": float(no_private_same_take_contrast_loss.detach().cpu()),
        "temporal_offset_contrast_loss": float(temporal_offset_contrast_loss.detach().cpu()),
        "no_private_temporal_offset_contrast_loss": float(no_private_temporal_offset_contrast_loss.detach().cpu()),
        "action_aware_contrast_loss": float(action_aware_contrast_loss.detach().cpu()),
        "no_private_action_aware_contrast_loss": float(no_private_action_aware_contrast_loss.detach().cpu()),
        "weight_self": weights["self"],
        "weight_swap": weights["swap"],
        "weight_kl": weights["kl"] * config.kl_weight,
        "weight_action_only": action_only_weight,
        "weight_action_contrast": contrast_weight,
        "weight_no_private_contrast": no_private_contrast_weight,
        "weight_random_code_contrast": random_code_contrast_weight,
        "weight_no_private_random_code_contrast": no_private_random_code_contrast_weight,
        "weight_zero_action_contrast": zero_action_contrast_weight,
        "weight_no_private_zero_action_contrast": no_private_zero_action_contrast_weight,
        "weight_action_consistency": consistency_weight,
        "weight_assignment_entropy": entropy_weight,
        "weight_slot_balance": slot_balance_weight,
        "weight_hard_usage_balance": hard_balance_weight,
        "weight_usage_capacity": capacity_weight,
        "weight_slot_diversity": slot_diversity_weight,
        "weight_motion_focus": motion_focus_weight,
        "weight_action_only_motion_focus": action_only_motion_focus_weight,
        "weight_motion_contrast": motion_contrast_weight,
        "weight_delta_focus": delta_focus_weight,
        "weight_action_only_delta_focus": action_only_delta_focus_weight,
        "weight_delta_contrast": delta_contrast_weight,
        "weight_no_private_delta_contrast": no_private_delta_contrast_weight,
        "weight_motion_gated_usage": motion_gated_usage_weight,
        "weight_same_take_contrast": same_take_contrast_weight,
        "weight_no_private_same_take_contrast": no_private_same_take_contrast_weight,
        "weight_temporal_offset_contrast": temporal_offset_contrast_weight,
        "weight_no_private_temporal_offset_contrast": no_private_temporal_offset_contrast_weight,
        "weight_action_aware_contrast": action_aware_contrast_weight,
        "weight_no_private_action_aware_contrast": no_private_action_aware_contrast_weight,
        "weight_take_uniform": take_uniform_weight,
        "weight_take_slot_uniform": take_slot_uniform_weight,
        "weight_take_pair_uniform": take_pair_uniform_weight,
    }
    for name, value in recon.items():
        logs[f"{name}/feature_mse"] = float(value.detach().cpu())
    return loss, logs
