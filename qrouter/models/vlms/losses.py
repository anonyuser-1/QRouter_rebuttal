from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_to_patch_target(
    ground_truth_mask: torch.Tensor,
    patch_hw: tuple[int, int],
) -> torch.Tensor:
    """Implements Eq. (41) with area averaging on the patch grid.

    Args:
        ground_truth_mask: [B,H,W] or [B,1,H,W], values in [0,1].
        patch_hw: output patch-grid height and width.
    Returns:
        Patch-level target prior with shape [B,N].
    """
    if ground_truth_mask.ndim == 3:
        ground_truth_mask = ground_truth_mask.unsqueeze(1)
    if ground_truth_mask.ndim != 4 or ground_truth_mask.shape[1] != 1:
        raise ValueError("ground_truth_mask must have shape [B,H,W] or [B,1,H,W].")
    target = F.adaptive_avg_pool2d(ground_truth_mask.float(), patch_hw)
    return target.flatten(1).clamp(0.0, 1.0)


def dice_loss(
    predicted_prior: torch.Tensor,
    target_prior: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    predicted_prior = predicted_prior.float()
    target_prior = target_prior.float()
    intersection = (predicted_prior * target_prior).sum(dim=-1)
    denominator = predicted_prior.sum(dim=-1) + target_prior.sum(dim=-1)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def cis_patch_prior_loss(
    predicted_prior: torch.Tensor,
    target_prior: torch.Tensor,
    dice_weight: float = 1.0,
    valid_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Eq. (42): BCE(a,a*) + lambda_Dice Dice(a,a*)."""
    if predicted_prior.shape != target_prior.shape:
        raise ValueError(
            f"Prior shapes differ: predicted={tuple(predicted_prior.shape)}, " f"target={tuple(target_prior.shape)}."
        )
    predicted_prior = predicted_prior.float().clamp(1e-6, 1.0 - 1e-6)
    target_prior = target_prior.float().clamp(0.0, 1.0)
    bce_per_sample = F.binary_cross_entropy(
        predicted_prior,
        target_prior,
        reduction="none",
    ).mean(dim=-1)
    dice_per_sample = dice_loss(predicted_prior, target_prior)

    if valid_samples is None:
        valid = torch.ones_like(bce_per_sample)
    else:
        valid = valid_samples.to(device=bce_per_sample.device, dtype=bce_per_sample.dtype)
    denominator = valid.sum().clamp_min(1.0)
    bce = (bce_per_sample * valid).sum() / denominator
    dice = (dice_per_sample * valid).sum() / denominator
    total = bce + float(dice_weight) * dice
    return total, {"cis_bce": bce.detach(), "cis_dice": dice.detach()}
