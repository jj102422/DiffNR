"""Spine-mask head and losses for joint SliceFixer reconstruction training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpineMaskHead(nn.Module):
    """Predict one-channel spine logits from the final VAE decoder feature map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(int(in_channels), 1, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.conv.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0e-6,
) -> torch.Tensor:
    """Mean soft Dice loss over samples with a non-empty ground-truth mask.

    Empty targets are intentionally excluded from Dice. They remain supervised by
    BCE in :func:`spine_segmentation_loss`.
    """

    if logits.shape != target.shape:
        raise ValueError(f"Spine logits/target shape mismatch: {logits.shape} vs {target.shape}")
    probabilities = torch.sigmoid(logits.float())
    target = target.float()
    reduce_dims = tuple(range(1, target.ndim))
    target_sum = target.sum(dim=reduce_dims)
    nonempty = target_sum > 0
    if not torch.any(nonempty):
        return logits.float().sum() * 0.0
    intersection = (probabilities * target).sum(dim=reduce_dims)
    probability_sum = probabilities.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (probability_sum + target_sum + smooth)
    return (1.0 - dice[nonempty]).mean()


def spine_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total spine loss plus its unweighted Dice and BCE components."""

    target = target.float()
    dice = soft_dice_loss(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="mean")
    return dice + float(bce_weight) * bce, dice, bce


def linear_warmup_weight(step: int, target_weight: float, warmup_steps: int) -> float:
    """Linearly ramp a loss weight to its target value."""

    if warmup_steps <= 0:
        return float(target_weight)
    progress = min(max(float(step), 0.0) / float(warmup_steps), 1.0)
    return float(target_weight) * progress


def binary_mask_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute batch-mean Dice, precision and recall from mask logits."""

    prediction = torch.sigmoid(logits.float()) >= float(threshold)
    target_bool = target >= 0.5
    reduce_dims = tuple(range(1, target.ndim))
    intersection = (prediction & target_bool).sum(dim=reduce_dims).float()
    predicted = prediction.sum(dim=reduce_dims).float()
    expected = target_bool.sum(dim=reduce_dims).float()
    dice = (2.0 * intersection + eps) / (predicted + expected + eps)
    precision = (intersection + eps) / (predicted + eps)
    recall = (intersection + eps) / (expected + eps)
    return {
        "dice": dice.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
    }
