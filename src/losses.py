from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    dice = num / den
    return 1.0 - dice.mean()


def focal_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    modulating = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * modulating * bce).mean()


def build_loss(loss_name: str) -> nn.Module:
    loss_name = loss_name.lower()
    bce = nn.BCEWithLogitsLoss()

    if loss_name == "bce":
        return bce

    class BCEDice(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bce(logits, targets) + dice_loss_from_logits(logits, targets)

    class BCEFocal(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bce(logits, targets) + focal_loss_from_logits(logits, targets)

    if loss_name == "bce_dice":
        return BCEDice()
    if loss_name == "bce_focal":
        return BCEFocal()
    raise ValueError(f"Unsupported loss: {loss_name}")
