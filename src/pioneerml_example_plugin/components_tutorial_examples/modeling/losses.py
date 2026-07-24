from __future__ import annotations

"""Loss plugin used by the tutorial LightningModule."""

import torch
import torch.nn.functional as F

from pioneerml.integration.pytorch.losses import BaseLoss
from pioneerml.integration.pytorch.losses.factory.registry import REGISTRY as LOSS_REGISTRY


@LOSS_REGISTRY.register("sensor_health_focal_bce")
class SensorHealthFocalBCELoss(BaseLoss):
    """Small BCE variant that emphasizes hard tutorial examples.

    This demonstrates that a loss plugin is just a torch module with a registry
    name. The config controls ``gamma`` and ``positive_weight``.
    """

    def __init__(self, *, gamma: float = 1.5, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight), dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=preds.dtype, device=preds.device)
        pos_weight = self.positive_weight.to(dtype=preds.dtype, device=preds.device)
        bce = F.binary_cross_entropy_with_logits(preds, target, pos_weight=pos_weight, reduction="none")
        prob = torch.sigmoid(preds)
        # Focal weighting down-weights examples the model already predicts well.
        pt = torch.where(target >= 0.5, prob, 1.0 - prob).clamp_min(1e-6)
        return ((1.0 - pt) ** self.gamma * bce).mean()


@LOSS_REGISTRY.register("sensor_health_bce")
class SensorHealthBCELoss(BaseLoss):
    """Plain weighted BCE for the tutorial's linearly learnable sensor task."""

    def __init__(self, *, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight), dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=preds.dtype, device=preds.device)
        pos_weight = self.positive_weight.to(dtype=preds.dtype, device=preds.device)
        return F.binary_cross_entropy_with_logits(preds, target, pos_weight=pos_weight)
