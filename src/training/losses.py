"""Loss functions for simultaneous gender classification and age regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossOutput:
    """Keep each task loss visible for curves and interpretation."""

    total: torch.Tensor
    gender: torch.Tensor
    age: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Combine classification and regression losses with a configurable weight."""

    def __init__(self, lambda_age: float = 0.01) -> None:
        super().__init__()
        if lambda_age < 0:
            raise ValueError("lambda_age no puede ser negativo.")
        self.lambda_age = lambda_age
        self.gender_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        gender_logits: torch.Tensor,
        age_predictions: torch.Tensor,
        gender_targets: torch.Tensor,
        age_targets: torch.Tensor,
    ) -> LossOutput:
        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        age_loss = self.age_criterion(age_predictions, age_targets)
        total_loss = gender_loss + self.lambda_age * age_loss
        return LossOutput(total=total_loss, gender=gender_loss, age=age_loss)
