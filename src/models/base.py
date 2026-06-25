"""Base class for models with gender and age outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseMultiTaskModel(nn.Module, ABC):
    """Common interface for all neural multitask strategies."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gender logits [B, 2] and age predictions [B]."""

    def count_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
