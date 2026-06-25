"""Exercise placeholder for the multitask MLP strategy."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskMLP(BaseMultiTaskModel):
    """TODO(alumno): implement E2 using PyTorch.

    Suggested steps:
    1. Flatten the RGB image.
    2. Build a shared fully connected representation.
    3. Add one head with two gender logits and one scalar age head.
    4. Expose dropout as a constructor argument so it can be ablated.
    """

    def __init__(self, dropout: float = 0.4, image_size: int = 224, **kwargs) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")
        if image_size <= 0:
            raise ValueError("image_size debe ser mayor que cero.")

        input_features = 3 * image_size * image_size
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(images)
        gender_logits = self.gender_head(features)
        age_pred = self.age_head(features).squeeze(1)
        return gender_logits, age_pred
