"""Exercise placeholder for ResNet transfer learning strategies."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """TODO(alumno): implement E4 and E5 with torchvision.models.resnet18.

    The implementation should support a frozen backbone and fine-tuning. It
    must replace the original classification layer with separate gender and age
    heads, and expose the number of unfrozen blocks for ablation studies.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        unfrozen_blocks: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        import torchvision.models as models

        if unfrozen_blocks < 0:
            raise ValueError("unfrozen_blocks no puede ser negativo.")

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in base.parameters():
                param.requires_grad = False

        blocks = [base.layer4, base.layer3, base.layer2, base.layer1]
        for block in blocks[: min(unfrozen_blocks, len(blocks))]:
            for param in block.parameters():
                param.requires_grad = True

        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        gender_logits = self.gender_head(features)
        age_pred = self.age_head(features).squeeze(1)
        return gender_logits, age_pred
