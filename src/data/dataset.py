"""PyTorch Dataset for UTKFace images."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.parser import UTKFaceRecord


class UTKFaceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Load face images and return tensors for both learning tasks."""

    def __init__(
        self,
        records: Sequence[UTKFaceRecord],
        transform: Callable | None = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError("La transformacion debe convertir la imagen a torch.Tensor.")

        gender = torch.tensor(record.gender, dtype=torch.long)
        age = torch.tensor(record.age, dtype=torch.float32)
        return image, gender, age
