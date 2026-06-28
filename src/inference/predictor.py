"""Load a trained multitask checkpoint and run PyTorch inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.transforms import TransformFactory
from src.models.cnn import MultiTaskCNN
from src.models.resnet_todo import MultiTaskResNet


@dataclass(frozen=True)
class Prediction:
    """Human-readable model output for one detected face."""

    gender_index: int
    gender_label: str
    gender_confidence: float
    estimated_age: float


class CNNPredictor:
    """Apply exactly the same deterministic preprocessing used during testing."""

    GENDER_LABELS = {
        0: "Masculino",
        1: "Femenino",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.image_size = image_size
        self.device = device
        self.transform = TransformFactory.inference(image_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> "CNNPredictor":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint {path}. Configure CNN_CHECKPOINT "
                "antes de usar Streamlit."
            )

        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
        model_name = checkpoint.get("model_name")
        model_kwargs = checkpoint.get("model_kwargs", {})
        if model_name == "cnn":
            model = MultiTaskCNN(**model_kwargs)
        elif model_name in {"resnet_frozen", "resnet_finetuning"}:
            model = MultiTaskResNet(
                **model_kwargs,
                pretrained=False,
            )
        else:
            raise ValueError(
                f"El checkpoint usa un modelo no soportado: {model_name!r}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        image_size = int(checkpoint.get("image_size", 224))
        return cls(model=model, image_size=image_size, device=device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        gender_logits, age_prediction = self.model(image_tensor)
        probabilities = torch.softmax(gender_logits, dim=1)
        confidence, gender_index = probabilities.max(dim=1)
        index = int(gender_index.item())
        return Prediction(
            gender_index=index,
            gender_label=self.GENDER_LABELS.get(index, str(index)),
            gender_confidence=float(confidence.item()),
            estimated_age=float(age_prediction.item()),
        )
