import pytest
import torch

from src.models.cnn import MultiTaskCNN


def test_cnn_returns_one_output_per_task() -> None:
    model = MultiTaskCNN(dropout=0.4)
    images = torch.randn(3, 3, 64, 64)

    gender_logits, age_predictions = model(images)

    assert gender_logits.shape == (3, 2)
    assert age_predictions.shape == (3,)


def test_cnn_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        MultiTaskCNN(dropout=1.0)
