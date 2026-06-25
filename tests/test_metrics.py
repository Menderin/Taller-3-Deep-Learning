import pytest
import torch

from src.evaluation.metrics import MultiTaskMetrics


def test_metrics_calculates_classification_and_regression_values() -> None:
    result = MultiTaskMetrics.calculate(
        gender_targets=torch.tensor([0, 0, 1, 1]),
        gender_predictions=torch.tensor([0, 1, 1, 1]),
        age_targets=torch.tensor([10.0, 20.0, 30.0, 40.0]),
        age_predictions=torch.tensor([12.0, 18.0, 33.0, 39.0]),
    )

    assert result.confusion_matrix == [[1, 1], [0, 2]]
    assert result.metrics["gender_accuracy"] == pytest.approx(0.75)
    assert result.metrics["gender_f1"] == pytest.approx(0.733333, rel=1e-5)
    assert result.metrics["age_mae"] == pytest.approx(2.0)
    assert result.metrics["age_rmse"] == pytest.approx(4.5**0.5)
    assert result.metrics["age_r2"] == pytest.approx(0.964)
