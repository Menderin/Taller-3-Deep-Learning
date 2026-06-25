"""PyTorch-based metrics for gender classification and age regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    """Metrics plus raw values needed to build diagnostic plots."""

    metrics: dict[str, float | int]
    confusion_matrix: list[list[int]]
    gender_targets: list[int]
    gender_predictions: list[int]
    age_targets: list[float]
    age_predictions: list[float]


class MultiTaskMetrics:
    """Calculate all required metrics without hiding their definitions."""

    AGE_RANGES = (
        ("0_12", 0.0, 13.0),
        ("13_19", 13.0, 20.0),
        ("20_39", 20.0, 40.0),
        ("40_59", 40.0, 60.0),
        ("60_plus", 60.0, float("inf")),
    )

    @classmethod
    def calculate(
        cls,
        gender_targets: torch.Tensor,
        gender_predictions: torch.Tensor,
        age_targets: torch.Tensor,
        age_predictions: torch.Tensor,
    ) -> EvaluationResult:
        gender_targets = gender_targets.to(torch.long).cpu()
        gender_predictions = gender_predictions.to(torch.long).cpu()
        age_targets = age_targets.to(torch.float32).cpu()
        age_predictions = age_predictions.to(torch.float32).cpu()

        if gender_targets.numel() == 0:
            raise ValueError("No se pueden calcular metricas sin muestras.")

        confusion = torch.zeros((2, 2), dtype=torch.long)
        for target, prediction in zip(gender_targets, gender_predictions):
            confusion[int(target), int(prediction)] += 1

        total = confusion.sum().item()
        accuracy = confusion.diag().sum().item() / total
        supports = confusion.sum(dim=1).to(torch.float32)
        predicted_counts = confusion.sum(dim=0).to(torch.float32)
        true_positives = confusion.diag().to(torch.float32)

        precision_per_class = cls._safe_divide(true_positives, predicted_counts)
        recall_per_class = cls._safe_divide(true_positives, supports)
        f1_per_class = cls._safe_divide(
            2 * precision_per_class * recall_per_class,
            precision_per_class + recall_per_class,
        )
        weights = supports / supports.sum()

        age_errors = age_predictions - age_targets
        mae = age_errors.abs().mean().item()
        rmse = torch.sqrt((age_errors**2).mean()).item()
        ss_res = (age_errors**2).sum()
        ss_tot = ((age_targets - age_targets.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else 0.0

        metrics: dict[str, float | int] = {
            "gender_accuracy": accuracy,
            "gender_precision": (precision_per_class * weights).sum().item(),
            "gender_recall": (recall_per_class * weights).sum().item(),
            "gender_f1": (f1_per_class * weights).sum().item(),
            "age_mae": mae,
            "age_rmse": rmse,
            "age_r2": r2,
            "samples": total,
        }
        metrics.update(cls._age_range_metrics(age_targets, age_predictions))

        return EvaluationResult(
            metrics=metrics,
            confusion_matrix=confusion.tolist(),
            gender_targets=gender_targets.tolist(),
            gender_predictions=gender_predictions.tolist(),
            age_targets=age_targets.tolist(),
            age_predictions=age_predictions.tolist(),
        )

    @staticmethod
    def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))

    @classmethod
    def _age_range_metrics(
        cls,
        age_targets: torch.Tensor,
        age_predictions: torch.Tensor,
    ) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        for label, lower, upper in cls.AGE_RANGES:
            mask = (age_targets >= lower) & (age_targets < upper)
            support = int(mask.sum().item())
            output[f"age_support_{label}"] = support
            if support > 0:
                output[f"age_mae_{label}"] = (
                    age_predictions[mask] - age_targets[mask]
                ).abs().mean().item()
            else:
                output[f"age_mae_{label}"] = float("nan")
        return output


class MultiTaskEvaluator:
    """Collect predictions from a PyTorch model and calculate metrics."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> EvaluationResult:
        model.eval()
        gender_targets: list[torch.Tensor] = []
        gender_predictions: list[torch.Tensor] = []
        age_targets: list[torch.Tensor] = []
        age_predictions: list[torch.Tensor] = []

        for images, batch_gender_targets, batch_age_targets in loader:
            images = images.to(self.device)
            gender_logits, batch_age_predictions = model(images)
            batch_gender_predictions = gender_logits.argmax(dim=1)

            gender_targets.append(batch_gender_targets.cpu())
            gender_predictions.append(batch_gender_predictions.cpu())
            age_targets.append(batch_age_targets.cpu())
            age_predictions.append(batch_age_predictions.cpu())

        if not gender_targets:
            raise RuntimeError("El DataLoader de evaluacion no contiene muestras.")

        return MultiTaskMetrics.calculate(
            gender_targets=torch.cat(gender_targets),
            gender_predictions=torch.cat(gender_predictions),
            age_targets=torch.cat(age_targets),
            age_predictions=torch.cat(age_predictions),
        )
