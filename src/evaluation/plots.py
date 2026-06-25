"""Generate diagnostic plots for completed experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import EvaluationResult
from src.evaluation.reporter import ExperimentResult, ExperimentStatus


class ResultPlotter:
    """Save curves and comparisons instead of relying on manual screenshots."""

    def __init__(self, plots_dir: Path) -> None:
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(
        self,
        history: list[dict[str, float]],
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        epochs = [int(row["epoch"]) for row in history]
        loss_pairs = [
            ("total_loss", "Perdida total"),
            ("gender_loss", "Perdida de genero"),
            ("age_loss", "Perdida de edad"),
        ]

        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, (key, title) in zip(axes, loss_pairs):
            axis.plot(epochs, [row[f"train_{key}"] for row in history], label="train")
            axis.plot(epochs, [row[f"val_{key}"] for row in history], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.legend()
            axis.grid(alpha=0.3)
        figure.tight_layout()

        output_path = output_dir / "training_curves.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_confusion_matrix(
        self,
        evaluation: EvaluationResult,
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        matrix = evaluation.confusion_matrix
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_xticks([0, 1], labels=["0", "1"])
        axis.set_yticks([0, 1], labels=["0", "1"])
        axis.set_xlabel("Genero predicho")
        axis.set_ylabel("Genero real")
        axis.set_title("Matriz de confusion")

        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
        figure.tight_layout()

        output_path = output_dir / "gender_confusion_matrix.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_age_predictions(
        self,
        evaluation: EvaluationResult,
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(
            evaluation.age_targets,
            evaluation.age_predictions,
            alpha=0.35,
            s=14,
        )
        all_ages = evaluation.age_targets + evaluation.age_predictions
        lower = min(all_ages)
        upper = max(all_ages)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
        axis.set_xlabel("Edad real")
        axis.set_ylabel("Edad predicha")
        axis.set_title("Edad real versus predicha")
        axis.grid(alpha=0.3)
        figure.tight_layout()

        output_path = output_dir / "age_real_vs_predicted.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_ablation_comparison(
        self,
        results: list[ExperimentResult],
        strategy_id: str,
    ) -> list[Path]:
        completed = [
            result
            for result in results
            if result.strategy_id == strategy_id
            and result.status == ExperimentStatus.COMPLETED
        ]
        if not completed:
            return []

        names = [result.experiment_name for result in completed]
        gender_accuracy = [float(result.metrics["gender_accuracy"]) for result in completed]
        gender_f1 = [float(result.metrics["gender_f1"]) for result in completed]
        age_mae = [float(result.metrics["age_mae"]) for result in completed]
        age_rmse = [float(result.metrics["age_rmse"]) for result in completed]

        output_paths = [
            self._bar_plot(
                names,
                [gender_accuracy, gender_f1],
                ["Accuracy", "F1"],
                f"{strategy_id}: metricas de genero",
                self.plots_dir / f"{strategy_id.lower()}_ablation_gender_metrics.png",
            ),
            self._bar_plot(
                names,
                [age_mae, age_rmse],
                ["MAE", "RMSE"],
                f"{strategy_id}: metricas de edad",
                self.plots_dir / f"{strategy_id.lower()}_ablation_age_metrics.png",
            ),
        ]
        return output_paths

    def _experiment_dir(self, experiment_name: str) -> Path:
        output_dir = self.plots_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _bar_plot(
        names: list[str],
        series: list[list[float]],
        labels: list[str],
        title: str,
        output_path: Path,
    ) -> Path:
        import numpy as np

        x = np.arange(len(names))
        width = 0.8 / len(series)
        figure, axis = plt.subplots(figsize=(max(7, len(names) * 1.5), 5))
        for index, (values, label) in enumerate(zip(series, labels)):
            offset = (index - (len(series) - 1) / 2) * width
            axis.bar(x + offset, values, width=width, label=label)
        axis.set_xticks(x, labels=names, rotation=30, ha="right")
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
