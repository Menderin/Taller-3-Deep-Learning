"""Generate diagnostic plots for completed experiments."""

from __future__ import annotations

import math
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

    def plot_age_residuals(
        self,
        evaluation: EvaluationResult,
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        residuals = [
            prediction - target
            for target, prediction in zip(
                evaluation.age_targets,
                evaluation.age_predictions,
            )
        ]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(residuals, bins=30, edgecolor="black", alpha=0.8)
        axes[0].axvline(0, color="black", linestyle="--")
        axes[0].set_xlabel("Error (edad predicha - real)")
        axes[0].set_ylabel("Muestras")
        axes[0].set_title("Distribucion de errores")

        axes[1].scatter(evaluation.age_targets, residuals, alpha=0.35, s=14)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set_xlabel("Edad real")
        axes[1].set_ylabel("Error")
        axes[1].set_title("Error por edad real")
        axes[1].grid(alpha=0.3)
        figure.tight_layout()

        output_path = output_dir / "age_residuals.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def plot_age_range_errors(
        self,
        evaluation: EvaluationResult,
        experiment_name: str,
    ) -> Path:
        output_dir = self._experiment_dir(experiment_name)
        labels = ["0-12", "13-19", "20-39", "40-59", "60+"]
        keys = ["0_12", "13_19", "20_39", "40_59", "60_plus"]
        raw_values = [
            float(evaluation.metrics[f"age_mae_{key}"])
            for key in keys
        ]
        values = [value if math.isfinite(value) else 0.0 for value in raw_values]
        supports = [
            int(evaluation.metrics[f"age_support_{key}"])
            for key in keys
        ]

        figure, axis = plt.subplots(figsize=(7, 4))
        bars = axis.bar(labels, values)
        axis.set_xlabel("Rango etario")
        axis.set_ylabel("MAE")
        axis.set_title("Error de edad por rango")
        axis.grid(axis="y", alpha=0.3)
        for bar, support in zip(bars, supports):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={support}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        figure.tight_layout()
        output_path = output_dir / "age_mae_by_range.png"
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

    def plot_global_comparison(
        self,
        results: list[ExperimentResult],
    ) -> list[Path]:
        completed = [
            result
            for result in results
            if result.status == ExperimentStatus.COMPLETED
        ]
        if not completed:
            return []

        names = [result.experiment_name for result in completed]
        output_paths = [
            self._horizontal_bar_plot(
                names,
                [
                    [float(result.metrics["gender_accuracy"]) for result in completed],
                    [float(result.metrics["gender_f1"]) for result in completed],
                ],
                ["Accuracy", "F1"],
                "Comparacion global: genero",
                self.plots_dir / "global_gender_comparison.png",
            ),
            self._horizontal_bar_plot(
                names,
                [
                    [float(result.metrics["age_mae"]) for result in completed],
                    [float(result.metrics["age_rmse"]) for result in completed],
                ],
                ["MAE", "RMSE"],
                "Comparacion global: edad",
                self.plots_dir / "global_age_comparison.png",
            ),
            self._plot_tradeoff(completed),
            self._plot_computational_cost(completed),
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

    @staticmethod
    def _horizontal_bar_plot(
        names: list[str],
        series: list[list[float]],
        labels: list[str],
        title: str,
        output_path: Path,
    ) -> Path:
        import numpy as np

        y = np.arange(len(names))
        height = 0.8 / len(series)
        figure, axis = plt.subplots(figsize=(11, max(6, len(names) * 0.45)))
        for index, (values, label) in enumerate(zip(series, labels)):
            offset = (index - (len(series) - 1) / 2) * height
            axis.barh(y + offset, values, height=height, label=label)
        axis.set_yticks(y, labels=names)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def _plot_tradeoff(self, results: list[ExperimentResult]) -> Path:
        figure, axis = plt.subplots(figsize=(9, 6))
        strategy_colors = {
            "E1": "tab:gray",
            "E2": "tab:orange",
            "E3": "tab:green",
            "E4": "tab:blue",
            "E5": "tab:red",
        }
        for result in results:
            axis.scatter(
                float(result.metrics["age_mae"]),
                float(result.metrics["gender_accuracy"]),
                color=strategy_colors.get(result.strategy_id, "black"),
                s=45,
            )
            axis.annotate(
                result.experiment_name,
                (
                    float(result.metrics["age_mae"]),
                    float(result.metrics["gender_accuracy"]),
                ),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        axis.set_xlabel("MAE de edad (menor es mejor)")
        axis.set_ylabel("Accuracy de genero (mayor es mejor)")
        axis.set_title("Compromiso entre las dos tareas")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        output_path = self.plots_dir / "global_multitask_tradeoff.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def _plot_computational_cost(self, results: list[ExperimentResult]) -> Path:
        names = [result.experiment_name for result in results]
        training_seconds = [float(result.training_seconds or 0.0) for result in results]
        parameters = [int(result.trainable_parameters or 0) for result in results]
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(14, max(6, len(names) * 0.45)),
        )
        axes[0].barh(names, training_seconds)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Segundos")
        axes[0].set_title("Tiempo de entrenamiento")
        axes[0].grid(axis="x", alpha=0.3)

        axes[1].barh(names, parameters)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Parametros entrenables")
        axes[1].set_title("Complejidad entrenable")
        axes[1].grid(axis="x", alpha=0.3)
        figure.tight_layout()
        output_path = self.plots_dir / "global_computational_cost.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
