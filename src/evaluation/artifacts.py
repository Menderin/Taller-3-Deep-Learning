"""Persist granular experiment outputs so completed work survives interruptions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import EvaluationResult
from src.evaluation.reporter import ExperimentResult


class ExperimentArtifactWriter:
    """Write machine-readable outputs for each individual experiment."""

    def __init__(self, experiments_dir: Path, reports_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.reports_dir = reports_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, selected_names: set[str]) -> None:
        self._write_json(
            self.reports_dir / "run_progress.json",
            {
                "status": "RUNNING",
                "selected_experiments": sorted(selected_names),
                "completed_results": [],
            },
        )

    def update_run(
        self,
        selected_names: set[str],
        results: list[ExperimentResult],
        status: str = "RUNNING",
    ) -> None:
        selected_results = [
            result.to_row()
            for result in results
            if result.experiment_name in selected_names
        ]
        self._write_json(
            self.reports_dir / "run_progress.json",
            {
                "status": status,
                "selected_experiments": sorted(selected_names),
                "completed_results": selected_results,
            },
        )

    def write_result(self, result: ExperimentResult) -> Path:
        output_path = self._experiment_dir(result.experiment_name) / "result.json"
        self._write_json(output_path, result.to_row())
        return output_path

    def write_history(
        self,
        experiment_name: str,
        history: list[dict[str, float]],
    ) -> list[Path]:
        output_dir = self._experiment_dir(experiment_name)
        json_path = output_dir / "training_history.json"
        csv_path = output_dir / "training_history.csv"
        self._write_json(json_path, history)

        columns = list(history[0]) if history else []
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(history)
        return [json_path, csv_path]

    def write_evaluation(
        self,
        experiment_name: str,
        evaluation: EvaluationResult,
        sample_names: list[str] | None = None,
    ) -> list[Path]:
        output_dir = self._experiment_dir(experiment_name)
        metrics_path = output_dir / "evaluation.json"
        predictions_path = output_dir / "predictions.csv"
        self._write_json(
            metrics_path,
            {
                "metrics": evaluation.metrics,
                "confusion_matrix": evaluation.confusion_matrix,
            },
        )

        columns = [
            "sample_index",
            "sample_name",
            "gender_target",
            "gender_prediction",
            "gender_correct",
            "age_target",
            "age_prediction",
            "age_error",
            "absolute_age_error",
        ]
        with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            if sample_names is not None and len(sample_names) != len(
                evaluation.gender_targets
            ):
                raise ValueError(
                    "La cantidad de nombres no coincide con las predicciones."
                )
            for index, values in enumerate(
                zip(
                    evaluation.gender_targets,
                    evaluation.gender_predictions,
                    evaluation.age_targets,
                    evaluation.age_predictions,
                )
            ):
                gender_target, gender_prediction, age_target, age_prediction = values
                age_error = age_prediction - age_target
                writer.writerow(
                    {
                        "sample_index": index,
                        "sample_name": sample_names[index] if sample_names else "",
                        "gender_target": gender_target,
                        "gender_prediction": gender_prediction,
                        "gender_correct": int(gender_target == gender_prediction),
                        "age_target": age_target,
                        "age_prediction": age_prediction,
                        "age_error": age_error,
                        "absolute_age_error": abs(age_error),
                    }
                )
        return [metrics_path, predictions_path]

    def _experiment_dir(self, experiment_name: str) -> Path:
        output_dir = self.experiments_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
