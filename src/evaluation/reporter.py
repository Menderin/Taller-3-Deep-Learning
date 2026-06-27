"""Export experiment states and metrics to reproducible CSV and Markdown files."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStatus(str, Enum):
    COMPLETED = "COMPLETADO"
    NOT_IMPLEMENTED = "NO_IMPLEMENTADO"
    NOT_EXECUTED = "NO_EJECUTADO"
    ERROR = "ERROR"


@dataclass
class ExperimentResult:
    """One row in the final experiment comparison."""

    strategy_id: str
    strategy_name: str
    experiment_name: str
    variant: str
    changed_component: str
    status: ExperimentStatus
    metrics: dict[str, float | int] = field(default_factory=dict)
    trainable_parameters: int | None = None
    training_seconds: float | None = None
    checkpoint: str = ""
    message: str = ""

    def to_row(self) -> dict[str, Any]:
        row = {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "experiment_name": self.experiment_name,
            "variant": self.variant,
            "changed_component": self.changed_component,
            "status": self.status.value,
            "trainable_parameters": self.trainable_parameters,
            "training_seconds": self.training_seconds,
            "checkpoint": self.checkpoint,
            "message": self.message,
        }
        row.update(self.metrics)
        return row


class MetricsReporter:
    """Write a global report and one ablation report per strategy."""

    BASE_COLUMNS = [
        "strategy_id",
        "strategy_name",
        "experiment_name",
        "variant",
        "changed_component",
        "status",
        "gender_accuracy",
        "gender_precision",
        "gender_recall",
        "gender_f1",
        "age_mae",
        "age_rmse",
        "age_r2",
        "samples",
        "trainable_parameters",
        "training_seconds",
        "checkpoint",
        "message",
    ]

    STRATEGY_FILENAMES = {
        "E1": "e1_classical_ablations",
        "E2": "e2_mlp_ablations",
        "E3": "e3_cnn_ablations",
        "E4": "e4_resnet_frozen_ablations",
        "E5": "e5_resnet_finetuning_ablations",
    }

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        results: list[ExperimentResult],
        environment_info: dict[str, Any],
    ) -> list[Path]:
        """Export all rows, including missing and unexecuted strategies."""

        output_paths: list[Path] = []
        output_paths.extend(
            self._write_table(results, self.reports_dir / "all_experiments_comparison")
        )

        for strategy_id, filename in self.STRATEGY_FILENAMES.items():
            strategy_results = [
                result for result in results if result.strategy_id == strategy_id
            ]
            output_paths.extend(
                self._write_table(strategy_results, self.reports_dir / filename)
            )

        environment_path = self.reports_dir / "environment.json"
        environment_path.write_text(
            json.dumps(environment_info, indent=2, default=str),
            encoding="utf-8",
        )
        output_paths.append(environment_path)
        return output_paths

    def write_run_metadata(self, metadata: dict[str, Any]) -> Path:
        """Persist the effective configuration without copying environment secrets."""

        output_path = self.reports_dir / "run_metadata.json"
        output_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path

    def _write_table(
        self,
        results: list[ExperimentResult],
        output_base: Path,
    ) -> list[Path]:
        rows = [result.to_row() for result in results]
        extra_columns = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in self.BASE_COLUMNS
            }
        )
        columns = self.BASE_COLUMNS + extra_columns

        csv_path = output_base.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: self._format_csv(row.get(key)) for key in columns})

        markdown_path = output_base.with_suffix(".md")
        markdown_path.write_text(
            self._to_markdown(rows, columns),
            encoding="utf-8",
        )
        return [csv_path, markdown_path]

    @classmethod
    def _to_markdown(cls, rows: list[dict[str, Any]], columns: list[str]) -> str:
        visible_columns = [
            column
            for column in columns
            if column
            in {
                "strategy_id",
                "experiment_name",
                "changed_component",
                "status",
                "gender_accuracy",
                "gender_f1",
                "age_mae",
                "age_rmse",
                "age_r2",
                "trainable_parameters",
                "training_seconds",
                "message",
            }
        ]
        header = "| " + " | ".join(visible_columns) + " |"
        separator = "| " + " | ".join("---" for _ in visible_columns) + " |"
        lines = ["# Reporte de experimentos", "", header, separator]
        for row in rows:
            values = [cls._format_markdown(row.get(column)) for column in visible_columns]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_csv(value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value if value is not None else ""

    @staticmethod
    def _format_markdown(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "/")
