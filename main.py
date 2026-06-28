"""Training orchestrator for the UTKFace laboratory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import sys
from pathlib import Path

from src.config import AppConfig
from src.evaluation.reporter import ExperimentStatus, MetricsReporter
from src.training.experiment_runner import ExperimentRunner, build_experiment_catalog
from src.utils import collect_environment_info, configure_cpu_threads, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena y compara estrategias multitarea sobre UTKFace."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--experiment",
        action="append",
        help="Nombre de un experimento. Se puede repetir la opcion.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta todos los experimentos implementados y reporta los pendientes.",
    )
    selection.add_argument(
        "--ablations",
        action="store_true",
        help="Ejecuta las bases y ablaciones implementadas de todas las estrategias.",
    )
    selection.add_argument(
        "--list",
        action="store_true",
        help="Muestra el catalogo completo sin entrenar.",
    )
    return parser


def print_catalog(catalog) -> None:
    print("Catalogo de experimentos")
    print("-" * 100)
    for spec in catalog.values():
        state = "IMPLEMENTADO" if spec.implemented else "NO_IMPLEMENTADO"
        print(
            f"{spec.strategy_id:>2}  {spec.name:<34} "
            f"{state:<15} {spec.changed_component}"
        )


def print_results(results) -> None:
    print("\nResumen")
    print("-" * 100)
    for result in results:
        metrics = result.metrics
        detail = ""
        if result.status == ExperimentStatus.COMPLETED:
            detail = (
                f"acc={metrics['gender_accuracy']:.4f} "
                f"f1={metrics['gender_f1']:.4f} "
                f"mae={metrics['age_mae']:.4f} "
                f"rmse={metrics['age_rmse']:.4f}"
            )
        elif result.message:
            detail = result.message
        print(f"{result.experiment_name:<34} {result.status.value:<16} {detail}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env()
    _thread_limiter = configure_cpu_threads(config.cpu_threads)
    config.ensure_artifact_directories()
    catalog = build_experiment_catalog(config)

    if args.list:
        print_catalog(catalog)
        return 0

    if args.all:
        selected_names = {name for name, spec in catalog.items() if spec.implemented}
    elif args.ablations:
        selected_names = {
            name
            for name, spec in catalog.items()
            if spec.implemented and spec.variant in {"base", "ablacion"}
        }
    elif args.experiment:
        selected_names = set(args.experiment)
    else:
        selected_names = {"cnn_base"}

    device = resolve_device(config.device)
    print(f"Device: {device}")
    print(f"CPU threads: {config.cpu_threads}")
    print(f"Dataset: {config.dataset_dir}")

    reporter = MetricsReporter(config.reports_dir)
    config_snapshot = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    reporter.write_run_metadata(
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_experiments": sorted(selected_names),
            "configuration": config_snapshot,
            "environment": collect_environment_info(device),
        }
    )

    runner = ExperimentRunner(config=config, device=device, catalog=catalog)
    results = runner.run(selected_names)

    report_paths = reporter.write_all(results, collect_environment_info(device))
    print_results(results)
    print(f"\nReportes generados en: {config.reports_dir}")
    print(f"Archivos escritos: {len(report_paths)}")

    selected_errors = [
        result
        for result in results
        if result.experiment_name in selected_names
        and result.status == ExperimentStatus.ERROR
    ]
    return 1 if selected_errors else 0


if __name__ == "__main__":
    sys.exit(main())
