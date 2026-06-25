from src.evaluation.reporter import ExperimentResult, ExperimentStatus, MetricsReporter


def test_reporter_keeps_not_implemented_experiments_visible(tmp_path) -> None:
    reporter = MetricsReporter(tmp_path)
    results = [
        ExperimentResult(
            strategy_id="E3",
            strategy_name="CNN simple multitarea",
            experiment_name="cnn_base",
            variant="base",
            changed_component="ninguno",
            status=ExperimentStatus.COMPLETED,
            metrics={
                "gender_accuracy": 0.8,
                "gender_f1": 0.79,
                "age_mae": 8.0,
                "age_rmse": 10.0,
                "age_r2": 0.5,
            },
        ),
        ExperimentResult(
            strategy_id="E2",
            strategy_name="MLP multitarea",
            experiment_name="mlp_base",
            variant="base",
            changed_component="ninguno",
            status=ExperimentStatus.NOT_IMPLEMENTED,
            message="Pendiente.",
        ),
    ]

    reporter.write_all(results, {"python": "test"})

    report = (tmp_path / "all_experiments_comparison.md").read_text(encoding="utf-8")
    assert "cnn_base" in report
    assert "mlp_base" in report
    assert "NO_IMPLEMENTADO" in report
