from src.evaluation.reporter import ExperimentResult, ExperimentStatus, MetricsReporter
from src.evaluation.artifacts import ExperimentArtifactWriter
from src.evaluation.metrics import EvaluationResult


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


def test_artifact_writer_preserves_predictions_and_history(tmp_path) -> None:
    writer = ExperimentArtifactWriter(
        tmp_path / "experiments",
        tmp_path / "reports",
    )
    evaluation = EvaluationResult(
        metrics={"gender_accuracy": 1.0, "age_mae": 2.0},
        confusion_matrix=[[1, 0], [0, 1]],
        gender_targets=[0, 1],
        gender_predictions=[0, 1],
        age_targets=[20.0, 40.0],
        age_predictions=[22.0, 38.0],
    )

    writer.write_evaluation(
        "cnn_base",
        evaluation,
        sample_names=["20_0_sample.jpg", "40_1_sample.jpg"],
    )
    writer.write_history(
        "cnn_base",
        [{"epoch": 1.0, "train_total_loss": 1.2, "val_total_loss": 1.0}],
    )

    predictions = (
        tmp_path / "experiments" / "cnn_base" / "predictions.csv"
    ).read_text(encoding="utf-8")
    assert "20_0_sample.jpg" in predictions
    assert "absolute_age_error" in predictions
    assert (
        tmp_path / "experiments" / "cnn_base" / "training_history.json"
    ).exists()
