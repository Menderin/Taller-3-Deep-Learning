from pathlib import Path

from src.config import AppConfig
from src.training.experiment_runner import build_experiment_catalog


def test_catalog_requires_ablations_for_every_strategy(tmp_path) -> None:
    config = AppConfig(
        project_root=tmp_path,
        dataset_dir=tmp_path / "UTKFace",
        artifacts_dir=tmp_path / "artifacts",
        cnn_checkpoint=tmp_path / "artifacts/checkpoints/cnn_base/best_model.pt",
    )

    catalog = build_experiment_catalog(config)

    for strategy_id in ("E1", "E2", "E3", "E4", "E5"):
        specs = [spec for spec in catalog.values() if spec.strategy_id == strategy_id]
        assert any(spec.variant == "base" for spec in specs)
        assert any(spec.variant == "ablacion" for spec in specs)

    assert catalog["cnn_base"].implemented is True
    assert catalog["mlp_base"].implemented is True
