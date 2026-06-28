"""Application configuration loaded from environment variables and a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding an extra dependency.

    Existing environment variables take precedence over values from the file.
    This parser intentionally supports the simple format used by this project.
    """

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _as_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _as_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class AppConfig:
    """Central configuration shared by training, evaluation and inference."""

    project_root: Path
    dataset_dir: Path
    artifacts_dir: Path
    cnn_checkpoint: Path
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_age: float = 0.01
    num_workers: int = 0
    cpu_threads: int = 4
    max_images: int = 0
    device: str = "auto"
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_dir / "plots"

    @property
    def splits_dir(self) -> Path:
        return self.artifacts_dir / "splits"

    @property
    def experiments_dir(self) -> Path:
        return self.artifacts_dir / "experiments"

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "AppConfig":
        """Build the configuration after reading the optional .env file."""

        project_root = Path.cwd().resolve()
        _load_env_file(project_root / Path(env_file))

        dataset_dir = Path(os.getenv("UTKFACE_DIR", "UTKFace")).expanduser()
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts")).expanduser()
        checkpoint = Path(
            os.getenv(
                "CNN_CHECKPOINT",
                "Frontend/models/best_model.pt",
            )
        ).expanduser()

        if not dataset_dir.is_absolute():
            dataset_dir = project_root / dataset_dir
        if not artifacts_dir.is_absolute():
            artifacts_dir = project_root / artifacts_dir
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint

        config = cls(
            project_root=project_root,
            dataset_dir=dataset_dir.resolve(),
            artifacts_dir=artifacts_dir.resolve(),
            cnn_checkpoint=checkpoint.resolve(),
            seed=_as_int("SEED", 42),
            image_size=_as_int("IMAGE_SIZE", 224),
            batch_size=_as_int("BATCH_SIZE", 32),
            epochs=_as_int("EPOCHS", 10),
            learning_rate=_as_float("LEARNING_RATE", 1e-3),
            weight_decay=_as_float("WEIGHT_DECAY", 1e-4),
            lambda_age=_as_float("LAMBDA_AGE", 0.01),
            num_workers=_as_int("NUM_WORKERS", 0),
            cpu_threads=_as_int("CPU_THREADS", 4),
            max_images=_as_int("MAX_IMAGES", 0),
            device=os.getenv("DEVICE", "auto").lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate values that would otherwise produce confusing errors."""

        if self.image_size <= 0:
            raise ValueError("IMAGE_SIZE debe ser mayor que cero.")
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE debe ser mayor que cero.")
        if self.epochs <= 0:
            raise ValueError("EPOCHS debe ser mayor que cero.")
        if self.max_images < 0:
            raise ValueError("MAX_IMAGES no puede ser negativo.")
        if self.cpu_threads <= 0:
            raise ValueError("CPU_THREADS debe ser mayor que cero.")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("Las fracciones de train y validacion deben dejar datos para test.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("DEVICE debe ser auto, cpu, cuda o mps.")

    def ensure_artifact_directories(self) -> None:
        """Create directories used by generated results."""

        for path in (
            self.checkpoints_dir,
            self.reports_dir,
            self.plots_dir,
            self.splits_dir,
            self.experiments_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
