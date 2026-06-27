"""Build reproducible train, validation and test DataLoaders."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import AppConfig
from src.data.dataset import UTKFaceDataset
from src.data.parser import UTKFaceFilenameParser, UTKFaceRecord
from src.data.transforms import TransformFactory


class UTKFaceDataModule:
    """Own the dataset discovery, split and DataLoader creation process."""

    IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")

    def __init__(self, config: AppConfig, use_augmentation: bool = True) -> None:
        self.config = config
        self.use_augmentation = use_augmentation
        self.parser = UTKFaceFilenameParser()
        self.train_dataset: UTKFaceDataset | None = None
        self.val_dataset: UTKFaceDataset | None = None
        self.test_dataset: UTKFaceDataset | None = None

    def setup(self) -> None:
        """Discover valid images and create deterministic record-level splits."""

        records = self._discover_records()
        train_records, val_records, test_records = self._split_records(records)

        train_transform = TransformFactory.training(
            self.config.image_size,
            use_augmentation=self.use_augmentation,
        )
        eval_transform = TransformFactory.evaluation(self.config.image_size)

        # Separate datasets prevent random training transforms from leaking into
        # validation and test sets.
        self.train_dataset = UTKFaceDataset(train_records, transform=train_transform)
        self.val_dataset = UTKFaceDataset(val_records, transform=eval_transform)
        self.test_dataset = UTKFaceDataset(test_records, transform=eval_transform)
        self._save_split_manifest(train_records, val_records, test_records)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.train_dataset, "train"), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.val_dataset, "validation"), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._require_dataset(self.test_dataset, "test"), shuffle=False)

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": len(self._require_dataset(self.train_dataset, "train")),
            "validation": len(self._require_dataset(self.val_dataset, "validation")),
            "test": len(self._require_dataset(self.test_dataset, "test")),
        }

    def test_sample_names(self) -> list[str]:
        dataset = self._require_dataset(self.test_dataset, "test")
        return [record.path.name for record in dataset.records]

    def _discover_records(self) -> list[UTKFaceRecord]:
        if not self.config.dataset_dir.exists():
            raise FileNotFoundError(
                f"No existe UTKFACE_DIR: {self.config.dataset_dir}. "
                "Configure la ruta en el archivo .env."
            )

        paths: list[Path] = []
        for pattern in self.IMAGE_PATTERNS:
            paths.extend(self.config.dataset_dir.glob(pattern))
        paths = sorted(set(paths))

        records: list[UTKFaceRecord] = []
        invalid_names = 0
        for path in paths:
            try:
                records.append(self.parser.parse(path))
            except ValueError:
                invalid_names += 1

        if self.config.max_images > 0:
            records = records[: self.config.max_images]

        if len(records) < 3:
            raise RuntimeError(
                "Se necesitan al menos tres imagenes UTKFace validas para crear los splits. "
                f"Encontradas: {len(records)}; nombres ignorados: {invalid_names}."
            )
        return records

    def _split_records(
        self,
        records: list[UTKFaceRecord],
    ) -> tuple[list[UTKFaceRecord], list[UTKFaceRecord], list[UTKFaceRecord]]:
        generator = torch.Generator().manual_seed(self.config.seed)
        order = torch.randperm(len(records), generator=generator).tolist()
        shuffled = [records[index] for index in order]

        n_train = int(len(shuffled) * self.config.train_fraction)
        n_val = int(len(shuffled) * self.config.val_fraction)
        n_train = max(1, n_train)
        n_val = max(1, n_val)

        # Preserve at least one sample for test when using tiny debugging subsets.
        if n_train + n_val >= len(shuffled):
            n_train = max(1, len(shuffled) - 2)
            n_val = 1

        train_records = shuffled[:n_train]
        val_records = shuffled[n_train : n_train + n_val]
        test_records = shuffled[n_train + n_val :]
        return train_records, val_records, test_records

    def _loader(self, dataset: UTKFaceDataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def _save_split_manifest(
        self,
        train_records: list[UTKFaceRecord],
        val_records: list[UTKFaceRecord],
        test_records: list[UTKFaceRecord],
    ) -> None:
        self.config.splits_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "seed": self.config.seed,
            "dataset_dir": str(self.config.dataset_dir),
            "train": [record.path.name for record in train_records],
            "validation": [record.path.name for record in val_records],
            "test": [record.path.name for record in test_records],
        }
        output_path = self.config.splits_dir / "utkface_split.json"
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def _require_dataset(dataset: UTKFaceDataset | None, name: str) -> UTKFaceDataset:
        if dataset is None:
            raise RuntimeError(f"El dataset de {name} no esta listo. Ejecute setup() primero.")
        return dataset
