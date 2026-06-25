"""Parse labels encoded in UTKFace image filenames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UTKFaceRecord:
    """A single image path and the labels extracted from its filename."""

    path: Path
    age: float
    gender: int
    race: int | None = None


class UTKFaceFilenameParser:
    """Convert names such as 25_1_2_20170116174525125.jpg into labels."""

    VALID_GENDERS = {0, 1}

    def parse(self, path: str | Path) -> UTKFaceRecord:
        image_path = Path(path)
        parts = image_path.stem.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Nombre UTKFace invalido: {image_path.name}. "
                "Se esperaba edad_genero_raza_fecha.jpg."
            )

        try:
            age = float(parts[0])
            gender = int(parts[1])
            race = int(parts[2]) if len(parts) >= 3 else None
        except ValueError as error:
            raise ValueError(f"Etiquetas invalidas en {image_path.name}.") from error

        if age < 0:
            raise ValueError(f"La edad no puede ser negativa: {image_path.name}.")
        if gender not in self.VALID_GENDERS:
            raise ValueError(f"Genero fuera de rango en {image_path.name}: {gender}.")

        return UTKFaceRecord(path=image_path, age=age, gender=gender, race=race)
