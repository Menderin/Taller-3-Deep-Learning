"""Generate a qualitative 2x3 grid from saved test predictions."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


GENDER_LABELS = {0: "M", 1: "F"}
REQUIRED_COLUMNS = {
    "sample_name",
    "gender_target",
    "gender_prediction",
    "age_target",
    "age_prediction",
    "absolute_age_error",
}


@dataclass(frozen=True)
class PredictionExample:
    """One test prediction and the location of its source image."""

    sample_name: str
    image_path: Path
    gender_target: int
    gender_prediction: int
    age_target: float
    age_prediction: float
    absolute_age_error: float

    @property
    def gender_correct(self) -> bool:
        return self.gender_target == self.gender_prediction


@dataclass(frozen=True)
class SelectedExample:
    """A prediction selected for one semantic panel of the figure."""

    category: str
    example: PredictionExample
    border_color: str


def load_predictions(
    predictions_path: Path,
    dataset_dir: Path,
) -> list[PredictionExample]:
    """Read predictions and retain rows whose source image exists."""

    if not predictions_path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {predictions_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset_dir}")

    with predictions_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas: " + ", ".join(missing_columns)
            )

        examples: list[PredictionExample] = []
        missing_images = 0
        for row_number, row in enumerate(reader, start=2):
            try:
                sample_name = Path(row["sample_name"]).name
                image_path = dataset_dir / sample_name
                gender_target = int(row["gender_target"])
                gender_prediction = int(row["gender_prediction"])
                if gender_target not in GENDER_LABELS:
                    raise ValueError(f"genero real invalido: {gender_target}")
                if gender_prediction not in GENDER_LABELS:
                    raise ValueError(
                        f"genero predicho invalido: {gender_prediction}"
                    )
                example = PredictionExample(
                    sample_name=sample_name,
                    image_path=image_path,
                    gender_target=gender_target,
                    gender_prediction=gender_prediction,
                    age_target=float(row["age_target"]),
                    age_prediction=float(row["age_prediction"]),
                    absolute_age_error=float(row["absolute_age_error"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Fila {row_number} invalida en {predictions_path}: {error}"
                ) from error

            if image_path.is_file():
                examples.append(example)
            else:
                missing_images += 1

    if not examples:
        raise RuntimeError(
            "No se encontro ninguna imagen del CSV dentro del dataset."
        )
    if missing_images:
        print(f"Advertencia: {missing_images} imagenes del CSV no fueron encontradas.")
    return examples


def _choose(
    examples: Iterable[PredictionExample],
    predicate: Callable[[PredictionExample], bool],
    used_names: set[str],
    rng: random.Random,
) -> PredictionExample | None:
    candidates = sorted(
        (
            example
            for example in examples
            if example.sample_name not in used_names and predicate(example)
        ),
        key=lambda example: example.sample_name,
    )
    if not candidates:
        return None
    selected = rng.choice(candidates)
    used_names.add(selected.sample_name)
    return selected


def _choose_with_fallback(
    examples: list[PredictionExample],
    predicates: list[Callable[[PredictionExample], bool]],
    used_names: set[str],
    rng: random.Random,
    category: str,
) -> PredictionExample:
    for predicate in predicates:
        selected = _choose(examples, predicate, used_names, rng)
        if selected is not None:
            return selected
    raise RuntimeError(f"No hay candidatos suficientes para: {category}")


def select_examples(
    examples: list[PredictionExample],
    seed: int = 42,
) -> list[SelectedExample]:
    """Select six unique cases with deterministic semantic criteria."""

    rng = random.Random(seed)
    used_names: set[str] = set()

    def accurate(gender: int) -> list[Callable[[PredictionExample], bool]]:
        return [
            lambda item: (
                item.gender_target == gender
                and item.gender_prediction == gender
                and item.absolute_age_error < 3
            ),
            lambda item: (
                item.gender_target == gender
                and item.gender_prediction == gender
                and item.absolute_age_error < 5
            ),
        ]

    selected: list[SelectedExample] = []
    for gender in (0, 1):
        example = _choose_with_fallback(
            examples,
            accurate(gender),
            used_names,
            rng,
            category=f"acierto {GENDER_LABELS[gender]}",
        )
        selected.append(
            SelectedExample(
                category=f"Acierto ({GENDER_LABELS[gender]})",
                example=example,
                border_color="#2e8b57",
            )
        )

    gender_errors = (
        (0, 1, "Error de g\u00e9nero (M a F)"),
        (1, 0, "Error de g\u00e9nero (F a M)"),
    )
    for target, prediction, category in gender_errors:
        example = _choose_with_fallback(
            examples,
            [
                lambda item, target=target, prediction=prediction: (
                    item.gender_target == target
                    and item.gender_prediction == prediction
                    and item.absolute_age_error < 3
                ),
                lambda item, target=target, prediction=prediction: (
                    item.gender_target == target
                    and item.gender_prediction == prediction
                ),
            ],
            used_names,
            rng,
            category=category,
        )
        selected.append(
            SelectedExample(
                category=category,
                example=example,
                border_color="#b22222",
            )
        )

    for preferred_gender in (0, 1):
        category = "Error grande de edad (60+)"
        example = _choose_with_fallback(
            examples,
            [
                lambda item, gender=preferred_gender: (
                    item.age_target >= 60
                    and item.absolute_age_error > 10
                    and item.gender_correct
                    and item.gender_target == gender
                ),
                lambda item: (
                    item.age_target >= 60
                    and item.absolute_age_error > 10
                    and item.gender_correct
                ),
                lambda item: (
                    item.age_target >= 60 and item.absolute_age_error > 10
                ),
                lambda item: item.absolute_age_error > 10 and item.gender_correct,
                lambda item: item.absolute_age_error > 10,
            ],
            used_names,
            rng,
            category=category,
        )
        selected.append(
            SelectedExample(
                category=category,
                example=example,
                border_color="#d17a00",
            )
        )

    return selected


def render_grid(
    selected: list[SelectedExample],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Render and save the six selected examples."""

    if len(selected) != 6:
        raise ValueError("La grilla requiere exactamente seis ejemplos.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    figure.suptitle(
        "Ejemplos cualitativos del modelo ResNet18 final",
        fontsize=17,
        fontweight="bold",
    )

    for axis, selection in zip(axes.flat, selected):
        example = selection.example
        try:
            with Image.open(example.image_path) as image_file:
                image = ImageOps.exif_transpose(image_file).convert("RGB")
        except OSError as error:
            raise RuntimeError(
                f"No se pudo abrir {example.image_path}: {error}"
            ) from error

        axis.imshow(image)
        axis.set_title(selection.category, fontsize=12, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(selection.border_color)
            spine.set_linewidth(3)

        target_gender = GENDER_LABELS[example.gender_target]
        predicted_gender = GENDER_LABELS[example.gender_prediction]
        caption = (
            f"Real: {target_gender}, {example.age_target:.0f} a\u00f1os | "
            f"Predicho: {predicted_gender}, {example.age_prediction:.0f} a\u00f1os\n"
            f"Error absoluto de edad: {example.absolute_age_error:.1f} a\u00f1os"
        )
        axis.text(
            0.5,
            -0.08,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    figure.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.90,
        bottom=0.08,
        wspace=0.18,
        hspace=0.34,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def print_selection(selected: list[SelectedExample]) -> None:
    """Print the selected rows for report traceability."""

    print("\nEjemplos seleccionados")
    print("-" * 100)
    for index, selection in enumerate(selected, start=1):
        item = selection.example
        print(
            f"{index}. {selection.category}: {item.sample_name} | "
            f"real={GENDER_LABELS[item.gender_target]},{item.age_target:.0f} | "
            f"pred={GENDER_LABELS[item.gender_prediction]},"
            f"{item.age_prediction:.1f} | error={item.absolute_age_error:.1f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una grilla 2x3 de ejemplos cualitativos usando "
            "predicciones ya guardadas."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV predictions.csv del modelo final.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Carpeta que contiene las imagenes originales de UTKFace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("informe/images/resnet_final_visual_examples.png"),
        help="Ruta del PNG de salida.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi debe ser mayor que cero.")

    examples = load_predictions(args.predictions, args.dataset)
    selected = select_examples(examples, seed=args.seed)
    render_grid(selected, args.output, dpi=args.dpi)
    print_selection(selected)
    print(f"\nFigura guardada en: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
