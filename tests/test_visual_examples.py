from pathlib import Path

from scripts.generate_visual_examples import (
    PredictionExample,
    select_examples,
)


def _example(
    name: str,
    gender_target: int,
    gender_prediction: int,
    age_target: float,
    age_prediction: float,
) -> PredictionExample:
    return PredictionExample(
        sample_name=name,
        image_path=Path(name),
        gender_target=gender_target,
        gender_prediction=gender_prediction,
        age_target=age_target,
        age_prediction=age_prediction,
        absolute_age_error=abs(age_prediction - age_target),
    )


def test_select_examples_covers_required_categories() -> None:
    examples = [
        _example("hit_m.jpg", 0, 0, 30, 31),
        _example("hit_f.jpg", 1, 1, 28, 27),
        _example("wrong_mf.jpg", 0, 1, 45, 44),
        _example("wrong_fm.jpg", 1, 0, 36, 38),
        _example("old_m.jpg", 0, 0, 72, 50),
        _example("old_f.jpg", 1, 1, 81, 55),
    ]

    selected = select_examples(examples, seed=42)

    assert len(selected) == 6
    assert len({item.example.sample_name for item in selected}) == 6
    assert selected[0].example.gender_target == selected[0].example.gender_prediction == 0
    assert selected[1].example.gender_target == selected[1].example.gender_prediction == 1
    assert (selected[2].example.gender_target, selected[2].example.gender_prediction) == (0, 1)
    assert (selected[3].example.gender_target, selected[3].example.gender_prediction) == (1, 0)
    assert all(item.example.age_target >= 60 for item in selected[4:])
    assert all(item.example.absolute_age_error > 10 for item in selected[4:])
