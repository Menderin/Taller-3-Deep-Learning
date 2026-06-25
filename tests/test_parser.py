from pathlib import Path

import pytest

from src.data.parser import UTKFaceFilenameParser


def test_parser_extracts_utkface_labels() -> None:
    parser = UTKFaceFilenameParser()

    record = parser.parse(Path("25_1_2_20170116174525125.jpg"))

    assert record.age == 25.0
    assert record.gender == 1
    assert record.race == 2


@pytest.mark.parametrize(
    "filename",
    [
        "invalid.jpg",
        "25_7_2_date.jpg",
        "age_1_2_date.jpg",
    ],
)
def test_parser_rejects_invalid_names(filename: str) -> None:
    parser = UTKFaceFilenameParser()

    with pytest.raises(ValueError):
        parser.parse(filename)
