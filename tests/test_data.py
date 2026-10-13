from pathlib import Path

import pandas as pd
import pytest

from recommender.data import validate_dataset


def _write_dataset(root: Path, ratings: str, movies: str) -> None:
    (root / "ratings.csv").write_text(ratings, encoding="utf-8")
    (root / "movies.csv").write_text(movies, encoding="utf-8")


def test_validate_dataset_returns_row_counts(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        "userId,movieId,rating,timestamp\n1,10,4.0,100\n",
        "movieId,title,genres\n10,Example,Drama\n",
    )

    result = validate_dataset(tmp_path)

    assert result.ratings_rows == 1
    assert result.movies_rows == 1


def test_validate_dataset_rejects_invalid_header(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        "userId,movieId,rating,time\n1,10,4.0,100\n",
        "movieId,title,genres\n10,Example,Drama\n",
    )

    with pytest.raises(ValueError, match="expected"):
        validate_dataset(tmp_path)


def test_validate_dataset_rejects_missing_required_file(tmp_path: Path) -> None:
    (tmp_path / "movies.csv").write_text(
        "movieId,title,genres\n10,Example,Drama\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="ratings.csv"):
        validate_dataset(tmp_path)


def test_validate_dataset_rejects_invalid_rating_and_duplicate_movie(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        "userId,movieId,rating,timestamp\n1,10,6.0,100\n",
        "movieId,title,genres\n10,Example,Drama\n10,Duplicate,Drama\n",
    )

    with pytest.raises(ValueError, match="rating must be"):
        validate_dataset(tmp_path)
