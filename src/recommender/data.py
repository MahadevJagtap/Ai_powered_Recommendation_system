"""Dataset discovery and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RATINGS_COLUMNS = ("userId", "movieId", "rating", "timestamp")
MOVIES_COLUMNS = ("movieId", "title", "genres")


@dataclass(frozen=True)
class DatasetValidation:
    """Summary returned after validating the required MovieLens files."""

    root: Path
    ratings_rows: int
    movies_rows: int


def _validate_header(path: Path, expected: tuple[str, ...]) -> None:
    """Validate a CSV header without loading the full file."""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    if tuple(header) != expected:
        raise ValueError(
            f"{path} has columns {header!r}; expected {list(expected)!r}"
        )


def validate_dataset(dataset_root: str | Path) -> DatasetValidation:
    """Validate required files and their basic structure.

    The full ratings file is read in chunks to verify row count and required
    value types without retaining the 25M-row dataset in memory.
    """

    root = Path(dataset_root)
    ratings_path = root / "ratings.csv"
    movies_path = root / "movies.csv"
    missing = [str(path) for path in (ratings_path, movies_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required dataset files are missing: {missing}")

    _validate_header(ratings_path, RATINGS_COLUMNS)
    _validate_header(movies_path, MOVIES_COLUMNS)

    ratings_rows = 0
    for chunk in pd.read_csv(ratings_path, usecols=list(RATINGS_COLUMNS), chunksize=250_000):
        if chunk.isna().any().any():
            raise ValueError("ratings.csv contains missing values in required columns")
        if not pd.api.types.is_numeric_dtype(chunk["userId"]):
            raise TypeError("ratings.csv userId must be numeric")
        if not pd.api.types.is_numeric_dtype(chunk["movieId"]):
            raise TypeError("ratings.csv movieId must be numeric")
        if not pd.api.types.is_numeric_dtype(chunk["rating"]):
            raise TypeError("ratings.csv rating must be numeric")
        if not pd.api.types.is_numeric_dtype(chunk["timestamp"]):
            raise TypeError("ratings.csv timestamp must be numeric")
        if (chunk[["userId", "movieId"]] <= 0).any().any():
            raise ValueError("ratings.csv userId and movieId must be positive")
        if not chunk["rating"].between(0.5, 5.0).all():
            raise ValueError("ratings.csv rating must be between 0.5 and 5.0")
        if (chunk["timestamp"] < 0).any():
            raise ValueError("ratings.csv timestamp must be non-negative")
        ratings_rows += len(chunk)

    movies = pd.read_csv(movies_path, usecols=list(MOVIES_COLUMNS))
    if movies.empty:
        raise ValueError("movies.csv is empty")
    if movies[["movieId", "title"]].isna().any().any():
        raise ValueError("movies.csv contains missing movieId or title values")
    if not pd.api.types.is_numeric_dtype(movies["movieId"]):
        raise TypeError("movies.csv movieId must be numeric")
    if (movies["movieId"] <= 0).any():
        raise ValueError("movies.csv movieId must be positive")
    if movies["movieId"].duplicated().any():
        raise ValueError("movies.csv contains duplicate movieId values")

    return DatasetValidation(root=root, ratings_rows=ratings_rows, movies_rows=len(movies))
