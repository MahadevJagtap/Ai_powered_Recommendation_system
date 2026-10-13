"""Validate the local MovieLens dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from recommender.data import validate_dataset


if __name__ == "__main__":
    result = validate_dataset(Path(__file__).parents[1] / "data" / "ml-25m")
    print(f"Validated {result.ratings_rows:,} ratings and {result.movies_rows:,} movies.")
