"""Run the training-only popularity baseline on MovieLens 25M."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from recommender.baseline import PopularityRecommender
from recommender.data import validate_dataset
from recommender.split import chronological_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data") / "ml-25m"
    )
    parser.add_argument("--results-path", type=Path, default=Path("results") / "baseline.json")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    validation = validate_dataset(args.dataset_root)
    ratings = pd.read_csv(
        args.dataset_root / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        usecols=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(args.dataset_root / "movies.csv")
    split = chronological_split(ratings)
    recommender = PopularityRecommender().fit(split.train, movies)
    metrics = recommender.evaluate(split.validation)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": {
            "root": str(validation.root),
            "ratings_rows": validation.ratings_rows,
            "movies_rows": validation.movies_rows,
        },
        "split": {
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
        },
        "positive_rating_threshold": recommender.positive_rating_threshold,
        "metrics": metrics.as_dict(),
    }
    args.results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.user_id is not None:
        print(recommender.recommend(args.user_id, args.k).to_string(index=False))


if __name__ == "__main__":
    main()
