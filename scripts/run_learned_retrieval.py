"""Train and evaluate the TensorFlow Recommenders two-tower retriever."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import tensorflow as tf

from recommender.baseline import PopularityRecommender
from recommender.data import validate_dataset
from recommender.retrieval import TwoTowerRetriever
from recommender.split import chronological_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "ml-25m")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "learned_retrieval")
    parser.add_argument("--results-path", type=Path, default=Path("results") / "learned_retrieval.json")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--validation-user-limit", type=int, default=1000)
    parser.add_argument(
        "--ratings-row-limit",
        type=int,
        default=0,
        help="Read only this many ratings for a bounded smoke run; use 0 for all rows.",
    )
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip the full CSV validation scan for bounded smoke runs.",
    )
    args = parser.parse_args()

    def progress(message: str) -> None:
        print(f"[learned-retrieval-cli] {message}", flush=True)

    started = time.perf_counter()
    progress("starting")
    validation = None
    if args.skip_dataset_validation:
        progress("skipping full dataset validation by request")
    else:
        progress("validating dataset")
        validation = validate_dataset(args.dataset_root)
    progress("loading ratings and movies")
    read_kwargs = {
        "usecols": ["userId", "movieId", "rating", "timestamp"],
        "dtype": {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    }
    if args.ratings_row_limit:
        read_kwargs["nrows"] = args.ratings_row_limit
    ratings = pd.read_csv(args.dataset_root / "ratings.csv", **read_kwargs)
    movies = pd.read_csv(args.dataset_root / "movies.csv")
    progress(f"loaded {len(ratings):,} ratings and {len(movies):,} movies")
    progress("creating chronological split")
    split = chronological_split(ratings)
    sample_size = None if args.sample_size == 0 else args.sample_size
    user_limit = None if args.validation_user_limit == 0 else args.validation_user_limit

    retriever = TwoTowerRetriever(dimension=args.dimension, seed=args.seed)
    retriever.fit(
        split.train,
        movies,
        sample_size=sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    progress("evaluating learned model")
    learned_metrics = retriever.evaluate(split.validation, user_limit=user_limit)
    baseline_validation = split.validation
    if user_limit is not None:
        positive_users = (
            split.validation.loc[split.validation["rating"] >= 4.0, "userId"]
            .drop_duplicates()
            .iloc[:user_limit]
        )
        baseline_validation = split.validation.loc[
            split.validation["userId"].isin(positive_users)
        ]
    progress("evaluating popularity baseline on the same users and candidate universe")
    candidate_movies = movies.loc[movies["movieId"].isin(retriever.movie_ids)].copy()
    baseline_metrics = PopularityRecommender().fit(split.train, candidate_movies).evaluate(
        baseline_validation
    )
    progress("saving artifacts")
    retriever.save(args.output_dir)
    progress("reloading artifacts")
    reloaded = TwoTowerRetriever.load(args.output_dir)
    user_for_reload = split.train["userId"].iloc[0]
    reload_check = reloaded.recommend(user_for_reload, 5)["movieId"].tolist() == retriever.recommend(
        user_for_reload, 5
    )["movieId"].tolist()
    payload = {
        "framework": {
            "backend": "tensorflow-recommenders-two-tower",
            "tensorflow_version": tf.__version__,
        },
        "training_mode": "chronological_training_partition_only",
        "validation_mode": "full" if user_limit is None else "smoke",
        "validation_user_limit": user_limit,
        "dataset": {
            "ratings_rows": len(ratings) if validation is None else validation.ratings_rows,
            "movies_rows": len(movies) if validation is None else validation.movies_rows,
        },
        "split": {"train_rows": len(split.train), "validation_rows": len(split.validation)},
        "learned_metrics": learned_metrics.as_dict(),
        "popularity_baseline_metrics": baseline_metrics.as_dict(),
        "candidate_universe": {
            "learned_movie_count": len(retriever.movie_ids),
            "baseline_movie_count": len(candidate_movies),
            "definition": "movies present in chronological training vocabulary",
        },
        "artifact_reload_check": reload_check,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    progress(f"wrote results to {args.results_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
