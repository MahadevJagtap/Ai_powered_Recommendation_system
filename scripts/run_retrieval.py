"""Train and evaluate the NumPy co-occurrence retrieval fallback."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from recommender.data import validate_dataset
from recommender.numpy_retrieval import CooccurrenceRetriever
from recommender.split import chronological_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "ml-25m")
    parser.add_argument("--results-path", type=Path, default=Path("results") / "retrieval.json")
    parser.add_argument("--artifacts-path", type=Path, default=Path("artifacts") / "retrieval")
    parser.add_argument(
        "--validation-user-limit",
        type=int,
        default=1000,
        help="Evaluate this many validation users; use 0 for the full validation set.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    validation = validate_dataset(args.dataset_root)
    ratings = pd.read_csv(args.dataset_root / "ratings.csv")
    movies = pd.read_csv(args.dataset_root / "movies.csv")
    split = chronological_split(ratings)
    retriever = CooccurrenceRetriever()
    retriever.fit(split.train, movies)
    user_limit = None if args.validation_user_limit == 0 else args.validation_user_limit
    metrics = retriever.evaluate(split.validation, user_limit=user_limit)
    retriever.save(args.artifacts_path)
    payload = {
        "framework": {"backend": "numpy-cooccurrence-fallback"},
        "tfrs_status": "not used: TensorFlow native Windows runtime was unstable",
        "training_mode": "full_training_partition",
        "validation_mode": "full" if user_limit is None else "smoke",
        "validation_user_limit": user_limit,
        "dataset": {"ratings_rows": validation.ratings_rows, "movies_rows": validation.movies_rows},
        "split": {"train_rows": len(split.train), "validation_rows": len(split.validation)},
        "positive_rating_threshold": 4.0,
        "metrics": metrics.as_dict(),
        "training_duration_seconds": round(time.perf_counter() - started, 3),
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
