"""Compare popularity, NumPy, and TFRS retrieval on one persisted sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import numpy as np

from recommender.baseline import PopularityRecommender
from recommender.data import validate_dataset
from recommender.evaluation import (
    build_evaluation_users,
    evaluate_recommendations,
)
from recommender.numpy_retrieval import CooccurrenceRetriever
from recommender.split import chronological_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "ml-25m")
    parser.add_argument("--ratings-row-limit", type=int, default=25_000)
    parser.add_argument(
        "--sampling-strategy",
        choices=["prefix", "uniform-users"],
        default="prefix",
    )
    parser.add_argument("--sample-users", type=int, default=2_000)
    parser.add_argument("--validation-user-limit", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-seeds",
        default=None,
        help="Comma-separated user-selection seeds; models are trained once.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "shared_evaluation")
    parser.add_argument("--skip-dataset-validation", action="store_true")
    args = parser.parse_args()
    import tensorflow as tf

    from recommender.retrieval import TwoTowerRetriever

    def progress(message: str) -> None:
        print(f"[shared-evaluation] {message}", flush=True)

    if args.ratings_row_limit <= 0 or args.validation_user_limit <= 0 or args.sample_users <= 0:
        raise ValueError("row limit, sample users, and validation-user-limit must be positive")
    evaluation_seeds = (
        [args.seed]
        if args.evaluation_seeds is None
        else [int(value) for value in args.evaluation_seeds.split(",") if value.strip()]
    )
    if not evaluation_seeds:
        raise ValueError("evaluation-seeds must contain at least one integer")
    started = time.perf_counter()
    progress(f"loading ratings with {args.sampling_strategy} sampling")
    validation = None if args.skip_dataset_validation else validate_dataset(args.dataset_root)
    read_options = {
        "usecols": ["userId", "movieId", "rating", "timestamp"],
        "dtype": {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    }
    if args.sampling_strategy == "prefix":
        ratings = pd.read_csv(
            args.dataset_root / "ratings.csv", nrows=args.ratings_row_limit, **read_options
        )
        selected_source_users = None
    else:
        user_chunks = pd.read_csv(
            args.dataset_root / "ratings.csv",
            usecols=["userId"],
            dtype={"userId": "int32"},
            chunksize=500_000,
        )
        all_users = pd.concat(user_chunks, ignore_index=True)["userId"].unique()
        rng = np.random.default_rng(args.seed)
        if args.sample_users > len(all_users):
            raise ValueError("sample-users exceeds the number of dataset users")
        selected_source_users = sorted(
            rng.choice(all_users, size=args.sample_users, replace=False).tolist()
        )
        rating_chunks = pd.read_csv(
            args.dataset_root / "ratings.csv", chunksize=500_000, **read_options
        )
        selected_set = set(selected_source_users)
        ratings = pd.concat(
            (chunk[chunk["userId"].isin(selected_set)] for chunk in rating_chunks),
            ignore_index=True,
        )
    movies = pd.read_csv(args.dataset_root / "movies.csv")
    split = chronological_split(ratings)
    candidate_ids = sorted(split.train["movieId"].unique().tolist())
    users = build_evaluation_users(
        split.train, split.validation, candidate_ids,
        user_limit=args.validation_user_limit, seed=evaluation_seeds[0],
        full_catalog_size=len(movies),
    )
    progress(
        f"selected {len(users.user_ids)} users, {len(users.candidate_movie_ids)} candidates, "
        f"{len(split.train):,} training rows"
    )
    candidate_movies = movies.loc[movies["movieId"].isin(users.candidate_movie_ids)].copy()
    methods: dict[str, object] = {}
    timings: dict[str, float] = {}

    phase = time.perf_counter()
    baseline = PopularityRecommender().fit(split.train, candidate_movies)
    methods["popularity"] = baseline
    timings["popularity_fit_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    numpy_retriever = CooccurrenceRetriever().fit(split.train, candidate_movies)
    methods["numpy_cooccurrence"] = numpy_retriever
    timings["numpy_fit_seconds"] = time.perf_counter() - phase

    phase = time.perf_counter()
    learned = TwoTowerRetriever(dimension=args.dimension, seed=args.seed).fit(
        split.train,
        candidate_movies,
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    methods["tfrs_two_tower"] = learned
    timings["tfrs_fit_seconds"] = time.perf_counter() - phase

    def bootstrap_interval(values: list[float], seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        array = np.asarray(values, dtype=float)
        samples = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
        return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]

    def paired_bootstrap_interval(
        left: list[float], right: list[float], seed: int
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        samples = rng.choice(
            differences, size=(2000, len(differences)), replace=True
        ).mean(axis=1)
        return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]

    evaluation_users_by_seed = {
        str(evaluation_seed): build_evaluation_users(
            split.train, split.validation, candidate_ids,
            user_limit=args.validation_user_limit, seed=evaluation_seed,
            full_catalog_size=len(movies),
        )
        for evaluation_seed in evaluation_seeds
    }
    results_by_seed: dict[str, Any] = {}
    for evaluation_seed in evaluation_seeds:
        users = evaluation_users_by_seed[str(evaluation_seed)]
        results_by_seed[str(evaluation_seed)] = {}
        for name, method in methods.items():
            progress(f"evaluating {name} with user seed {evaluation_seed}")
            phase = time.perf_counter()
            result = evaluate_recommendations(
                lambda user_id, k, model=method: model.recommend(user_id, k)["movieId"].tolist(),
                users, full_catalog_size=len(movies),
            )
            result_dict = result.as_dict()
            result_dict["recall_at_10_ci95"] = bootstrap_interval(
                [row["recall_at_10"] for row in result.per_user], evaluation_seed
            )
            result_dict["recall_at_50_ci95"] = bootstrap_interval(
                [row["recall_at_50"] for row in result.per_user], evaluation_seed + 1
            )
            result_dict["ndcg_at_10_ci95"] = bootstrap_interval(
                [row["ndcg_at_10"] for row in result.per_user], evaluation_seed + 2
            )
            results_by_seed[str(evaluation_seed)][name] = result_dict
            timings[f"{name}_evaluation_seed_{evaluation_seed}_seconds"] = (
                time.perf_counter() - phase
            )

        popularity_rows = results_by_seed[str(evaluation_seed)]["popularity"]["per_user"]
        for name in ("numpy_cooccurrence", "tfrs_two_tower"):
            method_rows = results_by_seed[str(evaluation_seed)][name]["per_user"]
            results_by_seed[str(evaluation_seed)][name]["paired_difference_vs_popularity"] = {
                "recall_at_10_mean": results_by_seed[str(evaluation_seed)][name]["recall_at_10"]
                - results_by_seed[str(evaluation_seed)]["popularity"]["recall_at_10"],
                "recall_at_10_ci95": paired_bootstrap_interval(
                    [row["recall_at_10"] for row in method_rows],
                    [row["recall_at_10"] for row in popularity_rows],
                    evaluation_seed + 10,
                ),
                "recall_at_50_mean": results_by_seed[str(evaluation_seed)][name]["recall_at_50"]
                - results_by_seed[str(evaluation_seed)]["popularity"]["recall_at_50"],
                "recall_at_50_ci95": paired_bootstrap_interval(
                    [row["recall_at_50"] for row in method_rows],
                    [row["recall_at_50"] for row in popularity_rows],
                    evaluation_seed + 11,
                ),
                "ndcg_at_10_mean": results_by_seed[str(evaluation_seed)][name]["ndcg_at_10"]
                - results_by_seed[str(evaluation_seed)]["popularity"]["ndcg_at_10"],
                "ndcg_at_10_ci95": paired_bootstrap_interval(
                    [row["ndcg_at_10"] for row in method_rows],
                    [row["ndcg_at_10"] for row in popularity_rows],
                    evaluation_seed + 12,
                ),
            }

    first_results = results_by_seed[str(evaluation_seeds[0])]

    payload = {
        "experiment": {
            "ratings_row_limit": (
                args.ratings_row_limit if args.sampling_strategy == "prefix" else None
            ),
            "sampling_strategy": args.sampling_strategy,
            "sample_users": args.sample_users if args.sampling_strategy == "uniform-users" else None,
            "validation_user_limit": args.validation_user_limit,
            "sample_size": args.sample_size,
            "dimension": args.dimension,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "evaluation_seeds": evaluation_seeds,
            "positive_rating_threshold": 4.0,
        },
        "dataset": {
            "ratings_loaded": len(ratings),
            "ratings_total": None if validation is None else validation.ratings_rows,
            "movies_total": len(movies),
            "selected_source_users": selected_source_users,
        },
        "split": {
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
        },
        "evaluation_users": {
            seed: evaluation_users_by_seed[str(seed)].user_ids
            for seed in evaluation_seeds
        },
        "evaluation_population_by_seed": {
            str(seed): {
                "users_considered": evaluation_users_by_seed[str(seed)].users_considered,
                "users_with_eligible_relevant_items": (
                    evaluation_users_by_seed[str(seed)].users_with_eligible_relevant_items
                ),
                "users_evaluated": len(evaluation_users_by_seed[str(seed)].user_ids),
            }
            for seed in evaluation_seeds
        },
        "relevance_by_evaluation_seed": {
            str(seed): {
                str(user_id): sorted(
                    evaluation_users_by_seed[str(seed)].relevant_by_user[user_id]
                )
                for user_id in evaluation_users_by_seed[str(seed)].user_ids
            }
            for seed in evaluation_seeds
        },
        "candidate_movie_ids": users.candidate_movie_ids,
        "methods": first_results,
        "methods_by_evaluation_seed": results_by_seed,
        "temporal_cutoffs": {
            name: {
                "min_timestamp": int(frame["timestamp"].min()) if not frame.empty else None,
                "max_timestamp": int(frame["timestamp"].max()) if not frame.empty else None,
                "rows": len(frame),
            }
            for name, frame in {
                "all_selected": ratings,
                "train": split.train,
                "validation": split.validation,
                "test": split.test,
            }.items()
        },
        "timings_seconds": timings,
        "runtime_seconds": time.perf_counter() - started,
        "memory_observation": "Peak process memory not measured; no full-dataset training was started.",
        "tensorflow_version": tf.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    progress(f"wrote {args.output_dir / 'experiment.json'}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
