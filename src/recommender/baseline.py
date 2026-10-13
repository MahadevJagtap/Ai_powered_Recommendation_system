"""Training-only popularity recommendation baseline and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable

import pandas as pd

from recommender.split import InteractionSplit


POSITIVE_RATING_THRESHOLD = 4.0
REQUIRED_RATING_COLUMNS = ("userId", "movieId", "rating")
REQUIRED_MOVIE_COLUMNS = ("movieId", "title", "genres")


@dataclass(frozen=True)
class EvaluationResult:
    """Macro-averaged ranking metrics for validation users."""

    users_evaluated: int
    recall_at_10: float
    recall_at_50: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "users_evaluated": self.users_evaluated,
            "recall_at_10": self.recall_at_10,
            "recall_at_50": self.recall_at_50,
            "ndcg_at_10": self.ndcg_at_10,
        }


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


class PopularityRecommender:
    """Recommend movies ordered by positive training-interaction popularity.

    A positive interaction is a rating greater than or equal to 4.0. Popularity
    counts, rating means, and user histories are all fitted from training data
    only. Movie metadata is used only to decorate returned recommendations.
    """

    def __init__(self, positive_rating_threshold: float = POSITIVE_RATING_THRESHOLD):
        if not 0 <= positive_rating_threshold <= 5:
            raise ValueError("positive_rating_threshold must be between 0 and 5")
        self.positive_rating_threshold = positive_rating_threshold
        self._ranked_movies: pd.DataFrame | None = None
        self._seen_by_user: dict[object, set[object]] = {}

    def fit(self, training: pd.DataFrame, movies: pd.DataFrame) -> "PopularityRecommender":
        _validate_columns(training, (*REQUIRED_RATING_COLUMNS,), "training")
        _validate_columns(movies, REQUIRED_MOVIE_COLUMNS, "movies")
        if training.empty:
            raise ValueError("training must not be empty")
        if movies.empty:
            raise ValueError("movies must not be empty")
        if movies["movieId"].duplicated().any():
            raise ValueError("movies contains duplicate movieId values")

        positive = training.loc[
            training["rating"] >= self.positive_rating_threshold,
            ["movieId", "rating"],
        ]
        stats = positive.groupby("movieId", as_index=False).agg(
            positive_interactions=("movieId", "size"),
            positive_mean_rating=("rating", "mean"),
        )
        ranked = movies.loc[:, list(REQUIRED_MOVIE_COLUMNS)].merge(
            stats, on="movieId", how="left"
        )
        ranked["positive_interactions"] = ranked["positive_interactions"].fillna(0).astype(
            "int64"
        )
        ranked["positive_mean_rating"] = ranked["positive_mean_rating"].fillna(0.0)
        self._ranked_movies = ranked.sort_values(
            ["positive_interactions", "positive_mean_rating", "movieId"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        self._seen_by_user = {
            user_id: set(user_rows["movieId"])
            for user_id, user_rows in training.groupby("userId", sort=False)
        }
        return self

    def _require_fitted(self) -> pd.DataFrame:
        if self._ranked_movies is None:
            raise RuntimeError("PopularityRecommender must be fitted before recommendation")
        return self._ranked_movies

    def _recommend_ids(self, user_id: object, k: int) -> list[object]:
        if k <= 0:
            raise ValueError("k must be positive")
        ranked = self._require_fitted()
        seen = self._seen_by_user.get(user_id, set())
        result: list[object] = []
        for movie_id in ranked["movieId"]:
            if movie_id not in seen:
                result.append(movie_id)
                if len(result) == k:
                    break
        return result

    def recommend(self, user_id: object, k: int = 10) -> pd.DataFrame:
        """Return up to ``k`` recommendations with original IDs and metadata."""

        ranked = self._require_fitted()
        ids = self._recommend_ids(user_id, k)
        if not ids:
            return ranked.iloc[0:0].copy()
        return ranked.set_index("movieId").loc[ids].reset_index()

    def evaluate(self, validation: pd.DataFrame) -> EvaluationResult:
        """Evaluate positive validation interactions using training-only ranking."""

        _validate_columns(validation, REQUIRED_RATING_COLUMNS, "validation")
        positive_validation = validation.loc[
            validation["rating"] >= self.positive_rating_threshold
        ]
        relevant_by_user = {
            user_id: set(user_rows["movieId"])
            for user_id, user_rows in positive_validation.groupby("userId", sort=False)
        }
        if not relevant_by_user:
            return EvaluationResult(0, 0.0, 0.0, 0.0)

        recall_10 = []
        recall_50 = []
        ndcg_10 = []
        for user_id, relevant in relevant_by_user.items():
            recommendations = self._recommend_ids(user_id, 50)
            hits_10 = len(set(recommendations[:10]) & relevant)
            hits_50 = len(set(recommendations) & relevant)
            recall_10.append(hits_10 / len(relevant))
            recall_50.append(hits_50 / len(relevant))
            dcg = sum(
                1 / log2(rank + 2)
                for rank, movie_id in enumerate(recommendations[:10])
                if movie_id in relevant
            )
            ideal_hits = min(len(relevant), 10)
            ideal_dcg = sum(1 / log2(rank + 2) for rank in range(ideal_hits))
            ndcg_10.append(dcg / ideal_dcg if ideal_dcg else 0.0)

        return EvaluationResult(
            users_evaluated=len(relevant_by_user),
            recall_at_10=sum(recall_10) / len(recall_10),
            recall_at_50=sum(recall_50) / len(recall_50),
            ndcg_at_10=sum(ndcg_10) / len(ndcg_10),
        )


def evaluate_baseline(split: InteractionSplit, movies: pd.DataFrame) -> EvaluationResult:
    """Fit the baseline on training only and evaluate on validation."""

    return PopularityRecommender().fit(split.train, movies).evaluate(split.validation)
