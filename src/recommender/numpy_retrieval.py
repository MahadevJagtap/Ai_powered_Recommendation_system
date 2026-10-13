"""Dependency-free personalized retrieval fallback for Windows."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
import json

import pandas as pd

from recommender.baseline import POSITIVE_RATING_THRESHOLD, PopularityRecommender


@dataclass(frozen=True)
class RetrievalEvaluation:
    users_evaluated: int
    recall_at_10: float
    recall_at_50: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "users_evaluated": self.users_evaluated,
            "recall_at_10": self.recall_at_10,
            "recall_at_50": self.recall_at_50,
        }


class CooccurrenceRetriever:
    """Personalized item-item retrieval using positive training co-occurrence.

    This is a reliable CPU fallback, not a TensorFlow Recommenders model.
    """

    def __init__(self, max_positive_items_per_user: int = 50) -> None:
        if max_positive_items_per_user <= 1:
            raise ValueError("max_positive_items_per_user must be greater than 1")
        self.max_positive_items_per_user = max_positive_items_per_user
        self.catalog: pd.DataFrame | None = None
        self.catalog_by_id: pd.DataFrame | None = None
        self.seen: dict[object, set[object]] = {}
        self.positive_seen: dict[object, set[object]] = {}
        self.related: dict[object, dict[object, int]] = {}
        self.fallback: PopularityRecommender | None = None
        self.popular_ids: list[object] = []

    def fit(self, training: pd.DataFrame, movies: pd.DataFrame) -> "CooccurrenceRetriever":
        self.catalog = movies.copy()
        self.catalog_by_id = self.catalog.set_index("movieId")
        catalog_ids = set(self.catalog["movieId"])
        positive = training.loc[
            (training["rating"] >= POSITIVE_RATING_THRESHOLD)
            & training["movieId"].isin(catalog_ids)
        ]
        self.seen = {
            user: set(rows["movieId"])
            for user, rows in training.groupby("userId", sort=False)
        }
        self.positive_seen = {
            user: set(rows["movieId"])
            for user, rows in positive.groupby("userId", sort=False)
        }
        self.related = {}
        for _, rows in positive.groupby("userId", sort=False):
            # Bound pair generation for users with very long histories.
            items = sorted(set(rows["movieId"]))[: self.max_positive_items_per_user]
            for item in items:
                self.related.setdefault(item, {})
                for other in items:
                    if item != other:
                        self.related[item][other] = self.related[item].get(other, 0) + 1
        self.fallback = PopularityRecommender().fit(training, movies)
        self.popular_ids = self.fallback._require_fitted()["movieId"].tolist()
        return self

    def _recommend_ids(self, user_id: object, k: int) -> list[object]:
        if self.catalog is None or self.fallback is None:
            raise RuntimeError("CooccurrenceRetriever must be fitted first")
        if user_id not in self.seen:
            return [movie_id for movie_id in self.popular_ids if movie_id not in self.seen.get(user_id, set())][:k]
        seen = self.seen[user_id]
        positive_seen = self.positive_seen.get(user_id, set())
        scores: dict[object, int] = {}
        for item in positive_seen:
            for candidate, score in self.related.get(item, {}).items():
                if candidate not in seen:
                    scores[candidate] = scores.get(candidate, 0) + score
        ranked = [
            item
            for item, _ in heapq.nsmallest(
                k, scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]
        if len(ranked) < k:
            ranked.extend(item for item in self.popular_ids if item not in ranked and item not in seen)
        return ranked[:k]

    def recommend(self, user_id: object, k: int = 10) -> pd.DataFrame:
        if k <= 0:
            raise ValueError("k must be positive")
        self._require_catalog()
        ranked = self._recommend_ids(user_id, k)
        return self.catalog_by_id.loc[ranked].reset_index()

    def explain(self, user_id: object, movie_ids: list[object], limit: int = 3) -> list[dict[str, object]]:
        """Return actual seen-item contributions for co-occurrence candidates."""
        seen = self.seen.get(user_id, set())
        positive_seen = self.positive_seen.get(user_id, set())
        explanations: list[dict[str, object]] = []
        for movie_id in movie_ids:
            contributors = [
                {
                    "movieId": item,
                    "title": (
                        str(self.catalog_by_id.loc[item, "title"])
                        if self.catalog_by_id is not None and item in self.catalog_by_id.index
                        else str(item)
                    ),
                    "score": score,
                }
                for item in positive_seen
                for candidate, score in self.related.get(item, {}).items()
                if candidate == movie_id
            ]
            contributors.sort(key=lambda item: (-int(item["score"]), item["movieId"]))
            explanations.append({"movieId": movie_id, "contributors": contributors[:limit]})
        return explanations

    def _require_catalog(self) -> pd.DataFrame:
        if self.catalog is None or self.fallback is None:
            raise RuntimeError("CooccurrenceRetriever must be fitted first")
        return self.catalog

    def evaluate(
        self, validation: pd.DataFrame, user_limit: int | None = None
    ) -> RetrievalEvaluation:
        if user_limit is not None and user_limit <= 0:
            raise ValueError("user_limit must be positive")
        positive = validation.loc[validation["rating"] >= POSITIVE_RATING_THRESHOLD]
        relevant = {
            user: set(rows["movieId"])
            for user, rows in positive.groupby("userId", sort=False)
        }
        if user_limit is not None:
            relevant = dict(list(relevant.items())[:user_limit])
        if not relevant:
            return RetrievalEvaluation(0, 0.0, 0.0)
        r10, r50 = [], []
        for user, expected in relevant.items():
            top50_ids = self._recommend_ids(user, 50)
            top50 = set(top50_ids)
            top10 = set(top50_ids[:10])
            r10.append(len(top10 & expected) / len(expected))
            r50.append(len(top50 & expected) / len(expected))
        return RetrievalEvaluation(len(relevant), sum(r10) / len(r10), sum(r50) / len(r50))

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "cooccurrence.json").write_text(
            json.dumps(
                {
                    "backend": "numpy-cooccurrence",
                    "users": len(self.seen),
                    "items": len(self.catalog) if self.catalog is not None else 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
