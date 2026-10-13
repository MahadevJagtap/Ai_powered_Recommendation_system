"""Application service that exposes the validated recommendation backends."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recommender.baseline import PopularityRecommender
from recommender.numpy_retrieval import CooccurrenceRetriever


@dataclass(frozen=True)
class RecommendationResponse:
    recommendations: list[dict[str, object]]
    source: str
    model_version: str
    explanations: list[dict[str, object]]


class RecommendationService:
    """Coordinate baseline and NumPy retrieval recommendations."""

    def __init__(self, model_version: str = "local-v1") -> None:
        self.model_version = model_version
        self.baseline: PopularityRecommender | None = None
        self.retrieval: CooccurrenceRetriever | None = None

    def fit(self, training: pd.DataFrame, movies: pd.DataFrame) -> "RecommendationService":
        self.baseline = PopularityRecommender().fit(training, movies)
        self.retrieval = CooccurrenceRetriever().fit(training, movies)
        return self

    def _require_fitted(self) -> tuple[PopularityRecommender, CooccurrenceRetriever]:
        if self.baseline is None or self.retrieval is None:
            raise RuntimeError("RecommendationService must be fitted first")
        return self.baseline, self.retrieval

    def recommend(
        self, user_id: int, k: int = 10, method: str = "retrieval"
    ) -> RecommendationResponse:
        if k <= 0:
            raise ValueError("k must be positive")
        if k > 100:
            raise ValueError("k must be at most 100")
        if method not in {"baseline", "retrieval"}:
            raise ValueError("method must be 'baseline' or 'retrieval'")
        baseline, retrieval = self._require_fitted()
        if method == "baseline":
            result = baseline.recommend(user_id, k)
            source = "popularity_fallback" if user_id not in baseline._seen_by_user else "baseline"
        else:
            result = retrieval.recommend(user_id, k)
            has_history = user_id in retrieval.positive_seen
            has_related = has_history and any(
                item in retrieval.related for item in retrieval.positive_seen[user_id]
            )
            source = "personalized_cooccurrence" if has_related else "popularity_fallback"
        recommendations = result.to_dict(orient="records")
        if method == "retrieval" and source == "personalized_cooccurrence":
            explanations = retrieval.explain(
                user_id, [item["movieId"] for item in recommendations]
            )
        elif method == "baseline":
            explanations = [
                {
                    "movieId": item["movieId"],
                    "algorithm": "positive training interactions",
                    "positive_interactions": item.get("positive_interactions", 0),
                }
                for item in recommendations
            ]
        else:
            explanations = []
        return RecommendationResponse(
            recommendations=recommendations,
            source=source,
            model_version=self.model_version,
            explanations=explanations,
        )
