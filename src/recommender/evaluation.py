"""Shared, leakage-safe evaluation for recommendation methods."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from recommender.baseline import POSITIVE_RATING_THRESHOLD


@dataclass(frozen=True)
class EvaluationUsers:
    """Persistable definition of the users and relevant items being evaluated."""

    user_ids: list[object]
    relevant_by_user: dict[object, set[object]]
    candidate_movie_ids: list[object]
    users_considered: int
    users_with_eligible_relevant_items: int


@dataclass(frozen=True)
class SharedEvaluation:
    users_considered: int
    users_with_eligible_relevant_items: int
    users_evaluated: int
    candidate_catalog_size: int
    catalog_coverage: float
    recall_at_10: float
    recall_at_50: float
    ndcg_at_10: float
    per_user: list[dict[str, float]]

    @property
    def users_with_relevant_items(self) -> int:
        """Backward-compatible alias for eligible relevant validation users."""
        return self.users_with_eligible_relevant_items

    def as_dict(self) -> dict[str, float | int]:
        return {
            "users_considered": self.users_considered,
            "users_with_eligible_relevant_items": self.users_with_eligible_relevant_items,
            "users_evaluated": self.users_evaluated,
            "users_with_relevant_items": self.users_with_eligible_relevant_items,
            "candidate_catalog_size": self.candidate_catalog_size,
            "catalog_coverage": self.catalog_coverage,
            "recall_at_10": self.recall_at_10,
            "recall_at_50": self.recall_at_50,
            "ndcg_at_10": self.ndcg_at_10,
            "per_user": self.per_user,
        }


def build_evaluation_users(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    candidate_movie_ids: Iterable[object],
    *,
    user_limit: int,
    seed: int,
    full_catalog_size: int,
) -> EvaluationUsers:
    """Select a deterministic positive-validation user sample.

    Relevance is restricted to the shared candidate universe. Users with no
    eligible relevant item are excluded from the evaluated population.
    """

    if user_limit <= 0:
        raise ValueError("user_limit must be positive")
    candidates = set(candidate_movie_ids)
    training_users = set(training["userId"])
    users_considered = validation["userId"].nunique()
    positive = validation.loc[
        (validation["rating"] >= POSITIVE_RATING_THRESHOLD)
        & validation["movieId"].isin(candidates)
        & validation["userId"].isin(training_users)
    ]
    relevant = {
        user_id: set(rows["movieId"])
        for user_id, rows in positive.groupby("userId", sort=True)
    }
    rng = np.random.default_rng(seed)
    users = sorted(relevant)
    users_with_eligible_relevant_items = len(users)
    if len(users) > user_limit:
        users = sorted(rng.choice(users, size=user_limit, replace=False).tolist())
    return EvaluationUsers(
        user_ids=users,
        relevant_by_user={user_id: relevant[user_id] for user_id in users},
        candidate_movie_ids=sorted(candidates),
        users_considered=users_considered,
        users_with_eligible_relevant_items=users_with_eligible_relevant_items,
    )


def evaluate_recommendations(
    recommend: Callable[[object, int], list[object]],
    evaluation_users: EvaluationUsers,
    *,
    full_catalog_size: int,
) -> SharedEvaluation:
    """Evaluate a recommender against a fixed user/relevance definition."""

    if not evaluation_users.user_ids:
        return SharedEvaluation(
            users_considered=evaluation_users.users_considered,
            users_with_eligible_relevant_items=evaluation_users.users_with_eligible_relevant_items,
            users_evaluated=0,
            candidate_catalog_size=len(evaluation_users.candidate_movie_ids),
            catalog_coverage=0.0,
            recall_at_10=0.0,
            recall_at_50=0.0,
            ndcg_at_10=0.0,
            per_user=[],
        )
    recalls_10: list[float] = []
    recalls_50: list[float] = []
    ndcgs_10: list[float] = []
    per_user: list[dict[str, float]] = []
    candidate_set = set(evaluation_users.candidate_movie_ids)
    for user_id in evaluation_users.user_ids:
        recommendations = recommend(user_id, 50)[:50]
        if any(movie_id not in candidate_set for movie_id in recommendations):
            raise ValueError("recommender returned an item outside the shared candidate catalog")
        if len(recommendations) != len(set(recommendations)):
            raise ValueError("recommender returned duplicate candidate IDs")
        relevant = evaluation_users.relevant_by_user[user_id]
        recalls_10.append(len(set(recommendations[:10]) & relevant) / len(relevant))
        recalls_50.append(len(set(recommendations) & relevant) / len(relevant))
        dcg = sum(
            1 / log2(rank + 2)
            for rank, movie_id in enumerate(recommendations[:10])
            if movie_id in relevant
        )
        ideal_count = min(len(relevant), 10)
        ideal_dcg = sum(1 / log2(rank + 2) for rank in range(ideal_count))
        ndcgs_10.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        per_user.append(
            {
                "user_id": user_id,
                "recall_at_10": recalls_10[-1],
                "recall_at_50": recalls_50[-1],
                "ndcg_at_10": ndcgs_10[-1],
            }
        )
    return SharedEvaluation(
        users_considered=evaluation_users.users_considered,
        users_with_eligible_relevant_items=evaluation_users.users_with_eligible_relevant_items,
        users_evaluated=len(evaluation_users.user_ids),
        candidate_catalog_size=len(evaluation_users.candidate_movie_ids),
        catalog_coverage=(
            len(evaluation_users.candidate_movie_ids) / full_catalog_size
        ),
        recall_at_10=float(np.mean(recalls_10)),
        recall_at_50=float(np.mean(recalls_50)),
        ndcg_at_10=float(np.mean(ndcgs_10)),
        per_user=per_user,
    )
