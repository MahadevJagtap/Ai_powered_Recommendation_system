import pandas as pd
import pytest

from recommender.evaluation import build_evaluation_users, evaluate_recommendations


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [5.0, 4.0, 5.0, 4.0],
            "timestamp": [1, 2, 1, 2],
        }
    )
    validation = pd.DataFrame(
        {
            "userId": [1, 1, 2, 3],
            "movieId": [30, 40, 20, 10],
            "rating": [5.0, 5.0, 3.0, 5.0],
            "timestamp": [3, 4, 3, 3],
        }
    )
    return training, validation


def test_evaluation_users_align_candidates_and_training_users() -> None:
    training, validation = _frames()
    selected = build_evaluation_users(
        training,
        validation,
        [10, 20, 30],
        user_limit=10,
        seed=42,
        full_catalog_size=4,
    )
    assert selected.user_ids == [1]
    assert selected.relevant_by_user == {1: {30}}
    assert selected.users_considered == 3
    assert selected.users_with_eligible_relevant_items == 1


def test_shared_metrics_use_ranked_candidates() -> None:
    training, validation = _frames()
    selected = build_evaluation_users(
        training,
        validation,
        [10, 20, 30, 40],
        user_limit=10,
        seed=42,
        full_catalog_size=4,
    )
    metrics = evaluate_recommendations(
        lambda user_id, k: [30, 20, 10],
        selected,
        full_catalog_size=4,
    )
    assert metrics.users_considered == 3
    assert metrics.users_with_eligible_relevant_items == 1
    assert metrics.users_evaluated == 1
    assert metrics.recall_at_10 == 0.5
    assert metrics.recall_at_50 == 0.5
    assert metrics.catalog_coverage == 1.0


def test_shared_metrics_reject_out_of_catalog_or_duplicate_results() -> None:
    training, validation = _frames()
    selected = build_evaluation_users(
        training,
        validation,
        [10, 20, 30],
        user_limit=10,
        seed=42,
        full_catalog_size=4,
    )
    with pytest.raises(ValueError, match="outside"):
        evaluate_recommendations(
            lambda user_id, k: [40],
            selected,
            full_catalog_size=4,
        )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_recommendations(
            lambda user_id, k: [30, 30],
            selected,
            full_catalog_size=4,
        )


def test_evaluation_counts_excluded_users_without_eligible_positive_items() -> None:
    training = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 20], "rating": [5.0, 5.0]}
    )
    validation = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4],
            "movieId": [30, 40, 10, 50],
            "rating": [5.0, 3.0, 5.0, 5.0],
        }
    )
    selected = build_evaluation_users(
        training,
        validation,
        [10, 20, 30],
        user_limit=10,
        seed=42,
        full_catalog_size=5,
    )

    assert selected.users_considered == 4
    assert selected.users_with_eligible_relevant_items == 1
    assert selected.user_ids == [1]
