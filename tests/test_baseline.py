import pandas as pd
import pytest

from recommender.baseline import PopularityRecommender


@pytest.fixture
def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Comedy", "Action", "Horror"],
        }
    )


@pytest.fixture
def training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 2, 3],
            "movieId": [10, 20, 20, 30, 30, 40],
            "rating": [5.0, 4.0, 5.0, 4.5, 3.0, 5.0],
        }
    )


def test_known_user_filters_seen_items_and_returns_metadata(training, catalog) -> None:
    result = PopularityRecommender().fit(training, catalog).recommend(1, k=3)

    assert result.movieId.tolist() == [40, 30]
    assert result.title.tolist() == ["D", "C"]
    assert result.genres.tolist() == ["Horror", "Action"]


def test_unknown_user_uses_popularity_order(training, catalog) -> None:
    result = PopularityRecommender().fit(training, catalog).recommend(999, k=4)

    assert result.movieId.tolist() == [20, 10, 40, 30]


def test_ranking_tie_breaks_by_mean_rating_then_movie_id(catalog) -> None:
    training = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4],
            "movieId": [10, 10, 20, 20],
            "rating": [4.0, 5.0, 4.0, 4.0],
        }
    )

    result = PopularityRecommender().fit(training, catalog).recommend(999, k=4)

    assert result.movieId.tolist() == [10, 20, 30, 40]


def test_evaluation_metrics(training, catalog) -> None:
    validation = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [30, 10, 99],
            "rating": [5.0, 5.0, 5.0],
        }
    )
    metrics = PopularityRecommender().fit(training, catalog).evaluate(validation)

    assert metrics.users_evaluated == 3
    assert metrics.recall_at_10 == pytest.approx(2 / 3)
    assert metrics.recall_at_50 == pytest.approx(2 / 3)
    assert metrics.ndcg_at_10 == pytest.approx(
        (1.0 + 1.0 / 1.584962500721156 + 0.0) / 3
    )


def test_evaluation_ignores_non_positive_validation_ratings(training, catalog) -> None:
    validation = pd.DataFrame(
        {"userId": [1], "movieId": [30], "rating": [3.5]}
    )

    metrics = PopularityRecommender().fit(training, catalog).evaluate(validation)

    assert metrics.users_evaluated == 0
