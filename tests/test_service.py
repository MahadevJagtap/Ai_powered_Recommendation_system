import pandas as pd
import pytest

from recommender.service import RecommendationService


@pytest.fixture
def service() -> RecommendationService:
    training = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 20, 30],
            "rating": [5.0, 4.0, 5.0, 4.0],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Comedy", "Action", "Horror"],
        }
    )
    return RecommendationService().fit(training, movies)


def test_service_returns_metadata_and_source(service: RecommendationService) -> None:
    response = service.recommend(1, k=2, method="retrieval")
    assert response.source == "personalized_cooccurrence"
    assert len(response.recommendations) == 2
    assert {"movieId", "title", "genres"} <= response.recommendations[0].keys()
    assert all(row["movieId"] not in {10, 20} for row in response.recommendations)


def test_service_unknown_user_uses_fallback(service: RecommendationService) -> None:
    response = service.recommend(999, k=2, method="retrieval")
    assert response.source == "popularity_fallback"
    assert len(response.recommendations) == 2


def test_service_low_rated_history_uses_popularity_fallback() -> None:
    training = pd.DataFrame(
        {
            "userId": [1, 2, 2, 3, 3],
            "movieId": [10, 10, 20, 10, 30],
            "rating": [2.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama"] * 4,
        }
    )
    response = RecommendationService().fit(training, movies).recommend(
        1, k=2, method="retrieval"
    )

    assert response.source == "popularity_fallback"
    assert response.explanations == []


def test_service_rejects_invalid_method(service: RecommendationService) -> None:
    with pytest.raises(ValueError, match="method"):
        service.recommend(1, method="ranker")
