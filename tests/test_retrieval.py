import pandas as pd
import pytest

from recommender.numpy_retrieval import CooccurrenceRetriever


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 20, 30, 30, 40],
            "rating": [5.0, 4.0, 5.0, 4.0, 5.0, 4.5],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50],
            "title": list("ABCDE"),
            "genres": ["Drama"] * 5,
        }
    )
    return training, movies


def test_mapping_and_seen_filtering(data) -> None:
    training, movies = data
    retriever = CooccurrenceRetriever().fit(training, movies)
    result = retriever.recommend(1, 3)
    assert set(result["movieId"]).isdisjoint({10, 20})
    assert set(result["movieId"]).issubset({30, 40, 50})
    assert all(movie_id in movies.movieId.values for movie_id in result.movieId)


def test_unknown_user_uses_fallback(data) -> None:
    training, movies = data
    retriever = CooccurrenceRetriever().fit(training, movies)
    result = retriever.recommend(999, 2)
    assert len(result) == 2
    assert result.movieId.tolist() == [20, 30]


def test_low_rated_history_does_not_seed_personalized_retrieval() -> None:
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
    retriever = CooccurrenceRetriever().fit(training, movies)

    assert retriever.recommend(1, 2)["movieId"].tolist() == [20, 30]
    assert retriever.explain(1, [20, 30]) == [
        {"movieId": 20, "contributors": []},
        {"movieId": 30, "contributors": []},
    ]


def test_unknown_training_movie_ids_are_not_recommended(data) -> None:
    training, movies = data
    training = pd.concat(
        [
            training,
            pd.DataFrame(
                {"userId": [1, 4], "movieId": [999, 999], "rating": [5.0, 5.0]}
            ),
        ],
        ignore_index=True,
    )
    retriever = CooccurrenceRetriever().fit(training, movies)

    result = retriever.recommend(1, 10)

    assert 999 not in result["movieId"].tolist()
    assert set(result["movieId"]).issubset(set(movies["movieId"]))


def test_retrieval_evaluation(data) -> None:
    training, movies = data
    validation = pd.DataFrame(
        {"userId": [1, 2], "movieId": [30, 40], "rating": [5.0, 5.0]}
    )
    retriever = CooccurrenceRetriever().fit(training, movies)
    metrics = retriever.evaluate(validation)
    assert metrics.users_evaluated == 2
    assert 0 <= metrics.recall_at_10 <= 1
    assert 0 <= metrics.recall_at_50 <= 1


def test_evaluation_user_limit(data) -> None:
    training, movies = data
    validation = pd.DataFrame(
        {"userId": [1, 2], "movieId": [30, 40], "rating": [5.0, 5.0]}
    )
    retriever = CooccurrenceRetriever().fit(training, movies)
    metrics = retriever.evaluate(validation, user_limit=1)
    assert metrics.users_evaluated == 1
