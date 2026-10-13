import pandas as pd
import pytest
from math import log2
from recommender.split import chronological_split

try:
    from recommender.retrieval import TwoTowerRetriever, calculate_retrieval_metrics
except ImportError as error:
    pytest.skip(f"TensorFlow runtime unavailable: {error}", allow_module_level=True)


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _fitted() -> TwoTowerRetriever:
    training, movies = _data()
    return TwoTowerRetriever(dimension=4, seed=7).fit(
        training, movies, sample_size=None, epochs=1, batch_size=2
    )


def test_mappings_and_candidate_generation() -> None:
    retriever = _fitted()
    assert retriever.user_ids == [1, 2, 3]
    assert retriever.movie_ids == [10, 20, 30, 40]
    result = retriever.recommend(1, k=2)
    assert len(result) == 2
    assert set(result["movieId"]).isdisjoint({10, 20})


def test_candidate_scores_have_stable_descending_seen_filtered_order() -> None:
    ranked = TwoTowerRetriever.rank_candidates(
        movie_ids=[10, 20, 30, 40],
        scores=[0.2, 0.9, 0.9, 0.1],
        seen=[20],
        k=3,
    )
    assert ranked == [30, 10, 40]


def test_candidate_score_length_is_validated() -> None:
    with pytest.raises(ValueError, match="same length"):
        TwoTowerRetriever.rank_candidates([10, 20], [1.0], [], 1)


def test_metric_calculation_is_deterministic() -> None:
    metrics = calculate_retrieval_metrics(
        {1: [30, 10, 20], 2: [40, 50, 60]},
        {1: {20, 30}, 2: {50}},
    )
    assert metrics.users_evaluated == 2
    assert metrics.recall_at_10 == 1.0
    assert metrics.recall_at_50 == 1.0
    first = (1.0 + 1 / log2(4)) / (1.0 + 1 / log2(3))
    assert metrics.ndcg_at_10 == (first + 1 / log2(3)) / 2


def test_training_vocabularies_exclude_future_interactions() -> None:
    interactions = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 20, 30, 10, 20, 40],
            "rating": [5.0] * 6,
            "timestamp": [1, 2, 3, 1, 2, 3],
        }
    )
    split = chronological_split(interactions)
    movies = pd.DataFrame(
        {"movieId": [10, 20, 30, 40], "title": list("ABCD"), "genres": ["Drama"] * 4}
    )
    retriever = TwoTowerRetriever(dimension=2, seed=1).fit(
        split.train, movies, sample_size=None, epochs=1, batch_size=2
    )
    assert set(retriever.user_ids) == set(split.train["userId"])
    assert set(retriever.movie_ids) == set(split.train["movieId"])
    assert set(retriever.movie_ids).isdisjoint(
        set(split.validation["movieId"]) - set(split.train["movieId"])
    )


def test_unknown_user_uses_popularity_fallback() -> None:
    retriever = _fitted()
    result = retriever.recommend(999, k=2)
    assert result["movieId"].tolist() == [20, 30]


def test_save_reload_preserves_candidates(tmp_path) -> None:
    retriever = _fitted()
    before = retriever.recommend(1, k=3)["movieId"].tolist()
    retriever.save(tmp_path)
    loaded = TwoTowerRetriever.load(tmp_path)
    assert loaded.item_embeddings is not None
    assert loaded.recommend(1, k=3)["movieId"].tolist() == before
    assert loaded.recommend(999, k=2)["movieId"].tolist() == [20, 30]


def test_evaluation_reports_same_positive_user_denominator() -> None:
    retriever = _fitted()
    validation = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4],
            "movieId": [30, 40, 50, 10],
            "rating": [5.0, 5.0, 3.0, 5.0],
        }
    )
    metrics = retriever.evaluate(validation)
    assert metrics.users_evaluated == 3
    assert 0 <= metrics.recall_at_10 <= 1
    assert 0 <= metrics.recall_at_50 <= 1
    assert 0 <= metrics.ndcg_at_10 <= 1
