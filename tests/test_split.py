import pandas as pd
import pytest

from recommender.split import chronological_split


def test_split_is_chronological_per_user_without_leakage() -> None:
    interactions = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2],
            "movieId": [10, 11, 12, 13, 14, 20, 21],
            "timestamp": [5, 1, 4, 2, 3, 2, 1],
        }
    )

    split = chronological_split(interactions, validation_fraction=0.2, test_fraction=0.2)

    assert split.train[split.train.userId == 1].timestamp.tolist() == [1, 2, 3]
    assert split.validation[split.validation.userId == 1].timestamp.tolist() == [4]
    assert split.test[split.test.userId == 1].timestamp.tolist() == [5]
    assert split.train[split.train.userId == 2].timestamp.tolist() == [1, 2]
    assert set(split.train.movieId) | set(split.validation.movieId) | set(split.test.movieId) == set(
        interactions.movieId
    )

    for user_id in interactions.userId.unique():
        train_max = split.train.loc[split.train.userId == user_id, "timestamp"].max()
        validation_min = split.validation.loc[
            split.validation.userId == user_id, "timestamp"
        ].min()
        test_min = split.test.loc[split.test.userId == user_id, "timestamp"].min()
        if pd.notna(validation_min):
            assert train_max <= validation_min
        if pd.notna(test_min):
            assert train_max <= test_min
            if pd.notna(validation_min):
                assert validation_min <= test_min


def test_short_user_histories_remain_in_training() -> None:
    interactions = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 20], "timestamp": [2, 1]}
    )

    split = chronological_split(interactions)

    assert len(split.train) == 2
    assert split.validation.empty
    assert split.test.empty


def test_split_rejects_invalid_input() -> None:
    interactions = pd.DataFrame({"userId": [1], "movieId": [1]})

    with pytest.raises(ValueError, match="Missing required"):
        chronological_split(interactions)

    valid = interactions.assign(timestamp=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        chronological_split(valid, validation_fraction=0)


def test_split_rejects_missing_or_non_numeric_keys() -> None:
    with pytest.raises(ValueError, match="must not contain missing"):
        chronological_split(
            pd.DataFrame(
                {"userId": [1], "movieId": [None], "timestamp": [1]}
            )
        )

    with pytest.raises(TypeError, match="timestamp must be numeric"):
        chronological_split(
            pd.DataFrame(
                {"userId": [1], "movieId": [10], "timestamp": ["later"]}
            )
        )


def test_split_is_reproducible_for_equal_timestamps() -> None:
    interactions = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [10, 11, 12],
            "timestamp": [100, 100, 100],
        }
    )

    first = chronological_split(interactions)
    second = chronological_split(interactions)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
