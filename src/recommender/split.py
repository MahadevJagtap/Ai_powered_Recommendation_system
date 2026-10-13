"""Leakage-safe chronological interaction splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = ("userId", "movieId", "timestamp")


@dataclass(frozen=True)
class InteractionSplit:
    """Per-user chronological train, validation, and test interactions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    interactions: pd.DataFrame,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> InteractionSplit:
    """Split each user's interactions in time order.

    The newest interactions are assigned to test, the preceding interactions
    to validation, and the oldest interactions to train. Users with at least
    three interactions receive at least one row in each split. Users with one
    or two interactions remain in train so no user's observed history is
    entirely removed from model training.
    """

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    missing = [column for column in REQUIRED_COLUMNS if column not in interactions.columns]
    if missing:
        raise ValueError(f"Missing required interaction columns: {missing}")
    if interactions.empty:
        raise ValueError("interactions must not be empty")
    if interactions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("interaction keys and timestamps must not contain missing values")
    for column in ("userId", "movieId", "timestamp"):
        if not pd.api.types.is_numeric_dtype(interactions[column]):
            raise TypeError(f"{column} must be numeric")

    # Preserve source order as an explicit deterministic tie-breaker when
    # multiple interactions have the same timestamp.
    ordered = interactions.assign(_source_order=range(len(interactions))).sort_values(
        ["userId", "timestamp", "_source_order"], kind="mergesort"
    )
    ordered = ordered.reset_index(drop=True)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_rows in ordered.groupby("userId", sort=False):
        count = len(user_rows)
        if count < 3:
            train_end = count
            validation_end = count
        else:
            test_count = max(1, int(count * test_fraction))
            validation_count = max(1, int(count * validation_fraction))
            if test_count + validation_count >= count:
                validation_count = 1
                test_count = 1
            train_end = count - validation_count - test_count
            validation_end = count - test_count

        train_parts.append(user_rows.iloc[:train_end])
        validation_parts.append(user_rows.iloc[train_end:validation_end])
        test_parts.append(user_rows.iloc[validation_end:])

    def _clean(part: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(part, ignore_index=True).drop(columns="_source_order")

    return InteractionSplit(
        train=_clean(train_parts),
        validation=_clean(validation_parts),
        test=_clean(test_parts),
    )
