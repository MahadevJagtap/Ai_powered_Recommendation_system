"""TensorFlow Recommenders two-tower candidate retrieval."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from recommender.baseline import POSITIVE_RATING_THRESHOLD, PopularityRecommender


ARTIFACT_VERSION = "two-tower-retrieval-v1"


def _json_id(value: object) -> int | float | str:
    """Convert a NumPy scalar or ordinary ID to a JSON-compatible value."""

    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True)
class RetrievalEvaluation:
    users_evaluated: int
    recall_at_10: float
    recall_at_50: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "users_evaluated": self.users_evaluated,
            "recall_at_10": self.recall_at_10,
            "recall_at_50": self.recall_at_50,
            "ndcg_at_10": self.ndcg_at_10,
        }


def calculate_retrieval_metrics(
    recommendations_by_user: dict[object, list[object]],
    relevant_by_user: dict[object, set[object]],
) -> RetrievalEvaluation:
    """Calculate macro metrics for a fixed set of ranked recommendations."""

    if not relevant_by_user:
        return RetrievalEvaluation(0, 0.0, 0.0, 0.0)
    recalls_10: list[float] = []
    recalls_50: list[float] = []
    ndcg_10: list[float] = []
    for user_id, expected in relevant_by_user.items():
        recommendations = recommendations_by_user[user_id][:50]
        hits_10 = len(set(recommendations[:10]) & expected)
        recalls_10.append(hits_10 / len(expected))
        recalls_50.append(len(set(recommendations) & expected) / len(expected))
        dcg = sum(
            1 / log2(rank + 2)
            for rank, movie_id in enumerate(recommendations[:10])
            if movie_id in expected
        )
        ideal_hits = min(len(expected), 10)
        ideal_dcg = sum(1 / log2(rank + 2) for rank in range(ideal_hits))
        ndcg_10.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    return RetrievalEvaluation(
        len(relevant_by_user),
        float(np.mean(recalls_10)),
        float(np.mean(recalls_50)),
        float(np.mean(ndcg_10)),
    )


class TwoTowerModel(tfrs.Model):
    """Minimal user/movie embedding model trained with sampled softmax."""

    def __init__(self, user_ids: Iterable[object], movie_ids: Iterable[object], dimension: int):
        super().__init__()
        self.user_lookup = tf.keras.layers.StringLookup(
            vocabulary=[str(value) for value in user_ids], mask_token=None
        )
        self.movie_lookup = tf.keras.layers.StringLookup(
            vocabulary=[str(value) for value in movie_ids], mask_token=None
        )
        self.user_tower = tf.keras.Sequential(
            [
                self.user_lookup,
                tf.keras.layers.Embedding(self.user_lookup.vocabulary_size(), dimension),
                tf.keras.layers.Dense(dimension, activation="relu"),
            ]
        )
        self.movie_tower = tf.keras.Sequential(
            [
                self.movie_lookup,
                tf.keras.layers.Embedding(self.movie_lookup.vocabulary_size(), dimension),
                tf.keras.layers.Dense(dimension, activation="relu"),
            ]
        )
        self.task = tfrs.tasks.Retrieval()

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        return self.user_tower(features["userId"])

    def compute_loss(self, features: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        user_embeddings = self.user_tower(features["userId"])
        movie_embeddings = self.movie_tower(features["movieId"])
        return self.task(user_embeddings, movie_embeddings)

    def user_embedding(self, user_id: object) -> np.ndarray:
        return self.user_tower(tf.constant([str(user_id)])).numpy()[0]

    def movie_embeddings(self, movie_ids: Iterable[object]) -> np.ndarray:
        values = tf.constant([str(value) for value in movie_ids])
        return self.movie_tower(values).numpy()


class TwoTowerRetriever:
    """Train-only two-tower model with persisted mappings and safe retrieval."""

    def __init__(self, dimension: int = 32, seed: int = 42):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.seed = seed
        self.model: TwoTowerModel | None = None
        self.user_ids: list[object] = []
        self.movie_ids: list[object] = []
        self.movie_catalog: pd.DataFrame | None = None
        self.item_embeddings: np.ndarray | None = None
        self.seen_by_user: dict[object, np.ndarray] = {}
        self.fallback: PopularityRecommender | None = None
        self.training_config: dict[str, object] = {}

    def fit(
        self,
        training: pd.DataFrame,
        movies: pd.DataFrame,
        sample_size: int | None = 200_000,
        epochs: int = 1,
        batch_size: int = 1024,
    ) -> "TwoTowerRetriever":
        required = {"userId", "movieId", "rating"}
        if not required.issubset(training.columns):
            raise ValueError(f"training is missing columns: {sorted(required - set(training.columns))}")
        if not {"movieId", "title", "genres"}.issubset(movies.columns):
            raise ValueError("movies must contain movieId, title, and genres")
        if training.empty or movies.empty:
            raise ValueError("training and movies must not be empty")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if sample_size is not None and sample_size <= 0:
            raise ValueError("sample_size must be positive or None")

        def progress(message: str) -> None:
            print(f"[learned-retrieval] {message}", flush=True)

        progress(f"preparing {len(training):,} training rows")
        tf.keras.utils.set_random_seed(self.seed)
        positive = training.loc[training["rating"] >= POSITIVE_RATING_THRESHOLD].copy()
        if positive.empty:
            raise ValueError("training has no positive interactions")
        self.user_ids = sorted(training["userId"].unique().tolist())
        self.movie_ids = sorted(training["movieId"].unique().tolist())
        self.movie_catalog = movies.loc[movies["movieId"].isin(self.movie_ids)].copy()
        self.seen_by_user = {
            user_id: np.sort(rows["movieId"].to_numpy())
            for user_id, rows in training.groupby("userId", sort=False)
        }
        self.fallback = PopularityRecommender().fit(training, movies)
        if sample_size is not None and len(positive) > sample_size:
            positive = positive.sample(sample_size, random_state=self.seed)
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "userId": positive["userId"].astype(str).to_numpy(),
                "movieId": positive["movieId"].astype(str).to_numpy(),
            }
        ).shuffle(min(len(positive), 100_000), seed=self.seed, reshuffle_each_iteration=False)
        progress(
            f"training {len(positive):,} positive rows, {len(self.user_ids):,} users, "
            f"{len(self.movie_ids):,} movies"
        )
        self.model = TwoTowerModel(self.user_ids, self.movie_ids, self.dimension)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
                progress(f"epoch {epoch + 1}/{epochs} started")

            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                loss = None if logs is None else logs.get("loss")
                progress(f"epoch {epoch + 1}/{epochs} complete; loss={loss}")

        self.model.fit(
            dataset.batch(batch_size),
            epochs=epochs,
            verbose=0,
            callbacks=[ProgressCallback()],
        )
        progress("exporting item embeddings")
        self.item_embeddings = self.model.movie_embeddings(self.movie_ids)
        self.training_config = {
            "dimension": self.dimension,
            "seed": self.seed,
            "sample_size": sample_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
            "training_rows": len(training),
            "positive_training_rows": len(positive),
        }
        return self

    def _require_fitted(self) -> TwoTowerModel:
        if (
            self.model is None
            or self.movie_catalog is None
            or self.fallback is None
            or self.item_embeddings is None
        ):
            raise RuntimeError("TwoTowerRetriever must be fitted or loaded before retrieval")
        return self.model

    @staticmethod
    def _contains(sorted_values: np.ndarray, value: object) -> bool:
        position = int(np.searchsorted(sorted_values, value, side="left"))
        return position < len(sorted_values) and sorted_values[position] == value

    @staticmethod
    def rank_candidates(
        movie_ids: Iterable[object],
        scores: np.ndarray,
        seen: Iterable[object],
        k: int,
    ) -> list[object]:
        """Rank scored candidates descending, stably, and remove seen items."""

        if k <= 0:
            raise ValueError("k must be positive")
        ids = list(movie_ids)
        values = np.asarray(scores)
        if len(ids) != len(values):
            raise ValueError("movie_ids and scores must have the same length")
        seen_values = np.sort(np.asarray(list(seen)))
        order = np.argsort(-values, kind="stable")
        return [
            ids[index]
            for index in order
            if not TwoTowerRetriever._contains(seen_values, ids[index])
        ][:k]

    def recommend(self, user_id: object, k: int = 10) -> pd.DataFrame:
        if k <= 0:
            raise ValueError("k must be positive")
        model = self._require_fitted()
        if user_id not in self.seen_by_user:
            return self.fallback.recommend(user_id, k)
        scores = self.item_embeddings @ model.user_embedding(user_id)
        seen = self.seen_by_user[user_id]
        selected = self.rank_candidates(self.movie_ids, scores, seen, k)
        result = self.movie_catalog.set_index("movieId").loc[selected].reset_index()
        score_by_movie = dict(zip(self.movie_ids, scores))
        result["retrieval_score"] = [float(score_by_movie[movie_id]) for movie_id in selected]
        return result

    def evaluate(self, validation: pd.DataFrame, user_limit: int | None = None) -> RetrievalEvaluation:
        positive = validation.loc[validation["rating"] >= POSITIVE_RATING_THRESHOLD]
        relevant = {
            user_id: set(rows["movieId"])
            for user_id, rows in positive.groupby("userId", sort=False)
        }
        if user_limit is not None:
            if user_limit < 0:
                raise ValueError("user_limit must be non-negative or None")
            relevant = dict(list(relevant.items())[:user_limit])
        recommendations = {
            user_id: self.recommend(user_id, 50)["movieId"].tolist()
            for user_id in relevant
        }
        return calculate_retrieval_metrics(recommendations, relevant)

    def save(self, directory: str | Path) -> None:
        model = self._require_fitted()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        model.save_weights(target / "two_tower.weights.h5")
        np.save(target / "item_embeddings.npy", self.item_embeddings)
        (target / "mappings.json").write_text(
            json.dumps(
                {
                    "user_ids": [_json_id(value) for value in self.user_ids],
                    "movie_ids": [_json_id(value) for value in self.movie_ids],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        user_offsets = np.zeros(len(self.user_ids) + 1, dtype=np.int64)
        seen_values: list[np.ndarray] = []
        for index, user_id in enumerate(self.user_ids):
            values = self.seen_by_user.get(user_id, np.array([], dtype=np.int64))
            seen_values.append(values)
            user_offsets[index + 1] = user_offsets[index] + len(values)
        np.savez(
            target / "seen_by_user.npz",
            user_offsets=user_offsets,
            seen_movie_ids=np.concatenate(seen_values) if seen_values else np.array([], dtype=np.int64),
        )
        self.movie_catalog.to_csv(target / "movie_catalog.csv", index=False)
        self.fallback._require_fitted().to_csv(target / "fallback_ranked_movies.csv", index=False)
        (target / "training_config.json").write_text(
            json.dumps(self.training_config, indent=2) + "\n", encoding="utf-8"
        )
        (target / "model_metadata.json").write_text(
            json.dumps(
                {
                    "artifact_version": ARTIFACT_VERSION,
                    "model_type": "tensorflow-recommenders-two-tower",
                    "tensorflow_version": tf.__version__,
                    "tfrs_version": getattr(tfrs, "__version__", "unknown"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[learned-retrieval] saved model and artifacts to {target}", flush=True)

    @classmethod
    def load(cls, directory: str | Path) -> "TwoTowerRetriever":
        source = Path(directory)
        metadata = json.loads((source / "model_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported two-tower artifact version")
        config = json.loads((source / "training_config.json").read_text(encoding="utf-8"))
        mappings = json.loads((source / "mappings.json").read_text(encoding="utf-8"))
        retriever = cls(int(config["dimension"]), int(config["seed"]))
        retriever.user_ids = mappings["user_ids"]
        retriever.movie_ids = mappings["movie_ids"]
        retriever.training_config = config
        retriever.movie_catalog = pd.read_csv(source / "movie_catalog.csv")
        retriever.item_embeddings = np.load(source / "item_embeddings.npy")
        compact_seen = source / "seen_by_user.npz"
        if compact_seen.is_file():
            arrays = np.load(compact_seen)
            offsets = arrays["user_offsets"]
            values = arrays["seen_movie_ids"]
            retriever.seen_by_user = {
                user_id: values[offsets[index] : offsets[index + 1]]
                for index, user_id in enumerate(retriever.user_ids)
            }
        else:
            seen = json.loads((source / "seen_by_user.json").read_text(encoding="utf-8"))
            retriever.seen_by_user = {
                next((user_id for user_id in retriever.user_ids if str(user_id) == key), key): np.asarray(movie_ids)
                for key, movie_ids in seen.items()
            }
        retriever.model = TwoTowerModel(retriever.user_ids, retriever.movie_ids, retriever.dimension)
        retriever.model.user_tower(tf.constant([str(retriever.user_ids[0])]))
        retriever.model.movie_tower(tf.constant([str(retriever.movie_ids[0])]))
        retriever.model.load_weights(source / "two_tower.weights.h5")
        fallback = PopularityRecommender()
        fallback._ranked_movies = pd.read_csv(source / "fallback_ranked_movies.csv")
        fallback._seen_by_user = retriever.seen_by_user
        retriever.fallback = fallback
        return retriever
