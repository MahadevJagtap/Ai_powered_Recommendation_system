# Individual project reflection

## What I built

I built and validated an offline MovieLens recommendation pipeline in small
stages. The pipeline validates the supplied dataset, creates a deterministic
per-user chronological split, fits a training-only popularity baseline, and
provides a sparse NumPy item-item co-occurrence retrieval fallback. Both
recommendation paths preserve original MovieLens IDs and return movie titles
and genres.

## Key engineering decisions

The chronological split was chosen instead of a random split to prevent future
interactions from leaking into training. A rating threshold of 4.0 provided a
simple, explicit definition of positive feedback for the offline experiment.
The retrieval fallback uses sparse dictionaries rather than a dense user-by-
movie matrix, which is more practical for the 25M-row dataset on Windows.

## Verification

The final test suite contains 18 passing tests. The full dataset was validated
at 25,000,095 ratings and 62,423 movies. The baseline result was measured on
150,363 validation users. The saved retrieval result is a clearly labeled
1,000-user smoke evaluation.

## Trade-offs and limitations

TensorFlow Recommenders was attempted in the isolated environment, but its
native Windows/Python 3.13 runtime was unstable. I therefore kept the
NumPy fallback rather than claiming an unverified deep-learning model. This
means the project does not yet contain learned embeddings, FAISS retrieval,
or a learned ranking stage.

The original infrastructure requirements—Spark, Redis, Airflow, Triton, and
production serving—were not implemented within the available time. The
MovieLens data contains explicit ratings rather than online clicks, so the
reported metrics are offline Recall and NDCG measurements and not CTR.

## What I would do next

I would move model training to Linux or WSL2, implement a tested two-tower
retriever and a learned ranker, compare both against the same temporal
validation users, and then add a small API with caching. Only after measuring
latency and failure behavior would I integrate production serving components.
