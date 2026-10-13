# Implementation status

Status labels: `implemented and tested`, `implemented but not fully
validated`, `configuration only`, and `not implemented`.

| Requirement | Status | Files / validation | Known limitation |
|---|---|---|---|
| Dataset validation | implemented and tested | `src/recommender/data.py`; `scripts/validate_dataset.py` | Full CSV validation is I/O-heavy |
| Chronological split | implemented and tested | `src/recommender/split.py`; `tests/test_split.py` | Per-user pandas implementation |
| Popularity baseline | implemented and tested | `src/recommender/baseline.py`; `tests/test_baseline.py` | Offline baseline |
| NumPy retrieval fallback | implemented and tested | `src/recommender/numpy_retrieval.py`; `tests/test_retrieval.py` | Not a learned two-tower model |
| Local recommendation service | implemented and tested | `src/recommender/service.py`; `tests/test_service.py` | Loads in-process artifacts |
| HTTP demo API | implemented and tested | `scripts/run_api.py`; `tests/test_api.py` | Full-dataset startup is expensive |
| Browser demo UI | implemented and tested | `scripts/run_demo_ui.py`; `tests/test_api.py` | Dependency-free local UI, not Streamlit |
| Learned ranking | not implemented | — | No DCN/Wide & Deep claim |
| Spark feature pipeline | not implemented | — | PySpark unavailable |
| TensorFlow Recommenders | implemented and bounded end-to-end smoke-tested in `.venv-tf312` | `src/recommender/retrieval.py`, `scripts/run_learned_retrieval.py`; 20,000-row smoke run, 100,000-rating experiment, and shared three-method harness | CPU-only on native Windows; full 25M-row run not completed |
| FAISS | not implemented | — | FAISS unavailable |
| Redis | not implemented | — | Redis client/server unavailable |
| Airflow workflow | configuration only | no tested DAG | Airflow import availability alone is insufficient |
| Triton serving | not implemented | — | No running Triton server tested |
| Production deployment | not implemented | — | Current API is a local demo |

## TensorFlow/TFRS diagnostic

The existing `.venv` contains TensorFlow 2.20.0 and TFRS 0.7.7. Importing
TFRS without the legacy Keras setting fails because the environment exposes
Keras 3.15.1:

```text
Package tensorflow_recommenders requires tf.keras to be Keras version 2
```

With `TF_USE_LEGACY_KERAS=1`, the actual tiny two-tower train/save smoke test
fails before model construction with:

```text
DLL load failed while importing _pywrap_tensorflow_internal:
A dynamic link library (DLL) initialization routine failed.
```

The separate `.venv-tf312` environment uses Python 3.12.7 with the pinned
TensorFlow 2.20.0, tf-keras 2.20.1, and TensorFlow Recommenders 0.7.7
dependencies. The bounded smoke command below trained, saved, reloaded, and
evaluated successfully:

```powershell
$env:TF_USE_LEGACY_KERAS="1"
$env:PYTHONPATH="D:\recommendation\src"
.\.venv-tf312\Scripts\python.exe scripts\run_learned_retrieval.py `
  --ratings-row-limit 20000 --sample-size 5000 --dimension 8 `
  --batch-size 256 --epochs 1 --validation-user-limit 20 `
  --skip-dataset-validation `
  --output-dir artifacts\learned_retrieval_smoke `
  --results-path results\learned_retrieval_smoke.json
```

That run took 6.908 seconds, evaluated 20 users, and reported learned
Recall@10/Recall@50/NDCG@10 of 0.01048/0.01798/0.01082 versus the popularity
baseline's 0.16274/0.25829/0.10968 on the same users. These are bounded smoke
metrics only, not MovieLens 25M results.

The previous full attempt reached model weights and item embeddings, but did
not write a results file. Its timestamps show the expensive phase was saving
the Python `dict[user, set[movieId]]` history as `seen_by_user.json` (224.7 MB);
that representation also caused the approximately 1.1 GB memory footprint.
The retriever now stores sorted histories in compact `seen_by_user.npz` arrays
and logs preparation, epochs, embedding export, saving, reload, and evaluation.
The original interrupted artifacts are preserved but are not treated as
verified because their reload was not completed.

Verification status:

- Original `.venv`: 23 tests passed; 1 TensorFlow-specific test skipped
  because native TensorFlow is unavailable in Python 3.13 on this Windows
  machine.
- `.venv-tf312`: 8 focused TensorFlow/retrieval tests passed.
- The 20,000-row smoke run is distinct from the completed 100,000-rating
  bounded experiment. The smoke run used 5,000 sampled positives, dimension 8,
  one epoch, and 20 validation users. The larger experiment used 25,000
  sampled positives, dimension 16, one epoch, and 100 validation users.

The quality audit also found a candidate-universe mismatch in the first
comparison: the learned model scored only training-vocabulary movies while
the baseline ranked the full catalog. The CLI now aligns both methods to the
training-vocabulary universe. Three 100,000-row bounded experiments still
left learned retrieval below popularity. The best tested configuration
(dimension 16, one epoch, 25,000 sampled positives, 100 validation users)
reported learned Recall@10/Recall@50/NDCG@10 of
0.03311/0.12521/0.02361 versus the aligned baseline's
0.08588/0.16936/0.06312. This is not a significance claim; full training
remains intentionally deferred.

Reproducible diagnostic command:

```powershell
$env:TF_USE_LEGACY_KERAS="1"
$env:PYTHONPATH="D:\recommendation\src"
.\.venv\Scripts\python.exe -c "import tensorflow as tf; import tensorflow_recommenders as tfrs; print(tf.__version__, tfrs.__version__)"
```

## Current verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The local suite includes the original tests plus service tests. Results must
be refreshed by running the command; no result is inferred from file presence.

## Uniform-user evaluation

Prefix sampling is retained for quick diagnostics, but it is not
representative because the MovieLens ratings file is grouped by user. The
harness now supports `--sampling-strategy uniform-users`: it selects users
with a seeded sample, streams every rating for those users, and then applies
the chronological split. This preserves each sampled user's time order and
leaves the test partition untouched.

The verified 2,000-user run used evaluation seeds 42, 43, and 44, evaluated
1,000 users per seed, and completed in 924.313 seconds. It loaded 330,988
ratings and produced 266,526 training, 32,231 validation, and 32,231 test
rows, with 29,021 candidates.

| Seed | Popularity R@10/R@50/NDCG | NumPy R@10/R@50/NDCG | TFRS R@10/R@50/NDCG |
|---:|---|---|---|
| 42 | 0.05340 / 0.17038 / 0.04421 | 0.06963 / 0.20915 / 0.05700 | 0.01701 / 0.08879 / 0.01317 |
| 43 | 0.05825 / 0.16825 / 0.04561 | 0.06708 / 0.19891 / 0.05420 | 0.02031 / 0.08845 / 0.01582 |
| 44 | 0.05016 / 0.15976 / 0.04298 | 0.06080 / 0.19909 / 0.05166 | 0.02609 / 0.09577 / 0.01833 |

Per-user metrics and paired bootstrap intervals for method-minus-popularity
differences are persisted in the result artifact. These are bounded
uncertainty intervals, not full-dataset significance claims. Peak memory was
not measured.

## Shared evaluation harness

`scripts/run_shared_evaluation.py` compares popularity, NumPy co-occurrence,
and TFRS two-tower retrieval using one chronological split, one seeded
validation-user sample, one training-vocabulary candidate catalog, one
positive-rating rule, and one seen-item filter. It persists configuration,
selected users, candidate IDs, metrics, and timings.

The 20,000-row smoke run completed in 4.408 seconds for 20 users and 3,824
candidates. Metrics were popularity 0.05354/0.15790/0.05969, NumPy
co-occurrence 0.09667/0.22580/0.08732, and TFRS two-tower
0.00000/0.05521/0.00000 for Recall@10/Recall@50/NDCG@10.

The 250,000-row bounded run completed in 22.979 seconds for 100 users and
11,893 candidates. Metrics were popularity 0.05827/0.15875/0.05241, NumPy
co-occurrence 0.07363/0.22094/0.05984, and TFRS two-tower
0.04698/0.09896/0.03250. Peak process memory was not measured; no
full-dataset training was started. These are diagnostic bounded results, not
significance tests.

Newly generated shared-evaluation artifacts report three separate population
counts: `users_considered` is the number of unique users present in the
validation partition before eligibility filtering; `users_with_eligible_relevant_items`
is the number remaining after requiring training history, positive validation
ratings, and candidate-catalog membership; and `users_evaluated` is the number
selected after the configured user limit. The legacy
`users_with_relevant_items` field remains as an alias for the second count.
Historical artifacts are not overwritten and may not contain the new fields.
