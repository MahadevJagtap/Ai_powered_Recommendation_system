# Model training status

## Working training paths

The validated project currently trains the following CPU-compatible models:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py
.\.venv\Scripts\python.exe scripts\run_retrieval.py
```

The second command trains the sparse NumPy item-item co-occurrence fallback.
It is not a learned embedding model.

## TensorFlow Recommenders bounded training path

`src/recommender/retrieval.py` contains the two-tower user and movie embedding
model. The bounded training path is runnable and tested in the separate
`.venv-tf312` environment, including training, artifact saving, artifact
reloading, candidate inference, and evaluation.

Installed versions in the existing environment:

- Python 3.13.15
- TensorFlow 2.20.0
- TensorFlow Recommenders 0.7.7

TFRS import requires:

```powershell
$env:TF_USE_LEGACY_KERAS="1"
```

The Python 3.13 `.venv` remains unable to initialize the native TensorFlow
runtime on this Windows machine. That limitation applies only to `.venv`;
the Python 3.12.7 `.venv-tf312` path is the validated TensorFlow environment.

## Compatible training environment

A separate Python 3.12.7 environment is now available at `.venv-tf312`.

Verify the interpreter before creating or recreating the environment:

```powershell
py -3.12 --version
py -3.12 -m venv .venv-tf312
```

Install the pinned optional requirements:

```powershell
.\.venv-tf312\Scripts\python.exe -m pip install -r requirements-tf.txt
$env:TF_USE_LEGACY_KERAS="1"
$env:PYTHONPATH="D:\recommendation\src"
```

The bounded two-tower train/save/reload/evaluation smoke test now passes in
this environment with TensorFlow 2.20.0 and TensorFlow Recommenders 0.7.7.
The full MovieLens 25M run is intentionally not started automatically.

Completed bounded experiment:

```powershell
$env:TF_USE_LEGACY_KERAS="1"
$env:PYTHONPATH="D:\recommendation\src"
.\.venv-tf312\Scripts\python.exe scripts\run_learned_retrieval.py `
  --ratings-row-limit 100000 --sample-size 25000 --dimension 16 `
  --batch-size 512 --epochs 1 --validation-user-limit 100 `
  --skip-dataset-validation `
  --output-dir artifacts\learned_retrieval_d16_s25k_e1 `
  --results-path results\learned_retrieval_d16_s25k_e1.json
```

This experiment used the first 100,000 ratings, 25,000 sampled positive
training interactions, dimension 16, one epoch, batch size 512, seed 42, and
100 validation users. It completed training, saving, reloading, inference,
and evaluation in 15.413 seconds.

| Metric | Learned two-tower | Aligned popularity |
|---|---:|---:|
| Recall@10 | 0.03311 | 0.08588 |
| Recall@50 | 0.12521 | 0.16936 |
| NDCG@10 | 0.02361 | 0.06312 |

These are small, diagnostic bounded results. They do not establish
full-dataset performance or statistical significance. The popularity baseline
remains the benchmark and fallback.

The previous full attempt spent substantial time and memory materializing and
serializing per-user seen histories as a 224.7 MB JSON file after model
weights/item embeddings had already been written. The current implementation
uses compact sorted NumPy history arrays and flushes progress messages.

## Retrieval quality audit

The training objective is TFRS sampled retrieval with in-batch negatives:
positive interactions are ratings `>= 4.0`, user and movie IDs are stringified
through separate `StringLookup` vocabularies, and the model scores candidates
with the dot product of the user and movie tower outputs. Candidates are
sorted by descending score with a stable tie-break and training-seen movies
are removed before Recall/NDCG calculation. Unknown users use the training-only
popularity fallback.

The chronological split is unchanged: training is the oldest per-user
partition, validation follows it, and test is newest. Model vocabularies and
fitting use training only. A previous comparison had an evaluation mismatch:
the learned model could score only training-vocabulary movies while the
baseline ranked the full catalog. The CLI now restricts both methods to the
same training-vocabulary candidate universe.

Bounded experiments on the first 100,000 ratings and the first 100 positive
validation users produced:

| Dimension | Sample | Epochs | Runtime (s) | Learned R@10 | Learned R@50 | Learned NDCG@10 | Baseline R@10 | Baseline R@50 | Baseline NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 25,000 | 1 | 12.992 | 0.01144 | 0.05477 | 0.01526 | 0.08588 | 0.16936 | 0.06312 |
| 16 | 25,000 | 1 | 15.413 | 0.03311 | 0.12521 | 0.02361 | 0.08588 | 0.16936 | 0.06312 |
| 16 | 25,000 | 2 | 16.099 | 0.02120 | 0.08756 | 0.01597 | 0.08588 | 0.16936 | 0.06312 |
| 16 | 42,374 available positives | 1 | 16.125 | 0.02183 | 0.08076 | 0.01873 | 0.08588 | 0.16936 | 0.06312 |

The sample-size run used all 42,374 positive rows available in the bounded
training prefix because fewer than the requested 50,000 existed. These
experiments are diagnostic bounded samples, not statistically significant
comparisons. They show that the low learned metrics persist after candidate
universe alignment; the evidence currently points to model/objective quality
and limited sampled training rather than a ranking or leakage bug. Full
training remains blocked until a larger bounded experiment demonstrates
credible improvement.

## Shared bounded evaluation

The common harness compares all three retrieval paths using the same
chronological split, seeded validation users, training-vocabulary candidate
catalog, relevance rule, and seen-item filtering:

```powershell
$env:TF_USE_LEGACY_KERAS="1"
$env:PYTHONPATH="D:\recommendation\src"
.\.venv-tf312\Scripts\python.exe scripts\run_shared_evaluation.py `
  --ratings-row-limit 250000 --sample-size 100000 --dimension 16 `
  --batch-size 512 --epochs 1 --validation-user-limit 100 --seed 42 `
  --skip-dataset-validation --output-dir results\shared_250k
```

The smoke run (20,000 ratings, 20 users) took 4.408 seconds. The 250,000-row
run (100 users) took 22.979 seconds. The 250,000-row metrics were:

| Method | Recall@10 | Recall@50 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.05827 | 0.15875 | 0.05241 |
| NumPy co-occurrence | 0.07363 | 0.22094 | 0.05984 |
| TFRS two-tower | 0.04698 | 0.09896 | 0.03250 |

The learned model did not beat either existing method in these bounded
comparisons. Peak process memory was not measured; full MovieLens training
remains deferred.

Prefix-based runs are useful diagnostics but are biased because the ratings
file is grouped by user. The shared harness also supports a seeded
uniform-user bounded sample that streams complete histories before splitting
chronologically. A verified 2,000-user sample evaluated 1,000 users under
seeds 42, 43, and 44 in 924.313 seconds. NumPy co-occurrence remained ahead
of popularity and TFRS for every seed. Per-user metrics and paired bootstrap
intervals are persisted; these bounded intervals do not establish
full-dataset significance.
