# 10-minute presentation outline

## 1. Problem and objective — 1 minute

- Recommend movies from MovieLens 25M.
- Demonstrate a leakage-safe offline recommendation workflow.
- Prioritize a working, testable Windows demo within the project time limit.

## 2. Dataset and data quality — 1 minute

- 25,000,095 ratings and 62,423 movies.
- Original `ratings.csv` and `movies.csv` are preserved.
- Ratings of at least 4.0 are treated as positive interactions.

## 3. Temporal split — 1 minute

- Interactions are sorted per user by timestamp.
- Oldest interactions are used for training.
- Later interactions are used for validation and test.
- Equal timestamps use deterministic source-order tie-breaking.
- Validation and test interactions are never used to fit recommendation statistics.

## 4. Baseline recommender — 1 minute

- Training-only movie popularity counts.
- Positive mean rating breaks popularity ties.
- Known-user recommendations exclude training-seen movies.
- Unknown users receive global popularity recommendations.

## 5. Retrieval fallback — 1.5 minutes

- Sparse item-item co-occurrence from positive training histories.
- Bounded user histories avoid excessive pair generation.
- Personalized candidates are supplemented by popularity when needed.
- This is a NumPy fallback, not a TensorFlow Recommenders two-tower model.

## 6. Demo — 1 minute

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --user-id 1 --k 5
```

Show the returned `movieId`, title, and genres. Explain that training-seen
movies are filtered and that unknown users use popularity fallback.

## 7. Evaluation results — 1.5 minutes

Baseline on 150,363 validation users:

- Recall@10: 0.0573
- Recall@50: 0.1713
- NDCG@10: 0.0509

Retrieval smoke result on 1,000 validation users:

- Recall@10: 0.0654
- Recall@50: 0.2049

The retrieval smoke population is smaller and must not be presented as a
full-dataset comparison.

## 8. Engineering verification — 1 minute

- 18 automated tests pass.
- Dataset validation passes.
- CLI commands and saved JSON results were verified.
- Results are reproducible from the project-root commands in `README.md`.

## 9. Limitations and next steps — 1 minute

Not implemented or validated:

- Learned ranking model
- TensorFlow Recommenders retrieval
- FAISS
- Spark
- Redis
- Airflow
- Triton
- Production API and deployment

Next step would be a Linux/WSL-based learned retrieval and ranking pipeline,
followed by serving and latency evaluation.
