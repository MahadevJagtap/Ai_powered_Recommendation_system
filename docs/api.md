# Local recommendation API

The project includes a dependency-free HTTP demo that calls the actual
popularity and NumPy co-occurrence implementations. It is not a production
API and does not use Redis, FastAPI, or Triton.

Start it from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py
```

The command loads the dataset and chronological training partition before
serving requests, so startup is I/O- and memory-intensive on MovieLens 25M.

Endpoints:

```text
GET /health
GET /ready
GET /recommendations/{user_id}?k=10&method=retrieval
GET /recommendations/{user_id}?k=10&method=baseline
```

Example PowerShell requests:

```powershell
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommendations/1?k=5&method=retrieval"
curl "http://127.0.0.1:8000/recommendations/999?k=5&method=baseline"
```

Responses include the recommendation source, model version, original movie
IDs, titles, genres, and an honest `explanations` payload. For retrieval,
contributors contain only actual positive-training co-occurrence signals; for
the baseline, supporting positive-interaction counts are included when
available. Unknown users are labeled
`popularity_fallback`. This local server has no authentication, rate limiting,
distributed caching, metrics backend, or production deployment controls.

## Browser demonstration

The dependency-free browser UI uses the same `RecommendationService` and
supports user ID, retrieval method, recommendation count, and movie metadata:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_ui.py
```

Open `http://127.0.0.1:8501/`. The UI is a local demonstration and has the
same full-dataset startup cost and non-production security limitations as the
HTTP API. It also exposes `GET /evaluation`, which reads saved JSON artifacts
without retraining. The page displays optional OMDb poster artwork when the server process has
`OMDB_API_KEY` configured directly or in a local `.env` file. The key is never
sent to browser code. MovieLens `links.csv` IMDb IDs are preferred, with title
and year used as a fallback. Poster metadata is cached under
`artifacts/posters/omdb_cache.json`, and an offline-safe placeholder is used
when OMDb is unavailable or returns `N/A`.
