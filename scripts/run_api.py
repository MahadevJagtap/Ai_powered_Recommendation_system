"""Run the local dependency-free recommendation HTTP API."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from recommender.data import validate_dataset
from recommender.service import RecommendationService
from recommender.split import chronological_split


def build_service(dataset_root: Path) -> RecommendationService:
    validate_dataset(dataset_root)
    ratings = pd.read_csv(
        dataset_root / "ratings.csv",
        usecols=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(dataset_root / "movies.csv")
    return RecommendationService().fit(chronological_split(ratings).train, movies)


def make_handler(service: RecommendationService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._write(200, {"status": "ok"})
                return
            if parsed.path == "/ready":
                self._write(200, {"status": "ready", "model_version": service.model_version})
                return
            prefix = "/recommendations/"
            if parsed.path.startswith(prefix):
                try:
                    user_id = int(parsed.path[len(prefix) :])
                    query = parse_qs(parsed.query)
                    k = int(query.get("k", ["10"])[0])
                    method = query.get("method", ["retrieval"])[0]
                    response = service.recommend(user_id, k=k, method=method)
                    self._write(
                        200,
                        {
                            "user_id": user_id,
                            "method": method,
                            "source": response.source,
                            "model_version": response.model_version,
                            "recommendations": response.recommendations,
                            "explanations": response.explanations,
                        },
                    )
                except (ValueError, RuntimeError, KeyError) as exc:
                    self._write(400, {"error": str(exc)})
                return
            self._write(404, {"error": "not found"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "ml-25m")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    service = build_service(args.dataset_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Recommendation API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
