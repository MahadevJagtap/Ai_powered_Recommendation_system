"""Run the local dependency-free recommendation HTTP API."""

from __future__ import annotations
import os
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

        def end_headers(self) -> None:
            self.send_header(
                "Access-Control-Allow-Origin",
                "https://recommendation-ui-65o5.onrender.com",
            )
            self.send_header(
                "Access-Control-Allow-Methods",
                "GET, OPTIONS",
            )
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type",
            )
            super().end_headers()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.end_headers()

        def _write(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Keep your existing do_GET() and log_message() methods unchanged.

    return Handler




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("DATASET_ROOT", "data/ml-25m")),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
    )

    args = parser.parse_args()
    service = build_service(args.dataset_root)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(service),
    )
    print(f"Recommendation API listening on http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
