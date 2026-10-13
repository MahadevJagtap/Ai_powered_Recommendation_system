import json
from http.client import HTTPConnection
from threading import Thread

import pandas as pd

from scripts.run_api import make_handler
from scripts.run_demo_ui import make_handler as make_ui_handler
from recommender.service import RecommendationService
from http.server import ThreadingHTTPServer


def _server() -> ThreadingHTTPServer:
    training = pd.DataFrame(
        {"userId": [1, 1, 2, 2], "movieId": [10, 20, 20, 30], "rating": [5, 4, 5, 4]}
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Comedy", "Action", "Horror"],
        }
    )
    service = RecommendationService().fit(training, movies)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def _get(server: ThreadingHTTPServer, path: str) -> tuple[int, dict]:
    connection = HTTPConnection(*server.server_address)
    connection.request("GET", path)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def test_health_and_recommendation_endpoint() -> None:
    server = _server()
    try:
        assert _get(server, "/health") == (200, {"status": "ok"})
        status, body = _get(server, "/recommendations/999?k=2&method=baseline")
        assert status == 200
        assert body["source"] == "popularity_fallback"
        assert len(body["recommendations"]) == 2
    finally:
        server.shutdown()
        server.server_close()


def test_api_rejects_invalid_requests() -> None:
    server = _server()
    try:
        status, body = _get(server, "/recommendations/1?k=101")
        assert status == 400
        assert "at most 100" in body["error"]
        assert _get(server, "/missing")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_browser_demo_serves_form_and_real_recommendations() -> None:
    training = pd.DataFrame(
        {"userId": [1, 1], "movieId": [10, 20], "rating": [5, 4]}
    )
    movies = pd.DataFrame(
        {"movieId": [10, 20, 30], "title": ["A", "B", "C"], "genres": ["Drama"] * 3}
    )
    service = RecommendationService().fit(training, movies)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_ui_handler(service))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        document = response.read().decode()
        assert "Reelwise" in document
        assert "cards" in document
        assert "NumPy co-occurrence" in document
        assert "Curating your slate" in document
        assert "movie-title" in document
        assert "poster-fallback" in document
        assert "Year unavailable" in document
        assert "titleById" in document
        assert "MovieLens ID" in document
        status, body = _get(server, "/recommendations?user_id=999&k=1")
        assert status == 200
        assert body["recommendations"][0]["movieId"] == 10
        assert "explanations" in body
        status, body = _get(server, "/evaluation")
        assert status == 200
        assert isinstance(body, dict)
    finally:
        server.shutdown()
        server.server_close()


def test_browser_demo_includes_server_side_poster_metadata() -> None:
    training = pd.DataFrame(
        {"userId": [1, 1], "movieId": [10, 20], "rating": [5, 4]}
    )
    movies = pd.DataFrame(
        {"movieId": [10, 20, 30], "title": ["A", "B", "C"], "genres": ["Drama"] * 3}
    )

    class FakePosterCatalog:
        def resolve(self, movie_id, title):
            return {
                "poster_url": "https://img.example/poster.jpg",
                "poster_provider": "OMDb",
            }

    service = RecommendationService().fit(training, movies)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_ui_handler(service, poster_catalog=FakePosterCatalog()),
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, body = _get(server, "/recommendations?user_id=999&k=1")
        assert status == 200
        assert body["recommendations"][0]["poster_provider"] == "OMDb"
        assert body["recommendations"][0]["poster_url"].startswith("https://")
    finally:
        server.shutdown()
        server.server_close()
