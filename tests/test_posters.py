import json
from urllib.parse import parse_qs, urlparse

from recommender.posters import PosterCatalog


def test_poster_catalog_is_offline_safe_without_api_key(tmp_path) -> None:
    catalog = PosterCatalog(tmp_path / "cache.json", api_key="")
    assert catalog.resolve(10, "Example Movie (2020)") == {}
    assert not (tmp_path / "cache.json").exists()


def test_poster_catalog_loads_key_from_local_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OMDB_API_KEY=test-only-key\n", encoding="utf-8")
    catalog = PosterCatalog(tmp_path / "cache.json", env_path=env_file)
    assert catalog.api_key == "test-only-key"


def test_poster_catalog_reuses_cached_metadata(tmp_path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps({"10": {"poster_url": "https://image.tmdb.org/a.jpg", "poster_provider": "OMDb"}}),
        encoding="utf-8",
    )
    catalog = PosterCatalog(cache, api_key="not-used")
    assert catalog.resolve(10, "Example Movie (2020)")["poster_provider"] == "OMDb"


def test_omdb_lookup_uses_imdb_id_and_caches_result(tmp_path, monkeypatch) -> None:
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"Response":"True","Poster":"https://img.example/poster.jpg"}'

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return Response()

    monkeypatch.setattr("recommender.posters.urlopen", fake_urlopen)
    links = tmp_path / "links.csv"
    links.write_text("movieId,imdbId,tmdbId\n10,0111161,278\n", encoding="utf-8")
    catalog = PosterCatalog(
        tmp_path / "cache.json",
        api_key="test-key",
        links_path=links,
    )

    assert catalog.resolve(10, "Ignored Title (1994)") == {
        "poster_url": "https://img.example/poster.jpg",
        "poster_provider": "OMDb",
    }
    assert catalog.resolve(10, "Ignored Title (1994)")["poster_provider"] == "OMDb"
    assert len(calls) == 1
    query = parse_qs(urlparse(calls[0][0]).query)
    assert query["i"] == ["tt0111161"]
    assert "apikey" in query


def test_omdb_title_year_fallback_and_missing_poster(tmp_path, monkeypatch) -> None:
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"Response":"True","Poster":"N/A"}'

    def fake_urlopen(request, timeout):
        requests.append(request.full_url)
        return Response()

    monkeypatch.setattr("recommender.posters.urlopen", fake_urlopen)
    catalog = PosterCatalog(tmp_path / "cache.json", api_key="test-key")
    assert catalog.resolve(11, "Example Movie (2020)") == {}
    assert catalog.resolve(11, "Example Movie (2020)") == {}
    assert len(requests) == 1
    query = parse_qs(urlparse(requests[0]).query)
    assert query["t"] == ["Example Movie"]
    assert query["y"] == ["2020"]


def test_omdb_failure_is_cached_and_offline_safe(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise TimeoutError("timed out")

    monkeypatch.setattr("recommender.posters.urlopen", fake_urlopen)
    catalog = PosterCatalog(tmp_path / "cache.json", api_key="test-key")
    assert catalog.resolve(12, "Unavailable (2020)") == {}
    assert catalog.resolve(12, "Unavailable (2020)") == {}
    assert len(calls) == 1
