"""Optional, cached OMDb movie-poster metadata enrichment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class PosterCatalog:
    """Resolve MovieLens titles through OMDb when an API key is configured.

    Poster lookup is deliberately optional. Without ``OMDB_API_KEY`` or when
    the service is unavailable, callers receive an empty result and can render
    an offline-safe placeholder.
    """

    def __init__(
        self,
        cache_path: str | Path = Path("artifacts") / "posters" / "omdb_cache.json",
        api_key: str | None = None,
        links_path: str | Path | None = None,
        env_path: str | Path = ".env",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.api_key = api_key or os.getenv("OMDB_API_KEY") or self._load_env_key(env_path)
        self.links = self._load_links(links_path)
        self._lock = Lock()
        self._cache = self._load_cache()

    @staticmethod
    def _load_env_key(env_path: str | Path) -> str | None:
        """Load only OMDB_API_KEY from a local dotenv-style file."""

        try:
            lines = Path(env_path).read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError):
            return None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "OMDB_API_KEY":
                value = value.strip().strip("\"'")
                return value or None
        return None

    @staticmethod
    def _load_links(links_path: str | Path | None) -> dict[str, str]:
        if links_path is None:
            return {}
        try:
            import csv

            with Path(links_path).open(newline="", encoding="utf-8") as handle:
                return {
                    str(row["movieId"]): str(row["imdbId"])
                    for row in csv.DictReader(handle)
                    if row.get("movieId") and row.get("imdbId")
                }
        except (FileNotFoundError, OSError, KeyError):
            return {}

    def _load_cache(self) -> dict[str, dict[str, str]]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _title_and_year(title: str) -> tuple[str, str | None]:
        match = re.search(r"\s+\((\d{4})\)\s*$", title)
        return (title[: match.start()].strip(), match.group(1)) if match else (title, None)

    def _query(self, movie_id: object, title: str) -> dict[str, str]:
        params = {"apikey": self.api_key or "", "r": "json"}
        imdb_id = self.links.get(str(movie_id), "").strip()
        if imdb_id:
            params["i"] = f"tt{imdb_id.zfill(7)}"
        else:
            clean_title, year = self._title_and_year(title)
            params["t"] = clean_title
            if year:
                params["y"] = year
        request = Request(
            "https://www.omdbapi.com/?" + urlencode(params),
            headers={"Accept": "application/json", "User-Agent": "Reelwise/1.0"},
        )
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("Response") != "True":
            return {}
        poster_url = payload.get("Poster")
        if not poster_url or poster_url == "N/A":
            return {}
        return {"poster_url": str(poster_url), "poster_provider": "OMDb"}

    def resolve(self, movie_id: object, title: str) -> dict[str, str]:
        key = str(movie_id)
        with self._lock:
            if key in self._cache:
                return dict(self._cache[key])
            if not self.api_key:
                return {}
            try:
                result = self._query(movie_id, title)
            except (OSError, ValueError, KeyError, TypeError):
                result = {}
            self._cache[key] = result
            try:
                self._save_cache()
            except OSError:
                pass
        return dict(result)
