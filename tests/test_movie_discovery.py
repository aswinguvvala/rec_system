"""Tests for src/movie_discovery.py: TMDb-sourced catalog broadening."""

import requests

from src.data_pipeline import GENRE_COLUMNS
from src.movie_discovery import (
    build_supplementary_catalog,
    discover_candidate_ids,
    fetch_movie_detail,
    _normalize_detail,
)


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


def _detail(imdb_id: str = "tt9999999", title: str = "Test Movie", genres=None, release_date="2019-05-01", status="Released"):
    return {
        "title": title,
        "release_date": release_date,
        "genres": [{"name": g} for g in (genres or [])],
        "external_ids": {"imdb_id": imdb_id},
        "status": status,
    }


class TestDiscoverCandidateIds:
    def test_no_results_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(
            "src.movie_discovery.requests.get",
            lambda *a, **k: _FakeResponse({"results": [], "total_pages": 1}),
        )
        assert discover_candidate_ids("te", "fake-key", max_pages=3) == []

    def test_collects_ids_across_pages_then_stops_at_total_pages(self, monkeypatch):
        calls = []

        def _get(*a, **k):
            page = k["params"]["page"]
            calls.append(page)
            return _FakeResponse({"results": [{"id": page * 10}], "total_pages": 2})

        monkeypatch.setattr("src.movie_discovery.requests.get", _get)

        ids = discover_candidate_ids("te", "fake-key", max_pages=5)

        assert ids == [10, 20]
        assert calls == [1, 2]  # stopped at total_pages, didn't run all 5 requested

    def test_network_failure_returns_whatever_was_collected_so_far(self, monkeypatch):
        def _raise(*a, **k):
            raise requests.exceptions.ConnectionError("simulated failure")

        monkeypatch.setattr("src.movie_discovery.requests.get", _raise)

        assert discover_candidate_ids("te", "fake-key", max_pages=3) == []


class TestFetchMovieDetail:
    def test_returns_payload_on_success(self, monkeypatch):
        monkeypatch.setattr(
            "src.movie_discovery.requests.get", lambda *a, **k: _FakeResponse(_detail())
        )
        detail = fetch_movie_detail(123, "fake-key")
        assert detail["external_ids"]["imdb_id"] == "tt9999999"

    def test_network_failure_returns_none(self, monkeypatch):
        def _raise(*a, **k):
            raise requests.exceptions.ConnectionError("simulated failure")

        monkeypatch.setattr("src.movie_discovery.requests.get", _raise)
        assert fetch_movie_detail(123, "fake-key") is None

    def test_http_error_status_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "src.movie_discovery.requests.get", lambda *a, **k: _FakeResponse({}, status_code=404)
        )
        assert fetch_movie_detail(123, "fake-key") is None


class TestNormalizeDetail:
    def test_maps_known_genres_and_drops_unknown_ones(self):
        row = _normalize_detail(_detail(genres=["Action", "Documentary", "Science Fiction"]), "Telugu")
        assert row["Action"] == 1
        assert row["Sci-Fi"] == 1  # TMDb's "Science Fiction" -> dataset's "Sci-Fi"
        assert sum(row[g] for g in GENRE_COLUMNS) == 2  # Documentary has no home, dropped

    def test_sets_languages_from_the_search_language_not_tmdb_metadata(self):
        row = _normalize_detail(_detail(), "Telugu")
        assert row["languages"] == "Telugu"

    def test_imdb_rating_is_always_na_not_backfilled_from_tmdb_vote_average(self):
        import pandas as pd

        row = _normalize_detail(_detail(), "Telugu")
        assert pd.isna(row["imdb_rating"])

    def test_missing_imdb_id_returns_none(self):
        detail = _detail()
        detail["external_ids"] = {}
        assert _normalize_detail(detail, "Telugu") is None

    def test_missing_title_returns_none(self):
        detail = _detail(title="")
        assert _normalize_detail(detail, "Telugu") is None

    def test_unreleased_movie_returns_none(self):
        detail = _detail(status="Post Production")
        assert _normalize_detail(detail, "Telugu") is None

    def test_release_year_parsed_from_release_date(self):
        row = _normalize_detail(_detail(release_date="2019-05-01"), "Telugu")
        assert row["release_year"] == 2019

    def test_missing_release_date_gives_na_year_not_a_crash(self):
        import pandas as pd

        row = _normalize_detail(_detail(release_date=""), "Telugu")
        assert pd.isna(row["release_year"])


class TestBuildSupplementaryCatalog:
    def test_no_api_key_returns_empty_catalog_without_network_call(self, monkeypatch):
        def _fail_if_called(*a, **k):
            raise AssertionError("hit the network despite a missing api_key")

        monkeypatch.setattr("src.movie_discovery.requests.get", _fail_if_called)

        catalog = build_supplementary_catalog(existing_movie_ids=[], api_key=None)

        assert catalog.empty
        assert list(catalog.columns) == ["movie_id", "title", "release_year", "imdb_rating", "languages", *GENRE_COLUMNS]

    def test_already_existing_movie_id_is_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "src.movie_discovery.discover_candidate_ids", lambda code, key, max_pages: [1]
        )
        monkeypatch.setattr(
            "src.movie_discovery.fetch_movie_detail",
            lambda tmdb_id, key: _detail(imdb_id="tt0000001", genres=["Drama"]),
        )

        catalog = build_supplementary_catalog(
            existing_movie_ids=["tt0000001"], api_key="fake-key", languages={"te": "Telugu"}
        )

        assert catalog.empty

    def test_new_movie_is_kept_and_deduped_across_languages(self, monkeypatch):
        monkeypatch.setattr(
            "src.movie_discovery.discover_candidate_ids", lambda code, key, max_pages: [1]
        )
        # Same TMDb id surfaces under two different language searches (e.g. a
        # dubbed title) -- should only appear once in the output, credited to
        # whichever language found it first.
        monkeypatch.setattr(
            "src.movie_discovery.fetch_movie_detail",
            lambda tmdb_id, key: _detail(imdb_id="tt0000002", genres=["Comedy"]),
        )

        catalog = build_supplementary_catalog(
            existing_movie_ids=[], api_key="fake-key", languages={"te": "Telugu", "ta": "Tamil"}
        )

        assert len(catalog) == 1
        assert catalog.iloc[0]["movie_id"] == "tt0000002"

    def test_detail_lookup_failure_is_skipped_not_raised(self, monkeypatch):
        monkeypatch.setattr(
            "src.movie_discovery.discover_candidate_ids", lambda code, key, max_pages: [1]
        )
        monkeypatch.setattr("src.movie_discovery.fetch_movie_detail", lambda tmdb_id, key: None)

        catalog = build_supplementary_catalog(
            existing_movie_ids=[], api_key="fake-key", languages={"te": "Telugu"}
        )

        assert catalog.empty
