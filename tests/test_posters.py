"""Tests for src/posters.py: TMDb title normalization and poster lookup."""

import requests

from src.posters import _normalize_title, get_poster_url, get_poster_urls


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


class TestNormalizeTitle:
    def test_plain_title_with_year(self):
        assert _normalize_title("Toy Story (1995)") == ("Toy Story", 1995)

    def test_relocated_article_the_is_moved_to_front(self):
        assert _normalize_title("Truth About Cats & Dogs, The (1996)") == (
            "The Truth About Cats & Dogs",
            1996,
        )

    def test_relocated_article_a_is_moved_to_front(self):
        assert _normalize_title("Room with a View, A (1986)") == ("A Room with a View", 1986)

    def test_title_without_year_suffix_is_returned_unchanged(self):
        assert _normalize_title("Untitled Movie") == ("Untitled Movie", None)


class TestGetPosterUrl:
    def test_no_api_key_returns_none_without_network_call(self, monkeypatch):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("get_poster_url hit the network despite a missing api_key")

        monkeypatch.setattr("src.posters.requests.get", _fail_if_called)

        assert get_poster_url("Toy Story (1995)", api_key=None) is None

    def test_returns_full_image_url_on_a_match(self, monkeypatch):
        monkeypatch.setattr(
            "src.posters.requests.get",
            lambda *a, **k: _FakeResponse({"results": [{"poster_path": "/abc123.jpg"}]}),
        )

        url = get_poster_url("Toy Story (1995)", api_key="fake-key")

        assert url == "https://image.tmdb.org/t/p/w342/abc123.jpg"

    def test_no_results_returns_none(self, monkeypatch):
        monkeypatch.setattr("src.posters.requests.get", lambda *a, **k: _FakeResponse({"results": []}))

        assert get_poster_url("Some Obscure Movie (1995)", api_key="fake-key") is None

    def test_result_missing_poster_path_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "src.posters.requests.get",
            lambda *a, **k: _FakeResponse({"results": [{"poster_path": None}]}),
        )

        assert get_poster_url("Toy Story (1995)", api_key="fake-key") is None

    def test_network_failure_returns_none_instead_of_raising(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise requests.exceptions.ConnectionError("simulated network failure")

        monkeypatch.setattr("src.posters.requests.get", _raise)

        assert get_poster_url("Toy Story (1995)", api_key="fake-key") is None

    def test_http_error_status_returns_none_instead_of_raising(self, monkeypatch):
        monkeypatch.setattr(
            "src.posters.requests.get", lambda *a, **k: _FakeResponse({}, status_code=401)
        )

        assert get_poster_url("Toy Story (1995)", api_key="bad-key") is None


class TestGetPosterUrls:
    def test_empty_input_returns_empty_dict_without_network_call(self, monkeypatch):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("get_poster_urls hit the network for an empty title list")

        monkeypatch.setattr("src.posters.requests.get", _fail_if_called)

        assert get_poster_urls([], api_key="fake-key") == {}

    def test_no_api_key_maps_every_title_to_none(self):
        result = get_poster_urls(["Toy Story (1995)", "Casablanca (1942)"], api_key=None)

        assert result == {"Toy Story (1995)": None, "Casablanca (1942)": None}

    def test_duplicate_titles_are_looked_up_once_each(self, monkeypatch):
        call_count = 0

        def _get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _FakeResponse({"results": [{"poster_path": "/x.jpg"}]})

        monkeypatch.setattr("src.posters.requests.get", _get)

        titles = ["Toy Story (1995)", "Toy Story (1995)", "Casablanca (1942)"]
        result = get_poster_urls(titles, api_key="fake-key")

        assert call_count == 2  # one per unique title
        assert result["Toy Story (1995)"] == "https://image.tmdb.org/t/p/w342/x.jpg"
        assert result["Casablanca (1942)"] == "https://image.tmdb.org/t/p/w342/x.jpg"
