"""Movie poster lookup via The Movie Database (TMDb) API.

Optional, best-effort enrichment for the Streamlit demo: look up a poster
image URL on TMDb, either by exact IMDb id (:func:`get_poster_url_by_imdb_id`
-- the Indian Regional Movie Dataset's ``movie_id`` *is* a real IMDb ``tt``
id, so this is the reliable, no-guessing path) or by fuzzy title search
(:func:`get_poster_url` -- kept for titles with no known IMDb id). This
module never raises out to its caller on a missing API key, network
failure, or unmatched title/id -- it always returns ``None`` instead, since
a broken poster lookup must never take down the recommendation demo
itself. The app is expected to fall back to a placeholder poster whenever a
lookup comes back empty.

This module does no Streamlit-specific caching -- callers (``app.py``)
wrap the batch functions in ``st.cache_data`` so repeat lookups are free
across reruns.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests

from src.utils import get_logger

logger = get_logger(__name__)

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_FIND_URL = "https://api.themoviedb.org/3/find/{imdb_id}"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"
REQUEST_TIMEOUT_SECONDS = 5
MAX_WORKERS = 8

# MovieLens moves a leading article to the end for alphabetization, e.g.
# "Truth About Cats & Dogs, The (1996)" or "Postino, Il (1994)". TMDb's
# search matches much better on the natural word order, so this undoes
# that transformation before searching.
_TITLE_YEAR_RE = re.compile(r"^(?P<base>.*?)(?:,\s*(?P<article>The|A|An))?\s*\((?P<year>\d{4})\)\s*$")


def _normalize_title(raw_title: str) -> tuple[str, int | None]:
    """Split a MovieLens title into a TMDb-searchable title and release year.

    Args:
        raw_title: Title as stored in ``movies_df``, e.g.
            ``"Truth About Cats & Dogs, The (1996)"``.

    Returns:
        ``(search_title, year)``. ``year`` is ``None`` if the title didn't
        end in the expected ``(YYYY)`` suffix.
    """
    match = _TITLE_YEAR_RE.match(raw_title.strip())
    if not match:
        return raw_title.strip(), None
    base, article, year = match["base"].strip(), match["article"], match["year"]
    search_title = f"{article} {base}" if article else base
    return search_title, int(year)


def get_poster_url(title: str, api_key: str | None) -> str | None:
    """Look up a single movie's poster image URL on TMDb.

    Args:
        title: Movie title in MovieLens format, e.g. ``"Toy Story (1995)"``.
        api_key: TMDb v3 API key. If falsy, no request is made.

    Returns:
        A full poster image URL, or ``None`` if no key was configured, the
        request failed, or no matching poster was found.
    """
    if not api_key:
        return None
    search_title, year = _normalize_title(title)
    try:
        response = requests.get(
            TMDB_SEARCH_URL,
            params={"api_key": api_key, "query": search_title, "year": year},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results and year is not None:
            # Release-year mismatches (region/rerelease dates) are common enough
            # in TMDb that it's worth one unfiltered retry before giving up.
            response = requests.get(
                TMDB_SEARCH_URL,
                params={"api_key": api_key, "query": search_title},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
    except requests.exceptions.RequestException as exc:
        logger.warning("TMDb lookup failed for %r: %s", title, exc)
        return None
    except ValueError as exc:  # malformed JSON body
        logger.warning("TMDb returned an unparseable response for %r: %s", title, exc)
        return None

    poster_path = next((r["poster_path"] for r in results if r.get("poster_path")), None)
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None


def get_poster_urls(titles: Iterable[str], api_key: str | None) -> dict[str, str | None]:
    """Look up poster URLs for many titles concurrently.

    Args:
        titles: Movie titles in MovieLens format. Duplicates are looked up once.
        api_key: TMDb v3 API key. If falsy, every title maps to ``None``
            without making any network requests.

    Returns:
        Mapping from each input title to its poster URL, or ``None`` where
        no poster could be found.
    """
    unique_titles = list(dict.fromkeys(titles))
    if not unique_titles:
        return {}
    if not api_key:
        return dict.fromkeys(unique_titles)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(unique_titles))) as executor:
        posters = list(executor.map(lambda t: get_poster_url(t, api_key), unique_titles))
    return dict(zip(unique_titles, posters))


def get_poster_url_by_imdb_id(imdb_id: str, api_key: str | None) -> str | None:
    """Look up a single movie's poster image URL on TMDb by its exact IMDb id.

    Strictly more reliable than :func:`get_poster_url`'s fuzzy title search
    when the caller already has a real IMDb ``tt`` id -- no title
    normalization or year-matching heuristics needed.

    Args:
        imdb_id: IMDb id, e.g. ``"tt5286444"``.
        api_key: TMDb v3 API key. If falsy, no request is made.

    Returns:
        A full poster image URL, or ``None`` if no key was configured, the
        request failed, or TMDb has no matching movie for this id.
    """
    if not api_key:
        return None
    try:
        response = requests.get(
            TMDB_FIND_URL.format(imdb_id=imdb_id),
            params={"api_key": api_key, "external_source": "imdb_id"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json().get("movie_results", [])
    except requests.exceptions.RequestException as exc:
        logger.warning("TMDb find-by-id lookup failed for %r: %s", imdb_id, exc)
        return None
    except ValueError as exc:  # malformed JSON body
        logger.warning("TMDb returned an unparseable response for %r: %s", imdb_id, exc)
        return None

    poster_path = next((r["poster_path"] for r in results if r.get("poster_path")), None)
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None


def get_poster_urls_by_imdb_id(imdb_ids: Iterable[str], api_key: str | None) -> dict[str, str | None]:
    """Look up poster URLs for many IMDb ids concurrently.

    Args:
        imdb_ids: IMDb ids, e.g. ``"tt5286444"``. Duplicates are looked up once.
        api_key: TMDb v3 API key. If falsy, every id maps to ``None``
            without making any network requests.

    Returns:
        Mapping from each input id to its poster URL, or ``None`` where no
        poster could be found. See :func:`get_poster_url_by_imdb_id`.
    """
    unique_ids = list(dict.fromkeys(imdb_ids))
    if not unique_ids:
        return {}
    if not api_key:
        return dict.fromkeys(unique_ids)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(unique_ids))) as executor:
        posters = list(executor.map(lambda i: get_poster_url_by_imdb_id(i, api_key), unique_ids))
    return dict(zip(unique_ids, posters))
