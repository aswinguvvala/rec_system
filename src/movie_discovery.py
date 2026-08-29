"""Discover additional Indian-regional movies from TMDb to broaden the catalog.

This is a one-time enrichment tool, not a live pipeline stage: unlike
``src/posters.py`` (which looks up a poster for whatever movie the app
already knows about, on every run), this module *finds new movies* the base
Kaggle dataset (see ``src/data_pipeline.py``) never had -- more onboarding
variety for the "pick movies you like" flow and more candidates for
content-based similarity, at the cost of never having real per-user ratings
(a TMDb-discovered movie will always be a cold-start item; it can only ever
surface through content/SVD-item-similarity or popularity, never through a
user's own collaborative signal -- see ``recommend_similar_to_picks`` in
``src/models.py``, which already handles the ~51% of the base catalog with
zero training ratings, so this isn't a new code path, just more movies
taking it).

Run standalone with ``python -m src.movie_discovery`` (requires
``TMDB_API_KEY`` in the environment) to regenerate
``catalog_supplement/tmdb_supplementary_movies.csv`` -- a small, git-tracked
CSV (same convention as ``results/metrics.json``: it's a real, reviewable
artifact, not something to silently regenerate on every deploy) that
``src/data_pipeline.py``'s ``run_pipeline`` merges into the movie catalog if
present. Its absence is never fatal: a fresh clone with no supplementary
file just gets the base 2,850-movie catalog, same as before this module
existed.

Output rows are normalized into the *exact* schema ``load_movies`` produces
(``movie_id``, ``title``, ``release_year``, ``imdb_rating``, ``languages``,
plus one binary column per genre in ``GENRE_COLUMNS``) so
``merge_supplementary_movies`` can concatenate them with zero special-casing.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.data_pipeline import GENRE_COLUMNS, load_movies
from src.utils import RAW_DATA_DIR, SUPPLEMENTARY_MOVIES_PATH, ensure_dir, get_logger

logger = get_logger(__name__)

TMDB_DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
TMDB_MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
REQUEST_TIMEOUT_SECONDS = 5
MAX_WORKERS = 8
RESULTS_PER_PAGE = 20
# A quality floor so obscure zero-data placeholder entries (unreleased
# titles, TV specials with a stub TMDb page) don't dilute the catalog --
# still returns thousands of real regional releases per language.
MIN_VOTE_COUNT = 5

# original_language ISO 639-1 code -> the exact language name string this
# dataset uses in its `language`/`languages` column (see
# src/data_pipeline.py's load_movies). Must match verbatim: `languages` is
# treated as free text downstream (e.g. `_primary_language` in
# src/models.py splits on ", " and compares full names, not codes).
# Limited to languages TMDb's discover endpoint actually recognizes as ISO
# 639-1 codes -- a few languages already present in the base dataset
# (Manipuri, Haryanvi, Chhattisgarhi, Konkani) have no such code and can't
# be targeted this way.
LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "pa": "Panjabi",
    "gu": "Gujarati",
    "ur": "Urdu",
    "or": "Oriya",
    "as": "Assamese",
    "ne": "Nepali",
}

# TMDb's genre vocabulary doesn't line up 1:1 with GENRE_COLUMNS (see
# src/data_pipeline.py's discovered 21-genre vocabulary). Only names present
# in both are mapped; a TMDb genre with no home here (Documentary, TV Movie)
# is silently dropped -- the same thing load_movies already does for any raw
# genre string outside GENRE_COLUMNS, so a supplementary movie with only an
# unmapped genre just gets an all-zero genre vector, same as ~24% of the
# base catalog already has.
TMDB_GENRE_NAME_MAP: dict[str, str] = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Drama": "Drama",
    "Family": "Family",
    "Fantasy": "Fantasy",
    "History": "History",
    "Horror": "Horror",
    "Music": "Music",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Science Fiction": "Sci-Fi",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western",
}

_EMPTY_SCHEMA_COLUMNS: list[str] = ["movie_id", "title", "release_year", "imdb_rating", "languages", *GENRE_COLUMNS]


def _empty_catalog() -> pd.DataFrame:
    """Return an empty DataFrame with the exact supplementary-catalog schema."""
    return pd.DataFrame(columns=_EMPTY_SCHEMA_COLUMNS)


def discover_candidate_ids(
    language_code: str, api_key: str, max_pages: int
) -> list[int]:
    """Discover candidate TMDb movie ids for one original language.

    Args:
        language_code: ISO 639-1 code, e.g. ``"te"`` for Telugu.
        api_key: TMDb v3 API key.
        max_pages: Number of result pages to fetch (20 movies/page),
            sorted by TMDb popularity descending.

    Returns:
        TMDb movie ids, most popular first. Empty on any request failure --
        this is best-effort enrichment, not a required step.
    """
    ids: list[int] = []
    for page in range(1, max_pages + 1):
        try:
            response = requests.get(
                TMDB_DISCOVER_URL,
                params={
                    "api_key": api_key,
                    "with_original_language": language_code,
                    "sort_by": "popularity.desc",
                    "vote_count.gte": MIN_VOTE_COUNT,
                    "page": page,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("TMDb discover failed for language=%s page=%d: %s", language_code, page, exc)
            break
        except ValueError as exc:  # malformed JSON body
            logger.warning("TMDb returned an unparseable discover response for language=%s: %s", language_code, exc)
            break

        results = payload.get("results", [])
        if not results:
            break
        ids.extend(r["id"] for r in results if r.get("id") is not None)
        if page >= payload.get("total_pages", page):
            break
    return ids


def fetch_movie_detail(tmdb_id: int, api_key: str) -> dict | None:
    """Fetch one movie's TMDb details, including its real IMDb id.

    Args:
        tmdb_id: TMDb's own numeric movie id (from :func:`discover_candidate_ids`).
        api_key: TMDb v3 API key.

    Returns:
        The raw TMDb ``/movie/{id}`` JSON payload (with ``external_ids``
        appended), or ``None`` on any request failure -- a single bad
        lookup must never abort the whole discovery run.
    """
    try:
        response = requests.get(
            TMDB_MOVIE_DETAILS_URL.format(tmdb_id=tmdb_id),
            params={"api_key": api_key, "append_to_response": "external_ids"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("TMDb detail lookup failed for tmdb_id=%d: %s", tmdb_id, exc)
        return None
    except ValueError as exc:  # malformed JSON body
        logger.warning("TMDb returned an unparseable detail response for tmdb_id=%d: %s", tmdb_id, exc)
        return None


def _normalize_detail(detail: dict, language_name: str) -> dict | None:
    """Turn one TMDb movie-detail payload into a row matching load_movies's schema.

    Args:
        detail: Payload from :func:`fetch_movie_detail`.
        language_name: The dataset-vocabulary language name this movie was
            discovered under (see :data:`LANGUAGE_CODE_TO_NAME`) -- used
            as-is for the `languages` column, since TMDb's
            ``original_language`` filter already pins the one language that
            matters for this dataset's "primary language" convention
            (see ``_primary_language`` in ``src/models.py``).

    Returns:
        A row dict, or ``None`` if the movie has no real IMDb id (every
        model and the poster lookup treat ``movie_id`` as a real ``tt`` id,
        so a movie without one can't be represented) or hasn't actually
        been released yet.
    """
    imdb_id = (detail.get("external_ids") or {}).get("imdb_id")
    title = (detail.get("title") or "").strip()
    if not imdb_id or not title:
        return None
    # TMDb's popularity ranking surfaces hyped unreleased/announced titles
    # (sequels, remakes) right alongside real releases -- `status` is the
    # authoritative "has this actually come out" signal TMDb provides,
    # more reliable than comparing release_date to today (which varies by
    # region/festival cut). A recommendation catalog entry nobody could
    # have actually watched yet doesn't belong here.
    if detail.get("status") != "Released":
        return None

    release_date = detail.get("release_date") or ""
    # float("nan") rather than pd.NA: keeps these columns float64 (matching
    # load_movies's own pd.to_numeric-derived dtypes) instead of degrading
    # to object dtype, which otherwise trips a pandas concat FutureWarning
    # in merge_supplementary_movies when a whole column is NA in one frame.
    release_year = float(release_date[:4]) if release_date[:4].isdigit() else float("nan")

    row = {
        "movie_id": imdb_id,
        "title": title,
        "release_year": release_year,
        # Deliberately NOT populated from TMDb's vote_average: this column
        # is documented (see load_movies) as IMDb's own aggregate score, and
        # conflating a different rating source into it -- even though
        # nothing currently reads this purely-informational column -- would
        # be a quiet, misleading data-quality regression for whoever reads
        # it next. NaN is the honest value here.
        "imdb_rating": float("nan"),
        "languages": language_name,
    }
    genre_names = {TMDB_GENRE_NAME_MAP[g["name"]] for g in detail.get("genres", []) if g.get("name") in TMDB_GENRE_NAME_MAP}
    for genre in GENRE_COLUMNS:
        row[genre] = int(genre in genre_names)
    return row


def build_supplementary_catalog(
    existing_movie_ids: Iterable[str],
    api_key: str,
    languages: dict[str, str] | None = None,
    max_pages_per_language: int = 4,
) -> pd.DataFrame:
    """Discover, fetch, and normalize new Indian-regional movies from TMDb.

    Args:
        existing_movie_ids: ``movie_id``s already in the base dataset --
            never re-added, so re-running this against a growing base
            catalog stays additive rather than duplicating work.
        api_key: TMDb v3 API key. If falsy, returns an empty catalog
            without making any network requests (mirrors src/posters.py's
            no-key-is-fine convention).
        languages: ``{iso_code: dataset_language_name}`` to search.
            Defaults to :data:`LANGUAGE_CODE_TO_NAME`.
        max_pages_per_language: Result pages (20 movies/page) to fetch per
            language, most popular first.

    Returns:
        DataFrame in the exact schema :func:`src.data_pipeline.load_movies`
        produces, deduplicated on ``movie_id`` against both
        ``existing_movie_ids`` and itself.
    """
    if not api_key:
        logger.warning("No TMDb API key provided -- returning an empty supplementary catalog.")
        return _empty_catalog()

    languages = languages or LANGUAGE_CODE_TO_NAME
    existing_ids = set(existing_movie_ids)

    candidate_tmdb_ids: dict[int, str] = {}  # tmdb_id -> language_name, first language wins
    for code, name in languages.items():
        ids = discover_candidate_ids(code, api_key, max_pages_per_language)
        logger.info("Discovered %d candidate(s) for language=%s (%s)", len(ids), name, code)
        for tmdb_id in ids:
            candidate_tmdb_ids.setdefault(tmdb_id, name)

    if not candidate_tmdb_ids:
        return _empty_catalog()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        details = list(
            executor.map(lambda item: (item[1], fetch_movie_detail(item[0], api_key)), candidate_tmdb_ids.items())
        )

    rows: list[dict] = []
    seen_movie_ids: set[str] = set()
    for language_name, detail in details:
        if detail is None:
            continue
        row = _normalize_detail(detail, language_name)
        if row is None:
            continue
        movie_id = row["movie_id"]
        if movie_id in existing_ids or movie_id in seen_movie_ids:
            continue
        seen_movie_ids.add(movie_id)
        rows.append(row)

    logger.info(
        "Built supplementary catalog: %d new movie(s) kept out of %d candidate(s) discovered",
        len(rows), len(candidate_tmdb_ids),
    )
    return pd.DataFrame(rows, columns=_EMPTY_SCHEMA_COLUMNS) if rows else _empty_catalog()


def _existing_movie_ids(raw_dir: Path) -> set[str]:
    """Load movie_ids already in the base dataset, so discovery stays additive."""
    return set(load_movies(raw_dir)["movie_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-pages-per-language", type=int, default=4,
        help="TMDb result pages (20 movies/page) to fetch per language (default: 4).",
    )
    parser.add_argument(
        "--out", type=Path, default=SUPPLEMENTARY_MOVIES_PATH,
        help=f"Output CSV path (default: {SUPPLEMENTARY_MOVIES_PATH}).",
    )
    args = parser.parse_args()

    tmdb_api_key = os.environ.get("TMDB_API_KEY")
    if not tmdb_api_key:
        raise SystemExit(
            "TMDB_API_KEY environment variable is not set. Get a free key at "
            "https://www.themoviedb.org/settings/api and set it before running this script."
        )

    catalog = build_supplementary_catalog(
        existing_movie_ids=_existing_movie_ids(RAW_DATA_DIR / "indian_movies"),
        api_key=tmdb_api_key,
        max_pages_per_language=args.max_pages_per_language,
    )
    ensure_dir(args.out.parent)
    catalog.to_csv(args.out, index=False)
    logger.info("Wrote %d supplementary movie(s) to %s", len(catalog), args.out)
