"""Download, clean, and split the Indian Regional Movie Dataset.

Source: Agarwal et al., "Indian Regional Movie Dataset for Recommender
Systems" (arXiv:1801.02203), mirrored on Kaggle as
``snathjr/indian-regional-movie``. Unlike the MovieLens 100K dataset this
pipeline originally used, there is no unauthenticated public URL for this
data -- Kaggle gates every dataset download behind its API, so a Kaggle
account and API token are required (see :func:`download_indian_movies_dataset`).

Two structural differences from MovieLens drive most of the design choices
below, and both are load-bearing enough to call out up front:

1. **Ratings are a ternary preference signal (-1/0/1), not 1-5 stars.**
   ``1`` = liked, ``0`` = disliked/not interested, ``-1`` = ambiguous/skipped.
   This pipeline does not force that onto a fake 1-5 scale; ``src/models.py``
   treats ``[-1, 1]`` as the native rating range, so RMSE/MAE here measure
   preference-*score* prediction, not star-rating prediction.
2. **No timestamp field exists**, so the chronological train/test split this
   pipeline used for MovieLens isn't possible -- see
   :func:`random_train_test_split` for the honest replacement.

Pipeline stages: :func:`download_indian_movies_dataset` (cached download +
extract) -> :func:`load_ratings` / :func:`load_movies` / :func:`load_users`
(parse the raw files) -> :func:`merge_supplementary_movies` (optionally
folds in a TMDb-sourced catalog of additional Indian-regional movies with no
ratings of their own -- see ``src/movie_discovery.py``) -> :func:`merge_data`
(join into one denormalized frame) -> :func:`random_train_test_split`
(per-user split).
:func:`run_pipeline` orchestrates all of the above, plus referential-integrity
cleanup (dropping ratings that reference a user/movie that got filtered out
elsewhere -- see its docstring), and writes the processed CSVs to
``data/processed/``.

Run standalone with ``python -m src.data_pipeline``.
"""

from __future__ import annotations

import json
import socket
import string
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SUPPLEMENTARY_MOVIES_PATH,
    ensure_dir,
    get_logger,
)

logger = get_logger(__name__)

KAGGLE_DATASET = "snathjr/indian-regional-movie"
# The kaggle package doesn't reliably apply its own per-request timeout across versions
# (see download_indian_movies_dataset) -- this bounds how long a stalled Kaggle API call
# can silently hang the whole app before failing with an actionable error instead.
KAGGLE_SOCKET_TIMEOUT_SECONDS = 30

# The dataset's real, discovered genre vocabulary (21 genres; see claude.md
# for how this was determined). Movies with no listed genre -- about a
# quarter of the catalog -- get an all-zero vector across these columns
# rather than a fabricated "unknown" placeholder.
GENRE_COLUMNS: list[str] = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical",
    "Mystery", "News", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western",
]

# ratings.json encodes each vote as a string; anything outside this map is
# logged and dropped rather than guessed at.
_RATING_VALUE_MAP: dict[str, int] = {"1": 1, "0": 0, "-1": -1}

_MIN_USER_ID_LENGTH = 3
_JUNK_ALPHABET_RUN_MIN_LENGTH = 5

# The dataset paper (Agarwal et al.) was submitted January 2018, so survey
# responses were collected in 2017. "age" is therefore an approximate
# as-of-2017 snapshot derived from date of birth, not a live age -- the
# same static-snapshot convention MovieLens's own age field used.
_SURVEY_COLLECTION_YEAR = 2017


class DataPipelineError(Exception):
    """Base exception for all data pipeline failures."""


class DataDownloadError(DataPipelineError):
    """Raised when the dataset cannot be downloaded from Kaggle or extracted."""


class DataFormatError(DataPipelineError):
    """Raised when a raw dataset file is missing or fails to parse."""


def download_indian_movies_dataset(dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Download and extract the Indian Regional Movie Dataset, using a local cache.

    Requires a Kaggle account and API token: either ``~/.kaggle/kaggle.json``
    or the ``KAGGLE_USERNAME``/``KAGGLE_KEY`` environment variables. Get a
    free token at https://www.kaggle.com/settings under "API".

    If ``dest_dir/indian_movies/ratings.json`` already exists, the download
    is skipped entirely so repeated runs (including every Streamlit app
    launch) are fast and don't hit the network.

    Args:
        dest_dir: Directory that will contain the extracted dataset files.

    Returns:
        Path to the directory containing ``movies.csv``, ``users.csv``, and
        ``ratings.json``.

    Raises:
        DataDownloadError: If Kaggle credentials aren't configured, or the
            dataset can't be downloaded or extracted.
    """
    extracted_dir = dest_dir / "indian_movies"
    if (extracted_dir / "ratings.json").exists():
        logger.info("Using cached Indian movie dataset at %s", extracted_dir)
        return extracted_dir

    ensure_dir(extracted_dir)
    logger.info("Downloading %s from Kaggle (this may take a moment)...", KAGGLE_DATASET)
    # The kaggle package's own HTTP calls (both its own import-time auth attempt and
    # dataset_download_files below) don't take an explicit per-request timeout in every
    # version, so a slow/unreachable Kaggle endpoint can hang this call indefinitely with
    # no error and no log output -- silently wedging the whole app. socket.setdefaulttimeout
    # is a blunt instrument (process-wide), so it's set right before this specific network
    # operation and restored immediately after, rather than left on for the app's lifetime.
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(KAGGLE_SOCKET_TIMEOUT_SECONDS)
    try:
        import kaggle  # local import: importing this module authenticates as a side effect

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(extracted_dir), unzip=True)
    except Exception as exc:  # noqa: BLE001 - the kaggle package doesn't expose a small, stable
        # set of exception types across versions for "no credentials" vs. "network failure" vs.
        # "dataset moved" vs. "timed out"; all surface as generic exceptions, so this boundary
        # is intentionally broad. What matters is that the resulting message stays actionable.
        raise DataDownloadError(
            f"Failed to download the Indian movie dataset ({KAGGLE_DATASET}) from Kaggle: {exc}. "
            "Make sure a Kaggle API token is configured at ~/.kaggle/kaggle.json (or the "
            "KAGGLE_USERNAME/KAGGLE_KEY env vars) -- get a free one at "
            "https://www.kaggle.com/settings under 'API'."
        ) from exc
    finally:
        socket.setdefaulttimeout(previous_timeout)

    if not (extracted_dir / "ratings.json").exists():
        raise DataDownloadError(
            f"Download reported success but {extracted_dir / 'ratings.json'} is missing; "
            "the dataset's file layout on Kaggle may have changed."
        )
    logger.info("Indian movie dataset ready at %s", extracted_dir)
    return extracted_dir


def _parse_json_string_list(raw: str) -> list[str]:
    """Parse a JSON-array-encoded CSV cell (e.g. ``'[ "Hindi" ]'``) into a list of strings."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def load_movies(raw_dir: Path) -> pd.DataFrame:
    """Parse ``movies.csv`` into a movie metadata frame.

    Args:
        raw_dir: Path to the extracted dataset directory.

    Returns:
        DataFrame with ``movie_id`` (the movie's IMDb ``tt`` id -- used
        as-is rather than remapped to a synthetic integer, since every
        model in ``src/models.py`` already treats ``movie_id`` as an
        opaque, hashable key), ``title``, ``release_year``, ``imdb_rating``
        (IMDb's own aggregate score, informational only -- not used by any
        model), ``languages``, and one binary column per genre in
        :data:`GENRE_COLUMNS`.

    Raises:
        DataFormatError: If the file is missing or fails to parse.
    """
    path = raw_dir / "movies.csv"
    try:
        raw = pd.read_csv(path, dtype={"movie_id": str})
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        raise DataFormatError(f"Failed to read movies file {path}: {exc}") from exc

    movies = pd.DataFrame({"movie_id": raw["movie_id"]})
    movies["title"] = raw["name"].fillna("").str.strip()
    movies["release_year"] = pd.to_datetime(raw["released"], errors="coerce").dt.year
    movies["imdb_rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    movies["languages"] = raw["language"].fillna("[]").apply(
        lambda s: ", ".join(_parse_json_string_list(s))
    )

    genre_lists = raw["genre"].fillna("[]").apply(_parse_json_string_list)
    for genre in GENRE_COLUMNS:
        movies[genre] = genre_lists.apply(lambda genres, g=genre: int(g in genres))

    return movies.drop_duplicates(subset="movie_id").reset_index(drop=True)


def load_supplementary_movies(path: Path = SUPPLEMENTARY_MOVIES_PATH) -> pd.DataFrame:
    """Load the optional TMDb-sourced supplementary movie catalog, if present.

    Built once by ``python -m src.movie_discovery`` (see that module) and
    committed to the repo -- absent by default until that's been run.
    Missing or unreadable is never fatal: :func:`run_pipeline` works fine
    without it, just with a smaller catalog, the same graceful-degradation
    convention ``src/posters.py`` uses for a missing TMDb key.

    Args:
        path: Path to the supplementary CSV, in the same schema
            :func:`load_movies` produces.

    Returns:
        DataFrame with ``movie_id``, ``title``, ``release_year``,
        ``imdb_rating``, ``languages``, and one column per
        :data:`GENRE_COLUMNS`. Empty (but correctly shaped) if the file is
        missing or fails to parse.
    """
    empty = pd.DataFrame(columns=["movie_id", "title", "release_year", "imdb_rating", "languages", *GENRE_COLUMNS])
    if not path.exists():
        return empty
    try:
        return pd.read_csv(path, dtype={"movie_id": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("Failed to read supplementary movie catalog at %s: %s -- ignoring it.", path, exc)
        return empty


def merge_supplementary_movies(base_movies: pd.DataFrame, supplementary_movies: pd.DataFrame) -> pd.DataFrame:
    """Union in supplementary movies, keeping the base dataset's row on any id collision.

    Args:
        base_movies: Output of :func:`load_movies` (the real Kaggle-sourced catalog).
        supplementary_movies: Output of :func:`load_supplementary_movies`.

    Returns:
        ``base_movies`` with any non-overlapping supplementary rows appended.
        A ``movie_id`` already present in ``base_movies`` always wins --
        the real per-user-rated dataset is authoritative for any title it
        already has an opinion about.
    """
    if supplementary_movies.empty:
        return base_movies
    combined = pd.concat([base_movies, supplementary_movies], ignore_index=True)
    combined = combined.drop_duplicates(subset="movie_id", keep="first").reset_index(drop=True)
    n_added = len(combined) - len(base_movies)
    n_skipped = len(supplementary_movies) - n_added
    logger.info(
        "Merged %d supplementary movie(s) into the catalog (%d already present, skipped)",
        n_added, n_skipped,
    )
    return combined


def _is_junk_user_id(user_id: str) -> bool:
    """Flag obvious survey test-submissions rather than real user handles.

    Two patterns showed up in a manual inspection of the raw data: 1-2
    character ids (``"n"``, ``"p"``) and a literal "ABCDEFGHI JKLM" -- someone
    typing the alphabet into the form. Both are caught generically here
    rather than as a hardcoded blocklist, since more of the same is likely
    in the long tail.
    """
    if len(user_id) < _MIN_USER_ID_LENGTH:
        return True
    letters_only = "".join(ch.lower() for ch in user_id if ch.isalpha())
    return (
        len(letters_only) >= _JUNK_ALPHABET_RUN_MIN_LENGTH
        and letters_only in string.ascii_lowercase
    )


def load_users(raw_dir: Path) -> pd.DataFrame:
    """Parse ``users.csv`` into a user demographics frame.

    Args:
        raw_dir: Path to the extracted dataset directory.

    Returns:
        DataFrame with ``user_id``, ``age`` (approximate, see
        :data:`_SURVEY_COLLECTION_YEAR`), ``gender``, ``occupation``, ``state``.
        Rows with a junk/placeholder id (see :func:`_is_junk_user_id`) are
        dropped, with the count logged.

    Raises:
        DataFormatError: If the file is missing or fails to parse.
    """
    path = raw_dir / "users.csv"
    try:
        raw = pd.read_csv(path, dtype={"_id": str})
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        raise DataFormatError(f"Failed to read users file {path}: {exc}") from exc

    users = pd.DataFrame({"user_id": raw["_id"].fillna("").str.strip()})
    n_before = len(users)
    users = users[~users["user_id"].apply(_is_junk_user_id)].copy()
    n_dropped = n_before - len(users)
    if n_dropped:
        logger.info("Dropped %d user row(s) with a junk/placeholder id (survey test submissions)", n_dropped)

    raw = raw.loc[users.index]
    birth_year = pd.to_datetime(raw["dob"], format="%d-%m-%Y", errors="coerce").dt.year
    users["age"] = _SURVEY_COLLECTION_YEAR - birth_year
    users["gender"] = raw["gender"].fillna("").replace("", "Unknown")
    users["occupation"] = raw["job"].fillna("Unknown").replace("", "Unknown")
    users["state"] = raw["state"].fillna("Unknown").replace("", "Unknown")

    return users.drop_duplicates(subset="user_id").reset_index(drop=True)


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """Parse ``ratings.json`` into a ratings frame.

    The file is mongoexport-style: one JSON object per line, each shaped
    like ``{"_id": "<user_id>", "rated": {"<movie_id>": ["1"], ..., "submit":
    ["submit"]}}``. The ``"submit"`` key is a form-submission artifact, not a
    movie rating, and is dropped.

    Args:
        raw_dir: Path to the extracted dataset directory.

    Returns:
        DataFrame with ``user_id``, ``movie_id``, ``rating`` (int, one of
        -1/0/1 -- see the module docstring for what these mean). No
        timestamp column: this export doesn't carry one.

    Raises:
        DataFormatError: If the file is missing or fails to parse.
    """
    path = raw_dir / "ratings.json"
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataFormatError(f"Failed to read ratings file {path}: {exc}") from exc

    rows: list[tuple[str, str, int]] = []
    skipped_values = 0
    for record in records:
        user_id = str(record.get("_id", "")).strip()
        if not user_id:
            continue
        for movie_id, value in record.get("rated", {}).items():
            if movie_id == "submit":
                continue
            raw_value = value[0] if isinstance(value, list) else value
            rating = _RATING_VALUE_MAP.get(str(raw_value))
            if rating is None:
                skipped_values += 1
                continue
            rows.append((user_id, movie_id, rating))

    if skipped_values:
        logger.warning(
            "Skipped %d rating entr(y/ies) with an unrecognized value (expected one of -1/0/1)",
            skipped_values,
        )

    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def merge_data(
    ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame
) -> pd.DataFrame:
    """Join ratings with movie metadata and user demographics.

    Args:
        ratings: Output of :func:`load_ratings`.
        movies: Output of :func:`load_movies`.
        users: Output of :func:`load_users`.

    Returns:
        A single denormalized DataFrame, one row per rating, with movie and
        user columns attached.
    """
    merged = ratings.merge(movies, on="movie_id", how="left")
    merged = merged.merge(users, on="user_id", how="left")
    return merged


def random_train_test_split(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
    min_ratings_for_test: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/test sets randomly, per user.

    Design decision: this dataset carries no timestamp (see
    :func:`load_ratings`), so the per-user *chronological* split this
    pipeline used against MovieLens -- reproducing the real deployment
    condition of predicting future ratings from past ones -- isn't possible
    here. A per-user **random** split is the honest next-best option: it
    still avoids the failure mode where a global (non-per-user) split would
    leave some users entirely out of train or entirely out of test, which
    would make per-user ranking metrics (Precision@K, Recall@K, NDCG@K)
    impossible to compute for them. What it can't reproduce is the
    "no peeking at the future" guarantee a timestamp would have given, so
    the metrics from this split are directly comparable to the MovieLens
    version's *ranking* story but not to its "honest production simulation"
    framing.

    Users with fewer than ``min_ratings_for_test`` ratings are kept entirely
    in train, matching the same cold-start rationale as before: holding out
    a test point for them would leave too little signal to predict from,
    and that's exactly the scenario the cold-start fallback handles.

    Args:
        ratings: Ratings frame with a ``user_id`` column.
        test_frac: Fraction of each eligible user's ratings to hold out.
        min_ratings_for_test: Minimum ratings a user must have before any
            of theirs are held out for test.
        random_state: Seed for the per-user shuffle, for reproducibility.

    Returns:
        ``(train_df, test_df)`` tuple of DataFrames with the same columns
        as ``ratings``.
    """
    rng = np.random.default_rng(random_state)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in ratings.groupby("user_id", sort=False):
        n_test = int(len(group) * test_frac) if len(group) >= min_ratings_for_test else 0
        if n_test == 0:
            train_parts.append(group)
            continue
        shuffled_positions = rng.permutation(len(group))
        test_df_positions = shuffled_positions[:n_test]
        train_df_positions = shuffled_positions[n_test:]
        train_parts.append(group.iloc[train_df_positions])
        test_parts.append(group.iloc[test_df_positions])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0]
    return train_df, test_df


def _processed_cache_is_usable(movies_path: Path) -> bool:
    """Check that a cached ``movies.csv`` actually matches this pipeline's schema.

    Guards against a real failure mode: ``data/processed/`` is gitignored, so
    it isn't wiped by a code-only redeploy -- a persistent host (e.g. a
    Streamlit Cloud container reused across deploys) can retain processed
    CSVs from a *previous, different* dataset. That bit this project for
    real when it moved off MovieLens: the leftover MovieLens ``movies.csv``
    (19 genre columns) satisfied the old "do the files exist" check, so it
    got loaded as-is, and every model choked with ``KeyError`` on the six
    genre columns (``Biography``, ``Family``, ``History``, ``Music``,
    ``News``, ``Sport``) that only exist in this dataset's taxonomy. Only
    the header needs reading, not the full file, so this check is cheap
    enough to run on every cache hit.

    Args:
        movies_path: Path to the cached ``movies.csv``.

    Returns:
        ``True`` if the file exists and its header contains every column in
        :data:`GENRE_COLUMNS`.
    """
    if not movies_path.exists():
        return False
    try:
        header_columns = set(pd.read_csv(movies_path, nrows=0).columns)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    return set(GENRE_COLUMNS).issubset(header_columns)


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    supplementary_movies_path: Path = SUPPLEMENTARY_MOVIES_PATH,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the full pipeline: download, load, clean, split, and persist.

    Idempotent: if the processed CSVs already exist *and match this
    pipeline's current schema* (see :func:`_processed_cache_is_usable`) and
    ``force`` is ``False``, they are loaded from disk instead of being
    recomputed. This is what lets ``app.py`` call this on every Streamlit
    launch cheaply.

    Args:
        raw_dir: Directory for the raw/cached dataset download.
        processed_dir: Directory to write/read processed CSVs.
        supplementary_movies_path: Path to the optional TMDb-sourced
            supplementary movie catalog (see :func:`load_supplementary_movies`
            and ``src/movie_discovery.py``). Exposed as a parameter (rather
            than always reading the module-level default) so tests can
            isolate a fake raw/processed dir without also picking up the
            real, git-tracked supplementary file sitting in the repo.
        force: If True, re-run and overwrite even if processed files exist.

    Returns:
        Dict with keys ``"train"``, ``"test"``, ``"movies"``, ``"users"``,
        each mapping to a DataFrame.

    Raises:
        DataDownloadError: If the raw dataset can't be obtained.
        DataFormatError: If a raw file can't be parsed.
    """
    ensure_dir(processed_dir)
    paths = {name: processed_dir / f"{name}.csv" for name in ("train", "test", "movies", "users")}

    if not force and all(p.exists() for p in paths.values()):
        if _processed_cache_is_usable(paths["movies"]):
            logger.info("Loading cached processed data from %s", processed_dir)
            return {
                name: pd.read_csv(p, dtype={"user_id": str, "movie_id": str})
                for name, p in paths.items()
            }
        logger.warning(
            "Cached data at %s doesn't match this pipeline's current schema (likely left over "
            "from a previous dataset) -- ignoring it and rebuilding from scratch.",
            processed_dir,
        )

    raw_dir_path = download_indian_movies_dataset(raw_dir)
    ratings = load_ratings(raw_dir_path)
    movies = load_movies(raw_dir_path)
    movies = merge_supplementary_movies(movies, load_supplementary_movies(supplementary_movies_path))
    users = load_users(raw_dir_path)
    logger.info(
        "Loaded %d ratings, %d movies, %d users (before referential-integrity cleanup)",
        len(ratings), len(movies), len(users),
    )

    n_before = len(ratings)
    ratings = ratings[
        ratings["movie_id"].isin(set(movies["movie_id"])) & ratings["user_id"].isin(set(users["user_id"]))
    ].reset_index(drop=True)
    if len(ratings) != n_before:
        logger.info(
            "Dropped %d rating(s) referencing a movie/user missing from movies.csv/users.csv "
            "after cleanup (e.g. a junk user id)",
            n_before - len(ratings),
        )

    train_df, test_df = random_train_test_split(ratings)
    logger.info(
        "Random per-user split: %d train ratings, %d test ratings (%.1f%% held out)",
        len(train_df), len(test_df), 100 * len(test_df) / len(ratings) if len(ratings) else 0.0,
    )

    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)
    movies.to_csv(paths["movies"], index=False)
    users.to_csv(paths["users"], index=False)
    logger.info("Wrote processed data to %s", processed_dir)

    return {"train": train_df, "test": test_df, "movies": movies, "users": users}


if __name__ == "__main__":
    data = run_pipeline()
    for name, df in data.items():
        logger.info("%s: %s rows, columns=%s", name, len(df), list(df.columns)[:6])
