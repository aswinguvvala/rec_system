"""Download, clean, and split the MovieLens 100K dataset.

Pipeline stages: :func:`download_movielens_100k` (cached download + extract)
-> :func:`load_ratings` / :func:`load_movies` / :func:`load_users` (parse
the raw ``u.*`` files) -> :func:`merge_data` (join into one denormalized
frame) -> :func:`chronological_train_test_split` (per-user, time-ordered
split). :func:`run_pipeline` orchestrates all of the above and writes the
processed CSVs to ``data/processed/``.

Run standalone with ``python -m src.data_pipeline``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_dir, get_logger

logger = get_logger(__name__)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRE_COLUMNS: list[str] = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
_MOVIE_COLUMNS: list[str] = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
] + GENRE_COLUMNS
_USER_COLUMNS: list[str] = ["user_id", "age", "gender", "occupation", "zip_code"]
_RATING_COLUMNS: list[str] = ["user_id", "movie_id", "rating", "timestamp"]


class DataPipelineError(Exception):
    """Base exception for all data pipeline failures."""


class DataDownloadError(DataPipelineError):
    """Raised when the MovieLens archive cannot be downloaded or extracted."""


class DataFormatError(DataPipelineError):
    """Raised when a raw MovieLens file is missing or fails to parse."""


def download_movielens_100k(dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Download and extract the MovieLens 100K dataset, using a local cache.

    If ``dest_dir/ml-100k/u.data`` already exists, the download is skipped
    entirely so repeated runs (including every Streamlit app launch) are
    fast and don't hit the network.

    Args:
        dest_dir: Directory that will contain the extracted ``ml-100k``
            folder. Created if it doesn't exist.

    Returns:
        Path to the extracted ``ml-100k`` directory.

    Raises:
        DataDownloadError: If the archive can't be downloaded, isn't a
            valid zip file, or doesn't contain the expected contents.
    """
    extracted_dir = dest_dir / "ml-100k"
    if (extracted_dir / "u.data").exists():
        logger.info("Using cached MovieLens 100K at %s", extracted_dir)
        return extracted_dir

    ensure_dir(dest_dir)
    zip_path = dest_dir / "ml-100k.zip"
    try:
        logger.info("Downloading MovieLens 100K from %s", MOVIELENS_URL)
        response = requests.get(MOVIELENS_URL, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    except requests.exceptions.RequestException as exc:
        raise DataDownloadError(
            f"Failed to download MovieLens 100K from {MOVIELENS_URL}: {exc}"
        ) from exc

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        raise DataDownloadError(
            f"Downloaded file at {zip_path} is not a valid zip archive: {exc}"
        ) from exc
    finally:
        zip_path.unlink(missing_ok=True)

    if not (extracted_dir / "u.data").exists():
        raise DataDownloadError(
            f"Extraction reported success but {extracted_dir / 'u.data'} is missing; "
            "the MovieLens archive layout may have changed."
        )
    logger.info("MovieLens 100K ready at %s", extracted_dir)
    return extracted_dir


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """Parse ``u.data`` into a ratings frame.

    Args:
        raw_dir: Path to the extracted ``ml-100k`` directory.

    Returns:
        DataFrame with columns ``user_id``, ``movie_id``, ``rating``,
        ``timestamp`` (Unix epoch seconds).

    Raises:
        DataFormatError: If the file is missing or fails to parse.
    """
    path = raw_dir / "u.data"
    try:
        return pd.read_csv(path, sep="\t", names=_RATING_COLUMNS, engine="python")
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        raise DataFormatError(f"Failed to read ratings file {path}: {exc}") from exc


def load_movies(raw_dir: Path) -> pd.DataFrame:
    """Parse ``u.item`` into a movie metadata frame.

    Args:
        raw_dir: Path to the extracted ``ml-100k`` directory.

    Returns:
        DataFrame with ``movie_id``, ``title``, ``release_date``,
        ``video_release_date``, ``imdb_url``, and one binary column per
        genre in :data:`GENRE_COLUMNS`.

    Raises:
        DataFormatError: If the file is missing or fails to parse.
    """
    path = raw_dir / "u.item"
    try:
        # u.item is Latin-1 encoded (not UTF-8) in the original MovieLens release.
        return pd.read_csv(
            path, sep="|", names=_MOVIE_COLUMNS, encoding="latin-1", engine="python"
        )
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        raise DataFormatError(f"Failed to read movies file {path}: {exc}") from exc


def load_users(raw_dir: Path) -> pd.DataFrame:
    """Parse ``u.user`` into a user demographics frame.

    Args:
        raw_dir: Path to the extracted ``ml-100k`` directory.

    Returns:
        DataFrame with ``user_id``, ``age``, ``gender``, ``occupation``,
        ``zip_code``.

    Raises:
        DataFormatError: If the file is missing or fails to parse.
    """
    path = raw_dir / "u.user"
    try:
        return pd.read_csv(path, sep="|", names=_USER_COLUMNS, engine="python")
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        raise DataFormatError(f"Failed to read users file {path}: {exc}") from exc


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


def chronological_train_test_split(
    ratings: pd.DataFrame, test_frac: float = 0.2, min_ratings_for_test: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/test sets chronologically, per user.

    Design decision: this pipeline uses a **per-user chronological split**
    rather than a random/stratified split. A random split lets the model
    train on a user's *later* ratings to predict their *earlier* ones,
    which the model could never actually observe at serving time — that's
    label leakage disguised as good accuracy. A chronological split
    reproduces the real deployment condition (predict future ratings from
    past ones), so the reported metrics are honest about how the system
    would perform in production, at the cost of being slightly less
    standard for apples-to-apples benchmarking against papers that use a
    random split.

    The split is done per user (not on the global timeline) so that every
    active user contributes to both sets — a global timeline split would
    put all of one user's ratings entirely in train or entirely in test
    whenever their activity clusters in time, making per-user ranking
    metrics (Precision@K, Recall@K, NDCG@K) impossible to compute for many
    users. Users with fewer than ``min_ratings_for_test`` ratings are kept
    entirely in train, since holding out a test point for them would leave
    too little signal to predict from and is exactly the cold-start
    scenario handled separately by the cold-start fallback.

    Args:
        ratings: Ratings frame with ``user_id`` and ``timestamp`` columns.
        test_frac: Fraction of each eligible user's most recent ratings to
            hold out for testing. Defaults to 0.2.
        min_ratings_for_test: Minimum number of ratings a user must have
            before any of their ratings are held out for test. Defaults to 5.

    Returns:
        ``(train_df, test_df)`` tuple of DataFrames with the same columns
        as ``ratings``.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in ratings.groupby("user_id", sort=False):
        ordered = group.sort_values("timestamp", kind="mergesort")
        n_test = int(len(ordered) * test_frac) if len(ordered) >= min_ratings_for_test else 0
        if n_test == 0:
            train_parts.append(ordered)
        else:
            train_parts.append(ordered.iloc[:-n_test])
            test_parts.append(ordered.iloc[-n_test:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0]
    return train_df, test_df


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the full pipeline: download, load, merge, split, and persist.

    Idempotent: if the processed CSVs already exist and ``force`` is
    ``False``, they are loaded from disk instead of being recomputed. This
    is what lets ``app.py`` call this on every Streamlit launch cheaply.

    Args:
        raw_dir: Directory for the raw/cached MovieLens download.
        processed_dir: Directory to write/read processed CSVs.
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
        logger.info("Loading cached processed data from %s", processed_dir)
        return {name: pd.read_csv(p) for name, p in paths.items()}

    raw_dir_path = download_movielens_100k(raw_dir)
    ratings = load_ratings(raw_dir_path)
    movies = load_movies(raw_dir_path)
    users = load_users(raw_dir_path)
    logger.info(
        "Loaded %d ratings, %d movies, %d users", len(ratings), len(movies), len(users)
    )

    train_df, test_df = chronological_train_test_split(ratings)
    logger.info(
        "Chronological per-user split: %d train ratings, %d test ratings (%.1f%% held out)",
        len(train_df), len(test_df), 100 * len(test_df) / len(ratings),
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
