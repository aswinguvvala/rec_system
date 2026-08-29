"""Shared logging configuration and filesystem path helpers.

Every other module in ``src`` imports its logger and its data paths from
here so that log formatting and directory locations stay consistent across
the pipeline, the models, and the evaluation code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
# Deliberately NOT under DATA_DIR: data/ is entirely gitignored and
# regenerated from a fresh download on every clone, but the supplementary
# movie catalog (see src/movie_discovery.py) is a small, one-time-generated,
# git-tracked artifact -- same "real artifact, not silently regenerated"
# convention as RESULTS_DIR/metrics.json.
SUPPLEMENTARY_CATALOG_DIR: Path = PROJECT_ROOT / "catalog_supplement"
SUPPLEMENTARY_MOVIES_PATH: Path = SUPPLEMENTARY_CATALOG_DIR / "tmdb_supplementary_movies.csv"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with consistent formatting.

    Safe to call repeatedly with the same ``name`` (e.g. on Streamlit
    re-runs): handlers are only attached once per logger instance.

    Args:
        name: Logger name, typically the calling module's ``__name__``.
        level: Logging level for this logger. Defaults to ``logging.INFO``.

    Returns:
        A configured ``logging.Logger`` that writes to stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Create a directory (and any missing parents) if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The same ``path``, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
