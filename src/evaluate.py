"""Evaluation metrics and the cross-model comparison harness.

Two families of metrics:

* **Rating prediction** (:func:`rmse`, :func:`mae`): how close
  ``model.predict(user, movie)`` is to the true held-out rating.
* **Ranking** (:func:`precision_at_k`, :func:`recall_at_k`,
  :func:`ndcg_at_k`): whether the movies a user actually rated highly in
  the test set appear near the top of ``model.recommend_for_user``'s
  output -- this is closer to what a recommender is actually used for
  than rating prediction is.

:func:`compare_models` fits and evaluates several models on the same
train/test split and returns a table; ``python -m src.evaluate`` runs this
for the four models this project ships (popularity baseline,
content-based, SVD, hybrid) and writes ``results/metrics.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.models import BaseRecommender
from src.utils import RESULTS_DIR, ensure_dir, get_logger

logger = get_logger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (5, 10)
# The Indian Regional Movie Dataset's ratings are ternary (-1/0/1, see
# src/data_pipeline.py), not 1-5 stars -- "relevant" means the user
# explicitly marked the movie liked (rating == 1).
DEFAULT_RELEVANCE_THRESHOLD = 1.0


class EvaluationError(Exception):
    """Raised when evaluation results can't be persisted."""


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error between true and predicted ratings.

    Args:
        y_true: Ground-truth ratings.
        y_pred: Predicted ratings, same shape as ``y_true``.

    Returns:
        RMSE as a float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error between true and predicted ratings.

    Args:
        y_true: Ground-truth ratings.
        y_pred: Predicted ratings, same shape as ``y_true``.

    Returns:
        MAE as a float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` recommendations that are relevant.

    If fewer than ``k`` items were recommended, the denominator is the
    actual number returned (not ``k``) so a short list isn't unfairly
    penalized for something the caller, not the ranking, controls.

    Args:
        recommended: Ranked movie IDs, most confident first.
        relevant: Set of movie IDs the user actually liked (ground truth).
        k: Cutoff rank.

    Returns:
        Precision@K in [0, 1].
    """
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & relevant)
    return hits / len(top_k)


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items captured in the top-``k`` recommendations.

    Args:
        recommended: Ranked movie IDs, most confident first.
        relevant: Set of movie IDs the user actually liked (ground truth).
        k: Cutoff rank.

    Returns:
        Recall@K in [0, 1]. 0.0 if ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K, with binary relevance.

    Rewards relevant items appearing earlier in the ranking: a hit at
    rank 1 contributes more than a hit at rank 10. Normalized by the
    ideal DCG (all relevant items packed at the top) so the score is
    comparable across users with different numbers of relevant items.

    Args:
        recommended: Ranked movie IDs, most confident first.
        relevant: Set of movie IDs the user actually liked (ground truth).
        k: Cutoff rank.

    Returns:
        NDCG@K in [0, 1]. 0.0 if ``relevant`` is empty.
    """
    top_k = list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, movie_id in enumerate(top_k)
        if movie_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rating_prediction(model: BaseRecommender, test_df: pd.DataFrame) -> dict[str, float]:
    """Compute RMSE and MAE for a model's rating predictions on the test set.

    Args:
        model: A fitted recommender.
        test_df: Held-out ratings with ``user_id``, ``movie_id``, ``rating``.

    Returns:
        Dict with keys ``"rmse"`` and ``"mae"``.
    """
    y_true = test_df["rating"].to_numpy(dtype=float)
    y_pred = np.array(
        [model.predict(u, m) for u, m in zip(test_df["user_id"], test_df["movie_id"])]
    )
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def evaluate_ranking(
    model: BaseRecommender,
    test_df: pd.DataFrame,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict[str, float]:
    """Compute Precision@K, Recall@K, and NDCG@K averaged over test users.

    A test rating counts as "relevant" (i.e. something the user actually
    liked) if it's at or above ``relevance_threshold``. Only users with at
    least one relevant test rating are included, since precision/recall
    are degenerate (always 0) for users with none -- including them would
    just dilute every model's score by the same constant rather than
    reveal anything about ranking quality.

    Args:
        model: A fitted recommender.
        test_df: Held-out ratings with ``user_id``, ``movie_id``, ``rating``.
        k_values: Cutoffs to compute metrics at.
        relevance_threshold: Minimum rating counted as "liked".

    Returns:
        Dict with keys like ``"precision@5"``, ``"recall@10"``,
        ``"ndcg@5"`` for every ``k`` in ``k_values``.
    """
    relevant_by_user = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    max_k = max(k_values)
    sums = {metric: {k: 0.0 for k in k_values} for metric in ("precision", "recall", "ndcg")}
    n_users = 0

    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        recs = model.recommend_for_user(user_id, n=max_k, exclude_seen=True)
        recommended_ids = [r.movie_id for r in recs]
        n_users += 1
        for k in k_values:
            sums["precision"][k] += precision_at_k(recommended_ids, relevant, k)
            sums["recall"][k] += recall_at_k(recommended_ids, relevant, k)
            sums["ndcg"][k] += ndcg_at_k(recommended_ids, relevant, k)

    if n_users == 0:
        logger.warning("No test users had a relevant (rating >= %.1f) item; ranking metrics are 0.", relevance_threshold)
        return {f"{metric}@{k}": 0.0 for metric in sums for k in k_values}

    return {f"{metric}@{k}": total / n_users for metric, per_k in sums.items() for k, total in per_k.items()}


def compare_models(
    models: dict[str, BaseRecommender],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, float]]:
    """Fit and evaluate several recommenders on the same split.

    Args:
        models: Mapping of display name -> unfitted recommender instance.
        train_df: Training ratings.
        test_df: Held-out test ratings.
        movies_df: Movie metadata.
        k_values: Ranking-metric cutoffs.

    Returns:
        Mapping of model name -> dict of all metrics (rmse, mae, and
        precision/recall/ndcg at each k).
    """
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        logger.info("Fitting %s", name)
        model.fit(train_df, movies_df)
        rating_metrics = evaluate_rating_prediction(model, test_df)
        ranking_metrics = evaluate_ranking(model, test_df, k_values)
        results[name] = {**rating_metrics, **ranking_metrics}
        logger.info("%s results: %s", name, {k: round(v, 4) for k, v in results[name].items()})
    return results


def save_metrics(results: dict[str, dict[str, float]], path: Path = RESULTS_DIR / "metrics.json") -> None:
    """Persist evaluation results as JSON.

    Args:
        results: Output of :func:`compare_models`.
        path: Destination file path.

    Raises:
        EvaluationError: If the file can't be written.
    """
    try:
        ensure_dir(path.parent)
        path.write_text(json.dumps(results, indent=2))
    except OSError as exc:
        raise EvaluationError(f"Failed to write metrics to {path}: {exc}") from exc
    logger.info("Wrote metrics to %s", path)


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.models import ContentBasedRecommender, HybridRecommender, PopularityRecommender, SVDRecommender

    data = run_pipeline()
    model_registry: dict[str, BaseRecommender] = {
        "popularity_baseline": PopularityRecommender(),
        "content_based": ContentBasedRecommender(),
        "svd": SVDRecommender(),
        "hybrid": HybridRecommender(ContentBasedRecommender(), SVDRecommender(), strategy="weighted", alpha=0.6),
    }
    all_results = compare_models(model_registry, data["train"], data["test"], data["movies"])
    save_metrics(all_results)

    table = pd.DataFrame(all_results).T.round(4)
    logger.info("\n%s", table.to_string())
