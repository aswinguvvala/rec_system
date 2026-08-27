"""Recommendation models: content-based, SVD collaborative filtering, a
weighted/switching hybrid, a popularity baseline, and a cold-start wrapper.

All models implement :class:`BaseRecommender` so they can be composed
interchangeably by :class:`HybridRecommender` and
:class:`ColdStartRecommender`, and evaluated uniformly by
``src/evaluate.py``. This module never touches the filesystem or the
network -- all I/O lives in ``src/data_pipeline.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_pipeline import GENRE_COLUMNS
from src.utils import get_logger

logger = get_logger(__name__)

# The Indian Regional Movie Dataset's ratings are a ternary preference signal
# (-1 = ambiguous/skipped, 0 = disliked, 1 = liked), not 1-5 stars -- see
# src/data_pipeline.py's module docstring. Every model below is written in
# terms of these two constants rather than a hardcoded range, so this is the
# only place that scale is defined.
RATING_MIN = -1.0
RATING_MAX = 1.0


@dataclass(frozen=True)
class Recommendation:
    """A single ranked recommendation.

    Attributes:
        movie_id: The recommended movie's IMDb ``tt`` id.
        score: Model-internal score used to rank this item. Scale varies
            by source (content/hybrid scores are similarities in roughly
            [0, 1]; SVD scores are predicted preference in [-1, 1]) --
            compare scores only within the same source.
        source: Which code path produced this item: ``"content"``,
            ``"svd"``, ``"hybrid"``, or ``"cold_start"``. Surfaced in the
            Streamlit app so a reviewer can see which model actually
            served each result.
    """

    movie_id: str
    score: float
    source: str


class BaseRecommender(ABC):
    """Common interface implemented by every recommender in this module."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Fit the model on training ratings and movie metadata.

        Args:
            train_df: Ratings with at least ``user_id``, ``movie_id``,
                ``rating`` columns.
            movies_df: Movie metadata with ``movie_id`` and genre columns.
        """

    @abstractmethod
    def predict(self, user_id: str, movie_id: str) -> float:
        """Predict a single user's rating for a single movie.

        Args:
            user_id: Raw user ID.
            movie_id: Raw movie ID (IMDb ``tt`` id).

        Returns:
            Predicted rating, clipped to ``[RATING_MIN, RATING_MAX]``.
        """

    @abstractmethod
    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[Recommendation]:
        """Return the top-``n`` recommended movies for a user.

        Args:
            user_id: Raw user ID.
            n: Number of recommendations to return.
            exclude_seen: If True, exclude movies the user rated in the
                training data.

        Returns:
            Up to ``n`` :class:`Recommendation` objects, highest score first.
        """


class ContentBasedRecommender(BaseRecommender):
    """Genre-based content filtering via cosine similarity.

    The Indian Regional Movie Dataset has no plot synopsis or free-text
    overview field, so the only substantive content signal available is
    each movie's genre one-hot vector (21 genres -- see
    ``src/data_pipeline.py``'s ``GENRE_COLUMNS``; roughly a quarter of the
    catalog has no listed genre at all and gets an all-zero vector). This
    model builds a per-user "taste profile" as the rating-weighted sum of
    the genre vectors of movies the user has rated, then ranks unseen
    movies by cosine similarity to that profile. This is deliberately
    simple (no TF-IDF over a text corpus that doesn't exist in this
    dataset) but is a fast, legitimate content-based signal that is
    completely independent of collaborative (rating-matrix) structure --
    which is exactly what makes it useful to combine with SVD in
    :class:`HybridRecommender`.
    """

    def __init__(self, genre_columns: list[str] | None = None) -> None:
        self._genre_columns = genre_columns or GENRE_COLUMNS
        self._movie_ids: np.ndarray | None = None
        self._genre_matrix: np.ndarray | None = None
        self._movie_id_to_idx: dict[str, int] = {}
        self._item_similarity: np.ndarray | None = None
        self._user_profiles: dict[str, np.ndarray] = {}
        self._seen: dict[str, set[str]] = {}

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self._movie_ids = movies_df["movie_id"].to_numpy()
        self._movie_id_to_idx = {mid: idx for idx, mid in enumerate(self._movie_ids)}
        self._genre_matrix = movies_df[self._genre_columns].to_numpy(dtype=float)
        self._item_similarity = cosine_similarity(self._genre_matrix)

        self._seen = {uid: set(g["movie_id"]) for uid, g in train_df.groupby("user_id")}

        self._user_profiles = {}
        for user_id, group in train_df.groupby("user_id"):
            mask = group["movie_id"].isin(self._movie_id_to_idx)
            idxs = [self._movie_id_to_idx[m] for m in group.loc[mask, "movie_id"]]
            if not idxs:
                continue
            weights = group.loc[mask, "rating"].to_numpy(dtype=float)
            profile = (self._genre_matrix[idxs] * weights[:, None]).sum(axis=0)
            self._user_profiles[user_id] = profile

        logger.info(
            "ContentBasedRecommender fit on %d users, %d movies",
            len(self._user_profiles), len(self._movie_ids),
        )

    def predict(self, user_id: str, movie_id: str) -> float:
        profile = self._user_profiles.get(user_id)
        idx = self._movie_id_to_idx.get(movie_id)
        if profile is None or idx is None or self._genre_matrix is None:
            return (RATING_MIN + RATING_MAX) / 2
        movie_vec = self._genre_matrix[idx]
        denom = np.linalg.norm(profile) * np.linalg.norm(movie_vec)
        similarity = float(np.dot(profile, movie_vec) / denom) if denom > 0 else 0.0
        similarity = max(0.0, min(1.0, similarity))
        return RATING_MIN + similarity * (RATING_MAX - RATING_MIN)

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[Recommendation]:
        profile = self._user_profiles.get(user_id)
        if profile is None or self._genre_matrix is None or self._movie_ids is None:
            return []
        profile_norm = np.linalg.norm(profile)
        if profile_norm == 0:
            return []
        norms = np.linalg.norm(self._genre_matrix, axis=1)
        sims = (self._genre_matrix @ profile) / (norms * profile_norm + 1e-12)

        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        ranked_idx = np.argsort(-sims)
        results: list[Recommendation] = []
        for idx in ranked_idx:
            mid = str(self._movie_ids[idx])
            if mid in seen:
                continue
            results.append(Recommendation(movie_id=mid, score=float(sims[idx]), source="content"))
            if len(results) >= n:
                break
        return results

    def similar_items(
        self, movie_id: str, n: int = 10, candidate_movie_ids: set[str] | None = None
    ) -> list[Recommendation]:
        """Return the ``n`` movies most similar to ``movie_id`` by genre.

        Bonus "more like this" lookup, independent of any user -- handy
        for a recruiter-facing demo of the raw content signal.

        Args:
            movie_id: Raw movie ID to find neighbors for.
            n: Number of similar movies to return.
            candidate_movie_ids: If given, only movies in this set are
                eligible to be returned as neighbors -- e.g. restricting to
                the query movie's primary language (see
                :func:`movie_ids_matching_languages`), since genre alone
                carries no language information and can otherwise tie a
                same-genre movie in a completely different language with
                one that's actually a close match. ``None`` considers the
                full catalog.

        Returns:
            Up to ``n`` :class:`Recommendation` objects, most similar first.
        """
        idx = self._movie_id_to_idx.get(movie_id)
        if idx is None or self._item_similarity is None or self._movie_ids is None:
            return []
        sims = self._item_similarity[idx]
        ranked_idx = np.argsort(-sims)
        results: list[Recommendation] = []
        for i in ranked_idx:
            if i == idx:
                continue
            mid = str(self._movie_ids[i])
            if candidate_movie_ids is not None and mid not in candidate_movie_ids:
                continue
            results.append(Recommendation(movie_id=mid, score=float(sims[i]), source="content"))
            if len(results) >= n:
                break
        return results


class SVDRecommender(BaseRecommender):
    """Matrix factorization via manual Funk-SVD (SGD).

    Learns latent user/item factor vectors plus per-user and per-item bias
    terms by stochastic gradient descent on the observed ratings,
    minimizing regularized squared error -- the same model family as the
    classic Netflix Prize "Funk SVD". Implemented directly on numpy rather
    than pulling in ``surprise`` or ``implicit`` (see ``claude.md`` for the
    library tradeoff: avoids `surprise`'s brittle Windows/CI builds and
    `implicit`'s implicit-feedback focus, at the cost of writing the SGD
    loop ourselves).

    Unseen users/items at prediction time fall back to bias-only or
    global-mean estimates rather than raising -- true cold-start handling
    is layered on top by :class:`ColdStartRecommender`.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

        self._user_id_to_idx: dict[str, int] = {}
        self._item_id_to_idx: dict[str, int] = {}
        self._item_ids: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._b_u: np.ndarray | None = None
        self._b_i: np.ndarray | None = None
        self._global_mean: float = (RATING_MIN + RATING_MAX) / 2
        self._seen: dict[str, set[str]] = {}

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame | None = None) -> None:
        user_ids = train_df["user_id"].unique()
        item_ids = train_df["movie_id"].unique()
        self._user_id_to_idx = {u: idx for idx, u in enumerate(user_ids)}
        self._item_id_to_idx = {m: idx for idx, m in enumerate(item_ids)}
        self._item_ids = item_ids

        n_users, n_items = len(user_ids), len(item_ids)
        rng = np.random.default_rng(self.random_state)
        self._P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self._Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        self._b_u = np.zeros(n_users)
        self._b_i = np.zeros(n_items)
        self._global_mean = float(train_df["rating"].mean())

        u_idx = train_df["user_id"].map(self._user_id_to_idx).to_numpy()
        i_idx = train_df["movie_id"].map(self._item_id_to_idx).to_numpy()
        ratings = train_df["rating"].to_numpy(dtype=float)

        lr, reg = self.learning_rate, self.regularization
        n_samples = len(ratings)
        order = np.arange(n_samples)

        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            sq_error_sum = 0.0
            for row in order:
                u, i, r = u_idx[row], i_idx[row], ratings[row]
                pred = self._global_mean + self._b_u[u] + self._b_i[i] + self._P[u] @ self._Q[i]
                err = r - pred
                sq_error_sum += err * err

                p_u = self._P[u].copy()
                q_i = self._Q[i].copy()
                self._b_u[u] += lr * (err - reg * self._b_u[u])
                self._b_i[i] += lr * (err - reg * self._b_i[i])
                self._P[u] += lr * (err * q_i - reg * p_u)
                self._Q[i] += lr * (err * p_u - reg * q_i)

            rmse = (sq_error_sum / n_samples) ** 0.5
            logger.info("SVD epoch %d/%d - train RMSE: %.4f", epoch + 1, self.n_epochs, rmse)

        self._seen = {uid: set(g["movie_id"]) for uid, g in train_df.groupby("user_id")}
        logger.info(
            "SVDRecommender fit: %d users, %d items, %d factors", n_users, n_items, self.n_factors
        )

    def predict(self, user_id: str, movie_id: str) -> float:
        u = self._user_id_to_idx.get(user_id)
        i = self._item_id_to_idx.get(movie_id)
        if self._b_u is None or self._b_i is None or self._P is None or self._Q is None:
            return self._global_mean
        if u is None and i is None:
            pred = self._global_mean
        elif u is None:
            pred = self._global_mean + self._b_i[i]
        elif i is None:
            pred = self._global_mean + self._b_u[u]
        else:
            pred = self._global_mean + self._b_u[u] + self._b_i[i] + float(self._P[u] @ self._Q[i])
        return float(np.clip(pred, RATING_MIN, RATING_MAX))

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[Recommendation]:
        u = self._user_id_to_idx.get(user_id)
        if u is None or self._item_ids is None or self._b_i is None or self._Q is None or self._b_u is None:
            return []
        scores = self._global_mean + self._b_u[u] + self._b_i + self._Q @ self._P[u]
        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        ranked_idx = np.argsort(-scores)
        results: list[Recommendation] = []
        for idx in ranked_idx:
            mid = str(self._item_ids[idx])
            if mid in seen:
                continue
            score = float(np.clip(scores[idx], RATING_MIN, RATING_MAX))
            results.append(Recommendation(movie_id=mid, score=score, source="svd"))
            if len(results) >= n:
                break
        return results

    def similar_items(
        self, movie_id: str, n: int = 10, candidate_movie_ids: set[str] | None = None
    ) -> list[Recommendation]:
        """Return the ``n`` movies most similar to ``movie_id`` by learned collaborative factors.

        The collaborative counterpart to
        :meth:`ContentBasedRecommender.similar_items`: two movies end up with
        similar item-factor vectors when the users who rated one tended to
        rate the other similarly, independent of genre -- a "people who liked
        this also liked that" signal genre similarity can't provide on its
        own. Used to build a real similar-to-your-picks recommendation for
        cold-start onboarding (see :func:`recommend_similar_to_picks`) instead
        of only ever blending overall popularity with one aggregated genre
        profile.

        Only available for movies that appeared in the training data at least
        once -- roughly half this dataset's catalog never received a training
        rating at all (see ``claude.md``), so this returns ``[]`` for those,
        the same "no signal, no crash" convention :meth:`predict` already uses
        for unseen items, rather than fabricating a collaborative signal that
        doesn't exist.

        Args:
            movie_id: Raw movie ID to find neighbors for.
            n: Number of similar movies to return.
            candidate_movie_ids: If given, only movies in this set are
                eligible to be returned as neighbors -- e.g. restricting to
                the query movie's primary language (see
                :func:`movie_ids_matching_languages`), since the learned
                rating-pattern similarity this method surfaces has no notion
                of language either. ``None`` considers the full catalog.

        Returns:
            Up to ``n`` :class:`Recommendation` objects (``source="svd"``),
            most similar first.
        """
        idx = self._item_id_to_idx.get(movie_id)
        if idx is None or self._Q is None or self._item_ids is None:
            return []
        item_vec = self._Q[idx]
        item_norm = np.linalg.norm(item_vec)
        if item_norm < 1e-12:
            return []
        norms = np.linalg.norm(self._Q, axis=1)
        sims = (self._Q @ item_vec) / (norms * item_norm + 1e-12)
        ranked_idx = np.argsort(-sims)
        results: list[Recommendation] = []
        for i in ranked_idx:
            if i == idx:
                continue
            mid = str(self._item_ids[i])
            if candidate_movie_ids is not None and mid not in candidate_movie_ids:
                continue
            results.append(Recommendation(movie_id=mid, score=float(sims[i]), source="svd"))
            if len(results) >= n:
                break
        return results


class PopularityRecommender(BaseRecommender):
    """Popularity baseline -- also used as the cold-start fallback.

    Ranks movies by a Bayesian-averaged "weighted rating" (the classic
    IMDB formula: ``WR = v/(v+m)*R + m/(v+m)*C``) rather than the raw mean
    rating, so a movie with 2 enthusiastic ratings doesn't outrank one with
    200 ratings that are solidly positive on average. Raw means are
    extremely noisy for low-count items, which is exactly the regime
    cold-start fallbacks operate in.
    """

    def __init__(self, min_votes_quantile: float = 0.6) -> None:
        self.min_votes_quantile = min_votes_quantile
        self._global_mean: float = (RATING_MIN + RATING_MAX) / 2
        self._movie_stats: pd.DataFrame | None = None
        self._genre_columns: list[str] = []
        self._genre_matrix: np.ndarray | None = None
        self._movie_id_to_idx: dict[str, int] = {}
        self._seen: dict[str, set[str]] = {}

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self._genre_columns = [c for c in GENRE_COLUMNS if c in movies_df.columns]
        stats = (
            train_df.groupby("movie_id")["rating"]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_ratings", "mean": "avg_rating"})
        )
        self._global_mean = float(train_df["rating"].mean())
        m = float(stats["n_ratings"].quantile(self.min_votes_quantile))
        stats["weighted_rating"] = (
            (stats["n_ratings"] / (stats["n_ratings"] + m)) * stats["avg_rating"]
            + (m / (stats["n_ratings"] + m)) * self._global_mean
        )

        movie_stats = movies_df[["movie_id"] + self._genre_columns].merge(
            stats[["n_ratings", "avg_rating", "weighted_rating"]], on="movie_id", how="left"
        )
        movie_stats["n_ratings"] = movie_stats["n_ratings"].fillna(0)
        movie_stats["weighted_rating"] = movie_stats["weighted_rating"].fillna(self._global_mean)
        self._movie_stats = movie_stats
        self._movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_stats["movie_id"])}
        self._genre_matrix = movie_stats[self._genre_columns].to_numpy(dtype=float)
        self._seen = {uid: set(g["movie_id"]) for uid, g in train_df.groupby("user_id")}
        logger.info("PopularityRecommender fit on %d movies (min-votes prior m=%.1f)", len(movie_stats), m)

    def predict(self, user_id: str, movie_id: str) -> float:
        if self._movie_stats is None:
            return self._global_mean
        idx = self._movie_id_to_idx.get(movie_id)
        if idx is None:
            return self._global_mean
        return float(self._movie_stats.iloc[idx]["weighted_rating"])

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[Recommendation]:
        return self.recommend_for_genre_profile(
            genre_weights=None, user_id=user_id, n=n, exclude_seen=exclude_seen
        )

    def recommend_for_genre_profile(
        self,
        genre_weights: np.ndarray | None,
        user_id: str | None = None,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movie_ids: set[str] | None = None,
    ) -> list[Recommendation]:
        """Rank movies by popularity, optionally blended with genre affinity.

        Args:
            genre_weights: A genre-preference vector (same column order as
                :data:`GENRE_COLUMNS`, e.g. built from a cold-start user's
                handful of ratings) to blend with popularity. ``None``
                ranks by pure popularity.
            user_id: If given and ``exclude_seen``, excludes movies this
                user has already rated.
            n: Number of recommendations to return.
            exclude_seen: Whether to exclude the user's seen movies.
            candidate_movie_ids: If given, restricts ranking to just these
                movie ids before scoring -- e.g. narrowing to movies sharing
                a language with a cold-start user's live picks (see
                :func:`movie_ids_matching_languages` and the Streamlit app's
                onboarding flow). Genre affinity alone doesn't carry language
                information (this dataset's 21 genres are language-agnostic),
                so a broad genre match can otherwise let a popular film in a
                different language outrank a less-popular one that actually
                matches what the user picked. ``None`` ranks the full
                catalog, unchanged from before this parameter existed.

        Returns:
            Up to ``n`` :class:`Recommendation` objects with
            ``source="cold_start"``, highest score first.
        """
        if self._movie_stats is None or self._genre_matrix is None:
            return []

        if candidate_movie_ids is not None:
            mask = self._movie_stats["movie_id"].isin(candidate_movie_ids).to_numpy()
            movie_stats = self._movie_stats[mask]
            genre_matrix = self._genre_matrix[mask]
        else:
            movie_stats = self._movie_stats
            genre_matrix = self._genre_matrix
        if movie_stats.empty:
            return []

        wr = movie_stats["weighted_rating"].to_numpy(dtype=float)
        wr_range = wr.max() - wr.min()
        wr_norm = (wr - wr.min()) / wr_range if wr_range > 1e-12 else np.zeros_like(wr)

        if genre_weights is not None and np.linalg.norm(genre_weights) > 0:
            genre_norms = np.linalg.norm(genre_matrix, axis=1)
            weight_norm = np.linalg.norm(genre_weights)
            affinity = (genre_matrix @ genre_weights) / (genre_norms * weight_norm + 1e-12)
            affinity_range = affinity.max() - affinity.min()
            affinity_norm = (
                (affinity - affinity.min()) / affinity_range if affinity_range > 1e-12 else np.zeros_like(affinity)
            )
            scores = 0.5 * wr_norm + 0.5 * affinity_norm
        else:
            scores = wr_norm

        seen = self._seen.get(user_id, set()) if (exclude_seen and user_id is not None) else set()
        movie_ids = movie_stats["movie_id"].to_numpy()
        ranked_idx = np.argsort(-scores)
        results: list[Recommendation] = []
        for idx in ranked_idx:
            mid = str(movie_ids[idx])
            if mid in seen:
                continue
            results.append(Recommendation(movie_id=mid, score=float(scores[idx]), source="cold_start"))
            if len(results) >= n:
                break
        return results


def genre_profile_from_movie_ids(
    movie_ids: list[str], movies_df: pd.DataFrame, genre_columns: list[str] | None = None
) -> np.ndarray | None:
    """Build a genre-preference vector from a set of movies a user says they like.

    This is the live-onboarding counterpart to
    :meth:`ColdStartRecommender._genre_profile_for_user`: that one builds a
    genre profile from a user's real *training ratings*; this one builds the
    same kind of vector from movies a brand-new user just picked in the UI,
    with no rating history at all -- e.g. the Streamlit app's "pick a few
    movies you like" cold-start onboarding flow. Both feed the same consumer,
    :meth:`PopularityRecommender.recommend_for_genre_profile`. Unlike the
    training-data version, a live pick carries no rating to weight by (there
    is no rating yet, just "I like this"), so each pick contributes an equal,
    unweighted share to the profile.

    Args:
        movie_ids: IDs of movies the user picked as "movies I like".
        movies_df: Movie metadata with one column per genre in ``genre_columns``.
        genre_columns: Genre column names, in the order the returned vector
            uses. Defaults to whichever of :data:`GENRE_COLUMNS` are actually
            present in ``movies_df`` (mirrors ``PopularityRecommender.fit``'s
            own defaulting, so this stays robust to a ``movies_df`` that
            doesn't carry every genre column -- e.g. a smaller test fixture,
            or a future dataset swap with a different genre set).

    Returns:
        A genre-preference vector in ``genre_columns`` order, or ``None`` if
        none of ``movie_ids`` were found in ``movies_df``.
    """
    columns = genre_columns or [c for c in GENRE_COLUMNS if c in movies_df.columns]
    matches = movies_df.loc[movies_df["movie_id"].isin(set(movie_ids)), columns]
    if matches.empty:
        return None
    return matches.to_numpy(dtype=float).sum(axis=0)


def _primary_language(cell: object) -> str | None:
    """Extract a movie's primary (first-listed) language from its ``languages`` cell.

    The ``languages`` column lists every language a title was released or dubbed
    in, in the order the source data gave them, with no explicit "this one is
    the original" flag -- but spot-checking well-known films against this
    dataset shows the first entry reliably *is* the original/primary industry:
    "Dil Se.." (a Hindi film dubbed into Telugu, Tamil, Urdu, and Assamese for
    wider release) lists ``"Hindi, Urdu, Assamese, Tamil, Telugu"``; "Baahubali:
    The Beginning" and "Eega" (both Telugu productions) list Telugu first even
    though they were dubbed into many more languages than Dil Se.. was.

    This matters because :func:`movie_ids_matching_languages` used to match on
    *any* shared language, and a handful of massively-popular, widely-dubbed
    Hindi hits like Dil Se.. kept slipping back into "Telugu-matched" results
    that way -- technically true (Telugu is one of its five language tags) but
    not what a user picking Telugu films actually wants. Matching on primary
    language only fixes that while barely shrinking the real candidate pool
    (in this dataset, Telugu-primary is 297 movies vs. 338 Telugu-anywhere).

    Args:
        cell: A ``languages`` column value, e.g. ``"Telugu, Tamil"``.

    Returns:
        The first language, or ``None`` if ``cell`` is empty/not a string.
    """
    if not isinstance(cell, str) or not cell:
        return None
    first = cell.split(",")[0].strip()
    return first or None


def movie_ids_matching_languages(movie_ids: list[str], movies_df: pd.DataFrame) -> set[str] | None:
    """Find every catalog movie whose primary language matches the given movies'.

    Exists because genre affinity alone is language-blind: this dataset's 21
    genres (Action, Comedy, Drama, ...) say nothing about which of its 18
    Indian regional languages a film is in, so a broad genre match (e.g.
    "Comedy") can happily surface a popular film in a completely different
    language than what the user actually picked -- popularity does the rest,
    since a handful of blockbuster hits in the catalog's dominant language
    outrank almost everything else on raw rating counts alone. Meant to be
    used as :meth:`PopularityRecommender.recommend_for_genre_profile`'s
    ``candidate_movie_ids``, narrowing the ranked pool to same-primary-language
    films before popularity and genre affinity ever come into it.

    Matches on *primary* language (see :func:`_primary_language`), not any
    language a title happens to be tagged with -- matching on any shared tag
    let widely-dubbed Hindi blockbusters slip back into results for someone
    who picked only Telugu films, since being dubbed into Telugu was enough
    to "share a language" even though the film isn't Telugu cinema.

    Args:
        movie_ids: IDs of movies to read the "wanted" language(s) from --
            typically a cold-start user's live picks.
        movies_df: Movie metadata with a comma-joined ``languages`` column
            (see ``src/data_pipeline.py``'s ``load_movies``).

    Returns:
        Set of matching movie ids (this always includes the input
        ``movie_ids`` themselves, since a film trivially matches its own
        primary language), or ``None`` if none of ``movie_ids`` had any
        parsed language info at all -- callers should treat ``None`` as
        "can't narrow by language" and fall back to the unrestricted catalog
        rather than recommending nothing.
    """
    picked = movies_df.loc[movies_df["movie_id"].isin(set(movie_ids)), "languages"]
    wanted_languages = {lang for cell in picked if (lang := _primary_language(cell))}
    if not wanted_languages:
        return None

    primary = movies_df["languages"].apply(_primary_language)
    return set(movies_df.loc[primary.isin(wanted_languages), "movie_id"])


class HybridRecommender(BaseRecommender):
    """Combines content-based and SVD scores, weighted or switching.

    Args:
        content_model: A :class:`ContentBasedRecommender`; refit inside
            :meth:`fit`.
        svd_model: A :class:`SVDRecommender`; refit inside :meth:`fit`.
        strategy: ``"weighted"`` blends both models' normalized scores
            every time using ``alpha``. ``"switching"`` uses SVD once a
            user has at least ``min_ratings_for_svd`` training ratings
            (collaborative signal becomes reliable) and uses content-based
            otherwise. This is a *separate, smaller* threshold from
            :class:`ColdStartRecommender`'s -- that wrapper handles users
            with very little or no history by falling back to popularity;
            this handles the middle ground where a user has some history
            but not enough for SVD's latent factors to have converged for
            them specifically.
        alpha: Weight on the SVD score in ``"weighted"`` mode, in [0, 1].
            The content-based score gets ``1 - alpha``. Default of 0.6
            favors collaborative signal, since it typically outperforms
            content-only on rating prediction once a user has enough history.
        min_ratings_for_svd: Threshold used only in ``"switching"`` mode.
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        svd_model: SVDRecommender,
        strategy: Literal["weighted", "switching"] = "weighted",
        alpha: float = 0.6,
        min_ratings_for_svd: int = 5,
    ) -> None:
        self.content_model = content_model
        self.svd_model = svd_model
        self.strategy = strategy
        self.alpha = alpha
        self.min_ratings_for_svd = min_ratings_for_svd
        self._user_rating_counts: dict[str, int] = {}
        self._movie_ids: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self.content_model.fit(train_df, movies_df)
        self.svd_model.fit(train_df, movies_df)
        self._user_rating_counts = train_df.groupby("user_id").size().to_dict()
        self._movie_ids = movies_df["movie_id"].to_numpy()
        logger.info("HybridRecommender fit (strategy=%s, alpha=%.2f)", self.strategy, self.alpha)

    def predict(self, user_id: str, movie_id: str) -> float:
        if self.strategy == "switching":
            if self._user_rating_counts.get(user_id, 0) < self.min_ratings_for_svd:
                return self.content_model.predict(user_id, movie_id)
            return self.svd_model.predict(user_id, movie_id)

        content_pred = self.content_model.predict(user_id, movie_id)
        svd_pred = self.svd_model.predict(user_id, movie_id)
        blended = self.alpha * svd_pred + (1 - self.alpha) * content_pred
        return float(np.clip(blended, RATING_MIN, RATING_MAX))

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[Recommendation]:
        if self.strategy == "switching":
            if self._user_rating_counts.get(user_id, 0) < self.min_ratings_for_svd:
                base = self.content_model.recommend_for_user(user_id, n, exclude_seen)
            else:
                base = self.svd_model.recommend_for_user(user_id, n, exclude_seen)
            return [Recommendation(r.movie_id, r.score, "hybrid") for r in base]

        if self._movie_ids is None:
            return []
        n_candidates = len(self._movie_ids)
        content_scores = {
            r.movie_id: r.score
            for r in self.content_model.recommend_for_user(user_id, n=n_candidates, exclude_seen=exclude_seen)
        }
        svd_scores = {
            r.movie_id: r.score
            for r in self.svd_model.recommend_for_user(user_id, n=n_candidates, exclude_seen=exclude_seen)
        }
        candidates = set(content_scores) | set(svd_scores)
        if not candidates:
            return []

        content_norm = _min_max_normalize(content_scores)
        svd_norm = _min_max_normalize(svd_scores)
        blended = {
            mid: self.alpha * svd_norm.get(mid, 0.0) + (1 - self.alpha) * content_norm.get(mid, 0.0)
            for mid in candidates
        }
        ranked = sorted(blended.items(), key=lambda kv: -kv[1])[:n]
        return [Recommendation(movie_id=mid, score=score, source="hybrid") for mid, score in ranked]


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score dict to [0, 1].

    Used to put content similarity (~[0, 1]) and SVD predicted preference
    (~[-1, 1]) on a comparable scale before blending in
    :meth:`HybridRecommender.recommend_for_user`.
    """
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def recommend_similar_to_picks(
    picked_movie_ids: list[str],
    content_model: ContentBasedRecommender,
    svd_model: SVDRecommender,
    popularity_model: PopularityRecommender,
    movies_df: pd.DataFrame,
    n: int = 10,
    neighbors_per_pick: int = 50,
    alpha: float = 0.5,
) -> list[Recommendation]:
    """Recommend movies genuinely similar to a set of picks -- content AND collaborative.

    This exists because the simpler cold-start path
    (:meth:`PopularityRecommender.recommend_for_genre_profile`) blends overall
    popularity with *one* genre-preference vector aggregated across every
    pick -- it never actually asks "what's similar to this specific movie the
    user picked", and it has no collaborative signal ("people who rated
    similar movies the way others did also liked these") at all. This
    function answers both, by looking up each pick's real nearest neighbors
    from two independent recommenders already in this module:

    * :meth:`ContentBasedRecommender.similar_items` -- genre overlap with
      this specific pick.
    * :meth:`SVDRecommender.similar_items` -- learned rating-pattern
      similarity to this specific pick. Only available for picks with at
      least one training rating (roughly half this dataset's catalog has
      none at all -- see ``claude.md`` -- so a niche/unrated pick
      legitimately contributes content signal only, rather than a fabricated
      collaborative one).

    Both neighbor searches are restricted to the picks' primary language (see
    :func:`movie_ids_matching_languages`) *before* ranking, not filtered
    afterward -- neither genre nor the learned collaborative factors carry
    any language information on their own, so an unrestricted search can
    easily tie or outrank a real same-language match with an unrelated film
    in a different language entirely (confirmed live: two Telugu picks
    surfacing Bengali/Hindi/Kannada/English results with no same-language
    result even in the top spot). A candidate's content/collaborative score
    is its *best* (max) similarity to any single pick, not a score against
    one blended profile -- scoring each pick individually is what actually
    answers "is this similar to something they picked"; averaging into one
    profile first can wash out a strong match to just one distinctive pick
    among several. The two signals are min-max normalized and blended by
    ``alpha``, the same normalize-then-blend approach
    :class:`HybridRecommender` already uses for content vs. SVD at the user
    level -- this is that same idea applied at the item level. Movies neither
    signal ever surfaces (rare, but possible for very niche picks) are
    backfilled with language-restricted popularity so the result doesn't fall
    short of ``n`` just because the neighbor search came up thin; if even
    that restricted backfill can't reach ``n`` (a pick in a language with only
    a handful of catalog movies), one final *unrestricted* popularity pass
    fills any remaining slots -- language narrows the search, but, per
    :func:`movie_ids_matching_languages`'s own contract, never causes fewer
    than ``n`` results to come back.

    Args:
        picked_movie_ids: IDs of movies the user picked as "movies I like".
        content_model: A fitted :class:`ContentBasedRecommender`.
        svd_model: A fitted :class:`SVDRecommender`.
        popularity_model: A fitted :class:`PopularityRecommender`, used only
            to backfill if the two similarity signals together don't surface
            enough candidates.
        movies_df: Movie metadata, for the popularity backfill's genre/language
            profile (see :func:`genre_profile_from_movie_ids` and
            :func:`movie_ids_matching_languages`).
        n: Number of recommendations to return.
        neighbors_per_pick: How many nearest neighbors to fetch per pick per
            signal, before aggregating and filtering the picks back out.
        alpha: Weight on the collaborative score in the blend, in [0, 1].
            The content score gets ``1 - alpha``.

    Returns:
        Up to ``n`` :class:`Recommendation` objects, ranked by the blended
        content/collaborative score (``source="hybrid"``) with any popularity
        backfill (``source="cold_start"``) ranked after. None of them are the
        user's own picks.
    """
    picked = set(picked_movie_ids)
    # Computed once, up front, and reused for both the primary neighbor search
    # (below) and the backfill (further down) -- None means "no parsed
    # language info at all for any pick", the same "can't narrow, don't
    # filter" contract movie_ids_matching_languages documents.
    language_candidates = movie_ids_matching_languages(picked_movie_ids, movies_df)

    content_scores: dict[str, float] = {}
    collab_scores: dict[str, float] = {}

    for pick in picked_movie_ids:
        for rec in content_model.similar_items(pick, n=neighbors_per_pick, candidate_movie_ids=language_candidates):
            if rec.movie_id in picked:
                continue
            content_scores[rec.movie_id] = max(content_scores.get(rec.movie_id, -1.0), rec.score)
        for rec in svd_model.similar_items(pick, n=neighbors_per_pick, candidate_movie_ids=language_candidates):
            if rec.movie_id in picked:
                continue
            collab_scores[rec.movie_id] = max(collab_scores.get(rec.movie_id, -1.0), rec.score)

    candidates = set(content_scores) | set(collab_scores)
    results: list[Recommendation] = []
    if candidates:
        content_norm = _min_max_normalize(content_scores)
        collab_norm = _min_max_normalize(collab_scores)
        blended = {
            mid: alpha * collab_norm.get(mid, 0.0) + (1 - alpha) * content_norm.get(mid, 0.0)
            for mid in candidates
        }
        ranked = sorted(blended.items(), key=lambda kv: -kv[1])
        results = [Recommendation(movie_id=mid, score=score, source="hybrid") for mid, score in ranked]

    genre_profile = genre_profile_from_movie_ids(picked_movie_ids, movies_df)

    if len(results) < n:
        already = picked | {r.movie_id for r in results}
        backfill = popularity_model.recommend_for_genre_profile(
            genre_weights=genre_profile,
            n=(n - len(results)) + len(already),
            exclude_seen=False,
            candidate_movie_ids=language_candidates,
        )
        for rec in backfill:
            if rec.movie_id in already:
                continue
            results.append(rec)
            already.add(rec.movie_id)
            if len(results) >= n:
                break

    if len(results) < n:
        # Even the language-restricted backfill came up short -- true only for a
        # pick in a language with a handful of catalog movies. Rather than
        # under-deliver, one final unrestricted popularity pass fills the rest:
        # language narrows the search, it never reduces how many results come
        # back.
        already = picked | {r.movie_id for r in results}
        more = popularity_model.recommend_for_genre_profile(
            genre_weights=genre_profile, n=(n - len(results)) + len(already), exclude_seen=False
        )
        for rec in more:
            if rec.movie_id in already:
                continue
            results.append(rec)
            already.add(rec.movie_id)
            if len(results) >= n:
                break

    return results[:n]


class ColdStartRecommender(BaseRecommender):
    """Wraps any recommender with cold-start fallback logic.

    This is the single place cold-start branching happens -- the wrapped
    model (content / SVD / hybrid) never needs its own if/else for "does
    this user or item have enough data". Two independent fallbacks:

    * **User cold-start**: users with fewer than ``min_user_ratings``
      training ratings get genre-matched popularity recommendations
      instead of the base model's output, built from whatever few ratings
      they do have (pure global popularity if they have none at all).
    * **Item cold-start**: movies with zero training ratings always get a
      global-popularity rating estimate from :meth:`predict`, since the
      base model has no signal to have learned anything about them.
    """

    def __init__(
        self,
        base_model: BaseRecommender,
        popularity_model: PopularityRecommender,
        min_user_ratings: int = 5,
    ) -> None:
        self.base_model = base_model
        self.popularity_model = popularity_model
        self.min_user_ratings = min_user_ratings
        self._user_rating_counts: dict[str, int] = {}
        self._item_rating_counts: dict[str, int] = {}
        self._genre_columns: list[str] = []
        self._genre_matrix: np.ndarray | None = None
        self._movie_id_to_idx: dict[str, int] = {}
        self._user_ratings: dict[str, pd.DataFrame] = {}

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self.base_model.fit(train_df, movies_df)
        self.popularity_model.fit(train_df, movies_df)
        self._user_rating_counts = train_df.groupby("user_id").size().to_dict()
        self._item_rating_counts = train_df.groupby("movie_id").size().to_dict()
        self._genre_columns = [c for c in GENRE_COLUMNS if c in movies_df.columns]
        self._genre_matrix = movies_df[self._genre_columns].to_numpy(dtype=float)
        self._movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df["movie_id"])}
        self._user_ratings = dict(tuple(train_df.groupby("user_id")))
        logger.info("ColdStartRecommender fit (min_user_ratings=%d)", self.min_user_ratings)

    def _is_cold_user(self, user_id: str) -> bool:
        return self._user_rating_counts.get(user_id, 0) < self.min_user_ratings

    def _is_cold_item(self, movie_id: str) -> bool:
        return self._item_rating_counts.get(movie_id, 0) == 0

    def _genre_profile_for_user(self, user_id: str) -> np.ndarray | None:
        group = self._user_ratings.get(user_id)
        if group is None or len(group) == 0 or self._genre_matrix is None:
            return None
        mask = group["movie_id"].isin(self._movie_id_to_idx)
        idxs = [self._movie_id_to_idx[m] for m in group.loc[mask, "movie_id"]]
        if not idxs:
            return None
        weights = group.loc[mask, "rating"].to_numpy(dtype=float)
        return (self._genre_matrix[idxs] * weights[:, None]).sum(axis=0)

    def predict(self, user_id: str, movie_id: str) -> float:
        if self._is_cold_item(movie_id) or self._is_cold_user(user_id):
            return self.popularity_model.predict(user_id, movie_id)
        return self.base_model.predict(user_id, movie_id)

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[Recommendation]:
        if self._is_cold_user(user_id):
            genre_profile = self._genre_profile_for_user(user_id)
            return self.popularity_model.recommend_for_genre_profile(
                genre_weights=genre_profile, user_id=user_id, n=n, exclude_seen=exclude_seen
            )
        return self.base_model.recommend_for_user(user_id, n=n, exclude_seen=exclude_seen)
