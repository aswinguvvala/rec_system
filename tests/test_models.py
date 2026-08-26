"""Tests for src/models.py: each recommender and the cold-start wrapper.

Fixtures use a tiny 3-genre catalog (not the real 21-genre Indian Regional
Movie Dataset taxonomy) so similarity/weighted-rating values can be
hand-computed and asserted exactly, rather than just checking "it doesn't
crash". Ratings are ternary (-1/0/1, see src/data_pipeline.py's module
docstring for why), not 1-5 stars. IDs are strings throughout, matching the
real dataset's IMDb ``tt`` ids and free-text user handles.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.models import (
    RATING_MAX,
    RATING_MIN,
    ColdStartRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    PopularityRecommender,
    SVDRecommender,
)

GENRES = ["Action", "Comedy", "Drama"]


@pytest.fixture
def movies_df() -> pd.DataFrame:
    # 1: Action, 2: Action+Comedy, 3: Comedy, 4: Drama, 5: Action+Drama
    return pd.DataFrame(
        {
            "movie_id": ["1", "2", "3", "4", "5"],
            "title": ["Action Only", "Action Comedy", "Comedy Only", "Drama Only", "Action Drama"],
            "Action": [1, 1, 0, 0, 1],
            "Comedy": [0, 1, 1, 0, 0],
            "Drama": [0, 0, 0, 1, 1],
        }
    )


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["1", "1", "2", "2", "2", "2", "2"],
            "movie_id": ["1", "2", "1", "2", "3", "4", "5"],
            "rating": [1, 1, 1, 0, 1, -1, 1],
        }
    )


class TestContentBasedRecommender:
    def test_recommend_ranks_unseen_movies_by_genre_similarity(self, movies_df, train_df):
        model = ContentBasedRecommender(genre_columns=GENRES)
        model.fit(train_df, movies_df)

        # user 1 liked movie 1 (Action, +1) and movie 2 (Action+Comedy, +1) -> both seen.
        # profile = 1*[1,0,0] + 1*[1,1,0] = [2, 1, 0]
        recs = model.recommend_for_user("1", n=3)
        recommended_ids = [r.movie_id for r in recs]

        # unseen candidates: 3 (Comedy [0,1,0]), 4 (Drama [0,0,1]), 5 (Action+Drama [1,0,1])
        # sim(profile, movie3) = 1 / (||[2,1,0]|| * 1) = 1/sqrt(5) = 0.4472
        # sim(profile, movie4) = 0
        # sim(profile, movie5) = 2 / (sqrt(5) * sqrt(2)) = 2/sqrt(10) = 0.6325
        # expected order: movie5 > movie3 > movie4
        assert recommended_ids == ["5", "3", "4"]

        profile_norm = math.sqrt(2**2 + 1**2)
        expected_sim_5 = 2 / (profile_norm * math.sqrt(2))
        assert recs[0].score == pytest.approx(expected_sim_5, rel=1e-4)
        assert recs[0].source == "content"

    def test_predict_scales_similarity_into_rating_range(self, movies_df, train_df):
        model = ContentBasedRecommender(genre_columns=GENRES)
        model.fit(train_df, movies_df)

        prediction = model.predict("1", "5")

        profile_norm = math.sqrt(2**2 + 1**2)
        expected_sim = 2 / (profile_norm * math.sqrt(2))
        expected_rating = RATING_MIN + expected_sim * (RATING_MAX - RATING_MIN)
        assert prediction == pytest.approx(expected_rating, rel=1e-4)

    def test_unknown_user_or_movie_returns_midpoint(self, movies_df, train_df):
        model = ContentBasedRecommender(genre_columns=GENRES)
        model.fit(train_df, movies_df)

        assert model.predict("unknown_user", "1") == pytest.approx((RATING_MIN + RATING_MAX) / 2)
        assert model.predict("1", "unknown_movie") == pytest.approx((RATING_MIN + RATING_MAX) / 2)

    def test_exclude_seen_false_includes_rated_movies(self, movies_df, train_df):
        model = ContentBasedRecommender(genre_columns=GENRES)
        model.fit(train_df, movies_df)

        recs = model.recommend_for_user("1", n=10, exclude_seen=False)
        assert {r.movie_id for r in recs} == {"1", "2", "3", "4", "5"}


class TestSVDRecommender:
    def test_predict_is_within_rating_bounds(self, train_df):
        model = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        model.fit(train_df)

        pairs = [("1", "1"), ("2", "5"), ("1", "unknown_movie"), ("unknown_user", "1"), ("unknown_user", "unknown_movie")]
        for user_id, movie_id in pairs:
            pred = model.predict(user_id, movie_id)
            assert RATING_MIN <= pred <= RATING_MAX

    def test_unknown_user_and_item_returns_global_mean(self, train_df):
        model = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        model.fit(train_df)

        assert model.predict("unknown_user", "unknown_movie") == pytest.approx(train_df["rating"].mean(), abs=1e-9)

    def test_recommend_excludes_seen_movies(self, movies_df, train_df):
        model = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        model.fit(train_df)

        # user 1 has only rated movies {1, 2} out of the 5-item universe in train_df,
        # so unseen candidates {3, 4, 5} must come back non-empty.
        recs = model.recommend_for_user("1", n=10)
        recommended_ids = {r.movie_id for r in recs}
        seen = set(train_df.loc[train_df["user_id"] == "1", "movie_id"])
        assert recommended_ids  # non-empty: proves the exclusion logic actually ran, not vacuously true
        assert recommended_ids.isdisjoint(seen)
        assert all(r.source == "svd" for r in recs)

    def test_same_random_state_is_reproducible(self, train_df):
        model_a = SVDRecommender(n_factors=4, n_epochs=5, random_state=7)
        model_b = SVDRecommender(n_factors=4, n_epochs=5, random_state=7)
        model_a.fit(train_df)
        model_b.fit(train_df)

        assert model_a.predict("1", "3") == pytest.approx(model_b.predict("1", "3"))


class TestPopularityRecommender:
    def test_weighted_rating_matches_hand_computed_formula(self, movies_df):
        train_df = pd.DataFrame(
            {
                "user_id": ["1", "2", "3", "1"],
                "movie_id": ["1", "1", "1", "2"],
                "rating": [1, 0, -1, 1],
            }
        )
        model = PopularityRecommender(min_votes_quantile=0.5)
        model.fit(train_df, movies_df)

        # movie 1: n=3, ratings [1,0,-1] -> mean=0; movie 2: n=1, mean=1. global_mean = (1+0-1+1)/4 = 0.25
        # m = median of [3, 1] = 2.0
        global_mean = 0.25
        m = 2.0
        expected_wr_movie1 = (3 / (3 + m)) * 0.0 + (m / (3 + m)) * global_mean
        assert model.predict(user_id="u0", movie_id="1") == pytest.approx(expected_wr_movie1)

    def test_unrated_movie_falls_back_to_global_mean(self, movies_df, train_df):
        model = PopularityRecommender()
        model.fit(train_df, movies_df)

        assert model.predict(user_id="u0", movie_id="unknown_movie") == pytest.approx(model._global_mean)

    def test_genre_profile_biases_ranking_toward_matching_genres(self, movies_df, train_df):
        model = PopularityRecommender()
        model.fit(train_df, movies_df)

        drama_weights = np.array([0.0, 0.0, 1.0])  # pure Drama preference, in GENRES order
        recs = model.recommend_for_genre_profile(genre_weights=drama_weights, n=1, exclude_seen=False)

        # movies 4 and 5 both have Drama=1; either is a defensible top pick, but a pure-Action
        # movie (1) or pure-Comedy movie (3) should never win a pure-Drama profile.
        assert recs[0].movie_id in {"4", "5"}
        assert recs[0].source == "cold_start"


class TestHybridRecommender:
    def test_weighted_alpha_one_matches_pure_svd(self, movies_df, train_df):
        content = ContentBasedRecommender(genre_columns=GENRES)
        svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        hybrid = HybridRecommender(content, svd, strategy="weighted", alpha=1.0)
        hybrid.fit(train_df, movies_df)

        standalone_svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        standalone_svd.fit(train_df)

        assert hybrid.predict("1", "3") == pytest.approx(standalone_svd.predict("1", "3"))

    def test_weighted_alpha_zero_matches_pure_content(self, movies_df, train_df):
        content = ContentBasedRecommender(genre_columns=GENRES)
        svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        hybrid = HybridRecommender(content, svd, strategy="weighted", alpha=0.0)
        hybrid.fit(train_df, movies_df)

        standalone_content = ContentBasedRecommender(genre_columns=GENRES)
        standalone_content.fit(train_df, movies_df)

        assert hybrid.predict("1", "3") == pytest.approx(standalone_content.predict("1", "3"))

    def test_switching_strategy_uses_content_below_threshold(self, movies_df, train_df):
        content = ContentBasedRecommender(genre_columns=GENRES)
        svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        hybrid = HybridRecommender(content, svd, strategy="switching", min_ratings_for_svd=5)
        hybrid.fit(train_df, movies_df)

        # user 1 has only 2 train ratings, below min_ratings_for_svd=5
        standalone_content = ContentBasedRecommender(genre_columns=GENRES)
        standalone_content.fit(train_df, movies_df)
        assert hybrid.predict("1", "3") == pytest.approx(standalone_content.predict("1", "3"))

        # user 2 has 5 train ratings, meets the threshold
        standalone_svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        standalone_svd.fit(train_df)
        assert hybrid.predict("2", "1") == pytest.approx(standalone_svd.predict("2", "1"))


class TestColdStartRecommender:
    """Covers the required "cold-start fallback triggers correctly" case.

    Uses its own fixtures (not the shared ``movies_df``/``train_df``) with a
    6th movie that nobody ever rates: in the shared fixture, user 2 has
    rated every single movie in the 5-movie catalog, which would make an
    "above-threshold uses the base model" assertion pass vacuously against
    an empty recommendation list. The 6th movie guarantees at least one
    unseen candidate is always available.
    """

    @pytest.fixture
    def cs_movies_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "movie_id": ["1", "2", "3", "4", "5", "6"],
                "title": ["Action Only", "Action Comedy", "Comedy Only", "Drama Only", "Action Drama", "Comedy Drama"],
                "Action": [1, 1, 0, 0, 1, 0],
                "Comedy": [0, 1, 1, 0, 0, 1],
                "Drama": [0, 0, 0, 1, 1, 1],
            }
        )

    @pytest.fixture
    def cs_train_df(self) -> pd.DataFrame:
        # movie 6 is never rated by anyone -> also exercises item cold-start.
        return pd.DataFrame(
            {
                "user_id": ["1", "1", "2", "2", "2", "2", "2"],
                "movie_id": ["1", "2", "1", "2", "3", "4", "5"],
                "rating": [1, 1, 1, 0, 1, -1, 1],
            }
        )

    def _build_wrapper(self, min_user_ratings: int = 5) -> ColdStartRecommender:
        content = ContentBasedRecommender(genre_columns=GENRES)
        svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        base = HybridRecommender(content, svd, strategy="weighted", alpha=0.5)
        return ColdStartRecommender(base, PopularityRecommender(), min_user_ratings=min_user_ratings)

    def test_user_below_threshold_triggers_popularity_fallback(self, cs_movies_df, cs_train_df):
        wrapper = self._build_wrapper(min_user_ratings=5)
        wrapper.fit(cs_train_df, cs_movies_df)

        # user 1 has only 2 train ratings -> below min_user_ratings=5 -> cold-start path
        recs = wrapper.recommend_for_user("1", n=3)
        assert recs
        assert all(r.source == "cold_start" for r in recs)

    def test_user_above_threshold_uses_base_model(self, cs_movies_df, cs_train_df):
        wrapper = self._build_wrapper(min_user_ratings=5)
        wrapper.fit(cs_train_df, cs_movies_df)

        # user 2 has 5 train ratings -> meets min_user_ratings=5 -> base model path.
        # Only movie 6 is unseen for user 2, so exactly one recommendation comes back.
        recs = wrapper.recommend_for_user("2", n=3)
        assert recs
        assert all(r.source == "hybrid" for r in recs)
        assert {r.movie_id for r in recs} == {"6"}

    def test_unknown_user_with_zero_ratings_triggers_cold_start(self, cs_movies_df, cs_train_df):
        wrapper = self._build_wrapper(min_user_ratings=5)
        wrapper.fit(cs_train_df, cs_movies_df)

        recs = wrapper.recommend_for_user("unknown_user", n=3)
        assert recs
        assert all(r.source == "cold_start" for r in recs)

    def test_item_with_zero_ratings_falls_back_to_global_popularity(self, cs_movies_df, cs_train_df):
        wrapper = self._build_wrapper(min_user_ratings=5)
        wrapper.fit(cs_train_df, cs_movies_df)

        standalone_popularity = PopularityRecommender()
        standalone_popularity.fit(cs_train_df, cs_movies_df)

        # movie 6 exists in the catalog but was never rated -> genuine item cold-start,
        # not just an out-of-catalog ID.
        assert wrapper._is_cold_item("6") is True
        assert wrapper.predict("2", "6") == pytest.approx(standalone_popularity.predict("2", "6"))

        # this movie id doesn't exist in the catalog at all -> same fallback path.
        assert wrapper.predict("2", "unknown_movie") == pytest.approx(
            standalone_popularity.predict("2", "unknown_movie")
        )
