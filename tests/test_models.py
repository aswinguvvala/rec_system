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
    _primary_language,
    genre_profile_from_movie_ids,
    movie_ids_matching_languages,
    recommend_similar_to_picks,
)

GENRES = ["Action", "Comedy", "Drama"]


@pytest.fixture
def movies_df() -> pd.DataFrame:
    # 1: Action, 2: Action+Comedy, 3: Comedy, 4: Drama, 5: Action+Drama
    # languages: 1=Telugu, 2=Telugu+Hindi, 3=Hindi, 4=Tamil, 5=Hindi
    return pd.DataFrame(
        {
            "movie_id": ["1", "2", "3", "4", "5"],
            "title": ["Action Only", "Action Comedy", "Comedy Only", "Drama Only", "Action Drama"],
            "Action": [1, 1, 0, 0, 1],
            "Comedy": [0, 1, 1, 0, 0],
            "Drama": [0, 0, 0, 1, 1],
            "languages": ["Telugu", "Telugu, Hindi", "Hindi", "Tamil", "Hindi"],
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

    def test_similar_items_candidate_restriction_excludes_other_language_neighbors(self, movies_df, train_df):
        model = ContentBasedRecommender(genre_columns=GENRES)
        model.fit(train_df, movies_df)

        # Unrestricted: movie 1 (Telugu, Action=[1,0,0]) ties in genre cosine
        # similarity with movie 2 (Telugu, Action+Comedy=[1,1,0]) and movie 5
        # (Hindi, Action+Drama=[1,0,1]) -- both are equally "similar" by genre alone.
        unrestricted = model.similar_items("1", n=10)
        assert {"2", "5"} <= {r.movie_id for r in unrestricted}

        # Restricting to Telugu-primary candidates must drop movie 5 (Hindi) even
        # though it's an exact genre-similarity tie with movie 2 (Telugu) -- genre
        # alone can't tell them apart, language restriction is what does.
        telugu_candidates = movie_ids_matching_languages(["1"], movies_df)  # {"1", "2"}
        restricted = model.similar_items("1", n=10, candidate_movie_ids=telugu_candidates)
        assert restricted
        assert all(r.movie_id in telugu_candidates for r in restricted)
        assert "5" not in {r.movie_id for r in restricted}


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

    def test_similar_items_excludes_the_movie_itself(self, train_df):
        model = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        model.fit(train_df)

        recs = model.similar_items("1", n=10)
        assert recs
        assert all(r.movie_id != "1" for r in recs)
        assert all(r.source == "svd" for r in recs)

    def test_similar_items_returns_empty_for_a_movie_with_no_training_ratings(self, train_df):
        model = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        model.fit(train_df)

        # "unknown_movie" never appears in train_df, so SVD never learned a factor
        # vector for it -- no collaborative signal exists, and this must say so
        # honestly (empty list) rather than guessing.
        assert model.similar_items("unknown_movie", n=10) == []

    def test_similar_items_candidate_restriction_limits_results_to_the_given_set(self, train_df):
        model = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        model.fit(train_df)

        restricted = model.similar_items("1", n=10, candidate_movie_ids={"2", "3"})
        assert restricted
        assert all(r.movie_id in {"2", "3"} for r in restricted)


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


class TestGenreProfileFromMovieIds:
    """Covers the live cold-start onboarding path: a brand-new user with no rating
    history picks a few movies they like in the UI, and this turns those picks into
    the same kind of genre-preference vector PopularityRecommender.recommend_for_genre_profile
    expects -- see the Streamlit app's "pick a few movies you like" flow.
    """

    def test_single_pick_returns_that_movies_genre_vector(self, movies_df):
        profile = genre_profile_from_movie_ids(["1"], movies_df, genre_columns=GENRES)  # 1: Action only
        assert list(profile) == [1.0, 0.0, 0.0]

    def test_multiple_picks_sum_genre_vectors(self, movies_df):
        # 1: Action [1,0,0], 4: Drama [0,0,1] -> summed [1,0,1]
        profile = genre_profile_from_movie_ids(["1", "4"], movies_df, genre_columns=GENRES)
        assert list(profile) == [1.0, 0.0, 1.0]

    def test_empty_picks_returns_none(self, movies_df):
        assert genre_profile_from_movie_ids([], movies_df, genre_columns=GENRES) is None

    def test_unknown_movie_ids_only_returns_none(self, movies_df):
        assert genre_profile_from_movie_ids(["does-not-exist"], movies_df, genre_columns=GENRES) is None

    def test_unknown_ids_mixed_with_known_ones_are_ignored(self, movies_df):
        profile = genre_profile_from_movie_ids(
            ["1", "does-not-exist"], movies_df, genre_columns=GENRES
        )
        assert list(profile) == [1.0, 0.0, 0.0]

    def test_feeds_recommend_for_genre_profile_end_to_end(self, movies_df, train_df):
        # Picking a pure-Drama movie should bias recommendations toward Drama,
        # exactly like the hand-verified genre_weights test above.
        model = PopularityRecommender()
        model.fit(train_df, movies_df)

        profile = genre_profile_from_movie_ids(["4"], movies_df, genre_columns=GENRES)  # 4: Drama Only
        recs = model.recommend_for_genre_profile(genre_weights=profile, n=1, exclude_seen=False)

        assert recs[0].movie_id in {"4", "5"}  # both have Drama=1
        assert recs[0].source == "cold_start"


class TestPrimaryLanguage:
    def test_single_language_returns_it(self):
        assert _primary_language("Telugu") == "Telugu"

    def test_multi_language_returns_the_first_one(self):
        # This is the exact real-world shape that motivated matching on primary
        # language at all: "Dil Se.." is a Hindi film, Hindi listed first.
        assert _primary_language("Hindi, Urdu, Assamese, Tamil, Telugu") == "Hindi"

    def test_strips_whitespace(self):
        assert _primary_language(" Telugu , Tamil") == "Telugu"

    def test_empty_string_returns_none(self):
        assert _primary_language("") is None

    def test_non_string_returns_none(self):
        assert _primary_language(None) is None
        assert _primary_language(float("nan")) is None


class TestMovieIdsMatchingLanguages:
    def test_single_pick_matches_movies_sharing_that_primary_language(self, movies_df):
        # 1: Telugu (primary) -> 2's primary is also Telugu ("Telugu, Hindi" -> Telugu first).
        matches = movie_ids_matching_languages(["1"], movies_df)
        assert matches == {"1", "2"}

    def test_does_not_match_on_a_non_primary_language_tag(self, movies_df):
        # Regression test for the real bug this caught in production: "Dil Se.." (a Hindi
        # film dubbed into Telugu among others) was slipping into Telugu-picked results
        # because it shared *a* language tag, even though Telugu wasn't its primary one.
        # Movie 2 here is built the same way: primary language Telugu, with Hindi as a
        # secondary tag ("Telugu, Hindi"). Picking movie 3 (Hindi) must NOT match it,
        # even though movie 2's tags do technically include "Hindi".
        matches = movie_ids_matching_languages(["3"], movies_df)  # 3: Hindi (primary)
        assert matches == {"3", "5"}
        assert "2" not in matches  # movie 2's primary language is Telugu, not Hindi

    def test_empty_picks_returns_none(self, movies_df):
        assert movie_ids_matching_languages([], movies_df) is None

    def test_unknown_movie_ids_only_returns_none(self, movies_df):
        assert movie_ids_matching_languages(["does-not-exist"], movies_df) is None

    def test_movie_with_no_language_info_is_never_a_match(self, movies_df):
        movies_df = movies_df.copy()
        movies_df.loc[movies_df["movie_id"] == "4", "languages"] = ""  # simulate missing language data
        # Picking movie 4 itself (now language-less) contributes nothing to "wanted".
        assert movie_ids_matching_languages(["4"], movies_df) is None
        # And movie 4 never matches anyone else's language, even when it should logically
        # never appear in a language-restricted candidate set once it has no language at all.
        matches = movie_ids_matching_languages(["3"], movies_df)  # Hindi
        assert "4" not in matches


class TestRecommendForGenreProfileCandidateRestriction:
    def test_candidate_movie_ids_restricts_the_ranked_pool(self, movies_df, train_df):
        model = PopularityRecommender()
        model.fit(train_df, movies_df)

        # Without restriction, movie 1 (highest popularity among Action movies in this
        # fixture) is a plausible top pick for a pure-Action profile.
        action_profile = np.array([1.0, 0.0, 0.0])
        unrestricted = model.recommend_for_genre_profile(genre_weights=action_profile, n=5, exclude_seen=False)
        assert any(r.movie_id == "1" for r in unrestricted)

        # Restricting to Hindi-primary candidates (3, 5 -- movie 2's primary language is
        # Telugu, not Hindi) must exclude movie 1 (Telugu) entirely, even though it would
        # otherwise be a strong Action match.
        hindi_candidates = movie_ids_matching_languages(["3"], movies_df)  # {"3", "5"}
        restricted = model.recommend_for_genre_profile(
            genre_weights=action_profile, n=5, exclude_seen=False, candidate_movie_ids=hindi_candidates
        )
        assert restricted
        assert all(r.movie_id in hindi_candidates for r in restricted)
        assert not any(r.movie_id == "1" for r in restricted)

    def test_empty_candidate_set_returns_no_recommendations(self, movies_df, train_df):
        model = PopularityRecommender()
        model.fit(train_df, movies_df)

        recs = model.recommend_for_genre_profile(genre_weights=None, n=5, candidate_movie_ids=set())
        assert recs == []


class TestRecommendSimilarToPicks:
    """Covers the live cold-start onboarding "similar to your picks" path: real
    content + collaborative nearest-neighbor similarity to each individual pick,
    not just overall popularity blended with one aggregated genre guess.
    """

    def _fitted_models(self, movies_df, train_df):
        content = ContentBasedRecommender(genre_columns=GENRES)
        svd = SVDRecommender(n_factors=4, n_epochs=5, random_state=42)
        popularity = PopularityRecommender()
        content.fit(train_df, movies_df)
        svd.fit(train_df, movies_df)
        popularity.fit(train_df, movies_df)
        return content, svd, popularity

    def test_excludes_the_users_own_pick(self, movies_df, train_df):
        content, svd, popularity = self._fitted_models(movies_df, train_df)

        recs = recommend_similar_to_picks(["1"], content, svd, popularity, movies_df, n=3)
        assert recs
        assert all(r.movie_id != "1" for r in recs)

    def test_excludes_all_picks_even_when_one_pick_is_a_neighbor_of_another(self, movies_df, train_df):
        content, svd, popularity = self._fitted_models(movies_df, train_df)

        # movie 1 (Action) and movie 2 (Action+Comedy) are genre-similar to each
        # other -- picking both must never recommend either back.
        recs = recommend_similar_to_picks(["1", "2"], content, svd, popularity, movies_df, n=5)
        assert recs
        assert all(r.movie_id not in {"1", "2"} for r in recs)

    def test_surfaces_a_genuine_genre_neighbor_not_just_popularity_order(self, movies_df, train_df):
        # movie 1 (pure Action) shares genre with movie 2 (Action+Comedy) and movie 5
        # (Action+Drama) -- a real content-similarity result, not the popularity
        # ranking, should surface at least one of them tagged as such.
        content, svd, popularity = self._fitted_models(movies_df, train_df)

        recs = recommend_similar_to_picks(["1"], content, svd, popularity, movies_df, n=5)
        assert {r.movie_id for r in recs} & {"2", "5"}
        assert any(r.source == "hybrid" for r in recs)

    def test_backfills_with_popularity_when_pick_has_no_similarity_signal(self, movies_df, train_df):
        # A pick that doesn't exist in the catalog at all has neither content nor
        # collaborative neighbors -- results must still come back via the popularity
        # backfill rather than an empty list.
        content, svd, popularity = self._fitted_models(movies_df, train_df)

        recs = recommend_similar_to_picks(["unknown-pick"], content, svd, popularity, movies_df, n=3)
        assert recs
        assert all(r.source == "cold_start" for r in recs)

    def test_language_restricts_the_primary_similarity_path_not_just_backfill(self, movies_df, train_df):
        # Movie 5 (Hindi, Action+Drama) ties movie 2 (Telugu, Action+Comedy) in raw
        # genre-cosine similarity with pick 1 (Telugu, Action) -- genre alone can't
        # tell them apart. Restricting the primary content/SVD search to
        # Telugu-primary candidates must keep movie 5 out of the real "hybrid"
        # similarity results entirely (it may still surface later as an
        # honestly-labeled popularity backfill/safety-net result once real
        # same-language candidates run out -- that's fine; being ranked as if it
        # were a genuine similarity match is the bug).
        content, svd, popularity = self._fitted_models(movies_df, train_df)

        recs = recommend_similar_to_picks(["1"], content, svd, popularity, movies_df, n=5)
        hybrid_ids = {r.movie_id for r in recs if r.source == "hybrid"}
        assert hybrid_ids == {"2"}

    def test_rare_language_pick_still_returns_n_results_via_unrestricted_safety_net(self, movies_df, train_df):
        # Movie 4's primary language (Tamil) matches no other catalog movie, so
        # both the language-restricted primary search and the language-restricted
        # backfill come up completely empty for this pick. The result must still
        # come back with a full n recommendations via one final *unrestricted*
        # popularity pass, rather than under-deliver just because the pick's
        # language happens to be rare in the catalog.
        content, svd, popularity = self._fitted_models(movies_df, train_df)

        recs = recommend_similar_to_picks(["4"], content, svd, popularity, movies_df, n=4)
        assert len(recs) == 4
        assert {r.movie_id for r in recs} == {"1", "2", "3", "5"}
        assert all(r.source == "cold_start" for r in recs)
