"""Tests for src/data_pipeline.py: parsing, cleaning, merging, and the split."""

import csv
import json

import pandas as pd
import pytest

from src.data_pipeline import (
    GENRE_COLUMNS,
    DataDownloadError,
    DataFormatError,
    _is_junk_user_id,
    _processed_cache_is_usable,
    download_indian_movies_dataset,
    load_movies,
    load_ratings,
    load_supplementary_movies,
    load_users,
    merge_data,
    merge_supplementary_movies,
    random_train_test_split,
    run_pipeline,
)


class TestRandomTrainTestSplit:
    def test_holds_out_the_expected_fraction_per_user(self):
        ratings = pd.DataFrame({"user_id": ["1"] * 10, "movie_id": [str(i) for i in range(10)], "rating": [1] * 10})
        train, test = random_train_test_split(ratings, test_frac=0.2, min_ratings_for_test=5, random_state=42)

        # 10 ratings * 20% = 2 held out
        assert len(test) == 2
        assert len(train) == 8
        # every held-out row really did come from the original set, and train/test don't overlap
        assert set(test["movie_id"]) <= set(ratings["movie_id"])
        assert set(train["movie_id"]).isdisjoint(set(test["movie_id"]))

    def test_users_below_threshold_are_entirely_in_train(self):
        ratings = pd.DataFrame({"user_id": ["1", "1", "1"], "movie_id": ["1", "2", "3"], "rating": [1, 0, -1]})
        train, test = random_train_test_split(ratings, test_frac=0.2, min_ratings_for_test=5)

        assert len(test) == 0
        assert len(train) == 3

    def test_multiple_users_split_independently(self):
        ratings = pd.DataFrame(
            {
                "user_id": ["1"] * 10 + ["2"] * 3,
                "movie_id": [str(i) for i in range(10)] + [str(i) for i in range(3)],
                "rating": [1] * 13,
            }
        )
        train, test = random_train_test_split(ratings, test_frac=0.2, min_ratings_for_test=5)

        assert len(test[test["user_id"] == "1"]) == 2
        assert len(test[test["user_id"] == "2"]) == 0
        assert len(train[train["user_id"] == "2"]) == 3

    def test_same_random_state_is_reproducible(self):
        ratings = pd.DataFrame({"user_id": ["1"] * 10, "movie_id": [str(i) for i in range(10)], "rating": [1] * 10})
        train_a, test_a = random_train_test_split(ratings, test_frac=0.2, random_state=7)
        train_b, test_b = random_train_test_split(ratings, test_frac=0.2, random_state=7)

        assert sorted(test_a["movie_id"]) == sorted(test_b["movie_id"])


def test_merge_data_joins_ratings_movies_users():
    ratings = pd.DataFrame({"user_id": ["u1"], "movie_id": ["tt10"], "rating": [1]})
    movies = pd.DataFrame({"movie_id": ["tt10"], "title": ["Test Movie"]})
    users = pd.DataFrame({"user_id": ["u1"], "age": [30]})

    merged = merge_data(ratings, movies, users)

    assert merged.loc[0, "title"] == "Test Movie"
    assert merged.loc[0, "age"] == 30


class TestLoadRatings:
    def test_parses_line_delimited_json_and_drops_the_submit_key(self, tmp_path):
        record = {"_id": "user1", "rated": {"tt001": ["1"], "tt002": ["0"], "submit": ["submit"]}}
        (tmp_path / "ratings.json").write_text(json.dumps(record) + "\n", encoding="utf-8")

        df = load_ratings(tmp_path)

        assert list(df.columns) == ["user_id", "movie_id", "rating"]
        assert len(df) == 2  # "submit" is not a rating
        assert set(df["movie_id"]) == {"tt001", "tt002"}
        assert df.loc[df["movie_id"] == "tt001", "rating"].iloc[0] == 1
        assert df.loc[df["movie_id"] == "tt002", "rating"].iloc[0] == 0

    def test_multiple_user_records(self, tmp_path):
        lines = [
            json.dumps({"_id": "user1", "rated": {"tt001": ["1"]}}),
            json.dumps({"_id": "user2", "rated": {"tt001": ["-1"], "tt002": ["1"]}}),
        ]
        (tmp_path / "ratings.json").write_text("\n".join(lines) + "\n", encoding="utf-8")

        df = load_ratings(tmp_path)

        assert len(df) == 3
        assert set(df["user_id"]) == {"user1", "user2"}

    def test_unrecognized_rating_value_is_skipped_not_raised(self, tmp_path):
        record = {"_id": "user1", "rated": {"tt001": ["1"], "tt002": ["not-a-real-value"]}}
        (tmp_path / "ratings.json").write_text(json.dumps(record) + "\n", encoding="utf-8")

        df = load_ratings(tmp_path)

        assert len(df) == 1
        assert df.iloc[0]["movie_id"] == "tt001"

    def test_missing_file_raises_data_format_error(self, tmp_path):
        with pytest.raises(DataFormatError):
            load_ratings(tmp_path)

    def test_malformed_json_line_raises_data_format_error(self, tmp_path):
        (tmp_path / "ratings.json").write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(DataFormatError):
            load_ratings(tmp_path)


def _write_movies_csv(tmp_path, rows: list[dict]) -> None:
    fieldnames = ["movie_id", "description", "language", "released", "rating", "writer", "director", "cast", "genre", "name"]
    with (tmp_path / "movies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestLoadMovies:
    def test_parses_title_year_rating_and_genres(self, tmp_path):
        _write_movies_csv(
            tmp_path,
            [
                {
                    "movie_id": "tt001",
                    "description": "A test movie.",
                    "language": json.dumps(["Hindi"]),
                    "released": "2016-02-19T00:00:00.000Z",
                    "rating": "7.9",
                    "writer": json.dumps(["A Writer"]),
                    "director": json.dumps(["A Director"]),
                    "cast": json.dumps(["An Actor"]),
                    "genre": json.dumps(["Drama", "Thriller"]),
                    "name": "Test Movie",
                }
            ],
        )

        df = load_movies(tmp_path)

        assert df.iloc[0]["title"] == "Test Movie"
        assert df.iloc[0]["release_year"] == 2016
        assert df.iloc[0]["imdb_rating"] == pytest.approx(7.9)
        assert df.iloc[0]["languages"] == "Hindi"
        assert df.iloc[0]["Drama"] == 1
        assert df.iloc[0]["Thriller"] == 1
        assert df.iloc[0]["Comedy"] == 0

    def test_empty_genre_list_gives_all_zero_vector(self, tmp_path):
        _write_movies_csv(
            tmp_path,
            [
                {
                    "movie_id": "tt002",
                    "description": "",
                    "language": "[]",
                    "released": "",
                    "rating": "",
                    "writer": "[]",
                    "director": "[]",
                    "cast": "[]",
                    "genre": "[]",
                    "name": "No Genre Movie",
                }
            ],
        )

        df = load_movies(tmp_path)

        assert df.iloc[0][GENRE_COLUMNS].sum() == 0

    def test_missing_file_raises_data_format_error(self, tmp_path):
        with pytest.raises(DataFormatError):
            load_movies(tmp_path)


class TestLoadSupplementaryMovies:
    def test_missing_file_returns_empty_correctly_shaped_frame(self, tmp_path):
        df = load_supplementary_movies(tmp_path / "does_not_exist.csv")

        assert df.empty
        assert list(df.columns) == ["movie_id", "title", "release_year", "imdb_rating", "languages", *GENRE_COLUMNS]

    def test_malformed_file_returns_empty_frame_instead_of_raising(self, tmp_path):
        path = tmp_path / "supplement.csv"
        path.write_text("", encoding="utf-8")  # empty file -> EmptyDataError

        df = load_supplementary_movies(path)

        assert df.empty

    def test_valid_file_is_loaded_with_string_movie_ids(self, tmp_path):
        path = tmp_path / "supplement.csv"
        pd.DataFrame(
            [{"movie_id": "tt0000123", "title": "Extra Movie", "release_year": 2020,
              "imdb_rating": float("nan"), "languages": "Telugu", **{g: 0 for g in GENRE_COLUMNS}}]
        ).to_csv(path, index=False)

        df = load_supplementary_movies(path)

        assert df.iloc[0]["movie_id"] == "tt0000123"
        assert isinstance(df.iloc[0]["movie_id"], str)


class TestMergeSupplementaryMovies:
    def _base_movies(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"movie_id": "tt0001", "title": "Base Movie", "release_year": 2015,
              "imdb_rating": 7.5, "languages": "Hindi", **{g: 0 for g in GENRE_COLUMNS}}]
        )

    def test_empty_supplement_returns_base_unchanged(self):
        base = self._base_movies()
        empty = pd.DataFrame(columns=base.columns)

        result = merge_supplementary_movies(base, empty)

        assert len(result) == 1
        assert result.iloc[0]["movie_id"] == "tt0001"

    def test_new_supplementary_movie_is_appended(self):
        base = self._base_movies()
        supplement = pd.DataFrame(
            [{"movie_id": "tt0002", "title": "New Movie", "release_year": 2021,
              "imdb_rating": float("nan"), "languages": "Telugu", **{g: 0 for g in GENRE_COLUMNS}}]
        )

        result = merge_supplementary_movies(base, supplement)

        assert set(result["movie_id"]) == {"tt0001", "tt0002"}

    def test_colliding_movie_id_keeps_the_base_row(self):
        base = self._base_movies()
        supplement = pd.DataFrame(
            [{"movie_id": "tt0001", "title": "Should Not Win", "release_year": 1999,
              "imdb_rating": float("nan"), "languages": "Tamil", **{g: 0 for g in GENRE_COLUMNS}}]
        )

        result = merge_supplementary_movies(base, supplement)

        assert len(result) == 1
        assert result.iloc[0]["title"] == "Base Movie"
        assert result.iloc[0]["languages"] == "Hindi"


def _write_users_csv(tmp_path, rows: list[dict]) -> None:
    fieldnames = ["_id", "languages", "job", "state", "dob", "gender"]
    with (tmp_path / "users.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestIsJunkUserId:
    @pytest.mark.parametrize("user_id", ["n", "p", "ab", "ABCDEFGHIJKLM", "abcdefghijklm", "abcdefghi jklm"])
    def test_flags_known_junk_patterns(self, user_id):
        assert _is_junk_user_id(user_id) is True

    @pytest.mark.parametrize("user_id", ["11megha89", "ANAND", "9953547227", "real_user_42"])
    def test_does_not_flag_real_looking_ids(self, user_id):
        assert _is_junk_user_id(user_id) is False


class TestLoadUsers:
    def test_parses_age_occupation_and_state(self, tmp_path):
        _write_users_csv(
            tmp_path,
            [{"_id": "realuser1", "languages": json.dumps(["Hindi"]), "job": "Student", "state": "Delhi", "dob": "16-06-2000", "gender": "Male"}],
        )

        df = load_users(tmp_path)

        assert df.iloc[0]["user_id"] == "realuser1"
        assert df.iloc[0]["age"] == 2017 - 2000  # _SURVEY_COLLECTION_YEAR - birth year
        assert df.iloc[0]["occupation"] == "Student"
        assert df.iloc[0]["state"] == "Delhi"
        assert df.iloc[0]["gender"] == "Male"

    def test_junk_ids_are_dropped(self, tmp_path):
        _write_users_csv(
            tmp_path,
            [
                {"_id": "realuser1", "languages": "[]", "job": "Student", "state": "Delhi", "dob": "16-06-2000", "gender": "Male"},
                {"_id": "n", "languages": "[]", "job": "Student", "state": "Delhi", "dob": "16-06-2000", "gender": "Male"},
            ],
        )

        df = load_users(tmp_path)

        assert len(df) == 1
        assert df.iloc[0]["user_id"] == "realuser1"

    def test_missing_file_raises_data_format_error(self, tmp_path):
        with pytest.raises(DataFormatError):
            load_users(tmp_path)


class TestDownloadIndianMoviesDataset:
    def test_uses_local_cache_without_hitting_kaggle(self, tmp_path, monkeypatch):
        extracted = tmp_path / "indian_movies"
        extracted.mkdir()
        (extracted / "ratings.json").write_text("{}\n", encoding="utf-8")

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("download_indian_movies_dataset hit Kaggle despite a valid cache")

        monkeypatch.setattr("kaggle.api.authenticate", _fail_if_called)

        result = download_indian_movies_dataset(tmp_path)

        assert result == extracted

    def test_kaggle_failure_raises_data_download_error(self, tmp_path, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("simulated: no Kaggle credentials configured")

        monkeypatch.setattr("kaggle.api.authenticate", _raise)

        with pytest.raises(DataDownloadError):
            download_indian_movies_dataset(tmp_path)


class TestProcessedCacheIsUsable:
    def test_missing_file_is_not_usable(self, tmp_path):
        assert _processed_cache_is_usable(tmp_path / "movies.csv") is False

    def test_file_with_every_genre_column_is_usable(self, tmp_path):
        movies_path = tmp_path / "movies.csv"
        pd.DataFrame(columns=["movie_id", "title"] + GENRE_COLUMNS).to_csv(movies_path, index=False)

        assert _processed_cache_is_usable(movies_path) is True

    def test_file_missing_genre_columns_is_not_usable(self, tmp_path):
        # Reproduces the real bug this check was added for: a cached movies.csv left over
        # from a previous, different dataset (e.g. MovieLens's 19-genre taxonomy) satisfies
        # "the file exists" but is missing several of this dataset's real genre columns.
        movies_path = tmp_path / "movies.csv"
        stale_columns = ["movie_id", "title", "Action", "Comedy", "Drama"]  # missing most of GENRE_COLUMNS
        pd.DataFrame(columns=stale_columns).to_csv(movies_path, index=False)

        assert _processed_cache_is_usable(movies_path) is False


class TestRunPipelineCacheInvalidation:
    def test_stale_processed_cache_is_rebuilt_not_trusted(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        extracted = raw_dir / "indian_movies"
        extracted.mkdir(parents=True)

        _write_movies_csv(
            extracted,
            [
                {
                    "movie_id": "tt001",
                    "description": "",
                    "language": "[]",
                    "released": "2020-01-01T00:00:00.000Z",
                    "rating": "8.0",
                    "writer": "[]",
                    "director": "[]",
                    "cast": "[]",
                    "genre": json.dumps(["Drama"]),
                    "name": "A Real Movie",
                }
            ],
        )
        _write_users_csv(
            extracted,
            [{"_id": "realuser1", "languages": "[]", "job": "Student", "state": "Delhi", "dob": "16-06-2000", "gender": "Male"}],
        )
        (extracted / "ratings.json").write_text(
            json.dumps({"_id": "realuser1", "rated": {"tt001": ["1"]}}) + "\n", encoding="utf-8"
        )

        # Simulate a processed/ dir left over from a previous, different dataset: present,
        # so the naive "do the files exist" check alone would trust it, but missing almost
        # every real genre column.
        processed_dir.mkdir(parents=True)
        pd.DataFrame(columns=["user_id", "movie_id", "rating"]).to_csv(processed_dir / "train.csv", index=False)
        pd.DataFrame(columns=["user_id", "movie_id", "rating"]).to_csv(processed_dir / "test.csv", index=False)
        pd.DataFrame(columns=["movie_id", "title", "Action", "Comedy", "Drama"]).to_csv(processed_dir / "movies.csv", index=False)
        pd.DataFrame(columns=["user_id", "age"]).to_csv(processed_dir / "users.csv", index=False)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("run_pipeline hit Kaggle even though a valid raw-data cache exists")

        monkeypatch.setattr("kaggle.api.authenticate", _fail_if_called)

        result = run_pipeline(
            raw_dir=raw_dir, processed_dir=processed_dir,
            supplementary_movies_path=tmp_path / "no_supplement_here.csv",
        )

        # Rebuilt from the real raw data, not the stale processed cache -- proven by the
        # rebuilt movies frame actually having the full real genre taxonomy.
        assert set(GENRE_COLUMNS).issubset(result["movies"].columns)
        assert list(result["movies"]["movie_id"]) == ["tt001"]
