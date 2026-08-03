"""Tests for src/data_pipeline.py: parsing, merging, and the chronological split."""

import pandas as pd
import pytest
import requests

from src.data_pipeline import (
    GENRE_COLUMNS,
    DataDownloadError,
    DataFormatError,
    chronological_train_test_split,
    download_movielens_100k,
    load_movies,
    load_ratings,
    load_users,
    merge_data,
)


class TestChronologicalTrainTestSplit:
    def test_splits_by_most_recent_timestamp_per_user(self):
        ratings = pd.DataFrame(
            {
                "user_id": [1] * 10,
                "movie_id": range(10),
                "rating": [3] * 10,
                "timestamp": range(10),
            }
        )
        train, test = chronological_train_test_split(ratings, test_frac=0.2, min_ratings_for_test=5)

        # 10 ratings * 20% = 2 held out -> the two most recent timestamps (8, 9)
        assert sorted(test["timestamp"].tolist()) == [8, 9]
        assert len(train) == 8
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_users_below_threshold_are_entirely_in_train(self):
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "movie_id": [1, 2, 3],
                "rating": [3, 4, 5],
                "timestamp": [1, 2, 3],
            }
        )
        train, test = chronological_train_test_split(ratings, test_frac=0.2, min_ratings_for_test=5)

        assert len(test) == 0
        assert len(train) == 3

    def test_multiple_users_split_independently(self):
        ratings = pd.DataFrame(
            {
                "user_id": [1] * 10 + [2] * 3,
                "movie_id": list(range(10)) + list(range(3)),
                "rating": [3] * 13,
                "timestamp": list(range(10)) + list(range(3)),
            }
        )
        train, test = chronological_train_test_split(ratings, test_frac=0.2, min_ratings_for_test=5)

        assert len(test[test["user_id"] == 1]) == 2
        assert len(test[test["user_id"] == 2]) == 0
        assert len(train[train["user_id"] == 2]) == 3


def test_merge_data_joins_ratings_movies_users():
    ratings = pd.DataFrame({"user_id": [1], "movie_id": [10], "rating": [4], "timestamp": [100]})
    movies = pd.DataFrame({"movie_id": [10], "title": ["Test Movie"]})
    users = pd.DataFrame({"user_id": [1], "age": [30]})

    merged = merge_data(ratings, movies, users)

    assert merged.loc[0, "title"] == "Test Movie"
    assert merged.loc[0, "age"] == 30


class TestLoadRatings:
    def test_parses_tab_separated_file(self, tmp_path):
        (tmp_path / "u.data").write_text("1\t2\t3\t880000000\n4\t5\t4\t880000001\n")

        df = load_ratings(tmp_path)

        assert list(df.columns) == ["user_id", "movie_id", "rating", "timestamp"]
        assert len(df) == 2
        assert df.iloc[0]["rating"] == 3

    def test_missing_file_raises_data_format_error(self, tmp_path):
        with pytest.raises(DataFormatError):
            load_ratings(tmp_path)


class TestLoadMovies:
    def test_parses_pipe_separated_file_with_genres(self, tmp_path):
        genre_values = "|".join(["1"] + ["0"] * (len(GENRE_COLUMNS) - 1))
        line = f"1|Test Movie (1995)|01-Jan-1995||http://example.com|{genre_values}\n"
        (tmp_path / "u.item").write_text(line, encoding="latin-1")

        df = load_movies(tmp_path)

        assert df.iloc[0]["title"] == "Test Movie (1995)"
        assert df.iloc[0][GENRE_COLUMNS[0]] == 1
        assert df.iloc[0][GENRE_COLUMNS[1]] == 0

    def test_missing_file_raises_data_format_error(self, tmp_path):
        with pytest.raises(DataFormatError):
            load_movies(tmp_path)


class TestLoadUsers:
    def test_parses_pipe_separated_file(self, tmp_path):
        (tmp_path / "u.user").write_text("1|25|M|student|12345\n")

        df = load_users(tmp_path)

        assert df.iloc[0]["age"] == 25
        assert df.iloc[0]["occupation"] == "student"

    def test_missing_file_raises_data_format_error(self, tmp_path):
        with pytest.raises(DataFormatError):
            load_users(tmp_path)


class TestDownloadMovielens100k:
    def test_uses_local_cache_without_hitting_network(self, tmp_path, monkeypatch):
        extracted = tmp_path / "ml-100k"
        extracted.mkdir()
        (extracted / "u.data").write_text("1\t1\t5\t100\n")

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("download_movielens_100k hit the network despite a valid cache")

        monkeypatch.setattr("src.data_pipeline.requests.get", _fail_if_called)

        result = download_movielens_100k(tmp_path)

        assert result == extracted

    def test_network_failure_raises_data_download_error(self, tmp_path, monkeypatch):
        def _raise(*args, **kwargs):
            raise requests.exceptions.ConnectionError("simulated network failure")

        monkeypatch.setattr("src.data_pipeline.requests.get", _raise)

        with pytest.raises(DataDownloadError):
            download_movielens_100k(tmp_path)
