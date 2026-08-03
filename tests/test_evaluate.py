"""Tests for src/evaluate.py metric functions against hand-computed values."""

import math

import numpy as np
import pytest

from src.evaluate import mae, ndcg_at_k, precision_at_k, recall_at_k, rmse


class TestRatingMetrics:
    def test_rmse_matches_hand_computed_value(self):
        y_true = np.array([3, 4, 5])
        y_pred = np.array([2, 4, 4])
        # squared errors: 1, 0, 1 -> mean = 2/3 -> sqrt
        assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(2 / 3))

    def test_mae_matches_hand_computed_value(self):
        y_true = np.array([3, 4, 5])
        y_pred = np.array([2, 4, 4])
        # abs errors: 1, 0, 1 -> mean = 2/3
        assert mae(y_true, y_pred) == pytest.approx(2 / 3)

    def test_rmse_and_mae_are_zero_for_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0


class TestRankingMetrics:
    # Shared fixture: 5 recommended items, ranked best-first.
    RECOMMENDED = [10, 20, 30, 40, 50]

    def test_precision_at_k_matches_hand_computed_value(self):
        relevant = {20, 40, 60}
        # top 3 = [10, 20, 30] -> 1 hit (20) / 3 recommended
        assert precision_at_k(self.RECOMMENDED, relevant, k=3) == pytest.approx(1 / 3)

    def test_precision_at_k_uses_actual_list_length_as_denominator(self):
        # only 2 items recommended even though k=5 -> denominator is 2, not 5
        assert precision_at_k([1, 2], {1}, k=5) == pytest.approx(0.5)

    def test_precision_at_k_empty_recommendations_is_zero(self):
        assert precision_at_k([], {1, 2}, k=5) == 0.0

    def test_recall_at_k_matches_hand_computed_value(self):
        relevant = {20, 40, 60}
        # top 3 = [10, 20, 30] -> 1 hit / 3 relevant
        assert recall_at_k(self.RECOMMENDED, relevant, k=3) == pytest.approx(1 / 3)
        # top 5 = all 5 -> hits {20, 40} -> 2 / 3 relevant
        assert recall_at_k(self.RECOMMENDED, relevant, k=5) == pytest.approx(2 / 3)

    def test_recall_at_k_empty_relevant_set_is_zero(self):
        assert recall_at_k(self.RECOMMENDED, set(), k=5) == 0.0

    def test_ndcg_at_k_matches_hand_computed_value(self):
        relevant = {20, 40}
        # hits at 0-indexed ranks 1 (movie 20) and 3 (movie 40)
        dcg = 1 / math.log2(1 + 2) + 1 / math.log2(3 + 2)
        # ideal: both relevant items packed at ranks 0 and 1
        idcg = 1 / math.log2(0 + 2) + 1 / math.log2(1 + 2)
        expected = dcg / idcg
        assert ndcg_at_k(self.RECOMMENDED, relevant, k=5) == pytest.approx(expected)

    def test_ndcg_at_k_perfect_ranking_is_one(self):
        relevant = {10, 20}
        assert ndcg_at_k(self.RECOMMENDED, relevant, k=2) == pytest.approx(1.0)

    def test_ndcg_at_k_empty_relevant_set_is_zero(self):
        assert ndcg_at_k(self.RECOMMENDED, set(), k=5) == 0.0
