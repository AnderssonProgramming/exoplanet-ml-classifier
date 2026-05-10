"""Unit tests for ``src.feature_selection``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.feature_selection import (
    compare_selectors,
    rank_features_embedded,
    rank_features_filter,
    rank_features_wrapper,
)


@pytest.fixture
def toy_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    columns = [f"feat_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


class TestFilter:
    def test_returns_one_row_per_feature(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_filter(X, y, k=4)
        assert len(ranking) == X.shape[1]
        assert set(ranking.columns) == {"feature", "score", "selected"}

    def test_selected_count_matches_k(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_filter(X, y, k=4)
        assert ranking["selected"].sum() == 4

    def test_scores_sorted_descending(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_filter(X, y, k=4)
        assert ranking["score"].is_monotonic_decreasing


class TestWrapper:
    def test_selected_count_matches_k(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_wrapper(X, y, k=3)
        assert ranking["selected"].sum() == 3

    def test_ranking_sorted_ascending(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_wrapper(X, y, k=3)
        assert ranking["ranking"].is_monotonic_increasing


class TestEmbedded:
    def test_returns_one_row_per_feature(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_embedded(X, y)
        assert len(ranking) == X.shape[1]

    def test_importance_non_negative(self, toy_dataset):
        X, y = toy_dataset
        ranking = rank_features_embedded(X, y)
        assert (ranking["importance"] >= 0).all()


class TestCompareSelectors:
    def test_shape_and_columns(self, toy_dataset):
        X, y = toy_dataset
        comparison = compare_selectors(X, y, k=4)
        assert len(comparison) == X.shape[1]
        assert {"filter", "wrapper", "embedded", "agreement"}.issubset(
            comparison.columns
        )

    def test_agreement_is_in_valid_range(self, toy_dataset):
        X, y = toy_dataset
        comparison = compare_selectors(X, y, k=4)
        assert comparison["agreement"].between(0, 3).all()

    def test_at_least_one_feature_picked_by_all_three(self, toy_dataset):
        X, y = toy_dataset
        comparison = compare_selectors(X, y, k=4)
        assert (comparison["agreement"] >= 1).any()
