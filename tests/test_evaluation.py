"""Unit tests for src/evaluation.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    build_metrics_table,
    compute_classification_metrics,
    find_optimal_threshold_f1,
    find_optimal_threshold_youden,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def perfect_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays where every prediction matches the truth."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)
    return y_true, y_pred, y_prob


@pytest.fixture()
def random_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays with non-trivial errors so all metrics are non-degenerate."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.random(size=n)
    # Bias the probabilities slightly towards the truth so AUC > 0.5
    y_prob = 0.7 * y_true + 0.3 * y_prob
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Tests for compute_classification_metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics."""

    EXPECTED_KEYS = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "youden_j",
    }

    def test_returns_all_required_keys(self, random_predictions) -> None:
        """Result dict must contain every expected metric key."""
        y_true, y_pred, y_prob = random_predictions
        result = compute_classification_metrics(y_true, y_pred, y_prob)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_perfect_predictions_yield_unit_metrics(
        self, perfect_predictions
    ) -> None:
        """For perfect predictions every standard metric must equal 1.0."""
        y_true, y_pred, y_prob = perfect_predictions
        result = compute_classification_metrics(y_true, y_pred, y_prob)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc"):
            assert result[key] == pytest.approx(1.0)

    def test_metric_values_in_expected_ranges(self, random_predictions) -> None:
        """All probability-style metrics live in [0, 1]; MCC lives in [-1, 1]."""
        y_true, y_pred, y_prob = random_predictions
        result = compute_classification_metrics(y_true, y_pred, y_prob)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert 0.0 <= result[key] <= 1.0
        assert -1.0 <= result["mcc"] <= 1.0

    def test_empty_array_raises(self) -> None:
        """Passing empty arrays must raise ValueError."""
        empty = np.array([], dtype=int)
        with pytest.raises(ValueError):
            compute_classification_metrics(empty, empty, empty.astype(float))

    def test_length_mismatch_raises(self) -> None:
        """Length mismatch across the three arrays must raise ValueError."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        y_prob = np.array([0.1, 0.9, 0.4])
        with pytest.raises(ValueError):
            compute_classification_metrics(y_true, y_pred, y_prob)


# ---------------------------------------------------------------------------
# Tests for build_metrics_table
# ---------------------------------------------------------------------------

class TestBuildMetricsTable:
    """Tests for build_metrics_table."""

    def test_returns_dataframe_sorted_by_f1(self) -> None:
        """The returned DataFrame must be sorted descending by F1."""
        results = {
            "model_a": {
                "accuracy": 0.9, "balanced_accuracy": 0.9, "precision": 0.9,
                "recall": 0.9, "f1": 0.5, "roc_auc": 0.95, "pr_auc": 0.9,
                "mcc": 0.8, "youden_j": 0.8,
            },
            "model_b": {
                "accuracy": 0.85, "balanced_accuracy": 0.85, "precision": 0.85,
                "recall": 0.85, "f1": 0.85, "roc_auc": 0.9, "pr_auc": 0.85,
                "mcc": 0.7, "youden_j": 0.7,
            },
        }
        df = build_metrics_table(results)
        assert isinstance(df, pd.DataFrame)
        assert df.index[0] == "model_b"
        assert df.index[1] == "model_a"

    def test_empty_results_raises(self) -> None:
        """Calling with an empty dict must raise ValueError."""
        with pytest.raises(ValueError):
            build_metrics_table({})


# ---------------------------------------------------------------------------
# Tests for threshold optimisation
# ---------------------------------------------------------------------------

class TestThresholdOptimisation:
    """Tests for both threshold-finding helpers."""

    def test_youden_threshold_in_unit_interval(
        self, random_predictions
    ) -> None:
        """The Youden threshold must lie in [0, 1] for probabilistic outputs."""
        y_true, _, y_prob = random_predictions
        thresh = find_optimal_threshold_youden(y_true, y_prob)
        assert 0.0 <= thresh <= 1.0

    def test_f1_threshold_in_unit_interval(
        self, random_predictions
    ) -> None:
        """The F1-optimal threshold must lie in [0, 1]."""
        y_true, _, y_prob = random_predictions
        thresh = find_optimal_threshold_f1(y_true, y_prob)
        assert 0.0 <= thresh <= 1.0

    def test_youden_empty_input_raises(self) -> None:
        """Empty inputs must raise ValueError."""
        with pytest.raises(ValueError):
            find_optimal_threshold_youden(np.array([]), np.array([]))

    def test_f1_empty_input_raises(self) -> None:
        """Empty inputs must raise ValueError."""
        with pytest.raises(ValueError):
            find_optimal_threshold_f1(np.array([]), np.array([]))

    def test_youden_length_mismatch_raises(self) -> None:
        """Length mismatch must raise ValueError."""
        with pytest.raises(ValueError):
            find_optimal_threshold_youden(
                np.array([0, 1, 0]), np.array([0.1, 0.9])
            )

    def test_f1_length_mismatch_raises(self) -> None:
        """Length mismatch must raise ValueError."""
        with pytest.raises(ValueError):
            find_optimal_threshold_f1(
                np.array([0, 1, 0]), np.array([0.1, 0.9])
            )
