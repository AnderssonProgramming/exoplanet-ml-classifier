"""Unit tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.constants import NUMERIC_FEATURES, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE
from src.preprocessing import apply_smote_if_imbalanced, split_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def balanced_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Return a balanced (50/50) feature matrix and target Series."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES,
    )
    y = pd.Series([0] * (n // 2) + [1] * (n // 2), name="target")
    return X, y


@pytest.fixture()
def imbalanced_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Return a strongly imbalanced (10% minority) feature matrix and target."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES,
    )
    n_minority = int(n * 0.10)
    y = pd.Series(
        [1] * n_minority + [0] * (n - n_minority), name="target"
    )
    return X, y


# ---------------------------------------------------------------------------
# Tests for split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    """Tests for the split_data function."""

    _TOLERANCE = 0.02  # ±2% tolerance on split proportions

    def test_split_sizes_correct(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Train/val/test sizes must be within ±2% of expected proportions."""
        X, y = balanced_xy
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, TEST_SIZE, VALIDATION_SIZE, RANDOM_SEED
        )
        total = len(X)
        expected_test = TEST_SIZE
        expected_val = (1 - TEST_SIZE) * VALIDATION_SIZE
        expected_train = 1 - expected_test - expected_val

        actual_test = len(X_test) / total
        actual_val = len(X_val) / total
        actual_train = len(X_train) / total

        assert abs(actual_test - expected_test) <= self._TOLERANCE, (
            f"Test proportion {actual_test:.3f} outside tolerance of {expected_test:.3f}"
        )
        assert abs(actual_val - expected_val) <= self._TOLERANCE, (
            f"Val proportion {actual_val:.3f} outside tolerance of {expected_val:.3f}"
        )
        assert abs(actual_train - expected_train) <= self._TOLERANCE, (
            f"Train proportion {actual_train:.3f} outside tolerance of {expected_train:.3f}"
        )

    def test_no_overlap_between_splits(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Train, val, and test index sets must be mutually exclusive."""
        X, y = balanced_xy
        X_train, X_val, X_test, *_ = split_data(
            X, y, TEST_SIZE, VALIDATION_SIZE, RANDOM_SEED
        )
        idx_train = set(X_train.index)
        idx_val = set(X_val.index)
        idx_test = set(X_test.index)

        assert len(idx_train & idx_val) == 0, "Train/val overlap detected."
        assert len(idx_train & idx_test) == 0, "Train/test overlap detected."
        assert len(idx_val & idx_test) == 0, "Val/test overlap detected."

    def test_stratification_preserved(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Positive class proportion must be similar across all splits."""
        X, y = balanced_xy
        _, _, _, y_train, y_val, y_test = split_data(
            X, y, TEST_SIZE, VALIDATION_SIZE, RANDOM_SEED
        )
        original_ratio = y.mean()
        for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
            split_ratio = y_split.mean()
            assert abs(split_ratio - original_ratio) <= 0.05, (
                f"{name} class ratio {split_ratio:.3f} deviates from "
                f"original {original_ratio:.3f}."
            )

    def test_returns_six_elements(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """split_data must return exactly 6 elements."""
        X, y = balanced_xy
        result = split_data(X, y, TEST_SIZE, VALIDATION_SIZE, RANDOM_SEED)
        assert len(result) == 6

    def test_invalid_test_size_raises(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """test_size outside (0, 1) must raise ValueError."""
        X, y = balanced_xy
        with pytest.raises(ValueError):
            split_data(X, y, 1.5, VALIDATION_SIZE, RANDOM_SEED)

    def test_invalid_val_size_raises(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """val_size outside (0, 1) must raise ValueError."""
        X, y = balanced_xy
        with pytest.raises(ValueError):
            split_data(X, y, TEST_SIZE, 0.0, RANDOM_SEED)


# ---------------------------------------------------------------------------
# Tests for apply_smote_if_imbalanced
# ---------------------------------------------------------------------------

class TestApplySmoteIfImbalanced:
    """Tests for the apply_smote_if_imbalanced function."""

    _DEFAULT_THRESHOLD = 0.3

    def test_does_not_apply_smote_when_balanced(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """SMOTE must NOT be applied when minority ratio is above threshold."""
        X, y = balanced_xy
        X_res, y_res = apply_smote_if_imbalanced(
            X, y, threshold=self._DEFAULT_THRESHOLD, random_state=RANDOM_SEED
        )
        assert len(X_res) == len(X), (
            "SMOTE was applied to balanced data — X length changed."
        )
        assert len(y_res) == len(y), (
            "SMOTE was applied to balanced data — y length changed."
        )

    def test_applies_smote_when_imbalanced(
        self, imbalanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """SMOTE must be applied and class counts balanced when ratio is low."""
        X, y = imbalanced_xy
        original_minority = (y == 1).sum()
        X_res, y_res = apply_smote_if_imbalanced(
            X, y, threshold=self._DEFAULT_THRESHOLD, random_state=RANDOM_SEED
        )
        new_minority = (y_res == 1).sum()
        assert new_minority > original_minority, (
            "SMOTE did not increase minority class count."
        )
        assert len(X_res) > len(X), "SMOTE did not increase dataset size."

    def test_smote_produces_only_binary_labels(
        self, imbalanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """After SMOTE the label array must contain only 0s and 1s."""
        X, y = imbalanced_xy
        _, y_res = apply_smote_if_imbalanced(
            X, y, threshold=self._DEFAULT_THRESHOLD, random_state=RANDOM_SEED
        )
        assert set(np.unique(y_res)).issubset({0, 1}), (
            f"Unexpected labels after SMOTE: {np.unique(y_res)}"
        )

    def test_invalid_threshold_raises(
        self, balanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """threshold outside (0, 1) must raise ValueError."""
        X, y = balanced_xy
        with pytest.raises(ValueError):
            apply_smote_if_imbalanced(X, y, threshold=1.5)

    def test_x_and_y_same_length_after_smote(
        self, imbalanced_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """X and y must have the same length after SMOTE is applied."""
        X, y = imbalanced_xy
        X_res, y_res = apply_smote_if_imbalanced(
            X, y, threshold=self._DEFAULT_THRESHOLD, random_state=RANDOM_SEED
        )
        assert len(X_res) == len(y_res)
