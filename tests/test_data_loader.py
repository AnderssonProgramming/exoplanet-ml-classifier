"""Unit tests for src/data_loader.py."""

import pandas as pd
import pytest

from src.constants import LEAKAGE_COLUMNS, NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COLUMN
from src.data_loader import (
    describe_dataset,
    drop_leakage_columns,
    get_feature_target_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Return a small mock DataFrame that mimics the KOI table structure."""
    n = 20
    data = {
        TARGET_COLUMN: [POSITIVE_CLASS] * 12 + [NEGATIVE_CLASS] * 8,
        "koi_period": list(range(n)),
        "koi_score": [0.9] * n,
        "koi_disposition": ["CONFIRMED"] * n,
        "koi_fpflag_nt": [0] * n,
        "koi_fpflag_ss": [0] * n,
        "koi_fpflag_co": [0] * n,
        "koi_fpflag_ec": [0] * n,
        "kepid": list(range(n)),
        "kepoi_name": [f"K{i:05d}" for i in range(n)],
        "kepler_name": [None] * n,
        "koi_comment": [None] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture()
def leakage_free_df(minimal_df: pd.DataFrame) -> pd.DataFrame:
    """Return minimal_df with leakage columns already removed."""
    cols_to_keep = [
        c for c in minimal_df.columns if c not in LEAKAGE_COLUMNS
    ]
    return minimal_df[cols_to_keep].copy()


# ---------------------------------------------------------------------------
# Tests for describe_dataset
# ---------------------------------------------------------------------------

class TestDescribeDataset:
    """Tests for the describe_dataset function."""

    EXPECTED_KEYS = {
        "n_observations",
        "n_features",
        "missing_per_column",
        "target_distribution",
        "duplicate_rows",
        "class_imbalance_ratio",
    }

    def test_returns_all_required_keys(self, minimal_df: pd.DataFrame) -> None:
        """describe_dataset must return a dict with all specified keys."""
        result = describe_dataset(minimal_df)
        assert isinstance(result, dict)
        assert self.EXPECTED_KEYS == set(result.keys())

    def test_n_observations_correct(self, minimal_df: pd.DataFrame) -> None:
        """n_observations must equal len(df)."""
        result = describe_dataset(minimal_df)
        assert result["n_observations"] == len(minimal_df)

    def test_n_features_correct(self, minimal_df: pd.DataFrame) -> None:
        """n_features must equal df.shape[1]."""
        result = describe_dataset(minimal_df)
        assert result["n_features"] == minimal_df.shape[1]

    def test_target_distribution_sums_to_one(
        self, minimal_df: pd.DataFrame
    ) -> None:
        """target_distribution values must sum to approximately 1.0."""
        result = describe_dataset(minimal_df)
        total = sum(result["target_distribution"].values())
        assert abs(total - 1.0) < 1e-9

    def test_missing_target_column_raises(self) -> None:
        """describe_dataset must raise KeyError when TARGET_COLUMN absent."""
        df_no_target = pd.DataFrame({"col_a": [1, 2, 3]})
        with pytest.raises(KeyError):
            describe_dataset(df_no_target)

    def test_imbalance_ratio_between_zero_and_one(
        self, minimal_df: pd.DataFrame
    ) -> None:
        """class_imbalance_ratio must be in (0, 1]."""
        result = describe_dataset(minimal_df)
        ratio = result["class_imbalance_ratio"]
        assert 0 < ratio <= 1.0

    def test_duplicate_rows_is_int(self, minimal_df: pd.DataFrame) -> None:
        """duplicate_rows must be a non-negative integer."""
        result = describe_dataset(minimal_df)
        assert isinstance(result["duplicate_rows"], int)
        assert result["duplicate_rows"] >= 0


# ---------------------------------------------------------------------------
# Tests for drop_leakage_columns
# ---------------------------------------------------------------------------

class TestDropLeakageColumns:
    """Tests for the drop_leakage_columns function."""

    def test_removes_all_present_leakage_columns(
        self, minimal_df: pd.DataFrame
    ) -> None:
        """All LEAKAGE_COLUMNS that exist in df must be removed."""
        result = drop_leakage_columns(minimal_df)
        for col in LEAKAGE_COLUMNS:
            assert col not in result.columns, (
                f"Leakage column '{col}' still present after drop."
            )

    def test_preserves_non_leakage_columns(
        self, minimal_df: pd.DataFrame
    ) -> None:
        """Non-leakage columns must be preserved."""
        result = drop_leakage_columns(minimal_df)
        assert TARGET_COLUMN in result.columns
        assert "koi_period" in result.columns

    def test_safe_when_leakage_columns_already_absent(
        self, leakage_free_df: pd.DataFrame
    ) -> None:
        """drop_leakage_columns must not raise if columns are already gone."""
        result = drop_leakage_columns(leakage_free_df)
        assert isinstance(result, pd.DataFrame)

    def test_raises_type_error_on_non_dataframe(self) -> None:
        """drop_leakage_columns must raise TypeError for non-DataFrame input."""
        with pytest.raises(TypeError):
            drop_leakage_columns({"not": "a dataframe"})  # type: ignore[arg-type]

    def test_returns_new_dataframe(self, minimal_df: pd.DataFrame) -> None:
        """drop_leakage_columns must return a new object, not mutate in place."""
        original_cols = set(minimal_df.columns)
        result = drop_leakage_columns(minimal_df)
        assert set(minimal_df.columns) == original_cols
        assert len(result.columns) < len(minimal_df.columns)


# ---------------------------------------------------------------------------
# Tests for get_feature_target_split
# ---------------------------------------------------------------------------

class TestGetFeatureTargetSplit:
    """Tests for the get_feature_target_split function."""

    def test_y_is_binary(self, leakage_free_df: pd.DataFrame) -> None:
        """y must contain only 0s and 1s."""
        _, y = get_feature_target_split(leakage_free_df)
        assert set(y.unique()).issubset({0, 1})

    def test_target_column_not_in_x(
        self, leakage_free_df: pd.DataFrame
    ) -> None:
        """TARGET_COLUMN must not appear in the returned X."""
        X, _ = get_feature_target_split(leakage_free_df)
        assert TARGET_COLUMN not in X.columns

    def test_positive_class_maps_to_one(
        self, leakage_free_df: pd.DataFrame
    ) -> None:
        """POSITIVE_CLASS rows must map to label 1."""
        _, y = get_feature_target_split(leakage_free_df)
        positive_mask = (
            leakage_free_df[TARGET_COLUMN] == POSITIVE_CLASS
        )
        assert (y[positive_mask.values] == 1).all()

    def test_negative_class_maps_to_zero(
        self, leakage_free_df: pd.DataFrame
    ) -> None:
        """NEGATIVE_CLASS rows must map to label 0."""
        _, y = get_feature_target_split(leakage_free_df)
        negative_mask = (
            leakage_free_df[TARGET_COLUMN] == NEGATIVE_CLASS
        )
        assert (y[negative_mask.values] == 0).all()

    def test_x_and_y_same_length(self, leakage_free_df: pd.DataFrame) -> None:
        """X and y must have the same number of rows."""
        X, y = get_feature_target_split(leakage_free_df)
        assert len(X) == len(y)

    def test_missing_target_column_raises(self) -> None:
        """Should raise KeyError if TARGET_COLUMN is absent."""
        df = pd.DataFrame({"feature_a": [1, 2, 3]})
        with pytest.raises(KeyError):
            get_feature_target_split(df)
