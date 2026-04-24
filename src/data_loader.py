"""
Data loading utilities for the exoplanet-ml-classifier project.

Handles reading the raw NASA KOI cumulative table, describing dataset
statistics, removing data-leakage columns, and producing the feature/
target split used by downstream modules.
"""

from pathlib import Path

import pandas as pd

from src.constants import (
    LEAKAGE_COLUMNS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)


def load_raw_koi_data(path: Path) -> pd.DataFrame:
    """Load the raw NASA KOI cumulative CSV from disk.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        DataFrame containing all rows and columns from the CSV.

    Raises:
        FileNotFoundError: If the CSV does not exist at ``path``.
        ValueError: If the file cannot be parsed as a CSV.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path, comment="#")
    print(f"Loaded dataset — shape: {df.shape}")
    print(f"First 3 rows:\n{df.head(3).to_string()}")
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the raw KOI dataset.

    Args:
        df: DataFrame (post-load, pre-cleaning).

    Returns:
        Dictionary with the following keys:
            - ``n_observations`` (int): number of rows.
            - ``n_features`` (int): number of columns.
            - ``missing_per_column`` (dict): column → missing count.
            - ``target_distribution`` (dict): class → relative frequency.
            - ``duplicate_rows`` (int): number of exact duplicate rows.
            - ``class_imbalance_ratio`` (float): minority / majority count.

    Raises:
        KeyError: If ``TARGET_COLUMN`` is not present in ``df``.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in DataFrame."
        )

    missing = _compute_missing(df)
    target_dist = _compute_target_distribution(df)
    imbalance = _compute_imbalance_ratio(target_dist, df)

    return {
        "n_observations": len(df),
        "n_features": df.shape[1],
        "missing_per_column": missing,
        "target_distribution": target_dist,
        "duplicate_rows": int(df.duplicated().sum()),
        "class_imbalance_ratio": imbalance,
    }


def _compute_missing(df: pd.DataFrame) -> dict:
    """Return a column → missing-count mapping for columns with missing data.

    Args:
        df: Input DataFrame.

    Returns:
        Dict mapping column name to integer missing count.
    """
    missing_series = df.isna().sum()
    return missing_series[missing_series > 0].to_dict()


def _compute_target_distribution(df: pd.DataFrame) -> dict:
    """Compute the normalised class frequencies of the target column.

    Args:
        df: DataFrame containing ``TARGET_COLUMN``.

    Returns:
        Dict mapping class label to its relative frequency (sums to 1).
    """
    return df[TARGET_COLUMN].value_counts(normalize=True).to_dict()


def _compute_imbalance_ratio(
    target_dist: dict, df: pd.DataFrame
) -> float:
    """Compute minority-to-majority class count ratio.

    Args:
        target_dist: Normalised target distribution dict.
        df: Original DataFrame (used for absolute counts).

    Returns:
        Float ratio of minority class count to majority class count.
    """
    counts = df[TARGET_COLUMN].value_counts()
    if len(counts) < 2:
        return 1.0
    minority = counts.min()
    majority = counts.max()
    return float(minority / majority)


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that encode the label or are post-hoc measurements.

    Only columns that actually exist in ``df`` are dropped so that the
    function is safe to call on already-cleaned DataFrames.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with leakage columns removed.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")

    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    print(f"Dropping {len(cols_to_drop)} leakage columns: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def get_feature_target_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into features X and binary target y.

    The target column is encoded as:
        - ``POSITIVE_CLASS`` (``"CANDIDATE"``) → 1
        - ``NEGATIVE_CLASS`` (``"FALSE POSITIVE"``) → 0

    Rows whose target value is neither class label are dropped with a
    warning.

    Args:
        df: DataFrame that must contain ``TARGET_COLUMN``.

    Returns:
        Tuple ``(X, y)`` where:
            - ``X`` is a DataFrame of all columns except ``TARGET_COLUMN``.
            - ``y`` is a binary integer Series (0 or 1).

    Raises:
        KeyError: If ``TARGET_COLUMN`` is not in ``df``.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in DataFrame."
        )

    df_clean = _filter_valid_target_rows(df)
    y = _encode_target(df_clean[TARGET_COLUMN])
    X = df_clean.drop(columns=[TARGET_COLUMN])
    return X, y


def _filter_valid_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose target value is a recognised class label.

    Args:
        df: DataFrame with ``TARGET_COLUMN``.

    Returns:
        Filtered DataFrame.
    """
    valid_mask = df[TARGET_COLUMN].isin([POSITIVE_CLASS, NEGATIVE_CLASS])
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(
            f"Warning: dropping {n_dropped} rows with unrecognised "
            f"target values."
        )
    return df[valid_mask].copy()


def _encode_target(series: pd.Series) -> pd.Series:
    """Encode the target string column to binary integers.

    Args:
        series: Series containing ``POSITIVE_CLASS`` / ``NEGATIVE_CLASS``.

    Returns:
        Integer Series (1 = CANDIDATE, 0 = FALSE POSITIVE).
    """
    mapping = {POSITIVE_CLASS: 1, NEGATIVE_CLASS: 0}
    return series.map(mapping).astype(int)
