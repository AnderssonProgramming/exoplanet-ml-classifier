"""
Preprocessing utilities for the exoplanet-ml-classifier project.

Provides the sklearn ColumnTransformer for imputation + scaling,
stratified train/val/test splitting, optional SMOTE oversampling,
and the full sklearn Pipeline factory used by training code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import NUMERIC_FEATURES, RANDOM_SEED


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that imputes then scales numeric features.

    The transformer applies the following steps to every column listed in
    ``NUMERIC_FEATURES``:
        1. Median imputation (``SimpleImputer(strategy="median")``).
        2. Z-score standardisation (``StandardScaler``).

    All columns not listed in ``NUMERIC_FEATURES`` are dropped
    (``remainder="drop"``).

    Returns:
        Unfitted ``ColumnTransformer`` ready for use inside a Pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split data into stratified train, validation, and test sets.

    The split is performed in two stages:
        1. Hold out ``test_size`` of all data as the test set.
        2. From the remaining data, hold out ``val_size`` as the
           validation set.

    Both splits use stratification on ``y`` to preserve class proportions.

    Args:
        X: Feature matrix.
        y: Binary target Series.
        test_size: Proportion of the full dataset to use as test set
            (e.g. ``0.2``).
        val_size: Proportion of the **remaining** data (after removing the
            test set) to use as validation set (e.g. ``0.1``).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``.

    Raises:
        ValueError: If ``test_size`` or ``val_size`` are outside (0, 1).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state,
    )
    _print_split_info(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _print_split_info(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Print stratified split sizes and class proportions.

    Args:
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.
    """
    total = len(y_train) + len(y_val) + len(y_test)
    print(
        f"Split sizes — train: {len(y_train)} "
        f"({len(y_train)/total:.1%}), "
        f"val: {len(y_val)} ({len(y_val)/total:.1%}), "
        f"test: {len(y_test)} ({len(y_test)/total:.1%})"
    )
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos_frac = y.mean()
        print(f"  {name} positive rate: {pos_frac:.3f}")


def apply_smote_if_imbalanced(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    threshold: float = 0.3,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling only when the class ratio is below threshold.

    The minority-to-majority ratio is computed on the raw counts of
    ``y_train``. SMOTE is applied only if that ratio falls below
    ``threshold``.

    Args:
        X_train: Training feature matrix (DataFrame or array).
        y_train: Training labels (Series or array).
        threshold: Minority/majority ratio below which SMOTE is applied.
            Default is ``0.3``.
        random_state: Random seed passed to SMOTE.

    Returns:
        Tuple ``(X_resampled, y_resampled)`` as numpy arrays.
        If SMOTE is not applied, the originals are returned converted to
        arrays.

    Raises:
        ValueError: If ``threshold`` is outside (0, 1).
    """
    if not 0 < threshold < 1:
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}."
        )

    y_arr = np.asarray(y_train)
    ratio = _compute_minority_ratio(y_arr)

    if ratio < threshold:
        print(
            f"Class ratio {ratio:.3f} < threshold {threshold} — "
            f"applying SMOTE."
        )
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_arr)
        print(
            f"After SMOTE — X shape: {X_resampled.shape}, "
            f"y distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}"
        )
        return X_resampled, y_resampled

    print(
        f"Class ratio {ratio:.3f} >= threshold {threshold} — "
        f"SMOTE not applied."
    )
    return np.asarray(X_train), y_arr


def _compute_minority_ratio(y: np.ndarray) -> float:
    """Compute the minority-class fraction relative to the majority class.

    Args:
        y: 1-D array of binary labels.

    Returns:
        Float in ``(0, 1]``: minority count / majority count.
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return 1.0
    return float(counts.min() / counts.max())


def get_full_pipeline(model) -> Pipeline:
    """Assemble a full sklearn Pipeline: preprocessor → classifier.

    Args:
        model: An unfitted sklearn-compatible estimator.

    Returns:
        ``Pipeline`` with steps ``"preprocessor"`` and ``"classifier"``.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )
