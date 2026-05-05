"""
Model evaluation utilities for the exoplanet-ml-classifier project.

Computes classification metrics, builds comparison tables, finds
optimal decision thresholds, and evaluates all models on the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from src.pipeline import Pipeline


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary with the following float-valued keys:
            - ``accuracy``
            - ``balanced_accuracy``
            - ``precision``
            - ``recall``
            - ``f1``
            - ``roc_auc``
            - ``pr_auc``
            - ``mcc``
            - ``youden_j`` (TPR − FPR at the default threshold)

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    _validate_arrays(y_true, y_pred, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "youden_j": float(tpr - fpr),
    }


def _validate_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Check that evaluation arrays are non-empty and length-compatible.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.

    Raises:
        ValueError: If any array is empty or lengths differ.
    """
    for name, arr in [("y_true", y_true), ("y_pred", y_pred), ("y_prob", y_prob)]:
        if len(arr) == 0:
            raise ValueError(f"Array '{name}' must not be empty.")
    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError(
            f"Array length mismatch: y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, y_prob={len(y_prob)}."
        )


def build_metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a DataFrame comparing metrics across models.

    Args:
        results: Dict mapping model name to its metrics dict (as returned
            by ``compute_classification_metrics``).

    Returns:
        DataFrame indexed by model name, sorted descending by ``f1``.
        Numeric values are rounded to four decimal places.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("'results' dict must not be empty.")

    df = pd.DataFrame(results).T
    df = df.sort_values("f1", ascending=False)
    return df.round(4)


def find_optimal_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find the decision threshold that maximises Youden's J statistic.

    Youden's J = TPR − FPR.  The threshold is selected from the ROC curve.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Float threshold value that maximises ``TPR − FPR``.

    Raises:
        ValueError: If inputs are empty or length-mismatched.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must be non-empty.")
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}."
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find the decision threshold that maximises the F1 score.

    Uses the precision-recall curve to evaluate all candidate thresholds.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Float threshold value that maximises F1.

    Raises:
        ValueError: If inputs are empty or length-mismatched.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must be non-empty.")
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}."
        )

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = _compute_f1_from_pr(precision, recall)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def _compute_f1_from_pr(
    precision: np.ndarray,
    recall: np.ndarray,
) -> np.ndarray:
    """Compute element-wise F1 from precision and recall arrays.

    Note: ``precision_recall_curve`` returns arrays of length n+1 while
    ``thresholds`` has length n, so the last element is excluded.

    Args:
        precision: Precision values from ``precision_recall_curve``.
        recall: Recall values from ``precision_recall_curve``.

    Returns:
        Array of F1 values, same length as ``thresholds`` (n elements).
    """
    denom = precision[:-1] + recall[:-1]
    safe_denom = np.where(denom == 0, 1, denom)
    f1 = 2 * precision[:-1] * recall[:-1] / safe_denom
    return f1


def evaluate_all_models(
    trained_pipelines: dict[str, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Evaluate every trained pipeline on the test set and return a summary.

    Args:
        trained_pipelines: Dict mapping model name to its fitted Pipeline.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Metrics DataFrame sorted descending by F1 (from
        ``build_metrics_table``).

    Raises:
        ValueError: If ``trained_pipelines`` is empty.
    """
    if not trained_pipelines:
        raise ValueError("'trained_pipelines' must not be empty.")

    y_test_arr = np.asarray(y_test)
    all_metrics: dict[str, dict] = {}

    for name, pipeline in trained_pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_prob = _get_positive_class_proba(pipeline, X_test)
        metrics = compute_classification_metrics(y_test_arr, y_pred, y_prob)
        all_metrics[name] = metrics
        print(f"{name} — F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    return build_metrics_table(all_metrics)


def _get_positive_class_proba(
    pipeline: Pipeline,
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Extract the positive-class probability column from a pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline with a classifier that exposes
            ``predict_proba``.
        X: Feature matrix.

    Returns:
        1-D array of probabilities for the positive class (index 1).

    Raises:
        AttributeError: If the classifier does not support ``predict_proba``.
    """
    if not hasattr(pipeline, "predict_proba"):
        raise AttributeError(
            "Pipeline classifier does not support predict_proba."
        )
    return pipeline.predict_proba(X)[:, 1]
