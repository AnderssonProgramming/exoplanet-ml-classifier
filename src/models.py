"""
Model factory, training, evaluation, and persistence utilities.

Provides a model zoo of five classifier families, cross-validation
helpers, GridSearchCV wrappers, and joblib-based save/load functions.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.constants import PRIMARY_METRIC, RANDOM_SEED


def build_model_zoo(random_state: int = RANDOM_SEED) -> dict[str, BaseEstimator]:
    """Instantiate all seven classifier families with sensible defaults.

    Returns Logistic Regression, k-NN, Decision Tree, SVM, Random Forest,
    XGBoost, and MLP — covering linear, instance-based, single-tree,
    margin-based, bagging, boosting, and neural inductive biases.

    Args:
        random_state: Integer seed passed to all stochastic estimators.

    Returns:
        Dictionary mapping each name in ``MODEL_NAMES`` to an unfitted
        estimator instance.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            max_iter=500,
            random_state=random_state,
        ),
    }


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Pipeline:
    """Fit a pipeline on the training data and return it.

    Args:
        pipeline: Unfitted sklearn Pipeline (preprocessor + classifier).
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        The same ``pipeline`` object, now fitted.

    Raises:
        ValueError: If ``X_train`` and ``y_train`` have incompatible lengths.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train length {len(X_train)} != y_train length {len(y_train)}."
        )
    pipeline.fit(X_train, y_train)
    return pipeline


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv: int,
    scoring: str,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Run k-fold cross-validation and return the results dictionary.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X: Full feature matrix (train + val combined for CV).
        y: Full label array.
        cv: Number of folds.
        scoring: Scorer name recognised by sklearn (e.g. ``"f1"``).
        random_state: Seed — currently unused but kept for API consistency.

    Returns:
        Dictionary from ``sklearn.model_selection.cross_validate`` with
        keys including ``fit_time``, ``score_time``, and ``test_<scoring>``.
    """
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    mean_score = results[f"test_{scoring}"].mean()
    std_score = results[f"test_{scoring}"].std()
    print(
        f"CV {scoring} — mean: {mean_score:.4f} ± {std_score:.4f}"
    )
    return results


def run_grid_search(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cv: int,
    scoring: str = PRIMARY_METRIC,
) -> GridSearchCV:
    """Fit a GridSearchCV over the classifier step of the pipeline.

    The function automatically prefixes all keys in ``param_grid`` with
    ``"classifier__"`` so callers can supply bare hyperparameter names.

    Args:
        pipeline: Unfitted sklearn Pipeline with a ``"classifier"`` step.
        param_grid: Dict of bare hyperparameter names → candidate values.
        X_train: Training feature matrix.
        y_train: Training labels.
        cv: Number of cross-validation folds.
        scoring: Metric used to select the best estimator.

    Returns:
        Fitted ``GridSearchCV`` object with ``best_params_`` and
        ``best_score_`` attributes set.

    Raises:
        ValueError: If the pipeline has no step named ``"classifier"``.
    """
    step_names = [name for name, _ in pipeline.steps]
    if "classifier" not in step_names:
        raise ValueError(
            f"Pipeline must contain a 'classifier' step. "
            f"Found steps: {step_names}."
        )

    prefixed_grid = _prefix_param_grid(param_grid)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=prefixed_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best {scoring}: {grid_search.best_score_:.4f}")
    return grid_search


def _prefix_param_grid(param_grid: dict) -> dict:
    """Add ``"classifier__"`` prefix to all keys in a hyperparameter grid.

    Args:
        param_grid: Dict whose keys are bare estimator parameter names.

    Returns:
        New dict with all keys prefixed by ``"classifier__"``.
    """
    return {f"classifier__{k}": v for k, v in param_grid.items()}


def save_model(pipeline: Pipeline, name: str, path: Path) -> None:
    """Persist a fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline to save.
        name: Base filename (without extension).
        path: Directory where the file will be written.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created or the file cannot be
            written.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_file = path / f"{name}.joblib"
    joblib.dump(pipeline, output_file)
    print(f"Model saved to {output_file}")


def load_model(name: str, path: Path) -> Pipeline:
    """Load a fitted pipeline from disk using joblib.

    Args:
        name: Base filename (without extension).
        path: Directory containing the saved file.

    Returns:
        Fitted ``Pipeline`` loaded from disk.

    Raises:
        FileNotFoundError: If ``path/name.joblib`` does not exist.
    """
    model_file = Path(path) / f"{name}.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"No saved model found at {model_file}.")
    pipeline = joblib.load(model_file)
    print(f"Model loaded from {model_file}")
    return pipeline
