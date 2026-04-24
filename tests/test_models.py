"""Unit tests for src/models.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.constants import MODEL_NAMES, NUMERIC_FEATURES, RANDOM_SEED
from src.models import build_model_zoo, run_grid_search
from src.preprocessing import get_full_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small synthetic feature matrix and binary target."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 60
    X = pd.DataFrame(
        rng.standard_normal((n, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES,
    )
    y = pd.Series(
        [0] * (n // 2) + [1] * (n // 2), name="target"
    )
    return X, y


@pytest.fixture()
def lr_pipeline() -> Pipeline:
    """Return a pipeline with a LogisticRegression classifier."""
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return get_full_pipeline(lr)


# ---------------------------------------------------------------------------
# Tests for build_model_zoo
# ---------------------------------------------------------------------------

class TestBuildModelZoo:
    """Tests for the build_model_zoo function."""

    def test_returns_all_model_names(self) -> None:
        """build_model_zoo must return a dict keyed by all MODEL_NAMES."""
        zoo = build_model_zoo(RANDOM_SEED)
        assert set(zoo.keys()) == set(MODEL_NAMES)

    def test_all_values_are_estimators(self) -> None:
        """Every value in the zoo must be a sklearn-compatible estimator."""
        from sklearn.base import BaseEstimator
        zoo = build_model_zoo(RANDOM_SEED)
        for name, estimator in zoo.items():
            assert isinstance(estimator, BaseEstimator), (
                f"Model '{name}' is not a BaseEstimator subclass."
            )

    def test_no_model_is_fitted(self) -> None:
        """All estimators must be unfitted (no fitted attributes yet)."""
        zoo = build_model_zoo(RANDOM_SEED)
        for name, estimator in zoo.items():
            assert not _is_fitted(estimator), (
                f"Model '{name}' appears to be pre-fitted."
            )

    def test_lr_has_max_iter_1000(self) -> None:
        """LogisticRegression in the zoo must have max_iter=1000."""
        zoo = build_model_zoo(RANDOM_SEED)
        lr = zoo["logistic_regression"]
        assert lr.max_iter == 1000

    def test_mlp_has_random_state(self) -> None:
        """MLPClassifier must have random_state set."""
        zoo = build_model_zoo(RANDOM_SEED)
        mlp = zoo["mlp"]
        assert mlp.random_state == RANDOM_SEED

    def test_xgboost_eval_metric(self) -> None:
        """XGBClassifier must have eval_metric='logloss'."""
        zoo = build_model_zoo(RANDOM_SEED)
        xgb = zoo["xgboost"]
        assert xgb.eval_metric == "logloss"


# ---------------------------------------------------------------------------
# Tests for get_full_pipeline
# ---------------------------------------------------------------------------

class TestGetFullPipeline:
    """Tests for the get_full_pipeline function."""

    def test_returns_pipeline_instance(self, lr_pipeline: Pipeline) -> None:
        """get_full_pipeline must return a sklearn Pipeline."""
        assert isinstance(lr_pipeline, Pipeline)

    def test_has_preprocessor_step(self, lr_pipeline: Pipeline) -> None:
        """Pipeline must contain a step named 'preprocessor'."""
        step_names = [name for name, _ in lr_pipeline.steps]
        assert "preprocessor" in step_names

    def test_has_classifier_step(self, lr_pipeline: Pipeline) -> None:
        """Pipeline must contain a step named 'classifier'."""
        step_names = [name for name, _ in lr_pipeline.steps]
        assert "classifier" in step_names

    def test_exactly_two_steps(self, lr_pipeline: Pipeline) -> None:
        """Pipeline must have exactly two steps."""
        assert len(lr_pipeline.steps) == 2

    def test_classifier_is_passed_model(self, lr_pipeline: Pipeline) -> None:
        """The 'classifier' step must be the model passed to the function."""
        assert isinstance(
            lr_pipeline.named_steps["classifier"], LogisticRegression
        )


# ---------------------------------------------------------------------------
# Tests for run_grid_search
# ---------------------------------------------------------------------------

class TestRunGridSearch:
    """Tests for the run_grid_search function."""

    _LR_SMALL_GRID = {"C": [0.1, 1.0], "penalty": ["l2"], "solver": ["lbfgs"]}

    def test_returns_grid_search_cv_instance(
        self, lr_pipeline: Pipeline, small_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """run_grid_search must return a GridSearchCV instance."""
        X, y = small_xy
        result = run_grid_search(
            lr_pipeline,
            self._LR_SMALL_GRID,
            X,
            y,
            cv=2,
            scoring="f1",
        )
        assert isinstance(result, GridSearchCV)

    def test_best_params_is_set(
        self, lr_pipeline: Pipeline, small_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Fitted GridSearchCV must have a non-empty best_params_ dict."""
        X, y = small_xy
        result = run_grid_search(
            lr_pipeline,
            self._LR_SMALL_GRID,
            X,
            y,
            cv=2,
            scoring="f1",
        )
        assert hasattr(result, "best_params_")
        assert isinstance(result.best_params_, dict)
        assert len(result.best_params_) > 0

    def test_best_params_are_prefixed(
        self, lr_pipeline: Pipeline, small_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """best_params_ keys must use the 'classifier__' prefix."""
        X, y = small_xy
        result = run_grid_search(
            lr_pipeline,
            self._LR_SMALL_GRID,
            X,
            y,
            cv=2,
            scoring="f1",
        )
        for key in result.best_params_:
            assert key.startswith("classifier__"), (
                f"best_params_ key '{key}' missing 'classifier__' prefix."
            )

    def test_best_score_is_float(
        self, lr_pipeline: Pipeline, small_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """GridSearchCV best_score_ must be a float."""
        X, y = small_xy
        result = run_grid_search(
            lr_pipeline,
            self._LR_SMALL_GRID,
            X,
            y,
            cv=2,
            scoring="f1",
        )
        assert isinstance(result.best_score_, float)

    def test_raises_if_no_classifier_step(
        self, small_xy: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """run_grid_search must raise ValueError if pipeline has no classifier."""
        from sklearn.preprocessing import StandardScaler
        bad_pipeline = Pipeline([("scaler", StandardScaler())])
        X, y = small_xy
        with pytest.raises(ValueError, match="classifier"):
            run_grid_search(bad_pipeline, self._LR_SMALL_GRID, X, y, cv=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_fitted(estimator) -> bool:
    """Return True if the estimator has been fitted (has learned attributes)."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False
