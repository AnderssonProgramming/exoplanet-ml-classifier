"""Smoke tests for src/visualization.py.

We use the Agg backend so figures never try to open a window during CI.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.constants import NUMERIC_FEATURES, RANDOM_SEED
from src.preprocessing import get_full_pipeline
from src.feature_selection import compare_selectors
from src.visualization import (
    plot_correlation_heatmap,
    plot_feature_selection_comparison,
    plot_gmm_bic_aic,
    plot_gmm_clusters_on_pca,
    plot_learning_curve,
    plot_overlay_pr_curves,
    plot_overlay_roc_curves,
    plot_pca_scatter,
    plot_target_distribution,
    plot_tsne_scatter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_all_figures():
    """Close any matplotlib figures opened by the test."""
    yield
    plt.close("all")


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """Return a DataFrame with all NUMERIC_FEATURES columns."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 80
    return pd.DataFrame(
        rng.standard_normal((n, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES,
    )


@pytest.fixture()
def binary_target() -> pd.Series:
    """Return a balanced binary target Series."""
    return pd.Series([0] * 40 + [1] * 40, name="target")


@pytest.fixture()
def trained_pipelines(numeric_df, binary_target) -> dict:
    """Return a small dict of trained pipelines for overlay-plot tests."""
    lr_pipeline = get_full_pipeline(
        LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    )
    lr_pipeline.fit(numeric_df, binary_target)
    return {"logistic_regression": lr_pipeline}


# ---------------------------------------------------------------------------
# Smoke tests — every plotting function must run without raising
# ---------------------------------------------------------------------------

class TestPlotsDoNotRaise:
    """Each plotting function must execute end-to-end without errors."""

    def test_target_distribution(self, binary_target: pd.Series) -> None:
        plot_target_distribution(binary_target, save=False)

    def test_correlation_heatmap(self, numeric_df: pd.DataFrame) -> None:
        plot_correlation_heatmap(numeric_df, save=False)

    def test_pca_scatter(self, numeric_df: pd.DataFrame, binary_target) -> None:
        plot_pca_scatter(numeric_df.values, binary_target.values, save=False)

    def test_overlay_roc_curves(
        self,
        trained_pipelines: dict,
        numeric_df: pd.DataFrame,
        binary_target: pd.Series,
    ) -> None:
        plot_overlay_roc_curves(
            trained_pipelines, numeric_df, binary_target, save=False
        )

    def test_overlay_pr_curves(
        self,
        trained_pipelines: dict,
        numeric_df: pd.DataFrame,
        binary_target: pd.Series,
    ) -> None:
        plot_overlay_pr_curves(
            trained_pipelines, numeric_df, binary_target, save=False
        )

    def test_gmm_bic_aic_returns_int(self, numeric_df: pd.DataFrame) -> None:
        best = plot_gmm_bic_aic(numeric_df.values, range(1, 4), save=False)
        assert isinstance(best, int)
        assert 1 <= best <= 3

    def test_gmm_clusters_on_pca(
        self, numeric_df: pd.DataFrame, binary_target: pd.Series
    ) -> None:
        plot_gmm_clusters_on_pca(
            numeric_df.values, binary_target.values, n_components=2, save=False
        )

    def test_tsne_scatter(
        self, numeric_df: pd.DataFrame, binary_target: pd.Series
    ) -> None:
        plot_tsne_scatter(
            numeric_df.values, binary_target.values, perplexity=10, save=False
        )

    def test_learning_curve(
        self,
        numeric_df: pd.DataFrame,
        binary_target: pd.Series,
    ) -> None:
        pipeline = get_full_pipeline(
            LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
        )
        plot_learning_curve(
            pipeline,
            numeric_df,
            binary_target,
            cv=3,
            scoring="f1",
            train_sizes=np.linspace(0.3, 1.0, 3),
            title="LR test curve",
            save=False,
        )

    def test_feature_selection_comparison(
        self, numeric_df: pd.DataFrame, binary_target: pd.Series
    ) -> None:
        comparison = compare_selectors(numeric_df, binary_target.values, k=4)
        plot_feature_selection_comparison(comparison, save=False)
