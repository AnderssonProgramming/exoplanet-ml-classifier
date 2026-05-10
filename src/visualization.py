"""
Visualization utilities for the exoplanet-ml-classifier project.

All functions produce matplotlib/seaborn figures and optionally save
them to FIGURES_PATH. Functions that require trained pipelines call
predict_proba internally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from src.constants import FIGURES_PATH, NUMERIC_FEATURES, RANDOM_SEED

# Consistent aesthetics across all plots
sns.set_theme(style="whitegrid", palette="muted")
_FIGURE_DPI = 150
_POSITIVE_COLOR = "#2196F3"
_NEGATIVE_COLOR = "#FF5722"


def plot_target_distribution(
    y: pd.Series,
    save: bool = True,
) -> None:
    """Plot a bar chart of class counts with percentage labels.

    Args:
        y: Binary target Series (0 / 1 or string labels).
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(6, 4), dpi=_FIGURE_DPI)
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=[_POSITIVE_COLOR, _NEGATIVE_COLOR],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, pct in zip(bars, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("KOI Target Variable Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.tight_layout()
    _maybe_save(fig, "target_distribution.png", save)
    plt.show()


def plot_missing_values_heatmap(
    df: pd.DataFrame,
    save: bool = True,
) -> None:
    """Plot a heatmap of missing values, sorted by descending missing count.

    Args:
        df: DataFrame to analyse for NaN values.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    missing_counts = df.isna().sum().sort_values(ascending=False)
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

    if not cols_with_missing:
        print("No missing values found — heatmap not generated.")
        return

    df_missing = df[cols_with_missing].isna()
    fig, ax = plt.subplots(
        figsize=(min(len(cols_with_missing) * 0.6 + 2, 20), 6),
        dpi=_FIGURE_DPI,
    )
    sns.heatmap(
        df_missing.T,
        cbar=False,
        cmap="Blues",
        ax=ax,
        yticklabels=True,
    )
    ax.set_title("Missing Values Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Row index", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    plt.tight_layout()
    _maybe_save(fig, "missing_values_heatmap.png", save)
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save: bool = True,
) -> None:
    """Plot an annotated Pearson correlation heatmap for NUMERIC_FEATURES.

    Args:
        df: DataFrame containing the numeric feature columns.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[available].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=_FIGURE_DPI)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Pearson Correlation — Numeric KOI Features",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _maybe_save(fig, "correlation_heatmap.png", save)
    plt.show()


def plot_feature_distributions_by_class(
    df: pd.DataFrame,
    target: pd.Series,
    save: bool = True,
) -> None:
    """Plot side-by-side boxplots of each numeric feature split by class.

    Args:
        df: DataFrame containing the numeric feature columns.
        target: Binary target Series aligned with ``df``.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    n = len(available)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    plot_df = df[available].copy()
    plot_df["_target"] = target.values

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5, nrows * 4), dpi=_FIGURE_DPI
    )
    axes_flat = axes.flatten()
    for idx, col in enumerate(available):
        sns.boxplot(
            data=plot_df,
            x="_target",
            y=col,
            hue="_target",
            palette={0: _NEGATIVE_COLOR, 1: _POSITIVE_COLOR},
            ax=axes_flat[idx],
            showfliers=False,
            legend=False,
        )
        axes_flat[idx].set_title(col, fontsize=10)
        axes_flat[idx].set_xlabel("Class (0=FP, 1=CAND)", fontsize=9)
        axes_flat[idx].set_ylabel("")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Feature Distributions by Class (Boxplots)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _maybe_save(fig, "feature_distributions_by_class.png", save)
    plt.show()


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    save: bool = True,
) -> None:
    """Project data to 2-D with PCA and scatter-plot coloured by class.

    The plot title includes the cumulative explained variance of the first
    two principal components.

    Args:
        X: Feature matrix (pre-processed, no NaNs).
        y: Binary label array.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    cum_var = pca.explained_variance_ratio_.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 5), dpi=_FIGURE_DPI)
    for cls, color, label in [
        (0, _NEGATIVE_COLOR, "FALSE POSITIVE"),
        (1, _POSITIVE_COLOR, "CANDIDATE"),
    ]:
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            alpha=0.4,
            s=12,
            label=label,
        )
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11
    )
    ax.set_title(
        f"PCA Scatter — cumulative variance: {cum_var:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    _maybe_save(fig, "pca_scatter.png", save)
    plt.show()


def plot_overlay_roc_curves(
    trained_pipelines: dict[str, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    save: bool = True,
) -> None:
    """Overlay ROC curves for all models on a single axes.

    A random-classifier diagonal is included for reference. AUC values
    are shown in the legend.

    Args:
        trained_pipelines: Dict mapping model name to its fitted Pipeline.
        X_test: Test feature matrix.
        y_test: True test labels.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=_FIGURE_DPI)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    for name, pipeline in trained_pipelines.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _maybe_save(fig, "overlay_roc_curves.png", save)
    plt.show()


def plot_overlay_pr_curves(
    trained_pipelines: dict[str, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    save: bool = True,
) -> None:
    """Overlay Precision-Recall curves for all models on a single axes.

    Args:
        trained_pipelines: Dict mapping model name to its fitted Pipeline.
        X_test: Test feature matrix.
        y_test: True test labels.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=_FIGURE_DPI)

    for name, pipeline in trained_pipelines.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP = {pr_auc:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curves — All Models",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    _maybe_save(fig, "overlay_pr_curves.png", save)
    plt.show()


def plot_confusion_matrix_grid(
    trained_pipelines: dict[str, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    save: bool = True,
) -> None:
    """Render a 1×N grid of confusion matrices, one per model.

    Each subplot title includes the model name and its F1 score.

    Args:
        trained_pipelines: Dict mapping model name to its fitted Pipeline.
        X_test: Test feature matrix.
        y_test: True test labels.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    n = len(trained_pipelines)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4), dpi=_FIGURE_DPI)
    if n == 1:
        axes = [axes]

    y_test_arr = np.asarray(y_test)
    for ax, (name, pipeline) in zip(axes, trained_pipelines.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test_arr, y_pred)
        f1 = f1_score(y_test_arr, y_pred, zero_division=0)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["FP", "CAND"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nF1={f1:.3f}", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Confusion Matrices — All Models",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _maybe_save(fig, "confusion_matrix_grid.png", save)
    plt.show()


def plot_feature_importance(
    model_name: str,
    pipeline: Pipeline,
    feature_names: list[str],
    save: bool = True,
) -> None:
    """Plot feature importances for tree-based models and coefficients for LR.

    For ``random_forest`` and ``xgboost``: plots a horizontal bar chart of
    the top-15 features by importance.
    For ``logistic_regression``: plots signed coefficient magnitudes.
    For ``knn`` and ``mlp``: prints a polite not-available message.

    Args:
        model_name: One of the values in ``MODEL_NAMES``.
        pipeline: Fitted sklearn Pipeline with a ``"classifier"`` step.
        feature_names: List of feature names matching the preprocessor output.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None

    Raises:
        ValueError: If the pipeline has no ``"classifier"`` step.
    """
    if "classifier" not in pipeline.named_steps:
        raise ValueError("Pipeline must have a 'classifier' step.")

    classifier = pipeline.named_steps["classifier"]

    if model_name in ("random_forest", "xgboost"):
        _plot_tree_feature_importance(
            model_name, classifier, feature_names, save
        )
    elif model_name == "logistic_regression":
        _plot_lr_coefficients(classifier, feature_names, save)
    else:
        print(
            f"Feature importance is not available for '{model_name}'. "
            f"Consider using SHAP values for model-agnostic explanations."
        )


def _plot_tree_feature_importance(
    model_name: str,
    classifier,
    feature_names: list[str],
    save: bool,
) -> None:
    """Plot top-15 feature importances for tree-based models.

    Args:
        model_name: Name used for the file stem.
        classifier: Fitted tree-based estimator with ``feature_importances_``.
        feature_names: List of feature names.
        save: Whether to save the figure.
    """
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=_FIGURE_DPI)
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=_POSITIVE_COLOR,
        edgecolor="white",
    )
    ax.set_title(
        f"Top-15 Feature Importances — {model_name}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Importance", fontsize=10)
    plt.tight_layout()
    _maybe_save(fig, f"feature_importance_{model_name}.png", save)
    plt.show()


def _plot_lr_coefficients(
    classifier,
    feature_names: list[str],
    save: bool,
) -> None:
    """Plot logistic regression coefficients as a horizontal bar chart.

    Args:
        classifier: Fitted LogisticRegression estimator.
        feature_names: List of feature names.
        save: Whether to save the figure.
    """
    coef = classifier.coef_[0]
    indices = np.argsort(np.abs(coef))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=_FIGURE_DPI)
    colors = [_POSITIVE_COLOR if c > 0 else _NEGATIVE_COLOR for c in coef[indices]]
    ax.barh(
        [feature_names[i] for i in indices],
        coef[indices],
        color=colors,
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Logistic Regression Coefficients",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Coefficient value", fontsize=10)
    plt.tight_layout()
    _maybe_save(fig, "feature_importance_logistic_regression.png", save)
    plt.show()


def plot_gmm_bic_aic(
    X: np.ndarray,
    component_range: range | None = None,
    save: bool = True,
) -> int:
    """Plot BIC and AIC for a range of GMM component counts.

    Follows the model-selection recipe shown in Hortua's Session 06
    slides — lower BIC/AIC is better, and the BIC minimum gives a
    conservative estimate of the number of Gaussian components.

    Args:
        X: Feature matrix (typically the 2-D PCA projection).
        component_range: Iterable of component counts to evaluate.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        Number of components that minimises the BIC.
    """
    if component_range is None:
        component_range = range(1, 9)

    bic_scores: list[float] = []
    aic_scores: list[float] = []
    for n in component_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=RANDOM_SEED,
        )
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    components = list(component_range)
    best_n = components[int(np.argmin(bic_scores))]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=_FIGURE_DPI)
    ax.plot(components, bic_scores, marker="o", label="BIC", color=_POSITIVE_COLOR)
    ax.plot(components, aic_scores, marker="s", label="AIC", color=_NEGATIVE_COLOR)
    ax.axvline(best_n, ls="--", color="grey", lw=1, label=f"BIC argmin = {best_n}")
    ax.set_xlabel("Number of Gaussian components", fontsize=11)
    ax.set_ylabel("Information criterion (lower = better)", fontsize=11)
    ax.set_title("GMM model selection — BIC & AIC", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _maybe_save(fig, "gmm_bic_aic.png", save)
    plt.show()
    return best_n


def plot_gmm_clusters_on_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    save: bool = True,
) -> None:
    """Fit a GMM on the 2-D PCA projection and overlay its ellipses.

    Args:
        X: Feature matrix (will be projected with PCA first).
        y: True binary labels (used to colour the points, not to fit
            the GMM).
        n_components: Number of Gaussian components.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=RANDOM_SEED,
    )
    gmm.fit(X_pca)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=_FIGURE_DPI)
    for cls, color, label in [
        (0, _NEGATIVE_COLOR, "FALSE POSITIVE"),
        (1, _POSITIVE_COLOR, "CANDIDATE"),
    ]:
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            alpha=0.35,
            s=12,
            label=label,
        )

    for mean, covar in zip(gmm.means_, gmm.covariances_):
        eigvals, eigvecs = np.linalg.eigh(covar)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigvals)  # 95 % confidence ellipse
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor="black",
            lw=1.8,
            ls="--",
        )
        ax.add_patch(ellipse)
        ax.scatter(mean[0], mean[1], marker="X", c="black", s=80)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"GMM on PCA — {n_components} Gaussian components",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best")
    plt.tight_layout()
    _maybe_save(fig, "gmm_pca_ellipses.png", save)
    plt.show()


def plot_tsne_scatter(
    X: np.ndarray,
    y: np.ndarray,
    perplexity: float = 30.0,
    save: bool = True,
) -> None:
    """Project the data to 2-D with t-SNE and scatter-plot by class.

    Args:
        X: Feature matrix (pre-processed, no NaNs).
        y: Binary label array.
        perplexity: t-SNE perplexity. The slide deck recommends a value
            between 5 and 50, with 30 being typical.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
    )
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=_FIGURE_DPI)
    for cls, color, label in [
        (0, _NEGATIVE_COLOR, "FALSE POSITIVE"),
        (1, _POSITIVE_COLOR, "CANDIDATE"),
    ]:
        mask = y == cls
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=color,
            alpha=0.4,
            s=12,
            label=label,
        )
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.set_title(
        f"t-SNE projection (perplexity = {perplexity:g})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    _maybe_save(fig, "tsne_scatter.png", save)
    plt.show()


def plot_umap_scatter(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    save: bool = True,
) -> None:
    """Project the data to 2-D with UMAP and scatter-plot by class.

    UMAP is imported lazily so the module still works for users who
    haven't installed ``umap-learn``.

    Args:
        X: Feature matrix.
        y: Binary label array.
        n_neighbors: Local-vs-global trade-off, mirroring t-SNE's
            perplexity.
        min_dist: Minimum point separation in the embedding.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None

    Raises:
        ImportError: If ``umap-learn`` is not installed.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is not installed. Run `pip install umap-learn`."
        ) from exc

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=RANDOM_SEED,
    )
    X_umap = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=_FIGURE_DPI)
    for cls, color, label in [
        (0, _NEGATIVE_COLOR, "FALSE POSITIVE"),
        (1, _POSITIVE_COLOR, "CANDIDATE"),
    ]:
        mask = y == cls
        ax.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            c=color,
            alpha=0.4,
            s=12,
            label=label,
        )
    ax.set_xlabel("UMAP dim 1", fontsize=11)
    ax.set_ylabel("UMAP dim 2", fontsize=11)
    ax.set_title(
        f"UMAP projection (n_neighbors = {n_neighbors}, min_dist = {min_dist})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    _maybe_save(fig, "umap_scatter.png", save)
    plt.show()


def plot_learning_curve(
    pipeline: Pipeline,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv: int = 5,
    scoring: str = "f1",
    train_sizes: np.ndarray | None = None,
    title: str = "Learning curve",
    save: bool = True,
) -> None:
    """Plot training- and CV-score curves as a function of train size.

    Direct counterpart of the bias-variance figure in Session 05.
    A widening gap means high variance; both curves saturating low
    means high bias.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X: Feature matrix.
        y: Target labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric name recognised by sklearn.
        train_sizes: Fractions of the training set to evaluate.
        title: Figure title (also used to build the filename).
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 6)

    sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=_FIGURE_DPI)
    ax.plot(sizes_abs, train_mean, marker="o", color=_POSITIVE_COLOR, label="Training")
    ax.fill_between(
        sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color=_POSITIVE_COLOR,
    )
    ax.plot(sizes_abs, val_mean, marker="s", color=_NEGATIVE_COLOR, label="Cross-val")
    ax.fill_between(
        sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color=_NEGATIVE_COLOR,
    )
    ax.set_xlabel("Number of training samples", fontsize=11)
    ax.set_ylabel(f"{scoring.upper()} score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    safe_name = title.lower().replace(" ", "_").replace("—", "-")
    _maybe_save(fig, f"learning_curve_{safe_name}.png", save)
    plt.show()


def plot_feature_selection_comparison(
    comparison: pd.DataFrame,
    save: bool = True,
) -> None:
    """Plot a heatmap comparing which features each selector retains.

    Args:
        comparison: DataFrame returned by
            ``feature_selection.compare_selectors`` — rows indexed by
            feature, boolean columns ``filter``, ``wrapper``,
            ``embedded``, and an integer ``agreement`` column.
        save: If ``True``, saves the figure to ``FIGURES_PATH``.

    Returns:
        None
    """
    methods = ["filter", "wrapper", "embedded"]
    matrix = comparison[methods].astype(int)

    fig, ax = plt.subplots(
        figsize=(5, max(4, 0.35 * len(matrix))),
        dpi=_FIGURE_DPI,
    )
    sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        ax=ax,
        fmt="d",
    )
    ax.set_title(
        "Feature selection — selector agreement",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    plt.tight_layout()
    _maybe_save(fig, "feature_selection_comparison.png", save)
    plt.show()


def _maybe_save(fig: plt.Figure, filename: str, save: bool) -> None:
    """Save a figure to ``FIGURES_PATH`` if ``save`` is ``True``.

    Args:
        fig: Matplotlib Figure to save.
        filename: Filename (with extension) for the saved file.
        save: Whether to perform the save operation.
    """
    if not save:
        return
    figures_dir = Path(FIGURES_PATH)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / filename
    fig.savefig(output_path, bbox_inches="tight", dpi=_FIGURE_DPI)
    print(f"Figure saved to {output_path}")
