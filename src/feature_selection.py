"""
Feature selection comparison utilities.

Implements the three families covered in Hortua's Session 05 slides
(18-20): filter, wrapper, and embedded methods. The goal is to rank the
13 KOI numeric features by their contribution to classification
performance and to show that very different selection strategies often
agree on the top features.

The three methods used here:

* ``SelectKBest`` with ANOVA F-statistic — **filter** method, ignores
  the downstream classifier and ranks each feature against the target
  in isolation.
* ``RFE`` with a Logistic Regression base learner — **wrapper** method,
  recursively drops the weakest feature according to model coefficients.
* ``SelectFromModel`` with a Random Forest — **embedded** method,
  derives importance from the trees themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
)
from sklearn.linear_model import LogisticRegression

from src.constants import RANDOM_SEED


def rank_features_filter(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int = 5,
) -> pd.DataFrame:
    """Rank features by ANOVA F-statistic (filter method).

    Args:
        X: Feature DataFrame with named columns.
        y: Binary target array aligned with ``X``.
        k: Number of top features to keep.

    Returns:
        DataFrame with columns ``feature``, ``score``, and ``selected``,
        sorted by descending score.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "score": selector.scores_,
            "selected": selector.get_support(),
        }
    )
    return ranking.sort_values("score", ascending=False).reset_index(drop=True)


def rank_features_wrapper(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Rank features by recursive elimination (wrapper method).

    Uses a Logistic Regression base estimator. The ``ranking`` column is
    1 for the selected features and grows with the elimination order
    (so 2 was the next-best dropped, 3 the one after, and so on).

    Args:
        X: Feature DataFrame.
        y: Binary target array.
        k: Number of features to retain.
        random_state: Seed for the base estimator.

    Returns:
        DataFrame with columns ``feature``, ``ranking``, and
        ``selected``, sorted by ascending ranking.
    """
    base = LogisticRegression(max_iter=1000, random_state=random_state)
    selector = RFE(estimator=base, n_features_to_select=k, step=1)
    selector.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "ranking": selector.ranking_,
            "selected": selector.support_,
        }
    )
    return ranking.sort_values("ranking").reset_index(drop=True)


def rank_features_embedded(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    threshold: str | float = "median",
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Rank features by Random Forest importance (embedded method).

    Args:
        X: Feature DataFrame.
        y: Binary target array.
        threshold: Importance threshold for ``SelectFromModel``. The
            default ``"median"`` keeps features whose importance is at
            least the median value.
        random_state: Seed for the Random Forest.

    Returns:
        DataFrame with columns ``feature``, ``importance``, and
        ``selected``, sorted by descending importance.
    """
    base = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    selector = SelectFromModel(estimator=base, threshold=threshold)
    selector.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": selector.estimator_.feature_importances_,
            "selected": selector.get_support(),
        }
    )
    return ranking.sort_values("importance", ascending=False).reset_index(drop=True)


def compare_selectors(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run all three selectors and return a side-by-side comparison.

    Args:
        X: Feature DataFrame.
        y: Binary target array.
        k: Number of features each selector should retain.
        random_state: Seed shared by the wrapper and embedded methods.

    Returns:
        DataFrame indexed by feature with three boolean columns —
        ``filter``, ``wrapper``, and ``embedded`` — plus a final
        ``agreement`` column counting how many of the three methods
        kept the feature.
    """
    f_rank = rank_features_filter(X, y, k=k).set_index("feature")["selected"]
    w_rank = rank_features_wrapper(
        X, y, k=k, random_state=random_state
    ).set_index("feature")["selected"]
    e_rank = rank_features_embedded(
        X, y, random_state=random_state
    ).set_index("feature")["selected"]

    comparison = pd.DataFrame(
        {
            "filter": f_rank.astype(bool),
            "wrapper": w_rank.astype(bool),
            "embedded": e_rank.astype(bool),
        }
    ).reindex(X.columns)
    comparison["agreement"] = comparison.sum(axis=1).astype(int)
    return comparison.sort_values("agreement", ascending=False)
