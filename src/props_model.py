"""
props_model.py

Trains one regression model per prop category (passing/rushing/receiving
yards, receptions) and produces a predicted mean plus a residual-based
standard deviation, which together let edge_finder.py compute an
over/under probability against any sportsbook line via a normal
approximation.

Supports multiple underlying model types (gradient boosting, random
forest, ridge regression, histogram gradient boosting) so different
algorithms can be compared for accuracy via scripts/backtest.py before
committing to one for production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from feature_engineering import FEATURE_COLUMNS, build_feature_set

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_TYPES = ("gbr", "rf", "ridge", "hgb")

# Empirically best model type per stat, from backtest.py comparisons
# against a real holdout season. Used when train_models.py is run with
# --model-type auto. Update this if a future backtest run finds a
# different winner.
BEST_MODEL_TYPES = {
    "passing_yards": "ridge",
    "rushing_yards": "rf",
    "receiving_yards": "gbr",
    "receptions": "rf",
}


def _make_regressor(model_type: str):
    if model_type == "gbr":
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
        )
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_depth=6, random_state=42, n_jobs=-1,
        )
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    if model_type == "hgb":
        return HistGradientBoostingRegressor(max_depth=6, random_state=42)
    raise ValueError(f"Unknown model_type '{model_type}'. Choose from {MODEL_TYPES}")


class PropModel:
    def __init__(self, stat: str, model_type: str = "gbr"):
        if stat not in FEATURE_COLUMNS:
            raise ValueError(f"Unsupported stat '{stat}'. Choose from {list(FEATURE_COLUMNS)}")
        self.stat = stat
        self.model_type = model_type
        self.features = FEATURE_COLUMNS[stat]
        self.model = _make_regressor(model_type)
        self.residual_std_: float | None = None

    def fit(self, df: pd.DataFrame) -> "PropModel":
        data = df.dropna(subset=self.features + [self.stat])
        X, y = data[self.features], data[self.stat]

        # Time-ordered CV so we never train on a player's future to predict
        # their past.
        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        for train_idx, test_idx in tscv.split(X):
            m = _make_regressor(self.model_type)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = m.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        logger.info("%s (%s): CV MAE across folds = %.2f (mean)", self.stat, self.model_type, np.mean(maes))

        self.model.fit(X, y)
        residuals = y - self.model.predict(X)
        self.residual_std_ = float(residuals.std())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.dropna(subset=self.features).copy()
        data["predicted_mean"] = self.model.predict(data[self.features])
        data["predicted_std"] = self.residual_std_
        return data

    def save(self):
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self, MODELS_DIR / f"{self.stat}_model.joblib")

    @classmethod
    def load(cls, stat: str) -> "PropModel":
        return joblib.load(MODELS_DIR / f"{stat}_model.joblib")


def train_all(
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    weather: pd.DataFrame,
    injuries: pd.DataFrame,
    model_type: str = "gbr",
) -> dict[str, PropModel]:
    """Trains and saves one model per supported prop stat."""
    trained = {}
    for stat in FEATURE_COLUMNS:
        logger.info("Building feature set for %s", stat)
        feat_df = build_feature_set(weekly, snaps, defense_allowed, weather, injuries, stat)
        model = PropModel(stat, model_type=model_type).fit(feat_df)
        model.save()
        trained[stat] = model
    return trained
