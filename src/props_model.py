"""
props_model.py

Trains one gradient-boosted regression model per prop category
(passing/rushing/receiving yards, receptions) and produces a predicted
mean plus a residual-based standard deviation, which together let
edge_finder.py compute an over/under probability against any sportsbook
line via a normal approximation.

Simpler than the MLB system's per-pitch model on purpose: one model type
(LightGBM/GradientBoosting), one feature pipeline shared across stat
categories, trained on 3-8 years of clean nflverse history instead of
custom-scraped data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from feature_engineering import FEATURE_COLUMNS, build_feature_set

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class PropModel:
    def __init__(self, stat: str):
        if stat not in FEATURE_COLUMNS:
            raise ValueError(f"Unsupported stat '{stat}'. Choose from {list(FEATURE_COLUMNS)}")
        self.stat = stat
        self.features = FEATURE_COLUMNS[stat]
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.residual_std_: float | None = None

    def fit(self, df: pd.DataFrame) -> "PropModel":
        data = df.dropna(subset=self.features + [self.stat])
        X, y = data[self.features], data[self.stat]

        # Time-ordered CV so we never train on a player's future to predict
        # their past -- the same discipline the MLB system's model_trainer
        # should have had, made non-optional here.
        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        for train_idx, test_idx in tscv.split(X):
            m = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = m.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        logger.info("%s: CV MAE across folds = %.2f (mean)", self.stat, np.mean(maes))

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
) -> dict[str, PropModel]:
    """Trains and saves one model per supported prop stat."""
    trained = {}
    for stat in FEATURE_COLUMNS:
        logger.info("Building feature set for %s", stat)
        feat_df = build_feature_set(weekly, snaps, defense_allowed, stat)
        model = PropModel(stat).fit(feat_df)
        model.save()
        trained[stat] = model
    return trained
