"""
td_model.py

Classification model for anytime-touchdown-scorer props. Structurally
similar to props_model.py's PropModel, but predicts a probability (via
predict_proba) instead of a mean/std for a normal approximation, since
anytime-TD lines are single-sided Yes/No markets, not Over/Under lines
with a point value.

Target: whether the player scored ANY touchdown (rushing or receiving --
not passing) that game.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from feature_engineering import TD_FEATURE_COLUMNS, build_td_feature_set

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET_COL = "any_td"
MODEL_TYPES = ("gbr", "rf", "hgb", "logistic")


def _make_classifier(model_type: str):
    if model_type == "gbr":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
        )
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    if model_type == "hgb":
        return HistGradientBoostingClassifier(max_depth=6, random_state=42)
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"Unknown model_type '{model_type}'. Choose from {MODEL_TYPES}")


class AnytimeTDModel:
    def __init__(self, model_type: str = "gbr"):
        self.model_type = model_type
        self.features = TD_FEATURE_COLUMNS
        self.model = _make_classifier(model_type)

    def fit(self, df: pd.DataFrame) -> "AnytimeTDModel":
        data = df.dropna(subset=self.features + [TARGET_COL])
        X, y = data[self.features], data[TARGET_COL]

        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for train_idx, test_idx in tscv.split(X):
            if y.iloc[train_idx].nunique() < 2:
                continue  # AUC undefined without both classes present
            m = _make_classifier(self.model_type)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = m.predict_proba(X.iloc[test_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(y.iloc[test_idx], proba))
            except ValueError:
                pass
        if aucs:
            logger.info("anytime_td (%s): CV AUC across folds = %.3f (mean)",
                        self.model_type, np.mean(aucs))
        else:
            logger.warning("anytime_td: could not compute CV AUC (insufficient class balance in folds)")

        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.dropna(subset=self.features).copy()
        data["predicted_prob"] = self.model.predict_proba(data[self.features])[:, 1]
        return data

    def save(self):
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self, MODELS_DIR / "anytime_td_model.joblib")

    @classmethod
    def load(cls) -> "AnytimeTDModel":
        return joblib.load(MODELS_DIR / "anytime_td_model.joblib")


def train_td_model(
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    weather: pd.DataFrame,
    injuries: pd.DataFrame,
    model_type: str = "gbr",
) -> AnytimeTDModel:
    feat_df = build_td_feature_set(weekly, snaps, defense_allowed, weather, injuries)
    model = AnytimeTDModel(model_type=model_type).fit(feat_df)
    model.save()
    return model
