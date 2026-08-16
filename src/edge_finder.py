"""
edge_finder.py

Compares model predictions (mean + std) against sportsbook prop lines to
find edges, using a normal-approximation over/under probability and
Kelly-criterion stake sizing. Consolidates what was split across
kelly_criterion.py and daily_picks.py in the MLB system into one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def american_to_prob(price: int) -> float:
    """Converts American odds to implied probability (with vig)."""
    if price > 0:
        return 100 / (price + 100)
    return -price / (-price + 100)


def american_to_decimal(price: int) -> float:
    if price > 0:
        return 1 + price / 100
    return 1 + 100 / -price


def prob_over(predicted_mean: float, predicted_std: float, line: float) -> float:
    """P(actual stat > line) under a normal approximation of the residual
    distribution learned during training."""
    if predicted_std <= 0:
        return float(predicted_mean > line)
    return float(1 - norm.cdf(line, loc=predicted_mean, scale=predicted_std))


def kelly_fraction(model_prob: float, decimal_odds: float, kelly_cap: float = 0.25) -> float:
    """Fractional Kelly stake as a fraction of bankroll, capped for variance
    control (full Kelly is too aggressive for a props model with imperfect
    calibration)."""
    b = decimal_odds - 1
    edge = model_prob * decimal_odds - 1
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return max(0.0, min(full_kelly, 1.0)) * kelly_cap


def find_edges(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    min_edge_pct: float = 0.03,
) -> pd.DataFrame:
    """
    predictions: one row per player/stat with predicted_mean, predicted_std
    odds: one row per player/stat/book/side with line, price

    Returns picks where the model's edge over the book's implied
    probability exceeds `min_edge_pct`, sorted by edge descending.

    Only handles Over/Under-style markets (yardage, receptions) -- use
    find_td_edges for the single-sided anytime-TD-scorer market.
    """
    odds = odds[odds["stat"] != "anytime_td"]
    if odds.empty:
        return odds

    merged = odds.merge(
        predictions[["player_name", "stat", "predicted_mean", "predicted_std"]],
        on=["player_name", "stat"],
        how="inner",
    )

    merged["model_prob_over"] = merged.apply(
        lambda r: prob_over(r["predicted_mean"], r["predicted_std"], r["line"]), axis=1
    )
    merged["model_prob"] = np.where(
        merged["side"] == "Over", merged["model_prob_over"], 1 - merged["model_prob_over"]
    )
    merged["implied_prob"] = merged["price"].apply(american_to_prob)
    merged["decimal_odds"] = merged["price"].apply(american_to_decimal)
    merged["edge"] = merged["model_prob"] - merged["implied_prob"]
    merged["kelly_stake_pct"] = merged.apply(
        lambda r: kelly_fraction(r["model_prob"], r["decimal_odds"]), axis=1
    )

    # Keep the best-priced book per player/stat/side, then filter by edge.
    best = (
        merged.sort_values("edge", ascending=False)
        .drop_duplicates(subset=["player_name", "stat", "side"])
    )
    picks = best[best["edge"] >= min_edge_pct].sort_values("edge", ascending=False)
    return picks.reset_index(drop=True)


def find_td_edges(
    td_predictions: pd.DataFrame,
    odds: pd.DataFrame,
    min_edge_pct: float = 0.03,
) -> pd.DataFrame:
    """
    Anytime-TD-scorer props are single-sided (Yes only, no line) -- the
    edge is just model probability of scoring vs. the book's implied
    probability from the American price directly, no normal-
    approximation needed since there's no Over/Under to model.

    td_predictions: one row per player with predicted_prob (from
    AnytimeTDModel.predict).
    """
    td_odds = odds[odds["stat"] == "anytime_td"].copy()
    if td_odds.empty:
        return td_odds

    merged = td_odds.merge(
        td_predictions[["player_name", "predicted_prob"]],
        on="player_name",
        how="inner",
    )
    if merged.empty:
        return merged

    merged["implied_prob"] = merged["price"].apply(american_to_prob)
    merged["decimal_odds"] = merged["price"].apply(american_to_decimal)
    merged["model_prob"] = merged["predicted_prob"]
    merged["edge"] = merged["model_prob"] - merged["implied_prob"]
    merged["kelly_stake_pct"] = merged.apply(
        lambda r: kelly_fraction(r["model_prob"], r["decimal_odds"]), axis=1
    )

    best = merged.sort_values("edge", ascending=False).drop_duplicates(subset=["player_name"])
    picks = best[best["edge"] >= min_edge_pct].sort_values("edge", ascending=False)
    return picks.reset_index(drop=True)
