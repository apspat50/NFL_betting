"""
weekly_picks.py

The main orchestrator -- run once a week during the season (e.g. Tuesday
morning after props post) via GitHub Actions or manually.

1. Loads current-season data through last week.
2. Builds this week's feature rows for every player likely to play.
3. Loads trained models and predicts mean/std per stat.
4. Pulls live sportsbook lines via The Odds API.
5. Finds edges and sends picks to Telegram.

Simpler than the MLB daily pipeline: one run per week, no intraday
re-scraping, no live-line polling.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import (
    load_weekly_stats, load_snap_counts, load_schedules, team_defense_allowed,
)
from feature_engineering import build_feature_set, FEATURE_COLUMNS
from props_model import PropModel
from odds_client import OddsClient
from edge_finder import find_edges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_this_week_features(season: int, upcoming_week: int) -> pd.DataFrame:
    """Builds feature rows for every player with recent history, attached
    to their upcoming opponent for this week."""
    seasons = tuple(range(season - 3, season + 1))  # trailing 3 seasons + current
    weekly = load_weekly_stats(seasons)
    snaps = load_snap_counts(seasons)
    defense = team_defense_allowed(seasons)
    sched = load_schedules((season,))

    matchups = sched[sched["week"] == upcoming_week][
        ["week", "home_team", "away_team"]
    ]
    home = matchups.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = matchups.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    this_week_matchups = pd.concat([home, away], ignore_index=True)
    this_week_matchups["season"] = season

    # Attach opponent to historical rows so opponent-context features compute,
    # then keep only the latest row per player (their upcoming matchup).
    weekly_sched = load_schedules(seasons)[["season", "week", "home_team", "away_team"]]
    h = weekly_sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    a = weekly_sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    hist_matchups = pd.concat([h, a], ignore_index=True)
    weekly = weekly.merge(hist_matchups, on=["season", "week", "recent_team"], how="left")

    feature_frames = []
    for stat in FEATURE_COLUMNS:
        feat = build_feature_set(weekly, snaps, defense, stat)
        latest = feat.sort_values(["season", "week"]).groupby("player_id").tail(1)
        latest = latest.merge(
            this_week_matchups[["recent_team", "opponent_team"]],
            on="recent_team", how="inner", suffixes=("", "_new"),
        )
        latest["stat"] = stat
        feature_frames.append(latest)

    return pd.concat(feature_frames, ignore_index=True)


def predict_all(features: pd.DataFrame) -> pd.DataFrame:
    preds = []
    for stat in FEATURE_COLUMNS:
        model = PropModel.load(stat)
        stat_rows = features[features["stat"] == stat]
        pred = model.predict(stat_rows)
        preds.append(pred[["player_name", "stat", "predicted_mean", "predicted_std"]])
    return pd.concat(preds, ignore_index=True)


def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram credentials not set -- skipping send, printing instead.")
        print(message)
        return
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=15,
    )
    resp.raise_for_status()


def format_picks(picks: pd.DataFrame) -> str:
    if picks.empty:
        return "No qualifying edges found this week."
    lines = ["*NFL Prop Picks*\n"]
    for _, r in picks.head(20).iterrows():
        lines.append(
            f"{r['player_name']} — {r['side']} {r['line']} {r['stat'].replace('_', ' ')} "
            f"({r['bookmaker']}, {r['price']:+d}) | edge: {r['edge']*100:.1f}% | "
            f"kelly: {r['kelly_stake_pct']*100:.1f}%"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--min-edge", type=float, default=0.03)
    args = parser.parse_args()

    logger.info("Building features for season=%s week=%s", args.season, args.week)
    features = build_this_week_features(args.season, args.week)
    predictions = predict_all(features)

    logger.info("Fetching sportsbook props")
    odds = OddsClient().fetch_week_props()

    picks = find_edges(predictions, odds, min_edge_pct=args.min_edge)
    logger.info("Found %d qualifying edges", len(picks))

    send_telegram(format_picks(picks))
    picks.to_csv(f"cache/picks_{args.season}_wk{args.week}.csv", index=False)


if __name__ == "__main__":
    main()
