"""
weekly_picks.py

The main orchestrator -- runs multiple times per week, once before each
NFL game night (Thursday, Sunday, Monday), via GitHub Actions or
manually.

1. Loads current-season data through last week.
2. Builds this week's feature rows for every player likely to play.
3. Loads trained models and predicts mean/std per stat.
4. Pulls live sportsbook lines via ParlayAPI.
5. Filters picks down to only the game(s) happening TODAY, so a
   Thursday run only sends the Thursday Night Football game, a Sunday
   run sends the full Sunday slate, and a Monday run sends Monday
   Night Football -- one script, three schedules, no separate logic
   needed per game night.
6. Finds edges and sends picks to Telegram.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
Path("cache").mkdir(exist_ok=True)

load_dotenv()

from data_loader import (
    load_weekly_stats, load_snap_counts, load_schedules, team_defense_allowed,
    get_current_season_and_week,
)
from feature_engineering import build_feature_set, FEATURE_COLUMNS
from props_model import PropModel
from odds_client import OddsClient
from edge_finder import find_edges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback if tzdata isn't installed
    EASTERN = None


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


def filter_to_game_day(odds: pd.DataFrame, target_date: dt.date) -> pd.DataFrame:
    """
    Keeps only props for games whose kickoff falls on `target_date`
    (Eastern time, since that's how NFL game nights are actually
    scheduled/discussed -- a game that shows as the next UTC day could
    still be "tonight" locally).
    """
    if odds.empty:
        return odds
    commence = pd.to_datetime(odds["commence_time"], utc=True)
    if EASTERN is not None:
        commence = commence.dt.tz_convert(EASTERN)
    local_date = commence.dt.date
    return odds[local_date == target_date].copy()


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


def format_picks(picks: pd.DataFrame, game_day_label: str) -> str:
    if picks.empty:
        return f"No qualifying edges found for {game_day_label}."
    lines = [f"*NFL Prop Picks — {game_day_label}*\n"]
    for _, r in picks.head(20).iterrows():
        lines.append(
            f"{r['player_name']} — {r['side']} {r['line']} {r['stat'].replace('_', ' ')} "
            f"({r['bookmaker']}, {r['price']:+d}) | edge: {r['edge']*100:.1f}% | "
            f"kelly: {r['kelly_stake_pct']*100:.1f}%"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None,
                         help="Defaults to auto-detected current season if omitted.")
    parser.add_argument("--week", type=int, default=None,
                         help="Defaults to auto-detected current week if omitted.")
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument(
        "--game-date", type=str, default=None,
        help="YYYY-MM-DD to filter picks to a specific game day (Eastern time). "
             "Defaults to today -- this is what makes the Thu/Sun/Mon schedule work "
             "without separate config per run.",
    )
    args = parser.parse_args()

    season, week = args.season, args.week
    if season is None or week is None:
        auto_season, auto_week = get_current_season_and_week()
        season = season or auto_season
        week = week or auto_week
        logger.info("Auto-detected season=%s week=%s", season, week)

    target_date = (
        dt.datetime.strptime(args.game_date, "%Y-%m-%d").date()
        if args.game_date else dt.date.today()
    )

    logger.info("Building features for season=%s week=%s", season, week)
    features = build_this_week_features(season, week)
    predictions = predict_all(features)

    logger.info("Fetching sportsbook props")
    odds = OddsClient().fetch_week_props()
    odds = filter_to_game_day(odds, target_date)
    logger.info("%d prop rows remain after filtering to %s", len(odds), target_date)

    if odds.empty:
        logger.info("No games scheduled for %s -- nothing to compare against.", target_date)
        picks = pd.DataFrame()
    else:
        picks = find_edges(predictions, odds, min_edge_pct=args.min_edge)
    logger.info("Found %d qualifying edges", len(picks))

    game_day_label = target_date.strftime("%A %b %d")
    send_telegram(format_picks(picks, game_day_label))
    picks.to_csv(f"cache/picks_{season}_wk{week}_{target_date}.csv", index=False)


if __name__ == "__main__":
    main()
