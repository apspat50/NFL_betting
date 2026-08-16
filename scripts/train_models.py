"""
train_models.py

Run once at the start of the season (and optionally re-run weekly as more
games accumulate). Trains and saves the prop models (yardage/receptions)
plus the anytime-TD-scorer model to models/.

Usage:
    python scripts/train_models.py --seasons 2019 2020 2021 2022 2023 2024
    python scripts/train_models.py --seasons 2019 2020 2021 2022 2023 2024 --model-type rf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from data_loader import (
    load_weekly_stats, load_snap_counts, load_schedules, load_injuries,
    team_defense_allowed, get_weather_features,
)
from props_model import train_all, MODEL_TYPES
from td_model import train_td_model

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                         help="Seasons to train on, e.g. --seasons 2019 2020 2021 2022 2023 2024")
    parser.add_argument("--model-type", choices=list(MODEL_TYPES) + ["auto"], default="gbr",
                             help="Underlying regressor/classifier type, or 'auto' to use the "
                                  "empirically best type per stat (see props_model.BEST_MODEL_TYPES). "
                                  "Use scripts/backtest.py with --model-types to compare before "
                                  "picking one for production.")
    args = parser.parse_args()
    seasons = tuple(args.seasons)

    weekly = load_weekly_stats(seasons)
    snaps = load_snap_counts(seasons)
    defense = team_defense_allowed(seasons)
    weather = get_weather_features(seasons)
    injuries = load_injuries(seasons)

    # weekly needs an opponent_team column for the opponent-context feature
    sched = load_schedules(seasons)[["season", "week", "home_team", "away_team"]]
    home = sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    matchups = pd.concat([home, away], ignore_index=True)
    weekly = weekly.merge(matchups, on=["season", "week", "recent_team"], how="left")

    train_all(weekly, snaps, defense, weather, injuries, model_type=args.model_type)
    train_td_model(weekly, snaps, defense, weather, injuries, model_type=args.model_type)
    print(f"Trained and saved models (type={args.model_type}) for seasons {seasons}. "
          f"See models/ directory.")


if __name__ == "__main__":
    main()
