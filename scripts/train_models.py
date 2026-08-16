"""
train_models.py

Run once at the start of the season (and optionally re-run weekly as more
games accumulate). Trains and saves the prop models to models/.

Usage:
    python scripts/train_models.py --seasons 2019 2020 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_weekly_stats, load_snap_counts, team_defense_allowed
from props_model import train_all

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                         help="Seasons to train on, e.g. --seasons 2019 2020 2021 2022 2023 2024")
    args = parser.parse_args()
    seasons = tuple(args.seasons)

    weekly = load_weekly_stats(seasons)
    snaps = load_snap_counts(seasons)
    defense = team_defense_allowed(seasons)

    # weekly needs an opponent_team column for the opponent-context feature
    from data_loader import load_schedules
    sched = load_schedules(seasons)[["season", "week", "home_team", "away_team"]]
    home = sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    import pandas as pd
    matchups = pd.concat([home, away], ignore_index=True)
    weekly = weekly.merge(matchups, on=["season", "week", "recent_team"], how="left")

    train_all(weekly, snaps, defense)
    print(f"Trained and saved models for seasons {seasons}. See models/ directory.")


if __name__ == "__main__":
    main()
