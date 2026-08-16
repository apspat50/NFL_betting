"""
backtest.py

Validates model accuracy against a real season the model never trained
on -- the closest thing to "how would this have done in a real season"
without waiting for actual games to happen.

Trains on --train-seasons, evaluates against --holdout-season (a season
NOT in the training set), and compares model error against a naive
baseline (just guessing the player's own season average). If the model
can't beat that naive baseline, it isn't adding value.

Usage:
    python scripts/backtest.py --train-seasons 2019 2020 2021 2022 2023 --holdout-season 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data_loader import load_weekly_stats, load_snap_counts, load_schedules, team_defense_allowed
from feature_engineering import build_feature_set, FEATURE_COLUMNS
from props_model import PropModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seasons", type=int, nargs="+", required=True)
    parser.add_argument("--holdout-season", type=int, required=True,
                         help="A season NOT in --train-seasons, used purely for evaluation.")
    args = parser.parse_args()

    if args.holdout_season in args.train_seasons:
        raise SystemExit("--holdout-season must not appear in --train-seasons "
                          "(that would let the model train on the data you're testing it against).")

    all_seasons = tuple(sorted(set(args.train_seasons) | {args.holdout_season}))
    weekly = load_weekly_stats(all_seasons)
    snaps = load_snap_counts(all_seasons)
    defense = team_defense_allowed(all_seasons)

    sched = load_schedules(all_seasons)[["season", "week", "home_team", "away_team"]]
    home = sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    matchups = pd.concat([home, away], ignore_index=True)
    weekly = weekly.merge(matchups, on=["season", "week", "recent_team"], how="left")

    print(f"\nTraining on seasons {args.train_seasons}, evaluating against "
          f"holdout season {args.holdout_season} (never seen during training)\n")
    print("=" * 70)

    for stat in FEATURE_COLUMNS:
        feat = build_feature_set(weekly, snaps, defense, stat)
        train = feat[feat["season"].isin(args.train_seasons)]
        holdout = feat[feat["season"] == args.holdout_season]

        if train.empty or holdout.empty:
            logger.warning("Skipping %s -- insufficient data for train or holdout set.", stat)
            continue

        model = PropModel(stat).fit(train)
        preds = model.predict(holdout)
        preds = preds.dropna(subset=["predicted_mean", stat, f"{stat}_szn_avg"])

        if preds.empty:
            logger.warning("Skipping %s -- no valid holdout predictions after filtering.", stat)
            continue

        mae = mean_absolute_error(preds[stat], preds["predicted_mean"])
        r2 = r2_score(preds[stat], preds["predicted_mean"])
        naive_mae = mean_absolute_error(preds[stat], preds[f"{stat}_szn_avg"])
        improvement_pct = (naive_mae - mae) / naive_mae * 100 if naive_mae else 0.0

        print(f"\n{stat} (n={len(preds)} holdout predictions)")
        print(f"  Model MAE:              {mae:.2f}")
        print(f"  Naive (season-avg) MAE: {naive_mae:.2f}")
        print(f"  R^2:                    {r2:.3f}")
        print(f"  Model {'BEATS' if mae < naive_mae else 'DOES NOT beat'} the naive baseline "
              f"({improvement_pct:+.1f}% {'improvement' if improvement_pct > 0 else 'worse'})")

    print("\n" + "=" * 70)
    print("A model that doesn't beat the naive baseline on most stats needs more\n"
          "features or more training data before it's worth trusting for real picks.")


if __name__ == "__main__":
    main()
