"""
backtest.py

Validates model accuracy against a real season the model never trained
on, and compares multiple underlying model types (gradient boosting,
random forest, ridge regression, histogram gradient boosting) so you can
see which one actually predicts best before committing to it for
production.

Also compares against a naive baseline (guessing the player's own
season average). If a model type can't beat that, it isn't adding value.

Usage:
    python scripts/backtest.py --train-seasons 2019 2020 2021 2022 2023 --holdout-season 2024
    python scripts/backtest.py --train-seasons 2019 2020 2021 2022 2023 --holdout-season 2024 --model-types gbr rf ridge hgb
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data_loader import (
    load_weekly_stats, load_snap_counts, load_schedules, load_injuries,
    team_defense_allowed, get_weather_features,
)
from feature_engineering import build_feature_set, FEATURE_COLUMNS
from props_model import PropModel, MODEL_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seasons", type=int, nargs="+", required=True)
    parser.add_argument("--holdout-season", type=int, required=True,
                         help="A season NOT in --train-seasons, used purely for evaluation.")
    parser.add_argument("--model-types", nargs="+", default=["gbr"], choices=MODEL_TYPES,
                         help="One or more model types to compare, e.g. --model-types gbr rf ridge hgb")
    args = parser.parse_args()

    if args.holdout_season in args.train_seasons:
        raise SystemExit("--holdout-season must not appear in --train-seasons "
                          "(that would let the model train on the data you're testing it against).")

    all_seasons = tuple(sorted(set(args.train_seasons) | {args.holdout_season}))
    weekly = load_weekly_stats(all_seasons)
    snaps = load_snap_counts(all_seasons)
    defense = team_defense_allowed(all_seasons)
    weather = get_weather_features(all_seasons)
    injuries = load_injuries(all_seasons)

    sched = load_schedules(all_seasons)[["season", "week", "home_team", "away_team"]]
    home = sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    matchups = pd.concat([home, away], ignore_index=True)
    weekly = weekly.merge(matchups, on=["season", "week", "recent_team"], how="left")

    print(f"\nTraining on seasons {args.train_seasons}, evaluating against "
          f"holdout season {args.holdout_season} (never seen during training)")
    print(f"Comparing model types: {args.model_types}\n")
    print("=" * 78)

    results = []
    for stat in FEATURE_COLUMNS:
        feat = build_feature_set(weekly, snaps, defense, weather, injuries, stat)
        train = feat[feat["season"].isin(args.train_seasons)]
        holdout = feat[feat["season"] == args.holdout_season]

        if train.empty or holdout.empty:
            logger.warning("Skipping %s -- insufficient data for train or holdout set.", stat)
            continue

        print(f"\n{stat}")
        naive_mae = None
        for model_type in args.model_types:
            model = PropModel(stat, model_type=model_type).fit(train)
            preds = model.predict(holdout)
            preds = preds.dropna(subset=["predicted_mean", stat, f"{stat}_szn_avg"])
            if preds.empty:
                print(f"  [{model_type}] no valid holdout predictions after filtering, skipping.")
                continue

            mae = mean_absolute_error(preds[stat], preds["predicted_mean"])
            r2 = r2_score(preds[stat], preds["predicted_mean"])
            if naive_mae is None:
                naive_mae = mean_absolute_error(preds[stat], preds[f"{stat}_szn_avg"])
            improvement = (naive_mae - mae) / naive_mae * 100 if naive_mae else 0.0

            print(f"  [{model_type:7s}] n={len(preds):5d}  MAE={mae:6.2f}  R^2={r2:6.3f}  "
                  f"vs naive({naive_mae:.2f}): {'BEATS' if mae < naive_mae else 'loses to'} "
                  f"({improvement:+.1f}%)")
            results.append({"stat": stat, "model_type": model_type, "mae": mae, "r2": r2})

    if len(args.model_types) > 1 and results:
        print("\n" + "=" * 78)
        print("Best model type per stat (lowest MAE):")
        df = pd.DataFrame(results)
        best = df.loc[df.groupby("stat")["mae"].idxmin()]
        for _, row in best.iterrows():
            print(f"  {row['stat']}: {row['model_type']} (MAE={row['mae']:.2f})")
        print("\nTrain production models with the winning type via:")
        print("  python scripts/train_models.py --seasons ... --model-type <best_type>")

    print("\n" + "=" * 78)
    print("A model that doesn't beat the naive baseline needs more features or\n"
          "training data before it's worth trusting for real picks. Beating the\n"
          "naive baseline is necessary but NOT sufficient to beat a real sportsbook\n"
          "line -- see scripts/backtest_odds.py for that harder test.")


if __name__ == "__main__":
    main()
