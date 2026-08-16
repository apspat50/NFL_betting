"""
debug_features.py

Diagnoses unexpected row-count drops in build_feature_set by printing
the row count after each pipeline step, so you can pinpoint exactly
which addition (weather, injuries, etc.) is dropping rows and why.

Usage:
    python scripts/debug_features.py --stat passing_yards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from data_loader import (
    load_weekly_stats, load_snap_counts, load_schedules, load_injuries,
    team_defense_allowed, get_weather_features,
)
from feature_engineering import (
    add_rolling_features, add_snap_share_trend, add_opponent_context,
    add_weather_features, add_injury_features, POSITION_ELIGIBLE, FEATURE_COLUMNS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat", default="passing_yards", choices=list(FEATURE_COLUMNS.keys()))
    parser.add_argument("--seasons", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024])
    args = parser.parse_args()
    stat = args.stat
    seasons = tuple(args.seasons)

    weekly = load_weekly_stats(seasons)
    snaps = load_snap_counts(seasons)
    defense = team_defense_allowed(seasons)
    weather = get_weather_features(seasons)
    injuries = load_injuries(seasons)

    print(f"\nRaw schedule weather coverage check:")
    print(f"  weather df shape: {weather.shape}")
    print(f"  weather df columns: {list(weather.columns)}")
    print(f"  temp non-null: {weather['temp'].notna().sum()} / {len(weather)}")
    print(f"  wind non-null: {weather['wind'].notna().sum()} / {len(weather)}")
    print(f"\nRaw injuries coverage check:")
    print(f"  injuries df shape: {injuries.shape}")
    print(f"  injuries df columns: {list(injuries.columns) if not injuries.empty else '(empty)'}")

    sched = load_schedules(seasons)[["season", "week", "home_team", "away_team"]]
    home = sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    matchups = pd.concat([home, away], ignore_index=True)
    weekly = weekly.merge(matchups, on=["season", "week", "recent_team"], how="left")

    df = weekly.copy()
    print(f"\n--- Pipeline for stat='{stat}' ---")
    print(f"start: {len(df)} rows")

    df = add_rolling_features(df, stat)
    print(f"after add_rolling_features: {len(df)} rows")

    df = add_snap_share_trend(df, snaps)
    print(f"after add_snap_share_trend: {len(df)} rows "
          f"(snap_pct_r3 non-null: {df['snap_pct_r3'].notna().sum()})")

    df = add_opponent_context(df, defense, stat)
    allowed_col = {"passing_yards": "pass_yds_allowed", "rushing_yards": "rush_yds_allowed",
                   "receiving_yards": "rec_yds_allowed", "receptions": "receptions_allowed"}[stat]
    print(f"after add_opponent_context: {len(df)} rows "
          f"({allowed_col}_szn_avg non-null: {df[f'{allowed_col}_szn_avg'].notna().sum()})")

    df = add_weather_features(df, weather)
    print(f"after add_weather_features: {len(df)} rows "
          f"(temp non-null: {df['temp'].notna().sum()}, wind non-null: {df['wind'].notna().sum()})")

    df = add_injury_features(df, injuries)
    print(f"after add_injury_features: {len(df)} rows "
          f"(injury_status_ordinal non-null: {df['injury_status_ordinal'].notna().sum()})")

    df = df[df[f"{stat}_games_played"] >= 1]
    print(f"after games_played>=1 filter: {len(df)} rows")

    df = df[df[f"{stat}_szn_avg"] > 0]
    print(f"after szn_avg>0 filter: {len(df)} rows")

    if "position" in df.columns and stat in POSITION_ELIGIBLE:
        df = df[df["position"].isin(POSITION_ELIGIBLE[stat])]
        print(f"after position filter ({POSITION_ELIGIBLE[stat]}): {len(df)} rows")

    print(f"\nFinal NaN count per feature column (would trigger dropna in PropModel.fit):")
    feats = FEATURE_COLUMNS[stat]
    print(df[feats].isna().sum())

    print(f"\nRows that survive dropna on ALL features + target: "
          f"{df.dropna(subset=feats + [stat]).shape[0]}")


if __name__ == "__main__":
    main()
