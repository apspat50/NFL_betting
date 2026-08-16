"""
backtest_odds.py

The real test of whether this system would have beaten the book: for
each week of a past season, generate the exact picks the system would
have made, compare them against the ACTUAL closing sportsbook lines
from that week, and score whether the actual outcome would have won.

This is a fundamentally harder and more meaningful test than
scripts/backtest.py, which only checks prediction accuracy against a
naive baseline -- that tells you if the model beats a dumb guess, not
if it beats an efficient market.

** IMPORTANT CAVEAT **
This depends on ParlayAPI's historical player-props data actually being
available and populated for past NFL seasons. ParlayAPI's own
documentation only showed a worked example for game-line (h2h) historical
data, not player props specifically -- coverage for player props
specifically is unverified. This script is written defensively (skips
weeks/players with no data rather than crashing) and reports coverage
stats so you can see how much real data it actually found. If coverage
comes back near-zero, that's ParlayAPI not having the historical player
props it advertises, not a bug in this script -- consider it inconclusive
in that case rather than a real backtest result.

Usage:
    python scripts/backtest_odds.py --season 2024 --weeks 1 2 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from data_loader import (
    load_weekly_stats, load_snap_counts, load_schedules, load_injuries,
    team_defense_allowed, get_weather_features,
)
from feature_engineering import build_feature_set, build_td_feature_set, FEATURE_COLUMNS
from props_model import PropModel
from td_model import AnytimeTDModel
from odds_client import OddsClient
from edge_finder import find_edges, find_td_edges, american_to_decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def score_pick(row: pd.Series, actuals: pd.DataFrame) -> float | None:
    """
    Returns profit in units staked (e.g. 0.91 for a -110 win, -1.0 for a
    loss) for one pick, given the actual stat line that player recorded
    that week. Returns None if we don't have the actual result to check.
    """
    match = actuals[
        (actuals["player_name"] == row["player_name"])
        & (actuals["season"] == row["season"])
        & (actuals["week"] == row["week"])
    ]
    if match.empty:
        return None
    actual_row = match.iloc[0]

    decimal_odds = american_to_decimal(row["price"])

    if row["stat"] == "anytime_td":
        won = (actual_row.get("rushing_tds", 0) or 0) + (actual_row.get("receiving_tds", 0) or 0) > 0
    else:
        actual_value = actual_row.get(row["stat"])
        if pd.isna(actual_value) or pd.isna(row["line"]):
            return None
        won = actual_value > row["line"] if row["side"] == "Over" else actual_value < row["line"]

    return (decimal_odds - 1) if won else -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True,
                         help="Past season to backtest against (models train on all OTHER seasons).")
    parser.add_argument("--weeks", type=int, nargs="+", required=True,
                         help="Which weeks of --season to test, e.g. --weeks 1 2 3 4 5")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=None,
                         help="Seasons to train on. Defaults to 5 seasons before --season.")
    parser.add_argument("--min-edge", type=float, default=0.03)
    args = parser.parse_args()

    train_seasons = args.train_seasons or list(range(args.season - 5, args.season))
    if args.season in train_seasons:
        raise SystemExit("--season must not appear in --train-seasons.")

    all_seasons = tuple(sorted(set(train_seasons) | {args.season}))
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

    actuals = weekly[weekly["season"] == args.season][
        ["player_name", "season", "week", "passing_yards", "rushing_yards",
         "receiving_yards", "receptions", "rushing_tds", "receiving_tds"]
    ]

    print(f"\nTraining on {train_seasons}, backtesting real picks against actual "
          f"{args.season} closing lines for weeks {args.weeks}\n")

    # Train models on everything except the test season.
    train_weekly = weekly[weekly["season"].isin(train_seasons)]
    models = {}
    for stat in FEATURE_COLUMNS:
        feat = build_feature_set(train_weekly, snaps, defense, weather, injuries, stat)
        models[stat] = PropModel(stat).fit(feat)
    td_feat = build_td_feature_set(train_weekly, snaps, defense, weather, injuries)
    td_model = AnytimeTDModel().fit(td_feat)

    client = OddsClient()
    all_picks = []
    weeks_with_data = 0

    for week in args.weeks:
        week_dates = sched[(sched["season"] == args.season) & (sched["week"] == week)]
        if week_dates.empty:
            continue
        # Query historical odds once per distinct game date in that week
        # (a week can span Thu/Sun/Mon -- multiple calendar dates).
        for date_val in pd.to_datetime(week_dates.get("gameday", pd.Series(dtype=str))).dt.date.unique():
            odds = client.fetch_historical_closing_props(str(date_val))
            if odds.empty:
                continue
            weeks_with_data += 1

            week_feat = weekly[(weekly["season"] == args.season) & (weekly["week"] == week)]
            preds = []
            for stat in FEATURE_COLUMNS:
                feat = build_feature_set(weekly, snaps, defense, weather, injuries, stat)
                this_week = feat[(feat["season"] == args.season) & (feat["week"] == week)]
                if this_week.empty:
                    continue
                pred = models[stat].predict(this_week)
                pred["stat"] = stat
                preds.append(pred[["player_name", "stat", "predicted_mean", "predicted_std"]])
            if not preds:
                continue
            predictions = pd.concat(preds, ignore_index=True)

            td_this_week = build_td_feature_set(weekly, snaps, defense, weather, injuries)
            td_this_week = td_this_week[
                (td_this_week["season"] == args.season) & (td_this_week["week"] == week)
            ]
            td_predictions = (
                td_model.predict(td_this_week)[["player_name", "predicted_prob"]]
                if not td_this_week.empty else pd.DataFrame(columns=["player_name", "predicted_prob"])
            )

            yardage_picks = find_edges(predictions, odds, min_edge_pct=args.min_edge)
            td_picks = find_td_edges(td_predictions, odds, min_edge_pct=args.min_edge)
            picks = pd.concat([yardage_picks, td_picks], ignore_index=True)
            if picks.empty:
                continue
            picks["season"] = args.season
            picks["week"] = week
            all_picks.append(picks)

    print("=" * 78)
    print(f"Historical odds data found for {weeks_with_data} game-date(s) across "
          f"weeks {args.weeks}.")
    if weeks_with_data == 0:
        print("\nNo historical player-props data was returned by ParlayAPI for any of\n"
              "these dates. This means either: (a) ParlayAPI doesn't actually have\n"
              "historical NFL player props despite advertising a props archive, or\n"
              "(b) this specific season/date range isn't covered. This is NOT a\n"
              "result -- there's nothing to conclude here except 'inconclusive'.")
        return

    if not all_picks:
        print("\nHistorical odds data was found, but no picks cleared the edge\n"
              "threshold in any tested week. Try a lower --min-edge to see raw\n"
              "coverage, or this may genuinely mean no edges existed.")
        return

    picks = pd.concat(all_picks, ignore_index=True)
    picks["profit"] = picks.apply(lambda r: score_pick(r, actuals), axis=1)
    scored = picks.dropna(subset=["profit"])

    print(f"\nTotal picks generated: {len(picks)}")
    print(f"Picks with a verifiable actual result: {len(scored)}")
    if scored.empty:
        print("No picks could be matched to an actual result -- can't compute win rate.")
        return

    win_rate = (scored["profit"] > 0).mean()
    roi = scored["profit"].sum() / len(scored)
    print(f"\nWin rate: {win_rate*100:.1f}%")
    print(f"ROI per unit staked: {roi*100:+.1f}%")
    print(f"(Breakeven at standard -110 odds is ~52.4% win rate)")
    print("\nThis reflects a SMALL sample from limited historical data availability --")
    print("treat this as a directional signal, not a statistically reliable result,")
    print("until you've accumulated many more weeks of real picks (paper-traded or")
    print("backtested) to test against.")


if __name__ == "__main__":
    main()
