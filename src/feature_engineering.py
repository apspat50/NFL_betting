"""
feature_engineering.py

Turns raw weekly stat lines into model-ready features:
  - rolling per-player averages (form)
  - season-to-date averages (baseline talent/role)
  - opponent defensive strength allowed in that stat category (matchup)
  - snap-share trend (opportunity / role signal)

Kept deliberately simple relative to the MLB system's pitch-level features --
NFL's weekly cadence means a handful of well-chosen rolling windows carries
most of the signal, without needing play-by-play complexity for a v1.
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOWS = (3, 5)

STAT_COLUMNS = {
    "passing_yards": "pass_yds_allowed",
    "rushing_yards": "rush_yds_allowed",
    "receiving_yards": "rec_yds_allowed",
    "receptions": "receptions_allowed",
}


def add_rolling_features(weekly: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Adds trailing rolling-average and season-to-date columns for `stat`,
    computed strictly from prior weeks (no leakage) via a shift(1)."""
    df = weekly.copy()
    grouped = df.groupby("player_id")[stat]

    df[f"{stat}_szn_avg"] = grouped.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    for w in ROLLING_WINDOWS:
        df[f"{stat}_r{w}"] = grouped.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
    df[f"{stat}_games_played"] = grouped.transform(
        lambda s: s.shift(1).expanding().count()
    )
    return df


def add_snap_share_trend(weekly: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Merges in trailing snap-share (offense_pct) as an opportunity signal.

    nflverse's snap-count data keys players by full display name (e.g.
    "Josh Allen"), while weekly stats' `player_name` is abbreviated
    ("J.Allen") -- must join on `player_display_name` instead, or every
    row silently comes back null.
    """
    snaps = snaps[["player", "season", "week", "team", "offense_pct"]].rename(
        columns={"player": "player_display_name", "team": "recent_team"}
    )
    df = weekly.merge(
        snaps, on=["player_display_name", "season", "week", "recent_team"], how="left"
    )
    df["snap_pct_r3"] = df.groupby("player_id")["offense_pct"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    return df


def add_opponent_context(weekly: pd.DataFrame, defense_allowed: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    Attaches how much of `stat` the upcoming opponent has allowed on
    average this season (through the prior week), as a matchup-strength
    feature. Requires an `opponent_team` column already on `weekly`.
    """
    allowed_col = STAT_COLUMNS[stat]
    defense = defense_allowed.sort_values(["season", "week"]).copy()
    defense[f"{allowed_col}_szn_avg"] = defense.groupby(["season", "team"])[
        allowed_col
    ].transform(lambda s: s.shift(1).expanding().mean())

    merged = weekly.merge(
        defense[["season", "week", "team", f"{allowed_col}_szn_avg"]],
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).drop(columns=["team"])
    return merged


def build_feature_set(
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    stat: str,
) -> pd.DataFrame:
    """Full feature pipeline for one target stat (e.g. 'passing_yards')."""
    df = add_rolling_features(weekly, stat)
    df = add_snap_share_trend(df, snaps)
    df = add_opponent_context(df, defense_allowed, stat)
    # Drop rows with no prior-game history -- can't predict a player's
    # week 1 debut off rolling averages, and that's fine for a props model.
    df = df[df[f"{stat}_games_played"] >= 1]
    return df


FEATURE_COLUMNS = {
    "passing_yards": [
        "passing_yards_szn_avg", "passing_yards_r3", "passing_yards_r5",
        "pass_yds_allowed_szn_avg", "snap_pct_r3",
    ],
    "rushing_yards": [
        "rushing_yards_szn_avg", "rushing_yards_r3", "rushing_yards_r5",
        "rush_yds_allowed_szn_avg", "snap_pct_r3",
    ],
    "receiving_yards": [
        "receiving_yards_szn_avg", "receiving_yards_r3", "receiving_yards_r5",
        "rec_yds_allowed_szn_avg", "snap_pct_r3",
    ],
    "receptions": [
        "receptions_szn_avg", "receptions_r3", "receptions_r5",
        "receptions_allowed_szn_avg", "snap_pct_r3",
    ],
}
