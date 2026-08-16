"""
feature_engineering.py

Turns raw weekly stat lines into model-ready features:
  - rolling per-player averages (form)
  - season-to-date averages (baseline talent/role)
  - opponent defensive strength allowed in that stat category (matchup)
  - snap-share trend (opportunity / role signal)
  - weather (temperature, wind, dome/outdoor)
  - injury report status
  - position-based population filtering, so e.g. the passing_yards model
    is actually trained/evaluated on QBs, not the whole league

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
    "any_td": "total_td_allowed",
}

# Which positions can realistically produce each stat. Without this, a
# model "predicting passing_yards" is really being trained on every
# offensive player, ~90% of whom never throw a pass -- which makes both
# the model and a naive baseline look artificially accurate by coasting
# on easy zeros, instead of being tested on the population real prop
# lines actually exist for.
POSITION_ELIGIBLE = {
    "passing_yards": ["QB"],
    "rushing_yards": ["QB", "RB", "WR", "FB"],
    "receiving_yards": ["QB", "RB", "WR", "TE", "FB"],
    "receptions": ["QB", "RB", "WR", "TE", "FB"],
    "any_td": ["QB", "RB", "WR", "TE", "FB"],
}

INJURY_ORDINAL = {"Out": 3, "Doubtful": 2, "Questionable": 1}


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


def add_weather_features(weekly: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merges in game-level weather (temperature, wind, dome/outdoor) for
    the player's own team that week. Wind in particular has a known
    effect on passing volume/accuracy; dome games neutralize weather
    entirely, which is why `is_dome` is included as its own signal
    rather than relying on temp/wind alone (a 0mph "wind" reading could
    mean "calm outdoor day" or "played in a dome" -- very different
    situations that temp/wind alone can't distinguish).
    """
    if weather.empty:
        df = weekly.copy()
        df["temp"] = pd.NA
        df["wind"] = pd.NA
        df["is_dome"] = 0
        return df
    return weekly.merge(weather, on=["season", "week", "recent_team"], how="left")


def add_injury_features(weekly: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Merges in the player's own injury report status for that week
    (Out/Doubtful/Questionable -> ordinal severity, healthy/unlisted -> 0).

    Defensive by design: if the injuries dataset's schema doesn't match
    what's expected (column names occasionally shift across nflverse
    releases), this falls back to treating everyone as healthy rather
    than crashing the whole pipeline -- a missing feature is a much
    smaller problem than a broken pipeline during a live game week.
    """
    df = weekly.copy()
    if injuries is None or injuries.empty or "report_status" not in injuries.columns:
        df["injury_status_ordinal"] = 0
        return df

    key_col = "gsis_id" if "gsis_id" in injuries.columns else None
    if key_col is None or "player_id" not in df.columns:
        df["injury_status_ordinal"] = 0
        return df

    inj = injuries[["season", "week", key_col, "report_status"]].rename(
        columns={key_col: "player_id"}
    )
    inj["injury_status_ordinal"] = inj["report_status"].map(INJURY_ORDINAL).fillna(0)
    inj = inj.drop(columns=["report_status"]).drop_duplicates(
        subset=["season", "week", "player_id"]
    )
    merged = df.merge(inj, on=["season", "week", "player_id"], how="left")
    merged["injury_status_ordinal"] = merged["injury_status_ordinal"].fillna(0)
    return merged


def build_feature_set(
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    weather: pd.DataFrame,
    injuries: pd.DataFrame,
    stat: str,
) -> pd.DataFrame:
    """Full feature pipeline for one target yardage/reception stat
    (e.g. 'passing_yards')."""
    df = add_rolling_features(weekly, stat)
    df = add_snap_share_trend(df, snaps)
    df = add_opponent_context(df, defense_allowed, stat)
    df = add_weather_features(df, weather)
    df = add_injury_features(df, injuries)

    # Drop rows with no prior-game history -- can't predict a player's
    # week 1 debut off rolling averages, and that's fine for a props model.
    df = df[df[f"{stat}_games_played"] >= 1]
    # Drop players with no real history of this stat.
    df = df[df[f"{stat}_szn_avg"] > 0]
    if "position" in df.columns and stat in POSITION_ELIGIBLE:
        df = df[df["position"].isin(POSITION_ELIGIBLE[stat])]
    return df


def build_td_feature_set(
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    weather: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feature pipeline for the anytime-touchdown-scorer target: whether a
    player scored ANY touchdown (rushing or receiving -- not passing,
    since anytime-TD-scorer props are about the player scoring
    themselves) that game. Reuses the same rolling/opponent/weather/
    injury infrastructure as the yardage stats, just on a binary target
    instead of a continuous one.
    """
    df = weekly.copy()
    df["any_td"] = (
        df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)
    ).gt(0).astype(int)

    df = add_rolling_features(df, "any_td")
    df = add_snap_share_trend(df, snaps)
    df = add_opponent_context(df, defense_allowed, "any_td")
    df = add_weather_features(df, weather)
    df = add_injury_features(df, injuries)

    df = df[df["any_td_games_played"] >= 1]
    if "position" in df.columns:
        df = df[df["position"].isin(POSITION_ELIGIBLE["any_td"])]
    return df


FEATURE_COLUMNS = {
    "passing_yards": [
        "passing_yards_szn_avg", "passing_yards_r3", "passing_yards_r5",
        "pass_yds_allowed_szn_avg", "snap_pct_r3",
        "temp", "wind", "is_dome", "injury_status_ordinal",
    ],
    "rushing_yards": [
        "rushing_yards_szn_avg", "rushing_yards_r3", "rushing_yards_r5",
        "rush_yds_allowed_szn_avg", "snap_pct_r3",
        "temp", "wind", "is_dome", "injury_status_ordinal",
    ],
    "receiving_yards": [
        "receiving_yards_szn_avg", "receiving_yards_r3", "receiving_yards_r5",
        "rec_yds_allowed_szn_avg", "snap_pct_r3",
        "temp", "wind", "is_dome", "injury_status_ordinal",
    ],
    "receptions": [
        "receptions_szn_avg", "receptions_r3", "receptions_r5",
        "receptions_allowed_szn_avg", "snap_pct_r3",
        "temp", "wind", "is_dome", "injury_status_ordinal",
    ],
}

TD_FEATURE_COLUMNS = [
    "any_td_szn_avg", "any_td_r3", "any_td_r5",
    "total_td_allowed_szn_avg", "snap_pct_r3",
    "temp", "wind", "is_dome", "injury_status_ordinal",
]
