"""
data_loader.py

Pulls NFL player/team data from nflverse (via nfl_data_py). No scraping,
no SSL-fragile custom clients -- nflverse publishes clean parquet files
on GitHub releases and updates them within a day or two of games being
played. This is the single source of truth for all model inputs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import nfl_data_py as nfl
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEEKLY_COLUMNS = [
    "player_id", "player_name", "player_display_name", "position",
    "recent_team", "season", "week", "season_type",
    "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "sacks",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fantasy_points_ppr",
]


@lru_cache(maxsize=8)
def load_weekly_stats(seasons: tuple[int, ...]) -> pd.DataFrame:
    """
    Weekly player-level stat lines for the given seasons.
    Cached in-process since this is re-used across every prop model.
    """
    logger.info("Loading weekly stats for seasons=%s", seasons)
    df = nfl.import_weekly_data(list(seasons), columns=WEEKLY_COLUMNS)
    df = df[df["season_type"] == "REG"].copy()
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


@lru_cache(maxsize=8)
def load_schedules(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Game-level schedule info: matchups, home/away, week, result."""
    return nfl.import_schedules(list(seasons))


@lru_cache(maxsize=8)
def load_rosters(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly rosters -- used to confirm active status / team for a given week."""
    return nfl.import_weekly_rosters(list(seasons))


@lru_cache(maxsize=8)
def load_snap_counts(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Snap-count share -- a strong proxy for role/opportunity."""
    return nfl.import_snap_counts(list(seasons))


@lru_cache(maxsize=8)
def load_injuries(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly injury report designations."""
    return nfl.import_injuries(list(seasons))


def get_current_week_slate(season: int) -> pd.DataFrame:
    """Upcoming games for the season -- used to know who plays whom this week."""
    sched = load_schedules((season,))
    upcoming = sched[sched["result"].isna()]
    return upcoming.sort_values(["week", "gameday"])


def team_defense_allowed(seasons: tuple[int, ...]) -> pd.DataFrame:
    """
    Per-team, per-week yards/TDs allowed by category, derived from opponents'
    weekly stat lines. This is the core opponent-strength signal used in
    feature_engineering.py.
    """
    weekly = load_weekly_stats(seasons)
    sched = load_schedules(seasons)[["season", "week", "home_team", "away_team"]]

    # Map each player-week row to the opponent team.
    home = sched.rename(columns={"home_team": "recent_team", "away_team": "opponent_team"})
    away = sched.rename(columns={"away_team": "recent_team", "home_team": "opponent_team"})
    matchups = pd.concat([home, away], ignore_index=True)

    merged = weekly.merge(matchups, on=["season", "week", "recent_team"], how="left")

    allowed = (
        merged.groupby(["season", "week", "opponent_team"])
        .agg(
            pass_yds_allowed=("passing_yards", "sum"),
            rush_yds_allowed=("rushing_yards", "sum"),
            rec_yds_allowed=("receiving_yards", "sum"),
            receptions_allowed=("receptions", "sum"),
            pass_td_allowed=("passing_tds", "sum"),
            rush_td_allowed=("rushing_tds", "sum"),
        )
        .reset_index()
        .rename(columns={"opponent_team": "team"})
    )
    return allowed
