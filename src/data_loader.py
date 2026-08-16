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


def _load_per_year_safe(loader, seasons: tuple[int, ...], label: str) -> pd.DataFrame:
    """
    Calls an nfl_data_py loader one year at a time and skips any year
    that isn't published yet (404), instead of one missing year failing
    the entire multi-season load. This is what makes the pipeline keep
    working during preseason, when the current season has no data yet.
    """
    frames = []
    for year in seasons:
        try:
            frames.append(loader([year]))
        except Exception as e:
            logger.warning("%s: season %s not available yet, skipping: %s", label, year, e)
    if not frames:
        raise RuntimeError(f"No {label} data available for any of seasons={seasons}")
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=8)
def load_weekly_stats(seasons: tuple[int, ...]) -> pd.DataFrame:
    """
    Weekly player-level stat lines for the given seasons.
    Cached in-process since this is re-used across every prop model.
    """
    logger.info("Loading weekly stats for seasons=%s", seasons)
    df = _load_per_year_safe(
        lambda yrs: nfl.import_weekly_data(yrs, columns=WEEKLY_COLUMNS),
        seasons, "weekly stats",
    )
    df = df[df["season_type"] == "REG"].copy()
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


@lru_cache(maxsize=8)
def load_schedules(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Game-level schedule info: matchups, home/away, week, result.
    Comes from one combined file (all seasons + the full upcoming
    schedule, known in advance), so this doesn't need the per-year
    skip logic the stat endpoints need."""
    return nfl.import_schedules(list(seasons))


@lru_cache(maxsize=8)
def load_rosters(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly rosters -- used to confirm active status / team for a given week."""
    return _load_per_year_safe(nfl.import_weekly_rosters, seasons, "rosters")


@lru_cache(maxsize=8)
def load_snap_counts(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Snap-count share -- a strong proxy for role/opportunity."""
    return _load_per_year_safe(nfl.import_snap_counts, seasons, "snap counts")


@lru_cache(maxsize=8)
def load_injuries(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Weekly injury report designations."""
    return _load_per_year_safe(nfl.import_injuries, seasons, "injuries")


def get_current_week_slate(season: int) -> pd.DataFrame:
    """Upcoming games for the season -- used to know who plays whom this week."""
    sched = load_schedules((season,))
    upcoming = sched[sched["result"].isna()]
    return upcoming.sort_values(["week", "gameday"])


def get_current_season_and_week(today=None) -> tuple[int, int]:
    """
    Determines the current NFL season and week from today's date by
    checking the real schedule -- so the weekly job never needs a
    hardcoded season/week and keeps working automatically every year.

    "Current week" = the week containing the next game that hasn't
    happened yet on or after today.
    """
    import datetime as _dt

    today = today or _dt.date.today()
    # NFL seasons start in September; games in Jan/early Feb belong to
    # the previous year's season (e.g. Feb 2026 games are the 2025 season).
    season = today.year if today.month >= 8 else today.year - 1

    sched = load_schedules((season,))
    sched = sched.copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.date

    upcoming = sched[sched["gameday"] >= today]
    if upcoming.empty:
        # Season's over -- fall back to the last played week.
        return season, int(sched["week"].max())
    return season, int(upcoming.sort_values("gameday")["week"].iloc[0])


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
