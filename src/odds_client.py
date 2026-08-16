"""
odds_client.py

Thin wrapper around ParlayAPI for NFL player-prop lines. Chosen over The
Odds API because ParlayAPI's free tier (1,000 requests/month, no card
required) actually includes player props -- The Odds API's free tier
covers game lines only and gates props behind a paid plan.

ParlayAPI advertises itself as drop-in compatible with The Odds API's v4
response shape (same JSON schema, same TOA-canonical market keys like
`player_pass_yds`), which is why this client's parsing logic barely
differs from a stock Odds API integration.

CAUTION: ParlayAPI is a newer, less-established provider than The Odds
API -- there's no long track record to point to for uptime or data
accuracy. Spot-check a handful of returned lines against an actual
sportsbook before trusting this for real picks, and treat this client
as swappable (that's why it's isolated behind one class) if it doesn't
hold up.

Env var required: PARLAYAPI_KEY
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://parlay-api.com/v1"
SPORT_KEY = "americanfootball_nfl"

# TOA-canonical market keys (ParlayAPI uses the same naming) -> our internal stat names
MARKET_MAP = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}


class OddsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("PARLAYAPI_KEY")
        if not self.api_key:
            raise RuntimeError("PARLAYAPI_KEY not set (env var or constructor arg)")
        self.headers = {"X-API-Key": self.api_key}

    def get_events(self) -> list[dict]:
        """Upcoming NFL games. Mainly useful for game metadata (kickoff
        time, home/away) -- props themselves come in one batched call."""
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT_KEY}/events",
            headers=self.headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def get_player_props(self, markets: list[str] | None = None) -> list[dict]:
        """
        Player props for the entire upcoming NFL slate in a single call --
        unlike The Odds API, ParlayAPI's /props endpoint isn't scoped to
        one event at a time, which is both simpler and cheaper on request
        budget.
        """
        markets = markets or list(MARKET_MAP.keys())
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT_KEY}/props",
            params={"markets": ",".join(markets), "regions": "us", "oddsFormat": "american"},
            headers=self.headers,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_week_props(self) -> "pd.DataFrame":  # noqa: F821 (lazy import below)
        """
        Pulls player-prop lines for every upcoming NFL game this week and
        flattens them into one DataFrame: one row per player/stat/book.
        """
        import pandas as pd

        rows = []
        for game in self.get_player_props():
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    stat = MARKET_MAP.get(market["key"])
                    if not stat:
                        continue
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "event_id": game.get("id"),
                            "commence_time": game.get("commence_time"),
                            "home_team": game.get("home_team"),
                            "away_team": game.get("away_team"),
                            "bookmaker": bookmaker["key"],
                            "stat": stat,
                            "player_name": outcome.get("description"),
                            "side": outcome.get("name"),  # "Over" / "Under"
                            "line": outcome.get("point"),
                            "price": outcome.get("price"),
                        })
        return pd.DataFrame(rows)

