# NFL Prop Betting System

Automated NFL player-prop model: predicts passing/rushing/receiving yards
and receptions, compares predictions against live sportsbook lines, and
surfaces the edges via Telegram. A weekly-cadence sibling to the MLB picks
system, built simpler and on cleaner data:

| | MLB system | This system |
|---|---|---|
| Data source | pybaseball + custom FanDuel scraper (broke on SSL) | [nflverse](https://github.com/nflverse/nflverse-data) parquet files via `nfl_data_py` — no scraping |
| Cadence | Daily | Weekly |
| Feature pipeline | Pitch-level Statcast features, per-prop scripts | One shared rolling-average + opponent-matchup pipeline across all stat types |
| Odds | The Odds API | The Odds API (same provider/key) |

## How it works

1. **`scripts/train_models.py`** — trains one gradient-boosted model per
   prop category on trailing seasons of nflverse weekly data. Run once
   preseason, re-run anytime you want to fold in more games.
2. **`scripts/weekly_picks.py`** — the weekly job. Builds this week's
   features for every player, predicts a mean + std per prop, pulls live
   lines from The Odds API, computes model-implied probability vs. the
   book's implied probability (normal approximation), and sends any pick
   clearing the edge threshold to Telegram with a Kelly-fraction stake size.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in PARLAYAPI_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
python scripts/train_models.py --seasons 2019 2020 2021 2022 2023 2024
python scripts/weekly_picks.py --season 2026 --week 1
```

## Odds provider

Uses **[ParlayAPI](https://parlay-api.com)** — free tier includes 1,000
requests/month with NFL player props, no credit card required. (The Odds
API was the original plan since the MLB system already uses it, but its
free tier only covers game lines; player props there require a paid
Business plan at ~$99/mo. ParlayAPI's free tier includes props outright,
and advertises a drop-in-compatible response schema with The Odds API's
v4 API, which is why `odds_client.py`'s parsing barely differs from a
stock Odds API integration.)

> **Caution:** ParlayAPI is a newer, less-established provider — there's
> no long track record for uptime or data accuracy the way there is for
> The Odds API. Spot-check a handful of returned lines against an actual
> sportsbook before trusting it for real picks. The client is isolated
> in one file specifically so it's easy to swap providers later if it
> doesn't hold up.

## Automation

`.github/workflows/weekly_picks.yml` runs every Tuesday at 9am ET during
the season. Add `ODDS_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`
as repo secrets, or trigger manually via `workflow_dispatch` with a
specific season/week.

## Project layout

```
src/
  data_loader.py        nflverse data pulls (weekly stats, schedules, snaps, injuries)
  feature_engineering.py rolling averages + opponent-strength features
  props_model.py         model training/prediction per prop category
  odds_client.py         The Odds API wrapper
  edge_finder.py         model-vs-book probability comparison + Kelly sizing
scripts/
  train_models.py        preseason/periodic model training
  weekly_picks.py         weekly orchestration + Telegram delivery
models/                   saved trained models (gitignored)
cache/                    weekly picks output CSVs (gitignored)
```

## Notes on accuracy vs. the MLB system

- **Time-ordered cross-validation** is enforced in `props_model.py`
  (`TimeSeriesSplit`) so models are never validated on data that would
  leak future games into past predictions.
- **Opponent defensive strength** (yards/receptions allowed, season-to-date)
  is a first-class feature, not an afterthought — a much stronger signal
  in NFL than in MLB's batter-vs-pitcher matchups given far fewer
  data points per matchup.
- **Snap-share trend** captures role/opportunity changes (a player
  suddenly seeing more snaps) which often move a prop line before the
  box score does.
