"""
data/odds_fetcher.py
--------------------
Fetches upcoming odds from TheRundown API.
Free tier: 20,000 requests/day.

Get a free key at: https://therundown.io
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://therundown.io/api/v1"

# Regular season + playoff IDs per sport (TheRundown splits them unlike The Odds API)
SPORT_IDS = {
    "basketball_nba":       [4, 24],   # NBA regular season + NBA Playoffs
    "basketball_wnba":      [8],
    "americanfootball_nfl": [2, 26],   # NFL regular season + NFL Playoffs
    "baseball_mlb":         [3, 31],   # MLB regular season + MLB Playoffs
    "icehockey_nhl":        [6, 28],   # NHL regular season + NHL Playoffs
}


class OddsFetcher:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self._auth_params = {"key": self.api_key}

    def get_upcoming_games(self, sport: str, bookmakers: list[str]) -> list[dict]:
        """
        Returns upcoming pre-parsed game dicts for a given sport.
        `bookmakers` param kept for interface compatibility but not used —
        TheRundown returns all available lines in one call.

        Cache strategy:
          - Cache < 90 min old  → always use it
          - Cache 90 min–6 hrs  → only refresh if a game is within 4h
          - Cache > 6 hrs old   → always refresh
        """
        cache_file = CACHE_DIR / f"odds_{sport}.json"
        now = datetime.now(timezone.utc)

        if cache_file.exists():
            age_minutes = (time.time() - cache_file.stat().st_mtime) / 60

            if age_minutes < 90:
                logger.info(f"Odds cache fresh ({age_minutes:.0f}min old) — {sport}")
                with open(cache_file) as f:
                    return json.load(f)

            if age_minutes < 360:
                with open(cache_file) as f:
                    cached = json.load(f)
                if not self._game_approaching(cached, now, hours=4):
                    logger.info(
                        f"Odds cache ({age_minutes:.0f}min old), no game within 4h"
                        f" — skipping API call for {sport}"
                    )
                    return cached

        if not self.api_key:
            logger.warning("No Rundown API key set — returning empty game list")
            return []

        sport_ids = SPORT_IDS.get(sport)
        if not sport_ids:
            logger.warning(f"Unknown sport key: {sport}")
            return []

        # Fetch today and tomorrow (UTC) across all sport IDs (regular season + playoffs).
        # Sleep 1.1s between requests to respect the free tier's 1 req/sec rate limit.
        raw_events: list[dict] = []
        for sport_id in sport_ids:
            for delta_days in (0, 1):
                date_str = (now + timedelta(days=delta_days)).strftime("%Y-%m-%d")
                url = f"{BASE_URL}/sports/{sport_id}/events/{date_str}"
                try:
                    resp = self.session.get(url, params=self._auth_params, timeout=10)
                    resp.raise_for_status()
                    raw_events.extend(resp.json().get("events", []))
                except Exception as e:
                    logger.error(f"Rundown API error for {sport} {date_str}: {e}")
                time.sleep(1.1)

        if not raw_events:
            logger.info(f"No events returned for {sport}")
            return []

        parsed, seen = [], set()
        for event in raw_events:
            game = self._parse_event(event, sport)
            if game and game["game_id"] not in seen:
                seen.add(game["game_id"])
                parsed.append(game)

        logger.info(f"Fetched {len(parsed)} games for {sport}")
        with open(cache_file, "w") as f:
            json.dump(parsed, f)
        return parsed

    def _parse_event(self, event: dict, sport_key: str) -> dict | None:
        """Convert a TheRundown event to the normalized game dict the bot expects."""
        try:
            event_id = event["event_id"]

            teams = event.get("teams_normalized") or event.get("teams", [])
            home_team = away_team = None
            for t in teams:
                if t.get("is_home"):
                    home_team = t["name"]
                else:
                    away_team = t["name"]

            if not home_team or not away_team:
                return None

            home_ml = away_ml = None
            total_line = over_odds = under_odds = None

            for aff_data in event.get("lines", {}).values():
                if home_ml is None:
                    ml = aff_data.get("moneyline", {})
                    h = ml.get("moneyline_home")
                    a = ml.get("moneyline_away")
                    # Real American odds always have abs >= 100; 0.0001 is TheRundown's "no line" sentinel
                    if h and a and abs(h) >= 100 and abs(a) >= 100:
                        home_ml = h
                        away_ml = a

                if total_line is None:
                    tot = aff_data.get("total", {})
                    tl = tot.get("total_over")
                    if tl:
                        total_line = tl
                        over_odds  = tot.get("total_over_money")
                        under_odds = tot.get("total_under_money")

                if home_ml is not None and total_line is not None:
                    break

            if home_ml is None:
                return None

            return {
                "_pre_parsed":   True,
                "game_id":       event_id,
                "sport":         sport_key,
                "home_team":     home_team,
                "away_team":     away_team,
                "commence_time": event.get("event_date", ""),
                "home_ml":       home_ml,
                "away_ml":       away_ml,
                "total_line":    total_line,
                "over_odds":     over_odds,
                "under_odds":    under_odds,
                "home_implied":  OddsFetcher.american_to_implied(home_ml),
                "away_implied":  OddsFetcher.american_to_implied(away_ml),
            }
        except Exception as e:
            logger.warning(f"Failed to parse Rundown event: {e}")
            return None

    def get_active_sport(self, sports: list[str], bookmakers: list[str]) -> tuple[str | None, list[dict]]:
        for sport in sports:
            games = self.get_upcoming_games(sport, bookmakers)
            if games:
                logger.info(f"Active sport: {sport} ({len(games)} upcoming games)")
                return sport, games
        logger.warning("No active sports found with upcoming games")
        return None, []

    @staticmethod
    def _game_approaching(games: list[dict], now: datetime, hours: int = 4) -> bool:
        cutoff = now + timedelta(hours=hours)
        for g in games:
            ct = g.get("commence_time") or g.get("start_time")
            if not ct:
                continue
            try:
                start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if now <= start <= cutoff:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def parse_game(game: dict) -> dict | None:
        """Pass-through for pre-parsed TheRundown games."""
        return game if game.get("_pre_parsed") else None

    @staticmethod
    def american_to_implied(american_odds: float) -> float:
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def implied_to_american(prob: float) -> float:
        if prob >= 0.5:
            return -(prob / (1 - prob)) * 100
        else:
            return ((1 - prob) / prob) * 100
