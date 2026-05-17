"""
data/odds_fetcher.py
--------------------
Fetches live and upcoming odds from The Odds API.
Free tier: 500 requests/month. We cache aggressively to stay within limits.

Get a free key at: https://the-odds-api.com
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Imported lazily to avoid circular imports
_action_network_fetcher = None


def _get_action_network():
    global _action_network_fetcher
    if _action_network_fetcher is None:
        from SportsBettingBot.data.action_network_fetcher import ActionNetworkFetcher
        _action_network_fetcher = ActionNetworkFetcher()
    return _action_network_fetcher

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"


class OddsFetcher:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_upcoming_games(self, sport: str, bookmakers: list[str]) -> list[dict]:
        """
        Returns upcoming games with odds for a given sport.
        sport examples: 'basketball_nba', 'americanfootball_nfl', 'baseball_mlb', 'icehockey_nhl'

        Cache strategy (saves API quota):
          - Cache < 90 min old  → always use it, no API call
          - Cache 90 min–6 hrs  → peek at cached commence_times; only call if a game
                                   is within 4 hours (odds are moving) or no game data exists
          - Cache > 6 hrs old   → always refresh (pick up newly scheduled games)
          - No cache            → always call
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
                        f"Odds cache ({age_minutes:.0f}min old), no game within 4h — skipping API call for {sport}"
                    )
                    return cached

        if not self.api_key or self.api_key == "your_odds_api_key_here":
            logger.warning("No Odds API key set — returning empty game list")
            return []

        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey":     self.api_key,
            "regions":    "us",
            "markets":    "h2h,totals",
            "bookmakers": ",".join(bookmakers),
            "oddsFormat": "american",
        }

        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code in (401, 422, 429):
                logger.warning(
                    f"Odds API quota/auth error ({resp.status_code}) — "
                    f"falling back to ActionNetwork for {sport}"
                )
                return _get_action_network().get_upcoming_games(sport)
            resp.raise_for_status()
            games = resp.json()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info(f"Fetched {len(games)} games for {sport} (API requests remaining: {remaining})")
            with open(cache_file, "w") as f:
                json.dump(games, f)
            return games
        except Exception as e:
            logger.error(f"Odds API error for {sport}: {e} — falling back to ActionNetwork")
            return _get_action_network().get_upcoming_games(sport)

    def get_active_sport(self, sports: list[str], bookmakers: list[str]) -> tuple[str | None, list[dict]]:
        """
        Checks each sport in order and returns the first one with upcoming games.
        Respects API quota by checking cache first.
        """
        for sport in sports:
            games = self.get_upcoming_games(sport, bookmakers)
            if games:
                logger.info(f"Active sport: {sport} ({len(games)} upcoming games)")
                return sport, games
        logger.warning("No active sports found with upcoming games")
        return None, []

    @staticmethod
    def _game_approaching(games: list[dict], now: datetime, hours: int = 4) -> bool:
        """Returns True if any game in the cached list starts within `hours` from now."""
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
        """
        Extracts a clean game record from raw Odds API response.
        Returns None if odds are missing.
        """
        try:
            home = game["home_team"]
            away = game["away_team"]
            commence = game["commence_time"]

            h2h_odds   = {}
            total_line = None
            over_odds  = None
            under_odds = None

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h" and not h2h_odds:
                        for outcome in market["outcomes"]:
                            h2h_odds[outcome["name"]] = outcome["price"]
                    if market["key"] == "totals" and total_line is None:
                        for outcome in market["outcomes"]:
                            if outcome["name"] == "Over":
                                total_line = outcome.get("point")
                                over_odds  = outcome["price"]
                            elif outcome["name"] == "Under":
                                under_odds = outcome["price"]

            if not h2h_odds:
                return None

            home_ml = h2h_odds.get(home)
            away_ml = h2h_odds.get(away)

            return {
                "game_id":      game["id"],
                "sport":        game["sport_key"],
                "home_team":    home,
                "away_team":    away,
                "commence_time": commence,
                "home_ml":      home_ml,
                "away_ml":      away_ml,
                "total_line":   total_line,
                "over_odds":    over_odds,
                "under_odds":   under_odds,
                "home_implied": OddsFetcher.american_to_implied(home_ml) if home_ml else None,
                "away_implied": OddsFetcher.american_to_implied(away_ml) if away_ml else None,
            }
        except Exception as e:
            logger.warning(f"Failed to parse game: {e}")
            return None

    @staticmethod
    def american_to_implied(american_odds: float) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def implied_to_american(prob: float) -> float:
        """Convert probability to American odds."""
        if prob >= 0.5:
            return -(prob / (1 - prob)) * 100
        else:
            return ((1 - prob) / prob) * 100
