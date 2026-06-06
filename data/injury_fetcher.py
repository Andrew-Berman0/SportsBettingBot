"""
data/injury_fetcher.py
----------------------
Fetches current injury reports from ESPN's unofficial API.
No API key required. Results cached per sport (2 hours by default).
"""

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SIGNIFICANT_STATUSES = {"Out", "Doubtful", "Questionable", "Day-To-Day"}

SPORT_ESPN_MAP = {
    "basketball_nba":       ("basketball", "nba"),
    "basketball_wnba":      ("basketball", "wnba"),
    "americanfootball_nfl": ("football",   "nfl"),
    "baseball_mlb":         ("baseball",   "mlb"),
    "icehockey_nhl":        ("hockey",     "nhl"),
}


class InjuryFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._cache: dict[str, dict[str, list]] = {}   # sport -> team -> injuries
        self._loaded_at: dict[str, float] = {}

    def get_team_injuries(self, team_name: str, sport: str = "basketball_nba",
                          max_age_minutes: int = 120) -> list[dict]:
        """
        Returns significant injuries for a team.
        Each entry: {"player": str, "status": str, "detail": str}
        Returns [] if team has no injuries (not an error).
        """
        age_minutes = (time.time() - self._loaded_at.get(sport, 0)) / 60
        if sport not in self._cache or age_minutes > max_age_minutes:
            self._fetch(sport)

        cache = self._cache.get(sport, {})
        name_lower = team_name.lower()
        if team_name in cache:
            return cache[team_name]
        for key in cache:
            if name_lower in key.lower() or key.lower() in name_lower:
                return cache[key]
        nickname = name_lower.split()[-1]
        for key in cache:
            if nickname in key.lower().split():
                return cache[key]
        return []

    def _fetch(self, sport: str):
        if sport not in SPORT_ESPN_MAP:
            self._cache[sport] = {}
            self._loaded_at[sport] = time.time()
            return

        league_sport, league = SPORT_ESPN_MAP[sport]
        url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/injuries"
        cache_file = CACHE_DIR / f"{league}_injuries.json"

        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            team_cache: dict[str, list] = {}
            for team_entry in data.get("injuries", []):
                team_display = team_entry.get("displayName", "")
                injuries = []
                for item in team_entry.get("injuries", []):
                    status = item.get("status", "")
                    if status not in SIGNIFICANT_STATUSES:
                        continue
                    player  = item.get("athlete", {}).get("displayName", "Unknown")
                    comment = item.get("shortComment", "")
                    injuries.append({"player": player, "status": status, "detail": comment})
                team_cache[team_display] = injuries
            self._cache[sport] = team_cache
            with open(cache_file, "w") as f:
                json.dump(team_cache, f)
            total = sum(len(v) for v in team_cache.values())
            logger.info(f"Fetched ESPN {league} injuries: {len(team_cache)} teams, {total} significant entries")
        except Exception as e:
            logger.warning(f"InjuryFetcher fetch error ({sport}): {e}")
            self._cache[sport] = {}
        finally:
            self._loaded_at[sport] = time.time()
