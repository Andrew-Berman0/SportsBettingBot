"""
data/injury_fetcher.py
----------------------
Fetches current NBA injury reports from ESPN's unofficial API.
No API key required. One request fetches all 30 teams; results cached 2 hours.
"""

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / "raw" / "nba_injuries.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

SIGNIFICANT_STATUSES = {"Out", "Doubtful", "Questionable", "Day-To-Day"}

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


class InjuryFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._cache: dict[str, list] = {}
        self._loaded_at: float = 0.0

    def get_team_injuries(self, team_name: str, max_age_minutes: int = 120) -> list[dict]:
        """
        Returns significant injuries for a team.
        Each entry: {"player": str, "status": str, "detail": str}
        Returns [] if team has no injuries (not an error).

        max_age_minutes: force-refresh if in-memory cache is older than this.
        Pass 30 for games within 6 hours to catch late scratches.
        """
        age_minutes = (time.time() - self._loaded_at) / 60
        if not self._cache or age_minutes > max_age_minutes:
            self._load()

        name_lower = team_name.lower()
        # 1. Exact match
        if team_name in self._cache:
            return self._cache[team_name]
        # 2. Substring match
        for key in self._cache:
            if name_lower in key.lower() or key.lower() in name_lower:
                return self._cache[key]
        # 3. Nickname match — last word of team name (e.g. "Clippers" matches "LA Clippers")
        nickname = name_lower.split()[-1]
        for key in self._cache:
            if nickname in key.lower().split():
                return self._cache[key]
        # Not found likely means no injuries (team not listed by ESPN)
        return []

    def _load(self):
        """Fetch from ESPN and update both file cache and in-memory cache."""
        self._fetch()
        self._loaded_at = time.time()

    def _fetch(self):
        try:
            resp = self.session.get(ESPN_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._cache = {}
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
                self._cache[team_display] = injuries
            with open(CACHE_FILE, "w") as f:
                json.dump(self._cache, f)
            total = sum(len(v) for v in self._cache.values())
            logger.info(f"Fetched ESPN NBA injuries: {len(self._cache)} teams, {total} significant entries")
        except Exception as e:
            logger.warning(f"InjuryFetcher fetch error: {e}")
            self._cache = {}
