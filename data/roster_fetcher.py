"""
data/roster_fetcher.py
----------------------
Fetches current NBA rosters from ESPN's unofficial API.
No API key required. Results cached for 6 hours.
Gives Claude an accurate current roster so trades/signings after its
training cutoff are reflected in analysis.
"""

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Same abbreviations used by InjuryFetcher
NBA_TEAM_ABBR = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}


class RosterFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_roster_string(self, team_name: str) -> str:
        """
        Returns a compact roster string for use in the Claude prompt.
        Format: "J. Tatum (F), J. Brown (G), N. Vucevic (C), ..."
        Returns empty string if team not found.
        """
        players = self._get_players(team_name)
        if not players:
            return "Not available"
        # Group by position for readability
        by_pos: dict[str, list[str]] = {}
        for p in players:
            pos = p["position"]
            by_pos.setdefault(pos, []).append(self._abbrev_name(p["name"]))
        parts = []
        for pos in ["PG", "SG", "G", "SF", "PF", "F", "C"]:
            if pos in by_pos:
                parts.append(", ".join(f"{n} ({pos})" for n in by_pos[pos]))
        # Any remaining positions
        for pos, names in by_pos.items():
            if pos not in ("PG", "SG", "G", "SF", "PF", "F", "C"):
                parts.append(", ".join(f"{n} ({pos})" for n in names))
        return ", ".join(parts)

    def _get_players(self, team_name: str) -> list[dict]:
        abbr = self._resolve_abbr(team_name)
        if not abbr:
            return []

        cache_file = CACHE_DIR / f"roster_{abbr}.json"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 6:
                with open(cache_file) as f:
                    return json.load(f)

        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{abbr}/roster"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            players = [
                {
                    "name":     a.get("displayName", ""),
                    "position": a.get("position", {}).get("abbreviation", "?"),
                }
                for a in data.get("athletes", [])
            ]
            with open(cache_file, "w") as f:
                json.dump(players, f)
            logger.info(f"Fetched roster for {team_name}: {len(players)} players")
            return players
        except Exception as e:
            logger.warning(f"Roster fetch error for {team_name} ({abbr}): {e}")
            return []

    @staticmethod
    def _abbrev_name(full_name: str) -> str:
        """'Jayson Tatum' → 'J. Tatum'"""
        parts = full_name.strip().split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return full_name

    def _resolve_abbr(self, team_name: str) -> str | None:
        if team_name in NBA_TEAM_ABBR:
            return NBA_TEAM_ABBR[team_name]
        name_lower = team_name.lower()
        for full, abbr in NBA_TEAM_ABBR.items():
            if name_lower in full.lower() or full.lower() in name_lower:
                return abbr
        # Nickname fallback (last word)
        nickname = name_lower.split()[-1]
        for full, abbr in NBA_TEAM_ABBR.items():
            if nickname in full.lower().split():
                return abbr
        logger.warning(f"RosterFetcher: no abbreviation for '{team_name}'")
        return None
