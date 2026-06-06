"""
data/roster_fetcher.py
----------------------
Fetches current rosters from ESPN's unofficial API.
No API key required. Results cached 6 hours per team.
"""

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPORT_ESPN_MAP = {
    "basketball_nba":       ("basketball", "nba"),
    "basketball_wnba":      ("basketball", "wnba"),
    "americanfootball_nfl": ("football",   "nfl"),
    "baseball_mlb":         ("baseball",   "mlb"),
    "icehockey_nhl":        ("hockey",     "nhl"),
}

# Priority positions shown per sport (keeps prompt concise for large rosters)
POSITION_PRIORITY = {
    "basketball_nba":       ["PG", "SG", "G", "SF", "PF", "F", "C"],
    "basketball_wnba":      ["PG", "SG", "G", "SF", "PF", "F", "C"],
    "americanfootball_nfl": ["QB", "RB", "WR", "TE", "K", "P"],
    "baseball_mlb":         ["SP", "RP", "CP", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"],
    "icehockey_nhl":        ["C", "LW", "RW", "D", "G"],
}

# NBA abbreviations kept for reliability — other sports resolved dynamically
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
        self._team_lookup: dict[str, dict[str, str]] = {}  # sport -> displayName -> abbr

    def get_roster_string(self, team_name: str, sport: str = "basketball_nba") -> str:
        """
        Returns a compact roster string for use in the Claude prompt.
        For NFL/MLB shows only key positions to keep the prompt manageable.
        """
        players = self._get_players(team_name, sport)
        if not players:
            return "Not available"

        priority = POSITION_PRIORITY.get(sport, [])
        by_pos: dict[str, list[str]] = {}
        for p in players:
            pos = p["position"]
            by_pos.setdefault(pos, []).append(self._abbrev_name(p["name"]))

        parts = []
        for pos in priority:
            if pos in by_pos:
                parts.append(", ".join(f"{n} ({pos})" for n in by_pos[pos]))

        # For NBA, also append any unlisted positions (full 15-man roster is small)
        if sport == "basketball_nba":
            for pos, names in by_pos.items():
                if pos not in priority:
                    parts.append(", ".join(f"{n} ({pos})" for n in names))

        return ", ".join(parts) or "Not available"

    def _get_players(self, team_name: str, sport: str) -> list[dict]:
        abbr = self._resolve_abbr(team_name, sport)
        if not abbr or sport not in SPORT_ESPN_MAP:
            return []

        league_sport, league = SPORT_ESPN_MAP[sport]
        cache_file = CACHE_DIR / f"roster_{league}_{abbr}.json"

        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 6:
                with open(cache_file) as f:
                    return json.load(f)

        url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/teams/{abbr}/roster"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            players = self._parse_athletes(data.get("athletes", []))
            with open(cache_file, "w") as f:
                json.dump(players, f)
            logger.info(f"Fetched roster for {team_name} ({sport}): {len(players)} players")
            return players
        except Exception as e:
            logger.warning(f"Roster fetch error for {team_name} ({abbr}, {sport}): {e}")
            return []

    @staticmethod
    def _parse_athletes(athletes_data) -> list[dict]:
        """
        Handles ESPN's two roster response shapes:
          - NBA: flat list of player dicts  (key: "displayName" at top level)
          - MLB/NFL/NHL: position-group dicts with a nested "items" list of player dicts
        """
        players = []
        if not isinstance(athletes_data, list):
            return players
        for a in athletes_data:
            if not isinstance(a, dict):
                continue
            if "items" in a:
                # Position-group format (MLB/NFL/NHL)
                for item in a.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    pos = item.get("position", {})
                    players.append({
                        "name":     item.get("displayName", ""),
                        "position": pos.get("abbreviation", "?") if isinstance(pos, dict) else "?",
                    })
            elif "displayName" in a:
                # Flat format (NBA)
                pos = a.get("position", {})
                players.append({
                    "name":     a.get("displayName", ""),
                    "position": pos.get("abbreviation", "?") if isinstance(pos, dict) else "?",
                })
        return players

    def _resolve_abbr(self, team_name: str, sport: str) -> str | None:
        if sport in ("basketball_nba",):
            if team_name in NBA_TEAM_ABBR:
                return NBA_TEAM_ABBR[team_name]
            name_lower = team_name.lower()
            for full, abbr in NBA_TEAM_ABBR.items():
                if name_lower in full.lower() or full.lower() in name_lower:
                    return abbr
            nickname = name_lower.split()[-1]
            for full, abbr in NBA_TEAM_ABBR.items():
                if nickname in full.lower().split():
                    return abbr
            logger.warning(f"RosterFetcher: no abbreviation for '{team_name}' (NBA)")
            return None
        return self._resolve_abbr_dynamic(team_name, sport)

    def _resolve_abbr_dynamic(self, team_name: str, sport: str) -> str | None:
        """Resolve team abbreviation via ESPN teams API, cached 7 days."""
        if sport not in SPORT_ESPN_MAP:
            return None
        league_sport, league = SPORT_ESPN_MAP[sport]

        if sport not in self._team_lookup:
            cache_file = CACHE_DIR / f"teams_{league}.json"
            if cache_file.exists():
                age_days = (time.time() - cache_file.stat().st_mtime) / 86400
                if age_days < 7:
                    with open(cache_file) as f:
                        self._team_lookup[sport] = json.load(f)

        if sport not in self._team_lookup:
            url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/teams?limit=100"
            try:
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                teams_map: dict[str, str] = {}
                for item in (data.get("sports") or [{}])[0].get("leagues", [{}])[0].get("teams", []):
                    team = item.get("team", {})
                    dn   = team.get("displayName", "")
                    abbr = team.get("abbreviation", "")
                    if dn and abbr:
                        teams_map[dn] = abbr
                self._team_lookup[sport] = teams_map
                with open(CACHE_DIR / f"teams_{league}.json", "w") as f:
                    json.dump(teams_map, f)
                logger.info(f"Fetched ESPN {league} team list: {len(teams_map)} teams")
            except Exception as e:
                logger.warning(f"ESPN teams lookup error ({league}): {e}")
                self._team_lookup[sport] = {}

        lookup = self._team_lookup.get(sport, {})
        name_lower = team_name.lower()
        if team_name in lookup:
            return lookup[team_name]
        for dn, abbr in lookup.items():
            if name_lower in dn.lower() or dn.lower() in name_lower:
                return abbr
        nickname = name_lower.split()[-1]
        for dn, abbr in lookup.items():
            if nickname in dn.lower().split():
                return abbr
        logger.warning(f"RosterFetcher: no abbreviation for '{team_name}' ({sport})")
        return None

    @staticmethod
    def _abbrev_name(full_name: str) -> str:
        parts = full_name.strip().split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return full_name
