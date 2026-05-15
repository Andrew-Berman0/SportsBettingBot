"""
data/action_network_fetcher.py
------------------------------
Fallback odds source using ActionNetwork's public scoreboard API.
No API key required. Used when The Odds API quota is exhausted.

Returns game dicts in the same format as OddsFetcher.parse_game() output
so the rest of the pipeline needs no changes.
"""

import logging

import requests

logger = logging.getLogger(__name__)

SCOREBOARD_URL = "https://api.actionnetwork.com/web/v1/scoreboard/{league}"

LEAGUE_MAP = {
    "basketball_nba":      "nba",
    "americanfootball_nfl": "nfl",
    "baseball_mlb":        "mlb",
    "icehockey_nhl":       "nhl",
}

# Preferred book IDs (ActionNetwork internal IDs for major US sportsbooks)
# We pick the first match in this order; fall back to any book with valid lines.
PREFERRED_BOOK_IDS = [
    69,   # FanDuel NJ
    75,   # BetMGM NJ
    1997, # DraftKings AS
    30,   # Open (consensus opening line)
]


class ActionNetworkFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept":     "application/json",
        })

    def get_upcoming_games(self, sport: str) -> list[dict]:
        """
        Returns upcoming games with odds as pre-parsed dicts
        (same keys as OddsFetcher.parse_game output, plus _pre_parsed=True).
        """
        league = LEAGUE_MAP.get(sport)
        if not league:
            logger.warning(f"ActionNetwork: no league mapping for {sport}")
            return []

        raw = self._fetch(league)
        games = []
        for g in raw:
            parsed = self._parse_game(g, sport)
            if parsed:
                games.append(parsed)

        logger.info(f"ActionNetwork: {len(games)} games for {sport}")
        return games

    def _fetch(self, league: str) -> list[dict]:
        # No date param — ActionNetwork returns today's games by default
        url = SCOREBOARD_URL.format(league=league)
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("games", [])
        except Exception as e:
            logger.warning(f"ActionNetwork fetch error ({league}): {e}")
            return []

    def _parse_game(self, g: dict, sport: str) -> dict | None:
        try:
            # Only process games that haven't started yet
            if g.get("status") != "scheduled":
                return None

            teams_by_id = {t["id"]: t for t in g.get("teams", [])}
            home_id = g.get("home_team_id")
            away_id = g.get("away_team_id")

            home_team = teams_by_id.get(home_id, {}).get("full_name")
            away_team = teams_by_id.get(away_id, {}).get("full_name")

            if not home_team or not away_team:
                return None

            # Normalize start_time: strip sub-second precision so fromisoformat
            # handles "2026-05-15T23:00:00.000Z" the same as "2026-05-15T23:00:00Z"
            raw_start = g.get("start_time") or ""
            if "." in raw_start and raw_start.endswith("Z"):
                raw_start = raw_start[:raw_start.index(".")] + "Z"
            if not raw_start:
                return None

            home_ml = away_ml = total_line = over_odds = under_odds = None

            odds_list = g.get("odds", [])
            # Try preferred books in order, fall back to first valid entry
            candidates = {o["book_id"]: o for o in odds_list if o.get("book_id") is not None}
            chosen = None
            for bid in PREFERRED_BOOK_IDS:
                if bid in candidates and candidates[bid].get("ml_home") is not None:
                    chosen = candidates[bid]
                    break
            if chosen is None:
                for o in odds_list:
                    if o.get("ml_home") is not None and o.get("ml_away") is not None:
                        chosen = o
                        break

            if chosen is None:
                return None

            home_ml    = chosen["ml_home"]
            away_ml    = chosen["ml_away"]
            total_line = chosen.get("total")
            over_odds  = chosen.get("over")
            under_odds = chosen.get("under")

            return {
                "game_id":      f"an_{g['id']}",
                "sport":        sport,
                "home_team":    home_team,
                "away_team":    away_team,
                "commence_time": raw_start,
                "home_ml":      home_ml,
                "away_ml":      away_ml,
                "total_line":   total_line,
                "over_odds":    over_odds,
                "under_odds":   under_odds,
                "home_implied": self._to_implied(home_ml),
                "away_implied": self._to_implied(away_ml),
                "_pre_parsed":  True,
            }
        except Exception as e:
            logger.warning(f"ActionNetwork parse error: {e}")
            return None

    @staticmethod
    def _to_implied(odds: float | None) -> float | None:
        if odds is None:
            return None
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)
