"""
data/stats_fetcher.py
---------------------
Fetches team stats for NBA (and other sports) to build features.

NBA: uses the free nba_api package (official NBA stats endpoint)
NFL/MLB/NHL: ESPN unofficial API (free, no key required)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class NBAStatsFetcher:
    """
    Fetches NBA team stats via nba_api.
    Covers: offensive/defensive rating, pace, net rating, recent form,
    back-to-back indicator, rest days, home/away splits.
    """

    def __init__(self):
        try:
            from nba_api.stats.endpoints import leaguedashteamstats, teamgamelogs
            from nba_api.stats.static import teams
            self._leaguedash   = leaguedashteamstats
            self._teamgamelogs = teamgamelogs
            self._teams        = teams
            self._available    = True
        except ImportError:
            logger.warning("nba_api not installed — run: pip install nba_api")
            self._available = False

    def get_team_stats(self, season: str = "2024-25") -> pd.DataFrame:
        """
        Returns a DataFrame of team stats for the season.
        Columns include: TEAM_ID, TEAM_NAME, W_PCT, OFF_RATING, DEF_RATING,
                         NET_RATING, PACE, TS_PCT, etc.
        """
        if not self._available:
            return pd.DataFrame()

        cache_file = CACHE_DIR / f"nba_team_stats_{season}.parquet"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 6:
                return pd.read_parquet(cache_file)

        try:
            time.sleep(0.6)   # NBA API rate limit
            dash = self._leaguedash.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            )
            df = dash.get_data_frames()[0]
            df.to_parquet(cache_file)
            logger.info(f"Fetched NBA team stats: {len(df)} teams ({season})")
            return df
        except Exception as e:
            logger.error(f"NBA team stats fetch error: {e}")
            return pd.DataFrame()

    def get_recent_form(self, team_id: int, n_games: int = 10, season: str = "2024-25") -> dict:
        """
        Returns recent form stats for a team: win%, avg point diff, back-to-back flag.
        """
        if not self._available:
            return {}

        try:
            time.sleep(0.6)
            logs = self._teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                last_n_games_nullable=n_games,
            )
            df = logs.get_data_frames()[0]
            if df.empty:
                return {}

            df = df.sort_values("GAME_DATE", ascending=False).head(n_games)
            wins    = (df["WL"] == "W").sum()
            avg_diff = df["PLUS_MINUS"].mean()

            # Back-to-back: last game was yesterday
            last_game_date = pd.to_datetime(df["GAME_DATE"].iloc[0])
            is_b2b = (datetime.today() - last_game_date).days <= 1

            # Rest days since last game
            rest_days = (datetime.today() - last_game_date).days

            return {
                f"win_pct_l{n_games}":    wins / n_games,
                f"avg_diff_l{n_games}":   float(avg_diff),
                "is_back_to_back":         int(is_b2b),
                "rest_days":               rest_days,
            }
        except Exception as e:
            logger.warning(f"Recent form fetch error (team {team_id}): {e}")
            return {}

    def get_team_id(self, team_name: str) -> int | None:
        """Look up NBA team ID by full or partial name."""
        if not self._available:
            return None
        try:
            all_teams = self._teams.get_teams()
            name_lower = team_name.lower()
            for t in all_teams:
                if (name_lower in t["full_name"].lower() or
                        name_lower in t["nickname"].lower() or
                        name_lower in t["abbreviation"].lower()):
                    return t["id"]
            return None
        except Exception:
            return None


class ESPNStatsFetcher:
    """
    Fetches team stats for NFL, MLB, NHL via ESPN's unofficial API.
    No key required, but use gently (cache everything).
    """

    SPORT_MAP = {
        "americanfootball_nfl": ("football",    "nfl"),
        "baseball_mlb":         ("baseball",    "mlb"),
        "icehockey_nhl":        ("hockey",      "nhl"),
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_team_stats(self, sport_key: str) -> pd.DataFrame:
        """Returns basic team stats (win%, point diff) for the given sport."""
        if sport_key not in self.SPORT_MAP:
            return pd.DataFrame()

        league_sport, league = self.SPORT_MAP[sport_key]
        cache_file = CACHE_DIR / f"espn_{league}_stats.parquet"

        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 6:
                return pd.read_parquet(cache_file)

        url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/standings"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rows = []
            for group in data.get("children", []):
                for entry in group.get("standings", {}).get("entries", []):
                    team_name = entry["team"]["displayName"]
                    stats = {s["name"]: s.get("value") for s in entry.get("stats", [])}
                    rows.append({"team": team_name, **stats})
            df = pd.DataFrame(rows)
            df.to_parquet(cache_file)
            logger.info(f"Fetched ESPN {league} standings: {len(df)} teams")
            return df
        except Exception as e:
            logger.error(f"ESPN stats fetch error ({league}): {e}")
            return pd.DataFrame()
