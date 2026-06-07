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


def current_nba_season() -> str:
    """Returns the current NBA season string, e.g. '2025-26'.
    NBA seasons start in October, so Oct–Dec belong to the new season year."""
    now = datetime.today()
    year = now.year
    if now.month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"

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

    def get_team_stats(self, season: str | None = None) -> pd.DataFrame:
        season = season or current_nba_season()
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

    def get_recent_form(self, team_id: int, n_games: int = 10, season: str | None = None) -> dict:
        season = season or current_nba_season()
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
            wins_l10  = (df["WL"] == "W").sum()
            wins_l5   = (df.head(5)["WL"] == "W").sum()
            diff_l10  = df["PLUS_MINUS"].mean()
            diff_l5   = df.head(5)["PLUS_MINUS"].mean()

            last_game_date = pd.to_datetime(df["GAME_DATE"].iloc[0])
            is_b2b    = (datetime.today() - last_game_date).days <= 1
            rest_days = (datetime.today() - last_game_date).days

            return {
                "win_pct_l10":     wins_l10 / max(len(df), 1),
                "win_pct_l5":      wins_l5  / max(min(5, len(df)), 1),
                "avg_diff_l10":    float(diff_l10),
                "avg_diff_l5":     float(diff_l5),
                "is_back_to_back": int(is_b2b),
                "rest_days":       rest_days,
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
        "basketball_wnba":      ("basketball", "wnba"),
        "americanfootball_nfl": ("football",   "nfl"),
        "baseball_mlb":         ("baseball",   "mlb"),
        "icehockey_nhl":        ("hockey",     "nhl"),
    }

    @staticmethod
    def resolve_espn_nick(team_name: str) -> str:
        """Return the last word of a team name to search in ESPN standings.
        Works correctly because odds_fetcher now stores full 'City Mascot' names
        (e.g. 'Oakland Athletics', 'Los Angeles Dodgers') so the last word is
        always the unique nickname."""
        return team_name.split()[-1]

    # Scoreboard map includes NBA (for series context only — stats come from nba_api)
    _SCOREBOARD_MAP = {
        "basketball_nba":       ("basketball", "nba"),
        "basketball_wnba":      ("basketball", "wnba"),
        "americanfootball_nfl": ("football",   "nfl"),
        "baseball_mlb":         ("baseball",   "mlb"),
        "icehockey_nhl":        ("hockey",     "nhl"),
    }

    # site.api.espn.com returns only a redirect link for some sports;
    # site.web.api.espn.com/apis/v2 has the full standings data
    _STANDINGS_BASE  = "https://site.web.api.espn.com/apis/v2/sports"
    _SCOREBOARD_BASE = "https://site.api.espn.com/apis/site/v2/sports"
    _SERIES_CACHE_TTL = 4 * 3600  # seconds

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._series_cache: dict = {}  # {(sport_key, home_nick, away_nick): (ts, str|None)}

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

        url = f"{self._STANDINGS_BASE}/{league_sport}/{league}/standings"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rows = []
            # Structure: data.children (conferences/leagues) → standings.entries (teams)
            for child in data.get("children", []):
                for entry in child.get("standings", {}).get("entries", []):
                    team_name = entry.get("team", {}).get("displayName", "")
                    stats = {s["name"]: s.get("value") for s in entry.get("stats", [])}
                    rows.append({"team": team_name, **stats})
            df = pd.DataFrame(rows)
            df.to_parquet(cache_file)
            logger.info(f"Fetched ESPN {league} standings: {len(df)} teams")
            return df
        except Exception as e:
            logger.error(f"ESPN stats fetch error ({league}): {e}")
            return pd.DataFrame()

    def get_series_context(self, sport_key: str, home_team: str, away_team: str) -> str | None:
        """
        Returns the current playoff series standing for a matchup, e.g. 'VGK leads series 2-1'.
        Scans the last 10 days of the ESPN scoreboard. Returns None for regular-season games.
        Result is cached in memory for 4 hours to avoid hammering ESPN on every bot wake.
        """
        if sport_key not in self._SCOREBOARD_MAP:
            return None

        home_nick = home_team.lower().split()[-1]
        away_nick = away_team.lower().split()[-1]
        cache_key = (sport_key, home_nick, away_nick)

        now_ts = time.time()
        if cache_key in self._series_cache:
            cached_ts, cached_ctx = self._series_cache[cache_key]
            if now_ts - cached_ts < self._SERIES_CACHE_TTL:
                return cached_ctx

        league_sport, league = self._SCOREBOARD_MAP[sport_key]
        from datetime import date, timedelta
        today = date.today()
        result = None

        for delta in range(10):
            d = today - timedelta(days=delta)
            url = (f"{self._SCOREBOARD_BASE}/{league_sport}/{league}/scoreboard"
                   f"?dates={d.strftime('%Y%m%d')}")
            try:
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                for event in resp.json().get("events", []):
                    for comp in event.get("competitions", []):
                        names = [
                            c.get("team", {}).get("displayName", "").lower()
                            for c in comp.get("competitors", [])
                        ]
                        if (any(home_nick in n for n in names) and
                                any(away_nick in n for n in names)):
                            series = comp.get("series", {})
                            # ESPN summary is the useful field ("VGK leads series 2-1");
                            # title is always the generic "Playoff Series"
                            summary = series.get("summary") or series.get("title", "")
                            if summary and summary.lower() != "playoff series":
                                result = summary
            except Exception as e:
                logger.debug(f"Series context fetch error ({league} {d}): {e}")

            if result:
                break
            time.sleep(0.2)

        self._series_cache[cache_key] = (now_ts, result)
        if result:
            logger.info(f"Series context for {away_team} @ {home_team}: {result}")
        return result
