"""
data/stats_fetcher.py
---------------------
Fetches team stats for NBA (and other sports) to build features.

NBA: uses the free nba_api package (official NBA stats endpoint)
NFL/MLB/NHL: ESPN unofficial API (free, no key required)
"""

import json
import logging
import time
from datetime import date as date_type, datetime, timedelta
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

try:
    import nfl_data_py as nfl_data
    _NFL_DATA_AVAILABLE = True
except ImportError:
    _NFL_DATA_AVAILABLE = False
    nfl_data = None

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

        # During the NBA playoff window, prefer playoff game logs so "recent form"
        # reflects the actual postseason run rather than stale regular-season games
        # (which ended in April). Fall back to regular season if the team has no
        # playoff games yet (e.g. play-in or eliminated teams).
        today = datetime.today()
        in_playoffs = (today.month == 4 and today.day >= 12) or today.month in (5, 6)
        season_types = ["Playoffs", "Regular Season"] if in_playoffs else ["Regular Season"]

        df = None
        for season_type in season_types:
            try:
                time.sleep(0.6)
                logs = self._teamgamelogs.TeamGameLogs(
                    team_id_nullable=team_id,
                    season_nullable=season,
                    season_type_nullable=season_type,
                    last_n_games_nullable=n_games,
                )
                d = logs.get_data_frames()[0]
                if not d.empty:
                    df = d
                    break
            except Exception as e:
                logger.warning(f"Recent form fetch error (team {team_id}, {season_type}): {e}")

        if df is None or df.empty:
            return {}

        try:
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
            logger.warning(f"Recent form compute error (team {team_id}): {e}")
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

    _CORE_BASE = "https://sports.core.api.espn.com/v2/sports"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._series_cache: dict = {}  # {(sport_key, home_nick, away_nick): (ts, str|None)}
        self._throws_cache: dict = {}  # {athlete_id: "L"|"R"|None}
        self._rates_cache:  dict = {}  # {athlete_id: {ip, gs, whip, k9, kbb}}

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
                    # Some fields (Home, Road, Last Ten Games, overall) carry their
                    # record only in displayValue ("4-1"); value is None. Fall back so
                    # those reach the prompt instead of rendering N/A.
                    stats = {}
                    for s in entry.get("stats", []):
                        v = s.get("value")
                        stats[s["name"]] = v if v is not None else s.get("displayValue")
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

    def get_starting_pitchers(self, home_team: str, away_team: str) -> dict:
        """
        Returns probable starting pitchers for an MLB game from the ESPN scoreboard.
        Checks today and tomorrow. Result: {"home": {...}, "away": {...}} or {}.
        Each side: {"name": str, "record": str, "era": str}
        """
        from datetime import date, timedelta
        home_nick = home_team.split()[-1].lower()
        away_nick = away_team.split()[-1].lower()

        for delta in range(2):
            d = date.today() + timedelta(days=delta)
            url = (f"{self._SCOREBOARD_BASE}/baseball/mlb/scoreboard"
                   f"?dates={d.strftime('%Y%m%d')}")
            try:
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                for event in resp.json().get("events", []):
                    for comp in event.get("competitions", []):
                        competitors = comp.get("competitors", [])
                        home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                        away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                        if not home_c or not away_c:
                            continue
                        h_name = home_c.get("team", {}).get("displayName", "").lower()
                        a_name = away_c.get("team", {}).get("displayName", "").lower()
                        if home_nick in h_name and away_nick in a_name:
                            result = {}
                            for side, comp_ in (("home", home_c), ("away", away_c)):
                                probs = comp_.get("probables", [])
                                if probs:
                                    p = probs[0]
                                    stats = {s["abbreviation"]: s["displayValue"]
                                             for s in p.get("statistics", [])}
                                    aid = p.get("athlete", {}).get("id")
                                    result[side] = {
                                        "name":   p.get("athlete", {}).get("fullName", "TBD"),
                                        "record": p.get("record", ""),
                                        "era":    stats.get("ERA", "?"),
                                        "wins":   stats.get("W", "?"),
                                        "losses": stats.get("L", "?"),
                                        "throws": self._fetch_pitcher_throws(aid),
                                        **self._fetch_pitcher_rates(aid),
                                    }
                                else:
                                    result[side] = {"name": "TBD", "record": "", "era": "?"}
                            if result:
                                logger.info(
                                    f"Starters: {away_team} ({result.get('away',{}).get('name','?')}) "
                                    f"@ {home_team} ({result.get('home',{}).get('name','?')})"
                                )
                                return result
            except Exception as e:
                logger.debug(f"Starting pitcher fetch error: {e}")
        return {}

    def _fetch_pitcher_throws(self, athlete_id) -> str | None:
        """Returns 'L' or 'R' (pitching hand) for an ESPN athlete id, or None.
        Cached in memory since handedness never changes within a run."""
        if not athlete_id:
            return None
        aid = str(athlete_id)
        if aid in self._throws_cache:
            return self._throws_cache[aid]
        hand = None
        try:
            r = self.session.get(
                f"{self._CORE_BASE}/baseball/leagues/mlb/athletes/{aid}", timeout=10
            )
            r.raise_for_status()
            abbr = r.json().get("throws", {}).get("abbreviation")
            if abbr in ("L", "R"):
                hand = abbr
        except Exception as e:
            logger.debug(f"Pitcher handedness fetch error ({aid}): {e}")
        self._throws_cache[aid] = hand
        return hand

    def _fetch_pitcher_rates(self, athlete_id) -> dict:
        """Season pitching peripherals for an ESPN athlete id: IP/GS (sample size)
        and WHIP/K9/K-BB (whether the ERA is earned). Cached in memory."""
        if not athlete_id:
            return {}
        aid = str(athlete_id)
        if aid in self._rates_cache:
            return self._rates_cache[aid]
        rates: dict = {}
        try:
            year = datetime.today().year
            r = self.session.get(
                f"{self._CORE_BASE}/baseball/leagues/mlb/seasons/{year}/types/2/athletes/{aid}/statistics",
                timeout=10,
            )
            r.raise_for_status()
            for cat in r.json().get("splits", {}).get("categories", []):
                if cat.get("name") != "pitching":
                    continue
                m = {x.get("name"): x.get("displayValue") for x in cat.get("stats", [])}
                rates = {
                    "ip":   m.get("innings"),
                    "gs":   m.get("gamesStarted"),
                    "whip": m.get("WHIP"),
                    "k9":   m.get("strikeoutsPerNineInnings"),
                    "kbb":  m.get("strikeoutToWalkRatio"),
                }
                break
        except Exception as e:
            logger.debug(f"Pitcher rates fetch error ({aid}): {e}")
        self._rates_cache[aid] = rates
        return rates


class NHLStatsFetcher:
    """
    Fetches NHL advanced stats from the official NHL Stats API (api.nhle.com).
    Adds: PP%, PK%, shots/game, and the team's top goalie save%/GAA.
    No API key required.
    Tries playoff data first (gameTypeId=3); falls back to regular season when
    a team hasn't played enough playoff games (< 4).
    """

    _TEAM_URL   = "https://api.nhle.com/stats/rest/en/team/summary"
    _GOALIE_URL = "https://api.nhle.com/stats/rest/en/goalie/summary"
    _TEAMS_URL  = "https://api.nhle.com/stats/rest/en/team"
    _CACHE_TTL_HOURS = 6

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._abbrev_map: dict | None = None  # teamId -> triCode (summary endpoint dropped its abbrev field)

    def _team_abbrev_map(self) -> dict:
        """teamId -> triCode. The team/summary endpoint no longer carries an
        abbreviation, but the goalie merge and rest-days lookups key on it."""
        if self._abbrev_map is None:
            self._abbrev_map = {}
            try:
                r = self.session.get(self._TEAMS_URL, timeout=10)
                r.raise_for_status()
                for t in r.json().get("data", []):
                    if t.get("id") is not None and t.get("triCode"):
                        self._abbrev_map[t["id"]] = t["triCode"]
            except Exception as e:
                logger.warning(f"NHL team abbrev map fetch failed: {e}")
        return self._abbrev_map

    def get_team_stats(self, season: str | None = None) -> pd.DataFrame:
        """
        Returns one row per NHL team:
          team, nhl_abbrev, pp_pct, pk_pct, shots_for_pg, shots_against_pg,
          goalie_name, goalie_sv_pct, goalie_gaa
        """
        season_id = season or self._current_season_id()
        cache_file = CACHE_DIR / f"nhl_advanced_stats_{season_id}.parquet"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < self._CACHE_TTL_HOURS:
                return pd.read_parquet(cache_file)
        try:
            rs = self._fetch_team_summary(season_id, game_type=2)
            po = self._fetch_team_summary(season_id, game_type=3)

            # Start from regular season; overlay playoff stats for teams with 4+ PO games
            stat_cols = ["pp_pct", "pk_pct", "shots_for_pg", "shots_against_pg"]
            if not rs.empty and not po.empty:
                po_lookup = po.set_index("team")[stat_cols + ["nhl_gp"]].rename(
                    columns={c: c + "_po" for c in stat_cols} | {"nhl_gp": "nhl_gp_po"}
                )
                df = rs.join(po_lookup, on="team", how="left")
                in_playoffs = df["nhl_gp_po"].fillna(0) >= 4
                for col in stat_cols:
                    po_col = col + "_po"
                    if po_col in df.columns:
                        df.loc[in_playoffs, col] = df.loc[in_playoffs, po_col]
                df = df.drop(columns=[c for c in df.columns if c.endswith("_po")], errors="ignore")
            elif not po.empty:
                df = po
            else:
                df = rs if not rs.empty else pd.DataFrame()

            if df.empty:
                return df

            # Goalie: prefer playoff starters (most games started), fall back to RS
            goalies = self._fetch_top_goalies(season_id, game_type=3)
            if goalies.empty:
                goalies = self._fetch_top_goalies(season_id, game_type=2)
            if not goalies.empty:
                df = df.merge(goalies, on="nhl_abbrev", how="left")

            df.to_parquet(cache_file)
            logger.info(f"Fetched NHL advanced stats: {len(df)} teams (season {season_id})")
            return df
        except Exception as e:
            logger.error(f"NHL stats fetch error: {e}")
            return pd.DataFrame()

    def _fetch_team_summary(self, season_id: str, game_type: int) -> pd.DataFrame:
        params = {
            "isAggregate": "false",
            "isGame":      "false",
            "start":       0,
            "limit":       32,
            "cayenneExp":  f"gameTypeId={game_type} and seasonId<={season_id} and seasonId>={season_id}",
        }
        try:
            r = self.session.get(self._TEAM_URL, params=params, timeout=10)
            r.raise_for_status()
            abbrev_map = self._team_abbrev_map()
            rows = []
            for t in r.json().get("data", []):
                rows.append({
                    "team":             t.get("teamFullName"),
                    "nhl_abbrev":       abbrev_map.get(t.get("teamId")),
                    "nhl_gp":           t.get("gamesPlayed"),
                    # API now returns these as fractions (0.205); the prompt expects a percent number.
                    "pp_pct":           (t.get("powerPlayPct")   * 100) if t.get("powerPlayPct")   is not None else None,
                    "pk_pct":           (t.get("penaltyKillPct") * 100) if t.get("penaltyKillPct") is not None else None,
                    "shots_for_pg":     t.get("shotsForPerGame"),
                    "shots_against_pg": t.get("shotsAgainstPerGame"),
                })
            return pd.DataFrame(rows).dropna(subset=["team"])
        except Exception as e:
            logger.warning(f"NHL team summary failed (gameType={game_type}): {e}")
            return pd.DataFrame()

    def _fetch_top_goalies(self, season_id: str, game_type: int) -> pd.DataFrame:
        params = {
            "isAggregate": "false",
            "isGame":      "false",
            "start":       0,
            "limit":       200,
            "cayenneExp":  f"gameTypeId={game_type} and seasonId<={season_id} and seasonId>={season_id}",
        }
        try:
            r = self.session.get(self._GOALIE_URL, params=params, timeout=10)
            r.raise_for_status()
            rows = []
            for g in r.json().get("data", []):
                rows.append({
                    "nhl_abbrev":   g.get("teamAbbrevs") or g.get("teamAbbrev"),
                    "goalie_name":  g.get("goalieFullName"),
                    "goalie_sv_pct": g.get("savePct"),
                    "goalie_gaa":   g.get("goalsAgainstAverage"),
                    "goalie_gp":    g.get("gamesStarted") or g.get("gamesPlayed") or 0,
                })
            df = pd.DataFrame(rows).dropna(subset=["nhl_abbrev"])
            if df.empty:
                return df
            # Top goalie per team = most games started
            df = (df.sort_values("goalie_gp", ascending=False)
                    .groupby("nhl_abbrev", as_index=False)
                    .first())
            return df[["nhl_abbrev", "goalie_name", "goalie_sv_pct", "goalie_gaa"]]
        except Exception as e:
            logger.warning(f"NHL goalie fetch failed (gameType={game_type}): {e}")
            return pd.DataFrame()

    def get_rest_days(self, abbrev: str, game_date: datetime) -> int | None:
        """
        Returns days since the team's last completed game before game_date.
        0 = back-to-back. None = schedule unavailable.
        """
        target = game_date.date() if hasattr(game_date, "date") else game_date
        months = [target.strftime("%Y-%m")]
        if target.day <= 7:
            prev = (target.replace(day=1) - timedelta(days=1))
            months.append(prev.strftime("%Y-%m"))

        played: list[date_type] = []
        for month in months:
            url = f"https://api-web.nhle.com/v1/club-schedule/{abbrev}/month/{month}"
            try:
                r = self.session.get(url, timeout=10)
                r.raise_for_status()
                for g in r.json().get("games", []):
                    # gameState 7 = final in the new API; "OFF" and "FINAL" are also used
                    if g.get("gameState") in ("OFF", "FINAL", "7", 7):
                        try:
                            gd = datetime.strptime(g["gameDate"], "%Y-%m-%d").date()
                            if gd < target:
                                played.append(gd)
                        except (KeyError, ValueError):
                            pass
            except Exception:
                pass

        if not played:
            return None
        return (target - max(played)).days

    @staticmethod
    def _current_season_id() -> str:
        today = datetime.today()
        start = today.year if today.month >= 10 else today.year - 1
        return f"{start}{start + 1}"


class WNBAStatsFetcher:
    """
    Augments ESPN standings data with per-team shooting and ball-control
    stats from ESPN's team statistics endpoint.
    Adds: FG%, 3PT%, assist-to-turnover ratio, turnovers/game, rebounds/game.
    No API key required. Fetches once per team per session (15 calls, cached 6h).
    """

    _TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams"
    _STATS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{tid}/statistics"
    _CACHE_TTL_HOURS = 6

    _SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{tid}/schedule"
    _ROSTER_URL   = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{tid}/roster"
    _CORE_BASE    = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._team_id_map: dict[str, str] = {}  # displayName.lower() -> team_id

    def get_rest_days(self, team_name: str, game_date: datetime) -> int | None:
        """
        Returns days since the team's last completed game before game_date.
        0 = back-to-back. None = schedule unavailable.
        """
        tid = self._resolve_team_id(team_name)
        if not tid:
            return None
        target = game_date.date() if hasattr(game_date, "date") else game_date
        try:
            r = self.session.get(self._SCHEDULE_URL.format(tid=tid), timeout=10)
            r.raise_for_status()
            played = []
            for e in r.json().get("events", []):
                comp = (e.get("competitions") or [{}])[0]
                if not comp.get("status", {}).get("type", {}).get("completed"):
                    continue
                try:
                    gd = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).date()
                    if gd < target:
                        played.append(gd)
                except (KeyError, ValueError):
                    pass
            if not played:
                return None
            return (target - max(played)).days
        except Exception as e:
            logger.warning(f"WNBA rest days failed for {team_name}: {e}")
            return None

    def _resolve_team_id(self, team_name: str) -> str | None:
        if not self._team_id_map:
            try:
                self._team_id_map = {
                    name.lower(): tid for tid, name in self._fetch_team_list()
                }
            except Exception as e:
                logger.warning(f"WNBA team list fetch failed: {e}")
                self._team_id_map = {}
        name = team_name.lower()
        if name in self._team_id_map:
            return self._team_id_map[name]
        nick = team_name.split()[-1].lower()
        return next(
            (tid for n, tid in self._team_id_map.items() if n.split()[-1] == nick),
            None,
        )

    def get_team_stats(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per WNBA team:
          team, fg_pct, three_pct, ast_to_ratio, avg_turnovers,
          avg_rebounds, avg_off_rebounds
        """
        cache_file = CACHE_DIR / f"wnba_advanced_stats_{datetime.today().year}.parquet"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < self._CACHE_TTL_HOURS:
                return pd.read_parquet(cache_file)

        try:
            teams = self._fetch_team_list()
            rows = []
            for tid, tname in teams:
                stats = self._fetch_team_stats(tid)
                if stats:
                    rows.append({"team": tname, **stats})
                time.sleep(0.2)
            df = pd.DataFrame(rows)
            df.to_parquet(cache_file)
            logger.info(f"Fetched WNBA advanced stats: {len(df)} teams")
            return df
        except Exception as e:
            logger.error(f"WNBA advanced stats fetch error: {e}")
            return pd.DataFrame()

    def _fetch_team_list(self) -> list[tuple[str, str]]:
        r = self.session.get(self._TEAMS_URL, timeout=10)
        r.raise_for_status()
        teams = (r.json().get("sports", [{}])[0]
                  .get("leagues", [{}])[0]
                  .get("teams", []))
        return [(t["team"]["id"], t["team"]["displayName"]) for t in teams]

    def _fetch_team_stats(self, tid: str) -> dict:
        try:
            r = self.session.get(self._STATS_URL.format(tid=tid), timeout=10)
            r.raise_for_status()
            cats = (r.json().get("results", {})
                     .get("stats", {})
                     .get("categories", []))
            flat = {}
            for cat in cats:
                for stat in cat.get("stats", []):
                    flat[stat["name"]] = stat.get("value")
            return {
                "fg_pct":           flat.get("fieldGoalPct"),
                "three_pct":        flat.get("threePointPct"),
                "ast_to_ratio":     flat.get("assistTurnoverRatio"),
                "avg_turnovers":    flat.get("avgTurnovers"),
                "avg_rebounds":     flat.get("avgRebounds"),
                "avg_off_rebounds": flat.get("avgOffensiveRebounds"),
                "avg_steals":       flat.get("avgSteals"),
                "avg_blocks":       flat.get("avgBlocks"),
            }
        except Exception:
            return {}

    def get_player_stats(self, team_name: str, top_n: int = 6) -> list[dict]:
        """Top season scorers for a team: [{name, pos, ppg, rpg, apg, mpg}], sorted by PPG.
        The WNBA persona weights individual players heavily (one star's absence can swing a
        game), but it was only given team aggregates + injury NAMES — no way to gauge how
        good an injured player is. This supplies per-player production so the persona is
        actionable. Cached on disk per team for 6h (season averages move slowly)."""
        tid = self._resolve_team_id(team_name)
        if not tid:
            return []
        cache_file = CACHE_DIR / f"wnba_players_{tid}.json"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < self._CACHE_TTL_HOURS:
                try:
                    with open(cache_file) as f:
                        return json.load(f)
                except Exception:
                    pass
        players: list[dict] = []
        try:
            r = self.session.get(self._ROSTER_URL.format(tid=tid), timeout=10)
            r.raise_for_status()
            for a in r.json().get("athletes", []):
                if not isinstance(a, dict):
                    continue
                aid, name = a.get("id"), a.get("displayName")
                if not aid or not name:
                    continue
                rates = self._fetch_player_rates(aid)
                if rates.get("ppg") is None:
                    continue
                players.append({"name": name, "pos": (a.get("position") or {}).get("abbreviation", ""), **rates})
            players.sort(key=lambda p: p.get("ppg") or 0, reverse=True)
            players = players[:top_n]
            with open(cache_file, "w") as f:
                json.dump(players, f)
        except Exception as e:
            logger.warning(f"WNBA player stats error for {team_name}: {e}")
        return players

    def _fetch_player_rates(self, athlete_id) -> dict:
        """Season per-game averages (PPG/RPG/APG/MPG) for a WNBA athlete id, from ESPN core."""
        year = datetime.today().year
        try:
            r = self.session.get(
                f"{self._CORE_BASE}/seasons/{year}/types/2/athletes/{athlete_id}/statistics",
                timeout=10,
            )
            r.raise_for_status()
            flat: dict = {}
            for cat in r.json().get("splits", {}).get("categories", []):
                for x in cat.get("stats", []):
                    flat[x.get("name")] = x.get("value")
            if flat.get("avgPoints") is None:
                return {}
            rnd = lambda v: round(v, 1) if isinstance(v, (int, float)) else None
            return {
                "ppg": rnd(flat.get("avgPoints")),
                "rpg": rnd(flat.get("avgRebounds")),
                "apg": rnd(flat.get("avgAssists")),
                "mpg": rnd(flat.get("avgMinutes")),
            }
        except Exception as e:
            logger.debug(f"WNBA player rates error ({athlete_id}): {e}")
            return {}


class MLBStatsFetcher:
    """
    Fetches advanced MLB team stats from the official stats.mlb.com API.
    Provides team ERA/WHIP/K9, bullpen ERA, and offensive OPS/OBP/SLG.
    No API key required.
    """

    _BASE = "https://statsapi.mlb.com/api/v1"
    _CACHE_TTL_HOURS = 6

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_team_stats(self, season: int | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame (one row per team) with:
          team, team_era, team_whip, team_k9, team_bb9, team_hr9,
          bullpen_era, team_ops, team_obp, team_slg, team_avg, team_runs, team_hr
        """
        year = season or datetime.today().year
        cache_file = CACHE_DIR / f"mlb_advanced_stats_{year}.parquet"

        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < self._CACHE_TTL_HOURS:
                return pd.read_parquet(cache_file)

        try:
            pitching_df = self._fetch_team_pitching(year)
            hitting_df  = self._fetch_team_hitting(year)
            bullpen_df  = self._fetch_bullpen_era(year)

            df = pitching_df.merge(hitting_df, on="team", how="outer")
            df = df.merge(bullpen_df, on="team", how="left")
            df.to_parquet(cache_file)
            logger.info(f"Fetched MLB advanced stats: {len(df)} teams ({year})")
            return df
        except Exception as e:
            logger.error(f"MLB advanced stats fetch error: {e}")
            return pd.DataFrame()

    def _fetch_team_pitching(self, year: int) -> pd.DataFrame:
        r = self.session.get(f"{self._BASE}/teams/stats", params={
            "season": year, "stats": "season", "group": "pitching",
            "gameType": "R", "sportId": 1,
        }, timeout=10)
        r.raise_for_status()
        rows = []
        for sp in r.json().get("stats", [{}])[0].get("splits", []):
            st = sp["stat"]
            rows.append({
                "team":      sp["team"]["name"],
                "team_era":  st.get("era"),
                "team_whip": st.get("whip"),
                "team_k9":   st.get("strikeoutsPer9Inn"),
                "team_bb9":  st.get("walksPer9Inn"),
                "team_hr9":  st.get("homeRunsPer9"),
            })
        return pd.DataFrame(rows)

    def _fetch_team_hitting(self, year: int) -> pd.DataFrame:
        r = self.session.get(f"{self._BASE}/teams/stats", params={
            "season": year, "stats": "season", "group": "hitting",
            "gameType": "R", "sportId": 1,
        }, timeout=10)
        r.raise_for_status()
        rows = []
        for sp in r.json().get("stats", [{}])[0].get("splits", []):
            st = sp["stat"]
            rows.append({
                "team":      sp["team"]["name"],
                "team_ops":  st.get("ops"),
                "team_obp":  st.get("obp"),
                "team_slg":  st.get("slg"),
                "team_avg":  st.get("avg"),
                "team_runs": st.get("runs"),
                "team_hr":   st.get("homeRuns"),
            })
        return pd.DataFrame(rows)

    def _fetch_bullpen_era(self, year: int) -> pd.DataFrame:
        """Aggregate ERA for relievers only (gamesStarted=0, gamesPlayed>=5) per team."""
        from collections import defaultdict
        r = self.session.get(f"{self._BASE}/stats", params={
            "stats": "season", "group": "pitching", "gameType": "R",
            "season": year, "sportId": 1, "limit": 500, "playerPool": "All",
        }, timeout=15)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])

        team_bp: dict = defaultdict(lambda: {"earned_runs": 0, "outs": 0.0})
        for sp in splits:
            st = sp["stat"]
            if st.get("gamesStarted", 1) != 0 or st.get("gamesPlayed", 0) < 5:
                continue
            team = sp.get("team", {}).get("name", "")
            if not team:
                continue
            try:
                ip_str = str(st.get("inningsPitched", "0"))
                parts = ip_str.split(".")
                innings = float(parts[0]) + (float(parts[1]) / 3 if len(parts) > 1 else 0)
                team_bp[team]["earned_runs"] += st.get("earnedRuns", 0)
                team_bp[team]["outs"] += innings * 3
            except Exception:
                continue

        rows = []
        for team, d in team_bp.items():
            bp_era = round((d["earned_runs"] * 27) / d["outs"], 2) if d["outs"] > 0 else None
            rows.append({"team": team, "bullpen_era": bp_era})
        return pd.DataFrame(rows)


class NFLStatsFetcher:
    """
    NFL advanced team stats from two sources:
      1. ESPN per-team statistics endpoint  — yards/game, sacks, turnover differential
      2. nfl_data_py (nflverse)             — offensive/defensive EPA per play
         Requires: pip install nfl_data_py  (gracefully skipped if unavailable)
    """

    _TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    _STATS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}/statistics"
    _CACHE_TTL_HOURS = 6
    _EPA_CACHE_TTL_HOURS = 24

    # ESPN full name → nflverse abbreviation (used for nfl_data_py lookups)
    _TEAM_ABBREVS: dict[str, str] = {
        "Buffalo Bills": "BUF",        "Miami Dolphins": "MIA",
        "New England Patriots": "NE",  "New York Jets": "NYJ",
        "Baltimore Ravens": "BAL",     "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",     "Pittsburgh Steelers": "PIT",
        "Houston Texans": "HOU",       "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX", "Tennessee Titans": "TEN",
        "Denver Broncos": "DEN",       "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",     "Los Angeles Chargers": "LAC",
        "Dallas Cowboys": "DAL",       "New York Giants": "NYG",
        "Philadelphia Eagles": "PHI",  "Washington Commanders": "WAS",
        "Chicago Bears": "CHI",        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",     "Minnesota Vikings": "MIN",
        "Atlanta Falcons": "ATL",      "Carolina Panthers": "CAR",
        "New Orleans Saints": "NO",    "Tampa Bay Buccaneers": "TB",
        "Arizona Cardinals": "ARI",    "Los Angeles Rams": "LA",
        "San Francisco 49ers": "SF",   "Seattle Seahawks": "SEA",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_team_stats(self, season: int | None = None) -> pd.DataFrame:
        """Returns one row per NFL team with ESPN stats merged with EPA."""
        season = season or self._current_nfl_season()
        cache_file = CACHE_DIR / f"nfl_advanced_stats_{season}.parquet"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < self._CACHE_TTL_HOURS:
                return pd.read_parquet(cache_file)
        try:
            espn_df = self._fetch_espn_stats()
            epa_df  = self._fetch_epa(season)
            if not espn_df.empty and not epa_df.empty:
                df = espn_df.merge(epa_df, on="team", how="left")
            else:
                df = espn_df if not espn_df.empty else epa_df
            if not df.empty:
                df.to_parquet(cache_file)
                logger.info(f"Fetched NFL advanced stats: {len(df)} teams (season {season})")
            return df
        except Exception as e:
            logger.error(f"NFL stats fetch error: {e}")
            return pd.DataFrame()

    def get_rest_days(self, team_name: str, game_date: datetime) -> int | None:
        """
        Returns days since the team's last completed game before game_date.
        Requires nfl_data_py. Returns None if unavailable.
        """
        if not _NFL_DATA_AVAILABLE:
            return None
        abbrev = (self._TEAM_ABBREVS.get(team_name) or
                  next((v for k, v in self._TEAM_ABBREVS.items()
                        if k.split()[-1] == team_name.split()[-1]), None))
        if not abbrev:
            return None
        target = game_date.date() if hasattr(game_date, "date") else game_date
        season = target.year if target.month >= 9 else target.year - 1
        try:
            schedules = nfl_data.import_schedules([season])
            team_games = schedules[
                ((schedules["home_team"] == abbrev) | (schedules["away_team"] == abbrev)) &
                schedules["home_score"].notna()
            ].copy()
            team_games["gameday"] = pd.to_datetime(team_games["gameday"]).dt.date
            played = team_games[team_games["gameday"] < target]["gameday"]
            if played.empty:
                return None
            return (target - played.max()).days
        except Exception as e:
            logger.warning(f"NFL rest days failed for {team_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fetch_espn_stats(self) -> pd.DataFrame:
        try:
            r = self.session.get(self._TEAMS_URL, timeout=10)
            r.raise_for_status()
            teams = (r.json().get("sports", [{}])[0]
                      .get("leagues", [{}])[0]
                      .get("teams", []))
            rows = []
            for t in teams:
                tid   = t["team"]["id"]
                tname = t["team"]["displayName"]
                stats = self._fetch_team_stats(tid)
                if stats:
                    rows.append({"team": tname, **stats})
                time.sleep(0.2)
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"NFL ESPN fetch error: {e}")
            return pd.DataFrame()

    def _fetch_team_stats(self, tid: str) -> dict:
        try:
            r = self.session.get(self._STATS_URL.format(tid=tid), timeout=10)
            r.raise_for_status()
            cats = (r.json().get("results", {})
                     .get("stats", {})
                     .get("categories", []))
            flat: dict = {}
            flat_pg: dict = {}
            for cat in cats:
                prefix = cat.get("name", "")
                for stat in cat.get("stats", []):
                    key = f"{prefix}_{stat['name']}"
                    flat[key] = stat.get("value")
                    pg = stat.get("perGameValue")
                    if pg is not None:
                        flat_pg[key] = pg
            return {
                "pass_yards_pg":    (flat.get("passing_avgYards")
                                     or flat.get("passing_passingYardsPerGame")),
                "rush_yards_pg":    (flat.get("rushing_avgYards")
                                     or flat.get("rushing_rushingYardsPerGame")),
                # Per-game sacks (ESPN supplies perGameValue) — the season total
                # mislabels the "3+ sacks/game" line protection signal.
                "sacks_allowed_pg": (flat_pg.get("passing_sacks")
                                     or flat_pg.get("general_sacksAllowed")),
                "def_sacks_pg":      flat_pg.get("defensive_sacks"),
                "giveaways":        (flat.get("general_giveaways")
                                     or flat.get("scoring_giveaways")),
                "takeaways":        (flat.get("general_takeaways")
                                     or flat.get("defensive_takeaways")),
                "to_differential":  (flat.get("general_turnoverRatio")
                                     or flat.get("general_turnoverDifferential")),
            }
        except Exception:
            return {}

    def _fetch_epa(self, season: int) -> pd.DataFrame:
        if not _NFL_DATA_AVAILABLE:
            return pd.DataFrame()
        epa_cache = CACHE_DIR / f"nfl_epa_{season}.parquet"
        if epa_cache.exists():
            age_hours = (time.time() - epa_cache.stat().st_mtime) / 3600
            if age_hours < self._EPA_CACHE_TTL_HOURS:
                return pd.read_parquet(epa_cache)
        try:
            pbp = nfl_data.import_pbp_data(
                years=[season],
                columns=["posteam", "defteam", "epa", "play_type"],
                downcast=True,
            )
            plays = pbp[pbp["play_type"].isin(["pass", "run"])].dropna(subset=["epa"])
            off  = plays.groupby("posteam")["epa"].mean().round(3)
            def_ = plays.groupby("defteam")["epa"].mean().round(3)
            _rev = {v: k for k, v in self._TEAM_ABBREVS.items()}
            df = (off.rename("off_epa_per_play")
                     .to_frame()
                     .join(def_.rename("def_epa_allowed_per_play"), how="outer")
                     .reset_index()
                     .rename(columns={"index": "abbrev", "posteam": "abbrev"}))
            df["team"] = df["abbrev"].map(_rev)
            df = df.dropna(subset=["team"])[["team", "off_epa_per_play", "def_epa_allowed_per_play"]]
            df.to_parquet(epa_cache)
            logger.info(f"Fetched NFL EPA: {len(df)} teams (season {season})")
            return df
        except Exception as e:
            logger.error(f"NFL EPA fetch error: {e}")
            return pd.DataFrame()

    @staticmethod
    def _current_nfl_season() -> int:
        today = datetime.today()
        return today.year if today.month >= 9 else today.year - 1
