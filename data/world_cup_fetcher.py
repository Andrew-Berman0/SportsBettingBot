"""
data/world_cup_fetcher.py
-------------------------
Fetches FIFA World Cup 2026 game data from ESPN's unofficial API.
Provides odds (via DraftKings embed), team form, rosters, injuries,
head-to-head history, and group standings — no API key required.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class WorldCupFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0"
        self._roster_cache: dict[str, tuple[float, list]] = {}  # team_id → (timestamp, athletes)
        self._team_id_cache: dict[str, str] = {}               # display_name.lower() → team_id
        self._parse_errors = 0   # events that crashed in _parse_event this fetch (drift signal)

    # ------------------------------------------------------------------
    # Odds / upcoming games
    # ------------------------------------------------------------------

    def get_upcoming_games(self) -> list[dict]:
        """
        Returns pre-parsed game dicts for today + tomorrow's World Cup matches.
        Moneyline odds (home/away/draw) sourced from ESPN's DraftKings embed.
        Cache strategy matches OddsFetcher: < 90 min → always use cache.
        """
        cache_file = CACHE_DIR / "odds_soccer_fifa_world_cup.json"
        now = datetime.now(timezone.utc)

        if cache_file.exists():
            age_min = (time.time() - cache_file.stat().st_mtime) / 60
            if age_min < 90:
                logger.info(f"World Cup odds cache fresh ({age_min:.0f}min) — skipping fetch")
                with open(cache_file) as f:
                    games = json.load(f)
                # The file cache survives restarts but the in-memory team-id cache does
                # not — repopulate it from the cached games so roster/injury lookups
                # (and thus the squad data-gap check) don't fail after a restart.
                self._cache_team_ids(games)
                return games

        games: list[dict] = []
        seen: set[str] = set()
        self._parse_errors = 0
        for delta in (0, 1):
            date_str = (now + timedelta(days=delta)).strftime("%Y%m%d")
            try:
                r = self.session.get(
                    f"{ESPN_BASE}/scoreboard",
                    params={"dates": date_str},
                    timeout=10,
                )
                r.raise_for_status()
                for event in r.json().get("events", []):
                    game = self._parse_event(event)
                    if game and game["game_id"] not in seen:
                        seen.add(game["game_id"])
                        games.append(game)
            except Exception as e:
                logger.error(f"WorldCupFetcher scoreboard error for {date_str}: {e}")

        logger.info(f"World Cup: fetched {len(games)} upcoming game(s)")
        from SportsBettingBot.notifications import push_notifier
        push_notifier.notify_parse_errors("World Cup", self._parse_errors)
        with open(cache_file, "w") as f:
            json.dump(games, f)
        return games

    def _cache_team_ids(self, games: list[dict]) -> None:
        """Populate the name→team-id cache from game dicts (which carry the ids),
        so roster/injury lookups work even when games came from the file cache."""
        for g in games:
            for side in ("home", "away"):
                name, tid = g.get(f"{side}_team"), g.get(f"{side}_team_id")
                if name and tid:
                    self._team_id_cache[name.lower()] = tid

    def _parse_event(self, event: dict) -> dict | None:
        try:
            comp = (event.get("competitions") or [{}])[0] or {}
            event_id = str(event["id"])

            home_team = away_team = None
            home_team_id = away_team_id = None
            home_form = away_form = ""

            for c in comp.get("competitors", []):
                team = c.get("team") or {}
                name = team.get("displayName")
                tid = str(team.get("id")) if team.get("id") is not None else None
                if not name:
                    continue
                form = c.get("form") or ""
                if c.get("homeAway") == "home":
                    home_team, home_team_id, home_form = name, tid, form
                else:
                    away_team, away_team_id, away_form = name, tid, form

            if not home_team or not away_team:
                return None

            # Cache team IDs for later roster lookup
            if home_team_id:
                self._team_id_cache[home_team.lower()] = home_team_id
            if away_team_id:
                self._team_id_cache[away_team.lower()] = away_team_id

            # Parse odds from ESPN/DraftKings embed
            home_ml = away_ml = draw_ml = None
            total_line = over_odds = under_odds = None

            # ESPN returns explicit null (not a missing key) for moneyline/total/sub-fields
            # on unpriced future games, so guard every level with (x or {}).
            for o in (comp.get("odds") or []):
                o = o or {}
                ml = o.get("moneyline") or {}
                home_ml = self._parse_ml(((ml.get("home") or {}).get("close") or {}).get("odds"))
                away_ml = self._parse_ml(((ml.get("away") or {}).get("close") or {}).get("odds"))
                draw_ml = self._parse_ml(((ml.get("draw") or {}).get("close") or {}).get("odds"))

                tot = o.get("total") or {}
                over_close  = (tot.get("over")  or {}).get("close") or {}
                under_close = (tot.get("under") or {}).get("close") or {}
                if over_close.get("line"):
                    try:
                        total_line = float(
                            str(over_close["line"]).lstrip("ou")
                        )
                    except ValueError:
                        pass
                    over_odds  = self._parse_ml(over_close.get("odds"))
                    under_odds = self._parse_ml(under_close.get("odds"))
                if home_ml is not None and away_ml is not None:
                    break  # use the first odds block that actually has a moneyline

            if home_ml is None or away_ml is None:
                return None

            # No-vig implied probabilities across all 3 outcomes
            raw_home = self._american_to_implied(home_ml)
            raw_away = self._american_to_implied(away_ml)
            raw_draw = self._american_to_implied(draw_ml) if draw_ml is not None else 0.0
            total_raw = raw_home + raw_away + raw_draw
            home_implied = raw_home / total_raw if total_raw else 0.5
            away_implied = raw_away / total_raw if total_raw else 0.5
            draw_implied = raw_draw / total_raw if total_raw else 0.0

            venue = comp.get("venue") or {}
            venue_str = ", ".join(
                filter(None, [venue.get("fullName"), (venue.get("address") or {}).get("city")])
            )

            return {
                "_pre_parsed":   True,
                "game_id":       event_id,
                "sport":         "soccer_fifa_world_cup",
                "home_team":     home_team,
                "away_team":     away_team,
                "home_team_id":  home_team_id,
                "away_team_id":  away_team_id,
                "commence_time": event.get("date", ""),
                "home_ml":       home_ml,
                "away_ml":       away_ml,
                "draw_ml":       draw_ml,
                "total_line":    total_line,
                "over_odds":     over_odds,
                "under_odds":    under_odds,
                "home_implied":  round(home_implied, 4),
                "away_implied":  round(away_implied, 4),
                "draw_implied":  round(draw_implied, 4),
                "home_form":     home_form,
                "away_form":     away_form,
                "venue":         venue_str,
            }
        except Exception as e:
            self._parse_errors += 1
            logger.warning(f"WorldCupFetcher._parse_event error: {e}")
            return None

    # ------------------------------------------------------------------
    # Team stats DataFrame
    # ------------------------------------------------------------------

    def get_team_stats(self) -> pd.DataFrame:
        """
        Returns a DataFrame (one row per team) with tournament record and
        goal stats. Values are zero at tournament start and fill in as
        group stage matches are played.
        """
        try:
            r = self.session.get(f"{ESPN_BASE}/teams", timeout=10)
            r.raise_for_status()
            rows = []
            for sp in r.json().get("sports", []):
                for lg in sp.get("leagues", []):
                    for entry in lg.get("teams", []):
                        t = entry["team"]
                        name = t.get("displayName", "")
                        stat_map: dict[str, float] = {}
                        for item in t.get("record", {}).get("items", []):
                            for s in item.get("stats", []):
                                stat_map[s["name"]] = s["value"]
                        rows.append({
                            "team":          name,
                            "wins":          stat_map.get("wins", 0),
                            "losses":        stat_map.get("losses", 0),
                            "ties":          stat_map.get("ties", 0),
                            "goals_for":     stat_map.get("pointsFor", 0),
                            "goals_against": stat_map.get("pointsAgainst", 0),
                            "goal_diff":     stat_map.get("pointDifferential", 0),
                            "points":        stat_map.get("points", 0),
                        })
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e:
            logger.warning(f"WorldCupFetcher.get_team_stats error: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Roster + injuries
    # ------------------------------------------------------------------

    def get_team_roster_and_injuries(self, team_name: str) -> tuple[list[dict], str]:
        """
        Returns (injuries, roster_string) for a team.
          injuries     – list of {player, status, detail} for non-Active players
          roster_string – all 26 squad members as a comma-separated string
        """
        team_id = self._team_id_cache.get(team_name.lower())
        if not team_id:
            return [], ""

        now_ts = time.time()
        cached = self._roster_cache.get(team_id)
        if cached and (now_ts - cached[0]) < 14400:  # 4-hour cache
            athletes = cached[1]
        else:
            try:
                r = self.session.get(
                    f"{ESPN_BASE}/teams/{team_id}/roster", timeout=10
                )
                r.raise_for_status()
                athletes = r.json().get("athletes", [])
                self._roster_cache[team_id] = (now_ts, athletes)
            except Exception as e:
                logger.warning(f"WorldCupFetcher roster error for {team_name}: {e}")
                return [], ""

        injuries: list[dict] = []
        all_names: list[str] = []

        for a in athletes:
            name = a.get("displayName", "")
            pos  = a.get("position", {}).get("abbreviation", "")
            all_names.append(f"{name} ({pos})" if pos else name)

            status_type = a.get("status", {}).get("type", "active")
            inj_list    = a.get("injuries", [])
            if status_type != "active" or inj_list:
                detail = (
                    inj_list[0].get("details", {}).get("detail", "")
                    if inj_list else ""
                )
                injuries.append({
                    "player": name,
                    "status": a.get("status", {}).get("name", "Unknown"),
                    "detail": detail,
                })

        return injuries, ", ".join(all_names)

    # ------------------------------------------------------------------
    # Group context for Claude (standings + H2H + draw odds)
    # ------------------------------------------------------------------

    def get_group_context(
        self,
        event_id: str,
        home_team: str,
        away_team: str,
        draw_ml: float | None,
        draw_implied: float,
    ) -> str | None:
        """
        Returns a formatted context string for the Claude prompt covering:
          - Draw probability from market odds
          - Group stage standings (when populated)
          - Recent head-to-head results
        """
        lines: list[str] = []

        if draw_ml is not None:
            draw_odds_str = f"+{int(draw_ml)}" if draw_ml > 0 else str(int(draw_ml))
            lines.append(
                f"Draw odds: {draw_odds_str} (market implies {draw_implied:.0%} draw probability)"
            )

        try:
            r = self.session.get(
                f"{ESPN_BASE}/summary", params={"event": event_id}, timeout=10
            )
            d = r.json()

            # Group standings
            for grp in d.get("standings", {}).get("groups", []):
                grp_name = grp.get("name", "Group")
                entries  = grp.get("standings", {}).get("entries", [])
                if entries:
                    lines.append(f"\n{grp_name} standings:")
                    for entry in entries:
                        tname    = entry.get("team", {}).get("displayName", "")
                        stat_map = {s["name"]: s["value"] for s in entry.get("stats", [])}
                        gp  = int(stat_map.get("gamesPlayed", 0))
                        pts = int(stat_map.get("points", 0))
                        gf  = int(stat_map.get("pointsFor", 0))
                        ga  = int(stat_map.get("pointsAgainst", 0))
                        lines.append(f"  {tname}: {gp} GP | {pts} pts | {gf}-{ga} goals")

            # Head-to-head
            h2h_results: list[str] = []
            for entry in d.get("headToHeadGames", []):
                for ev in entry.get("events", []):
                    comp = (ev.get("competitions") or [{}])[0]
                    parts = []
                    for t in comp.get("competitors", []):
                        parts.append(
                            f"{t.get('team', {}).get('displayName', '?')} {t.get('score', '?')}"
                        )
                    date = ev.get("date", "")[:10]
                    if parts:
                        h2h_results.append(" vs ".join(parts) + f" ({date})")
            if h2h_results:
                lines.append("\nRecent H2H: " + "; ".join(h2h_results[:3]))

        except Exception as e:
            logger.debug(f"WorldCupFetcher.get_group_context error: {e}")

        return "\n".join(lines) if lines else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ml(val) -> float | None:
        if val is None:
            return None
        try:
            return float(str(val))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _american_to_implied(odds: float) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)
