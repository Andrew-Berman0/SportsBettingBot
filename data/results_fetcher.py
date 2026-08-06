"""
data/results_fetcher.py
-----------------------
Fetches completed game scores from ESPN's unofficial scoreboard API.
Used to auto-settle open bets and label training data.
No API key required.
"""

import logging
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

SPORT_ESPN_MAP = {
    "basketball_nba":        ("basketball", "nba"),
    "basketball_wnba":       ("basketball", "wnba"),
    "baseball_mlb":          ("baseball",   "mlb"),
    "icehockey_nhl":         ("hockey",     "nhl"),
    "americanfootball_nfl":  ("football",   "nfl"),
    "soccer_fifa_world_cup": ("soccer",     "fifa.world"),
}


class ResultsFetcher:

    _STATSAPI = "https://statsapi.mlb.com/api/v1"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _mlb_games_on(self, date_str: str) -> list:
        """All MLB games on a date from statsapi (carries scores, status, and start time)."""
        try:
            r = self.session.get(f"{self._STATSAPI}/schedule",
                                 params={"sportId": 1, "date": date_str}, timeout=12)
            r.raise_for_status()
            dates = r.json().get("dates", [])
            return dates[0]["games"] if dates else []
        except Exception as e:
            logger.warning(f"MLB results fetch failed ({date_str}): {e}")
            return []

    def get_completed_games(self, days_back: int = 2,
                            sport: str = "basketball_nba") -> list[dict]:
        """
        Returns completed games from the past `days_back` days for the given sport.
        Each entry: {home_team, away_team, home_score, away_score, date}
        """
        if sport == "mma_ufc":
            return self._completed_mma(days_back)
        if sport == "baseball_mlb":
            return self._completed_mlb_games(days_back)
        if sport not in SPORT_ESPN_MAP:
            return []

        league_sport, league = SPORT_ESPN_MAP[sport]
        url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/scoreboard"

        results = []
        for days_ago in range(days_back + 1):
            date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")
            try:
                resp = self.session.get(url, params={"dates": date}, timeout=10)
                resp.raise_for_status()
                for event in resp.json().get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    if not comp.get("status", {}).get("type", {}).get("completed"):
                        continue
                    home, away = None, None
                    for team in comp.get("competitors", []):
                        info = {
                            "name":  team["team"]["displayName"],
                            "score": int(team.get("score") or 0),
                        }
                        if team["homeAway"] == "home":
                            home = info
                        else:
                            away = info
                    if home and away:
                        results.append({
                            "home_team":  home["name"],
                            "away_team":  away["name"],
                            "home_score": home["score"],
                            "away_score": away["score"],
                            "date":       date[:4] + "-" + date[4:6] + "-" + date[6:],
                        })
            except Exception as e:
                logger.warning(f"ResultsFetcher error for {sport} {date}: {e}")
        return results

    def _completed_mlb_games(self, days_back: int) -> list[dict]:
        """Completed MLB games from statsapi (for outcome logging): name/score/date."""
        results = []
        for days_ago in range(days_back + 1):
            d = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            for g in self._mlb_games_on(d):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                teams = g.get("teams", {})
                h, a = teams.get("home", {}), teams.get("away", {})
                if h.get("score") is None or a.get("score") is None:
                    continue
                results.append({
                    "home_team":  (h.get("team") or {}).get("name", ""),
                    "away_team":  (a.get("team") or {}).get("name", ""),
                    "home_score": int(h["score"]),
                    "away_score": int(a["score"]),
                    "date":       d,
                })
        return results

    def _settle_mlb(self, sport_bets: list, broker, now: datetime) -> int:
        """Settle/void MLB bets from statsapi.mlb.com. Keeps MLB off ESPN's rate-limited
        scoreboard, and uses each game's start time to disambiguate doubleheaders."""
        days_back = 3
        for b in sport_bets:
            ct = b.get("commence_time")
            if ct:
                try:
                    start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    days_back = max(days_back, (now - start).days + 1)
                except Exception:
                    pass
        days_back = min(days_back, 30)

        settled_count = 0
        remaining = list(sport_bets)
        for days_ago in range(days_back + 1):
            if not remaining:
                break
            d = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            for g in self._mlb_games_on(d):
                teams  = g.get("teams", {})
                h, a   = teams.get("home", {}), teams.get("away", {})
                h_name = (h.get("team") or {}).get("name", "")
                a_name = (a.get("team") or {}).get("name", "")
                status = g.get("status", {})
                matching = [
                    b for b in remaining
                    if self._teams_match(b["home_team"], h_name)
                    and self._teams_match(b["away_team"], a_name)
                    and self._game_has_started(b.get("commence_time"), now)
                    and self._start_time_aligns(g.get("gameDate"), b.get("commence_time"))
                ]
                if not matching:
                    continue
                gid = matching[0]["game_id"]
                if (status.get("abstractGameState") == "Final"
                        and h.get("score") is not None and a.get("score") is not None):
                    logger.info(f"Settling {a_name} @ {h_name} ({a['score']}-{h['score']}) [baseball_mlb]")
                    settled = broker.settle_bet(game_id=gid,
                                                home_score=int(h["score"]),
                                                away_score=int(a["score"]))
                    settled_count += len(settled)
                    remaining = [b for b in remaining if b["game_id"] != gid]
                elif status.get("detailedState", "") in ("Postponed", "Cancelled", "Canceled"):
                    reason = status["detailedState"].lower()
                    logger.info(f"Voiding {a_name} @ {h_name} — {reason} [baseball_mlb]")
                    voided = broker.void_bet(game_id=gid, reason=reason)
                    settled_count += len(voided)
                    remaining = [b for b in remaining if b["game_id"] != gid]
        return settled_count

    def _completed_mma(self, days_back: int) -> list[dict]:
        """
        Completed UFC fights. A card is one event with many fights, each decided by
        a `winner` flag (no scores). Emits a 1/0 score per fight, in BOTH fighter
        orderings, so outcome_tracker's order-sensitive name key matches either way.
        """
        url = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
        results = []
        for days_ago in range(days_back + 1):
            date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")
            date_iso = date[:4] + "-" + date[4:6] + "-" + date[6:]
            try:
                resp = self.session.get(url, params={"dates": date}, timeout=10)
                resp.raise_for_status()
                for event in resp.json().get("events", []):
                    for comp in event.get("competitions", []):
                        if not comp.get("status", {}).get("type", {}).get("completed"):
                            continue
                        competitors = comp.get("competitors", [])
                        if len(competitors) != 2:
                            continue
                        fa, fb = competitors[0], competitors[1]
                        na = fa.get("athlete", {}).get("displayName")
                        nb = fb.get("athlete", {}).get("displayName")
                        a_won, b_won = bool(fa.get("winner")), bool(fb.get("winner"))
                        if not na or not nb or a_won == b_won:
                            continue   # missing name or draw/no-contest
                        results.append({"home_team": na, "away_team": nb,
                                        "home_score": int(a_won), "away_score": int(b_won), "date": date_iso})
                        results.append({"home_team": nb, "away_team": na,
                                        "home_score": int(b_won), "away_score": int(a_won), "date": date_iso})
            except Exception as e:
                logger.warning(f"ResultsFetcher MMA completed error for {date}: {e}")
        return results

    _VOID_STATUSES = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED"}

    def settle_open_bets(self, broker) -> int:
        """
        Checks ESPN for completed games, settles any matching open bets.
        Also detects postponed/cancelled games and voids those bets (stake refunded).
        Handles all sports that have open bets.
        Returns the number of bets settled or voided.
        """
        if not broker.open_bets:
            return 0

        now = datetime.now(timezone.utc)

        # Group open bets by sport so we fetch each sport's scoreboard once
        bets_by_sport: dict[str, list] = defaultdict(list)
        for bet in broker.open_bets:
            bets_by_sport[bet.get("sport", "basketball_nba")].append(bet)

        settled_count = 0

        for sport, sport_bets in bets_by_sport.items():
            if sport == "mma_ufc":
                # MMA settles by winner (no scores), and a card is one event with many
                # fights — needs its own path rather than the team score-comparison logic.
                settled_count += self._settle_mma(sport_bets, broker, now)
                continue
            if sport == "baseball_mlb":
                # MLB settles from the official MLB API (statsapi), not ESPN's rate-limited
                # scoreboard — and its clean per-game start times disambiguate doubleheaders.
                settled_count += self._settle_mlb(sport_bets, broker, now)
                continue
            if sport not in SPORT_ESPN_MAP:
                logger.debug(f"No ESPN scoreboard configured for sport '{sport}' — skipping settlement")
                continue

            # Dynamic lookback from oldest open bet for this sport
            days_back = 3
            for b in sport_bets:
                ct = b.get("commence_time")
                if ct:
                    try:
                        start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                        days_old = (now - start).days + 1
                        days_back = max(days_back, days_old)
                    except Exception:
                        pass
            days_back = min(days_back, 30)

            league_sport, league = SPORT_ESPN_MAP[sport]
            url = f"https://site.api.espn.com/apis/site/v2/sports/{league_sport}/{league}/scoreboard"

            for days_ago in range(days_back + 1):
                date = (now - timedelta(days=days_ago)).strftime("%Y%m%d")
                try:
                    resp = self.session.get(url, params={"dates": date}, timeout=10)
                    resp.raise_for_status()
                    for event in resp.json().get("events", []):
                        comp   = event.get("competitions", [{}])[0]
                        status = comp.get("status", {}).get("type", {})
                        home, away = None, None
                        for team in comp.get("competitors", []):
                            info = {
                                "name":  team["team"]["displayName"],
                                "score": int(team.get("score") or 0),
                            }
                            if team["homeAway"] == "home":
                                home = info
                            else:
                                away = info
                        if not home or not away:
                            continue

                        result_date = date[:4] + "-" + date[4:6] + "-" + date[6:]

                        if status.get("completed"):
                            # Normal settlement
                            matching = [
                                b for b in sport_bets
                                if self._teams_match(b["home_team"], home["name"])
                                and self._teams_match(b["away_team"], away["name"])
                                and self._game_has_started(b.get("commence_time"), now)
                                and self._result_date_matches(result_date, b.get("commence_time"))
                                and self._start_time_aligns(event.get("date"), b.get("commence_time"))
                            ]
                            if matching:
                                game_id = matching[0]["game_id"]
                                logger.info(
                                    f"Settling {away['name']} @ {home['name']} "
                                    f"({away['score']}-{home['score']}) [{sport}]"
                                )
                                settled = broker.settle_bet(
                                    game_id=game_id,
                                    home_score=home["score"],
                                    away_score=away["score"],
                                )
                                settled_count += len(settled)
                                sport_bets = [b for b in sport_bets if b["game_id"] != game_id]

                        elif status.get("name") in self._VOID_STATUSES:
                            # Postponed / cancelled — void the bet
                            matching = [
                                b for b in sport_bets
                                if self._teams_match(b["home_team"], home["name"])
                                and self._teams_match(b["away_team"], away["name"])
                                and self._result_date_matches(result_date, b.get("commence_time"))
                                and self._start_time_aligns(event.get("date"), b.get("commence_time"))
                            ]
                            if matching:
                                game_id = matching[0]["game_id"]
                                reason  = status["name"].replace("STATUS_", "").lower()
                                logger.info(
                                    f"Voiding {away['name']} @ {home['name']} — {reason} [{sport}]"
                                )
                                voided = broker.void_bet(game_id=game_id, reason=reason)
                                settled_count += len(voided)
                                sport_bets = [b for b in sport_bets if b["game_id"] != game_id]

                except Exception as e:
                    logger.warning(f"ResultsFetcher error for {sport} {date}: {e}")

        return settled_count

    def _settle_mma(self, sport_bets: list, broker, now: datetime) -> int:
        """
        Settle UFC bets. A card is one ESPN event with many fights (competitions);
        each fight is decided by a `winner` flag, not a score. We synthesize a
        1/0 score for the bet's home/away fighter so the broker's normal moneyline
        settlement applies. Draws / no-contests void the bet.
        """
        settled = 0
        url = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"

        days_back = 3
        for b in sport_bets:
            ct = b.get("commence_time")
            if ct:
                try:
                    start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    days_back = max(days_back, (now - start).days + 1)
                except Exception:
                    pass
        days_back = min(days_back, 30)

        for days_ago in range(days_back + 1):
            if not sport_bets:
                break
            date = (now - timedelta(days=days_ago)).strftime("%Y%m%d")
            try:
                resp = self.session.get(url, params={"dates": date}, timeout=10)
                resp.raise_for_status()
                for event in resp.json().get("events", []):
                    for comp in event.get("competitions", []):
                        competitors = comp.get("competitors", [])
                        if len(competitors) != 2:
                            continue
                        by_name = {c.get("athlete", {}).get("displayName", ""): c for c in competitors}
                        if "" in by_name:
                            continue
                        fight_names = {self._norm_fighter(n) for n in by_name}
                        matching = [
                            b for b in sport_bets
                            if {self._norm_fighter(b["home_team"]),
                                self._norm_fighter(b["away_team"])} == fight_names
                            and self._game_has_started(b.get("commence_time"), now)
                        ]
                        if not matching:
                            continue
                        b = matching[0]
                        game_id = b["game_id"]
                        status = comp.get("status", {}).get("type", {})

                        if status.get("completed"):
                            winner = next((n for n, c in by_name.items() if c.get("winner")), None)
                            if winner is None:   # draw or no-contest
                                voided = broker.void_bet(game_id=game_id, reason="draw_or_no_contest")
                                settled += len(voided)
                                logger.info(f"Voiding UFC fight (draw/NC): {b['home_team']} vs {b['away_team']}")
                            else:
                                home_won = self._norm_fighter(b["home_team"]) == self._norm_fighter(winner)
                                logger.info(f"Settling UFC: {winner} def. opponent [{b['home_team']} vs {b['away_team']}]")
                                done = broker.settle_bet(
                                    game_id=game_id,
                                    home_score=1 if home_won else 0,
                                    away_score=0 if home_won else 1,
                                )
                                settled += len(done)
                            sport_bets[:] = [x for x in sport_bets if x["game_id"] != game_id]
                        elif status.get("name") in self._VOID_STATUSES:
                            reason = status["name"].replace("STATUS_", "").lower()
                            voided = broker.void_bet(game_id=game_id, reason=reason)
                            settled += len(voided)
                            sport_bets[:] = [x for x in sport_bets if x["game_id"] != game_id]
            except Exception as e:
                logger.warning(f"ResultsFetcher MMA error for {date}: {e}")

        return settled

    @staticmethod
    def _norm_fighter(name: str) -> str:
        n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
        return " ".join("".join(c for c in n.lower() if c.isalnum() or c == " ").split())

    @staticmethod
    def _teams_match(bet_name: str, espn_name: str) -> bool:
        b = bet_name.lower().split()[-1]
        e = espn_name.lower().split()[-1]
        return b == e

    @staticmethod
    def _game_has_started(commence_time: str | None, now: datetime) -> bool:
        if not commence_time:
            return True
        try:
            start = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            return start <= now
        except Exception:
            return True

    @staticmethod
    def _result_date_matches(result_date: str, commence_time: str | None) -> bool:
        """
        Returns True if the ESPN result date matches the game's calendar date in ET.
        ESPN always records the local (ET) date, while commence_time is UTC, so we
        convert before comparing. We also allow the next ET day to handle games that
        run past midnight ET (e.g. extra innings, overtime).
        """
        if not commence_time:
            return True
        try:
            from zoneinfo import ZoneInfo
            ET = ZoneInfo("America/New_York")
            game_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            game_et_date = game_dt.astimezone(ET).date()
            next_et_date = game_et_date + timedelta(days=1)
            return result_date in (
                game_et_date.strftime("%Y-%m-%d"),
                next_et_date.strftime("%Y-%m-%d"),
            )
        except Exception:
            return True

    @staticmethod
    def _start_time_aligns(event_start: str | None, commence_time: str | None,
                           tol_hours: float = 2.0) -> bool:
        """Disambiguates DOUBLEHEADERS: two same-day games between the same teams both
        match on name + date, so name-only matching settles the wrong game (this is how a
        Reds 2-0 win got booked as a 5-6 loss). Require the ESPN event's start time to be
        within tol_hours of the bet's commence_time. Falls back to True when either time is
        missing so single games (with minor provider skew) still settle."""
        if not event_start or not commence_time:
            return True
        try:
            es = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
            cs = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            return abs((es - cs).total_seconds()) <= tol_hours * 3600
        except Exception:
            return True
