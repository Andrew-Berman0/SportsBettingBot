"""
data/results_fetcher.py
-----------------------
Fetches completed game scores from ESPN's unofficial scoreboard API.
Used to auto-settle open bets and label training data.
No API key required.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

SPORT_ESPN_MAP = {
    "basketball_nba":       ("basketball", "nba"),
    "basketball_wnba":      ("basketball", "wnba"),
    "baseball_mlb":         ("baseball",   "mlb"),
    "icehockey_nhl":        ("hockey",     "nhl"),
    "americanfootball_nfl": ("football",   "nfl"),
}


class ResultsFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_completed_games(self, days_back: int = 2,
                            sport: str = "basketball_nba") -> list[dict]:
        """
        Returns completed games from the past `days_back` days for the given sport.
        Each entry: {home_team, away_team, home_score, away_score, date}
        """
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

    def settle_open_bets(self, broker) -> int:
        """
        Checks ESPN for completed games, settles any matching open bets.
        Handles all sports that have open bets.
        Returns the number of bets settled.
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

            completed = self.get_completed_games(days_back=days_back, sport=sport)
            if not completed:
                continue

            for result in completed:
                matching = [
                    b for b in sport_bets
                    if self._teams_match(b["home_team"], result["home_team"])
                    and self._teams_match(b["away_team"], result["away_team"])
                    and self._game_has_started(b.get("commence_time"), now)
                    and self._result_date_matches(result["date"], b.get("commence_time"))
                ]
                if not matching:
                    continue

                game_id = matching[0]["game_id"]
                logger.info(
                    f"Settling {result['away_team']} @ {result['home_team']} "
                    f"({result['away_score']}-{result['home_score']}) [{sport}]"
                )
                settled = broker.settle_bet(
                    game_id=game_id,
                    home_score=result["home_score"],
                    away_score=result["away_score"],
                )
                settled_count += len(settled)

        return settled_count

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
