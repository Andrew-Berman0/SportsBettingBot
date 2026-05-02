"""
data/results_fetcher.py
-----------------------
Fetches completed NBA game scores from ESPN's unofficial scoreboard API.
Used to auto-settle open bets and label training data.
No API key required.
"""

import logging
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


class ResultsFetcher:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_completed_games(self, days_back: int = 2) -> list[dict]:
        """
        Returns a list of completed games from the past `days_back` days.
        Each entry:
            {
                "home_team": str,
                "away_team": str,
                "home_score": int,
                "away_score": int,
                "date": str,   # YYYY-MM-DD
            }
        """
        results = []
        for days_ago in range(days_back + 1):
            date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")
            try:
                resp = self.session.get(ESPN_URL, params={"dates": date}, timeout=10)
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
                logger.warning(f"ResultsFetcher error for {date}: {e}")
        return results

    def settle_open_bets(self, broker) -> int:
        """
        Checks ESPN for completed games, settles any matching open bets.
        Returns the number of bets settled.
        """
        if not broker.open_bets:
            return 0

        completed = self.get_completed_games(days_back=3)
        if not completed:
            return 0

        settled_count = 0
        for result in completed:
            matching_bets = [
                b for b in broker.open_bets
                if self._teams_match(b["home_team"], result["home_team"])
                and self._teams_match(b["away_team"], result["away_team"])
            ]
            if not matching_bets:
                continue

            game_id = matching_bets[0]["game_id"]
            logger.info(
                f"Settling {result['away_team']} @ {result['home_team']} "
                f"({result['away_score']}-{result['home_score']})"
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
        """Fuzzy match on the team nickname (last word)."""
        b = bet_name.lower().split()[-1]
        e = espn_name.lower().split()[-1]
        return b == e
