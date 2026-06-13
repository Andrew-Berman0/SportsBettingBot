"""
data/outcome_tracker.py
-----------------------
Logs the actual final score for every evaluated game — bets and passes alike.
Stored in game_outcomes.jsonl at the project root (one JSON record per line).

Not shown on the dashboard. Used offline to calibrate edge thresholds per sport:
  - What edge did Claude see?
  - Did the side Claude leaned on actually win?
  - At what edge level does Claude's directional call become reliable?
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTCOMES_FILE = Path(__file__).parent.parent / "game_outcomes.jsonl"

# How long after game start before we try to fetch a result.
# Gives enough time for the game to finish across all sports.
_MIN_HOURS_AFTER_START = {
    "basketball_nba":       3.5,
    "basketball_wnba":      3.5,
    "baseball_mlb":         4.0,
    "icehockey_nhl":        3.5,
    "americanfootball_nfl": 4.0,
    "mma_ufc":              6.0,   # a card's fights all share the start time but run ~6h
}
_DEFAULT_HOURS = 4.0


class OutcomeTracker:

    def __init__(self, results_fetcher):
        self._fetcher = results_fetcher
        self._logged_ids: set[str] = self._load_logged_ids()

    def update(self, broker) -> int:
        """
        Resolves final scores for any evaluated games not yet logged.
        Closed bets already carry the score; passed games are looked up via ESPN.
        Returns the number of new records written.
        """
        written = 0
        written += self._log_closed_bets(broker.closed_bets)
        written += self._log_passed_games(broker.passed_games)
        if written:
            logger.info(f"OutcomeTracker: logged {written} new game outcome(s)")
        return written

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _log_closed_bets(self, closed_bets: list) -> int:
        written = 0
        for bet in closed_bets:
            gid = bet["game_id"]
            if gid in self._logged_ids or bet.get("home_score") is None:
                continue
            chp = bet.get("claude_home_prob", 0.5)
            bhp = bet.get("book_home_prob", 0.5)
            home_won = bet["home_score"] > bet["away_score"]
            edge_side = "home" if bet["bet_type"] == "home_ml" else "away"
            self._write({
                "game_id":          gid,
                "sport":            bet["sport"],
                "home_team":        bet["home_team"],
                "away_team":        bet["away_team"],
                "commence_time":    bet.get("commence_time"),
                "claude_home_prob": chp,
                "book_home_prob":   bhp,
                "home_edge":        round(chp - bhp, 4),
                "away_edge":        round((1 - chp) - (1 - bhp), 4),
                "bet_placed":       True,
                "bet_type":         bet["bet_type"],
                "bet_odds":         bet.get("odds"),
                "home_score":       bet["home_score"],
                "away_score":       bet["away_score"],
                "home_won":         home_won,
                "edge_side":        edge_side,
                "claude_correct":   (edge_side == "home") == home_won,
                "analyst_version":  bet.get("analyst_version"),
            })
            written += 1
        return written

    def _log_passed_games(self, passed_games: list) -> int:
        now = datetime.now(timezone.utc)
        pending_by_sport: dict[str, list[dict]] = defaultdict(list)

        for g in passed_games:
            gid = g["game_id"]
            if gid in self._logged_ids:
                continue
            ct = g.get("commence_time")
            if not ct:
                continue
            try:
                start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                min_wait = _MIN_HOURS_AFTER_START.get(g.get("sport", ""), _DEFAULT_HOURS)
                if now < start + timedelta(hours=min_wait):
                    continue  # game may not be finished yet
            except Exception:
                continue
            pending_by_sport[g["sport"]].append(g)

        written = 0
        for sport, games in pending_by_sport.items():
            completed = self._fetcher.get_completed_games(days_back=3, sport=sport)
            score_map = {
                (r["home_team"].lower().split()[-1],
                 r["away_team"].lower().split()[-1]): r
                for r in completed
            }
            for g in games:
                key = (g["home_team"].lower().split()[-1],
                       g["away_team"].lower().split()[-1])
                result = score_map.get(key)
                if not result:
                    continue

                home_won = result["home_score"] > result["away_score"]
                he = g.get("home_edge", 0)
                ae = g.get("away_edge", 0)

                if he > 0:
                    edge_side = "home"
                    correct: bool | None = home_won
                elif ae > 0:
                    edge_side = "away"
                    correct = not home_won
                else:
                    edge_side = None   # Claude was below market on both sides
                    correct = None

                self._write({
                    "game_id":          g["game_id"],
                    "sport":            g["sport"],
                    "home_team":        g["home_team"],
                    "away_team":        g["away_team"],
                    "commence_time":    g.get("commence_time"),
                    "claude_home_prob": g.get("claude_home_prob"),
                    "book_home_prob":   g.get("book_home_prob"),
                    "home_edge":        round(he, 4),
                    "away_edge":        round(ae, 4),
                    "bet_placed":       False,
                    "bet_type":         None,
                    "bet_odds":         None,
                    "home_score":       result["home_score"],
                    "away_score":       result["away_score"],
                    "home_won":         home_won,
                    "edge_side":        edge_side,
                    "claude_correct":   correct,
                    "analyst_version":  g.get("analyst_version"),
                })
                written += 1
        return written

    def _write(self, record: dict) -> None:
        record["logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(OUTCOMES_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._logged_ids.add(record["game_id"])

    def _load_logged_ids(self) -> set[str]:
        if not OUTCOMES_FILE.exists():
            return set()
        ids = set()
        with open(OUTCOMES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["game_id"])
                except Exception:
                    pass
        return ids
