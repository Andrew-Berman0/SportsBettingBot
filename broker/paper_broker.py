"""
broker/paper_broker.py
----------------------
Simulates a paper betting bankroll.
Tracks open bets, results, P&L, and ROI.
Persists state to JSON so it survives restarts.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent / "broker_state.json"


class PaperBroker:

    def __init__(self, starting_bankroll: float = 10000.0):
        self.starting_bankroll = starting_bankroll
        self.bankroll   = starting_bankroll
        self.open_bets  = []   # list of bet dicts
        self.closed_bets = []  # settled bets
        self._load()

    # ------------------------------------------------------------------
    # Betting
    # ------------------------------------------------------------------

    def place_bet(
        self,
        game_id:         str,
        sport:           str,
        home_team:       str,
        away_team:       str,
        bet_type:        str,    # "home_ml" | "away_ml" | "over" | "under"
        odds:            float,  # American odds
        stake:           float,  # USD amount to bet
        reasoning:       str = "",
        claude_home_prob: float | None = None,   # our estimated home win prob
        book_home_prob:   float | None = None,   # bookmaker's implied home prob
        features:         dict | None = None,    # feature snapshot for model training
        commence_time:    str | None = None,     # ISO-8601 game start time
    ) -> dict:
        """Place a paper bet. Returns the bet record."""
        if stake > self.bankroll:
            stake = self.bankroll
            logger.warning(f"Stake reduced to bankroll: ${stake:.2f}")

        bet = {
            "bet_id":      f"{game_id}_{bet_type}",
            "game_id":     game_id,
            "sport":       sport,
            "home_team":   home_team,
            "away_team":   away_team,
            "bet_type":    bet_type,
            "odds":        odds,
            "stake":       stake,
            "potential_payout": self._calc_payout(stake, odds),
            "placed_at":   datetime.now(timezone.utc).isoformat(),
            "commence_time": commence_time,
            "status":      "open",
            "reasoning":   reasoning,
            "claude_home_prob": claude_home_prob,
            "book_home_prob":   book_home_prob,
            "features":        features or {},
        }
        self.bankroll -= stake
        self.open_bets.append(bet)
        self._save()

        logger.info(
            f"BET PLACED: {away_team} @ {home_team} | {bet_type} {odds:+.0f} | "
            f"Stake: ${stake:.2f} | To win: ${bet['potential_payout'] - stake:.2f}"
        )
        return bet

    def settle_bet(self, game_id: str, home_score: int, away_score: int, total: float | None = None):
        """Settle all open bets for a game given the final score."""
        settled = []
        remaining = []
        for bet in self.open_bets:
            if bet["game_id"] != game_id:
                remaining.append(bet)
                continue

            won = self._evaluate_bet(bet, home_score, away_score, total)
            bet["status"]     = "won" if won else "lost"
            bet["settled_at"] = datetime.now(timezone.utc).isoformat()
            bet["home_score"] = home_score
            bet["away_score"] = away_score
            bet["home_won"]   = home_score > away_score   # training label
            if won:
                self.bankroll += bet["potential_payout"]
                bet["pnl"] = bet["potential_payout"] - bet["stake"]
            else:
                bet["pnl"] = -bet["stake"]

            logger.info(
                f"BET SETTLED: {bet['away_team']} @ {bet['home_team']} | "
                f"{bet['bet_type']} | {'WON' if won else 'LOST'} | P&L: ${bet['pnl']:+.2f}"
            )
            settled.append(bet)

        self.open_bets    = remaining
        self.closed_bets.extend(settled)
        self._save()
        return settled

    def kelly_stake(self, our_prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        Quarter Kelly criterion for bet sizing.
        Returns recommended stake in USD.
        """
        impl_prob = self._american_to_implied(odds)
        if our_prob <= impl_prob:
            return 0.0

        if odds > 0:
            b = odds / 100
        else:
            b = 100 / abs(odds)

        kelly = (b * our_prob - (1 - our_prob)) / b
        kelly = max(0.0, kelly) * fraction
        return round(self.bankroll * kelly, 2)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        total_staked  = sum(b["stake"] for b in self.closed_bets)
        total_pnl     = sum(b["pnl"] for b in self.closed_bets)
        wins          = sum(1 for b in self.closed_bets if b["status"] == "won")
        total_closed  = len(self.closed_bets)
        roi           = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

        return {
            "bankroll":       round(self.bankroll, 2),
            "starting":       self.starting_bankroll,
            "total_pnl":      round(total_pnl, 2),
            "roi_pct":        round(roi, 2),
            "win_rate":       round(wins / total_closed, 3) if total_closed > 0 else 0.0,
            "total_bets":     total_closed,
            "open_bets":      len(self.open_bets),
            "wins":           wins,
            "losses":         total_closed - wins,
        }

    def export_training_data(self, path: str = "training_data.csv") -> int:
        """
        Exports closed bets with features + outcome to a CSV for model training.
        Each row = one bet with full feature snapshot and home_won label.
        Returns number of rows written.
        """
        import csv
        rows = [b for b in self.closed_bets if b.get("features") and "home_won" in b]
        if not rows:
            logger.info("No closed bets with features yet — nothing to export.")
            return 0

        # Collect all feature keys across all rows
        feature_keys = sorted({k for b in rows for k in b["features"].keys()})
        meta_keys = ["game_id", "sport", "home_team", "away_team", "bet_type",
                     "odds", "placed_at", "claude_home_prob", "book_home_prob", "home_won"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=meta_keys + feature_keys, extrasaction="ignore")
            writer.writeheader()
            for b in rows:
                row = {k: b.get(k, "") for k in meta_keys}
                row.update({k: b["features"].get(k, "") for k in feature_keys})
                writer.writerow(row)

        logger.info(f"Exported {len(rows)} training rows → {path}")
        return len(rows)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_payout(stake: float, odds: float) -> float:
        if odds > 0:
            return stake + stake * (odds / 100)
        else:
            return stake + stake * (100 / abs(odds))

    @staticmethod
    def _american_to_implied(odds: float) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def _evaluate_bet(bet: dict, home_score: int, away_score: int, total: float | None) -> bool:
        bet_type = bet["bet_type"]
        if bet_type == "home_ml":
            return home_score > away_score
        elif bet_type == "away_ml":
            return away_score > home_score
        elif bet_type == "over" and total is not None:
            return (home_score + away_score) > total
        elif bet_type == "under" and total is not None:
            return (home_score + away_score) < total
        return False

    def _save(self):
        state = {
            "bankroll":    self.bankroll,
            "open_bets":   self.open_bets,
            "closed_bets": self.closed_bets,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.bankroll    = state.get("bankroll", self.starting_bankroll)
            self.open_bets   = state.get("open_bets", [])
            self.closed_bets = state.get("closed_bets", [])
            logger.info(
                f"PaperBroker loaded — Bankroll: ${self.bankroll:,.2f} | "
                f"Open: {len(self.open_bets)} | Closed: {len(self.closed_bets)}"
            )
        else:
            logger.info(f"PaperBroker initialized — Bankroll: ${self.bankroll:,.2f}")
