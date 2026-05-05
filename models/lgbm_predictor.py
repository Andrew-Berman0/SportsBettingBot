"""
models/lgbm_predictor.py
------------------------
Wraps the saved LightGBM model for use in the live bot.

Usage:
    predictor = LGBMPredictor.load()
    if predictor:
        prob = predictor.predict(home_team, away_team, home_stats, away_stats,
                                 home_implied_prob, away_implied_prob)
"""

import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "saved" / "lgbm_betting_model.pkl"


class LGBMPredictor:

    def __init__(self, model, feature_cols: list[str],
                 team_elos: dict, team_name_to_id: dict):
        self.model           = model
        self.feature_cols    = feature_cols
        self.team_elos       = team_elos        # {team_id: elo_float}
        self.team_name_to_id = team_name_to_id  # {team_name: team_id}

    @classmethod
    def load(cls) -> "LGBMPredictor | None":
        """Returns a loaded predictor, or None if the model file doesn't exist yet."""
        if not MODEL_PATH.exists():
            logger.info("LGBMPredictor: no saved model found — run lgbm_trainer.py --retrain-all")
            return None
        try:
            with open(MODEL_PATH, "rb") as f:
                art = pickle.load(f)
            logger.info(f"LGBMPredictor loaded ({len(art['feature_cols'])} features, "
                        f"{len(art['team_elos'])} teams with Elo)")
            return cls(art["model"], art["feature_cols"],
                       art["team_elos"], art["team_name_to_id"])
        except Exception as e:
            logger.warning(f"LGBMPredictor failed to load: {e}")
            return None

    def predict(self, home_team: str, away_team: str,
                home_stats: dict, away_stats: dict,
                home_implied_prob: float, away_implied_prob: float) -> float:
        """
        Returns the model's home win probability (float 0–1).
        Falls back to home_implied_prob if a feature can't be computed.
        """
        try:
            h_id  = self._team_id(home_team)
            a_id  = self._team_id(away_team)
            h_elo = self.team_elos.get(h_id, 1500.0) if h_id else 1500.0
            a_elo = self.team_elos.get(a_id, 1500.0) if a_id else 1500.0

            feat = {
                # Season strength
                "home_season_w_pct":  home_stats.get("W_PCT", 0.5),
                "away_season_w_pct":  away_stats.get("W_PCT", 0.5),
                "season_wpct_diff":   home_stats.get("W_PCT", 0.5) - away_stats.get("W_PCT", 0.5),
                # Elo
                "home_elo":           h_elo,
                "away_elo":           a_elo,
                "elo_diff":           h_elo - a_elo,
                # L10 form
                "home_win_pct_l10":   home_stats.get("win_pct_l10", 0.5),
                "away_win_pct_l10":   away_stats.get("win_pct_l10", 0.5),
                "win_pct_diff_l10":   home_stats.get("win_pct_l10", 0.5) - away_stats.get("win_pct_l10", 0.5),
                "home_avg_diff_l10":  home_stats.get("avg_diff_l10", 0.0),
                "away_avg_diff_l10":  away_stats.get("avg_diff_l10", 0.0),
                "avg_diff_diff_l10":  home_stats.get("avg_diff_l10", 0.0) - away_stats.get("avg_diff_l10", 0.0),
                # L5 form
                "home_win_pct_l5":    home_stats.get("win_pct_l5", 0.5),
                "away_win_pct_l5":    away_stats.get("win_pct_l5", 0.5),
                "win_pct_diff_l5":    home_stats.get("win_pct_l5", 0.5) - away_stats.get("win_pct_l5", 0.5),
                "home_avg_diff_l5":   home_stats.get("avg_diff_l5", 0.0),
                "away_avg_diff_l5":   away_stats.get("avg_diff_l5", 0.0),
                "avg_diff_diff_l5":   home_stats.get("avg_diff_l5", 0.0) - away_stats.get("avg_diff_l5", 0.0),
                # Rest / fatigue
                "home_rest_days":     home_stats.get("rest_days", 3),
                "away_rest_days":     away_stats.get("rest_days", 3),
                "rest_days_diff":     home_stats.get("rest_days", 3) - away_stats.get("rest_days", 3),
                "home_is_b2b":        home_stats.get("is_back_to_back", 0),
                "away_is_b2b":        away_stats.get("is_back_to_back", 0),
                # Market prior
                "home_implied_prob":  home_implied_prob,
                "away_implied_prob":  away_implied_prob,
                # Game context — NBA playoffs run mid-April through mid-June
                "is_playoff":         int(datetime.today().month in (4, 5, 6)),
            }

            X = np.array([[feat.get(c, 0.0) for c in self.feature_cols]])
            prob = float(self.model.predict_proba(X)[0, 1])
            return max(0.05, min(0.95, prob))

        except Exception as e:
            logger.warning(f"LGBMPredictor.predict failed: {e} — falling back to market prob")
            return home_implied_prob

    def _team_id(self, team_name: str) -> int | None:
        if team_name in self.team_name_to_id:
            return self.team_name_to_id[team_name]
        nick = team_name.lower().split()[-1]
        for name, tid in self.team_name_to_id.items():
            if nick in name.lower():
                return tid
        return None
