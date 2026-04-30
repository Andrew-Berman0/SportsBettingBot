"""
features/engineer.py
--------------------
Builds a feature vector for each upcoming game.

Features per game:
  - Odds-derived: implied probabilities, line movement proxy, juice
  - Team stats:   offensive/defensive rating, pace, net rating, win%
  - Form:         last-N win%, avg point diff, back-to-back, rest days
  - Matchup:      home advantage, rating differential
  - Time:         day of week, time of day, days into season
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:

    def build_game_features(self, game: dict, home_stats: dict, away_stats: dict) -> dict:
        """
        Builds a flat feature dict for a single game.
        game        : parsed game dict from OddsFetcher.parse_game()
        home_stats  : team stats dict for home team
        away_stats  : team stats dict for away team
        Returns a dict of feature_name → float.
        """
        features = {}

        # --- Odds features ---
        home_impl = game.get("home_implied") or 0.5
        away_impl = game.get("away_implied") or 0.5
        # Remove bookmaker juice to get true implied probs
        total_impl = home_impl + away_impl
        features["home_implied_prob"] = home_impl / total_impl
        features["away_implied_prob"] = away_impl / total_impl
        features["bookmaker_juice"]   = total_impl - 1.0   # vig

        home_ml = game.get("home_ml") or 0
        away_ml = game.get("away_ml") or 0
        features["home_is_favorite"] = int(home_ml < 0)
        features["favorite_strength"] = abs(home_ml - away_ml) / 100

        if game.get("total_line"):
            features["total_line"] = float(game["total_line"])
        else:
            features["total_line"] = np.nan

        # --- Team stat differentials (home - away) ---
        stat_pairs = [
            ("off_rating",   "OFF_RATING"),
            ("def_rating",   "DEF_RATING"),
            ("net_rating",   "NET_RATING"),
            ("pace",         "PACE"),
            ("win_pct",      "W_PCT"),
            ("ts_pct",       "TS_PCT"),
        ]
        for feat_name, stat_key in stat_pairs:
            h_val = home_stats.get(stat_key)
            a_val = away_stats.get(stat_key)
            if h_val is not None and a_val is not None:
                features[f"{feat_name}_diff"] = float(h_val) - float(a_val)
                features[f"home_{feat_name}"] = float(h_val)
                features[f"away_{feat_name}"] = float(a_val)
            else:
                features[f"{feat_name}_diff"] = np.nan
                features[f"home_{feat_name}"] = np.nan
                features[f"away_{feat_name}"] = np.nan

        # --- Recent form ---
        for prefix, stats in [("home", home_stats), ("away", away_stats)]:
            features[f"{prefix}_win_pct_l10"] = stats.get("win_pct_l10", np.nan)
            features[f"{prefix}_avg_diff_l10"] = stats.get("avg_diff_l10", np.nan)
            features[f"{prefix}_is_b2b"]       = stats.get("is_back_to_back", 0)
            features[f"{prefix}_rest_days"]    = stats.get("rest_days", 2)

        features["rest_days_diff"] = (
            features["home_rest_days"] - features["away_rest_days"]
        )

        # --- Home advantage (always present) ---
        features["home_advantage"] = 1.0

        # --- Time features ---
        try:
            from datetime import datetime, timezone
            commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
            features["day_of_week"] = commence.weekday()
            features["hour_of_day"] = commence.hour
            features["is_weekend"]  = int(commence.weekday() >= 5)
        except Exception:
            features["day_of_week"] = np.nan
            features["hour_of_day"] = np.nan
            features["is_weekend"]  = np.nan

        return features

    def features_to_df(self, feature_list: list[dict]) -> pd.DataFrame:
        """Converts a list of feature dicts to a DataFrame, filling NaNs."""
        df = pd.DataFrame(feature_list)
        # Fill NaN with column medians for model input
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        return df
