"""
models/lgbm_trainer.py
-----------------------
Trains a LightGBM model to predict NBA home-win probability.

Walk-forward split: train 2021-22/2022-23, evaluate 2023-24.
Includes a simulated Kelly-bet ROI on the test set so you can see
whether the model actually adds edge over the closing line.

Run:
  python models/lgbm_trainer.py               # evaluate only
  python models/lgbm_trainer.py --retrain-all # retrain on all data, save final model
  python models/lgbm_trainer.py --no-odds     # debug: exclude market odds from features
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb

logger = logging.getLogger(__name__)

DATA_PATH  = Path(__file__).parent.parent / "data" / "raw" / "nba_training_dataset.parquet"
SAVE_DIR   = Path(__file__).parent / "saved"
MODEL_PATH = SAVE_DIR / "lgbm_betting_model.pkl"
SAVE_DIR.mkdir(exist_ok=True)

# Columns used as model inputs — all are known BEFORE tip-off
FEATURE_COLS = [
    # Season-level strength
    "home_season_w_pct",   "away_season_w_pct",   "season_wpct_diff",
    # Elo (opponent-adjusted rating)
    "home_elo",            "away_elo",             "elo_diff",
    # Recent form — L10 and L5
    "home_win_pct_l10",    "away_win_pct_l10",     "win_pct_diff_l10",
    "home_win_pct_l5",     "away_win_pct_l5",      "win_pct_diff_l5",
    "home_avg_diff_l10",   "away_avg_diff_l10",    "avg_diff_diff_l10",
    "home_avg_diff_l5",    "away_avg_diff_l5",     "avg_diff_diff_l5",
    # Rest / fatigue
    "home_rest_days",      "away_rest_days",        "rest_days_diff",
    "home_is_b2b",         "away_is_b2b",
    # Market closing line (strong prior)
    "home_implied_prob",   "away_implied_prob",
]
TARGET        = "home_won"
TRAIN_SEASONS = ["2021-22", "2022-23"]
TEST_SEASONS  = ["2023-24"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(require_odds: bool = True) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python data/historical_builder.py"
        )

    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows | seasons: {dict(df['season'].value_counts().sort_index())}")

    if require_odds:
        df = df[df["home_implied_prob"].notna()].copy()
        logger.info(f"  After odds filter:  {len(df):,} rows")

    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + [TARGET])
    logger.info(f"  After NaN drop:     {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        num_leaves=15,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


def train(df: pd.DataFrame, feat_cols: list[str], seasons: list[str]):
    subset = df[df["season"].isin(seasons)]
    X = subset[feat_cols].values
    y = subset[TARGET].values
    logger.info(f"Training on {seasons}: {len(subset):,} rows, home-win rate={y.mean():.3f}")
    # Isotonic calibration via 5-fold CV — aligns model probabilities with true frequencies
    # so that edge = (model_prob - market_implied_prob) is meaningful
    calibrated = CalibratedClassifierCV(make_model(), cv=5, method="isotonic")
    calibrated.fit(X, y)
    return calibrated


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 time-ordered split used when only one season is available."""
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    cut = int(len(df) * 0.70)
    return df.iloc[:cut], df.iloc[cut:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_seasons(model: lgb.LGBMClassifier, df: pd.DataFrame,
                      feat_cols: list[str], test_seasons: list[str]) -> None:
    test_df = df[df["season"].isin(test_seasons)].reset_index(drop=True)
    if test_df.empty:
        logger.warning("No test rows — skipping evaluation")
        return
    _evaluate_df(model, test_df, feat_cols, label=str(test_seasons))


def _evaluate_df(model: lgb.LGBMClassifier, test_df: pd.DataFrame,
                 feat_cols: list[str], label: str = "") -> None:
    X = test_df[feat_cols].values
    y = test_df[TARGET].values
    probs = model.predict_proba(X)[:, 1]

    auc        = roc_auc_score(y, probs)
    brier      = brier_score_loss(y, probs)
    ll         = log_loss(y, probs)
    base_brier = float(np.mean((y.mean() - y) ** 2))

    logger.info(f"\n{'─'*42}")
    logger.info(f"Test metrics  {label}")
    logger.info(f"{'─'*42}")
    logger.info(f"  AUC-ROC:     {auc:.4f}  (0.5 = random)")
    logger.info(f"  Brier:       {brier:.4f}  (baseline={base_brier:.4f})")
    logger.info(f"  Log-loss:    {ll:.4f}")

    _print_calibration(probs, y)
    _print_feature_importance(model, feat_cols)
    _simulate_roi(probs, test_df)


def _print_calibration(probs: np.ndarray, y: np.ndarray, n_bins: int = 5) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    logger.info(f"\n  Calibration (predicted → actual):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        actual = y[mask].mean()
        pred   = probs[mask].mean()
        logger.info(f"    {lo:.1f}–{hi:.1f}:  pred={pred:.3f}  actual={actual:.3f}  n={mask.sum()}")


def _print_feature_importance(model, feat_cols: list[str]) -> None:
    # CalibratedClassifierCV wraps estimators — average importances across folds
    if hasattr(model, "calibrated_classifiers_"):
        importances = np.mean(
            [c.estimator.feature_importances_ for c in model.calibrated_classifiers_], axis=0
        )
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return
    pairs = sorted(zip(feat_cols, importances), key=lambda x: -x[1])
    logger.info(f"\n  Feature importances (top 10):")
    for feat, imp in pairs[:10]:
        bar = "█" * int(imp / max(imp for _, imp in pairs) * 20)
        logger.info(f"    {feat:<26} {bar}  {imp:.0f}")


def _simulate_roi(probs: np.ndarray, test_df: pd.DataFrame,
                  edge_thresh: float = 0.04, kelly_frac: float = 0.25) -> None:
    bankroll  = 1000.0
    start     = bankroll
    bets      = wins = 0
    pnl_per_bet: list[float] = []

    for i, p_home in enumerate(probs):
        row      = test_df.iloc[i]
        p_away   = 1.0 - p_home
        h_imp    = row.get("home_implied_prob", np.nan)
        a_imp    = row.get("away_implied_prob", np.nan)
        h_ml     = row.get("home_ml", np.nan)
        a_ml     = row.get("away_ml", np.nan)
        won_home = bool(row[TARGET])

        for p, imp, ml, outcome in [
            (p_home, h_imp, h_ml, won_home),
            (p_away, a_imp, a_ml, not won_home),
        ]:
            if np.isnan(imp) or np.isnan(ml):
                continue
            edge = p - imp
            if edge < edge_thresh:
                continue
            b = ml / 100 if ml > 0 else 100 / abs(ml)
            k = max(0.0, (b * p - (1 - p)) / b) * kelly_frac
            stake = min(bankroll * k, bankroll)
            bankroll -= stake
            if outcome:
                pnl = stake * b
                bankroll += stake + pnl
                wins += 1
            else:
                pnl = -stake
            pnl_per_bet.append(pnl)
            bets += 1
            break  # only one side per game

    roi = (bankroll - start) / start * 100

    # Sharpe: mean P&L / std P&L, annualized assuming test set ≈ 1 NBA season
    if len(pnl_per_bet) > 1 and np.std(pnl_per_bet) > 0:
        sharpe = (np.mean(pnl_per_bet) / np.std(pnl_per_bet)) * np.sqrt(len(pnl_per_bet))
    else:
        sharpe = 0.0

    logger.info(f"\n  Simulated betting (edge≥{edge_thresh:.0%}, {kelly_frac:.0%} Kelly):")
    if bets:
        logger.info(f"    Bets: {bets}  |  Wins: {wins}  |  Win-rate: {wins/bets:.1%}")
        logger.info(f"    Start: ${start:,.2f}  →  End: ${bankroll:,.2f}  |  ROI: {roi:+.1f}%")
        logger.info(f"    Sharpe: {sharpe:.2f}  (>0.5 = solid, >1.0 = exceptional)")
    else:
        logger.info(f"    No bets placed (try lowering --edge-thresh)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-all", action="store_true",
                        help="Retrain on all seasons after evaluation and save final model")
    parser.add_argument("--no-odds", action="store_true",
                        help="Exclude closing-line implied-prob features (shows raw model value)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    feat_cols = [c for c in FEATURE_COLS if not (args.no_odds and "implied" in c)]
    require_odds = not args.no_odds

    df = load_data(require_odds=require_odds)

    available = sorted(df["season"].unique().tolist())
    train_seasons = [s for s in TRAIN_SEASONS if s in available]
    test_seasons  = [s for s in TEST_SEASONS  if s in available]

    if not train_seasons:
        # Only one season available (e.g. --fast run) — use a 70/30 chronological split
        logger.info(f"Only season(s) available: {available} — using 70/30 chronological split")
        train_df, test_df = chronological_split(df)
        X_train = train_df[feat_cols].values
        y_train = train_df[TARGET].values
        logger.info(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
        model = make_model()
        model.fit(X_train, y_train)
        # Patch evaluate to use our pre-split test_df
        _evaluate_df(model, test_df, feat_cols, label=f"{available} (70/30 split)")
    else:
        model = train(df, feat_cols, train_seasons)
        _evaluate_seasons(model, df, feat_cols, test_seasons)

    if args.retrain_all:
        all_seasons = sorted(df["season"].unique().tolist())
        model = train(df, feat_cols, all_seasons)

        # Extract latest pre-game Elo per team from the full raw dataset
        df_raw = pd.read_parquet(DATA_PATH)
        records = []
        for _, row in df_raw.iterrows():
            records.append({"team_id": int(row["home_team_id"]), "name": row["home_team"],
                             "elo": row["home_elo"], "date": row["GAME_DATE"]})
            records.append({"team_id": int(row["away_team_id"]), "name": row["away_team"],
                             "elo": row["away_elo"], "date": row["GAME_DATE"]})
        rec_df = pd.DataFrame(records).sort_values("date")
        latest = rec_df.groupby("team_id").last()
        team_elos       = latest["elo"].to_dict()
        team_name_to_id = {row["name"]: int(tid) for tid, row in latest.iterrows()}

        artifact = {
            "model":           model,
            "feature_cols":    feat_cols,
            "team_elos":       team_elos,
            "team_name_to_id": team_name_to_id,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"\n✓ Final model saved → {MODEL_PATH}")
        logger.info(f"  Trained on:  {all_seasons}")
        logger.info(f"  Features:    {feat_cols}")
        logger.info(f"  Teams with Elo: {len(team_elos)}")
    else:
        logger.info(f"\n(Run with --retrain-all to save a production model)")


if __name__ == "__main__":
    main()
