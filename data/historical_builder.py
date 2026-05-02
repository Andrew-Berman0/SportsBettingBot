"""
data/historical_builder.py
--------------------------
Builds a historical NBA training dataset by combining:
  1. nba_api   — game results + rolling team stats (per season game logs)
  2. ESPN API  — closing moneyline odds per game (free, no key)

Features added beyond raw stats:
  - L5 and L10 rolling windows for win% and point differential
  - Season-to-date win rate (expanding window)
  - Elo ratings (opponent-adjusted, resets partially each season)
  - Rest days / back-to-back flags

Output: data/raw/nba_training_dataset.parquet

Run:
  python data/historical_builder.py              # all seasons
  python data/historical_builder.py --fast       # 2023-24 only (quick test)

Estimated time: ~35 min first run, near-instant on reruns (fully cached).
Safe to interrupt and resume — everything is cached per date/game.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR   = Path(__file__).parent / "raw" / "historical"
OUTPUT_PATH = Path(__file__).parent / "raw" / "nba_training_dataset.parquet"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEASONS_FULL = ["2021-22", "2022-23", "2023-24", "2024-25"]
SEASONS_FAST = ["2023-24"]

ESPN = requests.Session()
ESPN.headers.update({"User-Agent": "Mozilla/5.0"})


# ---------------------------------------------------------------------------
# Phase 1: nba_api game logs
# ---------------------------------------------------------------------------

def fetch_game_logs(seasons: list[str]) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog

    frames = []
    for season in seasons:
        cache = CACHE_DIR / f"gamelog_{season}.parquet"
        if cache.exists():
            logger.info(f"  [{season}] loaded from cache")
            frames.append(pd.read_parquet(cache))
            continue

        logger.info(f"  [{season}] fetching from nba_api...")
        time.sleep(0.8)
        df = LeagueGameLog(season=season, direction="ASC").get_data_frames()[0]
        df["season"] = season
        df.to_parquet(cache)
        logger.info(f"    → {len(df):,} team-game rows")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 2: rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    logs = logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["is_home"]   = logs["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)
    logs["won"]       = (logs["WL"] == "W").astype(int)

    logs = logs.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    g = logs.groupby("TEAM_ID")

    # Shift(1) so the current game is never included in its own rolling window
    logs["win_pct_l10"]  = g["won"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    logs["win_pct_l5"]   = g["won"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    logs["avg_diff_l10"] = g["PLUS_MINUS"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    logs["avg_diff_l5"]  = g["PLUS_MINUS"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    # Season-to-date win rate before this game (expanding window)
    logs["season_w_pct"] = g["won"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    logs["prev_date"] = g["GAME_DATE"].transform(lambda x: x.shift(1))
    logs["rest_days"] = (logs["GAME_DATE"] - logs["prev_date"]).dt.days.clip(1, 14).fillna(7)
    logs["is_b2b"]    = (logs["rest_days"] == 1).astype(int)

    return logs


# ---------------------------------------------------------------------------
# Phase 3: pivot to one row per game
# ---------------------------------------------------------------------------

def pivot_to_games(logs: pd.DataFrame) -> pd.DataFrame:
    home = logs[logs["is_home"] == 1].copy()
    away = logs[logs["is_home"] == 0].copy()

    h_rename = {
        "TEAM_NAME": "home_team", "TEAM_ID": "home_team_id",
        "PTS": "home_pts", "PLUS_MINUS": "home_margin",
        "season_w_pct": "home_season_w_pct",
        "won": "home_won",
        "win_pct_l10": "home_win_pct_l10", "win_pct_l5": "home_win_pct_l5",
        "avg_diff_l10": "home_avg_diff_l10", "avg_diff_l5": "home_avg_diff_l5",
        "rest_days": "home_rest_days", "is_b2b": "home_is_b2b",
    }
    a_rename = {
        "TEAM_NAME": "away_team", "TEAM_ID": "away_team_id",
        "PTS": "away_pts", "PLUS_MINUS": "away_margin",
        "season_w_pct": "away_season_w_pct",
        "won": "away_won",
        "win_pct_l10": "away_win_pct_l10", "win_pct_l5": "away_win_pct_l5",
        "avg_diff_l10": "away_avg_diff_l10", "avg_diff_l5": "away_avg_diff_l5",
        "rest_days": "away_rest_days", "is_b2b": "away_is_b2b",
    }

    home = home.rename(columns=h_rename)
    away = away.rename(columns=a_rename)

    h_keep = ["GAME_ID", "GAME_DATE", "season"] + list(h_rename.values())
    a_keep = ["GAME_ID"] + list(a_rename.values())

    games = home[[c for c in h_keep if c in home.columns]].merge(
            away[[c for c in a_keep if c in away.columns]],
            on="GAME_ID")

    # Differential features
    games["win_pct_diff_l10"]  = games["home_win_pct_l10"]  - games["away_win_pct_l10"]
    games["win_pct_diff_l5"]   = games["home_win_pct_l5"]   - games["away_win_pct_l5"]
    games["avg_diff_diff_l10"] = games["home_avg_diff_l10"] - games["away_avg_diff_l10"]
    games["avg_diff_diff_l5"]  = games["home_avg_diff_l5"]  - games["away_avg_diff_l5"]
    games["rest_days_diff"]    = games["home_rest_days"]     - games["away_rest_days"]
    games["season_wpct_diff"]  = games["home_season_w_pct"] - games["away_season_w_pct"]

    return games.sort_values("GAME_DATE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 3.5: Elo ratings (opponent-adjusted strength)
# ---------------------------------------------------------------------------

def add_elo_ratings(games: pd.DataFrame,
                    k: float = 20.0,
                    home_bonus: float = 100.0,
                    season_regress: float = 0.75) -> pd.DataFrame:
    """
    Adds pre-game Elo for each team. Processes games chronologically.
    At each season boundary, Elos regress 25% toward 1500 to account for
    roster changes over the summer.
    """
    games = games.sort_values("GAME_DATE").reset_index(drop=True)
    elos: dict[int, float] = {}
    prev_season = None
    home_elos, away_elos = [], []

    for _, row in games.iterrows():
        season = row["season"]
        if prev_season is not None and season != prev_season:
            for tid in elos:
                elos[tid] = 1500.0 + (elos[tid] - 1500.0) * season_regress
        prev_season = season

        h_id  = int(row["home_team_id"])
        a_id  = int(row["away_team_id"])
        h_elo = elos.get(h_id, 1500.0)
        a_elo = elos.get(a_id, 1500.0)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Expected home win (home court bonus baked into Elo difference)
        exp_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - home_bonus) / 400.0))
        actual   = float(row["home_won"])
        elos[h_id] = h_elo + k * (actual - exp_home)
        elos[a_id] = a_elo + k * ((1.0 - actual) - (1.0 - exp_home))

    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    return games


# ---------------------------------------------------------------------------
# Phase 4: ESPN closing odds
# ---------------------------------------------------------------------------

def _espn_scoreboard(date_str: str) -> list[dict]:
    """Returns list of completed games for a YYYYMMDD date string."""
    cache = CACHE_DIR / f"sb_{date_str}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    time.sleep(0.3)
    r = ESPN.get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        params={"dates": date_str}, timeout=10
    )
    result = []
    for ev in r.json().get("events", []):
        comp = ev.get("competitions", [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        teams = {t["homeAway"]: t for t in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        result.append({
            "espn_id":   ev["id"],
            "home_team": teams["home"]["team"]["displayName"],
            "away_team": teams["away"]["team"]["displayName"],
        })

    with open(cache, "w") as f:
        json.dump(result, f)
    return result


def _espn_odds(espn_id: str) -> dict | None:
    """Returns {home_ml, away_ml} closing line or None."""
    cache = CACHE_DIR / f"odds_{espn_id}.json"
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
            return data if data else None  # {} is a cached miss

    time.sleep(0.3)
    url = (f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
           f"/events/{espn_id}/competitions/{espn_id}/odds")
    try:
        r = ESPN.get(url, timeout=10)
    except Exception:
        return None
    if r.status_code != 200:
        with open(cache, "w") as f:
            json.dump({}, f)
        return None

    items = r.json().get("items", [])
    for item in items:
        home_ml = item.get("homeTeamOdds", {}).get("moneyLine")
        away_ml = item.get("awayTeamOdds", {}).get("moneyLine")
        if home_ml is not None and away_ml is not None:
            result = {"home_ml": home_ml, "away_ml": away_ml}
            with open(cache, "w") as f:
                json.dump(result, f)
            return result

    with open(cache, "w") as f:
        json.dump({}, f)
    return None


def _implied(ml: float) -> float:
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def fetch_odds_for_games(games: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(games["GAME_DATE"].dt.strftime("%Y%m%d").unique())
    logger.info(f"  Fetching odds for {len(games):,} games across {len(dates)} dates...")

    odds_records = []
    for i, date_str in enumerate(dates):
        if i % 100 == 0:
            logger.info(f"    {i}/{len(dates)} dates processed...")

        espn_games = _espn_scoreboard(date_str)
        day_mask   = games["GAME_DATE"].dt.strftime("%Y%m%d") == date_str

        for _, row in games[day_mask].iterrows():
            home_nick = row["home_team"].split()[-1].lower()
            away_nick = row["away_team"].split()[-1].lower()

            match = next(
                (eg for eg in espn_games
                 if home_nick in eg["home_team"].lower()
                 and away_nick in eg["away_team"].lower()),
                None
            )
            if not match:
                continue

            odds = _espn_odds(match["espn_id"])
            if not odds:
                continue

            h_raw = _implied(odds["home_ml"])
            a_raw = _implied(odds["away_ml"])
            total = h_raw + a_raw

            odds_records.append({
                "GAME_ID":           row["GAME_ID"],
                "espn_id":           match["espn_id"],
                "home_ml":           odds["home_ml"],
                "away_ml":           odds["away_ml"],
                "home_implied_prob": round(h_raw / total, 4),
                "away_implied_prob": round(a_raw / total, 4),
            })

    if not odds_records:
        logger.warning("  No odds data found.")
        return games

    odds_df = pd.DataFrame(odds_records)
    merged  = games.merge(odds_df, on="GAME_ID", how="left")
    found   = merged["home_ml"].notna().sum()
    logger.info(f"  Odds matched: {found:,} / {len(games):,} games ({found/len(games):.1%})")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="One season only (quick test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    seasons = SEASONS_FAST if args.fast else SEASONS_FULL
    logger.info(f"=== NBA Historical Dataset Builder ===")
    logger.info(f"Seasons: {seasons}")

    logger.info("\nPhase 1/5 — nba_api game logs")
    logs = fetch_game_logs(seasons)
    logger.info(f"Total team-game rows: {len(logs):,}")

    logger.info("\nPhase 2/5 — rolling features (L5, L10, season)")
    logs = add_rolling_features(logs)

    logger.info("\nPhase 3/5 — pivot to game rows")
    games = pivot_to_games(logs)
    logger.info(f"Total games: {len(games):,}  |  Home win rate: {games['home_won'].mean():.3f}")

    logger.info("\nPhase 4/5 — Elo ratings")
    games = add_elo_ratings(games)
    logger.info(f"  Elo range: {games['home_elo'].min():.0f} – {games['home_elo'].max():.0f}")

    logger.info("\nPhase 5/5 — ESPN closing odds")
    games = fetch_odds_for_games(games)

    games.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\n✓ Saved → {OUTPUT_PATH}")
    logger.info(f"  Shape:      {games.shape}")
    logger.info(f"  Date range: {games['GAME_DATE'].min().date()} → {games['GAME_DATE'].max().date()}")
    logger.info(f"  Columns:    {list(games.columns)}")


if __name__ == "__main__":
    main()
