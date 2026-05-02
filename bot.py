"""
bot.py
------
Main loop for the sports betting bot.

Every hour:
  1. Check which sport is currently in season with upcoming games
  2. Fetch odds for upcoming games
  3. Fetch team stats + recent form
  4. Build features for each game
  5. Ask Claude to analyze each game and estimate win probability
  6. Compare our probability to bookmaker implied probability
  7. If edge > threshold → place paper bet (Kelly-sized)
  8. Settle any bets from games that have finished

Run:
  python bot.py
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG
from data.odds_fetcher import OddsFetcher
from data.stats_fetcher import NBAStatsFetcher, ESPNStatsFetcher
from data.injury_fetcher import InjuryFetcher
from data.roster_fetcher import RosterFetcher
from data.results_fetcher import ResultsFetcher
from features.engineer import FeatureEngineer
from models.claude_analyst import ClaudeAnalyst
from models.lgbm_predictor import LGBMPredictor
from broker.paper_broker import PaperBroker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ]
)
logger = logging.getLogger("bot")


def get_team_stats(sport: str, team_name: str, nba_fetcher: NBAStatsFetcher,
                   espn_fetcher: ESPNStatsFetcher, nba_stats_df=None) -> dict:
    """Returns stat dict for a team depending on sport."""
    if sport == "basketball_nba" and nba_stats_df is not None and not nba_stats_df.empty:
        row = nba_stats_df[nba_stats_df["TEAM_NAME"].str.contains(
            team_name.split()[-1], case=False, na=False
        )]
        if not row.empty:
            stats = row.iloc[0].to_dict()
            team_id = int(stats.get("TEAM_ID", 0))
            if team_id:
                form = nba_fetcher.get_recent_form(team_id)
                stats.update(form)
            return stats
    return {}


def evaluate_game(game_raw: dict, sport: str, nba_fetcher: NBAStatsFetcher,
                  espn_fetcher: ESPNStatsFetcher, injury_fetcher: InjuryFetcher,
                  roster_fetcher: RosterFetcher, nba_stats_df,
                  engineer: FeatureEngineer, claude: ClaudeAnalyst,
                  broker: PaperBroker, lgbm: LGBMPredictor | None = None) -> None:
    """Run the full analysis pipeline for a single game and place bets if value found."""
    game = OddsFetcher.parse_game(game_raw)
    if not game:
        return

    # Skip games starting in less than 1 hour (too late to bet reliably)
    try:
        commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        hours_until = (commence - datetime.now(timezone.utc)).total_seconds() / 3600
        if hours_until < 1 or hours_until > 48:
            return
    except Exception:
        return

    # Skip if already bet on this game (open or already settled)
    existing_ids = {b["game_id"] for b in broker.open_bets} | {b["game_id"] for b in broker.closed_bets}
    if game["game_id"] in existing_ids:
        return

    # Skip if too many open bets
    if len(broker.open_bets) >= CONFIG.bankroll.max_open_bets:
        logger.info("Max open bets reached — skipping new games")
        return

    home_team = game["home_team"]
    away_team = game["away_team"]

    home_stats = get_team_stats(sport, home_team, nba_fetcher, espn_fetcher, nba_stats_df)
    away_stats = get_team_stats(sport, away_team, nba_fetcher, espn_fetcher, nba_stats_df)

    # Injury report + current roster (NBA only — ESPN API)
    home_injuries, away_injuries = [], []
    home_roster, away_roster = "", ""
    if sport == "basketball_nba":
        home_injuries = injury_fetcher.get_team_injuries(home_team)
        away_injuries = injury_fetcher.get_team_injuries(away_team)
        home_roster   = roster_fetcher.get_roster_string(home_team)
        away_roster   = roster_fetcher.get_roster_string(away_team)
        if home_injuries or away_injuries:
            logger.info(
                f"  Injuries — {home_team}: {len(home_injuries)} | {away_team}: {len(away_injuries)}"
            )

    # Build feature snapshot (saved with bet for future model training)
    features = engineer.build_game_features(game, home_stats, away_stats)

    # Base probability: LightGBM model if available, else market implied
    book_home_prob = game.get("home_implied") or 0.5
    book_away_prob = game.get("away_implied") or 0.5

    if lgbm and sport == "basketball_nba":
        lgbm_prob = lgbm.predict(
            home_team, away_team, home_stats, away_stats,
            book_home_prob, book_away_prob,
        )
        logger.info(f"  LightGBM: home={lgbm_prob:.1%}  |  Market: home={book_home_prob:.1%}")
        base_home_prob = lgbm_prob
    else:
        base_home_prob = book_home_prob

    # Claude analysis — receives model probability as starting point
    logger.info(f"Analyzing: {away_team} @ {home_team} ({hours_until:.1f}h away)")
    analysis = claude.analyze_game(game, home_stats, away_stats, base_home_prob,
                                   home_injuries=home_injuries, away_injuries=away_injuries,
                                   home_roster=home_roster, away_roster=away_roster)

    our_home_prob = analysis["adjusted_home_prob"]
    our_away_prob = 1 - our_home_prob

    home_edge = our_home_prob - book_home_prob
    away_edge = our_away_prob - book_away_prob

    logger.info(
        f"  Claude: home={our_home_prob:.1%} (edge={home_edge:+.1%}) | "
        f"away={our_away_prob:.1%} (edge={away_edge:+.1%}) | "
        f"confidence={analysis['confidence']} | rec={analysis['bet_recommendation']}"
    )
    logger.info(f"  Reasoning: {analysis['reasoning']}")

    # --- Place bets where edge exceeds threshold ---
    min_edge = CONFIG.bankroll.min_edge

    if home_edge >= min_edge and game.get("home_ml") is not None:
        stake = broker.kelly_stake(our_home_prob, game["home_ml"], CONFIG.bankroll.kelly_fraction)
        max_stake = broker.bankroll * CONFIG.bankroll.max_bet_pct
        stake = min(stake, max_stake)
        if stake >= 5.0:
            broker.place_bet(
                game_id=game["game_id"], sport=sport,
                home_team=home_team, away_team=away_team,
                bet_type="home_ml", odds=game["home_ml"], stake=stake,
                reasoning=analysis["reasoning"],
                claude_home_prob=our_home_prob,
                book_home_prob=book_home_prob,
                features=features,
                commence_time=game.get("commence_time"),
            )

    elif away_edge >= min_edge and game.get("away_ml") is not None:
        stake = broker.kelly_stake(our_away_prob, game["away_ml"], CONFIG.bankroll.kelly_fraction)
        max_stake = broker.bankroll * CONFIG.bankroll.max_bet_pct
        stake = min(stake, max_stake)
        if stake >= 5.0:
            broker.place_bet(
                game_id=game["game_id"], sport=sport,
                home_team=home_team, away_team=away_team,
                bet_type="away_ml", odds=game["away_ml"], stake=stake,
                reasoning=analysis["reasoning"],
                claude_home_prob=our_home_prob,
                book_home_prob=book_home_prob,
                features=features,
                commence_time=game.get("commence_time"),
            )
    else:
        logger.info(f"  No value found — passing on {away_team} @ {home_team}")


def run_loop():
    odds_fetcher    = OddsFetcher(api_key=CONFIG.odds_api_key)
    nba_fetcher     = NBAStatsFetcher()
    espn_fetcher    = ESPNStatsFetcher()
    injury_fetcher  = InjuryFetcher()
    roster_fetcher  = RosterFetcher()
    results_fetcher = ResultsFetcher()
    engineer        = FeatureEngineer()
    claude          = ClaudeAnalyst(api_key=CONFIG.claude.api_key, model=CONFIG.claude.model)
    broker          = PaperBroker(starting_bankroll=CONFIG.bankroll.starting_bankroll)
    lgbm            = LGBMPredictor.load()  # None if model not yet trained

    logger.info("=" * 60)
    logger.info("Sports Betting Bot started [PAPER MODE]")
    logger.info(f"Sports: {CONFIG.sports.sports}")
    logger.info(f"Min edge: {CONFIG.bankroll.min_edge:.0%}  |  Kelly fraction: {CONFIG.bankroll.kelly_fraction}")
    logger.info("=" * 60)

    while True:
        try:
            now = datetime.now(timezone.utc)
            logger.info(f"--- Loop tick: {now.strftime('%Y-%m-%d %H:%M UTC')} ---")

            # Settle any completed games first
            if broker.open_bets:
                n_settled = results_fetcher.settle_open_bets(broker)
                if n_settled:
                    logger.info(f"Auto-settled {n_settled} bet(s) from completed games.")
                    broker.export_training_data("training_data.csv")

            # Find active sport with upcoming games
            sport, games_raw = odds_fetcher.get_active_sport(
                CONFIG.sports.sports, CONFIG.sports.bookmakers
            )

            if sport and games_raw:
                # Fetch team stats once per loop
                nba_stats_df = None
                if sport == "basketball_nba":
                    nba_stats_df = nba_fetcher.get_team_stats()

                for game_raw in games_raw:
                    evaluate_game(
                        game_raw, sport, nba_fetcher, espn_fetcher,
                        injury_fetcher, roster_fetcher, nba_stats_df,
                        engineer, claude, broker, lgbm,
                    )
            else:
                logger.info("No active sports right now — no bets to evaluate")

            # Print summary every loop
            summary = broker.summary()
            logger.info(
                f"BANKROLL: ${summary['bankroll']:,.2f} | "
                f"P&L: ${summary['total_pnl']:+,.2f} | "
                f"ROI: {summary['roi_pct']:+.1f}% | "
                f"Record: {summary['wins']}-{summary['losses']} | "
                f"Open bets: {summary['open_bets']}"
            )

        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)

        logger.info(f"Sleeping {CONFIG.loop_interval_seconds // 60} minutes...")
        time.sleep(CONFIG.loop_interval_seconds)


if __name__ == "__main__":
    run_loop()
