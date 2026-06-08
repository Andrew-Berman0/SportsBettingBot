"""
bot.py
------
Event-driven sports betting bot.

On each wake:
  1. Settle any completed bets via ESPN
  2. Fetch odds for all configured sports (smart cache — only hits the API when needed)
  3. Evaluate any game within the 2h analysis window
  4. Sleep until the next meaningful event:
       - 2h before the next unevaluated game  (pre-game analysis)
       - 4h after tip-off for any open bet    (settlement check)
       - midnight ET daily                    (morning discovery refresh)

Run:
  python bot.py
"""

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_here = Path(__file__).parent
sys.path.insert(0, str(_here if (_here / "SportsBettingBot").is_dir() else _here.parent))

from SportsBettingBot.config import CONFIG
from SportsBettingBot.data.odds_fetcher import OddsFetcher
from SportsBettingBot.data.stats_fetcher import NBAStatsFetcher, ESPNStatsFetcher
from SportsBettingBot.data.injury_fetcher import InjuryFetcher
from SportsBettingBot.data.roster_fetcher import RosterFetcher
from SportsBettingBot.data.results_fetcher import ResultsFetcher
from SportsBettingBot.data.outcome_tracker import OutcomeTracker
from SportsBettingBot.features.engineer import FeatureEngineer
from SportsBettingBot.models.claude_analyst import ClaudeAnalyst
from SportsBettingBot.broker.paper_broker import PaperBroker

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


def _hours_until(g: dict) -> float | None:
    ct = g.get("commence_time")
    if not ct:
        return None
    try:
        return (
            datetime.fromisoformat(ct.replace("Z", "+00:00")) - datetime.now(timezone.utc)
        ).total_seconds() / 3600
    except Exception:
        return None


def _next_sleep_seconds(all_games_raw: list, broker: PaperBroker, now: datetime) -> int:
    """
    Returns seconds until the next meaningful event:
      - 2h before the next unevaluated game
      - 4h after tip-off for any open bet (settlement check)
      - midnight ET the next day (morning discovery)
    Minimum 5 minutes.
    """
    candidates: list[tuple[str, datetime]] = []

    evaluated = broker.evaluated_game_ids
    bet_ids = {b["game_id"] for b in broker.open_bets} | {b["game_id"] for b in broker.closed_bets}

    for _sport, g in all_games_raw:
        ct = g.get("commence_time")
        if not ct:
            continue
        gid = str(g.get("game_id") or g.get("id") or "")
        if gid and (gid in evaluated or gid in bet_ids):
            continue
        try:
            commence = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            wake_at = commence - timedelta(hours=2)
            if wake_at > now + timedelta(minutes=5):
                label = f"pre-game {g.get('away_team','?')} @ {g.get('home_team','?')}"
                candidates.append((label, wake_at))
        except Exception:
            pass

    for bet in broker.open_bets:
        ct = bet.get("commence_time")
        if not ct:
            continue
        try:
            start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            settle_at = start + timedelta(hours=4)
            if settle_at > now + timedelta(minutes=5):
                candidates.append((f"settle {bet['away_team']} @ {bet['home_team']}", settle_at))
        except Exception:
            pass

    # Daily morning refresh at midnight ET (start of new ET day)
    now_et = now.astimezone(_ET)
    morning = (now_et.replace(hour=0, minute=0, second=0, microsecond=0)
               + timedelta(days=1)).astimezone(timezone.utc)
    if morning <= now + timedelta(minutes=5):
        morning += timedelta(days=1)
    candidates.append(("morning refresh", morning))

    label, next_wake = min(candidates, key=lambda x: x[1])
    sleep_secs = max(300, int((next_wake - now).total_seconds()))
    logger.info(
        f"Sleeping until {next_wake.strftime('%Y-%m-%d %H:%M UTC')} "
        f"({sleep_secs / 3600:.1f}h) — {label}"
    )
    return sleep_secs


def get_team_stats(sport: str, team_name: str, nba_fetcher: NBAStatsFetcher,
                   nba_stats_df=None, espn_stats_df=None) -> dict:
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
    elif espn_stats_df is not None and not espn_stats_df.empty:
        # Prefer exact full-name match ("Boston Red Sox" → "Boston Red Sox").
        # Fall back to last-word nick for relocated teams whose ESPN name lost the city
        # ("Oakland Athletics" → ESPN "Athletics").
        row = espn_stats_df[espn_stats_df["team"].str.lower() == team_name.lower()]
        if row.empty:
            nick = ESPNStatsFetcher.resolve_espn_nick(team_name)
            row = espn_stats_df[espn_stats_df["team"].str.contains(nick, case=False, na=False)]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {}


def evaluate_game(game_raw: dict, sport: str, nba_fetcher: NBAStatsFetcher,
                  espn_fetcher: ESPNStatsFetcher, injury_fetcher: InjuryFetcher,
                  roster_fetcher: RosterFetcher, nba_stats_df, espn_stats_df,
                  engineer: FeatureEngineer, claude: ClaudeAnalyst,
                  broker: PaperBroker) -> None:
    """Run the full analysis pipeline for a single game and place a bet if value found."""
    game = game_raw if game_raw.get("_pre_parsed") else OddsFetcher.parse_game(game_raw)
    if not game:
        return

    # Analysis window: bot wakes at 2h before, accept 0.5–2.5h to handle timing drift
    try:
        commence = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        hours_until = (commence - datetime.now(timezone.utc)).total_seconds() / 3600
        if hours_until < 0.5 or hours_until > 2.5:
            return
    except Exception:
        return

    # Skip if already evaluated or already have a bet on this game
    existing_ids = {b["game_id"] for b in broker.open_bets} | {b["game_id"] for b in broker.closed_bets}
    if game["game_id"] in existing_ids or game["game_id"] in broker.evaluated_game_ids:
        return

    # Skip if we already have an open bet on this matchup (series game N+1 before N settles)
    open_matchups = {
        (b["home_team"].lower().split()[-1], b["away_team"].lower().split()[-1])
        for b in broker.open_bets
    }
    this_matchup = (game["home_team"].lower().split()[-1], game["away_team"].lower().split()[-1])
    if this_matchup in open_matchups:
        logger.info(
            f"Open bet already exists for {game['away_team']} @ {game['home_team']} — skipping until settled"
        )
        return

    if len(broker.open_bets) >= CONFIG.bankroll.max_open_bets:
        logger.info("Max open bets reached — skipping new games")
        return

    home_team = game["home_team"]
    away_team = game["away_team"]

    home_stats = get_team_stats(sport, home_team, nba_fetcher, nba_stats_df, espn_stats_df)
    away_stats = get_team_stats(sport, away_team, nba_fetcher, nba_stats_df, espn_stats_df)

    home_injuries = injury_fetcher.get_team_injuries(home_team, sport=sport, max_age_minutes=30)
    away_injuries = injury_fetcher.get_team_injuries(away_team, sport=sport, max_age_minutes=30)
    home_roster   = roster_fetcher.get_roster_string(home_team, sport=sport)
    away_roster   = roster_fetcher.get_roster_string(away_team, sport=sport)
    if home_injuries or away_injuries:
        logger.info(f"  Injuries — {home_team}: {len(home_injuries)} | {away_team}: {len(away_injuries)}")

    features = engineer.build_game_features(game, home_stats, away_stats)

    book_home_prob = game.get("home_implied") or 0.5
    book_away_prob = game.get("away_implied") or 0.5

    base_home_prob = book_home_prob

    # Fetch playoff series context for NBA and NHL so Claude knows the series score
    series_context = None
    if sport in ("basketball_nba", "icehockey_nhl"):
        series_context = espn_fetcher.get_series_context(sport, home_team, away_team)

    # Fetch probable starting pitchers for MLB — biggest single factor in game outcome
    starting_pitchers = {}
    if sport == "baseball_mlb":
        starting_pitchers = espn_fetcher.get_starting_pitchers(home_team, away_team)

    logger.info(f"Analyzing: {away_team} @ {home_team} ({hours_until:.1f}h away)")
    analysis = claude.analyze_game(game, home_stats, away_stats, base_home_prob,
                                   home_injuries=home_injuries, away_injuries=away_injuries,
                                   home_roster=home_roster, away_roster=away_roster,
                                   sport=sport, series_context=series_context,
                                   starting_pitchers=starting_pitchers)

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

    min_edge   = CONFIG.bankroll.min_edge_by_sport.get(sport, CONFIG.bankroll.min_edge)
    claude_rec = analysis["bet_recommendation"]

    stake = round(broker.bankroll * CONFIG.bankroll.flat_bet_pct, 2)
    logger.info(f"  Flat stake: ${stake:.2f} ({CONFIG.bankroll.flat_bet_pct:.1%} of bankroll)")

    bet_placed = False
    if home_edge >= min_edge and claude_rec == "home_ml" and game.get("home_ml") is not None:
        if stake >= 5.0:
            broker.place_bet(
                game_id=game["game_id"], sport=sport,
                home_team=home_team, away_team=away_team,
                bet_type="home_ml", odds=game["home_ml"], stake=stake,
                reasoning=analysis["reasoning"],
                claude_home_prob=our_home_prob, book_home_prob=book_home_prob,
                features=features, commence_time=game.get("commence_time"),
            )
            bet_placed = True
    elif away_edge >= min_edge and claude_rec == "away_ml" and game.get("away_ml") is not None:
        if stake >= 5.0:
            broker.place_bet(
                game_id=game["game_id"], sport=sport,
                home_team=home_team, away_team=away_team,
                bet_type="away_ml", odds=game["away_ml"], stake=stake,
                reasoning=analysis["reasoning"],
                claude_home_prob=our_home_prob, book_home_prob=book_home_prob,
                features=features, commence_time=game.get("commence_time"),
            )
            bet_placed = True
    elif claude_rec == "pass" and (home_edge >= min_edge or away_edge >= min_edge):
        logger.info(f"  Edge found but Claude says pass — skipping {away_team} @ {home_team}")
    else:
        logger.info(f"  No value found — passing on {away_team} @ {home_team}")

    broker.mark_evaluated(game["game_id"])

    if not bet_placed:
        broker.record_pass(
            game_id=game["game_id"], sport=sport,
            home_team=home_team, away_team=away_team,
            commence_time=game.get("commence_time"),
            reasoning=analysis["reasoning"],
            claude_home_prob=our_home_prob, book_home_prob=book_home_prob,
            home_edge=home_edge, away_edge=away_edge,
            home_ml=game.get("home_ml"), away_ml=game.get("away_ml"),
        )


def run_loop():
    odds_fetcher     = OddsFetcher(api_key=CONFIG.odds_api_key)
    nba_fetcher      = NBAStatsFetcher()
    espn_fetcher     = ESPNStatsFetcher()
    injury_fetcher   = InjuryFetcher()
    roster_fetcher   = RosterFetcher()
    results_fetcher  = ResultsFetcher()
    outcome_tracker  = OutcomeTracker(results_fetcher)
    engineer        = FeatureEngineer()
    claude          = ClaudeAnalyst(api_key=CONFIG.claude.api_key, model=CONFIG.claude.model)
    broker          = PaperBroker(starting_bankroll=CONFIG.bankroll.starting_bankroll)

    logger.info("=" * 60)
    logger.info("Sports Betting Bot started [PAPER MODE]")
    logger.info(f"Sports: {CONFIG.sports.sports}")
    edge_summary = ", ".join(
        f"{s.split('_')[1].upper()}: {e:.0%}"
        for s, e in CONFIG.bankroll.min_edge_by_sport.items()
    )
    logger.info(f"Min edge: [{edge_summary}]  |  Flat bet: {CONFIG.bankroll.flat_bet_pct:.1%} of bankroll")
    logger.info("=" * 60)

    all_games_raw: list[tuple[str, dict]] = []

    while True:
        try:
            now = datetime.now(timezone.utc)
            logger.info(f"--- Wake: {now.strftime('%Y-%m-%d %H:%M UTC')} ---")

            # 1. Settle any completed bets and log all game outcomes
            if broker.open_bets:
                n_settled = results_fetcher.settle_open_bets(broker)
                if n_settled:
                    logger.info(f"Auto-settled {n_settled} bet(s) from completed games.")
                    broker.export_training_data("training_data.csv")
            outcome_tracker.update(broker)

            # 2. Fetch all sports (cache handles rate limiting; pre-game wake forces fresh call)
            all_games_raw = []
            for sport in CONFIG.sports.sports:
                games = odds_fetcher.get_upcoming_games(sport, CONFIG.sports.bookmakers)
                for g in games:
                    all_games_raw.append((sport, g))

            total_games = len(all_games_raw)
            logger.info(f"Upcoming games across all sports: {total_games}")

            # 3. Pre-fetch stats for any sport with a game in the analysis window
            nba_stats_df = None
            espn_stats_cache: dict = {}
            for sport, g in all_games_raw:
                h = _hours_until(g)
                if h is None or not (0.5 <= h <= 2.5):
                    continue
                if sport == "basketball_nba" and nba_stats_df is None:
                    nba_stats_df = nba_fetcher.get_team_stats()
                elif sport != "basketball_nba" and sport not in espn_stats_cache:
                    espn_stats_cache[sport] = espn_fetcher.get_team_stats(sport)

            # 4. Evaluate games in analysis window
            for sport, game_raw in all_games_raw:
                evaluate_game(
                    game_raw, sport, nba_fetcher, espn_fetcher,
                    injury_fetcher, roster_fetcher,
                    nba_stats_df if sport == "basketball_nba" else None,
                    espn_stats_cache.get(sport),
                    engineer, claude, broker,
                )

            # 5. Summary
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

        time.sleep(_next_sleep_seconds(all_games_raw, broker, now))


if __name__ == "__main__":
    run_loop()
