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
import math
import re
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
from SportsBettingBot.data.world_cup_fetcher import WorldCupFetcher
from SportsBettingBot.data.ufc_fetcher import UFCFetcher
from SportsBettingBot.data.stats_fetcher import NBAStatsFetcher, ESPNStatsFetcher, MLBStatsFetcher, NHLStatsFetcher, WNBAStatsFetcher, NFLStatsFetcher
from SportsBettingBot.data.weather_fetcher import WeatherFetcher
from SportsBettingBot.data.injury_fetcher import InjuryFetcher
from SportsBettingBot.data.roster_fetcher import RosterFetcher
from SportsBettingBot.data.results_fetcher import ResultsFetcher
from SportsBettingBot.data.outcome_tracker import OutcomeTracker
from SportsBettingBot.features.engineer import FeatureEngineer
from SportsBettingBot.models.claude_analyst import ClaudeAnalyst, analyst_version_for
from SportsBettingBot.broker.paper_broker import PaperBroker
from SportsBettingBot.notifications import push_notifier

_LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
_LOG_DATE = "%H:%M:%S"
_ANSI_RE  = re.compile(r"\x1b\[[0-9;]*m")

class _PlainFormatter(logging.Formatter):
    """Strips ANSI color codes so the log file stays human-readable."""
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_RE.sub("", super().format(record))

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE))

_file_handler = logging.FileHandler("bot.log")
_file_handler.setFormatter(_PlainFormatter(_LOG_FMT, datefmt=_LOG_DATE))

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger("bot")

_SPORT_SHORT: dict[str, str] = {
    "basketball_nba":        "NBA",
    "basketball_wnba":       "WNBA",
    "baseball_mlb":          "MLB",
    "icehockey_nhl":         "NHL",
    "americanfootball_nfl":  "NFL",
    "soccer_fifa_world_cup": "WC",
    "mma_ufc":               "UFC",
}

_SPORT_TAG: dict[str, str] = {
    "basketball_nba":        "\033[95mNBA\033[0m",
    "basketball_wnba":       "\033[33mWNBA\033[0m",
    "baseball_mlb":          "\033[34mMLB\033[0m",
    "icehockey_nhl":         "\033[36mNHL\033[0m",
    "americanfootball_nfl":  "\033[32mNFL\033[0m",
    "soccer_fifa_world_cup": "\033[93mWC\033[0m",
    "mma_ufc":               "\033[91mUFC\033[0m",
}

def _sport_tag(sport: str) -> str:
    return _SPORT_TAG.get(sport, sport.split("_")[-1].upper())


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
            elif start < now:
                # 4h check already passed but bet still open — game ran long or ESPN was delayed;
                # retry every 45 minutes until settled
                retry_at = now + timedelta(minutes=45)
                candidates.append((f"retry-settle {bet['away_team']} @ {bet['home_team']}", retry_at))
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


# Per-sport stats that should be populated if their data stream succeeded.
# A None/NaN/placeholder here means a fetch or merge silently failed — Claude
# would otherwise analyze with a hole in exactly the inputs its persona weights.
# Basketball per-player fetchers return up to 6 top scorers; a healthy fetch yields
# the core rotation. Alert if a partial fetch drops below this, so a degraded result
# (not just a fully-empty one) still surfaces — a missing star skews the analysis.
_MIN_KEY_PLAYERS = 4

_CRITICAL_STAT_FIELDS: dict[str, dict[str, str]] = {
    "basketball_nba": {
        "NET_RATING":   "net rating",
        "win_pct_l10":  "last-10 form",
        "avg_diff_l10": "last-10 point diff",
        # key_players handled separately (count threshold, not empty/non-empty) below
    },
    "baseball_mlb": {
        "team_era":    "team ERA",
        "bullpen_era": "bullpen ERA",
        "team_ops":    "team OPS",
        "streak":      "streak",
    },
    "icehockey_nhl": {
        "pp_pct":      "power play %",
        "pk_pct":      "penalty kill %",
        "goalie_name": "top goalie",
        "pointsFor":   "goals for",
        "rest_days":   "rest days",
        "streak":      "streak",
    },
    "basketball_wnba": {
        "fg_pct":         "FG%",
        "ast_to_ratio":   "A/TO ratio",
        "Last Ten Games": "last-10 record",
        # key_players handled separately (count threshold, not empty/non-empty) below
        "rest_days":      "rest days",
        "streak":         "streak",
    },
    "americanfootball_nfl": {
        "off_epa_per_play":         "offensive EPA",
        "def_epa_allowed_per_play": "defensive EPA",
        "sacks_allowed_pg":         "sacks allowed/game",
        "pointsFor":                "points for",
        "rest_days":                "rest days",
        "streak":                   "streak",
    },
    "soccer_fifa_world_cup": {
        "form": "recent form",
    },
    "mma_ufc": {
        "record":        "record",
        "strikeLPM":     "strikes landed/min",
        "strikeAccuracy": "striking accuracy",
        "takedownAvg":   "takedown average",
        "reach":         "reach",
    },
}


def _is_missing(v) -> bool:
    """True if a stat value represents 'not found' — None, NaN, or a blank/placeholder string."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().upper() in ("", "N/A", "NA", "NONE", "TBD", "?"):
        return True
    if isinstance(v, (list, dict, tuple)) and len(v) == 0:   # e.g. key_players came back empty
        return True
    return False


def _detect_missing_data(sport: str, home_team: str, away_team: str,
                         home_stats: dict, away_stats: dict,
                         home_roster: str, away_roster: str,
                         starting_pitchers: dict, weather: dict | None,
                         game: dict) -> list[str]:
    """Returns human-readable descriptions of any critical data stream that came back empty."""
    missing: list[str] = []
    field_map = _CRITICAL_STAT_FIELDS.get(sport, {})

    for team, stats in ((home_team, home_stats), (away_team, away_stats)):
        if not stats:
            missing.append(f"{team}: all team stats")
            continue
        absent = [label for key, label in field_map.items() if _is_missing(stats.get(key))]
        if absent:
            missing.append(f"{team}: {', '.join(absent)}")

    if sport in ("basketball_nba", "basketball_wnba"):
        for team, stats in ((home_team, home_stats), (away_team, away_stats)):
            if not stats:
                continue   # already reported as "all team stats" above
            n = len(stats.get("key_players") or [])
            if n < _MIN_KEY_PLAYERS:
                missing.append(f"{team}: key players ({n}/{_MIN_KEY_PLAYERS} fetched)")

    if sport == "soccer_fifa_world_cup":
        if not home_roster:
            missing.append(f"{home_team}: squad")
        if not away_roster:
            missing.append(f"{away_team}: squad")
        if _is_missing(game.get("venue")):
            missing.append("venue")

    if sport == "baseball_mlb":
        for side, team in (("home", home_team), ("away", away_team)):
            p = (starting_pitchers or {}).get(side, {})
            if _is_missing(p.get("name")):
                missing.append(f"{team}: starting pitcher")
            elif _is_missing(p.get("throws")):
                missing.append(f"{team}: pitcher handedness")

    if sport == "americanfootball_nfl" and weather is None:
        missing.append("weather")

    return missing


def evaluate_game(game_raw: dict, sport: str, nba_fetcher: NBAStatsFetcher,
                  espn_fetcher: ESPNStatsFetcher, injury_fetcher: InjuryFetcher,
                  roster_fetcher: RosterFetcher, nba_stats_df, espn_stats_df,
                  engineer: FeatureEngineer, claude: ClaudeAnalyst,
                  broker: PaperBroker,
                  nhl_fetcher: NHLStatsFetcher | None = None,
                  nfl_fetcher: NFLStatsFetcher | None = None,
                  weather_fetcher: WeatherFetcher | None = None,
                  world_cup_fetcher: WorldCupFetcher | None = None,
                  wnba_fetcher: WNBAStatsFetcher | None = None,
                  ufc_fetcher: UFCFetcher | None = None,
                  mlb_fetcher: MLBStatsFetcher | None = None) -> None:
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

    if sport == "mma_ufc" and ufc_fetcher is not None:
        # Individual sport — stats come per-fighter from ESPN core (id stored on the fight).
        home_stats = ufc_fetcher.get_fighter_stats(game.get("home_team_id"))
        away_stats = ufc_fetcher.get_fighter_stats(game.get("away_team_id"))
        home_stats["record"] = game.get("home_record")
        away_stats["record"] = game.get("away_record")
    else:
        home_stats = get_team_stats(sport, home_team, nba_fetcher, nba_stats_df, espn_stats_df)
        away_stats = get_team_stats(sport, away_team, nba_fetcher, nba_stats_df, espn_stats_df)

    if sport == "icehockey_nhl" and nhl_fetcher is not None:
        for stats in (home_stats, away_stats):
            abbrev = stats.get("nhl_abbrev")
            if abbrev:
                stats["rest_days"] = nhl_fetcher.get_rest_days(abbrev, commence)

    weather = None
    if sport == "americanfootball_nfl":
        if nfl_fetcher is not None:
            for team_name, stats in ((home_team, home_stats), (away_team, away_stats)):
                stats["rest_days"] = nfl_fetcher.get_rest_days(team_name, commence)
        if weather_fetcher is not None:
            weather = weather_fetcher.get_game_weather(home_team, commence)
            if weather:
                label = "dome" if weather.get("is_dome") else f"{weather.get('temp')}°F, {weather.get('wind_speed')} mph wind"
                logger.info(f"  Weather: {label}")

    if sport == "basketball_nba":
        for team_name, stats in ((home_team, home_stats), (away_team, away_stats)):
            stats["key_players"] = nba_fetcher.get_player_stats(team_name)

    if sport == "basketball_wnba" and wnba_fetcher is not None:
        for team_name, stats in ((home_team, home_stats), (away_team, away_stats)):
            rd = wnba_fetcher.get_rest_days(team_name, commence)
            if rd is not None:
                stats["rest_days"] = rd
            stats["key_players"] = wnba_fetcher.get_player_stats(team_name)

    if sport == "soccer_fifa_world_cup" and world_cup_fetcher:
        home_injuries, home_roster = world_cup_fetcher.get_team_roster_and_injuries(home_team)
        away_injuries, away_roster = world_cup_fetcher.get_team_roster_and_injuries(away_team)
        home_stats["form"] = game.get("home_form", "")
        away_stats["form"] = game.get("away_form", "")
    else:
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

    # Series / group context for Claude. NBA/NHL/MLB/WNBA all decide playoff rounds
    # by series, so feed Claude the current series standing (e.g. "leads series 2-1").
    # Returns None for regular-season games, so it is safe to call year-round.
    series_context = None
    if sport in ("basketball_nba", "icehockey_nhl", "baseball_mlb", "basketball_wnba"):
        series_context = espn_fetcher.get_series_context(sport, home_team, away_team)
    elif sport == "soccer_fifa_world_cup" and world_cup_fetcher:
        series_context = world_cup_fetcher.get_group_context(
            game["game_id"], home_team, away_team,
            game.get("draw_ml"), game.get("draw_implied", 0.0),
        )

    # Fetch probable starting pitchers for MLB — biggest single factor in game outcome
    starting_pitchers = {}
    if sport == "baseball_mlb" and mlb_fetcher is not None:
        starting_pitchers = mlb_fetcher.get_starting_pitchers(home_team, away_team)

    # Alert (admin push) if any critical data stream came back empty before Claude analyzes
    _missing = _detect_missing_data(
        sport, home_team, away_team, home_stats, away_stats,
        home_roster, away_roster, starting_pitchers, weather, game,
    )
    if _missing:
        logger.warning(f"  [{_sport_tag(sport)}] Data gap for {away_team} @ {home_team}: {_missing}")
        push_notifier.notify_data_missing(
            f"{away_team} @ {home_team}",
            _SPORT_SHORT.get(sport, sport.split("_")[-1].upper()),
            _missing,
        )

    logger.info(f"[{_sport_tag(sport)}] Analyzing: {away_team} @ {home_team} ({hours_until:.1f}h away)")
    analysis = claude.analyze_game(game, home_stats, away_stats, base_home_prob,
                                   home_injuries=home_injuries, away_injuries=away_injuries,
                                   home_roster=home_roster, away_roster=away_roster,
                                   sport=sport, series_context=series_context,
                                   starting_pitchers=starting_pitchers,
                                   weather=weather)

    our_home_prob = analysis["adjusted_home_prob"]
    # In a 3-way market (soccer), the draw is a separate outcome an ML bet loses on.
    # Claude returns only P(home win), so 1 - P(home) would fold the entire draw mass
    # into the away side and fabricate an away edge. Hold the market's no-vig draw
    # probability fixed and give the away side the remainder.
    if sport == "soccer_fifa_world_cup":
        draw_prob = game.get("draw_implied", 0.0)
        our_away_prob = max(0.0, 1 - our_home_prob - draw_prob)
    else:
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
    min_odds   = CONFIG.bankroll.min_odds   # skip heavy favorites (odds more negative than this)
    max_edge   = CONFIG.bankroll.max_edge_by_sport.get(sport)   # None = uncapped; pass implausible divergences
    max_div    = CONFIG.bankroll.max_divergence_by_sport.get(sport)  # hard cap on |claude - market| (magnitude discipline)
    divergence = abs(our_home_prob - book_home_prob)
    claude_rec = analysis["bet_recommendation"]

    stake = round(broker.bankroll * CONFIG.bankroll.flat_bet_pct, 2)
    logger.info(f"  Flat stake: ${stake:.2f} ({CONFIG.bankroll.flat_bet_pct:.1%} of bankroll)")

    home_ml = game.get("home_ml")
    away_ml = game.get("away_ml")
    bet_placed = False
    if home_edge >= min_edge and claude_rec == "home_ml" and home_ml is not None:
        if max_edge is not None and home_edge > max_edge:
            logger.info(f"  Skipping {home_team} — edge {home_edge:+.0%} exceeds max_edge ({max_edge:.0%}); divergence implausibly large for {_SPORT_SHORT.get(sport, sport)}")
        elif max_div is not None and divergence > max_div:
            logger.info(f"  Skipping {home_team} — divergence {divergence:.0%} exceeds max_divergence ({max_div:.0%}); over-repricing the market for {_SPORT_SHORT.get(sport, sport)}")
        elif home_ml < min_odds:
            logger.info(f"  Skipping {home_team} — odds {home_ml:.0f} past min_odds ({min_odds:.0f}); favorite too heavy")
        elif stake >= 5.0:
            broker.place_bet(
                game_id=game["game_id"], sport=sport,
                home_team=home_team, away_team=away_team,
                bet_type="home_ml", odds=home_ml, stake=stake,
                reasoning=analysis["reasoning"],
                claude_home_prob=our_home_prob, book_home_prob=book_home_prob,
                home_edge=home_edge, away_edge=away_edge,
                features=features, commence_time=game.get("commence_time"),
                analyst_version=analyst_version_for(sport),
            )
            bet_placed = True
    elif away_edge >= min_edge and claude_rec == "away_ml" and away_ml is not None:
        if max_edge is not None and away_edge > max_edge:
            logger.info(f"  Skipping {away_team} — edge {away_edge:+.0%} exceeds max_edge ({max_edge:.0%}); divergence implausibly large for {_SPORT_SHORT.get(sport, sport)}")
        elif max_div is not None and divergence > max_div:
            logger.info(f"  Skipping {away_team} — divergence {divergence:.0%} exceeds max_divergence ({max_div:.0%}); over-repricing the market for {_SPORT_SHORT.get(sport, sport)}")
        elif away_ml < min_odds:
            logger.info(f"  Skipping {away_team} — odds {away_ml:.0f} past min_odds ({min_odds:.0f}); favorite too heavy")
        elif stake >= 5.0:
            broker.place_bet(
                game_id=game["game_id"], sport=sport,
                home_team=home_team, away_team=away_team,
                bet_type="away_ml", odds=away_ml, stake=stake,
                reasoning=analysis["reasoning"],
                claude_home_prob=our_home_prob, book_home_prob=book_home_prob,
                home_edge=home_edge, away_edge=away_edge,
                features=features, commence_time=game.get("commence_time"),
                analyst_version=analyst_version_for(sport),
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
            analyst_version=analyst_version_for(sport),
        )


def run_loop():
    odds_fetcher     = OddsFetcher(api_key=CONFIG.odds_api_key)
    world_cup_fetcher = WorldCupFetcher()
    ufc_fetcher      = UFCFetcher(odds_api_key=CONFIG.the_odds_api_key)
    nba_fetcher      = NBAStatsFetcher()
    espn_fetcher     = ESPNStatsFetcher()
    mlb_fetcher      = MLBStatsFetcher()
    nhl_fetcher      = NHLStatsFetcher()
    nfl_fetcher      = NFLStatsFetcher()
    wnba_fetcher     = WNBAStatsFetcher()
    weather_fetcher  = WeatherFetcher()
    injury_fetcher   = InjuryFetcher()
    roster_fetcher   = RosterFetcher()
    results_fetcher  = ResultsFetcher()
    outcome_tracker  = OutcomeTracker(results_fetcher)
    engineer        = FeatureEngineer()
    claude          = ClaudeAnalyst(api_key=CONFIG.claude.api_key, model=CONFIG.claude.model)
    broker          = PaperBroker(starting_bankroll=CONFIG.bankroll.starting_bankroll)

    logger.info("=" * 60)
    logger.info("Sports Betting Bot started [PAPER MODE]")
    logger.info(f"Sports: {[_sport_tag(s) for s in CONFIG.sports.sports]}")
    edge_summary = ", ".join(
        f"{_sport_tag(s)}: {e:.0%}"
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
                if sport == "soccer_fifa_world_cup":
                    games = world_cup_fetcher.get_upcoming_games()
                elif sport == "mma_ufc":
                    games = ufc_fetcher.get_upcoming_fights()
                else:
                    games = odds_fetcher.get_upcoming_games(sport, CONFIG.sports.bookmakers)
                for g in games:
                    all_games_raw.append((sport, g))

            total_games = len(all_games_raw)
            logger.info(f"Upcoming games across all sports: {total_games}")

            # Persist today's (ET) scheduled games for the dashboard dropdown
            today_et = now.astimezone(_ET).date()
            today_games = []
            for _sport, _g in all_games_raw:
                _ct = _g.get("commence_time", "")
                if not _ct:
                    continue
                try:
                    _game_et_date = datetime.fromisoformat(
                        _ct.replace("Z", "+00:00")
                    ).astimezone(_ET).date()
                    if _game_et_date == today_et:
                        today_games.append({
                            "game_id":      _g.get("game_id", ""),
                            "sport":        _sport,
                            "home_team":    _g.get("home_team", ""),
                            "away_team":    _g.get("away_team", ""),
                            "commence_time": _ct,
                        })
                except Exception:
                    pass
            broker.set_upcoming_games(today_games)

            # 3. Pre-fetch stats for any sport with a game in the analysis window
            nba_stats_df = None
            espn_stats_cache: dict = {}
            for sport, g in all_games_raw:
                h = _hours_until(g)
                if h is None or not (0.5 <= h <= 2.5):
                    continue
                if sport == "basketball_nba" and nba_stats_df is None:
                    nba_stats_df = nba_fetcher.get_team_stats()
                elif sport == "mma_ufc":
                    continue   # fighter stats fetched per-fight in evaluate_game
                elif sport == "soccer_fifa_world_cup" and sport not in espn_stats_cache:
                    espn_stats_cache[sport] = world_cup_fetcher.get_team_stats()
                elif sport != "basketball_nba" and sport not in espn_stats_cache:
                    espn_df = espn_fetcher.get_team_stats(sport)
                    if sport == "baseball_mlb":
                        mlb_adv = mlb_fetcher.get_team_stats()
                        if not mlb_adv.empty and not espn_df.empty:
                            # Merge on full name first, fall back to nickname for relocated teams
                            merged = espn_df.merge(mlb_adv, on="team", how="left")
                            unmatched = merged["team_era"].isna()
                            if unmatched.any():
                                mlb_adv_nick = mlb_adv.copy()
                                mlb_adv_nick["_nick"] = mlb_adv_nick["team"].str.split().str[-1]
                                merged["_nick"] = merged["team"].str.split().str[-1]
                                fallback = merged[unmatched][["team", "_nick"]].merge(
                                    mlb_adv_nick.drop(columns="team"), on="_nick", how="left"
                                )
                                adv_cols = [c for c in mlb_adv.columns if c != "team"]
                                for col in adv_cols:
                                    merged.loc[unmatched, col] = fallback[col].values
                            espn_df = merged.drop(columns=["_nick"], errors="ignore")
                    elif sport == "americanfootball_nfl":
                        nfl_adv = nfl_fetcher.get_team_stats()
                        if not nfl_adv.empty and not espn_df.empty:
                            merged = espn_df.merge(nfl_adv, on="team", how="left")
                            unmatched = merged["pass_yards_pg"].isna()
                            if unmatched.any():
                                nfl_adv_nick = nfl_adv.copy()
                                nfl_adv_nick["_nick"] = nfl_adv_nick["team"].str.split().str[-1]
                                merged["_nick"] = merged["team"].str.split().str[-1]
                                fallback = merged[unmatched][["team", "_nick"]].merge(
                                    nfl_adv_nick.drop(columns="team"), on="_nick", how="left"
                                )
                                adv_cols = [c for c in nfl_adv.columns if c != "team"]
                                for col in adv_cols:
                                    merged.loc[unmatched, col] = fallback[col].values
                            espn_df = merged.drop(columns=["_nick"], errors="ignore")
                    elif sport == "icehockey_nhl":
                        nhl_adv = nhl_fetcher.get_team_stats()
                        if not nhl_adv.empty and not espn_df.empty:
                            merged = espn_df.merge(nhl_adv, on="team", how="left")
                            unmatched = merged["pp_pct"].isna()
                            if unmatched.any():
                                nhl_adv_nick = nhl_adv.copy()
                                nhl_adv_nick["_nick"] = nhl_adv_nick["team"].str.split().str[-1]
                                merged["_nick"] = merged["team"].str.split().str[-1]
                                fallback = merged[unmatched][["team", "_nick"]].merge(
                                    nhl_adv_nick.drop(columns="team"), on="_nick", how="left"
                                )
                                adv_cols = [c for c in nhl_adv.columns if c != "team"]
                                for col in adv_cols:
                                    merged.loc[unmatched, col] = fallback[col].values
                            espn_df = merged.drop(columns=["_nick"], errors="ignore")
                    elif sport == "basketball_wnba":
                        wnba_adv = wnba_fetcher.get_team_stats()
                        if not wnba_adv.empty and not espn_df.empty:
                            espn_df = espn_df.merge(wnba_adv, on="team", how="left")
                    espn_stats_cache[sport] = espn_df

            # 4. Evaluate games in analysis window
            for sport, game_raw in all_games_raw:
                evaluate_game(
                    game_raw, sport, nba_fetcher, espn_fetcher,
                    injury_fetcher, roster_fetcher,
                    nba_stats_df if sport == "basketball_nba" else None,
                    espn_stats_cache.get(sport),
                    engineer, claude, broker,
                    nhl_fetcher=nhl_fetcher,
                    nfl_fetcher=nfl_fetcher,
                    weather_fetcher=weather_fetcher,
                    world_cup_fetcher=world_cup_fetcher,
                    wnba_fetcher=wnba_fetcher,
                    ufc_fetcher=ufc_fetcher,
                    mlb_fetcher=mlb_fetcher,
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
