import os
from dataclasses import dataclass, field


@dataclass
class SportsConfig:
    sports: list = field(default_factory=lambda: ["basketball_nba", "basketball_wnba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_fifa_world_cup"])
    primary_sport: str = "basketball_nba"
    bet_types: list = field(default_factory=lambda: ["h2h", "totals"])  # moneyline + over/under
    bookmakers: list = field(default_factory=lambda: ["fanduel", "draftkings", "betmgm"])


@dataclass
class BankrollConfig:
    starting_bankroll: float = 10000.0   # paper bankroll
    flat_bet_pct: float = 0.025          # 2.5% of current bankroll per bet
    min_edge: float = 0.03               # fallback if sport not in min_edge_by_sport
    min_edge_by_sport: dict = field(default_factory=lambda: {
        "basketball_nba":        0.03,
        "basketball_wnba":       0.03,
        "americanfootball_nfl":  0.03,
        "baseball_mlb":          0.05,   # 3-5% edges hit <50% historically; require stronger conviction
        "icehockey_nhl":         0.03,
        "soccer_fifa_world_cup": 0.04,
        "mma_ufc":               0.05,   # high variance — demand a larger edge
    })
    # Upper edge cap: in an efficient market a divergence this large is more likely a
    # model error than real value, so pass it. Sports absent here are uncapped — NBA/UFC
    # are left out on purpose (legitimate large edges: playoff form, soft fight markets).
    # WNBA added in v4: its away leans hit ~40% and a 13pt fade of a known-injury favorite
    # got through uncapped — back the persona's magnitude rule with a hard ceiling.
    max_edge_by_sport: dict = field(default_factory=lambda: {
        "baseball_mlb":          0.12,
        "americanfootball_nfl":  0.12,
        "soccer_fifa_world_cup": 0.12,
        "basketball_wnba":       0.12,
    })
    max_open_bets: int = 15              # max simultaneous bets
    min_odds: float = -300               # avoid heavy favorites (implied > 75%)


@dataclass
class ModelConfig:
    retrain_every_days: int = 7
    val_frac: float = 0.2
    min_games_to_train: int = 200
    confidence_threshold: float = 0.60


@dataclass
class ClaudeConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))


@dataclass
class BettingBotConfig:
    sports:    SportsConfig   = field(default_factory=SportsConfig)
    bankroll:  BankrollConfig = field(default_factory=BankrollConfig)
    model:     ModelConfig    = field(default_factory=ModelConfig)
    claude:    ClaudeConfig   = field(default_factory=ClaudeConfig)
    odds_api_key: str         = field(default_factory=lambda: os.getenv("THE_RUNDOWN_API_KEY", ""))
    the_odds_api_key: str     = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))  # The Odds API — UFC odds
    loop_interval_seconds: int = 3600   # check for new games every hour
    state_file: str = "bot_state.json"
    log_file:   str = "bot.log"


CONFIG = BettingBotConfig()
