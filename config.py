import os
from dataclasses import dataclass, field


@dataclass
class SportsConfig:
    sports: list = field(default_factory=lambda: ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl"])
    primary_sport: str = "basketball_nba"
    bet_types: list = field(default_factory=lambda: ["h2h", "totals"])  # moneyline + over/under
    bookmakers: list = field(default_factory=lambda: ["fanduel", "draftkings", "betmgm"])


@dataclass
class BankrollConfig:
    starting_bankroll: float = 10000.0   # paper bankroll
    kelly_fraction: float = 0.25         # quarter Kelly (conservative)
    max_bet_pct: float = 0.05            # never more than 5% on one game
    min_edge: float = 0.03               # only bet if edge > 3%
    max_open_bets: int = 5               # max simultaneous bets
    min_odds: float = -300               # avoid heavy favorites (implied > 75%)


@dataclass
class ModelConfig:
    retrain_every_days: int = 7
    val_frac: float = 0.2
    min_games_to_train: int = 200
    confidence_threshold: float = 0.60


@dataclass
class ClaudeConfig:
    model: str = "claude-opus-4-7"
    max_tokens: int = 1024
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))


@dataclass
class BettingBotConfig:
    sports:    SportsConfig   = field(default_factory=SportsConfig)
    bankroll:  BankrollConfig = field(default_factory=BankrollConfig)
    model:     ModelConfig    = field(default_factory=ModelConfig)
    claude:    ClaudeConfig   = field(default_factory=ClaudeConfig)
    odds_api_key: str         = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    loop_interval_seconds: int = 3600   # check for new games every hour
    state_file: str = "bot_state.json"
    log_file:   str = "bot.log"


CONFIG = BettingBotConfig()
