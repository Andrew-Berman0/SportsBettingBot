# SportsBettingBot

A paper-trading sports betting bot that combines a LightGBM statistical model with Claude AI analysis to find edges over bookmaker closing lines. Runs on a remote server and evaluates upcoming NBA (and other sport) games every hour.

## How it works

1. **Odds fetcher** pulls upcoming games and closing moneylines from The Odds API (FanDuel, DraftKings, BetMGM).
2. **LightGBM model** produces a baseline home win probability from pre-game features: Elo ratings, rolling win%, point differential (L5/L10), season win rate, rest/back-to-back, and market implied probability.
3. **Claude** (`claude-opus-4-7`) receives the model estimate alongside team stats, injury reports, and current rosters. It adjusts the probability using qualitative reasoning the model can't capture.
4. **Kelly criterion** (quarter Kelly, capped at 5% of bankroll) sizes the stake when `our_prob − market_implied_prob > 3%`.
5. **Results fetcher** auto-settles open bets once a game completes using ESPN's scoreboard API.
6. All state (bankroll, open/closed bets, P&L) persists in `broker_state.json` and survives restarts.

## Project structure

```
SportsBettingBot/
├── bot.py                      # Main loop (runs every hour)
├── config.py                   # All tunable parameters
├── requirements.txt
│
├── data/
│   ├── historical_builder.py   # Builds NBA training dataset (nba_api + ESPN odds)
│   ├── odds_fetcher.py         # The Odds API wrapper
│   ├── stats_fetcher.py        # NBA team stats + recent form (nba_api)
│   ├── injury_fetcher.py       # ESPN injury reports
│   ├── roster_fetcher.py       # ESPN current rosters
│   └── results_fetcher.py      # ESPN scoreboard for bet settlement
│
├── models/
│   ├── lgbm_trainer.py         # Train / evaluate the LightGBM model
│   ├── lgbm_predictor.py       # Wraps the saved model for the live bot
│   ├── claude_analyst.py       # Claude probability adjustment
│   └── saved/
│       └── lgbm_betting_model.pkl
│
├── features/
│   └── engineer.py             # Feature engineering for live games
│
├── broker/
│   └── paper_broker.py         # Bankroll, Kelly sizing, bet tracking
│
└── dashboard/
    ├── server.py               # FastAPI dashboard server
    └── static/index.html
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
ODDS_API_KEY=your_odds_api_key        # https://the-odds-api.com
ANTHROPIC_API_KEY=your_anthropic_key
```

### Train the model (required before first run)

Build the historical dataset (~35 min first run, cached on reruns):

```bash
python data/historical_builder.py
```

Train and save the model:

```bash
python models/lgbm_trainer.py --retrain-all
```

To evaluate walk-forward performance (train 2021-23, test 2023-24) without saving:

```bash
python models/lgbm_trainer.py
```

### Run the bot

```bash
python bot.py
```

The bot evaluates games every 60 minutes. It logs to both stdout and `bot.log`.

## Configuration

All parameters live in [config.py](config.py):

| Parameter | Default | Description |
|---|---|---|
| `starting_bankroll` | $10,000 | Paper bankroll |
| `kelly_fraction` | 0.25 | Quarter Kelly (conservative) |
| `max_bet_pct` | 0.05 | Max 5% of bankroll per bet |
| `min_edge` | 0.03 | Minimum edge to place a bet |
| `max_open_bets` | 5 | Max simultaneous open bets |
| `loop_interval_seconds` | 3600 | How often to check for new games |

## Model details

- **Algorithm**: LightGBM with isotonic calibration (5-fold CV)
- **Features**: Elo ratings, L5/L10 rolling win% and point differential, season win rate, rest days, back-to-back flag, market closing implied probability
- **Training**: Seasons 2021-22 and 2022-23
- **Validation**: Season 2023-24 (walk-forward, no lookahead)
- **Test AUC**: ~0.73

The model's primary role is to give Claude a calibrated statistical starting point. The market implied probability is intentionally included as a feature — it's a strong prior that the model learns to lean on while adding signal from team strength and form.

## Deployment (Linux / systemd)

Copy files to your server and create `/etc/systemd/system/sportsbettingbot.service`:

```ini
[Unit]
Description=Sports Betting Bot
After=network.target

[Service]
WorkingDirectory=/root/SportsBettingBot
ExecStart=/root/SportsBettingBot/venv/bin/python3 bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable sportsbettingbot
systemctl start sportsbettingbot
journalctl -u sportsbettingbot -f
```
