"""
web/app.py
----------
WWAID — What Would AI Do?
Public dashboard for the AI paper-trading sports betting experiment.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

STATE_FILE        = Path(__file__).parent.parent / "broker_state.json"
STARTING_BANKROLL = 10_000.0

SPORT_LABELS = {
    "basketball_nba":       "NBA",
    "basketball_wnba":      "WNBA",
    "baseball_mlb":         "MLB",
    "icehockey_nhl":        "NHL",
    "americanfootball_nfl": "NFL",
}


def _load_state() -> dict:
    with open(STATE_FILE) as f:
        return json.load(f)


def _fmt_odds(odds: float) -> str:
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def _fmt_date(iso: str, fmt: str = "%b %d") -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime(fmt)
    except Exception:
        return iso[:10] if iso else "—"


def _prepare_bet(bet: dict) -> dict:
    b = dict(bet)
    sport = b.get("sport", "")
    b["sport_label"] = SPORT_LABELS.get(sport, sport.upper())
    b["sport_key"]   = sport
    b["matchup"]     = f"{b['away_team']} @ {b['home_team']}"
    b["bet_team"]    = b["home_team"] if b["bet_type"] == "home_ml" else b["away_team"]
    b["bet_label"]   = "Home ML" if b["bet_type"] == "home_ml" else "Away ML"
    b["odds_str"]    = _fmt_odds(b.get("odds", 0))
    b["stake_str"]   = f"${b['stake']:.2f}"
    b["placed_date"] = _fmt_date(b.get("placed_at", ""), "%b %d, %H:%M UTC")
    b["to_win"]      = b.get("potential_payout", 0) - b.get("stake", 0)

    if b.get("status") in ("won", "lost"):
        b["won"]          = b["status"] == "won"
        b["pnl"]          = b.get("pnl", 0)
        pnl               = b["pnl"]
        b["pnl_str"]      = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        b["settled_date"] = _fmt_date(b.get("settled_at", ""))
        hs = b.get("home_score")
        as_ = b.get("away_score")
        b["score_str"] = f"{b['home_team']} {hs}, {b['away_team']} {as_}" if hs is not None else ""
    return b


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = _load_state()

    closed = [_prepare_bet(b) for b in state["closed_bets"]]
    closed.sort(key=lambda b: b.get("settled_at", ""), reverse=True)

    open_bets = [_prepare_bet(b) for b in state["open_bets"]]

    # Overall stats
    total_staked = sum(b["stake"] for b in closed)
    total_pnl    = sum(b["pnl"] for b in closed)
    wins         = sum(1 for b in closed if b["won"])
    losses       = len(closed) - wins
    roi          = (total_pnl / total_staked * 100) if total_staked else 0.0
    win_rate     = (wins / len(closed) * 100) if closed else 0.0

    stats = {
        "bankroll":   state["bankroll"],
        "total_pnl":  total_pnl,
        "roi":        roi,
        "wins":       wins,
        "losses":     losses,
        "total_bets": len(closed),
        "win_rate":   win_rate,
    }

    # Bankroll history (chronological for chart)
    history_bets = sorted(closed, key=lambda b: b.get("settled_at", ""))
    bankroll_labels  = ["Start"]
    bankroll_values  = [STARTING_BANKROLL]
    running = STARTING_BANKROLL
    for b in history_bets:
        running += b["pnl"]
        bankroll_labels.append(b["settled_date"])
        bankroll_values.append(round(running, 2))

    # P&L bar chart (chronological)
    pnl_labels = [b["matchup"] for b in history_bets]
    pnl_values = [round(b["pnl"], 2) for b in history_bets]
    pnl_colors = ["#10b981" if v >= 0 else "#ef4444" for v in pnl_values]

    # Per-sport breakdown
    sport_stats: dict[str, dict] = {}
    for b in closed:
        sl = b["sport_label"]
        if sl not in sport_stats:
            sport_stats[sl] = {"wins": 0, "losses": 0, "pnl": 0.0}
        sport_stats[sl]["wins" if b["won"] else "losses"] += 1
        sport_stats[sl]["pnl"] += b["pnl"]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "stats":           stats,
            "closed_bets":     closed,
            "open_bets":       open_bets,
            "sport_stats":     sport_stats,
            "bankroll_labels": json.dumps(bankroll_labels),
            "bankroll_values": json.dumps(bankroll_values),
            "pnl_labels":      json.dumps(pnl_labels),
            "pnl_values":      json.dumps(pnl_values),
            "pnl_colors":      json.dumps(pnl_colors),
            "last_updated":    datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC"),
        },
    )
