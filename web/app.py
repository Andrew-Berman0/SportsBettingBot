"""
web/app.py
----------
WWAID — What Would AI Do?
Public dashboard for the AI paper-trading sports betting experiment.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

STATE_FILE        = Path(__file__).parent.parent / "broker_state.json"
STARTING_BANKROLL = 10_000.0

SPORT_LABELS = {
    "basketball_nba":        "NBA",
    "basketball_wnba":       "WNBA",
    "baseball_mlb":          "MLB",
    "icehockey_nhl":         "NHL",
    "americanfootball_nfl":  "NFL",
    "soccer_fifa_world_cup": "WC",
}


def _load_state() -> dict:
    with open(STATE_FILE) as f:
        return json.load(f)


def _fmt_date(iso: str, fmt: str = "%b %d", tz=None) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if tz:
            dt = dt.astimezone(tz)
        return dt.strftime(fmt)
    except Exception:
        return iso[:10] if iso else "—"


def _fmt_odds(odds: float | None) -> str:
    if odds is None:
        return "—"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def _prepare_pass(g: dict) -> dict:
    p = dict(g)
    sport = p.get("sport", "")
    p["sport_label"] = SPORT_LABELS.get(sport, sport.upper())
    p["matchup"]     = f"{p['away_team']} @ {p['home_team']}"
    p["home_ml_str"] = _fmt_odds(p.get("home_ml"))
    p["away_ml_str"] = _fmt_odds(p.get("away_ml"))
    try:
        ct = p.get("commence_time", "")
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(ET)
        p["game_time"] = dt.strftime("%-I:%M %p ET · %b %-d")
    except Exception:
        p["game_time"] = ""
    p["home_edge_pct"] = f"{p.get('home_edge', 0) * 100:+.1f}%"
    p["away_edge_pct"] = f"{p.get('away_edge', 0) * 100:+.1f}%"
    p["claude_pct"]    = f"{p.get('claude_home_prob', 0) * 100:.0f}%"
    p["market_pct"]    = f"{p.get('book_home_prob', 0) * 100:.0f}%"
    return p


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
    b["placed_date"] = _fmt_date(b.get("placed_at", ""), "%-m/%-d %-I:%M %p ET", tz=ET)
    try:
        ct = b.get("commence_time", "")
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(ET)
        b["game_time"] = dt.strftime("%-I:%M %p ET · %b %-d")
    except Exception:
        b["game_time"] = ""
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


@app.get("/OneSignalSDKWorker.js")
async def onesignal_worker():
    return FileResponse(
        Path(__file__).parent / "static" / "OneSignalSDKWorker.js",
        media_type="application/javascript",
    )


@app.get("/google5a1c10f341626ed7.html", response_class=PlainTextResponse)
async def google_verification():
    return "google-site-verification: google5a1c10f341626ed7.html"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = _load_state()

    closed = [_prepare_bet(b) for b in state["closed_bets"]]
    closed.sort(key=lambda b: b.get("commence_time", ""), reverse=True)

    open_bets = [_prepare_bet(b) for b in state["open_bets"]]

    # Today's passed games (ET date)
    today_et = datetime.now(ET).date()
    passed_today = []
    for g in state.get("passed_games", []):
        try:
            passed_date = datetime.fromisoformat(g["passed_at"]).astimezone(ET).date()
            if passed_date == today_et:
                passed_today.append(_prepare_pass(g))
        except Exception:
            pass
    passed_today.sort(key=lambda g: g.get("commence_time", ""))
    today_label = datetime.now(ET).strftime("%A, %B %-d")

    # Games scheduled today that Claude has NOT yet evaluated — shown in CTA dropdown
    evaluated_ids = (
        set(state.get("evaluated_game_ids", []))
        | {b["game_id"] for b in state["open_bets"]}
        | {b["game_id"] for b in state["closed_bets"]}
        | {g["game_id"] for g in state.get("passed_games", [])}
    )
    upcoming_games = []
    for g in state.get("upcoming_games", []):
        if g.get("game_id") in evaluated_ids:
            continue
        ct = g.get("commence_time", "")
        try:
            if datetime.fromisoformat(ct.replace("Z", "+00:00")) < datetime.now(timezone.utc):
                continue
        except Exception:
            pass
        try:
            game_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(ET)
            game_time_str = game_dt.strftime("%-I:%M %p ET")
            eval_dt = game_dt - timedelta(hours=2)
            eval_time_str = eval_dt.strftime("%-I:%M %p ET")
        except Exception:
            game_time_str = ""
            eval_time_str = ""
        sport = g.get("sport", "")
        upcoming_games.append({
            "matchup":     f"{g.get('away_team', '')} @ {g.get('home_team', '')}",
            "game_time":   game_time_str,
            "eval_time":   eval_time_str,
            "sport_label": SPORT_LABELS.get(sport, sport.upper()),
            "commence_time": ct,
        })
    upcoming_games.sort(key=lambda g: g.get("commence_time", ""))

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
    pnl_colors = ["#39FF14" if v >= 0 else "#ff4444" for v in pnl_values]

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
            "passed_today":    passed_today,
            "today_label":     today_label,
            "upcoming_games":  upcoming_games,
            "last_updated":    datetime.now(ET).strftime("%b %d, %Y %-I:%M %p ET"),
        },
    )
