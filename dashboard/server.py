"""
dashboard/server.py
-------------------
FastAPI dashboard server. Reads state files from both bots and serves
a live HTML dashboard at http://<server>:8000

Run: uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
"""

import json
import re
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE           = Path(__file__).parent.parent
BETTING_STATE  = BASE / "broker_state.json"
BETTING_LOG    = BASE / "bot.log"
TRADING_DIR    = Path("/root/TradingBot")
TRADING_LOG    = TRADING_DIR / "bot.log"
TRADING_META   = TRADING_DIR / "models" / "saved" / "lgbm_model.json"
TRADING_STATE  = TRADING_DIR / "bot_state.json"

app = FastAPI(title="Bot Dashboard")
app.mount("/static", StaticFiles(directory=BASE / "dashboard" / "static"), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict | list:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def tail_log(path: Path, n: int = 60) -> list[str]:
    try:
        with open(path) as f:
            return list(deque(f, maxlen=n))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/betting/summary")
def betting_summary():
    state = read_json(BETTING_STATE)
    if not state:
        return {"bankroll": 10000, "total_pnl": 0, "roi_pct": 0,
                "win_rate": 0, "wins": 0, "losses": 0, "open_bets": 0, "total_bets": 0}

    closed = state.get("closed_bets", [])
    open_b = state.get("open_bets", [])
    bankroll = state.get("bankroll", 10000)
    starting = 10000.0

    total_staked = sum(b["stake"] for b in closed)
    total_pnl    = sum(b.get("pnl", 0) for b in closed)
    wins         = sum(1 for b in closed if b.get("status") == "won")
    losses       = len(closed) - wins
    roi          = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    return {
        "bankroll":   round(bankroll, 2),
        "starting":   starting,
        "total_pnl":  round(total_pnl, 2),
        "roi_pct":    round(roi, 2),
        "win_rate":   round(wins / len(closed), 3) if closed else 0,
        "wins":       wins,
        "losses":     losses,
        "open_bets":  len(open_b),
        "total_bets": len(closed),
    }


@app.get("/api/betting/open")
def betting_open():
    state = read_json(BETTING_STATE)
    return state.get("open_bets", [])


@app.get("/api/betting/history")
def betting_history():
    state = read_json(BETTING_STATE)
    closed = state.get("closed_bets", [])
    return list(reversed(closed[-50:]))  # last 50, newest first


@app.get("/api/betting/equity")
def betting_equity():
    """Bankroll over time derived from closed bets."""
    state = read_json(BETTING_STATE)
    closed = sorted(state.get("closed_bets", []), key=lambda b: b.get("settled_at", ""))
    bankroll = 10000.0
    points = [{"time": "Start", "bankroll": bankroll}]
    for bet in closed:
        bankroll += bet.get("pnl", 0)
        points.append({
            "time":     bet.get("settled_at", "")[:10],
            "bankroll": round(bankroll, 2),
            "bet":      f"{bet['away_team']} @ {bet['home_team']}",
            "result":   bet.get("status", ""),
        })
    return points


@app.get("/api/betting/log")
def betting_log():
    return tail_log(BETTING_LOG, n=80)


@app.get("/api/trading/summary")
def trading_summary():
    try:
        lines = tail_log(TRADING_LOG, n=300)
        equity = None
        drawdown = None
        signal = None
        confidence = None
        proba = None

        starting_equity = None
        for line in reversed(lines):
            if "Starting equity:" in line and starting_equity is None:
                m = re.search(r"Starting equity: \$([\d,]+\.?\d*)", line)
                if m:
                    starting_equity = float(m.group(1).replace(",", ""))
                    # If no Equity: line found yet, bot restarted recently — use this as current equity
                    if equity is None:
                        equity = starting_equity
                        drawdown = 0.0
            if "Equity:" in line and "Starting equity:" not in line and equity is None:
                m = re.search(r"Equity: \$([\d,]+\.?\d*)\s+Drawdown: ([+\-\d.]+)%", line)
                if m:
                    equity   = float(m.group(1).replace(",", ""))
                    drawdown = float(m.group(2))
            if "Signal:" in line and signal is None:
                m = re.search(r"Signal: (\w+)\s+confidence: ([\d.]+)%", line)
                if m:
                    signal     = m.group(1)
                    confidence = float(m.group(2))
            if "Probabilities" in line and proba is None:
                m = re.search(r"SELL: ([\d.]+)%\s+HOLD: ([\d.]+)%\s+BUY: ([\d.]+)%", line)
                if m:
                    proba = {"sell": float(m.group(1)), "hold": float(m.group(2)), "buy": float(m.group(3))}
            if equity is not None and signal is not None:
                break

        # Pull model metadata (MCC, Sharpe, threshold) from saved JSON
        meta = read_json(TRADING_META)
        state = read_json(TRADING_STATE)

        effective_equity = equity if equity is not None else (starting_equity if starting_equity is not None else 100000)
        return {
            "equity":      effective_equity,
            "drawdown":    drawdown or 0,
            "signal":      signal or "HOLD",
            "confidence":  confidence or 0,
            "proba":       proba,   # None until bot logs a Probabilities line
            "starting":    100000,
            "pnl":         round(effective_equity - 100000, 2),
            "mcc":         meta.get("mcc", 0),
            "sharpe":      meta.get("sharpe", 0),
            "threshold":   meta.get("optimal_threshold", 0.65),
            "n_train":     meta.get("n_train", 0),
            "last_retrain": state.get("last_retrain", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trading/equity")
def trading_equity():
    """Parse equity values from trading bot log to build equity curve."""
    lines = tail_log(TRADING_LOG, n=500)
    points = []
    for line in lines:
        if "Equity:" in line:
            m = re.search(r"(\d{2}:\d{2}:\d{2}).*Equity: \$([\d,]+\.?\d*)", line)
            if m:
                points.append({
                    "time":    m.group(1),
                    "equity":  float(m.group(2).replace(",", "")),
                })
    return points


@app.get("/api/trading/log")
def trading_log():
    return tail_log(TRADING_LOG, n=80)


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(BASE / "dashboard" / "static" / "index.html") as f:
        return f.read()
