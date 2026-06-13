"""
scripts/calibration_reminder.py
-------------------------------
Run from cron (daily). Once analyst_version 1 has accumulated >= THRESHOLD MLB
games in game_outcomes.jsonl, sends a one-time OneSignal admin push (same player
id as the data-gap alerts) with a compact v1 MLB calibration summary — reminding
you to compare against the legacy baseline. Fires once via a sentinel file.

30 games is a "did the home-fade behavior change" checkpoint, not a results
verdict (win-rate/Brier need ~100+); the push says so.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for analyst_calibration
sys.path.insert(0, str(REPO.parent))                       # for the SportsBettingBot package

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from analyst_calibration import _stats
from SportsBettingBot.notifications import push_notifier

THRESHOLD = 30
SPORT     = "baseball_mlb"
VERSION   = 1
SENTINEL  = REPO / "data" / "raw" / ".calib_reminder_mlb_v1.sent"


def main() -> None:
    outcomes = REPO / "game_outcomes.jsonl"
    if not outcomes.exists():
        print("no outcomes file"); return

    recs = []
    for line in outcomes.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    v1 = [r for r in recs if r.get("sport") == SPORT and r.get("analyst_version") == VERSION]
    n = len(v1)
    if n < THRESHOLD:
        print(f"v{VERSION} {SPORT} games: {n}/{THRESHOLD} — waiting")
        return
    if SENTINEL.exists():
        print("reminder already sent")
        return

    s = _stats(v1)
    rec = f"{s.get('bet_wins', 0)}-{s.get('bet_losses', 0)}" if s.get("n_bets") else "no bets yet"
    body = (
        f"v1 MLB hit {n} games. Behavioral check — home prob: Claude "
        f"{s['avg_claude_home']:.0%} vs market {s['avg_market_home']:.0%} vs actual "
        f"{s['home_win_rate']:.0%} (was 48/53/54 pre-change). Bets {rec}. "
        f"Results verdict still needs ~100+. Run: "
        f"python3 scripts/analyst_calibration.py --sport baseball_mlb"
    )
    push_notifier.notify_admin("MLB calibration ready (v1)", body)
    SENTINEL.write_text("sent\n")
    print(f"reminder sent (n={n})")


if __name__ == "__main__":
    main()
