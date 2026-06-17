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
from SportsBettingBot.models.claude_analyst import analyst_version_for

THRESHOLD    = 30
# Sports whose analyst was recently changed and we want a 30-game checkpoint on.
# Each is watched at its CURRENT version with its own per-version sentinel, so the
# ping follows the latest logic and fires once.
WATCH_SPORTS = ["baseball_mlb", "basketball_wnba"]


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

    for sport in WATCH_SPORTS:
        version  = analyst_version_for(sport)
        sentinel = REPO / "data" / "raw" / f".calib_reminder_{sport}_v{version}.sent"
        cur = [r for r in recs if r.get("sport") == sport and r.get("analyst_version") == version]
        n = len(cur)
        if n < THRESHOLD:
            print(f"{sport} v{version}: {n}/{THRESHOLD} — waiting")
            continue
        if sentinel.exists():
            print(f"{sport} v{version}: reminder already sent")
            continue

        s = _stats(cur)
        SH = sport.split("_")[-1].upper()
        rec = f"{s.get('bet_wins', 0)}-{s.get('bet_losses', 0)}" if s.get("n_bets") else "no bets yet"
        body = (
            f"v{version} {SH} hit {n} games. Home prob Claude {s['avg_claude_home']:.0%} vs "
            f"market {s['avg_market_home']:.0%} vs actual {s['home_win_rate']:.0%}. "
            f"Divergence avg {s['avg_divergence']:.0%}, {s['big_divergence']} >10pt. Bets {rec}. "
            f"Run: python3 scripts/analyst_calibration.py --sport {sport}"
        )
        push_notifier.notify_admin(f"{SH} calibration ready (v{version})", body)
        sentinel.write_text("sent\n")
        print(f"{sport} v{version}: reminder sent (n={n})")


if __name__ == "__main__":
    main()
