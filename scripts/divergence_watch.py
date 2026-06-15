"""
scripts/divergence_watch.py
---------------------------
Early-warning for the MLB-style failure mode (overconfidence vs an efficient
market), for ANY sport — without waiting for bets to settle. Divergence
|Claude - market| is logged on every evaluated game (passes included), so a
sport straying implausibly far from the market is detectable immediately.

Run from cron (daily). For each sport, looks at its most recent WINDOW evaluated
games; if >=THRESHOLD of them diverged >10pt from the market, pushes a one-time
admin alert (same channel as the data-gap alerts). Re-arms silently when the
sport settles back down, so it alerts on the transition, not every day.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # analyst_calibration
sys.path.insert(0, str(REPO.parent))                       # SportsBettingBot package

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from analyst_calibration import _stats
from SportsBettingBot.notifications import push_notifier

WINDOW    = 15     # each sport's most recent N evaluated games
MIN_GAMES = 10     # need at least this many to judge
THRESHOLD = 0.20   # alert if >=20% of recent games diverged >10pt from market
STATE     = REPO / "data" / "raw" / ".divergence_watch_state.json"


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

    by_sport: dict = defaultdict(list)
    for r in recs:
        if {"claude_home_prob", "book_home_prob", "home_won"} <= r.keys():
            by_sport[r.get("sport", "?")].append(r)

    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}
    changed = False

    for sport, rs in by_sport.items():
        rs = sorted(rs, key=lambda x: x.get("logged_at", ""))[-WINDOW:]
        if len(rs) < MIN_GAMES:
            print(f"{sport}: {len(rs)} recent games (<{MIN_GAMES}) — skip")
            continue
        s = _stats(rs)
        rate = s["big_divergence"] / s["n"]
        alerted = state.get(sport, False)

        if rate >= THRESHOLD and not alerted:
            push_notifier.notify_admin(
                f"⚠ {sport.split('_')[-1].upper()} divergence spike",
                f"{s['big_divergence']}/{s['n']} of recent games diverged >10pt from the "
                f"market (avg {s['avg_divergence']:.0%}) — possible overconfidence vs an "
                f"efficient market. Review: python3 scripts/analyst_calibration.py --sport {sport}",
            )
            state[sport] = True; changed = True
            print(f"ALERT {sport}: {rate:.0%} >10pt over last {s['n']}")
        elif rate < THRESHOLD and alerted:
            state[sport] = False; changed = True   # recovered — re-arm silently
            print(f"recovered {sport}: {rate:.0%}")
        else:
            print(f"{sport}: {rate:.0%} >10pt ({'alerted' if alerted else 'ok'})")

    if changed:
        STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
