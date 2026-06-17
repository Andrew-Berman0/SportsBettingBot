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
import argparse
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
# Two failure modes, two triggers:
#  - TAIL: an EGREGIOUS spike of big single-game divergences (the MLB ERA-bet kind).
#    Coarse "go look" trigger above where healthy sports sit (NBA/WNBA ~20%, MLB ~13%).
#  - BIAS: a SYSTEMATIC directional drift — Claude consistently rating home above/below
#    the market (the WNBA home-fade kind). The tail check misses this because it's not
#    about outliers; it's the center of the distribution shifting.
THRESHOLD      = 0.30   # tail: >=30% of recent games diverged >10pt from market
BIAS_THRESHOLD = 0.06   # bias: avg Claude-home off the market by >=6pt
STATE          = REPO / "data" / "raw" / ".divergence_watch_state.json"


def _check(state: dict, key: str, bad: bool, title: str, body: str, dry: bool) -> bool:
    """Alert once on entering a bad state; re-arm silently on recovery. Returns changed."""
    alerted = state.get(key, False)
    if bad and not alerted:
        if dry:
            print(f"WOULD ALERT {key}: {title}")
            return False
        push_notifier.notify_admin(title, body)
        state[key] = True
        print(f"ALERT {key}")
        return True
    if not bad and alerted:
        if not dry:
            state[key] = False
        print(f"recovered {key}")
        return not dry
    print(f"{key}: {'alerted' if alerted else 'ok'}{' [dry]' if dry else ''}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print what would alert, send nothing")
    dry = ap.parse_args().dry_run
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
        SH = sport.split("_")[-1].upper()
        rate = s["big_divergence"] / s["n"]
        bias = s["avg_claude_home"] - s["avg_market_home"]

        changed |= _check(
            state, f"{sport}:tail", rate >= THRESHOLD,
            f"⚠ {SH} divergence spike",
            f"{s['big_divergence']}/{s['n']} of recent games diverged >10pt from the market "
            f"(avg {s['avg_divergence']:.0%}) — possible overconfidence. "
            f"Review: python3 scripts/analyst_calibration.py --sport {sport}",
            dry,
        )
        side = "home" if bias < 0 else "away"
        changed |= _check(
            state, f"{sport}:bias", abs(bias) >= BIAS_THRESHOLD,
            f"⚠ {SH} systematic {side} lean",
            f"Claude rates home {s['avg_claude_home']:.0%} vs market {s['avg_market_home']:.0%} "
            f"({bias * 100:+.0f}pt) over last {s['n']} — persistent {side} bias vs the market "
            f"(actual home win {s['home_win_rate']:.0%}). "
            f"Review: python3 scripts/analyst_calibration.py --sport {sport}",
            dry,
        )

    if changed and not dry:
        STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
