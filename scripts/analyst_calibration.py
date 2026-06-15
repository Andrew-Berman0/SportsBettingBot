"""
scripts/analyst_calibration.py
------------------------------
Reads game_outcomes.jsonl and reports, per sport, broken out by analyst_version:
  - Home bias        : avg Claude home prob vs market vs actual home win rate
  - Brier score      : Claude vs market (predictive value; lower is better)
  - Pick-winner acc  : Claude vs market
  - Edge accuracy    : home-lean vs away-lean hit rate (tests home-fade bias)
  - Accuracy by edge size (3-5% / 5-8% / 8%+)
  - Bet record + flat-unit ROI

Stamping analyst_version on each result lets you compare logic versions cleanly:
records logged before versioning have no version (shown as "legacy").

Usage (from repo root):
    python3 scripts/analyst_calibration.py
    python3 scripts/analyst_calibration.py --sport baseball_mlb
    python3 scripts/analyst_calibration.py --version 1
    python3 scripts/analyst_calibration.py --sport baseball_mlb --version legacy
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

OUTCOMES_FILE = Path(__file__).resolve().parent.parent / "game_outcomes.jsonl"


def _american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def _stats(records: list) -> dict:
    n = len(records)
    out: dict = {"n": n, "n_bets": 0}
    if not n:
        return out
    hw = sum(1 for r in records if r["home_won"])
    out["home_win_rate"]   = hw / n
    out["avg_claude_home"] = sum(r["claude_home_prob"] for r in records) / n
    out["avg_market_home"] = sum(r["book_home_prob"] for r in records) / n
    out["avg_divergence"]  = sum(abs(r["claude_home_prob"] - r["book_home_prob"]) for r in records) / n
    out["big_divergence"]  = sum(1 for r in records if abs(r["claude_home_prob"] - r["book_home_prob"]) > 0.10)
    out["brier_claude"]    = sum((r["claude_home_prob"] - r["home_won"]) ** 2 for r in records) / n
    out["brier_market"]    = sum((r["book_home_prob"]  - r["home_won"]) ** 2 for r in records) / n
    out["acc_claude"]      = sum(1 for r in records if (r["claude_home_prob"] > 0.5) == r["home_won"]) / n
    out["acc_market"]      = sum(1 for r in records if (r["book_home_prob"]  > 0.5) == r["home_won"]) / n

    for side in ("home", "away"):
        g = [r for r in records if r.get("edge_side") == side]
        out[f"{side}_lean_n"]   = len(g)
        out[f"{side}_lean_acc"] = (sum(1 for r in g if r.get("claude_correct")) / len(g)) if g else None

    buckets: dict = defaultdict(list)
    for r in records:
        if not r.get("edge_side"):
            continue
        e = max(r.get("home_edge", 0), r.get("away_edge", 0))
        key = "3-5%" if e < 0.05 else ("5-8%" if e < 0.08 else "8%+")
        buckets[key].append(1 if r.get("claude_correct") else 0)
    out["edge_buckets"] = {k: (sum(v), len(v)) for k, v in buckets.items()}

    bets = [r for r in records if r.get("bet_placed") and r.get("bet_odds") is not None]
    out["n_bets"] = len(bets)
    if bets:
        w = sum(1 for r in bets if r.get("claude_correct"))
        pnl = sum((_american_to_decimal(r["bet_odds"]) - 1) if r.get("claude_correct") else -1 for r in bets)
        out["bet_wins"], out["bet_losses"] = w, len(bets) - w
        out["bet_units"], out["bet_roi"]   = pnl, pnl / len(bets)
    return out


def _print_group(label: str, s: dict) -> None:
    print(f"\n  {label}: {s['n']} games, {s['n_bets']} bets")
    if not s["n"]:
        return
    better = "Claude better" if s["brier_claude"] < s["brier_market"] else "market better"
    print(f"    Home bias : Claude {s['avg_claude_home']:.0%} | market {s['avg_market_home']:.0%} | actual {s['home_win_rate']:.0%}")
    print(f"    Divergence: avg |Claude-market| {s['avg_divergence']:.0%} | >10pt divergences: {s['big_divergence']}/{s['n']}")
    print(f"    Brier     : Claude {s['brier_claude']:.4f} | market {s['brier_market']:.4f}  ({better})")
    print(f"    Pick-win  : Claude {s['acc_claude']:.0%} | market {s['acc_market']:.0%}")
    hl = f"{s['home_lean_acc']:.0%}" if s["home_lean_acc"] is not None else "—"
    al = f"{s['away_lean_acc']:.0%}" if s["away_lean_acc"] is not None else "—"
    print(f"    Edge acc  : home-lean {hl} (n={s['home_lean_n']}) | away-lean {al} (n={s['away_lean_n']})")
    if s["edge_buckets"]:
        bk = " | ".join(f"{k} {c}/{t} ({c/t:.0%})" for k, (c, t) in sorted(s["edge_buckets"].items()))
        print(f"    By edge   : {bk}")
    if s["n_bets"]:
        print(f"    Bets      : {s['bet_wins']}-{s['bet_losses']} | {s['bet_units']:+.2f}u | ROI {s['bet_roi'] * 100:+.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport",   help="filter to one sport key (e.g. baseball_mlb)")
    ap.add_argument("--version", help="filter to one analyst_version (int, or 'legacy' for unstamped)")
    args = ap.parse_args()

    if not OUTCOMES_FILE.exists():
        print(f"No outcomes file at {OUTCOMES_FILE}")
        return

    records = []
    for line in OUTCOMES_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if args.sport:
        records = [r for r in records if r.get("sport") == args.sport]
    if args.version:
        want = None if args.version.lower() == "legacy" else int(args.version)
        records = [r for r in records if r.get("analyst_version") == want]

    print(f"=== analyst calibration — {len(records)} records from {OUTCOMES_FILE.name} ===")
    if not records:
        return

    by_ver: dict = defaultdict(list)
    for r in records:
        by_ver[r.get("analyst_version")].append(r)

    # None (legacy) sorts first, then ascending version
    for ver in sorted(by_ver, key=lambda v: (v is not None, v if v is not None else -1)):
        label = "legacy (no version)" if ver is None else f"analyst_version {ver}"
        recs  = by_ver[ver]
        print(f"\n{'=' * 64}\n{label} — {len(recs)} records\n{'=' * 64}")
        by_sport: dict = defaultdict(list)
        for r in recs:
            by_sport[r.get("sport", "?")].append(r)
        for sport in sorted(by_sport, key=lambda s: -len(by_sport[s])):
            _print_group(sport, _stats(by_sport[sport]))


if __name__ == "__main__":
    main()
