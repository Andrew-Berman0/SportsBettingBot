"""
data/ufc_fetcher.py
-------------------
Fetches upcoming UFC fights, moneyline odds, and per-fighter stats.

Sources (all reachable from the production server):
  - Fight card + fighter IDs + stats: ESPN core MMA API (free, no key)
      sports.core.api.espn.com/v2/sports/mma/leagues/ufc/events → competitions
      → competitors → athlete → statistics
  - Moneyline odds: The Odds API (`mma_mixed_martial_arts`, h2h). The feed is all
    MMA, not UFC-only, so it is matched to the ESPN card by fighter name — the
    name match IS the UFC filter.

To control Odds API cost, odds are only fetched when a card is within 48h.
A UFC card's fights all share the event start time, so the bot evaluates the
whole card in one wake ~2h before it begins.
"""

import json
import logging
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CORE_BASE   = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"
CACHE_DIR   = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_ODDS_WINDOW_HOURS = 48   # only spend an Odds API credit when a card is this close


class UFCFetcher:

    def __init__(self, odds_api_key: str = ""):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0"
        self.odds_api_key = odds_api_key
        self._athlete_ref: dict[str, str] = {}        # athlete_id -> $ref
        self._stats_cache: dict[str, tuple[float, dict]] = {}  # athlete_id -> (ts, stats)
        self._parse_errors = 0   # fights that crashed in _build_fight this fetch (drift signal)

    # ------------------------------------------------------------------
    # Upcoming fights (pre-parsed game dicts, compatible with the bot pipeline)
    # ------------------------------------------------------------------

    def get_upcoming_fights(self) -> list[dict]:
        cache_file = CACHE_DIR / "odds_mma_ufc.json"
        now = datetime.now(timezone.utc)

        if cache_file.exists():
            age_min = (time.time() - cache_file.stat().st_mtime) / 60
            if age_min < 90:
                logger.info(f"UFC cache fresh ({age_min:.0f}min) — skipping fetch")
                with open(cache_file) as f:
                    return json.load(f)

        try:
            events = self._upcoming_events(now)
        except Exception as e:
            logger.error(f"UFC events fetch error: {e}")
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
            return []

        soon = [e for e in events
                if 0 <= (e["date"] - now).total_seconds() <= _ODDS_WINDOW_HOURS * 3600]
        if not soon:
            logger.info("UFC: no card within 48h — skipping odds fetch")
            with open(cache_file, "w") as f:
                json.dump([], f)
            return []

        odds_map = self._fetch_ufc_odds()
        fights: list[dict] = []
        seen: set[str] = set()
        self._parse_errors = 0
        for ev in soon:
            for comp in ev["competitions"]:
                try:
                    fight = self._build_fight(ev, comp, odds_map)
                except Exception as e:
                    self._parse_errors += 1
                    logger.warning(f"UFC build fight error: {e}")
                    continue
                if fight and fight["game_id"] not in seen:
                    seen.add(fight["game_id"])
                    fights.append(fight)

        logger.info(f"UFC: {len(fights)} fight(s) with odds on {soon[0]['name']}")
        from SportsBettingBot.notifications import push_notifier
        push_notifier.notify_parse_errors("UFC", self._parse_errors)
        with open(cache_file, "w") as f:
            json.dump(fights, f)
        return fights

    def _upcoming_events(self, now: datetime) -> list[dict]:
        r = self._get(f"{CORE_BASE}/events", params={"limit": 12})
        out = []
        for it in r.get("items", []):
            ev = self._get(it["$ref"])
            ds = ev.get("date")
            if not ds:
                continue
            try:
                d = datetime.fromisoformat(ds.replace("Z", "+00:00"))
            except ValueError:
                continue
            if d < now - timedelta(hours=6):   # already over
                continue
            out.append({
                "name":         ev.get("name", ""),
                "date":         d,
                "date_str":     ds,
                "competitions": ev.get("competitions", []),
            })
        out.sort(key=lambda e: e["date"])
        return out

    def _build_fight(self, ev: dict, comp: dict, odds_map: dict) -> dict | None:
        if "competitors" not in comp and comp.get("$ref"):
            comp = self._get(comp["$ref"])
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            return None

        fighters = []
        for c in competitors:
            aref = c.get("athlete", {}).get("$ref")
            if not aref:
                return None
            aid = aref.split("/athletes/")[1].split("?")[0]
            self._athlete_ref[aid] = aref
            athlete = self._get(aref)
            name = athlete.get("displayName")
            if not name:
                return None
            fighters.append((aid, name, self._competitor_record(c)))

        (a_id, a_name, a_rec), (b_id, b_name, b_rec) = fighters
        odds = odds_map.get(frozenset([self._norm(a_name), self._norm(b_name)]))
        if not odds:
            logger.debug(f"UFC: no odds match for {a_name} vs {b_name}")
            return None
        a_ml = odds.get(self._norm(a_name))
        b_ml = odds.get(self._norm(b_name))
        if a_ml is None or b_ml is None:
            return None

        ra = self._american_to_implied(a_ml)
        rb = self._american_to_implied(b_ml)
        tot = ra + rb
        return {
            "_pre_parsed":   True,
            "game_id":       str(comp.get("id") or f"{a_id}-{b_id}"),
            "sport":         "mma_ufc",
            "home_team":     a_name,
            "away_team":     b_name,
            "home_team_id":  a_id,
            "away_team_id":  b_id,
            "home_record":   a_rec,
            "away_record":   b_rec,
            "commence_time": ev["date_str"],
            "home_ml":       a_ml,
            "away_ml":       b_ml,
            "home_implied":  round(ra / tot, 4) if tot else 0.5,
            "away_implied":  round(rb / tot, 4) if tot else 0.5,
            "total_line":    None,
            "event":         ev["name"],
        }

    def _competitor_record(self, competitor: dict) -> str | None:
        """W-L-D record lives on the competitor (fight-card context), not the athlete."""
        rec = competitor.get("record")
        ref = rec.get("$ref") if isinstance(rec, dict) else None
        if not ref:
            return None
        try:
            items = self._get(ref).get("items", [])
            if items:
                return items[0].get("summary") or items[0].get("displayValue")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Per-fighter stats (called for fights in the analysis window)
    # ------------------------------------------------------------------

    def get_fighter_stats(self, athlete_id) -> dict:
        if not athlete_id:
            return {}
        aid = str(athlete_id)
        cached = self._stats_cache.get(aid)
        if cached and (time.time() - cached[0]) < 21600:   # 6h
            return cached[1]

        aref = self._athlete_ref.get(aid) or f"https://sports.core.api.espn.com/v2/sports/mma/athletes/{aid}"
        stats: dict = {}
        try:
            a = self._get(aref)
            stats["name"]   = a.get("displayName")
            stats["reach"]  = a.get("reach")
            stats["stance"] = (a.get("stance") or {}).get("text")
            stats["height"] = a.get("displayHeight")
            stats["weight"] = a.get("displayWeight")
            dob = a.get("dateOfBirth")
            if dob:
                try:
                    d = datetime.fromisoformat(dob.replace("Z", "+00:00"))
                    stats["age"] = int((datetime.now(timezone.utc) - d).days / 365.25)
                except ValueError:
                    pass

            rec = a.get("record")
            rec_ref = rec.get("$ref") if isinstance(rec, dict) else None
            if rec_ref:
                try:
                    items = self._get(rec_ref).get("items", [])
                    if items:
                        stats["record"] = items[0].get("summary") or items[0].get("displayValue")
                except Exception:
                    pass

            sref = (a.get("statistics") or {}).get("$ref") if isinstance(a.get("statistics"), dict) else None
            if sref:
                sd = self._get(sref)
                for cat in sd.get("splits", {}).get("categories", []):
                    for x in cat.get("stats", []):
                        v = x.get("value")
                        stats[x["name"]] = v if v is not None else x.get("displayValue")
        except Exception as e:
            logger.warning(f"UFC fighter stats error ({aid}): {e}")

        self._stats_cache[aid] = (time.time(), stats)
        return stats

    # ------------------------------------------------------------------
    # Odds (The Odds API — costs 1 credit, only when a card is within 48h)
    # ------------------------------------------------------------------

    def _fetch_ufc_odds(self) -> dict:
        if not self.odds_api_key:
            logger.warning("No Odds API key set — UFC fights will have no odds")
            return {}
        try:
            r = self.session.get(ODDS_API_URL, params={
                "apiKey":     self.odds_api_key,
                "regions":    "us",
                "markets":    "h2h",
                "oddsFormat": "american",
            }, timeout=15)
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining")
            out: dict = {}
            for g in r.json():
                bks = g.get("bookmakers") or []
                if not bks:
                    continue
                mkts = bks[0].get("markets") or []
                if not mkts:
                    continue
                outs = mkts[0].get("outcomes") or []
                if len(outs) != 2:
                    continue
                d = {self._norm(o["name"]): o.get("price") for o in outs}
                out[frozenset(d.keys())] = d
            logger.info(f"UFC odds: {len(out)} matchups from The Odds API "
                        f"(credits remaining: {remaining})")
            return out
        except Exception as e:
            logger.error(f"UFC odds fetch error: {e}")
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict:
        # ESPN core returns http:// refs; force https to avoid redirect churn.
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _norm(name: str) -> str:
        if not name:
            return ""
        n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        return " ".join("".join(c for c in n.lower() if c.isalnum() or c == " ").split())

    @staticmethod
    def _american_to_implied(odds: float) -> float:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)
