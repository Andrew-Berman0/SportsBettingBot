"""
notifications/push_notifier.py
-------------------------------
Sends web push notifications via OneSignal REST API.
Requires ONESIGNAL_APP_ID and ONESIGNAL_REST_API_KEY in .env.
Fails silently — a broken push should never crash the bot.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://onesignal.com/api/v1/notifications"
_PARSE_ALERT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _headers() -> dict | None:
    key = os.getenv("ONESIGNAL_REST_API_KEY", "")
    if not key:
        return None
    return {
        "Authorization": f"Basic {key}",
        "Content-Type": "application/json",
    }


def _app_id() -> str:
    return os.getenv("ONESIGNAL_APP_ID", "")


def _fmt_odds(odds: float) -> str:
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def _send(title: str, body: str, admin_only: bool = False) -> None:
    headers = _headers()
    app_id = _app_id()
    if not headers or not app_id:
        logger.debug("OneSignal not configured — skipping push")
        return

    payload: dict = {
        "app_id":   app_id,
        "headings": {"en": title},
        "contents": {"en": body},
        "url":      "https://wwaid.live",
    }

    if admin_only:
        admin_id = os.getenv("ONESIGNAL_ADMIN_PLAYER_ID", "")
        if not admin_id:
            logger.debug("ONESIGNAL_ADMIN_PLAYER_ID not set — skipping admin push")
            return
        payload["include_player_ids"] = [admin_id]
    else:
        payload["included_segments"] = ["All"]

    try:
        r = requests.post(_API_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        # OneSignal returns HTTP 200 even when a push reaches nobody (e.g. a stale
        # player id -> {"errors": ["All included players are not subscribed"]}).
        # Surface that loudly instead of masking it as "0 subscribers".
        errors = data.get("errors")
        if errors:
            logger.warning(f"Push NOT delivered ({title}): {errors}")
        elif not data.get("id"):
            logger.warning(f"Push returned no id ({title}): {data}")
        else:
            recipients = data.get("recipients")
            who = f"{recipients} subscriber(s)" if recipients is not None else "targeted device"
            logger.info(f"Push sent to {who}: {title}")
    except Exception as e:
        logger.warning(f"Push notification failed: {e}")


def notify_admin(title: str, body: str) -> None:
    """Admin-only push (same player id as the data-gap alerts)."""
    _send(title=title, body=body, admin_only=True)


def notify_parse_errors(source: str, n: int) -> None:
    """Admin alert when a fetcher drops events at the PARSE stage (likely API drift) —
    these never reach the data-gap detector, which runs after parsing. Deduped per
    source per UTC day so a persistent issue can't spam."""
    if n <= 0:
        return
    logger.warning(f"{source}: {n} event(s) failed to parse — possible API change")
    key = "".join(c if c.isalnum() else "_" for c in source.lower())
    sentinel = _PARSE_ALERT_DIR / f".parse_alert_{key}.day"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        if sentinel.read_text().strip() == today:
            return
    except Exception:
        pass
    try:
        _PARSE_ALERT_DIR.mkdir(parents=True, exist_ok=True)
        notify_admin(
            f"⚠ {source} parse failures",
            f"{n} {source} event(s) failed to parse and were dropped from evaluation — "
            f"likely an API shape change. Check the fetcher's parser.",
        )
        sentinel.write_text(today)
    except Exception as e:
        logger.warning(f"Parse-error alert failed ({source}): {e}")


def notify_data_missing(matchup: str, sport_label: str, missing: list[str]) -> None:
    if not missing:
        return
    _send(
        title=f"⚠ Data gap — {matchup}",
        body=f"[{sport_label}] Missing: {' | '.join(missing)} — Claude analyzed with incomplete data",
        admin_only=True,
    )


def notify_bet_placed(bet: dict, bankroll: float) -> None:
    away  = bet["away_team"]
    home  = bet["home_team"]
    team  = home if bet["bet_type"] == "home_ml" else away
    odds  = _fmt_odds(bet["odds"])
    stake = bet["stake"]
    to_win = bet["potential_payout"] - stake

    _send(
        title="WWAID just dropped a pick",
        body=f"{team} {odds} vs {away if bet['bet_type'] == 'home_ml' else home} — tap to see the reasoning",
    )
