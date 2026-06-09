"""
notifications/push_notifier.py
-------------------------------
Sends web push notifications via OneSignal REST API.
Requires ONESIGNAL_APP_ID and ONESIGNAL_REST_API_KEY in .env.
Fails silently — a broken push should never crash the bot.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://onesignal.com/api/v1/notifications"


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


def _send(title: str, body: str) -> None:
    headers = _headers()
    app_id = _app_id()
    if not headers or not app_id:
        logger.debug("OneSignal not configured — skipping push")
        return
    try:
        r = requests.post(
            _API_URL,
            json={
                "app_id":            app_id,
                "included_segments": ["All"],
                "headings":          {"en": title},
                "contents":          {"en": body},
                "url":               "https://wwaid.live",
            },
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        recipients = r.json().get("recipients", 0)
        logger.info(f"Push sent to {recipients} subscriber(s): {title}")
    except Exception as e:
        logger.warning(f"Push notification failed: {e}")


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
