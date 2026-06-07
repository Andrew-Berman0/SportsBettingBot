"""
notifications/email_notifier.py
--------------------------------
Sends bet placement and settlement emails via SendGrid.
Fails silently — a broken email should never crash the bot.
"""

import logging
import os

logger = logging.getLogger(__name__)

FROM_EMAIL = "andrewberman2015@gmail.com"
TO_EMAIL   = "bloplanopepper@gmail.com"


def _client():
    try:
        from sendgrid import SendGridAPIClient
        key = os.getenv("SENDGRID_KEY", "")
        if not key:
            return None
        return SendGridAPIClient(key)
    except Exception:
        return None


def _send(subject: str, body: str) -> None:
    sg = _client()
    if not sg:
        logger.warning("SendGrid not configured — skipping email")
        return
    try:
        from sendgrid.helpers.mail import Mail
        msg = Mail(
            from_email=FROM_EMAIL,
            to_emails=TO_EMAIL,
            subject=subject,
            plain_text_content=body,
        )
        sg.send(msg)
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.warning(f"Email send failed: {e}")


def _fmt_odds(odds: float) -> str:
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def notify_bet_placed(bet: dict, bankroll: float) -> None:
    away   = bet["away_team"]; home = bet["home_team"]
    side   = "HOME" if bet["bet_type"] == "home_ml" else "AWAY"
    team   = home if bet["bet_type"] == "home_ml" else away
    sport  = bet.get("sport", "").upper().replace("_", " ")
    odds   = _fmt_odds(bet["odds"])
    stake  = bet["stake"]
    payout = bet["potential_payout"]
    to_win = payout - stake

    subject = f"Bet Placed — {team} {odds} ({away} @ {home})"

    body = f"""WWAID — Bet Placed
==================
Sport:    {sport}
Game:     {away} @ {home}
Bet:      {side} ({team}) {odds}
Stake:    ${stake:.2f}
To win:   ${to_win:.2f}
Payout:   ${payout:.2f}

Reasoning:
{bet.get("reasoning", "N/A")}

Bankroll after bet: ${bankroll:,.2f}
"""
    _send(subject, body)


def notify_bet_settled(bet: dict, bankroll: float) -> None:
    away   = bet["away_team"]; home = bet["home_team"]
    won    = bet["status"] == "won"
    pnl    = bet.get("pnl", 0)
    team   = home if bet["bet_type"] == "home_ml" else away
    sport  = bet.get("sport", "").upper().replace("_", " ")
    odds   = _fmt_odds(bet["odds"])
    score  = ""
    if bet.get("home_score") is not None:
        score = f"{home} {bet['home_score']}, {away} {bet['away_score']}"

    result  = "WON" if won else "LOST"
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    subject = f"Bet {result} {pnl_str} — {team} ({away} @ {home})"

    body = f"""WWAID — Bet Settled
===================
Result:   {result}
Sport:    {sport}
Game:     {away} @ {home}
{"Score:    " + score if score else ""}
Bet:      {team} {odds}
Stake:    ${bet['stake']:.2f}
P&L:      {pnl_str}

Bankroll: ${bankroll:,.2f}
"""
    _send(subject, body)
