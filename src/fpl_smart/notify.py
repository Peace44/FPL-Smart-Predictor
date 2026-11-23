"""Tell the owner what happened, for someone who does not open WSL: a message to a Discord channel,
when DISCORD_WEBHOOK_URL is set in `.env`. Best effort: a failure here never stops a run.
"""
from __future__ import annotations

import datetime
import os
import time

import requests

DISCORD_WEBHOOKS = ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
COLOURS = {"ok": 0x2ECC71, "info": 0x3498DB, "warn": 0xE67E22, "error": 0xE74C3C}


def _clip(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def discord_payload(title: str, lines=(), problems=(), kind: str | None = None, fields: dict | None = None,
                    footer: str | None = None) -> dict:
    """One embed, kept inside Discord's size limits (6000 characters in all)."""
    kind = kind or ("warn" if problems else "info")
    embed: dict = {"title": _clip(title, 250), "color": COLOURS.get(kind, COLOURS["info"]),
                   "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if lines:
        embed["description"] = _clip("\n".join(f"• {x}" for x in lines), 2400)
    out = []
    if problems:
        out.append({"name": "Needs your attention", "value": _clip("\n".join(f"• {x}" for x in problems), 1000)})
    for k, v in (fields or {}).items():
        if v:
            out.append({"name": _clip(k, 100), "value": _clip(v, 600)})
    if out:
        embed["fields"] = out[:4]
    if footer:
        embed["footer"] = {"text": _clip(footer, 200)}
    return {"username": "FPL Autopilot", "embeds": [embed], "allowed_mentions": {"parse": []}}


def discord(title: str, lines=(), problems=(), kind: str | None = None, fields: dict | None = None,
            footer: str | None = None) -> bool:
    """Post a report to the Discord channel behind DISCORD_WEBHOOK_URL. True if Discord accepted it."""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip().strip("\"'")
    if not url.startswith(DISCORD_WEBHOOKS):      # never send team details anywhere but a Discord webhook
        return False
    try:
        body = discord_payload(title, lines, problems, kind, fields, footer)
        for attempt in range(3):
            r = requests.post(url, params={"wait": "true"}, json=body, timeout=15)
            if r.status_code == 429 and attempt < 2:      # rate limited: Discord says how long to wait
                try:
                    wait = float(r.json().get("retry_after", 1))
                except (ValueError, AttributeError):
                    wait = 1.0
                time.sleep(min(max(wait, 0.5), 5.0))
                continue
            return 200 <= r.status_code < 300
    except Exception:  # noqa: BLE001  (a report that cannot be sent must not stop a run)
        return False
    return False
