"""Tell the owner what happened, for someone who does not open WSL.

All best effort (a failure here never stops a run):
* a copy of the status and the latest plan in a Windows folder (default: Documents\\FPL);
* a message to a Discord channel, when DISCORD_WEBHOOK_URL is set in `.env`.

Outside WSL (no powershell.exe) the copy does nothing.
"""
from __future__ import annotations

import base64
import datetime
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests


def _powershell(script: str, timeout: int = 20) -> str | None:
    exe = shutil.which("powershell.exe")
    if not exe:
        return None
    try:
        enc = base64.b64encode(script.encode("utf-16-le")).decode()
        r = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", enc], capture_output=True,
                           timeout=timeout, cwd="/mnt/c" if os.path.isdir("/mnt/c") else None)
        return r.stdout.decode("utf-8", "replace").replace("\r", "").strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def windows_dir() -> Path | None:
    """Where copies go: FPL_WINDOWS_DIR (a WSL path), else Documents\\FPL. "off" disables copies."""
    custom = os.environ.get("FPL_WINDOWS_DIR")
    if custom:
        return None if custom.lower() in ("off", "none", "0") else Path(custom)
    docs = _powershell("[Environment]::GetFolderPath('MyDocuments')")
    if not docs or not shutil.which("wslpath"):
        return None
    try:
        r = subprocess.run(["wslpath", "-u", docs], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(r.stdout.strip()) / "FPL" if r.returncode == 0 and r.stdout.strip() else None


def publish(files: dict[str, str]) -> Path | None:
    """Write {file name: text} into the Windows folder. Returns the folder, or None if unavailable."""
    d = windows_dir()
    if d is None:
        return None
    try:
        d.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
        return d
    except OSError:
        return None


# ----------------------------------------------------------------------------- Discord
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
