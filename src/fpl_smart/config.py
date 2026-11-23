"""Paths, environment and season constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

PAST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = "2026-27"

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()


def team_id() -> int:
    return int(os.environ["FPL_TEAM_ID"])


def odds_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY")
