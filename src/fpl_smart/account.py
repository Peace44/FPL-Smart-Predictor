"""Signed-in access to your own FPL account: read "my team", submit transfers and line-ups.

FPL signs users in through OpenID Connect at account.premierleague.com, and its API takes the
resulting access token in an `X-API-Authorization: Bearer ...` header. Access tokens are
short-lived; a refresh token buys new ones and is itself replaced on every use, so the newest
pair is kept in data/auth/token.json (never committed, readable only by you).

The first refresh token comes from your browser (README, "Letting it act for you") and goes in
`.env` as FPL_REFRESH_TOKEN. After that the token file is the source of truth; pasting a new
value into `.env` replaces it.

None of these endpoints are documented by the Premier League, so nothing here is trusted on its
own: every write is followed by a read that checks the account is in the state that was asked
for (execute.py).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import requests

from .config import DATA

API = "https://fantasy.premierleague.com/api"
TOKEN_URL = "https://account.premierleague.com/as/token"
CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"      # the FPL website's own public client id
STORE = DATA / "auth" / "token.json"
SITE = "https://fantasy.premierleague.com"


class AuthError(RuntimeError):
    """The sign-in is missing or no longer valid: a new refresh token is needed."""


class ApiError(RuntimeError):
    """FPL answered with an error, or could not be reached."""

    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(msg)
        self.status, self.body = status, body


def parse_refresh_token(pasted: str) -> str:
    """Accept either the bare token or the whole JSON value of the browser's `oidc.user:...` entry."""
    s = (pasted or "").strip().strip("'")
    if s.startswith("{"):
        try:
            tok = json.loads(s).get("refresh_token")
            if isinstance(tok, str) and tok:
                return tok
        except ValueError:
            pass
    return s.strip('"')


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class Account:
    def __init__(self, entry_id: int, seed: str | None = None, store: Path = STORE, http=None):
        self.entry_id = int(entry_id)
        self.seed = parse_refresh_token(seed) if seed else None
        self.store = Path(store)
        self.http = http or requests.Session()

    @classmethod
    def from_env(cls) -> "Account | None":
        """None when no sign-in has ever been set up."""
        from .config import team_id
        seed = os.environ.get("FPL_REFRESH_TOKEN") or None
        if not seed and not STORE.exists():
            return None
        return cls(team_id(), seed)

    # ------------------------------------------------------------------------- tokens
    def _read(self) -> dict:
        try:
            return json.loads(self.store.read_text())
        except (OSError, ValueError):
            return {}

    def _write(self, d: dict) -> None:
        """Atomic, private. A refresh token that was used but not saved is gone for good."""
        self.store.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.store.parent, 0o700)
        tmp = self.store.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.store)

    @contextlib.contextmanager
    def _locked(self):
        """One process at a time: two refreshing with the same token would strand one of them."""
        self.store.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store.parent / ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            yield

    def _exchange(self, refresh_token: str) -> dict | None:
        """New tokens for a refresh token; None when FPL says the token is no longer good."""
        try:
            r = self.http.post(TOKEN_URL, timeout=30, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
                               data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CLIENT_ID})
        except requests.RequestException as e:
            raise ApiError(f"could not reach the FPL sign-in service: {e!r}") from e
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code == 200 and body.get("access_token"):
            return body
        if r.status_code in (400, 401) and body.get("error") in ("invalid_grant", "invalid_request", "invalid_token"):
            return None
        raise ApiError(f"FPL sign-in service answered HTTP {r.status_code}", r.status_code, r.text[:300])

    def access_token(self, force: bool = False) -> str:
        with self._locked():
            st = self._read()
            if not force and st.get("access_token") and st.get("expires_at", 0) > time.time() + 30:
                return st["access_token"]
            # A token pasted since the last run goes first. One already exchanged is never sent
            # again: replaying a used refresh token can make the server revoke the whole sign-in.
            fresh_seed = self.seed if self.seed and _sha(self.seed) != st.get("seed_sha") else None
            for tok, is_seed in ((fresh_seed, True), (st.get("refresh_token"), False)):
                if not tok:
                    continue
                got = self._exchange(tok)
                if got is None:
                    if is_seed:      # remember it failed, so it is not retried on every run
                        self._write({**st, "seed_sha": _sha(tok)})
                        st = self._read()
                    continue
                self._write({
                    "access_token": got["access_token"],
                    "expires_at": time.time() + float(got.get("expires_in") or 300),
                    "refresh_token": got.get("refresh_token") or tok,
                    "seed_sha": _sha(tok) if is_seed else st.get("seed_sha"),
                    "refreshed": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return got["access_token"]
        raise AuthError("the FPL sign-in has expired or was never set up: copy a new refresh token from "
                        "your browser into FPL_REFRESH_TOKEN in .env (README, \"Letting it act for you\")")

    # ------------------------------------------------------------------------- API
    def _call(self, method: str, path: str, body: dict | None = None):
        url = f"{API}/{path}"
        last = None
        for attempt in range(2):
            headers = {"X-API-Authorization": f"Bearer {self.access_token(force=attempt > 0)}",
                       "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            if body is not None:
                headers.update({"Content-Type": "application/json", "Origin": SITE, "Referer": SITE + "/"})
            try:
                r = self.http.request(method, url, headers=headers, json=body, timeout=30)
            except requests.RequestException as e:
                raise ApiError(f"could not reach FPL ({method} {path}): {e!r}") from e
            if r.status_code in (401, 403) and attempt == 0:
                last = r
                continue                      # the access token may have just lapsed: refresh once
            if r.status_code >= 400:
                if r.status_code in (401, 403):
                    raise AuthError(f"FPL refused the sign-in for {path} (HTTP {r.status_code}): {r.text[:200]}")
                raise ApiError(f"FPL rejected {method} {path} (HTTP {r.status_code}): {r.text[:400]}",
                               r.status_code, r.text[:2000])
            try:
                return r.json() if r.text.strip() else {}
            except ValueError:
                return {}
        raise AuthError(f"FPL refused the sign-in for {path} (HTTP {last.status_code})")

    def me(self) -> dict:
        return self._call("GET", "me/")

    def my_team(self) -> dict:
        """Picks with selling prices, bank, free transfers and chip status for the upcoming deadline."""
        return self._call("GET", f"my-team/{self.entry_id}/")

    def post_transfers(self, gw: int, transfers: list[dict], chip: str | None = None) -> dict:
        """transfers: [{"element_in", "element_out", "purchase_price", "selling_price"}].
        chip: "wildcard" / "freehit" to play it with these transfers."""
        return self._call("POST", "transfers/", {"chip": chip, "entry": self.entry_id, "event": int(gw),
                                                 "transfers": transfers})

    def post_picks(self, picks: list[dict], chip: str | None = None) -> dict:
        """picks: all 15 as {"element", "position" (1-11 start, 12 bench GK, 13-15 bench order),
        "is_captain", "is_vice_captain"}. chip: "bboost" / "3xc" to play it this gameweek."""
        return self._call("POST", f"my-team/{self.entry_id}/", {"picks": picks, "chip": chip})


# ----------------------------------------------------------------------------- reading "my team"
def shape_problem(my_team) -> str | None:
    """What is wrong with a "my team" answer, if it is not laid out the way this code reads it.
    The endpoint is undocumented and can change; acting on a misread answer is how points get lost."""
    if not isinstance(my_team, dict):
        return "the answer is not an object"
    picks = my_team.get("picks")
    if not isinstance(picks, list) or len(picks) != 15:
        return "it does not list 15 picks"
    for p in picks:
        if not isinstance(p, dict) or not all(isinstance(p.get(k), int) for k in ("element", "position", "selling_price")):
            return "a pick lacks its player, position or selling price"
    if sorted(p["position"] for p in picks) != list(range(1, 16)):
        return "the picks are not in positions 1 to 15"
    tr = my_team.get("transfers")
    if not isinstance(tr, dict) or not all(k in tr for k in ("bank", "limit", "made", "cost", "status")):
        return "the transfer details (bank, limit, made, cost, status) are incomplete"
    if tr["status"] not in ("cost", "unlimited"):
        return f"unknown transfer status {tr['status']!r}"
    chips = my_team.get("chips")
    if not isinstance(chips, list) or not all(
            isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("status_for_entry"), str)
            for c in chips):
        return "the chips are not listed with a name and a status"
    return None


def active_chip(my_team: dict) -> str | None:
    """The chip already played for the upcoming deadline, if any."""
    return next((c.get("name") for c in my_team.get("chips") or []
                 if c.get("status_for_entry") == "active" or c.get("is_pending") is True), None)


def points_spent(my_team: dict) -> int:
    """Points already deducted for transfers made for the upcoming deadline. In FPL's data `cost`
    is the price of one extra transfer (4), `limit` the free transfers the gameweek started with
    and `made` the transfers made so far.

    Nothing is charged while a Wildcard or Free Hit is active (playing one gives back hits already
    taken that week), whatever those fields say: with the Wildcard active in GW11 FPL still showed
    `status: "cost"` and a limit, so a count of the free transfers in `made` would read as a charge."""
    if active_chip(my_team) in ("wildcard", "freehit"):
        return 0
    tr = my_team.get("transfers") or {}
    if tr.get("status") != "cost" or tr.get("limit") is None:
        return 0
    return int(tr.get("cost") or 4) * max(0, int(tr.get("made") or 0) - int(tr["limit"]))


def chip_available(my_team: dict, name: str) -> bool:
    return any(c.get("name") == name and c.get("status_for_entry") in ("available", "active")
               for c in my_team.get("chips") or [])


def unlimited_transfers(my_team: dict) -> bool:
    return (my_team.get("transfers") or {}).get("status") == "unlimited" or active_chip(my_team) in ("wildcard", "freehit")


def free_transfers_left(my_team: dict) -> int | None:
    """Free transfers still unused for the upcoming deadline; None when FPL does not say."""
    tr = my_team.get("transfers") or {}
    if tr.get("limit") is None:
        return None
    return max(0, int(tr["limit"]) - int(tr.get("made") or 0))
