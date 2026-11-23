"""Current state of an FPL entry: squad, selling prices, bank, free transfers, chips.

From the public API alone (no login) free transfers have to be reconstructed from the transfer
history. When the signed-in "my team" data is passed in (account.py), the squad, selling prices,
bank, free transfers and any chip already played for the deadline are read from it instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import fpl_api

MAX_FT = 5


@dataclass
class TeamState:
    team_id: int
    next_gw: int
    squad: list[int]                      # 15 element ids
    buy_price: dict[int, int]             # tenths of a million
    sell_price: dict[int, int]
    bank: int
    free_transfers: int
    chips: list[dict] = field(default_factory=list)  # unused chips: {"name", "gws" it can be played in}
    captain: int | None = None
    vice: int | None = None
    lineup: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    total_points: int = 0
    overall_rank: int | None = None
    active_chip: str | None = None        # chip already played for the upcoming deadline (signed-in data only)


def selling_price(buy: int, now: int) -> int:
    """FPL keeps half of any profit (rounded down to 0.1m)."""
    return buy + (now - buy) // 2 if now > buy else now


def free_transfers_from_history(history: dict, started_gw: int, next_gw: int) -> int:
    """Replay the season: +1 a week up to five banked; a wildcard or free hit week leaves the
    count unchanged; taking a hit means all free transfers were used."""
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    by_gw = {g["event"]: g for g in history.get("current", [])}
    ft = 1
    for gw in range(started_gw + 1, next_gw):
        g = by_gw.get(gw)
        if g is None:
            ft = min(MAX_FT, ft + 1)
            continue
        if chips.get(gw) in ("wildcard", "freehit"):
            continue
        used = g["event_transfers"]
        ft = 1 if g["event_transfers_cost"] > 0 else min(MAX_FT, max(ft - used, 0) + 1)
    return ft


def load(team_id: int, bs: dict | None = None, free_transfers: int | None = None,
         private: dict | None = None) -> TeamState:
    """`private`: the signed-in "my team" response, when there is one."""
    bs = bs or fpl_api.bootstrap()
    els = {e["id"]: e for e in bs["elements"]}
    cur, nxt = fpl_api.current_and_next_gw(bs)
    if nxt is None:
        raise RuntimeError("Season is over.")
    ent = fpl_api.entry(team_id, 0.0)
    hist = fpl_api.entry_history(team_id, 0.0)
    transfers = fpl_api.entry_transfers(team_id, 0.0)
    notes: list[str] = []
    started = ent.get("started_event") or 1

    used_chip = {c["event"]: c["name"] for c in hist.get("chips", [])}
    base_gw = cur
    if base_gw is not None and used_chip.get(base_gw) == "freehit":
        base_gw -= 1          # a free hit squad reverts, so the real squad is the previous one
        notes.append(f"GW{cur} was a Free Hit; using the GW{base_gw} squad.")
    if base_gw is None or base_gw < started:
        raise RuntimeError("This entry has no completed gameweek yet; pick an initial squad first.")
    picks = fpl_api.entry_picks(team_id, base_gw, 0.0)
    squad = [p["element"] for p in picks["picks"]]
    lineup = [p["element"] for p in picks["picks"] if p["position"] <= 11]
    captain = next((p["element"] for p in picks["picks"] if p["is_captain"]), None)
    vice = next((p["element"] for p in picks["picks"] if p["is_vice_captain"]), None)
    bank = picks["entry_history"]["bank"]

    active = next((c.get("name") for c in (private or {}).get("chips") or []
                   if c.get("status_for_entry") == "active" or c.get("is_pending") is True), None)
    # transfers already confirmed for the upcoming gameweek (under an active free hit they build
    # the one-week squad, not this one)
    pending = [] if active == "freehit" else [t for t in transfers if t["event"] == nxt]
    for t in sorted(pending, key=lambda t: t["time"]):
        if t["element_out"] in squad:
            squad[squad.index(t["element_out"])] = t["element_in"]
            bank += t["element_out_cost"] - t["element_in_cost"]
    if pending:
        notes.append(f"{len(pending)} transfer(s) already made for GW{nxt} are included.")

    buy: dict[int, int] = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        buy[t["element_in"]] = t["element_in_cost"]
    for p in squad:
        if p not in buy:  # original pick: bought at the price when the squad was created
            e = els[p]
            if started <= 1:
                buy[p] = e["now_cost"] - e["cost_change_start"]
            else:
                h = fpl_api.element_summary(p)["history"]
                at = [r["value"] for r in h if r["round"] == started]
                buy[p] = at[0] if at else e["now_cost"] - e["cost_change_start"]
    sell = {p: selling_price(buy[p], els[p]["now_cost"]) for p in squad}

    ft_read = None
    if private and active != "freehit":
        # the account itself: exact squad, selling prices and bank
        mine = sorted(private["picks"], key=lambda p: p["position"])
        squad = [p["element"] for p in mine]
        lineup = [p["element"] for p in mine if p["position"] <= 11]
        captain = next((p["element"] for p in mine if p.get("is_captain")), captain)
        vice = next((p["element"] for p in mine if p.get("is_vice_captain")), vice)
        buy = {p["element"]: p.get("purchase_price", buy.get(p["element"], els[p["element"]]["now_cost"])) for p in mine}
        sell = {p["element"]: p.get("selling_price", sell.get(p["element"], els[p["element"]]["now_cost"])) for p in mine}
        tr = private.get("transfers") or {}
        bank = tr.get("bank", bank)
        if active != "wildcard" and tr.get("status") != "unlimited" and tr.get("limit") is not None:
            ft_read = max(0, int(tr["limit"]) - int(tr.get("made") or 0))
    if active:
        notes.append(f"The {active} chip is already active for GW{nxt}.")

    if free_transfers is not None:
        ft = free_transfers
    elif ft_read is not None:
        ft = ft_read
    else:
        ft = free_transfers_from_history(hist, started, nxt)
        if used_chip.get(nxt) not in ("wildcard", "freehit") and not active:
            ft = max(ft - len(pending), 0)
        notes.append(f"Free transfers ({ft}) are reconstructed from your transfer history; "
                     "check the number on the FPL site and override with --ft if it differs.")

    # Each chip comes twice a season (one per half); each copy has its own window and expires.
    chips: list[dict] = []
    for c in bs.get("chips", []):
        window = list(range(max(c["start_event"], nxt), c["stop_event"] + 1))
        used = any(u["name"] == c["name"] and c["start_event"] <= u["event"] <= c["stop_event"]
                   for u in hist.get("chips", []))
        if not used and window:
            chips.append({"name": c["name"], "gws": window})

    return TeamState(team_id, nxt, squad, {p: buy[p] for p in squad}, sell, bank, ft, chips,
                     captain, vice, lineup, notes, ent.get("summary_overall_points") or 0,
                     ent.get("summary_overall_rank"), active)
