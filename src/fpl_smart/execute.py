"""Carry out a recommendation on the real FPL account: transfers, chip, line-up, captain.

Transfers cannot be undone and a wrong request could cost points, so this is deliberately
suspicious of everything, including itself:

* Nothing is sent unless the plan passes the game's own rules here first (squad shape, three per
  club, budget at the account's real selling prices, a legal eleven).
* It works from what the account looks like *now*, not from what the plan assumed, and refuses
  if the points cost would be higher than the plan (or the caller's cap) allows.
* A wildcard or free hit is played with a single transfer first, one that a free transfer would
  cover anyway. Only when FPL then reports unlimited transfers are the rest sent. If the chip did
  not register, it stops having spent nothing.
* After every write the account is read back and compared with what was asked for.
* Anything unexpected stops the run and is reported; it never tries to "fix" a surprise.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import account as acc
from .config import REPORTS
from .planner import Recommendation
from .solver import MAX_PLAY, MIN_PLAY, SQUAD, GWPlan

MIN_MINUTES = 10          # never start writing this close to the deadline
CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}


class Refused(RuntimeError):
    """A safety check failed before the request it guards was sent."""


class Mismatch(RuntimeError):
    """FPL accepted a request, but the account is not in the state that was asked for."""


@dataclass
class Outcome:
    gw: int
    dry_run: bool = False
    actions: list[str] = field(default_factory=list)     # what was done (or would be), in order
    warnings: list[str] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)   # every request body sent (or that would be)
    changed: bool = False                                # something on the account was altered
    verified: bool = False                               # the final read-back matched the plan
    error: str | None = None
    points_cost: int | None = None                       # points spent on transfers this gameweek, per FPL, afterwards
    planned_hits: int = 0                                # -4s these transfers were expected to cost
    when: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


# ----------------------------------------------------------------------------- checks
def check_target(w: GWPlan, els: dict) -> None:
    """The planned squad and line-up must be legal under the game's rules before anything is sent."""
    missing = [p for p in w.squad if p not in els]
    if missing:
        raise Refused(f"the plan includes players FPL does not list: {missing}")
    pos = {p: els[p]["element_type"] for p in w.squad}
    if len(w.squad) != 15 or len(set(w.squad)) != 15:
        raise Refused("the planned squad is not 15 different players")
    count = {k: sum(1 for p in w.squad if pos[p] == k) for k in SQUAD}
    if count != SQUAD:
        raise Refused(f"the planned squad has the wrong shape: {count}")
    clubs: dict[int, int] = {}
    for p in w.squad:
        clubs[els[p]["team"]] = clubs.get(els[p]["team"], 0) + 1
    if max(clubs.values()) > 3:
        raise Refused("the planned squad has more than three players from one club")
    if len(w.lineup) != 11 or sorted(w.lineup + w.bench) != sorted(w.squad):
        raise Refused("the planned line-up and bench do not add up to the squad")
    n = {k: sum(1 for p in w.lineup if pos[p] == k) for k in SQUAD}
    if any(not MIN_PLAY[k] <= n[k] <= MAX_PLAY[k] for k in SQUAD):
        raise Refused(f"the planned line-up is not a legal formation: {n}")
    if pos[w.bench[0]] != 1 or any(pos[p] == 1 for p in w.bench[1:]):
        raise Refused("the planned bench must be the spare goalkeeper followed by three outfield players")
    if w.captain not in w.lineup or w.vice not in w.lineup or w.captain == w.vice:
        raise Refused("the planned captain and vice-captain must be two different starters")


def pair_transfers(out: list[int], inn: list[int], els: dict, sell: dict) -> list[dict]:
    """FPL takes transfers as like-for-like swaps: pair each sale with a purchase in the same position."""
    pairs = []
    for k in SQUAD:
        o = sorted((p for p in out if els[p]["element_type"] == k), key=lambda p: (-sell[p], p))
        i = sorted((p for p in inn if els[p]["element_type"] == k), key=lambda p: (-els[p]["now_cost"], p))
        if len(o) != len(i):
            raise Refused("the planned transfers do not swap like for like by position")
        pairs += [{"element_in": b, "element_out": a, "purchase_price": int(els[b]["now_cost"]),
                   "selling_price": int(sell[a])} for a, b in zip(o, i)]
    return pairs


def first_swap(sold: list[int], bought: list[int], current: list[int], els: dict, sell: dict, bank: int) -> dict:
    """The single transfer a wildcard / free hit is played with. On its own it must leave a legal
    squad (in budget, three per club), so take the swap that frees the most money."""
    clubs: dict[int, int] = {}
    for p in current:
        clubs[els[p]["team"]] = clubs.get(els[p]["team"], 0) + 1
    best = None
    for o in sold:
        for i in bought:
            if els[o]["element_type"] != els[i]["element_type"]:
                continue
            after = clubs.get(els[i]["team"], 0) + 1 - (els[o]["team"] == els[i]["team"])
            money = bank + sell[o] - els[i]["now_cost"]
            if after <= 3 and money >= 0 and (best is None or money > best[0]):
                best = (money, o, i)
    if best is None:
        raise Refused("no single transfer keeps the squad legal on its own; play the chip by hand on the FPL site")
    return {"element_in": best[2], "element_out": best[1], "purchase_price": int(els[best[2]]["now_cost"]),
            "selling_price": int(sell[best[1]])}


def build_picks(w: GWPlan, els: dict) -> list[dict]:
    order = sorted(w.lineup, key=lambda p: (els[p]["element_type"], p)) + list(w.bench)
    return [{"element": int(p), "position": i + 1, "is_captain": p == w.captain, "is_vice_captain": p == w.vice}
            for i, p in enumerate(order)]


def _lineup_state(my_team: dict) -> tuple:
    picks = sorted(my_team["picks"], key=lambda p: p["position"])
    return (frozenset(p["element"] for p in picks if p["position"] <= 11),
            tuple(p["element"] for p in picks if p["position"] > 11),
            next((p["element"] for p in picks if p.get("is_captain")), None),
            next((p["element"] for p in picks if p.get("is_vice_captain")), None))


def _wanted_lineup(w: GWPlan) -> tuple:
    return frozenset(w.lineup), tuple(w.bench), w.captain, w.vice


# ----------------------------------------------------------------------------- apply
def apply(rec: Recommendation, account, bs: dict, dry_run: bool = False, max_new_hits: int | None = None,
          now: pd.Timestamp | None = None) -> Outcome:
    """Make the account match gameweek one of `rec.plan`. Never raises for a refusal or a failed
    request: the Outcome says what was done before it stopped and why."""
    w = rec.plan.weeks[0]
    now = now or pd.Timestamp.now(tz="UTC")
    out = Outcome(gw=w.gw, dry_run=dry_run, when=str(now))
    try:
        _apply(w, account, bs, out, max_new_hits, now)
    except (Refused, Mismatch, acc.AuthError, acc.ApiError) as e:
        out.error = f"{type(e).__name__}: {e}"
    except Exception as e:  # an answer from FPL in a shape this code does not know: stop, do not guess
        out.error = f"unexpected {type(e).__name__}: {e}"
    return out


def _apply(w: GWPlan, account, bs: dict, out: Outcome, max_new_hits: int | None, now: pd.Timestamp) -> None:
    els = {e["id"]: e for e in bs["elements"]}
    name = lambda p: str(els[p]["web_name"])  # noqa: E731
    ev = next((e for e in bs["events"] if e["id"] == w.gw), None)
    if ev is None or not ev.get("is_next"):
        raise Refused(f"GW{w.gw} is not the upcoming gameweek any more")
    left = (pd.Timestamp(ev["deadline_time"]) - now).total_seconds() / 60.0
    if left < MIN_MINUTES:
        raise Refused(f"the GW{w.gw} deadline is {left:.0f} minutes away; too late to act safely")
    check_target(w, els)

    def send(kind: str, body: dict, call):
        out.requests.append({"kind": kind, **body})
        if out.dry_run:
            return None
        res = call()
        out.changed = True
        return res

    def read() -> dict:
        got = account.my_team()
        wrong = acc.shape_problem(got)
        if wrong:
            raise (Mismatch if out.changed else Refused)(
                f"FPL's answer about your team is not laid out as expected ({wrong}); not acting on it")
        return got

    mt = read()
    current = [p["element"] for p in mt["picks"]]
    sell = {p["element"]: p.get("selling_price", els[p["element"]]["now_cost"]) for p in mt["picks"]}
    active = acc.active_chip(mt)
    squad_chip = w.chip if w.chip in ("wildcard", "freehit") else None
    team_chip = w.chip if w.chip in ("bboost", "3xc") else None
    if active in ("wildcard", "freehit") and w.chip != active:
        raise Refused(f"the {CHIP_NAMES[active]} is already active for GW{w.gw} but the plan does not use it")

    # ---- transfers
    sold = [p for p in current if p not in w.squad]
    bought = [p for p in w.squad if p not in current]
    if sold:
        tr = mt.get("transfers") or {}
        unlimited = acc.unlimited_transfers(mt)
        gone = [name(p) for p in bought if els[p].get("status") == "u"]
        if gone:
            raise Refused(f"cannot buy {', '.join(gone)}: no longer available in the game")
        spare = int(tr.get("bank") or 0) + sum(sell[p] for p in sold) - sum(els[p]["now_cost"] for p in bought)
        if spare < 0:
            raise Refused(f"prices have moved: the transfers are £{-spare / 10:.1f}m over budget")
        cap = (bs.get("game_settings") or {}).get("transfers_cap")
        if cap and int(tr.get("made") or 0) + len(sold) > int(cap):
            raise Refused(f"more than the {cap} transfers FPL allows in one gameweek")
        free_left = acc.free_transfers_left(mt)
        spent_before = acc.points_spent(mt)
        new_hits = 0
        if not unlimited and not squad_chip:
            if free_left is None:
                raise Refused("FPL did not say how many free transfers are left")
            new_hits = max(0, len(sold) - free_left)
            if new_hits > w.hits:
                raise Refused(f"{len(sold)} transfers would cost {4 * new_hits} points but the plan expected "
                              f"{4 * w.hits}: the account has {free_left} free transfer(s)")
            if max_new_hits is not None and new_hits > max_new_hits:
                raise Refused(f"{len(sold)} transfers would cost {4 * new_hits} points; the limit for "
                              f"unattended moves is {4 * max_new_hits}")
        pairs = pair_transfers(sold, bought, els, sell)
        moves = "; ".join(f"{name(t['element_out'])} → {name(t['element_in'])}" for t in pairs)
        out.planned_hits = new_hits

        if squad_chip and not unlimited:
            if not acc.chip_available(mt, squad_chip):
                raise Refused(f"FPL does not show the {CHIP_NAMES[squad_chip]} as available")
            if not free_left:
                raise Refused(f"no free transfer left to play the {CHIP_NAMES[squad_chip]} safely; "
                              "play it by hand on the FPL site")
            one = first_swap(sold, bought, current, els, sell, int(tr.get("bank") or 0))
            first = [one]
            rest = pair_transfers([p for p in sold if p != one["element_out"]],
                                  [p for p in bought if p != one["element_in"]], els, sell)
            send("transfers", {"event": w.gw, "chip": squad_chip, "transfers": first},
                 lambda: account.post_transfers(w.gw, first, squad_chip))
            out.actions.append(f"Played the {CHIP_NAMES[squad_chip]} with a first transfer: "
                               f"{name(first[0]['element_out'])} → {name(first[0]['element_in'])}")
            if not out.dry_run:
                mt = read()
                if not acc.unlimited_transfers(mt):
                    raise Mismatch(f"FPL took the first transfer but does not show the {CHIP_NAMES[squad_chip]} "
                                   "as active. Stopped there: one free transfer used, no points spent")
                if acc.active_chip(mt) != squad_chip:
                    out.warnings.append(f"FPL shows unlimited transfers but does not label the {CHIP_NAMES[squad_chip]} "
                                        "as active; check the chip on the FPL site")
            if rest:
                # The chip must be named again: a transfers request with no chip cancels a chip that is
                # still pending for the deadline, and every transfer in it is then charged (GW11: -40).
                send("transfers", {"event": w.gw, "chip": squad_chip, "transfers": rest},
                     lambda: account.post_transfers(w.gw, rest, squad_chip))
                out.actions.append(f"Made the other {len(rest)} transfer(s) under the {CHIP_NAMES[squad_chip]}: "
                                   + "; ".join(f"{name(t['element_out'])} → {name(t['element_in'])}" for t in rest))
        else:
            keep = active if active in ("wildcard", "freehit") else None   # never send "no chip" over a pending one
            send("transfers", {"event": w.gw, "chip": keep, "transfers": pairs},
                 lambda: account.post_transfers(w.gw, pairs, keep))
            how = "free under the active chip" if unlimited else (f"-{4 * new_hits} points" if new_hits else "free")
            out.actions.append(f"Made {len(pairs)} transfer(s) ({how}): {moves}")

        if not out.dry_run:
            mt = read()
            have = {p["element"] for p in mt["picks"]}
            if have != set(w.squad):
                raise Mismatch("after the transfers the squad on FPL is not the planned one; "
                               f"unexpected: {sorted(name(p) for p in have - set(w.squad))}, "
                               f"missing: {sorted(name(p) for p in set(w.squad) - have)}")
            out.points_cost = acc.points_spent(mt)
            chip_lost = (squad_chip or active in ("wildcard", "freehit")) and not acc.unlimited_transfers(mt)
            if chip_lost or out.points_cost > spent_before + 4 * new_hits:
                lost = CHIP_NAMES[squad_chip or active]
                raise Mismatch(f"FPL shows {out.points_cost} points spent on transfers; "
                               f"{spent_before + 4 * new_hits} was expected"
                               + (f". The {lost} is no longer active: play it on the FPL site before the "
                                  "deadline and the points for this week's transfers are given back" if chip_lost else ""))
    elif bought:
        raise Refused("the squad on FPL has fewer players than the plan expects")

    # ---- line-up, captain, bench order, bench boost / triple captain
    active = acc.active_chip(mt)
    want = _wanted_lineup(w)
    same_team = not sold and _lineup_state(mt) == want
    chip_change = (team_chip and active != team_chip) or (active in ("bboost", "3xc") and active != team_chip)
    if team_chip and active != team_chip and not acc.chip_available(mt, team_chip):
        raise Refused(f"FPL does not show the {CHIP_NAMES[team_chip]} as available")
    if not same_team or chip_change:
        picks = build_picks(w, els)
        send("picks", {"chip": team_chip, "picks": picks}, lambda: account.post_picks(picks, team_chip))
        bench = ", ".join(name(p) for p in w.bench)
        out.actions.append(f"Set the line-up: captain {name(w.captain)}, vice-captain {name(w.vice)}; bench: {bench}")
        if team_chip and active != team_chip:
            out.actions.append(f"Played the {CHIP_NAMES[team_chip]}")
        if not out.dry_run:
            mt = read()
            if _lineup_state(mt) != want:
                raise Mismatch("after saving, the line-up, bench order or captain on FPL is not the planned one")
            now_active = acc.active_chip(mt)
            held = squad_chip or (active if active in ("wildcard", "freehit") else None)
            if held and not acc.unlimited_transfers(mt):
                raise Mismatch(f"after saving the line-up FPL no longer shows the {CHIP_NAMES[held]} as active; "
                               "play it again on the FPL site before the deadline")
            if team_chip and now_active != team_chip:
                raise Mismatch(f"the line-up was saved but FPL does not show the {CHIP_NAMES[team_chip]} as active")
            if not team_chip and now_active in ("bboost", "3xc"):
                out.warnings.append(f"The {CHIP_NAMES[now_active]} is still active although the plan no longer "
                                    "plays it this week; cancel it on the FPL site if you agree")
    if not out.actions:
        out.actions.append("Nothing to change: the account already matches the plan")
    out.verified = not out.dry_run
    if out.points_cost is None:
        out.points_cost = acc.points_spent(mt)


# ----------------------------------------------------------------------------- record
def save(outcome: Outcome) -> dict:
    """Keep a full record of every run that touched (or tried to touch) the account."""
    d = REPORTS / f"gw{outcome.gw:02d}"
    d.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S")
    f = d / f"{'dryrun' if outcome.dry_run else 'applied'}-{stamp}.json"
    f.write_text(json.dumps(asdict(outcome), indent=1))
    with open(REPORTS / "actions.log", "a") as log:
        tag = "DRY RUN " if outcome.dry_run else ""
        for a in outcome.actions:
            log.write(f"{stamp} GW{outcome.gw} {tag}{a}\n")
        for wn in outcome.warnings:
            log.write(f"{stamp} GW{outcome.gw} {tag}WARNING {wn}\n")
        if outcome.error:
            log.write(f"{stamp} GW{outcome.gw} {tag}STOPPED {outcome.error}\n")
    return {"json": f}
