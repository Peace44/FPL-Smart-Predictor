"""Turn a recommendation into a readable Markdown report."""
from __future__ import annotations

import pandas as pd

from .config import POS
from .planner import Recommendation
from .solver import GWPlan, Plan

CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
STATUS = {"d": "doubtful", "i": "injured", "s": "suspended", "u": "unavailable", "n": "unavailable"}


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


class _View:
    """Lookups shared by every section."""

    def __init__(self, rec: Recommendation):
        self.rec, self.P = rec, rec.projection
        self.pl, self.xp = self.P.players, self.P.xp
        self.gws = [g for g in self.xp.columns if g >= rec.state.next_gw][: rec.horizon]
        fx = self.P.fixtures
        self.fix = {}
        for r in fx.sort_values("kickoff").itertuples():
            self.fix.setdefault((r.team, int(r.gw)), []).append(f"{r.opp} ({'H' if r.home else 'A'})")

    def name(self, p: int) -> str:
        return str(self.pl.at[p, "name"])

    def tag(self, p: int) -> str:
        r = self.pl.loc[p]
        return f"{r['name']} ({r['team']}, {POS[int(r['pos'])]}, £{r['price']:.1f}m)"

    def names(self, ids) -> str:
        return ", ".join(self.name(p) for p in ids) or "none"

    def fixture(self, p: int, gw: int) -> str:
        return " + ".join(self.fix.get((self.pl.at[p, "team"], gw), [])) or "no match"

    def x(self, p: int, gw: int) -> float:
        return float(self.xp.at[p, gw]) if gw in self.xp.columns else 0.0

    def flag(self, p: int) -> str:
        r = self.pl.loc[p]
        if r["status"] == "a" and (pd.isna(r["chance"]) or r["chance"] >= 100):
            return ""
        chance = "" if pd.isna(r["chance"]) else f" {int(r['chance'])}%"
        news = f": {r['news']}" if isinstance(r["news"], str) and r["news"] else ""
        return f"{STATUS.get(r['status'], 'flagged')}{chance}{news}"


def _chip_line(plan: Plan) -> str:
    c = plan.chips
    return ", ".join(f"{CHIP_NAMES[k]} GW{c[k]}" for k in CHIP_NAMES if k in c) or "none"


def _lineup_table(v: _View, w: GWPlan) -> str:
    rows = []
    for p in w.lineup:
        role = " (C)" if p == w.captain else (" (V)" if p == w.vice else "")
        rows.append([POS[int(v.pl.at[p, "pos"])], v.name(p) + role, v.pl.at[p, "team"], v.fixture(p, w.gw),
                     f"{v.x(p, w.gw):.1f}", v.flag(p)])
    for i, p in enumerate(w.bench):
        slot = "GK" if int(v.pl.at[p, "pos"]) == 1 else str(sum(1 for q in w.bench[:i] if int(v.pl.at[q, 'pos']) != 1) + 1)
        rows.append([f"bench {slot}", v.name(p), v.pl.at[p, "team"], v.fixture(p, w.gw), f"{v.x(p, w.gw):.1f}", v.flag(p)])
    return _table(["", "Player", "Team", "Fixture", "xP", "Flag"], rows)


def _xp_rows(v: _View, ids, gws, label=None) -> list[list]:
    rows = []
    for p in ids:
        r = [label(p) if label else v.name(p), v.pl.at[p, "team"], POS[int(v.pl.at[p, "pos"])], f"£{v.pl.at[p, 'price']:.1f}m"]
        r += [f"{v.x(p, g):.1f}" for g in gws] + [f"{sum(v.x(p, g) for g in gws):.1f}"]
        rows.append(r)
    return rows


def render(rec: Recommendation) -> str:
    v = _View(rec)
    st, plan, hold = rec.state, rec.plan, rec.hold
    w, h0 = plan.weeks[0], hold.weeks[0]
    gw, last = st.next_gw, v.gws[-1]
    L: list[str] = []
    add = L.append

    title = f"# Gameweek {gw} plan" + (f" for {rec.meta['team_name']}" if rec.meta.get("team_name") else "")
    add(title)
    add("")
    bits = []
    if rec.deadline is not None:
        bits.append(f"**Deadline:** {rec.deadline.strftime('%a %d %b %Y, %H:%M')} UTC")
    if st.overall_rank:
        total = rec.meta.get("total_players")
        bits.append(f"**Rank:** {st.overall_rank:,}" + (f" of {total:,}" if total else "") + f" ({st.total_points} pts)")
    bits.append(f"**Bank:** £{st.bank / 10:.1f}m")
    bits.append(f"**Free transfers:** {st.free_transfers}")
    add(" · ".join(bits))
    add("")

    # ---------------------------------------------------------------- the decision
    add("## What to do")
    add("")
    fh_in = [p for p in w.squad if p not in st.squad]
    fh_out = [p for p in st.squad if p not in w.squad]
    if w.chip == "freehit":
        add(f"- **Chip:** play the **Free Hit**: a squad for this gameweek only; your own squad returns for GW{gw + 1}.")
        add(f"- **In for this week ({len(fh_in)}):** {v.names(fh_in)}")
        add(f"- **Left out ({len(fh_out)}):** {v.names(fh_out)}")
    elif w.chip == "wildcard":
        add("- **Chip:** play the **Wildcard**: unlimited free transfers this gameweek.")
        add(f"- **Sell ({len(w.transfers_out)}):** {v.names(w.transfers_out)}")
        add(f"- **Buy ({len(w.transfers_in)}):** {v.names(w.transfers_in)}")
    else:
        add(f"- **Chip:** play the **{CHIP_NAMES[w.chip]}**." if w.chip else "- **Chip:** none this week.")
        if w.transfers_in:
            add(f"- **Transfers out:** {v.names(w.transfers_out)}")
            add(f"- **Transfers in:** {v.names(w.transfers_in)}")
            if w.hits:
                add(f"- **Points hit:** -{4 * w.hits} ({w.hits} extra transfer{'s' if w.hits > 1 else ''}); "
                    "the plan projects it pays back.")
        else:
            add("- **Transfers:** none; roll the free transfer.")
    add(f"- **Captain:** {v.name(w.captain)}; vice-captain {v.name(w.vice)}.")
    if w.chip != "freehit":
        add(f"- **Money left:** £{w.bank / 10:.1f}m.")
    nxt = plan.weeks[1] if len(plan.weeks) > 1 else None
    if nxt is not None and nxt.chip in ("wildcard", "freehit"):
        add(f"- **Next week:** the plan plays the {CHIP_NAMES[nxt.chip]} in GW{nxt.gw}; this week's advice assumes that.")
    add("")
    gain_now, gain_all = w.xp - h0.xp, plan.total_xp - hold.total_xp
    add(f"Projected: **{w.xp:.1f} points this gameweek** ({gain_now:+.1f} against your current team unchanged) "
        f"and **{plan.total_xp:.0f} over GW{gw}-{last}** ({gain_all:+.0f} against never making a transfer or "
        f"playing a chip).")
    add("")
    add(_lineup_table(v, w))
    add("")

    # ---------------------------------------------------------------- why these moves
    if w.transfers_in and w.chip != "freehit":
        near = v.gws[:6]
        add("## Why these moves")
        add("")
        add(f"Expected points per gameweek (GW{near[0]}-{near[-1]}):")
        add("")
        rows = _xp_rows(v, w.transfers_out, near, lambda p: "OUT " + v.name(p)) + \
            _xp_rows(v, w.transfers_in, near, lambda p: "IN " + v.name(p))
        add(_table(["", "Team", "Pos", "Price"] + [f"GW{g}" for g in near] + ["Total"], rows))
        add("")
        flagged = [p for p in w.transfers_out if v.flag(p)]
        if flagged:
            add("Flags on players being sold: " + "; ".join(f"{v.name(p)} ({v.flag(p)})" for p in flagged) + ".")
            add("")

    # ---------------------------------------------------------------- confidence
    rb = rec.robust
    if rb is not None:
        add("## How sure is this")
        add("")
        add(f"The plan was re-solved {rb.n} times with every projection randomly shaken (players, single weeks "
            "and whole fixtures). Share of those runs that made the same choice for this gameweek:")
        add("")
        if w.chip in ("wildcard", "freehit"):
            picks = sorted(w.squad, key=lambda p: -rb.squad.get(p, 0.0))
            add("- **Squad picks** (how often each made the squad): "
                + ", ".join(f"{v.name(p)} {_pct(rb.squad.get(p, 0.0))}" for p in picks))
            alt = [(p, sh) for p, sh in rb.squad.items() if p not in w.squad and sh >= 0.25][:8]
            if alt:
                add("- **Others that often made it:** " + ", ".join(f"{v.name(p)} {_pct(sh)}" for p, sh in alt))
            add("- A low share means other players are about as good for that slot, not that the pick is poor; "
                "the listed alternatives are fair swaps if you prefer them.")
        elif w.transfers_in:
            add(f"- **Exactly these transfers:** {_pct(rb.base_kept)}")
            rows = [["sell " + v.name(p), _pct(rb.sold.get(p, 0.0))] for p in w.transfers_out]
            rows += [["buy " + v.name(p), _pct(rb.bought.get(p, 0.0))] for p in w.transfers_in]
            others = [(p, sh) for p, sh in rb.bought.items() if p not in w.transfers_in and sh >= 0.15][:5]
            rows += [["alternative buy: " + v.name(p), _pct(sh)] for p, sh in others]
            others = [(p, sh) for p, sh in rb.sold.items() if p not in w.transfers_out and sh >= 0.15][:5]
            rows += [["alternative sale: " + v.name(p), _pct(sh)] for p, sh in others]
            add("")
            add(_table(["Move", "Share of runs"], rows))
            add("")
        else:
            add(f"- **No transfer:** {_pct(rb.n_transfers.get(0, 0.0))}")
        others = ", ".join(f"{v.name(p)} {_pct(sh)}" for p, sh in rb.captain.items() if p != w.captain and sh >= 0.1)
        add(f"- **Captain:** {v.name(w.captain)} {_pct(rb.captain.get(w.captain, 0.0))}"
            + (f"; alternatives: {others}" if others else ""))
        add("")

    # ---------------------------------------------------------------- the weeks ahead
    add("## The weeks ahead")
    add("")
    add("Later weeks are re-planned every gameweek as prices, injuries and odds change; treat them as direction.")
    add("")
    rows = []
    for k in plan.weeks:
        if k.chip in ("wildcard", "freehit"):
            moves = f"{len(k.transfers_in)} changes" if k.chip == "wildcard" else "one-week squad"
        else:
            moves = "; ".join(f"{v.name(o)} → {v.name(i)}" for o, i in _pair(v, k)) or "-"
        rows.append([f"GW{k.gw}", CHIP_NAMES.get(k.chip, "-"), moves + (f" (-{4 * k.hits})" if k.hits else ""),
                     v.name(k.captain), f"{k.xp:.1f}", f"£{k.bank / 10:.1f}m"])
    add(_table(["GW", "Chip", "Transfers", "Captain", "xP", "Bank"], rows))
    add("")

    # ---------------------------------------------------------------- chips
    if rec.search is not None and len(rec.search.scenarios) > 1:
        add("## Chip timing")
        add("")
        exp = [CHIP_NAMES[c] for c, i in rec.search.windows.items() if i["gws"] and i["expires"]]
        if exp:
            ends = max(g for i in rec.search.windows.values() if i["expires"] for g in i["gws"])
            add(f"These chips expire after GW{ends} and are lost if unused: {', '.join(exp)}.")
            add("")
        best = rec.search.best
        shown = rec.search.scenarios[:6]
        for extra in (plan, rec.search.without_squad_chips()):
            if extra is not None and not any(extra is q for q in shown):
                shown = shown + [extra]
        rows = []
        for p in shown:
            c = p.chips
            rows.append([f"GW{c['wildcard']}" if "wildcard" in c else "-", f"GW{c['freehit']}" if "freehit" in c else "-",
                         f"GW{c['bboost']}" if "bboost" in c else "-", f"GW{c['3xc']}" if "3xc" in c else "-",
                         f"{p.total_xp:.0f}", f"{p.objective - best.objective:+.1f}", "recommended" if p is plan else ""])
        add(_table(["Wildcard", "Free Hit", "Bench Boost", "Triple Captain", f"Points GW{gw}-{last}", "Score vs best", ""], rows))
        add("")
        if plan is not best:
            add("The top-scoring option plays a Wildcard or Free Hit this week, but the recommended one is within "
                "the margin of error of it and keeps both in hand, which cannot be undone once played.")
            add("")
        add("\"Score\" weights near gameweeks more than far ones, because far projections are less reliable. "
            "Differences under about 2 are too close to call.")
        add("")

    # ---------------------------------------------------------------- squad and market
    near = v.gws[:6]
    add("## Your squad")
    add("")
    mine = sorted(st.squad, key=lambda p: (int(v.pl.at[p, "pos"]), -sum(v.x(p, g) for g in near)))
    rows = _xp_rows(v, mine, near, lambda p: v.name(p) + (" ⚑" if v.flag(p) else ""))
    add(_table(["Player", "Team", "Pos", "Price"] + [f"GW{g}" for g in near] + ["Total"], rows))
    flags = [f"{v.name(p)}: {v.flag(p)}" for p in mine if v.flag(p)]
    if flags:
        add("")
        add("Flagged: " + "; ".join(flags) + ".")
    add("")
    add("## Best projected players you do not own")
    add("")
    tot = v.xp[near].sum(axis=1)
    rows = []
    for pos in (1, 2, 3, 4):
        ids = [p for p in tot[(v.pl["pos"] == pos) & ~v.pl.index.isin(st.squad)].nlargest(5).index]
        rows += _xp_rows(v, ids, near)
    add(_table(["Player", "Team", "Pos", "Price"] + [f"GW{g}" for g in near] + ["Total"], rows))
    add("")

    # ---------------------------------------------------------------- notes
    add("## Notes")
    add("")
    fx = v.P.fixtures
    odds_gws = sorted(int(g) for g in fx.loc[fx["has_odds"], "gw"].unique())
    if odds_gws:
        add(f"- Bookmaker odds are available for GW{odds_gws[0]}" + (f"-{odds_gws[-1]}" if len(odds_gws) > 1 else "")
            + "; later gameweeks use the team ratings alone.")
    else:
        add("- No bookmaker odds were available; all fixtures use the team ratings alone.")
    counts = fx.groupby(["team", "gw"]).size()
    doubles = sorted({int(g) for (_, g), n in counts.items() if n > 1 and g in v.gws})
    teams = set(fx["team"])
    blanks = sorted({g for g in v.gws for t in teams if (t, g) not in counts.index})
    if doubles:
        add(f"- Double gameweeks in the plan: {', '.join('GW' + str(g) for g in doubles)}.")
    if blanks:
        add(f"- Gameweeks where some teams have no match: {', '.join('GW' + str(g) for g in blanks)}.")
    for n in st.notes:
        add(f"- {n}")
    if plan.gap is not None:
        add(f"- The optimiser proved this plan is within {max(100 * plan.gap, 0.1):.1f}% of the best possible score. "
            "Several squads usually sit that close together, so two runs can differ in a few picks.")
    if rec.search is not None:
        for n in rec.search.notes:
            add(f"- {n}")
    add(f"- Generated {pd.Timestamp(rec.meta.get('generated')).strftime('%d %b %Y %H:%M')} UTC. "
        "Projections are averages: any single gameweek can land far from them.")
    return "\n".join(L) + "\n"


def _pair(v: _View, k: GWPlan) -> list[tuple[int, int]]:
    """Match each sale with a purchase in the same position where possible (display only)."""
    ins, pairs = list(k.transfers_in), []
    for o in k.transfers_out:
        same = [i for i in ins if v.pl.at[i, "pos"] == v.pl.at[o, "pos"]]
        i = (same or ins)[0]
        ins.remove(i)
        pairs.append((o, i))
    return pairs
