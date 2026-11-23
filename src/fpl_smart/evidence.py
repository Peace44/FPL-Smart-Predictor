"""The model's errors pooled over every review of the season, with standard errors: the evidence that
aims code changes.

One gameweek's team result is mostly luck: one manager's parts (squad, selection, captaincy) swing by
5 to 10 points from week to week, so a few weeks of them cannot say which part of the model to fix.
Each review also checks the projection against every likely starter, about 280 a week. Pooled over
the reviews, those checks are the strongest evidence the loop has of where the model is wrong.

Each review stores, per check, the number of cases and the sum of their errors (actual minus
projected) for each club. Pooling gives a mean error with a standard error clustered by club and
gameweek: players of one club share its clean sheets and goals, so they are not independent. From
three gameweeks on, the spread between gameweeks is used instead when it is larger (a whole weekend can
run high). A check is worth acting on when it is at least Z_ACT standard errors from zero, at least the
smallest size worth a change, and spans at least MIN_GW_SPREAD gameweeks. About 25 checks are made, so
Z_ACT = 3 keeps a false lead to about one run in twenty (a Bonferroni bound); 2 to 3 standard errors is
worth watching, not acting on.

Fewer gameweeks are at most worth watching, however large the z: within one week the clustering cannot
see a shock shared by every club. GW11 showed it: the ten highest projections scored 2.85 points a
player below projection, z -6.5 from that week alone, while the walk-forward replays find the top
projections run slightly low.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_ACT = 3.0
Z_WATCH = 2.0
MIN_GW_SPREAD = 3          # gameweeks needed before the spread between them is measured


@dataclass(frozen=True)
class Check:
    label: str
    unit: str              # "pts" | "goals" | "share"
    min_size: float        # the smallest mean error worth a code change
    where: str             # the code that moves it
    factor: str            # the review's part it feeds


_COMP = {   # component: (label, where, factor)
    "app": ("Appearance points", "minutes model and flags (minutes.py, xp.availability)", "selection"),
    "goals": ("Goal points", "goal shares and team attack (player_model.rates, team_model)", "squad"),
    "assists": ("Assist points", "assist shares and team attack (player_model.rates, team_model)", "squad"),
    "cs": ("Clean-sheet points", "clean-sheet chance from team defence (team_model, player_model)", "squad"),
    "gc": ("Goals-conceded points", "goals conceded from team defence (team_model, player_model)", "squad"),
    "saves": ("Save points", "saves (player_model, player.k_sv)", "squad"),
    "defcon": ("Defensive-contribution points", "defensive contributions (player_model, player.k_dc)", "squad"),
    "bonus": ("Bonus points", "bonus (player_model, player.k_bps / k_bres)", "captaincy"),
    "cards": ("Cards, own goals and missed pens", "cards (player_model, player.k_yc)", "squad"),
}
POSITIONS = ("GKP", "DEF", "MID", "FWD")


def check(key: str) -> Check | None:
    kind, _, arg = key.partition(":")
    if kind == "comp" and arg in _COMP:
        label, where, factor = _COMP[arg]
        return Check(f"{label} per likely starter", "pts", 0.05, where, factor)
    if kind == "pos" and arg in POSITIONS:
        return Check(f"{arg} points per likely starter", "pts", 0.1,
                     "calibration by position (player_model.fit_calibration)", "squad")
    if kind == "top10":
        return Check("Points of the ten highest projections, per player", "pts", 0.3,
                     "top-end calibration (player_model.fit_calibration, by price band)", "captaincy")
    if kind == "team_xg":
        return Check("Team expected goals against the chances created (xG), per club and match", "goals", 0.1,
                     "team ratings and the odds blend (team_model, odds.py)", "squad")
    if kind == "play":
        return Check(f"Share who played, P(plays) in {arg}", "share", 0.05,
                     "P(plays) (minutes.py, xp.availability)", "selection")
    return None


# ----------------------------------------------------------------------------- what a review stores
def _by_team(team: pd.Series, err: pd.Series) -> dict[str, list]:
    agg = pd.DataFrame({"team": team.astype(str), "e": err.astype(float)}).groupby("team")["e"].agg(["size", "sum"])
    return {str(t): [int(r["size"]), float(r["sum"])] for t, r in agg.iterrows()}


def collect(likely: pd.DataFrame, allp: pd.DataFrame, top: pd.DataFrame, teams: pd.DataFrame | None,
            components: list[str], bins: list[float], pos_name: dict) -> dict[str, dict[str, list]]:
    """This gameweek's errors (actual minus projected), as {check: {club: [cases, sum of errors]}}.

    likely: likely starters with projected components (`c`), actual ones (`a_c`), `xp`, `pts`, `pos`, `team`.
    allp: every projected player with `p_play`, `played`, `team`. top: the ten highest projections.
    teams: per club `projected` expected goals and `xg` created (index = club), or None.
    """
    out: dict[str, dict[str, list]] = {}
    for c in components:
        if c in likely and f"a_{c}" in likely and check(f"comp:{c}"):
            out[f"comp:{c}"] = _by_team(likely["team"], likely[f"a_{c}"] - likely[c])
    for p, g in likely.groupby("pos"):
        name = pos_name.get(int(p))
        if name:
            out[f"pos:{name}"] = _by_team(g["team"], g["pts"] - g["xp"])
    if len(top):
        out["top10"] = _by_team(top["team"], top["pts"] - top["xp"])
    if teams is not None and len(teams):
        out["team_xg"] = _by_team(pd.Series(teams.index, index=teams.index), teams["xg"] - teams["projected"])
    cut = pd.cut(allp["p_play"], bins, right=False)
    for b, g in allp.groupby(cut, observed=True):
        out[f"play:{b.left:.0%}-{min(b.right, 1.0):.0%}"] = _by_team(g["team"], g["played"] - g["p_play"])
    return out


# ----------------------------------------------------------------------------- pooling
@dataclass
class Finding:
    key: str
    check: Check
    mean: float            # actual minus projected: positive = the projection was too low
    se: float
    n: int
    gws: list[int]

    @property
    def z(self) -> float:
        return self.mean / self.se if self.se > 0 else 0.0

    @property
    def verdict(self) -> str | None:
        if abs(self.mean) < self.check.min_size:
            return None
        if abs(self.z) >= Z_ACT and len(self.gws) >= MIN_GW_SPREAD:
            return "act"
        return "watch" if abs(self.z) >= Z_WATCH else None

    def size_text(self) -> str:
        way = "too low" if self.mean > 0 else "too high"
        if self.check.unit == "share":
            return f"projection {way} by {abs(self.mean):.0%} ± {self.se:.0%}"
        return f"projection {way} by {abs(self.mean):.2f} ± {self.se:.2f} {self.check.unit}"


def pool(reviews: list[dict]) -> list[Finding]:
    """Every check pooled over the reviews that stored errors, strongest evidence first."""
    rows: dict[str, list[tuple[int, int, float]]] = {}
    for r in reviews:
        for key, teams in ((r.get("model") or {}).get("errors") or {}).items():
            for n, s in teams.values():
                if n:
                    rows.setdefault(key, []).append((int(r["gw"]), int(n), float(s)))
    out = []
    for key, rs in rows.items():
        c = check(key)
        gw = np.array([x[0] for x in rs])
        n = np.array([x[1] for x in rs], float)
        s = np.array([x[2] for x in rs])
        N, G = n.sum(), len(rs)
        if c is None or G < 2 or N < 2:
            continue
        m = s.sum() / N
        se = math.sqrt(G / (G - 1) * float(((s - n * m) ** 2).sum())) / N       # clustered by club and gameweek
        weeks = sorted(set(gw.tolist()))
        if len(weeks) >= MIN_GW_SPREAD:
            wk = np.array([s[gw == g].sum() / n[gw == g].sum() for g in weeks])
            se = max(se, float(wk.std(ddof=1)) / math.sqrt(len(weeks)))
        out.append(Finding(key, c, float(m), se, int(N), weeks))
    return sorted(out, key=lambda f: (f.verdict != "act", f.verdict != "watch", -abs(f.z)))


def leads(findings: list[Finding]) -> list[Finding]:
    """The findings worth acting on or watching, strongest first."""
    return [f for f in findings if f.verdict]


def table(findings: list[Finding], limit: int | None = None) -> list[list[str]]:
    rows = []
    for f in leads(findings)[:limit]:
        g = f.gws
        span = f"GW{g[0]}" if len(g) == 1 else f"GW{g[0]}-{g[-1]} ({len(g)})"
        rows.append([f.check.label, f.size_text(), f"{f.z:+.1f}", span, f.verdict, f.check.where, f.check.factor])
    return rows


HEADERS = ["Check", "Error (± s.e.)", "z", "Gameweeks", "Verdict", "Where in the code", "Part it moves"]


def summary(findings: list[Finding]) -> str:
    """One sentence on how much evidence there is."""
    if not findings:
        return "No pooled model errors yet: each review stores them when a projection was saved before its deadline."
    gws = sorted({g for f in findings for g in f.gws})
    n_act = sum(f.verdict == "act" for f in findings)
    n_watch = sum(f.verdict == "watch" for f in findings)
    rest = len(findings) - n_act - n_watch
    span = f"GW{gws[0]}" if len(gws) == 1 else f"GW{gws[0]}-{gws[-1]}"
    return (f"{len(findings)} checks of the projection pooled over {len(gws)} gameweek{'s' if len(gws) != 1 else ''} "
            f"({span}): {n_act} to act on (at least {Z_ACT:g} standard errors over {MIN_GW_SPREAD} gameweeks or more, "
            f"and big enough to matter), {n_watch} to watch (at least {Z_WATCH:g}, or fewer gameweeks), "
            f"{rest} within noise or too small.")
