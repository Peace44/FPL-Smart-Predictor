"""Score a picked team against what actually happened, following FPL's rules:
armband doubling (vice steps in if the captain did not play) and automatic substitutions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_FORMATION = {1: 1, 2: 3, 3: 2, 4: 1}


@dataclass
class GWScore:
    points: float
    lineup: list[int]                 # the eleven that counted, after substitutions
    subs: list[tuple[int, int]] = field(default_factory=list)   # (off, on)
    captain: int | None = None        # who actually got the armband bonus


def score_gw(lineup: list[int], bench: list[int], captain: int, vice: int,
             points: dict[int, float], minutes: dict[int, float], pos: dict[int, int],
             chip: str | None = None) -> GWScore:
    """`bench` is in substitution order (the keeper may sit anywhere in it). `points` and
    `minutes` are gameweek totals; a player missing from them did not play."""
    played = lambda p: minutes.get(p, 0) > 0  # noqa: E731
    pts = lambda p: float(points.get(p, 0.0))  # noqa: E731

    if chip == "bboost":
        final, subs = list(lineup) + list(bench), []
    else:
        final, subs = list(lineup), []
        available = [b for b in bench if played(b)]
        for out in [p for p in lineup if not played(p)]:
            for b in available:
                if pos[out] == 1 or pos[b] == 1:
                    ok = pos[out] == pos[b]           # keepers only swap with keepers
                else:
                    trial = [q for q in final if q != out] + [b]
                    count = {k: sum(1 for q in trial if pos[q] == k) for k in MIN_FORMATION}
                    ok = all(count[k] >= n for k, n in MIN_FORMATION.items())
                if ok:
                    final[final.index(out)] = b
                    available.remove(b)
                    subs.append((out, b))
                    break

    total = sum(pts(p) for p in final)
    armband = captain if played(captain) else (vice if played(vice) else None)
    if armband is not None and armband in final:
        total += pts(armband) * (2 if chip == "3xc" else 1)
    return GWScore(total, final, subs, armband)
