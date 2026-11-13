"""Compare optimiser settings by replaying past seasons (no chips).

    .venv/bin/python scripts/tune_optimiser.py [grid] [season ...]

Each row is one setting; columns are season totals. All settings start from the same initial
squad. Season totals are noisy (one haul by a captain is 20+ points), so the script also prints
each setting's average weekly margin over the reference with its standard error.
Weekly detail is saved to data/processed/tuning_<grid>.csv.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fpl_smart import backtest  # noqa: E402
from fpl_smart.config import PROCESSED  # noqa: E402
from fpl_smart.optimise import worker_pool  # noqa: E402
from fpl_smart.solver import SolverOptions  # noqa: E402

GRIDS = {
    "horizon": {
        "h1": (1, {}), "h2": (2, {}), "h3": (3, {}), "h5": (5, {}), "h8": (8, {}),
        "h3 decay .75": (3, {"decay": 0.75}), "h5 decay .75": (5, {"decay": 0.75}), "h8 decay .75": (8, {"decay": 0.75}),
        "h5 decay .65": (5, {"decay": 0.65}), "h5 no hits": (5, {"max_hits_per_gw": 0}),
        "h8 no hits": (8, {"max_hits_per_gw": 0}), "h1 no hits": (1, {"max_hits_per_gw": 0}),
        "h5 ft 2.0": (5, {"ft_value": 2.0}), "h5 ft 0.5": (5, {"ft_value": 0.5}),
    },
    "decay": {
        "h1": (1, {}), "h1 no hits": (1, {"max_hits_per_gw": 0}), "h1 ft .5": (1, {"ft_value": 0.5}),
        "h8 d.86": (8, {}), "h8 d.75": (8, {"decay": 0.75}), "h8 d.65": (8, {"decay": 0.65}),
        "h8 d.55": (8, {"decay": 0.55}), "h8 d.86 ft.5": (8, {"ft_value": 0.5}),
        "h8 d.75 ft.5": (8, {"decay": 0.75, "ft_value": 0.5}), "h8 d.65 ft.5": (8, {"decay": 0.65, "ft_value": 0.5}),
        "h5 d.65 ft.5": (5, {"decay": 0.65, "ft_value": 0.5}),
    },
}


def job(args):
    grid, season, label = args
    h, kw = GRIDS[grid][label]
    data = backtest.project_season(season, 8)
    r = backtest.simulate(data, season, "plan", h, SolverOptions(**kw))
    return r.weeks.reset_index().assign(season=season, setting=label)


if __name__ == "__main__":
    grid = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in GRIDS else "decay"
    seasons = [a for a in sys.argv[1:] if a not in GRIDS] or ["2024-25", "2025-26"]
    labels = list(GRIDS[grid])
    for s in seasons:                       # build the projection caches first (once per season)
        backtest.project_season(s, 8)
    with worker_pool(10) as ex:
        weeks = pd.concat(ex.map(job, [(grid, s, lab) for s in seasons for lab in labels]), ignore_index=True)
    weeks.to_csv(PROCESSED / f"tuning_{grid}.csv", index=False)
    pts = weeks.pivot_table(index="setting", columns="season", values="points", aggfunc="sum").loc[labels]
    pts["total"] = pts.sum(axis=1)
    pts["hits"] = weeks.groupby("setting")["hits"].sum()
    pts["transfers"] = weeks.groupby("setting")["transfers"].sum()
    wide = weeks.pivot_table(index=["season", "gw"], columns="setting", values="points")
    ref = labels[0]
    d = wide.sub(wide[ref], axis=0)
    pts[f"per GW vs {ref}"] = d.mean()
    pts["s.e."] = d.std() / np.sqrt(len(d))
    print(pts.round(2).to_string())
