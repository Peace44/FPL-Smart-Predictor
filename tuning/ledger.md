# Self-improvement ledger

One entry per run of `fpl improve`: the limiting factor the review found, what was tried, and what changed. `tuning/params.json` holds the values in effect; `fpl improve --revert ID` undoes a change.

## Five-step pass (2026-11-23): settings experiments no longer follow the limiting factor
- Every experiment (points_model, top_end, availability, planning) now runs each week, whatever the factor, until one keeps a change. Z_ADOPT 3.0 -> 3.2: 63 comparisons (39 candidate settings, 24 of them judged by top_end too) instead of at most 35; 3.2 keeps the chance of keeping one by luck near 4% a run.
- Revisit at GW16: the settled test (gwreview.certainty). If no factor has settled by then, propose deleting certainty, part_noise and deficit_cov.
