# Model or system? Filter on / off with per-scan logging (2026-11-27)

`scripts/run_model_or_system.sh`: `goalflow_l3_circles_scanlog` (filter on; identical to the full run, seed 0 L0
reproduces it) and `goalflow_l3_circles_filteroff` (filter verdict computed and logged, not applied; never brakes;
memory on), seeds 0-99 at L0 / L2 / L4, front-axle mount, constant 3 m/s. `scans.csv` per run: top-ranked candidate
rejected by the filter (y/n), its true clearance, brake, the executed path's true clearance.

**Code versions:** the filter-on run started from 8a6db41. The filter-off run spawned its workers later, from newer
code; the only difference that touches it is the added `top3_goal_score_spread` column (logging only; planning
unchanged). `scanlog_spread/` is a deterministic rerun of the filter-on config with that column, used for lambda3.
