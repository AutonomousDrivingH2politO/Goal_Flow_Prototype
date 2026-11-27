# Model or system? Filter on / off with per-scan logging (2026-11-27)

`scripts/run_model_or_system.sh`: `goalflow_l3_circles_scanlog` (filter on; identical to the full run, seed 0 L0
reproduces it) and `goalflow_l3_circles_filteroff` (filter verdict computed and logged, not applied; never brakes;
memory on), seeds 0-99 at L0 / L2 / L4, front-axle mount, constant 3 m/s. `scans.csv` per run: top-ranked candidate
rejected by the filter (y/n), its true clearance, brake, the executed path's true clearance.

**Code versions:** the filter-on run started from 8a6db41. The filter-off run spawned its workers later, from newer
code; the only difference that touches it is the added `top3_goal_score_spread` column (logging only; planning
unchanged). `scanlog_spread/` is a deterministic rerun of the filter-on config with that column, used for lambda3.

## Results (seeds 0-99; % of feasible scenes)

| config | level | success % | collision % (n) | stalled % | exits |
|---|---|---|---|---|---|
| filter on | L0 | 94.8 | 0.0 (0) | 4.2 | 1 |
| filter on | L2 | 92.7 | 0.0 (0) | 7.3 | 0 |
| filter on | L4 | 80.2 | 1.0 (1) | 18.8 | 0 |
| filter off | L0 | 90.6 | 8.3 (8) | 0.0 | 1 |
| filter off | L2 | 89.6 | 9.4 (9) | 0.0 | 1 |
| filter off | L4 | 89.6 | 9.4 (9) | 0.0 | 1 |

(L2 and L4 filter-off totals coincide; the colliding seeds differ.)

Per scan (`scans.csv`): top-ranked candidate rejected by the filter, and its true clearance:

| run | level | scans | top rejected | truly colliding | truly 0-0.12 m | truly clear >= 0.12 m | executed path truly colliding |
|---|---|---|---|---|---|---|---|
| on | L0 | 12756 | 548 (4.3%) | 353 | 191 | 4 | 0 |
| on | L2 | 12980 | 757 (5.8%) | 423 | 180 | 154 | 0 |
| on | L4 | 12695 | 1299 (10.2%) | 708 | 156 | 435 | 10 |
| off | L0 | 12139 | 421 (3.5%) | 217 | 201 | 3 | 217 |
| off | L2 | 12149 | 465 (3.8%) | 249 | 159 | 57 | 249 |
| off | L4 | 12100 | 626 (5.2%) | 209 | 173 | 244 | 210 |

**Reading:** the low collision rate is the system, not the model alone. The model's top-ranked path truly collides in
~2% of scans (filter off: 217-249 of ~12,100), which is enough for 8-9 collisions per 100 episodes; the filter removes
them (0 / 0 / 1). With the filter on, the rejected top candidates are mostly true collisions (64 / 56 / 55% truly
colliding, nearly all the rest within the 0.12 m margin); false rejections (truly clear >= 0.12 m) are 1% at L0 but
20% at L2 and 33% at L4, which is where the stalls come from.
