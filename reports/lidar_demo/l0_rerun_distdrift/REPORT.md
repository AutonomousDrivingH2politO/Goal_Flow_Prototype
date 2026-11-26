# Observed corridor-lattice baselines with distance-proportional drift, L1-L4, seeds 0-499 (2026-11-26)

`scripts/rerun_distdrift_parallel.sh` (`chain.log`); odometry drift now scales with distance driven (DEVIATIONS);
L0 has no drift and is unchanged (`../l0_rerun/`). Cells: success / collision / stalled % of feasible scenes /
track-exit count (477 feasible scenes per level). "time" = the earlier time-based drift (`../l0_rerun/`).

| Config | Drift | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| (a) pre-L3 + stub fix | time | 97.1 / 0.8 / 1.9 / 1 | 97.1 / 1.3 / 1.7 / 0 | 96.6 / 0.8 / 2.5 / 0 | 93.7 / 2.3 / 3.6 / 2 |
| (a) | distance | **97.9** / 0.6 / 1.3 / 1 | **98.7** / 0.0 / 1.3 / 0 | **97.3** / 0.8 / 1.9 / 0 | **93.1** / 2.3 / 4.4 / 1 |
| (b) stub fix + L3 | time | 97.3 / 0.0 / 2.7 / 0 | 95.8 / 0.6 / 3.6 / 0 | 92.5 / 1.0 / 6.5 / 0 | 74.8 / 1.9 / 22.4 / 4 |
| (b) | distance | 97.1 / 0.0 / 2.7 / 1 | 95.8 / 0.4 / 3.6 / 1 | 91.0 / 0.4 / 8.6 / 0 | 78.0 / 3.1 / 18.0 / 4 |
| (c) (b) without edge memory | time | 96.9 / 0.0 / 3.1 / 0 | 94.3 / 1.0 / 4.6 / 0 | 89.5 / 0.6 / 9.2 / 3 | 77.1 / 1.5 / 20.8 / 3 |
| (c) | distance | 95.6 / 0.2 / 3.8 / 2 | 95.0 / 1.0 / 4.0 / 0 | 87.8 / 1.0 / 11.1 / 0 | 78.0 / 2.9 / 18.7 / 2 |
| (d) (b) without tracking | time | 97.5 / 0.0 / 2.5 / 0 | 94.8 / 0.2 / 4.8 / 1 | 74.4 / 1.3 / 24.1 / 1 | 37.1 / 2.5 / 60.2 / 1 |
| (d) | distance | 96.6 / 0.4 / 2.7 / 1 | 95.2 / 0.0 / 4.6 / 1 | 73.4 / 0.8 / 25.6 / 1 | 39.6 / 0.8 / 59.3 / 1 |

- The drift model changes the baselines by at most ~1.5 points per cell, except (b) at L4 (+3.2 success) and (d) at L4.
- **Best classical config per level (success): (b) at L0 (unchanged, no drift), (a) at L1-L4.** At L1 this changes
  from (b) (97.3, time-based) to (a) (97.9 vs 97.1; a 4-scene margin), re-picked by the pre-registered rule from the
  baseline results alone.
