# Observed corridor-lattice rerun, L0-L4, seeds 0-499 (2026-11-25)

`scripts/overnight_l0_rerun.sh` (`chain.log`), no time budget. These replace every pre-fix observed number
(`reports/lidar_demo/l0/`, 3513584). Per-config tables: `<config>/summary.md`; per-episode rows:
`<config>/results.csv`.

| Config | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| pre-fix (3513584) | 91.2 / 0.4 / 7.1 | 97.1 / 0.2 / 2.7 | 98.1 / 0.8 / 0.8 | 94.8 / 2.3 / 2.5 | 70.0 / 1.7 / 27.7 |
| (a) pre-L3 + goal-line stub fix | 94.8 / 0.4 / 4.0 | 97.1 / 0.8 / 1.9 | 97.1 / 1.3 / 1.7 | 96.6 / 0.8 / 2.5 | 93.7 / 2.3 / 3.6 |
| (b) stub fix + L3 | 97.5 / 0.0 / 1.9 | 97.3 / 0.0 / 2.7 | 95.8 / 0.6 / 3.6 | 92.5 / 1.0 / 6.5 | 74.8 / 1.9 / 22.4 |
| (c) (b) without edge memory | 95.4 / 0.0 / 4.0 | 96.9 / 0.0 / 3.1 | 94.3 / 1.0 / 4.6 | 89.5 / 0.6 / 9.2 | 77.1 / 1.5 / 20.8 |
| (d) (b) without tracking | 97.9 / 0.0 / 1.7 | 97.5 / 0.0 / 2.5 | 94.8 / 0.2 / 4.8 | 74.4 / 1.3 / 24.1 | 37.1 / 2.5 / 60.2 |

Cells: success / collision / stalled, % of feasible scenes. Exact track-exit counts (feasible scenes, of 477
per level; L0 / L1 / L2 / L3 / L4): pre-fix 6 / 0 / 1 / 2 / 3; (a) 4 / 1 / 0 / 0 / 2; (b) 3 / 0 / 0 / 0 / 4;
(c) 3 / 0 / 0 / 3 / 3; (d) 2 / 0 / 1 / 1 / 1. L3 = tracking, edge memory, kept-plan hysteresis +
fallback, hold, failed-search cache (`l3_config()`).

- **L3 helps at low noise, hurts at high noise.** L0-L1: (b)-(d) are best, with zero collisions. L3-L4: they
  stall far more than (a) (L4: 21-60% vs 3.6%); without tracking, (d) collapses at L4. Suspected cause (not yet
  tested): hold (two consecutive successful plans needed after a brake) and the failed-search cache under noisy
  scans. Next: ablate hold and cache in (b).
- **Goal-cut guard refusals** (episodes with >= 1 refusal, of 500, L0-L4): (a) 2/1/0/0/0, (b) 2/1/0/1/2,
  (c) 3/0/2/4/2, (d) 2/1/1/3/1. They fire at L3/L4, rarely; too few to explain the stalls.
- **L2, (a) vs pre-fix (98.1 -> 97.1%).** 10 seeds went success -> failure (82, 190, 274, 308 stopped;
  105, 107, 288, 299, 302, 385 collided), mid-course, zero guard refusals. The 50e9ad9 planner (edge-fill fix,
  no stub fix) fails all 10 the same way, so these come from the edge-fill fix. Under (b) 8 of them succeed
  (274, 308 stop); (c) 6 (274, 299, 308 stop, 302 collides); (d) 6 (107, 190, 299 stop).
- **L4, pre-fix -> (a) (70.0 -> 93.7%), measured:** the 50e9ad9 planner (edge-fill fix, no stub fix) reaches
  89.5% at L4 (collision 2.3, exit 2.3, stalled 5.9; `pre_L4/`). So the edge-fill fix gives +19.5 points and the
  stub fix +4.2 (it re-cuts edges near the goal line without refusing; 0 refusals at L4).
- Config (a) was run twice: the first run finished and was deleted by mistake at a relaunch; with no time
  budget the rerun is deterministic (see `chain.log`).

Renders: `renders/` (one still per episode, (a) | (b); rules in each folder's RULE.txt).
