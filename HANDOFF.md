# HANDOFF (2026-11-27, 03:20)

## 1. Full closed-loop run: finished
- `goalflow_l3_circles`, seeds 0-499, L0-L4, 5 workers, front-axle LiDAR mount. Started 22:38 from commit **3fe727b**,
  ended 03:12.
- Output: `reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles/` with `summary.md`, `results.csv`, `plan_times.csv`
  (git-ignored) and `COMPARISON.md`; log `full_goalflow_l3_circles.log`, `full_goalflow_l3_circles.chain.log`.
- `COMPARISON.md` is built by `python scripts/build_comparison.py reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles`.
- Mid-run-edit check passed: one spawn pool created at start, workers never replaced (no maxtasksperchild), so no code
  was reloaded; 2,500 rows, all from the start code, none with the later `mount` field.

## 2. Repo
- Fork `ozaydincan/Goal_Flow_Prototype`, branch `feature/lidar-sim-demo`, **PR #4** open into `feature/phase0-audit`.
- Last pushed commit: **4b70bbf** (full-run results). After it: 8e101c4 (EMX baselines script) and this handoff, pushed
  with this file.
- **Running now: EMX baselines** (b) -> (a) -> (c), seeds 0-499, level EMX, front-axle mount, 7 workers, detached:
  - command: `nohup ./scripts/run_emx_baselines.sh > /dev/null 2>&1 &` (each config: `caffeinate -i .venv/bin/python
    scripts/eval_corridor.py --planners <observed_l3|observed_stubfix|observed_l3_noedge> --levels EMX --seeds 0-499
    --workers 7 --out reports/lidar_demo/emx_baselines/<b|a|c>`), started 03:13 from 8e101c4;
  - process group 95152 (chain script 95163; eval 95170 for config b); `RUNNING.lock` in the config's folder holds the
    eval PID; progress `reports/lidar_demo/emx_baselines/<b|a|c>.log`, `chain.log` ends with "done";
  - resumable (finished configs are skipped); to stop: `kill -- -95152`, then confirm with ps.
  - Nothing else may run until it has finished.
- Uncommitted: `docs/report/report.pdf` and `report.tex` (Can's edits, not touched by Claude); `reports/status/`
  handoff notes; `reports/lidar_demo/emx_baselines/` (run in progress).

- EMX baselines finished 03:36 (`reports/lidar_demo/emx_baselines/`): (b) 97.7/0.0/1.9, (a) 98.7/0.0/1.0, (c)
  97.9/0.0/1.9; EMX baseline = (a) by the rule. The model-or-system runs started 03:36 (item 1b).

## 3. Frozen baselines (per level, chosen from baseline runs alone)
- (b) stub fix + L3 at **L0**: `reports/lidar_demo/l0_rerun/b/`.
- (a) pre-L3 + stub fix at **L1-L4**: `reports/lidar_demo/l0_rerun_distdrift/a/` (distance-proportional drift).
- (c) (b) without edge memory, reported alongside: `l0_rerun/c/` (L0), `l0_rerun_distdrift/c/` (L1-L4).
- EMX level: chosen by the same rule from `reports/lidar_demo/emx_baselines/` once it exists.

## 4. COMPARISON.md: reviewed (Can, 2026-11-27)
Headline: GoalFlow is the safest planner at every level and matches (c) on success at L3-L4 with fewer collisions; the
success gap is stalls (L0-L1 mostly truly blocked, L2-L4 mostly filter false alarms). Door flips at L0-L2 trigger
lambda3 (pre-registered); track exits 11 / 10 at L0 / L1 vs 3 / 1.

## 5. Queue (2026-11-27 14:45; one run at a time)

**Running:** combined run `goalflow_l3_combined` = circles + (a2) memory fix + (c) edge critic (vectorised), seeds
0-499 x L0-L4, started 14:30 from dd68c5a, ~4.5 h; process group 2469 (eval PID 2483), lock
`reports/lidar_demo/l3_goalflow/full_goalflow_l3_combined/RUNNING.lock`, progress `full_goalflow_l3_combined.log`, chain
log ends with "end" after COMPARISON.md is built. Stop: `kill -- -2469`, then confirm with ps.

**Next, in order (each waits for the previous; GoalFlow runs at 5 workers):**
1. Combined-run COMPARISON.md: report vs the frozen baselines and vs `goalflow_l3_circles`.
2. COMPARISON.md additions (`scripts/build_comparison.py`): paired McNemar on success vs the baseline and vs (c) per
   level; stall overlap (GoalFlow stalls on seeds where the baseline also stalls), truly blocked vs false alarm.
3. The full run's two L4 collisions ("safety-filter miss, obstacle in hit memory"): render, find why the remembered
   obstacle did not block; fix if a bug. Recheck on the combined run.
4. Latency: single-worker per-component profile (the 241 ms is 5 workers sharing the GPU; single worker ~143 ms).
5. `juno_bangbang` speed mode: final GoalFlow config and the frozen baselines, seeds 0-99 at L0/L2/L4, vs 3 m/s.
6. EMX pitch check (EMX, roof mount, pitch 5-9 deg; hits per cone/bollard at 10-30 m; circle fitted?).
7. EMX shift check (`scripts/emx_shift_check.py`, seeds 250000+); then GoalFlow at EMX after review; roof-mount shift
   check before any retraining. EMX numbers stay unquoted until artefact rates are measured from Juno recordings.
8. Scene-overlap check (dev 0-499 vs training 100000+ by obstacle layout).
9. Smaller: brake-rate column in the selection table; tracker birth rule + duplicate merging; seed 190/399 render fix;
   Orin expansion budget (after Can's measurement); MPPI only as fallback.

**Combined run finished 19:14 (COMPARISON.md built): collisions ROSE vs goalflow_l3_circles: 2 -> 15 (L0-L4: 0.2 /
0.4 / 0.6 / 0.8 / 1.0% vs 0 / 0 / 0 / 0 / 0.4%), 13 of them "safety-filter miss (obstacle in hit memory)"; success +0 to
+2.1 points, stalls down at L2-L4, exits 11 / 10 -> 4 / 2 at L0 / L1. Flagged to Can; the per-change test (seeds 0-99)
did not show it. Needs Can's decision on the final config.**

**Done today** (details in DEVIATIONS 2026-11-27): EMX baselines ((a) = EMX baseline); model or system (the filter
provides the safety); lambda3 = 1.29 measured; beams-only (denser beams alone do not help); false-alarm diagnostic
(remembered circles dominate); per-change runs ((a), (c), (d), (a2) accepted; lambda3 not; (a2) rerun after a guard
bug); decisions (combined = (a2) + (c)); edge critic vectorised.

## 6. Can's TODO
- LiDAR optical centre x (from the rear axle), height, pitch.
- Windshield top edge and base (x, height); nose top (x, height); front and rear overhang.
- At first power-on: check the point cloud for a fixed cluster on the glass or hood.
- Body crop box in AD-SEM at bring-up.
- Speed control: from the SEM stop-sign recording (if it exists) brake decel and command-to-braking delay, otherwise a
  short logged brake test at 8 km/h; a coast-down from 8 km/h with the valve closed (coast decel); valve-open accel in
  the 6-8 km/h band and the valve delay.

## 7. Rules (verbatim)
- validation and dev seeds only;
- never tune on test seeds, and the test split never drives decisions;
- seeds: dev 0-499, train 100000+, val 200000+, EMX shift 250000+, spent test 300000+, fresh test 400000+;
- read logs before any delete or relaunch;
- kill the whole process group and confirm with ps before relaunching; one run per output folder (RUNNING.lock);
- keep >=5% CPU idle;
- public repo: no raw recording names, no personal names, no images of people; don't commit the Juno data sheet;
- never modify the AD-SEM repo, launch files or Nav2 configs; don't delete raw data;
- no force-push over Can's commits (rebase);
- commits end with Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com> plus the session line; messages lean.

## Chain after the combined run (approved by Can 2026-11-27; one at a time, no review stops)

Order: 2 McNemar + stall overlap (in COMPARISON.md) -> 3 the two L4 collisions (diagnose, report, no fix) -> (a)+filter
(seeds 0-499 x L0-L4) -> 4 latency profile (single worker) -> 6 EMX pitch check (roof) -> 5 juno_bangbang (combined
GoalFlow + (a), (b), (c); seeds 0-99 x L0/L2/L4) -> 8 scene overlap. Script `scripts/run_chain_after_combined.sh`, log
`reports/lidar_demo/chain_after_combined.log`. Not approved (wait for Can): item 7 (EMX shift check, GoalFlow at EMX,
roof-mount shift), item 9. Progress lines below are appended by the script as items finish.

- 11-27 19:14 item 2 McNemar + stall overlap: exit 0 (COMPARISON.md of combined and circles runs)
- 11-27 19:14 item 3 L4 collisions diagnosed: exit 0 (reports/lidar_demo/l3_goalflow/l4_collisions/)
