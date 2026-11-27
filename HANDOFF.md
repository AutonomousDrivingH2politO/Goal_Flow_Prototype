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

## 3. Frozen baselines (per level, chosen from baseline runs alone)
- (b) stub fix + L3 at **L0**: `reports/lidar_demo/l0_rerun/b/`.
- (a) pre-L3 + stub fix at **L1-L4**: `reports/lidar_demo/l0_rerun_distdrift/a/` (distance-proportional drift).
- (c) (b) without edge memory, reported alongside: `l0_rerun/c/` (L0), `l0_rerun_distdrift/c/` (L1-L4).
- EMX level: chosen by the same rule from `reports/lidar_demo/emx_baselines/` once it exists.

## 4. COMPARISON.md: reviewed (Can, 2026-11-27)
Headline: GoalFlow is the safest planner at every level and matches (c) on success at L3-L4 with fewer collisions; the
success gap is stalls (L0-L1 mostly truly blocked, L2-L4 mostly filter false alarms). Door flips at L0-L2 trigger
lambda3 (pre-registered); track exits 11 / 10 at L0 / L1 vs 3 / 1.

## 5. Queue, in order (behind the EMX baselines, not alongside)
1. EMX baselines (running) -> per-level EMX baseline by the rule.
2. COMPARISON.md additions (in `scripts/build_comparison.py`): paired McNemar on success, GoalFlow vs baseline and vs
   (c), per level; stall overlap (GoalFlow stalls on seeds where the baseline also stalls), split truly blocked vs filter
   false alarm.
3. The two L4 collisions ("safety-filter miss, obstacle in hit memory"): render both, find why the remembered obstacle
   did not block (memory position drift, circle radius, or filter lookup); fix if it is a bug.
4. Pre-registered changes (DEVIATIONS, 2026-11-27), in this order; lambda3 and the edge critic go with the margin
   fixes, not behind them: (a) filter false alarms (69 of 99 L4 stalls): circle radius capped at class radius + a small
   allowance, clusters enter the filter after 2 scans except the near-field memory, 0.12 m clearance kept; (b) lambda3
   on; (c) track-edge critic on; (d) refinement against circles for the L0-L1 precision stalls. Each alone on seeds 0-99
   at L0, L2, L4, accepted only if collisions do not rise AND stalls drop (or success rises); per-change numbers in one
   small table; then the accepted set combined on 0-499 x L0-L4.
5. Latency: single-worker per-component profile (step 5); 241 ms with 5 workers sharing the GPU is not the number.
6. EMX pitch check (roof mount, 5-9 deg); EMX shift check (`scripts/emx_shift_check.py`, seeds 250000+); GoalFlow at
   EMX after review; roof-mount shift check before any retraining.
7. Brake-rate column in the selection table; tracker birth rule + duplicate merging; seed 190/399 render fix; Orin
   expansion budget; MPPI (item 16) only as fallback.

## 6. Can's TODO
- LiDAR optical centre x (from the rear axle), height, pitch.
- Windshield top edge and base (x, height); nose top (x, height); front and rear overhang.
- At first power-on: check the point cloud for a fixed cluster on the glass or hood.
- Body crop box in AD-SEM at bring-up.

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
