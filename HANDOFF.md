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
- **Running now:** EMX baselines (b), (a), (c), seeds 0-499, `scripts/run_emx_baselines.sh`, detached, started 03:13,
  output `reports/lidar_demo/emx_baselines/` (RUNNING.lock while running, `chain.log`). Check its log before anything
  else; don't start another run until it has finished.
- Uncommitted: `docs/report/report.pdf` and `report.tex` (Can's edits, not touched by Claude); `reports/status/`
  handoff notes; `reports/lidar_demo/emx_baselines/` (run in progress).

## 3. Frozen baselines (per level, chosen from baseline runs alone)
- (b) stub fix + L3 at **L0**: `reports/lidar_demo/l0_rerun/b/`.
- (a) pre-L3 + stub fix at **L1-L4**: `reports/lidar_demo/l0_rerun_distdrift/a/` (distance-proportional drift).
- (c) (b) without edge memory, reported alongside: `l0_rerun/c/` (L0), `l0_rerun_distdrift/c/` (L1-L4).
- EMX level: chosen by the same rule from `reports/lidar_demo/emx_baselines/` once it exists.

## 4. COMPARISON.md review needs
Per level vs those baselines: success / collision / stalled / exits; brake rate; door flips (if GoalFlow flips more than
the baseline, lambda3 is turned on and the run repeated: pre-registered, triggered at L0-L2); stall diagnostics (best
true clearance at the last brake scan). Also flagged: GoalFlow track exits 11 / 10 at L0 / L1 vs 3 / 1 (edge critic off).

## 5. Queue, in order
1. COMPARISON.md review.
2. EMX pitch check (EMX preset, roof mount, pitch 5-9 deg; hits per cone and bollard at 10-30 m; circle fitted?).
3. EMX shift check (`scripts/emx_shift_check.py`, seeds 250000+), then baselines at EMX (running now), then GoalFlow
   at EMX after review.
4. Roof-mount shift check (current model, `roof` preset, seeds 250000+, same protocol) before any retraining.
5. Margin fixes (DEVIATIONS Part C item 18): refinement against the fitted circles, then a pre-registered clearance
   critic (soft, toward 0.25 m, weight fixed in advance), MPPI (item 16) only as fallback.
6. Latency profile (step 5); brake-rate column in the selection table (step 6); tracker birth rule + duplicate merging
   (step 7).
7. Seed 190/399 render fix; Orin expansion budget (after Can's expansions/s measurement).

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
