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

## 5. Queue (updated 2026-11-27 21:10; nothing running, nothing approved)

**Chain finished 20:52.** Replay of the 13 combined-run filter-miss collisions done (9 re-anchored to an offset
observation, 3 dropped by the free-ray rule, 1 drifted; `reports/lidar_demo/l3_goalflow/combined_collisions/`).

**Decided (Can):** `goalflow_l3_combined` is not the final config; final GoalFlow config decided after the replay.
Recommended planner for the car: corridor-lattice planner + the safety layer ((a)+filter). GoalFlow = research track.

**Waiting for Can (nothing starts without approval):**
- Final GoalFlow config after the replay (e.g. circles + edge critic, or the memory fix without re-anchoring).
- Freeze both final configs (GoalFlow and (a)+filter) and run each once on the fresh test seeds 400000+, all levels
  (planned, not approved).
- GoalFlow's juno_bangbang collapse (35 of 40 collisions "no safe candidate"): a separating test (zero brake delay or
  speed input fixed at 3 m/s) if wanted.
- Item 7 for (a)+filter: EMX shift check, roof-mount check ("GoalFlow at EMX" dropped).
- Item 9: brake-rate column, seed 190/399 render fix, Orin expansion budget (tracker birth rule dropped).

**Results today (details in DEVIATIONS and the report folders):**
- Combined run (`reports/lidar_demo/l3_goalflow/full_goalflow_l3_combined/COMPARISON.md`, McNemar + stall overlap
  included): success vs (a) significantly lower at every level (p <= 1e-3); vs (c) not significant at L0 / L3 / L4.
- L4 collisions of the circles run (seeds 93, 443): memory position drift, remembered circle 0.26-0.31 m off; the
  filter saw 0.16-0.19 m clearance, the truth was -0.08 to -0.11 m; radius and lookup fine (`l4_collisions/`).
- **(a)+filter** (`reports/lidar_demo/a_filter/`, seeds 0-499): success / collision / stalled % L0 94.8 / 0.0 / 4.4,
  L1 96.2 / 0.0 / 3.6, L2 96.4 / 0.0 / 3.6, L3 95.2 / 0.0 / 4.8, L4 85.5 / 0.2 / 13.8 (vs (a) L4 93.1 / 2.3 / 4.4). The
  safety transfers: collisions fall to ~0 for a cost of 0-2.3 points of success at L0-L3 and 7.6 at L4 (stalls). It
  beats `goalflow_l3_combined` on both success and collisions at every level.
- Latency, single worker (`l3_goalflow/latency/goalflow_l3_combined.json`, 12 episodes, 1,479 replans): 152 ms per
  replan; rollout 86 ms (56%), flow sampling 33, safety filter 10, encoder 6, edge critic 6, scorer 4, raster 2.5,
  circles + memory 0.9, goal selection 0.8, other 3.
- EMX pitch check, roof mount (`reports/lidar_demo/mounting/pitch_check.md`): pitch 5-8 deg give the same hits and circle
  rates (cones 34 / 21 / 15 / 11 / 8 returns at 10-30 m, circle 100 / 100 / 96 / 96 / 82%; bollards 100%); at 9 deg far
  bollards lose returns (circle 95-98% at 15-25 m, 90% at 30 m). The 2D sim does not model vertical beam count, so this
  only separates the top-beam cut-off; recommendation 7-8 deg stands.

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
- 11-27 19:45 (a)+filter run: exit 0 (reports/lidar_demo/a_filter/)
- 11-27 19:50 item 4 latency profile: exit 0 (reports/lidar_demo/l3_goalflow/latency/)
- 11-27 19:50 item 6 EMX pitch check: exit 0 (reports/lidar_demo/mounting/pitch_check.md)
- 11-27 20:46 item 5 juno_bangbang: GoalFlow exit 0, baselines last exit 0 (reports/lidar_demo/juno/)
- 11-27 20:52 item 8 scene overlap: exit 0 (reports/lidar_demo/scene_overlap/)
- 11-27 20:52 chain done
