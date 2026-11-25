# L2 seed-0: selection variants on validation (2026-11-25)

Validation only (the L1 test split is spent). Frozen checkpoints `runs/l2_full_seed0/goalflow_B_best.pt`
(encoder and goal scorer = `goalflow_A_best.pt`) and `regression_best.pt`. 11,331 frames: the fixed 4,000-frame
subset plus every multi-route frame (7,777; 4,864 with endpoint-distinct routes). M = 128 samples, same noise
for every variant. Metrics and rules were pre-registered in `scripts/eval_selection.py` before the run; code at
launch a6c0c43, clean tree. Full numbers: `selection_val.json` (per-frame dumps `frames_*.npz` stay local);
goal scorer: `goal_scorer_val.json`; safety filter: `safety_calibration.json`, `safety_misses.json`.
Collision = Juno's footprint along the selected executed path overlaps a true obstacle (open loop).

## Decision: GoalFlow configuration for L3

**NMS-K3 goals + safety filter, shadow path off, 1 Euler step, M = 128.**

| Evidence (all 11,331 frames) | Collision rate | Paired with M3 (frames only M3 collides / only the variant collides) |
|---|---|---|
| M3 (single g*, shadow on, as trained and reported) | 0.79% | |
| M3 + safety filter | 0.79% (89 vs 90 frames) | identical except 1 frame |
| NMS-K3 + safety, shadow on | 0.59% | 41 / 18, exact McNemar p = 0.004 |
| NMS-K3 + safety, main path only (shadow off) | 0.36% | vs M3 main path (0.66%): 37 / 3, p = 2e-8 |

- **The safety filter needs alternatives.** With single g* the samples collapse onto one goal, so when they
  are flagged there is no safe candidate to switch to (M3 + safety = M3). With three NMS goals the filter can
  pick another route.
- **Shadow off.** The shadow switch fires in 305 frames (2.7%), all with a near g* (median 1.5 m ahead,
  ~10 m from the driven endpoint). There the shadow path collides in 23 frames against 8 for the main path, and
  it bypasses the safety filter. It lowers ADE (the main path crawls toward the near g*: ADE 3.3 m vs 0.45 m
  in those frames), but ADE against the driven route is not the criterion here. The near-g* frames are a
  closed-loop stall risk and are examined next (below).
- **Why "off" holds for the L3 configuration** (shadow sweep on NMS-K3 + safety):

  | Shadow threshold | Fires (all frames) | Collisions (all frames) | Executed ADE (subset) |
  |---|---|---|---|
  | off | 0 | 0.36% | 0.213 m |
  | 2.0 m | 47 | 0.40% | 0.202 m |
  | 1.5 m (as trained) | 103 | 0.59% | 0.200 m |

  For single-g* M3 a 2 m threshold was worth it (ADE 0.288 m with the shadow off, from the near-g* crawl), but
  NMS-K3 already removes most of that crawl (ADE 0.213 m off), so the shadow's small ADE gain no longer pays
  for its collisions.
- **Multiple goals give real alternatives at no cost.** With NMS-K3 the candidate set covers every teacher
  route in 83% of multi-route frames (single g*: 33%); by homotopy, the best candidate per goal covers every
  route in 69% (single g*: 27%). ADE and on-route rates are unchanged. The chosen path stays on the g* route
  (the l4 goal-score term ranks g*'s samples first), so top-10 coverage of the chosen candidates barely moves
  (0.32 -> 0.35).

## Euler steps: keep 1

| n | Raw samples within kappa_max (1.0x / 1.1x, subset) | Executed ADE (subset) | Mac throughput (batch 32, MPS) |
|---|---|---|---|
| 1 | 63% / 72% | 0.198 m | 22.7 ms per frame |
| 2 | 91% / 94% | 0.195 m | 38.6 ms per frame |
| 4 | 95% / 97% | 0.196 m | 71.0 ms per frame |

More steps make the raw samples much more drivable, but candidates are rolled out through pure pursuit + the
bicycle model before scoring, so the executed paths are drivable either way, and ADE and collisions do not
change. Latency here is Mac throughput, not the per-frame Orin latency of the 50 ms criterion.

## Route-based goal selection: parked

Route goals from the observed corridor planner (single scan) displace g*: g* collision 2.8% (K = 2) / 0.6%
(K = 3) vs 0.2% for NMS-K, and executed collisions on multi-route frames 4.6% / 1.8% vs 0.6%. Likely cause:
the corridors of reconverging routes share the region past the obstacles, so a goal is rarely exclusive to one
route (about half the slots fell back to "best goal anywhere in the corridor"), and the per-route pick replaces
the scorer's best goal with a worse one inside a corridor.

## Baselines and the honest gap

| Planner (all 11,331 frames) | Collision rate | Notes |
|---|---|---|
| Corridor-lattice observed, cold (single scan, pre-L3) | 0.07% | brakes in 3.8% of frames; ADE 0.53 m |
| GoalFlow NMS-K3 + safety, main path | 0.36% | |
| GoalFlow NMS-K3 + safety, shadow on | 0.59% | |
| Regression | 3.17% | mode averaging; best open-loop ADE 0.193 m (subset) |
| M0 (mean of goal-free samples) | 5.28% | |

**The success criterion "collisions <= either baseline" is not met open loop:** the observed corridor planner
collides less (0.07% vs 0.36%), partly because it brakes instead (3.8% of frames). GoalFlow beats regression
clearly (0.36% vs 3.17%). The closed-loop comparison in L3, with the same tracked perception and safety inputs
for both, decides it.

## Safety filter

On all M3 candidates: recall 90.3% of truly colliding candidates; false alarms 0.75% on collision-free
candidates, 0.16% on driven routes, 2.0% on all teacher routes (cell slack 0; the pre-set slack rule was
changed on validation because recall is flat, see DEVIATIONS). Per frame (calibration set), it flags the
colliding candidates in 20 of the 21 frames that have one; the miss is a fully occluded bollard.

## Next

The near-g* stall risk, on the L3 configuration (NMS-K3 + safety, main path), not M3: of the 305 frames where
M3's g* is a near goal, how many still have a near selected goal under NMS-K3, and for those: the teacher's
speed and progress over the 4 s, the main path's progress, and whether the main path is safe.
