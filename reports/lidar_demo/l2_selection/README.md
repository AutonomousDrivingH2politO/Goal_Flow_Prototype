# L2 seed-0: selection variants on validation (2026-11-25)

Validation only (the L1 test split is spent). Frozen checkpoints `runs/l2_full_seed0/goalflow_B_best.pt`
(encoder and goal scorer = `goalflow_A_best.pt`) and `regression_best.pt`. 11,331 frames: the fixed 4,000-frame
subset plus every multi-route frame (7,777; 4,864 with endpoint-distinct routes). M = 128 samples, same noise
for every variant. Metrics and rules were pre-registered in `scripts/eval_selection.py` before the run; code at
launch a6c0c43, clean tree. Full numbers: `selection_val.json`; goal scorer: `goal_scorer_val.json`; safety
filter: `safety_calibration.json`, `safety_misses.json`.

## Headlines

1. **Two Euler steps.** n = 2 raises the share of raw samples within kappa_max from 63% to 91% (n = 4: 95%)
   with no ADE cost (executed ADE 0.198 -> 0.195 m on the subset). Throughput cost on the Mac (batch 32, MPS):
   22.7 -> 38.6 ms per frame (n = 4: 71 ms). Not an Orin number.
2. **Multiple goals give the critics real alternatives, at no cost.** With NMS-K goals (1.5 m), the candidate
   set covers every teacher route in 71% (K = 2) / 83% (K = 3) of multi-route frames, against 33% for single
   g*; by homotopy, the best candidate per goal covers every route in 58% / 69% (single g*: 27%). Collision,
   ADE and on-route rates are unchanged. The chosen path itself stays on the g* route: the l4 goal-score term
   ranks g*'s samples first, so top-k coverage of the chosen candidates does not move (cover@10 0.32 -> 0.35).
   That is expected until the critics (L3) choose across routes.
3. **Route-based goals from the observed planner are worse than NMS-K.** Corridors of reconverging routes
   share the region past the obstacles, so about half the goal slots fall back to "best goal anywhere in the
   corridor". Coverage is lower (K = 3: 64% vs 83%), g* collision is higher (0.6% vs 0.2%; K = 2: 2.8%), and
   executed collisions rise (multi-route frames, K = 3: 1.8% vs 0.6%).
4. **Mode averaging shows up as collisions.** Executed-path collision rate on the 4,000-frame subset:
   regression 3.1%, M0 (mean of goal-free samples) 4.6%, M3 1.0%, M3 NMS-K3 + safety 0.7%. Near obstacles
   (2,875 frames): regression 4.0%, M0 5.9%, M3 1.3%, M3 NMS-K3 + safety 1.0%. Regression keeps the best
   open-loop ADE (0.193 m vs 0.198 m).
5. **The shadow switch adds collisions.** It fires in 3% of frames. In those frames (305) the scorer's g* is
   a near goal (median 1.5 m ahead, ~10 m from the driven endpoint), so the main path crawls (ADE 3.3 m); the
   shadow restores progress (ADE 0.45 m) but collides 3x as often (7.5% vs 2.6%), because it is not safety
   filtered. Sweep on the subset (M3): collision 0.97% at 1.5 m, 0.75% from 2.0 m up, 0.75% with the shadow
   off; ADE 0.198 / 0.202 / 0.288 m. A threshold of >= 2 m keeps the ADE benefit without the added collisions.
   The near-goal g* is a scorer failure mode (2.7% of frames) to look at before L3.
6. **Safety filter.** On all M3 candidates: recall 90.3% of truly colliding candidates; false alarms 0.75% on
   collision-free candidates, 0.16% on driven routes, 2.0% on all teacher routes. Its effect on the chosen path
   is small (collision 1.0% -> 1.0% single g*; 1.0% -> 0.7% with NMS-K3), since most collisions come from the
   shadow path and unseen obstacles. Per frame (calibration set), it flags the colliding candidates in 20 of 21
   frames; the miss is a fully occluded bollard.
7. **The cold corridor planner** (single scan, the route source): 0.0% collisions, 3.8% brakes, ADE 0.53 m
   (it often takes a different line than the teacher).

## Suggested settings for L3 (validation-based)

n = 2 Euler steps; NMS-K3 goals with the safety filter; shadow threshold 2 m (or the shadow passed through the
safety filter); M = 128. Route-based goal selection from single-scan corridors is not pursued in this form.
