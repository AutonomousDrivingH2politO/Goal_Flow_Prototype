# Deviations and findings

Every departure from the GoalFlow paper (`docs/GoalFlow.pdf`, arXiv 2503.05689v6) and from the
agreed designs in `PROMPT_LIDAR_DEMO.md`, with the reason. Newest findings first within each part.

## Part A. Findings from Phase L0 (simulator and corridor planners)

### A1. Funnel smoothing + pure pursuit is not feasible with a 10° steering limit

The agreed corridor design smooths the A* door sequence with line-of-sight (funnel) smoothing and
then checks curvature. Implemented as specified (`src/gfsem/planners/corridor.py`, kept as the
baseline **"corridor-funnel (original design)"**), it fails on scenes that are kinematically
feasible for Juno:

* The funnel path is a polyline that hugs the inflated obstacle corners. Juno can turn at most
  R = 1.60 m / tan 10° = 9.07 m. Lateral shifts at doors need an S-curve that starts about 10 m
  before the door. A polyline aimed straight at the door corner turns too late, so the rollout
  collides and the door is blocked even though a smooth path through it exists.
* Replanning at 10 Hz from the current pose makes it worse. Each new funnel path starts from
  wherever the car is, so the car is never on the early-turn line, and it can drift into a state
  with no feasible continuation.
* Measured on seeds 0–149 at L0, with the version that also reused plans and remembered blocked
  doors ("step 7"): 139/144 oracle-feasible scenes succeeded (96.5%), 0 collisions. Every failure
  braked at spawn with every door blocked by the curvature check (seeds 11, 58, 124, 134, 149).
  Pure original design at the single 0.25 m margin, seeds 0–499 at L0 (privileged, true
  scene): **361/477 (75.7%)**, all failures braking with doors blocked by the curvature check.
  The corridor-lattice teacher gets 474/477 (99.4%) on the same scenes
  (`reports/lidar_demo/l0/summary.md`).

**This also applies to the real AD-SEM stack.** Any planner that hands a corner-hugging polyline to
pure pursuit on Juno with the 10° limit will, in tight door sequences, either cut corners into
bollards or brake. The smoother has to be curvature-aware: it must plan in the space of
paths Juno can drive, not smooth geometrically and check afterwards.

### A2. Corridor-lattice planner (replaces step 5 of the agreed design)

`src/gfsem/planners/corridor_lattice.py`. Steps 1–4 and 6 are unchanged: Delaunay doors and walls,
A* with narrow-door and tentative-door costs, braking when there is no route. Step 5 is replaced:

* A* gives the top routes, as distinct door sequences found by blocking one obstacle door at a time.
* For each route, a forward-arc lattice search runs inside the route's **homotopy corridor**: a
  flood fill of triangles from the start that crosses only edges with no obstacle endpoint, or
  edges equivalent to one of the route's doors. All left-edge vertices count as one "L" vertex and
  all right-edge vertices as one "R".
  An earlier version used "chain triangles + every triangle sharing a vertex". That leaked
  around shared bollards, so alternative routes collapsed into the same path.
* Curvature ladder: 0.8 κ_max, then 0.9, then 0.95 (the oracle's limit). The level each plan needed
  is logged. On seeds 0–499: 469 plans needed 0.8, 3 needed 0.9, 2 needed 0.95.
* Curvature continuity: at most 2 curvature steps (of 7) between consecutive 1 m primitives, so pure
  pursuit can track the path.
* Validation is a closed-loop simulation: pure pursuit with a 2.5 m lookahead, the bicycle model,
  dt = 0.05 s. If that fails, the same route is retried with a stronger clearance preference in the
  lattice cost (cost shaping only, the margin is unchanged).
* **Teacher (privileged)**: the world is static and fully known, so it plans once per episode. It
  replans only if the car leaves the plan by more than 0.5 m (never happened at L0).
* **Observed baseline**: event-triggered. A kept plan is re-checked against each new observation
  and replanned only when it collides (0.25 m), ends more than 4 m short of the horizon, or the car
  deviates. Replans warm-start from the valid prefix of the old plan (dropping 4 m before the first
  invalid pose). The beam is 300, against 800 for the teacher.
* Multimodal labels: the lattice inside each of the top-K distinct routes, deduplicated by homotopy
  signature. This replaces the "±15% edge-cost noise" in the prompt.
* Plans are extended 5 m straight past the goal line. Without that, the kept plan ran out
  next to the goal and the replan failed.

### A3. Margins: plan with 0.25 m, accept tracked rollouts at 0.12 m

* The oracle, the door rule, the lattice and the warm-start validity check all keep **0.25 m** from
  obstacles.
* A **tracked rollout** (pure pursuit following the plan) is accepted if it keeps **0.12 m**.
  Pure-pursuit corner cutting of a few cm is expected (measured: 0.1–7.5 cm into the 0.25 m band on
  the paths that failed a single-margin check). With one 0.25 m margin everywhere, the teacher
  reached 464/477 (97.3%), with tracking error as the main failure cause.
* Episode success is unchanged: goal line reached with no contact and no track exit. Minimum
  clearance is reported separately (5th percentile in the results table).

### A4. Definition of "feasible" (the scene oracle)

`src/gfsem/sim/scene.py::oracle_feasible`. A scene is feasible if a forward-arc lattice
(1 m arcs, 7 curvatures up to **0.95 κ_max**) reaches the goal line with Juno's box keeping
**0.25 m** from every obstacle disc and staying inside the track (0 m to the edge).
Two changes from the first L0 version, both recorded here:

1. The obstacle inflation went from 0.15 m to **0.25 m**, so "feasible" matches the door rule
   (gap − 2r ≥ width + 2 × 0.25 m).
2. The box is now checked **every 0.25 m** along each arc (it was the middle and end of each 1 m
   arc). One "feasible" scene (seed 58 at the time) existed only because its path broke the margin
   by 1.5 cm between samples.

About 5% of scenes are generated without a feasible route (half of them as full bollard walls),
and their label comes from the same oracle.

### A5. Door rule against the track edge

The prompt's door rule is for gaps between obstacles. For a gap between an obstacle and the track
edge, the planners now measure the distance to the edge **line** (not to the nearest 1 m edge
sample) and require width + 0.25 m. That is clearance on the obstacle side only, matching the
oracle, which keeps 0 m from the edge. Found on seed 170: its only route passed a bollard 1.62 m
from the edge, which the two-sided rule (1.665 m) called a wall.

### A6. Open issue: observed-lattice is worse at L0 than at L1/L2

Observed corridor-lattice succeeds on 91.2% of feasible scenes at L0, but 97.1% at L1 and 98.1%
at L2. 33 seeds fail only at L0, almost all stalling late (12–15 s), next to the goal line.
Hypothesis: with noise-free edges the goal-line endpoints coincide exactly with edge points,
giving degenerate geometry where the edges are cut and triangulated at the goal. Any noise
breaks the tie. To fix before the L3 comparison, because it understates the baseline at L0.

### A6b. Bug found and fixed: the 12 m edge fill put points inside the track (L1–L4)

The opposite-edge fill offset each noisy edge point by the track width along a tangent from its
two neighbours. With noisy edges that tangent swung wildly, and the 12 m lever turned it into fill
points up to ~4 m inside the track (and behind the car). Both the observed planners' corridor and
the BEV drivable channel were affected. Fix (`sim/edges.py`): fit a curve to the source edge in its
chord frame (degree by length: line under 12 m, quadratic under 25 m, cubic above, one round of
outlier rejection), offset the *fitted* curve, and reject fill points closer than width − 1 m
to the source edge. Worst per-edge error at L4 is now 1.31 m (from observed points; median 0.69 m).
Test: `tests/test_sim_data.py::test_edge_fill_rejects_bad_offsets_on_noisy_edges`.

Effect: of the 18 seeds checked that succeeded at L3 but stalled at L4 in the evaluation, 16 now
succeed at L4. **The L1–L4 observed numbers in `reports/lidar_demo/l0/summary.md` predate this fix**
and will be regenerated together with the goal-stall fix (Part C, item 1). The L1 shards are
unaffected (raw observations, rasterised at load time).

### A7. Why corridor-funnel is worse on the true scene (75.7%) than observed (81.6%)

On the 58 scenes funnel-observed solves and funnel-privileged does not, the observed run sees 92%
of the obstacles at spawn (those behind rows appear late) and plans only to its ~25 m edge
horizon, so its curvature check blocks fewer doors. Its successes there keep a median clearance
of 0.17 m (p10 0.04 m), below the 0.25 m the privileged run requires. Its higher collision rate
is mostly a separate effect: 14 of its 18 collisions are on scenes the privileged run solves
(noisy detections), only 4 on scenes it fails.

### A8. Other simulator choices

* Juno's overhangs are not in the data sheet. Assumed symmetric: 0.6375 m front and rear of the
  axles (`configs/vehicle.yaml`, TODO(Can)). The ego frame origin is the rear-axle centre.
* The LiDAR is mounted on the front axle (x = 1.60 m), 1.10 m high, pitched 7° down. The cone
  visibility limit (≈ 2.03 m) and occlusion follow from the beam geometry. Cones do not hide
  bollards behind them, since beams pass over the cone. A beam exactly level with a bollard top
  does not count as passing over it.
* Free space in the raster is a Nav2-style raytrace. Rays without a return clear out to the
  maximum range, because ground returns would. Dropped returns clear nothing.
* **Observed edges.** Where one road edge is not observed (outside the 120° FOV near the car, or
  hidden behind a bollard row), it is filled from the opposite observed edge offset by the nominal
  track width (12 m). Observed points always take precedence. This is a map prior. Without it the
  observed planners braked on the inner edge of curves. The BEV raster's drivable channel uses the
  same function (`sim/edges.py::observed_edges`), so the learned planner and the corridor planners
  see the same road (`tests/test_sim_data.py`). The L1 shards store raw observations and are
  rasterised at load time, so the change needs no regeneration.
* BEV goal channel (5): the goal segment is clipped to the raster and drawn 2 px thick. If no part
  of it is inside, the far edge (rows 0–1, max x) is drawn inside the drivable corridor. Earlier it
  was a 1 px row that vanished in renders, and a goal segment beside the raster on a curve drew
  nothing. Test: channel 5 non-empty in every frame of a shard.
* Localisation drift is a random walk of the odom-frame offset, updated at each replan. It affects
  the goal line, the track-relative ego status and anything a planner keeps in the odom frame.
* Noise presets L0–L4 are listed in `src/gfsem/sim/config.py`. L0 is noise-free but still has
  occlusion and limited FOV and range, so "observed L0" is not "privileged".

## Part B. Planned model against the GoalFlow paper (checked before L1, model not built yet)

| Item | Paper | Plan (PROMPT_LIDAR_DEMO §3) | Status |
|---|---|---|---|
| Input / perception | Transfuser camera + LiDAR fusion into F_bev, aux losses on HD map (CE) + 3D boxes (CE, L1), w = 10/1/10 (Eq. 13) | 6-channel BEV raster from simulated LiDAR + edges, small ResNet → 16×16×D; optional aux head on true obstacle discs + drivable area (BCE) | **Deviation** (simulator has no camera; the raster mirrors the Nav2 costmap) |
| Hidden size | 256 baseline, 1024 best (Table 5) | D = 256 | Same as the paper's baseline; scaling not planned |
| Goal vocabulary | k-means on training endpoints (x, y, θ); N = 4096 or 8192 | k-means on teacher endpoints, θ scaled 1 rad = 2 m; N = 1024, sweep {256, 1024, 4096} | **Deviation**: smaller N (4 s horizons in a 12 m track give a much smaller endpoint space); the θ scaling is ours (paper does not state one) |
| Distance score target | softmax(−‖g_i − g_gt‖₂), no temperature (Eq. 3) | temperature T = 0.5 m | **Deviation** (sharper target for a small, dense vocabulary); listed in the prompt |
| DAC target | binary; all 4 corners of the shadow vehicle inside the drivable polygon | full Juno polygon inside track polygon minus *true obstacle discs* | **Deviation**: full polygon (corners alone miss a bollard between them) and obstacles included (the paper's DAC ignores them) |
| Scorer decoder | query = F_V + F_ego, key/value = F_bev, transformer decoder, two MLP heads | same, 3 decoder layers | Layer count not given in the paper |
| Goal selection | δ̂_final = w1 log δ̂_dis + w2 log δ̂_dac | same | w1, w2 not given in the paper; to be tuned on validation |
| Goal loss | L_goal = w4 L_dis + w5 L_dac, w4 = 1.0, w5 = 0.005; L_dis soft-target CE (Eq. 15), L_dac BCE (Eq. 16) | **w5 = 0.1 for the main seed-0 run**; 0.005 (paper) kept as an ablation on the P16v; sweep w5 ∈ {0.005, 0.1, 1.0} | **Deviation**: with w5 = 0.005 the shared encoder gets almost no obstacle signal (L_dac is ~0.5 against L_dis ~5), and in an early render g* landed next to an obstacle between the two modes. Our DAC target includes obstacles (the paper's does not), which makes it the encoder's main obstacle signal |
| Rectified flow | x0 ~ N(0, σ²I), σ = 0.1 best (Table 4); x_t = (1−t)x0 + tτ_norm; v = τ_norm − x0; τ_norm = H(τ_gt); L1 loss (Eq. 17) | same, σ = 0.1, L1, sweep σ ∈ {0.05, 0.1, 0.2} | Same. The intro notes use σ = 1 and MSE, which is the generic form |
| Normalisation H | not specified | per-axis affine to ≈ [−1, 1] from 1st/99th percentiles, stored in the checkpoint | Our choice |
| Trajectory | 2 Hz, 4 s → 8 waypoints (NAVSIM) | 4 Hz, 4 s → 16 waypoints (x, y), heading by finite differences | **Deviation** (finer for 3 m/s door manoeuvres) |
| Generator network | x_t linear, t and goal sinusoidal; F_env = E_env(Q, F_bev+F_ego, …) with learnable Q; F_all = concat(F_env, F_goal, F_traj, F_t) → N attention layers (Eqs. 7–9) | 16 waypoint tokens, 1 time, 1 goal, 16 env queries cross-attending to F_bev + F_ego, 4 encoder layers | Same structure; N not given in the paper |
| Condition dropout | "follows classifier-free guidance, conditioning set randomly masked"; probability not given | goal token → learned null token with p = 0.1 | Our probability; we mask only the goal (needed for the shadow trajectory) |
| Timestep distribution (training) | not stated | switchable uniform / logit-normal | Our choice; the intro notes use uniform |
| Sampling | Euler, Eq. 11; "smoothing method in [9] that re-scales the timesteps nonlinearly" (SD3 timestep shift) | Euler, n ∈ {1, 2, 5, 10}, uniform steps | **Deviation** unless we add the SD3 shift; to be ablated (matters only for n > 1) |
| Samples | 128 / 256 | M = 128 | Same |
| Trajectory scorer | f = −λ1 Φ(f_dis) + λ2 Φ(f_pg), Φ = min–max (Eq. 12) | same + λ3 · Φ(distance to the previously chosen path) | **Deviation**: λ3 is ours (commit / consistency); report λ3 = 0 and tuned |
| Shadow trajectory | goal masked; used if it "deviates significantly" from the main one | 16 shadow samples, threshold tuned on validation | Count and threshold not given in the paper |
| Ablation M3 | Table 2: M3 = M2 + trajectory scorer (the text says "builds upon M1", a typo by the table) | M3 = M2 + scorer + shadow | Follows the table |
| Training stages | perception first, then goal constructor; generator schedule not explicit | Stage A encoder + goal scorer, Stage B generator with A frozen, optional joint fine-tune | Our schedule |
| Goal selection | single best goal g* (argmax of δ̂_final) | **planned for L3**: best goal per route for the top-K routes (K = 2–3), M/K samples toward each, critics choose across all | **Extension** (not in tonight's run): one g* that lands between two modes, or next to an obstacle, commits every sample to it |
| Selection with critics | not in the paper | MPPI-style critics and softmax within the best homotopy cluster | **Extension** (team's idea) |
| Safety filter | not in the paper | drop candidates whose swept box touches hit/unknown cells; brake if none survive | **Extension**; results reported with it on and off |
| Executed-path horizon | — | the rollout is sampled at the candidate's own 16 timestamps (4 s); its speed command comes from a smoothed copy of the candidate, capped at 4.5 m/s, and the executed path is truncated at the smoothed candidate's arc length | Ours: a zigzag raw candidate implied much higher speeds than its net progress, so the executed path reached further than the candidate. Used for scoring, the shadow comparison and every ADE/FDE |
| Candidate feasibility | candidates scored as generated | every candidate rolled out through pure pursuit (2.5 m) + bicycle with the 10° limit before scoring; the rolled-out path is scored and returned; share of raw samples within κ_max logged (`kappa_ok_raw`) | **Deviation** (`models/rollout.py`): the paper's vehicle is not steering-limited like Juno; scoring raw samples would rank paths Juno cannot drive. Applies to M0–M3 (averaged trajectories are rolled out too), the scorer, the shadow comparison and, later, the critics |
| Goal-score evaluation | DAC/EP and PDMS | before ADE: share of multi-mode frames whose top-k goals cover every teacher mode (cover@5, cover@10) and whose distance score puts ≥ 10% mass near every mode | Ours (the multimodality question this demo is about) |
| Teacher | human driving logs | privileged corridor-lattice planner on the true scene | **Deviation**: student learns from our own planner; on clean input it can at best match it |

Docs note: the flow-matching notes are in `docs/` as
`Flow Matching_ A visual introduction _ Peter’s Notes.pdf` (the prompt calls it
`docs/flow_matching_intro.pdf`).

## Part C. Future work and queued items

### Future work (decided not to do now)

* **Track-edge margin of 0.10 m** in the oracle and planners. It would remove the seed 58/170-type
  cases (oracle paths 3–6 mm from the edge). Not done, so the current oracle and the L1 shards
  stay valid (decision 2026-11-24).

### Queued for Phase L3

1. Fix the observed-lattice L0 goal stall (A6), then rerun observed-lattice only (L0–L4, which also
   picks up the edge-fill fix A6b) and update `reports/lidar_demo/l0/summary.md`. Remaining L4 stall
   pattern to look at: fitted circles jitter at 45% dropout, so kept plans keep failing the 0.25 m
   check and replans alternate between a path and a brake (seed 288 crawls for 15 s).
2. **Obstacle tracking for the observed planners**: track fitted obstacles over time in the odom
   frame (smoothed positions, short persistence through dropouts) instead of re-fitting every
   scan from scratch.
3. **Kept-plan validation at the 0.12 m acceptance margin with hysteresis** (not 0.25 m), so
   jittering fits do not invalidate a good plan every scan (the seed 288 crawl).
4. **Hold position instead of creeping** while no valid plan exists (no alternating brake/plan).
5. GoalFlow's safety filter uses the same tracked obstacles and the same hysteresis, so the race
   is fair.
6. Rerun observed corridor-lattice at L0–L4 after the edge-fill (A6b) and goal-stall (A6) fixes and
   **replace every pre-fix number** (summary, chart, README, DEVIATIONS).
7. **Per-route goal selection** (our extension; the paper uses only the single best goal):
   * Group the vocabulary goals by route: by the homotopy class of a short teacher-style path to each
     goal, or by which side of each obstacle the goal lies on.
   * Take the best-scoring goal in each of the top-K routes (K = 2–3), generate M/K samples toward each,
     roll them all out, and let the critics choose across all of them.
   * Compare against single-g* selection on the g* collision rate, success at L0–L4, door flips per run,
     and latency (one extra sample batch per extra route).
8. Same real-time budget for every planner (fixed per-replan time cap; when the cap is hit,
   keep the previous plan or brake).
9. Failed-search caching: after a replan fails, skip the full retry ladder until the observation
   changes materially (today a failing L4 replan can cost seconds).
10. Numba (or vectorised) lattice expansion and collision checks.
11. Try routes strictly in A* cost order, stopping at the first that validates (no route lookahead).
12. Lookahead sweep on the corridor-funnel baseline: 2.5 / 4 / 6 m.

### Curvature-feasibility metric (reference values)

`kappa_ok_raw` is the share of raw samples whose discrete curvature (16 waypoints, 0.25 s) stays within
κ_max = tan 10° / 1.60 m at every interior point; `kappa_ok_raw_1.1` allows 1.1 × κ_max. Reference: the
teacher's own driven trajectories (val_0001, 2,573 paths) pass 98.1% at 1.0 × and 100% at 1.1 × (the gap
is discretisation where the teacher's steering saturates at 10°). The metric is very sensitive to
waypoint jitter: a 3 cm lateral wiggle at 0.75 m spacing already implies about κ_max.

### Phase L2 plan (decided 2026-11-24)

* Full training (seed 0, Stage A, Stage B, regression) runs on the Mac (MPS) as soon as the L1
  train split is complete (the P16v is not set up yet). Full training-state checkpoints at least every
  30 min, mid-stage resume, early stopping on the validation split only (the test split is never
  read). The 3-seed runs and ablations come later on the P16v.
* Goal-scorer diagnostics: g* collision rate (Juno box at g* overlapping a true obstacle, from the
  scene index), DAC accuracy on frames with an obstacle within 2 m of the driven path, balanced DAC
  accuracy, and the mean distance of sample endpoints from g* (raw and executed).
* Report order after training: cover@5 / cover@10, **g* collision rate**, mode-mass share, share of
  raw samples within κ_max, ADE / FDE (raw and executed, GoalFlow and regression), training time and
  checkpoint path.
* Open-loop metrics are reported for both the raw path and the executed path (after the pure-pursuit +
  bicycle rollout), for GoalFlow and the regression baseline alike.
