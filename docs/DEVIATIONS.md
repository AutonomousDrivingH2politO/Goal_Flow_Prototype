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

### A6. Observed-lattice is worse at L0 than at L1/L2: two edge-test artefacts (analysed 2026-11-25)

Observed corridor-lattice succeeded on 91.2% of feasible scenes at L0 but 97.1% at L1 and 98.1% at L2
(3513584); 33 seeds failed only at L0. Full analysis: `reports/lidar_demo/l0/a6/README.md` (original
planner, worktree at bb0a9c1 = 50e9ad9 planner code, no L3 additions). With the edge-fill fix (A6b) 9 of the
33 already succeed at L0. The rest:

* **Goal-line edge stub (10 seeds; first brake 0.1-1.6 m before the goal).** On a curved track the goal
  line is oblique in the ego frame; after the observed edges are cut at it, one side keeps a 2-point stub
  (~0.15 m) at the line's end. The track-exit test in `CollisionWorld.clearance` (corner behind the inward
  normal of the nearest edge point) then puts all four corners "outside": the car's own pose is a track
  exit and every replan is rejected.
* **Fill edge inside the true edge beside the car (7 seeds; 3.7-20 m before the goal).** The forward
  LiDAR sees no edge beside the car; the 12 m fill lies 0.21-1.94 m inside the true edge there, and the
  car's corners (truly 0.15-0.2 m inside the track) fail the 0.12 m check against it.
* **Plan fails, own pose OK (11 seeds; 5 near the goal, 6 mid-course; incl. collisions 287 and 451).**
  The lattice finds no curvature-feasible path. Still open.
* One other own-pose failure (397) and 4 seeds that never brake (106 exits).

**Why only at L0:** the same own-pose failures occur at L1/L2 (8 and 6 of these seeds), but noisy edges
change from scan to scan, so a later scan passes; at L0 the faulty geometry repeats in every scan and the
brake is permanent.

**Rejected hypothesis:** smaller fitted radii at L1/L2 making tight gaps look wider. The planner fits only
circle centres (radius from the camera class); class flips are 4.9% / 8.2% at L1 / L2, and the narrowest
route gap looks 2-3 mm *narrower* than the truth, not wider.

**Approved fixes (2026-11-25):** (1) fix the goal-line stub at its source, where the edges are cut at the
goal line (`corridor.py`), observed mode only; the privileged teacher must keep 474/477 feasible successes,
0 collisions, 23/23 correct brakes on seeds 0-499 at L0. (2) Edge memory beside the car, in the L3 config
only (start relief is not used as a stand-in). Note for the L3 report: the corridor baseline then has edge
memory while GoalFlow's BEV stays single-scan. After both, the A6 table is rerun and the 11 plan-failure
seeds are classified (collisions 287 and 451 first).

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
and will be regenerated together with the A6 fixes (Part C, item 1). The L1 shards are
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
| Scorer decoder | query = F_V + F_ego, key/value = F_bev, transformer decoder, two MLP heads | 3 layers of **cross-attention + FFN only, no self-attention among the goal queries** (each goal is scored independently given the BEV); the standard decoder is kept as `scorer_self_attn=True` for an ablation | **Deviation** (found when the first full run thrashed): self-attention over the 1,024 vocabulary queries is a 1024 × 1024 map per head; at batch 64 stage A needed 17.65 GB of MPS memory on the 16 GB Mac (119 s/step). The paper does not say whether its decoder has goal self-attention. Layer count not given in the paper |
| Batch size (stage A) | — | effective batch 64 as specified, as **2 accumulated micro-batches of 32** (4.65 GB, ~0.83 s/step on MPS); stages B and regression at batch 64 directly (1.4 GB) | Memory on the Mac. Gradients verified equal to a full batch (float64, max relative difference 3e-6); BatchNorm statistics are per micro-batch |
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
13. **Safety filter on fitted/tracked obstacle circles.** The open-loop filter checks the raw hits channel.
    On 1,600 validation frames it flags the colliding candidates in 20 of the 21 frames that have one
    (95%; per candidate 1,686 of 1,814, 92.9%). All 128 missed candidates are one obstacle in one frame
    (val frame 22919, noise L1), clipped at the end of the horizon (first contact at 3.75 s): a bollard 11.8 m ahead fully occluded by a nearer
    bollard at the same bearing (no hit cell on it; nearest hit 1.79 m from the collision point). No miss
    was "seen but clipped on an unobserved side" (`reports/lidar_demo/l2_selection/safety_misses.json`).
    Occluded obstacles need memory: feed the filter the same fitted/tracked circles as the observed planner
    (`tracking.py`), so both planners use identical perception. Not implemented yet.
14. **Terminal feasibility for the observed lattice.** Accept a plan only if its end state admits a
    continuation, or extend plans through the next door. The 4 seeds still failing under the L3 config
    (108, 190, 203, 399) are dead-end states reached on a kept plan; for 190 and 399 the blocking door was
    observed from t = 0 at 22-35 m (`reports/lidar_demo/l0/a6/README.md`).
15. **Circle fit with flank-only hits.** `fit_circle_fixed_r` pushes the centre by r along the sensor ray,
    which assumes front-face hits. When a cluster's front face is below the lowest beam or in the blind zone,
    use a two-point geometric solution or keep the tracked centre (the 287 / 451 cone grazes, 0.30-0.33 m
    offset). Config (a) stays as it is.
16. **MPPI refinement around the chosen GoalFlow sample.** In tight gaps GoalFlow's best candidate clears the
    true obstacles by ~0.01 m (seed 17), so the safety filter brakes; a local MPPI refinement of the chosen sample
    could add margin. Not implemented.
17. **Stall diagnosis and a "second chance" before braking (queued 2026-11-25, not run).** For the 6 goalflow_l3
    smoke-test stalls (L0: 4, 36; L4: 1, 18, 28, 32), at the stalled scans: is baseline (a)'s route (homotopy
    signature of its driven path vs the true obstacles) among the routes of GoalFlow's 3 NMS goals, and what is the
    best true clearance per goal? If another route was available but not sampled, try a "second chance" before
    braking: resample once with K = 6 goals, excluding goals whose candidates were all flagged; brake only if that
    also yields no safe candidate. Order vs item 16 (MPPI refinement): run this diagnosis first (replays only); if
    (a)'s route is mostly missing from the 3 goals, try the second chance first (one extra model call on ~3% of
    replans, targets route choice, e.g. seed 28); if it is mostly present but its candidates clear by only a few cm,
    MPPI refinement first (targets precision in tight gaps, e.g. seed 17).
18. **Candidate margin against the fitted circles (queued 2026-11-26, rules to be pre-registered before running):**
    (1) the gradient refinement (step C variant b) run against the fitted obstacle circles instead of raw hits;
    (2) a clearance critic: among candidates passing the 0.12 m circle filter, prefer larger clearance to the circles
    (soft, saturating toward 0.25 m), weight fixed in advance. MPPI refinement (item 16) only if both fail. Then
    steps 5 (latency profile), 6 (brake-rate column in the selection table) and 7 (tracker birth rule).
    **Plan (decided 2026-11-25):** this diagnosis first, before the full run. If it points to the second chance or
    to MPPI, implement that one and rerun the same 80-episode smoke test (seeds 0-39, L0 and L4) under a rule fixed
    here before the run: fewest stalls + collisions on feasible scenes, ties to fewer collisions; goalflow_l3 is kept
    unless the new variant wins. The full run then uses the winner, on Can's go.

### L2 findings on validation (2026-11-25)

**Goal scorer: not a single-route predictor** (`reports/lidar_demo/l2_selection/goal_scorer_val.json`,
full validation split, 66,063 frames; replaces the earlier "scorer keeps one mode" and "vocabulary range"
diagnoses). The vocabulary covers every teacher route (nearest goal median 0.015–0.02 m, p95 <= 0.07 m, 0%
beyond 0.5 m); the y in [-3.56, 3.58] m in the training log is the trajectory normaliser, not the
vocabulary (y in [-6.56, 6.57] m). The driven route ranks first; with 1.5 m suppression the other route is
the 2nd distinct pick in 76% (all routes) / 61% (endpoint-distinct) and within 5 in 96.5% / 94%. Suppressed
cover@2/3/5 = 0.71/0.84/0.96 (all) and 0.57/0.75/0.94 (distinct), against plain cover@5 0.35/0.02: plain
top-k on the dense vocabulary is used up by neighbours of the best goal. The l_dis plateau (5.83) is mostly
the entropy floor (mixture gap 0.29 nats; non-driven routes 1.26). The sample collapse in the renders comes
from conditioning all samples on one g*. Hence per-route / multi-goal selection (Part C item 7) is the lever,
not the vocabulary.

**Safety-filter slack: pre-set rule changed on validation.** `scripts/calibrate_safety.py` (1,600
validation frames) pre-set the rule "largest slack with driven-route false alarms <= 1%", which picks
0.05 m. The rule ignored recall, and recall turned out flat: 92.9% (1,686 of 1,814 truly colliding M3
candidates) at slack 0, 0.05 and 0.1 m. False alarms rise with slack: driven routes 0.25% / 0.88% / 2.3%,
all teacher routes 0.84% / 1.56% / 3.06%, collision-free candidates 1.06% / 2.36% / 3.85%. Slack 0 dominates
on every rate, so the filter uses cell_slack = 0 (changed after seeing these numbers, for that reason).



### L2 selection decisions (2026-11-25; `reports/lidar_demo/l2_selection/REPORT.md`)

* **GoalFlow configuration for L3: NMS-K3 goals + safety filter, shadow path off, 1 Euler step, M = 128.**
  All 11,331 validation frames, open loop: collisions M3 0.79% -> NMS-K3 + safety 0.59% (paired 41 vs 18
  frames, exact McNemar p = 0.004); main path only 0.36% (vs M3's main path 0.66%: 37 vs 3, p = 2e-8).
  M3 + safety = M3 (collapsed samples leave no safe alternative). The shadow switch fires in 305 frames (near
  g*), collides there in 23 frames vs 8 for the main path, and bypasses the safety filter, so it is off.
* **Route-based goal selection (item 7): parked.** Route goals from single-scan observed corridors displace
  g* (g* collision 2.8% vs 0.2% for NMS-K). Likely cause: corridors of reconverging routes share the region
  past the obstacles, so few goals are exclusive to one route; about half the slots fell back to the best goal
  anywhere in the corridor, replacing the scorer's best goal with a worse one.
* **Euler steps: keep n = 1.** Raw samples within kappa_max: 63% / 91% / 95% for n = 1 / 2 / 4 (1.1x: 72% /
  94% / 97%); Mac throughput (batch 32, MPS) 22.7 / 38.6 / 71.0 ms per frame. Executed paths are drivable for
  every n (rollout before scoring); ADE and collisions do not change.
* **Honest gap:** the observed corridor planner (cold, single scan) collides in 0.07% of frames (braking in
  3.8%) against 0.36-0.59% for GoalFlow, so "collisions <= either baseline" is not met open loop. GoalFlow
  beats regression (3.17%). The closed-loop L3 comparison, with identical tracked perception for both, decides.

### Pre-registered L3 comparison baseline (2026-11-25)

GoalFlow (L3 configuration) is compared, at each noise level, against the **best classical observed config at
that level**: currently (a) pre-L3 + stub fix at L2-L4 and (b) stub fix + L3 at L0-L1
(`reports/lidar_demo/l0_rerun/REPORT.md`). Config (c), (b) without edge memory, is reported alongside as the
same-information baseline (single-scan edges like GoalFlow's BEV, with tracked obstacles). If the stall
ablation changes which config is best at a level, the best config is re-picked on these validation scenes
before any GoalFlow closed-loop result exists, and recorded here.
**Re-pick after the drift change (2026-11-26):** with distance-proportional drift the best classical config is (b) at
L0 (unchanged; no drift) and **(a) at L1-L4** (L1: (a) 97.9% vs (b) 97.1%, a 4-scene margin; previously (b) 97.3% vs
(a) 97.1%). Chosen from the baseline rerun alone (`reports/lidar_demo/l0_rerun_distdrift/REPORT.md`); the only
GoalFlow closed-loop results so far are 40-seed smoke tests used for variant selection, not for this choice.

### L3 observed additions: findings and frozen baselines (2026-11-25)

`reports/lidar_demo/l0_rerun/REPORT.md`, `reports/lidar_demo/l0_rerun/ablation_L4/`. The L3 set (tracking, edge
memory, kept-plan hysteresis + fallback, hold, failed-search cache) helps at L0-L1 (zero collisions) but stalls
at L3-L4. Stall ablation of (b) at L4 (success / collision / exit / stalled %, seeds 0-499):

| L4 | Success | Collision | Exit | Stalled |
|---|---|---|---|---|
| (a) pre-L3 + stub fix | 93.7 | 2.3 | 0.4 | 3.6 |
| (b) stub fix + L3 | 74.8 | 1.9 | 0.8 | 22.4 |
| (b) without failed-search cache | 74.8 | 1.9 | 0.8 | 22.4 (identical in all 500: the cache never fires at L4) |
| (b) without hold | 86.8 | 3.8 | 0.8 | 8.6 |
| (b) without hysteresis + fallback | 77.8 | 0.8 | 0.4 | 21.0 |

* The cache hypothesis is falsified: under L4 noise the tracked positions change every scan, so its "tracks
  unchanged" condition never holds (0 hits; it fires at L0-L3: 477 / 76 / 36 / 7 hits, mostly in no-route
  scenes, without changing outcomes). Kept for now; decided with the real-time budget after the Orin
  measurement.
* Hold is a filter, not a bug: in (b) a stalled episode spends 40 braked scans, 11 with a plan found but
  held, 29 with no plan. Without hold, stalls drop (22.4 -> 8.6%) but collisions double (1.9 -> 3.8%): it
  rejects flickering bad plans.
* "No route (A*)" stalls at L4 (perception check, `perception_noroute_*.json`): at the failing scans the planner
  holds ~21 tracked obstacles in view against ~11 true ones: ~5.2 phantoms per scan (no true obstacle within
  0.6 m; in 99% of failing scans, close to the L4 clutter rate of 5 per scan) and, by difference, ~4.9 extra
  tracks within 0.6 m of true obstacles (likely duplicate tracks of displaced fits); 0.2 true obstacles missed;
  matched centre error 0.15 m. Not fixed yet.
* **Stop rule applied:** no L3 variant beats (a) at L3-L4, so the L3-level rerun is skipped and the comparison
  baselines are frozen as pre-registered: (a) at L2-L4, (b) at L0-L1, (c) reported as the same-information
  baseline. Next: the closed-loop GoalFlow planner.

### Closed-loop GoalFlow: hit memory in the safety filter (2026-11-25)

GoalFlow's safety filter checks candidates against the current scan's observed hits plus remembered hits: a map
of 0.2 m cells in the odom frame (believed, noisy, drifting pose; never the true pose), a cell used once hit in
>= 2 scans and while it is ahead of the rear bumper and within 4 m of the car (position-based, no time limit).
First version (1.5 s time window) replaced after the smoke test: the cone collisions came 2.05-2.15 s after the
cone vanished below the lowest beam, with the car crawling (1.1-1.2 m travelled); remembered single-scan clutter
blocked candidates at L4. A cone-only memory was rejected: it would let a learned camera label gate which
obstacles the filter keeps (team rule), drop ~1 in 5 cones at L4 (label flips), and forget bollards beside the
car. Seed 17's L0 stall is mostly correct braking: the best candidate cleared the true obstacles by a median
0.010 m, and only ~0.9 of 128 per stalled scan by more than 0.12 m (candidate quality in tight gaps).

Position-based memory without a time limit removed all collisions but stalled 18 of 39 feasible L4 scenes
(seeds 0-39). Replay of L4 stall seeds 1, 4, 9: the blocking remembered cells lie where the current scan cannot
see (~6 per stalled scan vs 0.1-0.6 visible), and most are not on real obstacles (median 0.51 / 0.14 / 0.96 m from
the nearest true surface): cells stored 10-20 m ahead are displaced by accumulated heading drift when mapped back.

**Pre-set rule (2026-11-25, before the smoke test):** two memory variants run the same 80-episode smoke test
(seeds 0-39, L0 and L4): (i) visibility gate (remembered cells used only where the current scan cannot see), and
(ii) near-observation memory (only hits observed within 4 m of the sensor are remembered; position-based
retention and the 2-scan rule in both). The full run uses the variant with the fewest collisions + stalls over
those 80 episodes (feasible scenes; ties: fewer collisions).
**Result:** near-observation memory: L0 37 / 0 / 2 / 0, L4 35 / 0 / 4 / 0 (success / collision / stalled / exit,
feasible scenes), 6 collisions + stalls, "no safe candidate" brakes 3.0% of replans; visibility gate: L0
38 / 0 / 1 / 0, L4 19 / 0 / 20 / 0, 21 collisions + stalls, 7.6%. **The full run uses the near-observation memory**
(`eval_corridor.py --planners goalflow_l3`, the default; the losing variant is `goalflow_l3_visgate`). The BEV input stays single-scan, as trained. Reason: a 0.40 m cone falls below the lowest
beam within ~2.03 m of the sensor, so a single scan loses it exactly when the car reaches it (the closed-loop
smoke test's only collisions were such cones), while the corridor baseline (a) has implicit memory through its
kept plan. If no candidate survives the filter, the planner brakes.

### Closed-loop GoalFlow: critics off, and the tight-gap precision rule (2026-11-26)

**Critics off.** GoalFlow's trajectory score uses only the distance to the candidate's goal, progress and the goal's
own score (NMS-K); the lambda3 consistency term (distance to the previous chosen path), a smoothness critic and a
track-edge critic are off. Reasons: lambda3 = 0 was the validated configuration (open loop has no previous path to
tune it on); every candidate is already a rolled-out, curvature-feasible executed path, so smoothness is largely
enforced by the rollout; track exits stayed at 0 in the smoke tests, so no edge critic was needed yet. Check in the
full run: door flips per run for every planner, with the baselines' definition (the chosen path's side signature
changing between consecutive replans). If GoalFlow flips more than the baseline, lambda3 is turned on and the run
repeated.

**Tight-gap precision, pre-set rule (before any run).** Variants, cheapest first, each on top of the step-2 filter
(exact box-to-point, 0.12 m): (a) goal jitter: per NMS goal also goals shifted +-0.3 m sideways, M = 128 in total
(3 goals x 3 offsets, ~14 samples each); (b) a few gradient steps on the chosen candidates' waypoints away from the
hit points (exact distances), then re-rollout. Both run the same 80-episode smoke test (seeds 0-39, L0 and L4). The
winner is the one with the fewest stalls + collisions on feasible scenes, ties to fewer collisions; if neither beats
the step-2 variant, the step-2 variant is kept. MPPI refinement (Part C item 16) only if both fail.
**Amendment (2026-11-26, before running), split by noise model:** the L0 half of that smoke test (seeds 0-39) runs
now, since L0 has no odometry drift; the L4 half waits for the distance-proportional drift. Variant names:
`goalflow_l3_exact_jitter` (a; +-0.3 m) and `goalflow_l3_exact_refine` (b; 8 gradient steps, only when no candidate
is safe). Reference: `goalflow_l3_exact`. The rule is applied once both halves exist; the L0 half alone is
reported, not used to choose.
**Final choice rule (2026-11-26, before the L4 half):** candidates `goalflow_l3_exact`, `goalflow_l3_exact_jitter`,
`goalflow_l3_exact_refine`. Score = stalls + collisions on feasible scenes over seeds 0-39 at L0 and L4 (80
episodes), L4 run with the distance-proportional drift; the L0 halves already run are reused (no drift at L0, so
identical under either drift model). Lowest score wins; ties go to fewer collisions; a remaining tie keeps
`goalflow_l3_exact`. The winner is the GoalFlow variant for the full run.
**Result:** exact L0 37/0/2/0, L4 27/0/12/0 (score 14); jitter 38/0/1/0, 23/0/16/0 (17); refine 36/0/3/0, 28/0/11/0
(14) (success / collision / stalled / exit, feasible scenes). Exact and refine tie at 14 with 0 collisions each, so
the rule keeps **`goalflow_l3_exact`** for the full run. Note: at L4 its stalls rose from 8 (time-based drift) to 12
with the distance-proportional drift; the drift fix did not remove the L4 stalls.

**Halo diagnosis and `goalflow_l3_circles` (pre-registered 2026-11-26, before running).** Replays of 5 of the 12 L4
stalls (1, 13, 21, 29, 35; `reports/lidar_demo/l3_goalflow/diagnose_halo.json`): the points that block truly clear
candidates lie a median 0.17-0.36 m outside the true obstacle surface (on the free/gap side), are 65-100%
remembered, come from 23-80 scans per obstacle, and sit in the outer part of the scatter (above the obstacle's own
median protrusion, but only 5-39% in its outermost decile). L4 range noise scatters raw hits and the memory keeps their
union, so each obstacle grows a ~0.3 m halo that closes tight gaps; the corridor baseline fits circles, which
averages the noise. **Variant:** the safety filter checks the exact 0.12 m clearance (Juno's box to the circle
surface) against obstacle circles fitted per scan by the same `cluster_obstacles` code as baseline (a), plus a
near-field memory of circles: circles observed within 4 m of the sensor update it (centre averaged over sightings,
associated within 0.6 m, in the believed odom frame), kept until behind the rear bumper or > 4 m away; position
only, no time limit. **Rule:** 80-episode smoke test (seeds 0-39, L0 and L4; L4 with the distance-proportional
drift; L0 is rerun because the filter changes L0 too), score = stalls + collisions on feasible scenes, compared with
`goalflow_l3_exact` (score 14); lower wins, ties to fewer collisions, otherwise `goalflow_l3_exact` is kept. Full run
only after that, on Can's go.
**Result:** circles L0 36/0/3/0, L4 31/0/8/0 (score 11, 0 collisions; brake 2.9% / 7.5%, 215 ms per replan) vs exact
14 -> the rule picks **`goalflow_l3_circles`**. L4 still trails (a) on these seeds (37/1/1/0). Phantom check on the 8
L4 stalls (`diagnose_phantom_circles.json`): 30-65% of the filter's circles are phantoms (no true obstacle within
0.5 m), but none of the circles blocking truly clear candidates is a phantom; every blocker is a real obstacle's
fitted circle, slightly off the true disc, against candidates that clear the truth by only a little. A 2-sighting
rule would not remove these stalls; the remaining lever is candidate margin.

### Odometry drift proportional to distance driven (decided 2026-11-26, before any rerun)

**Change.** The closed-loop noise model's odometry drift was a time-based random walk: every replan (10 Hz) added
N(0, drift_xy) to x and y and N(0, drift_yaw) to the heading, whatever the car did. It now scales with the distance
driven since the previous replan: each increment is multiplied by sqrt(ds / 0.30 m), so the variance grows in
proportion to distance. **Calibration:** 0.30 m is one replan at 3 m/s, so at cruising speed the drift per scan
equals the old value and each noise level keeps its meaning; only slow and stopped driving change (a stopped car
no longer drifts). Same random draws as before (the streams stay aligned); only their scale changes.
**Reason.** With the time-based walk a car crawling at 0.5 m/s accrued ~6x more drift per metre than at 3 m/s, and a
stopped car kept drifting; real wheel odometry error grows with distance. The GoalFlow filter memory exposed it:
80-100% of the remembered points blocking truly clear candidates at the L4 stalls had drifted while moving slowly,
and at seed 1 drifted points (median 0.28 m off the true surface) let a path through a remembered cone
(`reports/lidar_demo/l3_goalflow/diagnose_drift_exposure.json`). A zero-velocity update was rejected: it would
remove at most ~15-20% of that drift.
**Scope.** Applies to every planner (corridor baselines and GoalFlow). The baselines (a)-(d) are rerun at L1-L4
(L0 has no drift); the old L1-L4 tables stay, labelled "time-based drift". Training data is **not** regenerated:
drift affects only map-derived inputs there (the goal line and the track-relative ego status), and the shards
used the time-based model. `run_episode(drift_per_distance=False)` reproduces the old model (the 50e9ad9 planner
regression test uses it).
**Real-car lesson.** The filter memory is only as good as the odometry: on Juno it needs wheel odometry + IMU in the
EKF (the ZED visual odometry read 10-60% low).

### Run hygiene

Run hygiene (house rules, 2026-11-26): (1) stopping a chain stops its whole process group (`kill -- -<PGID>`,
or `pkill -f` on eval_corridor for that output folder), then `ps` confirms no eval_corridor / worker processes remain
before anything is relaunched (killing a bash chain script leaves its running child evaluation alive; on 2026-11-26
config (a) briefly ran twice into one folder because of this); (2) never two runs writing to the same output folder:
`scripts/eval_corridor.py` holds `RUNNING.lock` in its output folder and refuses to start if a live process holds it;
(3) more load on request still keeps >= 5% CPU idle, about 7 CPU-bound workers in total on the 8-core Mac.

### EMX sensor level (2026-11-27)

**Sensor facts (RoboSense EMX user guide, Table 1 "EMX Specifications", p. 8; generationrobots.com copy):** range
1.5-200 m (200 m @ 10% NIST), blind zone <= 1.5 m; horizontal FOV 120 deg (-60 to +60), resolution 0.08 deg; vertical
FOV 20 deg (-12 to +8), resolution 0.1 deg (192 lines); accuracy +-5 cm @ 1 sigma (typical); frame rate 10 Hz;
2,880,000 pts/s. These match `LidarConfig` (blind zone 1.5 m, 120 deg, -12/+8 deg, 10 Hz), so L0-L4 are unchanged.
A reseller's 140 deg horizontal FOV contradicts the manufacturer's guide; 120 deg is used. The guide gives 0.1 deg
vertical resolution but no per-beam table: uniform spacing is assumed (the simulator models the vertical span as
continuous, which 0.1 deg spacing approximates).

**Preset `EMX`** (added alongside L0-L4, never replacing them; `NOISE_LEVELS["EMX"]`, not in `LADDER`):

| Parameter | EMX value | Source |
|---|---|---|
| range noise (1 sigma) | 0.05 m | spec (accuracy +-5 cm @ 1 sigma) |
| horizontal ray step | 0.08 deg (1,501 rays over 120 deg) | spec (horizontal resolution) |
| vertical span | continuous, -12 to +8 deg | spec (0.1 deg over 20 deg) |
| frame rate | 10 Hz | spec; Juno's planning rate |
| bearing jitter | 0.05 deg | **assumption** (L1 value) |
| point dropout | 5% | **assumption** (L1) |
| clutter | 0.5 clusters / scan | **assumption** (L1) |
| edge noise / edge dropouts | 0.10 m / 0.3 per side per scan | **assumption** (L1) |
| class flips | 5% | **assumption** (L1) |
| odometry drift | 0.005 m, 0.05 deg per 0.30 m driven | **assumption** (L1; distance-proportional model) |

Unit check (2D simulator, one return per horizontal ray per obstacle; not comparable to the 3D 2.88 M pts/s): hits per
bollard EMX vs L2 at 3 / 10 / 20 m from the sensor: 104 vs 40 / 32 vs 15 / 18 vs 11 (mean of 20 scans); per scan on
scene 3: 238 vs 87 points, simulation 0.7 vs 0.5 ms, rasterisation 4.7 vs 2.5 ms. `observation_from_shard` now
derives the ray geometry from the stored scan length, so 0.08 deg shards rebuild correctly (0.2 deg shards
unchanged).

**Pre-registered (2026-11-27, before any EMX result):** EMX is the realistic headline level of the final comparison;
L0-L4 remain robustness and stress levels (L4 in particular is far noisier and sparser than the real sensor). The
per-level baseline rule (best classical config per level, chosen from the baseline run alone) applies to EMX too.
GoalFlow runs at EMX only after the baselines at EMX and the open-loop distribution-shift check
(`scripts/emx_shift_check.py`: GoalFlow was trained on 0.2 deg rays; same validation scenes, seeds 250000+, at EMX vs L2;
a clear drop is a sim-to-real warning and puts retraining with EMX-level data in the queue).

**EMX mounting (analytic; height 1.10 m, vertical FOV -12/+8 deg about the pitched axis, blind zone 1.5 m, Juno's body
front at 2.24 m from the rear axle, assuming symmetric 0.6375 m overhangs, still unmeasured):**

| mount_x [m] | pitch [deg] | lowest / highest beam [deg] | blind zone past bumper [m] | 0.40 m cone invisible within [m from sensor / past bumper] | lowest beam hits ground [m from sensor / past bumper] | 0.40 m cone hit up to | 1.10 m bollard hit up to |
|---|---|---|---|---|---|---|---|
| 1.60 (front axle) | 10 | -22 / -2 | +0.86 | 1.73 / +1.10 | 2.72 / +2.09 | 31.5 m | 31.5 m |
| 1.60 | 8 | -20 / +0 | +0.86 | 1.92 / +1.29 | 3.02 / +2.38 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.60 | 7 | -19 / +1 | +0.86 | 2.03 / +1.40 | 3.19 / +2.56 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.60 | 4 | -16 / +4 | +0.86 | 2.44 / +1.80 | 3.84 / +3.20 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.60 | 0 | -12 / +8 | +0.86 | 3.29 / +2.66 | 5.18 / +4.54 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.00 | 10 | -22 / -2 | +0.26 | 1.73 / +0.50 | 2.72 / +1.49 | 31.5 m | 31.5 m |
| 1.00 | 8 | -20 / +0 | +0.26 | 1.92 / +0.69 | 3.02 / +1.78 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.00 | 7 | -19 / +1 | +0.26 | 2.03 / +0.80 | 3.19 / +1.96 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.00 | 4 | -16 / +4 | +0.26 | 2.44 / +1.20 | 3.84 / +2.60 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 1.00 | 0 | -12 / +8 | +0.26 | 3.29 / +2.06 | 5.18 / +3.94 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 0.50 | 10 | -22 / -2 | -0.24 | 1.73 / -0.00 | 2.72 / +0.99 | 31.5 m | 31.5 m |
| 0.50 | 8 | -20 / +0 | -0.24 | 1.92 / +0.19 | 3.02 / +1.28 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 0.50 | 7 | -19 / +1 | -0.24 | 2.03 / +0.30 | 3.19 / +1.46 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 0.50 | 4 | -16 / +4 | -0.24 | 2.44 / +0.70 | 3.84 / +2.10 | beyond sim max range (40 m) | beyond sim max range (40 m) |
| 0.50 | 0 | -12 / +8 | -0.24 | 3.29 / +1.56 | 5.18 / +3.44 | beyond sim max range (40 m) | beyond sim max range (40 m) |

Far coverage (geometric; height 1.1 m): at pitch <= 8 deg the top beam is at or above the horizontal, so cones and
bollards are hit beyond the simulator's 40 m max range (real reach is limited by the EMX's rated range on
low-reflectivity targets, 200 m @ 10% NIST, and by beam density); above 8 deg every beam points down
and nothing standing on the ground is hit beyond where the top beam meets the ground (1.1 m / tan(pitch - 8 deg): 31.5
m at 10 deg). Every pitch below 8 deg leaves some beams above the horizontal (+1 / +4 / +8 deg at 7 / 4 / 0 deg pitch).
**Far range needed:** a gap must be seen ~10 m before the swerve at 3 m/s, plus planning, so ~15-20 m; geometrically
the steepest pitch that keeps 20 m is 11.1 deg (15 m: 12.2 deg), so 10 deg still meets it (31.5 m) while
cutting the cone blind range to 1.73 m from the sensor; this ignores reflectivity and beam density at range. Mounting the
sensor further back shrinks the blind region in front of the bumper, and more downward pitch shrinks the region where
low cones vanish. But both cost far range and upward coverage, and at 1.10 m height a rearward sensor may have its
near field hidden by Juno's own body (not modelled).

### Vehicle data and the roof LiDAR mount (2026-11-27)

**Juno data (Juno data sheet, internal spreadsheet, not committed; values recorded here):** overall height 1,165 mm,
ground clearance 100 mm, length 2,875 mm, width 1,165 mm, wheelbase 1,600 mm, track 1,000 mm, tyre radius 0.2792 m. The
sheet's 5.5 m turning radius is mechanical; the planners keep the 10 deg software steering limit (~9.1 m radius). From
a side photo: nose ~2.35 m ahead of the rear axle (front overhang ~0.75 m); the AD-SEM footprint says +2.445 m. The
simulator still assumes symmetric 0.6375 m overhangs (body front 2.2375 m) for every run so far.

**Roof mount.** The EMX will sit on the roof at the base of the ZED support, at the top edge of the windshield,
outside the glass (the ZED stays on top of the support): ~1.0-1.1 m ahead of the rear axle, optical centre ~1.20 m
(roof 1.165 m + half the housing). The LiDAR mount is one config entry, `LidarConfig` (mount_x from the rear axle,
height, pitch), with presets in `LIDAR_MOUNTS`: `front_axle` (1.60 m, 1.10 m, 7 deg; the default, every run so far)
and `roof` (1.05 m, 1.20 m, 7 deg; provisional until measured). Sim, raster and planners take the mount from it;
`eval_corridor.py --mount roof` selects it.

**Self-occlusion (placeholder geometry, assumptions until measured).** The windshield and hood are a polyline in the
sensor's vertical plane: roof edge 3.5 cm below the optical centre just ahead of it, windshield sloping 20 deg below
the horizontal (from the photo) over 0.6 m, a mid-hood point, the nose top at 0.55 m height at the body front
(`placeholder_body_profile`). Beams below its clearance angle hit the car; with a profile set, the simulator clips the
lowest beam there (`LidarConfig.body_clearance_rad`; the `roof` preset sets it, `front_axle` does not). Removing beams
that hit the car is a simulator simplification: the real EMX returns points on the glass and hood, so AD-SEM needs a
body crop box that drops them (car bring-up item, not simulation work). Roof preset:
clearance -22.9 deg, so the steepest usable pitch is ~10.9 deg (lowest beam = -12 deg - pitch). It depends on the
unmeasured windshield length: 0.4 / 0.6 / 0.9 / 1.2 m of glass give -24.3 / -22.9 / -21.9 / -21.5 deg, i.e. a steepest
pitch of 12.3 / 10.9 / 9.9 / 9.5 deg; a long 20 deg windshield tends to ~8 deg.

**Roof mounting table (analytic; blind zone 1.5 m; -12/+8 deg about the pitched axis; placeholder body profile):**

Summary (full table: `reports/lidar_demo/mounting/roof_table.csv`, 30 rows; mount_x 1.0 / 1.1 m, height 1.15 /
1.20 / 1.25 m, pitch 5-9 deg, body front 2.35 / 2.445 m): blind zone past the bumper 0.06-0.25 m; 0.40 m cone
invisible past the bumper 0.5-1.5 m (smaller with more pitch, larger with more height; ~1.0 m at 7 deg); with the
placeholder windshield no lowest beam hits the car in any row; cones and bollards are hit beyond the sim's 40 m max
range up to 8 deg pitch and up to 66-72 m at 9 deg (geometric); 12 vertical beams on a cone at 20 m.

Beams on a cone at 20 m: vertical count at 0.1 deg spacing (12 for every row; the 2D simulator models one row).
**Recommended pitch:** 7-8 deg. 8 deg is the steepest that keeps the top beam at the horizontal (far reach beyond the sim's 40 m, limited in reality by
the EMX's rated range on dark targets)
and still clears a 20 deg windshield of any length, but with no margin against an unmeasured steeper glass or mounting
tolerance; 7 deg leaves 1 deg. Final choice after the measurement and the simulated pitch check.

**All results so far use the front-axle mount (1.60 m, 1.10 m, 7 deg).** The final comparison uses the measured roof
mount: it moves the blind area and the cone-invisible zone in GoalFlow's input, so the front-axle numbers do not
transfer as they are. Before any retraining, a roof-mount shift check (current model, `roof` preset, validation seeds
250000+, same protocol as the EMX shift check, `scripts/emx_shift_check.py`); retrain only if it degrades. Queued after
the EMX shift check.

**Measurements needed (Can):** LiDAR optical-centre position (x from the rear axle) and height, and its pitch;
windshield top edge and base (x, height); nose top (x, height); front and rear overhang. At first power-on: check the
point cloud for a fixed cluster on the windshield or hood (self-hits).

**Queued after the full run:** pitch check in simulation: EMX preset, roof mount, pitch 5-9 deg; hits per cone and
bollard at 10-30 m, and whether `cluster_obstacles` returns a circle for each.

### After the full run: pre-registered changes (2026-11-27, before running; dev seeds only)

Full run reviewed: GoalFlow is the safest planner at every level and matches (c) on success at L3-L4 with fewer
collisions; the success gap is stalls. Changes, in this order (lambda3 and the edge critic go with the margin fixes, not
behind them), each tested alone on seeds 0-99 at L0, L2 and L4; **a change is accepted only if collisions do not
rise AND stalls drop (or success rises)** on those seeds and levels; the per-change numbers go in one small table; then
the accepted set combined on seeds 0-499 x L0-L4:
(a) filter false alarms (69 of 99 L4 stalls had a candidate truly clear by >= 0.12 m): circle radius capped at the class radius + an allowance instead of the fitted radius;
clusters enter the filter after 2 scans, except the near-field memory; the 0.12 m clearance is kept.
(b) lambda3 on (door flips at L0-L2 exceed the baseline: 0.30 / 0.25 / 0.27 vs 0.05 / 0.14 / 0.26, per the rule above).
(c) track-edge critic on (exits 11 / 10 at L0 / L1 vs 3 / 1; the smoke-test reason for leaving it off no longer holds).
(d) refinement against the fitted circles for the L0-L1 precision stalls (item 18).
Weights and allowances are fixed in this file before the first of these runs.

### Is GoalFlow's low collision rate the model or the system around it? (2026-11-27)

**Input check (code-traced, `sim/lidar.py`, `sim/raster.py`):** none of the 6 BEV channels uses privileged information.
hits = simulated LiDAR returns (noise, dropout, clutter, occlusion, blind zone, lowest-beam cut-off); free = per-ray
clearing from the same scan; unknown = cells no ray reached; drivable = observed noisy road-edge points within 25 m with
the nominal 12 m width fill (a constant); class = camera label per hit with flips; goal line = the map's goal line
through the believed (drifting) pose. The ego vector (speed, steer, yaw rate, heading and lateral offset relative to the
track) uses the believed pose too. The true scene, teacher routes and true edges appear only as training targets.

**Dev seeds are optimistic.** Seeds 0-499 drove design decisions (safety filter, memory, NMS goals, selection rules),
so every dev-seed result is optimistic. The unbiased number is the fresh test set (seeds 400000+), which stays untouched
until the final configuration is frozen.

**Queued behind the EMX baselines:** (1) filter-off ablation: goalflow_l3_circles with the safety filter off (memory on),
seeds 0-99 at L0 / L2 / L4, vs the full run on the same seeds; per level, the share of scans where the top-ranked
candidate was rejected by the filter and how many of those would truly have collided (needs replays: the full run logs
episodes, not scans). (2) Scene overlap: for each dev scene 0-499, the nearest training scene (100000+) by obstacle
layout; report any near-copy.

### Real-stack note: near-field visibility of the EMX (2026-11-25)

With the sensor at the front axle (1.60 m ahead of the rear axle, 1.10 m high, pitched 7 deg down, lower beam
-12 deg): the 1.5 m blind zone applies to every obstacle and reaches ~0.86 m past the front bumper; in addition,
a 0.40 m cone is invisible within 0.70 / tan 19 deg = 2.03 m of the sensor, because its top is below the
lowest beam (-19 deg absolute). Obstacles near the car therefore need persistence (tracking), not just the
current scan; in simulation this caused the 287 / 451 cone grazes. This bears on the EMX mounting decision and
on the AD-SEM costmap's obstacle persistence (no changes to AD-SEM from here).

### Curvature-feasibility metric (reference values)

`kappa_ok_raw` is the share of raw samples whose discrete curvature (16 waypoints, 0.25 s) stays within
κ_max = tan 10° / 1.60 m at every interior point; `kappa_ok_raw_1.1` allows 1.1 × κ_max. Reference: the
teacher's own driven trajectories (val_0001, 2,573 paths) pass 98.1% at 1.0 × and 100% at 1.1 × (the gap
is discretisation where the teacher's steering saturates at 10°). The metric is very sensitive to
waypoint jitter: a 3 cm lateral wiggle at 0.75 m spacing already implies about κ_max.

### Phase L2 plan (decided 2026-11-24)

* Full training (seed 0, Stage A, Stage B, regression) runs on the Mac (MPS) as soon as the L1
  train split is complete (the P16v is not set up yet). Full training-state checkpoints at least every
  30 min, mid-stage resume, early stopping on the validation split only. The 3-seed runs and
  ablations come later on the P16v.
* **Correction (2026-11-25): the L1 test split is spent.** It was read once after training
  (`reports/lidar_demo/l2_full/test_report.json`, frozen seed-0 checkpoints), and a decision then
  used that result: giving per-route goal selection priority for L3, because test cover@k (0.32 /
  0.34) confirmed the validation finding. Its numbers can no longer count as held-out. The split is
  frozen (kept, never re-read for any choice), and **before L4 a fresh test set is generated with
  new, disjoint scene seeds**; every final number in L4 comes from that set, read once at the end.
  **Fresh test seeds (decided 2026-11-26): 400000+.** 300000+ is the spent L1 test split; training uses
  100000+, validation 200000+, and closed-loop development seeds 0-499.
  Until then all comparisons (selection variants, thresholds, the shadow trigger) use validation only.
* **Training-target mode (checked 2026-11-25).** `SimFrames` is built with `driven_only=False`
  (`train.py:453`), so each time a frame is loaded one teacher mode is drawn uniformly
  (`sim_torch.py:122`), and both the goal target (Stage A, `train.py:95`) and the trajectory target
  (Stage B and regression, `train.py:104`, `108`) come from that same mode. Stage A therefore sees a
  single-mode distance target per step and the mixture only across epochs. The validation loader drew
  modes the same way, and with persistent workers the draw changed between evaluations, so the
  Stage A selection metric (validation `L_goal`) was measured against a different mode on the ~11%
  multi-mode frames at each evaluation (noise in early stopping). Stage B and regression select on
  `ADE_driven` (mode 0), which is unaffected. Planned fix: a mixture distance target over all
  modes (deterministic on validation) and random-mode sampling for Stage B, behind config flags.
* Model selection / early stopping: stage A on validation L_goal; stage B and regression on validation
  ADE of the executed path against the driven teacher mode. (Stage A first used goal error; it was
  switched mid-run on 2026-11-25 at step 8,422 because goal error stopped improving while L_goal,
  DAC accuracy and the g* collision rate kept improving: it penalises choosing the other valid mode.)
* Goal-scorer diagnostics: g* collision rate (Juno box at g* overlapping a true obstacle, from the
  scene index), DAC accuracy on frames with an obstacle within 2 m of the driven path, balanced DAC
  accuracy, and the mean distance of sample endpoints from g* (raw and executed).
* Report order after training: cover@5 / cover@10, **g* collision rate**, mode-mass share, share of
  raw samples within κ_max, ADE / FDE (raw and executed, GoalFlow and regression), training time and
  checkpoint path.
* Open-loop metrics are reported for both the raw path and the executed path (after the pure-pursuit +
  bicycle rollout), for GoalFlow and the regression baseline alike.
