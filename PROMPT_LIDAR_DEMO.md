# Claude Code prompt — GoalFlow + LiDAR demo in simulation

Nov 24, 2026 · @Can

## How to use

This is a second prompt for the existing `Goal_Flow_Prototype` repo. It builds a LiDAR-input GoalFlow planner in a fast 2D simulator of the Shell Eco-marathon obstacle course, and races it against the team's corridor planner. The aim is a demo that shows potential before CARLA and before the next SEM. Save it as `PROMPT_LIDAR_DEMO.md` in the repo root and start a new branch `feature/lidar-sim-demo` off the current one. Then tell Claude Code: *"Read PROMPT\_LIDAR\_DEMO.md, PROMPT.md and docs/, then start Phase L0."*

The video pipeline from `PROMPT.md` is paused, not deleted. The iPhone track-walk clips stay reserved for perception tests.

Vehicle values come from the Juno data sheet, which is final (it comes from Juno's designers and the strategy team): wheelbase 1.60 m, length 2.875 m, width 1.165 m, track 1.00 m, tyre radius 0.279 m, turning radius 5.5 m. `configs/vehicle.yaml` currently has wrong values (1.65 / 2.55 / 1.85 m). The first step of Phase L0 is to correct it.

Hardware: the Orin for runtime and latency checks, the desktop (Quadro P2000, 5 GB) for quick runs, and the team's new Lenovo ThinkPad P16v Gen 2 for training (RTX 2000 Ada laptop GPU with 8 GB, 64 GB RAM, 2 TB SSD). The P16v ships with Windows 11 Pro, so run the Python stack under WSL2 with CUDA, or on a native Ubuntu install.

Development can also run on an M1 MacBook Pro (16 GB), which is the plan for Phase L0–L1. Keep all code device-agnostic: pick `cuda`, then `mps`, then `cpu` automatically, use fp32 on MPS, and set `PYTORCH_ENABLE_MPS_FALLBACK=1`. macOS starts worker processes with `spawn`, so every multiprocessing entry point needs an `if __name__ == "__main__":` guard and picklable arguments. Latency numbers count only when measured on the Orin.

## 1. Goal and what "shows potential" means

You are building a demo for Team H2politO's autonomous driving division. **The question: given the same LiDAR observations, can a GoalFlow planner (goal scorer plus rectified flow) drive Juno through the SEM obstacle course as well as or better than the team's Delaunay corridor planner, fast enough for the AGX Orin?** Read `docs/GoalFlow.pdf` (Sections 3.1–3.2.5, Tables 2–4) and `docs/flow_matching_intro.pdf` first.

**Why it could win.** The corridor planner is hand-written logic over detected bollard centres. When it gets noisy, partial or occluded detections, it can choose a door that isn't there or flip between doors. GoalFlow can be trained by a *privileged teacher*: the corridor planner running on the true, complete scene. The student sees only what the LiDAR sees. So the student can learn to handle noise and occlusion better than the same corridor planner running on the same noisy input. That is the honest win condition. On perfect input the student will at best match its teacher, and the report must say so.

**Success criteria, fixed before any training.** The demo shows potential if, on 1,000 held-out closed-loop scenes with realistic noise:

1. GoalFlow's success rate (goal line reached, no collision, no track exit) is at least that of both baselines (noisy corridor and MPPI-style), and higher at the top two noise levels.
2. Its collision rate is at most either baseline's.
3. It visibly keeps both left and right options. Where a deterministic regression planner on the same input collides by averaging the two, GoalFlow does not: a figure plus a collision count.
4. It flips between doors no more often than the corridor planner (left/right switches per run).
5. End-to-end planning takes ≤ 50 ms on the Orin at one denoising step with 128 samples.

If any criterion fails, report it plainly. A clear negative result is also a deliverable.

## 2. Simulator

Write a pure-Python 2D simulator (`numpy`, `shapely`, `scipy.spatial`) under `src/gfsem/sim/`. It should run at least 50 closed-loop scenes per minute on one CPU core, so that thousands of comparisons are cheap. Every observation and every plan is in the ego frame (x forward, y left).

**Scene generator.** Put all values in one dataclass config.

| Parameter | Default | Note |
| --- | --- | --- |
| Track centreline | straight or constant curvature, \|κ\| ≤ 1/30 m⁻¹ | sample per scene |
| Track width | 12 m (the SEM track) | confirmed by Can; keep it a config value |
| Obstacle field | 25 m long, starting 8–15 m ahead of the spawn |  |
| Bollards | 6–14, radius 0.225 m, height 1.10 m | minimum centre spacing 1.0 m |
| Cones | 0–6, radius 0.25 m, height 0.40 m | values from `configs/vehicle.yaml` |
| Juno | wheelbase 1.60 m, length 2.875 m, width 1.165 m | 2023 data sheet; steering hard-limited to 10° in the AD-SEM stack, so the minimum turning radius is about 9.1 m |
| Door rule | gap − 2r ≥ width + 2 · 0.25 m | about 2.12 m between bollard centres at width 1.165 m |
| Speed | 3.0 m/s target | the planner may command braking |

Keep about 5% of scenes with no feasible route, to test braking. "Feasible" means drivable within the 10° steering limit, not just geometrically open.

**LiDAR model (RoboSense EMX at 1.10 m, pitched 7° down).**

- Horizontal FOV 120°, ray-cast at 0.2° steps; the real sensor is finer, so make the step a config value. Vertical span −12° to +8° around the pitched axis. Blind zone 1.5 m, maximum range 40 m (config value).
- At this mount height the sensor is level with the bollard tops, so occlusion is 2D. Ray-cast against obstacle circles in 2D. An obstacle is visible at distance d only if some beam elevation intersects its height band \[0, h\]. Cones closer than about 2 m fall under the beams; compute this from geometry rather than hard-coding it.
- Output a point set: the visible *front arc* of each obstacle, not a full disc, just as the real costmap sees it. Add per-point range noise (σ = 3 cm), angular jitter, random point dropout, and false-positive clutter near the track edges (grass, kerbs).
- Road edges: points along both edges out to 25 m with σ = 10 cm, plus occasional 1–2 m dropouts. This stands in for the agreed fusion of segmentation and LiDAR ground intensity.
- Camera labels: each detected cluster carries a class (bollard or cone), wrong with probability p\_mis = 0.05. Detection never depends on the class, matching the stack's rule.

**Noise levels.** Define five presets, L0 (clean) to L4 (harsh). Each raises range noise, dropout, clutter, edge noise, p\_mis, and the localisation drift of the odom frame between replans. Training draws a mix of levels; evaluation reports every level separately.

**BEV raster (input to the learned planner).** 128 × 128 cells at 0.2 m, covering x ∈ \[−3.2, 22.4\] m and y ∈ \[−12.8, 12.8\] m. Channels, float32 in \[0, 1\]:

1. LiDAR hits, binned, with no inflation.
2. Free space: cells crossed by a ray before its hit (the Nav2 obstacle layer's raytrace clearing).
3. Unknown: cells no ray reached, including occlusion shadows (the layer's `track_unknown_space`).
4. Drivable area from the observed road edges.
5. Class hint: 1 for bollard, 0.5 for cone, on the hit cells.
6. Goal-line hint: the line across the track at the far end of the field, or at the raster edge if farther.

These channels mirror the team's live Nav2 costmap on purpose. Write the rasteriser against a small `Observation` dataclass (points, edge points, labels, ego pose), so a CARLA LiDAR or a real costmap can feed it later.

**Ego status.** Speed, steering angle, yaw rate, heading relative to the track, and lateral offset from the centreline.

**Corridor planner** (`src/gfsem/planners/corridor.py`). Reimplement the agreed design:

1. Delaunay triangulation over obstacle centres plus track-edge points sampled every 1 m.
2. Each edge is a door or a wall, by the rule above. Shadows behind obstacles are worst-case unknown, so doors there are only tentative.
3. A\* over triangles to the goal line, with a penalty that grows as doors narrow.
4. Commit only to the next door.
5. Line-of-sight smoothing, then a curvature check. The path must stay within |κ| ≤ tan(10°) / 1.60 ≈ 0.11 m⁻¹ (a 9.1 m minimum radius), with a margin. A door that needs a sharper turn counts as blocked.
6. Brake when there is no route.

It runs in two modes. *Privileged* uses the true scene and is the teacher. *Observed* gets obstacle centres from the noisy point clusters (Euclidean clustering, then a circle fit with the known radius for the cluster's class) and is the baseline to beat.

**Closed loop.** A kinematic bicycle model with the 1.60 m wheelbase and the steering limit. A pure-pursuit controller with a 2.5 m lookahead tracks the planned path. Replan at 10 Hz. Every planner gets exactly the same seeds, noise draws and controller.

**Training data.**

- Run the privileged teacher closed-loop through each scene and sample frames at 5 Hz. Each frame stores the *observed* BEV and ego status, plus the teacher's future trajectory: 16 poses at 0.25 s, a 4 s horizon of about 12 m.
- Multimodality: replan the teacher K = 8 times with ±15% random edge-cost noise and a random left/right tie-break bias. Keep the distinct homotopy classes (which side of each obstacle the path passes), and sample one per frame.
- Recovery states: start about 30% of frames from perturbed poses (lateral offset ±1.0 m, heading ±15°).
- Start with 15k scenes for training and 1.5k each for validation and test, split by seed. Shard to `.npz`, deterministic and parallel.

**Storage budget: 20 GB for the whole dataset.** At float32, 1M frames of 6 × 128 × 128 BEV would take about 390 GB, so don't store rasters. Store the compact `Observation` per frame (points, edge points, labels, ego state) plus the targets, and rasterise on the fly in the DataLoader. If that is too slow, cache uint8, compressed rasters instead. Report the measured size per frame in Phase L1.

**Viewer.** `scripts/show_sim_scene.py` shows the true scene with doors, walls and the K teacher paths, next to the six BEV channels. It is the first thing Can checks.

## 3. Model (BEV GoalFlow)

PyTorch, about 5–10 M parameters, hidden size D = 256, every size a config value. Train on the P16v (RTX 2000 Ada, AMP) by default; the Orin (AMP fp16) and the desktop P2000 (fp32 only) also work. Put it in `src/gfsem/models/`, beside the existing data code.

**Normalisation H.** A per-axis affine transform to roughly \[−1, 1\], fitted on training percentiles (1st and 99th) and stored in the checkpoint. Predict (x, y) for each of the 16 waypoints and derive heading from finite differences.

**Perception.**

- BEV encoder: a small ResNet-style CNN (6 input channels) giving a 16 × 16 × D map. Add 2D sinusoidal position encoding and flatten to 256 tokens, F\_bev.
- Ego encoder: an MLP on the ego-status vector, giving F\_ego.
- Optional auxiliary head: predict the *true* obstacle discs and drivable area from F\_bev (BCE). This teaches the encoder to fill in occluded and noisy regions.

**Goal vocabulary V.** Run k-means on the (x, y, θ) endpoints of the teacher trajectories, with θ scaled so that 1 rad counts as 2 m. Default N = 1024; sweep {256, 1024, 4096}.

**Goal scorer** (paper Section 3.2.3).

- Queries: sinusoidal encoding of each vocabulary point, then an MLP, plus F\_ego. Run 3 transformer-decoder layers that cross-attend to F\_bev, then two MLP heads.
- Distance target: softmax over the vocabulary of −‖g\_i − g\_gt‖ / T, with T = 0.5 m.
- DAC target: 1 if a Juno-sized box at (x\_i, y\_i, θ\_i) lies entirely inside the track polygon minus the *true* obstacle discs. Test the full polygon, not just the corners. Deviation from the paper, which ignores obstacles.
- Loss: L\_goal = w4 · L\_dis + w5 · L\_dac, with w4 = 1.0 and w5 = 0.005 (paper); sweep w5 ∈ {0.005, 0.1, 1.0}.
- Selection: g\* = argmax over i of w1 · log δ̂\_dis + w2 · log δ̂\_dac.

**Trajectory generator** (rectified flow, paper Section 3.2.4).

- Tokens: 16 waypoint tokens (a linear layer on x\_t plus a learned position embedding), one time token, one goal token, and 16 environment queries that cross-attend to F\_bev + F\_ego. Run 4 transformer-encoder layers and read the velocity v̂ off the waypoint tokens.
- Training: x0 \~ N(0, σ²I) with σ = 0.1; x\_t = (1 − t)·x0 + t·τ\_norm; target v = τ\_norm − x0; L1 loss. During training the goal is the ground-truth endpoint.
- Condition dropout: with p = 0.1, replace the goal token with a learned null token, which enables the shadow trajectory. Make the timestep distribution switchable between uniform and logit-normal.
- Schedule: Stage A trains the encoder and goal scorer. Stage B trains the generator with Stage A frozen, followed by an optional joint fine-tune.

**Inference.**

1. Pick g\*.
2. Draw M = 128 noise samples and integrate with Euler for n ∈ {1, 2, 5, 10} steps, batched.
3. Score each candidate: f = −λ1 · Φ(endpoint distance to g\*) + λ2 · Φ(progress along the track) − λ3 · Φ(distance to the previously chosen path). Φ is min–max normalisation over the candidates.
4. Shadow trajectory: generate 16 samples with the goal masked. If the best of them differs from the chosen path by more than a threshold (tuned on validation), output it instead.

The λ3 term is our addition. It plays the role of the corridor planner's "commit to the next door" rule. Report results with λ3 = 0 and with λ3 tuned.

**Critic-based selection (MPPI-style; the team's core idea).** The flow model proposes candidate paths and a set of critics chooses among them, as in Nav2's MPPI controller. Implement `src/gfsem/planners/critics.py`. Each critic scores every candidate along its swept Juno footprint:

- Obstacle: costmap cost along the footprint, plus a clearance term. Lethal or unknown cells cause a hard reject (this is the safety filter).
- Goal progress: distance gained toward the goal line.
- Goal score: GoalFlow's own δ̂\_final at the path's endpoint.
- Constraints: a heavy penalty for steering beyond 10° or lateral acceleration beyond a configurable limit.
- Smoothness: change in curvature, and jerk.
- Consistency: distance to the previously chosen path (the λ3 term above).
- Track edge: distance to the road edges.

Keep the weights in YAML. Start from the MPPI critic weights in the AD-SEM Nav2 config used at SEM 2026 where they map, and document the mapping.

Selection rules, switchable:

1. Lowest total cost.
2. MPPI-style softmax weighting with a temperature, applied only *within the cluster (homotopy class) of the best candidate*. Averaging across clusters would bring back the mode-averaging failure.
3. The paper's scorer alone.

Stretch goal: cost guidance during sampling. At each Euler step, add −∇cost to the predicted velocity, so samples are pushed away from obstacles before the critics see them.

**Safety filter (our extension).** Drop any candidate whose swept Juno box touches LiDAR hit cells or unknown cells; if none survive, brake. Report every closed-loop result with the filter on and off. Off shows what the model learned. On is how it would actually run on the car.

**Readable outputs.** Reuse the earlier spec in `PROMPT.md` Section 4:

- A control vector (v, ω, steering angle δ) with its 3 × 3 covariance across the samples, plus a left/right `ambiguous` flag.
- Goal-score maps (δ̂\_dac, δ̂\_dis, δ̂\_final) and the density of the samples, exported as 0–100 costmap layers. Include the int8 `OccupancyGrid` conversion test.

## 4. Experiments and demo artefacts

**Sanity first.** Add the 2D left/right toy from `PROMPT.md` Section 5 as a test if it isn't there yet: a disc in front of a start point, with paths passing it left or right 50/50. Regression must average through the disc; goal-conditioned flow must separate the modes.

**Planners compared.** All of them run closed-loop on the same 1,000 held-out scenes at every noise level, with the same controller.

| Planner | Input | Role |
| --- | --- | --- |
| Corridor, privileged | true scene | ceiling (the teacher) |
| Corridor, observed | noisy LiDAR clusters | **the baseline to beat** (what we would field) |
| Regression | same BEV as GoalFlow | shows mode averaging |
| M0: rectified flow, no goal, mean of samples | same BEV | paper Table 2 |
| M1: + distance score | same BEV | paper Table 2 |
| M2: + DAC score | same BEV | paper Table 2 |
| M3: + scorer and shadow trajectory (full GoalFlow) | same BEV | the candidate |
| M3 + λ3 commit, safety filter on and off | same BEV | the deployable variant |

**Two more planners.**

1. **MPPI-style baseline, no learning:** the same critics and selection rule, but the candidates come from 128 Gaussian perturbations of the previous steering sequence, rolled out through the bicycle model. This is closest to what the team drove at SEM 2026.
2. **GoalFlow + critics:** flow samples, scored by the same critics.

Comparing these two with identical critics answers the cleanest question in the study: does a learned proposal distribution beat Gaussian sampling? Also ablate the selection rule: the paper's scorer, the critics, or both.

**Closed-loop metrics, per noise level.**

- Success rate.
- Collision rate, and track-exit rate.
- Minimum clearance to any obstacle.
- Door flips per run (left/right switches).
- Peak lateral acceleration and jerk.
- Time to the goal line.
- Correct braking on the no-route scenes.

Also report open-loop ADE/FDE and minADE over M against the teacher.

**Ablations.** Denoising steps n ∈ {1, 2, 5, 10} against latency (paper Table 3). σ ∈ {0.05, 0.1, 0.2} (paper Table 4). Vocabulary size, w5, λ3, the auxiliary head on or off, and the noise mix used in training. Run 3 seeds for every configuration and report mean ± std.

**Latency.** Per component and end to end, on the Orin (PyTorch fp16 and ONNX Runtime; TensorRT is a stretch goal) and on the desktop. For comparison, time the observed corridor planner end to end, including clustering and circle fits.

**Demo artefacts** (`reports/lidar_demo/`):

1. **Side-by-side video, 60–90 s.** The same scenes driven by corridor-observed (left) and GoalFlow (right) at noise L3. Each panel shows the 128 samples, the chosen path, the δ̂\_final goal heatmap and the LiDAR hits. Pick scenes by a fixed rule (seeds 0–19, for example), not by hand, so the video is not cherry-picked.
2. **Mode-averaging figure.** For one two-way scene: regression goes through the bollard; GoalFlow shows two clusters and picks one.
3. **Results table.** The success criteria from Section 1, each marked pass or fail, at every noise level.
4. **A one-page `summary.md`** for the team: what was shown, what wasn't (simulation only, the teacher is our own planner, no real sensor data yet), and the next step, which is CARLA.

## 5. CARLA readiness, phases and rules

**Getting ready for CARLA without installing it.** Write `src/gfsem/sim/carla_adapter.py` as an interface stub. It turns a CARLA LiDAR point cloud plus the ego transform into the same `Observation` dataclass. Document the mapping from the EMX to CARLA's `sensor.lidar.ray_cast` attributes: `horizontal_fov` 120, `upper_fov`/`lower_fov` from the pitched −12°/+8° span, `channels`, `range`, `points_per_second`, `rotation_frequency`. Note that `sensor.lidar.ray_cast_semantic` gives obstacle labels for free. CARLA has no SEM bollards, so they will need custom static props (blue cylinders, 0.45 m × 1.10 m). CARLA's current docs ask for at least an 8 GB, RTX 2070-class GPU ([source](https://carla.readthedocs.io/en/latest/getting_started/)), and the desktop's P2000 has 5 GB. The new P16v (RTX 2000 Ada laptop GPU, 8 GB) is borderline and could only try low-quality settings. The first CARLA session therefore runs on Bylogix's machine (see \`PROMPT\_CARLA.md\`). Keep the 2D simulator as the fast benchmark either way.

**Phases.** Stop at each checkpoint, summarise, and wait for Can's go-ahead.

| Phase | Work | Checkpoint: done when |
| --- | --- | --- |
| L0 | Simulator, LiDAR model, corridor planner (both modes), closed loop, viewer | Privileged corridor succeeds on ≥ 98% of feasible L0 scenes. Observed corridor degrades visibly from L0 to L4. Can has reviewed 20 scenes in the viewer. |
| L1 | Dataset generation, toy test | Shards are reproducible from a seed. The homotopy-class counts per scene are logged. The toy test passes. |
| L2 | Regression, M0–M3 | All train on 3 seeds; validation curves are logged. |
| L3 | Closed-loop evaluation, ablations, latency | Full tables at all noise levels; Orin latency measured; the ONNX export matches PyTorch within 1e-3. |
| L4 | Demo artefacts | Video, figure, table and `summary.md` exist; each success criterion is marked pass or fail. |
| L5 (optional) | CARLA adapter stub | The interface and the attribute mapping are documented, with a unit test on a synthetic point cloud. |

**Rules.**

- Leave the video pipeline in `PROMPT.md` and its code untouched. Reuse `configs/vehicle.yaml`, after correcting it to the Juno data sheet values above. Those values are final.
- The Orin disk is 94% full. Put datasets and checkpoints under a configurable `GFSEM_DATA` path (an external SSD) and keep them git-ignored.
- The repo is public. Commit no raw recording names, personal names or images of people.
- Python 3.10. Code must be ONNX-friendly (opset 17, no custom ops). Use type hints and pytest, make everything reproducible from a config and a seed, and commit once per finished step.
- Record every departure from the paper in `docs/DEVIATIONS.md`, with the reason: BEV input instead of camera + LiDAR fusion, obstacles in the DAC area, the privileged teacher, λ3, the safety filter, the distance temperature.
- Never tune on the test seeds. Pick demo scenes by a fixed rule.
- Report negative results plainly. If GoalFlow does not beat the observed corridor planner, say so, and say at which noise levels.
