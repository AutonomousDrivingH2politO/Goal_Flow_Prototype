# Claude Code prompt — GoalFlow prototype for AD-SEM

Nov 18, 2026 · @Can

## How to use

This prompt has Claude Code build a GoalFlow-style planner (a goal-point scorer plus a rectified-flow trajectory generator) that learns from footage you already have: the ZED recordings from track tests (SVO, or exported MP4) and the Shell Eco-marathon videos. The model sees image features only. Its training labels come from the ego-motion recovered from the same videos. Export this doc as Markdown, save it as `PROMPT.md` in a fresh repo, and put both PDFs in `docs/`: `GoalFlow.pdf` (Xing et al., arXiv 2503.05689) and `flow_matching_intro.pdf` (Roelants). Put or symlink the videos under `data/raw/zed/` and `data/raw/shell/`. Then tell Claude Code: *"Read PROMPT.md and the papers in docs/, then start Phase 0."*

Before you run it, fill in the values marked **TODO(Can)**: Juno's length and wheelbase, the ZED mount height and pitch (used if they can't be recovered from the SVOs), the path to the team's segmentation model, and whether any rosbags with ECU speed and steering match the recordings. Everything from the next section on is the prompt.

## 1. Role, goal and scope

You are building a research prototype for Team H2politO's autonomous driving division (AD-SEM stack: ROS 2 Humble, ZED 2i camera, Jetson AGX Orin on the car). The question is: **can goal-conditioned flow matching (GoalFlow), trained only on our existing camera footage, predict sensible local trajectories through the Shell Eco-marathon track and obstacle course, and is it fast enough for the Orin?** Read `docs/GoalFlow.pdf` (method, Sections 3.1–3.2.5, Tables 2–4) and `docs/flow_matching_intro.pdf` (rectified flow basics) before writing code.

**The setting.** Juno drives one at a time at 10–12 km/h (about 3 m/s). The obstacle course is about 25 m long. It has blue inflatable cylindrical bollards (about 110 cm tall, 45 cm diameter) and small orange cones, and they move between runs. There is no HD map. A pure-pursuit controller with a 2–3 m lookahead tracks whatever path the planner publishes.

**Data constraint.** For now the only input is video: the team's ZED recordings from track tests and the Shell videos. The model's input is RGB frames only, through image feature extraction. Stereo, IMU and depth inside the SVO files may be used *offline* to compute labels (ego-motion, ground plane), never as model input. No LiDAR, point clouds or simulator.

**Why GoalFlow fits.** The course is multimodal by nature: every bollard can be passed left or right. Plain regression averages the two into a path straight through the bollard. A standalone generative model gives scattered paths. GoalFlow first picks a goal point from a vocabulary, scored on where the expert ended up and on whether a car-shaped box at that goal stays drivable. It then generates trajectories with rectified flow conditioned on that goal, and picks one with a light scorer. The paper reports that one denoising step costs only about 1.6% in score, which is what makes the Orin plausible.

**Expectations.** There are probably at most a few hours of footage, with heavily correlated frames and one driven trajectory per moment. Treat this as a feasibility study with open-loop evaluation. Report data volume and the limits honestly, and prefer frozen pretrained features with small trainable heads.

**In scope.**

- A data audit and ingestion pipeline for the videos.
- Ego-motion labels, plus pseudo-BEV drivable and obstacle masks (labels only).
- Cached image features from a frozen backbone.
- The GoalFlow heads: goal vocabulary and scorer, rectified-flow generator, trajectory scorer with shadow trajectory.
- Open-loop evaluation, baselines, the paper's ablations, and overlay videos.
- Latency measurement and ONNX export.
- An optional, isolated ROS 2 node on the ZED image topic (Phase 5).

**Out of scope.**

- LiDAR, point clouds, depth or stereo as model input, simulators, and NAVSIM/OpenScene.
- Any change to the existing AD-SEM packages, launch files or Nav2 configs.
- Using the learned planner as a safety layer. It proposes paths. Geometric checks stay authoritative, in line with the stack's rule that learned components never gate obstacle detection.

## 2. Data: videos, ego-motion labels and pseudo-labels

The learning signal is imitation: what the car actually did over the next 4 s is the target trajectory. Everything in this section runs offline on the desktop (Quadro P2000, 64 GB RAM).

**Audit first (Phase 0).** Write `scripts/audit_videos.py`. For every file under `data/raw/` it records container (SVO/SVO2/MP4/MKV), resolution, fps, duration, whether the frame is side-by-side stereo, whether IMU and timestamps are embedded, and the camera intrinsics if available. Write the results to `data/inventory.csv`, plus one contact sheet of 12 frames per file. Sort every file into one of these classes:

| Class | Example | Ego-motion source | Use |
| --- | --- | --- | --- |
| A: ZED SVO/SVO2 | track tests | ZED SDK positional tracking (stereo + IMU) | train and test |
| B: ZED MP4, side-by-side stereo | exported track tests | stereo VO with ZED 2i calibration | train and test |
| C: monocular onboard MP4 | Shell runs filmed from the car | ground-plane visual odometry (below) | train and test if it passes the quality gate |
| D: not an ego view | spectator or broadcast footage | none | exclude |

Stop after Phase 0 and report the inventory: hours per class, and how many obstacle-course passes it contains. The rest of the plan depends on it.

**Ego-motion.**

- Class A: open with `pyzed` (`InitParameters.set_from_svo_file`). Turn on positional tracking with IMU fusion and use floor-plane detection to get camera height and pitch. Export, per frame: timestamp, the rectified left image, the pose in a gravity-aligned frame, and the intrinsics.
- Class B: split the frames and run stereo VO using the ZED 2i calibration (from a matching SVO or the factory calibration file, TODO(Can)).
- Class C: Juno's track is flat. Default to ground-plane VO: track features on the road region between consecutive frames, fit a homography, and decompose it with known camera height (TODO(Can), or the median from class A) into metric planar motion (x, y, yaw). Offer SfM or a learned monocular VO only as an alternative, with scale fixed by the same ground plane.
- Quality gate on every segment: speed between 0 and 20 km/h, bounded jerk and yaw-rate spikes, and no tracking-loss flags. Where a run exists as both SVO and MP4, check that class B/C motion agrees with class A within 5% in distance and 3° in heading over 10 s. Drop segments that fail and log why.
- If rosbags with ECU speed and steering exist for the same runs (TODO(Can)), align them by timestamp and use them for ego status and as a check on scale.

**Samples.** Sample at 5 Hz. Each sample has:

- The rectified front image at time t, plus optionally t − 0.5 s and t − 1.0 s.
- Ego status computed from past poses only (speed, yaw rate, acceleration), so no future information leaks in.
- The target: 16 future poses at 0.25 s (a 4 s horizon of about 12 m), interpolated on timestamps, as (x, y, θ) in the ego frame at t (x forward, y left).

Drop samples without a full 4 s future. Keep only 1 in 5 frames where speed is below 0.5 m/s.

**Pseudo-BEV masks (labels and evaluation only, never model input).**

- Drivable area: run the team's segmentation model (TODO(Can): path and checkpoint). If it is unavailable, use a pretrained road segmentation model (for example SegFormer trained on Cityscapes). Also mark the car's own future footprint as drivable, since it actually drove there.
- Obstacles: blue bollards and orange cones by HSV thresholds plus connected components. Take the bottom-centre of each blob as its ground contact point. Add a small detector later only if the thresholds fail.
- Projection: inverse perspective mapping (IPM) with the intrinsics and camera height and pitch. The BEV grid is 0.1 m, x ∈ \[0, 25\] m, y ∈ \[−10, 10\] m. Draw obstacles as discs at true radius (0.225 m bollards, cone TODO(Can)). Accumulate ±1 s of neighbouring frames using the ego poses to fill holes.
- Store a confidence channel. IPM is reliable only on flat ground and degrades with range, so mark cells beyond about 15 m and cells the camera cannot see as unknown.

**Image features (model input).** Use the rectified left image, resized for the backbone. Default backbone: frozen DINOv2 ViT-S/14 at 518 × 294, giving about 37 × 21 patch tokens of size 384. Alternative: an ImageNet ResNet-34. Precompute features once and cache them as fp16 on disk, so training on the P2000 is cheap. Also cache horizontally flipped frames. A flip mirrors y, θ and yaw rate in the labels, which balances left and right passes. Provide an online mode (no cache) for colour-jitter augmentation and for later fine-tuning of the last backbone blocks.

**Splits.** Split by recording session, never by frame: neighbouring frames are nearly identical and would leak. Hold out at least one track-test session and, if one exists, one Shell obstacle-course run as the test set. Store shards as `zarr` or `.npz`, reproducible from `inventory.csv` and a seed.

**Viewer.** Write `scripts/show_sample.py`. It shows the image with the future trajectory projected onto it, the pseudo-BEV with the trajectory and obstacles, and the speed and yaw-rate trace. Check a few dozen samples by eye before training anything.

## 3. Model

PyTorch, with about 5–10 M trainable parameters on top of the frozen backbone. Training runs on a Quadro P2000 (5 GB, Pascal): use fp32 by default, because that chip has no bf16 and fp16 is slow on it. The model must also run on the AGX Orin. Use hidden size D = 256 throughout and make every size a config value.

**Normalisation H.** Map trajectories to roughly \[−1, 1\] with a per-axis affine transform fitted on training-set percentiles (1st and 99th). Store it in the checkpoint. Predict (x, y) for each of the 16 waypoints and derive heading from finite differences.

**Perception.**

- Image tokens: cached DINOv2 patch tokens, projected from 384 to D with a linear layer, plus 2D sinusoidal position encoding, giving F\_img (about 777 tokens). F\_img plays the role of the paper's F\_bev everywhere below.
- Temporal option: average-pool each past frame to 19 × 11 and add a learned frame embedding.
- Ego encoder: an MLP on the ego-status vector, giving F\_ego.
- Auxiliary BEV head (in the spirit of the paper's perception losses, Section 3.2.5): a 50 × 40 grid of learnable queries at 0.5 m, each carrying its sinusoidal (x, y) encoding. They cross-attend to F\_img and predict drivable and obstacle logits. The loss is BCE against the pseudo-BEV, masked by its confidence. This forces the image features to encode ground geometry, and it is one of the costmap outputs in Section 4.

**Goal vocabulary V.** Run k-means on the (x, y, θ) endpoints of all training trajectories (flipped copies included), with θ scaled so that 1 rad counts as 2 m. Default N = 256 because the dataset is small; sweep {128, 256, 1024}. Save it as `vocab.npy` and plot it.

**Goal scorer** (paper Section 3.2.3, Fig. 3a).

- Queries: sinusoidal encoding of each vocabulary point, then an MLP, giving F\_V. Add F\_ego. Run 3 transformer-decoder layers that cross-attend to F\_img. Because the queries carry metric (x, y, θ), cross-attention has to learn the mapping from image to ground.
- Two MLP heads give a distance logit and a DAC logit per point.
- Distance target: softmax over the vocabulary of −‖g\_i − g\_gt‖ / T, with temperature T = 0.5 m as a config value. The paper has no temperature; at metre scale one is needed.
- DAC target: place a car-sized box ("shadow vehicle") at (x\_i, y\_i, θ\_i) in the pseudo-BEV. The target is 1 if the whole box lies in drivable area minus obstacle discs. Mask the loss when the box touches unknown or low-confidence cells. Deviation from the paper: its drivable area ignores obstacles, but here they are subtracted.
- Losses: soft-target cross-entropy for distance, binary cross-entropy for DAC.

```latex
L_{goal} = w_4 L_{dis} + w_5 L_{dac}, \quad w_4 = 1.0,\; w_5 = 0.005 \text{ (paper)}
```

The paper tuned w5 on NAVSIM, so sweep w5 ∈ {0.005, 0.1, 1.0}.

- Selection at test time:

```latex
\hat{\delta}^{final}_i = w_1 \log \hat{\delta}^{dis}_i + w_2 \log \hat{\delta}^{dac}_i, \qquad g^* = \arg\max_i \hat{\delta}^{final}_i
```

**Trajectory generator** (rectified flow, paper Section 3.2.4, Fig. 4).

- Tokens: 16 waypoint tokens (Linear on x\_t per waypoint, plus a learned position embedding); one time token (sinusoidal t, then MLP); one goal token (sinusoidal g, then MLP); and 16 environment tokens. The environment tokens are learnable queries Q that cross-attend to F\_img + F\_ego (paper eq. 7).
- Concatenate all tokens, run 4 transformer-encoder layers (8 heads), and read the velocity v̂ off the waypoint tokens.
- Training:

```latex
x_0 \sim \mathcal{N}(0, \sigma^2 I),\; \sigma = 0.1 \qquad x_t = (1-t)\,x_0 + t\,\tau^{norm} \qquad v_t = \tau^{norm} - x_0 \qquad L_{planner} = \lvert v_t - \hat{v}_\theta(x_t, t, g, F_{env}) \rvert_1
```

- During training the goal is the ground-truth endpoint (teacher forcing, paper Section 4.4).
- Condition dropout: with p = 0.1, replace the goal token with a learned null token. This is what makes the shadow trajectory possible.
- Timestep distribution: uniform or logit-normal (as in SD3, the paper's reference \[9\]). Inference schedule: uniform or the same nonlinear rescaling. Make both switchable.
- Use an L1 loss (paper) with an L2 option (Roelants notebook).
- Training schedule: Stage A trains the projection, auxiliary BEV head and goal scorer. Stage B trains the generator with Stage A frozen, then optionally fine-tunes jointly. Also support a single joint run with a summed loss.

**Inference.**

1. Pick g\*.
2. Draw M = 128 noise samples and integrate with Euler for n ∈ {1, 2, 5, 10, 20} steps, in one batched forward pass per step.
3. Denormalise the results.

**Trajectory scorer and shadow trajectory** (paper Section 3.2.4).

```latex
f(\hat{\tau}_i) = -\lambda_1\, \Phi(f_{dis}(\hat{\tau}_i)) + \lambda_2\, \Phi(f_{pg}(\hat{\tau}_i))
```

- f\_dis is the L2 distance from the trajectory endpoint to g\*. f\_pg is forward progress along ego x, since there is no map direction. Φ is min–max normalisation over the M candidates. Default λ1 = λ2 = 1.
- Shadow trajectory: also generate 16 samples with the goal masked and take the best one by progress. If its mean waypoint distance to the chosen trajectory is above a threshold (default 1.0 m, tuned on validation), output the shadow instead. Log how often this triggers.

**Safety filter (our extension, off by default).** Before scoring, drop any candidate whose swept car box touches obstacle or unknown cells. In evaluation those cells come from the pseudo-BEV; on the car they would come from the perception stack. If no candidate survives, output a braking trajectory. Report results with the filter on and off.

## 4. Readable outputs: control vector and costmap

The model's internal features are not readable as they stand. Expose what it has learned in two forms the stack already understands, and make these the main thing to look at in every evaluation video.

**A. Control vector with a covariance matrix.**

- From the selected trajectory, compute at t + 0.5 s: forward speed v, yaw rate ω, curvature κ, and the steering angle δ = atan(L · κ) from a kinematic bicycle model (L = wheelbase, TODO(Can)). Also compute the pure-pursuit steering for 2.5 m lookahead, the command the car's controller would actually produce.
- Uncertainty matrix: across the M sampled trajectories (and the shadow set), compute the mean and the 3 × 3 covariance Σ of (v, ω, δ), plus the 2 × 2 covariance of the lookahead point (x, y). Report how much of the spread comes from the goal choice (between top-k goals) versus noise (within one goal).
- A bimodal spread, such as a left/right split at a bollard, shows up as a large Σ\_δδ or two clusters. Detect it with a 2-component GMM on δ and flag it as an explicit `ambiguous` boolean rather than averaging it away.
- Flow-field view: for one frame, record v̂ at every Euler step and draw it as quiver arrows on the waypoints, coloured by t. This is the trajectory-space version of the Roelants velocity heatmap. Also plot the 1D version for the lateral offset of the final waypoint (t on the horizontal axis, colour = velocity), matching the notebook's figure.
- ROS shape for later: `geometry_msgs/TwistWithCovarianceStamped` (v, ω and their covariance) plus a small custom message or `std_msgs/Float32MultiArray` for δ, Σ and the `ambiguous` flag.

**B. Learned costmap layers.**

- Goal-score maps (paper Fig. 3b and Fig. 6): evaluate the goal scorer on a dense grid of goal points at 0.25 m (θ set to the direction from the ego), not only on the vocabulary. Render three layers: δ̂\_dac (learned drivability), δ̂\_dis (intent: where the expert would go), and δ̂\_final.
- Auxiliary BEV head: drivable and obstacle probabilities on the 0.5 m grid.
- Sample density: a kernel density estimate of all M trajectories' waypoints on the BEV grid, showing where the generative model puts its mass.
- Convert each layer to a Nav2-style cost in 0–100 (cost = 100 × (1 − p), −1 for unknown) on a 0.1 m grid in the ego frame, matching the live local costmap (0.1 m cells). Write `to_occupancy_grid()` so it later publishes `nav_msgs/OccupancyGrid`. Note that `OccupancyGrid.data` is int8 with −1 meaning unknown, and the stack has already been bitten by an int8/uint8 cast that made free space read as lethal, so add a unit test for this conversion.
- Check the learned layers against the pseudo-BEV: IoU of thresholded δ̂\_dac and the auxiliary drivable mask against pseudo-drivable, and obstacle recall within 0.3 m, both by range bin (0–5, 5–10, 10–15 m).

**Dashboard video.** `scripts/render_eval.py` writes an MP4 per test run. Each frame has four panels:

1. The camera image with all M samples projected (thin), the selected trajectory (thick) and the shadow trajectory (dashed).
2. The BEV with the δ̂\_final costmap, obstacles and g\*.
3. The control readout: v, ω and δ, a Σ ellipse, and the `ambiguous` flag.
4. A strip chart of predicted versus actual steering and speed over the run.

This video is the main artefact Can will review.

## 5. Training and evaluation

**Sanity check before real data.** Reproduce the Roelants 1D bimodal toy with the project's own flow-matching code. Then build a 2D toy: a disc in front of a start point, with expert trajectories passing left or right 50/50. Show that:

- the unconditioned model produces both modes,
- regression produces the average, straight through the disc,
- goal conditioning separates the modes cleanly.

Keep both toys as tests in `tests/`.

**Training.** AdamW, learning rate 3e-4 with cosine decay, weight decay 0.01, batch size 64 on cached features. Early-stop on validation ADE of the selected trajectory. Small data gives high variance, so run every configuration with 3 seeds and report mean ± std. Log to TensorBoard.

**Metrics** (held-out sessions, open-loop). Report each one for all test frames and for the obstacle subset: frames with at least one detected bollard or cone within 15 m.

| Metric | Definition |
| --- | --- |
| ADE / FDE @1, 2, 3, 4 s | selected trajectory against the driven trajectory |
| minADE over M and over the top-5 goals | coverage: does any sample match what the driver did |
| Goal error, top-k hit rate | ‖g\* − g\_gt‖; share of frames with a top-k goal within 1 m |
| Control error @0.5 s | MAE of v, ω and δ from Section 4A against the measured values (ECU if rosbags exist, else derived from ego-motion) |
| Pseudo-DAC, pseudo-collision | share of selected trajectories whose swept box stays in pseudo-drivable; share that hits an obstacle disc (confident cells only) |
| Uncertainty calibration | Spearman correlation of trace(Σ) with ADE; precision of the `ambiguous` flag on frames with a bollard within 8 m ahead |
| Costmap quality | IoU and obstacle recall from Section 4B, by range bin |
| Comfort | peak lateral acceleration and jerk against fixed bounds |
| Latency | per component and end to end, M = 128, n ∈ {1, 5}, on the P2000 and, when available, the Orin (PyTorch and ONNX Runtime; TensorRT is a stretch goal). Target ≤ 100 ms end to end at n = 1 on the Orin. |

**Baselines.** Constant velocity, constant yaw rate, an ego-status MLP, and a deterministic regression head on the same image features (L1 loss, Transfuser-style).

**Ablations, mirroring the paper.**

1. Components (paper Table 2): M0 is rectified flow with no goal, averaging the samples. M1 adds the distance score. M2 adds the DAC score. M3 adds the trajectory scorer and shadow trajectory. Add an oracle run that uses the ground-truth endpoint as the goal (the paper's GoalFlow†) to bound what a better goal scorer could gain.
2. Denoising steps n ∈ {1, 2, 5, 10, 20} against latency (paper Table 3).
3. Noise σ ∈ {0.05, 0.1, 0.2, 0.3} (paper Table 4).
4. Ours: vocabulary size, w5, temperature T, DINOv2 versus ResNet-34, temporal frames on or off, auxiliary BEV head on or off, flip augmentation on or off, safety filter on or off.

**Report.** `scripts/make_report.py` writes `reports/results.md`: every table above (mean ± std), the toy plots, goal-score and costmap figures, and links to the dashboard videos. End it with a limitations section: open-loop only, noise in the pseudo-labels, the amount of data, correlated test frames, and how much the class C ego-motion can be trusted.

## 6. Repo, phases and rules

**Layout.**

```
goalflow-sem/
  PROMPT.md  docs/  configs/            # YAML + dataclasses, one file per experiment
  src/gfsem/
    data/     audit.py zed_svo.py stereo_vo.py ground_vo.py quality.py
              samples.py pseudo_bev.py features.py dataset.py
    models/   encoders.py aux_bev.py goal_scorer.py flow_generator.py traj_scorer.py baselines.py
    outputs/  control.py costmap.py      # Section 4
    train.py  eval.py  export_onnx.py
  scripts/    audit_videos.py show_sample.py render_eval.py make_report.py bench_latency.py
  tests/      toys, normalisation round-trip, ego-frame transforms, IPM, OccupancyGrid conversion, ONNX parity
  ros2/gfsem_ros/                        # Phase 5 only
  reports/  data/ (git-ignored)
```

**Phases.** Stop at each checkpoint, summarise, and wait for Can's go-ahead before continuing.

| Phase | Work | Checkpoint: done when |
| --- | --- | --- |
| 0 | Video audit | `inventory.csv` and contact sheets exist; you report hours per class and the number of obstacle passes. If under about 20 min of usable ego footage, propose options (more recordings, pretraining on public driving video) before going on. |
| 1 | Flow-matching core on the toys | Both toy tests pass; the plots show regression averaging and goal conditioning separating the modes. |
| 2 | Data pipeline | Ego-motion passes the quality gate on class A; the class B/C agreement check is reported; features are cached; Can has reviewed 30 samples from `show_sample.py`. |
| 3 | Model and training | Baselines and M0–M3 train on 3 seeds; validation curves are logged. |
| 4 | Evaluation and outputs | `results.md` is complete; dashboard videos exist for every test run; control and costmap outputs are tested; latency is benchmarked; the ONNX export matches PyTorch within 1e-3. |
| 5 (optional) | ROS 2 node | Package `gfsem_ros` subscribes to the ZED left image and odometry. It publishes `nav_msgs/Path`, `TwistWithCovarianceStamped` and `OccupancyGrid` layers under `/gfsem/*` only, never wired to the controller. It is tested by replaying a rosbag on the Orin. |

**Rules.**

- Python 3.10 (matching ROS 2 Humble) and `uv` for environments. Keep model code ONNX-friendly (opset 17, no custom ops). On the Orin, use the PyTorch build that JetPack provides.
- Never modify the AD-SEM repository, its launch files or its Nav2 configs. Never commit raw videos or caches. Never delete or overwrite anything in `data/raw/`.
- Make everything reproducible from a config and a seed. Use type hints, pytest, and one commit per finished step on a feature branch.
- Record every departure from the paper in `docs/DEVIATIONS.md`, with the reason: image tokens instead of a BEV fusion backbone, obstacles inside the DAC area, pseudo-labels, the distance temperature, vocabulary size, the auxiliary BEV head, and the safety filter.
- Mark unknown vehicle or camera values as `TODO(Can)` in the config and ask, rather than guessing silently.
- When data contradicts an assumption in this prompt (for example, the Shell videos turn out not to be onboard footage), stop and report it. Do not work around it.
- Report negative results plainly. If GoalFlow does not beat the regression baseline on the obstacle subset, that is a finding.
