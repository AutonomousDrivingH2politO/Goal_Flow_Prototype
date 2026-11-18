# Phase 0 — data audit (revision 2)

2026-11-18 · `scripts/audit_videos.py`, `scripts/count_obstacle_passes.py`, manual review in `configs/audit_overrides.yaml`

Revision 1 covered only the three videos in `data/raw/{zed,shell}` (7.5 min, no obstacles). After Can confirmed rosbags exist,
the whole disk was searched: 7 unique ZED SVO2 files and 66 ROS 2 bags were found and symlinked (read-only) into
`data/raw/svo/` and `data/raw/bags/`. Outputs (git-ignored): `data/inventory.csv`, `data/audit/*_contact.jpg`,
`*_timeline.csv`, `obstacle_events.csv`, `events/*.jpg`.

## Values from Can (in `configs/vehicle.yaml`)

Wheelbase 1.65 m; ZED 2i (S/N 39159302) at 1.25 m, pitch 0°; cone r = 0.25 m, h = 0.40 m; all footage from the team ZED 2i;
segmentation = TwinLiteNetPlus (`nano/small/medium/large.pth` present). HD720 rectified intrinsics read from the SVO2 headers:
fx = fy = 533.7, cx = 638.4, cy = 362.1 (≈100° HFOV); baseline 0.120 m. `/ECU/speed` is km/h (verified: bag "saw_15kmh_max" peaks at 15.08).

## New classes for bags

| Class | Meaning | Ego-motion source |
|---|---|---|
| R1 | ZED rectified RGB + `/zed/zed_node/odom` (+ IMU, ECU speed) | ZED stereo-inertial odometry, as class A |
| R2 | ZED rectified RGB + `/ECU/speed`, no odometry/IMU/tf | ECU speed for scale + visual rotation for yaw |
| X | camera stream < 2 Hz or absent | none (excluded) |

## Inventory (usable = moving > 1.8 km/h with images present, minus excluded ranges)

| Domain | Class | Files | Total | Usable |
|---|---|---|---|---|
| **track** | A (SVO2) | 1 | 0.6 min | 0.4 min |
| **track** | R1 | 13 | 14.4 min | 7.5 min |
| **track** | R2 | 12 | 22.9 min | 15.3 min |
| **track** | C (mono video) | 4 | 8.6 min | 7.8 min |
| urban (campus roads) | A + R1 | 19 | 13.7 min | 5.7 min |
| handheld (bike path SVO2) | D | 4 | 6.7 min | 0 |
| lab / static | D | 3 | 1.9 min | 0 |
| unusable bags | X / ERR | 19 | 37.1 min | 0 |
| duplicate copies | dup | 3 | 9.8 min | 0 |

**Track total: ≈ 31 min usable, of which ≈ 23 min has metric ego-motion (A/R1/R2).** Urban adds 5.7 min. This is above the
20-minute threshold, but it is spread over ~25 short sessions (median ≈ 40 s) on 4 sites (Morano, Lombardone, the kart circuit,
the Shell circuit).

## Obstacles

* **There are no blue bollards or orange cones in any recording.** All obstacles are **people standing on the runway**
  (Morano 21 May, 26 May, 5 June 2026; Lombardone 2025 has people at the track edge). Every HSV bollard/cone hit was checked by
  eye: blue jeans, kerbs, blue fencing, pit-lane floor.
* YOLOv8m person events in path (2–20 m ahead, |y| ≤ 3 m), track recordings only: **26 events — 13 stopped in front,
  5 passed (2 left, 3 right), 8 unresolved.** Bag names suggest a few more avoidances inside long "stopped" events
  (e.g. `long_run_with_obs_avoided`, `obs_avoided_very_nays`). Realistic estimate: **about 5–10 avoidance passes and
  about 13 stop events**, to be counted exactly from lateral ego-motion in Phase 2.

## Assumptions in PROMPT.md that the data contradicts

1. **Obstacles are people, not bollards/cones.** HSV obstacle pseudo-labels won't work; obstacles must come from an offline
   person detector (labels only). The obstacle disc radius should be about 0.3 m for a person. Many runs are *stop* tests
   (the car brakes in front of the person), not pass-left/pass-right, so the multimodal "pass on either side" case is rare.
2. **No measured steering.** Only commands exist: `/commands/KalmanAngle` (R2 bags, ~1.5 Hz) and `/cmd_vel`. The "control error"
   metric must use steering derived from ego-motion.
3. **Scale disagreement.** Where both exist, ZED odometry distance is 10–60 % *below* integrated ECU speed (e.g. 61 vs 111 m,
   48 vs 119 m in `turning_with_pushing`), and drifts badly in one near-static bag (166 m vs 6 m). Either the ECU wheel-speed
   calibration is off or ZED tracking drops out. This must be settled before trusting either for R2 labels.
4. **Low camera rate in most R1 bags.** 3–6 Hz (the 26 May and 5 June sessions), so 5 Hz samples there fall back to image
   timestamps. R2 bags are 13–15 Hz.
5. **The best obstacle session (Morano 21 May) is R2.** No odometry/IMU, so ego-motion relies on ECU speed + visual yaw. The
   R1 bags (with both ZED odometry and ECU speed) can validate that method with the 5 % / 3° check in place of class A.
6. **Over-exposed (white) frames** at the start of many bags (auto-exposure). The quality gate needs an exposure check.
7. **`shell/output.mp4` looks unrectified** (curved horizon), so it needs rectifying with the factory calibration of
   S/N 39159302. `morano_video.mp4` is *not* the 2025-11-18 SVO2 (frames don't match).
8. **Hardware.** Everything runs on this AGX Orin (JetPack R36.4.7, PyTorch 2.7 + CUDA, pyzed 5.3, ROS Humble). Disk is
   94 % full (30 GB free). `uv` is not installed, so system Python 3.10 was used.

## Proposed plan (needs Can's go-ahead)

* **Training set:** track A/R1/R2 (≈ 23 min) as the core. Track C videos (≈ 8 min) only if they pass the quality gate.
  Urban (≈ 6 min) as optional extra/pretraining data.
* **Test set:** hold out the Morano 21 May session (the most avoidance passes), or one 21 May bag plus one 5 June bag.
* **Obstacles:** person detector → ground contact point → BEV discs, replacing the HSV path. Keep HSV for future cone/bollard runs.
* **Still worth recording:** a dedicated session with real bollards/cones, passing each side about 15 times, recorded as SVO2
  (or a bag with ZED odometry) at ≥ 10 Hz camera rate, to answer the original multimodality question directly.

## Still open (TODO(Can))

Juno **length and width** (for the shadow-vehicle box); whether people are the intended obstacle class for this study or a
bollard/cone session is planned; which of ZED odometry or ECU speed is trusted for scale (has the ECU wheel circumference
been calibrated?).
