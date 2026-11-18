# Phase 0 — video audit

2026-11-18 · produced by `scripts/audit_videos.py` (+ manual review recorded in `configs/audit_overrides.yaml`)

Outputs (git-ignored): `data/inventory.csv`, `data/audit/<stem>_contact.jpg`, `<stem>_timeline.csv`, `<stem>_obstacles.jpg`.

## Inventory

| File | Class | Container | Res | fps | Duration | Usable (moving, not excluded) | Obstacle passes |
|---|---|---|---|---|---|---|---|
| shell/output.mp4 | C | MP4 / H.264 | 1280×720 | 29.97 | 175.8 s | 175.6 s | 0 |
| zed/morano_video.mp4 | C | MP4 / MPEG-4 | 1280×720 | 5.00 | 101.6 s | 101.0 s | 0 |
| zed/video_cerrina1.webm | C | WebM / VP8 | 1280×720 | 30.00 | 221.5 s | 175.6 s | 0 |

| Class | Files | Total | Usable |
|---|---|---|---|
| A (SVO/SVO2) | 0 | 0 | 0 |
| B (side-by-side stereo) | 0 | 0 | 0 |
| C (monocular onboard) | 3 | 8.3 min | **7.5 min** |
| D (not ego view) | 0 | 0 | 0 |

**Obstacle-course passes: 0.** The HSV hint detector flagged 6 short events in the kart video; all were checked by eye and are
false positives (red/white kerbs, blue fence, blue pit-lane floor, people). No blue bollards or orange cones appear in any file.

Rough sample count at 5 Hz with a full 4 s future: ≈ 2 200 samples (≈ 4 400 with flips), from 3 sessions on 3 different tracks
with 3 different cameras/mounts. A session-level split leaves one session for test.

## Assumptions in PROMPT.md that the data contradicts

1. **No class A or B data.** No SVO/SVO2 and no side-by-side frames (all files are 1280×720, 16:9). Consequences: no ZED
   positional tracking, no IMU, no floor-plane estimate of camera height/pitch, no intrinsics, and the class B/C-vs-A agreement
   check in the quality gate cannot be run. All ego-motion would have to come from monocular ground-plane VO with camera height
   and intrinsics supplied by hand (TODO(Can)).
2. **No obstacle course anywhere.** The project's core question (left/right multimodality around bollards) cannot be trained or
   evaluated on this footage — the obstacle-subset metrics, `ambiguous`-flag precision and pseudo-collision would all be empty.
3. **The "Shell" video is onboard footage** (good, class C), but from a very low wide-angle/fisheye camera that is not the ZED, on
   an open circuit. It is a different camera model and mount from the other two files.
4. **Under the 20-minute threshold** (7.5 min usable) → options below, per the Phase 0 checkpoint.
5. **Hardware:** this workspace is the Jetson AGX Orin (JetPack R36.4.7, 12 cores, ~30 GB RAM, PyTorch 2.7 with CUDA, pyzed 5.3),
   not the Quadro P2000 desktop. Disk is 94 % full (30 GB free). `uv` is not installed; system Python 3.10 was used.

Detector note: the left/right-half correlation alone is not discriminative on real road scenes (mono frames score 0.47–0.91
because sky and road are self-similar), so SBS detection is gated on aspect ratio ≥ 3 (a ZED HD720 SBS frame is 2560×720).

## Options (need Can's decision)

1. **Record class A data (recommended, required for the obstacle question).** SVO2 from Juno's ZED 2i at HD720/15–30 fps with
   IMU, on the obstacle course or a mock-up (bollards + cones on any flat lot). Target ≥ 30 passes, moving the obstacles between
   passes and passing each side; plus ≥ 20 min of open-track driving. SVO gives metric ego-motion, camera height/pitch and
   intrinsics for free. A rosbag with ECU speed/steering recorded alongside would also settle scale and control labels.
2. **Look for originals.** `morano_video.mp4` is 5 fps, which suggests an export from an SVO or rosbag — if the source exists it
   becomes class A. Same question for the 2023 kart recording.
3. **Pretrain on public driving video, fine-tune on ours.** For example comma2k19 (front camera + GNSS/CAN poses) or the nuScenes
   front camera, using only low-speed segments and a distance-matched horizon (≈12 m) to reduce the domain gap. This helps the
   image→ground mapping and goal scorer but cannot supply bollard passes.
4. **Proceed with Phase 1 now regardless.** The toy flow-matching work is data-independent, so it can run while data is collected.

## Open TODO(Can) values

Juno length and wheelbase; ZED mount height and pitch; which camera/vehicle recorded each of the three files (and the
Shell camera's model, for intrinsics); ZED 2i serial number (factory calibration); cone radius; path to the team segmentation
model; whether rosbags with ECU speed/steering exist for any of these runs.
