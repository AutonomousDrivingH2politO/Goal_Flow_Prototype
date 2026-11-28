# GoalFlow for Juno

Prototype of a learned path planner for **Juno**, Team H2politO's Shell Eco-marathon car, in the autonomous
obstacle course. It adapts **GoalFlow** (goal-driven flow matching; Xing et al., 2025,
[arXiv:2503.05689](https://arxiv.org/abs/2503.05689)) to Juno's 10° steering limit and RoboSense EMX LiDAR, and
compares it in closed loop against a classical corridor-lattice planner in a purpose-built 2D simulator.

> **Status (28 Nov 2026):** simulator study complete, confirmed on untouched test seeds. **Recommendation: drive Juno
> with the corridor-lattice planner plus the safety layer developed here, "(a)+filter"** (1 collision in about 2,365
> test runs, no collisions under Juno's real speed control). GoalFlow is as safe but stalls 2-3 times as often and
> fails below its 3 m/s training speed; it stays a research track. Details:
> [`docs/report/report.pdf`](docs/report/report.pdf).

## How it works

```
noisy LiDAR scan → 6-channel bird's-eye view (128×128, 0.2 m)
   → goal scorer over a 1,024-goal vocabulary → 3 goals ≥ 1.5 m apart (NMS-K3)
   → rectified-flow generator: 128 candidate paths (1 Euler step)
   → rollout with pure pursuit + bicycle model (10° steering)
   → safety filter: reject any executed path within 0.12 m of an obstacle circle (current or remembered)
   → critics (goal distance, progress, goal score, track edge) → drive the best, or brake
```

- **Teacher–student:** a privileged corridor-lattice planner that sees the true scene produces the training
  paths; GoalFlow sees only noisy LiDAR.
- **Classical baseline:** the same corridor-lattice planner on the noisy observation (Delaunay doors, door rule,
  A* routes, steerable arc lattice; plan with 0.25 m clearance, accept 0.12 m while following).
- **Noise ladder L0–L4:** from a perfect sensor to very noisy (range noise, dropout, clutter, edge noise, class
  flips, odometry drift proportional to distance).

## Results (fresh test seeds 400000–400499, frozen configs, front-axle mount, 3 m/s)

Success / collision, % of the 473 feasible scenes per level
([`FINAL_COMPARISON.md`](reports/lidar_demo/final_test/FINAL_COMPARISON.md)):

| Level | GoalFlow final | (a)+filter | (a) | (b) | (c) |
|---|---|---|---|---|---|
| L0 | 87.9 / 0.0 | 93.2 / 0.0 | 94.3 / 0.0 | **96.8** / 0.0 | 94.1 / 0.0 |
| L1 | 89.0 / 0.0 | 94.9 / 0.0 | **97.0** / 0.6 | 97.0 / 0.0 | 96.2 / 0.2 |
| L2 | 86.3 / 0.0 | 93.2 / 0.0 | **96.0** / 1.1 | 96.2 / 0.0 | 93.9 / 0.8 |
| L3 | 81.0 / 0.2 | 91.3 / 0.2 | **95.3** / 1.9 | 88.6 / 0.8 | 87.5 / 0.8 |
| L4 | 73.4 / 0.2 | 86.0 / 0.0 | **90.9** / 2.5 | 77.0 / 2.1 | 76.5 / 2.5 |

(a) classical baseline; (b) (a) + tracking, edge memory, hysteresis, hold; (c) (b) without edge memory (same
information as GoalFlow); (a)+filter = (a) + GoalFlow's safety layer.

- GoalFlow is as safe as (a)+filter but stalls 2-3 times as often; it is significantly behind every classical
  opponent (paired McNemar), except (c) at L4.
- The safety comes from the geometric filter and near-field memory, not the network: with the filter off, GoalFlow
  collides in 8-9 of 100 episodes. On the classical planner the same layer brings collisions to about zero.
- Under Juno's real speed control (6-8 km/h valve control, 0.3 s brake delay) (a)+filter has no collisions; GoalFlow
  collapses (the brake delay causes its collisions, the lower speed its stalls), so it would need retraining.
- Planning takes about 152 ms per step on an M1 Mac (classical: 8-42 ms); the on-car target is 50 ms on a Jetson
  AGX Orin.

## Setup

Python 3.10 (`requires-python = ">=3.10,<3.11"`).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch scipy matplotlib        # PyTorch build for your platform (CUDA, MPS or CPU)
```

Code picks `cuda`, then `mps`, then `cpu`. On Apple silicon use fp32 and set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
Multiprocessing uses `spawn` on macOS, so keep the `if __name__ == "__main__":` guards.

Generated data (`data/`), checkpoints (`runs/`) and evaluation logs are git-ignored; regenerate them with the
commands below.

## Common commands

```bash
# Look at a scene (true scene, doors, teacher paths, and what the LiDAR sees)
python scripts/show_sim_scene.py --seed 3 --noise L2 --t 2.0

# Closed-loop evaluation of planners on fixed seeds and noise levels
python scripts/eval_corridor.py --planners observed_stubfix,goalflow_final \
    --seeds 0-499 --levels L0,L1,L2,L3,L4 --workers 7 --out reports/lidar_demo/my_run

# Training data: generate shards, then the scene index and memory-mapped arrays
python scripts/gen_sim_dataset.py --splits val,train --workers 7
python scripts/prepare_training.py --splits val,train --scenes --workers 2
python scripts/prepare_training.py --splits val,train --memmap

# Train (stage A: encoder + goal scorer; stage B: flow generator; plus a regression baseline)
SEED=0 ./scripts/launch_full_training.sh

# Open-loop selection comparison, latency profile, comparison table
python scripts/eval_selection.py --run runs/l2_full_seed0
python scripts/profile_latency.py --planner goalflow_final
python scripts/build_comparison.py reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles

# Tests (the closed-loop regression checks are marked slow)
pytest -m "not slow"
```

Planner names for `--planners`:

| Name | What it is |
|---|---|
| `teacher` | corridor-lattice on the true scene (reference) |
| `observed_stubfix` | classical baseline (a) |
| `observed_l3` | classical baseline (b): (a) + tracking, edge memory, hysteresis, hold |
| `observed_l3_noedge` | classical baseline (c): (b) without edge memory (same information as GoalFlow) |
| `observed_stubfix_filter` | (a) + GoalFlow's safety layer |
| `goalflow_final` | GoalFlow, frozen final configuration (circle filter + near-field memory + edge critic) |

Other options of `eval_corridor.py`: `--mount front_axle|roof` (LiDAR mount preset) and
`--speed-mode constant|juno_bangbang` (Juno's 6–8 km/h valve control).

## Repository map

| Path | Contents |
|---|---|
| `src/gfsem/sim/` | simulator: scenes, LiDAR model and noise, edges, BEV raster, vehicle, closed loop |
| `src/gfsem/planners/` | `corridor_lattice.py` (classical), `goalflow_planner.py`, `safety_wrap.py`, `tracking.py` |
| `src/gfsem/models/` | encoder, goal scorer, flow generator, rollout, `goalflow.py` (planning step and losses) |
| `src/gfsem/data/` | simulator dataset, scene index, recording tools |
| `src/gfsem/train.py` | training stages A, B and the regression baseline |
| `scripts/` | data generation, evaluation, diagnostics, renders, run chains |
| `reports/lidar_demo/` | results, comparison tables and renders (each render folder has a `RULE.txt`) |
| `docs/report/` | full report (`report.tex`, `report.pdf`) |
| `docs/DEVIATIONS.md` | every design decision, pre-registered rule and deviation from the plan |
| `HANDOFF.md` | current state, running jobs and queue |

## Ground rules

- **Seeds:** dev 0–499 · train 100000+ · validation 200000+ · EMX shift check 250000+ · spent test 300000+ ·
  fresh test 400000+.
- Decisions use dev and validation seeds only; selection rules are written into `docs/DEVIATIONS.md` before the
  runs they govern; the test split never drives a decision.
- Demo frames are chosen by a fixed rule, recorded in each folder's `RULE.txt`.
- One evaluation run per output folder (`RUNNING.lock`); keep some CPU idle on shared machines.
- Negative results are reported plainly.

## Credits

GoalFlow: Zebin Xing et al., *GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in
End-to-End Autonomous Driving*, 2025, [arXiv:2503.05689](https://arxiv.org/abs/2503.05689). This repository is an
independent prototype by Team H2politO (Politecnico di Torino), Autonomous Driving division.
