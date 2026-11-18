# Feature-extraction feasibility (DINOv2 ViT-S/14, frozen)

2026-11-18 · `scripts/feature_feasibility.py --sessions 2026-06-05T142247 2026-05-26T125445 2025-05-28T111357 2026-05-21T170951 --holdout 2026-05-21T170951 --seeds 1`
Raw numbers: `reports/feature_feasibility/summary.json`; examples: `reports/feature_feasibility/probe_examples.jpg` (not in git, shows people).

**Answer: yes.** A frozen DINOv2 ViT-S/14 runs on the Orin well inside budget. Its patch tokens linearly encode the
drivable area on an unseen session much better than colour + position does, and the full cache fits on disk.

## Setup
- 4 track sessions at 1 Hz, over-exposed frames dropped (7). Train: Morano 5 Jun (105 frames), Morano 26 May (45),
  Lombardone 28 May 2025 (270). **Held out:** Morano 21 May `long_run_with_obs_avoided` (160 frames).
- Input 518×294 → 21×37 tokens × 384. Target: TwinLiteNetPlus (large) drivable mask averaged per 14×14 patch.
- Probe: one linear layer per patch (BCE), standardised inputs, 30 epochs, 1 seed. Baseline: the same classifier on per-patch
  RGB/HSV mean+std + (row, col).

## Results
| | Value |
|---|---|
| DINOv2 latency, Orin, batch 1, fp32 / fp16 | 45.0 / **23.3 ms** |
| DINOv2 fp16, batch 8 | 8.3 ms per frame |
| Extraction incl. preprocessing, fp16 | 38 ms per frame |
| Token size (fp16) | 597 KB per frame |
| Full track cache at 5 Hz (≈31 min usable) | ≈ 5.6 GB, **≈ 11 GB with flipped copies** (30 GB free) |

| Probe on held-out session | Patch IoU | Patch accuracy |
|---|---|---|
| colour + position, linear | 0.797 | 0.918 |
| **DINOv2 ViT-S/14, linear** | **0.954** | **0.984** |

## What this does and doesn't show
- It shows the frozen tokens carry ground layout (road vs grass vs sky, road edges and curves) that a linear map can read.
  That is the information the goal scorer and the auxiliary BEV head need.
- The target is a pseudo-label (TwinLiteNetPlus), not ground truth. The held-out session is a different day but the same
  site (Morano) as two training sessions. One seed, about 420 training frames.
- **It says nothing about bollards/cones**: there are none in the footage. Whether these tokens separate a 45 cm bollard
  at 10–15 m (about 2 patches wide at 10 m) can only be tested on a session that contains them.
- It does not yet test whether the tokens predict the future *trajectory*. That needs ego-motion labels from ECU speed
  (Phase 2).
