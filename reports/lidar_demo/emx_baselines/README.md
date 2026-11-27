# Baselines at the EMX sensor level (2026-11-27)

Seeds 0-499, level `EMX`, **front-axle mount (1.60 m, 1.10 m, 7 deg), constant 3 m/s** (roof-mount and `juno_bangbang`
checks come later). Command: `scripts/run_emx_baselines.sh`, start commit 8e101c4.

| Config | Success | Collision | Stalled | Exits | Brake rate | Correct brakes | Door flips / run | ms / replan |
|---|---|---|---|---|---|---|---|---|
| (b) stub fix + L3 | 97.7% | 0.0% | 1.9% | 2 | 1.7% | 22/23 | 0.12 | 11 |
| **(a) pre-L3 + stub fix** | **98.7%** | 0.0% | 1.0% | 1 | 1.4% | 22/23 | 0.14 | 11 |
| (c) (b) without edge memory | 97.9% | 0.0% | 1.9% | 1 | 1.6% | 22/23 | 0.12 | 11 |

EMX baseline = (a), by the per-level rule (best classical config, chosen from the baseline run alone).

**What the EMX preset contains** (`NOISE_LEVELS["EMX"]`; every component on):

| Component | EMX | L1 | L2 | Source |
|---|---|---|---|---|
| range noise (1 sigma) | 0.05 m | 0.03 | 0.05 | spec |
| horizontal beam step | 0.08 deg | 0.2 | 0.2 | spec |
| vertical span / blind zone | -12 to +8 deg about the pitch / 1.5 m | same | same | spec |
| bearing jitter | 0.05 deg | 0.05 | 0.10 | assumption (L1) |
| point dropout | 5% | 5% | 15% | assumption (L1) |
| clutter | 0.5 clusters/scan | 0.5 | 1.5 | assumption (L1) |
| edge noise / edge dropouts | 0.10 m / 0.3 per side | 0.10 / 0.3 | 0.15 / 0.6 | assumption (L1) |
| class flips | 5% | 5% | 8% | assumption (L1) |
| odometry drift (per 0.30 m) | 0.005 m / 0.05 deg | same | 0.010 / 0.10 | assumption (L1) |

So EMX = L2 range noise + L1 artefacts + a 2.5x denser beam pattern. None is off, so no "EMX + L2 artefacts" variant
is queued.
