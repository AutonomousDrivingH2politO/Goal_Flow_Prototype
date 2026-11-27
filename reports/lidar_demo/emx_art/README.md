# Beams-only comparison: EMX beam pattern with L2 / L4 artefacts (2026-11-27)

`scripts/run_emx_art.sh`, start commit 11f432a; seeds 0-499; front-axle mount, constant 3 m/s. `emx_L2art` / `emx_L4art`
= EMX beam pattern (0.08 deg) and range noise (0.05 m) with the L2 / L4 artefacts. Success / collision / stalled % of
feasible scenes, brake rate; L2 / L4 from `l0_rerun_distdrift/`, EMX from `emx_baselines/`.

| config | L2 | emx_L2art | L4 | emx_L4art | EMX |
|---|---|---|---|---|---|
| (a) | 98.7 / 0.0 / 1.3, brakes 2.6% | 96.9 / 0.4 / 2.5, brakes 2.4% | 93.1 / 2.3 / 4.4, brakes 9.5% | 91.6 / 2.9 / 5.2, brakes 9.3% | 98.7 / 0.0 / 1.0, brakes 1.4% |
| (c) | 95.0 / 1.0 / 4.0, brakes 3.7% | 95.6 / 0.4 / 3.8, brakes 3.4% | 78.0 / 2.9 / 18.7, brakes 17.9% | 78.6 / 1.5 / 19.9, brakes 17.5% | 97.9 / 0.0 / 1.9, brakes 1.6% |

Hits per bollard at 2-5 / 8-12 / 18-22 m (`scripts/hits_per_obstacle.py`): L2 29.6 / 10.4 / 5.0, emx_L2art 74.9 / 26.0 /
12.6; L4 17.8 / 6.2 / 2.7, emx_L4art 49.2 / 16.9 / 7.8; EMX 82.8 / 29.0 / 13.9 (cones similar; DEVIATIONS).

**Reading:** the denser beam pattern does not help the classical baselines: with the same artefacts, emx_L2art is about
L2 and emx_L4art about L4 (2.5x the hits per obstacle changes success by -1.8 to +0.6 points, collisions within one or
two episodes). The artefacts set the result, so EMX scoring like L1-L2 comes from its assumed L1 artefact values, not
from the sensor's resolution. The EMX headline therefore stands or falls with the real dropout, clutter, edge noise and
class-flip rates, which should be measured from real Juno recordings before the EMX numbers are quoted as realistic.
