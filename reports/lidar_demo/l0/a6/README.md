# A6: why observed corridor-lattice fails at L0 on 33 seeds that succeed at L1 and L2

Analysis 2026-11-25, **before any planner change**. Every replay here uses the original observed planner
(worktree at bb0a9c1; its planner code equals 50e9ad9, the edge-fill fix A6b). None of the L3 additions
(commit 505d6a8) were active. The 33 seeds are the L0-only failures in `../results.csv` (3513584, which
predates the edge-fill fix). Scripts: `scripts/a6_table.py`, `scripts/a6_classify.py`; data: `a6_table.json`.

## Result

With the edge-fill fix already in, 9 of the 33 now succeed at L0 (L1: 30, L2: 29). The failures and brakes
fall into distinct mechanisms. Two of them are the same flaw in the planner's track-exit test
(`CollisionWorld.clearance`: a footprint corner is "outside" when it lies behind the inward normal of the
nearest observed edge point). Both make **the car's own current pose** fail the check, so every replan is
rejected.

| Mechanism | Seeds | Where | What happens |
|---|---|---|---|
| Goal-line edge stub | 10 (6 stopped, 4 recovered) | first brake 0.1–1.6 m before the goal line | The goal line is oblique in the ego frame (curved track); after the edges are cut at it, one side keeps a 2-point stub ~0.15 m long at the line's end. Its normal is meaningless, so all four corners test "outside": own pose = track exit, every plan rejected. This is the original A6 hypothesis (degenerate geometry where the edges meet the goal line), in specific form. |
| Fill edge inside the true edge beside the car | 7 (6 stopped, 1 recovered) | 3.7–20 m before the goal | Beside the car the forward LiDAR sees no edge; the 12 m fill (opposite edge offset across the track) lies 0.21–1.94 m inside the true edge there. The car's corners, truly 0.15–0.2 m inside the track, fail the 0.12 m check against it. |
| Plan fails, own pose OK | 11 | 5 near the goal (0.5–3.5 m), 6 mid-course | Not an edge-test artefact: the lattice finds no curvature-feasible path in any route corridor. Includes the 2 collisions (287, 451) and 3 exits. Not analysed further yet. |
| Own pose fails 0.12 m, other | 1 (397, exit) | 1.9 m before goal | |
| No brake | 4 (106 exit; 200, 359, 493 succeed) | | |

At L1/L2 the same own-pose failures occur (8 and 6 of these seeds brake with the car's own pose failing),
but the noisy edges change from scan to scan, so a later scan passes and the car continues. At L0 the faulty
geometry is identical in every scan, so the brake is permanent. That is why the failures show up only at L0.

## The radius hypothesis (rejected)

`cluster_obstacles` does not fit a radius: it fits only the centre, with the radius fixed by the camera class
(bollard / cone). So the radius can differ only through class flips. On these 33 scenes, over all scans
(fitted clusters associated with the true obstacle within 0.6 m):

| Level | Class-flip share | Mean radius error | Median centre error | Narrowest route pair: observed − true gap (median) |
|---|---|---|---|---|
| L0 | 0.0% | +0.0 mm | 0 mm | +0.0 mm |
| L1 | 4.9% | +0.7 mm | 14 mm | -1.8 mm |
| L2 | 8.2% | +1.1 mm | 27 mm | -3.4 mm |

Observed gaps at L1/L2 are slightly *narrower* than the truth (−2 to −3 mm), not wider. L1/L2 are not
optimistic through the fitted circles; the L0 failures are the edge-test artefacts above.

## Proposed fixes (not implemented; for review)

1. **Edge test only within each edge's extent**: judge a corner against an edge only if its projection falls
   on the polyline (no extrapolation past its ends), and drop edge pieces shorter than 1 m. Targets the stub
   (10 seeds).
2. **Edge memory beside the car** (L3 item 2, coded behind a flag in 505d6a8): use edge points observed in
   earlier scans instead of the fill. Targets the fill error (7 seeds). Start relief (also coded) is a
   fallback for the same case.
3. The 11 "plan fails, own pose OK" seeds need their own look before the overnight rerun.

## Per-seed table (L0 unless noted)

| Seed | L0 outcome | 1st brake [s] | To goal [m] | v [m/s] | Steer [°] | Nearest obstacle [m] | Narrowest route gap − width [m] (kind) | Own pose fails 0.12 m | Observed-edge error near car [m] | Mechanism | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | stopped | 8.2 | 16.4 | 3.0 | 2.5 | 1.39 | 1.57 (edge) | True | 0.30 | fill edge inside true edge | success | success |
| 202 | stopped | 6.1 | 18.9 | 3.0 | 6.2 | 3.07 | 1.84 (edge) | True | 0.70 | fill edge inside true edge | success | success |
| 225 | stopped | 6.6 | 20.3 | 3.0 | 2.6 | 2.17 | 0.82 (pair) | True | 0.21 | fill edge inside true edge | collision | success |
| 273 | stopped | 7.6 | 18.1 | 3.0 | 5.9 | 1.20 | 2.85 (pair) | True | 1.48 | fill edge inside true edge | success | success |
| 384 | success | 6.6 | 18.7 | 3.0 | 6.4 | 0.60 | 1.97 (pair) | True | 1.94 | fill edge inside true edge | success | success |
| 434 | stopped | 10.3 | 3.7 | 3.0 | 2.6 | 10.25 | 3.54 (pair) | True | 0.46 | fill edge inside true edge | success | success |
| 486 | stopped | 11.2 | 6.9 | 3.0 | 2.7 | 5.78 | 4.36 (edge) | True | 0.80 | fill edge inside true edge | success | stopped |
| 27 | stopped | 12.1 | 1.6 | 3.0 | -0.2 | 2.79 | 2.34 (edge) | True | – | goal-line edge stub | success | success |
| 71 | stopped | 12.2 | 1.6 | 3.0 | -3.0 | 2.46 | 1.53 (pair) | True | – | goal-line edge stub | success | success |
| 82 | success | 12.5 | 0.5 | 3.0 | -0.1 | 6.63 | 1.16 (pair) | True | – | goal-line edge stub | stopped | stopped |
| 86 | stopped | 12.9 | 1.5 | 3.0 | -1.2 | 3.87 | 1.83 (edge) | True | – | goal-line edge stub | success | success |
| 102 | success | 13.3 | 1.0 | 3.0 | -5.3 | 1.27 | 1.74 (pair) | True | – | goal-line edge stub | success | success |
| 146 | stopped | 12.5 | 1.6 | 3.0 | -1.1 | 3.64 | 2.32 (pair) | True | – | goal-line edge stub | success | success |
| 155 | success | 12.5 | 1.1 | 3.0 | 5.8 | 3.26 | 1.94 (edge) | True | -0.00 | goal-line edge stub | success | success |
| 189 | stopped | 12.8 | 1.4 | 3.0 | -0.0 | 2.10 | 1.62 (pair) | True | – | goal-line edge stub | success | success |
| 394 | stopped | 12.6 | 1.3 | 3.0 | 4.2 | 3.39 | 2.45 (edge) | True | – | goal-line edge stub | success | success |
| 432 | success | 12.4 | 0.1 | 3.0 | 1.3 | 3.65 | 2.28 (pair) | True | – | goal-line edge stub | success | success |
| 106 | exit | – | – | – | – | – | 2.34 (pair) | – | – | no brake | success | success |
| 200 | success | – | – | – | – | – | 2.18 (pair) | – | – | no brake | success | success |
| 359 | success | – | – | – | – | – | 2.26 (edge) | – | – | no brake | success | success |
| 493 | success | – | – | – | – | – | 2.21 (pair) | – | – | no brake | success | success |
| 397 | exit | 13.4 | 1.9 | 3.0 | -6.0 | 3.77 | 2.78 (pair) | True | 0.00 | own pose fails 0.12, other | success | success |
| 17 | exit | 12.4 | 2.4 | 3.0 | -0.0 | 7.32 | 0.65 (pair) | False | -0.00 | plan fails, own pose OK | success | success |
| 108 | stopped | 11.9 | 2.8 | 3.0 | -0.1 | 2.36 | 1.51 (pair) | False | – | plan fails, own pose OK | success | success |
| 190 | stopped | 6.6 | 22.1 | 3.0 | 4.7 | 4.62 | 0.90 (pair) | False | – | plan fails, own pose OK | success | stopped |
| 203 | exit | 11.2 | 2.9 | 3.0 | -0.3 | 3.70 | 1.96 (pair) | False | -0.00 | plan fails, own pose OK | success | success |
| 287 | collision | 7.7 | 17.3 | 3.0 | 3.0 | 1.03 | 1.45 (pair) | False | – | plan fails, own pose OK | success | success |
| 308 | stopped | 8.6 | 11.0 | 3.0 | 3.7 | 1.10 | 0.77 (pair) | False | 0.51 | plan fails, own pose OK | success | stopped |
| 333 | success | 14.3 | 0.5 | 3.0 | -2.7 | 5.55 | 2.59 (pair) | False | 0.00 | plan fails, own pose OK | success | success |
| 373 | exit | 12.7 | 3.5 | 3.0 | 0.3 | 0.87 | 1.72 (pair) | False | 0.00 | plan fails, own pose OK | success | success |
| 399 | stopped | 4.3 | 29.0 | 3.0 | 1.0 | 2.82 | 1.85 (pair) | False | – | plan fails, own pose OK | success | success |
| 437 | stopped | 6.2 | 17.0 | 3.0 | -0.6 | 0.53 | 0.66 (pair) | False | 0.50 | plan fails, own pose OK | stopped | success |
| 451 | collision | 9.0 | 14.7 | 3.0 | -3.7 | 0.43 | 1.49 (pair) | False | -0.00 | plan fails, own pose OK | success | success |