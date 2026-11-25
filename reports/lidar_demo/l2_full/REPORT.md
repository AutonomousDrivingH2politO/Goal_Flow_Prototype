# L2 seed-0 results (2026-11-25)

Validation subset of 4,000 frames (fixed), best checkpoints. Full numbers: `summary.json`; log: `train.log`.

## 1. Goal-score multimodality (446 frames with 2+ teacher modes)

| Metric | Value |
|---|---|
| cover@5 | 0.32 |
| cover@10 | 0.35 |
| g* collision rate | 0.1% |
| Mode-mass share | 0.79 |

The scorer avoids obstacles (DAC accuracy near obstacles 0.94, balanced 0.94).

**Correction (2026-11-25, full validation split, `../l2_selection/goal_scorer_val.json`).** The low plain
cover@k above does not mean the scorer predicts a single route. Plain top-k on a dense vocabulary (1,024
goals) is used up by neighbours of the best goal. Findings:

- The vocabulary covers every teacher route: nearest goal median 0.015–0.02 m, p95 <= 0.07 m, 0% beyond
  0.5 m, driven and non-driven alike. (The y in [-3.56, 3.58] m in the log is the trajectory normaliser's
  1st/99th percentile, not the vocabulary, which spans y in [-6.56, 6.57] m.)
- The driven route ranks first. With 1.5 m suppression, the other route is the 2nd distinct pick in 76% of
  cases (all routes) / 61% (endpoint-distinct routes), and within the top 5 in 96.5% / 94%.
- Suppressed cover@2 / 3 / 5 on multi-route frames: 0.71 / 0.84 / 0.96 (all routes, 7,777 frames) and
  0.57 / 0.75 / 0.94 (endpoint-distinct, 4,864 frames), against plain cover@5 of 0.35 / 0.02.
- The l_dis plateau (~5.8) is mostly the target's entropy floor: the gap to the mixture target is 0.29
  nats; for non-driven routes alone it is 1.26.
- The collapse in the renders (e.g. frame 40, all 128 samples within 1 m of one mode) comes from
  conditioning every sample on the single g*, not from the scorer.

## 2. Share of raw samples within κ_max (teacher reference: 98.1% at 1.0 ×, 100% at 1.1 ×)

| Model | 1.0 × | 1.1 × |
|---|---|---|
| GoalFlow (flow samples) | 64% | 72% |
| Regression | 84% | 89% |

GoalFlow's share rose from 9% early in stage B. Executed paths are drivable for both models by construction
(pure-pursuit + bicycle rollout, 10° limit).

## 3. ADE / FDE against the driven teacher mode

| | ADE executed | FDE executed | ADE raw | FDE raw | Best-of-modes ADE (exec / raw) |
|---|---|---|---|---|---|
| GoalFlow (M = 32) | 0.199 m | 0.481 m | 0.292 m | 0.676 m | 0.188 / 0.279 m |
| Regression | 0.193 m | 0.506 m | 0.190 m | 0.505 m | 0.185 / 0.182 m |

Regression is slightly better on executed ADE and clearly better on raw paths; GoalFlow is slightly better
on executed FDE. ADE against the driven mode favours an averaging model when it picks the same side; the
multi-mode advantage GoalFlow is meant to have only shows up in the closed loop (L3). Sample endpoints land
0.04 m from g* (executed). Open question for L3: GoalFlow's selected path (0.20 m) beats its best individual
sample (0.27 m), which suggests the shadow switch (untuned 1.5 m threshold) fires often.

## 4. Training time and checkpoints

Wall time 02:04 to 10:21 (8 h 17 min; a failed first attempt 01:32–02:00 excluded):

| Stage | Time | Steps |
|---|---|---|
| A (encoder + goal scorer) | ~3 h 08 min (incl. a restart at 04:06) | 12,784 |
| B (generator, A frozen) | 3 h 36 min | 86,603 |
| Regression | 1 h 31 min | 15,875 |

No stage stopped early.

- `runs/l2_full_seed0/goalflow_B_best.pt`: full GoalFlow model, 10.12 M parameters
- `runs/l2_full_seed0/regression_best.pt`
- Full training states for resume: `runs/l2_full_seed0/state_{A,B,reg}.pt`
- Renders: this folder, 12 frames by the fixed rule in `RULE.txt` (redrawn 2026-11-25 with M = 128, the one
  sample count now used everywhere; the table above used M = 32 in the training-time evaluation)
- The L1 test split was read once (`test_report.json`) and is now spent; final numbers come from a fresh
  test set before L4 (DEVIATIONS, Phase L2 plan)

## Overnight interventions (details in docs/DEVIATIONS.md)

1. First run thrashed on memory (17.65 GB at batch 64 in stage A): goal scorer switched to cross-attention
   only, stage A uses gradient accumulation (2 × 32); 4.65 GB, 0.84 s/step.
2. Stage A checkpoint selection switched from goal error to validation goal loss (goal error penalises the
   other valid mode and had stalled while every other metric improved).
3. After training the generator was restarted: its original workers had died while paused.
