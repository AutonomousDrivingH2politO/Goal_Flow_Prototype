# L2 seed-0 results (2026-11-25)

Validation subset of 4,000 frames (fixed), best checkpoints. Full numbers: `summary.json`; log: `train.log`.

## 1. Goal-score multimodality (446 frames with 2+ teacher modes)

| Metric | Value |
|---|---|
| cover@5 | 0.32 |
| cover@10 | 0.35 |
| g* collision rate | 0.1% |
| Mode-mass share | 0.79 |

The scorer avoids obstacles (DAC accuracy near obstacles 0.94, balanced 0.94) but does not keep both teacher
modes: in two thirds of multi-mode frames the top 10 goals miss a mode, and coverage stayed flat around 0.3
for the whole run (see frame 6 in the renders). Mode mass is high only because the distance scores are
fairly spread out.

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
- Renders: this folder, 12 frames by the fixed rule in `RULE.txt`

## Overnight interventions (details in docs/DEVIATIONS.md)

1. First run thrashed on memory (17.65 GB at batch 64 in stage A): goal scorer switched to cross-attention
   only, stage A uses gradient accumulation (2 × 32); 4.65 GB, 0.84 s/step.
2. Stage A checkpoint selection switched from goal error to validation goal loss (goal error penalises the
   other valid mode and had stalled while every other metric improved).
3. After training the generator was restarted: its original workers had died while paused.
