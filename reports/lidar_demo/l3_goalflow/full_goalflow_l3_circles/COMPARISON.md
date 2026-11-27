# Full closed-loop run: goalflow_l3_circles vs the frozen baselines (seeds 0-499; success / collision / stalled % of feasible / exits)

## L0
- GoalFlow (goalflow_l3_circles): 91.8 / 0.0 / 5.9 / 11 | brakes 3.3% | correct brake 22/23 | flips 0.30 | ms 241
- baseline (b): 97.5 / 0.0 / 1.9 / 3 | brakes 1.3% | correct brake 23/23 | flips 0.05 | ms 8
- (c) same-information: 95.4 / 0.0 / 4.0 / 3 | brakes 1.6% | correct brake 23/23 | flips 0.05 | ms 8
- GoalFlow stalls 28: best true clearance at the last brake scan: median -0.056 m; truly blocked (<0) 16, clear by >= 0.12 m 1
## L1
- GoalFlow (goalflow_l3_circles): 91.6 / 0.0 / 6.3 / 10 | brakes 3.7% | correct brake 21/23 | flips 0.25 | ms 241
- baseline (a): 97.9 / 0.6 / 1.3 / 1 | brakes 1.7% | correct brake 22/23 | flips 0.14 | ms 10
- (c) same-information: 95.6 / 0.2 / 3.8 / 2 | brakes 2.4% | correct brake 23/23 | flips 0.15 | ms 10
- GoalFlow stalls 30: best true clearance at the last brake scan: median -0.037 m; truly blocked (<0) 16, clear by >= 0.12 m 3
## L2
- GoalFlow (goalflow_l3_circles): 88.7 / 0.0 / 10.7 / 3 | brakes 4.5% | correct brake 22/23 | flips 0.27 | ms 242
- baseline (a): 98.7 / 0.0 / 1.3 / 0 | brakes 2.6% | correct brake 22/23 | flips 0.26 | ms 14
- (c) same-information: 95.0 / 1.0 / 4.0 / 0 | brakes 3.7% | correct brake 23/23 | flips 0.25 | ms 15
- GoalFlow stalls 51: best true clearance at the last brake scan: median 0.121 m; truly blocked (<0) 18, clear by >= 0.12 m 26
## L3
- GoalFlow (goalflow_l3_circles): 85.7 / 0.0 / 14.3 / 0 | brakes 5.5% | correct brake 23/23 | flips 0.30 | ms 243
- baseline (a): 97.3 / 0.8 / 1.9 / 0 | brakes 5.1% | correct brake 23/23 | flips 0.69 | ms 26
- (c) same-information: 87.8 / 1.0 / 11.1 / 0 | brakes 9.4% | correct brake 23/23 | flips 0.45 | ms 29
- GoalFlow stalls 68: best true clearance at the last brake scan: median 0.178 m; truly blocked (<0) 20, clear by >= 0.12 m 40
## L4
- GoalFlow (goalflow_l3_circles): 78.8 / 0.4 / 20.8 / 0 | brakes 7.3% | correct brake 23/23 | flips 0.32 | ms 242
- baseline (a): 93.1 / 2.3 / 4.4 / 1 | brakes 9.5% | correct brake 23/23 | flips 1.61 | ms 42
- (c) same-information: 78.0 / 2.9 / 18.7 / 2 | brakes 17.9% | correct brake 23/23 | flips 1.07 | ms 42
- GoalFlow stalls 99: best true clearance at the last brake scan: median 0.235 m; truly blocked (<0) 18, clear by >= 0.12 m 69
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 2}
