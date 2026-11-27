# Closed-loop comparison: full_goalflow_l3_circles vs the frozen baselines (seeds 0-499; success / collision / stalled % of feasible / exits)

## L0
- GoalFlow (full_goalflow_l3_circles): 91.8 / 0.0 / 5.9 / 11 | brakes 3.3% | correct brake 22/23 | flips 0.30 | ms 241
- baseline (b): 97.5 / 0.0 / 1.9 / 3 | brakes 1.3% | correct brake 23/23 | flips 0.05 | ms 8
- (c) same-information: 95.4 / 0.0 / 4.0 / 3 | brakes 1.6% | correct brake 23/23 | flips 0.05 | ms 8
- GoalFlow stalls 28: best true clearance at the last brake scan: median -0.056 m; truly blocked (<0) 16, clear by >= 0.12 m 1
- vs baseline (b): McNemar on success: GoalFlow-only 6, baseline (b)-only 33, p = 1.43e-05; stall overlap: 5 of 28 GoalFlow stalls on seeds where baseline (b) also stalls (all GoalFlow stalls: truly blocked 16, false alarm 1, 0-0.12 m 11; overlapping ones: 3 / 0 / 2)
- vs (c): McNemar on success: GoalFlow-only 15, (c)-only 32, p = 0.0186; stall overlap: 6 of 28 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 16, false alarm 1, 0-0.12 m 11; overlapping ones: 4 / 0 / 2)
## L1
- GoalFlow (full_goalflow_l3_circles): 91.6 / 0.0 / 6.3 / 10 | brakes 3.7% | correct brake 21/23 | flips 0.25 | ms 241
- baseline (a): 97.9 / 0.6 / 1.3 / 1 | brakes 1.7% | correct brake 22/23 | flips 0.14 | ms 10
- (c) same-information: 95.6 / 0.2 / 3.8 / 2 | brakes 2.4% | correct brake 23/23 | flips 0.15 | ms 10
- GoalFlow stalls 30: best true clearance at the last brake scan: median -0.037 m; truly blocked (<0) 16, clear by >= 0.12 m 3
- vs baseline (a): McNemar on success: GoalFlow-only 5, baseline (a)-only 35, p = 1.38e-06; stall overlap: 4 of 30 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 16, false alarm 3, 0-0.12 m 11; overlapping ones: 2 / 0 / 2)
- vs (c): McNemar on success: GoalFlow-only 14, (c)-only 33, p = 0.00794; stall overlap: 7 of 30 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 16, false alarm 3, 0-0.12 m 11; overlapping ones: 4 / 1 / 2)
## L2
- GoalFlow (full_goalflow_l3_circles): 88.7 / 0.0 / 10.7 / 3 | brakes 4.5% | correct brake 22/23 | flips 0.27 | ms 242
- baseline (a): 98.7 / 0.0 / 1.3 / 0 | brakes 2.6% | correct brake 22/23 | flips 0.26 | ms 14
- (c) same-information: 95.0 / 1.0 / 4.0 / 0 | brakes 3.7% | correct brake 23/23 | flips 0.25 | ms 15
- GoalFlow stalls 51: best true clearance at the last brake scan: median 0.121 m; truly blocked (<0) 18, clear by >= 0.12 m 26
- vs baseline (a): McNemar on success: GoalFlow-only 2, baseline (a)-only 50, p = 6.12e-13; stall overlap: 4 of 51 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 18, false alarm 26, 0-0.12 m 7; overlapping ones: 1 / 2 / 1)
- vs (c): McNemar on success: GoalFlow-only 16, (c)-only 46, p = 0.000176; stall overlap: 8 of 51 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 18, false alarm 26, 0-0.12 m 7; overlapping ones: 4 / 1 / 3)
## L3
- GoalFlow (full_goalflow_l3_circles): 85.7 / 0.0 / 14.3 / 0 | brakes 5.5% | correct brake 23/23 | flips 0.30 | ms 243
- baseline (a): 97.3 / 0.8 / 1.9 / 0 | brakes 5.1% | correct brake 23/23 | flips 0.69 | ms 26
- (c) same-information: 87.8 / 1.0 / 11.1 / 0 | brakes 9.4% | correct brake 23/23 | flips 0.45 | ms 29
- GoalFlow stalls 68: best true clearance at the last brake scan: median 0.178 m; truly blocked (<0) 20, clear by >= 0.12 m 40
- vs baseline (a): McNemar on success: GoalFlow-only 3, baseline (a)-only 58, p = 3.29e-14; stall overlap: 8 of 68 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 20, false alarm 40, 0-0.12 m 8; overlapping ones: 4 / 3 / 1)
- vs (c): McNemar on success: GoalFlow-only 36, (c)-only 46, p = 0.32; stall overlap: 22 of 68 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 20, false alarm 40, 0-0.12 m 8; overlapping ones: 10 / 10 / 2)
## L4
- GoalFlow (full_goalflow_l3_circles): 78.8 / 0.4 / 20.8 / 0 | brakes 7.3% | correct brake 23/23 | flips 0.32 | ms 242
- baseline (a): 93.1 / 2.3 / 4.4 / 1 | brakes 9.5% | correct brake 23/23 | flips 1.61 | ms 42
- (c) same-information: 78.0 / 2.9 / 18.7 / 2 | brakes 17.9% | correct brake 23/23 | flips 1.07 | ms 42
- GoalFlow stalls 99: best true clearance at the last brake scan: median 0.235 m; truly blocked (<0) 18, clear by >= 0.12 m 69
- vs baseline (a): McNemar on success: GoalFlow-only 15, baseline (a)-only 83, p = 1.4e-12; stall overlap: 13 of 99 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 18, false alarm 69, 0-0.12 m 12; overlapping ones: 4 / 6 / 3)
- vs (c): McNemar on success: GoalFlow-only 55, (c)-only 51, p = 0.771; stall overlap: 42 of 99 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 18, false alarm 69, 0-0.12 m 12; overlapping ones: 9 / 27 / 6)
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 2}
