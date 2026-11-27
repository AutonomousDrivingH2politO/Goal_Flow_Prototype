# Closed-loop comparison: full_goalflow_l3_combined vs the frozen baselines (seeds 0-499; success / collision / stalled % of feasible / exits)

## L0
- GoalFlow (full_goalflow_l3_combined): 93.3 / 0.2 / 5.7 / 4 | brakes 3.3% | correct brake 22/23 | flips 0.30 | ms 254
- baseline (b): 97.5 / 0.0 / 1.9 / 3 | brakes 1.3% | correct brake 23/23 | flips 0.05 | ms 8
- (c) same-information: 95.4 / 0.0 / 4.0 / 3 | brakes 1.6% | correct brake 23/23 | flips 0.05 | ms 8
- GoalFlow stalls 27: best true clearance at the last brake scan: median -0.087 m; truly blocked (<0) 17, clear by >= 0.12 m 1
- vs baseline (b): McNemar on success: GoalFlow-only 7, baseline (b)-only 27, p = 0.000821; stall overlap: 5 of 27 GoalFlow stalls on seeds where baseline (b) also stalls (all GoalFlow stalls: truly blocked 17, false alarm 1, 0-0.12 m 9; overlapping ones: 3 / 0 / 2)
- vs (c): McNemar on success: GoalFlow-only 16, (c)-only 26, p = 0.164; stall overlap: 6 of 27 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 17, false alarm 1, 0-0.12 m 9; overlapping ones: 4 / 0 / 2)
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 1}
## L1
- GoalFlow (full_goalflow_l3_combined): 91.6 / 0.4 / 7.5 / 2 | brakes 3.8% | correct brake 21/23 | flips 0.26 | ms 252
- baseline (a): 97.9 / 0.6 / 1.3 / 1 | brakes 1.7% | correct brake 22/23 | flips 0.14 | ms 10
- (c) same-information: 95.6 / 0.2 / 3.8 / 2 | brakes 2.4% | correct brake 23/23 | flips 0.15 | ms 10
- GoalFlow stalls 36: best true clearance at the last brake scan: median -0.048 m; truly blocked (<0) 19, clear by >= 0.12 m 10
- vs baseline (a): McNemar on success: GoalFlow-only 5, baseline (a)-only 35, p = 1.38e-06; stall overlap: 4 of 36 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 19, false alarm 10, 0-0.12 m 7; overlapping ones: 3 / 0 / 1)
- vs (c): McNemar on success: GoalFlow-only 13, (c)-only 32, p = 0.00661; stall overlap: 7 of 36 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 19, false alarm 10, 0-0.12 m 7; overlapping ones: 5 / 1 / 1)
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 2}
## L2
- GoalFlow (full_goalflow_l3_combined): 89.1 / 0.6 / 10.1 / 1 | brakes 4.5% | correct brake 22/23 | flips 0.28 | ms 252
- baseline (a): 98.7 / 0.0 / 1.3 / 0 | brakes 2.6% | correct brake 22/23 | flips 0.26 | ms 14
- (c) same-information: 95.0 / 1.0 / 4.0 / 0 | brakes 3.7% | correct brake 23/23 | flips 0.25 | ms 15
- GoalFlow stalls 48: best true clearance at the last brake scan: median 0.092 m; truly blocked (<0) 19, clear by >= 0.12 m 22
- vs baseline (a): McNemar on success: GoalFlow-only 2, baseline (a)-only 48, p = 2.27e-12; stall overlap: 4 of 48 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 19, false alarm 22, 0-0.12 m 7; overlapping ones: 1 / 3 / 0)
- vs (c): McNemar on success: GoalFlow-only 16, (c)-only 44, p = 0.000394; stall overlap: 7 of 48 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 19, false alarm 22, 0-0.12 m 7; overlapping ones: 4 / 2 / 1)
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 3}
## L3
- GoalFlow (full_goalflow_l3_combined): 87.8 / 0.8 / 11.1 / 1 | brakes 5.0% | correct brake 22/23 | flips 0.33 | ms 253
- baseline (a): 97.3 / 0.8 / 1.9 / 0 | brakes 5.1% | correct brake 23/23 | flips 0.69 | ms 26
- (c) same-information: 87.8 / 1.0 / 11.1 / 0 | brakes 9.4% | correct brake 23/23 | flips 0.45 | ms 29
- GoalFlow stalls 53: best true clearance at the last brake scan: median 0.127 m; truly blocked (<0) 15, clear by >= 0.12 m 28
- vs baseline (a): McNemar on success: GoalFlow-only 5, baseline (a)-only 50, p = 2.14e-10; stall overlap: 5 of 53 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 15, false alarm 28, 0-0.12 m 10; overlapping ones: 3 / 1 / 1)
- vs (c): McNemar on success: GoalFlow-only 38, (c)-only 38, p = 1; stall overlap: 18 of 53 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 15, false alarm 28, 0-0.12 m 10; overlapping ones: 7 / 7 / 4)
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 3, 'no safe candidate': 1}
## L4
- GoalFlow (full_goalflow_l3_combined): 80.5 / 1.0 / 18.4 / 0 | brakes 6.7% | correct brake 23/23 | flips 0.38 | ms 253
- baseline (a): 93.1 / 2.3 / 4.4 / 1 | brakes 9.5% | correct brake 23/23 | flips 1.61 | ms 42
- (c) same-information: 78.0 / 2.9 / 18.7 / 2 | brakes 17.9% | correct brake 23/23 | flips 1.07 | ms 42
- GoalFlow stalls 88: best true clearance at the last brake scan: median 0.201 m; truly blocked (<0) 17, clear by >= 0.12 m 55
- vs baseline (a): McNemar on success: GoalFlow-only 16, baseline (a)-only 76, p = 1.6e-10; stall overlap: 12 of 88 GoalFlow stalls on seeds where baseline (a) also stalls (all GoalFlow stalls: truly blocked 17, false alarm 55, 0-0.12 m 16; overlapping ones: 2 / 5 / 5)
- vs (c): McNemar on success: GoalFlow-only 60, (c)-only 48, p = 0.29; stall overlap: 34 of 88 GoalFlow stalls on seeds where (c) also stalls (all GoalFlow stalls: truly blocked 17, false alarm 55, 0-0.12 m 16; overlapping ones: 6 / 20 / 8)
- GoalFlow collision causes: {'safety-filter miss (obstacle in hit memory)': 4, 'no safe candidate': 1}
