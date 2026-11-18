# VOI note: P5.1 R51.1 `nfloor` unit 3 (`submit_unit3.sbatch`; rule RULE_B.md, unchanged)

* leaf: P5.1
* route: R51.1
* tier: T2
* decision: RULE_B after unit 2 (job 761860, `results/score_unit2.json`): Delta(A) 0.1713, Delta(B) 0.0677, Delta(C) 0.0155, Delta(D) 0.1469; Delta(B)/Delta(A) = 0.395 <= 0.62 and Delta(C)/Delta(B) = 0.229 <= 0.62 -> NOISE-FLOOR (fitted exponent a = 0.87); Delta(D)/Delta(A) = 0.858 > 0.75 -> not STEP-NOISE. The release order requires a second seed (9500) of every arm whose margin to a threshold is within 2 replica spreads (0.030): C (0.026 to NOISE-FLOOR's second condition), D (0.018) and A (0.025) to STEP-NOISE. Verdicts after unit 3 (seed means): NOISE-FLOOR kept and not STEP-NOISE -> R51.7 killed (as already recorded), leaf proceeds with the samples-per-step scaling route R51.8; NOISE-FLOOR lost (Delta(C) mean > 0.62 Delta(B)) -> AMBIGUOUS, R51.1's exponent withdrawn, R51.8's rule unchanged (it uses its own arms); STEP-NOISE gained (Delta(D) mean <= 0.75 Delta(A) mean) -> R51.7 revived to a T1 run of the decaying schedule.
* prediction: verdict unchanged: NOISE-FLOOR kept, STEP-NOISE not reached.
* p: 0.93
* prediction-changes-decision: no
* cheapest-kill-tried: stored-data power estimate. (i) C: to lose NOISE-FLOOR the seed-mean Delta(C) must exceed 0.0420, i.e. the second seed must land 0.053 below the first in R; the replica spread 0.015 was measured at N = 16,384 with held-out step selection (R38), and at 16N with a fixed step the spread is smaller (sd of the per-step dR over C's last 20 steps is 0.0012, and C's R(t) is monotone to within 0.002), so P(flip) < 0.01. (ii) STEP-NOISE: the seed-mean d = Delta(D) - 0.75 Delta(A) is now +0.018; with per-seed sd sqrt(0.015^2 + (0.75 x 0.015)^2) = 0.019 the second seed must land below -0.018, z = -1.9, P about 0.03. (iii) Both changed decisions together have P about 0.04-0.07.
* cost-cpuh: 6.75 (cap 3 tasks x 2.25 h x 1 CPU; expected use 4.0)
* cheapest-version: only the three arms inside the margin clause; B is outside it. Fewer steps would change the scored statistic (last-20 mean at 120/240 steps).
* convergence: the seed-9000 arms ran their full step counts in unit 2 within these caps (A 43, C 95.5, D 97.9 CPU-min).
* wall-time-basis: in-job rates of job 761860 (kim partition): C 95.5 CPU-min for 120 steps, D 97.9 for 240, A 43.3 for 120 (job 759974); longest 98 x 1.3 = 127 min -> --time 02:15:00, --cpu-cap 130.
* power: P(any verdict change) about 0.05 from the spreads above; the margins are 0.9-1.8 replica spreads, which is why the clause fires, but the predicted outcome is not in doubt at the skip threshold.

## Correction (per-leaf critic, 2026-11-17; the refusal stands)
* The p 0.93 above is overstated. (ii) used the observed d = +0.018 as the true value; the predictive sd of the second seed's d is sqrt(2) x 0.019 = 0.027, which gives P(STEP-NOISE flip) about 0.09, not 0.03. (i) used C's within-run step-to-step spread (0.0012) as a stand-in for the between-seed spread, which accumulates along the run, so P(C flip) is larger than 0.01.
* corrected p: about 0.88-0.90 (verdict unchanged), still >= 0.8, so the G2 refusal of unit 3 is procedurally valid.
* Recorded on R51.1's card: the pre-registered release order of RULE_B (second seeds of A, C, D) was overridden by the compute gate, not run.
