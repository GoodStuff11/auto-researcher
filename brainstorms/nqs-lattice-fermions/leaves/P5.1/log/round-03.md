# P5.1 round 3 (2026-11-17): R51.1 `nfloor` unit 1 scored; unit 2 prepared (worker)

Task (orchestrator): unit 1 of RULE_B (job 759974: arms A and B) is done. Apply `tests/P5.1_noise_floor_r01/RULE_B.md`
exactly (R_plat, Delta = R_pol - R_plat, truncation clause, release order), state unit 1's verdict, prepare unit 2
(gated, not submitted), and say what unit 1 implies for P4.1's sampled-optimiser question.

## Unit 1 data (job 759974, `results/N16384_tau0.005_s9000/run.json`, `results/N65536_tau0.005_s9000/run.json`)
Both tasks on kim-cpu-04. R_pol = 0.7475. R_plat = mean R over the arm's last 20 steps; tail slope = mean dR there.

| arm | N | steps done | CPU-min (in-job per step) | R(last) | R_plat | tail slope | rises | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A | 16,384 | 120 / 120 | 43.3 (0.359) | 0.5865 | 0.5762 (steps 101-120) | 0.0012 | 18/120 (max 0.0018) | 0.1713 |
| B | 65,536 | 83 / 120, TRUNCATED by `--cpu-cap 66` | 65.9 (0.790) | 0.6499 | 0.6334 (steps 64-83) | 0.0018 | 2/83 | ≤ 0.1141 |

Trajectory (R at step 20/40/60/80/83/100/120): A 0.364/0.444/0.494/0.534/0.536/0.563/0.587; B 0.463/0.555/0.608/
0.645/0.650/-/-. Mean dR per 20-step block: A 0.0182, 0.0040, 0.0025, 0.0020, 0.0014, 0.0012; B 0.0231, 0.0046,
0.0027, 0.0019 (61-80).

Why B was truncated: the in-job time per step was 1.62x (A) and 1.79x (B) the interactive pre-flight (0.221, 0.441),
so the cap of 66 CPU-min, set from the pre-flight x1.3, covered 83 steps. Lesson for every gated job here: time per
step must be measured on the partition's nodes, not the interactive node (proposed library entry below).

## Verdict per RULE_B (unit 1)
* REACHES: no. R_plat(A) = 0.5762 < 0.7475.
* B is INCOMPLETE (stopped at 83 of 120). Its tail slope 0.0018 is below the rule's 0.002 "not yet plateaued" flag,
  but it was rising when cut (dR 0.0016 at step 83), so the truncation clause is applied as the orchestrator directs:
  R_plat(B) = 0.6334 is a lower bound, Delta(B) ≤ 0.1141 an upper bound, and NOT-NOISE is not declared from it.
* Delta(B)/Delta(A) ≤ 0.1141 / 0.1713 = 0.666: above NOISE-FLOOR's 0.62, below NOT-NOISE's 0.85. NOT-NOISE is
  excluded (ratio < 0.85 even at its upper bound); NOISE-FLOOR's first condition is not met by the scored numbers
  but could be once B completes (the bound can only fall).
* **Unit 1 verdict: undecided (neither REACHES nor NOT-NOISE) -> unit 2 released per the release order.**
* Reported only (not the pre-registered statistic): at equal step count 83, A's steps 64-83 mean is 0.5241, Delta 0.2234,
  ratio 0.511.
* Caveat for the card: A's tail slope 0.0012 passes the rule's plateau test but A is still rising (+0.024 over steps
  101-120). RULE_B's Delta therefore measures the remaining shortfall after a fixed step count, not a true floor; a
  NOISE-FLOOR verdict will mean "4-16x samples buy proportionally faster progress", which is what the leaf's
  total-samples question needs, but it will not show that a floor exists.

## What unit 1 implies for P4.1's sampled-optimiser question
* At equal steps the fixed short step does not beat R38's held-out ladder: A's R(60) = 0.494 against R38's 0.536,
  0.512, 0.508 (same N = 16,384; R38 also spent held-out draws). R38's larger early steps win early.
* It keeps going where R38 stalled: A reaches 0.587 at 120 steps (R38 realised 0.0020/step over 41-60 and the kill
  read its flat end as a floor). So R38's "plateau at 0.51" was not the guided estimator's floor; a fixed short step
  continues at 0.0012-0.0020/step from there.
* 4N at fixed length beats R38 at equal steps: B's R(60) = 0.608. But per sample, A wins: A at 120 steps (1.97M
  draws) R = 0.587 against B at 30 steps (same 1.97M draws) R = 0.516. At this stage more samples per step buy less
  than more steps; unit 2's C arm and the total-samples comparison decide whether that reverses near R_pol.
* The one-step counterfactual overstated the sustained rate: T1a (RULE_A) gave 0.0078/step for tau 0.005 at R38's
  states (R about 0.51-0.54); A, at its own states with similar R (steps 60-80), gained 0.0020-0.0025/step. A step
  rate measured at another trajectory's states is not the rate the fixed-step optimiser sustains.
* Practical recipe for P4.1 now: large held-out-chosen steps early (R38) then a fixed short length, i.e. a
  decaying schedule, which is R51.7's premise; R51.7's gate stays arm D.

## Unit 2 preparation
* Required beyond C and D: completing B to 120 steps (RULE_B scores Delta(B) at its last 20 of 120 steps, and both
  NOISE-FLOOR conditions need it). `nf_run.py` resumes from `run.json`, so this is 37 steps, not a rerun.
* Arm C as written was expensive: 2-step interactive pre-flight (`logs/preflight_N262144.txt`) 1.21 CPU-min/step and
  17.6 GB maxrss (N x P arrays); in-job (x1.79) 2.17/step, 120 steps x1.3 = 5.6 h, so a 3-task array at that limit
  would cap at 17 CPU-h (audit). Cheapest version that still decides: `nf_run.py --dedup` (new option, off by default;
  A, B, D do not use it) sums the weights of repeated draws before forming S_T and f, an exact algebraic rewrite.
  Verified by `dedup_check.py` (`logs/dedup_check.txt`): at arm B's step-1 draw, max relative difference S_T 3.6e-16,
  f 9.3e-15, step 5.7e-13; PASS. Dedup pre-flight (`logs/preflight_N262144_dedup.txt`): identical R at steps 1-2
  (0.5654, 0.5797), 0.46 CPU-min/step, 3.3 GB.
* Wall time from unit 1's in-job rates: B 37 x 0.790 + 0.6 = 30 min; C 120 x 0.82 + 1 = 99 min; D 240 x 0.359 + 0.5 =
  87 min. Longest x1.3 = 129 -> `--time 02:20:00`; `--cpu-cap 135` per task, so no arm is truncated by its cap unless
  its node is more than 35% slower than kim-cpu-04 (it then stops cleanly and saves).
* Files: `tests/P5.1_noise_floor_r01/submit_unit2.sbatch` (array 0 = B resume, 1 = C dedup, 2 = D),
  `voi_unit2.md` (prediction NOISE-FLOOR p 0.60, AMBIGUOUS 0.30, NOT-NOISE 0.05, INCOMPLETE 0.05; STEP-NOISE p 0.4).
* Gate: `compute_gate.py check <wd> tests/P5.1_noise_floor_r01/submit_unit2.sbatch --voi .../voi_unit2.md --pre-audit`
  -> cap = 3 array x 2.33 h x 1 cpu = 7.0 CPU-h; leaf committed 1.9 / 300; GATE: PASS. Not submitted.

## Compute this round
Interactive: arm C pre-flights 2.57 + 1.06 CPU-min, dedup check about 1 CPU-min: 0.08 CPU-h. Unit 1 used 1.82 CPU-h.

## Proposed library entries (orchestrator owns library.md)
* Pre-flight timing on the interactive node under-predicted in-job time per step by 1.6-1.8x (job 759974 on kim-cpu-04),
  truncating a capped arm at 83 of 120 steps; set caps from in-job rates or measure a step inside a short job.
* A one-step counterfactual gain measured at another optimiser's states overstated the sustained fixed-step rate 3-4x
  (RULE_A 0.0078/step vs 0.0020-0.0025 realised at similar R, P5.1 R51.1).
* (revision of R38's lesson) Guided SR's 0.51 "plateau" was the held-out step choice, not a floor: a fixed short step
  continued to 0.587 (120 steps), 4x samples to 0.650 (83 steps).
