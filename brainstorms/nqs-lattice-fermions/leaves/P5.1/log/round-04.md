# P5.1 round 4 (2026-11-17): R51.1 scored after unit 2; size-scaling route generated and gated (worker)

Task (orchestrator): (1) score R51.1 under RULE_B exactly with unit 2 (job 761860) done; state the verdict and
whether unit 3 is required by the margin clause, and prepare it if so; fix the R51.3 card and summary row (T1d is
done, MARGINAL). (2) Generate and gate the cheapest route that measures how the samples needed to reach a fixed
fraction of the representable minimum grow with system size, whether duplicate aggregation's advantage vanishes,
and whether the existing arms give the comparison with SR/MinSR; pre-registered rule, sbatch, voi, gate check.

## 1. R51.1 under RULE_B (`tests/P5.1_noise_floor_r01/score_unit2.py` -> `results/score_unit2.json`)
R_pol = 0.7475. R_plat = mean R over the last 20 steps (RULE_B's estimator); Delta = R_pol - R_plat; slope = mean dR
over the last 20 steps.

| arm | N | tau | steps | R(last) | R_plat | slope | rises | Delta | CPU-min (job) | mean unique draws |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A | 16,384 | 0.005 | 120 | 0.5865 | 0.5762 | 0.0012 | 18 | 0.1713 | 43.3 (759974) | 6,257 |
| B | 65,536 | 0.005 | 120 | 0.6868 | 0.6798 | 0.0009 | 9 | 0.0677 | 65.9 + 32.7 (759974 + 761860) | 14,923 |
| C | 262,144 (dedup) | 0.005 | 120 | 0.7384 | 0.7320 | 0.0007 | 10 | 0.0155 | 95.5 (761860) | 31,697 |
| D | 16,384 | 0.0025 | 240 | 0.6059 | 0.6006 | 0.0005 | 13 | 0.1469 | 97.9 (761860) | 6,256 |

* REACHES: no (R_plat(A) 0.576 < 0.7475).
* NOISE-FLOOR: Delta(B)/Delta(A) = 0.395 ≤ 0.62 and Delta(C)/Delta(B) = 0.229 ≤ 0.62 -> **NOISE-FLOOR**. Fitted
  exponent (least squares on ln Delta vs ln N, A-C) a = 0.87; pairwise 0.67 (A->B), 1.06 (B->C).
* NOT-NOISE: no (ratios far below 0.85 / 0.75).
* STEP-NOISE: Delta(D) = 0.1469 > 0.75 Delta(A) = 0.1285 (ratio 0.858) -> **not STEP-NOISE** (D's slope ≥ 0 holds).
* No arm exceeds the 0.002 "not yet plateaued" slope, so RULE_B treats all R_plat as plateau estimates; the card
  keeps the caveat that no arm has levelled off (slopes 0.0005-0.0012), so Delta is the shortfall after a fixed
  step count.
* Margin clause ("a second seed (9500) of the arms whose verdict margin is within 2 replica spreads", 2 x 0.015 =
  0.030 in R): NF1 margins A 0.062, B 0.038; NF2 margins C 0.026, B 0.043; STEP-NOISE margins D 0.018, A 0.025.
  Required by the rule's letter: **unit 3 = A, C, D at seed 9500.**
* Unit 3 prepared: `submit_unit3.sbatch` (3 tasks, --time 02:15:00, cap 6.75 CPU-h), `voi_unit3.md` (prediction:
  verdict unchanged, p 0.93; C's last-20 dR sd is 0.0011, so a flip needing 0.053 lower R is implausible; STEP-NOISE
  flip needs the second seed 1.9 per-seed spreads below the first, P about 0.03). Gate:
  `compute_gate.py check ... submit_unit3.sbatch --voi voi_unit3.md --pre-audit` -> cap 6.8 CPU-h; leaf committed
  5.7 / 300; **GATE: REFUSED - G2 predicted, don't run: p=0.93 ≥ 0.8 and the predicted verdict does not change the
  decision.** The recorded prediction stands in place of unit 3.
* Consequences (voi_unit2 decisions): NOISE-FLOOR -> samples-per-step / variance routes first, conditioning routes
  (R51.3) deprioritised; not STEP-NOISE -> **R51.7 KILLED** (card rewritten, kill-index row added).
* Noise model (T0 for part 2): Delta = Delta_0 + b/N fitted on A and B gives Delta_0 = 0.033, b = 0.138 (N in units
  of 16,384) and predicts Delta(C) = 0.042 against 0.0155 observed: a stationary noise ball around a fixed
  deterministic shortfall is wrong; noise slows the path itself. "Gap proportional to 1/N" is not what the data
  show either: the local exponent rises from 0.67 to 1.06.
* Equal total samples (R at the step where N x steps matches): 0.98M A 0.494, B 0.426, D 0.433, C 0.290; 1.97M
  A 0.587, B 0.516, D 0.516, C 0.403; 3.93M B 0.608, D 0.606, C 0.493; 7.86M B 0.687, C 0.591. First reach of
  0.9 R_pol: B at step 103 (6.75M draws), C at 55 (14.4M); of 0.95 R_pol: only C, step 84 (22.0M).
* R51.3 card and summary row fixed: T1d MARGINAL, rho 2.37 (1.70 with the score build charged in full; job 760095,
  `tests/P5.1_rowsample_r01/results/t1d_cont.json`), cross-machine confound (R51.3 steps 0.71-0.76 CPU-min on the
  node vs 0.44 interactive; the ladder ran interactively). Predicted PASS (p 0.8) missed.

## 2. Size-scaling question: generation batch (Operation B; Analogy Map extended with B5)
Analogy Map additions (problem.md): B2 gained "statistical learning theory (ridge regression sample complexity)";
new B5 "how the noisy measurements per update needed to stay on the noise-free path grow with size", sources:
ridge/kernel regression sample complexity (tried R51.10), Monte Carlo support coverage / effective sample size
(used as R51.8's exponential reference and the duplicate-aggregation T0), critical batch size in deep learning
(McCandlish et al. 2018; untried).

| route | code | one line | outcome |
| :--- | :--- | :--- | :--- |
| R51.8 | `nscale` | R51.1's optimiser at S10 (5x2) and S12 (4x3), each against its exact-sum (N = infinity) run; N_needed(V) at 5% shortfall; compare with V^3 and with the effective-support growth; Born arms give the SR/MinSR comparator | LIVE, T2 unit 1 prepared, gate PASS (2.5 + 8.0 CPU-h) |
| R51.9 | `phiscale` | one-step fraction of the exact step's gain at each size's start as a proxy for the multi-step need | KILLED at T0 |
| R51.10 | `kappa` | cross-field (ridge-regression learning theory): exact noise-to-signal ratio kappa of the SR step from one state predicts the size trend of N_needed | LIVE, T1 partial (kappa at both starts), scored on R51.8 |

### Gates
* R51.8. G0: no measurement of samples-to-representable-minimum versus size for fermionic NQS (round-01 searches);
  SR floors are known (Wang & Liu 2026), not their size scaling: PASS as a measurement. G1: no claim beyond the
  sizes tested: PASS. G2: the statistic is relative to each size's own exact run, which no cheap state can game;
  the trivial expectation (N proportional to P) is one of the outcomes it can confirm: PASS. G3: doped,
  non-stoquastic, projected sector retained: PASS (small clusters, L39/L40 caveats in the rule). G4: per-sample cost
  polynomial; N_needed(V) is what it measures; over 1.2-1.4 in V it can exclude support-like growth, not prove a
  polynomial; this route is itself the three-size Gate 4 check (third size in unit 2): PASS.
* R51.9. Killed at T0 (Gate 2 / T0): one-step quantities miss the multi-step need by 2-4x in both directions at
  the only calibrated size (one-step rate 0.0078 vs sustained 0.0020-0.0025; one-step phi = 0.45, 0.57, 0.82 at
  N, 4N, 16N from the SR start, so N_phi(0.9) > 262,144, while the multi-step run reached 95% of R_pol at an
  interpolated 115,000). Card written.
* R51.10. G0: no use of ridge sample-complexity constants to predict VMC sample needs found: PASS. G1, G3: PASS.
  G2: it is scored against measured needs at two sizes, not against itself: PASS. G4: one exact metric build per
  size, sampled at 24 sites: PASS.

### T0 arguments (no compute)
1. Noise model (above): the need is set by the step's direction fidelity along the path; ridge theory makes that
   polynomial (effective dimension x residual variance / signal) unless the residual's tails grow with size.
2. Duplicate aggregation: unique draws saturate at the effective support exp(H[p]); measured 3035 (12 sites) and
   781 (10 sites) at the starts, x3.9 per 2 sites; extrapolated to about 1e7 at 24 sites, so N/n_unique -> 1 and
   R51.1's 0.8 CPU-min/step at 262,144 draws (31,700 unique) is a small-cluster artefact; the honest 12-site cost is
   the un-aggregated 2.2 CPU-min/step (in-job). Answer to "does dedup's advantage vanish at larger Hilbert spaces":
   yes, by this argument; nothing needs to be run for it.
3. Comparator: the existing arms do not give the SR/MinSR comparison - all are guided SR at a fixed step. Plain SR
   exists only as R21's held-out run (R 0.195 at 15 steps, 16,384 samples) and as the SR-trained start (1024
   samples x 1000 steps, R = 0 by definition). At N ≥ P MinSR's step equals SR's with the same shift (Woodbury), so
   "MinSR" is a separate regime only at N < P; R51.8 includes Born arms at 1024 < P = 1832 at S10 for that.

### T1 measurements this round (interactive)
* `ns_run.py` (new; generalises nf_run.py to any open strip, exact arm, Born or guided draws, duplicate
  aggregation always on): reproduces the 12-site start energy -7.805526273788163 exactly.
* `ns_run.py --info`: S12 sector 627,264, 157,216 representatives, P 2637, exp(H) 3035, unique guided draws 6.1k /
  14.8k / 31.3k / 58.2k at 16k / 64k / 256k / 1M; exact rows 1.09 CPU-min. S10 sector 44,100, 11,100
  representatives, P 1832, exp(H) 781, unique guided 2.9k / 5.1k / 7.4k / 9.2k; exact rows 0.048 CPU-min. Lowest
  (+,+) levels (any spin): S12 -7.828637 (the S = 0 E0 -7.827899 is higher), S10 -6.550697; start eps 3.0e-3 (S12)
  and 5.5e-4 (S10) against them.
* S10 start: `vmc.py --mode train --Lx 5 --Ly 2 --Ne 8 --pg 1,1 --iters 1000` with the 12-site pilot's hypers,
  1 thread: 9.9 CPU-min, final exact E -6.547116 (last-25% average). `logs/start_5x2.txt`.
* Pre-flights (CPU-min/step, interactive): S12 exact 1.67 (14.5 GB in kappa.py, 11 GB in ns_run), S12 Born 262,144
  0.28; S10 exact 0.075, guided 1M 0.066, guided 16k 0.032, Born 1k 0.020. Step 1 at S12 from the start: exact
  dE = -2.499e-3 (0.112 of the gap); Born 262,144 -1.713e-3.
* `kappa.py` (R51.10): S12 kappa_guided 1.24e5, kappa_born 1.48e9, d_eff 999, xSx 0.0434 (2.7 CPU-min); S10 1.08e5,
  2.04e8, 526, 0.0028 (0.13 CPU-min). Predicted N_needed ratio (12 over 10 sites): 1.15.

### Pre-registered rules and gated jobs
* `tests/P5.1_size_scaling_r04/RULE.md` (R51.8): POLY-CONSISTENT if r ≤ 1.728, SUPPORT-LIKE if r ≥ gH = 3.89,
  AMBIGUOUS between, UNRESOLVABLE if gH < 2.59; comparator GUIDED-FEWER / MIXED / NO-GAIN at 2 / 1.25; what it does
  and does not show, and the representativeness clause, are in the rule.
* `RULE_kappa.md` (R51.10): calibrate on S12, predict S10; PASS within x1.5, KILLED beyond x2.5.
* `submit_unit1_s10.sbatch` + `voi_unit1a.md`: cap 2.5 CPU-h; leaf committed 5.7 / 300; GATE: PASS.
* `submit_unit1_ref12.sbatch` + `voi_unit1b.md`: cap 8.0 CPU-h; leaf committed 5.7 / 300; GATE: PASS.
  (10.5 CPU-h together; each job is below the 10 CPU-h auditor threshold.) Not submitted.
* Deferred (unit 1c, only if needed): S12 Born 16k/64k/256k (comparator at 12 sites; about 2.5 CPU-h) and S12 guided
  1,048,576 (bracket, about 2.5 CPU-h). Unit 2: S14 (7x2, 9.0M configurations; exact metric about 22 CPU-min per
  build) and S12' (6x2) - roughly 20-40 CPU-h, auditor needed.

## Compute this round
Interactive: S10 start 9.9 CPU-min; pre-flights 3.5 + 0.7 + about 1.0 (S10 four arms); info 1.5; kappa 2.8; scoring
seconds: about 0.33 CPU-h. No job submitted.

## Proposed library entries (orchestrator owns library.md)
* Pre-flight timing on the interactive node under-predicted in-job time per step by 1.6-1.8x (jobs 759974, 761860,
  760095 at 0.71-0.76 vs 0.44 CPU-min/step), truncating one capped arm; set caps from in-job rates or a short
  in-job probe, and never compare rates measured on different machines (R51.3 T1d's confound).
* A one-step counterfactual measured at another optimiser's states overstated the sustained fixed-step rate 3-4x
  (0.0078 vs 0.0020-0.0025 per step), and a one-step fidelity at the start overstated the multi-step sample need by
  more than 2x: one-step quantities do not transfer to multi-step needs without calibration (R51.1, R51.9).
* R38's 0.51 "plateau" was its held-out step choice, not a floor: a fixed short step continued to 0.732 of the gap
  at 16x samples (R51.1), so a plateau verdict must be checked against a fixed-step control before it is read as an
  estimator floor.
* (new) At a fixed step count, the shortfall of fixed-step sampled SR fell as N^-0.87 and a stationary noise-ball
  model mis-predicted it by 2.7x; at equal total samples, fewer samples per step won at every budget tested
  (R51.1): more steps beat bigger batches in this regime.
* (new) Aggregating duplicate draws is an exact rewrite that cut a 12-site sampled step 2.7x in time and 5x in
  memory, but its saving is bounded by the effective support exp(H[|ψ|²]) (3035 at 12 sites, 781 at 10) and
  vanishes once exp(H) >> N; small-cluster per-step costs measured with it do not extrapolate.

## 3. Compute auditor: CHANGES REQUIRED for R51.8 unit 1 (applied, nothing submitted)
Auditor items (summarised by the orchestrator): (1) same-geometry primary ratio S10 (5x2) vs S12' (6x2), series with
S14 (7x2); 4x3 demoted to a geometry check; (2) cheap T1s first (exact kappa at S12' and S14, measured pre-flights,
trained S12' start), and R51.10 as written is not independent of R51.8 (m = r / 1.15) - recast as a 3-size kappa
trend; (3) stage spending using R51.1's bound on the 4x3 need, optionally drop the 1M rung; (4) 1b refused until
ns_run.py saves a resumable state on --cpu-cap.

What was done:
* RULE.md revision 1 (before any arm ran): primary r' = N_needed(S12') / N_needed(S10), three-size series with S14
  (POLY-CONSISTENT-3 if N_needed(S14)/N_needed(S10) ≤ 2.744); 4x3 is a geometry check bought only if kappa or the
  ladders show geometry matters. The 4x3 bound is stated as conditional: (65,536, 262,144] holds if the 4x3 exact
  arm's R_plat lies in [0.732, 0.770] (B's s ≥ 0.071, C's s ≤ 0.05); an exact-limit fit through A, B, C gives
  Delta_inf = -0.038 (R_ref about 0.785), in which case C's s = 0.067 and the bound moves up. Staged thresholds:
  N_needed(S10) ≥ 151,700 or censored above 262,144 -> r ≤ 1.728; ≤ 16,850 -> r ≥ 3.89. The 1M rung is dropped.
* RULE_kappa.md revision 1: old test withdrawn as a verdict (not independent); recast as KAPPA-POLY / SUPPORT /
  AMBIGUOUS on kappa(6x2)/kappa(5x2) and kappa(7x2)/kappa(5x2), a prediction recorded before R51.8's ladders, killed
  as a predictor by a class disagreement where R51.8 is decided; geometry flag kappa(6x2)/kappa(4x3) outside 1.5x.
* ns_run.py: theta and history saved together atomically (state.npz, os.replace) after every step; clean stop before
  a step that would pass --cpu-cap, or after the current step on SIGUSR1/SIGTERM; status DONE / TRUNCATED in
  run.json; cumulative CPU across resumes. Test at S10 (guided 16k): capped at step 2, resumed to step 4; energies
  at steps 1-4 identical to an uninterrupted run (-6.547307, -6.547400, -6.547471, -6.547500). Sbatch scripts use
  `--signal=USR1@600` (not B:) and append output with >> (tee would die on the signal).
* kappa.py: chunked two-pass version (S14 cannot hold 2.25e6 x 3590 rows); reproduces the one-pass S10 values to
  1e-11 (kappa_born 204426537.6348 vs .6353; kappa_guided 108115.48360).
* Interactive probes (threads = 1 exported, ulimit -t 600 each): vmc.py 20-iteration training at 6x2 48 CPU-s
  (1.2 s/iteration), at 7x2 48 CPU-s (1.7 s/iteration): 1000-iteration starts cost 20 and 29 CPU-min, so both are
  gated jobs. 6x2 (probe start): 156,816 representatives, P 2637, exp(H) 3864, eps 3.5e-2 against the lowest (+,+)
  level (untrained probe); exact step 1.69 CPU-min (11 GB), guided 262,144 0.425 CPU-min/step (32.9k unique).
  7x2 (probe start): sector 9,018,009, 2,255,421 representatives, P 3590, Hr nnz 7.6e7, exp(H) 19,360, one exact
  energy 1.24 CPU-min, rows 1.02e-5 CPU-min each (23.0 min per exact pass), peak 16.3 GB. A first 7x2 probe was
  killed by this session's 20 GB cgroup limit while a 6x2 pre-flight ran in parallel (that 6x2 exact pre-flight lost
  its second step to the same kill); rerun alone it completed.
* Jobs prepared (not submitted) and gate outputs (`compute_gate.py check ... --pre-audit`):
  - `submit_unit1_s10.sbatch` + `voi_unit1a.md` (staged decision stated): cap 2.2 CPU-h; GATE: PASS; digest
    26e826d7d689 (the auditor re-reviews 1a).
  - `submit_quick_s12p.sbatch` + `voi_quick_s12p.md` (6x2 start + info + exact kappa): cap 1.0 CPU-h [quick note];
    GATE: PASS; digest 2397cf0b3dfc.
  - `submit_kappa_s14.sbatch` + `voi_kappa_s14.md` (7x2 start + info + exact kappa, 32G): cap 3.2 CPU-h; GATE: PASS;
    digest d17f0257df99.
  - `submit_unit1_ref12.sbatch`: demoted to the geometry check, resumable/signal fix applied, not re-gated
    (voi_unit1b.md must be rewritten for the geometry-check decision before any submission).
* Cards R51.8, R51.10 and the summary's portfolio and Next step updated. Interactive compute for this part: about
  0.15 CPU-h (probes and tests).
