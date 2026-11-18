# P5.1 round 5 — generation batch independent of the running jobs (worker, 2026-11-17)

Task (orchestrator): generate 5–7 routes, ≥2 cross-field, aimed at the scalable gap (everything that worked needs
exact sums or brute samples), gate them, run cheapest falsifiers (T0 first, T1 ≤10 CPU-min interactive, bigger →
sbatch + voi + gate check, not submitted). Must not depend on R51.8's 5x2 ladder (763968) or R51.10's κ jobs
(763902/763903). Coordinator note mid-round: do not edit summary.md, problem.md or cards R51.1–R51.10 (critic fixes
being applied); critic findings: R38's plateau was not caused by held-out step selection; no arm has compared
against SR/MinSR at equal total samples; R51.5 reopened, R51.7 shelved.

Read first: library index, `bs.py context P5.1 --role worker`, cards R51.7, P4.1 R15, R33, R34, R35, R37, R38;
kill_index grep for control variate / SVRG / surrogate / MinSR.

## Routes generated (8 cards, R51.11–R51.18)

| Route | Kind | Outcome |
| :--- | :--- | :--- |
| R51.11 `qopt` κ-optimal sampling distribution | cross-field (optimal subsampling, statistics) | LIVE: T1 MARGINAL, T1b PASS, T2 prepared |
| R51.12 `sketchlev` sketched ridge leverage for sampled metric rows | cross-field (randomised NLA) | KILLED T0 |
| R51.13 `svrg` SVRG/STORM anchor control variate on the direction | cross-field (stochastic optimisation) | KILLED T1 |
| R51.14 `levreuse` reweighted reuse of high-leverage configurations | Operation A of P4.1 R34 | KILLED T0 |
| R51.15 `adaptN` noise-scale-driven samples-per-step schedule | cross-field (dynamic sample size), Op. A of R51.7 | LIVE: T1 PASS (optimistic bound) |
| R51.16 `cmp` plain SR / MinSR comparator at equal total samples | the Question's comparator | LIVE: T2 prepared |
| R51.17 `reshuffle` without-replacement pool reshuffling | Operation B (cross-field, random reshuffling) | KILLED T0 |
| R51.18 `storm-sf` STORM recursion on S and f | translation repair of R51.13 | KILLED T1 |

## G0 searches (quick WebSearch, 2026-11-17)
* "VMC optimal sampling distribution ... SR importance sampling NQS": Misery, arXiv:2507.05352 (MLST Jan 2026):
  adaptive |ψ|^α overdispersed sampling tuned to gradient variance; Neural importance resampling arXiv:2507.20510.
  No SR-direction-optimal sampler found.
* "SVRG variance reduction stochastic reconfiguration VMC NQS control variate previous parameters": nothing.
* "adaptive sample size schedule VMC NQS": nothing direct (sample-size requirement analyses of first vs second
  order methods exist; no schedule).

## T0 arguments
* R51.12: in sampled VMC every drawn row is needed for the force, so metric row selection saves only the Gram
  (≤ about 10–15% of a step; local energy 85–90% at 24 sites, P4.1 R34); leverage of N sampled rows costs O(NP²),
  the Gram's cost; metric from fewer rows breaks S–f cancellation (R35, R37; L59). KILL.
* R51.14: R34's cost bound (reuse saves only drawing, 5–8%) plus R51.13's measured weight growth (χ² 1.7e-4, 0.074,
  1.7e4 at 1, 5, 20 steps; max weight 1.9e10 at 20). KILL.
* R51.17: accumulated K-step error of reshuffled mini-batches = K × pool error, variance Kσ²/N, equal to fresh
  draws; reweighting adds variance; no compute saving (R34). KILL.

## T1 / T1b results (all interactive, OMP/OPENBLAS/MKL=1, ulimit -t 600; `tests/P5.1_r05_gen/`)

### R51.15 adaptN (RULE_adaptN.md, `envelope.py`, stored R51.1 data, 4x3)
Single arms, draws to 0.80/0.90/0.95 R_pol: A —/—/—; B 3.67M/6.75M/—; C 8.39M/14.4M/22.0M; D 3.70M/—/—.
Bin-by-bin oracle envelope: 1.69M (2.17×), 4.80M (1.41×), 12.06M (1.83×) → PASS (≥1.5 at 0.95).
Diagnostics (not scored): best monotone A→B→C schedule (`envelope_mono.py`) 2.57M (1.43×, breaks 0.58/0.59),
5.46M (1.24×, 0.58/0.67), 12.14M (1.81×, 0.58/0.68). Second size, R51.8's finished 5x2 guided arms, targets
relative to its exact arm's R at step 120 (`envelope_s10.py`): monotone gains 1.27, 1.07, 1.94. The bin-by-bin switch
alternates arms below $R\approx0.5$, i.e. it picks noise; the 0.95 gain arises where small-N arms are censored at 120 steps.

### R51.11 qopt T1 (RULE_qopt.md, `kq.py`)
κ (draws for unit relative step error) at the starts. 5x2 (11,076 reps, 0.9 CPU-min with anchors): born 2.04e8,
guided 1.07e5, |ψ|^{2α} α=0.5 2.25e4, 0.6 8.83e4, 0.7 4.74e5, 0.8 3.17e6, 0.9 2.43e7; mix0.5 4.43e4, mix0.35
6.25e4; residual q∝p(0.1ē+|e|) 6.37e4, p(ē+|e|) 1.15e5; amplitude-only optimum 10 bins 4600, 40 bins 2200;
oracle 977; Born top-0.1% share 0.934. 4x3 (156,967 reps, `logs/kq_S4x3_q.txt`): born 1.48e9, guided 1.24e5,
α=0.5 2.63e4, 0.6 1.12e5, 0.7 7.74e5, 0.8 7.74e6, 0.9 9.90e7; mix0.5 5.16e4, mix0.35 7.28e4; residual 2.40e4,
3.02e4; amp-only 10 bins 13,509, 40 bins 5402; oracle 1353; top-0.1% share 0.993.
r_amp 48.6 / 23.0, r_orc 109 / 92 → MARGINAL (≥3 at both sizes but falling).
A first 4x3 run crashed at an assertion (one representative had exactly zero amplitude at the anchor only) after
the κ part, whose output was lost through `tail`; `kq.py` was patched (common representatives, early save) and rerun.

### R51.11 qopt T1b (RULE_qopt_T1b.md, `storm.py --a 1.0 --qmix b`, 5x2, N 4096, 120 steps, seeds 9000+n)
R_plat: b=1.0 (q∝|ψ|) 0.3969 (3.6 CPU-min), b=0.5 0.3501 (2.7); foils guided 4096 0.2740, guided 16384 0.4026
(stored R51.8 arms; only R_plat read). PASS (≥0.3383). Training gain follows the κ order (4.7× and 2.4× lower κ).

### R51.13 svrg (RULE_svrg.md one-step bound; RULE_storm.md multi-step)
One-step, exact anchor, guided draws, 5x2: ρ(1)=0.044 (c_opt 0.84), ρ(5)=0.448, ρ(20)=0.991 → MARGINAL
(not KILL: ρ(1)≤0.25; not PASS: ρ(5)>0.25). 4x3 k=1 run hit the 600 s CPU limit after the representative-set
recompute (`logs/kq_S4x3_k1.txt`); not needed, the verdict was fixed by 5x2.
`storm.py` identity check (a=1, N 4096, 3 steps) reproduced ns_run.py's stored energies to 15 digits.
STORM on the direction, N 2048, a 0.1: R_plat 0.2112 (1.8 CPU-min) vs plain 4096 0.2740 (G = 0.1286) → KILL.
### R51.18 storm-sf (RULE_storm.md addendum, written before the run)
Recursion on S and f, N 2048, a 0.1: diverged before step 10 (recursed S indefinite, metric length negative, NaN
sampling distribution; `logs/storm_N2048_a0.1_sf.txt`) → KILL. Translation repair failed; both cards record why.

## Prepared job (not submitted): `tests/P5.1_r05_gen/submit_cmp.sbatch` + `voi_cmp.md`
Task 0: born 16k, 824 steps. Task 1: born 65k τ 0.005 and τ 0.01 (206 steps), born 262k (51), guided 16k continued
from R51.1 A's step-120 θ (292 steps). Task 2 (R51.11 T2): q∝|ψ| 16k (412 steps), 65k (103). Rules RULE_cmp.md,
RULE_qopt_T2.md; scorer score_cmp.py. Pre-flights: born 16k 0.135, |ψ| 16k 0.20, |ψ| 65k 0.39 CPU-min/step.
`compute_gate.py check --pre-audit`: cap 3 × 5.33 h = 16.0 CPU-h; leaf committed 11.5/300; GATE: PASS PENDING AUDIT,
VOI digest da2f6097605b. (An earlier two-task version checked at 10.7 CPU-h, digest a05eab8c6164, superseded.)
The guided 16k continuation also gives R51.15 its Sharpest-Gap answer (does a long small-N run reach 0.95 as cheaply
as the schedule?).

## Compute this round
Interactive about 22 CPU-min: kq 5x2 0.9; kq 4x3 crashed run about 4, κ rerun about 4, k=1 run 10 (CPU limit); storm 1.8 + 0.1
(diverged) + identity 0.2; T1b 6.3; pre-flights 0.6 + 1.0; envelope scripts negligible.

## Proposed library entries
1. Row subsampling of the metric pays only where rows are computed for the metric alone; in sampled VMC every drawn
   row feeds the force, so the saving is bounded by the Gram share (R51.12; with R51.2's B4 timing).
2. Variance-reduced recursions from finite-sum SGD (STORM on the direction; STORM on S, f) lost to fresh draws at
   equal CPU on a 10-site state: direction recursion carries ridge-solve errors forward; metric recursion loses
   positive definiteness at N < P (R51.13, R51.18; one seed, one a).
3. Without-replacement reuse / reshuffling gains only relative to a finite pool; for a population objective at equal
   per-sample cost it matches fresh draws at first order (R51.17).
4. (candidate, after T2) For SR on determinant NQS, κ (step-error second moment) ranks samplers correctly: q∝|ψ| has
   4.7× lower κ than 0.8|ψ|²+0.2|ψ| at 10 and 12 sites and matched 4× the samples in 120 steps at 10 sites.

## For the summary (not edited this round; coordinator's instruction)
Portfolio rows and Next step are in the worker's return to the orchestrator.

## Audit revision (compute auditor CHANGES REQUIRED on digest da2f6097605b; applied 2026-11-17, before any data)
1. Split: R51.16 keeps `submit_cmp.sbatch` + `voi_cmp.md`; R51.11 T2 has its own `submit_qopt.sbatch` + `voi_qopt.md`
   and `score_qopt.py` (needs cmp task 0 for T_guided); shared statistic in `score_common.py`.
2. MinSR at N < P: scoped out (RULE_cmp.md rev. 1 (a)); every arm has N at least P, where MinSR = SR. An arm with
   N up to 2048 at 4x3 would need about 7,000 steps to spend the budget; moved to a larger-P instance, later unit.
3. Dropped Born 262k and Born 65k tau 0.005; p(GUIDED-FEWER) raised to 0.75 (κ ratio 1.2e4 at 4x3 vs 1.9e3 at 5x2,
   where Born needed about 9.4x guided's samples by the auditor's reading).
4. Sustained crossing (trailing 20-step mean of R reaches R*) in both rules. R51.1 re-scored (`score_common.py`,
   `results/rescore_R51.1_sustained.json`): at 0.80/0.90/0.95 R_pol, A never/never/never; B 4.33M/7.41M/never;
   C 11.0M/17.3M/24.6M; D 3.85M/never/never. Budgets re-sized: Born 14.8M (= 2 T_B), guided continuation 332 steps
   (A + continuation = 7.41M), q arms 5.93M (= 0.8 T_B).
5. Wall times from R51.1's in-job rates (0.36 at 16k, 0.27 at 65k; q ~ |ψ| scaled by 2.4 from interactive);
   guided continuation first in task 0; per-run --cpu-cap = each run's share, summing to the task's bound.
6. Signal: `--signal=B:USR1@600`, python in the background, trap forwards USR1, later runs skipped; tested with
   `tests/P5.1_r05_gen/sigtest.sh` (child saved on USR1, next run skipped).
7. Card R51.16 now states the 14.3 CPU-h cap and digest.
Gate checks (`--pre-audit`): cmp cap 2 x 7.17 h = 14.3 CPU-h, PASS PENDING AUDIT, digest 21b01ca1b458;
qopt cap 5.8 CPU-h, PASS, digest f2e46c43f548.
