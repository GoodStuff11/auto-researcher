# P4.1 R43: pre-registered rule (spin-flip-even `nnbf` against R42's `che`, 24 sites, equal compute)

Written 2026-11-17, round 29, after `r43.py --mode check` (logs/check_local.txt) and `--mode probe`
(logs/probe_local.txt), which only verify code and time steps. Written **before any training run**. Card:
`leaves/P4.1/cards/R43.md`. Asked for by the per-leaf critic of round 28 (Major 2): R42's comparator `nnbf` ended
with <S^2> 1.1-1.2, so its loss may be symmetry restoration alone (library L1).

## Question

Does `che` (charge-only backflow, exact S = 0) still beat the spin-reading backflow once the spin-reading network is
given the cheap symmetry a practitioner would add, at equal CPU? The baseline is `nnbf_even`:
psi_even(x) = psi(n_up, n_dn) + psi(n_dn, n_up), the global spin-flip symmetrisation of R42's `nnbf`. At S^z = 0
with N_up = N_dn this keeps only even S (S = 0, 2, 4, ...), removing the S = 1 and S = 3 channels; it costs 2 amplitudes
(measured 1.98x `nnbf` per step, 1.01x `che`), so at equal CPU it gets about `che`'s step count. It does not remove
S = 2, 4 (exact projection to S = 0 stays exponential for a spin-reading network, commutant lemma). Not tested:
a spin penalty, other sectors, other sizes.

## Instance, arm, optimiser, budget

Everything is R42's (`../P4.1_cheap_su2_r42/RULE.md`): open 8x3, N_e = 20, U = 8, t' = -0.25, P_G (+,+), H = 16,
P = 9711, same initialisation per seed (`nnbf_even` uses `nnbf`'s parameters and init, so at a given seed its start
is the flip-symmetrised R42 `nnbf` start), guided SR (R38 mixture, F target 0.8, Woodbury SR, shift 1e-3, max step
0.5, lr 0.05 then cosine to 0.005 in CPU fraction, Polyak average over the last 25% of CPU), **180 CPU-min on one
core** per (arm, seed), seeds 0 and 1, 1024 chains. `r43.py` is a copy of `r42.py` with the new arm only (`FlipModel`),
plus a SIGUSR1 checkpoint-and-exit.

**`che` is not rerun.** R42's `che` results (job 759907, `../P4.1_cheap_su2_r42/results/che_seed{0,1}.json`) are
used as they are: the `che` code path is byte-identical in `r43.py`, the scoring is identical, and the job is pinned
to the node R42 ran on (kim-cpu-01, all four R42 tasks) to avoid the cross-machine step-rate confound the leaf met
in P5.1 R51.3 T1d. Rerunning `che` would cost 7.6 CPU-h for no new information. Comparability check (reported, not
scored): `nnbf_even`'s in-job CPU-min per step against 2 x R42 `nnbf`'s 0.143 (expected ratio 1.0, from the
interactive 1.98x); if outside [0.8, 1.25], the write-up states that the step rates differ from R42's and by how much.

## Reference energy

R42's DMRG bracket, both ends: E_ref in {E_last = -16.58860, E_extrap - sigma = -16.62058} (chi <= 1000,
`../P4.1_cheap_su2_scaling/dmrg_irrep/dmrg_S8x3_N20_Sz20_U8_tp-0.25.json`); ground state assumed singlet.

## Scores

**Lead number** (reported first, unscored): E_che,s - E_even,s per seed (reference-free), and E_even,s - E_nnbf,s
(what flip symmetry bought at equal CPU).

Per seed s (`che` seed s paired with `nnbf_even` seed s; same init seed):
rho'_s = (E_che,s - E_ref) / (E_even,s - E_ref), from the Polyak-averaged parameters, 131,072 |psi|^2 samples,
chain-blocked errors (R42's numbers for `che`). rho' = mean over the two seeds. Reported, not scored: `nnbf_even`
<S^2> (32,768 samples; even-S content only), steps, the same rho' at half budget (raw parameters at 90 CPU-min,
32,768 samples, against R42's `che` half-budget energies), `nnbf_even`'s energy at `che`'s final step count.

| verdict | condition (must hold at both reference ends) | what it means |
| :-- | :-- | :-- |
| CHE-STILL-WINS | every rho'_s <= 0.80 | `che` beats the spin-filtered network by at least 20% of the gap at 3 CPU-hours per run |
| CONTAMINATION-EXPLAINS | rho' >= 1.00 and every rho'_s >= 0.90 | R42's win was symmetry restoration: a spin-filtered network does as well or better |
| MARGINAL | rho' > 0.80 and rho' < 1.00, classes agree at both ends | `che`'s practical lead over a spin-filtered network is under 20% of the gap |
| UNSETTLED | the class differs between the two reference ends, or no row's seed condition holds | numbers only |

Seed spread: R42's two seeds differed by 0.007-0.008 in rho and 0.002-0.008 in E; the 24-site run-to-run spread is
about 0.01-0.02 in E (R15b), i.e. about 0.05-0.1 in rho'. **Stage-2 clause**: seed 2 of `nnbf_even` and of `che` (R42
ran no seed 2; 2 tasks, 5.25 h + 4.5 h, new gated job, cap 9.75 CPU-h) is requested only if the two seeds give different
classes, or any rho'_s is within 0.05 of 0.80 or 1.00 at either end. Otherwise the stage-1 verdict is final.

**BUDGET-DEPENDENT** (overrides every class, as in R42): if the half-budget class differs from the full-budget class,
the numbers are reported, the leaf takes the MARGINAL action below, and any Answer wording says "at 3 CPU-hours per
run" with both reads.

**INVALID-EARLY** (fail-fast, every task): at 20 CPU-min of training the task aborts, writes
`results/nnbf_even_seed<s>_INVALID-EARLY.json` and exits 3 if the median guided mass fraction F so far is outside
[0.7, 0.9], the median effective sample size is below 512, or any step energy is non-finite. Any INVALID-EARLY task
makes the stage INVALID.

**INVALID** (changes nothing; rerun only through a new gate): median F over the last 25% of steps outside [0.7, 0.9],
or median ESS below 512; a task cut before scoring (INCOMPLETE: resubmit, it resumes from its checkpoint; a SIGUSR1
at 10 min before the limit writes the checkpoint and exits 4); a final Polyak energy above the half-budget energy by
more than 3 combined sigma (divergence); `nnbf_even` <S^2> per-sample estimator non-finite.

## What each verdict does to the leaf (ledger row 10)

* **CHE-STILL-WINS**: the 24-site practical statement stays, now against the fair baseline: the Answer may say "it
  ended with x of the energy error of the same network reading spins, with or without spin-flip symmetry imposed"
  (with "at 3 CPU-hours per run, one core" and the two-seed, one-strip scope). Row 10 closes; the leaf takes exit 3
  (EXHAUSTED at this budget) as the orchestrator decided, with the practical statement in the certificate.
* **CONTAMINATION-EXPLAINS**: the practical statement is replaced by "at 24 sites and equal CPU it did no better than
  the spin-reading network with spin-flip symmetry imposed (rho' = x); R42's lead came from the unsymmetrised
  network's spin contamination". The practical part of the Answer becomes a No/neutral; exit 3 with that wording.
* **MARGINAL** (and BUDGET-DEPENDENT): the Answer keeps the R42 numbers but adds "its lead over the same network with
  spin-flip symmetry was only rho' = x (under 20%)"; "half the energy error" may no longer stand alone. Exit 3.
* **UNSETTLED / INVALID**: numbers only; the R42 statement stays scoped as the critic narrowed it, with "an unrun
  spin-filtered comparison" named in Limits. Exit 3.
