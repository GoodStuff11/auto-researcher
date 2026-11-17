# P4.1 R42: pre-registered rule (practitioner comparison at 24 sites, equal compute)

Written 2026-11-17, round 27, after `r42.py --mode check` (logs/check_local.txt) and `--mode probe`
(logs/probe_local.txt), which only verify code and time steps. Written **before any training run**. Card:
`leaves/P4.1/cards/R42.md`.

## Framing: what this decides, and what it does not

This answers a **narrower question than expressivity**: *at equal effort, does the cheap exact-S network do as well in
practice?* A practitioner choosing between the charge-only, exactly spin-projected network (`che`) and the usual
spin-reading network (`nnbf`, which cannot be exactly spin-projected at 24 sites) trains one of them with the
same sampled optimiser for the same CPU time and keeps the lower energy. This run measures that choice. It does
**not** measure what either network could reach if converged (that is the leaf's expressivity question, which at 24
sites no available optimiser can answer: R38, R21, R15b). Neither arm is converged at the end; by design they are
scored at equal compute (L35), so the result is an optimiser-and-ansatz comparison (L33) and is labelled as such.

## Instance and arms

Open 8x3 strip, N_e = 20 (N_up = N_dn = 10), U = 8, t' = -0.25, point-group projection P_G with characters (+,+) on
both arms (the DMRG reference irrep). Both are `vmc.Model` at H = 16, P = 9711, identical initialisation for a
given seed (tight-binding Fermi sea, Gutzwiller Jastrow, small random backflow weights).

| arm | backflow input | spin symmetry | cost per step (measured, 1 core, 1024 samples) |
| :-- | :-- | :-- | :-- |
| `che` | charges (doublon, holon) | exact S = 0 (Che closed form) | 0.166 CPU-min |
| `nnbf` | spins (n_up, n_dn) | S^z = 0 only; no spin projection (exact P_S costs the spin-orbit sum, exponential); spin-flip parity not imposed (L56: gives a parity only) | 0.090 CPU-min |

## Optimiser (identical for both arms)

Guided sample-space SR: the R38 mixture q = 0.8 |psi|^2 + 0.2 |psi| (normalised), sampled by Metropolis on
|psi|^2 + lam |psi| with lam adapted every step so the |psi|^2 mass fraction is 0.8; self-normalised weights
|psi|^2/q; vmc.py's Woodbury SR (diag shift 1e-3, max step 0.5); lr 0.05 for the first half of the CPU budget, then
cosine to 0.005; Polyak average over the last 25% of the CPU budget. R38's held-out step-length ladder is **not**
ported: it needs four extra local-energy passes per step (the local energy is 85-90% of a step, R34), which would
quarter the steps. **Porting cost** of the guided sampler: about 60 lines in `r42.py` (Metropolis target and weights),
checked on 3x3 (N_e 6) against exact energies (z = +0.25, +0.27). R38's 2x step efficiency was measured at 12
sites with i.i.d. exact draws; at 24 sites with Metropolis it is unmeasured, and both arms share whatever it gives.

## Budget and seeds

Each (arm, seed) trains for **180 CPU-min on one core** (equal wall time); then scoring, not counted in the budget.
Expected steps: `che` about 1080, `nnbf` about 2000. Stage 1: seeds 0 and 1 (4 tasks). Stage 2 (seed 2, both arms,
a new gated job) only if the stage-2 clause below fires.

## Reference energy

No converged 24-site reference exists (687302 was cancelled). The best available is the chi <= 1000 DMRG in
`../P4.1_cheap_su2_scaling/dmrg_irrep/dmrg_S8x3_N20_Sz20_U8_tp-0.25.json` ((+,+) characters 0.999):
E_last (chi = 1000) = -16.58860, E_extrap = -16.60459 +- 0.01599. Every ratio is computed at **both ends of the
bracket** E_ref in {E_last = -16.58860, E_extrap - sigma = -16.62058}. (Earlier leaf text called -16.60459 the
chi = 1000 energy; it is the extrapolation.) DMRG with S^z = 0 does not fix S; the ground state is assumed singlet.

## Scores

**Lead number** (reported first in every write-up, unscored): the reference-free difference E_che - E_nnbf per seed
and its mean, with errors. Also reported, unscored: each arm's guided step energy against step count (every step
is logged), and `nnbf`'s energy at `che`'s final step count (mean of `nnbf`'s logged step energies over a 100-step
window centred there), which separates the per-step comparison from the equal-CPU one.

Per seed s: rho_s = (E_che,s - E_ref) / (E_nnbf,s - E_ref), from the Polyak-averaged parameters, 131,072 |psi|^2
samples, chain-blocked errors. rho = median over seeds (mean of two at stage 1). Reported, not scored: <S^2> of
`nnbf` (32,768 samples), the reference-free difference E_che - E_nnbf, steps taken, and the same rho at half
budget (raw parameters, 32,768 samples).

| verdict | condition (must hold at both reference ends) | what it means |
| :-- | :-- | :-- |
| CHE-NO-WORSE | rho <= 1.10, and every rho_s <= 1.30 | at this budget the cheap exact-S network is at least as accurate as the spin-reading one |
| CHE-LOSES | rho >= 1.50, and every rho_s >= 1.20 | the charge-only restriction costs accuracy even in practice at 24 sites |
| INTERMEDIATE | rho strictly between, classes agree | a loss of 10-50%, reported as a number |
| UNSETTLED | the class differs between the two reference ends, or the seed condition fails | no change to the Answer beyond the numbers |

**Stage-2 clause**: seed 2 is requested (new gated job, cap 9 CPU-h) only if the two stage-1 seeds give different
classes, or either rho_s is within 0.15 of 1.10 or 1.50 at either reference end. Otherwise the stage-1 verdict is
final. **BUDGET-DEPENDENT** label: if the half-budget class differs from the full-budget class, the verdict is stated
"at 3 CPU-hours per run" and no claim about larger budgets is made.

**INVALID-EARLY** (fail-fast, every task, auditor change 1): at 20 CPU-min of training the task aborts, writes
`results/<arm>_seed<s>_INVALID-EARLY.json` and exits if the median guided mass fraction F over the steps so far is
outside [0.7, 0.9], or the median effective sample size is below 512, or any step energy is non-finite. Any
INVALID-EARLY task makes the stage INVALID.

**INVALID** (changes nothing): a `che` sample with |S^2_loc| > 1e-6; the median guided mass fraction F over the
last 25% of steps outside [0.7, 0.9], or median effective sample size below 512; any task cut by Slurm before
scoring (INCOMPLETE; resubmission resumes from its checkpoint); a final Polyak energy above the half-budget energy
by more than 3 combined sigma (divergence).

## What each verdict does to the leaf

**BUDGET-DEPENDENT overrides every class below** (auditor change 2): if the half-budget class differs from the
full-budget class, the numbers are reported only, and neither the ANSWERED proposal nor the EXHAUSTED exit is
triggered, whatever the full-budget class.

* CHE-NO-WORSE: the Answer gains "at 24 sites, trained by the same sampled optimiser for the same CPU time (3 CPU-h
  per run, one core), the cheap exact-S network was as accurate as the unprojected spin-reading network (whose
  <S^2> was x)". Any Answer wording for this class must contain "at 3 CPU-h per run, one core". The
  "What it would take" item "the like-for-like comparison at 24+ sites" is met for practice; the expressivity part
  stays open. The orchestrator may then propose an ANSWERED exit for the practical question.
* CHE-LOSES: the Answer's accuracy clause becomes a 24-site, equal-compute statement: "No in practice: even
  unprojected, the spin-reading network beat it by rho at equal CPU". The leaf goes to its EXHAUSTED exit with no
  further charge-only accuracy work.
* INTERMEDIATE: the number is added to the Answer; no change of next step.
