# P4.1 round 29 (worker): R43 `flip-even-24`, the named falsifier for R42 (critic r28 Major 2)

2026-11-17. Task: prepare (not submit) a spin-flip-even `nnbf` arm in a copy of R42's harness, equal compute, seeds 0–1,
compared against R42's `che`. summary.md and existing cards not edited (an editor is applying the critic fixes).

## 1. What was built

* `tests/P4.1_cheap_su2_r43/r43.py`: copy of `r42.py` (R42's folder untouched). New class `FlipModel(V.Model)`:
  `logpsi` returns log|ψ_G(n↑,n↓) + ψ_G(n↓,n↑)| and its sign (exact cancellations to −inf, as vmc.py does for
  point-group zeros); `logderiv` maps NaN/inf gradients to 0 (point-group symmetry zeros are θ-independent, so 0 is the
  exact derivative; checked by finite differences). Because vmc.Model's `logpsi_batched`, `logderiv` and
  `local_energy` all call `logpsi`, the sampler, SR step, scorer and S² estimator all see ψ_even unchanged. Arm
  `nnbf_even` uses nnbf's parameters/init (P = 9711). Added SIGUSR1 handler: write checkpoint, exit 4 (INCOMPLETE).
  New check (d)/(e) in `--mode check`.
* Sign convention: Sector stores ψ as an (up, dn) matrix in up-first mode order; with N↑ = N↓ the swap is ψ → ψᵀ.
  Derivation: the operator c↑ ↔ c↓ gives ψ'(x) = (−1)^{N↑N↓} ψ(flip x) and acts on |S, 0⟩ as (−1)^{S+N/2}, so
  ψ(flip x) = (−1)^S ψ(x) for every N↑ = N↓ = n (n + n² even). Hence ψ + ψ(flip) keeps even S. Verified numerically
  on the 3×3 (3+3) cluster (exact singlet ground state flip-even, residual 5e-13).

## 2. Checks (`logs/check_local.txt`, kim-compute-03, 1 thread, 10.6 s)

| check | result |
| :-- | :-- |
| (a) guided energy of ψ_even vs exact, 3×3 | −5.84076 ± 0.04345 vs −5.87456, z +0.78, PASS |
| (b) S² estimator on ψ_even, exact average | 0.1427739852 = exact, PASS |
| (d) S-sector weights, bare nnbf | S=0 0.834, S=1 0.145, S=2 0.020, S=3 0.0011 (odd 0.146) |
| (d) S-sector weights, nnbf_even | S=0 0.976, S=1 2e-17, S=2 0.024, S=3 3e-19 (odd 2.3e-17), PASS |
| (d) ψ_even = ψ + ψ(flip) on all 7056 configs | max rel dev 3.0e-16, PASS |
| (d) exact GS (E0 −6.894707) | ⟨S²⟩ 1.7e-17, flip residual 5.4e-13, PASS |
| (e) gradient vs central differences | 4.2e-10 (9 sites), 1.4e-8 (24 sites), PASS |

Smoke tests on 4×3 (/tmp, not kept): fail-fast abort branch (256 chains: median ESS 239 < 512, exit 3); pass branch
with 1024 chains (check passed at 0.5 CPU-min, step 14); SIGUSR1 at 60 s → checkpoint at step 35, exit 4; rerun
resumed and finished (76 steps, E −7.754 ± 0.049, ⟨S²⟩ 0.014 ± 0.014 — even-S only).

## 3. Timing (`logs/probe_local.txt`, kim-compute-03, load ~10/20, 1 thread, 4 guided steps each)

nnbf 0.0847 CPU-min/step, nnbf_even 0.1676 (1.98×; R42's che probe 0.166), start-up 0.69. R42 in-job per-step:
che 0.252, nnbf 0.143 (1.52×, 1.69× their probes). Training is capped at 180 CPU-min of process time, so the in-job
slowdown cuts steps (expected ~600–640 for nnbf_even vs che's 714/719), not wall time; R42 wall = CPU to 1% (sacct
759907). Uncapped parts ×1.8: start-up 1.2 min, scoring 17 × 1.98 × 1.8 = 61 min (R42 in-job nnbf scoring 23.9 min
× 1.98 = 47). 242 min × 1.3 = 315 → `--time=05:15:00`. RSS 0.86 GB → 3G.

## 4. Pre-registration and job

* `RULE.md` written before any training: ρ'_s = gap(che_s)/gap(even_s) at both DMRG ends; CHE-STILL-WINS every ρ'_s ≤
  0.80; CONTAMINATION-EXPLAINS ρ' ≥ 1.00 and every ρ'_s ≥ 0.90; MARGINAL between; UNSETTLED; BUDGET-DEPENDENT → MARGINAL
  action; stage 2 within 0.05 of a line; R42's INVALID-EARLY/INVALID rules; leaf actions per verdict (all exit 3).
* `che` not rerun: identical code path and scoring; job pinned to kim-cpu-01, where all four R42 tasks ran, to avoid
  the cross-machine step-rate confound (P5.1 R51.3 T1d). In-job step rate reported against 2 × 0.143.
* `submit_r43.sbatch`: array 0–1 (seed = task id), partition kim, nodelist kim-cpu-01, 1 CPU, 3G, 05:15:00,
  `--signal=B:USR1@600` with `exec python` (the brief said `USR1@600`; without `B:` Slurm signals job steps only, and
  this script has no srun step, so the B: form is what actually reaches python).
* For the CHE-STILL-WINS bar: che gaps 0.157 (E_last), 0.189 (extrap end); nnbf_even needs gap ≥ 0.196 / 0.236.
  R42 nnbf: 0.306 / 0.338; at step ~715 nnbf was at −16.18 (gap 0.41).
* `voi.md`: prediction CHE-STILL-WINS, p 0.50 (MARGINAL 0.20, CONTAMINATION-EXPLAINS 0.18, UNSETTLED/INVALID 0.12).

Gate output (`compute_gate.py check ... --pre-audit`):

```
cap = 2 array x 5.25 h x (1 tasks x 1 cpus) = 10.5 CPU-h
leaf P4.1: committed 26.9 / 300 CPU-h; brainstorm committed 44.5 / 2000.0 CPU-h
skip threshold: base 0.8 (1 scored predictions); leaf no-change streak: 0; script crash run: 0; VOI digest: ca84421165a9
GATE: PASS PENDING AUDIT (cap > 10 CPU-h; the orchestrator gets the auditor's approval for digest ca84421165a9 before submitting)
```

## 5. Notes for the orchestrator

* summary.md needs (by the editor/orchestrator): ledger row 10 = R43 (prepared, gate PASS PENDING AUDIT), portfolio row R43.
* A cheaper non-deciding read exists: post-hoc flip symmetrisation of R42's trained nnbf θ (scoring only, ~1 CPU-h);
  L1 says post-hoc restoration does not stand in for in-sector training, so it was not substituted.
* Possible stronger cheap baseline not tested: an S² penalty λ⟨S²⟩ at equal CPU (needs a λ scan, so not "fair" at one run).
