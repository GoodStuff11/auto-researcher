# P4.1 round 27 (worker): R42, the practitioner comparison at 24 sites

Date 2026-11-17. Task from the orchestrator: design the comparison the leaf never ran: at 24+ sites, `che`
(charge-only, exactly projected) against the spin-reading network trained by the same sampled optimiser for the
same compute, scored by energy (and S^2 for the spin-reading arm). Job 759446 (row 4) untouched. summary.md not
edited (the orchestrator owns it this round).

## 1. Framing

The question decided is narrower than the leaf's: "at equal effort, does the cheap exact-S network do as well in
practice?", not "does the charge-only restriction sacrifice expressivity?". Stated on card R42 and in RULE.md.
Question (verbatim from `bs.py show . P4.1`): "Can the total-spin sector $S$ be imposed on a determinant- or
Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without
sacrificing backflow expressivity?" Current Answer: "Partly ... On a 12-site cluster that restriction cost about 1.7
times in accuracy in unconverged training ...". How each verdict would change it: RULE.md, last section.

## 2. What existed at 24 sites

* Code: `tests/P4.1_cheap_su2_scaling/vmc.py` already has both arms at any size: `che` and `nnbf` (spin-resolved
  input, no spin projection), identical layout and init, P = 9711 at H = 16 on 8x3, with P_G. `t2b.py` drives
  `che`/`pcg`/`r1` only. No 24-site `nnbf` run exists beyond a 40-step smoke (`scaling/logs/train_check`).
* Reference: 687302 never produced an 8x3 value. The best available is the $\chi\le1000$ DMRG in
  `scaling/dmrg_irrep/dmrg_S8x3_N20_Sz20_U8_tp-0.25.json`: E_last ($\chi=1000$, truncerr 7e-5) -16.58860,
  E_extrap -16.60459 +- 0.01599; the energy still moved 0.2 between $\chi$ 400 and 600, so it is used as a bracket
  [-16.62058, -16.58860]. Correction: legacy.md lines 1764 and 1939 call -16.60459 the $\chi=1000$ energy; it is the
  extrapolated value.
* `che` 24-site plain SR, 3000 steps (t2b, 4 threads, 11,861 s): -16.4559 +- 0.0021 (gap 0.13-0.16).

## 3. Optimiser port (guided SR at 24 sites)

`tests/P4.1_cheap_su2_r25/guided.py` cannot run at 24 sites: it draws i.i.d. from the exact full vector (12-site
enumeration) and scores steps on exact energies. Port, in `tests/P4.1_cheap_su2_r42/r42.py` (imports vmc.py read
only): Metropolis on $\lvert\psi\rvert^2+\lambda\lvert\psi\rvert$ (unnormalised, so no partition functions are needed), $\lambda$ adapted each step
so the self-normalised $\lvert\psi\rvert^2$ mass fraction is 0.8 (= R38's 0.8 p + 0.2 p_flat); weights $\lvert\psi\rvert^2/q$ in vmc.py's
Woodbury SR; lr schedule written in CPU fraction so both arms share it in CPU time; Polyak average over the last
25% of the budget. Cost of the port: about 60 lines plus an S^2 local estimator (S^- S^+ at S^z = 0, Jordan-Wigner
signs in the up-then-down mode order). Not ported: R38's held-out step-length ladder (4 extra local-energy passes
per step, where the local energy is 85-90% of a step, R34).

## 4. Checks and probe (interactive, 1 thread, ulimit -t 600, BLAS threads pinned before import)

`logs/check_local.txt` (17 s): on 3x3 (N_e 6, closed shell; 4x3 exact enumeration was too slow for the S^2 sum and
was aborted after about 3 CPU-min; 3x3 N_e 8 is open shell):
(a) guided-weighted energy vs exact: che -5.98887 +- 0.04026 vs -5.99886 (z +0.25); nnbf -5.37310 +- 0.03972 vs
-5.38393 (z +0.27); F about 0.79. (b) S^2 estimator averaged exactly over $\lvert\psi\rvert^2$ vs `sec.S2`: che 0 vs 0, nnbf
0.4248383369 vs 0.4248383369. (c) che on 8x3, 64 samples: max |S^2_loc| 5.9e-13.
`logs/probe_local.txt` (219 s), 8x3, 1024 chains, 4 steps each: che 0.166 CPU-min per step (guided and plain equal),
start 0.7-1.0; nnbf 0.090, start 0.4; ratio 1.84. E_loc + S^2 on 128 samples: che 0.036, nnbf 0.017 CPU-min. Peak
RSS 0.87 GB. (The probe's S^2 numbers used amplitudes from before the last update and are meaningless; fixed in the
code afterwards; timing unaffected.) Train-path smoke on 4x3 (22 s, /tmp): resumable checkpoint, half-budget theta,
final JSON written. Interactive total about 7.5 CPU-min.

## 5. Design (RULE.md, voi.md, submit_r42.sbatch)

Seeds 0-1 (stage 1); seed 2 only on the stage-2 clause. 180 CPU-min training per (arm, seed) on one core; scoring
131,072 samples (energy), 32,768 (S^2), 32,768 (half-budget energy). Verdict classes and INVALID clauses in RULE.md.
Prediction: CHE-NO-WORSE 0.45, INTERMEDIATE 0.25, CHE-LOSES 0.15, UNSETTLED/INVALID 0.15. `pcg` left out: row 4
(job 759446) decides `pcg` vs its ablation, and a third arm at 3.6x `che`'s step cost would add about 50% to the cap.
Optional later: `nnbf` with spin-flip averaging (2x cost per amplitude; a parity only, L56).

## 6. Gate check

```
cap = 4 array x 4.50 h x (1 tasks x 1 cpus) = 18.0 CPU-h
leaf P4.1: committed 15.0 / 300 CPU-h; brainstorm committed 20.5 / 2000.0 CPU-h
GATE: PASS PENDING AUDIT (cap > 10 CPU-h; the orchestrator gets the auditor's approval for digest 809d84c8c67e before submitting)
```

## 7. For the orchestrator

* Summary rows to add: ledger row for R42 (source: orchestrator round 27; outcome OPEN, gated PASS PENDING AUDIT).
* Proposed library note (not an entry yet): the leaf's 24-site reference is a bracket, not -16.60459.

## 8. Auditor CHANGES REQUIRED on digest 809d84c8c67e, applied

(1) Fail-fast in `r42.py` and RULE.md: at 20 CPU-min of training each task aborts with `<tag>_INVALID-EARLY.json`
(exit 3) if the median F is outside [0.7, 0.9], median ESS < 512, or a step energy is non-finite; abort and pass
paths smoke-tested on 4x3 (/tmp, deleted). (2) RULE.md: BUDGET-DEPENDENT overrides every class (numbers only, no
ANSWERED proposal, no EXHAUSTED exit); CHE-NO-WORSE Answer wording must say "at 3 CPU-h per run, one core". (3) Every
step's guided energy is saved (`energy_per_step`); unscored reports: energy vs step per arm, `nnbf` at `che`'s final
step count (100-step window); the lead number is the reference-free E_che - E_nnbf. voi.md: modal p unchanged 0.45;
BUDGET-DEPENDENT about 0.2, so exit-triggering outcomes are about 0.35 (NO-WORSE) and 0.12 (LOSES). Re-gate: PASS
PENDING AUDIT, cap 18.0 CPU-h, digest 4ed16de4e3f2.
