# R45 T1 rule (written 2026-11-17, before `t1_fit.py` was run)

Premise under test ("fit traction"): supervised infidelity fitting to the exact ground state moves a 12-site exact-S
state that exact-gradient energy L-BFGS has left crawling, so a fit-to-plateau capacity comparison is affordable.

* Start: `che` drift4000 seed 0 (open 4x3, 5+5, U=8, t'=-0.25, (+,+), S=0; eps 3.0585e-4, 1-F 3.18e-4). Over its last
  500 energy L-BFGS iterations eps fell 0.18% per 50 iterations (stored trajectory), i.e. about 0.13% per 40 evaluations.
* Run: fresh L-BFGS-B (maxcor 30) on -log F, 40 function evaluations, one thread, under `ulimit -t 600`.
* Metric: d = 1 - (1-F)_best / (1-F)_start over the 40 evaluations; eps of the best-fidelity iterate reported.
* TRACTION: d >= 0.10 (about 75x the energy route's relative rate). Action: prepare the T2 (both arms, fit to a plateau).
* NO-TRACTION: d < 0.02. Action: R45 KILLED(T1): the infidelity landscape stalls like the energy one, so supervised
  fitting does not remove the convergence confound at affordable cost.
* MARGINAL otherwise: no T2 prepared this round; recorded.
* INVALID if any check fails: g P_G-invariant to 1e-10; <g|P phi> = <g|phi> to 1e-10 relative; start eps and 1-F
  reproduce the record to 1e-8; finite-difference gradient to 1e-6 relative.
* Prediction (before the run): MARGINAL or TRACTION, p(NO-TRACTION) about 0.3: the fidelity loss is a different
  landscape, but both inherit the same near-singular parameter metric.
