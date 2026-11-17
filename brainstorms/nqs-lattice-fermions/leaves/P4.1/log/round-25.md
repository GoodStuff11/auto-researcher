# P4.1 round 25 (worker, 2026-11-17): R40 (charge-only vs spin-reading) and ledger row 4 cheaper version

Task (orchestrator, after the round-23 per-leaf critic, objections 1, 2, 4): (1) a new route for the decisive 12-site
comparison of a charge-only and a spin-reading backflow, both exactly projected; (2) cost the cheaper version of
ledger row 4. Route id: first written as R39, renamed R40 on the coordinator's instruction (R39 is `js-force`).
No job submitted; summary.md and existing cards not edited.

## 1. Correction to the critic's premise (for the orchestrator)

The critic says the only direct datum is 508440's `che_k8` beating `nnbf_SG` by 35%, "SR-trained". Two corrections:
* Those runs were exact-gradient L-BFGS (2000 iterations, 3 seeds), not SR.
* There is a more direct datum pointing the other way: in the same runs (job 451886, reused by 508440's report),
  `che` (the identical network with charge input) had 3.43e-4 against `nnbf_SG`'s 2.07e-4, i.e. the charge-only
  restriction cost 1.66-fold (seed ranges disjoint; history.md "Spin-resolved conditioning is worth about 1.7x").
  `che_k8` won only by adding rank, which R1 killed at G4. The critic's proposed Answer sentence ("a charge-only
  network beat a spin-reading one by 35%") should name both: the like-for-like network lost 1.66-fold; an
  8-component (non-scalable) mixture won by 35%; neither converged.

## 2. R40 (`che-vs-spin`), Operation A on R1 / 508440

* Arm choice: `che` vs `nnbf_SG`, same NNBF (H 16, P 2637), only the input differs; the ratio is the cost of the
  restriction itself. `che_k8` rejected (adds rank, a finite-size gain, L10); `pcg` rejected (adds the pairing frame
  and was polished from an SR start; its r22 endpoint is reported beside the result only).
* Code: `tests/P4.1_cheap_su2_r40/r40.py` (reuses r19 polish / r11 realise / r22 ladder protocol; new: block-diagonal
  exact S = 0 projector over charge configurations, blocks $\le$ 252, sum of d^2 = 20.5 M; nnbf energy, exact gradient,
  ladder build with P_S applied per column block after the P_G reduction).
* Checks (local, `logs/check_local.txt`, `logs/timing_local.txt`): P_S vs sec.projS(0) 1.6e-14; che drift4000 start
  vs record 1.8e-15; nnbf gradient vs FD 3.6e-11 rel; projection pipeline 3.7e-15; log-derivatives vs FD 1.4e-10.
* Finding during checks: the E_init recorded by 451886 (che -6.27701, nnbf_SG -6.28260) is not reproduced by the
  current legacy fix_test.py either (it gives che -6.28046, nnbf_SG -6.28247 at the same theta as V.Model.init(0)),
  so the 451886-era init differed slightly. The trained che drift4000 theta reproduces to 1e-15, so the model is the
  same; the check compares against the current legacy code (`logs/legacy_init.txt`, `legacy_init_check.py`), and
  pretraining validity uses eps at iteration 2000 inside the widened legacy 3-seed range.
* Timings (1 thread): nnbf energy 0.025, energy+gradient 0.0645 CPU-min; ladder build 2.40 (extrapolated from a
  partial build), dense S/H 0.70, eigh 0.05, line search $\le$ 0.38 -> 3.6-3.9 per step (4.7 with margin); che 2.9
  (742206). 3-iteration pretraining smoke: 0.35 CPU-min (about 1.7 evaluations per early iteration).
* Rule: `RULE.md` (NO-LOSS $\le$ 1.1, LOSS $\ge$ 1.5, INTERMEDIATE between; r22 amend2 guard; 30-step cap; one common
  absolute stop g = 2.5e-4 |E0|). Prediction (voi.md): LOSS 0.30, INTERMEDIATE 0.28, NO-LOSS 0.05, AMBIGUOUS 0.32,
  INVALID/INCOMPLETE 0.05; p 0.32.
* 12-site caveats (card, RULE.md): P/dim about 0.06 (below L36), but L39 applies: no spin correlation longer than
  the cluster; LOSS is informative, NO-LOSS does not transfer; matched parameters, not matched cost (L35).
* Gate: `compute_gate.py check ... submit_r40.sbatch --voi voi.md --pre-audit` ->
  cap 11.0 CPU-h; GATE: PASS PENDING AUDIT (digest 51b4814ff5fa). Auditor needed (> 10 CPU-h).

## 3. Ledger row 4, cheaper version (`tests/P4.1_cheap_su2_r22row4/`)

* RULE_amend3.md (written before any step): read rho_abl only; rank5 alone to STOP-RULE / STOP-NOGAIN / 30-step cap
  (amend1, amend2 guard); then pcg one step per call only until eps_pcg $\le$ 0.60 min(eps_rank5_end, eps_rank5_inf)
  (PAIRING-KEPT by dominance, since ladder steps never raise E and eps_inf $\le$ eps_end); otherwise pcg to its own stop
  and the unchanged rule. che and rho_pol dropped (rho_pol stays provisional 0.586).
* rank5 state: 8 steps, eps 7.693e-4, predicted gain/g 0.100, 0.069, 0.047 (x0.68/step) -> STOP-RULE in about
  10-11 more steps (17 at x0.8); fit eps_inf 5.98e-4 -> dominance line about 3.6-3.8e-4; pcg at 4.230e-4 needs about
  2-4 steps.
* Prediction: PAIRING-KEPT 0.70, ABL-CONFOUNDED 0.18, RETRACTED 0.02, AMBIGUOUS 0.06, INCOMPLETE 0.04; p 0.70.
* Cost: worst 403 CPU-min (rank5 22 x 8.84 + 0.7; pcg 21 calls x 9.9) -> --time 07:00:00, 1 CPU = 7.0 CPU-h;
  expected 2-3.3 CPU-h. Under 10 CPU-h: no auditor.
* Gate: `compute_gate.py check ... submit_row4.sbatch --voi voi_row4.md --pre-audit` -> cap 7.0 CPU-h; GATE: PASS
  (digest a589c8b73ac7).

## 4. Compute this round (interactive, 1 thread, under the 10 CPU-minute allowance)

check 0.54 x 2, timing 1.59 + 0.87, legacy-init check about 1.5, debug import 0.5, gemm timing 0.1, pretrain smoke 0.48,
row4 import test $<$ 0.1: about 6.6 CPU-min.

## 5. Proposed summary changes (orchestrator owns summary.md)

* Portfolio: add R40 (T0, GATED PASS PENDING AUDIT, 11.0 CPU-h) and ledger row 4 -> "cheaper version gated PASS
  (7.0 CPU-h, RULE_amend3)"; drop "GATE-REFUSED(G2)" from row 4 (critic objection 1).
* Answer accuracy clause: name both data (section 1), and R40 in "What it would take".
