# P4.1 round 21: R22 `spin-tree` falsifier 2 implemented, validated, gated (not submitted)

2026-11-17. Continuation worker (previous worker stopped mid-way). Test dir `tests/P4.1_cheap_su2_r23/`.

* Inherited and kept (not broken): `spin_tree.py` (coupling-tree basis, recoupling H, point-group maps) and
  `tree_nqs.py` ($\psi = P_G[D(c) F_\theta(c,T)]$, stage A Adam, stage B r22 ladder, sign diagnostic, verdict).
* Required validations, all PASS (`logs/validate.log`, `results/validate.json`):
  (a) tree-basis ED (+,+) = -7.8278990342718, |diff from E0| 9.8e-15 (below 1e-10); [H,U_g] 2.2e-16.
  (b) recoupling H vs brute-force Fock transformation: chain4+t', 2x2, 2x3, 4x2 (mirror tree), N_e 2-6, S 0-1:
      max diff 3.6e-15, orthonormality 6.7e-16; U_g brute force 6.7e-16.
  (c) connected T' counts vs r21 `tree_count.hop_count`: 5,225,220 (state, hop) pairs, 0 mismatches; 0 accidental
      zeros. Sector dim 169,884; (+,+) irrep 42,756. No T0 kill: recoupling is polynomial as claimed and the basis
      carries the charge sector (D(c) on fermion-ordered slots) at no extra cost; H build 68 s.
* `tree_nqs.py --mode check` (`logs/check.log`): P = 2672 (+1.3% vs 2637), analytic-vs-autograd gradient 1.1e-14,
  orbit-reduction identities < 5e-15. OK.
* RULE.md amended before any tree-arm energy: eps_ref = min(1.217e-3, ladder-polished `che` from 742206), the same
  min for the guard. 742206 `che` is final (STOP-CAP): eps 7.217e-4, guard 6.930e-4 -> eps_ref 7.217e-4; PASS-T1
  needs $\epsilon_{\rm tree} \le 4.33\times10^{-4}$. `tree_nqs.py verdict` patched to use the min. Erratum: N_A = 7000 (cost-set).
* Timing (`logs/time.log`, 1 thread): Adam step 0.94 CPU-s; ladder step 0.54 CPU-min. Whole test ~6 CPU-h, over 10
  CPU-min, so not run interactively.
* Pre-run basis probe (`probe_basis_sign.py`, 0.1 CPU-min, no training): exact ground state in the tree basis has
  48.1% of weight on negative amplitudes; |psi0| has eps 0.956; 99% weight on 6,250 of 42,756 representatives. The
  tree basis does not reduce the sign structure; F's sign head must learn it (card's Hard Core). Informs the
  prediction only.
* Job: `submit_tree.sbatch` (1 task x 3 CPU, 02:40:00; 3 seeds x 7000 Adam in parallel, then ladder polish, sign,
  verdict). VOI `voi.md`: prediction KILL (G2) p 0.75 / AMBIGUOUS 0.17 / PASS 0.08.
  `compute_gate.py check ... --pre-audit`: cap 8.0 CPU-h; leaf 3.7/300; GATE: PASS (digest 781b0d4b55e5). NOT submitted.
* Summary and cards untouched this round (compactor rewriting them).

## Restructure (coordinator request, same round): seed 0 first

* `RULE_amend1.md` (written before any run): seed 0 alone; one-seed KILL (G2) if polished seed-0 eps (endpoint and
  guard) is at least 2.5 eps_ref = 1.804e-3, else release seeds 1-2 and apply RULE.md's 3-seed rule.
* Factor 2.5, not 2: on 12 multi-seed arms of this leaf (r3, r5, r6) worst/best seed eps is 1.08-1.81 (median 1.24;
  max r3 `che` N8). Factor 2 is only 10% above that maximum, and a learned spin-sector sign can widen the spread.
* `tree_nqs.py` gained `--mode verdict1` (smoke-tested on a fake state). `submit_tree.sbatch` now runs 1 CPU,
  02:40:00, seed 0, with polish/sign/verdict1 in `results/oneseed/`.
* VOI: P(one-seed KILL) 0.6, P(RELEASE) 0.4. Gate: cap 2.7 CPU-h; leaf 3.7/300; GATE: PASS (digest 0063f561e528).
  NOT submitted.
