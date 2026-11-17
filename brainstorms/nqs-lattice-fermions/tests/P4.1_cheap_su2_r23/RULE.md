# P4.1 r23: pre-registered rule (R22 `spin-tree`, falsifier 2: matched-parameter test against polished `che`)

Written 2026-11-17, before any r23 code existed and before any tree-arm energy. Source: the R22 card, "Cheapest
falsifier 2" and ranked item 6 (`/home/jek354/.claude/jobs/91298e5c/tmp/P4.1_round20.md`); the polish protocol is
r22's (`../P4.1_cheap_su2_r22/RULE.md`, `ladder.py`); falsifier 1 passed in r21 (`../P4.1_cheap_su2_r21/results/
tree_count.json`: connected-state exponent 2.09 / 2.19 <= 2.6).

Card text (verbatim): "at 12 sites with exact sums, train D(c)F_theta(c,T) with P ~ 2637 (an MLP on (c,T), with a
sign head), and polish it with the R19 ladder. Pre-registered rule: KILL (G2) if eps_tree >= eps_che,pol =
1.217e-3. PASS-T1 if eps_tree <= 0.6 eps_che,pol. AMBIGUOUS otherwise."

## System

Open 4x3 Hubbard strip, N_e = 10, S = 0, U = 8, t = 1, t' = -0.25 (H = -sum T_ij c^dag_i c_j + U sum n_up n_dn, the
`vmc.py` convention), point-group irrep (+,+). E0 = -7.827899034271753. eps = (E - E0)/|E0|. All sums exact.

## Tree arm (`tree`)

* Basis |c, T>: c in {0,1,2}^12 with sum c = 10; singly occupied sites carry spin 1/2; T = intermediate spins of a
  fixed binary coupling tree over the 12 sites, root spin 0. The tree is a reflection-covariant spatial tree
  (halves x in {0,1} | {2,3}; each half = [middle row pair, [bottom row pair, top row pair]]), chosen so that both
  mirrors act as signed permutations of basis states (needed for P_G). It differs from r21's k-d tree on 4x3,
  which is not mirror-covariant; falsifier 1's exponent is a large-L statement and is unaffected.
* Fermion order = depth-first leaf order of the tree, up before down on a doubly occupied site.
* Amplitude psi = P_G [D(c) F_theta(c,T)], P_G = (1/4) sum_g chi_g U_g over {E, Mx, My, MxMy}.
  - D(c) = det Phi[occupied slots, :], Phi in R^{24 x 10}: slot 2p is the first electron on the p-th site in fermion
    order, slot 2p+1 the second (used only when doubly occupied). Charge-only.
  - F_theta(c,T) = exp(a) tanh(b), (a, b) = W2 tanh(W1 x + b1) + b2, one hidden layer of width 30; x = one-hot of
    every site occupation (36) and one-hot of every non-root internal node spin (42) = 78 inputs. b is the sign head.
  - P = 240 + 78*30 + 30 + 30*2 + 2 = 2672 (matched to `che`'s 2637 within 1.3%; equal to r22 `rank5`).
* Hop matrix elements are computed by recoupling (per-node reduced matrix elements, 6j-type node coefficients and
  one rank-0 coupling at the lowest common ancestor), never by expanding into the Fock basis (that is used only to
  validate).

## Protocol

* Stage A (pretraining; `che` had SR pretraining before its polish): exact-sum Adam on the exact (+,+) energy,
  3 seeds, cosine learning rate 3e-3 -> 1e-5, N_A steps per seed. N_A is set by cost alone (target: the whole job
  within the 10 CPU-h worker cap) and is written in the erratum below before any stage-A energy is read. The seed
  with the lowest exact energy at the end of stage A is polished; the other seeds are reported.
* Stage B (polish): r22's ladder protocol, unchanged: per step one exact S, H build and one tangent
  eigendecomposition; cutoffs {1e-5, ..., 1e-9}; LM candidate with |z| <= 0.05; line search s in {1, 1/2, 1/4}
  taking the first s that lowers eps by > 1e-7; best rung accepted; STOP-NOGAIN if none; STOP-RULE if the accepted
  step's predicted gain is below 1e-3 g, with g = E(theta_A) - E0 the tree arm's own gap at the start of polish
  (as r22 used each arm's own pre-polish gap); STOP-CAP at 12 steps. Convergence guard: r22's geometric tail.

## Denominator

eps_che = the r22 `che` arm endpoint (`../P4.1_cheap_su2_r22/results/state_che.json`), with its r22 guard tail
(eps_che,inf). The card's 1.217e-3 is the r19 endpoint, which r21 showed was a frozen polish (FROZEN-ARTEFACT,
probe_che.json); the card's intent is "polished `che`", so the r22 ladder endpoint replaces it. The ratio against
1.217e-3 is also printed and decides nothing. If r22's `che` arm has no final stop, the verdict is INCOMPLETE.

## Verdict (rho = eps_tree / eps_che)

| verdict | condition |
| :-- | :-- |
| PASS-T1 | rho <= 0.60 |
| KILL (G2) | rho >= 1.00 (the card: eps_tree >= eps_che,pol) |
| AMBIGUOUS | 0.60 < rho < 1.00 |

The card gives thresholds, so the default 0.85 KILL line of the worker brief is not used; it is printed for
information only. rho is classified with endpoint eps and with guard-extrapolated eps (both arms); if the classes
differ the result is UNSETTLED (resume, no change).

## Gate-2 sign diagnostic (pre-registered reading, does not change the verdict)

At the tree endpoint, with psi0 the exact (+,+) ground state in the tree basis (global sign by positive overlap):

* eps_signfix: energy of |psi_tree| * sign(psi0) (tree magnitudes, exact signs).
* eps_ampfix: energy of |psi0| * sign(psi_tree) (exact magnitudes, tree signs).
* w_wrong: psi0-weight on basis states where sign(psi_tree) != sign(psi0).
* Basis control: eps of |psi0| (all signs +) and of |psi0| * sign(D(c)) with the endpoint D; how much sign
  structure the tree basis leaves for F to learn.

Class: SIGN-DOMINATED if eps_signfix <= 0.5 eps_tree; NOT-SIGN-LIMITED if eps_signfix >= 0.8 eps_tree; MIXED
otherwise.

## What each outcome does

| verdict | R22 |
| :-- | :-- |
| KILL | R22 killed at Gate 2 (12 sites, matched P). The sign class says why: SIGN-DOMINATED confirms the card's Hard Core (no Gaussian reference for the spin-sector sign); otherwise the loss is amplitude expressivity of a determinant-free spin sector. |
| AMBIGUOUS | R22 stays LIVE at T1-ambiguous; the 12-site question is not re-run at this budget; any next rung must say what it adds beyond P. |
| PASS-T1 | R22 to T2: the 24-site question needs a sampler in the tree basis (autoregressive root-down, card) and the R21 optimiser verdict. |

## INVALID (no verdict)

* Tree-basis ED (+,+) ground energy differs from E0 by more than 1e-10.
* Recoupling hop matrix elements differ from a brute-force Fock-basis transformation by more than 1e-12 on any test
  cluster (4-site chain, 2x2 with t', 2x3 with t'; several N_e), or the tree states fail orthonormality to 1e-12.
* The recoupling code's connected T' count differs from r21 `tree_count.hop_count` on the same tree for any hop of
  any basis state of the 12-site sector (the r21 count is the triangle-allowed count; accidental zeros of the
  recoupling coefficient are reported separately).
* U_g does not commute with the tree-basis H to 1e-12, or the brute-force check of U_g on 2x2 fails.
* P differs from 2637 by more than 2%.
* r22's `che` state carries invalid checks.

## Erratum / settings fixed after timing (before any stage-A energy was read)

(filled in below if needed)
