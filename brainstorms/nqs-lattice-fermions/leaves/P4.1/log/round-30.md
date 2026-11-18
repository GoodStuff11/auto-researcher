# P4.1 round 30 — generation batch: does the charge-only restriction cost expressivity at sizes that matter?

Worker round, 2026-11-17. The leaf stays DORMANT (ledger row 10 = R43, job 764949, running and not touched). `summary.md` was not edited; the orchestrator adds the rows. New cards R44–R49.

## 1. Scope
This round targets the part of the Question that stays open whatever R43 returns: whether reading only charges, which cheap exact $S$ needs, costs expressivity. Facts in hand: 12-site exact-gradient L-BFGS gave 1.66 / 2.10 (unconverged, L33); R40 (polished 12-site comparison) was refused by the auditor; R42 is a practical comparison at 24 sites against an unprojected comparator; P5.1 R51.1 (guided SR reaches 99% of polished at 262k samples/step) and R51.3 (leverage-row metric, 3.53× the exact ladder rate, coordinator message mid-round).

## 2. Structural observation (T0, applies to the whole batch)
Exact projection of a spin-reading network costs $\binom{N_s}{N_s/2}$ amplitudes per amplitude (commutant lemma), so **any expressivity comparison against an exactly projected spin-reading arm is ED-bound**. Beyond ED only one-sided comparisons are possible: a spin-reading arm that is unprojected, cheaply symmetrised or penalised gives a feasible point (or a lower bound on fidelity) of the projected family, so it can show LOSS for the restriction but never NO-LOSS against the projected family. Proposed library entry (from R44).

## 3. Routes
| Route | Item | Verdict | One line |
| :--- | :--- | :--- | :--- |
| R44 `conv-16` | (a) | KILLED G4 | sampled Hubbard 4x4 ~18 CPU-h per step; the affordable form (t-J 4x4, exact sums, 1.68M configurations) is an ED cluster on another model, so it says nothing beyond 16 sites |
| R45 `sup-fit` | (b), cross-field (ML teacher-student) | LIVE (GATED) | T1 TRACTION; T2 prepared, PASS PENDING AUDIT |
| R46 `pen-bound-24` | cross-field (augmented Lagrangian) | KILLED G2 | one-sided; detection p ~0.1 given R42; reopenable on R43 CONTAMINATION-EXPLAINS |
| R47 `distill-24` | cross-field (knowledge distillation) | LIVE (T0) | the only route that says something at 24 sites: fit students to a DMRG MPS; $F(P_S\psi)\ge F(\psi)$ for an $S=0$ teacher makes `nnbf`-wins a valid LOSS |
| R48 `moment-feat` | (c) | KILLED G2 | SU(2)-scalar moments of $\Phi_c$ are functions of $(c,\theta)$: no new information |
| R49 `angle-gcm` | (c), cross-field (GCM) | KILLED G1 | angle-dependent Gaussians break exactness; exact sub-cases are `che` and R1 |

Item (c) finding: under the commutant lemma and Lemma B, cheap exact-S constructions are c-indexed sums of projected spin-conserving Gaussians; spin information enters only via the Gaussian's type (UHF determinant `che`, pairing R2, non-collinear RG3) and their number (R1, R5: G4). R48 and R49 were the two new slot-fillers; both reduce to existing slots. No (c) route survives.

## 4. R45 T1 (interactive, 5.9 CPU-min, `tests/P4.1_cheap_su2_r45/`)
Rule `RULE_T1.md` written before the run. Start `che` drift4000 seed 0: eps $3.0585\times10^{-4}$, $1-F=3.1804\times10^{-4}$. 41 L-BFGS-B evaluations on $-\log F$: best $1-F=2.5813\times10^{-4}$ ($d=0.188$, TRACTION, line 0.10); eps of that state $3.340\times10^{-4}$ (rose 9%); still falling about 0.25% per evaluation at the end. Reference: energy L-BFGS lowered eps 0.18% per 50 iterations at iteration 4000 (stored trajectory). Checks: target (+,+)-invariant $2.0\times10^{-13}$; $\langle g|P\phi\rangle=\langle g|\phi\rangle$ to $3.6\times10^{-14}$; FD gradient $5.0\times10^{-8}$ relative; start eps and $1-F$ reproduce the record to $2\times10^{-15}$. Prediction was MARGINAL or TRACTION (p NO-TRACTION 0.3): hit. Reading: the energy-trained state is far from its fidelity optimum, so energy-trained rankings need not carry to capacity; it does not show that fits plateau (T2's flag reports that). Log `logs/t1_fit.txt`, `results/t1_fit.json`.

## 5. R45 T2 (prepared, not submitted)
`r45.py` (modes check / train / verdict), `RULE_T2.md`, `submit_r45.sbatch`, `voi.md`. `--mode check` (1.6 CPU-min, `logs/check_local.txt`): FD gradients `che` $4.1\times10^{-10}$, `nnbf` $2.5\times10^{-10}$; per evaluation `che` 0.117, `nnbf` 0.065 CPU-min; initial $1-F$ 0.425 / 0.423 (common init recipe). Train/resume/verdict smoke-tested (results removed). Peak RSS 1.66 GB. Design: seed 0, 2000 evaluations each, single-seed NO-LOSS $\kappa\le1.0$, LOSS $\ge1.65$, half-budget agreement, NOT-PLATEAUED flag.

`compute_gate.py check --pre-audit`: cap = 2 array x 9.00 h x 1 CPU = 18.0 CPU-h; leaf P4.1 committed 37.4 / 300; brainstorm 53.7 / 2000; VOI digest 08d370f1663e; **GATE: PASS PENDING AUDIT**.

## 6. Coordinator input: P5.1 R51.3 and R40's refusal reasons
R51.3 (leverage-score row sampling of the exact-sum metric, 3.53× the ladder rate per CPU-min at 12 sites, 2.22× with the score build charged) against R40's refusal: (1) decision value: unchanged; a 12-site result still does not carry (L39), and NO-LOSS still would not transfer. (2) Convergence: about 3× the steps per budget, but R40's fitted tails were 20–100% of eps, so AMBIGUOUS stays the likeliest class unless the start is much closer. (3) Cost: R51.3 does not touch the unmeasured `nnbf_SG` pretraining (about 440 CPU-min of L-BFGS), and for `nnbf_SG` the projected tangent needs $P_S$ applied to full columns before rows can be sampled (the projection mixes rows inside each charge block, blocks $\le252$), so the saving is smaller than for `che` unless whole charge blocks are sampled. Composition proposed: R45's fitted endpoints replace R40's pretraining as starts, then an energy polish of both arms with R51.3 (fit-then-polish). That is the cheapest converged 12-site energy comparison; prepare only after R45's T2, since a T2 NO-LOSS or LOSS with plateaued fits may make it unnecessary.

## 7. Proposed library entry
"An expressivity comparison against an exactly projected network that reads a non-invariant input is ED-bound when the projection is an orbit sum; beyond ED only one-sided tests (unprojected, symmetrised or penalised comparator; fidelity to a symmetric teacher) are possible, and they can show a loss but not its absence." Sources: R44, R46, R47.

## 8. For other leaves
P5.1: the infidelity fit is a cheap exact-sum way to get a near-capacity start (T1: 18.8% of $1-F$ in 41 evaluations from an energy-converged state); fit-then-polish may be a better reference-state recipe than polish alone.

## 9. Compute
Interactive: T1 5.9 CPU-min, check 1.6, smoke about 2.5, RSS probe about 0.4: about 10.4 CPU-min across four separate runs, each under `ulimit -t` 600 or 300.
