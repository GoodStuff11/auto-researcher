# Subproblem: P4.1 Exact spin-SU(2) for fermionic NQS at sub-projection cost
**Parent Chain**: P4 -> P4.1  **Status**: DORMANT

## Route Portfolio
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| none live | — | — | ledger rows 4 (job 759446 running), 8 (R41 worker running) | — | each row's outcome is checked against the DORMANT exits |

All routes are KILLED or SHELVED (`kill_index.md`). Round history: logs `round-23`, `round-24`, `round-25-critic`.

## Reopening Ledger
| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| 1. sample reuse lifting the sampled plateau | closure attempt, item 5 | R34: 24-site cost ratio | KILLED(G4): (sampling + row)/row 1.05–1.08 |
| 2. exact 12-site optimiser $\ge3\times$ the ladder | closure attempt, item 5 | R28 hybrid metric refresh | KILLED(T0): bounded at 1.3–1.8× |
| 3. force shrinkage (R36's translation repair) | closure attempt, item 5 | R39 James–Stein force shrinkage | KILLED(T1): Rbar 0.005, −0.008 vs base 0.016 |
| 4. 12-site convergence of R2, control and ablation | What would close it | R19/R20 resume, 13.0 CPU-h; cheaper: rank-5 arm alone, ~1.4 CPU-h | OPEN: cheaper version (rank-5 arm, then `pcg` to the dominance line; RULE_amend3) gated PASS, 7.0 CPU-h cap; job 759446 running |
| 5. spin-tree basis beyond seed 0 | What would close it | R22 seeds 1–2, 5.4 CPU-h | GATE-REFUSED(G2: predicted KILL p 0.99) |
| 6. charge-only extensive terms at the polished parent | What would close it | R16, R17 tangent rule | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
| 7. sampled optimiser reaching the polished 12-site state | What would close it | R21, R38 | R21 KILLED(G2, dominated by R38); R38 KILLED(T1) |
| 8. R30's translation repair | per-leaf critic r23, item 11 | Anderson on the LM direction before the line search | OPEN: R41 generated; worker running |
| 9. charge-only vs exactly projected spin-reading network, converged | per-leaf critic r23, item 4 | R40, 12 sites | GATE-REFUSED(auditor), closed by refusal (not a G2 refusal): REFUSED by the compute auditor 2026-11-17 (11.0 CPU-h; seed-robust LOSS p about 0.1, AMBIGUOUS likeliest, NO-LOSS not carried to 24+ sites; 30-step cap buys UNSETTLED; pretraining unmeasured, about 440 CPU-min against a 360 cap; seed margin 0.1 vs 0.15); unshelve terms on card R40 |

## Leaf Status Summary
* **State**: DORMANT
* **Updated**: 2026-11-17 (round 25: R40 REFUSED by the compute auditor, row 9 closed; row 4's cheaper version running as job 759446; R41 generated for row 8; accuracy evidence corrected: 451886/508440 were exact-gradient L-BFGS).
* **Entered because**: a closure certificate was attempted but could not be filled honestly: ledger rows 1–3 had no outcome (now KILLED) and row 4's refusal was invalid (now running); rows 8–9 were added (row 9 since auditor-refused).
* **Question**: Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
* **Answer**: Partly. Exact total spin is cheap if the backflow network reads only where the charges are, and exponential, with no cheaper method known, if it reads the up/down spin pattern directly. On a 12-site cluster that restriction cost about 1.7 times in accuracy in unconverged training, recovered only by a mixture that does not scale.
* **What we showed**: Cost: the charge-only form stays 1.3 to 3.6 times the simplest exactly projected version at 12 to 48 sites. Accuracy: with the same network and only its input changed, exact-gradient L-BFGS (2000–4000 iterations, 3 seeds, 12 sites; jobs 451886/508440, not SR) gave the charge-only version $3.43\times10^{-4}$ against $2.07\times10^{-4}$ when reading spins, 1.66 times worse with disjoint seed ranges; neither arm converged or was polished, so this is optimiser-confounded (L33) and 12-site only (L39). Only an 8-component charge-only mixture beat the spin-reading network (by 35%), and it does not scale (G4). A single pairing wave function whose orbitals follow the charges had 45% less error than the same network cut back to a product of up and down orbitals, neither converged. A basis with exact spin by construction did 53 times worse (signs not learned). Sampled optimisers removed about half the energy error, exact ones three quarters.
* **What it would take**: First an affordable optimiser that converges these networks (exact at 12 sites, sampled at 24+; none known that reaches the polished energies), then the like-for-like charge-only vs spin-reading comparison, converged, at 24+ sites. The 12-site version (R40) was refused by the auditor as not decisive. For the pairing form, row 4's 12-site rank-5 comparison is running (job 759446).
* **Conclusion**: Exact $S$ is cheap only when every spin-breaking ingredient sits inside a Gaussian per charge configuration (commutant lemma plus Lemma B). Within that class R2 holds a provisional, unconverged 12-site lead ($\rho_{\mathrm{pol}}=0.586$ against `che`, $\rho_{\mathrm{abl}}=0.550$ against its rank-5 truncation), still an optimiser comparison (L33); no scalable optimiser reaches the polished baseline.

**The question in plain terms.** Neural quantum states (NQS) for the doped Hubbard model build a determinant or Pfaffian whose orbitals depend on the configuration through a neural backflow (NNBF). The ground state has a definite total spin $S$; imposing it exactly helps accuracy and is needed for sector-resolved physics. Spin projection by quadrature over $N_{\mathrm{grid}}$ rotations is cheap only when every correlator reads charges alone. Can exact $S$ be had at polynomial cost without giving up spin information?

**Mechanism and boundary.** Commutant lemma: a factor diagonal in the up/down occupation basis commutes with spin rotations only if it reads the charge configuration $c$ alone, so projecting a spin-reading network sums over all $\binom{N_s}{N_s/2}$ spin arrangements of $c$ ($N_s$ singly occupied sites); no lower bound over all algorithms is claimed. Lemma B: if the state at each $c$ is a sum of $K$ spin-conserving Gaussians, exact projection costs $\mathcal O(KN^3)$ (Che, arXiv:2608.22925). Ten spin-reading routes died at Gates 0–1. R22 escaped the lemma in another basis (a balanced spin-coupling tree, exact and polynomial, $\mathcal O(L^{2.1})$ connected states) but lost on the sign.

**Chain of evidence** (open $4\times3$, $5+5$, $U=8$, $t'=-0.25$, exact sums; relative energy error $\epsilon$). (1) Exact-gradient L-BFGS, matched parameters (jobs 451886/508440): the same NNBF reading charges lost 1.66-fold to it reading spins (exactly projected); only a rank-8 charge-only mixture (R1) won, by 35%, and it died at G4. (2) R2 `pcg` projects one charge-backflowed pairing Gaussian exactly at 3.6 times `che`'s per-step cost at 48 sites; after SR, $\epsilon=1.28\times10^{-3}$ vs $2.86\times10^{-3}$ (`che`), T2a PASS. (3) At 24 sites SR plateaued (R15b). (4) Exact linear-method (LM) steps showed the plateau is the optimiser: polished `che` reached $1.32\times10^{-3}$, erasing (2). (5) Cutoff-ladder polish (job 742206): `pcg` $4.23\times10^{-4}$, `che` $7.22\times10^{-4}$, rank-5 ablation $7.69\times10^{-4}$, all still falling (fitted tails 22–100% of $\epsilon$); the 13 CPU-h resume awaits the auditor. (6) Faster exact optimisers lost to the ladder per CPU-minute (R27–R30; R28 bounded at 1.8×). (7) Sampled SR: R21 R(15) 0.195; guided sampling (R38) 0.364 at equal cost, plateau R(60) 0.512 against 0.7475 exact; force noise limits the step (R32, R35–R37); R39 and R34 do not help. (8) R22 seed 0: 52.6 times `che`'s error. (9) At the polished parents the squeezed frame gains at most 0.0097 of the gap against a 0.05 bar (R16); plaquette charge Jastrows stay below 1.3% (R17).

**Why alternatives fail.** Exact singlets by construction are 4–7 times worse than projecting a broken-symmetry state (job 451886; R0.1, R3.4, R4.2; R22). Mixtures and rational heads gained nothing (R5.5, R5.4h, R3iii). Size-extending refills die on Gate 4 or cost (R4–R12, RG1–RG3, R23–R26).

**Lessons beyond the leaf** (`library.md`): L11, L10, L33, L29, L52, L55, L1, L56, L59 (force noise dominates one sampled SR step), L61.

**Strongest objection.** "The 12-site lead is real: 0.586 already sits on the keep line." Arms whose fitted tails are 22–100% of their error cannot attribute a 41% gap to the ansatz rather than the optimiser (L33), and a 12-site result cannot show a per-site gain; unresolved.

**Limits.** Every accuracy result is 12-site, one filling; 24-site SR runs are not capacity statements.

* **Scope**: open $4\times3$, $5+5$ and $4+4$, $U=8$, $t'=-0.25$, $S=0$, exact sums; 24-site SR at 1024 samples; cost to 576 sites; tree counts to 4096.
* **Compute spent**: 6,979 CPU-hours in 20 legacy Slurm jobs; 8.0 in five gated jobs of the 300 CPU-hour leaf budget; about 2.3 interactive since r21, over-allowance runs disclosed in log `round-23` section 8.
* **Most promising direction**: R2 `pcg` (SHELVED; row 4's cheaper resume running, job 759446): one size-consistent pairing Gaussian, exactly projected at bounded cost ratio. For: T2a PASS, cost PASS, provisional $\rho_{\mathrm{pol}}=0.586$, $\rho_{\mathrm{abl}}=0.550$. Against: unconverged arms, no 24-site optimiser.
* **Next step**: read job 759446 against RULE_amend3 (row 4); take R41's worker result through the gates (row 8); then check the DORMANT exits (row 9 is closed by refusal, so if rows 4 and 8 close, the exit is EXHAUSTED at this budget).
* **What would close it**: rows 4 and 8 closed (row 9 GATE-REFUSED): EXHAUSTED at this scale if all KILLED, at this budget if one is GATE-REFUSED; a surviving row reopens the leaf with that route. PROVEN-IMPOSSIBLE is out of reach: the hardness results bound the spin-reading class only.
