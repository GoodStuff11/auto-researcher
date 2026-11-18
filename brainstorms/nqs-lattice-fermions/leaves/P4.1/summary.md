# Subproblem: P4.1 Exact spin-SU(2) for fermionic NQS at sub-projection cost
**Parent Chain**: P4 -> P4.1  **Status**: DORMANT

## Route Portfolio
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| none live | — | — | ledger row 10 (spin-flip-even `nnbf` arm, being prepared as a gated job) | its RULE.md (card TBD) | row 10 closes → exit 3 (EXHAUSTED at this budget) unless it changes the Answer's scope |

All routes are KILLED or SHELVED (`kill_index.md`); R42 is complete (CHE-NO-WORSE). Round history: logs `round-23` to `round-29-critic`.

## Reopening Ledger
| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| 1–3. sample reuse; exact optimiser ≥3× the ladder; force shrinkage | closure attempt, item 5 | R34, R28, R39 | KILLED(G4), KILLED(T0), KILLED(T1) |
| 4. R2 vs its rank-5 ablation, converged at 12 sites | What would close it | job 759446, RULE_amend3 | DIAGNOSTIC-HOLDS (pairing gain unattributed; label ABL-CONFOUNDED) |
| 5. spin-tree basis beyond seed 0 | What would close it | R22 seeds 1–2, 5.4 CPU-h | GATE-REFUSED(G2) |
| 6. charge-only extensive terms | What would close it | R16; R17 | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
| 7. sampled optimiser reaching the polished 12-site state | What would close it | R21, R38 | KILLED(G2), KILLED(T1) |
| 8. R30's translation repair | critic r23, item 11 | R41 | KILLED(T1) |
| 9. converged charge-only vs projected spin-reading | critic r23, item 4 | R40 (11.0 CPU-h), recast at 24 sites as R42 | GATE-REFUSED(auditor); R42 CHE-NO-WORSE (job 759907) |
| 10. spin-flip-even nnbf baseline at 24 sites (named falsifier from the per-leaf critic, Major 2; being prepared as a gated job, card number TBD by the worker) | critic r29, Major 2 | r42 harness, 2 seeds, equal compute | OPEN |

## Leaf Status Summary
* **State**: DORMANT
* **Updated**: 2026-11-17 (round 29: per-leaf critic refused the scoped ANSWERED; Answer narrowed; ledger row 10 opened; EXHAUSTED-at-this-budget certificate drafted, pending row 10).
* **Entered because**: a closure certificate was attempted but could not be filled honestly: a named falsifier (row 10, the spin-flip-even baseline) is unrun.
* **Question**: Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
* **Answer**: Partly. Exact total spin is cheap if the backflow network reads only where the charges are, and exponential, with no cheaper method known, if a network built from factors that each read a configuration's up/down spin pattern does; in one test (a 24-site strip at one filling, two seeds, the same optimiser, at 3 CPU-h per run, one core, i.e. three CPU-hours), the charge-only network ended with about half the energy error of the same network reading spins with no spin symmetry imposed, whose state drifted away from the ground state's expected total spin. Whether the restriction costs expressivity is not settled: in unconverged exact-gradient training the charge-only network had 1.7 to 2.1 times the energy error on the two 12-site fillings tried, against the spin-reading network projected exactly at exponential cost.
* **What we showed**: Cost: the charge-only form costs 1.3 to 3.6 times the simplest exactly projected version at 12 to 48 sites. Practice: on a 24-site strip (20 electrons, two seeds), same sampled optimiser and equal CPU time, the projected charge-only network's energy error was 0.51–0.56 of the spin-reading network's (also at half the time), and the spin-reading network had total spin squared 1.1–1.2 where the ground state is expected to have 0. Bases built to have exact total spin give it at polynomial cost but were 4 to 53 times less accurate. A pairing wave function whose orbitals follow the charges had 38% less error than the same code cut back to separate up and down orbitals, short of the preset bar for crediting pairing.
* **What it would take**: First, the same 24-site test against the spin-reading network symmetrised under a global spin flip, which removes its largest spin contamination at twice the cost per amplitude; it is being prepared. For expressivity: the same network reading charges versus spins, both converged, at 24 or more sites, which needs a sampled optimiser that converges there; one reaches 99% of the best 12-site energy with 262,000 samples per step, untested at 24 sites. The practical result would be overturned by the spin-reading network winning at a larger budget, size or filling, or with a cheap spin penalty or symmetrisation at equal CPU time.
* **Conclusion**: For networks whose spin dependence enters as factors diagonal in the up/down occupation basis, exact $S$ is cheap only when every spin-breaking ingredient sits inside a Gaussian per charge configuration (commutant lemma plus Lemma B). Within that class `che` beats the unprojected NNBF at equal compute on 24 sites ($\rho$ 0.51–0.56); the spin-flip-even baseline (row 10) is owed. The 12-site matched-parameter data (1.66, 2.10) are optimiser comparisons (L33) that need not transfer (L39). R2's pairing gain over its rank-5 ablation ($\rho_{\mathrm{abl}}=0.623$) is unattributed.

**The question in plain terms.** Neural quantum states (NQS) for the doped Hubbard model build a determinant or Pfaffian whose orbitals depend on the configuration through a neural backflow (NNBF). The ground state has a definite total spin $S$; imposing it exactly helps accuracy and is needed for sector-resolved physics. Spin projection by quadrature over $N_{\mathrm{grid}}$ rotations is cheap only when every correlator reads charges alone. Can exact $S$ be had at polynomial cost without giving up spin information?

**Mechanism and boundary.** Commutant lemma: a factor diagonal in the up/down occupation basis commutes with spin rotations only if it reads the charge configuration $c$ alone, so projecting a spin-reading network of such factors sums over all $\binom{N_s}{N_s/2}$ spin arrangements of $c$ ($N_s$ singly occupied sites); no lower bound over all algorithms is claimed. Lemma B: a sum of $K$ spin-conserving Gaussians per $c$ projects exactly at $\mathcal O(KN^3)$ (Che, arXiv:2608.22925).

**Chain of evidence** (12 sites: open $4\times3$, $U=8$, $t'=-0.25$, exact sums; 24 sites: open $8\times3$, $N_e=20$). (1) L-BFGS, matched parameters (jobs 451886/508440; pilot 411494): charge-reading lost 1.66-fold ($5+5$) and 2.10-fold ($4+4$) to spin-reading (both projected); only the non-scaling rank-8 mixture R1 won. (2) R2 `pcg` costs 3.6× `che` per step at 48 sites; its SR lead vanished once `che` was polished (brought to the stop rule of exact linear-method steps). (3) Row 4 (job 759446): $\rho_{\mathrm{abl}}=0.623$, between the credit (0.60) and retraction (0.85) lines. (4) No sampled optimiser reaches polished 12-site states (R21, R38; R32–R39, R41 do not help). (5) R42 (job 759907): guided SR, seeds 0–1, $\rho$ 0.51–0.56 at both DMRG bracket ends, half budget 0.53–0.58, `nnbf` $\langle S^2\rangle$ 1.11–1.18 (log `round-29-critic` section 3).

**Why alternatives fail.** Exact singlets by construction are 4–53 times worse (R0.1, R0.4, R3.4, R4.2, R22: sign not learned). Mixtures and rational heads gained nothing (R5.5, R5.4h, R3iii). Size-extending refills die on Gate 4 or cost (R4–R12, RG1–RG3, R23–R26). Ten spin-reading routes died at Gates 0–1.

**Lessons beyond the leaf** (`library.md`): L1, L11, L10, L33, L35, L39, L29, L52, L55, L56, L59, L61.

**Strongest objection.** (i) L1 (broken symmetry inflates the error of an unsymmetrised network): `che`'s win may be the removal of `nnbf`'s spin-1 contamination, which a spin-flip-symmetrised `nnbf` also removes at about equal step count. Not yet answered; it is row 10. (ii) "Equal-compute, unconverged training compares optimisers, not ansätze (L33)." Agreed for expressivity, which stays open; the claim is the practical one, holding at half budget, both reference ends and both seeds.

**Limits.** One 24-site instance, one filling, one budget, two seeds; the comparator is the unprojected spin-reading network; the optimiser recipe (R38 mixture, `vmc.py` step and shift defaults) was developed on `che` and not retuned for `nnbf`; every expressivity datum is 12-site and unconverged.

**Closure certificate**: EXHAUSTED at this budget, drafted in log `round-29-critic` section 2, pending ledger row 10 (refused steps: R40, R22 seeds 1–2, R17).

* **Scope**: open $4\times3$ ($5+5$, $4+4$) exact sums; open $8\times3$, $N_e=20$, guided SR at 1024 samples per step, 3 CPU-h per run; $U=8$, $t'=-0.25$, $S=0$; cost to 576 sites.
* **Compute spent**: 6,979 CPU-hours in 20 legacy Slurm jobs; 26.9 in seven gated jobs of the 300 CPU-hour leaf budget; about 2.3 interactive since r21 (log `round-23` section 8).
