# Subproblem: P4.1 Exact spin-SU(2) for fermionic NQS at sub-projection cost
**Parent Chain**: P4 -> P4.1  **Status**: DORMANT

## Route Portfolio
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R47 (distil a 24-site DMRG state into both networks) | T0 (T1 designed) | LIVE | T1 code, local check, voi note, gate check (no submission yet) | `tests/P4.1_cheap_su2_r47/RULE_T1.md` | PASS → T2 at 24 sites (~26 CPU-h) with the teacher $\langle S^2\rangle$ correction; KILL → KILLED(T1), certificate adopted; UNSETTLED → numbers, orchestrator decides a seed |

All other routes are KILLED or SHELVED (`kill_index.md`); R42 (CHE-NO-WORSE), its falsifier R43 (CHE-STILL-WINS) and R45 (LOSS, job 769413) are complete. History: logs `round-23` to `round-33`.

## Reopening Ledger
| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| 1–3. sample reuse; exact optimiser ≥3× the ladder; force shrinkage | closure attempt | R34, R28, R39 | KILLED(G4), KILLED(T0), KILLED(T1) |
| 4. R2 vs its rank-5 ablation, converged at 12 sites | What would close it | job 759446, RULE_amend3 | DIAGNOSTIC-HOLDS (pairing gain unattributed; label ABL-CONFOUNDED) |
| 5. spin-tree basis beyond seed 0 | What would close it | R22 seeds 1–2, 5.4 CPU-h | GATE-REFUSED(G2) |
| 6. charge-only extensive terms | What would close it | R16; R17 | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
| 7. sampled optimiser reaching the polished 12-site state | What would close it | R21, R38 | KILLED(G2), KILLED(T1) |
| 8. R30's translation repair | critic r23, item 11 | R41 | KILLED(T1) |
| 9. converged charge-only vs projected spin-reading | critic r23, item 4 | R40 (11.0 CPU-h), recast at 24 sites as R42 | GATE-REFUSED(auditor); R42 CHE-NO-WORSE (job 759907) |
| 10. spin-flip-even nnbf baseline at 24 sites | critic r29, Major 2 | R43 (job 764949), 2 seeds, equal compute | CHE-STILL-WINS: R42 stands |
| 11. 12-site capacity loss at 24 sites | R45 LOSS (job 769413) | R47 T1 (~7.5 CPU-h), then T2 (~26) | open: T1 designed |

## Leaf Status Summary
* **State**: DORMANT
* **Updated**: 2026-11-18 (R45 LOSS, job 769413, upheld by critic; R47 released, T1 designed; logs `round-33`, `round-34-critic`).
* **Entered because**: closure certificate blocked by unrun row 10; row 10 has run (R42 stands); DORMANT while R47's T1 is prepared (released by R45's LOSS).
* **Question**: Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
* **Answer**: Partly. Exact total spin is cheap if the backflow network reads only where the charges are, and exponential, with no cheaper method known, if a network built from factors that each read a configuration's up/down spin pattern does; in one test (a 24-site strip at one filling, two seeds, the same optimiser, tuned on the charge-only network, at 3 CPU-h per run, one core), the charge-only network ended with about half the energy error (0.49 to 0.56 of it) of the same network reading spins, with or without spin-flip symmetry imposed; even the symmetrised version still had total spin squared 0.6 to 0.7 where the ground state's is expected to be 0. On 12 sites, from one random start and at equal fitting effort (one seed; its spread is unmeasured), the charge-only network ended 2.2 times further from the exact ground state than the exactly projected spin-reading network, and the gap widened with effort (1.6 to 2.2 between 500 and 2000 steps); neither had converged.
* **What we showed**: Cost: the charge-only form is 1.3 to 3.6 times the simplest exactly projected version at 12 to 48 sites. Practice: on a 24-site strip (20 electrons, two seeds), same sampled optimiser and equal CPU time, the projected charge-only network's energy error was 0.51–0.56 of the spin-reading network's, whose total spin squared was 1.1–1.2 instead of 0; imposing global spin-flip symmetry on it lowered that to 0.6–0.7 but not its energy (ratio 0.49–0.55). Bases built to have exact total spin give it at polynomial cost but were 4 to 53 times less accurate. Capacity, 12 sites, one random start: fitted to the exact ground state for 2,000 exact-gradient steps each, the charge-only network's infidelity (one minus the squared overlap) was $2.9\times10^{-4}$ against $1.3\times10^{-4}$ for the exactly projected spin-reading network (1.85 times at half of each network's fitting time).
* **What it would take**: Beyond 12 sites: fit both networks to a near-exact 24-site state, the spin-reading one unprojected; only a clear win for it is informative (about 26 CPU-hours, after a 7.5 CPU-hour 12-site rehearsal that unprojected fits still show the loss). Converged 24-site energy training needs a sampled optimiser that converges there (one reaches 99% of the best 12-site energy; untested at 24). The practical result falls if the spin-reading network, with or without spin-flip symmetry, wins at a larger budget, size or filling, or with a cheap spin penalty at equal CPU.
* **Conclusion**: For networks whose spin dependence enters as factors diagonal in the up/down occupation basis, exact $S$ is cheap only when every spin-breaking ingredient sits inside a Gaussian per charge configuration (commutant lemma plus Lemma B). Within that class `che` beats the unprojected NNBF ($\rho$ 0.51–0.56) and its spin-flip-even form ($\rho'$ 0.49–0.55; row 10) at equal compute on 24 sites. At 12 sites the restriction costs fit accuracy at equal effort, with a gap that widens with budget ($\kappa$ 1.64 → 2.23 from 500 to 2000 evaluations; 2.62 at equal CPU): $\kappa=(1-F)_{\rm che}/(1-F)_{\rm nnbf}$ (R45, seed 0; chain item 7), cf. energy ratios 1.66, 2.10 (L33); 24 sites (L39) is R47's question. R2's pairing gain over its rank-5 ablation ($\rho_{\mathrm{abl}}=0.623$) is unattributed.

**The question in plain terms.** Neural quantum states (NQS) for the doped Hubbard model build a determinant or Pfaffian whose orbitals depend on the configuration through a neural backflow (NNBF). Imposing the ground state's total spin $S$ exactly helps accuracy and is needed for sector-resolved physics. Quadrature projection over $N_{\mathrm{grid}}$ rotations is cheap only if every correlator reads charges alone: can exact $S$ be had at polynomial cost without giving up spin information?

**Mechanism and boundary.** Commutant lemma: a factor diagonal in the up/down occupation basis commutes with spin rotations only if it reads the charge configuration $c$ alone, so projecting a spin-reading network of such factors sums over all $\binom{N_s}{N_s/2}$ spin arrangements of $c$ ($N_s$ singly occupied sites); no lower bound over all algorithms is claimed. Lemma B: $K$ spin-conserving Gaussians per $c$ project exactly at $\mathcal O(KN^3)$ (Che, arXiv:2608.22925).

**Chain of evidence** (12 sites: open $4\times3$, $U=8$, $t'=-0.25$, exact sums; 24 sites: open $8\times3$, $N_e=20$). (1) L-BFGS, matched parameters (jobs 451886/508440): charge-reading lost 1.66-fold ($5+5$) and 2.10-fold ($4+4$) to spin-reading (both projected); only the non-scaling rank-8 mixture R1 won. (2) R2 `pcg` costs 3.6× `che` per step at 48 sites; its SR lead vanished once `che` was polished. (3) Row 4 (job 759446): $\rho_{\mathrm{abl}}=0.623$, between the credit (0.60) and retraction (0.85) lines. (4) No sampled optimiser reaches polished 12-site states (R21, R38; R32–R39, R41 do not help). (5) R42 (job 759907): guided SR, seeds 0–1, $\rho$ 0.51–0.56 at both DMRG bracket ends, half budget 0.53–0.58, `nnbf` $\langle S^2\rangle$ 1.11–1.18. (6) R43 (job 764949): `nnbf` made spin-flip-even (even $S$ only), same harness and CPU, seeds 0–1: $\rho'$ 0.49–0.55 at both ends, half budget 0.46–0.49, $\langle S^2\rangle$ 0.61–0.68, energy no lower than bare `nnbf` at equal CPU (`round-31`). (7) R45 (job 769413): L-BFGS on $-\log F$ to the exact 12-site ($5+5$) state, 2000 evaluations, seed 0: $\kappa=2.23$ (half 1.85); last-quarter drops `che` 9.0%, under the pre-registered 10% line (12% over the last third, 21% over the last half; still falling), `nnbf` 14.3% (raises $\kappa$ only if `che` does not improve faster, which the trajectory shows); $\kappa$ 1.55, 1.64, 1.95, 2.13, 2.23 at 250–2000 evaluations: LOSS (logs `round-33`, `round-34-critic`).

**Why alternatives fail.** Exact singlets by construction are 4–53 times worse (R0.1, R0.4, R3.4, R4.2, R22: sign not learned). Mixtures, rational heads: no gain (R5.5, R5.4h). Size-extending refills die on Gate 4 or cost (R4–R12, RG1–RG3, R23–R26). Ten spin-reading routes died at G0–G1.

**Lessons beyond the leaf** (`library.md`): L1, L11, L10, L33, L35, L39, L29, L52, L55, L56, L59, L61.

**Strongest objection.** (i) L1 (broken symmetry inflates the error of an unsymmetrised network): `che`'s win may be removal of `nnbf`'s spin-1 contamination, which spin-flip symmetry also removes. Answered by row 10 (R43, chain item 6); a spin penalty is untried. (ii) Unconverged equal-compute training compares optimisers, not ansätze (L33): agreed; the 24-site claim is practical (holds at half budget, both reference ends, both seeds); expressivity rests on chain item 7.

**Limits.** One 24-site instance, one filling, one budget, two seeds; no spin-penalised comparator; the optimiser was tuned on `che`, not `nnbf`; every expressivity datum is 12-site, one seed, unconverged.

**Closure certificate**: EXHAUSTED at this budget, drafted in log `round-29-critic` §2; pending R47 (row 11) (refused steps: R40, R22 seeds 1–2, R17).

* **Scope**: open $4\times3$ ($5+5$, $4+4$) exact sums; open $8\times3$, $N_e=20$, guided SR at 1024 samples per step, 3 CPU-h per run; $U=8$, $t'=-0.25$, $S=0$; cost to 576 sites.
* **Compute spent**: 6,979 CPU-hours in 20 legacy Slurm jobs; 44.9 in ten completed gated jobs of the 300 CPU-hour leaf budget; about 2.3 interactive since r21 (log `round-23` section 8).
