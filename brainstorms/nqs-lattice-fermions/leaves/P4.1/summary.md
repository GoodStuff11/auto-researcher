# Subproblem: P4.1 Exact spin-SU(2) for fermionic NQS at sub-projection cost
**Parent Chain**: P4 -> P4.1  **Status**: ANSWERED (practical question, scoped; per-leaf critic sign-off owed)

## Route Portfolio
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R42 practitioner-24 | T2 | complete: CHE-NO-WORSE | — | — | supports the scoped ANSWERED exit |

All other routes are KILLED or SHELVED (`kill_index.md`). Round history: logs `round-23` to `round-28`.

## Reopening Ledger
All rows closed (log `round-28` section 3): 1 R34 KILLED(G4); 2 R28 KILLED(T0); 3 R39 KILLED(T1); 4 R2 vs its rank-5 ablation, ABL-CONFOUNDED (job 759446); 5 R22 seeds 1–2 GATE-REFUSED(G2); 6 R16 KILLED(T1), R17 GATE-REFUSED(G2); 7 R21 KILLED(G2), R38 KILLED(T1); 8 R41 KILLED(T1); 9 R40 GATE-REFUSED(auditor), recast at 24 sites as R42: CHE-NO-WORSE (job 759907).

## Leaf Status Summary
* **State**: ANSWERED (practical question, scoped; per-leaf critic sign-off owed)
* **Updated**: 2026-11-17 (round 28: job 759907 CHE-NO-WORSE at 24 sites; job 759446 closed ledger row 4 ABL-CONFOUNDED; every ledger row closed).
* **Question**: Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
* **Answer**: Partly. Exact total spin is cheap when the backflow network reads only where the charges are (exponential, with no cheaper method known, when it reads the up/down spin pattern), and in practice this cheap form won: on 24 sites, trained the same way at 3 CPU-h per run, one core, it ended with about half the energy error of the usual spin-reading network, which also lost the spin symmetry. Whether the restriction costs expressivity is not settled: on 12 sites, in unconverged exact-gradient training, the charge-only network was 1.7 times less accurate than the same network reading spins.
* **What we showed**: Cost: the charge-only form costs 1.3 to 3.6 times the simplest exactly projected version at 12 to 48 sites. Practice: on a 24-site strip (20 electrons, two random seeds), with the same sampled optimiser and equal CPU time, the exactly projected charge-only network's energy error, measured against the best available reference, was 0.51–0.56 of the spin-reading network's (the same at half the time), and the spin-reading network had total spin squared 1.1–1.2 where the ground state has 0. Expressivity, 12 sites only: with only the network's input changed and exact-gradient training, reading charges gave 1.66 times the error of reading spins (neither converged), and only an 8-term charge-only mixture that does not scale beat the spin-reading network. A pairing wave function whose orbitals follow the charges had 38% less error than the same code cut back to separate up and down orbitals, short of the preset bar for crediting pairing.
* **What it would take**: For expressivity: the same network reading charges versus spins, both converged, at 24 or more sites; that needs a sampled optimiser that reaches converged energies there. The optimisation leaf found one that reaches 99% of the polished 12-site state with 262,000 samples per step, the kind of tool this comparison would need at scale. The practical answer would be overturned by the spin-reading network winning at a larger budget, another filling or size, or with a cheap spin penalty at equal CPU time.
* **Conclusion**: Exact $S$ is cheap only when every spin-breaking ingredient sits inside a Gaussian per charge configuration (commutant lemma plus Lemma B). Within that class the Che-projected charge-only backflow (`che`) beats the unprojected spin-reading NNBF at equal compute on 24 sites ($\rho$ 0.51–0.56), while the 12-site matched-parameter datum (1.66) stays an optimiser comparison (L33) that need not transfer (L39). R2's pairing gain over its rank-5 ablation ($\rho_{\mathrm{abl}}=0.623$) is unattributed.

**The question in plain terms.** Neural quantum states (NQS) for the doped Hubbard model build a determinant or Pfaffian whose orbitals depend on the configuration through a neural backflow (NNBF). The ground state has a definite total spin $S$; imposing it exactly helps accuracy and is needed for sector-resolved physics. Spin projection by quadrature over $N_{\mathrm{grid}}$ rotations is cheap only when every correlator reads charges alone. Can exact $S$ be had at polynomial cost without giving up spin information?

**Mechanism and boundary.** Commutant lemma: a factor diagonal in the up/down occupation basis commutes with spin rotations only if it reads the charge configuration $c$ alone, so projecting a spin-reading network sums over all $\binom{N_s}{N_s/2}$ spin arrangements of $c$ ($N_s$ singly occupied sites); no lower bound over all algorithms is claimed. Lemma B: a sum of $K$ spin-conserving Gaussians per $c$ projects exactly at $\mathcal O(KN^3)$ (Che, arXiv:2608.22925). Ten spin-reading routes died at Gates 0–1.

**Chain of evidence** (12 sites: open $4\times3$, $5+5$, $U=8$, $t'=-0.25$, exact sums; 24 sites: open $8\times3$, $N_e=20$). (1) L-BFGS, matched parameters (jobs 451886/508440): charge-reading lost 1.66-fold to spin-reading (both exactly projected); only the rank-8 mixture R1 won, and it died at G4. (2) R2 `pcg` projects one charge-backflowed pairing Gaussian at 3.6× `che`'s step cost at 48 sites; its SR lead vanished once `che` was polished by exact linear-method steps. (3) Ladder polish to the predicted-gain stop (jobs 742206, 759446): `pcg` $\epsilon=2.77\times10^{-4}$ against its rank-5 ablation $4.44\times10^{-4}$, $\rho_{\mathrm{abl}}=0.623$, ABL-CONFOUNDED (R20). (4) No sampled optimiser reaches polished 12-site states (R21 0.195, R38 plateau 0.512 of the gap; R32–R39, R41 do not help). (5) R42 (job 759907): guided SR, 180 CPU-min one core per run, seeds 0–1: $E_{\mathrm{che}}-E_{\mathrm{nnbf}}=-0.149$, $-0.155$; $\rho$ 0.51–0.56 at both ends of the DMRG bracket, half budget 0.53–0.58; `nnbf` $\langle S^2\rangle$ 1.11–1.18; all validity checks pass.

**Why alternatives fail.** Exact singlets by construction are 4–53 times worse (R0.1, R3.4, R4.2, R22: sign not learned). Mixtures and rational heads gained nothing (R5.5, R5.4h, R3iii). Size-extending refills die on Gate 4 or cost (R4–R12, RG1–RG3, R23–R26).

**Lessons beyond the leaf** (`library.md`): L11, L10, L33, L35, L39, L29, L52, L55, L56, L59, L61; new entries proposed in log `round-28`.

**Strongest objection.** "Equal-compute, unconverged training compares optimisers, not ansätze (L33)." Agreed for expressivity, which stays open; the scoped claim is the practical one, and it holds at half budget, both reference ends and both seeds, about 0.5 below the 1.10 line.

**Limits.** One 24-site instance, one filling, one budget, two seeds; the comparator is the unprojected spin-reading network; every expressivity datum is 12-site and unconverged.

**Closure certificate** (per-leaf critic sign-off owed):
1. *Claim*: (a) on the open $8\times3$ strip ($N_e=20$, $U=8$, $t'=-0.25$, $S^z=0$, point group (+,+)), two seeds, guided SR for 3 CPU-h per run on one core, the exactly projected charge-only backflow had 0.51–0.56 of the energy gap of the unprojected spin-reading backflow (same parameter count), which was spin-contaminated ($\langle S^2\rangle$ 1.1–1.2); (b) at 12 sites: cost ratio 1.3–3.6 (12–48 sites), a 1.66× restriction cost in unconverged exact-gradient training, R2's pairing gain unattributed. Not claimed: converged expressivity.
2. *Decisive evidence*: job 759907 under `tests/P4.1_cheap_su2_r42/RULE.md` (CHE-NO-WORSE; auditor-approved, digest 4ed16de4e3f2); job 759446 under RULE_amend3; jobs 451886/508440; cost gate on card R2. Per-leaf critic review owed.
3. *Why no doubt remains* (claim a): optimiser confound: the claim is equal effort, and half budget agrees; unconverged DMRG reference: both bracket ends; seeds: both, far from the lines; validity: mass fraction 0.80, effective samples 900–912, `che` $|S^2_{\mathrm{loc}}|\le7.8\times10^{-10}$; `che` also leads at equal step count (log `round-28`).
4. *Falsifiers run*: ledger rows 1–9 above; rows 5, 6 (R17) and 9 were refused and bear on expressivity or the pairing form, not on claim (a).
5. *Missing ingredients*: the converging optimiser (rows 2, 3, 7, 8 killed) and the converged 12-site comparison (row 9 refused) leave expressivity open, as the Answer states.
6. *What would reopen it*: a spin-reading network (penalised or projected) beating `che` at equal CPU, at a larger budget, size or other filling; or a 24-site optimiser reaching converged energies, enabling the expressivity comparison (R40 at 12 sites: 11.0 CPU-h, refused as not decisive).

* **Scope**: open $4\times3$ ($5+5$, $4+4$) exact sums; open $8\times3$, $N_e=20$, guided SR at 1024 samples per step, 3 CPU-h per run; $U=8$, $t'=-0.25$, $S=0$; cost to 576 sites.
* **Compute spent**: 6,979 CPU-hours in 20 legacy Slurm jobs; 26.9 in seven gated jobs of the 300 CPU-hour leaf budget; about 2.3 interactive since r21 (log `round-23` section 8).
