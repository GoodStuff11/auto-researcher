# P4.1 round 25: per-leaf critic of round 23 applied (editor, 2026-11-17)

Source: per-leaf critic report, round 23 (orchestrator's transcription, objections 1–16, CHECKED BY list), with the orchestrator's decisions. No jobs run, no commits. State stays DORMANT.

## 1. What was applied, by objection

1. r22 resume refusal was not a G2 refusal (predicted AMBIGUOUS p 0.55 is below the 0.8 threshold; gate PASS PENDING AUDIT, digest b2803f541c2d; 13 CPU-h goes to the compute auditor). Ledger row 4 REOPENED. Cards R2, R19, R20: Outcome "SHELVED (pending the compute auditor)"; Why shelved "SHELVED pending: the 13 CPU-h resume needs the compute auditor; a cheaper decisive version (rank-5 arm alone, ~1.4 CPU-h) is being costed". New ledger row 9 for R40 (charge-only vs exactly projected spin-reading comparison at 12 sites, being designed by another worker; card not created here).
2. One position on the 12-site resume: R2 no longer says a 12-site verdict cannot change the decision; summary "What it would take" names about 13 CPU-h of the existing exact optimiser (a cheaper version being costed).
3. Answer hardness clause replaced with the critic's text (sum over exponentially many spin arrangements, no cheaper method known); "Mechanism and boundary" now says the commutant lemma covers factors diagonal in the up/down occupation basis only, with no lower bound over all algorithms, and that R22 is exact and polynomial in another basis.
4. Answer accuracy clause replaced with the critic's text (one hint: 35% at 12 sites, same unconverged training, never converged or repeated), merged into the second sentence to keep the Answer at "Partly." plus two sentences. "What it would take" names that comparison (R40, row 9).
5. "What we showed": sampled optimisers "removed about half of the energy error, against three quarters for exact optimisation" (R38 R(60) 0.512 vs 0.7475).
6. R16 stays KILLED; Why this kills it now rests on magnitude (route gain at most 0.0097 g at both polished parents, PASS bar 0.05 g); Strongest case states the direction-count mismatch (foil 120 and 144 new independent directions vs squeezed 90 and 96; tangent ranks 284 vs 314 on `che`, 314 vs 362 on `pcg`), that per direction the squeezed term beat the foil, the single foil draw (seed 7), route–foil gaps no larger than the negative controls, and both parents' first-order residuals (0.0023 g `che`, 0.0177 g `pcg`).
7. R17: cost of the refused training tier added as a post-hoc estimate (about 2.5–3 CPU-h: about 10 ladder steps per arm at 7–8 CPU-min, as job 742206 took 8–12), predicted trained gain below 0.05 g, p about 0.9 (orchestrator's prediction; no cost was stated at the refusal).
8. R21: KILLED at G2 by dominance (orchestrator decision): R38 runs the same protocol at the same 0.41 CPU-min per step and reached R(15) 0.364 against R21's 0.195. R21's own cost: job 748762, 19.0 CPU-min (trsr 6.0, shift 6.1, ladder 6.9).
9. R32, R36: Why this kills it now cites the measured per-variant values against the KILL line base + 0.02 = 0.036 (blocked −0.002, −0.007; Ledoit–Wolf 0.016, NERCOME 0.021, factor −0.011), not the class bound. L59 narrowed: title "In one sampled SR step at about 6 samples per parameter (one start point, 16,384 samples), force noise dominates: the exact force recovers 60% of the exact step, the exact metric none."; the "bounded by the exact-metric arm" sentence replaced; scope and counter-evidence updated (R39, R34 killed).
10. `tests/P4.1_cheap_su2_r25/RULE_B.md`: dated erratum appended mapping old route numbers (R30 → R35, R29 → R34, R33 → R37, R27-blocked → R32; old R31 → R36 by variant; old R34 unmapped).
11. R30's translation repair (Anderson on the LM direction before the line search) added as ledger row 8, OPEN; no honest T0 dismissal was available (the card itself lists it as not ruled out).
12. R22: the amended one-seed rule quoted (KILL factor 2.5 on endpoint and extrapolation, `tests/P4.1_cheap_su2_r23/RULE_amend1.md`).
13. "What we showed": pairing credit now from the like-for-like rank-5 ablation (45% less error, rho_abl 0.550); "in one training run" added to the 53 times sentence.
14. "Entered because": rows 1–3 open at the time (now KILLED) and row 4's invalid refusal (now REOPENED), not the R17/R22 refusals.
15. Analogy Map (problem.md): R22 SHELVED (was LIVE); R30 tried and killed (was parked), translation gap restated, repair untried (row 8); force shrinkage tried (R39, killed); R38 killed.
16. Compute disclosures appended to `round-23.md` section 8 (R27–R29 batch 34.6 CPU-min interactive; guided run 12.0 CPU-min with unpinned BLAS; repeat BLAS failure; no result invalidated).

Rows 1 and 3 closed as kills in round 24 (R34 KILLED at G4, R39 KILLED at T1); the ledger shows them. Cards R34, R39 and any R40 file were not edited.

## 2. Checked by

Set to "per-leaf critic, 2026-11-17: upheld" on R27, R29, R31, R33, R35, R37, R38; with notes on R30 (objection 11), R32 and R36 (objection 9), R16 (reason corrected, objection 6); R22 (seeds 1–2 refusal valid); R17 (refusal incomplete, objection 7); R21 (reclassified KILLED by dominance, objection 8); R2, R19, R20 (resume refusal invalid, objection 1).

## 3. Text moved out of working files

Moved from cards to keep them under 900 tokens:

Card R21.md, Lineage before round 25:

* **Lineage**: Operation A on R15d, R18 and the r20 sampled-LM probes, Round 20: the "SR or minSR variant" branch of r20's NOT-VIABLE decision tree. Gates at T0: G0 (SR with a relative eigenvalue cutoff is a known tool, used here as the leaf's optimiser, not claimed as a solution), G1 and G3 PASS, G2 (L19 says sampler-only changes leave infinite-sample SR fixed points unchanged; truncation changes the flow, so L19 does not predict the outcome), G4 (cost of SR or minSR per step, both scale to 24 sites). Prediction: NOT-VIABLE $p\approx0.5$, MARGINAL 0.3, VIABLE 0.2. Cheapest falsifier (T1, about 12 CPU-minutes, so a quick-note batch job of at most 1 CPU-hour): from `che`'s trained parameters, 15 SR steps with 16,384 i.i.d. exact samples per step, relative cutoff $10^{-4}$, metric step length 0.05, against a control arm with the training's absolute shift $10^{-3}$; every step scored by the exact energy, no in-sample line search (L52). Pre-registered: VIABLE if $R_{\mathrm{trSR}}(15)\ge0.20$, at least 0.15 above the control, and no step raises the exact energy by more than 0.02 of the gap; NOT-VIABLE if $R_{\mathrm{trSR}}(15)<0.08$; MARGINAL otherwise. VIABLE sets the 24-site protocol (with the critic's agreement); NOT-VIABLE closes sampled polishing at about 6 samples per parameter, and R2 can then be decided only at 12 sites (R19, R20).

Card R22.md, Lineage before round 25:

* **Lineage**: translation repair of R0.4, restoring sparsity by changing the coupling tree; cross-field translation from lattice gauge theory and SU(2) tree tensor networks, Round 20; the "tree spin factor" R2.5 left open. Gates at T0: G0 narrow PASS (no fermionic balanced-tree version with a charge determinant found); G1, G3 PASS ($F$ must learn the spin-sector sign, the Hard Core); G2 OPEN, the main risk (exact-singlet forms were 4–7 times worse in job 451886); G4 PASS on paper. Falsifier 1 rule: KILL if the connected-state exponent exceeds 3, PASS if $\le2.6$ (`tests/P4.1_cheap_su2_r21/`). Falsifier 2: 3-seed rule in `tests/P4.1_cheap_su2_r23/RULE.md` (KILL if at least the polished `che`'s error, PASS-T1 if $\le0.6$ of it), amended to a one-seed KILL (Outcome).

Card R22 also lost two citations from The idea and Source analogy for space: "R0.4 died because sequential coupling (GUGA, Vieijra et al.) ..." (now "Sequential coupling (R0.4) ...") and "Jacobi trees in nuclear few-body physics"; prior-art list "arXiv:1905.06034, 2104.14869, 2604.06841" (1905.06034 and 2104.14869 remain in the Analogy Map).

Leaf summary as it stood before this round (superseded; detail dropped from the chain of evidence, such as the fixed-cutoff polish rho_pol 0.761 of job 731462, R20 step 1's 5.49-fold rise, and the per-optimiser rates L-BFGS 0.13–0.17x, truncated Newton 0.05x, Anderson 0.80x, whitened L-BFGS 1.12x, is on cards R18pp, R19, R20, R27–R30):

> # Subproblem: P4.1 Exact spin-SU(2) for fermionic NQS at sub-projection cost
> **Parent Chain**: P4 -> P4.1  **Status**: DORMANT
>
> ## Route Portfolio
> | Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
> | :--- | :--- | :--- | :--- | :--- | :--- |
> | none live | — | — | — | — | every ledger row closed (rows 1–3 KILLED in round 24); orchestrator decides the exit |
>
> All routes are KILLED or SHELVED (`kill_index.md`). Round history: logs `round-23`, `round-24`, `round-24-R28`.
>
> ## Reopening Ledger
> | Item | Source | Route or test | Outcome |
> | :--- | :--- | :--- | :--- |
> | 1. sample reuse lifting the sampled plateau | closure attempt, item 5 | R34: 24-site cost ratio | KILLED(G4): (sampling + row)/row 1.05–1.08 for `che`, `pcg`; local energy is 85–90% of a step |
> | 2. exact 12-site optimiser $\ge3\times$ the ladder | closure attempt, item 5 | R28 hybrid (metric refresh every $k$ gradient steps) | KILLED(T0): best schedule bounded at 1.3–1.8× from round 22's trajectory (log `round-24-R28`) |
> | 3. force shrinkage (R36's translation repair) | closure attempt, item 5 | R39 James–Stein force shrinkage, one-step screen (`RULE_E.md`) | KILLED(T1): Rbar 0.005 and −0.008 vs base 0.016 (exact force 0.171) |
> | 4. 12-site convergence of R2, control and ablation | What would close it | R19/R20 resume, 13.0 CPU-h | GATE-REFUSED(G2: predicted AMBIGUOUS p 0.55) |
> | 5. spin-tree basis beyond seed 0 | What would close it | R22 seeds 1–2, 5.4 CPU-h | GATE-REFUSED(G2: predicted KILL p 0.99) |
> | 6. charge-only extensive terms at the polished parent | What would close it | R16, R17 tangent rule | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
> | 7. sampled optimiser reaching the polished 12-site state | What would close it | R21, R38 | R21 MARGINAL, superseded; R38 KILLED(T1) |
>
> ## Leaf Status Summary
> * **State**: DORMANT
> * **Updated**: 2026-11-17 (round 24: R34 KILLED(G4), R39 force shrinkage KILLED(T1), R28 hybrid KILLED(T0); every ledger row closed; round 23: ladder resume refused; R21 MARGINAL, R38 FAILS; R16, R30 KILLED; R17, R22 SHELVED).
> * **Entered because**: a closure certificate (EXHAUSTED at this scale, as pre-stated below) was attempted but items 4–5 cannot be filled honestly: R17 and R22 were refused, not killed, and ledger rows 1–3 have neither a result nor a refusal. Per-leaf critic sign-off owed.
> * **Question**: Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
> * **Answer**: Partly. Exact total spin stays cheap (1.3 to 3.6 times the cost of the simplest exactly projected version at 12 to 48 sites) if the backflow network reads only where the charges are, and we proved that letting it read the spins makes exact projection exponentially costly in the worst case. Whether the charge-only network gives up accuracy is not known, because no optimiser we have brings the competing networks to their best, even on a 12-site cluster, and none that works from samples gets close enough to test larger ones.
> * **What we showed**: On a 12-site cluster a sum of 8 charge-only components beat an exactly projected network that reads spins by 35%, but a fixed number of components gains only a bounded total energy. The size-independent form, a single pairing wave function whose orbitals follow the charges, had 41% less energy error than the comparable charge-only network without pairing after the longest exact optimisation we could afford, but neither had stopped improving. A basis in which every state has exact spin by construction did 53 times worse, because the network could not learn the signs, and optimisers that work from samples stopped at about half the improvement exact optimisation reaches.
> * **What it would take**: An optimiser that works from samples and reaches the exactly optimised 12-site energies, then the same comparison at 24 sites; or an exact optimiser at least three times faster to finish the 12-site comparison (about 4 CPU-hours). Neither exists; the last cheap candidates were killed in round 24.
> * **Conclusion**: Exact $S$ is cheap only when every spin-breaking ingredient sits inside a Gaussian per charge configuration (commutant lemma plus Lemma B). Within that class the lead route R2 holds a provisional, unconverged 12-site lead ($\rho_{\mathrm{pol}}=0.586$ against `che`, $\rho_{\mathrm{abl}}=0.550$ against its own rank-5 truncation), still an optimiser comparison (L33); no scalable optimiser reaches the polished baseline, so no extensivity test can run.
>
> **The question in plain terms.** Neural quantum states (NQS) for the doped Hubbard model build a determinant or Pfaffian whose orbitals depend on the configuration through a neural backflow (NNBF). The ground state has a definite total spin $S$; imposing it exactly helps accuracy and is needed for sector-resolved physics. The standard tool, spin projection by quadrature over $N_{\mathrm{grid}}$ rotations, is cheap only when every correlator reads charges alone. Can exact $S$ be had at polynomial cost without giving up the spin information that makes spin-resolved networks accurate?
>
> **Mechanism and boundary.** A commutant lemma: a diagonal factor commutes with spin rotations only if it depends on the charge configuration $c$ alone, so projecting a spin-reading network sums over all $\binom{N_s}{N_s/2}$ spin arrangements of $c$ ($N_s$ singly occupied sites), #P-hard in the worst case. Lemma B: if the state at each $c$ is a sum of $K$ spin-conserving Gaussians, exact projection costs $K$ closed-form evaluations, $\mathcal O(KN^3)$ (Che, arXiv:2608.22925). Ten routes bringing spin information in otherwise died at Gates 0–1 (`kill_index.md`). R22 escaped the lemma with a balanced spin-coupling-tree basis, exact by construction and sparse ($\mathcal O(L^{2.1})$ connected states up to $L=1024$), but lost on the sign.
>
> **Chain of evidence** (open $4\times3$, $5+5$, $U=8$, $t'=-0.25$, exact sums; relative energy error $\epsilon$). (1) Rank-8 mixture (R1) beat a projected spin-resolved NNBF by 35% at matched parameters (job 508440), but died at G4. (2) R2 `pcg` projects one charge-backflowed singlet-plus-triplet pairing Gaussian exactly at 3.6 times `che`'s per-step cost at 48 sites; after SR, $\epsilon=1.28\times10^{-3}$ vs $2.86\times10^{-3}$ (`che`), T2a PASS. (3) At 24 sites SR plateaued (W FAIL, R15b KILL). (4) Exact linear-method (LM) steps showed the plateau is the optimiser: polished `che` reached $1.32\times10^{-3}$ (R18''), erasing (2). (5) A fixed-cutoff polish gave $\rho_{\mathrm{pol}}=0.761$ (job 731462) but was frozen (R19); truncating `pcg`'s pairing to UHF rank raised $\epsilon$ 5.49 times (R20 step 1). (6) Cutoff-ladder polish (job 742206): `pcg` $4.23\times10^{-4}$, `che` $7.22\times10^{-4}$, rank-5 ablation $7.69\times10^{-4}$, all still falling (fitted tails 22–100% of $\epsilon$); the 13 CPU-h resume was refused as predictably AMBIGUOUS. (7) Faster exact optimisers lost to the ladder per CPU-minute: L-BFGS 0.13–0.17×, truncated Newton 0.05×, Anderson 0.80×, whitened L-BFGS 1.12×, and any metric-refresh schedule of it at most 1.8× by a T0 bound (R27–R30). (8) Sampled SR: truncation no better than a shift, R(15) 0.195 (R21); guided sampling reaches that at step 3 but plateaus at R(60) 0.512 against 0.7475 exact (R38); force noise limits the step (R32, R35–R37); force shrinkage (R39) and sample reuse (R34) do not help. (9) R22 seed 0: 52.6 times `che`'s error. (10) At the polished parents the squeezed frame gains no more than generic directions (R16 killed); plaquette charge Jastrows stay below 1.3% of the gap (R17).
>
> **Why alternatives fail.** Exact singlets by construction are 4–7 times worse than projecting a broken-symmetry state (job 451886; R0.1, R3.4, R4.2; R22 worse still). Mixtures and rational heads gained nothing (R5.5, R5.4h, R3iii). Size-extending refills die on Gate 4 or cost (R4–R12, RG1–RG3); R23–R26 at G1 or G2.
>
> **Lessons beyond the leaf** (`library.md`): L11 (exact non-Abelian symmetry is polynomial only inside a closed-form covariant object), L10, L33, L29, L52, L55, L1, L56; proposed in round 23: force noise limits sampled SR; exact multisecant optimisers do not beat the ladder; guided sampling plateaus at half the gap.
>
> **Strongest objection.** "The 12-site lead is real: 0.586 already sits on the keep line." Arms whose fitted tails are 22–100% of their error cannot attribute a 41% gap to the ansatz rather than the optimiser (L33), and a 12-site result cannot show a per-site gain; unresolved, which is why the leaf is DORMANT, not closed.
>
> **Limits.** Every positive accuracy result is 12-site, one filling; 24-site SR runs are not capacity statements; no per-site trend.
>
> * **Scope**: open $4\times3$, $5+5$ and $4+4$, $U=8$, $t'=-0.25$, $S=0$, exact sums; 24-site SR at 1024 samples; cost to 576 sites; tree counts to 4096.
> * **Compute spent**: 6,979 CPU-hours in 20 legacy Slurm jobs; 8.0 in five gated jobs of the 300 CPU-hour leaf budget; about 2.3 interactive since r21 (6.7 CPU-min in round 24) (1.5 unledgered in round 22; an 18.8 CPU-min run against a 10 cap; log `round-23` section 8).
> * **Most promising direction**: R2 `pcg` (SHELVED): one size-consistent broken-symmetry pairing Gaussian, exactly projected at bounded cost ratio. For: T2a PASS, cost PASS, provisional $\rho_{\mathrm{pol}}=0.586$ and $\rho_{\mathrm{abl}}=0.550$, pairing used. Against: unconverged arms, no 24-site optimiser.
> * **Next step**: no open ledger row; exit 3 (EXHAUSTED at this budget) applies, for the orchestrator to take with the closure certificate.
> * **What would close it**: rows 1–3 are now KILLED, so exit 3, EXHAUSTED at this budget (rows 4–5 are G2 refusals), with the closure certificate. A surviving row reopens the leaf with that route only. PROVEN-IMPOSSIBLE is out of reach: the hardness results bound the spin-reading class only.

Card R30 Source analogy citations shortened for space; full forms: Pulay, Chem. Phys. Lett. 73, 393 (1980); Anderson, J. ACM 12, 547 (1965).
