# Subproblem: P4.1 Exact spin-SU(2) for fermionic NQS at sub-projection cost
**Parent Chain**: P4 -> P4.1  **Status**: DORMANT

## Route Portfolio
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R34 `reuse` | T1 bound PASS | LIVE (ledger row 1) | 24-site cost ratio unmeasured | ratio (sampling + row)/(row) $<2$ → KILLED(G4) | $\ge2$ → T1: reuse on top of guided SR, does R(60) pass 0.60? |
| R28 `plbfgs` | T1 AMBIGUOUS (1.12×) | LIVE (ledger row 2) | hybrid not written | hybrid $<1.5\times$ the ladder's rate → KILLED(T1); $\ge3\times$ → SURVIVED-T1 | SURVIVED → resubmit the 12-site convergence resume (about 4 CPU-h) to the gate |

All other routes are KILLED or SHELVED (`kill_index.md`). Round history: log `round-23`.

## Reopening Ledger
| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| 1. sample reuse lifting the sampled plateau | closure attempt, item 5 | R34: 24-site cost ratio (`tests/P4.1_cheap_su2_scaling/vmc.py`, minutes), then T1 | open |
| 2. exact 12-site optimiser $\ge3\times$ the ladder | closure attempt, item 5 | R28 hybrid (metric refresh every $k$ gradient steps), about 30 CPU-min | open |
| 3. force shrinkage (R36's translation repair) | closure attempt, item 5 | new route through the gates; one-step screen in `noise_split.py` (under 10 CPU-min) | open |
| 4. 12-site convergence of R2, control and ablation | What would close it | R19/R20 resume, 13.0 CPU-h | GATE-REFUSED(G2: predicted AMBIGUOUS p 0.55) |
| 5. spin-tree basis beyond seed 0 | What would close it | R22 seeds 1–2, 5.4 CPU-h | GATE-REFUSED(G2: predicted KILL p 0.99) |
| 6. charge-only extensive terms at the polished parent | What would close it | R16, R17 tangent rule | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
| 7. sampled optimiser reaching the polished 12-site state | What would close it | R21, R38 | R21 MARGINAL, superseded; R38 KILLED(T1) |

## Leaf Status Summary
* **State**: DORMANT
* **Updated**: 2026-11-17 (round 23: ladder resume refused; R21 MARGINAL, R38 FAILS; R16, R30 KILLED; R17, R22 SHELVED).
* **Entered because**: a closure certificate (EXHAUSTED at this scale, as pre-stated below) was attempted but items 4–5 cannot be filled honestly: R17 and R22 were refused, not killed, and ledger rows 1–3 have neither a result nor a refusal. Per-leaf critic sign-off owed.
* **Question**: Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
* **Answer**: Partly. Exact total spin stays cheap (1.3 to 3.6 times the cost of the simplest exactly projected version at 12 to 48 sites) if the backflow network reads only where the charges are, and we proved that letting it read the spins makes exact projection exponentially costly in the worst case. Whether the charge-only network gives up accuracy is not known, because no optimiser we have brings the competing networks to their best, even on a 12-site cluster, and none that works from samples gets close enough to test larger ones.
* **What we showed**: On a 12-site cluster a sum of 8 charge-only components beat an exactly projected network that reads spins by 35%, but a fixed number of components gains only a bounded total energy. The size-independent form, a single pairing wave function whose orbitals follow the charges, had 41% less energy error than the comparable charge-only network without pairing after the longest exact optimisation we could afford, but neither had stopped improving. A basis in which every state has exact spin by construction did 53 times worse, because the network could not learn the signs, and optimisers that work from samples stopped at about half the improvement exact optimisation reaches.
* **What it would take**: An optimiser that works from samples and reaches the exactly optimised 12-site energies, then the same comparison at 24 sites; or an exact optimiser at least three times faster to finish the 12-site comparison (about 4 CPU-hours). Neither exists; three cheap candidates remain in the Reopening Ledger.
* **Conclusion**: Exact $S$ is cheap only when every spin-breaking ingredient sits inside a Gaussian per charge configuration (commutant lemma plus Lemma B). Within that class the lead route R2 holds a provisional, unconverged 12-site lead ($\rho_{\mathrm{pol}}=0.586$ against `che`, $\rho_{\mathrm{abl}}=0.550$ against its own rank-5 truncation), still an optimiser comparison (L33); no scalable optimiser reaches the polished baseline, so no extensivity test can run.

**The question in plain terms.** Neural quantum states (NQS) for the doped Hubbard model build a determinant or Pfaffian whose orbitals depend on the configuration through a neural backflow (NNBF). The ground state has a definite total spin $S$; imposing it exactly helps accuracy and is needed for sector-resolved physics. The standard tool, spin projection by quadrature over $N_{\mathrm{grid}}$ rotations, is cheap only when every correlator reads charges alone. Can exact $S$ be had at polynomial cost without giving up the spin information that makes spin-resolved networks accurate?

**Mechanism and boundary.** A commutant lemma: a diagonal factor commutes with spin rotations only if it depends on the charge configuration $c$ alone, so projecting a spin-reading network sums over all $\binom{N_s}{N_s/2}$ spin arrangements of $c$ ($N_s$ singly occupied sites), #P-hard in the worst case. Lemma B: if the state at each $c$ is a sum of $K$ spin-conserving Gaussians, exact projection costs $K$ closed-form evaluations, $\mathcal O(KN^3)$ (Che, arXiv:2608.22925). Ten routes bringing spin information in otherwise died at Gates 0–1 (`kill_index.md`). R22 escaped the lemma with a balanced spin-coupling-tree basis, exact by construction and sparse ($\mathcal O(L^{2.1})$ connected states up to $L=1024$), but lost on the sign.

**Chain of evidence** (open $4\times3$, $5+5$, $U=8$, $t'=-0.25$, exact sums; relative energy error $\epsilon$). (1) Rank-8 mixture (R1) beat a projected spin-resolved NNBF by 35% at matched parameters (job 508440), but died at G4. (2) R2 `pcg` projects one charge-backflowed singlet-plus-triplet pairing Gaussian exactly at 3.6 times `che`'s per-step cost at 48 sites; after SR, $\epsilon=1.28\times10^{-3}$ vs $2.86\times10^{-3}$ (`che`), T2a PASS. (3) At 24 sites SR plateaued (W FAIL, R15b KILL). (4) Exact linear-method (LM) steps showed the plateau is the optimiser: polished `che` reached $1.32\times10^{-3}$ (R18''), erasing (2). (5) A fixed-cutoff polish gave $\rho_{\mathrm{pol}}=0.761$ (job 731462) but was frozen (R19); truncating `pcg`'s pairing to UHF rank raised $\epsilon$ 5.49 times (R20 step 1). (6) Cutoff-ladder polish (job 742206): `pcg` $4.23\times10^{-4}$, `che` $7.22\times10^{-4}$, rank-5 ablation $7.69\times10^{-4}$, all still falling (fitted tails 22–100% of $\epsilon$); the 13 CPU-h resume was refused as predictably AMBIGUOUS. (7) Faster exact optimisers lost to the ladder per CPU-minute: L-BFGS 0.13–0.17×, truncated Newton 0.05×, Anderson 0.80×, whitened L-BFGS 1.12× (R27–R30). (8) Sampled SR: truncation no better than a shift, R(15) 0.195 (R21); guided sampling reaches that at step 3 but plateaus at R(60) 0.512 against 0.7475 exact (R38); force noise, not metric noise, limits the step (R32, R35–R37). (9) R22 seed 0: 52.6 times `che`'s error. (10) At the polished parents the squeezed frame gains no more than generic directions (R16 killed); plaquette charge Jastrows stay below 1.3% of the gap (R17).

**Why alternatives fail.** Exact singlets by construction are 4–7 times worse than projecting a broken-symmetry state (job 451886; R0.1, R3.4, R4.2; R22 worse still). Mixtures and rational heads gained nothing (R5.5, R5.4h, R3iii). Size-extending refills die on Gate 4 or cost (R4–R12, RG1–RG3); R23–R26 at G1 or G2.

**Lessons beyond the leaf** (`library.md`): L11 (exact non-Abelian symmetry is polynomial only inside a closed-form covariant object), L10, L33, L29, L52, L55, L1, L56; proposed in round 23: force noise limits sampled SR; exact multisecant optimisers do not beat the ladder; guided sampling plateaus at half the gap.

**Strongest objection.** "The 12-site lead is real: 0.586 already sits on the keep line." Arms whose fitted tails are 22–100% of their error cannot attribute a 41% gap to the ansatz rather than the optimiser (L33), and a 12-site result cannot show a per-site gain; unresolved, which is why the leaf is DORMANT, not closed.

**Limits.** Every positive accuracy result is 12-site, one filling; 24-site SR runs are not capacity statements; no per-site trend.

* **Scope**: open $4\times3$, $5+5$ and $4+4$, $U=8$, $t'=-0.25$, $S=0$, exact sums; 24-site SR at 1024 samples; cost to 576 sites; tree counts to 4096.
* **Compute spent**: 6,979 CPU-hours in 20 legacy Slurm jobs; 8.0 in five gated jobs of the 300 CPU-hour leaf budget; about 2.2 interactive since r21 (1.5 unledgered in round 22; an 18.8 CPU-min run against a 10 cap; log `round-23` section 8).
* **Most promising direction**: R2 `pcg` (SHELVED): one size-consistent broken-symmetry pairing Gaussian, exactly projected at bounded cost ratio. For: T2a PASS, cost PASS, provisional $\rho_{\mathrm{pol}}=0.586$ and $\rho_{\mathrm{abl}}=0.550$, pairing used. Against: unconverged arms, no 24-site optimiser.
* **Next step**: thinking-queue work on ledger rows 1–3: R34's cost-ratio measurement, the force-shrinkage route through the gates, then R28's hybrid only if rows 1 and 3 leave the 12-site comparison as the only path.
* **What would close it**: if rows 1–3 close as KILLED or GATE-REFUSED, exit 3, EXHAUSTED at this budget (rows 4–5 are G2 refusals), with the closure certificate. A surviving row reopens the leaf with that route only. PROVEN-IMPOSSIBLE is out of reach: the hardness results bound the spin-reading class only.
