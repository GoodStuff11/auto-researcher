### Round 4 — Mutation & Replenishment (2026-11-11)
* **Inputs carried over (binding)**: the round-3 per-leaf critic returned MUTATE with ten objections.
  - (1) Gate 0: for a symmetric charge-conditioned pairing $F_c$, $\det F_c[U_s\cup D_b,D_s\cup D_b]=\det F_c[D_b,D_b]\,\det S_c[U_s,D_s]$, where $D_b$ are the doubly occupied sites and $S_c$ is the (symmetric) Schur complement. That is SQ-RVB's spin factor, so SQ-RVB is the round-1-killed Tocchio family.
  - (2) Holon-dressed RVB already exists (Weng group: arXiv:1808.06173, PRB 99, 205128, arXiv:2106.14898; SA-NNBF 2604.06841). The real barrier is that one AGP per charge configuration may be too restrictive.
  - (3) The cost gate used the wrong competitor: the same exact-$S$ class costs one determinant (Tocchio/R1).
  - (4) Che's projection has a closed form, so its fair slope is 3, not 4.
  - (5) SCF's cost was measured at a toy $D=13$; in 2D the required $D$ grows as $e^{\mathcal O(L_y)}$.
  - (6) The cost and accuracy parameterisations differed.
  - (7) Completeness of one AGP per $c$ was untested.
  - (8) About 21% of SQ-RVB's parameters were dead, and $g_{\mathrm{mech}}$ was confounded.
  - (9) Width-4 accuracy measures only capacity per parameter.
  - (10) The 42-task array could not decide the cost claim.
* **Running elsewhere, not duplicated**: the orchestrator's 9-task width-16 pilot (job 411494: `sqrvb`, `nnbf_SG`, `che`, 3 seeds) measures charge-only against spin-resolved conditioning, both exactly projected.
#### Step 1a — honest per-amplitude cost
* **Code**: `tests/P4.1_cheap_su2_r4/r4_test.py --mode cost` (single core, `OMP_NUM_THREADS=1`; `logs/cost_output.txt`, `results/cost_r4.json`).
  - Every arm uses the same hidden width $H=8$ and gathers only the field entries it needs: $\mathcal O(Hn^2)$ for pairings and $\mathcal O(HnN_e)$ for orbitals, with $n=N_e/2$. This answers objection 6.
  - `che_closed` is the closed form of objection 4. With $\mu$ the eigenvalues of $D_m^{-1}CA^{-1}B$, the amplitude is $\langle x\rvert P_0\lvert\Phi\rangle=\det A\det D_m\int_0^1\prod_i\big(x+(1-x)\mu_i\big)\,dx=\det A\det D_m\sum_ke_k(\mu)\,\frac{k!(n-k)!}{(n+1)!}$. Here $A,B,C,D_m$ are the up/down orbital blocks of round 3, $e_k$ is the $k$-th elementary symmetric polynomial, and the integral is a degree-$n$ polynomial, so it is computed exactly from $\mu$ in $\mathcal O(n^2)$. The total is one eigen-solve, $\mathcal O(n^3)$.
  - Smoke checks (`logs/smoke_output.txt`): `che_closed` agrees with round 3's LU-reused quadrature to $1.1\times10^{-14}$, with the $e_k$ form to $8\times10^{-17}$, and with the full-vector $P_S$ to $1.1\times10^{-14}$.
  - The single-determinant symmetric-$F_c$ singlet arm is `tocchio`, i.e. `kagp` with $K=1$.

| $L$ | $N_e$ | NNBF, unprojected | `tocchio` (1 det) | `kagp2` | `kagp4` | SQ-RVB | Che closed form | Che quadrature (round 3) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 144 | 126 | 0.17 ms | 0.23 ms | 0.45 ms | 0.89 ms | 1.01 ms | 1.11 ms | 1.42 ms |
| 256 | 224 | 0.50 ms | 0.65 ms | 1.29 ms | 2.71 ms | 1.67 ms | 3.12 ms | 5.20 ms |
| 400 | 350 | 1.23 ms | 1.50 ms | 3.09 ms | 6.25 ms | 4.64 ms | 9.53 ms | 17.7 ms |
| 576 | 504 | 2.80 ms | 3.30 ms | 6.73 ms | 13.5 ms | 10.9 ms | 20.8 ms | 53.8 ms |
| slope in $N_e$ | | 2.13 | 2.00 | 2.04 | 1.97 | 2.31 | 2.34 | 2.88 |
| ratio to `tocchio` at $L=576$ | | 0.8 | 1.0 | 2.0 | 4.1 | 3.3 | 6.3 | 16.3 |

* **Reading of the honest cost table**:
  - The cheapest exact-$S$ construction is the standard one: a symmetric charge-conditioned pairing at one $n\times n$ determinant. It costs $1.2\times$ the unprojected spin-resolved NNBF.
  - SQ-RVB is $3.3\times$ `tocchio` for the same per-$c$ spin class (objection 1). Against the fair Che it is only $1.9\times$ cheaper, not the $4.7$–$6.2\times$ claimed in round 3.
  - Closed-form Che is $6.3\times$ `tocchio`, and its slope (2.34, pre-asymptotic, BLAS-bound) is close to that of the determinant arms. The gap is a constant factor, as objection 4 said.
  - The exponential orbit sum for the spin-resolved NNBF costs 2.8 s per amplitude at $L=24$ (`logs/lemma_output.txt`), against milliseconds for all of the above.
  - SCF is dropped from every cost claim (objection 5).
#### Step 1b — completeness of one AGP per charge configuration
* **Dimension count** (`--mode dims`, `logs/dims_output.txt`). For a charge configuration with $N_s$ singles and no doublons, the spin factor lives in the singlet space of dimension $\mathrm{Cat}(N_s/2)$, the Catalan number. The table gives the Jacobian rank, at random parameters, of the tabular per-$c$ maps. A tabular map is one free parameter set per $c$, so it is the completeness ceiling of any charge-conditioned network. The Tocchio map's output was also checked to be an exact singlet ($\le4\times10^{-16}$).

| $N_s$ | singlet dimension | one AGP (Tocchio / SQ-RVB) | 2 AGPs | 3 AGPs | projected UHF (Che) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 4 | 2 | 2 | 2 | 2 | 2 |
| 6 | 5 | 5 | 5 | 5 | 5 |
| 8 | 14 | **13** | 14 | 14 | 14 |
| 10 | 42 | **26** | 42 | 42 | 42 |
| 12 | 132 | **43** | 86 | 129 | 62 |

* **Reading**:
  - One AGP per $c$ is incomplete from $N_s=8$ on: its image is a hypersurface (rank 13 of 14). At $N_s=10$ it covers only a 26-dimensional cone of the 42-dimensional singlet space.
  - Two AGPs, or one projected UHF, are locally complete up to $N_s=10$.
  - For every fixed $K$ the $K$-AGP family is incomplete beyond some $N_s$. The singlet dimension grows as $4^{N_s/2}$, while each family has only polynomially many parameters per $c$. No polynomial family is complete per $c$ at scale, so completeness can only be a small-cluster diagnostic.
* **Tabular fit to $\psi_0$** (Slurm job **411774**, `submit_agp.sbatch`, partition `kim`, 4 h). The $4\times3$ open cluster, $U=8$, $t'=-0.25$, at $N_e=8$, 10 and 12. For every charge configuration the job fits the best per-$c$ member of each family to $\psi_0$'s spin factor $\chi_c$. It reports the per-$N_s$ minimum fidelity, the weighted infidelity $1-\sum_cp_cF_c$ (with $p_c$ the weight of $c$ in $\psi_0$ and $F_c$ the best per-$c$ fidelity), and the energy error $\epsilon_{\mathrm{tab}}$ of the assembled tabular state.
  - Smoke on the $4\times2$ open cluster, $N_e=6$ ($N_s\le6$): all three families reach $F_c=1$ for every $c$ and total infidelity $\le4\times10^{-16}$, as the dimension count predicts (`logs/agp_smoke_4x2_N6.txt`).
  - Results: AGP_RESULTS_PLACEHOLDER
#### Saturation diagnostic (Gate 0 applied to the formulation itself)
* **What is already solved**: "exact $S$ below projection cost" is met by standard constructions.
  - A singlet pairing with charge-only correlators and charge-conditioned backflow (Tocchio et al., PRB 78, 041101; R1) is exact for every $\theta$ at one $n\times n$ determinant, $1.2\times$ the cost of an unprojected NNBF (table above).
  - For broken-symmetry references with charge-only backflow, Che's projection has an $\mathcal O(n^3)$ closed form, at $6.3\times$ the single-determinant cost.
  - The formulation of Section 1 ("cost below $N_{\mathrm{grid}}$ evaluations") is therefore **saturated**. Every route that targeted it (R1, PCG, SQ-RVB, SCF, Route 3.1) either is one of these two constructions or loses to them on cost.
* **What remains open**:
  - (a) Exact $S$ for **spin-resolved** backflow. The round-2 conjecture said this needs the exponential orbit sum, but it was never stated precisely or proved.
  - (b) Whether charge-only conditioning loses accuracy against spin-resolved conditioning once both are exactly projected. The orchestrator's pilot 411494 (`che` against `nnbf_SG`) measures this. It is not repeated here.
  - (c) Whether one AGP per charge configuration is too restrictive. The dimension count proves it is incomplete from $N_s=8$. The tabular job measures how much this costs in energy on $\psi_0$. A cheap exact-$S$ family beyond one AGP per $c$ is needed only if that loss is material.
* **Refined formulation (P4.1-R4, recorded; the Invariants are untouched)**:
  - Inputs: exact-$S$ amplitudes with per-amplitude cost polynomial in $N_e$, which are compared at equal parameterisation with the two standard baselines, `tocchio` (cost 1) and closed-form Che (cost $\approx6$).
  - "Solved" now means one of two things:
    - (i) a precise complexity boundary for exact $S$ under spin-resolved conditioning, i.e. which spin-reading backflows admit polynomial exact projection, with a constructive member beyond charge-only conditioning; or
    - (ii) an exact-$S$ family whose per-$c$ spin factor exceeds one AGP, priced below closed-form Che, that measurably reduces the energy error of `tocchio` at matched parameters on a cluster where $N_s\ge10$ configurations carry weight.
  - The decisive instance moves to $4\times3$ open, $5+5$ ($N_s=10$ is the dominant no-doublon sector at $\delta=1/6$). The $4+4$ cluster has $N_s\le8$, where one AGP per $c$ misses only a codimension-1 set.
#### Generation Batch 4
* **Route 4.0 (SQ-RVB, round-3 survivor)**:
  - Gate 0: **FAIL**. By the Schur-complement identity it is the Tocchio family with a reparameterised charge factor (objection 1).
  - Cost: it is $3.3\times$ the one-determinant member of its own class.
  - *Verdict*: **KILLED**. It stays only as a reported arm in job 411494.
* **Route 4.0b (SCF, round-3 second survivor)**:
  - Gate 1: **FAIL** under the refined formulation. The per-$c$ MPS is beyond one AGP, but the bond dimension it needs grows as $e^{\mathcal O(L_y)}$ in 2D (objection 5), so the cost is exponential in width. The round-3 cost row was a toy $D=13$ with random tensors.
  - *Verdict*: **KILLED** for P4.1-R4. The area-law caveat is recorded in the autopsy.
* **Route 4.1 (Operation B.3 from number projection, plus Operation A on the round-2 dichotomy conjecture): Spin-feature dichotomy.** The route has two halves, a hardness theorem and a constructive bucketing algorithm.
  - *Theorem, hardness half*: let the backflow class contain every orbital map $x\mapsto\Phi(x)$ computable by a polynomial-size Boolean circuit of the spin-resolved occupations. ReLU MLPs of polynomial width and depth have this property on Boolean inputs. Then computing the exact amplitude $\langle x\rvert P_{S=0}\lvert\phi_\Phi\rangle$ is #P-hard.
    - *Proof (dual rail)*: take $L=2n_v$ sites at half filling, with sites $(2a,2a+1)$ encoding variable $a$, and let $x_{\mathrm{ref}}$ have the up spins on the even sites.
    - In the site-major basis $\langle\sigma\rvert P_0\lvert\sigma'\rangle=K_k=(-1)^kk!(m-k)!/(m+1)!\neq0$, where $k$ is the number of up spins moved, which equals the Hamming weight of the assignment.
    - Choose orbitals that select the occupied sites, with one entry set to $g(x')=s_{\mathrm{ord}}(x')\,\mathrm{SAT}(x')/K_{k(x')}$ on valid dual-rail strings and 0 otherwise. Here $s_{\mathrm{ord}}$ is the site-major-to-up-then-down order sign and $\mathrm{SAT}(x')\in\{0,1\}$ is the formula evaluated on the assignment that $x'$ encodes. A polynomial-size circuit computes $g$.
    - Then $(P_0\phi)(x_{\mathrm{ref}})=s_{\mathrm{ord}}(x_{\mathrm{ref}})\,\#\mathrm{SAT}$.
    - *Verified in code* (`--mode lemma`, `logs/lemma_output.txt`): 30 random 3-CNF formulas with $n_v=3,4$, the orbitals built and the full-vector Löwdin projector applied. Every value equals #SAT to $10^{-12}$.
    - It is a worst-case statement. It does not say that a particular trained NNBF is hard.
  - *Constructive half*: let the backflow read spin only through $r$ integer features $w_a$ that are linear in the single-site spins, e.g. $w=$ the number of up-singles on sublattice $A$, with $R$ values each. Then exact $S$ costs $R^{2r}$ closed-form projections, $\mathcal O(R^{2r}n^3)$.
    - *Mechanism*: write $\Pi_w=\frac1P\sum_je^{-i\theta_jw}e^{i\theta_j(W'+N_{sA})/2}$, where $W'=\sum_{i\in A}(n_{i\uparrow}-n_{i\downarrow})$, $N_{sA}$ is the number of singles on $A$, $P=N_{sA}+1$ and $\theta_j=2\pi j/P$. The operator $e^{i\theta W'/2}$ is one-body, i.e. a row phase on the orbitals, so each term is one `che_closed` evaluation. The orbit sum splits by feature value.
    - The arm `sfb` takes $r=1$ and reads $s_A=(2w-N_{sA})/\max(N_{sA},1)$.
    - `sfb_single` matches the full-vector $P_S$ to $5\times10^{-14}$ with doublons present (smoke). It costs 15 ms at $L=24$, against 2.8 s for the orbit sum, and 0.34 s at $L=64$, where the orbit sum is out of reach.
    - Corollary: charge-only conditioning (Tocchio, Che) is the $r=0$ case. The cost of exact $S$ is set by the **number of independent spin features** the backflow reads, not by whether it reads spin at all. This upgrades round 2's conjecture to a theorem in both directions.
  - Gate 0 (Novelty & Adoption): **PASS (provisional)**.
    - The WebSearch queries "#P-hard symmetry projection neural network quantum state spin SU(2)" and "exact spin projection backflow depends on spin configuration Fourier one-body Slater" found no complexity result for spin projection of NQS and no spin-feature-bucketed projection.
    - The closest work is SA-NNBF (2604.06841: spatial-only backflow, CP-compressed spin functions) and the SU(2)-equivariant spin networks (2104.14869), which cite superpolynomial irrep multiplicities but prove no hardness.
    - The Fourier filter is the standard particle-number-projection device. What is new is its use inside the spin projector to condition on spin features.
    - *Adoption barrier*: the field treats "reads spin" against "charge-only" as binary (Tocchio, Che, SA-NNBF all go charge-only). The bucketing shows there is a graded middle at polynomial cost.
  - Gate 1 (Conservation & Asymptotics): **PASS**. Exact $S$ is shown for every $\theta$ (smoke $\langle\hat S^2\rangle=-7\times10^{-17}$; finite-difference gradient error $5\times10^{-10}$). The cost is $\mathcal O(N^5)$ for one global feature, or $\mathcal O(R_bN^4)$ with $R_b$ bins.
  - Gate 2 (Baseline Gameability): **PASS (decided by the test)**. The trivial foil is `che` ($r=0$, same network). If one spin feature buys nothing, `sfb` FAILs.
  - Gate 3 (Hard-Core Shield): **PASS, sign half only**. There is no Marshall or bipartite assumption: the sublattice is used only as a feature, and $t'\neq0$.
  - *Verdict*: **SURVIVED** (structural survivor plus a constructive arm).
* **Route 4.2 (Operation A on the "one AGP per $c$" wall): charge-conditioned $K$-AGP sum (KAGP)**, $\psi(x)=e^{J(c)}\sum_{k<K}\det F^{(k)}_c[U(x),D(x)]$ with every $F^{(k)}_c=\mathrm{sym}\big(F^{(k)}_0+\sum_hh_h(c)V^{(k)}_h\big)$ symmetric and one shared charge network $h(c)$.
  - *Mechanism*: each term is an exact singlet, so their sum is too, with no projector. By the dimension count, $K=2$ is locally complete per $c$ up to $N_s=10$, exactly where one AGP fails. The cost is $K$ determinants: $2.0\times$ `tocchio` and $0.32\times$ closed-form Che at $L=576$.
  - Gate 0 (Novelty & Adoption): **PASS (provisional, narrow)**.
    - Linear combinations of AGPs exist: LC-AGP (Uemura & Nakajima arXiv:1906.02478; Uemura, Kasamatsu & Sugino; Scuseria group's non-orthogonal AGP sums, arXiv:2101.08911), and AGP-CI (2604.14115). Neural geminals also exist (2605.17083).
    - None conditions the AGPs on the charge configuration, so in all of them the sum must be complete globally rather than per $c$.
    - *Adoption barrier*: LC-AGP needs many terms, and optimising them jointly is ill-conditioned (Uemura and the AGP-CI paper both report this).
    - *KAGP's fix*: the charge network supplies the configuration dependence, so $K$ only has to cover the per-$c$ spin deficit, which the dimension count bounds ($K=2$ up to $N_s=10$).
  - Gate 1 (Conservation & Asymptotics): **PASS with stated limit**.
    - Exact $S$: smoke $\langle\hat S^2\rangle=3\times10^{-17}$; finite-difference gradient error $5.5\times10^{-10}$; `kagp_single` agrees with the batched form to $3\times10^{-17}$.
    - *Limit*: per-$c$ completeness needs $K$ to grow with $N_s$ (rank 129 < 132 at $K=3$, $N_s=12$). At scale KAGP is an accuracy-per-cost bet, not a completeness fix.
  - Gate 2 (Baseline Gameability): **PASS (decided by the test)**. The foils are `tocchio` ($K=1$, same parameterisation, parameter-matched) and `che`.
  - Gate 3 (Hard-Core Shield): **PASS, sign half only**.
  - *Verdict*: **SURVIVED (conditional on the tabular premise from job 411774)**.
* **Route 4.3 (Operation B.1): exact projection of a spin-Jastrow times a charge-conditioned determinant.**
  - Gate 1: **FAIL**. With $e^{\sum J_{ij}\sigma_i\sigma_j}$ the orbit sum is a fixed-magnetisation Ising partition function weighted by $K_k$. Interpolating in a one-body field recovers the fixed-magnetisation Ising sums, which are #P-hard for general $J$.
  - It is the two-body special case of Route 4.1's hardness half.
  - *Verdict*: **KILLED**.
* **Route 4.4 (Operation B.3 from hidden-fermion determinants): hidden spinful ancillas in the spinon AGP, contracted into singlets.**
  - Gate 2: **FAIL**. Grassmann integration of Gaussian ancillas gives the Schur complement $A_{vv}-A_{vh}(A_{hh}+B)^{-1}A_{hv}$, where $v$ and $h$ label the visible and hidden modes, $A$ is the pairing matrix and $B$ is the ancilla singlet pairing. That is again one AGP per $c$.
  - Gaussian ancillas cannot leave the one-AGP class. Leaving it needs a sum (4.2), a projection (Che), or non-Gaussian structure (MPS, area law).
  - *Verdict*: **KILLED**.
### Top Surviving Route & Sharpest Gap (Round 4)
* **Survivor 1 (Route 4.1, spin-feature dichotomy)**:
  - Exact-$S$ projection of a spin-resolved backflow is #P-hard in the worst case (dual-rail reduction, verified in code).
  - It is polynomial, $\mathcal O(R^{2r}n^3)$, when the backflow reads $r$ bounded integer spin features (bucketing plus Che's closed form, verified exact).
  - The constructive arm `sfb` ($r=1$) is the first exact-$S$ backflow that reads spin at polynomial cost.
* **Survivor 2 (Route 4.2, KAGP)**: $K$ charge-conditioned symmetric AGPs, exact $S$ with no projector, at $K$ determinants.
  - $K=2$ closes the per-$c$ deficit of one AGP up to $N_s=10$.
  - It costs $2\times$ the Tocchio baseline and $1/3$ of closed-form Che.
* **Sharpest Gap**:
  - (4.1) Does one global spin feature measurably beat charge-only conditioning? The theorem covers worst-case universal networks, not the trained ones.
  - (4.2) Is the per-$c$ incompleteness of one AGP energetically material on $\psi_0$? The tabular job answers this. If $\epsilon_{\mathrm{tab}}(K{=}1)<10^{-4}$, the network parameterisation, not the per-$c$ class, limits `tocchio`, and Route 4.2 has no target on ED clusters.
### 24-Hour Decisive Test (Round 4)
* **Cost half (done, login node)**: the table in Step 1a, from `logs/cost_output.txt` and `results/cost_r4.json`, at equal $H$ with gathered fields.
  - $g_{\mathrm{cost}}$ for 4.2: $t_{\mathrm{kagp2}}<t_{\mathrm{che\_closed}}$ at the largest size. Measured 6.7 ms against 20.8 ms: **holds**.
  - Route 4.1 makes no cheaper-than-Che claim. Its cost claim is polynomial against exponential: 15 ms against 2.8 s at $L=24$, which also **holds**.
* **Instance (accuracy half)**: the $4\times3$ open cluster, $N_\uparrow=N_\downarrow=5$, $U=8$, $t'=-0.25$, Hilbert dimension 627,264, 58,278 charge configurations. Sign half only.
* **Arms**: all are parameter-matched to NNBF width 16 (2556–2653 parameters), trained with $P_G$ ($\{1,M_x,M_y,M_xM_y\}$), with exact sums, L-BFGS with an exact gradient, 2000 iterations for every arm, seeds 0, 1 and 2, and medians.

| Arm | Description | Projection | Parameters | s per energy + gradient (8 CPUs) |
| :--- | :--- | :--- | :--- | :--- |
| `tocchio` | 1 symmetric charge-conditioned det | $P_G$ | 2587 | 4.9 |
| `kagp2` | Route 4.2 | $P_G$ | 2556 | 9.8 |
| `che` | Che family, charge NNBF | $P_GP_S$ | 2637 | 11.7 |
| `sfb` | Route 4.1, `che` plus one spin feature | $P_GP_S$ | 2653 | 11.8 |
| `nnbf_SG` | projected control, spin-resolved NNBF | $P_GP_S$ | 2637 | 11.7 |

* **Metric**: $\epsilon=(E-E_0)/\lvert E_0\rvert$ per arm, with $\mathrm m(\cdot)$ the median over 3 seeds and the min/max spread reported. Exactness requires $\max\lvert\langle\hat S^2\rangle\rvert<10^{-10}$ for every arm.
* **Pre-committed rule** (implemented in `--mode report`; checked on synthetic results in `logs/check_report_r4_output.txt`, where four scenarios give the intended verdicts):
  - Route 4.2 premise, from job 411774: $\epsilon_{\mathrm{tab}}(K{=}1)\ge10^{-4}$ and $\epsilon_{\mathrm{tab}}(K{=}2)\le0.5\,\epsilon_{\mathrm{tab}}(K{=}1)$. If it fails, the verdict is NO-TARGET on this cluster.
  - The verdicts are evaluated in the order listed.

| Route | PASS | FAIL | AMBIGUOUS |
| :--- | :--- | :--- | :--- |
| 4.2 `kagp2` | $g_{\mathrm{cost}}$, and $\mathrm m(\epsilon_{\mathrm{kagp2}})\le0.7\,\mathrm m(\epsilon_{\mathrm{tocchio}})$ and $\le\mathrm m(\epsilon_{\mathrm{che}})$ | $g_{\mathrm{cost}}$ violated, or $\mathrm m(\epsilon_{\mathrm{kagp2}})\ge0.9\,\mathrm m(\epsilon_{\mathrm{tocchio}})$, or $>\mathrm m(\epsilon_{\mathrm{che}})$ | otherwise, or fewer than 3 seeds |
| 4.1 `sfb` | $\mathrm m(\epsilon_{\mathrm{sfb}})\le0.7\,\mathrm m(\epsilon_{\mathrm{che}})$ | $\mathrm m(\epsilon_{\mathrm{sfb}})\ge0.95\,\mathrm m(\epsilon_{\mathrm{che}})$ | otherwise |

* **Reported, not gating**: `che` against `nnbf_SG`, which is open question (b) and is also measured by pilot 411494; and $\mathrm m(\epsilon_{\mathrm{nnbf\_SG}})$ against $10^{-4}$.
* **Pre-committed action on AMBIGUOUS**: rerun the array on the $4\times3$ cylinder (`--bc-x pbc`). Promote a route only if it is not FAIL on both clusters.
* **Pilot sizing**:
  - The tabular job (411774) sets the 4.2 premise.
  - Seed 0 of every arm (`--array=0,3,6,9,12`) runs first as the pilot. The remaining ten tasks run only if no arm fails exactness and $\mathrm m(\epsilon_{\mathrm{tocchio}})\ge10^{-4}$.
  - The measured cost is 4.9–11.8 s per evaluation (`logs/timing_output.txt`), i.e. at most about 7.5 h for 2000 iterations within a 12 h allocation.
* **Execution (not submitted)**: `tests/P4.1_cheap_su2_r4/submit_test.sbatch`.
  - Pilot: `sbatch --array=0,3,6,9,12 /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4/submit_test.sbatch`.
  - Then the remaining tasks: the same command with `--array=1,2,4,5,7,8,10,11,13,14`.
  - Afterwards, from the test directory: `python3 r4_test.py --mode report --Ne 10`.
* **Smoke** (`cd tests/P4.1_cheap_su2_r4 && /share/apps/software/anaconda3/bin/python3 r4_test.py --mode smoke`; 0.5 s):

```
[15:20:30] 2x3 torus Ne=6 dim=400
[15:20:30]   tocchio  P= 307 <S^2>=-3.86e-17 grad rel.err=1.6e-09 [PASS]
[15:20:30]   kagp2    P= 268 <S^2>=+3.34e-17 grad rel.err=5.5e-10 [PASS]
[15:20:30]   che      P= 294 <S^2>=-6.47e-17 grad rel.err=2.9e-10 [PASS]
[15:20:30]   sfb      P= 298 <S^2>=-6.76e-17 grad rel.err=5.1e-10 [PASS]
[15:20:30]   kagp_single vs batched: 2.8e-17
[15:20:30]   2x3 pbc/pbc Ne=6 che: |single - full-vector P_S| 1.1e-14, |closed - GL| 1.1e-14, |closed - e_k| 8.3e-17 (max|amp| 4.05e-01) [PASS]
[15:20:30]   2x3 pbc/pbc Ne=6 sfb: |single - full-vector P_S| 7.3e-15 (max|amp| 1.98e-01) [PASS]
[15:20:30]   4x2 obc/obc Ne=6 che: |single - full-vector P_S| 1.7e-16, |closed - GL| 1.9e-16, |closed - e_k| 4.2e-17 (max|amp| 2.40e-01) [PASS]
[15:20:30]   4x2 obc/obc Ne=6 sfb: |single - full-vector P_S| 5.2e-14 (max|amp| 6.65e-01) [PASS]
[15:20:30] SMOKE PASS  (0.5s)
```

### Local Autopsy Log (Round 4 additions)
* **Critic objections and how each was handled**:
  - (1) Accepted: SQ-RVB was killed (Route 4.0).
  - (2) Accepted: the barrier is now stated as per-$c$ completeness, and it is measured by the dimension count and job 411774.
  - (3) Accepted: `tocchio` is the cost baseline.
  - (4) Accepted: the closed form is implemented and verified, and the fair slopes are 2.0–2.3.
  - (5) Accepted: SCF was killed for the refined formulation (Route 4.0b).
  - (6) Accepted: every cost arm uses the same $H$ and gathered fields, and the accuracy arms use the same families.
  - (7) Accepted: completeness is now measured, by the dimension count and job 411774.
  - (8) Moot, since SQ-RVB is retired. KAGP has no dead parameters of that kind: the doublon diagonal entries of $F_c$ do enter the determinant.
  - (9) Accepted: the narrow width was dropped, and the test runs at width 16 only.
  - (10) Accepted: the cost claim is decided on the login node and the array decides accuracy only.
* **New kills**:
  - 4.0: SQ-RVB, Gate 0 and cost-dominated.
  - 4.0b: SCF, area law in 2D.
  - 4.3: projected spin-Jastrow, #P-hard Ising sum.
  - 4.4: Gaussian ancillas, which collapse to one AGP per $c$.
* **New anti-pattern**: claiming "exact $S$ below projection cost" as novel. Tocchio-type singlet pairing (1 det) and closed-form Che ($\mathcal O(n^3)$) already saturate it, and any new exact-$S$ route must be priced against `tocchio` and `che_closed` at equal parameterisation.
