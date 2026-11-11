
### Round 3 — Mutation & Replenishment (2026-11-11)
**Binding inputs and how each is handled.**
* *R2.0 revival is illegitimate (accepted).* The round-1 Gate 0 kill of the frozen-backflow control variate stands. R2.0 is not a route this round. The frozen ratio appears only inside diagnostics.
* *Hybrid cost model (accepted).* Every number in the round-3 verdict is measured on one instance at one $N$. The test reports two cost models, and a route must pass under both:
    * measured single-thread wall-clock per sample of the code that runs;
    * the operation count of the same kernels at the instance's own $N$, $M$ and hidden widths.

  Variances come from genuine Monte Carlo on that instance's trained states. No $\kappa$ from another $N$ enters anything. The critic's "falling ratio is built in" point is answered by the metric: the denominator is the best control on the same state and seed (exact enumeration, screening, semistochastic), not exact enumeration alone.
* *R2.4 and R2.5 (accepted).* Both are killed on their own pre-checks: $M=1.03$–$1.08$ and $0.98$–$1.02$. R2.5 is Neyman allocation.
* *H4 at $N=4$ with an s-only basis (accepted).* The test uses H$_6$ ($N=6$) and H$_8$ ($N=8$) rings in a 6-31G** basis, with p functions built in numpy (McMurchie–Davidson). The active spaces are the $N$ valence MOs, then the p-type MOs, then the rest. They contain 2–8 MOs whose Mulliken p weight is at least 0.995. Integral checks:
    * the s part reproduces the round-2 integrals to $7\times10^{-16}$;
    * the RHF energy is invariant under a random rotation and translation to $3\times10^{-12}$;
    * the ERI tensor is positive semidefinite;
    * H$_2$/6-31G** at $1.4$ bohr gives $E_{\mathrm{RHF}}=-1.131284$.
* *Round-2 kills (accepted):* positivisation, the EN tail proposal, LRX summed over all terms, active-space backflow, sketching, Cholesky contraction, and multilevel over basis levels. Round 3 starts from the boundary condition they leave: on near-converged states, every within-row sampled residual is limited by the surrogate's magnitude error.
* *Correction to round 2.* R2.6 cited arXiv:2302.11588 as "active-space NNBF". That paper is Robledo Moreno, Cohn, Sels & Motta's orbital-rotation NQS, which includes the first active-space NQS calculation. It is not a backflow paper. R2.6's Gate 3 kill (no external pair correlation) stands on its own.

**Pre-checks used for the kills below.** Each pre-check names its instance. The kills marked "H4" use the round-2 instance (H$_4$ ring, $R=2.8$, s basis, $M=24$, NNBF width 32, 600 L-BFGS iterations, $\Delta E=0.05$ mHa, $\mathrm{Var}_\pi(E_L)=1.7\times10^{-4}$). They are kill evidence only. None of them enters the round-3 verdict. Scripts: `tests/P2.2_stochastic_local_energy_r3/scratch/pre{1..6}.py`.

* **Generation Batch 3**:
  - **Route 3.0 (Operation A on the frozen surrogate, the component that failed in R2.0)**: tangent-Slater surrogate. Dress the frozen ratio with the backflow's own first-order occupation response, $\tilde r\to\tilde r\,e^{2(g_a+g_b-g_i-g_j)}$ with $g=\partial_{\mathbf{x}}\ln\lvert\det\Phi(\mathbf{x})_{[\mathrm{occ}\,\mathbf{n}]}\rvert$. Here $\mathbf{x}=2\mathbf{n}-1$ is the backflow input, $i,j$ are the vacated and $a,b$ the newly occupied spin-orbitals of the excitation, and one backward pass gives all of $g$. The dressed surrogate is the frozen ratio of a fugacity-weighted determinant, so its full sum contracts like the AFQMC local energy.
    - Gate 1 (Conservation & Asymptotics): **FAIL (measured, H4)**. $g\propto\mathrm{tr}(A^{-1}\partial A)$, where $A=\Phi(\mathbf{n})_{[\mathrm{occ}\,\mathbf{n}]}$, diverges near the nodes of $A$. The exponential form overflows. The linearized form has residual fraction $\rho=1.26$ (frozen surrogate: 0.59) and $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)=4\times10^9$.
    - *Verdict*: KILLED.
  - **Route 3.1′ (Operation B.3, cross-field translation of the Jackson–Feenberg kinetic estimator)**: Cholesky sum-of-squares energy estimator. Write $\hat H=\hat h'+\tfrac12\sum_\gamma\hat L_\gamma^2$, with $\hat L_\gamma=\sum_{pq}L^\gamma_{pq}\hat E_{pq}$ and $(pq\vert rs)=\sum_\gamma L^\gamma_{pq}L^\gamma_{rs}$. By Hermiticity, $E=\mathbb{E}_\pi\big[\hat h'_{\mathrm{loc}}+\tfrac12\sum_\gamma\ell_\gamma(\mathbf{n})^2\big]$, where $\ell_\gamma(\mathbf{n})=(\hat L_\gamma\psi)(\mathbf{n})/\psi(\mathbf{n})$ needs only the $\mathcal{O}(NM)$ single excitations, not the $\mathcal{O}(N^2M^2)$ doubles. The identity was verified to $10^{-14}$.
    - Gate 1 (Conservation & Asymptotics): **FAIL (measured, H4)**. The variance contains $\sum_{\mathbf{n}}(\hat L\psi)(\mathbf{n})^4/\psi(\mathbf{n})^2$, which diverges where $\psi\to0$. $\mathrm{Var}/\mathrm{Var}(E_L)$ was $6.5\times10^{6}$ to $8\times10^{14}$, and $2\times10^{34}$ for the exact $\psi_0$. Importance sampling from $q\propto\psi^2+\tfrac12\sum_\gamma(\hat L_\gamma\psi)^2$ makes the variance finite, but $\mathcal{O}(E_2^2)$ rather than $\to0$: the zero-variance principle is lost.
    - *Verdict*: KILLED. This is a new anti-pattern for the master list.
  - **Route 3.2′ (Operation B.3, multifidelity Monte Carlo across samples; Peherstorfer–Willcox–Gunzburger)**: debias a cheap per-sample local energy $E^S_L$ with exact rows on a random subset of samples. The estimator is $\bar E^S_B+\overline{(E_L-E^S_L)}_{B'}$, and its variance is $[\mathrm{Var}(E_L)-\mathrm{Var}(\delta)]/B+\mathrm{Var}(\delta)/B'$ with $\delta=E_L-E^S_L$. No within-row sampling takes place, so the magnitude dispersion $\mathrm{Var}(x/\lvert y\rvert)$ never appears. Cheap models tried: frozen, LRX, screened sums, the orbital-window head, and regressions on all of them.
    - Gate 0 (Novelty & Adoption): PASS. No NQS use was found.
    - Gate 2 (Baseline Gameability): **FAIL (measured, H4)**.
        * $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ is 466 for the frozen surrogate, 19.5 for LRX, and 2.3–48 for frozen-surrogate screening with completion.
        * For the orbital-window head, the multifidelity cost ratio is at best $0.96$ ($M=32$, window 10).
        * The regression $R^2$ of $E_L$ on all cheap features is $0.13$.

      The surrogate's error is strongly sample-dependent, so it does not average out across samples. One trend is recorded for later: the window-head ratio fell from 1.59 to 0.96 as $M$ went from 24 to 32.
    - *Verdict*: KILLED.
  - **Route 3.3′ (Operation A on R2.0's surrogate)**: row-calibrated surrogate. In each row, fit per-orbital factors $y\,(1+\sum_pc_pf_p)$ or $y\,e^{\sum_pc_pf_p}$ on the exactly evaluated head, then use them as the control variate on the tail. This is unbiased, because the fit uses only the head.
    - Gate 2 (Baseline Gameability): **FAIL (measured, H4)**. The tail residual fraction rises from 1.07 (frozen) to 1.26 and 1.46. Head-fitted factors extrapolate worse than none.
    - *Verdict*: KILLED.
  - **Route 3.4′ (the brief's amortization idea)**: cache connected amplitudes across the batch or along the Markov chain.
    - Gate 2 (Baseline Gameability): FAIL. For two configurations at excitation distance 2, $\lvert\mathcal{C}(\mathbf{n})\cap\mathcal{C}(\mathbf{n}')\rvert=\mathcal{O}(NM)$, against $\lvert\mathcal{C}\rvert=\mathcal{O}(N^2M^2)$; at distance 4 the overlap is $\mathcal{O}(1)$. On any ED-feasible instance the union of connected sets saturates at the Hilbert-space dimension, so a measured speed-up there would be a finite-size artefact.
    - Gate 0 (Novelty & Adoption): FAIL. Deduplication of unique samples and sample-aware evaluation are standard (NNQS-Transformer, arXiv:2306.16705).
    - *Verdict*: KILLED.
  - **Route 3.5′**: Horvitz–Thompson (Poisson or priority) sampling of terms.
    - Gate 2 (Baseline Gameability): FAIL. The variance is $\sum_jx_j^2(1/\pi_j-1)$, where $\pi_j$ is the inclusion probability of term $j$. It is bounded by the same magnitude dispersion as multinomial sampling. The cross-term it removes is negligible for many small tail terms.
    - *Verdict*: KILLED.
  - **Route 3.6′ (the brief's question)**: an exact local energy at reduced cost via low-rank (THC or Cholesky) structure.
    - Gate 1 (Conservation & Asymptotics): FAIL for NNBF. Exact contraction exists only when the orbitals do not depend on $\mathbf{n}$, which is the AFQMC local energy (Motta & Zhang, arXiv:1711.02242). For unrestricted backflow, P2.1's F1 bound gives $\Omega(N^2)$ per connected amplitude.
    - *Verdict*: KILLED. The answer to the brief's question is yes: this is the standard AFQMC local energy.
  - **Route 3.7′ (Operation A.4 on R2.0)**: window-exact partially-frozen surrogate as a stand-alone control variate.
    - Gate 2 (Baseline Gameability): FAIL as a stand-alone route on near-converged states (measured, H4). $\rho=0.25$, 0.21 and 0.05 for windows of 4, 6 and 8 orbitals, but the oracle single-proposal floor of the residual is $104$–$369\times\mathrm{Var}(E_L)$.
    - *Verdict*: KILLED on H4. It survives below as R3.2, re-gated on the $N=6$ instance, where the states are realistically converged (Section "Pre-check on the decisive instance").
  - **Route 3.1 (Operation A on R2.6, killed at Gate 3 for "no external pair correlation"; changes the frame from estimating $E_L$ to an ansatz whose exact $E_L$ is cheap)**: **window-input backflow × all-orbital Jastrow (WIBJ)**.
    $$\psi(\mathbf{n})=J(\mathbf{n})\,\det\Phi(\mathbf{n}_W)_{[\mathrm{occ}\,\mathbf{n}]},\qquad \ln J=\mathbf{u}\cdot\mathbf{n}+\sum_{p<q}v_{pq}n_pn_q+\mathbf{a}\cdot\tanh(W_J\mathbf{x}+\mathbf{b}_J).$$
    Here $\mathbf{n}_W$ is the occupation of the window $W$ (the $N$ valence spatial MOs, both spins), and $\Phi\in\mathbb{R}^{M\times N}$ covers **all** orbitals. The Jastrow parameters are the one-body fields $\mathbf{u}$, the pair couplings $v_{pq}$, and a one-hidden-layer network of width 32 ($W_J$, $\mathbf{b}_J$, $\mathbf{a}$). Mechanism:
    * Every connected $\mathbf{n}'$ with the same window change shares $\Phi(\mathbf{n}'_W)$.
    * Per window key the code builds the $N$ occupied rows once, $\mathcal{O}(N^2h+N^3)$ with $h$ the backflow hidden width, plus one $\mathcal{O}(Nh)$ row per distinct added external orbital.
    * Each connected term then costs a $2\times2$ minor of $G_k=\Phi_kA_k^{-1}$ plus an $\mathcal{O}(h_J)$ Jastrow ratio, where $\Phi_k$ is the backflow matrix of key $k$, $A_k$ its occupied block and $h_J=32$ the Jastrow hidden width.
    * The number of keys is set by the window, not by $M$ (139 keys against 416 connected terms at $N=6$, $K=12$). The local energy is **exact**, so the zero-variance property is kept.
    - Gate 0 (Novelty & Adoption): PASS.
        * Searches (three queries: backflow input restricted to a subset of orbitals; Green's-function ratios with active-orbital backflow and a Jastrow; low-rank multi-determinant NNBF) found NNBF for chemistry (arXiv:2403.03286, 2502.18843), truncated local energies (2502.18843, 2603.14775), semistochastic energies (2507.19276), orbital-rotation active-space NQS (2302.11588), and low-rank backflow updates (lrux arXiv:2602.05255; Huang, Ou & Ozolins arXiv:2609.26544, Hubbard only). None restricts the backflow input to make the exact local energy key-groupable.
        * The adoption barrier of active-space NQS is missing external correlation. The mechanism that removes it: valence-dressed orbitals over all $M$ rows, plus an all-orbital neural Jastrow whose ratio is $\mathcal{O}(h_J)$ per term. This keeps the exact local energy cheap.
        * It differs from P2.1's low-rank families (SCALE, lrux), which cut the cost *per amplitude*. WIBJ cuts the *number* of backflow evaluations.
    - Gate 1 (Conservation & Asymptotics): PASS.
        * It conserves $N$ and $S_z$, and is exact.
        * The smoke test shows that the keyed ratio equals a direct determinant to $3\times10^{-14}$. On $N=6$, $\max\lvert E^{\mathrm{KEYX}}_L-E^{\mathrm{EXACT}}_L\rvert\le1.3\times10^{-15}$ over all samples.
        * Operation count: $\mathcal{O}(N_{\mathrm{key}}(N^2h+N^3)+\lvert\mathcal{C}\rvert(N^2+h_J))$ against $\mathcal{O}(\lvert\mathcal{C}\rvert(N^2h+N^3))$, with $N_{\mathrm{key}}$ the number of distinct window keys. $N_{\mathrm{key}}$ does not grow with the number of external orbitals, while $\lvert\mathcal{C}\rvert$ does.
    - Gate 2 (Baseline Gameability): {{G2_R31}}
    - Gate 3 (Hard-Core Shield): PASS (conditional). The multireference character sits in the valence window, which keeps full backflow. External orbitals are correlated only through valence-dressed orbitals and the Jastrow, so the external sign structure is mean-field-like by construction. The test charges this through $\Delta E$ and $\mathrm{Var}(E_L)$ against the unrestricted NNBF-J of the same seed. Coverage is the sign half only.
    - *Verdict*: {{V_R31}}
  - **Route 3.2 (Operation A on R2.0's failed component, the surrogate; the λ-continuation of R3.1)**: **window-keyed partially-frozen control variate (PFCV) on the unrestricted NNBF-J**.
    * The surrogate evaluates the *same* unrestricted network with the window input of $\mathbf{n}'$ and the external input frozen at $\mathbf{n}$, so it costs the same per key as R3.1.
    * It is exact for every window-internal excitation; its error is only the response of the backflow to external occupations.
    * The surrogate is summed exactly over all terms. Terms with $\lvert y\rvert>\varepsilon$ are evaluated exactly. The rest get $K$ residual draws from $p\propto0.9\lvert y\rvert+0.1\lvert H\rvert$.
    * SHRJ, a shrinkage $\mu\lVert W_1[:,\mathrm{ext}]\rVert^2$ on the external input weights (the continuation between R3.2 at $\mu=0$ and R3.1 at $\mu=\infty$), is reported as a diagnostic only.
    - Gate 0 (Novelty & Adoption): PASS (narrow). The construction "exact deterministic surrogate plus sampled residual" is textbook, as the critic said of R2.0. A search for multifidelity control variates in VMC or NQS found nothing in NQS (the closest is multilevel control functionals, arXiv:2305.12996, in Bayesian inference).
        * What is new is the surrogate. It is defined by which network *inputs* are frozen, not by freezing the orbitals; its cost is per window key; and its error vanishes identically on the window-internal terms that carry the multireference weight.
        * The round-2 frozen surrogate missed exactly those terms ($\rho_{\mathrm{FB}}=0.39$–$0.79$).
        * The Gate 0 risk is recorded, not hidden.
    - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased; a 20 000-draw test on one row gives $z=0.14$. It conserves $N$ and $S_z$, and the SR force stays conditionally unbiased.
    - Gate 2 (Baseline Gameability): {{G2_R32}}
    - Gate 3 (Hard-Core Shield): PASS (conditional). Nothing is restricted; the residual is sampled. Coverage is the sign half only.
    - *Verdict*: {{V_R32}}

**Pre-check on the decisive instance** (H$_6$ ring, $R=2.8$ bohr, 6-31G**, $N=6$, one seed, 400 L-BFGS iterations, 400 samples; `precheck_N6_K8-12.log`).
* HF weight 0.84 and frustrated fraction 0.144–0.148 at every $K$.
* Each cell gives $C/C_{\mathrm{EXACT}}$ on the arm's own state, under wall-clock and operation-count cost.
* $M$ is the verdict ratio against the best control on the NNBFJ state.

{{TABLE}}

{{READING}}

### Top Surviving Route & Sharpest Gap (Round 3)
{{TOP}}

### 24-Hour Decisive Test (Round 3)
* **Instance**:
    * H$_6$ ring with $K=8,10,12,14$ ($M=16$–$28$, dimension up to 132 496, 2–8 p-type MOs).
    * H$_8$ ring with $K=9,10,11$ ($M=18$–$22$, dimension up to 108 900, 1–3 p-type MOs).
    * Both at $R=2.8$ bohr in 6-31G**.
    * Window $K_W=N$ valence MOs.
    * Sign half of the Hard Core only. The same-sector competitor is not examined; rule 1 (lattice projection) does not apply.
* **Arms**: NNBFJ (unrestricted backflow, hidden width 48, plus the Jastrow), WIBJ (R3.1, same widths), and SHRJ (diagnostic, $\mu=10^{-2}$). Each is trained by L-BFGS on the exact energy for 1500 iterations, with **seeds 0, 1, 2**.
* **Estimators**: run per sample with the real kernels on $B=2000$ iid samples from $\lvert\psi\rvert^2$. Sampling and training use the full vector; neither counts toward any estimator's cost.
    * Controls on NNBFJ: EXACT, SCREEN ($\varepsilon$ grid; admissible if $\lvert\mathrm{bias}\rvert+2\,\mathrm{SE}\le0.16$ mHa), and SS ($\varepsilon$, $K$ grid).
    * Routes: KEYX on WIBJ (R3.1) and PFCV on NNBFJ (R3.2).
    * Tunables are selected on the first half of the samples and scored on the second half.
    * A configuration is admissible only if its variance increment is estimable: the top 1% of samples carry at most 50% of it. This applies to routes and controls alike.
* **Scalar metric**:
    * Cost: $C_X=\mathrm{var}_X\,t_X$, with $\mathrm{var}_X=\mathrm{Var}_\pi(E_L)+(s^2_X-s^2_{\mathrm{EXACT}})$. Here $\mathrm{Var}_\pi(E_L)$ is the exact $\pi$-variance and $s^2$ the sample variance, paired on the same samples.
    * Ratios: $M_{\mathrm{R3.1}}=C_{\mathrm{KEYX}}(\mathrm{WIBJ})/C_{\mathrm{ctrl}}(\mathrm{NNBFJ})$ and $M_{\mathrm{R3.2}}=C_{\mathrm{PFCV}}(\mathrm{NNBFJ})/C_{\mathrm{ctrl}}(\mathrm{NNBFJ})$, paired by seed; medians over seeds, with the spread reported.
    * Both are computed under wall-clock and under operation-count cost.
* **Thresholds** (per route, per $N$, under each cost model; implemented in `--aggregate`):
    * $\Delta E$ check: $\Delta E_{\mathrm{route}}(K_{\max})\le\max(1.25\,\Delta E_{\mathrm{NNBFJ}},\,\Delta E_{\mathrm{NNBFJ}}+0.5\ \mathrm{mHa})$, on medians. It holds by construction for R3.2.
    * PASS if the $\Delta E$ check holds, $M(K_{\max})\le0.5$, and $M(K_{\max})<M(K_{\min})$.
    * FAIL if the $\Delta E$ check fails or $M(K_{\max})\ge1$.
    * AMBIGUOUS otherwise.
    * A route passes at a given $N$ only if it passes under both cost models, and fails if it fails under either.
    * Global: PASS only if it passes at $N=6$ and $N=8$; FAIL if it fails at either.
    * Pre-committed action for AMBIGUOUS: rerun $N=6$ with `--n-active 15`; PASS if and only if $M(15)\le\min(M(14),0.5)$, else KILL.
* **Execution**:
    * Code: `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r3/micro_test.py` and `submit_test.sbatch`.
    * Job: array 0–20 (7 instances × 3 arms), partition `kim`, 2 CPUs, 24 GB, 24 h.
    * Expected cost: {{COST}}.
    * Command (not run): `sbatch /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r3/submit_test.sbatch`.
    * Verdict: `python3 micro_test.py --aggregate <test dir>`.
* **Smoke test**: command `/share/apps/software/anaconda3/bin/python3 micro_test.py --smoke` ({{SMOKET}}). It checks the integrals, the exact incremental and keyed ratios, the gradients and the unbiasedness of the estimators, then runs H$_4$ with $K=5$ and $K=7$ and three seeds. Literal output, head and verdict lines (the full output is in `smoke_output.txt`):

{{SMOKE}}

### Local Autopsy Log (Round 3 additions)
* **R2.0, R2.4, R2.5**: killed as the critic stated. R2.0 is not revived again. Its frozen ratio survives only as a diagnostic.
* **R3.0 (tangent-Slater surrogate)**: the first-order log-response of a determinant diverges at its nodes, so the surrogate is worse than the frozen one.
* **R3.1′ (Cholesky sum-of-squares estimator)**: it needs only $\mathcal{O}(NM)$ amplitudes, but its variance is infinite because the fourth moment diverges at $\psi\to0$. The finite-variance importance-sampled version loses the zero-variance principle.
* **R3.2′ (cross-sample multifidelity)**: every cheap model's error depends on the sample, with $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ between 2 and 466.
* **R3.3′ (row-calibrated surrogate)**: factors fitted on the head extrapolate badly to the tail.
* **R3.4′ (amplitude caching / batch union)**: neighbour overlap is $\mathcal{O}(NM)$ out of $\mathcal{O}(N^2M^2)$, and the technique is standard.
* **R3.5′ (Horvitz–Thompson)**: same magnitude-dispersion floor.
* **R3.6′ (exact low-rank $E_L$)**: this is the AFQMC local energy, and the P2.1 F1 bound applies to backflow.
* **R3.7′ (stand-alone partially-frozen control variate on near-exact H4 states)**: residual floor of $10^2\times\mathrm{Var}(E_L)$. Re-gated as R3.2 on the realistic $N=6$ states.
* **Boundary condition for the next generation (measured)**: the pre-checks separate two regimes.
    * On near-exact states ($\Delta E\lesssim0.05$ mHa, H4), every sampled residual and every cross-sample correction loses to exact enumeration.
    * On the $N=6$ states with p functions ($\Delta E=0.3$–$1$ mHa, $\mathrm{Var}(E_L)\sim10^{-3}$), the PFCV residual adds about 1–2% to $\mathrm{Var}(E_L)$, and the cost is set by the deterministic part.
    * Remaining progress must come from making the deterministic part cheaper (keys, not terms), not from better sampling.
