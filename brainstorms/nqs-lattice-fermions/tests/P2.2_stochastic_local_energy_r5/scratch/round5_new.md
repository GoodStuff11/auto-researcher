### Round 5 — Mutation & Replenishment (2026-11-12)
**Binding inputs and how each is handled.**
* *Pilot 411919 kills R4.1 (CGK).* On H$_4$ at $R=4.4$ bohr (HF weight 0.46–0.49) and $K=20$, the median paired loss is $\delta_{\mathrm{CGK}}=0.691$ mHa, above $\tau=0.2$. Coarse external shells lose the dynamic pair correlation, which is the externally contracted MRCI failure. Accepted; R4.1 is not revived.
* *Pilot side result.* At $K=20$, WIBJ (killed in round 3 as "prefactor only") reached $\Delta E=0.33$–$0.43$ mHa against NNBFJ's $0.27$–$0.33$:
    * paired median $\delta=+0.097$ mHa;
    * best-of-seeds $\delta=+0.060$ mHa;
    * variance ratio 1.22.
* *The same pilot shows a basis-size trend for WIBJ.* Its median absolute $\Delta E$ over seeds 0–2 is 0.022, 0.142 and 0.405 mHa at $K=12$, 16 and 20. NNBFJ is not monotone over the same $K$: its medians are 0.361, 0.945 and 0.329 mHa, and seed 1 is under-trained at $K=12$ and 16. So the pilot $\delta$ (−0.339, −0.827, +0.097 mHa) does not show the trend, but the absolute WIBJ error does.
* *Round-4 critic items carried forward:*
    * accuracy is measured at a large external/window ratio (H$_4$, $K=24$: 20 external orbitals per spin against a 4-orbital window);
    * best-of-seeds scoring and a noise-floor arm (a second NNBFJ seed set) are in the verdict;
    * the SCREEN grid runs to $\varepsilon=3\times10^{-2}$ and the SS grid to $K_d=256$;
    * the cost model is honest: calibrated operation counts, a wall model with per-call overhead, and a wall-clock CI whose upper bound must be at most 0.8.
* *Per-leaf critic, round 5 (all applied in this revision):*
    1. The Saturation Diagnostic misused Godambe–Joshi. It is restated below as a query lower bound plus the inclusion-probability variance bound, and F$_A$ is split into F$_{A1}$ (cheaper per term) and F$_{A2}$ (sub-$\lvert\mathcal{C}\rvert$ by contraction).
    2. WIBJ's re-entry is downgraded to CONDITIONAL, with its prior art and adoption barrier stated (Route 5.1).
    3. Part W now measures the accuracy trend: H$_4$ at $K=12$, 16, 20, 24 and H$_6$ at $K=12$, 14. It reports WIBJ's absolute $\Delta E(K)$ beside $\delta$, uses the Part L noise-floor definition, and adds a pre-committed trend FAIL.
    4. Minimal-contrast foils were added to Parts W and S: NNBFJ-h4, NNBFJ-h8 and CAS×J.
    5. Part S cost honesty:
        * SCREEN points at $\varepsilon=1.5$, 2 and $2.5\times10^{-3}$;
        * the Part L wall model with per-call overhead;
        * $n_{\mathrm{wall}}=60$;
        * WIBJ+SCREEN reported against NNBFJ+SCREEN;
        * trained-state samples where feasible (Part W) and the limitation stated for Part S.
    6. Part L is dropped from the array, and R5.2 is recorded as untested in its decisive regime.
    7. A first-row instance (LiH, BeH$_2$) is not feasible with this code. The limitation is stated in the decisive test.
* *Boundary conditions:* every earlier kill, listed in the round brief (single-proposal sampling through caching and batch union).

**Saturation Diagnostic (corrected).**
* *Question.* Is "an exact or unbiased local energy cheaper than enumeration, with no accuracy loss, for a general NNBF" impossible?
* *Model (black-box query).* The estimator knows the integrals, $\psi(\mathbf{n})$ and any surrogate for free. Each connected term $x_j=H_{\mathbf{n}\mathbf{n}_j}\psi(\mathbf{n}_j)/\psi(\mathbf{n})$ is revealed only by an oracle call $\psi(\mathbf{n}_j)$. Let $s_j$ be the uncertainty of $x_j$ given the free information.
* *Correction.* The earlier version cited the Godambe–Joshi bound. That bound limits the anticipated variance of design-unbiased estimators under a superpopulation model, so it says nothing about an exact computation. It is replaced by two statements that do apply.
* *Statement 1 (exact: query lower bound).* If the oracle may change $\psi(\mathbf{n}_j)$ alone without changing any other answer, an algorithm that never queries $\mathbf{n}_j$ cannot tell the two cases apart. So an exact $E_L$ needs $\Omega(\lvert\mathcal{C}\rvert)=\Omega(N^2M^2)$ oracle calls for a generic NNBF.
* *Statement 2 (unbiased: inclusion-probability bound).* Consider an estimator that is unbiased for every $x$ consistent with the free information, and that queries term $j$ with inclusion probability $\pi_j$. Its worst-case variance is at least $\sum_js_j^2(1/\pi_j-1)$, the Horvitz–Thompson form. With $q=\sum_j\pi_j$ expected queries, Cauchy–Schwarz gives
$$\mathrm{Var}\ \ge\ \frac{\big(\sum_js_j\big)^2}{q}-\sum_js_j^2 .$$
    * The single-proposal floor is the case where the surrogate is $\lvert x\rvert$ alone.
    * Every control-variate kill (R2.0, PFCV, R3.2′, R3.3′) had a $\sum_js_j$ that was not small: the measured residual fractions were 0.21–0.82.
* *Statement 3 (amortization, per fresh sample).* Reusing one set of $BT$ samples for $T$ SR steps injects noise of variance $T^2\sigma^2/(BT)=T\sigma^2/B$ per epoch, the same as $T$ independent steps of $B$ samples. So there is no first-order gain while optimization is noise-limited.
* *What escapes the bounds.* Both bounds assume black-box amplitudes. An architecture that exposes algebraic structure is not a black box, and it can beat $\lvert\mathcal{C}\rvert$ in two different ways:
    * **F$_{A1}$, cheaper per term.** Each term still has to be touched, but it costs $\mathcal{O}(1)$–$\mathcal{O}(N^2)$ arithmetic instead of a forward pass. KEYX on WIBJ is of this kind, and the $\lvert\mathcal{C}\rvert$ floor stays.
    * **F$_{A2}$, sub-$\lvert\mathcal{C}\rvert$ by contraction.** If the dependence on the external indices is sum-factorizable, whole classes of terms can be summed without enumerating them. An example is Slater–Jastrow with a Jastrow that factorizes across the window boundary: a Wick plus Cholesky (or THC) contraction sums the (window → external) double class at a cost below its term count. The round-4 R4.0 contraction is an instance; its identity was verified, and the measured contraction/enumeration ratio at $K=40$ was 0.87.
* *Verdict: saturated in the black-box model.* The leaf as first stated (unbiased, $C/C_{\mathrm{ex}}\to0$, general NNBF, black-box amplitudes) is impossible. Every kill so far is an instance of Statement 1 or 2. The leaf is refined to four regimes where a gain is not excluded:
    * **F$_{A1}$:** success means accuracy within $\tau$ of the unrestricted NNBF-J at a lower per-term cost;
    * **F$_{A2}$:** success means an exact sum over a term class at a cost sublinear in its size, under an architecture that loses no accuracy;
    * **F$_B$ (amortization):** make $s_j$ small with free information, namely the same network at an earlier parameter. Statement 3 limits the gain to step-limited optimization;
    * **F$_C$ (biased with certificate):** the bias is provably below $\tau$.

**Gate 0 searches (round 5, WebSearch).**
* *Queries:*
    * a variance-reduced local energy that reuses earlier iterates (SVRG, NQS);
    * stochastic variance-reduced VMC with anchor parameters;
    * fixed-sample reweighting across VMC iterations;
    * active-space neural backflow with a Jastrow, and its local-energy cost;
    * the Godambe–Joshi bound.
* *Found:*
    * importance sampling for VMC gradients (Misery et al., arXiv:2507.05352; $\lvert\psi\rvert^n$ sampling, arXiv:2302.01941);
    * gradient-level control variates in VMC, called SVRG in an NQS tomography paper (arXiv:2307.01840);
    * truncated local energies (arXiv:2502.18843);
    * active-space transformer backflow (arXiv:2509.25720) and spin-adapted active-space NNBF (arXiv:2604.06841).
* *Not found:* neither SVRG applied to the *inner* connected-term sum of $E_L$, nor NNBF input restricted to a window under an all-orbital Jastrow.
* *Known adjacent prior art:*
    * correlated sampling and fixed-sample optimization (Umrigar, Wilson & Wilkins 1988), which is unstable for energy minimization;
    * zero-variance, zero-bias and correlated-sampling estimators of Assaraf & Caffarel;
    * the linear method's $\partial_\theta E_L$ (Toulouse & Umrigar 2007), which is ALE1's anchor vector;
    * CAS × Jastrow in QMC: multideterminant Slater–Jastrow with a CAS expansion (Neuscamman and coworkers' variational optimization of multi-determinant Jastrow wave functions) and the Jastrow antisymmetrized geminal power, JAGP (Casula & Sorella; Neuscamman).

**Generation Batch 5.**
* *Pre-check setup.* H$_4$, $R=4.4$, $K=12$, NNBF-J width 32. Warm start: 100 L-BFGS iterations, reaching $\Delta E=1.153$ mHa. $B=500$, budget 100 EXACT steps, seed 0. Run on the login node.
* *Logs.* `tests/P2.2_stochastic_local_energy_r5/scratch/pre_K12_*.log` and `stale_K12.log`.
* **Route 5.1 (R3.1′ re-entered under F$_{A1}$): window-input backflow × all-orbital Jastrow (WIBJ).** The round-3 kill ("prefactor only") is moot under Statement 1, because every exact scheme in the black-box model is prefactor-only. What is left is whether a prefactor of order $h$ (a forward pass per term replaced by $\mathcal{O}(N^2)$ arithmetic) costs accuracy.
    - Gate 0 (Novelty & Adoption): CONDITIONAL.
        * This is a revival without a new mechanism. The construction is the round-3 one; only the reason for its kill was removed.
        * Prior art is close: active-space backflow (arXiv:2509.25720, 2604.06841), and CAS × Jastrow in QMC (Neuscamman; JAGP). The new claim is only the measured accuracy parity at a large external/window ratio.
        * *Adoption barrier.* External dynamic correlation enters only through the Jastrow and the fixed orbital rows, so it may be lost as the basis grows. That is the externally contracted MRCI failure that killed CGK. The pilot trend (0.022 → 0.142 → 0.405 mHa over $K=12$–20) is what this barrier predicts. Part W now tests it with a pre-committed trend FAIL.
        * *Minimal contrast.* If a CAS × Jastrow with no network, or a narrow unrestricted NNBFJ, matches WIBJ at no greater cost, then the window-input network adds nothing. Part W and Part S now carry those foils.
    - Gate 1 (Conservation & Asymptotics): PASS. The local energy is exact (KEYX equals EXACT to $10^{-14}$ in the pilot). Amplitude work is $\mathcal{O}(N_{\mathrm{key}}MNh)$ on top of the $\mathcal{O}(\lvert\mathcal{C}\rvert N^2)$ arithmetic floor.
    - Gate 2 (Baseline Gameability): PASS on the pre-check, which is not decisive. The pilot accuracy at $K=20$ was combined with round-4 Part S at $K=40$ through the round-4 aggregate:
        * $M_{\mathrm{op}}=0.21$;
        * $M_{\mathrm{wall}}=0.44$, CI $[0.42,0.47]$;
        * the control was SCREEN at the largest admissible $\varepsilon=10^{-3}$ (bias 0.046 mHa). At $\varepsilon=3\times10^{-3}$ the bias is 0.27 mHa, so that point is inadmissible. The grid between them is now filled in.
        * That wall model priced an SS draw at the average per-amplitude time, with no per-call overhead, and left out generation. Both are fixed in Part S.
    - Gate 3 (Hard-Core Shield): PASS (conditional; sign half only). HF weight 0.46, frustrated fraction 0.20. The multireference weight sits in the window, which keeps full backflow. Outside the window the sign structure is mean field dressed by the Jastrow; the test charges this through $\delta$ and the variance ratio.
    - *Verdict*: CONDITIONAL. It is carried into the decisive test, where it can PASS only against the trend rule and the foils.
* **Route 5.2 (Operation B.3; cross-field translation of SVRG, Johnson & Zhang 2013, onto the inner sum; F$_B$): anchored local energy (ALE).**
    * An SR epoch of $T$ steps reuses one sample set, reweighted by $(\psi(\mathbf{n})/\tilde\psi(\mathbf{n}))^2$.
    * At the anchor $\tilde\theta$ the $\tilde x_j$ are enumerated and stored. At $\theta$,
$$E^{\mathrm{ALE0}}=H_{\mathbf{n}\mathbf{n}}+\sum_j\tilde x_j+\frac{1}{K_d}\sum_k\frac{x_{j_k}(\theta)-\tilde x_{j_k}}{p_{j_k}} .$$
    * ALE1 adds $\tilde g(\mathbf{n})\cdot\Delta\theta$, where $\tilde g=\nabla_\theta E_L$ at the anchor costs one backward pass through the anchor enumeration. It subtracts the per-draw JVP.
    * The free information is the earlier iterate, so $s_j=\lvert H\rvert\,\lvert\Delta x_j\rvert\to0$ as $\Delta\theta\to0$, and the sign floor does not apply.
    - Gate 0 (Novelty & Adoption): PASS (narrow). No SVRG local energy on the inner sum was found. The closest prior art is correlated sampling (Umrigar, Wilson & Wilkins 1988), the correlated-sampling and zero-variance estimators of Assaraf & Caffarel, and gradient-level SVRG in NQS (arXiv:2307.01840). The adoption wall is the fixed-sample instability of correlated-sampling energy optimization.
    - Gate 1 (Conservation & Asymptotics): PASS.
        * It is unbiased given the samples, for any $\theta$: smoke $z=2.3$ (ALE0) and $1.3$ (ALE1) over 3000 draws.
        * $\sum_j\mathrm{JVP}_j=\tilde g\cdot\Delta\theta$ to $3\times10^{-10}$.
        * Variance increment over $\mathrm{Var}_\pi(E_L)$ along an EXACT-SR trajectory, $K_d=4$, anchor age 1–16: ALE0 $5\times10^{-3}$ to $6\times10^{-2}$; ALE1 $3\times10^{-6}$ to $1\times10^{-4}$.
    - Gate 2 (Baseline Gameability): **FAIL on the one-seed pre-check, inside the seed noise** (table after this batch). Under the pre-committed rule, $\Delta E_{\mathrm{ALE}}(C_b)=1.046>D_{\mathrm{ctrl}}+\nu=1.039$: a FAIL by 0.007 mHa on one seed.
        * That pre-check ran near convergence (warm start), where Statement 3 already predicts no gain.
        * The regime where ALE could win is a cold start with step-limited optimization, and nothing has tested it.
    - Gate 3 (Hard-Core Shield): PASS (conditional). $\Delta x_j$ is unrelated to the sign structure. HF weight 0.49.
    - *Verdict*: UNTESTED in its decisive regime. Part L (68 tasks, warm start) is dropped from the array because it measures the regime Statement 3 already settles. R5.2 needs a cold-start, step-limited micro-pilot (EXACT against ALE0/ALE1 +B at equal cost from the random initialization, about 3 seeds on H$_4$ $K=12$), which is deferred. The Part L code is kept in `micro_test.py` for that pilot.
* **Route 5.3 (F$_C$): Hadamard-certified screening.** A configuration-independent bound $\lvert\det\Phi(\mathbf{n}')_{[\mathrm{occ}]}\rvert\le\prod_p\big(\lVert\Phi_{0,p}\rVert+\sum_k\lVert W_{2,p,:,k}\rVert\big)$ (using $\lvert\tanh\rvert\le1$), times the exact Jastrow ratio, bounds every dropped term without a forward pass.
    - Gate 2 (Baseline Gameability): **FAIL (measured, $K=12$ warm state).**
        * The certified bias is 140–1000× the actual dropped sum.
        * It already exceeds $\tau$ at $\varepsilon=10^{-4}$: 0.62 mHa certified against $2\times10^{-4}$ mHa actual. At that $\varepsilon$, SCREEN keeps 86% of the terms.
        * So certification is affordable only where screening saves nothing.
    - *Verdict*: KILLED.
* **Route 5.4 (F$_C$): statistical bias certificate.** Sample the dropped tail and bound the bias with a one-sided CI.
    - Gate 0 (Novelty & Adoption): FAIL. This is semistochastic sampling with a confidence statement (arXiv:2507.19276; SHCI PT2), already among the controls.
    - *Verdict*: KILLED.
* **Route 5.5 (F$_B$ without correction): stale-plus-linear $E_L\approx\tilde E_L+\tilde g\cdot\Delta\theta$.**
    - Gate 1 (Conservation & Asymptotics): FAIL. The $\mathcal{O}(\Delta\theta^2)$ bias has no certificate, and the residual that removes it costs only $K_d$ amplitudes (ALE1). R5.2 dominates it.
    - *Verdict*: KILLED.
* **Route 5.6 (F$_B$): learned local-energy regressor, debiased on a subset.**
    - Gate 2 (Baseline Gameability): FAIL. This is R3.2′ with a learned model. R3.2′ measured $R^2=0.13$ on cheap features and $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ between 2 and 466.
    - *Verdict*: KILLED.

**R5.2 pre-check** (H$_4$, $K=12$, seed 0, warm start). Final $\Delta E$ in mHa at equal operation-count cost. "+B" means the arm spends its saving on samples per step instead of on steps.

| Arm | Mode | $\Delta E(C_b)$ | $\Delta E(C_b/2)$ |
| :--- | :--- | :--- | :--- |
| EXACT / EXACTNF | steps | 1.071 / 1.076 | 1.094 / 1.098 |
| SCREEN $10^{-2}$ | steps | 1.196 | 1.208 |
| SCREEN $3\times10^{-3}$ | +B, $B=2000$ | **1.034** | 1.055 |
| SS $10^{-2}$, $K_d=16$ | steps / +B | 1.312 / 1.082 | 1.441 / 1.121 |
| ALE0 $T=8$, $K_d=16$ | steps | 1.145 | 1.082 |
| ALE0 $T=4$, $K_d=16$ | steps / +B, $B=1829$ | 1.236 / **1.046** | 1.073 / 1.057 |
| ALE1 $T=8$, $K_d=16$ | steps | 2.406 (drifts) | 1.139 |

The pattern matches Statement 3:
* every arm that spends its saving on more steps is worse than EXACT;
* every arm that spends it on more samples is better;
* ALE in +B mode only ties SCREEN+B.

### Top Surviving Route & Sharpest Gap (Round 5)
* **No unconditional survivor.**
* **R5.1 (WIBJ with an exact keyed local energy, F$_{A1}$): CONDITIONAL.**
    * Accuracy at $K=20$: $\delta_{\mathrm{med}}=+0.097$ mHa and $\delta_{\mathrm{best}}=+0.060$ mHa.
    * But its absolute median $\Delta E$ rises 0.022 → 0.142 → 0.405 mHa over $K=12$, 16, 20.
    * Cost pre-check at $K=40$: $M_{\mathrm{op}}=0.21$, and the upper bound of the $M_{\mathrm{wall}}$ CI is 0.47. That wall model had no per-call overhead and no generation.
* **R5.2 (ALE, F$_B$): untested in its decisive regime** (cold start, step-limited). Its micro-pilot is deferred.
* **Sharpest gap (R5.1).** Does WIBJ's error relative to NNBF-J stay flat as the external space grows? The test is $\delta_{\mathrm{med}}(K)$ on H$_4$ at $K=12$–24, extrapolated to $K=40$, the smallest Part S size where the cost is claimed. A second question: does any cheaper known construction (CAS × J, a narrow NNBFJ) do as well?
* **Unexplored regime.** F$_{A2}$ (sub-$\lvert\mathcal{C}\rvert$ by contraction) is open. The R4.0 contraction is exact, but it needs a factorizable Jastrow, whose accuracy has not been measured.

### 24-Hour Decisive Test (Round 5)
* **Part W (R5.1 accuracy): ED, L-BFGS on the exact energy.**
    * *Instances.* Distorted H$_4$ at $R=4.4$, $K=12$, 16, 20, 24 (dimension up to 76 176), for the $K$-trend. Distorted H$_6$ at $R=3.6$, $K=12$, 14 (dimension up to 132 496). Every instance has HF weight below 0.6, checked in code.
    * *Arms*, each with three seeds:
        * NNBFJ (seeds 0–2);
        * NNBFJNF (seeds 3–5), the noise floor;
        * WIBJ (seeds 0–2);
        * NNBFJ-h4 and NNBFJ-h8: unrestricted NNBFJ with hidden width 4 or 8 and the same neural-Jastrow width. This is the minimal-contrast foil whose per-term cost is of the order of WIBJ's keyed per-term cost;
        * CASJ: the minimal-contrast foil with no network. It is a CAS($N$, $K_W$) expansion with all $\binom{K_W}{N/2}^2$ window determinants (36 for H$_4$, 400 for H$_6$), trainable orbital rows for both spins (so external occupations enter through orbital mixing), and the all-orbital pair Jastrow without the neural part.

      Width 64 ($h_J=32$) for NNBFJ and WIBJ, 6000 L-BFGS iterations for every arm.
    * *Controls on NNBFJ.* Exact SCREEN biases over the extended grid (now including $\varepsilon=1.5$, 2 and $2.5\times10^{-3}$); SS increments (half A / half B, tail rule); TRUNC.
    * *Trained-state samples.* Every arm reports per-sample operation counts on $\lvert\psi\rvert^2$ samples of its own trained state. WIBJ reports KEYX and KEYX restricted to SCREEN-kept terms.
* **Part S (R5.1 cost): ED-free.**
    * *Instance.* Distorted H$_8$ chain, $R=2.0$, $K=40$, 72, 104, 136.
    * *Arms.* NNBFJ and WIBJ, timed in the same task, plus the three foils (operation counts).
    * *Wall model* (the Part L calibration):
        * generation is timed per row and fitted as $t=a_g+b_g\lvert\mathcal{C}\rvert$;
        * `ratio_exact` is timed on prefixes of $m=1,\dots,\lvert\mathcal{C}\rvert$ terms and fitted as $t=a+bm$, where $a$ is the per-call overhead;
        * EXACT, KEYX and TRUNC are charged generation of all $\lvert\mathcal{C}\rvert$ terms. SCREEN and SS are charged generation of kept terms only, which is generous to them (HCI-sorted integrals);
        * an SS draw call is charged $a+bK_d$;
        * $n_{\mathrm{wall}}=60$ samples.
    * *Reported beside the verdict:* WIBJ+SCREEN (KEYX on the kept terms) against NNBFJ+SCREEN at $\varepsilon^*$, in operations and wall time; each foil's cost over WIBJ's.
    * *Limitation.* At $K\ge40$ there is no ED and no trained state, so Part S samples come from a CISD-like proposal (HF plus MP1-weighted singles and doubles). The trained-state cross-check is the Part W operation count at $K\le24$.
* **R5.1 thresholds** (implemented in `verdict_R51`; $\tau=0.2$ mHa).
    * *Accuracy*, at each instance's largest $K$:
        * $\delta_{\mathrm{med}}$ is the paired median of $\Delta E_{\mathrm{WIBJ}}-\Delta E_{\mathrm{NNBFJ}}$ over seeds 0–2;
        * $\delta_{\mathrm{best}}$ is the best of the 3 WIBJ seeds minus the best of all 6 NNBFJ seeds (0–5). It is asymmetric on purpose, against WIBJ;
        * $\nu_W=\max(\lvert\mathrm{med\,NNBFJ}_{0\text{–}2}-\mathrm{med\,NNBFJ}_{3\text{–}5}\rvert,\ \tfrac12\,\mathrm{range}_{0\text{–}5})$, the Part L noise-floor definition;
        * premise: HF weight below 0.6, median NNBFJ $\Delta E$ below 1.6 mHa, and $\nu_W<\tau$;
        * ACC_OK needs $\delta_{\mathrm{med}}\le\tau$ and $\delta_{\mathrm{best}}\le\tau$ on every instance, with every premise holding;
        * ACC_FAIL if $\delta_{\mathrm{med}}>\tau$ on any premise-valid instance.
    * *Trend*, on every instance with at least three $K$ values (H$_4$): TREND_FAIL if $\delta_{\mathrm{med}}(K)$ rises strictly monotonically **and** its least-squares line in $K$, extrapolated to $K=40$, exceeds $\tau$. WIBJ's absolute median $\Delta E(K)$ is printed beside $\delta_{\mathrm{med}}(K)$. H$_6$ has two $K$ values and enters only through the accuracy rule.
    * *Foils:* FOIL_FAIL if some foil $F$ meets both of these:
        * on every instance, at the largest $K$, its median $\Delta E$ is within the noise floor of WIBJ's ($\mathrm{med}\,\Delta E_F\le\mathrm{med}\,\Delta E_{\mathrm{WIBJ}}+\nu_W$);
        * at Part S $K_{\max}$ it is no more expensive: $C_F=\min(\mathrm{ops}_{\mathrm{EXACT}},\mathrm{ops}_{\mathrm{SCREEN}(\varepsilon^*)})\,v_F\le C_{\mathrm{WIBJ}}=\min(\mathrm{ops}_{\mathrm{KEYX}},\mathrm{ops}_{\mathrm{KEYX+SCREEN}(\varepsilon^*)})\,v_{\mathrm{WIBJ}}$, where $v$ is the variance ratio to NNBFJ.

      CASJ is priced generously: Sherman–Morrison minor updates per changed spin plus an $n_s$ or $n_s^2$ contraction per term.
    * *Cost*, from Part S:
        * $M_{\mathrm{op}}(K)$ is the variance-corrected KEYX operation count of WIBJ divided by the cheapest NNBFJ control (EXACT, SCREEN($\varepsilon^*$), SS with its measured increment, or admissible TRUNC);
        * $M_{\mathrm{wall}}(K)$ is the same ratio for the wall model, against the control with the smallest mean wall time, with a paired bootstrap 95% CI.
    * **PASS**: ACC_OK, no TREND_FAIL, no FOIL_FAIL, $M_{\mathrm{op}}(136)\le0.5$, $M_{\mathrm{op}}$ falling with $K$, and an upper CI bound of $M_{\mathrm{wall}}(136)$ of at most 0.8.
    * **FAIL**: ACC_FAIL, TREND_FAIL, FOIL_FAIL, $M_{\mathrm{op}}(136)\ge1$, or a lower CI bound of $M_{\mathrm{wall}}(136)$ above 1.
    * **AMBIGUOUS** otherwise. Part S at $K=168$ then decides: PASS iff $M_{\mathrm{op}}\le0.5$ and the upper CI bound is at most 0.8, else KILL.
* **R5.2.** Not in this test (verdict printed as UNTESTED).
* **First-row instance: not feasible.** The integral code builds hydrogen chains only: nuclear charge 1, hydrogen s and p shells, and electrons equal to atoms. LiH or BeH$_2$ would need Li/Be basis sets and core handling (frozen core). This limitation is stated rather than worked around.
* **Can PASS and can FAIL.**
    * The `verdict_R51` self-test on synthetic results returns PASS, FAIL (accuracy), FAIL (trend) and FAIL (foil).
    * The smoke aggregate returns ACC_OK and a cost-AMBIGUOUS verdict at smoke scale.
* **Execution.**
    * *Code.* `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/micro_test.py`, with `keyx_core.py` (a verbatim copy of the round-4 code) and `submit_test.sbatch`.
    * *Job.* Array 0–43, partition `kim`, 2 CPUs, 64 GB, 24 h:
        * tasks 0–35 are Part W, one per (instance, arm);
        * tasks 36–43 are Part S, one per basis and arm group (main or foils).

      Expect at most about 10 h per task (CASJ on H$_6$ $K=14$; 0.7 s per L-BFGS iteration at $K=12$, measured).
    * *Command (not run).* `sbatch /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/submit_test.sbatch`.
    * *Verdict.* `python3 micro_test.py --aggregate <test dir>`.
* **Smoke test.** Command: `/share/apps/software/anaconda3/bin/python3 micro_test.py --smoke` (43 s). It runs the checks, tiny Part W on H$_4$ at $K=6$, 7, 8 with all six arms, tiny Part S (both arm groups, $K=20$ and 36), and the aggregate. Below are the literal check lines and the verdict lines; the full output is in `smoke_output.txt`.

```
  logderiv vs FD rel err=4.7e-09; VJP vs per-sample O err=7.1e-15
  CASJ N=4 K=6: dets=36 P=162 grad rel err=5.3e-10 per-sample ratio err=2.3e-13; cofactor err=2.1e-10
  CASJ N=6 K=7: dets=400 P=589 grad rel err=3.3e-09 per-sample ratio err=1.6e-12; cofactor err=2.1e-10
  verdict_R51 self-test on synthetic results: {'pass': 'PASS', 'fail': 'FAIL', 'trend': 'FAIL', 'foil': 'FAIL'} (must be {'pass': 'PASS', 'fail': 'FAIL', 'trend': 'FAIL', 'foil': 'FAIL'})
SMOKE CHECKS PASS
  "accuracy": "ACC_OK",
  "trend_fail": false,
  "foil_fail": false,
  "verdict": "AMBIGUOUS (pre-committed: Part S --basis-series 6,5; PASS iff M_op<=0.5 and M_wall CI upper<=0.8, else KILL)"
  "verdict": "UNTESTED (deferred: needs a cold-start, step-limited micro-pilot)"
```

* *Smoke-scale read-out* (width 8, 150 iterations; plumbing, not evidence):
    * CASJ already reaches 0.09–0.14 mHa on H$_4$ at $K=6$–8, against 0.60–0.67 mHa for NNBFJ and WIBJ at this width. Its accuracy clause holds, but its cost ratio to WIBJ is 1.31 at $K=36$, so it does not trigger FOIL_FAIL.
    * With the new wall model, WIBJ's $M_{\mathrm{wall}}$ at $K=36$ is 0.99 (CI 0.96–1.02) at width 8, against $M_{\mathrm{op}}=0.78$.

### Local Autopsy Log (Round 5 additions)
* **R4.1 (CGK)**: killed by pilot 411919 ($\delta=0.691$ mHa at $K=20$). Coarse external occupation counts cannot carry pair correlation that grows with the basis.
* **Saturation (new boundary condition for the leaf)**: in the black-box query model, exact local energies need $\Omega(\lvert\mathcal{C}\rvert)$ oracle calls, and unbiased ones obey the inclusion-probability variance bound. Every future route must say which of F$_{A1}$, F$_{A2}$, F$_B$ or F$_C$ it targets.
* **Critic correction (Saturation Diagnostic)**: the first version cited Godambe–Joshi, which bounds the anticipated variance of design-unbiased estimators under a superpopulation model and does not apply to exact computation. It was replaced by the query lower bound and the inclusion-probability bound, and F$_A$ was split.
* **Critic correction (R5.1)**: re-entry downgraded to CONDITIONAL. Removing the reason for a kill is not a new mechanism. The external-correlation barrier (the CGK failure) is now tested by a pre-committed trend rule.
* **Critic correction (R5.2)**: the warm-start Part L tested only the regime Statement 3 already settles. It was dropped, and R5.2 is recorded as untested in its decisive regime.
* **R5.3 (Hadamard-certified screening)**: the configuration-independent bound is 140–1000× loose.
* **R5.4, R5.5, R5.6**: semistochastic relabel, uncertified bias, and R3.2′ relabel respectively.
* **Measured lesson (Part L pre-check)**: near convergence, a cheaper local energy helps SR only when the saving buys more fresh samples. Spending it on more steps at fixed $B$ made SCREEN, SS and ALE worse than EXACT. Any cost metric for a cheap local energy inside VMC must let the controls spend their saving on samples.
* **New anti-patterns proposed for the master list**:
    * "a cheap local-energy estimator compared at a fixed number of samples per step, which lets a cheaper control convert its saving only into extra noisy steps";
    * "a restricted ansatz judged at one basis size, without its error trend in $K$ and without a minimal-contrast foil (a known cheaper construction)".
