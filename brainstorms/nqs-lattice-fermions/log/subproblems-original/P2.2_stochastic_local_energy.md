# Subproblem: P2.2 Unbiased cheap local energy for second-quantized chemistry
**Parent Chain**: P2 -> P2.2  **Status**: EXHAUSTED at this budget (2026-11-17: R7.4, certified screening in a localized basis, KILLED at T0 on cost; R5.2 SHELVED by the compute gate's G2 rule; every reopening-ledger row closed; per-leaf critic SIGN-OFF-WITH-CHANGES 2026-11-17, applied; see the Leaf Status Summary at the end). Earlier: DORMANT from 2026-11-16. History: round 5 decisive test (job 452238): R5.1 FAIL on cost (pre-committed); per-leaf critic KILL, no mutation (screened WIBJ is a post-hoc rescue); leaf closed with the Saturation Diagnostic; closed-leaf falsifiers and the reopening review (2026-11-16) did not reopen it. CGK (R4.1) was killed by pilot 411919. A Saturation Diagnostic (black-box query lower bound plus the inclusion-probability variance bound) shows that the leaf as first stated (an unbiased local energy with $C/C_{\mathrm{ex}}\to0$ for a general black-box NNBF) is impossible. The leaf is refined to four regimes: F$_{A1}$ (cheaper per term), F$_{A2}$ (sub-$\lvert\mathcal{C}\rvert$ by contraction), F$_B$ (amortization) and F$_C$ (certified bias).
* Round-5 snapshot (before job 452238; superseded by the results below): no unconditional survivor. R5.1 (WIBJ, F$_{A1}$) is CONDITIONAL: a revival without a new mechanism.
    * Pilot WIBJ median $\Delta E$ grows with the basis: 0.022, 0.142 and 0.405 mHa at $K=12$, 16 and 20.
    * Its adoption barrier is the loss of external dynamic correlation as the basis grows, the failure that killed CGK.
* R5.2 (anchored local energy, F$_B$) is untested in its decisive regime; its micro-pilot is deferred.
* Four kills: certified screening, the statistical certificate, the stale-linear estimate and the learned regressor.
* The decisive test is rewritten and smoke-tested, not submitted: 44 tasks. Part W measures the WIBJ accuracy trend at $K=12$–24 with two kinds of minimal-contrast foil, and Part S measures the ED-free cost up to $K=136$ with a calibrated wall model.
### 1. Mathematical Abstraction & Barriers
* **Formulation**: Let $\pi(\mathbf{n})=\psi(\mathbf{n})^2/\sum_{\mathbf{m}}\psi(\mathbf{m})^2$ be the VMC sampling distribution of a real NQS $\psi$ on $M$ spin-orbitals with $N$ electrons. Let $\mathcal{C}(\mathbf{n})$ be the set of configurations $\mathbf{n}'\neq\mathbf{n}$ with $H_{\mathbf{n}\mathbf{n}'}\neq0$; for a dense two-body tensor $\lvert\mathcal{C}(\mathbf{n})\rvert=\mathcal{O}(N^2(M-N)^2)$. Write $r(\mathbf{n}')=\psi(\mathbf{n}')/\psi(\mathbf{n})$ and $x_{\mathbf{n}'}=H_{\mathbf{n}\mathbf{n}'}r(\mathbf{n}')$, so
$$E_L(\mathbf{n})=H_{\mathbf{n}\mathbf{n}}+\sum_{\mathbf{n}'\in\mathcal{C}(\mathbf{n})}x_{\mathbf{n}'} .$$
*Input*: $h_{pq}$, $V_{pqrs}$, the network $\psi_\theta$, and a sample $\mathbf{n}\sim\pi$. *Output*: an estimator $\hat{E}(\mathbf{n})$ with $\mathbb{E}[\hat{E}\mid\mathbf{n}]=E_L(\mathbf{n})$ built from $K$ draws, with conditional single-draw variance $\tau^2(\mathbf{n})$. By the law of total variance the per-sample variance is
$$\sigma^2_{\mathrm{tot}}=\mathrm{Var}_\pi(E_L)+\mathbb{E}_\pi[\tau^2(\mathbf{n})]/K .$$
The cost to reach statistical error $\epsilon$ on $E$ is $C=\sigma^2_{\mathrm{tot}}\,\bar{c}/\epsilon^2$, where $\bar{c}=\mathbb{E}_\pi[c(\mathbf{n})]$ is the mean per-sample cost counted in NQS forward passes (one per amplitude $\psi(\mathbf{n}')$ actually evaluated, plus $\kappa$ per cheap arithmetic term, $\kappa\ll1$). Exact enumeration has $C_{\mathrm{ex}}=\mathrm{Var}_\pi(E_L)(1+\mathbb{E}_\pi\lvert\mathcal{C}\rvert)/\epsilon^2$. *Solved* means: unbiased (or bias below $0.1\times1.6$ mHa), $C/C_{\mathrm{ex}}\to0$ as $M$ grows, at a wavefunction quality near chemical accuracy, on a Hard-Core instance, and below the cost of the existing screening and semistochastic baselines.
Gradients: the SR force $f_k=2\,\mathrm{Cov}_\pi(O_k,E_L)$ with $O_k=\partial_{\theta_k}\ln\psi$ is linear in $E_L$, and $S_{kl}=\mathrm{Cov}_\pi(O_k,O_l)$ does not contain $E_L$. With subsampling noise independent across samples, $\mathbb{E}[\hat{f}\mid\{\mathbf{n}\}]=f$ and $\mathbb{E}[S^{-1}\hat{f}\mid\{\mathbf{n}\}]=S^{-1}f$, so the SR step stays conditionally unbiased. Only the nonlinear uses are biased: the variance estimate, which is inflated by $\mathbb{E}_\pi[\tau^2]/K$ (and therefore the V-score and the variance extrapolations used by P3.1 and P3.2). The fix is a two-replica estimator $\hat{E}^{(1)}\hat{E}^{(2)}$ from independent draws, which is unbiased for $E_L^2$. The real damage is not bias. It is that the force noise no longer vanishes as $\psi\to\psi_0$.
* **The Node Barrier**: *Zero-variance mismatch with a sign floor.* As $\psi\to\psi_0$, $\mathrm{Var}_\pi(E_L)\to0$, but $\tau^2$ does not. For any unbiased single-draw importance estimator of $\sum_j x_j$ with proposal $p$, the minimum over $p$ is reached at $p_j\propto\lvert x_j\rvert$ and equals
$$\tau^2_{\min}(\mathbf{n})=\Big(\sum_j\lvert x_j\rvert\Big)^2-\Big(\sum_j x_j\Big)^2 ,$$
which is zero if and only if all $x_j=H_{\mathbf{n}\mathbf{n}'}\psi(\mathbf{n}')/\psi(\mathbf{n})$ have the same sign, i.e. if the row is sign-free (stoquastic in this basis for $\psi_0$). For a non-stoquastic fermionic Hamiltonian, every purely stochastic local-energy estimator, even with oracle knowledge of $\psi(\mathbf{n}')$, therefore has $C/C_{\mathrm{ex}}\to\infty$ as the variational error goes to zero. The Hard Core (sign structure) appears directly as an irreducible variance. Only a *deterministic* subtraction that carries the sign can escape it. Existing approaches halt as follows:
    - Exact enumeration with GPU parallelism (Zhao, Stokes & Veerapaneni arXiv:2208.05637; NNQS-Transformer arXiv:2306.16705) keeps zero variance but pays $\mathcal{O}(M^4)$ forward passes per sample (practical barrier).
    - Sample-restricted or cached local-energy surrogates (Malyshev, Schmitt & Lvovsky arXiv:2408.07625) and target-space truncation (Liu & Clark arXiv:2502.18843; solids arXiv:2603.14775) are biased. The bias is controlled only by how well a small set of important configurations captures $\psi$, and that set grows with multireference character (Hard Core).
    - Semistochastic local energies, with exact large $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$ and a sampled tail (hybrid TN/NQS arXiv:2507.19276; transformer backflow arXiv:2509.25720; spin-adapted NNBF arXiv:2604.06841), are unbiased but inherit the floor on the tail. They are adopted only as a speed-up at a loose tolerance, not as a zero-variance-preserving estimator. The heat-bath and semistochastic ideas come from SHCI (arXiv:1610.06660, arXiv:1809.04600), where the sampled sum is a PT2 correction, not the variational energy.
    - Orbital-space VMC with Jastrow–Slater states (Sabzevari & Sharma arXiv:1807.10633) makes all $\mathcal{O}(M^4)$ ratios cheap arithmetic through Green's-function determinant ratios. This works only because the ansatz has no configuration-dependent orbitals. It is the arithmetic that a surrogate can reuse.
    - *Fundamental*: the sign floor $\tau^2_{\min}$. *Practical*: the cost per forward pass, batching, and the $\kappa$ ratio.
    - **Anti-Solution Space**: (i) any purely stochastic sampler, whatever the proposal: heat-bath, factorized/Cholesky, learned. All are bounded below by $\tau^2_{\min}$. (ii) Truncations presented as unbiased. (iii) Speed-ups measured on poorly converged states, where $\mathrm{Var}_\pi(E_L)$ is large and hides $\tau^2$. (iv) Gains shown only near equilibrium, where the HF weight is above 0.9. (v) Gains that disappear once $\kappa$ is set to a realistic value.
### 1b. Analogy Map
Written in the 2026-11-17 documentation pass. It records retroactively which fields the routes of rounds 1–5 and the reopening review drew on, and adds the untried sources found then. Each entry is marked untried, tried (R<id>) or repaired (R<id>); the R<id> are route cards in the Route Register. Symbols are those of section 1: the row sum $\sum_jx_j$ with $x_j=H_{\mathbf{n}\mathbf{n}_j}\psi(\mathbf{n}_j)/\psi(\mathbf{n})$, the connected set $\mathcal{C}$, and a cheap surrogate $y_j$ of $x_j$.
* **Barrier B1 (sampling floor), field-neutral**: estimate a sum of $n$ terms of both signs whose magnitudes span orders of magnitude, where each term costs one expensive black-box evaluation, from far fewer than $n$ evaluations, without bias, and with a variance that should vanish when the sum's summands become "converged" (here: when $\psi\to\psi_0$ the exact sum has zero variance over samples, but any sampling of its terms does not).
    * *Survey sampling (Horvitz–Thompson estimation, Neyman allocation).* Source problem: estimate a population total from a sample of units drawn with unequal inclusion probabilities. Correspondence: population unit ↔ connected configuration $\mathbf{n}_j$; unit value ↔ $x_j$; auxiliary size variable known for every unit ↔ surrogate $\lvert y_j\rvert$; strata ↔ samples $\mathbf{n}$ (across-sample allocation). Technique: inclusion probabilities $\pi_j\propto$ size variable, and Neyman allocation of draws across strata in proportion to stratum spread. Why it worked there: the size variable is nearly proportional to the unit value, so $x_j/\pi_j$ is nearly constant. Translation gap: here the best cheap size variable (frozen backflow) has residual fraction 0.39–0.79, and the HT variance $\sum_jx_j^2(1/\pi_j-1)$ keeps that magnitude dispersion. Citation: Horvitz & Thompson, JASA 47, 663 (1952); Neyman, JRSS 97, 558 (1934). Status: tried (R3.5′ Horvitz–Thompson; R2.5 Neyman-type allocation); its variance form became Statement 2 of the Saturation Diagnostic.
    * *Monte Carlo variance reduction (control variates, positivisation).* Source problem: integrate a signed function by importance sampling. Correspondence: integrand ↔ $x_j$; control function with known integral ↔ surrogate $y_j$ summed exactly; positive and negative parts ↔ sign strata of $y$. Technique: subtract a control variate with known sum and sample only the residual; positivisation samples the positive and negative parts with separate proposals, which removes the single-proposal floor $(\sum\lvert x\rvert)^2-(\sum x)^2$. Why it worked there: the control is a smooth, accurate model of the integrand, and the sign regions are known. Translation gap: the residual $x_j-y_j$ of every cheap NQS surrogate is large (residual fractions 0.21–0.88), and the floor left after positivisation is the magnitude dispersion, not the sign. Citation: Owen & Zhou, JASA 95, 135 (2000), DOI:10.1080/01621459.2000.10473909. Status: tried (R1.4 and R2.0 control variate; R2.1 positivisation; R3.3′ calibrated control; R3.7′ and R3.2 partially frozen control).
    * *Multilevel and multifidelity Monte Carlo.* Source problem: estimate an expectation of an expensive simulation (for example an SDE path at fine time step) using a hierarchy of cheaper, correlated models. Correspondence: fine model ↔ exact amplitude ratio; coarse models ↔ frozen and linearized backflow surrogates, or a coarser orbital basis; levels ↔ fidelity of the surrogate. Technique: telescoping sum of level differences, with draws allocated by variance-to-cost ratio. Why it worked there: the level differences have variance that decays geometrically while their cost grows geometrically, so most work goes to cheap levels. Translation gap: the surrogate hierarchy has only two or three levels, the expensive setup of the linearized level ($\kappa_{\mathrm{S,LRX}}=12.4$ forward passes per sample) eats the gain, and across samples the surrogate error is sample-dependent ($\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ between 2 and 466). Citation: Giles, Oper. Res. 56, 607 (2008) (arXiv:1304.5472 review); Peherstorfer, Willcox & Gunzburger, SIAM J. Sci. Comput. 38, A3163 (2016). Status: tried (R2.4 surrogate-fidelity MLMC; R2.9 basis-level MLMC; R3.2′ cross-sample multifidelity; R5.6 learned regressor).
    * *Projector quantum Monte Carlo (FCIQMC, semistochastic, heat-bath selected CI).* Source problem: apply a huge sparse Hamiltonian to a stochastic wavefunction, or sum a second-order perturbative correction over a huge external space. Correspondence: deterministic subspace ↔ large-$\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$ terms evaluated exactly; stochastic tail ↔ sampled small terms; heat-bath proposal ↔ $p\propto\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$. Technique: exact treatment of a small important set, stochastic treatment of the rest. Why it worked there: in projector QMC the stochastic noise is averaged over long imaginary-time trajectories, and in SHCI the sampled sum is a small correction, not the variational energy. Translation gap: in VMC the sampled sum is the whole local energy on every sample, so the tail variance does not vanish at convergence and competes with a zero-variance exact sum. Citation: Petruzielo, Holmes, Changlani, Nightingale & Umrigar, PRL 109, 230201 (2012); Sharma, Holmes, Jeanmairet, Alavi & Umrigar, arXiv:1610.06660. Status: tried (R1.1 heat-bath and semistochastic; R5.4 statistical certificate; R7.5 anchored tail).
    * *Randomized numerical linear algebra (sketching, approximate matrix products).* Source problem: approximate an inner product or a matrix product from a few sampled or sketched coordinates. Correspondence: inner product $\langle h,r\rangle$ ↔ $E_L-H_{\mathbf{n}\mathbf{n}}$, with $h$ the Hamiltonian row and $r$ the amplitude-ratio vector. Technique: column sampling with $p_j\propto\lVert a_j\rVert\lVert b_j\rVert$, or a dense random sketch $S$. Why it worked there: both factors are cheap to read in full, so norms are known and a dense sketch is cheap. Translation gap: here reading $r$ costs one forward pass per coordinate, so a dense sketch needs every amplitude and a sparse one is single-proposal sampling. Citation: Drineas, Kannan & Mahoney, SIAM J. Comput. 36, 132 (2006), DOI:10.1137/S0097539704442684. Status: tried (R2.7).
    * *Algorithmic query complexity (adversary lower bounds).* Source problem: prove how many oracle calls any algorithm needs to compute a function of a hidden input. Correspondence: hidden input ↔ the amplitudes $\psi(\mathbf{n}_j)$; oracle call ↔ forward pass. Technique: an adversary changes an unqueried input without changing any answer. Why it worked there: the oracle model is exactly the cost model. Translation gap: none within the black-box model; it says nothing about architectures that expose structure. Citation: standard adversary argument (e.g. Ambainis, J. Comput. Syst. Sci. 64, 750 (2002), for the quantum version). Status: tried (Statement 1 of the Saturation Diagnostic; not a route).
* **Barrier B2 (per-term cost), field-neutral**: evaluate an expensive nonlinear function at $n$ inputs that each differ from a reference input by a small, structured change (two bits moved), when the function's internal state could in principle be updated rather than recomputed.
    * *Quantum Monte Carlo determinant updates (Sherman–Morrison, table method).* Source problem: evaluate all single and double excitation ratios of a Slater–Jastrow wavefunction. Correspondence: reference determinant ↔ backflow matrix at $\mathbf{n}$; excitation ↔ connected $\mathbf{n}'$; $2\times2$ minor of the Green's function ↔ keyed ratio. Technique: precompute $A^{-1}$ once, then each ratio is a small minor. Why it worked there: the orbitals do not depend on the configuration. Translation gap: neural backflow orbitals do depend on it, so the frozen ratio is only a surrogate (residual 0.39–0.82), unless the backflow input is restricted (WIBJ), which costs accuracy. Citation: Clark, Morales, McMinis, Kim & Scuseria, JCP 135, 244105 (2011); Filippi, Assaraf & Moroni, JCP 144, 194105 (2016); Sabzevari & Sharma, arXiv:1807.10633. Status: tried (R1.4 frozen surrogate; R3.1 and R5.1 WIBJ keyed ratios; R4.2 cross-spin keys).
    * *Configuration interaction occupation-class blocking (GAS, RAS) and externally contracted MRCI.* Source problem: organize a huge CI space so that all determinants sharing an occupation pattern of a reference space are handled together. Correspondence: occupation class ↔ window key; external space ↔ orbitals outside the window. Technique: block the work by class and contract the external part. Why it worked there: the external correlation is perturbative and the contraction error is accepted as a known approximation. Translation gap: the contraction error grows with the external space, which is exactly what killed CGK (R4.1) and WIBJ (R5.1). Citation: Olsen, Roos, Jørgensen & Jensen, JCP 89, 2185 (1988) (restricted active space CI). Status: tried (R3.1, R4.1, R5.1; the round-4 critic named the source).
    * *Auxiliary-field QMC local energy (Cholesky and tensor-hypercontraction contractions).* Source problem: compute the local energy of a non-orthogonal determinant walker with a dense two-body tensor. Correspondence: Cholesky vectors $L^\gamma_{pq}$ ↔ the same; walker Green's function ↔ frozen backflow Green's function. Technique: contract $\sum_{pqrs}V_{pqrs}G_{pr}G_{qs}$ through $N_\gamma$ vectors at $\mathcal{O}(N_\gamma MN)$ instead of enumerating $\mathcal{O}(N^2M^2)$ excitations. Why it worked there: one set of orbitals serves every excitation. Translation gap: with backflow, the orbitals change per class, and per-class contraction costs the same order as enumeration because $N_\gamma\approx7K$. Citation: Motta & Zhang, arXiv:1711.02242. Status: tried (R2.8, R3.6′, R4.0, R6.1).
    * *Automatic differentiation and first-order surrogates (linearization, tangent models).* Source problem: cheap approximate re-evaluation of a network under small input changes (sensitivity analysis). Correspondence: input change ↔ the two-bit excitation; Jacobian-vector product ↔ linearized backflow response. Technique: first-order Taylor surrogate of the changed rows. Why it worked there: the input change is small in a smooth function. Translation gap: a two-orbital excitation is not small, and the log-determinant response diverges at nodes (R3.0); the exact two-row update plus linearized rest (LRX) is accurate but costs 0.21 forward passes per term. Citation: Griewank & Walther, *Evaluating Derivatives*, SIAM (2008). Status: tried (R2.3, R2.4, R3.0, R5.5).
* **Barrier B3 (noise in the optimizer), field-neutral**: in stochastic optimization where each objective sample is itself an expensive inner sum, reuse earlier evaluations or cheapen the inner sum without letting the optimizer's noise floor rise.
    * *Variance-reduced stochastic gradients (SVRG).* Source problem: finite-sum minimization where a full gradient is expensive. Correspondence: anchor full gradient ↔ anchor enumeration of $\tilde x_j$ at an earlier parameter; per-step correction ↔ sampled $x_j(\theta)-\tilde x_j$. Technique: anchor plus control variate refreshed each epoch. Why it worked there: the finite sum is fixed, so the anchor stays exact for the same data, and the correction variance shrinks as the iterate converges. Translation gap: in VMC the samples are redrawn from $\psi_\theta^2$ (reweighting is unstable), and while optimization is noise-limited reuse gives no first-order gain (Statement 3). Citation: Johnson & Zhang, NeurIPS 2013. Status: tried (R5.2, R7.5).
    * *Correlated sampling and fixed-sample reweighting in QMC.* Source problem: compare energies of nearby wavefunctions on one sample set. Correspondence: reweighting factor $(\psi/\tilde\psi)^2$ ↔ the same. Technique: reuse samples across parameters. Why it worked there: for small parameter changes, correlated noise cancels in energy differences. Translation gap: energy minimization on a fixed sample is unstable. Citation: Umrigar, Wilson & Wilkins, PRL 60, 1719 (1988). Status: tried (R5.2).
    * *Doubly stochastic gradients (kernel machine learning).* Source problem: gradients where both the data point and a random feature are sampled. Correspondence: sampled random feature ↔ sampled connected term. Technique: use unbiased doubly stochastic gradients with decaying step size. Why it worked there: the objective is convex and step-size decay gives convergence. Translation gap: the inner noise inherits the sign floor and is amplified by $S^{-1}$ in stochastic reconfiguration, and step decay forfeits the speed-up. Citation: Dai, Xie, He, Liang, Raj, Balcan & Song, NeurIPS 2014. Status: tried (R1.2).
* **Barrier B4 (certifying a truncation), field-neutral**: drop many small signed terms from a quadratic form and certify the result is still an upper bound, or certify the size of the dropped part, at a cost below evaluating the dropped part.
    * *Lattice fixed-node approximation.* Source problem: make a non-stoquastic projector Monte Carlo variational. Correspondence: sign-violating hops ↔ dropped small elements; diagonal compensation ↔ the same. Technique: move dropped off-diagonal elements onto the diagonal so that $H'\succeq H$. Why it worked there: only the sign-violating hops are moved, and they are few. Translation gap: screening drops terms of both signs, and every sign-favourable dropped pair costs at least twice its magnitude with no cancellation (Statement 4). Citation: ten Haaf, van Bemmel, van Leeuwen & van Saarloos, PRB 51, 13039 (1995). Status: tried (R7.1, R7.4).
    * *Integral screening bounds in quantum chemistry (Schwarz, Hadamard-type).* Source problem: drop two-electron integrals below a threshold with a rigorous error bound. Correspondence: integral bound ↔ bound on a dropped term. Technique: Cauchy–Schwarz bound $\lvert(pq\vert rs)\rvert\le\sqrt{(pq\vert pq)(rs\vert rs)}$. Why it worked there: the bound is tight for the integrals that matter. Translation gap: a configuration-independent bound on a backflow determinant is 140–1000 times loose. Citation: Häser & Ahlrichs, J. Comput. Chem. 10, 104 (1989). Status: tried (R5.3, a Hadamard bound of the same kind).
    * *Positive semidefinite matrix completion.* Source problem: complete a partially specified Hermitian matrix to a positive semidefinite one. Correspondence: kept matrix elements ↔ specified entries; the compensator ↔ the completion. Technique: chordal-graph completion results and determinant-maximizing completions. Why it worked there: the specified pattern has chordal structure. Translation gap: unknown; this is the non-diagonal compensator named as the next step of R7.4 (C3′), and it has not been derived. Citation: Grone, Johnson, Sá & Wolkowicz, Linear Algebra Appl. 58, 109 (1984), DOI:10.1016/0024-3795(84)90207-6. Status: untried.
    * *Linear-scaling electronic structure (nearsightedness, local correlation).* Source problem: make the number of significant terms grow linearly with system size. Correspondence: localized orbitals ↔ Löwdin site orbitals; decay of couplings with distance ↔ decay of dropped $H_{\mathbf{n}\mathbf{n}'}$. Technique: work in a localized basis so that small terms can be dropped with an extensive saving. Why it worked there: insulating electronic structure is nearsighted. Translation gap: the saving belongs to the basis, not to the estimator, and whether a trained NNBF stays accurate in that basis is a P2.1 question. Citation: Goedecker, RMP 71, 1085 (1999); for NQS in localized orbitals, arXiv:2305.12776. Status: tried (R7.3, R7.4).
### 2. Candidate Routes & Falsification Gates
* **Route 1**: Heat-bath / semistochastic stochastic local energy. Exact on $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert>\varepsilon$, with $K$ draws $p\propto\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$ on the tail. Point of departure: SHCI PT2 transferred to VMC.
  - Gate 0 (Novelty & Adoption): FAIL. It is already used in NQS chemistry (arXiv:2507.19276, arXiv:2509.25720, arXiv:2604.06841). It is not standard because of the zero-variance loss, and this route has no mechanism against the $\tau^2_{\min}$ floor.
  - Gate 1 (Conservation & Asymptotics): FAIL. Its cost relative to exact enumeration diverges as $\psi\to\psi_0$ (sign floor).
  - Gate 2 (Baseline Gameability): FAIL. At near-converged states the deterministic part does all the work, and screening alone matches it (smoke data below: SS $0.932\,C_{\mathrm{ex}}$ vs SCREEN $0.911\,C_{\mathrm{ex}}$).
  - Gate 3 (Hard-Core Shield): FAIL. The sign floor grows with the frustrated fraction.
  - *Verdict*: KILLED
* **Route 2**: Doubly stochastic optimization. Use a $K=1$ sampled $E_L$ only inside the SR force (linear, hence unbiased) and exact $E_L$ only for final energy evaluation. Point of departure: doubly stochastic gradients in ML.
  - Gate 0 (Novelty & Adoption): PASS. No NQS-chemistry paper found that uses sampled $E_L$ only in the force with an exact final evaluation. The adoption barrier is the noise floor, addressed only by learning-rate decay.
  - Gate 1 (Conservation & Asymptotics): FAIL. The force noise $\mathbb{E}_\pi[\tau^2\,\lvert O_k\rvert^2]/K$ does not vanish as $\psi\to\psi_0$ (sign floor), and $S^{-1}$ amplifies it along small-eigenvalue directions. At fixed step size the stationary energy error is set by this floor, and removing it needs step-size decay. The cost to reach 1.6 mHa then scales with the floor, not with the vanishing exact variance.
  - Gate 2 (Baseline Gameability): not evaluated (killed at Gate 1).
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED
* **Route 3**: Factorized-Hamiltonian sampling. Write $V_{pqrs}=\sum_L L^L_{pr}L^L_{qs}$ (Cholesky / double factorization) and sample $(L,pr,qs)$ with probability proportional to factor magnitudes. Point of departure: low-rank sampling of Hamiltonian terms.
  - Gate 0 (Novelty & Adoption): PASS (marginal). Factorized integrals are standard in AFQMC and quantum algorithms. No NQS local-energy use was found.
  - Gate 1 (Conservation & Asymptotics): PASS. The estimator is unbiased and symmetry-preserving.
  - Gate 2 (Baseline Gameability): FAIL. It is just another proposal $p(\mathbf{n}'\mid\mathbf{n})$, still needs one forward pass per drawn $\mathbf{n}'$, and is bounded below by the oracle floor. In smoke data, even the oracle proposal costs $145.6\,C_{\mathrm{ex}}$ at a 0.17 mHa state (H$_6$, $R=3.0$), so exact enumeration with screening wins.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED
* **Route 4**: **Frozen-backflow control variate (SCV)**. For a determinant-based NQS $\psi(\mathbf{n})=\det\Phi(\mathbf{n})_{[\mathrm{occ}(\mathbf{n})]}$, where $\Phi(\mathbf{n})\in\mathbb{R}^{M\times N}$ are configuration-dependent backflow orbitals, define the surrogate ratio $\tilde{r}(\mathbf{n}')=\det\Phi(\mathbf{n})_{[\mathrm{occ}(\mathbf{n}')]}/\psi(\mathbf{n})$. This freezes the orbitals at the sampled $\mathbf{n}$. Every $\tilde{r}$ is an $\mathcal{O}(1)$ Green's-function determinant ratio after an $\mathcal{O}(MN^2)$ setup (Sabzevari & Sharma arXiv:1807.10633 arithmetic), so the full $\mathcal{O}(M^4)$ sum costs $\kappa$ per term with no forward passes. The estimator is
$$\hat{E}=H_{\mathbf{n}\mathbf{n}}+\sum_{\mathbf{n}'\in\mathcal{D}}x_{\mathbf{n}'}+\sum_{\mathbf{n}'\notin\mathcal{D}}H_{\mathbf{n}\mathbf{n}'}\tilde{r}(\mathbf{n}')+\frac{1}{K}\sum_{k=1}^{K}\frac{H_{\mathbf{n}\mathbf{n}'_k}\big(r(\mathbf{n}'_k)-\tilde{r}(\mathbf{n}'_k)\big)}{p(\mathbf{n}'_k)},$$
with $\mathcal{D}=\{\mathbf{n}':\lvert H_{\mathbf{n}\mathbf{n}'}\tilde{r}\rvert>\varepsilon\}$ evaluated exactly, and draws $\mathbf{n}'_k\notin\mathcal{D}$ from the defensive mixture $p=\alpha\lvert H\tilde{r}\rvert/Z_1+(1-\alpha)\lvert H\rvert/Z_2$. Point of departure: semistochastic estimators plus control variates. The deterministic part carries the sign of the network's own nodal structure at $\mathbf{n}$, so the floor applies only to the residual $x-\tilde{x}$.
  - Gate 0 (Novelty & Adoption): PASS. Semistochastic local energies exist (arXiv:2507.19276, arXiv:2509.25720, arXiv:2604.06841). Truncated local energies exist (arXiv:2502.18843). Frozen-orbital ratios are the standard Jastrow–Slater machinery (arXiv:1807.10633). Searches found no use of the network's own frozen backflow as an exactly summed control variate. The adoption barrier of semistochastic methods is the non-vanishing tail variance, and this route attacks exactly that: the residual vanishes identically when $\Phi$ is independent of $\mathbf{n}$, and is small when the backflow responds weakly to a double excitation.
  - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased by construction ($\mathbb{E}[\hat{E}\mid\mathbf{n}]=E_L$; the mixture has full support). The surrogate lives in the same $N$, $S_z$ sector, so it conserves what $\hat{H}$ conserves. In the Slater-determinant limit $\tau^2=0$ exactly and the zero-variance principle is recovered. SR forces are conditionally unbiased, and the two-replica estimator gives an unbiased variance (Section 1). Verified numerically: the residual fraction is $0$ at zero backflow and $0.006$ at backflow scale $0.01$ (unit check on H$_4$).
  - Gate 2 (Baseline Gameability): PASS (a priori). Screening must evaluate each kept $\mathbf{n}'$ at full forward-pass cost and is biased, while SCV covers every term at cost $\kappa$. The decisive test computes the screening baseline at $\lvert\mathrm{bias}\rvert\le0.16$ mHa explicitly, together with HB and SS, and the metric is the ratio to the best of them.
  - Gate 3 (Hard-Core Shield): PASS (conditional). The deterministic part uses the network's sign structure, not a mean-field reference, so nothing in the construction is limited to weak correlation. Whether the residual stays small when the multireference character is carried by the backflow is exactly what the test measures. The smoke runs give a warning (Section 3).
  - *Verdict*: SURVIVED
### 3. Top Surviving Route & Sharpest Gap
* **Mechanism**: Route 4 (SCV). Sum the local energy over all $\mathcal{O}(M^4)$ connected configurations with cheap frozen-backflow Green's-function ratios. Evaluate exactly the few terms with large $\lvert H\tilde{r}\rvert$. Estimate the residual $H(r-\tilde{r})$ with $K$ defensive importance draws, each needing one forward pass. The estimator is unbiased and SR-compatible, and it restores zero variance in the no-backflow limit.
* **Sharpest Gap**: An unproven lemma: the residual $\rho=\mathbb{E}_\pi\sum_{\mathbf{n}'}\lvert H_{\mathbf{n}\mathbf{n}'}(r-\tilde{r})\rvert/\mathbb{E}_\pi\sum_{\mathbf{n}'}\lvert H_{\mathbf{n}\mathbf{n}'}r\rvert$ should stay well below 1 for an energy-optimized NNBF in the strongly correlated regime. Smoke-scale evidence points against it: $\rho=0.56$–$0.82$ on H$_4$ and $\rho=0.69$–$0.80$ on H$_6$ at $R=3.0$ bohr, and the surrogate alone is biased by $100$–$180$ mHa. Energy optimization puts the multireference correlation into the configuration dependence of $\Phi(\mathbf{n})$, which is exactly what a frozen surrogate discards. A possible repair, untested, is an architecture whose backflow is provably smooth under double excitations (dependency on P1.2 and P2.1). The route is also native to determinant/Pfaffian NQS only: autoregressive chemistry NQS (arXiv:2208.05637, arXiv:2408.07625) have no frozen surrogate.
### 4. 24-Hour Decisive Test & Compute Execution
* **Minimal Instance**: Linear H$_8$ chain, STO-3G with analytic $s$-Gaussian integrals computed in numpy, transformed to the RHF canonical-MO basis. This is a dense all-to-all $V_{pqrs}$ with $M=16$ spin-orbitals and $N=8$, in the $S_z=0$ sector of dimension 4900. The Hard-Core instance is $R=3.0$ bohr, with $R=1.8$ bohr as the weak-correlation control and H$_6$ for the size trend. Hard-Core evidence on H$_6$ at $R=3.0$: HF weight $0.573$ in $\psi_0$, and the frustrated fraction $F=\sum\lvert H\psi_0\psi_0\rvert\,[H_{\mathbf{n}\mathbf{n}'}\psi_0(\mathbf{n})\psi_0(\mathbf{n}')>0]/\sum\lvert H\psi_0\psi_0\rvert=0.214$, so the instance is strongly multireference and non-stoquastic. A real NNBF with $\Phi(\mathbf{n})=\Phi_0+\mathrm{reshape}(W_2\tanh(W_1(2\mathbf{n}-1)+b_1))$ and hidden width 32 is optimized by L-BFGS on the exact energy (analytic gradient, checked to $5.8\times10^{-9}$ relative error). Snapshots are analyzed at iterations 50, 300, 1000 and 3000. All variances are computed exactly, analytically over $\pi$ and over the proposals, so the metric carries no Monte Carlo noise.
* **Scalar Metric $M$**: For each method $X$, $C_X=\min_{K,\varepsilon}\big[\mathrm{Var}_\pi(E_L^{X})+\mathbb{E}_\pi\tau_X^2/K\big]\,\bar{c}_X$, with $K\in\{1,\dots,256\}$, 30 thresholds $\varepsilon\in[10^{-7},1]$, cost in forward passes, $\kappa=10^{-3}$ per surrogate term, and $+1$ for the Green's-function setup. Then
$$M=\frac{\min(C_{\mathrm{SCV}},C_{\mathrm{SCVSS}})}{\min(C_{\mathrm{HB}},C_{\mathrm{SS}},C_{\mathrm{SCREEN}})}$$
on the final snapshot of H$_8$ at $R=3.0$. SCREEN is admissible only when $\lvert\mathrm{bias}\rvert\le1.6\times10^{-4}$ Ha. Reported alongside: $C_{\mathrm{SCVbest}}/C_{\mathrm{EXACT}}$, $\rho$, the surrogate-only bias, and an oracle-IS lower bound.
* **Execution Plan**: `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy/micro_test.py` and `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy/submit_test.sbatch` (partition `kim`, CPU, 4 cores, 12 GB, 8 h; expected wall time under 1 h). Command (not run): `sbatch /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy/submit_test.sbatch`. Output goes to `results_<jobid>.json` and `p22_<jobid>.out` next to the script.
* **Thresholds**: PASS if $M\le0.3$ on H$_8$ at $R=3.0$, and $M(\mathrm{H}_8)\le M(\mathrm{H}_6)$, and $C_{\mathrm{SCVbest}}/C_{\mathrm{EXACT}}<1$. FAIL if $M\ge1$ or $C_{\mathrm{SCVbest}}/C_{\mathrm{EXACT}}\ge1$. AMBIGUOUS otherwise. Pre-committed action for AMBIGUOUS: rerun with `--sizes 8 10 --bond-lengths 3.0`; PASS iff $M(\mathrm{H}_{10})\le\min(M(\mathrm{H}_8),0.5)$, else KILL the route. The script prints the verdict automatically.
* **Smoke-test evidence**: Command `/share/apps/software/anaconda3/bin/python3 micro_test.py --smoke` (H$_2$ integral check, H$_4$, gradient check, about 1 s). The FCI value differs from the textbook $-1.137284$ by $8\,\mu$Ha; the RHF value matches to $10^{-6}$. Literal output, minus the args line:

```
H2 STO-3G R=1.4: E_RHF=-1.116714 (ref -1.116714)  E_FCI=-1.137276 (ref -1.137284)

=== H4 chain R=1.4: dim=36  E_RHF=-2.098382  E_FCI=-2.139443  HF weight=0.974  frustrated frac=0.070  nnz/row=27  (0.0s)
  gradient check: max rel err = 5.80e-09
  it=   30 dE=    0.959 mHa  VarEx=2.66e-03  <nconn>=14  resid.frac=0.819  FROZEN bias=31.334 mHa  (0.0s)
     HB        C/C_exact=     31.45  K=16
     SS        C/C_exact=    0.8074  K=1 eps=0.0356
     SCREEN    C/C_exact=    0.7398  eps=0.0356 bias=2.29e-06
     SCV       C/C_exact=     6.391  K=16
     SCVSS     C/C_exact=    0.3536  K=1 eps=0.00386
     ORACLE_IS C/C_exact=     2.597  K=4
     -> M = C_SCVbest/C_baseline = 0.478;  C_SCVbest/C_EXACT = 0.3536
  [H4 R=1.4 it=150 and H4 R=3.0 blocks omitted here for length; final line:]
VERDICT: {"R_stretched": 3.0, "sizes": [4], "M_by_size": [1.0741382362864895], "M": 1.0741382362864895, "SCV_over_EXACT": 0.9503597988183358, "verdict": "FAIL"}
```

A 3 s pre-check on the small Hard-Core instance (H$_6$, $R=3.0$, hidden 16, 200 iterations; not the decisive run): command `python3 micro_test.py --sizes 6 --bond-lengths 3.0 --maxiter 200 --snapshots 40 --hidden 16`:

```
=== H6 chain R=3.0: dim=400  E_RHF=-2.675432  E_FCI=-2.957646  HF weight=0.573  frustrated frac=0.214  nnz/row=118  (0.2s)
  it=   40 dE=   13.021 mHa  VarEx=6.32e-03  <nconn>=60  resid.frac=0.685  FROZEN bias=178.237 mHa  (0.5s)
     HB        C/C_exact=     21.98  K=32
     SS        C/C_exact=    0.9239  K=1 eps=0.00672
     SCREEN    C/C_exact=    0.8967  eps=0.00672 bias=1.69e-05
     SCV       C/C_exact=     6.501  K=32
     SCVSS     C/C_exact=    0.6569  K=4 eps=0.00221
     ORACLE_IS C/C_exact=     4.485  K=16
     -> M = C_SCVbest/C_baseline = 0.7326;  C_SCVbest/C_EXACT = 0.6569
  it=  200 dE=    0.166 mHa  VarEx=2.03e-04  <nconn>=60  resid.frac=0.797  FROZEN bias=175.417 mHa  (0.5s)
     HB        C/C_exact=     656.8  K=256
     SS        C/C_exact=    0.9323  K=1 eps=0.00127
     SCREEN    C/C_exact=    0.9108  eps=0.00221 bias=-3.92e-06
     SCV       C/C_exact=     513.9  K=256
     SCVSS     C/C_exact=    0.9397  K=1 eps=4.52e-05
     ORACLE_IS C/C_exact=     145.6  K=128
     -> M = C_SCVbest/C_baseline = 1.032;  C_SCVbest/C_EXACT = 0.9397
VERDICT: {"R_stretched": 3.0, "sizes": [6], "M_by_size": [1.0317545952193892], "M": 1.0317545952193892, "SCV_over_EXACT": 0.9396934632444816, "verdict": "FAIL"}
```

Reading of the pre-check: once the state is within chemical accuracy (0.17 mHa), every unbiased stochastic estimator, the oracle included, is 145 to 657 times more expensive than exact enumeration. This is the sign floor of Section 1. SCV helps only on poorly converged states ($M=0.73$ at 13 mHa) and loses its edge at convergence ($M=1.03$). The decisive H$_8$ run decides whether the size trend reverses this.
### 5. Local Autopsy Log
* **Route 1 (heat-bath / semistochastic)**: Already in use (arXiv:2507.19276, arXiv:2509.25720, arXiv:2604.06841) with no mechanism against the tail-variance floor $\tau^2_{\min}=(\sum\lvert x\rvert)^2-(\sum x)^2$. That floor is nonzero for non-stoquastic rows, so the cost relative to exact enumeration diverges as $\psi\to\psi_0$ (measured: HB costs $657\times C_{\mathrm{ex}}$ at 0.17 mHa on H$_6$, $R=3.0$).
* **Route 2 (doubly stochastic SR force)**: The force noise inherits the sign floor and does not vanish at convergence, and $S^{-1}$ amplifies it. Fixed-step SR then stalls at a noise-set energy error, and decaying the step forfeits the speed-up.
* **Route 3 (factorized-Hamiltonian proposal)**: It is only a different $p(\mathbf{n}'\mid\mathbf{n})$, still one forward pass per draw, and bounded below by the oracle-IS floor. The oracle itself is $145.6\times C_{\mathrm{ex}}$ at 0.17 mHa, so exact enumeration with screening dominates.

### Round 2 — Mutation & Replenishment (2026-11-11)
**Binding inputs.** The critic made two Major objections. (i) The "variance floor" $(\sum_j\lvert x_j\rvert)^2-(\sum_jx_j)^2$ holds only for a single importance proposal (Kahn–Marshall; Owen's text). Positivisation (Owen & Zhou, JASA 95, 135, 2000, DOI:10.1080/01621459.2000.10473909) uses one sampler for the positive part and one for the negative part and can reach zero variance. (ii) "Surrogate control variate, exact sum plus sampled residual" is the standard semistochastic-plus-control-variate construction. Also, H$_8$/STO-3G ($M=16$, $\lvert\mathcal{C}\rvert\approx100$) cannot test a scaling claim in $M$, and the verdict rested on an assumed $\kappa=10^{-3}$. Round 2 treats both as boundary conditions. Every route below is judged on a new test that grows the basis to $M=48$ spin-orbitals and measures $\kappa$.

**What positivisation removes, and what it does not (correction to P2.2′).** Write a connected row as $x_j=H_{\mathbf{n}\mathbf{n}_j}\psi(\mathbf{n}_j)/\psi(\mathbf{n})$, where $j$ runs over the tail $T$ of terms not evaluated exactly. Let $y_j$ be any cheap surrogate of $x_j$, and split $T$ into strata $T_\pm=\{j\in T:\pm y_j>0\}$. Sample stratum $s$ with $p_s(j)=\lvert y_j\rvert/Z_s$, where $Z_s=\sum_{j\in T_s}\lvert y_j\rvert$. The single-draw variance of the stratum estimator $Z_s\,x_j/\lvert y_j\rvert$ is
$$\tau_s^2=Z_s\sum_{j\in T_s}\frac{x_j^2}{\lvert y_j\rvert}-\Big(\sum_{j\in T_s}x_j\Big)^2=Z_s^2\,\mathrm{Var}_{p_s}\!\big(x_j/\lvert y_j\rvert\big).$$
This is zero if and only if $x_j=c_s\lvert y_j\rvert$ on $T_s$ for a constant $c_s$. That requires (a) every term in a stratum to have the stratum's sign, *and* (b) the true magnitudes to be proportional to the surrogate magnitudes. The critic's claim that "the variance scales with the sign-mismatch weight" holds only for oracle magnitudes. With any cheap surrogate, the residual floor is the magnitude dispersion $\mathrm{Var}_{p_s}(x/\lvert y\rvert)$ plus the mismatch. Two diagnostics in the new test separate the parts:
* ORACLE_SIGN uses true signs with Epstein–Nesbet magnitudes.
* ISFB is a single proposal $\propto\lvert y_{\mathrm{FB}}\rvert$ with no strata.

On the pre-check (table below), ORACLE_SIGN equals PT2IS (EN magnitudes, no sign information) to within 0.3%, and POS_FB equals ISFB to within 2%. **On these instances the sign is not the binding part of the floor; the magnitude is.**

**New test reach.**
* *Instance.* H$_4$ ($N=4$ electrons) in an s-Gaussian basis built in numpy: STO-3G 1s plus $n_s-1$ uncontracted s shells per atom. This gives $K=4n_s$ spatial orbitals and $M=2K=16$–$48$ spin-orbitals. The mean connectivity $\langle\lvert\mathcal{C}\rvert\rangle$ grows from 101 to about 2500 (as $N^2(M-N)^2/4$ at fixed $N$).
* *Geometries.* Square ring and chain, both at $R=2.8$ bohr. On the ring, the HF weight is 0.44 and the frustrated fraction is 0.13.
* *State.* A real NNBF (hidden width 64) trained by L-BFGS on the exact energy to $\Delta E=0.03$–$0.5$ mHa.
* *$\kappa$, measured.* All kernels are batched numpy, timed in units of one exact NNBF amplitude of a connected configuration:
    * Hamiltonian element (Slater–Condon gather + phase), $\kappa_H$.
    * Frozen-backflow ratio, $\kappa_{\mathrm{FB}}$.
    * Epstein–Nesbet term, $\kappa_{\mathrm{EN}}$.
    * LRX ratio, $\kappa_{\mathrm{LRX}}$.
    * Per-sample setups, $\kappa_{\mathrm{S}}$ and $\kappa_{\mathrm{S,LRX}}$.
* *Decisive and sensitivity $\kappa$.* The decisive cost model times the same kernels on a chemistry-scale NNBF ($N=20$, $M=80$, hidden 256). Measured: $\kappa_H=8.2\times10^{-4}$, $\kappa_{\mathrm{FB}}=3.7\times10^{-3}$, $\kappa_{\mathrm{EN}}=2.3\times10^{-4}$, $\kappa_{\mathrm{LRX}}=0.21$, $\kappa_{\mathrm{S}}=0.13$, $\kappa_{\mathrm{S,LRX}}=12.4$; one amplitude takes 225 µs. The instance's own $\kappa$ ($N=4$: $\kappa_H\approx0.02$, $\kappa_{\mathrm{FB}}\approx0.023$, $\kappa_{\mathrm{LRX}}\approx0.55$) is re-reported as a sensitivity, and every verdict is printed under both.
* *Cost metric.* For each estimator $X$, $C_X=\big(\mathrm{Var}_\pi(E_L^X)+\mathbb{E}_\pi[\tau_X^2]/K\big)\,\bar c_X$, minimized over the number of draws $K$, the threshold $\varepsilon$, the defensive weight $\alpha$ and the P-space window $\Delta_P$. $\mathrm{Var}_\pi(E_L)$ is the exact full-space value, and all $\tau^2$ are analytic per row.
* *Controls.* Exact enumeration (EXACT); screening by $\lvert H\rvert$, $\lvert y_{\mathrm{FB}}\rvert$, $\lvert y_{\mathrm{EN}}\rvert$ and $\lvert y_{\mathrm{LRX}}\rvert$ (biased, admissible only if $\lvert\mathrm{bias}\rvert\le0.16$ mHa); deterministic surrogate *completion* of the screened tail (COMP_FB, COMP_EN, COMP_LRX; biased); heat-bath (HB); semistochastic (SS). Screening and SS are charged Hamiltonian generation only for the terms they keep, which is generous to the controls.
* *Hard Core.* The instance carries non-stoquastic sign structure and strong multireference character. Whether the competitor at $\Delta=0.016$ Ha in the $S_z=0$ sector shares the ground state's spin sector was not checked, so **the test covers only the sign half of the Hard Core**.

**Pre-check data** (ring, $R=2.8$; login-node runs of the decisive code with 600 L-BFGS iterations; $M=24$ and $32$ from `precheck_ns3_ns4_ring.log`, $M=16$ from the smoke run). The table gives $C_X/C_{\mathrm{EXACT}}$ under the decisive $\kappa$. The ratio-free columns give the residual fraction $\rho_y=\mathbb{E}_\pi\sum_j\lvert x_j-y_j\rvert/\mathbb{E}_\pi\sum_j\lvert x_j\rvert$ and the sign-mismatch weight $\mu_y$, the $\lvert x\rvert$-weighted fraction of terms with $\mathrm{sgn}\,y_j\ne\mathrm{sgn}\,x_j$.

| $M$ | $\Delta E$ (mHa) | $\langle\lvert\mathcal{C}\rvert\rangle$ | best non-SCV control | SCV (R2.0) | POS_FB (R2.1) | PT2IS (R2.2) | SCV_LRX (R2.3) | MLMC (R2.4) | SCV_ADA (R2.5) | $\rho_{\mathrm{FB}}$, $\rho_{\mathrm{LRX}}$, $\rho_{\mathrm{EN}}$ | $\mu_{\mathrm{FB}}$, $\mu_{\mathrm{LRX}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 16 | 0.52 | 101 | 0.501 (COMP_EN) | 0.451 | 0.440 | 0.533 | 0.572 | 0.489 | 0.458 | 0.79, 0.10, 0.88 | 0.12, 0.005 |
| 24 | 0.030 | 270 | 0.461 (COMP_EN) | 0.270 | 0.262 | 0.480 | 0.412 | 0.279 | 0.271 | 0.48, 0.04, 0.88 | 0.17, 0.002 |
| 32 | 0.25 | 518 | 0.174 (COMP_EN) | 0.104 | 0.104 | 0.293 | 0.281 | 0.112 | 0.101 | 0.39, 0.03, 0.89 | 0.04, 0.001 |

Reading:
* The frozen-backflow control variate (SCV) costs 0.45, 0.27 and 0.10 of exact enumeration as $M$ goes from 16 to 32, so $C_{\mathrm{SCV}}/C_{\mathrm{EXACT}}$ falls with $M$ under the measured $\kappa$. Round 1 never reached this regime.
* SCV beats every non-SCV control by a factor of 1.1, 1.7 and 1.7 across the three sizes.
* The strongest control is not screening. It is COMP_EN, the screened sum plus a deterministic per-sample Epstein–Nesbet completion of the tail. COMP_EN is biased, and at $M=32$ its bias, $1.5\times10^{-4}$ Ha, is at the admissibility limit.
* Positivisation, multilevel and adaptive allocation each land within 7% of plain SCV.

* **Generation Batch 2**:
  - **Route 2.0 (Route 1.4 revived under the Stage 4 Autopsy & Re-Loop rule: the round-1 kill was a measurement at $M\le16$ with an assumed $\kappa$, which the corrected test removes)**: frozen-backflow semistochastic control variate (SCV). Mechanism as in round 1. The deterministic set is $\mathcal{D}=\{\lvert y_{\mathrm{FB}}\rvert>\varepsilon\}$. The sum $\sum_{\mathcal{C}\setminus\mathcal{D}}y_{\mathrm{FB}}$ is evaluated in full at $\kappa_{\mathrm{FB}}$ per term. The residual $x-y_{\mathrm{FB}}$ gets $K$ draws from $p\propto\alpha\lvert y_{\mathrm{FB}}\rvert+(1-\alpha)\lvert H\rvert$.
    - Gate 0 (Novelty & Adoption): PASS (narrow; the critic's objection is answered, not dismissed). The generic construction is textbook Monte Carlo, as is positivisation, which the critic proposed. The NQS question is why practice does not use it.
        - Searches found no control variate in any NQS local energy. Practice splits two ways. One branch is biased truncation with a sign-blind proxy: Liu & Clark, arXiv:2502.18843 (target-space truncation; Gumbel top-$k$ weights, "not fully unbiased"), and arXiv:2603.14775 (proxy $\max_i\lvert\psi(x_i)H_{ij}\rvert$, pruned terms dropped). The other branch is semistochastic tails with no control variate (arXiv:2507.19276, 2509.25720, 2604.06841).
        - The adoption barrier is that unbiased tails are noisier than the truncation bias. The new mechanism that removes it is quantitative and measured. At chemistry-scale $\kappa_{\mathrm{FB}}=3.7\times10^{-3}$, the full surrogate sum costs less than one amplitude per 270 terms, and the residual variance stays below the screening cost at every $M$ tested.
        - The construction is narrowly a determinant/Pfaffian NQS tool (no frozen surrogate for autoregressive NQS). That restriction is kept as a limitation, not hidden.
    - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased, has full support, conserves $N$ and $S_z$, and keeps the SR force conditionally unbiased (round-1 Section 1). The cost floor is $C/C_{\mathrm{EXACT}}\ge(\kappa_H+\kappa_{\mathrm{FB}})/(1+\kappa_H)\approx4.5\times10^{-3}$ at chemistry scale, so the claim is "$\to$ a small constant", not "$\to0$". Reaching $\to0$ needs the $\mathcal{O}(N_\gamma MN)$ Cholesky contraction of the frozen sum (Route 2.8).
    - Gate 2 (Baseline Gameability): PASS on the pre-check. $M_{\mathrm{R2.0}}=C_{\mathrm{SCV}}/\min(\text{non-SCV controls})=0.90$, 0.59 and 0.60 at $M=16$, 24 and 32, against controls that include surrogate screening and EN completion.
    - Gate 3 (Hard-Core Shield): PASS (conditional). The surrogate carries the network's own nodal structure. $\rho_{\mathrm{FB}}$ falls from 0.79 to 0.39 as $M$ grows, because the added orbitals are weakly backflow-coupled. The risk sits in $N$, not $M$: at $N=4$ only two of four rows change per double excitation, and the frozen error at chemistry $N$ is unmeasured.
    - *Verdict*: SURVIVED.
  - **Route 2.1 (P2.2′ from the critic pass; mutated from Route 1.4 via Operation A.1: the adoption wall "tail variance from cancellation" attacked by stratifying the tail by the surrogate sign)**: positivised semistochastic local energy. Strata come from $\mathrm{sgn}\,y_{\mathrm{FB}}$ (POS_FB) or from $\mathrm{sgn}\,y_{\mathrm{FB}}$ with EN magnitudes (POS_HYB). Each stratum gets proposal $\propto\alpha\lvert y\rvert+(1-\alpha)\lvert H\rvert$, an integer allocation $K_s\propto$ stratum mass, and at least one draw if the stratum is non-empty.
    - Gate 0 (Novelty & Adoption): PASS. Searches for positivisation or sign-split sampling in VMC, NQS and FCIQMC local energies returned nothing; the closest NQS work is sign-blind (above).
    - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased and stratified with full support. Variance is zero if and only if $x_j=c_s\lvert y_j\rvert$ on each stratum.
    - Gate 2 (Baseline Gameability): **FAIL (measured)**. Its own single-proposal ablation ISFB matches it to within 1–3% at every $M$: 0.440 vs 0.453, 0.262 vs 0.264, 0.1035 vs 0.1053. ORACLE_SIGN, with *true* signs, does no better than having no sign information. The sign-mismatch weight is $\mu_{\mathrm{FB}}=0.04$–$0.17$, while the magnitude dispersion ($\rho_{\mathrm{FB}}=0.39$–$0.79$) carries the floor.
    - Gate 3 (Hard-Core Shield): not reached.
    - *Verdict*: KILLED. P2.2′ is closed: on these instances positivisation removes only the sign part of the floor, and that part is small.
  - **Route 2.2 (Operation B.3, cross-field translation from selected-CI perturbation theory)**: P/Q-partitioned local energy with an Epstein–Nesbet tail proposal (PT2IS). The exact P-space is $\{\lvert y_{\mathrm{EN}}\rvert>\varepsilon\}\cup\{H_{jj}<E+\Delta_P\}$ with $y_{\mathrm{EN},j}=H_{\mathbf{n}j}^2/\min(E-H_{jj},-\delta)$. The Q tail is sampled $\propto\lvert y_{\mathrm{EN}}\rvert$. The proposal is Hamiltonian-only and $\mathcal{O}(1)$ per term through the diagonal update, $H_{jj}-H_{\mathbf{n}\mathbf{n}}=f_a+f_b-f_i-f_j+G_{ij}+G_{ab}-G_{ai}-G_{aj}-G_{bi}-G_{bj}$, with $f_p=h_{pp}+\sum_{k\in\mathrm{occ}}G_{pk}$ and $G_{pq}=\langle pq\Vert pq\rangle$. That formula is verified to $2\times10^{-14}$.
    - Gate 0 (Novelty & Adoption): PASS. EN-PT2 enters NQS only as a global post-hoc correction (arXiv:2404.09280, appendix; arXiv:2601.21310, deterministic NNBF plus EN-PT2). It has not been used as a per-sample local-energy proposal. The route is architecture-agnostic, which fixes round 1's "determinant-only" gap.
    - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased. The asymptotic argument at fixed $N$: high-virtual tail terms are sign-coherent ($x_j\approx H_{\mathbf{n}j}^2/(E-H_{jj})<0$), so the tail mass converges with the basis while $\lvert\mathcal{C}\rvert$ grows.
    - Gate 2 (Baseline Gameability): PASS a priori; the test includes the deterministic EN completion and EN screening.
    - Gate 3 (Hard-Core Shield): **FAIL (measured)**. The interference ratio $\eta_j=\sum_{m\ne\mathbf{n}}H_{jm}\psi(m)/(H_{j\mathbf{n}}\psi(\mathbf{n}))$ is $\mathcal{O}(1)$ in the multireference regime, so $\rho_{\mathrm{EN}}=0.88$–$0.89$ at every $M$. $M_{\mathrm{R2.2}}$ grows from 1.18 to 1.78 to 2.82 with $M$.
    - *Verdict*: KILLED. The same EN magnitudes are worth more as a deterministic, biased completion (COMP_EN, the strongest control) than as an unbiased proposal.
  - **Route 2.3 (Mutated from Route 1.4 via Operation A, on the exact sub-component that failed: the frozen surrogate)**: the LRX surrogate. For a single or double excitation $i,j\to a,b$:
    - Form the two new rows of $\Phi(\mathbf{n}')$ exactly, with the exact hidden update $\tanh(z+W_1\Delta x)$ at $\mathcal{O}(Nh)$ each, where $z$ is the hidden pre-activation and $h$ the hidden width. Put them into a $2\times2$ minor of $\Phi_{\mathrm{new}}A^{-1}$, where $A=\Phi(\mathbf{n})[\mathrm{occ}\,\mathbf{n}]$.
    - Linearize the response $\Delta$ of the $N-2$ unchanged rows in adjugate form, $\det(B+\Delta)\approx\det B\,(1+\sum_{k}Z_{kk}\cdot\delta h)+\mathrm{tr}\big(\mathrm{adj}(W)\,\cdots\big)$. Here $B$ is $A$ with the two new rows inserted, $W$ is the $2\times2$ block of $G_j=BA^{-1}$, $Z[m,c,h]=\sum_l W_2[\mathrm{occ}_m,l,h]A^{-1}_{lc}$, and $\delta h$ is the change of the hidden layer. The adjugate form avoids dividing by $\det B$, so the error stays $\mathcal{O}(\lvert\Delta\rvert^2)$ near nodes of $B$.
    - Verified: with the backflow scaled by $10^{-2}$ and $10^{-3}$, the frozen error is $1.5$ and $0.15$, while the LRX error is $9.4\times10^{-3}$ and $9.4\times10^{-5}$.
    - Gate 0 (Novelty & Adoption): PASS. Exact low-rank amplitude updates (lrux, arXiv:2602.05255; SCALE, arXiv:2604.25775) need low-rank or local backflow. For unrestricted backflow the P2.1 bound gives $\Omega(N^2)$ work per hop. LRX does not need exactness, because the sampled residual restores unbiasedness. Searches found no linearized-backflow surrogate used as a control variate.
    - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased as a control variate and costs $\mathcal{O}(Nh)$ per term against $\mathcal{O}(N^2h+N^3)$ per amplitude.
    - Gate 2 (Baseline Gameability): **FAIL (measured, on cost)**. It is the best surrogate tested, with $\rho_{\mathrm{LRX}}=0.03$–$0.10$ and $\mu_{\mathrm{LRX}}=0.001$–$0.005$. But summing it over all of $\mathcal{C}$ costs $\kappa_{\mathrm{LRX}}\lvert\mathcal{C}\rvert=0.21\lvert\mathcal{C}\rvert$, so SCV beats it by 1.5–2.7.
    - *Verdict*: KILLED as a stand-alone sum; mutated into Route 2.4, which evaluates LRX only on drawn terms.
  - **Route 2.4 (Operation B.3, cross-field translation of multilevel Monte Carlo, Giles, arXiv:1304.5472, to surrogate fidelity levels instead of discretization levels)**: three-level estimator.
    $$\hat{E}=H_{\mathbf{n}\mathbf{n}}+\sum_{\mathcal{D}}x+\sum_{T}y_{\mathrm{FB}}+\frac{1}{K_1}\sum_{k=1}^{K_1}\frac{(y_{\mathrm{LRX}}-y_{\mathrm{FB}})_{j_k}}{p(j_k)}+\frac{1}{K_2}\sum_{k=1}^{K_2}\frac{(x-y_{\mathrm{LRX}})_{j'_k}}{p(j'_k)}.$$
    Here $T$ is the tail, and $p\propto\alpha\lvert y_{\mathrm{FB}}\rvert+(1-\alpha)\lvert H\rvert$ for both corrections, so $y_{\mathrm{LRX}}$ is never needed outside the drawn terms. The variance is $\tau_1^2/K_1+\tau_2^2/K_2$ and the cost is $K_1\kappa_{\mathrm{LRX}}+\kappa_{\mathrm{S,LRX}}+K_2$.
    - Gate 0 (Novelty & Adoption): PASS. Searches found no multilevel Monte Carlo in VMC local energies, nor a basis-set or fidelity hierarchy for them.
    - Gate 1 (Conservation & Asymptotics): PASS. The sum telescopes, so the estimator is unbiased.
    - Gate 2 (Baseline Gameability): PASS (conditional). It must beat SCV, which is now a control. Pre-check: $M_{\mathrm{R2.4}}=1.08$, 1.03 and 1.07. The LRX setup $Z$ costs $\kappa_{\mathrm{S,LRX}}=12.4$ amplitudes per sample, and at $N=4$ the per-term saving of LRX over an exact amplitude is small. Both improve with $N$ (setup amortized over $\lvert\mathcal{C}\rvert\propto N^2M^2$; per-term ratio $\sim5/N$).
    - Gate 3 (Hard-Core Shield): PASS (conditional; $\rho_{\mathrm{LRX}}\le0.10$ on the multireference ring).
    - *Verdict*: SURVIVED (the pre-check leans AMBIGUOUS at $N=4$).
  - **Route 2.5 (Operation B.2, active probing: allocate amplitude evaluations across samples, not only within a row)**: cost-aware cross-sample allocation. Sample $\mathbf{n}$ gets $K_{\mathbf{n}}=\max(1,\mathrm{round}(K\,Z_{\mathbf{n}}/\mathbb{E}_\pi Z))$ residual draws, where $Z_{\mathbf{n}}=\sum_{T}\lvert y_{\mathrm{FB}}\rvert$ is known before any amplitude is evaluated. Neyman allocation is $K_{\mathbf{n}}\propto\tau_{\mathbf{n}}$, and $Z_{\mathbf{n}}$ is its cheap proxy. A deterministic allocation given $\mathbf{n}$ keeps the estimator unbiased.
    - Gate 0 (Novelty & Adoption): PASS. Searches found no per-sample allocation of local-energy draws in NQS; adaptive importance sampling for NQS (arXiv:2507.05352) concerns the distribution of $\mathbf{n}$, not of $\mathbf{n}'$.
    - Gate 1 (Conservation & Asymptotics): PASS. It is unbiased; the gain is bounded by $\mathbb{E}[\tau^2]/(\mathbb{E}\tau)^2$.
    - Gate 2 (Baseline Gameability): PASS (conditional). Pre-check: $M_{\mathrm{R2.5}}=1.02$, 1.00 and 0.98, so it gains only where $\tau_{\mathbf{n}}$ is heavy-tailed across samples. The combined R2.4+R2.5 is also reported.
    - Gate 3 (Hard-Core Shield): PASS. The allocation is sign-agnostic.
    - *Verdict*: SURVIVED (the pre-check leans AMBIGUOUS).
  - **Route 2.6 (Operation B.1, frame transformation: active-space-restricted backflow input, so that excitations outside the active set have exact cheap ratios and $E_L$ stays exact)**:
    - Gate 3 (Hard-Core Shield): FAIL. For a fixed active configuration the state is a single determinant in the external orbitals. It has no pair correlation into external virtuals, which dominates the correlation energy as the basis grows, so in that space it is the foil.
    - Gate 0 (Novelty & Adoption): FAIL as well. Active-space NNBF with orbital rotations exists (arXiv:2302.11588).
    - *Verdict*: KILLED.
  - **Route 2.7 (Operation B.3, randomized linear algebra: sketch the row, $E_L-H_{\mathbf{n}\mathbf{n}}=\langle h,r\rangle\approx\langle Sh,Sr\rangle$)**:
    - Gate 1 (Conservation & Asymptotics): FAIL. A dense sketch $Sr$ needs every amplitude, and a sparse sketch is single-proposal sampling, which is the global anti-pattern.
    - *Verdict*: KILLED.
  - **Route 2.8 (Operation A on Route 1.3: exact Cholesky/THC contraction of the frozen surrogate sum, $\sum_{pqrs}V_{pqrs}(G_{pr}G_{qs}-G_{ps}G_{qr})$ in $\mathcal{O}(N_\gamma MN)$, where $N_\gamma$ is the number of Cholesky vectors)**:
    - Gate 0 (Novelty & Adoption): FAIL as a stand-alone route. It is the standard AFQMC local energy (Motta & Zhang, arXiv:1711.02242).
    - *Verdict*: KILLED as a route; kept as the implementation of R2.0's level-0 sum that would remove the $\kappa_{\mathrm{FB}}\lvert\mathcal{C}\rvert$ floor.
  - **Route 2.9 (Operation B.3: multilevel Monte Carlo across basis-set levels)**:
    - Gates 0 and 2: FAIL. The coarse-basis local energy of a fine-basis NQS is a truncation of $\mathcal{C}$, i.e. semistochastic with an orbital-energy deterministic set, which the SS arm already covers.
    - *Verdict*: KILLED.
### Top Surviving Route & Sharpest Gap (Round 2)
* **Survivors**:
    * R2.0, the revived frozen-backflow control variate: the cheapest unbiased estimator measured, and the only one whose cost relative to exact enumeration falls steeply with $M$.
    * R2.4, the multilevel surrogate hierarchy FB→LRX→exact. It is not the round-1 failure because the multireference residual is carried by the LRX level, with $\rho_{\mathrm{LRX}}\le0.10$ against $\rho_{\mathrm{FB}}\le0.79$.
    * R2.5, cost-aware cross-sample allocation.
* **Top route**: R2.0. It is the only survivor with a pre-check margin ($M_{\mathrm{R2.0}}=0.59$–$0.60$ at $M=24$–$32$). R2.4 and R2.5 are refinements, judged against R2.0 and currently within 7% of it.
* **Sharpest gap (R2.0)**: two things are unmeasured.
    * Does $M_{\mathrm{R2.0}}$ keep falling to $M=48$ on both geometries? Between $M=24$ and $M=32$ it was flat (0.59 to 0.60), because the EN completion control improved as fast as SCV did.
    * Does $\rho_{\mathrm{FB}}$ stay below about 0.5 at chemistry $N$? The variance structure comes from $N=4$, where a double excitation leaves half the rows unchanged; only $\kappa$ is measured at $N=20$.
    * Remaining Gate 0 exposure: a referee may still call R2.0 standard. The defence is the measured regime, not the construction.
### 24-Hour Decisive Test (Round 2)
* **Instance**:
    * H$_4$, $n_s=2,3,4,5,6$ ($M=16$–$48$, Hilbert dimension up to 76 176, $\langle\lvert\mathcal{C}\rvert\rangle$ up to about 2500).
    * Array task 0 is the square ring at $R=2.8$ bohr; task 1 is the chain at $R=2.8$ bohr.
    * NNBF with hidden width 64, 2000 L-BFGS iterations, snapshots at 200 and 2000.
    * Rows: all configurations up to dimension 15 000; above that, the top-weight rows holding $1-10^{-3}$ of the mass plus 3000 defensive draws $q\propto\sqrt{\pi}$ with importance weights.
    * The Hard Core is covered only on its sign half.
* **Scalar metric**: $M_X(n_s)=C_X/\min_{c\in\mathrm{controls}}C_c$ at the final snapshot.
    * Controls are EXACT, SCREEN_H, SCREEN_FB, SCREEN_EN, SCREEN_LRX, COMP_FB, COMP_EN, COMP_LRX, HB and SS. SCVSS is added as a control for every route except R2.0.
    * Decisive $\kappa$ is measured at chemistry scale; instance $\kappa$ is the sensitivity.
    * Diagnostics: ORACLE_SIGN, ISFB, ORACLE_IS, $\rho_y$, $\mu_y$.
* **Thresholds** (per route, per geometry):
    * PASS if $M_X(n_s{=}6)\le0.5$, $M_X(6)<M_X(2)$, and $\Delta E(6)\le10$ mHa.
    * FAIL if $M_X(6)\ge1$.
    * AMBIGUOUS otherwise. Pre-committed action: rerun with `--ns 6 7 --hidden 64`; PASS if and only if $M_X(7)\le\min(M_X(6),0.5)$, else KILL.
    * Global verdict: PASS only if both geometries PASS; FAIL if either FAILs.
* **Execution**:
    * Code: `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r2/micro_test.py` and `submit_test.sbatch` (partition `kim`, array 0–1, 8 CPUs, 48 GB, 24 h limit).
    * Measured cost: the smallest real instance takes 54 s with the decisive settings. The expected total is 2–3 h per task.
    * Command (not run): `sbatch /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r2/submit_test.sbatch`.
    * Read the verdict with `python3 micro_test.py --aggregate <test dir>`.
* **Smoke test**: command `/share/apps/software/anaconda3/bin/python3 micro_test.py --smoke` (9.3 s). It runs these checks:
    * H$_2$ reference energies.
    * The Slater–Condon builder against the round-1 independent $E_{pq}$ builder, elementwise.
    * The EN diagonal update.
    * The frozen $2\times2$ ratio.
    * The fast amplitude kernel.
    * The LRX error order.
    * The NNBF gradient.

  It then runs $n_s=1,2$ with a tiny network and chemistry-scale $\kappa$ timed at $N=8$, $M=32$. Literal output (head and the $n_s=2$ block; full output in `smoke_output.txt`):

```
H2 STO-3G R=1.4: E_RHF=-1.116714 (ref -1.116714)  E_FCI=-1.137276 (ref -1.137284)
H4 ring ns=2 (K=8, dim=784): max |H_SC - H_Epq| = 6.63e-14
  EN diagonal update err=1.93e-14  frozen-ratio err=2.66e-15  fast-forward err=1.78e-15
  backflow x1e-2: |FB-exact|=1.50e+00 |LRX-exact|=9.37e-03;  x1e-3: |FB-exact|=1.52e-01 |LRX-exact|=9.42e-05
  gradient check: max rel err = 2.09e-10   (0.4s)
chemistry-scale kappa (N=8, M=32, hidden=64, 4000 terms): kH=0.01 kFB=0.00989 kEN=0.00503 kLRX=0.266 kSetup=1.26 kSetupLRX=4.47 t_fwd=10.23 us (0.3s)
=== H4 ring R=2.8 ns=2: K=8 M=16 dim=784 E_RHF=-1.878565 E_FCI=-2.038176 Ecorr=-159.6 mHa gap(S_z=0)=0.0169 HF wt=0.436 frustr=0.129 nnz/row=101 (build 0.2s)
  it=  150 dE=   0.522 mHa VarEx=7.89e-04 VarRows=7.89e-04 <|C|>=101 rows=784 ELchk=8.0e-13 fwdchk=5.3e-16 (opt 0.2s, total 7.8s)
     decisive kappa (chemistry scale): kH=0.01 kFB=0.00989 kEN=0.00503 kLRX=0.266 kSetup=1.26 kSetupLRX=4.47 (t_fwd=10.23 us/amp)
     instance kappa: kH=0.147 kFB=0.179 kEN=0.127 kLRX=3.99 kSetup=2.61 kSetupLRX=3.27 (t_fwd=0.37 us/amp)
     sign mismatch: FB=0.119 LRX=0.005 EN=0.128  resid frac: FB=0.786 LRX=0.098 EN=0.883
     EXACT       C/C_exact=         1  
     SCREEN_H    C/C_exact=    0.5184  eps=0.00316 nD=52 bias=3.56e-05
     COMP_EN     C/C_exact=    0.5011  eps=0.0001 delta_P=0 nD=46.1 bias=0.000111
     SS          C/C_exact=    0.5287  K=1 eps=0.00316 alpha=0 nD=52
     SCVSS       C/C_exact=    0.4511  K=4 eps=0.001 alpha=0.5 nD=31.3
     POS_FB      C/C_exact=    0.4404  K=4 eps=0.001 alpha=0.5 nD=31.3
     PT2IS       C/C_exact=    0.5331  K=4 eps=0.0001 alpha=0.5 delta_P=0 nD=46.1
     SCV_LRX     C/C_exact=    0.5716  K=4 eps=0.00316 alpha=0.5 nD=15.2
     MLMC        C/C_exact=    0.4892  K=1 K1=16 eps=0.001 alpha=0.5 nD=31.3
     SCV_ADA     C/C_exact=    0.4582  K=4 eps=0.001 alpha=0.5 nD=31.3
     ISFB        C/C_exact=    0.4531  K=4 eps=0.001 alpha=0.5 nD=31.3
     ORACLE_SIGN C/C_exact=    0.5297  K=1 eps=0.0001 alpha=0.5 delta_P=0 nD=46.1
     -> R2.0 (revived SCV) vs best non-SCV control (COMP_EN): M_R2.0 = 0.9003
     -> control best = SCVSS;  M_R2.1 = 0.9761;  M_R2.2 = 1.182;  M_R2.3 = 1.267;  M_R2.4 = 1.084;  M_R2.5 = 1.016;  M_(R2.4+R2.5) = 1.09
     [sensitivity, instance kappa] M_R2.0=1.371 ctrl=SCREEN_H M_R2.1=1.361 M_R2.2=1.349 M_R2.3=7.635 M_R2.4=1.521 M_R2.5=1.364 M_(R2.4+R2.5)=1.521
real	0m9.320s
```

  (Lines for the other arms are omitted here; they are in `smoke_output.txt`. The $\kappa$ values are wall-clock timings, so their digits, and the cost ratios that use them, vary slightly between runs; `results_smoke.json` comes from a later rerun.) The $M=24$ and $M=32$ pre-check, `precheck_ns3_ns4_ring.log`, was run with the decisive code at 600 iterations. Its own printed verdict (2 sizes only, not the decisive verdict) was: R2.0 AMBIGUOUS ($0.586\to0.596$), R2.1 AMBIGUOUS, R2.2 FAIL, R2.3 FAIL, R2.4 FAIL ($1.03\to1.07$), R2.5 AMBIGUOUS ($1.00\to0.98$).
### Local Autopsy Log (Round 2 additions)
* **Route 1.4 (frozen-backflow control variate), revived as R2.0.** The round-1 kill rested on $M\le16$ instances and an assumed $\kappa$. Under the measured chemistry-scale $\kappa$, its cost relative to exact enumeration falls with $M$, from 0.45 to 0.27 to 0.10, and it leads every non-SCV control by a factor of 1.7 at $M=24$–$32$. The critic's novelty objection stays open as a Gate 0 risk.
* **Route 2.1 (P2.2′, positivisation)**: killed at Gate 2 by measurement.
    * Stratifying by surrogate sign removes only the sign part of the floor $Z_s^2\mathrm{Var}_{p_s}(x/\lvert y\rvert)$.
    * The sign-mismatch weight is 0.04–0.17, and true-sign strata (ORACLE_SIGN) do no better than none.
    * Positivisation equals single-proposal surrogate sampling to within 3%. The floor is magnitude dispersion.
* **Route 2.2 (Epstein–Nesbet tail proposal)**: killed at Gate 3 by measurement. In the multireference regime the interference of other parents, $\eta=\mathcal{O}(1)$, gives $\rho_{\mathrm{EN}}\approx0.88$ at every $M$, and $M_{\mathrm{R2.2}}$ grows with $M$ (1.18, 1.78, 2.82). Its magnitudes survive only as a biased deterministic completion (COMP_EN), now the strongest control.
* **Route 2.3 (LRX summed over all terms)**: killed at Gate 2 on cost ($\kappa_{\mathrm{LRX}}=0.21$ at chemistry scale). It is the best surrogate measured ($\rho\le0.10$), so it was mutated into Route 2.4.
* **Routes 2.6–2.9**:
    * 2.6, active-space backflow: no external pair correlation (Gate 3); prior art (Gate 0).
    * 2.7, row sketching: needs all amplitudes or reduces to single-proposal sampling (Gate 1).
    * 2.8, Cholesky contraction: the standard AFQMC local energy (Gate 0); kept as an implementation detail.
    * 2.9, multilevel Monte Carlo over basis levels: semistochastic in disguise (Gates 0 and 2).
* **Boundary condition for the next generation**: on these instances, any estimator built on a single cheap surrogate is limited by the surrogate's magnitude error, not its sign error. The deterministic EN completion is a biased but strong competitor that every future unbiased route must beat.

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
    - Gate 2 (Baseline Gameability): **FAIL on its own pre-check (measured, $N=6$).** At $K=12$, $M_{\mathrm{R3.1}}=1.32$ under wall-clock cost and $1.33$ under operation-count cost (one seed, 400 iterations).
        * The own-state cost of the exact keyed local energy falls with $K$: $1.00\to0.82\to0.66$ of exact enumeration by operation count, because the number of keys stays at 127–139 while $\langle\lvert\mathcal{C}\rvert\rangle$ grows from 134 to 416.
        * The gain is lost to the cost of the restriction. $\Delta E$ is $2.20$ mHa against $1.73$ mHa for NNBFJ, and $\mathrm{Var}(E_L)$ is $1.4\times$ larger.
        * Shrinking the window (4–5 valence MOs) makes it worse: $\Delta E=3.4$–$4.6$ mHa and $M=1.6$.
        * Carried to the decisive scaling test only as a **conditional candidate**, not a survivor.
    - Gate 3 (Hard-Core Shield): PASS (conditional). The multireference character sits in the valence window, which keeps full backflow. External orbitals are correlated only through valence-dressed orbitals and the Jastrow, so the external sign structure is mean-field-like by construction. The test charges this through $\Delta E$ and $\mathrm{Var}(E_L)$ against the unrestricted NNBF-J of the same seed. Coverage is the sign half only.
    - *Verdict*: NOT A SURVIVOR (Gates 0, 1 and 3 pass; Gate 2 fails on the pre-check at $K\le12$). The decisive test decides whether $K=14$–$16$ and $N=8$ reverse this.
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
    - Gate 2 (Baseline Gameability): **FAIL on its own pre-check (measured, $N=6$).** $M_{\mathrm{R3.2}}=1.65$, $1.43$ and $1.23$ (wall-clock) and $1.28$, $1.85$ and $1.04$ (operation count) at $K=8$, $10$ and $12$.
        * The residual is not the problem. The variance increment over $\mathrm{Var}_\pi(E_L)$ is about 1–2% at the selected setting, and no heavy tail was found (top-1% share 0.25–0.35).
        * The cost is the deterministic part: one backflow evaluation per key, a floor of about $N_{\mathrm{key}}/\lvert\mathcal{C}\rvert=0.33$ at $K=12$.
        * Against it stands semistochastic sampling, which on these realistically converged states costs $0.55$–$0.70$ of exact enumeration.
    - Gate 3 (Hard-Core Shield): PASS (conditional). Nothing is restricted; the residual is sampled. Coverage is the sign half only.
    - *Verdict*: NOT A SURVIVOR (Gates 0, 1 and 3 pass; Gate 2 fails on the pre-check at $K\le12$, though $M$ falls toward 1). Carried to the decisive scaling test as a conditional candidate.

**Pre-check on the decisive instance** (H$_6$ ring, $R=2.8$ bohr, 6-31G**, $N=6$, one seed, 400 L-BFGS iterations, 400 samples; `precheck_N6_K8-12.log`).
* HF weight 0.84 and frustrated fraction 0.144–0.148 at every $K$.
* Each cell gives $C/C_{\mathrm{EXACT}}$ on the arm's own state, under wall-clock and operation-count cost.
* $M$ is the verdict ratio against the best control on the NNBFJ state.

| $K$ ($M$) | p-MOs | $\langle\lvert\mathcal{C}\rvert\rangle$ / keys | $\Delta E$ NNBFJ / WIBJ (mHa) | $\mathrm{Var}(E_L)$ NNBFJ / WIBJ | KEYX on WIBJ, own state | best control on NNBFJ | PFCV on NNBFJ, own state | $M_{\mathrm{R3.1}}$ | $M_{\mathrm{R3.2}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 8 (16) | 2 | 134 / 127 | 0.304 / 0.325 | $1.10$ / $1.19\times10^{-3}$ | 1.25 / 1.00 | 0.94 / 0.93 (SCREEN) | 1.55 / 1.19 | 1.38 / 1.16 | 1.65 / 1.28 |
| 10 (20) | 4 | 236 / 130 | 0.589 / 0.895 | $2.22$ / $3.47\times10^{-3}$ | 1.08 / 0.82 | 0.91 (SCREEN) / 0.55 (SS) | 1.31 / 1.01 | 1.40 / 2.35 | 1.43 / 1.85 |
| 12 (24) | 6 | 416 / 139 | 1.729 / 2.197 | $6.19$ / $8.76\times10^{-3}$ | 0.93 / 0.66 | 0.93 (SCREEN) / 0.70 (SS) | 1.15 / 0.73 | 1.32 / 1.33 | 1.23 / 1.04 |

Each cell pair is wall-clock / operation count. At $K=12$ with windows of 5 and 4 MOs, WIBJ has $\Delta E=3.38$ and $4.60$ mHa, its own-state cost by operation count is $0.53$ and $0.40$, and $M_{\mathrm{R3.1}}=1.60$ and $1.63$ (`precheck_N6_K12_W{4,5}.log`).

Reading:
* **Neither route passes Gate 2 on its pre-check**, under either cost model, at any $K\le12$. In the critic's terms this is the R2.4/R2.5 situation, and it is reported as such. Neither route is listed as a survivor.
* **What did change**: on these realistically converged states ($\Delta E\approx0.3$–$1.7$ mHa, $\mathrm{Var}(E_L)\sim10^{-3}$), the sampled residual is no longer the obstacle. PFCV's variance increment is about 1–2%. The cost is now the deterministic work, and it scales with the number of window keys, not with $\lvert\mathcal{C}\rvert$.
* The ratio $N_{\mathrm{key}}/\lvert\mathcal{C}\rvert$ at the HF row is $0.33$ at $K=12$ and $0.28$ at $K=16$ for $N=6$. For $N=8$ it is $0.55$ at $K=12$. Ring symmetry zeros out many terms, so $\lvert\mathcal{C}\rvert$ grows slowly at ED-feasible $K$. This is why a pass at $M\le0.5$ looks unlikely within the decisive sizes.
* The scaling argument ($N_{\mathrm{key}}$ fixed by the window, $\lvert\mathcal{C}\rvert\propto M^2$) predicts a ratio of about $(W/M)^2$ at chemistry scale. That is unmeasured here and is not used in any verdict.
* **The controls are strong.** Semistochastic sampling with all large terms exact costs $0.55$–$0.70$ of exact enumeration on these states. Screening costs $0.91$–$0.94$, at a bias of about $0.01$ mHa.

### Top Surviving Route & Sharpest Gap (Round 3)
* **Survivors**: none passes all four gates. The Status line reflects this; under the critic's rule (R2.4/R2.5), a route that fails its own pre-check is not listed as a survivor.
* **Conditional candidates carried to the decisive test**:
    * **R3.2** (PFCV on the unrestricted NNBF-J): closest to passing ($M=1.04$ by operation count at $K=12$, falling). Paired on the same state as the controls, so no accuracy confound.
    * **R3.1** (WIBJ, exact keyed local energy): the own-state cost falls fastest ($0.66$ at $K=12$), but the restricted ansatz loses accuracy.
    * They answer the critic's objections as follows. Neither revives R2.0, and both are new mechanisms: keyed backflow evaluation, and freezing network *inputs* rather than orbitals. Every number is from the same instance at the same $N$, under two cost models. $N=6$–$8$ with p functions. Three seeds. Genuine sampled arms with a heavy-tail admissibility rule. The pre-check gates are applied honestly.
* **Sharpest gap**: whether $N_{\mathrm{key}}/\lvert\mathcal{C}\rvert$ falls fast enough with the external space to beat semistochastic sampling by $2\times$ before ED stops being feasible.
    * At $N=6$ the measured floor goes $0.95\to0.55\to0.33\to0.28$ ($K=8\to16$), while the best control sits at $0.55$–$0.70$.
    * For R3.1 there is a second gap: valence-only backflow gives up $0.5$ mHa at $K=12$, and the loss grows with the external space ($\Delta E$ ratio $1.07\to1.52\to1.27$).
* **What would change the verdict**:
    * a window chosen by natural-orbital occupation rather than by valence count;
    * or a test on non-symmetric molecules, where $\lvert\mathcal{C}\rvert$ is not thinned by point-group zeros.

  Both are proposed for round 4, not assumed here.

### 24-Hour Decisive Test (Round 3)
* **Instance**:
    * H$_6$ ring with $K=8,10,12,14,16$ ($M=16$–$32$, dimension up to 313 600, 2–10 p-type MOs).
    * H$_8$ ring with $K=9,10,11,12$ ($M=18$–$24$, dimension up to 245 025, 1–4 p-type MOs).
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
    * Job: array 0–26 (9 instances × 3 arms), partition `kim`, 2 CPUs, 32 GB, 24 h.
    * Expected cost: about 2.5 h per task for the largest instances (N = 6, K = 16, dimension 313 600; N = 8, K = 12, dimension 245 025), scaled from the measured pre-check (0.2 s per L-BFGS iteration at dimension 48 400).
    * Command (not run): `sbatch /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r3/submit_test.sbatch`.
    * Verdict: `python3 micro_test.py --aggregate <test dir>`.
* **Smoke test**: command `/share/apps/software/anaconda3/bin/python3 micro_test.py --smoke` (18 s). It checks the integrals, the exact incremental and keyed ratios, the gradients and the unbiasedness of the estimators, then runs H$_4$ with $K=5$ and $K=7$ and three seeds. Literal output, head and verdict lines (the full output is in `smoke_output.txt`):

```
H2 6-31G** R=1.4: E_RHF=-1.131284 (lit. -1.13128); H4 ring: ERI symmetry err=1.1e-16, min eig(ERI)=-1.0e-15, rotation invariance |dE|=3.4e-12
  exact incremental ratio err: full=2.0e-14 window=2.8e-14; window-keyed ratio err: WIB (must be ~0)=2.8e-14, full-input surrogate (nonzero)=4.9e+00; gradient rel err: full=5.0e-10 window=5.5e-10
  unbiasedness (20000 draws, one row): SS z=0.83  PFCV z=0.14   (11.4s)
SMOKE CHECKS PASS
=== H4 ring R=2.8 6-31G** active K=7 M=14 dim=441 E_RHF=-1.909344 E_FCI=-2.034686 Ecorr=-125.3 mHa gap=0.0072 HF wt=0.471 frustr=0.106 p-MOs=3 mean p-weight(external MOs)=0.995 window K_W=4 nnz/row=71 (build 0.9s)
  seed=0 NNBFJ dE=   2.403 mHa Var(EL)=1.01e-03 <|C|>=72 <keys>=44.1 (multi 17.9) ELchk=8.9e-16 KEYX-EXACT max=1.9e-02 |W1_ext|=0.901 (train 0s)
      C/C_EXACT(own state), wall-clock: EXACT=1 PFDET(biased)=1.39 SCREEN=0.923 SS=0.985 PFCV=1.65
      C/C_EXACT(own state), operation count: EXACT=1 KEYX=1.03 SCREEN=0.479 SS=0.629 PFCV=1.16
      SCREEN eps=0.001 K=0 bias=-0.0022+-0.0006 mHa top1%share=0.12
[smoke-scale verdict logic, H4 only, not decisive] R3.1: {"verdict": "FAIL", "wall_clock": {"verdict": "FAIL", "M_by_K": {"5": 5.767, "7": 3.982}}, "op_count": {"verdict": "FAIL", "M_by_K": {"5": 10.418, "7": 6.635}}}
[smoke-scale verdict logic, H4 only, not decisive] R3.2: {"verdict": "FAIL", "wall_clock": {"verdict": "FAIL", "M_by_K": {"5": 1.393, "7": 1.791}}, "op_count": {"verdict": "FAIL", "M_by_K": {"5": 2.992, "7": 2.815}}}
```

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
### Round 4 — Mutation & Replenishment (2026-11-11)
**Binding inputs (round-3 per-leaf critic) and how each is handled.**
* *1. The test could not PASS globally (accepted).* The cost verdict now comes from an ED-free Part S at $K=40$–$136$ spatial MOs, where the key/term ratio is not pinned near 0.5. The smoke run shows that the aggregate logic reaches both PASS and FAIL.
* *2. ED-sized symmetric rings cannot show the $M$-scaling (accepted).* Part S uses a distorted (no point group) H$_8$ chain with s+p bases of growing size. Part A uses distorted chains too.
* *3. WIBJ changes only the prefactor (partly rejected, with a measurement).* The keys change the $M$-dependence of the amplitude work. Exact enumeration costs $\mathcal{O}(\lvert\mathcal{C}\rvert(N^2h+N^3))=\mathcal{O}(N^4M^2h)$. Keyed evaluation costs $\mathcal{O}(N^{\mathrm{ext}}_{\mathrm{key}}MNh+\lvert\mathcal{C}\rvert c_{\mathrm{term}})$. Here $N^{\mathrm{ext}}_{\mathrm{key}}=\mathcal{O}(N^2W)$ is the number of keys with external terms, $W$ the window size in spin-orbitals, $h$ the backflow width, and $c_{\mathrm{term}}=\mathcal{O}(N^2)$ arithmetic per term. That is $\mathcal{O}(N^3WMh+N^2M^2)$: linear in $M$ up to $M\sim NWh$. The critic is right that every term is still visited. The derivation below shows that the proposed contraction does not remove that $\mathcal{O}(N^2M^2)$ floor either.
* *4. Gate 0 (partly accepted).*
    * The $2\times2$ minor is the table method of Clark et al. and of Filippi, Assaraf & Moroni, and key sharing is caching. Both are now listed as implementation, not novelty.
    * WIBJ's Gate 0 exposure (active-space backflow × Jastrow) is accepted. WIBJ is carried only as the $G=0$ end of R4.1 (R3.1′) and is not a route.
    * PFCV (R3.2) is dropped.
* *5. The ΔE tolerance (accepted).* There is now one absolute tolerance, $\tau=0.2$ mHa. The route's accuracy loss against the NNBF-J of the same seed must be at most $\tau$, and every biased control's $\lvert\mathrm{bias}\rvert$ must be at most $\tau$. The premise is that NNBF-J reaches $\Delta E<1.6$ mHa.
* *6. Missing controls (accepted).*
    * TRUNC is the sample-set-truncated local energy of arXiv:2502.18843: the sum runs over the unique configurations of the current batch of $B_U$ samples.
    * SCREEN and SS are charged Hamiltonian generation only for kept terms, which emulates HCI-style pre-sorted integrals and is generous to the controls.
    * The wall-clock veto is now one-sided. It applies only if the bootstrap 95% CI of $M_{\mathrm{wall}}$ lies entirely above 1.
* *7. Leakage (accepted).*
    * SCREEN and TRUNC biases are now exact sums over $\pi$ (SCREEN: $\langle\psi\rvert H^{\varepsilon}-H\lvert\psi\rangle/\langle\psi\vert\psi\rangle$ with the sparse screened $H^{\varepsilon}$; TRUNC: exact per-sample truncation errors for five independent batches), so no Monte Carlo sample enters them.
    * SS is selected on half A and scored on half B. The heavy-tail rule must hold on **both** halves.
    * PFCV is gone.
* *8. Cholesky anti-pattern (accepted).* The round-3 proposal "infinite variance" is withdrawn in favour of the critic's support condition $\mathrm{supp}(L_\gamma\psi)\subseteq\mathrm{supp}(\psi)$.
* *9. Weak correlation (accepted).* The Part A instances are A1, H$_6$ at $R=3.6$ bohr (HF weight 0.532), and A2, H$_8$ at $R=3.4$ bohr (HF weight 0.470). Both are distorted chains in 6-31G**. The code aborts if the HF weight is $\ge0.6$.

**Derivation: the critic's per-key contraction (Operation A on WIBJ).** Fix a sample $\mathbf{n}$ and a backflow key $\kappa$, i.e. the window occupation of $\mathbf{n}'$. Suppose the Jastrow is factorizable across the window boundary: one-body terms, window–window and window–external pairs, a neural term on window features, and **no external–external pair**. Then every $\mathbf{n}'$ with key $\kappa$ has
$$\frac{J(\mathbf{n}')}{J(\mathbf{n})}=e^{c_\kappa}\prod_{p\in\mathrm{ext}}e^{\lambda_p(\kappa)(n'_p-n_p)},\qquad \psi(\mathbf{n}')\propto\det\big[D_\kappa\Phi_\kappa\big]_{[\mathrm{occ}\,\mathbf{n}']},$$
with $D_\kappa=\mathrm{diag}(e^{\lambda_p(\kappa)})$ a fugacity dressing of the key's orbitals. For the dominant class, $i,j$ removed from the window and $a,b$ added in the external space, the generalized Wick theorem with $\tilde G=D_\kappa\Phi_\kappa A_\kappa^{-1}$, $A_\kappa=\Phi_\kappa[\mathrm{occ}\,\mathbf{n}]$, and Cholesky vectors $(pq\vert rs)=\sum_\gamma L^\gamma_{pq}L^\gamma_{rs}$ gives
$$\sum_{a<b}H_{\mathbf{n}\mathbf{n}'}\frac{\psi(\mathbf{n}')}{\psi(\mathbf{n})}=\frac{\det A_\kappa}{\det A_{\mathbf{n}}}e^{c_\kappa}\sum_\gamma\Big[\Big(\sum_{a:\sigma_a=\sigma_i}L^\gamma_{ai}\tilde G_{a i}\Big)\Big(\sum_{b:\sigma_b=\sigma_j}L^\gamma_{bj}\tilde G_{b j}\Big)-\Big(\sum_{a:\sigma_a=\sigma_j}L^\gamma_{aj}\tilde G_{a i}\Big)\Big(\sum_{b:\sigma_b=\sigma_i}L^\gamma_{bi}\tilde G_{b j}\Big)\Big].$$
* The exchange bracket survives for **opposite** spins $\sigma_i\ne\sigma_j$, because the backflow determinant is a generalized, spin-mixing one ($\tilde G_{a\beta,i\alpha}\ne0$). Omitting it gave errors of up to 27× in the first smoke run.
* With it, the identity holds to $1.2\times10^{-10}$ over 30 keys, including samples with external spectators (`--smoke`).
* **Cost.** Per key, $\mathcal{O}(MNh)$ for the key's orbitals (needed either way) plus $2N_\gamma M_{\mathrm{ext}}$ for the contraction. Enumerating the class costs about $(M_{\mathrm{ext}}^2/4)\,c_{\mathrm{term}}$.
* **Measured Cholesky rank**, $\tau_{\mathrm{chol}}=10^{-6}$: $N_\gamma=6.4K$ (H$_4$, $K=20$), $7.7K$ (H$_4$, $K=36$) and $6.7K$ (H$_8$, $K=40$).
* **Measured cost ratio**, contraction over enumeration on the same keys: 0.67, 0.69 and 0.87.

**Generation Batch 4** (round 4; pre-check data are in the table below):
* **Route 4.0 (Operation A on WIBJ, the critic's direction): factorized-Jastrow keyed Cholesky contraction (FJKC).**
    * Gate 0 (Novelty & Adoption): FAIL as stated. Generalized-Wick local energies with half-rotated Cholesky vectors are the AFQMC standard (Motta & Zhang, arXiv:1711.02242; Cholesky local energy, arXiv:2508.13036). Fast multi-Slater–Jastrow local energies in orbital space cost $\mathcal{O}(n^5+n_c)$ (Mahajan & Sharma, arXiv:2008.06477). Integral screening is Sabzevari & Sharma (arXiv:1807.10633). The fugacity dressing per key is new, but it only makes the standard contraction applicable.
    * Gate 1 (Conservation & Asymptotics): **FAIL (derived and measured).** The claimed change of scaling does not exist. Each key's external sum is a bilinear form in an $M_{\mathrm{ext}}\times M_{\mathrm{ext}}$ block. A rank-$N_\gamma$ factorization with $N_\gamma\approx7K$ costs the same $\mathcal{O}(M^2)$ as enumerating the block. Over all keys that is $\mathcal{O}(N^2N_\gamma M)=\mathcal{O}(N^2M^2)$, the same order as the enumeration floor; measured ratio 0.67–0.87. It does not touch the per-key orbital build $\mathcal{O}(MNh)$, which is where the keys already won.
    * Accuracy cost, worked out as asked: with no external–external pair term, the amplitude of $ij\to ab$ within one key is $\tilde G_{ai}\tilde G_{bj}-\tilde G_{aj}\tilde G_{bi}$, **rank 2 in $(a,b)$ for each pair $ij$**. First-order pair amplitudes $t^{ij}_{ab}$ are full rank. Part A measures the loss as the FJ arm.
    * *Verdict*: KILLED as a route. The identity is kept in the code and FJ is kept as a diagnostic arm.
* **Route 4.1 (Operation A on WIBJ's Gate 0 and accuracy walls): coarse-grained-key backflow (CGK).** The backflow input is the window occupations plus $2G$ coarse external counts $g_{\sigma s}(\mathbf{n})=\sum_{p\in\mathrm{shell}\,s}n_{p\sigma}$, the electrons of spin $\sigma$ in each of $G$ external energy shells. The neural Jastrow sees the same features, so it is evaluated once per key. The pair Jastrow covers all pairs and costs $\mathcal{O}(1)$ per term.
    * The orbitals now respond to *where in the external spectrum* electrons went, the energy scale of dynamic correlation. That makes it not active-space backflow.
    * A double excitation $ij\to ab$ has key $(\mathbf{n}_W-ij,\ g+e_{s(a)}+e_{s(b)})$. The key count grows by at most $\mathcal{O}(G^2)$ and stays independent of $M$.
    * $G$ is the λ-continuation: $G=0$ is WIBJ, and $G=K_{\mathrm{ext}}$ (one shell per orbital) is the unrestricted NNBF.
    - Gate 0 (Novelty & Adoption): PASS (narrow).
        * Searches (three queries: Jastrow/Slater/Cholesky fast local energies; NNBF fast local energies; occupation-input and opposite-spin backflow) found NNBF with truncated local energies (arXiv:2502.18843, 2603.14775), parallel exact enumeration (arXiv:2509.25720), multi-Slater–Jastrow local energies (arXiv:2008.06477) and spin-resolved NNBF (Luo & Clark, PRL 122, 226401).
        * None designs the backflow input so that the exact local energy is key-groupable. None uses coarse external occupation counts as backflow input.
        * Recorded risk: a referee may call this feature-engineered NNBF. The defence is the measured cost structure, not the construction.
    - Gate 1 (Conservation & Asymptotics): PASS. It is exact and conserves $N$ and $S_z$. KEYX equals EXACT to $2\times10^{-15}$ (relative) on the $K=40$ samples and to $2\times10^{-13}$ on A1. The amplitude cost is $\mathcal{O}(N^2WG^2\cdot MNh)$, plus the $\mathcal{O}(\lvert\mathcal{C}\rvert)$ floor.
    - Gate 2 (Baseline Gameability): **PASS on the pre-check, with no margin.** At $K=40$, $M_{\mathrm{op}}=0.49$ against SCREEN at the largest admissible $\varepsilon=3\times10^{-3}$, which keeps 17.5% of terms. The wall-clock $M_{\mathrm{wall}}=0.94$, CI $[0.88,1.03]$, does not veto. TRUNC is inadmissible: its bias is 3–25 mHa at $B_U=16\,000$–$1\,000$.
    - Gate 3 (Hard-Core Shield): PASS (conditional; sign half only). A1 has HF weight 0.53 and frustrated fraction 0.24. At $K=8$, CGK's accuracy loss is $-0.001$ mHa. At $K=10$ (two seeds, 1000 iterations), $\delta=-0.04$ and $+0.07$ mHa, with variance ratio 1.0.
    - *Verdict*: SURVIVED (conditional on the decisive test).
* **Route 4.2 (Operation B.1, frame transformation from occupation-window keys to spin-sector keys): cross-spin-keyed backflow (XSK).** $\psi=J\det\Phi^\alpha(\mathbf{n}^\beta_W)[\mathrm{occ}\,\alpha]\det\Phi^\beta(\mathbf{n}^\alpha_W)[\mathrm{occ}\,\beta]$.
    * Every same-spin excitation of spin $\alpha$ leaves the $\alpha$-orbital key unchanged. So all $\alpha\alpha$ doubles share one set of $\alpha$ orbitals, and only one $N/2$ determinant ratio per $\alpha$-window change is needed for the $\beta$ block.
    * The number of $M$-row orbital builds per sample drops from $\mathcal{O}(N^2W)$ to $\mathcal{O}(NW)$.
    * Cross-field basis: Luo & Clark (PRL 2019) found that their spin-resolved NNBF learns mostly cross-spin weights. XSK imposes this as a hard constraint.
    - Gate 0 (Novelty & Adoption): PASS (narrow). No paper restricts backflow to cross-spin, window-only input. The opposite-spin tensor backflow of arXiv:2308.11823 is a Hubbard-model form without the key structure.
    - Gate 1 (Conservation & Asymptotics): PASS. Exact, and KEYX equals EXACT to $6\times10^{-16}$. The cost at $K=40$ is 0.040 of EXACT, but the $\lvert\mathcal{C}\rvert$ floor is now 61% of its operations.
    - Gate 2 (Baseline Gameability): **FAIL on the pre-check.** $M_{\mathrm{op}}=8.3$, because its $\mathrm{Var}(E_L)$ is 64× that of NNBF-J.
    - Gate 3 (Hard-Core Shield): **FAIL on the pre-check.** The accuracy loss is 4.0 mHa at A1, $K=8$ (and 3.5–4.8 mHa at smoke scale). Same-spin backflow is not dispensable at HF weight 0.5. At $K=10$, $\delta=+3.8$ and $+10.2$ mHa.
    - *Verdict*: NOT A SURVIVOR. It is carried in the decisive test as a conditional candidate, because the pre-check is one seed at 1500 iterations.
* **R3.1′ (WIBJ, the $G=0$ end of R4.1)**: carried for the continuation only. At $K=40$, $M_{\mathrm{op}}=0.69$; at A1, $K=8$, the accuracy loss is $+0.009$ mHa and the variance ratio 1.6. At $K=10$, $\delta=-0.22$ and $+0.03$ mHa.

**Pre-checks** (login node, not decisive): Part S at $K=40$ only (`--basis-series 2,1 --n-samples 20 --n-wall 4`; `precheck_S_K40.log`); Part A on A1 at $K=8$ (one seed, 1500 iterations, 400 samples; `precheck_A1_K8.log`) and at $K=10$ (seeds 0 and 1, 1000 iterations; `precheck_A1_K10.log`).

| Quantity | NNBFJ (control) | WIBJ (R3.1′) | CGK (R4.1) | XSK (R4.2) | FJ (R4.0 diag.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Part S, $K=40$: KEYX/EXACT, operations | 1.80 | 0.074 | 0.092 | 0.040 | 0.074 |
| Part S, $K=40$: KEYX/EXACT, wall-clock | 1.13 | 0.15 | 0.17 | 0.20 | 0.14 |
| Part S, $K=40$: $\lvert\mathcal{C}\rvert$-floor share of KEYX operations | 0.01 | 0.23 | 0.19 | 0.61 | 0.23 |
| A1, $K=8$: $\Delta E$ (mHa), seed 0 | 0.014 | 0.023 | 0.013 | 4.019 | 0.023 |
| A1, $K=10$: $\Delta E$ (mHa), seeds 0, 1 | 0.735, 0.700 | 0.512, 0.727 | 0.698, 0.770 | 4.54, 10.91 | not run |
| A1, $K=10$: $\mathrm{Var}(E_L)$ ($10^{-3}$), seeds 0, 1 | 3.32, 3.18 | 2.33, 3.32 | 3.19, 3.50 | 5.27, 8.77 | not run |

Controls on the NNBFJ state (A1):
* **SCREEN exact bias**: $\le0.015$ mHa for $\varepsilon\le10^{-3}$, and $0.18$–$0.19$ mHa at $3\times10^{-3}$, so it is admissible up to $3\times10^{-3}$. It keeps 17.5% of the terms at $K=40$.
* **TRUNC batch bias**: 23–25, 7.2–7.7 and 2.7–3.1 mHa at $B_U=1000$, 4000 and 16 000. It is inadmissible everywhere.
* **SS**: the heavy-tail rule rejects most grid points at $\varepsilon\le3\times10^{-4}$, where the increments are tiny and dominated by single samples. The selected setting has an increment ratio of $0$–$0.009$ on half B.
* **Combined smoke-scale ratio** (Part S at $K=40$ with the A1, $K=8$ accuracy data, through the aggregate code): $M_{\mathrm{op}}=0.49$ (R4.1), 0.69 (R3.1′) and 8.3 (R4.2). For R4.1, $M_{\mathrm{wall}}=0.94$, CI $[0.88,1.03]$.

### Top Surviving Route & Sharpest Gap (Round 4)
* **Survivor**: R4.1 (CGK). It is the only route that passes all four gates on its pre-check, and it does so without margin.
* **Conditional candidate**: R4.2 (XSK). It fails Gates 2 and 3 on a one-seed pre-check, so it is not listed as a survivor; the decisive run settles it.
* **Killed this round**: R4.0 (the critic's contraction; Gate 1, a constant-factor change only), and PFCV (dropped).
* **Sharpest gap (R4.1)**: whether CGK's cost relative to the best admissible control keeps falling with $K$.
    * Its own ratio to EXACT falls with $K$, as $\mathcal{O}(W/(NM))$.
    * The admissible SCREEN keeps a shrinking fraction of terms as the basis grows (0.42 at $\varepsilon=10^{-3}$ and 0.175 at $3\times10^{-3}$, $K=40$).
    * Either can win. The 0.49 at $K=40$ is not a margin.
    * The second gap is the accuracy of $G=2$ shells at $K=12$ (A1) and $K=11$ (A2), where 4–6 external orbitals per spin are coarse-grained.

### 24-Hour Decisive Test (Round 4)
* **Part S (scaling, ED-free).**
    * Instance: a distorted H$_8$ chain ($R=2.0$ bohr, displacement $0.25$ bohr, seed 1, no symmetry) in s+p bases with $(n_s,n_p)=(2,1),(3,2),(4,3),(5,4)$, i.e. $K=40$, 72, 104 and 136 MOs ($M\le272$). $K=168$ is the pre-committed extension. Integrals are built in numpy by McMurchie–Davidson; rows use spatial ERIs, and configurations are Zobrist-hashed for $M>62$.
    * Samples: 200 from a CISD-like proposal (HF 0.4, singles 0.1, doubles 0.5 with weight $\propto\lvert t^{ij}_{ab}\rvert^2$ from MP1).
    * Arms: all five, each at its initial parameters plus a fixed random perturbation. Cost is structural, so no training is needed.
    * Measured per sample: the KEYX operation count and wall-clock; EXACT, SCREEN (kept terms only) and TRUNC wall-clock on 12 samples; the KEYX-vs-EXACT deviation; and $N_\gamma$ with the R4.0 contraction ratio.
* **Part A (accuracy, ED).**
    * Instances: A1 (H$_6$, $R=3.6$, HF weight 0.532, $K=8,10,12$) and A2 (H$_8$, $R=3.4$, HF weight 0.470, $K=9,10,11$).
    * Training: every arm by L-BFGS on the exact energy, width $h=64$, 3000 iterations, seeds 0, 1 and 2.
    * Measured: $\Delta E$ and $\mathrm{Var}(E_L)$ per arm; for the controls, the exact SCREEN and TRUNC biases and the SS variance increments (half A / half B).
* **Metric and thresholds** (implemented in `--aggregate`; $\tau=0.2$ mHa):
    * $\delta_R=$ the median over seeds of $\Delta E_R-\Delta E_{\mathrm{NNBFJ}}$, paired by seed, at each instance's largest $K$. The premise is HF weight $<0.6$ and median $\Delta E_{\mathrm{NNBFJ}}<1.6$ mHa.
    * $C_R(K)=\mathrm{ops}_{\mathrm{KEYX}}(R,K)\,v_R$, where $v_R$ is the largest median variance ratio over instances. $C_{\mathrm{ctrl}}(K)$ is the minimum of EXACT, SCREEN at the largest $\varepsilon$ admissible on every instance, SS times $(1+\text{measured increment})$, and TRUNC at the smallest admissible $B_U$.
    * $M_{\mathrm{op}}=C_R/C_{\mathrm{ctrl}}$, and $M_{\mathrm{wall}}$ is the same ratio with measured seconds and a bootstrap CI.
    * **PASS**: $\delta_R\le\tau$ on every instance, $M_{\mathrm{op}}(K_{\max})\le0.5$, $M_{\mathrm{op}}(K_{\max})<M_{\mathrm{op}}(K_{\min})$, and no wall-clock veto (CI lower bound $>1$).
    * **FAIL**: $\delta_R>\tau$ on any premise-valid instance, or $M_{\mathrm{op}}(K_{\max})\ge1$, or a wall-clock veto.
    * **AMBIGUOUS** otherwise. Pre-committed actions: a failed premise means rerun Part A with `--hidden 96 --maxiter 6000`; an ambiguous cost means rerun Part S with `--basis-series 6,5` ($K=168$), and the route PASSES if and only if $M_{\mathrm{op}}(168)\le0.5$, else it is KILLED.
* **Can PASS and can FAIL**: the smoke-scale aggregate (H$_4$) returns PASS for R3.1′ and FAIL for R4.1 and R4.2 (output below). On the pre-checks, R4.1 sits at $M_{\mathrm{op}}=0.49$ and R4.2 at 8.3.
* **Execution**:
    * Code: `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r4/micro_test.py` and `submit_test.sbatch`.
    * Job: array 0–33 (4 Part S tasks, 30 Part A tasks), partition `kim`, 2 CPUs, 64 GB, 24 h.
    * Expected time: at most about 5 h for the largest Part A task and about 3 h for $K=136$.
    * Command (not run): `sbatch /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r4/submit_test.sbatch`.
    * Verdict: `python3 micro_test.py --aggregate <test dir>`.
* **Smoke test**: command `/share/apps/software/anaconda3/bin/python3 micro_test.py --smoke` (14 s). Literal output, check lines and the verdict lines; the full output is in `smoke_output.txt`:

```
H2 R=1.4: E_RHF 6-31G**=-1.131284 (lit. -1.13128); (ns,np)=(4,3) E_RHF=-1.132223 (must be lower; HF limit -1.13363)
  spatial vs spin-orbital Slater-Condon rows: max |diff| = 4.2e-17
  NNBFJ: exact-ratio err=2.8e-14 keyed-ratio err=8.5e-14 grad rel err=1.5e-09 blocks=1 features=[12] jastrow features=12
  WIBJ : exact-ratio err=2.3e-13 keyed-ratio err=1.7e-13 grad rel err=1.5e-09 blocks=1 features=[8] jastrow features=8
  CGK  : exact-ratio err=3.6e-14 keyed-ratio err=9.5e-14 grad rel err=1.4e-09 blocks=1 features=[12] jastrow features=12
  XSK  : exact-ratio err=1.5e-11 keyed-ratio err=1.4e-11 grad rel err=3.3e-10 blocks=2 features=[4, 4] jastrow features=8
  FJ   : exact-ratio err=2.3e-13 keyed-ratio err=2.1e-12 grad rel err=3.2e-10 blocks=1 features=[8] jastrow features=8
  R4.0 Cholesky contraction == enumerated class sum: max rel dev = 1.2e-10 over 30 keys (Cholesky reconstruction err 1.7e-16)
  TRUNC per-sample truncation error, sparse formula vs direct rows: max |diff| = 2.2e-16
  (1.7s)
SMOKE CHECKS PASS
=== [A] H4 distorted chain R=4.4 d=0.25 K=5 dim=100 E_RHF=-1.767174 E_FCI=-1.984735 HF wt=0.490 frustr=0.192 nnz/row=55 (0.6s)
=== [A] H4 distorted chain R=4.4 d=0.25 K=6 dim=225 E_RHF=-1.767174 E_FCI=-1.984758 HF wt=0.490 frustr=0.192 nnz/row=93 (0.6s)
=== [S] H4 distorted chain R=2.0 d=0.25 basis(ns,np)=(2,1) K=20 M=40 E_RHF=-2.155268 N_chol(1e-6)=128 (6.40 K) N_chol(1e-4)=91 (build 1s)
  <|C|>=1674  kept fraction by eps: 0.0001:0.861 0.001:0.575 0.01:0.167
  NNBFJ ops KEYX/EXACT=2.0767 floor share=0.10 wall KEYX/EXACT=2.8456 KEYX-EXACT rel dev=4.7e-16
  WIBJ  ops KEYX/EXACT=0.3165 floor share=0.68 wall KEYX/EXACT=1.2069 KEYX-EXACT rel dev=3.5e-16
  CGK   ops KEYX/EXACT=0.3646 floor share=0.59 wall KEYX/EXACT=1.2898 KEYX-EXACT rel dev=8.4e-16
  XSK   ops KEYX/EXACT=0.3500 floor share=0.79 wall KEYX/EXACT=1.3188 KEYX-EXACT rel dev=4.3e-16
  FJ    ops KEYX/EXACT=0.3165 floor share=0.68 wall KEYX/EXACT=1.1193 KEYX-EXACT rel dev=1.8e-16 | R4.0 contraction/enumeration ops=0.67 (identity dev 2.3e-05)
=== [S] H4 distorted chain R=2.0 d=0.25 basis(ns,np)=(3,2) K=36 M=72 E_RHF=-2.157986 N_chol(1e-6)=276 (7.67 K) N_chol(1e-4)=193 (build 4s)
  <|C|>=5881  kept fraction by eps: 0.0001:0.863 0.001:0.498 0.01:0.069
  NNBFJ ops KEYX/EXACT=2.5174 floor share=0.09 wall KEYX/EXACT=4.0948 KEYX-EXACT rel dev=7.3e-16
  WIBJ  ops KEYX/EXACT=0.2602 floor share=0.83 wall KEYX/EXACT=0.9169 KEYX-EXACT rel dev=6.2e-16
  CGK   ops KEYX/EXACT=0.2785 floor share=0.77 wall KEYX/EXACT=1.0462 KEYX-EXACT rel dev=3.5e-16
  XSK   ops KEYX/EXACT=0.3353 floor share=0.83 wall KEYX/EXACT=1.0136 KEYX-EXACT rel dev=2.9e-16
  FJ    ops KEYX/EXACT=0.2602 floor share=0.83 wall KEYX/EXACT=0.8940 KEYX-EXACT rel dev=8.3e-16 | R4.0 contraction/enumeration ops=0.69 (identity dev 2.8e-04)
[smoke-scale verdict logic, H4 only, not decisive]
{"R4.1": {"verdict": "FAIL", "accuracy": "ACC_FAIL", "M_op_by_K": {"20": 0.63, "36": 0.556}}, "R4.2": {"verdict": "FAIL", "accuracy": "ACC_FAIL", "M_op_by_K": {"20": 0.631, "36": 0.698}}, "R3.1'": {"verdict": "PASS", "accuracy": "ACC_OK", "M_op_by_K": {"20": 0.314, "36": 0.298}}}
controls transferred: {"SCREEN_eps": "0.001", "TRUNC_BU": null}
```

### Local Autopsy Log (Round 4 additions)
* **R4.0 (per-key Cholesky contraction under a factorizable Jastrow)**: an exact identity, verified, but the contraction of an $M_{\mathrm{ext}}\times M_{\mathrm{ext}}$ block with $N_\gamma\approx7K$ vectors costs the same order as enumerating it (measured 0.67–0.87). Forbidding external–external pairs also makes every pair amplitude rank 2. **Boundary condition**: with molecular ERIs, no factorization of the integrals removes the $\mathcal{O}(N^2M^2)$ term floor of an exact local energy whose orbitals depend on the excitation's window part. Only the amplitude work can be cut, which the keys do.
* **R4.2 (cross-spin keys)**: the cheapest keyed form (0.040 of EXACT at $K=40$), but on the one-seed pre-check it loses 4 mHa and multiplies $\mathrm{Var}(E_L)$ by 64 on the multireference chain.
* **PFCV (R3.2)**: dropped (critic item 4).
* **New anti-pattern proposed for the master list**: "a factorized-integral (Cholesky/THC) contraction of a local-energy class whose orbitals change per class costs $\mathcal{O}(N_\gamma M)$ per class, the same order as enumerating its $\mathcal{O}(M^2)$ terms, because $N_\gamma\propto K$". Contractions save only when one set of orbitals serves many $(i,j)$ (the AFQMC case).
* **Correction to the round-3 anti-pattern proposal**: the Cholesky sum-of-squares failure is the support condition $\mathrm{supp}(L_\gamma\psi)\subseteq\mathrm{supp}(\psi)$, as the critic stated.
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
    * **F$_{A2}$, sub-$\lvert\mathcal{C}\rvert$ by contraction.** If the dependence on the external indices is sum-factorizable, whole classes of terms can be summed without enumerating them. An example is Slater–Jastrow with a Jastrow that factorizes across the window boundary: a Wick plus Cholesky (or THC) contraction sums the (window → external) double class at a cost below its term count. The round-4 R4.0 contraction is an instance: its identity held to $2\times10^{-4}$, and the measured contraction/enumeration ratio at $K=40$ was 0.87.
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
    * CAS × Jastrow in QMC: multideterminant (CAS) Slater–Jastrow wave functions, and the Jastrow antisymmetrized geminal power, JAGP (Casula & Sorella in real space; Neuscamman in Hilbert space).

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

### Round 5 Decisive Result (job 452238, 2026-11-14)

**Verdict (pre-committed `verdict_R51`, `python3 micro_test.py --aggregate .`): R5.1 FAIL on cost.** R5.2 (ALE) is UNTESTED (deferred).

* **Run note.** Tasks 18–23 (H$_4$ at $K=24$) failed at setup: H$_4$ in 6-31G** has only 20 MOs, so $K=24$ does not exist. That is a design error, not a crash. The trend rule needs at least three $K$ values; H$_4$ keeps $K=12$, 16 and 20, so the rule is still evaluable.
* **Accuracy: ACC_AMBIG.**

| instance | $K$ | med $\Delta E$ NNBFJ | med $\Delta E$ WIBJ | $\delta_{\mathrm{med}}$ | $\delta_{\mathrm{best}}$ | $\nu_W$ | best foil (med $\Delta E$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| H$_4$ | 12 | 0.360 | 0.013 | $-0.344$ | $-0.346$ | 2.20 | CASJ 0.390 |
| H$_4$ | 16 | 0.699 | 0.064 | $-0.652$ | $-0.331$ | 1.99 | CASJ 0.638 |
| H$_4$ | 20 | 0.131 | 0.187 | $+0.068$ | $+0.147$ | 0.147 | NNBFJh8 0.346 |
| H$_6$ | 12 | 0.349 | 0.112 | $-0.237$ | $-0.036$ | 0.394 | NNBFJh8 0.968 |
| H$_6$ | 14 | 1.203 | 0.647 | $-0.550$ | $-0.423$ | 0.333 | NNBFJh8 1.333 |

  All energies are in mHa. H$_4$ at $K=20$ passes both thresholds ($\tau=0.2$). H$_6$ at $K=14$ fails the premise ($\nu_W=0.33>\tau$), hence ACC_AMBIG.
* **Trend: no TREND_FAIL.** $\delta_{\mathrm{med}}(K)=[-0.34,-0.65,+0.07]$ is not monotone. **Warning:** WIBJ's absolute median $\Delta E$ rises monotonically, 0.013, 0.064 and 0.187 mHa (about ×3 per $+4$ in $K$), and crosses NNBFJ at $K=20$.
* **Foils: no FOIL_FAIL.** No foil is within the noise floor of WIBJ everywhere. At $K=136$ the foils cost 4.3× (NNBFJh8), 6.4× (NNBFJh4) and 52× (CASJ) WIBJ's cost.
* **Cost (decides the FAIL).**

| $K$ | $M_{\mathrm{op}}$ | $M_{\mathrm{wall}}$ (95% CI) | WIBJ+SCREEN / NNBFJ+SCREEN, op | WIBJ+SCREEN / NNBFJ+SCREEN, wall |
| :--- | :--- | :--- | :--- | :--- |
| 40 | 0.38 | 0.60 (0.59–0.61) | 0.21 | 0.21 |
| 72 | 0.37 | 1.29 (1.26–1.33) | 0.11 | 0.23 |
| 104 | 0.52 | 2.32 (2.25–2.39) | 0.10 | 0.24 |
| 136 | 0.70 | 4.03 (3.91–4.15) | 0.10 | 0.28 |

  The FAIL clause is: lower CI bound of $M_{\mathrm{wall}}(136)$ above 1. $M_{\mathrm{op}}$ also rises with $K$ instead of falling.
* **Design flaw (recorded, not used to overturn the verdict).** As pre-registered, $M_{\mathrm{op}}$ and $M_{\mathrm{wall}}$ charge WIBJ with **unscreened** KEYX, against the cheapest NNBFJ control, which is **SCREEN($\varepsilon^*=0.002$)** at every $K$. The comparison is therefore asymmetric. The symmetric pairing (both arms screened at the same $\varepsilon^*$) was printed only "beside the verdict"; it is 0.10 (op) and 0.28 (wall) at $K=136$, inside the PASS cost thresholds. SCREEN's admissibility for WIBJ (energy bias at $\varepsilon^*$) was never checked, so that number is not a pass.
* **Reading.** Unscreened keyed exactness loses wall-clock to plain screening as $K$ grows: KEYX skips forward passes, but it still touches every connected term, while SCREEN drops most of them. Whatever advantage WIBJ has is a per-kept-term prefactor on top of screening. Accuracy at fixed width degrades with $K$ faster than NNBFJ's.

### Round 5 per-leaf critic (2026-11-14)
**Verdict: KILL R5.1, and do not mutate it to screened WIBJ (R5.1s).** P2.2 closes with the corrected Saturation Diagnostic as its result. R5.2's cold-start micro-pilot and the F$_{A2}$ contraction regime stay as named follow-ups, not as survivors.

* **Code and data checked directly.**
    * `python3 micro_test.py --aggregate .` reproduces the verdict and every number in the cost table: $M_{\mathrm{wall}}(136)=4.03$, CI $[3.91,4.15]$, `ctrl_wall = SCREEN`, $\varepsilon^*=0.002$, $v_W=1.336$.
    * The critic check `critic_screen_bias.py` (job 621074, output in `tests/P2.2_stochastic_local_energy_r5/critic_r5/`) retrains NNBFJ and WIBJ exactly as Part W does. It reproduces the 452238 $\Delta E$ of every seed to the printed digit, for example WIBJ at $K=12$: 0.013, 0.011, 0.349 mHa. So its states are the decisive-test states.

**1. Is the FAIL correct under the rule as written? Yes. No bug found, and the FAIL is robust.**
* *Generation term.* `verdict_R51` charges the measured full-row generation `tg` to KEYX (and to EXACT and TRUNC), and the fitted kept-term generation `gm(kept)` to SCREEN, as pre-registered. Both sides pay generation. At $K=136$ the generation share is small: KEYX 402.5 ms plus `tg` 54.9 ms, against NNBFJ SCREEN 148.7 ms plus 3.0 ms.
* *Variance ratio.* $v_W$ multiplies the WIBJ side only, which is correct because it converts WIBJ's sample count to NNBFJ's error bar. The value 1.336 is the largest per-instance median, from H$_4$ $K=20$. That choice is conservative against WIBJ: the H$_6$ $K=14$ median is 0.52.
* *Robustness.* Dropping generation gives $402.5\times1.336/151.7=3.5$. Taking the smallest instance median, $v_W=0.52$, still gives about 1.4. The lower bound stays above 1 in every variant.
* *Op/wall gap, an implementation constant.* The wall ratio KEYX/EXACT is 0.15, but the operation ratio is 0.028, so the Python KEYX kernel is about 5× less efficient per operation than the batched forward pass. This does not rescue R5.1. $M_{\mathrm{op}}$ alone is 0.38, 0.37, 0.52 and 0.70 over $K=40$–136. It rises, and at $K=136$ it is above 0.5. With an ideal kernel the verdict would be AMBIGUOUS, and the pre-committed $K=168$ step (PASS iff $M_{\mathrm{op}}\le0.5$) would KILL.
* *Generous to the control.* SCREEN admissibility allows a bias up to $\tau$ itself (0.2 mHa). That was pre-registered and is not changed after the fact.

**2. Is "mutate to R5.1s" legitimate? No. It is a post-hoc rescue.**
* *(a) It is a constant factor.* The symmetric ratio WIBJ+SCREEN over NNBFJ+SCREEN is 0.21, 0.11, 0.10 and 0.10 in operations, and 0.21, 0.23, 0.24 and 0.28 in wall time, over $K=40$–136.
    * This is KEYX's per-term prefactor surviving on the kept terms. It is exactly the F$_{A1}$ effect that the leaf's Statement 1 calls the most any exact scheme can get.
    * The operation advantage is flat, and the wall advantage shrinks with $K$.
    * A constant 3.6× wall factor is a code optimization. It does not change the scaling class, and the leaf's refined F$_{A1}$ criterion asks for accuracy within $\tau$ of NNBF-J, which item (c) rules out.
* *(b) SCREEN bias on WIBJ, now measured.* The exact bias $\psi^{\mathsf T}(H_\varepsilon-H_{\mathrm{off}})\psi/\psi^{\mathsf T}\psi$ on the retrained states, in mHa, as max $\lvert\mathrm{bias}\rvert$ over seeds 0–2:

| H$_4$ $K$ | NNBFJ @ $10^{-3}$ | WIBJ @ $10^{-3}$ | NNBFJ @ $\varepsilon^*=2\times10^{-3}$ | WIBJ @ $\varepsilon^*$ | NNBFJ @ $2.5\times10^{-3}$ | WIBJ @ $2.5\times10^{-3}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 12 | 0.002 | 0.002 | 0.003 | 0.001 | 0.008 | 0.016 |
| 16 | 0.012 | 0.012 | 0.023 | 0.015 | 0.010 | 0.007 |
| 20 | 0.048 | 0.044 | 0.117 | 0.104 | 0.226 | 0.202 |

* *Reading of (b).*
    * SCREEN at $\varepsilon^*$ is admissible for WIBJ on H$_4$. Its bias is the same as NNBFJ's or slightly smaller, so objection (b) does not hold, and the symmetric pairing is fair on H$_4$. H$_6$ was not rechecked.
    * The SCREEN bias at $\varepsilon^*$ grows with $K$ for both arms: about 0.003, 0.02 and 0.11 mHa at $K=12$, 16 and 20. It is positive at $K=20$.
    * So WIBJ+SCREEN at $K=20$ carries $0.187+0.10\approx0.29$ mHa of total error, above $\tau$. NNBFJ+SCREEN's best seed carries $0.028+0.117=0.145$ mHa.
    * Part S applies $\varepsilon^*$ fixed from $K\le20$ at $K=40$–136. If the bias keeps growing, SCREEN at $\varepsilon^*$ is probably inadmissible there for both arms. That shared caveat applies to the control as well, so it does not rescue R5.1.
* *Data.* `critic_r5/aggregate_output.txt`, with all 18 per-seed rows. All 18 retrained $\Delta E$ match job 452238.

* *(c) The accuracy trend is the binding barrier.*
    * WIBJ's median absolute $\Delta E$ on H$_4$ is 0.013, 0.064 and 0.187 mHa at $K=12$, 16 and 20.
    * At $K=20$ its three seeds are tight (0.175, 0.187, 0.199 mHa), so this is an expressivity floor, not optimizer noise. NNBFJ's best seed there is 0.028 mHa.
    * On H$_6$ the WIBJ median goes from 0.112 to 0.647 mHa between $K=12$ and 14.
    * Extrapolating the H$_4$ medians to $K=40$:
        * linear least squares gives 0.61 mHa;
        * the last-segment slope gives 0.80 mHa;
        * a power law gives 7.3 mHa.

      All three exceed $\tau=0.2$ mHa. ($K=20$ is the whole 6-31G** space for H$_4$, so $K=40$ stands for a larger external space, not this basis.)
    * *"No TREND_FAIL" is not evidence of flatness.* The rule was applied to $\delta_{\mathrm{med}}(K)$, and $\nu_W$ is 2.20 and 1.99 mHa at $K=12$ and 16, so the premise fails at both points. $\delta(K)$ is NNBFJ optimizer noise there, so the trend rule could not be evaluated. The only clean signal, WIBJ's own error, rises by about ×3 per $+4$ orbitals. That is the externally contracted MRCI failure that Gate 0 named as the adoption barrier, and that killed CGK.
    * The Part S cost is claimed at $K=40$–136, where WIBJ's accuracy is not measured and is projected to violate $\tau$.
* *(d) Gate 0 for keyed or screened local energies (WebSearch, 2026-11-14).*
    * Spin-adapted active-space NNBF (arXiv:2604.06841) already splits $E_L$ by $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$: kept terms are deterministic and the tail is stochastic. It reports a cost cut of one to three orders of magnitude with bias below the statistical resolution. That is WIBJ+SCREEN without KEYX.
    * The truncated local energy with precomputed amplitudes (Liu & Clark, arXiv:2502.18843) and VQ-NQS redundancy reuse in local-energy evaluation (Sharir, Chan & Anandkumar, arXiv:2212.11296) cover cached or keyed amplitude reuse.
    * The autoregressive chemistry NQS (Barrett, Malyshev & Lvovsky, arXiv:2109.12606) is adjacent, not overlapping.
    * Each component of R5.1s is prior art. The only new element would be a constant-factor combination.

**3. Verdict and what P2.2 returns.**
* **R5.1: KILLED.** It fails the cost rule as written, which is robust to the generation term and to $v_W$. Its symmetric advantage is a constant F$_{A1}$ prefactor that shrinks in wall time with $K$. Its accuracy trend projects past $\tau$ in exactly the regime where the cost is claimed. R5.1s fails Gate 0 as a combination of 2604.06841-style screening with KEYX. Screening itself is admissible for WIBJ, as (b) shows, so that is not a kill reason.
* **P2.2 result.** The Saturation Diagnostic (Statements 1–3) is the leaf's result. In the black-box model, exact local energies need $\Omega(\lvert\mathcal{C}\rvert)$ amplitude queries. Unbiased ones obey the inclusion-probability bound. Fixed-sample amortization gives no first-order gain while optimization is noise-limited. Every kill logged in rounds 1–5 is an instance of one of these. The one structure-exploiting route measured (F$_{A1}$, WIBJ) buys a constant factor at an accuracy cost that grows with the external space.
* **Follow-ups, not survivors:**
    * the R5.2 cold-start, step-limited micro-pilot (F$_B$; about 3 seeds on H$_4$ at $K=12$, partition `kim`, under 24 h);
    * F$_{A2}$, whose first step is to measure the accuracy of a factorizable Jastrow before any cost work.
* **Revival condition for any window-restricted ansatz.** A pre-registered absolute-error trend clause on the restricted ansatz itself, with the median $\Delta E(K)$ extrapolated to $K_{\mathrm{cost}}$ at most $\tau$. It must not use $\delta$ against a noisy unrestricted control.

### Closed-leaf falsifier (2026-11-16)
Idle-time thinking-queue item (SKILL Stage 3.5.3, step 3): the two named follow-ups of the Saturation Diagnostic, gated at T0 and run at T1. A falsified diagnostic reopens P2.2 with that route only. Script: `tests/P2.2_closed_falsifier/closed_falsifier.py` (CLI docstring); it reuses the round-5 `micro_test.py` code unchanged. Caps: at most 10 CPU-minutes per run and 30 per leaf, single-threaded.
#### Pre-registration (written before any run)
* **R5.2 (ALE, regime F$_B$), T0 gate.** Statement 3 bounds amortization only while optimization is noise-limited. Far from convergence, SR is step-limited, the anchor drift $\Delta\theta$ per step is large, and the ALE correction variance grows like $\lvert\Delta\theta\rvert^2$, so the ESS refresh rule fires often. Nothing at T0 decides which effect wins, so the route passes T0 and goes to T1.
* **R5.2, T1 design.** H$_4$ at $R=4.4$, $K=12$, NNBF-J hidden 32 and $h_J=16$, one fixed **cold** initial $\theta_0$ (no L-BFGS warm-up) for every arm. The budget is $n_{\mathrm{ref}}=50$ EXACT steps at $B=500$, in operations and in calibrated wall time. Every arm runs in "steps" mode, so a cheaper local energy buys more SR steps (cap $8n_{\mathrm{ref}}$). Arms: EXACT, EXACTNF (noise floor), SCREEN $3\times10^{-3}$, SCREEN $10^{-2}$, SS $10^{-2}$ with $K_d=16$ (controls); ALE0 $T=4$ and $T=8$, ALE1 $T=4$, all with $K_d=16$ (routes). Three sampling seeds per arm.
* **R5.2, rule.** This is `verdict_L` with its warm-start premise replaced by a step-limited one.
    * $\nu$ is the larger of the EXACT-versus-EXACTNF median gap and half the range of the six EXACT and EXACTNF values at $C_b$. $D$ is the best control median at $C_b$. $R^*$ is the ALE arm with the lowest median at $C_b/2$, with medians $R_h$ at $C_b/2$ and $R_f$ at $C_b$.
    * Premise STEP-LIMITED: the EXACT median falls by more than $2\nu$ between $C_b/2$ and $C_b$, and is above 1.6 mHa at $C_b$.
    * On each cost axis: FAIL if $R_f>D+\nu$; PASS if $R_h\le D$ and the best $R^*$ seed at $C_b/2$ is no worse than the best control seed at $C_b$; AMBIGUOUS otherwise.
    * Overall: INCONCLUSIVE if the premise fails; FAIL if either axis fails; PASS if both pass; AMBIGUOUS otherwise.
    * Consequence: PASS reopens P2.2 with R5.2 only. FAIL kills R5.2 in its decisive regime. AMBIGUOUS is recorded, with no further T1 run committed.
* **F$_{A2}$, T0 gate.** Round 4 already showed that a contraction whose orbitals change per class (R4.0) costs the same order as enumeration ($N_\gamma\approx7K$; measured 0.67–0.87). A sub-$\lvert\mathcal C\rvert$ contraction therefore needs orbitals that do not depend on the configuration (one orbital set serving every $(i,j)$, the AFQMC case). It also needs a Jastrow ratio that factorizes over the contracted class, which means no external–external pair. Inside the code base the largest such family is CASJ (window multi-determinant with trained orbital rows) with the external–external pair entries removed. It is a subset of CASJ, whose three seeds at $K=12$ end at 0.386–0.390 mHa (job 452238), already above $\tau$. The prediction is therefore KILL, and the T1 run checks it.
* **F$_{A2}$, T1 design and rule.** Masked CASJ (window = the first $N$ spatial MOs, as for WIBJ and R4.0), $K=12$, seed 0, trained exactly as Part W (L-BFGS, 6000 iterations), plus the unmasked CASJ as a pipeline control.
    * INVALID if the control misses its 452238 value (0.390 mHa) by more than 0.01 mHa.
    * KILL-F$_{A2}$ if the masked CASJ ends above $\tau=0.2$ mHa. Restricted ansätze measured so far get worse with $K$ (WIBJ, CASJ), so failing at $K=12$ is decisive for the revival condition ($\Delta E\le\tau$ at $K_{\mathrm{cost}}\ge40$).
    * PREMISE-HOLDS otherwise, in which case F$_{A2}$ goes to the $K$-trend test at $K=16$ and 20 (not run here).
#### Run 1 results (2026-11-16; `aggregate_output.txt`)
* **F$_{A2}$: KILL-F$_{A2}$.** The control reproduces 452238 (unmasked CASJ 0.3899 mHa against 0.390). The masked CASJ, with the 120 external–external pair entries removed (312 of 432 parameters free), ends at **0.3885 mHa**, above $\tau=0.2$ mHa, as the nesting argument predicted. Removing the external–external pairs costs nothing measurable, so the binding limit is the configuration-independent orbitals of a 4-orbital window, not the factorizable Jastrow. Not tested: a larger CAS window, whose determinant count grows combinatorially and which is the multi-Slater–Jastrow fast local energy of Mahajan & Sharma (arXiv:2008.06477), already prior art at Gate 0.
* **R5.2 (cold start): INCONCLUSIVE, premise STEP-LIMITED fails.** Cold-start $\Delta E$ is 577.6 mHa; $C_b$ is 50 EXACT steps.
    * EXACT is bimodal. At $C_b$ the seeds end at 44.7, 220.5 and 221.1 mHa (EXACTNF: 28.5, 189.6, 67.9), and the 220 mHa seeds sit in a trap from which more EXACT steps do not move them. The median falls only from 223.6 to 220.5 mHa between $C_b/2$ and $C_b$, and $\nu=153$ mHa (half the range), so the premise fails and both axes read AMBIGUOUS.
    * Medians at $C_b$ (operations axis), in mHa: ALE0 $T=8$ 24.0, ALE0 $T=4$ 27.8, ALE1 $T=4$ 221.7 (trapped), against SS 9.4, SCREEN $3\times10^{-3}$ 11.9, SCREEN $10^{-2}$ 14.2, EXACT 220.5.
    * Descriptive reading (not the verdict): ALE0 escapes the trap that EXACT falls into, but every cheap control that also spends its saving on steps (SCREEN, SS) ends below every ALE arm. The best ALE seed is 22.8 mHa; the best control seed is 7.4 mHa.
#### Second T1 run, pre-registered after run 1 (written before it runs)
* **Why a second run.** Run 1 was INCONCLUSIVE for a design reason: a noise floor built from the half-range of a bimodal EXACT sample. The second run keeps the instance, the cold initial state and the verdict clauses. It changes three things:
    * $n_{\mathrm{ref}}=100$ (step cap 800);
    * five seeds per arm (0–4; EXACTNF 10–14);
    * the spread term of $\nu$ becomes half the interquartile range of the ten EXACT and EXACTNF values (`--nu iqr`).
* **Arms.** EXACT, EXACTNF, SCREEN $3\times10^{-3}$ and SS $10^{-2}$ with $K_d=16$ (the two best controls of run 1); ALE0 $T=8$ and $T=4$ with $K_d=16$. ALE1 is dropped: it fell into the trap in all three run-1 seeds.
* **Rule.** As in the first pre-registration, with the new $\nu$. PASS reopens P2.2 with R5.2 only. FAIL kills R5.2 in its decisive regime. INCONCLUSIVE or AMBIGUOUS closes the follow-up as untestable at T1 on this instance, and no third T1 run is made.
#### Second T1 run, result (2026-11-16; `aggregate_output_r2.txt`, logs `log_r2_*.txt`)
**Verdict: R5.2 cold-start AMBIGUOUS on both cost axes.** The STEP-LIMITED premise now holds: the EXACT median falls from 220.5 to 28.0 mHa between $C_b/2$ and $C_b$, and $\nu=17.6$ mHa. As pre-registered, the follow-up is closed as untestable at T1 on this instance, and no third run is made. P2.2 stays closed. The two runs used about 25 CPU-minutes in total.

| arm (operations axis, $C_b$ = 100 EXACT steps) | steps | median $\Delta E(C_b)$ (mHa) | seed range (mHa) |
| :--- | :--- | :--- | :--- |
| SS $10^{-2}$, $K_d=16$ (ctrl$^*$) | 689–706 | 7.0 | 4.8–7.6 |
| SCREEN $3\times10^{-3}$ | 505–555 | 9.0 | 7.3–12.1 |
| EXACTNF | 102 | 16.7 | 13.1–60.6 |
| ALE0 $T=4$, $K_d=16$ | 351–355 | 16.5 | 11.7–32.5 |
| ALE0 $T=8$, $K_d=16$ ($R^*$) | 604–631 | 17.0 | 10.6–21.1 |
| EXACT | 102 | 28.0 | 19.4–219.5 |

* **Why not FAIL.** $R_f=17.0\le D+\nu=24.6$ mHa: the noise floor, still widened by EXACT seeds caught in the 220 mHa trap, absorbs the gap.
* **Descriptive reading (not the verdict).** ALE converts its saving into steps and beats EXACT, as the step-limited argument predicts. But every SS seed (4.8–7.6 mHa) ends below every ALE0 $T=8$ seed (10.6–21.1 mHa), and the wall axis gives the same order. In the step-limited regime, plain screening with a stochastic tail spends the same saving better than anchored reuse. A tighter noise floor would most likely turn this into a FAIL rather than a PASS.
* **The Saturation Diagnostic stands.** Neither named follow-up falsifies it. F$_{A2}$ is killed at T1 on accuracy (0.3885 mHa against $\tau=0.2$). R5.2 gives no PASS in its decisive regime, and the cheap screening controls beat it there.

### Reopening review (2026-11-16)
Prompt: "closed" means neither solved nor proven unsolvable. The Saturation Diagnostic proves only the black-box statements (1–3); its escape regimes were not exhausted. This review turns the named missing ingredients into new routes and tests them at T0 and T1 only. No job is submitted. Caps: at most 4 threads, 10 CPU-minutes per run and 30 CPU-minutes in total, enforced in code. Script: `tests/P2.2_reopening_review/reopen_t1.py` (CLI docstring).
#### 1. Missing ingredients and untried regimes named by the diagnostic
* **I1 (all statements).** The bounds hold only for black-box amplitudes. Any gain needs structure that is free information: integrals, a surrogate, or an algebraic form.
* **I2 (F$_{A2}$).** A sub-$\lvert\mathcal{C}\rvert$ contraction needs configuration-independent orbitals and a factorizable Jastrow. Measured limit: masked CASJ with a 4-orbital window is stuck at 0.3885 mHa. Untried: anything beyond one configuration-independent orbital set other than a larger CAS (prior art, arXiv:2008.06477).
* **I3 (F$_B$).** Statement 3 leaves only step-limited optimization. R5.2 is AMBIGUOUS there, and SS beat every ALE seed (medians 7.0 against 17.0 mHa).
* **I4 (F$_C$).** Certified bias was tried only as a magnitude certificate: Hadamard (R5.3, 140–1000× loose) and statistical (R5.4, Gate 0). **Untried: a certificate of the sign of the bias**, which keeps the reported screened energy a rigorous upper bound, so a variational number needs no magnitude certificate.
* **I5 (orbital frame).** Every instance used canonical RHF orbitals, where $V_{pqrs}$ is dense. The diagnostic never asked how the term count at fixed bias scales with system size in another orbital frame (Gate 4 for screening itself).
#### 2. New candidate routes
* **C1 (Operation A on R5.3, from I4 and I1): variational product-weighted screening (VPS).** For each dropped off-diagonal pair $\{\mathbf{n},\mathbf{n}'\}$ with element $h=H_{\mathbf{n}\mathbf{n}'}$ and any $w>0$, the $2\times2$ block with diagonal $(\lvert h\rvert w,\lvert h\rvert/w)$ and off-diagonal $-h$ is positive semidefinite (determinant zero, trace positive). So dropping the pair and adding $\lvert h\rvert w$ to $D(\mathbf{n})$ and $\lvert h\rvert/w$ to $D(\mathbf{n}')$ gives
$$H'=H_\varepsilon+\mathrm{diag}(D)\ \succeq\ H\quad\Rightarrow\quad\frac{\langle\psi\rvert H'\lvert\psi\rangle}{\langle\psi\vert\psi\rangle}\ \ge\ \frac{\langle\psi\rvert H\lvert\psi\rangle}{\langle\psi\vert\psi\rangle}\ \ge\ E_0\quad\text{for every }\psi .$$
    * The weights are product ratios $w_{\mathbf{n}\to\mathbf{n}'}=g(\mathbf{n}')/g(\mathbf{n})$ with $g(\mathbf{n})=\exp(\sum_p\alpha_pn_p)$, consistent from both ends of a pair. Then $D(\mathbf{n})=\sum_{\mathrm{dropped}}\lvert h\rvert e^{\alpha\cdot(\mathbf{n}'-\mathbf{n})}$ factorizes over the annihilated pair $(i,j)$ and the created pair $(a,b)$, so an upper bound on it is an $\mathcal{O}(N^2)$ lookup of precomputed sums, with no amplitude.
    * The certified excess $\mathbb{E}_\pi[D]-\mathbb{E}_\pi[\sum_{\mathrm{dropped}}x]$ is convex in $\alpha$, so $\alpha$ can be fitted on VMC samples.
    * Two screening criteria: VPS1 drops $\lvert h\rvert\le\varepsilon$; VPSg drops $\lvert h\rvert\,g_0(\mathbf{n})g_0(\mathbf{n}')/\max g_0^2\le\varepsilon$ (HCI-like), with $g_0$ built from the one-body occupations.
* **C2 (Operation A on F$_{A2}$, from I2): a sum of $R$ configuration-independent Slater–Jastrows,** each contractible as in AFQMC.
* **C3 (Operation B.1, frame transformation, from I5): localized-orbital screening.** Work in a site-local orthonormal basis (Löwdin $S^{-1/2}$ orbitals), where $V_{pqrs}$ decays with distance, so the kept terms per row at fixed bias may grow more slowly than $\lvert\mathcal{C}\rvert$.
    * C3′ (merge of C3 with C1): VPS in the localized basis, where the dropped elements, and with them the certified excess per dropped term, decay with distance.
* **C4 (Operation A on R5.2, from I3): ALE on the SS tail only**, the stale-anchor control variate applied to the semistochastic tail draws.
#### 3. Gates and T0
* **Gate 0 (WebSearch, 2026-11-16).**
    * *Queries:* NQS chemistry with localized orbitals and local-energy screening; variational upper bounds from truncated Hamiltonians with the dropped off-diagonals moved to the diagonal; Boys or Pipek–Mezey bases for NQS on hydrogen chains.
    * *Found:* NQS calculations in Boys-localized bases (arXiv:2305.12776); semistochastic or screened local energies (arXiv:2507.19276, 2604.06841, 2502.18843); Gershgorin-type diagonal-shift bounds (arXiv:1605.07239). The closest construction is lattice fixed-node (ten Haaf, van Bemmel, van Leeuwen & van Saarloos, PRB 51, 13039, 1995), which moves the *sign-violating* hops onto the diagonal to get an upper bound for a projector method.
    * *Not found:* a VMC local energy that drops *small* elements of either sign with a product-weighted diagonal compensator, so that the screened energy stays a rigorous upper bound.
* **C1 (VPS).**
    * Gate 0: PASS (narrow). The bound is the lattice fixed-node inequality applied to a different set of terms. The adoption barrier of screening (arXiv:2604.06841) is that its bias has unknown sign and size, so admissibility needs a reference or a stochastic tail (R5.4). VPS removes the sign uncertainty: the reported energy stays a variational upper bound.
    * Gate 1: PASS. $H'\succeq H$ holds exactly; $H'$ conserves $N$ and $S_z$; SR on $H'$ is ordinary VMC of a Hermitian operator.
    * Gate 2: open. A sign-favourable dropped term costs about $4\lvert x\rvert$ of excess instead of $2\lvert x\rvert$ of signed bias, so VPS needs a smaller $\varepsilon$ than plain SCREEN at the same error. Foil: plain SCREEN at $\lvert\mathrm{bias}\rvert\le\tau$, uncertified, in the better basis.
    * Gate 3: PASS (conditional). Kept terms carry exact signs, and the bound holds for any sign structure.
    * Gate 4: in the canonical basis no argument says the kept fraction at fixed excess falls with $N$. In a localized basis (C3′) the dropped elements decay with distance, which predicts a kept count growing more slowly than $\lvert\mathcal{C}\rvert$ (an extensive gain). The compensator costs $\mathcal{O}(N^2)$ per sample after an $\mathcal{O}(M^4)$ precomputation. Checked at T1 on three sizes.
    * *Verdict:* SURVIVED to T1, in both bases.
* **C2.** Gate 0 FAIL: Jastrow times non-orthogonal multi-determinant (resonating Hartree–Fock) and multi-Slater–Jastrow fast local energies (arXiv:2008.06477) exist. Gate 4 FAIL: a fixed number $R$ of mean-field terms is the cross-leaf anti-pattern "a summed or fixed-rank construction is a finite-size device". *Verdict:* KILLED at T0.
* **C3 (plain localized screening).** Gate 0 FAIL: localized bases are used in NQS chemistry (arXiv:2305.12776), and a basis choice plus screening adds no new mechanism. It is kept only as a foil arm (SCREEN in the localized basis). *Verdict:* KILLED as a route.
* **C3′ (VPS in the localized basis).** Gates as for C1, with the Gate 4 argument above. *Verdict:* SURVIVED to T1.
* **C4.** Gate 2 FAIL at T0 from data already on disk: the SS tail adds only about 1–2% to $\mathrm{Var}(E_L)$ on converged states (round 3) and SS costs $0.93\,C_{\mathrm{ex}}$, so removing tail variance buys at most a few percent; in the step-limited regime the SS tail did not limit SS (R5.2 run 2). Statement 3 covers the rest. *Verdict:* KILLED at T0.
#### 4. T1 pre-registration (written before any run)
* **Instances.** Linear H$_N$ chains, STO-3G, $N=6$, 8, 10 (sector dimensions 400, 4900 and 63 504), $S_z=0$. Bond length: the largest $R$ in the list 3.0, 2.8, 2.6, 2.4, 2.2, 2.0 bohr with gap $E_1-E_0\ge0.02$ Ha at all three sizes, asserted in code (the gap is in Ha, the chemistry unit here). The instance is INVALID if the canonical HF weight at that $R$ is 0.7 or more at any size, or if the frustrated fraction is below 0.1.
* **Bases.** CAN (canonical RHF) and LOW (Löwdin $S^{-1/2}$, site-local). The spectrum is the same in both.
* **Arms**, all on the exact ground state $\psi_0$ (the state a converged NQS approximates):
    * SCREEN (foil): drop $\lvert h\rvert\le\varepsilon$; error $\lvert b_S\rvert=\lvert\psi_0^{\mathsf T}(H_\varepsilon-H)\psi_0\rvert$, uncertified;
    * VPS1 and VPSg: error $b_V=\psi_0^{\mathsf T}(H'-H)\psi_0\ge0$, certified, with $\alpha$ minimized (convex) at each $\varepsilon$. The $D$ used is the exact-dropped-set one, which is optimistic (the practical $\mathcal{O}(N^2)$ lookup over-counts), so it is valid for a kill but not sufficient for a pass.
* **Metric.** For each arm, basis and size, $f=n_{\mathrm{kept}}/n_{\mathrm{EX}}$ at the largest $\varepsilon$ on a 41-point log grid ($10^{-7}$ to $10^{-1}$ Ha) whose error is at most $\tau=0.2$ mHa. Here $n_{\mathrm{kept}}$ is $\mathbb{E}_{\pi_0}$ of the number of kept off-diagonal terms in a row, and $n_{\mathrm{EX}}$ is the same count for all terms in the CAN basis (common denominator). Also printed: the floor $F_V=E_0(H')-E_0$ at the chosen $\varepsilon$, a lower bound for any ansatz.
* **Rule, per basis, using the better VPS arm.**
    * KILL if $f_{\mathrm{VPS}}(10)\ge0.5$, or if $f_{\mathrm{VPS}}$ does not fall strictly over $N=6$, 8, 10.
    * PASS if $f_{\mathrm{VPS}}(10)\le0.5$, it falls strictly, and $f_{\mathrm{VPS}}(10)\le3f_S^*(10)$, where $f_S^*$ is the better uncertified SCREEN over both bases (the foil, with a certificate allowance of 3×).
    * AMBIGUOUS otherwise.
* **Consequence.** PASS in either basis sends that route to the value-of-information note and a T2 design (trained NNBF in that basis, practical $\mathcal{O}(N^2)$ compensator). KILL in both bases logs C1 and C3′, and the leaf stays closed.
#### 5. Changes after the smoke run (before the counted run)
* Elements with $\lvert h\rvert\le10^{-12}$ (symmetry zeros of the canonical chain) are no longer counted as terms. This changes the denominator only.
* The VPSg key is $\lvert h\rvert$ times a normalized product weight, which reaches far below $10^{-7}$. Its grid was extended to $10^{-15}$–$10^{-1}$ (113 points, same density). The rule is unchanged.
#### 6. T1 result (2026-11-16; `t1_output.txt`, `t1_output_H10.txt`, `results/`)
* **Instance.** $R=3.0$ bohr, the first value of the list that passes. Gaps $E_1-E_0$ are 0.041, 0.033 and 0.0275 Ha at $N=6$, 8 and 10, all above 0.02 Ha. Canonical HF weights are 0.573, 0.465 and 0.375, and frustrated fractions 0.214, 0.252 and 0.279, so the instance is valid.
* **Checks (smoke, H$_4$ and H$_6$).** $\lambda_{\min}(H'-H)\ge-9\times10^{-17}$ in every case, and $b_V\ge0$ at every $\varepsilon$, so the bound holds numerically.
* **Run note.** The first counted run hit the 600 CPU-s cap during H$_{10}$ LOW (charged at the cap). H$_{10}$ was rerun alone with `--sizes 10 --R 3.0 --arms SCREEN VPS1 --no-floor`, which drops only VPSg and the floor. The $N=6$ and 8 rows come from the first run, and the code is deterministic. Ledger: 982 of 1800 CPU-s.
* **Kept fraction $f$ at error $\le\tau$** (denominator: all canonical terms):

| $N$ | CAN SCREEN | CAN VPS1 | CAN VPSg | LOW SCREEN | LOW VPS1 (= VPSg) | LOW VPS / LOW SCREEN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 6 | 0.891 | 0.917 | 1.000 | 0.306 | 1.003 | 3.3 |
| 8 | 0.910 | 0.949 | 1.000 | 0.144 | 0.682 | 4.7 |
| 10 | 0.846 | 0.976 | 1.000 | 0.079 | 0.477 | 6.1 |

* **Pre-registered verdicts.**
    * **C1 (VPS, CAN): KILL.** $f_{\mathrm{VPS}}(10)=0.976\ge0.5$, and it rises with $N$ (0.917, 0.949, 0.976). VPSg drops only negligible terms. The floor $E_0(H')-E_0$ equals $b_V$ to three digits at $N\le8$, so a re-optimized ansatz would not recover it.
    * **C3′ (VPS, LOW): AMBIGUOUS.** $f_{\mathrm{VPS}}$ falls strictly (1.003, 0.682, 0.477) and is below 0.5 at $N=10$, but it is 6.1× the uncertified localized screen ($f_S^*=0.079$), outside the 3× certificate allowance. The ratio grows with $N$ (3.3, 4.7, 6.1), so a larger chain would move it away from PASS. The VPS numbers are also optimistic: they use the exact dropped set, while the practical $\mathcal{O}(N^2)$ lookup over-counts.
* **Why the certificate is expensive (T0, checked by the data).** For a dropped pair with $e=2h\psi(\mathbf{n})\psi(\mathbf{n}')$, the compensator contributes at least $\lvert e\rvert$ for any $w$, so the certified excess is at least $\sum_{\mathrm{dropped}}(\lvert e\rvert-e)$. That is zero for frustrated pairs and $2\lvert e\rvert$ for sign-favourable ones. So a sign certificate pays twice the magnitude of every favourable dropped term, with no cancellation. Plain screening's signed bias cancels across terms, and the cancellation grows with $N$.
* **Side observation (C3, prior art, not a route).** Plain screening in the Löwdin basis has an extensive gain at fixed (uncertified) bias: $f=0.306$, 0.144, 0.079 against 0.89–0.91 in the canonical basis. This is the known locality argument; arXiv:2305.12776 already runs NQS in localized orbitals. Whether a trained NNBF stays as accurate in that basis has not been measured.
#### 7. Recommendation: STAY-CLOSED
* No route passed T1, so there is no value-of-information note and no T2 design. Results: C1 killed; C3′ AMBIGUOUS, failing the foil clause by a margin that grows with $N$; C2, C3 and C4 killed at T0. As pre-registered, no further T1 run is made.
* **What the review adds to the Saturation Diagnostic.**
    * **Statement 4 (F$_C$, sign certificate).** Any screened local energy kept variational by a diagonal positive-semidefinite compensator pays a certified excess of at least $\sum_{\mathrm{dropped}}(\lvert e\rvert-e)$. On H$_6$–H$_{10}$ that costs 3.3–6.1× the kept terms of uncertified screening at the same error, in the basis where screening works, and the ratio grows with $N$. With R5.3 (magnitude certificate, 140–1000× loose), both kinds of certificate have now been measured, and both lose to uncertified screening.
    * **Refinement of I5.** The saving that grows with $N$ belongs to the orbital frame, not to the estimator. Localized-basis screening is prior art. If P2.2 reopens, it should be on a different question: how accurate a trained NNBF is in a localized basis, relative to canonical, at matched cost. That question belongs to P2.1 (amplitude cost and accuracy), not to the local-energy estimator.
* **What would reopen P2.2.** A compensator whose excess goes below $\sum(\lvert e\rvert-e)$ on favourable terms. That needs off-diagonal (non-diagonal) PSD completions, for example $2\times2$ blocks kept in a low-rank form. Or a demonstration that $f_{\mathrm{VPS}}/f_S$ stops growing beyond $N=10$. Neither is predicted by the T0 floor above.
* **Compute.** About 16 CPU-minutes in this review, on the allocation already held by the session (job 451618). No `sbatch` and no `scancel`.

### R7.4 T1: SDP bound on the best kept-pattern compensator (2026-11-17)
* **Question and rule** (`tests/P2.2_stochastic_local_energy_r8/RULE.md`, written before any run). R7.4's diagonal compensator pays at least $\sum_{\mathrm{dropped}}(\lvert e\rvert-e)$ (Statement 4). A compensator that stays cheap can have off-diagonal entries only on pairs that are already kept. The test computes the best certified kept fraction any such compensator can reach: minimise $\psi_0^{\mathsf T}(C-H_{\mathrm{drop}})\psi_0$ subject to $C-H_{\mathrm{drop}}\succeq0$, with $C$ supported on the diagonal plus the kept pairs. Instance identical to the reopening-review T1: linear H$_6$, STO-3G, $R=3.0$ bohr, Löwdin orbitals, full $S_z=0$ sector (400 determinants), exact ground state $\psi_0$, $\tau=0.2$ mHa, the same 41-point grid and canonical-basis denominator. Rule: ratio $=f_{\mathrm{opt}}/f_S^*(6)$ with $f_S^*=0.306$; KILL R7.4 if the ratio is at least 3; PASS-T1 if below 3, which licenses a drafted H$_8$ job of at most 1 CPU-h. Stated in advance: the diagonal ratio at H$_6$ is already only 3.3, so a PASS at H$_6$ is weak evidence.
* **Result** (480 CPU-s, interactive, no Slurm; cvxpy in an isolated environment). The dual problem reduces exactly to a positive-semidefinite completion on the connected components of the kept graph; the kept graph is connected at every grid point up to $\varepsilon=0.1$ Ha, so the optimal certified excess is 0 everywhere. The full $400\times400$ primal at the screening threshold $\varepsilon^*=2.24\times10^{-3}$ Ha ($f=0.306$) certifies an excess of $2.3\times10^{-11}$ Ha after a PSD repair, with free entries up to 3.9 Ha (30 times the largest $\lvert h\rvert$). Kept fractions at $\tau$: uncertified SCREEN 0.306; VPS1 (product-weight diagonal) 1.003 (3.28 times); best diagonal floor $\sum(\lvert e\rvert-e)$ 0.596 (1.95 times); optimum 0.134 at the end of the grid (0.44 times). **Verdict: PASS-T1** (ratio 0.44 < 3). Reading: most of VPS1's 3.3-times penalty comes from its product-weight form, and the rest from being diagonal.
* **Robustness caveat (not part of the rule).** The optimum is tuned to $\psi_0$: its PSD slack $Y$ annihilates $\psi_0$. Evaluated on $\psi_0(1+\delta\xi)$ (median of 5 draws), the same $Y$ has excess $1.8\times10^{-6}$ Ha at $\delta=10^{-3}$ and $1.84\times10^{-4}$ Ha at $\delta=10^{-2}$, which is 0.92 of $\tau$ at $f=0.306$, so with a realistic 1% amplitude error the optimum is barely admissible at ratio 1. Building it also needs reference amplitudes on dropped configurations, so its cheapness is not established. An explicit triangle construction (`gadget.py`) is incomplete: 5924 of 20244 dropped pairs have no common kept neighbour, so its off-$\psi_0$ numbers are not a certificate.
* **Next step: the H$_8$ robustness test** (`voi.md`, drafted, not submitted). H$_8$ (4900 determinants), compensator built from $\psi_{\mathrm{ref}}=\psi_0(1+\delta\xi)$ at $\delta=10^{-2}$, 5 draws. KILL R7.4 if the robust kept fraction $f_{\mathrm{rob}}(8)\ge3f_S^*(8)=0.433$; PASS if below and the ratio to $f_S^*$ does not grow from H$_6$; AMBIGUOUS otherwise. Predicted KILL, $p\approx0.6$; cost about 1 CPU-h. **Blocked**: it needs a complete sparse PSD-completion construction (the full SCS primal does not scale to 4900 determinants), which another worker is writing in `tests/P2.2_stochastic_local_energy_r8/`; the job then goes through `compute_gate.py`. If no complete sparse construction exists, the test is not run and R7.4 is reported AMBIGUOUS-blocked.

### R7.4 H$_8$ robustness test with the path gadget (2026-11-17)
* **Rule** (`voi.md`, written before the run; interactive, about 160 CPU-s, below the 10 CPU-minute T1 line, so no Slurm job). H$_8$ linear chain, STO-3G, $R=3.0$ bohr, Löwdin orbitals, full $S_z=0$ sector (4900 determinants), exact $\psi_0$. Metric: $f_{\mathrm{rob}}$, the kept fraction (canonical-basis denominator) at the largest grid $\varepsilon$ whose median certified excess over 5 draws of $\psi_0(1+\delta\xi)$ at $\delta=10^{-2}$ is at most $\tau=0.2$ mHa. KILL R7.4 if $f_{\mathrm{rob}}(8)\ge3f_S^*(8)=0.433$; PASS if below and $f_{\mathrm{rob}}/f_S^*$ does not grow from H$_6$; AMBIGUOUS otherwise. Predicted KILL, $p\approx0.6$.
* **T0 construction that unblocked it** (`path_gadget.py`). Every positive-semidefinite slack that annihilates $\psi_0$ has the form $Y=D^{-1}LD^{-1}$, with $D=\mathrm{diag}(\psi_0)$ and $L$ a signed graph Laplacian on the kept graph; under relative noise $\psi_0(1+\delta\xi)$ the expected excess is exactly $\delta^2\,\mathrm{tr}\,L$. Routing each dropped element $e$ along a shortest kept path of $m$ edges gives $\mathrm{tr}\,L=\sum_{e>0}e+\sum_{e<0}(m^2-1)\lvert e\rvert$. The old H$_6$ SCS primal's 0.92$\tau$ came from a poor $Y$ ($\mathrm{tr}\,L\approx1.8$ Ha, 400 times worse). Checks: H$_6$ certified excess $\approx10^{-19}$ Ha, $Y\succeq0$ to $10^{-17}$, excess $4.5\times10^{-7}$ Ha at $f=0.306$ and $6.5\times10^{-6}$ Ha (0.03$\tau$) at $f=0.134$; H$_8$: 880,000 dropped pairs on paths of 2–13 edges, excess $1.0\times10^{-5}$ Ha (0.05$\tau$) at $\varepsilon=0.1$.
* **Result.** $f_{\mathrm{rob}}(8)=0.0602<0.433$, and $f_{\mathrm{rob}}/f_S^*=0.42$ at H$_8$ against 0.44 at H$_6$ (not growing): **PASS-T1** under the pre-registered metric. Prediction MISS.
* **Caveats** (recorded with the verdict). $f_{\mathrm{rob}}$ is set by the end of the $\varepsilon$ grid, not by robustness. The relative-noise model sees only $\mathrm{tr}\,L$; under additive noise of the same norm the same $Y$ has excess 0.66$\tau$ (H$_6$, grid end), 23$\tau$ (H$_8$, $\varepsilon=0.1$) and about 125$\tau$ (H$_8$, $\varepsilon=2.24\times10^{-3}$), because $Y_{ii}$ reaches $7\times10^4$–$5\times10^5$ on determinants where $\psi_0$ is small; the largest off-diagonal $\lvert Y\rvert$ is 1375 times $\max\lvert h\rvert$ at H$_8$ against 22 at H$_6$ (a variance cost). 16 determinants with $\psi_0=0$ by symmetry (5608 dropped pairs) were excluded. So the PASS holds only if NNBF errors are relative.

### R7.4 T1b on real trained states, and the T0 cost count (2026-11-17)
* **T1b rule** (`REAL_ERROR.md`, written before `real_error.py` ran; about 340 CPU-s interactive). NNBF-J (hidden 64, Jastrow hidden 32) regenerated on the exact R7.4 instance by full-sum L-BFGS (no stored trained states existed). Statistic: median over seeds of the excess $u^{\mathsf T}Lu$, $u=\psi_{\mathrm{NN}}/\psi_0$, at $\varepsilon_S=2.24\times10^{-3}$; a state votes only if $\Delta E<1.6$ mHa. KILL if above $\tau$ on any instance; PASS-T1b if at most $0.3\tau$ on every instance; AMBIGUOUS otherwise.
* **T1b result.** H$_6$, 3 seeds, $\Delta E\approx10^{-7}$ mHa: excess $4.5\times10^{-9}$ Ha (0.00$\tau$). H$_8$, 2 seeds, $\Delta E\le0.001$ mHa: $1.8\times10^{-6}$ Ha (0.01$\tau$). Grid end at most $3.4\times10^{-7}$ Ha. **PASS-T1b.** Addendum A1 (registered after the main runs, before its own run, labelled post hoc): an under-parameterised H$_8$ NNBF-J (916 parameters against 4900 determinants), $\Delta E=0.029$ mHa, excess $5.8\times10^{-6}$ Ha (0.03$\tau$): PASS. Diagnostic only: the first L-BFGS iterates with $\Delta E\le0.74$ mHa give 0.2–0.4$\tau$ (H$_6$, hidden 64), 0.33$\tau$ (H$_8$, hidden 4), and 9$\tau$ and 16$\tau$ (H$_8$, hidden 64, early), so the PASS relies on smooth converged states; VMC sampling noise was not tested. Real states sit between the two noise models: excess 25–1000 times the relative-model prediction and 30–1000 times below the additive one.
* **T0 cost count** (`opcount_H8.py`, H$_8$, $\psi_0^2$-weighted, 5 CPU-s). Per sample at $\varepsilon_S$: 359.6 connected terms, 26.7 kept, 332.9 dropped. EXACT needs about 360 network ratios and SCREEN about 27. The compensated estimator also needs about 27 network calls, but its compensator $Y$ is global: dropped elements are routed along kept paths of up to 5 edges at $\varepsilon_S$ (13 at the grid end), and the kept 3-ball of a sample holds about 1352 of the 4900 determinants. With a frozen NQS as the reference amplitude that is about 1349 reference evaluations plus 27, about 3.8 times EXACT. A cheap Jastrow–Hartree–Fock reference (about 0.1 times EXACT) has relative error $\delta\approx0.3$–1 (estimate, not measured), against the $\delta\le\sqrt{\tau/\mathrm{tr}\,L}=0.17$ ($\varepsilon_S$) or 0.045 (grid end) the bound needs, so its excess would be 3–35$\tau$ (estimate). HF or CAS references vanish on most determinants, which forces the diagonal floor (0.596, 1.95 times plain screening at H$_6$). $Y\succeq0$ holds for any nonzero reference, so the bound stays valid; its size is set by how much $\psi_\theta/\psi_{\mathrm{ref}}$ varies along the kept paths.
* **Verdict (orchestrator, 2026-11-17): R7.4 KILLED at T0 (cost).** The certified saving does not survive any practical reference: the compensator's reference work is global (a precomputed $L$ exists only at ED size), a frozen-NQS reference costs 3.8 times EXACT, and a cheap reference is estimated (not measured) to break the $\delta$ bound by 2–20 times. The accuracy side (PASS-T1b) is recorded on the card under what the kill does not rule out. With R5.2 SHELVED, every reopening-ledger row is closed.

## Route Register
One card per route ever generated in this leaf, written in the 2026-11-17 documentation pass from the round texts above; ids follow the leaf's own labels (Route 1–4 of round 1 are R1.1–R1.4; the closed-leaf falsifier route is R6.1; the reopening-review candidates C1–C4 are R7.1–R7.5). Shared vocabulary, used in every card: $E_L(\mathbf{n})=H_{\mathbf{n}\mathbf{n}}+\sum_jx_j$ is the local energy of a sample $\mathbf{n}$, with $x_j=H_{\mathbf{n}\mathbf{n}_j}\psi(\mathbf{n}_j)/\psi(\mathbf{n})$ over the $\lvert\mathcal{C}\rvert=\mathcal{O}(N^2M^2)$ connected configurations ($N$ electrons, $M$ spin-orbitals, $K=M/2$ spatial orbitals). NNBF is a neural-network backflow determinant, NNBF-J (NNBFJ in tables) the same times a Jastrow factor. EXACT enumerates every term (one forward pass each). SCREEN drops terms with $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert\le\varepsilon$ (biased, admissible only if its bias is at most $\tau=0.2$ mHa). SS (semistochastic) evaluates large terms exactly and samples the tail; HB (heat-bath) samples with $p\propto\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$. A route's figure of merit $M_X$ is its cost to a fixed error bar (variance times cost per sample) divided by that of the best control on the same state; $M_X<1$ means the route is cheaper. $\Delta E$ is the energy error against exact diagonalization (ED). The residual fraction $\rho_y$ of a surrogate $y$ is $\mathbb{E}\sum_j\lvert x_j-y_j\rvert/\mathbb{E}\sum_j\lvert x_j\rvert$. Kill codes are those of SKILL.md.

### R1.1 — Heat-bath / semistochastic stochastic local energy (Route 1) (KILLED)
* **The idea**: Evaluate exactly the connected terms with $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert>\varepsilon$ and estimate the remaining tail from $K_d$ draws with probability proportional to $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert$, so each sample costs far fewer forward passes than EXACT. It transfers the semistochastic and heat-bath ideas of selected configuration interaction (SHCI) perturbation theory to the VMC local energy, expecting that the tail is small and cheap to sample.
* **Lineage**: initial batch, round 1 (2026-11-10). No successor; its failure mechanism (the single-proposal floor) became the anti-solution space for every later sampled route.
* **Outcome**: KILLED at G0 (2026-11-10); it also failed Gates 1–3.
* **Kill rule**: Gate 0: "is it known, and is the reason it is not standard practice fixed?"; Gate 1 in addition asked whether its cost relative to EXACT stays bounded as $\psi\to\psi_0$.
* **Evidence**: Prior art: semistochastic NQS local energies (arXiv:2507.19276, 2509.25720, 2604.06841). Derivation (section 1): any single importance proposal has per-draw variance at least $(\sum_j\lvert x_j\rvert)^2-(\sum_jx_j)^2$, nonzero for a mixed-sign row. Measurement (section 4 pre-check, H$_6$ chain, STO-3G, $R=3.0$ bohr, state 0.17 mHa from the ground state): HB costs $657\times$ EXACT; SS $0.932\times$ against SCREEN $0.911\times$.
* **Why this kills it**: The method is already in use, and its known adoption barrier (tail variance that does not vanish at convergence) is untouched; near convergence its deterministic part does all the work, so plain screening matches it.
* **What the kill does not rule out**: semistochastic schemes with a strong control variate on the tail (tested as R1.4 and R2.0), or multi-proposal samplers (R2.1); poorly converged states, where the floor is hidden by $\mathrm{Var}_\pi(E_L)$.
* **Strongest case against the kill**: The floor holds only for a single proposal; positivisation can remove it (round-1 critic, Owen & Zhou 2000). Answered: the kill stands on Gate 0 alone, and positivisation was tested separately as R2.1.
* **Checked by**: round-1 critic re-audit (master Cross-Pollination, 2026-11-11): "P2.2-R1 dies at Gate 0 (prior art)".
* **Lesson**: L12 (`library.md`)

### R1.2 — Doubly stochastic SR force (Route 2) (KILLED)
* **The idea**: Use a one-draw sampled local energy only inside the stochastic-reconfiguration (SR) force $f_k=2\,\mathrm{Cov}_\pi(O_k,E_L)$, which is linear in $E_L$ and therefore stays unbiased, and use exact $E_L$ only for the final energy. It borrows doubly stochastic gradients from machine learning, expecting cheap unbiased forces to drive the optimization.
* **Lineage**: initial batch, round 1; cross-field translation from doubly stochastic gradients in kernel machine learning. No successor.
* **Source analogy**: kernel methods with random features (Dai et al., NeurIPS 2014) sample both data and features and still converge with decaying steps, because the problem is convex. Correspondence: random feature ↔ sampled connected term. Gap: here the inner noise inherits the sign floor and is amplified by $S^{-1}$, the inverse SR metric.
* **Outcome**: KILLED at G1 (2026-11-10).
* **Kill rule**: Gate 1: does the method's cost stay bounded as $\psi\to\psi_0$?
* **Evidence**: Derivation: the force noise $\mathbb{E}_\pi[\tau^2\lvert O_k\rvert^2]/K_d$ ($\tau^2$ the per-draw variance, $O_k=\partial_{\theta_k}\ln\psi$) does not vanish at convergence; $S^{-1}$ amplifies it along small-eigenvalue directions; at fixed step the stationary energy error is set by this floor.
* **Why this kills it**: Reaching 1.6 mHa then needs step-size decay, and the cost to target scales with the floor rather than with the vanishing exact variance, so the saving per step is lost.
* **What the kill does not rule out**: step-limited, far-from-converged optimization, where noise is not the bottleneck (the regime later probed by R5.2).
* **Strongest case against the kill**: not raised by a critic: the argument is asymptotic, and at finite target accuracy the cheaper force might still win; no measurement was made.
* **Checked by**: round-1 critic re-audit (2026-11-11): "P2.2-R2 dies on noise amplification by $S^{-1}$".
* **Lesson**: L12 (`library.md`)

### R1.3 — Factorized-Hamiltonian (Cholesky) proposal (Route 3) (KILLED)
* **The idea**: Write the two-electron integrals as $V_{pqrs}=\sum_LL^L_{pr}L^L_{qs}$ (Cholesky or double factorization) and draw connected terms with probability proportional to the factor magnitudes, a cheap proposal that follows the Hamiltonian's low-rank structure instead of $\lvert H\rvert$ element by element.
* **Lineage**: initial batch, round 1. The factorization reappears as an exact contraction in R2.8, R3.6′ and R4.0.
* **Outcome**: KILLED at G2 (2026-11-10).
* **Kill rule**: Gate 2: does a cheap baseline reach the same metric?
* **Evidence**: It is still one proposal $p(\mathbf{n}'\mid\mathbf{n})$ with one forward pass per draw, so it is bounded below by the oracle proposal $p\propto\lvert x\rvert$. Measured on H$_6$, $R=3.0$, at 0.17 mHa: the oracle proposal itself costs $145.6\times$ EXACT, while screening costs $0.911\times$.
* **Why this kills it**: No proposal can beat the oracle, and the oracle loses to exact enumeration with screening by more than two orders of magnitude.
* **What the kill does not rule out**: the factorization used as an exact contraction rather than as a proposal (R2.8, R4.0).
* **Strongest case against the kill**: not raised by a critic: the comparison is on one small instance ($\lvert\mathcal{C}\rvert\approx118$); answered by the oracle bound, which holds for any instance at convergence.
* **Checked by**: round-1 critic re-audit (2026-11-11), which lists P2.2-R3 under the single-proposal anti-pattern.
* **Lesson**: L12 (`library.md`)

### R1.4 — Frozen-backflow semistochastic control variate, SCV (Route 4) (KILLED)
* **The idea**: For a determinant NQS $\psi(\mathbf{n})=\det\Phi(\mathbf{n})_{[\mathrm{occ}\,\mathbf{n}]}$, freeze the backflow orbitals at the sample, $\tilde r(\mathbf{n}')=\det\Phi(\mathbf{n})_{[\mathrm{occ}\,\mathbf{n}']}/\psi(\mathbf{n})$. Every frozen ratio is a cheap Green's-function minor, so the full surrogate sum is computed for almost nothing; only the residual $H(r-\tilde r)$ is sampled. The surrogate carries the network's own sign structure, so the floor should apply only to a small residual.
* **Lineage**: initial batch, round 1. Revived in round 2 as R2.0; its surrogate was repaired as R2.3 (linearized), R3.0 (tangent), R3.3′ (calibrated) and R3.7′/R3.2 (partially frozen).
* **Outcome**: KILLED at CRITIC (2026-11-11).
* **Kill rule**: round-1 critic item 12: is the construction new, and is the residual fraction "well below 1"? The worker's own pre-registered FAIL clause was $M\ge1$.
* **Evidence**: Residual fraction $\rho_{\mathrm{FB}}=0.56$–0.82 on H$_4$ and 0.69–0.80 on H$_6$ ($R=3.0$); the surrogate alone is biased by 100–180 mHa. Pre-check $M=1.07$ (H$_4$) and $M=1.03$ (H$_6$ at 0.17 mHa) against the best of HB, SS and SCREEN; $M=0.73$ only at 13 mHa.
* **Why this kills it**: Energy optimization puts the multireference correlation into the configuration dependence of the orbitals, which the frozen surrogate discards, so the residual stays large and the route is no cheaper than its controls; the construction itself is the standard semistochastic-plus-control-variate one.
* **What the kill does not rule out**: larger bases where the added orbitals couple weakly to the backflow (R2.0 found $\rho_{\mathrm{FB}}$ falling to 0.39 at $M=32$), and better surrogates (R2.3).
* **Strongest case against the kill**: the instances ($M\le16$, assumed per-term cost $\kappa=10^{-3}$) were too small to show scaling in $M$; round 2 revived it on exactly this ground (R2.0). That revival was itself ruled illegitimate by the round-2 critic.
* **Checked by**: round-1 per-leaf critic (item 12, 2026-11-11).
* **Lesson**: L8 (`library.md`)

### R2.0 — Revived frozen-backflow control variate at measured per-term cost (KILLED)
* **The idea**: Re-run R1.4 on bases up to $M=48$ with the per-term cost $\kappa$ of each kernel measured rather than assumed, arguing that the full frozen-surrogate sum costs less than one forward pass per 270 terms at chemistry scale, so the method's cost relative to EXACT falls as the basis grows.
* **Lineage**: revival of R1.4 under the Stage-4 re-loop rule, round 2 (2026-11-11). Not revived again; its frozen ratio survives only as a diagnostic.
* **Outcome**: KILLED at CRITIC (2026-11-11).
* **Kill rule**: round-2 per-leaf critic: is the revival legitimate, and does the PASS rest on a consistent cost model?
* **Evidence**: Pre-check (H$_4$ ring, $R=2.8$): $C/C_{\mathrm{EXACT}}=0.45$, 0.27, 0.10 at $M=16$, 24, 32, and $M_{\mathrm{R2.0}}=0.90$, 0.59, 0.60 against the best non-SCV control. But $\kappa$ came from $N=20$, $M=80$ while variances came from $N=4$ (a hybrid model), and $M_{\mathrm{R2.0}}$ was flat between $M=24$ and 32.
* **Why this kills it**: The revival brought no new mechanism, and the falling ratio came from mixing a chemistry-scale cost with a toy-scale variance, so the original round-1 kill stands.
* **What the kill does not rule out**: a frozen-surrogate control variate measured consistently at one $N$, and the large-$M$ regime itself.
* **Strongest case against the kill**: a real 1.7-fold lead over every non-SCV control at $M=24$–32; unresolved, because no consistent single-instance test was ever run for this exact route.
* **Checked by**: round-2 per-leaf critic (2026-11-11).
* **Lesson**: L51 (`library.md`)

### R2.1 — Positivised (sign-stratified) semistochastic local energy, P2.2′ (KILLED)
* **The idea**: Split the sampled tail by the sign of a cheap surrogate (the frozen-backflow ratio) and sample each sign stratum with its own proposal. With two proposals, the single-proposal floor $(\sum\lvert x\rvert)^2-(\sum x)^2$ no longer applies; the variance should fall to the small sign-mismatch weight between surrogate and true terms.
* **Lineage**: proposed by the round-1 critic as P2.2′ (cross-field translation of positivisation from Monte Carlo statistics), generated in round 2 as Operation A.1 on R1.4. No successor; it closed P2.2′ in the master.
* **Source analogy**: Owen & Zhou (JASA 95, 135, 2000) integrate a signed function with separate proposals for its positive and negative parts, reaching zero variance when each part is proportional to its proposal. Correspondence: positive and negative parts ↔ sign strata of $y_{\mathrm{FB}}$; proposal ↔ $\lvert y\rvert$. Why it worked there: the proposals match the magnitudes. Gap: here the surrogate magnitudes are poor.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2: does a cheaper baseline (its own single-proposal ablation) reach the same metric?
* **Evidence**: H$_4$ ring, $M=16$, 24, 32: positivised cost $0.440$, $0.262$, $0.1035$ of EXACT against $0.453$, $0.264$, $0.1053$ for one proposal $\propto\lvert y_{\mathrm{FB}}\rvert$ (within 1–3%). With true signs (ORACLE_SIGN) the cost equals that with no sign information to 0.3%. Sign-mismatch weight 0.04–0.17; residual fraction 0.39–0.79.
* **Why this kills it**: The stratum variance is $Z_s^2\mathrm{Var}_{p_s}(x/\lvert y\rvert)$, zero only if magnitudes are proportional within a stratum; the measured floor is magnitude dispersion, which sign splitting cannot touch.
* **What the kill does not rule out**: surrogates with accurate magnitudes but frequent sign errors; instances where the mismatch weight is large.
* **Strongest case against the kill**: the instance is small ($N=4$, s-only basis); the mismatch weight might grow in the multireference regime at larger $N$. Unresolved, but the oracle-sign test bounds the gain on these instances.
* **Checked by**: round-2 per-leaf critic (2026-11-11) and the master's round-2 anti-pattern entry.
* **Lesson**: route-specific — sign-aware sampling failed because this surrogate's magnitude error, not its sign error, dominated on these molecules; that is a property of the surrogate, not a general mechanism.

### R2.2 — Epstein–Nesbet perturbative tail proposal, PT2IS (KILLED)
* **The idea**: Partition connected terms into an exact P-space and a sampled Q tail, with the tail proposal $\propto\lvert H_{\mathbf{n}j}^2/(E-H_{jj})\rvert$, the Epstein–Nesbet (EN) first-order estimate of each term computed from integrals only in $\mathcal{O}(1)$. It needs no amplitude to propose, works for any architecture, and should match high-virtual terms, which are sign-coherent.
* **Lineage**: round 2, Operation B.3, cross-field translation from selected-CI perturbation theory. Its EN magnitudes survive as the deterministic completion control COMP_EN.
* **Source analogy**: in SHCI and EN-PT2, $H_{\mathbf{n}j}^2/(E-H_{jj})$ estimates the first-order amplitude of external determinants (arXiv:1610.06660). Correspondence: external determinant ↔ connected $\mathbf{n}_j$. Why it worked there: the reference dominates and other parents interfere weakly. Gap: in the multireference regime other parents interfere at order one.
* **Outcome**: KILLED at G3 (2026-11-11).
* **Kill rule**: Gate 3: does the route survive with the Hard Core (strong multireference character) present?
* **Evidence**: interference ratio $\eta_j=\mathcal{O}(1)$; residual fraction $\rho_{\mathrm{EN}}=0.88$–0.89 at every $M$ (H$_4$ ring, HF weight 0.44); $M_{\mathrm{R2.2}}=1.18$, 1.78, 2.82 at $M=16$, 24, 32.
* **Why this kills it**: The proposal is poor exactly when correlation is strong, and the route gets worse with basis size.
* **What the kill does not rule out**: EN magnitudes as a biased deterministic completion (COMP_EN was the strongest control), and weakly correlated systems.
* **Strongest case against the kill**: not raised by a critic: one geometry family at $N=4$; the measured trend is monotone against it.
* **Checked by**: round-2 per-leaf critic (2026-11-11), kills accepted in round 3.
* **Lesson**: route-specific — an integral-only proposal ignores the wavefunction, so it fails wherever the reference does not dominate.

### R2.3 — Linearized-backflow surrogate (LRX) summed over all terms (KILLED)
* **The idea**: Repair the frozen surrogate by forming the two changed backflow rows exactly and linearizing the response of the other rows in adjugate form, giving a surrogate accurate to second order in the backflow change; sum it over all connected terms and sample the residual.
* **Lineage**: round 2, Operation A on R1.4's failed component (the surrogate). Mutated into R2.4.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2: does a cheaper baseline (the frozen control variate) reach the same cost?
* **Evidence**: best surrogate measured, $\rho_{\mathrm{LRX}}=0.03$–0.10, sign mismatch 0.001–0.005; but per-term cost $\kappa_{\mathrm{LRX}}=0.21$ forward passes at chemistry scale, so the frozen control variate beats it by 1.5–2.7 (H$_4$ ring, $M=16$–32).
* **Why this kills it**: Summing an accurate surrogate over all $\lvert\mathcal{C}\rvert$ terms costs a fifth of EXACT before any residual is sampled.
* **What the kill does not rule out**: evaluating LRX only on drawn terms (R2.4).
* **Strongest case against the kill**: $\kappa_{\mathrm{LRX}}$ was a Python timing; a compiled kernel could be cheaper. Unresolved, but R2.4 used LRX more cheaply and still failed.
* **Checked by**: round-2 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — an accurate surrogate is useful only if its per-term cost is far below one forward pass.

### R2.4 — Multilevel Monte Carlo over surrogate fidelity, FB to LRX to exact (KILLED)
* **The idea**: A three-level telescoping estimator: frozen surrogate summed exactly, then a sampled LRX-minus-frozen correction, then a sampled exact-minus-LRX correction. Most draws go to the cheap middle level, and the expensive exact level sees only the small LRX residual.
* **Lineage**: round 2, Operation B.3, cross-field translation of multilevel Monte Carlo; mutation of R2.3.
* **Source analogy**: Giles' multilevel Monte Carlo (Oper. Res. 56, 607, 2008) estimates an SDE expectation by telescoping over time-step levels whose differences shrink geometrically. Correspondence: discretization level ↔ surrogate fidelity. Gap: only three levels, and the LRX setup costs 12.4 forward passes per sample.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2 against its control (R2.0, SCV), with the round-2 FAIL clause $M\ge1$; the round-2 critic ruled that a route failing its own pre-check is not a survivor.
* **Evidence**: $M_{\mathrm{R2.4}}=1.08$, 1.03, 1.07 at $M=16$, 24, 32 (H$_4$ ring).
* **Why this kills it**: It is never cheaper than the plain frozen control variate it refines.
* **What the kill does not rule out**: larger $N$, where the setup amortizes over more terms (per-term LRX saving $\sim5/N$).
* **Strongest case against the kill**: the gain was predicted to appear only at larger $N$; unresolved, never tested.
* **Checked by**: round-2 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — a per-sample setup cost dominates when the hierarchy has few levels.

### R2.5 — Cost-aware cross-sample allocation of residual draws (KILLED)
* **The idea**: Give each sample a number of residual draws proportional to its surrogate tail mass $Z_{\mathbf{n}}$, known before any amplitude is evaluated, a cheap proxy for Neyman allocation; spending draws where the residual is large should lower total variance at fixed cost while staying unbiased.
* **Lineage**: round 2, Operation B.2 (active probing); the critic identified it as Neyman allocation from survey sampling.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2 against its control (SCV), FAIL clause $M\ge1$ applied to its own pre-check by the round-2 critic.
* **Evidence**: $M_{\mathrm{R2.5}}=1.02$, 1.00, 0.98 at $M=16$, 24, 32; the possible gain is bounded by $\mathbb{E}[\tau^2]/(\mathbb{E}\tau)^2$, which is near 1 when the per-sample residual is not heavy-tailed.
* **Why this kills it**: The per-sample residual is too uniform for allocation to matter.
* **What the kill does not rule out**: instances with heavy-tailed per-sample residuals.
* **Strongest case against the kill**: the 0.98 at $M=32$ is a falling trend; unresolved, but a 2% gain cannot meet any PASS threshold.
* **Checked by**: round-2 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — allocation gains need heavy-tailed variance across samples.

### R2.6 — Active-space-restricted backflow input (KILLED)
* **The idea**: Let the backflow see only an active orbital set, so that excitations outside it have exact cheap ratios and the local energy stays exact while costing arithmetic instead of forward passes.
* **Lineage**: round 2, Operation B.1 (frame change). Repaired in round 3 as R3.1 (WIBJ) by adding all-orbital rows and an all-orbital Jastrow.
* **Outcome**: KILLED at G3 (2026-11-11); also Gate 0.
* **Kill rule**: Gate 3: does it keep the Hard Core's correlation?
* **Evidence**: Argument: for a fixed active configuration the state is a single determinant in the external orbitals, with no pair correlation into external virtuals, which dominates the correlation energy as the basis grows; in that space it is the mean-field foil. (The Gate 0 citation arXiv:2302.11588 was corrected in round 3: it is orbital-rotation NQS, not backflow.)
* **Why this kills it**: It buys cheapness by removing external correlation, so it cannot reach chemical accuracy in growing bases.
* **What the kill does not rule out**: restricted backflow with an all-orbital Jastrow and all-orbital rows (R3.1).
* **Strongest case against the kill**: not raised by a critic: the argument is qualitative; R3.1 and R5.1 later measured the loss, which grew with $K$.
* **Checked by**: round-2 per-leaf critic (2026-11-11); Gate 0 citation corrected in round 3.
* **Lesson**: L8 (`library.md`)

### R2.7 — Row sketching by randomized linear algebra (KILLED)
* **The idea**: Write $E_L-H_{\mathbf{n}\mathbf{n}}=\langle h,r\rangle$ (Hamiltonian row times amplitude ratios) and approximate the inner product by a random sketch $\langle Sh,Sr\rangle$, as randomized numerical linear algebra does for large inner products.
* **Lineage**: round 2, Operation B.3, cross-field translation from randomized numerical linear algebra.
* **Source analogy**: approximate matrix products by sampling or sketching coordinates (Drineas, Kannan & Mahoney, SIAM J. Comput. 36, 132, 2006). Correspondence: vector coordinates ↔ amplitude ratios. Why it worked there: both vectors are cheap to read in full. Gap: each coordinate of $r$ costs a forward pass.
* **Outcome**: KILLED at G1 (2026-11-11).
* **Kill rule**: Gate 1: does it beat the query and variance limits?
* **Evidence**: Argument: a dense sketch $Sr$ needs every amplitude; a sparse sketch is single-proposal sampling, bounded by the R1.1 floor.
* **Why this kills it**: Both versions reduce to known losers (EXACT or single-proposal sampling).
* **What the kill does not rule out**: sketches of structured amplitude vectors that can be formed without per-coordinate queries.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-2 per-leaf critic (2026-11-11), kills accepted in round 3.
* **Lesson**: route-specific — the source field's assumption of free coordinate access fails.

### R2.8 — Exact Cholesky/THC contraction of the frozen surrogate sum (KILLED)
* **The idea**: Remove the per-term floor of the frozen surrogate sum by contracting $\sum_{pqrs}V_{pqrs}(G_{pr}G_{qs}-G_{ps}G_{qr})$ through $N_\gamma$ Cholesky vectors in $\mathcal{O}(N_\gamma MN)$, where $G$ is the frozen Green's function, instead of enumerating $\mathcal{O}(N^2M^2)$ terms.
* **Lineage**: round 2, Operation A on R1.3. Kept as an implementation detail of R2.0; reappears as R3.6′ and R4.0.
* **Outcome**: KILLED at G0 (2026-11-11).
* **Kill rule**: Gate 0: is it known, and is its adoption barrier fixed?
* **Evidence**: it is the standard auxiliary-field QMC local energy (Motta & Zhang, arXiv:1711.02242).
* **Why this kills it**: As a route it adds nothing new; it only accelerates the surrogate part of another route.
* **What the kill does not rule out**: its use as an implementation inside a control variate.
* **Strongest case against the kill**: not raised by a critic; the kill is a novelty ruling, not a performance one.
* **Checked by**: round-2 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — a known technique, kept as an implementation.

### R2.9 — Multilevel Monte Carlo across basis-set levels (KILLED)
* **The idea**: Use the local energy of the same network in a coarser orbital basis as the cheap level of a multilevel estimator, correcting it with sampled fine-basis differences, so most work happens in the small basis.
* **Lineage**: round 2, Operation B.3, cross-field translation of multilevel Monte Carlo (second variant, after R2.4).
* **Source analogy**: Giles' multilevel Monte Carlo (Oper. Res. 56, 607, 2008) with discretization levels. Correspondence: mesh level ↔ basis-set level. Gap: a coarse-basis local energy of a fine-basis state is just a truncation of $\mathcal{C}$.
* **Outcome**: KILLED at G2 (2026-11-11); also Gate 0.
* **Kill rule**: Gate 2: does an existing control already cover it?
* **Evidence**: Argument: the coarse level is semistochastic sampling with an orbital-energy deterministic set, which the SS arm already measures.
* **Why this kills it**: It is a relabelled control.
* **What the kill does not rule out**: hierarchies whose coarse level is a genuinely different, correlated model.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-2 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — the coarse level coincides with a control.

### R3.0 — Tangent-Slater surrogate (KILLED)
* **The idea**: Dress the frozen ratio with the backflow's first-order response to the occupation change, $\tilde r\,e^{2(g_a+g_b-g_i-g_j)}$ with $g=\partial_{\mathbf{x}}\ln\lvert\det\Phi(\mathbf{x})_{[\mathrm{occ}\,\mathbf{n}]}\rvert$ from one backward pass; the dressed surrogate is still contractible like the AFQMC local energy, and should capture the configuration dependence the frozen one misses.
* **Lineage**: round 3, Operation A on R1.4's surrogate.
* **Outcome**: KILLED at G1 (2026-11-11).
* **Kill rule**: Gate 1: is the surrogate well-defined and finite near nodes?
* **Evidence**: $g\propto\mathrm{tr}(A^{-1}\partial A)$ diverges at the nodes of $A=\Phi(\mathbf{n})_{[\mathrm{occ}\,\mathbf{n}]}$; the exponential form overflows; the linearized form has $\rho=1.26$ (frozen: 0.59) and $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)=4\times10^9$ (H$_4$, $M=24$).
* **Why this kills it**: The surrogate is worse than the one it repairs.
* **What the kill does not rule out**: surrogates whose correction is bounded near nodes (adjugate forms, as in LRX).
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-3 per-leaf critic (2026-11-11), no objection.
* **Lesson**: route-specific — log-derivative dressings of determinants diverge at nodes.

### R3.1′ — Cholesky sum-of-squares energy estimator (Jackson–Feenberg form) (KILLED)
* **The idea**: Write $\hat H=\hat h'+\tfrac12\sum_\gamma\hat L_\gamma^2$ with one-body Cholesky operators $\hat L_\gamma$; by Hermiticity $E=\mathbb{E}_\pi[\hat h'_{\mathrm{loc}}+\tfrac12\sum_\gamma\ell_\gamma^2]$, where $\ell_\gamma=(\hat L_\gamma\psi)/\psi$ needs only the $\mathcal{O}(NM)$ single excitations, not the $\mathcal{O}(N^2M^2)$ doubles. (In round 4 the leaf reuses the label "R3.1′" for WIBJ; that is R3.1 here.)
* **Lineage**: round 3, Operation B.3, cross-field translation of the Jackson–Feenberg kinetic-energy estimator from continuum QMC.
* **Source analogy**: in continuum VMC the kinetic energy can be estimated as $\tfrac12\lvert\nabla\ln\psi\rvert^2$ instead of $-\tfrac12\nabla^2\psi/\psi$ by integrating by parts (Jackson–Feenberg identity; used e.g. in arXiv:1604.05804). Correspondence: gradient ↔ one-body operator $\hat L_\gamma$; integration by parts ↔ Hermiticity. Why it worked there: the wavefunction's support is closed under the gradient. Gap: $\hat L_\gamma\psi$ is nonzero where $\psi$ vanishes.
* **Outcome**: KILLED at G1 (2026-11-11).
* **Kill rule**: Gate 1: is the estimator's variance finite and vanishing at convergence?
* **Evidence**: The variance contains $\sum_{\mathbf{n}}(\hat L\psi)^4/\psi^2$; measured $\mathrm{Var}/\mathrm{Var}(E_L)$ from $6.5\times10^6$ to $8\times10^{14}$, and $2\times10^{34}$ for the exact $\psi_0$ (H$_4$). Importance sampling from $\psi^2+\tfrac12\sum_\gamma(\hat L_\gamma\psi)^2$ makes it finite but loses the zero-variance principle.
* **Why this kills it**: The identity is exact but its estimator is useless; the round-3 critic restated the mechanism as the support condition $\mathrm{supp}(L_\gamma\psi)\subseteq\mathrm{supp}(\psi)$, which fails.
* **What the kill does not rule out**: sum-of-squares rewritings for operators that preserve the support of $\psi$.
* **Strongest case against the kill**: the round-3 critic rejected the worker's "infinite variance" framing as too broad; answered by adopting the support condition.
* **Checked by**: round-3 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — the support condition $\mathrm{supp}(\hat L\psi)\subseteq\mathrm{supp}(\psi)$ is a known requirement of sum-of-squares estimators, and this construction violates it.

### R3.2′ — Cross-sample multifidelity correction (KILLED)
* **The idea**: Compute a cheap local energy $E^S_L$ on every sample and the exact one only on a random subset, combining them as $\bar E^S_B+\overline{(E_L-E^S_L)}_{B'}$; no within-row sampling happens, so the magnitude dispersion that killed R2.1 never enters.
* **Lineage**: round 3, Operation B.3, cross-field translation of multifidelity Monte Carlo.
* **Source analogy**: Peherstorfer, Willcox & Gunzburger (SIAM J. Sci. Comput. 38, A3163, 2016) debias a cheap model's statistics with occasional high-fidelity runs. Correspondence: high-fidelity model ↔ exact $E_L$; low-fidelity ↔ surrogate local energy. Why it worked there: low and high fidelity are strongly correlated across inputs. Gap: here the surrogate error depends strongly on the sample.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2: does it beat exact enumeration on its own pre-check?
* **Evidence**: H$_4$, $M=24$: $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)=466$ (frozen), 19.5 (LRX), 2.3–48 (screened completions); best multifidelity cost ratio 0.96 (orbital-window head, $M=32$); regression $R^2$ of $E_L$ on all cheap features 0.13.
* **Why this kills it**: The correction variance exceeds the variance it is meant to save.
* **What the kill does not rule out**: a surrogate whose error correlates with $E_L$ across samples (the window-head ratio fell from 1.59 to 0.96 between $M=24$ and 32).
* **Strongest case against the kill**: that falling window-head trend; unresolved, never followed.
* **Checked by**: round-3 per-leaf critic (2026-11-11), no objection.
* **Lesson**: L25 (`library.md`)

### R3.3′ — Row-calibrated surrogate (KILLED)
* **The idea**: In each row, fit per-orbital correction factors to the frozen surrogate on the exactly evaluated head terms, then use the calibrated surrogate as the control variate on the tail; fitting only on the head keeps the estimator unbiased.
* **Lineage**: round 3, Operation A on R1.4's surrogate.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2: does it reduce the residual relative to the uncalibrated control?
* **Evidence**: tail residual fraction rose from 1.07 (frozen) to 1.26 and 1.46 for the two factor forms (H$_4$, $M=24$).
* **Why this kills it**: Head-fitted factors extrapolate worse than none.
* **What the kill does not rule out**: calibrations fitted across many rows (which would become R3.2′-type models).
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-3 per-leaf critic (2026-11-11), no objection.
* **Lesson**: route-specific — head and tail terms differ systematically.

### R3.4′ — Amplitude caching across the batch or Markov chain (KILLED)
* **The idea**: Cache amplitudes of connected configurations and reuse them across samples in a batch or along the Markov chain, since neighbouring samples share connected configurations; each shared amplitude would then be paid for once instead of once per sample.
* **Lineage**: round 3, the brief's amortization idea.
* **Outcome**: KILLED at G2 (2026-11-11); also Gate 0.
* **Kill rule**: Gate 2: does the reuse fraction stay finite as the system grows?
* **Evidence**: Argument: two configurations at excitation distance 2 share $\mathcal{O}(NM)$ of $\mathcal{O}(N^2M^2)$ connected configurations, and $\mathcal{O}(1)$ at distance 4; on ED-sized instances the union saturates at the Hilbert-space dimension, so measured savings there are finite-size artefacts. Prior art: deduplication in NNQS-Transformer (arXiv:2306.16705).
* **Why this kills it**: The overlap fraction vanishes as $1/(NM)$.
* **What the kill does not rule out**: caching for very small systems or very correlated chains.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-3 per-leaf critic (2026-11-11), no objection.
* **Lesson**: route-specific — overlap counting.

### R3.5′ — Horvitz–Thompson (Poisson or priority) sampling of terms (KILLED)
* **The idea**: Include each connected term independently with probability $\pi_j$ and weight it by $1/\pi_j$ (Poisson or priority sampling), which removes the multinomial cross-term and might lower variance.
* **Lineage**: round 3, cross-field translation from survey sampling.
* **Source analogy**: Horvitz & Thompson (JASA 47, 663, 1952) estimate population totals with unequal inclusion probabilities. Correspondence: unit ↔ connected term. Gap: the variance keeps the magnitude dispersion.
* **Outcome**: KILLED at G2 (2026-11-11).
* **Kill rule**: Gate 2: does it beat multinomial sampling?
* **Evidence**: Derivation: variance $\sum_jx_j^2(1/\pi_j-1)$, bounded by the same magnitude dispersion; the removed cross-term is negligible for many small terms.
* **Why this kills it**: No variance gain over R2.1 or single-proposal sampling.
* **What the kill does not rule out**: none within the black-box model; this form became Statement 2 of the Saturation Diagnostic.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-3 per-leaf critic (2026-11-11), no objection.
* **Lesson**: route-specific — absorbed into the leaf's Statement 2.

### R3.6′ — Exact low-rank local energy (KILLED)
* **The idea**: Answer the brief's question: compute the exact local energy at reduced cost through the low-rank (Cholesky or tensor-hypercontraction) structure of the integrals, avoiding enumeration of the doubles.
* **Lineage**: round 3, the brief's question; relative of R2.8 and R4.0.
* **Outcome**: KILLED at G1 (2026-11-11).
* **Kill rule**: Gate 1: does an exact contraction exist for a configuration-dependent backflow?
* **Evidence**: It exists only when the orbitals do not depend on $\mathbf{n}$ (the AFQMC local energy, arXiv:1711.02242); for unrestricted backflow P2.1's F1 bound gives $\Omega(N^2)$ work per connected amplitude.
* **Why this kills it**: The structure it needs is absent in NNBF.
* **What the kill does not rule out**: architectures with configuration-independent orbitals (tested as R6.1).
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: round-3 per-leaf critic (2026-11-11), no objection.
* **Lesson**: route-specific — the answer to the brief's question is the AFQMC local energy.

### R3.7′ — Window-exact partially-frozen control variate on near-exact states (MERGED)
* **The idea**: Freeze only the backflow input from orbitals outside a window, so the surrogate is exact for window-internal excitations, and use it as a stand-alone control variate. On H$_4$ ($M=24$, $\Delta E=0.05$ mHa) it gave $\rho=0.25$, 0.21, 0.05 for windows of 4, 6, 8 orbitals, but a residual floor of 104–369 times $\mathrm{Var}(E_L)$, so it failed Gate 2 there.
* **Lineage**: round 3, Operation A.4 on R2.0; re-gated on realistic $N=6$ states as R3.2.
* **Outcome**: MERGED into R3.2.

### R3.1 — Window-input backflow × all-orbital Jastrow, WIBJ, round 3 (KILLED)
* **The idea**: Let the backflow network see only the occupations of a window of $N$ valence orbitals while producing rows for all $M$ orbitals, and multiply by an all-orbital neural Jastrow. Every connected term with the same window change shares one orbital matrix (a "key"), so the exact local energy costs one backflow evaluation per key plus cheap arithmetic per term; keys do not grow with $M$.
* **Lineage**: round 3, Operation A on R2.6 (repair of its missing external correlation). Carried in round 4 as the $G=0$ end of R4.1 (labelled "R3.1′" there); re-entered in round 5 as R5.1.
* **Outcome**: KILLED at G2 (2026-11-11), on its own pre-check.
* **Kill rule**: Gate 2 with the round-3 rule that a route failing its own pre-check is not a survivor ($M\ge1$ fails).
* **Evidence**: H$_6$ ring, 6-31G**, $N=6$: $M_{\mathrm{R3.1}}=1.38$, 1.40, 1.32 (wall) at $K=8$, 10, 12; own-state cost falls to 0.66 of EXACT by operations at $K=12$, but $\Delta E=2.20$ mHa against 1.73 for NNBF-J and $\mathrm{Var}(E_L)$ is 1.4 times larger; smaller windows give $\Delta E=3.4$–4.6 mHa.
* **Why this kills it**: The cost gain is eaten by the accuracy and variance lost to the restriction.
* **What the kill does not rule out**: larger bases where keys are a smaller fraction of terms (tested as R5.1).
* **Strongest case against the kill**: the round-3 critic argued WIBJ changes only the prefactor and is active-space backflow times a Jastrow (Gate 0); round 5 argued every exact scheme is prefactor-only in the black-box model, which led to R5.1.
* **Checked by**: round-3 per-leaf critic (2026-11-11).
* **Lesson**: route-specific — superseded by R5.1, whose lesson covers it.

### R3.2 — Window-keyed partially-frozen control variate, PFCV (KILLED)
* **The idea**: Evaluate the same unrestricted NNBF-J with the window input of $\mathbf{n}'$ and the external input frozen at $\mathbf{n}$. The surrogate is exact for window-internal excitations (which carry the multireference weight), costs one evaluation per key, is summed exactly, and its residual is sampled.
* **Lineage**: round 3, Operation A on R2.0's surrogate; re-gating of R3.7′. Dropped in round 4.
* **Outcome**: KILLED at CRITIC (2026-11-11).
* **Kill rule**: round-3 per-leaf critic: is PFCV a relabel of R2.0? Also the round-3 own-pre-check rule ($M\ge1$ fails).
* **Evidence**: H$_6$ ring, $N=6$: $M_{\mathrm{R3.2}}=1.65$, 1.43, 1.23 (wall) and 1.28, 1.85, 1.04 (operations) at $K=8$, 10, 12. The residual adds only 1–2% to $\mathrm{Var}(E_L)$; the cost is the deterministic part, a floor of about 0.33 of EXACT, while SS costs 0.55–0.70.
* **Why this kills it**: It is the revived frozen control variate with a new surrogate, and it never beat semistochastic sampling on its own pre-check.
* **What the kill does not rule out**: larger external spaces where the key fraction falls (about $(W/M)^2$ predicted, never measured).
* **Strongest case against the kill**: $M$ was falling toward 1 with $K$ (1.04 by operations at $K=12$); unresolved.
* **Checked by**: round-3 per-leaf critic (2026-11-11).
* **Lesson**: L26 (`library.md`)

### R4.0 — Factorized-Jastrow keyed Cholesky contraction, FJKC (KILLED)
* **The idea**: Within one key, if the Jastrow has no external–external pair term, every term's ratio is a fugacity-dressed determinant ratio, so the whole (window to external) double-excitation class can be summed by a generalized Wick theorem through Cholesky vectors instead of enumerated, aiming to break the $\mathcal{O}(N^2M^2)$ term floor.
* **Lineage**: round 4, Operation A on WIBJ, following the round-3 critic's direction. Its accuracy question went to the FJ arm (R4.0d) and later to R6.1.
* **Outcome**: KILLED at G1 (2026-11-11); also Gate 0.
* **Kill rule**: Gate 1: does the contraction change the scaling class?
* **Evidence**: The identity holds to $1.2\times10^{-10}$ over 30 keys. But each key's external sum is a bilinear form on an $M_{\mathrm{ext}}\times M_{\mathrm{ext}}$ block, and with $N_\gamma\approx6.4K$–$7.7K$ Cholesky vectors the contraction costs the same order; measured contraction over enumeration 0.67, 0.69, 0.87. Gate 0: generalized-Wick Cholesky local energies are the AFQMC standard (arXiv:1711.02242, 2508.13036).
* **Why this kills it**: A constant-factor saving on one class does not escape the term floor.
* **What the kill does not rule out**: contractions where one orbital set serves all $(i,j)$ (the AFQMC case, tested as R6.1).
* **Strongest case against the kill**: not raised by a critic (the round-4 critic called the kill sound and narrowed the anti-pattern to per-class dressing with $N_\gamma\propto K$).
* **Checked by**: round-4 per-leaf critic (2026-11-12).
* **Lesson**: route-specific — the growth of the per-class Cholesky rank with the basis is a property of this one contraction scheme.

### R4.0d — Factorized-Jastrow diagnostic arm, FJ (MERGED)
* **The idea**: Train WIBJ with the external–external pair Jastrow removed, the architecture that R4.0's contraction requires, to measure what the factorization costs in accuracy; within one key each pair amplitude is then rank 2 in the external indices, while first-order pair amplitudes are full rank.
* **Lineage**: diagnostic arm of R4.0 (round 4); its Part A never ran because the round-4 array was replaced by pilot 411919; the question was answered by the closed-leaf falsifier R6.1 (masked CASJ: removing external–external pairs cost nothing measurable).
* **Outcome**: MERGED into R6.1.

### R4.1 — Coarse-grained-key backflow, CGK (KILLED)
* **The idea**: Feed the backflow the window occupations plus $2G$ coarse counts of electrons per spin in $G$ external energy shells, so orbitals respond to where in the external spectrum electrons went; keys grow only by $\mathcal{O}(G^2)$, independent of $M$, and $G$ interpolates between WIBJ ($G=0$) and the unrestricted NNBF.
* **Lineage**: round 4, Operation A on WIBJ's Gate 0 and accuracy walls.
* **Outcome**: KILLED at T2 (2026-11-12, pilot 411919).
* **Kill rule**: pre-registered by the round-4 critic: "kill R4.1 if the median $\delta_{\mathrm{CGK}}>\tau$ at $K=20$", where $\delta$ is the seed-paired difference of $\Delta E$ against NNBF-J.
* **Evidence**: pilot 411919 (6.4 CPU-h), H$_4$ distorted chain at $R=4.4$ bohr (HF weight 0.46–0.49), $K=20$, three seeds: median $\delta_{\mathrm{CGK}}=0.691$ mHa against $\tau=0.2$. The round-4 pre-check had passed at $M_{\mathrm{op}}=0.49$, but at $K=8$ CGK equalled NNBF-J (one orbital per shell), so it was tautological.
* **Why this kills it**: Coarse external counts cannot carry the dynamic pair correlation that grows with the basis, the externally contracted MRCI failure.
* **What the kill does not rule out**: many shells ($G$ close to $K_{\mathrm{ext}}$), which returns to the unrestricted cost.
* **Strongest case against the kill**: training noise is about $\tau$ (round-4 critic); answered by the margin (3.5 times $\tau$).
* **Checked by**: round-4 per-leaf critic (rule) and the pilot read-out (master, 2026-11-12).
* **Lesson**: L8 (`library.md`)

### R4.2 — Cross-spin-keyed backflow, XSK (KILLED)
* **The idea**: Let the $\alpha$-spin orbitals depend only on the $\beta$ window occupations and vice versa. Every same-spin excitation then leaves its own spin's orbitals unchanged, so all same-spin doubles share one orbital set and the number of orbital builds per sample falls from $\mathcal{O}(N^2W)$ to $\mathcal{O}(NW)$; Luo and Clark found spin-resolved NNBF learns mostly cross-spin weights.
* **Lineage**: round 4, Operation B.1 (frame change from occupation-window keys to spin-sector keys). Carried as a "conditional candidate" for the round-4 decisive array, which was never submitted (replaced by pilot 411919); not carried into round 5.
* **Outcome**: KILLED at G3 (measured on the pre-check) (2026-11-11).
* **Kill rule**: Gate 3: does it keep accuracy within $\tau=0.2$ mHa of NNBF-J on a multireference instance? Also Gate 2 ($M_{\mathrm{op}}<1$).
* **Evidence**: A1 instance (H$_6$ chain, $R=3.6$, HF weight 0.53), three pre-check runs against a converged NNBF-J control (0.70–0.74 mHa): accuracy losses 4.0 mHa at $K=8$ (seed 0) and 3.8 and 10.2 mHa at $K=10$ ($\Delta E=4.54$ and 10.91 mHa against NNBF-J's 0.735 and 0.700), that is 19 to 51 times $\tau=0.2$ mHa. $\mathrm{Var}(E_L)$ was 64 times NNBF-J's at $K=8$ only; at $K=10$ it was 1.6 to 2.8 times. $M_{\mathrm{op}}=8.3$ at the combined pre-check. Cost alone was the lowest of all keyed forms (0.040 of EXACT at $K=40$).
* **Why this kills it**: Gate 3 is decided by measurement, not argument: removing the same-spin channel loses 19–51 times $\tau$ on all three pre-check runs, each against a converged control, so same-spin backflow is not dispensable at HF weight 0.5. The variance blow-up is not part of the inference (64 times at $K=8$ only).
* **What the kill does not rule out**: weakly correlated systems; mixed forms with a small same-spin channel.
* **Strongest case against the kill**: the round-4 worker kept it alive because the pre-check was one seed at 1500 iterations (under-trained); the $K=10$ two-seed data answer this, but no critic reviewed the route and no decisive run happened.
* **Checked by**: per-leaf critic, 2026-11-17: upheld; three runs all ≥19tau against a converged control; G3 by measurement; variance 64x is K=8 only.
* **Lesson**: route-specific — a symmetry restriction of the backflow that the target does not obey.

### R5.1 — WIBJ re-entered under F$_{A1}$ with an exact keyed local energy (KILLED)
* **The idea**: Re-enter WIBJ (R3.1) in regime F$_{A1}$ of the Saturation Diagnostic: every exact scheme in the black-box model is prefactor-only, so the question is whether replacing a forward pass per term by $\mathcal{O}(N^2)$ arithmetic per term (through keys) costs accuracy. Tested at large external-to-window ratios against minimal-contrast foils.
* **Lineage**: round 5 re-entry of R3.1 (a revival, downgraded to CONDITIONAL by the round-5 critic); mutation to screened WIBJ (R5.1s) rejected.
* **Outcome**: KILLED at T3 (2026-11-14, job 452238).
* **Kill rule**: pre-committed `verdict_R51`: FAIL if ACC_FAIL, TREND_FAIL, FOIL_FAIL, $M_{\mathrm{op}}(136)\ge1$, "or a lower CI bound of $M_{\mathrm{wall}}(136)$ above 1".
* **Evidence**: Job 452238 (147 CPU-h). Cost (distorted H$_8$ chain, $K=40$–136): $M_{\mathrm{wall}}=0.60$, 1.29, 2.32, 4.03 (95% CI at $K=136$: 3.91–4.15), $M_{\mathrm{op}}=0.38$, 0.37, 0.52, 0.70, both against SCREEN at $\varepsilon^*=0.002$. Accuracy (H$_4$, $R=4.4$): WIBJ median $\Delta E=0.013$, 0.064, 0.187 mHa at $K=12$, 16, 20, seeds tight at $K=20$ (0.175–0.199), against NNBF-J's best 0.028 mHa.
* **Why this kills it**: The wall-clock FAIL clause fires with margin; even an ideal kernel gives $M_{\mathrm{op}}$ rising above 0.5, which the pre-committed $K=168$ step would kill; and WIBJ's own error extrapolates to 0.61–7.3 mHa at $K=40$, above $\tau$, where the cost is claimed.
* **What the kill does not rule out**: a constant-factor per-term saving on top of screening (R5.1s); window choices by natural-orbital occupation; first-row molecules.
* **Strongest case against the kill**: the pre-registered metric charged WIBJ unscreened against a screened control (asymmetric). Answered by the critic: the symmetric ratio is a flat constant (0.10 operations, 0.28 wall at $K=136$), and WIBJ plus screening carries about 0.29 mHa at $K=20$, above $\tau$.
* **Checked by**: round-5 per-leaf critic (2026-11-14), with reproduction job 621074.
* **Lesson**: L8 (`library.md`)

### R5.1s — Screened WIBJ, WIBJ plus SCREEN (KILLED)
* **The idea**: After R5.1's FAIL, combine WIBJ's cheap keyed terms with screening, so WIBJ is charged only for the terms SCREEN keeps; the symmetric pairing printed beside the verdict was 0.10 in operations and 0.28 in wall time at $K=136$, inside the PASS thresholds.
* **Lineage**: proposed mutation of R5.1 after job 452238; rejected by the round-5 critic.
* **Outcome**: KILLED at CRITIC (2026-11-14).
* **Kill rule**: round-5 per-leaf critic: is the mutation a legitimate new route or a post-hoc rescue, and does it fail Gate 0?
* **Evidence**: the symmetric ratio is flat in operations (0.21, 0.11, 0.10, 0.10) and its wall advantage shrinks with $K$ (0.21 to 0.28); measured screening bias on WIBJ at $\varepsilon^*$ grows to 0.104 mHa at $K=20$, so WIBJ plus screening carries about 0.29 mHa there; its components are prior art (screened active-space NNBF, arXiv:2604.06841; amplitude reuse, arXiv:2502.18843, 2212.11296).
* **Why this kills it**: It is a constant-factor combination of known pieces, chosen after the result, whose accuracy already exceeds $\tau$.
* **What the kill does not rule out**: a compiled keyed kernel inside screened NNBF codes as an engineering speed-up.
* **Strongest case against the kill**: screening was admissible for WIBJ on H$_4$ (its bias equals or undercuts NNBF-J's), so the pairing is fair; the critic agreed, and killed on the constant factor and accuracy instead. H$_6$ was not rechecked.
* **Checked by**: round-5 per-leaf critic (2026-11-14, job 621074).
* **Lesson**: L51 (`library.md`)

### R5.2 — Anchored local energy, ALE (SHELVED)
* **The idea**: Translate stochastic variance-reduced gradients (SVRG) onto the inner sum: at an anchor parameter enumerate and store all $\tilde x_j$; at later parameters estimate $E_L=H_{\mathbf{n}\mathbf{n}}+\sum_j\tilde x_j+$ a few sampled corrections $x_j(\theta)-\tilde x_j$ (ALE0), plus a first-order term (ALE1). The correction vanishes as the parameter change shrinks, so the sign floor does not apply.
* **Lineage**: round 5, Operation B.3, cross-field translation of SVRG (regime F$_B$); its tail-only variant is R7.5.
* **Source analogy**: SVRG (Johnson & Zhang, NeurIPS 2013) anchors a full gradient and corrects it with sampled differences; it works because the finite sum is fixed. Correspondence: anchor gradient ↔ anchor enumeration; data reuse ↔ reweighted samples. Gap: VMC samples move with $\theta$, and in noise-limited optimization reuse has no first-order gain (Statement 3).
* **Outcome**: SHELVED (G2 predicted outcome, 2026-11-17). T1 reached, AMBIGUOUS twice on its one instance; the next step, a cold-start T1 on a new instance, was refused by the compute gate's G2 rule. Not refuted.
* **Evidence**: Warm start (H$_4$, $K=12$, one seed): $\Delta E_{\mathrm{ALE}}=1.046$ against the control's $1.039$ mHa bound, a 0.007 mHa miss inside seed noise. Cold start, five seeds, budget 100 EXACT steps: median $\Delta E=17.0$ mHa (best ALE) against SS 7.0, SCREEN 9.0, EXACT 28.0; every SS seed (4.8–7.6) below every ALE seed (10.6–21.1); not FAIL only because the noise floor (17.6 mHa) was widened by EXACT seeds trapped near 220 mHa.
* **Why shelved**: GATE-REFUSED (G2), reviewer triage 2026-11-17. On the cold-start test semistochastic sampling (SS) beat the best anchored arm on all five seeds (worst SS seed 7.6 mHa against best ALE seed 10.6 mHa), so a cold-start run on a new instance is predicted to FAIL against SS with $p\approx0.85$; at T1 (at most 10 CPU-minutes) that outcome changes no action, because the leaf already recommends SS or screening for this regime. It is recorded as shelved, not killed: the AMBIGUOUS verdicts came from a noise floor widened by trapped EXACT seeds, not from a rule that fired.
* **What would unshelve it**: evidence that optimisation on some target is step-limited rather than noise-limited (where Statement 3's no-first-order-gain argument does not bind), or a new instance on which SS loses its seed-by-seed lead over EXACT; either makes the anchored arm's outcome uncertain enough to pass G2.
* **Lesson**: L26 (`library.md`)

### R5.3 — Hadamard-certified screening (KILLED)
* **The idea**: Bound every dropped term without a forward pass by a configuration-independent Hadamard bound on the backflow determinant (using $\lvert\tanh\rvert\le1$) times the exact Jastrow ratio, giving screening with a certified bias (regime F$_C$).
* **Lineage**: round 5, regime F$_C$ of the Saturation Diagnostic. Repaired as a sign certificate in R7.1.
* **Outcome**: KILLED at G2 (2026-11-12).
* **Kill rule**: Gate 2: does certification remain affordable where screening saves work?
* **Evidence**: H$_4$, $K=12$, warm state: certified bias 140–1000 times the actual dropped sum; at $\varepsilon=10^{-4}$ it is already 0.62 mHa certified against $2\times10^{-4}$ mHa actual, while SCREEN there keeps 86% of terms.
* **Why this kills it**: Certification is affordable only where screening saves nothing.
* **What the kill does not rule out**: sign certificates (R7.1) and configuration-dependent bounds.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: not separately reviewed; the round-5 per-leaf critic (2026-11-14) raised no objection.
* **Lesson**: L27 (`library.md`)

### R5.4 — Statistical bias certificate (KILLED)
* **The idea**: Screen, then sample the dropped tail to bound the screening bias with a one-sided confidence interval, giving a statistically certified biased local energy whose admissibility no longer needs an exact reference.
* **Lineage**: round 5, regime F$_C$.
* **Outcome**: KILLED at G0 (2026-11-12).
* **Kill rule**: Gate 0: is it known, and is its barrier fixed?
* **Evidence**: it is semistochastic sampling with a confidence statement (arXiv:2507.19276; SHCI PT2), already a control.
* **Why this kills it**: A relabelled control.
* **What the kill does not rule out**: deterministic certificates (R5.3, R7.1).
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: not separately reviewed; the round-5 per-leaf critic (2026-11-14) raised no objection.
* **Lesson**: route-specific — relabel.

### R5.5 — Stale-plus-linear local energy without correction (KILLED)
* **The idea**: Approximate $E_L(\theta)\approx\tilde E_L+\tilde g\cdot\Delta\theta$ from an anchor's local energy and its parameter gradient, with no sampled correction, so no amplitude is evaluated between anchors and each step costs only a dot product.
* **Lineage**: round 5, regime F$_B$ without correction; dominated by R5.2.
* **Outcome**: KILLED at G1 (2026-11-12).
* **Kill rule**: Gate 1: is the bias controlled?
* **Evidence**: the $\mathcal{O}(\Delta\theta^2)$ bias has no certificate, and the residual that removes it costs only $K_d$ amplitudes (ALE1).
* **Why this kills it**: An uncertified bias that a cheap correction would remove.
* **What the kill does not rule out**: R5.2.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: not separately reviewed; the round-5 per-leaf critic (2026-11-14) raised no objection.
* **Lesson**: route-specific — dominated by R5.2.

### R5.6 — Learned local-energy regressor, debiased on a subset (KILLED)
* **The idea**: Train a regressor that predicts $E_L$ from cheap features and debias it with exact local energies on a random subset of samples, a learned cheap model inside the multifidelity estimator.
* **Lineage**: round 5, regime F$_B$; a learned version of R3.2′.
* **Outcome**: KILLED at G2 (2026-11-12).
* **Kill rule**: Gate 2: can the cheap model's error variance be small enough to pay?
* **Evidence**: R3.2′ measured $R^2=0.13$ on all cheap features and $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ from 2 to 466.
* **Why this kills it**: A learned model of the same features cannot beat their measured information content.
* **What the kill does not rule out**: features that include the amplitude network's internal state.
* **Strongest case against the kill**: not raised by a critic: a flexible model could beat linear $R^2$; never tested.
* **Checked by**: not separately reviewed; the round-5 per-leaf critic (2026-11-14) raised no objection.
* **Lesson**: L25 (`library.md`)

### R6.1 — Contractible CAS × Jastrow without external–external pairs (F$_{A2}$ closed-leaf falsifier) (KILLED)
* **The idea**: Test regime F$_{A2}$ (sub-$\lvert\mathcal{C}\rvert$ cost by contraction): a contraction needs configuration-independent orbitals and a Jastrow that factorizes over the contracted class. The largest such family in the code is CASJ (all window determinants, trainable orbital rows, pair Jastrow) with the external–external pairs masked; if it is accurate enough, F$_{A2}$ is open.
* **Lineage**: closed-leaf falsifier (2026-11-16), from the Saturation Diagnostic's F$_{A2}$ ingredient; answers R4.0d.
* **Outcome**: KILLED at T1 (2026-11-16).
* **Kill rule**: pre-registered: "KILL-F$_{A2}$ if the masked CASJ ends above $\tau=0.2$ mHa"; INVALID if the unmasked control misses 0.390 mHa by more than 0.01.
* **Evidence**: H$_4$, $R=4.4$, $K=12$, seed 0, 6000 L-BFGS iterations: masked CASJ 0.3885 mHa; unmasked control 0.3899 (valid).
* **Why this kills it**: Even before contraction, the contractible family misses $\tau$ at the smallest basis, and restricted ansätze get worse with $K$.
* **What the kill does not rule out**: larger CAS windows, which are multi-Slater–Jastrow fast local energies (arXiv:2008.06477, prior art); one seed only.
* **Strongest case against the kill**: one seed; answered partly by the three CASJ seeds of job 452238 (0.386–0.390), which bound the family from above.
* **Checked by**: leaf-summary critic sign-off (2026-11-16).
* **Lesson**: L8 (`library.md`)

### R7.1 — Variational product-weighted screening in canonical orbitals, VPS (C1) (KILLED)
* **The idea**: Screen small elements of either sign, but move each dropped pair onto the diagonal with product weights so that $H'=H_\varepsilon+\mathrm{diag}(D)\succeq H$; the screened energy then stays a rigorous variational upper bound, and the compensator costs an $\mathcal{O}(N^2)$ lookup per sample. This certifies the sign of the bias instead of its size.
* **Lineage**: reopening review (2026-11-16), Operation A on R5.3, from ingredient I4; translation of lattice fixed-node (cross-field translation repair of R5.3's certificate).
* **Source analogy**: lattice fixed-node (ten Haaf et al., PRB 51, 13039, 1995) moves sign-violating hops to the diagonal to get an upper bound. Correspondence: sign-violating hop ↔ dropped small element. Gap: screening drops elements of both signs, and favourable ones cost twice their size.
* **Outcome**: KILLED at T1 (2026-11-16).
* **Kill rule**: pre-registered: "KILL if $f_{\mathrm{VPS}}(10)\ge0.5$, or if $f_{\mathrm{VPS}}$ does not fall strictly over $N=6$, 8, 10", where $f$ is the kept-term fraction at error at most $\tau$.
* **Evidence**: linear H$_N$ chains, STO-3G, $R=3.0$, exact ground states: $f_{\mathrm{VPS}}=0.917$, 0.949, 0.976 at $N=6$, 8, 10 against uncertified SCREEN 0.891, 0.910, 0.846.
* **Why this kills it**: The certified kept fraction rises with $N$ and stays near 1.
* **What the kill does not rule out**: the same certificate in a localized basis (R7.4), and non-diagonal compensators.
* **Strongest case against the kill**: exact ground states rather than trained NNBF; the floor $E_0(H')-E_0$ equals the excess, so re-optimization would not help.
* **Checked by**: leaf-summary critic sign-off (2026-11-16).
* **Lesson**: L27 (`library.md`)

### R7.2 — Sum of $R$ configuration-independent Slater–Jastrows (C2) (KILLED)
* **The idea**: Extend F$_{A2}$ beyond one orbital set: a sum of $R$ Slater–Jastrow terms, each with configuration-independent orbitals, so each term's local energy contracts as in AFQMC while the sum adds expressivity.
* **Lineage**: reopening review, Operation A on F$_{A2}$ (ingredient I2).
* **Outcome**: KILLED at G0 (2026-11-16); also Gate 4.
* **Kill rule**: Gate 0 (known, barrier unfixed) and Gate 4 (does the gain survive size growth?).
* **Evidence**: Jastrow times non-orthogonal multi-determinant (resonating Hartree–Fock) and multi-Slater–Jastrow fast local energies exist (arXiv:2008.06477); a fixed $R$ of mean-field terms is the cross-leaf anti-pattern "a summed or fixed-rank construction is a finite-size device".
* **Why this kills it**: Known, and its expressivity gain does not grow with size.
* **What the kill does not rule out**: $R$ growing with size, at growing cost.
* **Strongest case against the kill**: not raised by a critic; none found.
* **Checked by**: leaf-summary critic sign-off (2026-11-16).
* **Lesson**: L10 (`library.md`)

### R7.3 — Plain screening in localized orbitals (C3) (KILLED)
* **The idea**: Change the orbital frame to site-local Löwdin orbitals, where two-electron couplings decay with distance, so the number of terms kept by screening at fixed bias should grow more slowly than $\lvert\mathcal{C}\rvert$, an extensive gain.
* **Lineage**: reopening review, Operation B.1 (frame transformation), from ingredient I5; merged with R7.1 into R7.4; kept as a foil arm.
* **Outcome**: KILLED at G0 (2026-11-16).
* **Kill rule**: Gate 0: is it known, and does it add a new mechanism?
* **Evidence**: NQS chemistry already runs in Boys-localized bases (arXiv:2305.12776); measured as a foil, its kept fraction is 0.306, 0.144, 0.079 at $N=6$, 8, 10 against 0.85–0.91 canonical.
* **Why this kills it**: A basis choice plus screening adds no estimator mechanism; the gain is real but known.
* **What the kill does not rule out**: whether a trained NNBF stays as accurate in a localized basis (a P2.1 question).
* **Strongest case against the kill**: the extensive gain is the largest effect seen in this leaf; it is kept as the foil and handed to P2.1.
* **Checked by**: leaf-summary critic sign-off (2026-11-16).
* **Lesson**: L9 (`library.md`)

### R7.4 — Variational product-weighted screening in localized orbitals (C3′) (KILLED)
* **The idea**: Combine R7.1's variational certificate with R7.3's localized basis: in Löwdin orbitals the dropped elements decay with distance, so the certified excess per dropped term falls too, and the certified kept fraction could fall with size while the energy stays a rigorous upper bound. Its non-diagonal form adds a compensator $C$ supported on the diagonal and the kept pairs, with $C-H_{\mathrm{drop}}\succeq0$, so that screening keeps a variational guarantee.
* **Lineage**: reopening review, merge of C3 (R7.3) with C1 (R7.1); its non-diagonal follow-up (the best compensator on the diagonal plus the kept pairs) was tested by the T1 SDP bound, the H$_8$ path-gadget robustness test, the T1b on real trained states and the T0 operation count, all on 2026-11-17. No successor.
* **Outcome**: KILLED at T0 (2026-11-17, cost).
* **Kill rule**: the leaf's acceptance condition, fixed in its question before any route: an estimator qualifies only if its cost to a fixed error bar is below EXACT's and falls relative to EXACT as the basis grows, while its bias stays at most $\tau=0.2$ mHa; at T0 the cost is an operation count of the real algorithm per sample (L47). No numeric rule beyond "cheaper than EXACT at admissible bias" was written before the count; the verdict was applied by the orchestrator.
* **Evidence**: H$_6$ SDP bound (exact $\psi_0$): best kept-pattern compensator certifies a kept fraction of 0.134, 0.44 times plain localized screening (0.306): PASS-T1. H$_8$ path gadget (4900 determinants, relative noise $\delta=10^{-2}$, 5 draws): $f_{\mathrm{rob}}=0.0602$ against the KILL line 0.433, ratio 0.42 against 0.44 at H$_6$: PASS-T1; under additive noise of the same norm the same compensator has excess 0.66$\tau$ (H$_6$) and 23–125$\tau$ (H$_8$). T1b on regenerated NNBF-J states: excess 0.00$\tau$ (H$_6$, 3 seeds) and 0.01$\tau$ (H$_8$, 2 seeds), 0.03$\tau$ for an under-parameterised H$_8$ network: PASS-T1b. T0 count (H$_8$, $\varepsilon_S=2.24\times10^{-3}$): 359.6 connected terms per sample, 26.7 kept; the compensator needs reference amplitudes on the kept 3-ball, about 1352 determinants per sample. Frozen-NQS reference: about 3.8 times EXACT. Jastrow–HF reference (about 0.1 times EXACT): $\delta\approx0.3$–1 against the needed $\le0.17$, excess 3–35$\tau$ (estimate, not measured; a measurement on H$_8$ is running in `tests/P2.2_stochastic_local_energy_r8/jhf_reference.*`). At the grid end ($\varepsilon=0.1$) the reference work is about 525 evaluations plus 11 network calls against 360, about 1.5 times EXACT. HF or CAS references: zero on most determinants, so only the diagonal floor (1.95 times plain screening) remains (`tests/P2.2_stochastic_local_energy_r8/`).
* **Why this kills it**: The certificate is cheap only if the slack $Y=D^{-1}LD^{-1}$ can be evaluated per sample. $Y$ couples each sample to determinants several kept edges away, so evaluating it needs a reference amplitude on about 1352 determinants at H$_8$, against 360 network calls for EXACT. Every reference considered either costs more than EXACT (frozen NQS, 3.8 times), is estimated to be too inaccurate for the bound to stay below $\tau$ (Jastrow–HF, 3–35$\tau$, an estimate under measurement), or vanishes where it is needed (HF, CAS), which falls back to the diagonal certificate that already failed. So no admissible version is cheaper than EXACT, and the leaf's acceptance condition fails.
* **What the kill does not rule out**: the accuracy side holds only on states trained to $\Delta E\le0.03$ mHa, where the certified excess is 0.00–0.03$\tau$ (PASS-T1b); at the leaf's working accuracy ($\approx0.74$ mHa) H$_8$ excess reached 9–16$\tau$. So a cheap, nonzero reference amplitude (relative error below about 0.17 at about 0.1 times EXACT) would revive the route only if the certificate also stays below $\tau$ at the working $\Delta E$. ED-size calculations where $L$ can be precomputed are also not covered, and neither is amortising reference amplitudes across a Markov chain on systems where the kept balls of successive samples overlap strongly.
* **Strongest case against the kill**: the reference work could be cached across samples, since many samples share kept neighbourhoods. At H$_8$ the 3-ball is 28% of the sector, so caching it amounts to near-full enumeration, which is an ED-size statement; beyond ED the ball grows with the number of excitations per path, faster than $\lvert\mathcal C\rvert$, so the count should only worsen; this scaling is argued, not measured. The 1352-determinant count is a conservative triangle-gadget 3-ball (the path gadget routes along paths of up to 5 edges at $\varepsilon_S$), and even at the grid end the reference work is about 1.5 times EXACT (525 plus 11 against 360). Not measured: VMC-trained states (early L-BFGS iterates already reach 9–16$\tau$ at H$_8$) and a dedicated cheap reference network. Unresolved for a reference designed for this purpose.
* **Checked by**: orchestrator decision, 2026-11-17; per-leaf critic, 2026-11-17: SIGN-OFF-WITH-CHANGES (T0 cost kill upheld under the leaf's acceptance condition; accuracy re-scoped to $\Delta E\le0.03$ mHa; Jastrow–HF numbers labelled as estimates).
* **Lesson**: L27 and L9 (`library.md`, extended 2026-11-17).

### R7.5 — Anchored local energy on the semistochastic tail only (C4) (KILLED)
* **The idea**: Apply R5.2's stale-anchor control variate only to the sampled tail of a semistochastic estimator, keeping large terms exact, so the anchored correction removes tail variance at little cost.
* **Lineage**: reopening review, Operation A on R5.2 (ingredient I3).
* **Outcome**: KILLED at G2 (2026-11-16), at T0 from data on disk.
* **Kill rule**: Gate 2: is the tail variance worth removing compared with plain SS?
* **Evidence**: the SS tail adds about 1–2% to $\mathrm{Var}(E_L)$ on converged states (round 3), and SS costs $0.93\times$ EXACT, so removing it buys a few percent; in the step-limited cold start the tail did not limit SS (R5.2 second run).
* **Why this kills it**: There is nothing left to reduce.
* **What the kill does not rule out**: instances with heavy SS tails.
* **Strongest case against the kill**: not raised by a critic; the 1–2% comes from $N=6$ rings and may differ elsewhere.
* **Checked by**: leaf-summary critic sign-off (2026-11-16).
* **Lesson**: L26 (`library.md`)

## Leaf Status Summary
* **State**: EXHAUSTED at this budget
* **Updated**: 2026-11-17 (R7.4 KILLED at T0 on cost after its H$_8$ robustness test passed (PASS-T1) and its real-state check passed (PASS-T1b); with R5.2 SHELVED by G2, every reopening-ledger row is closed and one is GATE-REFUSED: DORMANT → EXHAUSTED at this budget, section 3b exit 3; per-leaf critic SIGN-OFF-WITH-CHANGES 2026-11-17, changes applied: accuracy claim re-scoped to $\Delta E\le0.03$ mHa, Jastrow–HF numbers labelled as estimates, grid-end count added)
* **Question**: Can $E_{\mathrm{loc}}$ be replaced by an unbiased stochastic estimator over the $\mathcal{O}(M^4)$ connected configurations, with variance bounded so that the total cost to reach chemical accuracy scales better than exact enumeration?
* **Answer**: No, not in general. If each amplitude costs one network evaluation, we proved that an exact local energy cannot skip terms, and an unbiased random one has a variance floor set by the terms it leaves out. On hydrogen chains and rings (accuracy checked up to 20 orbitals, cost up to 136), all 41 ideas that used extra structure either lost accuracy or cost more than existing methods. Dropping small terms in localized orbitals, which is already known, stayed best. Reusing earlier samples during optimisation was not refuted; its test was judged predictable and not run.
* **Conclusion**: When each connected amplitude costs one network call (the black-box model), no exact local energy can skip connected terms and no unbiased one can beat an inclusion-probability variance bound; this is proven within that model, and each of the 41 routes tried either falls under it or lost accuracy or cost when it used structure to escape. The last certified route, variational screening in localized orbitals (R7.4), is admissible on states trained to $\Delta E\le0.03$ mHa; at the leaf's working accuracy ($\approx0.74$ mHa) H$_8$ excess reached 9–16$\tau$, and it needs reference amplitudes on a global neighbourhood, so it costs more than exact enumeration with any accurate reference. Plain screening in localized orbitals (prior art) remains the best estimator found; anchored reuse in step-limited optimisation (R5.2) was not refuted, only refused by the compute gate as predictable, so the leaf is closed at this budget.
* **Scope**: Hydrogen chains and rings (H$_4$–H$_{10}$; STO-3G, s-only, 6-31G**), Hartree–Fock (HF) weight 0.37–0.57; accuracy from exact diagonalization (ED) up to $K=20$ spatial orbitals, cost ED-free up to $K=136$; real neural backflow with a Jastrow (NNBF-J) trained on the exact energy (full sums, no VMC noise); R7.4 on linear H$_6$ and H$_8$ in Löwdin orbitals; only the sign half of the Hard Core (non-stoquastic rows, no near-degenerate competitor).
* **Compute spent**: about 162 CPU-h (jobs 411919, 452238 and 621074, plus about 0.95 CPU-h of capped interactive T1 runs, including the 480 CPU-s SDP bound, the 160 CPU-s H$_8$ path-gadget test and the 340 CPU-s real-state check and operation count). All Slurm spend predates the compute gate.

**The question.** Variational Monte Carlo (VMC) for a neural quantum state $\psi$ averages the local energy $E_L(\mathbf{n})=H_{\mathbf{n}\mathbf{n}}+\sum_{\mathbf{n}'\in\mathcal{C}(\mathbf{n})}H_{\mathbf{n}\mathbf{n}'}\psi(\mathbf{n}')/\psi(\mathbf{n})$ over samples $\mathbf{n}\sim\psi^2$. With dense chemistry integrals $\lvert\mathcal{C}\rvert=\mathcal{O}(N^2M^2)$ ($N$ electrons, $M$ spin-orbitals), each term costs a forward pass, and exact enumeration (EXACT) dominates the cost. The leaf asked for an estimator, unbiased or biased by at most $\tau=0.2$ mHa, whose cost to a fixed error bar falls relative to EXACT as $M$ grows and beats screening (SCREEN: drop $\lvert H_{\mathbf{n}\mathbf{n}'}\rvert\le\varepsilon$), semistochastic (SS) and heat-bath sampling.

**The result: the Saturation Diagnostic.** One importance proposal has per-draw variance at least $(\sum\lvert x_j\rvert)^2-(\sum x_j)^2$ over the terms $x_j$, nonzero for mixed-sign rows; on H$_6$ near convergence even oracle sampling cost 146 times EXACT (R1.1, R1.3), and sign splitting gained under 3% because surrogate magnitude error sets the floor (R2.1). In general: if an amplitude is revealed only by an oracle call, an adversary can change any unqueried one, so an exact $E_L$ needs $\Omega(\lvert\mathcal{C}\rvert)$ calls (Statement 1); an unbiased estimator querying term $j$ with probability $\pi_j$ has worst-case variance at least $\sum_js_j^2(1/\pi_j-1)$, with $s_j$ the uncertainty of $x_j$ given free information (Statement 2). Both are proofs within the black-box query model, a PROVEN-IMPOSSIBLE sub-result scoped to that model and held inside this leaf. Sample reuse gives no first-order gain while optimization is noise-limited (Statement 3). Escapes need structure: cheaper terms (F$_{A1}$), contraction (F$_{A2}$), amortization (F$_B$), certified bias (F$_C$).

**Evidence on the escapes** (pre-registered rules, controls on the same state). F$_{A1}$: window-input backflow (WIBJ, R5.1) failed decisive test 452238 on cost, wall ratio 4.03 (95% CI 3.91–4.15) at $K=136$ spatial orbitals, and its own error rose 0.013, 0.064, 0.187 mHa over $K=12$–20. F$_{A2}$: the largest contractible family reached 0.3885 mHa (R6.1). F$_B$: from a cold start the best anchored arm had median 17.0 mHa against SS 7.0 and EXACT 28.0 (five seeds), AMBIGUOUS only through a noise floor widened by trapped EXACT seeds (R5.2). F$_C$: a magnitude certificate was 140–1000 times loose (R5.3); a variational sign certificate kept 0.976 of terms in canonical orbitals (R7.1) and 0.477 in localized ones, 6.1 times plain localized screening (R7.4, diagonal). The best compensator allowed off-diagonal entries on kept pairs did far better. Its optimal slack has the closed form $Y=D^{-1}LD^{-1}$ ($D=\mathrm{diag}\,\psi_0$, $L$ a signed graph Laplacian along kept paths), and under relative amplitude noise it certified a kept fraction of 0.134 on H$_6$ (0.44 times plain screening) and 0.060 on H$_8$ (0.42 times, not growing; PASS-T1). On NNBF-J states trained to $\Delta E\le0.03$ mHa its excess was 0.00–0.03$\tau$ (PASS-T1b), but at the leaf's working accuracy ($\approx0.74$ mHa, early L-BFGS iterates) the H$_8$ excess reached 9–16$\tau$. Then the T0 operation count killed it on cost: at H$_8$ a sample has 360 connected terms, of which 27 are kept, but $Y$ links it to about 1352 of the 4900 determinants (a conservative triangle-gadget 3-ball count; the path gadget routes along paths of up to 5 edges at $\varepsilon_S$), so a frozen-NQS reference costs about 3.8 times EXACT; even at the end of the grid ($\varepsilon=0.1$) the reference work is about 525 evaluations plus 11 network calls against 360, about 1.5 times EXACT. A cheap Jastrow–HF reference (about 0.1 times EXACT) would be 2–20 times too inaccurate for the bound (relative error $\delta\approx0.3$–1, excess 3–35$\tau$) (estimate, not measured; a measurement on H$_8$ is running in `tests/P2.2_stochastic_local_energy_r8/jhf_reference.*`) (R7.4 card).

**Why the alternatives fail.** Surrogate control variates keep residual fractions of 0.21–0.88 (R1.4, R2.0, R3.2); accurate surrogates cost 0.21 forward passes per term (R2.3); cross-sample corrections have error variance 2–466 times $\mathrm{Var}_\pi(E_L)$ (R3.2′, R5.6); per-class contractions cost as much as enumeration (R4.0); window or coarse backflow inputs lose external pair correlation (R2.6, R4.1, R4.2). Plain screening wins throughout, and in localized orbitals its kept fraction falls with size (R7.3, prior art).

**What this leaf taught beyond itself** (entries of `library.md`):
* L12 (R1.1, R1.2, R1.3): single-proposal sampling of a non-stoquastic sum has a variance floor that does not vanish at convergence.
* L8 (R1.4, R2.6, R4.1, R5.1, R6.1): cheap-local-energy ansätze use configuration-independent or window-restricted orbitals and lose the external pair correlation that grows with the basis; judge them by their own error trend against a minimal-contrast foil.
* L9 (R7.3, R7.4): the size-extensive screening saving comes from the localized orbital frame, not the estimator; certifying it is cheap only with a precomputed global compensator.
* L25 (R3.2′, R5.6): surrogate errors vary too much from sample to sample for cross-sample debiasing to pay.
* L26 (R3.2, R5.2, R7.5): near convergence the deterministic work sets the cost, so a cheaper local energy helps only if its saving buys fresh samples.
* L27 (R5.3, R7.1, R7.4): certificates that ignore sign cancellations make certified screening keep almost every term in canonical orbitals; a kept-pattern compensator escapes this in localized orbitals and is admissible on states trained to $\Delta E\le0.03$ mHa; at the leaf's working accuracy ($\approx0.74$ mHa) H$_8$ excess reached 9–16$\tau$, but it needs reference amplitudes on a global neighbourhood, and every practical reference either costs more than EXACT or (by estimate) breaks the bound.
* L51 (R2.0, R5.1s): a revival or a post-hoc rescue is not a new mechanism.
* Route-specific: sign-aware sampling under magnitude-dominated error (R2.1), the sum-of-squares support condition (R3.1′) and the per-class Cholesky contraction cost (R4.0).

**Strongest objection and answer.** "The bounds cover only black-box amplitudes, and two escapes were never killed by measurement." The first part is accepted as scope: the impossibility statements hold inside the query model, and every structured escape was then tested on its own terms. R7.4 is now killed, but on cost, not accuracy: its certificate is admissible on states trained to $\Delta E\le0.03$ mHa; at the leaf's working accuracy ($\approx0.74$ mHa) H$_8$ excess reached 9–16$\tau$, so a cheap reference amplitude that is accurate at the working $\Delta E$ would revive it (unresolved; VMC-trained states were not tested). That the cost wall worsens beyond ED size is argued (the kept ball grows with the number of excitations per path), not measured. R5.2 is shelved, not refuted: plain semistochastic sampling beat it on every cold-start seed, so its next run was refused as predictable. That WIBJ was charged unscreened against a screened control is answered in R5.1s: the symmetric ratio is a flat constant and WIBJ plus screening is 0.29 mHa off at $K=20$.

**Limits.** Hydrogen only; accuracy at $K\le20$, cost at $K\ge40$; Python kernels; R5.2 on one instance; R7.4 on H$_6$ and H$_8$ with full-sum trained states (no VMC noise) and an operation count, not a timed kernel.

### Closure certificate
* **Claim**: (i) In the black-box query model, an exact local energy needs $\Omega(\lvert\mathcal C\rvert)$ network calls and an unbiased one has worst-case variance at least $\sum_js_j^2(1/\pi_j-1)$ (Statements 1 and 2, proofs in the Saturation Diagnostic). (ii) On hydrogen chains and rings (H$_4$–H$_{10}$, accuracy to $K=20$, cost to $K=136$), no tested escape (cheaper terms, contraction, certified bias) reached a cost to a fixed error bar below plain screening or semistochastic sampling at bias at most $\tau=0.2$ mHa: WIBJ lost on cost (wall ratio 4.03 at $K=136$) and accuracy; contractible families lost accuracy (0.3885 mHa); magnitude and canonical sign certificates kept almost every term; the localized kept-pattern certificate is admissible on states trained to $\Delta E\le0.03$ mHa; at the leaf's working accuracy ($\approx0.74$ mHa) H$_8$ excess reached 9–16$\tau$, and costs more than EXACT with any accurate reference (3.8 times with a frozen NQS at $\varepsilon_S$, about 1.5 times even at the grid end $\varepsilon=0.1$). Not settled: anchored reuse (R5.2) in step-limited optimisation, whose next step was refused at this budget.
* **Decisive evidence**: job 452238 (WIBJ, T3 KILL); R6.1 at T1 (DIAGNOSTIC-HOLDS); R5.3, R7.1 (kills); R7.4 T1 SDP bound, H$_8$ path-gadget test, T1b and T0 count (`tests/P2.2_stochastic_local_energy_r8/`, 2026-11-17); R5.2's two cold-start T1 runs (AMBIGUOUS, SS ahead on 5 of 5 seeds). Reviewed by the round-5 critic and the leaf-summary critic (2026-11-16); the R7.4 kill and this certificate by the per-leaf critic (2026-11-17, SIGN-OFF-WITH-CHANGES).
* **Why no doubt remains**:
    * *Only the black-box model is covered* (round-5 critic): accepted as the scope of (i); (ii) tests structured escapes directly.
    * *WIBJ charged unscreened* (critic): answered by R5.1s (WIBJ plus screening 0.29 mHa off at $K=20$).
    * *R7.4 robustness to amplitude errors*: answered only on states trained to $\Delta E\le0.03$ mHa (PASS-T1b, 0.00–0.03$\tau$); at the working accuracy ($\approx0.74$ mHa) the H$_8$ excess reached 9–16$\tau$. The cost kill does not depend on it.
    * *Grid end and gadget count* (per-leaf critic, 2026-11-17): at $\varepsilon=0.1$ the reference work is about 525 evaluations plus 11 network calls against 360 for EXACT, about 1.5 times EXACT, so the cost kill holds across the grid; the 1352-determinant triangle-gadget 3-ball is conservative (the path gadget uses paths of up to 5 edges at $\varepsilon_S$). That the cost worsens beyond ED is argued, not measured.
    * *Jastrow–HF reference numbers*: $\delta\approx0.3$–1 and 3–35$\tau$ are estimates, not measurements; a measurement on H$_8$ is running. The kill also rests on the frozen-NQS count (3.8 times EXACT), which is measured.
    * *Relative versus additive noise* (H$_8$ caveat): real states sit between the models (25–1000 times the relative prediction, 30–1000 times below the additive one); again the cost kill does not depend on it.
    * *Reference caching across samples*: at H$_8$ the reference ball is 28% of the sector, so caching is near-full enumeration; stated as a limit, not measured beyond ED.
    * *R5.2's AMBIGUOUS verdicts*: came from a noise floor widened by trapped EXACT seeds; SS beat ALE on every seed. Listed as the refused step below.
* **Falsifiers run**: F$_{A2}$ follow-up (R6.1): DIAGNOSTIC-HOLDS. R7.4 on exact H$_6$ (SDP bound): PASS-T1, then H$_8$ robustness: PASS-T1, then real trained states: PASS-T1b, then the T0 count: KILLED(T0). R5.2 cold-start follow-up on a new instance: GATE-REFUSED(G2), listed below.
* **Missing ingredients exhausted**: F$_{A1}$ cheaper per term → R5.1 KILLED(T3); F$_{A2}$ contraction → R6.1 KILLED(T1), R7.2 KILLED(G0); F$_B$ step-limited amortisation → R5.2 GATE-REFUSED(G2), R7.5 KILLED(G2); F$_C$ magnitude certificate → R5.3 KILLED(G2), R5.4 KILLED(G0); F$_C$ sign certificate, canonical → R7.1 KILLED(T1); F$_C$ sign certificate, localized → R7.4 KILLED(T0); orbital frame → R7.3 KILLED(G0), accuracy handed to P2.1.
* **What would reopen it**: (1) The refused step: a cold-start T1 of R5.2 on a new instance (at most 10 CPU-minutes, interactive; capped cost about 0.2 CPU-h, so any budget funds it). Refused under G2: FAIL against SS predicted with $p\approx0.85$, and that outcome changes no action. It reopens with evidence that optimisation on some target is step-limited rather than noise-limited, or an instance where SS loses its seed-by-seed lead. (2) R7.4: a reference amplitude that is nonzero everywhere, has relative error below about 0.17 along kept paths, and costs about 0.1 times EXACT per sample, together with a certificate that stays below $\tau$ on states at the leaf's working accuracy ($\Delta E\approx0.74$ mHa), not only with a cheap reference; or a timed kernel showing reference evaluations far cheaper than network ratios. (3) Any NQS architecture that exposes amplitude structure making Statements 1 and 2 inapplicable.
* **Certificate sign-off**: "P2.2 per-leaf critic, 2026-11-17: SIGN-OFF-WITH-CHANGES. R7.4 T0 cost kill upheld under the leaf's acceptance condition. R5.2 G2 refusal valid. EXHAUSTED at this budget (exit 3) stands. Accuracy claim re-scoped to ΔE ≤ 0.03 mHa (9–16τ at working accuracy). Jastrow–HF δ labelled estimate or measured." Changes applied by the editor on 2026-11-17.

**Reopening ledger (closed 2026-11-17).**

| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| F$_{A1}$, cheaper per term | Saturation Diagnostic | R5.1 (WIBJ), job 452238 | KILLED(T3) |
| F$_{A2}$, contraction | Saturation Diagnostic; review item I2 | R6.1 masked CASJ; R7.2 | KILLED(T1); KILLED(G0) |
| F$_B$, step-limited amortization | Saturation Diagnostic; review item I3 | R5.2 cold start (two T1 runs), then a new-instance cold start; R7.5 | GATE-REFUSED(G2: SS beat ALE 5/5 seeds, FAIL predicted $p\approx0.85$, changes no action); KILLED(G2) |
| F$_C$, magnitude certificate | Saturation Diagnostic | R5.3; R5.4 | KILLED(G2); KILLED(G0) |
| F$_C$, sign certificate, canonical frame | review item I4 | R7.1 | KILLED(T1) |
| F$_C$, sign certificate, localized frame | review items I4 and I5 | R7.4 (diagonal, AMBIGUOUS); T1 SDP bound (PASS-T1 on exact H$_6$); H$_8$ path-gadget robustness test (PASS-T1, relative-noise caveat); T1b on real trained states (PASS-T1b); T0 operation count | KILLED(T0): cost; the compensator's reference work is global (frozen-NQS reference 3.8 times EXACT at $\varepsilon_S$, about 1.5 times at the grid end; a cheap Jastrow–HF reference would break the bound 2–20 times, an estimate) |
| Orbital frame | review item I5 | R7.3 | KILLED(G0); accuracy question handed to P2.1 |
| Falsifier: F$_{A2}$ follow-up | round-5 critic | R6.1 at T1 | DIAGNOSTIC-HOLDS |
| Falsifier: R5.2 cold-start follow-up | round-5 critic | R5.2 at T1 (two runs); new-instance run | GATE-REFUSED(G2, as the F$_B$ row) |

Exits of section 3b checked on 2026-11-17 (after the R7.4 T0 kill): no row is SURVIVED-T1 with a next step that passes the gate, and none is DIAGNOSTIC-FALSIFIED; every row is closed and the F$_B$ rows are GATE-REFUSED(G2), so exit 3 applies: EXHAUSTED at this budget.

Critic sign-off (2026-11-16): approved, State stays DORMANT (R5.2 and C3′ are AMBIGUOUS, not killed, so EXHAUSTED is blocked). Numbers checked against the body and the Compute Ledger; corrected the near-convergence ALE claim (a one-seed miss by 0.007 mHa, not a tie), scoped the WIBJ screening-bias answer to H$_4$, scoped the Hartree–Fock weights and defined the sign half of the Hard Core. The 2026-11-17 documentation pass restructured this summary without changing any verdict; it is not yet re-signed by a critic. 2026-11-17 (later): R7.4 KILLED at T0 (cost) by the orchestrator and the leaf moved to EXHAUSTED at this budget. Per-leaf critic, 2026-11-17: SIGN-OFF-WITH-CHANGES (accuracy claim re-scoped to $\Delta E\le0.03$ mHa, Jastrow–HF numbers labelled as estimates, grid-end count and argued-not-measured scaling stated); applied.
