# Problem: P4.2 Size-transferable fermionic NQS

### 1. Mathematical Abstraction & Barriers
* **Formulation**: Let $\Lambda_L$ be a lattice of linear size $L$ (here the $2\times L$ ladder, $V_L=2L$ sites) with Hamiltonian $\hat H_L$ (same couplings $t,t',J$ at every size), boundary phase $\phi=\pm1$, and electron number $N_e(L)=(1-\delta)V_L$ at fixed hole doping $\delta$. An ansatz family is a set of parameter spaces $\Theta_L$ and amplitudes $\psi^{(L)}_\theta:\{0,1\}^{2V_L}\to\mathbb{R}$, together with a **size map** $T_{L\to L'}:\Theta_L\to\Theta_{L'}$ (here $L'=2L$). Define the energy $E_{L'}(\theta)=\langle\psi_\theta\rvert\hat H_{L'}\lvert\psi_\theta\rangle/\langle\psi_\theta\vert\psi_\theta\rangle$, the converged energy $E^{\star}_{L'}=\min_{\theta\in\Theta_{L'}}E_{L'}(\theta)$ (in practice the best of long optimizations), the source optimum $\theta^\star_L$, and the zero-shot excess per site
$$e_{\rm tr}=\frac{E_{L'}\big(T_{L\to L'}\theta^\star_L\big)-E^\star_{L'}}{V_{L'}} .$$
The Minimal Contrast Foil transferred to $L'$ is the transferred correlator $J$ times a *fresh* mean-field geminal on $\Lambda_{L'}$, with excess $e_{\rm foil}$ defined the same way. **Solved** means: (a) $e_{\rm tr}<\tfrac12 e_{\rm foil}$ and $e_{\rm tr}$ small in absolute terms; (b) the deciding observable (long-range pair correlation $P(d)=\langle\hat\Delta_0^\dagger\hat\Delta_d\rangle$) of the transferred state is no worse than the foil's; (c) the number of optimization steps from $T\theta^\star_L$ to $E^\star_{L'}$ is a fraction of the cost from the foil or from scratch; (d) all of this on an instance retaining the Hard Core (doped, $t'\neq0$, non-trivial fermion sign).
* **The Node Barrier**: The bosonic parts (translation-equivariant CNN/transformer correlators, Jastrow $v(r)$) have size-independent parameters, so $T$ is the identity on them. The antisymmetric layer has no such map:
    * *Determinant layer.* A translation-invariant Slater determinant has orbitals fixed by Bloch's theorem to $e^{ik\cdot r}u_k$; its only content is the occupied set $K_{\rm occ}(L,N_e,\phi)$, which jumps discontinuously with $L$ (shell structure, open-shell degeneracies). Backflow orbitals are $N_e\times M$ matrices whose shape changes with $N_e$ and $M$.
    * *Pair-function layer (fundamental).* For any translation-invariant geminal $F_L(r)$ on a ring of length $L$, Poisson summation gives $F_L(x)=\sum_m\phi^m F_\infty(x+mL)=\tfrac1L\sum_{k\in\mathcal K_L}F_k e^{ikx}$, where $\mathcal K_L$ is the $L$-point momentum grid and $F_\infty$ the infinite-lattice pair function. The source cluster therefore determines only the **aliased** values of $F_\infty$; its range beyond $L/2$ is not identifiable from $L$ (a Nyquist-type barrier). If $F_\infty$ has algebraic tails (Fermi surface, $d$-wave nodes in 2D), no finite source fixes it.
    * *Identifiability at ED-sized sources (fundamental in practice).* With a few hundred configurations and tens of parameters the source optimum is highly degenerate, so the optimizer has no reason to select the size-intrinsic parameters; the mean-field part and any learned correction co-adapt.
    * *Practical barriers.* $\delta V_L$ must be an integer, the stripe wavelength $\lambda$ must divide $L$, closed shells are needed for a non-degenerate Slater reference (avoidable with a BCS geminal or twists), and the correlator's receptive field must be much smaller than $L_{\rm src}$ (otherwise it learns cluster-global features; observed in our smoke tests, where a 3-rung window on a 4-rung ring transferred badly).
    * *Where existing approaches stop.* Zen et al. (PRE 101, 053301, 2020; arXiv:1908.09883) transfer RBM parameters for spin and bosonic models; there is no antisymmetric layer. Staged $6\times6\to36\times36$ CNN transfer for spins (arXiv:2204.07816) has the same limitation. Gerard, Scherbela, Sutterud, Foulkes & Grohs (arXiv:2405.07599) transfer *continuum* neural wavefunctions for solids by mapping the new supercell's Hartree–Fock orbitals through a network optimized jointly over twists, lattice constants and supercell sizes. They report convergence to local minima and the difficulty of splitting compute across geometries; the route relies on multi-system training. Foundation NQS (Rende et al., arXiv:2502.09488) train one transformer over many Hamiltonians, in spin models, which is the excluded massive-compute route. The hidden-fermion Pfaffian (Chen et al., arXiv:2507.10705) keeps translation equivariance when it reshapes network outputs into the pairing matrix, but nobody has transferred it across sizes. mVMC (Misawa et al., arXiv:1711.11418) reduces $f_{ij}$ to $\mathcal O(N_s)$ parameters with sublattice symmetry, but refits every size. Kim et al. (Commun. Phys. 2024, doi:10.1038/s42005-024-01613-w) use size-independent pairing-orbital parameters for continuum cold Fermi gases. Our searches found no published lattice-fermion NQS transferred from $L$ to $2L$.
    * *Anti-Solution Space.* (i) Multi-size or multi-Hamiltonian pretraining (foundation models): excluded by the compute budget. (ii) Transferring only the bosonic part with a fresh determinant: this *is* the foil. (iii) Validating on a 1D open chain, at half filling with $t'=0$, or in the non-interacting limit. (iv) Claiming transfer from energy alone without checking $P(d)$. (v) "Transferring" a determinant by zero-padding or interpolating orbitals in $k$, which reintroduces the shell jump. (vi) A route whose $L$-cluster training needs all particle-number sectors.
### 1b. Analogy Map
Written 2026-11-17, after the rounds below, as T0 work: it re-reads the leaf's barriers in field-neutral terms and marks which cross-field ideas the leaf already tried. Status tags: *untried*, *tried (R<id>)* (the route card in the Route Register), *repaired (R<id>)* (the first port failed and was translated again).
* **B1. Extrapolate a local rule to a larger domain that contains local patterns never seen in training.** Field-neutral: learn a size-independent rule on small domains whose output on a larger domain stays accurate, although the larger domain has local configurations (here: 3- and 4-hole environments, bulk sites far from any edge) that no small domain contains.
    * *Graph-network size generalisation* (Yehudai, Fetaya, Meirom, Chechik & Maron, ICML 2021, arXiv:2010.08853).
        * Correspondence: graph $\leftrightarrow$ lattice cluster; a node's $d$-hop neighbourhood pattern $\leftrightarrow$ a site's hole environment; small training graphs $\leftrightarrow$ the 8-site sources; large test graphs $\leftrightarrow$ the $4\times4$ target.
        * Technique and why it worked: they prove that when local patterns differ between sizes, small-graph optima can be "bad global minima" on large graphs, and they fix it with a self-supervised task on the large graphs' local patterns. It works because classification labels do not depend on the unseen patterns once the encoder represents them.
        * Translation gap: here the only supervision at the target is the target energy, i.e. the expensive optimisation the transfer was meant to avoid. Showing the target's environments to the source is the environment library.
        * Status: *tried (R5.2, R9.4)*; both died (Gate 0, and dominated by scratch training at the target).
    * *Length generalisation of transformers* (Anil et al., NeurIPS 2022, arXiv:2207.04901).
        * Correspondence: sequence length $\leftrightarrow$ $L$; fine-tuned model $\leftrightarrow$ copied NQS; scratchpad $\leftrightarrow$ an explicit iterative growth procedure.
        * Technique and why it worked: plain fine-tuning does not length-generalise at any scale; making the model execute a repeated local step with visible intermediate states does, because the step itself is length independent.
        * Translation gap: a wavefunction amplitude has no supervised intermediate states. The closest analogue is growing the system step by step (gluing, ring surgery), which hits a conservation law or the metastable-branch anti-pattern.
        * Status: *tried (R3, R2.1)* in its growth form; the scratchpad form is *untried* and has no lattice-fermion translation we could write.
    * *Equivariant CNN and graph-network NQS for spins* (Kochkov et al., arXiv:2110.06390; Liang et al., arXiv:2204.07816).
        * Correspondence: shared convolution weights $\leftrightarrow$ size-independent parameters; spin amplitude $\leftrightarrow$ fermion amplitude.
        * Technique and why it worked: copying the weights to a larger lattice gives a good warm start because spin amplitudes are positive (or Marshall-signed) and their correlations are local.
        * Translation gap: the fermion antisymmetric layer has no size map, and a receptive field that grows with $L$ is the published transformer or CNN backflow route.
        * Status: *tried (R2.2, R5.6, R9.3)*.
* **B2. Recover a translation-invariant kernel's long-range tail from a periodic domain shorter than the tail (aliasing).** Field-neutral: a function sampled on a coarse grid of a fixed domain must be evaluated on a finer grid, and the source fixes only its aliased values.
    * *Wannier interpolation in band theory* (Marzari, Mostofi, Yates, Souza & Vanderbilt, Rev. Mod. Phys. 84, 1419, 2012).
        * Correspondence: coarse $\mathbf k$-mesh $\leftrightarrow$ source momentum grid; maximally localised Wannier function $\leftrightarrow$ pair kernel $\kappa(\mathbf r)$; fine mesh $\leftrightarrow$ target grid.
        * Technique and why it worked: a short-range real-space object fixes its Fourier series from a coarse mesh. It works because Wannier functions of insulators decay exponentially.
        * Translation gap: nodal $d$-wave or Fermi-surface pair kernels decay algebraically, and at ED sizes the whole kernel is inside the aliasing range.
        * Status: *tried (R2.2, R3.1)*: the momentum-field kernel and the grid-complete source.
    * *Neural operators* (Li et al., Fourier neural operator, arXiv:2010.08895).
        * Correspondence: fixed physical domain refined in resolution $\leftrightarrow$ fixed Brillouin zone refined when $L$ doubles.
        * Technique and why it worked: a network on a fixed domain is invariant to grid refinement, because the target function is smooth on that domain.
        * Translation gap: the first port (real-space modes $2\pi j/L$) changes physical momentum when $L$ doubles; the repair puts the neural field on the Brillouin zone.
        * Status: *repaired (R2.2)*; the real-space reading is *tried (R2.4)*.
    * *Analytic finite-size corrections in quantum Monte Carlo* (Chiesa, Ceperley, Martin & Holzmann, PRL 97, 076404, 2006).
        * Correspondence: missing long-wavelength part of the structure factor $\leftrightarrow$ missing long-range part of the pair function; random-phase approximation at small $k$ $\leftrightarrow$ the mean-field BCS kernel regenerated at the target.
        * Technique and why it worked: the long-wavelength physics is universal (plasmons), so the part the supercell cannot hold is computed analytically from one simulation.
        * Translation gap: the pairing tail is not fixed by a universal small-$k$ law; the mean-field regeneration covers only the one-body part.
        * Status: *tried (R1)*, whose mean-field part is exactly this idea.
    * *Multigrid prolongation* (Briggs, Henson & McCormick, A Multigrid Tutorial, SIAM 2000).
        * Correspondence: coarse grid $\leftrightarrow$ source; prolongation operator $\leftrightarrow$ size map.
        * Technique and why it worked: smooth error components are represented on the coarse grid and interpolated.
        * Translation gap: rescaling real space changes the Fermi wavevector, i.e. the density.
        * Status: *tried (R2.4)*.
* **B3. Pick, from a large manifold of exact fits to a small data set, the one whose extrapolation is right (identifiability).** Field-neutral: the training loss is flat along most parameter directions, and those directions decide the prediction off the training set.
    * *L2-SP transfer regularisation* (Li, Grandvalet & Davoine, ICML 2018, arXiv:1802.01483).
        * Correspondence: pre-trained weights $\leftrightarrow$ mean-field prior; fine-tuning $\leftrightarrow$ source fit; target task $\leftrightarrow$ $4\times4$ target.
        * Technique and why it worked: an explicit penalty toward a reference point keeps the fit near a solution known to be reasonable, because pre-trained features are already close to the target task.
        * Translation gap: the anchor fixes the null directions, but the family's capacity at the target (5.7% error) makes a faster route to it worthless.
        * Status: *tried (R5.1)*.
    * *Minimum-norm (Tikhonov) selection in inverse problems* (Tikhonov & Arsenin, Solutions of Ill-Posed Problems, 1977).
        * Correspondence: the null space of the forward map $\leftrightarrow$ the source-null directions of the SR metric.
        * Technique and why it worked: in a linear problem the minimum-norm solution is unique and independent of the starting point.
        * Translation gap: in a nonlinear network the norm is not reparameterisation invariant; measured seeds were further apart with the penalty (1.30 to 2.00) than without (1.05 to 1.75).
        * Status: *tried (R3.1)*; its linearised repair is *tried (R5.b)*.
    * *Query-by-committee active learning* (Seung, Opper & Sompolinsky, COLT 1992).
        * Correspondence: committee members $\leftrightarrow$ source seeds; disagreement region $\leftrightarrow$ configurations whose transferred sign differs between seeds.
        * Technique and why it worked: disagreement locates informative inputs when the hypothesis class contains the truth.
        * Translation gap: measured recall of the sign error in the disagreement set was 0.345 at a cost of 0.394 of the weight (premise rule: at least 0.5 at most 0.25).
        * Status: *tried (R2.3)*.
    * *Domain randomisation and multi-geometry training* (Tobin et al., arXiv:1703.06907; Gerard et al., arXiv:2405.07599).
        * Correspondence: randomised simulators $\leftrightarrow$ several source clusters and twists.
        * Technique and why it worked: many sources pin the shared parameters, because the target is inside the training distribution.
        * Translation gap: on lattices the sources are a handful of 8-site clusters, and the method is the published one.
        * Status: *tried (R3.3, R3.1)*.
* **B4. A parameterisation tied across positions (so it transfers) must still represent position-dependent structure near edges.** Field-neutral: a weight-shared model has the capacity of a few lookup tables, while the target's error sits in position-specific structure.
    * *Boundary layers and matched asymptotic expansions* (Van Dyke, Perturbation Methods in Fluid Mechanics, 1975), and boundary-layer meshes in finite elements.
        * Correspondence: inner solution near a wall $\leftrightarrow$ edge sites; outer solution $\leftrightarrow$ the bulk kernel; layer width $\leftrightarrow$ the depth cut-off $c$.
        * Technique and why it worked: the layer has a fixed width in scaled variables, so one inner rule serves every domain size.
        * Translation gap: on ED-sized open clusters every site is within depth 1 of an edge, so there is no bulk to separate from.
        * Status: *tried (R5.5, R9.1)*.
    * *Spectral graph filters and their transferability* (Defferrard, Bresson & Vandergheynst, arXiv:1606.09375; Levie, Huang, Bucci, Bronstein & Kutyniok, JMLR 2021, arXiv:1907.12972).
        * Correspondence: graph Laplacian $\leftrightarrow$ the cluster's hopping matrix; filter coefficients $\leftrightarrow$ size-independent pair-kernel parameters.
        * Technique and why it worked: a polynomial filter of the graph operator transfers between graphs that sample the same continuum; boundary resolution comes from the operator itself.
        * Translation gap: transfer works by construction, but a 14-coefficient channel adds no capacity (gain 1.02 times at step 300).
        * Status: *tried (R9.2)*.
    * *Symmetry restoration by projection in nuclear structure* (projected Hartree–Fock–Bogoliubov; Ring & Schuck, The Nuclear Many-Body Problem, 1980).
        * Correspondence: a symmetry-broken intrinsic state $\leftrightarrow$ the mirror-breaking free pair matrix; angular-momentum projection $\leftrightarrow$ the irrep projector.
        * Technique and why it worked: a broken-symmetry mean field captures correlations cheaply, and projection restores the quantum numbers.
        * Translation gap: the gain from projection is a finite-size gain, and the broken pattern needs a size map (a fixed period that is not a known order at $\delta=1/4$).
        * Status: *tried (R9.7)*.
    * *Real-space renormalisation and MERA* (Corboz & Vidal, PRB 80, 165129, 2009; Li & Wang, PRL 121, 260601, 2018).
        * Correspondence: blocking $\leftrightarrow$ the size map run backwards.
        * Technique and why it worked: at a fixed point, coarse-graining is scale-free.
        * Translation gap: a doped $t$–$J$ ground state is not a fixed point, and blocked sites have a 9-dimensional local space.
        * Status: *tried (R2.5)*.
* **B5. Follow the lowest state through a change of the problem when the lowest branch crosses or avoids others.** Field-neutral: a local optimiser warm-started from a related problem's solution stays on that solution's branch.
    * *Homotopy continuation* (Allgower & Georg, Numerical Continuation Methods, 1990).
        * Correspondence: homotopy parameter $\leftrightarrow$ seam coupling $\lambda$; path tracking $\leftrightarrow$ warm-started SR.
        * Technique and why it worked: isolated roots are tracked with predictor–corrector steps, and all roots are tracked, so none is lost.
        * Translation gap: VMC tracks one root, and avoided crossings with tiny gaps are crossed diabatically.
        * Status: *tried (R2.1)*.
    * *Variation after projection* (Ring & Schuck, 1980).
        * Correspondence: comparing projected energies in each quantum-number sector $\leftrightarrow$ the zero-shot sector rule.
        * Technique and why it worked: the sector is chosen by an energy comparison, not by following a branch.
        * Translation gap: the comparison is only as good as the copied state; the rule picked the right irrep in 2 of 3 seeds.
        * Status: *tried (R3.1)*.
    * *Replica exchange* (Swendsen & Wang, PRL 57, 2607, 1986; Hukushima & Nemoto, J. Phys. Soc. Jpn. 65, 1604, 1996).
        * Correspondence: replicas at several temperatures $\leftrightarrow$ several copied states or seeds; exchange moves $\leftrightarrow$ swapping parameters between runs.
        * Technique and why it worked: hot replicas cross barriers and feed cold ones, because the landscape is a Boltzmann weight.
        * Translation gap: a VMC energy surface has no temperature, and $K$ replicas cost $K$ times the target optimisation the transfer was meant to save.
        * Status: *untried*.
