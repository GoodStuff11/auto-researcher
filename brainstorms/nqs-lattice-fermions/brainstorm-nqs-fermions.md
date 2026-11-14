# Hard-Problem Brainstorm: Neural Quantum States for Discrete Fermion Systems
## Summary
Goal: find novel, low-compute research directions that make neural quantum states (NQS) reliable for discrete fermion Hamiltonians (lattice Hubbard / $t$–$J$ models and second-quantized quantum chemistry). Mode B (parallel fan-out) was chosen by the user: every leaf subproblem is attacked at once by an independent worker, and the results are compiled in the Master Leaf Synthesis Matrix below.
**Outcome**: of nine leaf routes, six were killed after the adversarial critic (five at Gate 0 as relabels of published constructions), two are conditional on test redesign (P1.3 basin hopping, P3.1 A-Ritz for off-diagonal order parameters), and one diagnostic survives as the critical path: P1.1, which measures whether the sign or the amplitudes is the binding error of fermionic NQS on doped Hubbard clusters. Useful by-products are the $\Omega(N_e^3)$ local-energy lower bound for unrestricted backflow (P2.1), and the measurement that the foil's sign alone costs $0.104t$ on a $3\times4$ $t$–$t'$–$J$ cluster (P1.2 preview). Constraint from the user: routes that only work with massive compute (scaling up parameter count, foundation-model pretraining, multi-GPU-week runs) are out of scope.
Working files: this index, worker artifacts in `.subproblems/P{i}_<slug>.md`, and the decisive-test scripts in `tests/`.

---

## Invariants
<!-- SPEC_INVARIANTS: Immutable problem contract. Once filled, DO NOT modify during critic rounds or worker runs. -->
### 1. Mathematical Statement of the Problem
* **Input space**: A number-conserving fermionic Hamiltonian on $M$ spin-orbitals in second quantization,
$$\hat{H}=\sum_{pq}h_{pq}\,\hat{c}^\dagger_p\hat{c}_q+\tfrac{1}{2}\sum_{pqrs}V_{pqrs}\,\hat{c}^\dagger_p\hat{c}^\dagger_q\hat{c}_s\hat{c}_r ,$$
where $\hat{c}^\dagger_p$ ($\hat{c}_p$) creates (annihilates) a fermion in spin-orbital $p\in\{1,\dots,M\}$, $h_{pq}$ are one-body integrals and $V_{pqrs}$ two-body integrals. Two instance families: (i) lattice models, e.g. the Hubbard model $\hat{H}=-\sum_{ij\sigma}t_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma}+U\sum_i\hat{n}_{i\uparrow}\hat{n}_{i\downarrow}$ on an $L_x\times L_y$ lattice with hopping $t_{ij}$ (nearest $t$, next-nearest $t'$), on-site repulsion $U$, density $\hat{n}_{i\sigma}=\hat{c}^\dagger_{i\sigma}\hat{c}_{i\sigma}$, and hole doping $\delta=1-N_e/(L_xL_y)$; (ii) molecular Hamiltonians in a finite orbital basis. $N_e$ is the conserved electron number.
* **Ansatz**: an NQS is a parametrized map $\psi_\theta:\{0,1\}^M\to\mathbb{C}$ on occupation-number configurations $\mathbf{n}=(n_1,\dots,n_M)$ with $\sum_p n_p=N_e$, with parameters $\theta\in\mathbb{R}^{P}$, optimized by variational Monte Carlo (VMC), i.e. by minimizing $E(\theta)=\langle\psi_\theta\rvert\hat{H}\lvert\psi_\theta\rangle/\langle\psi_\theta\vert\psi_\theta\rangle$ estimated from samples $\mathbf{n}\sim\lvert\psi_\theta(\mathbf{n})\rvert^2$.
* **Target output**: the ground-state energy $E_0$ *and* ground-state observables that decide the physics, namely long-range correlation functions $C(r)=\langle\hat{O}_0^\dagger\hat{O}_r\rangle$ for spin, charge and pairing order parameters $\hat{O}_r$ (e.g. $d$-wave pair operator $\hat{\Delta}_r$) and their finite-size scaling.
* **Success criterion**: for system volume $V=L_xL_y$ (or $M$ orbitals),
$$\frac{\lvert E_\theta-E_0\rvert}{\lvert E_0\rvert}<\epsilon_E,\qquad \max_r\lvert C_\theta(r)-C(r)\rvert<\epsilon_C,\qquad \text{cost}=\mathcal{O}(\mathrm{poly}(V))\ \text{with a low-degree polynomial},$$
where the reference $E_0,C(r)$ is exact diagonalization (ED) where available and the best controlled method (DMRG on cylinders, constrained-path AFQMC with error bars) otherwise. **Compute budget**: a credible result must be reachable on at most a few GPU-days per data point; routes that need more are classified as massive compute.
### 2. The Hard Core
**Fermionic sign structure under strong correlation.** In the occupation-number basis, a fermionic Hamiltonian away from special points (half filling with particle–hole symmetry, $t'=0$) is non-stoquastic: its off-diagonal matrix elements carry both signs because of anticommutation. As a result, the ground-state sign function $s(\mathbf{n})=\mathrm{sgn}\,\psi_0(\mathbf{n})$ is not given by any efficiently computable proxy (the nodes of a mean-field Slater determinant are wrong in the doped, strongly correlated regime). At the same time, competing orders (stripes, $d$-wave superconductivity, phase separation) are separated by energies of order $10^{-3}t$ per site, which is below typical variational errors. No toy model may remove either half: non-stoquastic sign structure, or near-degenerate competing orders.
### 3. The Unacceptable Outcome (Failure Threshold)
A result meeting any of the following is worthless:
* It reaches a competitive variational energy but gets the deciding observable wrong: wrong long-range pairing or stripe correlation $C(r)$, or an order parameter set by the ansatz's built-in mean-field reference rather than by the data.
* Its runtime, sample count, or parameter count scales exponentially in $V$ or $N_e$, or its claimed accuracy requires massive compute (many GPU-weeks, scaling laws, pretraining on large datasets).
* It works only where the sign problem is absent or trivially cured: half filling with $t'=0$ on bipartite lattices, the 1D open chain (where the Jordan–Wigner sign is trivial), or the non-interacting limit.
* Its claimed improvement is matched within error bars by the Minimal Contrast Foil (below).
* It re-labels a published technique without fixing the reason that technique is not standard practice.
### 4. Minimal Contrast Foil & Known Baselines
* **Minimal Contrast Foil**: a Jastrow–Slater / Gutzwiller-projected BCS (Pfaffian) VMC wave function, i.e. a symmetry-projected mean-field reference times a two-body Jastrow factor $J(\mathbf{n})=\exp\big(\sum_{ij}v_{ij}n_in_j\big)$, optimized by stochastic reconfiguration (SR). This is the pre-NQS state of the art for doped Hubbard VMC and costs $\mathcal{O}(N_e^3)$ per configuration.
* **Known partial solutions (NQS side)**: neural-network backflow (NNBF; Luo & Clark, PRL 122, 226401, 2019; Loehr & Clark arXiv:2510.26906); hidden-fermion determinant states (HFDS; Robledo Moreno et al., PNAS 119, e2122059119, 2022); neural-network-augmented Pfaffians (Chen et al., arXiv:2507.10705); transformer backflow (Gu et al., arXiv:2507.02644; Viteritti et al., arXiv:2604.21978; Rende, Viteritti & Georges, arXiv:2608.12465); $t$–$J$ NQS (Lange et al., PRL 135, 136504, 2025); autoregressive fermionic NQS (PRR 7, 013122, 2025); second-quantized chemistry NQS (Zhao, Stokes & Veerapaneni, arXiv:2208.05637; arXiv:2408.07625); MinSR / SPRING optimizers (Chen & Heyl, Nat. Phys. 20, 1476, 2024; Goldshlager, Abrahamsen & Lin).
* **Non-NQS partial solutions**: DMRG on width-4 to width-8 cylinders, constrained-path AFQMC (Qin et al., PRX 10, 031016, 2020; Xu et al., Science 384, eadh7691, 2024), fixed-node DMC, iPEPS.
* **Why the known solutions are not universally adopted**:
    * Antisymmetric NQS layers (determinant, Pfaffian, backflow) introduce a **mean-field bias** and an $\mathcal{O}(N_e^3)$ per-amplitude cost; low-rank updates (Chen & Roth, Lrux, arXiv:2602.05255) mitigate this but do not remove it (Rigo et al. review, arXiv:2608.21291, §5.1).
    * Different architectures with comparable energies give **different intertwined orders** (filled versus half-filled stripes, with or without superconductivity: Gu et al. 2507.02644 vs. Roth et al. 2511.07566), i.e. the variational bias decides the physics (Viteritti et al. 2604.21978).
    * State-of-the-art accuracy has come from $\sim10^6$-parameter transformers and large GPU budgets; the "Scaling laws for NQS" paper (Rende et al., arXiv:2606.02794) finds V-score improving as a power law in training compute, with an exponent that gets worse with frustration. This is exactly the massive-compute route this brainstorm excludes.
    * Increasing NNBF network size saturates in practice; multi-determinant expansions, Lanczos steps or DMC give similar gains (Loehr & Clark 2510.26906). The bottleneck is the antisymmetric structure, not the network width.
    * In quantum chemistry, the local energy $E_{\mathrm{loc}}(\mathbf{n})=\sum_{\mathbf{n}'}H_{\mathbf{n}\mathbf{n}'}\psi(\mathbf{n}')/\psi(\mathbf{n})$ has $\mathcal{O}(M^4)$ connected configurations, which makes the cost per sample prohibitive (Zhao et al. 2208.05637).
### 5. Verified No-Go Theorems & Boundaries
* **Stoquasticity hardness**: deciding whether a local Hamiltonian can be cured of non-stoquasticity even by restricted local basis rotations is NP-complete (Marvian, Lidar & Hen, Nat. Commun. 10, 1571, 2019). Stoquastic local Hamiltonians form their own complexity class (Bravyi et al., quant-ph/0606140). Consequence: there is no general efficient sign "cure", and any route must exploit structure of the instance family.
* **Ground-state energy hardness**: the local Hamiltonian problem is QMA-complete, including for fermionic Hubbard-type models with site-dependent parameters. No route may claim poly-time convergence for arbitrary instances.
* **NQS entanglement bound**: the bipartite entanglement an NQS can carry is bounded as $S_A=\mathcal{O}(k\log N)$ in the number of scalar nonlinearities $k$ and system size $N$ (Paul, PRL 136, 120403, 2026). 2D fermions with a Fermi surface obey $S_A\sim L\log L$, so $k$ must grow at least as $\Omega(L)$.
* **Sign-learning complexity**: frustrated sign structures are not featureless worst cases: they contain exploitable Boolean-Fourier structure (Schurov et al., arXiv:2508.09870), and many-body sign structures behave like non-glassy Ising models (Westerhout, Katsnelson & Bagrov, Commun. Phys. 6, 275, 2023). This is evidence that structured routes are possible, not a guarantee.
* **tVMC is ill-defined in its conventional form** (Sinibaldi et al., Quantum 7, 1131, 2023). This is relevant only to dynamics, which was pruned below.

---

## Stage 1 — Survey & Two-Pass Decomposition
Primary field-consensus source: Rigo, Wurst, Nutakki, Schmitt & Kennes, *Neural quantum states in condensed matter: advances, best practices, and prospects*, arXiv:2608.21291 (Aug 2026), §5 "Future Directions" (roadblocks A: sign/phase learning, B: symmetry and representation constraints, C: dynamics). The extracted text is saved at `.subproblems/.rigo2026_review.txt`.
### Pass 1: Unpruned Candidate Pool
| ID | Dimension | Candidate subproblem | Evidence |
| :--- | :--- | :--- | :--- |
| C1 | Formal | Antisymmetric layers inherit mean-field nodal bias; no unbiased fermionic sign representation | Review §5.1 goal (i); Viteritti et al. 2604.21978 |
| C2 | Formal | Representability: entanglement and expressivity lower bounds for fermionic NQS with a Fermi surface | Paul PRL 136, 120403 (2026); review §5.2 |
| C3 | Formal | "Solved" is ill-defined when competing orders are closer in energy than the variational error | Review §5.1 last paragraph; Gu et al. vs. Roth et al. |
| C4 | Algorithmic | $\mathcal{O}(N_e^3)$ determinant/Pfaffian cost per amplitude | Review §5.1 goal (ii); Chen & Roth 2602.05255 |
| C5 | Algorithmic | Optimization trapped in wrong-sign local minima ("residual sign problem") | Szabó & Castelnovo PRR 2, 033075 (2020); review §5.2 |
| C6 | Algorithmic | MCMC autocorrelation and ergodicity in fixed-$N_e$ sectors; constrained autoregressive sampling | PRR 7, 013122 (2025); Zhao et al. 2208.05637 |
| C7 | Algorithmic | Local-energy cost $\mathcal{O}(M^4)$ connected elements in second-quantized chemistry | Zhao et al. 2208.05637; arXiv:2408.07625 |
| C8 | Validation | Energy-optimized NQS can get long-range order parameters wrong; no error bar on $C(r)$ | Viteritti et al. 2604.21978; Roth et al. 2511.07566 |
| C9 | Validation | No controlled extrapolation / error estimate for NQS (V-score is energy-only) | Wu et al., Science 386, 296 (2024); Loehr & Clark 2510.26906 |
| C10 | Validation | Symmetry enforcement versus expressivity and optimization stability (SU(2), point group, momentum) | Review §5.2 |
| C11 | Asymptotic | Finite-size scaling and size transfer: each $L$ is retrained from scratch; boundary artifacts mimic physics | Review §3.2 (stripe crossover was a boundary effect) |
| C12 | Asymptotic | Dynamics and spectral functions: tVMC ill-defined; parameter gap between time evolution ($10^3$ to $10^5$) and ground state ($10^6$) | Review §5.3; Sinibaldi et al. Quantum 7, 1131 |

### Pass 2: Bottleneck Audit
| ID | Blocking test | Root or symptom | Field consensus | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| C1 | Blocks C3, C8, C11 (wrong nodes give wrong order) | Root | Explicit (review §5.1) | **Keep → P1** |
| C5 | Blocks C1 in practice | Root of C1's optimization side | Explicit in spins; `[unverified]` for fermions | **Merge into P1** |
| C2 | Bounds C1 but does not block practice | Partly root | Explicit bound (Paul 2026) | Merge into P1 as context |
| C4 | Blocks C11 (large $L$) and the no-massive-compute constraint | Root | Explicit (review §5.1 goal ii) | **Keep → P2** |
| C7 | Blocks every chemistry application | Root | Explicit (Zhao et al.) | **Merge into P2** |
| C6 | Blocks large-$L$ runs | Symptom of C4 and C7 cost plus sector constraints | Partial | Merge into P2 |
| C3 | Blocks interpretation of every result | Root of validation | Explicit (review §5.1; conflicting stripe results) | **Keep → P3** |
| C8 | Blocks the Unacceptable-Outcome check | Root | Explicit (Viteritti et al. 2604.21978) | **Merge into P3** |
| C9 | Blocks C3 | Root | Explicit (Wu et al.; Loehr & Clark) | **Merge into P3** |
| C10 | Blocks C3 (sector mixing) and C11 | Root | Explicit (review §5.2) | **Keep → P4** |
| C11 | Blocks thermodynamic-limit claims | Symptom of C4 plus C10 | Explicit (review §3.2) | Merge into P4 |
| C12 | Blocks nothing in the ground-state tree | Separate problem; downstream of sign/phase learning (C1) | Explicit (review §5.3) | **Pruned**: ground-state reliability comes first, and NQS dynamics at useful sizes is not low-compute today |

### Final Tree
**P1 — Sign structure beyond mean-field nodal bias.**
* *Statement*: Fermionic NQS get their antisymmetry from determinant, Pfaffian or backflow layers whose nodal structure is anchored to a mean-field orbital set, and optimization does not reliably escape that anchor.
* *Why the field cares*: An unbiased sign representation would turn architecture-dependent intertwined orders into answers. It blocks P3 (certification) and the thermodynamic-limit part of P4.
* *Evidence*: Review §5.1 goal (i); Viteritti et al. 2604.21978; Loehr & Clark 2510.26906 (NNBF width saturates); Szabó & Castelnovo PRR 2, 033075 (residual sign problem).
* *Current direction and why*: Larger backflow/transformer networks feeding a single determinant or Pfaffian, plus multi-determinant or Lanczos corrections. This direction is chosen because it is drop-in compatible with existing VMC codes.
* *Feasibility*: Open and contested. The general case is hard (Marvian et al. 2019), but structured instance families are not ruled out (Schurov et al. 2508.09870).
* **P1.1 [leaf] — Error budget: sign versus amplitude.** For a state-of-the-art fermionic NQS (HFDS or NNBF) on a small doped Hubbard cluster, what fraction of the energy error and of the pairing-correlation error comes from wrong signs $\mathrm{sgn}\,\psi_\theta\ne\mathrm{sgn}\,\psi_0$ versus wrong amplitudes? This decides whether P1 is the binding constraint at all.
* **P1.2 [leaf] — Antisymmetric layers without a fixed nodal anchor.** Is there an antisymmetric construction at $\mathcal{O}(N_e^3)$ cost (or slightly above) whose nodal surface is provably more expressive than a backflow single determinant or Pfaffian, without a sum over many determinants?
* **P1.3 [leaf] — Escaping wrong-sign basins during optimization.** Can VMC optimization be made to change the sign structure of a fermionic NQS cheaply, rather than staying in the basin set by the initial mean-field state?

**P2 — Cost of fermionic NQS evaluation.**
* *Statement*: Per-sample cost is dominated by the $\mathcal{O}(N_e^3)$ antisymmetric layer (lattice) or the $\mathcal{O}(M^4)$ local-energy sum (chemistry), which together with sampling sets the compute bill.
* *Why the field cares*: Large-$L$ studies (three-band cuprates, nickelates, moiré Hubbard; review §5.1) and chemistry beyond ~100 orbitals become reachable without massive compute. It blocks P4's finite-size scaling.
* *Evidence*: Review §5.1 goal (ii); Chen & Roth 2602.05255; PRX Intelligence 1, 023001 (2026) convolutional backflow $\mathcal{O}(N^4)\to\mathcal{O}(N^3)$; Zhao et al. 2208.05637.
* *Current direction and why*: Low-rank updates, GPU batching, and local-energy screening heuristics. These are engineering-compatible, but they keep the asymptotic cost.
* *Feasibility*: Tractable to open, depending on whether exactness is required.
* **P2.1 [leaf] — Sub-cubic antisymmetric amplitudes and fast local updates.** Can the amplitude ratios $\psi(\mathbf{n}')/\psi(\mathbf{n})$ needed for local energies of a backflow-type state be computed in $o(N_e^3)$ per configuration (amortized), for example by exploiting locality of backflow, without changing the variational family?
* **P2.2 [leaf] — Unbiased cheap local energy for second-quantized chemistry.** Can $E_{\mathrm{loc}}$ be replaced by an unbiased stochastic estimator over the $\mathcal{O}(M^4)$ connected configurations, with variance bounded so that the total cost to reach chemical accuracy scales better than exact enumeration?

**P3 — Certifying what an NQS says about competing orders.**
* *Statement*: There is no controlled way to tell whether a variational NQS result for an order parameter is physics or ansatz bias when competing orders lie within the variational energy error.
* *Why the field cares*: Without this, NQS cannot settle the doped-Hubbard question it is being applied to (Gu et al. vs. Roth et al. vs. Rende, Viteritti & Georges). It blocks the use of any P1/P2 improvement as evidence.
* *Evidence*: Viteritti et al. 2604.21978 ("variational bias"); Wu et al., Science 386, 296 (2024) (V-score is energy-only); Loehr & Clark 2510.26906 (Lanczos and DMC corrections).
* *Current direction and why*: Comparing several architectures, plus V-score and variance extrapolation of the energy. These are cheap to report but do not bound order-parameter error.
* *Feasibility*: Open.
* **P3.1 [leaf] — Order-parameter error from cheap information.** Can the error of a long-range correlation $C_\theta(r)$ be bounded or extrapolated from quantities a VMC run already has (energy variance, a single Lanczos or power step, response to a small pinning or pairing field), with a validated relation on ED-sized doped clusters?
* **P3.2 [leaf] — Bias-diagnosing biased-ensemble tests.** Can a deliberately biased pair of runs (initialized or constrained toward each competing order, e.g. filled versus half-filled stripe) be turned into a controlled decision rule using energy-variance extrapolation of each branch to zero variance, in the spirit of level-crossing detection?

**P4 — Symmetry and system-size transfer.**
* *Statement*: Enforcing the fermionic symmetries (SU(2) spin, lattice point group, momentum) and moving from one lattice size to the next both cost either expressivity, optimization stability, or a full retraining.
* *Why the field cares*: Symmetry-resolved energies separate competing sectors (review §5.2), and size transfer is the cheapest route to finite-size scaling without massive compute.
* *Evidence*: Review §5.2; Rende, Viteritti & Georges 2608.12465 (symmetry-preserving NQS for $t$–$t'$ Hubbard); review §3.2 (boundary artifacts).
* *Current direction and why*: Group-averaged projection (cost $\propto\lvert G\rvert$ for symmetry group $G$) and translation-equivariant transformers, chosen because they reuse standard tooling.
* *Feasibility*: Tractable for Abelian symmetries; open for SU(2) with determinant layers and for size transfer.
* **P4.1 [leaf] — Exact spin-SU(2) for fermionic NQS at sub-projection cost.** Can the total-spin sector $S$ be imposed on a determinant- or Pfaffian-based NQS exactly, at cost below the $\mathcal{O}(N_{\mathrm{grid}})$ spin-projection quadrature, without sacrificing backflow expressivity?
* **P4.2 [leaf] — Size-transferable fermionic NQS.** Can a translation-equivariant fermionic NQS trained at $L$ initialize (or directly evaluate) $2L$ with an energy error per site that is already close to converged, so that finite-size scaling needs a fraction of the per-size retraining cost?

---

## Stage 1.5 — Execution Routing
| Leaf | Title | Parent |
| :--- | :--- | :--- |
| P1.1 | Sign versus amplitude error budget | P1 |
| P1.2 | Antisymmetric layers without a fixed nodal anchor | P1 |
| P1.3 | Escaping wrong-sign basins during optimization | P1 |
| P2.1 | Sub-cubic antisymmetric amplitudes and fast local updates | P2 |
| P2.2 | Unbiased cheap local energy for second-quantized chemistry | P2 |
| P3.1 | Order-parameter error from cheap information | P3 |
| P3.2 | Bias-diagnosing biased-ensemble tests | P3 |
| P4.1 | Exact spin-SU(2) for fermionic NQS at sub-projection cost | P4 |
| P4.2 | Size-transferable fermionic NQS | P4 |

**Routing decision**: Mode B (parallel fan-out) over all nine leaves, as the user asked when invoking the skill.
Compute environment: Slurm cluster (`sbatch` at `/usr/local/slurm/current/bin/sbatch`). Workers write `tests/P{i}_*/micro_test.py` and `submit_test.sbatch` and do not submit. The orchestrator submits only the critical-path test (see Cross-Pollination).

---

## Master Leaf Synthesis Matrix
All nine workers finished, each with one route that passed its own gates. The adversarial critic pass then killed six and made two conditional (see Revision & Critic Log). The Status column shows the post-critic verdict; the other columns record what the worker proposed. "Pre-check" is the login-node smoke or preview result: it checks the code, not the verdict, but where it already leans FAIL that is recorded. No decisive test has run yet.

| Node | Subproblem | Status | Top surviving route | Sharpest gap | 24-hour decisive test (metric; PASS) | Pre-check | Cross-node dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1.1** | Sign versus amplitude budget | **PASS (job 329057): sign is the binding error** | Fixed-sign split: $E_{\mathrm{sign}}=\min_{a\ge0}a^{\mathsf T}SHSa/a^{\mathsf T}a$ with $S=\mathrm{diag}(\mathrm{sgn}\,\psi_\theta)$, so $\Delta E_{\mathrm{tot}}=\Delta E_{\mathrm{sign}}+\Delta E_{\mathrm{amp}}$, both non-negative | Transfer of the budget from ED sizes to $8\times8$; no bound on $E_{\mathrm{sign}}-E_{\mathrm{stoq}}$ | Three $4\times3$ clusters, $t'=-0.25$, NNBF widths 8 to 128; $M=f_E=\Delta E_{\mathrm{sign}}/\Delta E_{\mathrm{tot}}$; PASS (sign binding) if $M\ge0.6$ on all | 6-site: amplitudes dominate (uninformative size) | Gates P1.2, P1.3; feeds P3.1 |
| **P1.2** | Antisymmetry without nodal anchor | KILLED (Gate 0 after critic) | Residual Walsh sign layer $\psi=s_{\mathrm{foil}}\,c(\mathbf{n})\,A(\mathbf{n})$ with plaquette-local degree-3 sign terms | Is the foil's residual sign error local and low-degree in fixed-node energy? | $4\times4$ $t$–$t'$–$J$, 2 holes; $M_{\mathrm{local}}=\Delta E_{\mathrm{FN}}[s_{\mathrm{foil}}c]/\Delta E_{\mathrm{FN}}[s_{\mathrm{foil}}]$; PASS if $M\le0.5$ and beats non-local null by 0.1 | $3\times4$: $M_{\mathrm{local}}=1.03$ | Needs P1.3 to learn $c$ variationally |
| **P1.3** | Escaping wrong-sign basins | CONDITIONAL (test needs a matched-restart null arm) | Orbital-swap basin hopping (OSBH): occupied/virtual reference-orbital swaps with short SR relaxation, under an equal-budget comparison | Do mean-field starts actually land in a wrong-sign basin ($G>0$)? | $4\times3$ PBC, $t'=-0.3$, $U=8$, exact-sum SR; $M=(\epsilon_{\mathrm{SR}}-\epsilon_{\mathrm{OS}})/G$; PASS if $G>0.25\epsilon_{\mathrm{SR}}$, $M\ge0.5$ | $3\times2$: no swaps accepted (uninformative) | Premise set by P1.1; shares ranking risk with P3.2 |
| **P2.1** | Sub-cubic antisymmetric amplitudes | KILLED (Gates 0 and 2 after critic) | Local + Global-Bottleneck backflow (LGB-BF): finite-range rows plus rank-$2K$ global channels; Woodbury Green's function gives $E_{\mathrm{loc}}$ in $\mathcal{O}(N_e^2)$ | Does fixed small $K$ close the local-versus-global backflow accuracy gap (Liu–Clark: low-rank corrections underperform)? | $L=8$ to 32 timing; $M$ = slope of $E_{\mathrm{loc}}$ time in $N_e$; PASS if $M\le2.3$ with ratio error below $10^{-9}$ | Ratios exact to $10^{-14}$; timing AMBIGUOUS | Cheap carrier for P1.2, P4.1 layers |
| **P2.2** | Cheap local energy for chemistry | KILLED (measured residual fraction); successor P2.2′ opened | Frozen-backflow control variate: exact sum with configuration-frozen orbitals plus importance-sampled residual | Residual fraction must be well below 1; measured 0.56 to 0.82 | H6/H8 STO-3G at $R=3.0$; $M$ = cost to fixed error relative to best of heat-bath/semistochastic/screening; PASS if $M\le0.3$ | H4 $M=1.07$; H6 $M=1.03$ near convergence | Only exists for determinant NQS (P1.2, P2.1) |
| **P3.1** | Order-parameter error from cheap info | CONDITIONAL (off-diagonal $A$ only; test redesign required) | A-Ritz: Rayleigh–Ritz in $\mathrm{span}\{\psi,H\psi,A\psi\}$ with the order operator $A$, zero-variance extrapolation of $\langle A\rangle$, abstain unless variance halves | Does $(A-a)\psi$ keep $\mathcal{O}(1)$ low-energy weight as $V$ grows? | Doped $4\times3$/$6\times2$, $U=8$; $M$ = trusted-and-within-factor-2 fraction; PASS if $M\ge0.7$, $M-M_L\ge0.2$, false-certificate rate $G\le0.05$ | Code checks only (no near-degenerate states in smoke) | Needs exact symmetry (P4.1); complements P3.2 |
| **P3.2** | Biased-ensemble decision rule | KILLED (Gates 0 and 3 after critic) | Joint two-state optimization (determinant of two states), roots labelled by measured order, per-root variance extrapolation | Clean root character and near-linear $E$–$\sigma^2$ paths near a crossing with gap far below the variational error | $6\times2$ ladder, $t'$ sweep across a real level crossing; $M$ = wrong-commit rate; PASS if $M\le0.02$ and decision rate $\ge0.5$ | Code checks only | Needs P3.1 for labels; same mechanism as P1.3 |
| **P4.1** | Exact SU(2) below projection cost | KILLED (Gate 0 after critic) | Charge-only backflow into a singlet pairing $F_c=B_cF_0B_c^{\mathsf T}$, plus spin-invariant bond-exchange dressing $\alpha_b(c)\hat{X}_b$; $S$ exact for all parameters | Large $U$ near half filling: charge configuration carries almost no information | $3\times4$ torus, $t'=-0.3$, $U=8$, $N_e=10$ singlet; $M=\epsilon_{\mathrm{r1x}}/\epsilon_{\mathrm{foil}}$; PASS if $M\le0.5$ and correlations no worse | $\langle S^2\rangle$ exact to $10^{-16}$; small-cluster accuracy favourable but under-determined | Lemma constrains P1.2 and P4.2 designs |
| **P4.2** | Size-transferable fermionic NQS | KILLED (Gate 3 after critic; pre-check $M=1.30$) | Residual-geminal transfer: mean-field $d$-wave geminal rebuilt per size, plus learned short-range residual $\Delta f$ carried over | Is the optimal $\Delta f$ short-ranged and size-stable (nodal $d$-wave predicts power-law tails)? | $t$–$J$ ladder $2\times4\to2\times8$; $M=e_{\mathrm{tr}}/e_{\mathrm{foil}}$; PASS if $M<0.5$ | $2\times4\to2\times6$: $M=1.30$ | Uses $S^z$ features, which conflicts with P4.1 |

### Pipeline State (pipelined orchestration, Mode B step 0)

| Leaf | State | Blocked on | Running job / agent |
| :--- | :--- | :--- | :--- |
| P1.1 | RESULT (PASS, qualified) | nothing | none |
| P1.1-P | RESULT: SYMMETRY-DOMINATED (all instances) | nothing | none |
| P1.2 | SUBMITTED (round 5 on 4×4 OBC: frdK AMBIGUOUS, rho_sign 0.84, rho_tot 0.92, 3/3 seed wins, disjoint ranges, beats separately trained ablation; frdX FAIL) | final 5-seed verdict blocked on seeds 44 and 55 | Slurm 553724 (8 tasks) |
| P1.3 | WAITING (width-6 after resume: still UNCLEAR-CONV; W 0.0115 vs cold best 0.0117, gap 10× below the 0.0029 BASIN margin, drift −6e-5; oracle W0 0.0195 is worse than cold, so fidelity ≠ sign. Per rule, the width trend now decides) | width-4 and width-10 runs | Slurm 507179 |
| P2.1 | SUBMITTED (decisive array; second instance qualified: 4×4 OBC 5+5 dE_sign 0.0147t, 5×3 OBC 6+6 0.0159t, both above the 0.01t bar; inst3 used per rule) | nothing | Slurm 526080 (85 tasks: inst0 + inst3) |
| P2.2 | SUBMITTED (round 5 reduced: accuracy trend + equal-cost foils + calibrated cost; no unconditional survivor) | nothing | Slurm 452238 (44 tasks) |
| P3.1 | PILOT (ESR0: set A 16/21 G=0 but set W 8/19 and a grade-H real false certificate; Gate 0 FAIL as method, PASS as diagnostic) | step 2 blocked on excited-state pilot (GO: eps/omega at most 0.4 per level, out-of-window weight at most 0.03) | Slurm 456094 (excited pilot); EHX gate 452146 recorded GATE-FAIL (EHX = Route 2.3 restated) |
| P3.2 | CRITIC (round 5: Eckart decision killed by exact pre-check, eps/Delta 0.19–0.55 so certified DEC 0/12; survivor LCC, linear crossing location from smooth branch energies) | nothing | per-leaf critic |
| P4.1 | SUBMITTED (rank ladder of projected UHF mixtures, K=1–8, drift control) | nothing | Slurm 508440 (22 tasks: che_rbm M=1–3, che_k K=2,4,8, containing foil, che drift to 4000 it) |
| P4.2 | SUBMITTED (capacity pilot MUTATE: 426-param MFNG-O plateaus at 6.0% rel. error, above 3%; projected NNBF reaches 0.35% at the same step count; re-pilot with enlarged family) | array blocked on enlarged re-pilot (at most 3%) | Slurm 604157 (channels 8, bf-hidden 32) |

Each leaf advances on its own as soon as it returns: lint, then its own critic, then fix or mutation, then submission. No leaf waits for its siblings.
* **P3.2 round 3: first Hard-Core-scale same-irrep competitor (2026-11-11)**:
    * **Where the crossing is.** With $t'$ and $U$ alone, the 4×3 torus has no exact same-irrep crossing (the gap valley runs to the excluded free limit $U\to0$); the 6×2 ladder reaches only $0.090t$. Adding anisotropic hopping $t_y$ as a second knob produces exact conical intersections, as expected from the codimension-2 rule.
    * **The test path.** On $4\times3$ PBC, $4+4$, $U=8$, $t_y=1.11394$, inside the irrep ($K=0$, $M_x=M_y=+1$, $S=0$; the global ground-state sector, dimension 1618):
        * the crossing is at $t'_c=-0.470482$;
        * the gap is $1.4\times10^{-3}t$ at $t'_c\pm0.0002$ and $0.010t$ at $\pm0.002$, which is the Hard-Core scale;
        * the two states are a period-4 spin stripe and a near-Néel state ($c_0$ flips sign across $t'_c$), with similar $d$-wave correlation.
    * **Other intersections:** $4+4$ at $U=8$ ($t'=-0.515243$, $t_y=1.143944$); $4+4$ at $U=5$; $5+5$ at $U=8$ ($t'=-0.5925$, $t_y=0.8442$, sector dimension 3752). They are in `tests/P3.2_biased_ensemble_r3/scan_logs/`, and were broadcast to the running P2.1, P4.1 and P4.2 workers.
    * **Survivors (critic pending).**
        * SO-EC: same-order eigenvector continuation from projected snapshots trained far from the crossing.
        * AT: affine-transport upper bounds.
    * **Kills.**
        * Kink/jump crossing detector, Gate 2: a $2\times2$ lemma puts every variational-energy rule on R0+'s decision boundary.
        * Curvature/susceptibility, Gate 1.
        * Berry phase, Gates 2 and 3.
        * In-irrep zero-variance extrapolation, Gate 0 (mVMC practice).
* **P4.1 round-4 per-leaf critic (10 objections, 7 Major)**:
    * **The #P-hardness result is correct but folklore-level.** An exponential sum over a circuit-computable $g$ is #P-hard by definition (cf. RBM normalisation, Long & Servedio). It is restated as "worst-case hardness plus a polynomial subclass, linear-$S^z$ features at $r=\mathcal{O}(1)$"; "dichotomy" and "theorem in both directions" are withdrawn.
    * **`sfb` is narrow.** Its single feature is the staggered magnetisation of the singly occupied sites, close kin to multi-component projected Hartree–Fock (PHF/ResHF), and it adds 16 parameters.
    * **KAGP is incremental.** It is Tocchio with $K>1$, and Gate 0 must cover multi-determinant NNBF (arXiv:2510.26906) and arXiv:2608.12465.
    * **The completeness count is local.** Full Jacobian rank does not imply that $\psi_0$ is reachable.
    * **The premise gate tests a tabular ceiling, not the trained capacity**, and `report()` does not enforce its premises.
    * **The thresholds are below seed noise**, and the initialisation is asymmetric between arms.
    * **Action.** Fix-then-submit, blocked on two jobs:
        * 411996 (tabular AGP fit). If $\epsilon_{\mathrm{tab}}(K{=}1)<10^{-4}$, or $<0.5\,m(\epsilon_{\mathrm{tocchio}})$, KAGP goes back to mutation.
        * 411494 (che vs nnbf_SG). If charge-only matches spin-resolved after projection, `sfb` has no headroom.
* **P2.2 round-4 per-leaf critic (10 objections, 6 Major)**:
    * **The pre-check was tautological.** At $K=8$ there is one external orbital per shell, so CGK equals NNBFJ, and $v$ is training noise. The honest $M_{\mathrm{op}}\approx0.54$ would be AMBIGUOUS.
    * **Accuracy is tested at no more than 3 orbitals per shell; cost is certified at 16–64.** The information lost is exactly the dynamic pair correlation that grows with the basis. This is the known failure mode of externally contracted MRCI.
    * **Training noise is about $\tau$**, and NNBFJ is under-trained.
    * **The controls are capped or on a knife edge.** The SS $\epsilon$ grid stops at $3\times10^{-3}$, and SCREEN's bias of $0.18$–$0.19$ mHa sits right at $\tau=0.2$.
    * **The op-count model favours KEYX about $2\times$ over wall-clock**, and the veto is toothless.
    * **Gate 0 is narrower than claimed.** The key structure is GAS/RAS occupation-class blocking from CI (Olsen; LUCITA), and the barrier is the contraction error of externally contracted MRCI. The R4.0 kill is sound; its anti-pattern is narrowed to per-class dressing with $N_\gamma\propto K$.
    * **Action.** The 34-task array is not submitted. A large-$K$ accuracy pilot runs as job 411919: H4 at $R=4.4$ bohr, HF weight about 0.49, $K=12/16/20$ (8–12 orbitals per shell), arms NNBFJ/CGK/WIBJ, 3 seeds. Pre-committed rule:
        * kill R4.1 if the median $\delta_{\mathrm{CGK}}>\tau$ at $K=20$;
        * otherwise, before any array runs: recompute $M_{\mathrm{op}}$ honestly, extend the SS and SCREEN grids, add the noise-floor arm and best-of-seeds scoring, require a wall-clock advantage, and add the H4 large-$K$ instance to Part A.
* **P4.1 round-3 per-leaf critic (12 objections, 9 Major)**:
    * **SQ-RVB fails Gate 0.** For a symmetric charge-backflow pairing, $\det F_c[U_s\cup D_b,D_s\cup D_b]=\det F_c[D_b,D_b]\,\det S_c[U_s,D_s]$, where $D_b$ is the doublon set and the Schur complement $S_c$ is symmetric. SQ-RVB is therefore the round-1 Tocchio family, reparameterised, and costs two determinants against Tocchio's one.
    * **Che's projection was benchmarked sub-optimally.** It is closed-form via elementary symmetric polynomials of the eigenvalues of $D_m^{-1}X$, at $\mathcal{O}(N_e^3)$, so the cost "advantage" is a constant factor.
    * **SCF's cost was measured at a toy bond dimension**, and the cost and accuracy parameterisations differ.
    * **Completeness is untested.** Whether one AGP per charge configuration can reach $\psi_0$ was never checked.
    * **Width-4 accuracy measures only capacity per parameter.**
    * **Action.** The 42-task array is not submitted.
        * A 9-task width-16 diagnostic pilot runs as job 411494 (sqrvb, nnbf_SG and che, 3 seeds). It measures charge-only against spin-resolved conditioning, both exactly projected.
        * P4.1 goes to round 4 with a **Saturation Diagnostic**: "exact $S$ below projection cost" is already met by standard constructions (singlet pairing with charge-only correlators at one determinant; Che's closed form at $\mathcal{O}(N^3)$).
        * The leaf is refined to what remains open: exact $S$ for spin-resolved backflow, the cost of charge-only conditioning, and exact-$S$ forms beyond one AGP per charge configuration.
* **P2.2 round-3 per-leaf critic (10 objections, 6 Major)**:
    * **The test could not PASS globally.** The H8 key/term ratio is 0.55.
    * **ED-sized symmetric rings cannot show the $M$-scaling.**
    * **WIBJ only changes the prefactor.** It is active-space backflow times a Jastrow, and its 2×2 minors are the standard table trick.
    * **PFCV is a relabel of R2.0.**
    * **The ΔE tolerance looks tuned after the fact.**
    * **The controls lack the truncated local energy** (arXiv:2502.18843) and HCI-style heat-bath sampling.
    * **The proposed Cholesky "infinite-variance" anti-pattern is rejected.** The correct statement is the support condition $\mathrm{supp}(L_\gamma\psi)\subseteq\mathrm{supp}(\psi)$.
    * **Action.** Not submitted; sent to round-4 mutation with the critic's direction: a key-wise AFQMC-type contraction (Cholesky/THC) of the external-excitation sum, at about $\mathcal{O}(N_{\mathrm{key}}N_\gamma NM)$, under a Jastrow factorizable across the window boundary. The test has two parts: ED-free scaling at $K=50$–$200$, and accuracy on genuinely multireference instances (HF weight below 0.6).
* **Cross-leaf finding (2026-11-11): the error left after projection is amplitude-borne.** P1.1-P width-8 budgets on both $5+5$ tori, trained with projection (3 seeds): $\Delta E_{\mathrm{tot}}=0.9$–$20\times10^{-4}$, $\Delta E_{\mathrm{sign}}\le5\times10^{-5}$, $f_E=0.004$–$0.023$, wrong-sign weight $10^{-8}$–$10^{-6}$. The $4+4$ instance is complete and SYMMETRY-DOMINATED; the widest width on $5+5$ is still running. P2.1's critic, reading the P1.2 pilot's interim logs, finds the same on the 4×3 OBC and cylinder clusters. Consequences:
    * On ED-sized Hubbard clusters, symmetry-projected NNBF has essentially no sign error. What remains is amplitude error. P1.1's original "sign is binding" result was symmetry contamination.
    * Every leaf whose premise is "projected NQS keeps a sign error" (P1.2, P1.3, P2.1's premise gate, part of P3.1) is very likely NO-TARGET at ED size. The premise pilots will confirm or refute this per leaf.
    * The live question moves to sizes beyond ED: does a nodal error appear when the projected sector dimension greatly exceeds what the network can represent (P1.1's size-transfer gap)? Testing that needs sampled VMC with an independent reference (AFQMC or DMRG) on lattices of $6\times6$ or more, a different and more expensive test.
* **P2.1 round-3 per-leaf critic (12 objections, 8 Major)**:
    * **Gate 0, SK-LR.** It is HFDS with $K$ blocks and a frozen hidden block, linearised at $r=1$. Equivalently it is a neural-CI singles expansion with the Clark et al. table method, and its per-ratio cost is the lrux kernel.
    * **SK-AGP.** It does not fix the HFPS $\mathcal{O}(N^3)$ barrier, because no sweep is implemented.
    * **"The restriction bites" is false.** $\det(A_0+\phi c^{\mathsf T})=\det A_0+c^{\mathsf T}\mathrm{adj}(A_0)\phi$ with an unrestricted $c(\mathbf{n})$, so a rank-1 term already represents an arbitrary scalar times the reference.
    * **The control is handicapped.** NNBF has no Jastrow while the route arms do.
    * **The rank dominance test is weak.** Unpaired, $P\approx0.20$ under the null.
    * **Cost timings are interpreter-dominated.**
    * **The verdict can rest on a single instance.**
    * **Action.** The 118-task array is held. A control-only pilot runs as job 410988 (the nnbf arm, seed 11, three instances). Pre-committed rule:
        * if projected $\Delta E_{\mathrm{sign}}<0.003t$ on instances 0 and 1, the verdict is NO-TARGET and P2.1 goes back to mutation (no width halving);
        * if it is at least $0.01t$ on both, apply fixes 3, 5 and 6 and the Gate 0 restatements, then submit;
        * otherwise run 2 more seeds first.
* **P4.2 round-3 per-leaf critic (11 objections, 5 Major)**:
    * **Prior art.** The critic confirmed the worker: arXiv:2505.00233 keeps the lattice size fixed, and no lattice-fermion NQS size transfer is published. Gate 0 must still cite and distinguish arXiv:2606.17045 (single-hole, exactly solvable, local-rule size transfer) and arXiv:2512.11962 (a $t$–$V$ foundation NQS conditioned on Hamiltonian parameters).
    * **Route 3.2 is a relabel.** Freezing the determinant and refitting a Jastrow is standard fixed-node practice, and its sign verdict is decided by route 3.1's zero-shot object. It is merged into 3.1 as a diagnostic, which leaves P4.2 with one survivor.
    * **The minimum-norm selection is contradicted by its own pre-check.** With the penalty, the seeds are further apart (1.30–2.00 against 1.05–1.75). The norm of a nonlinear network is not reparameterisation-invariant, and about 110 source-null directions act on 3- and 4-hole features the source never sees.
    * **Capacity dominates the criteria.** They should score the excess over $\epsilon_{\mathrm{opt}}$ and tighten VOID.
    * **The P1 premise passes almost automatically**, because scrB is scrL after 20 steps. The gate should be on steps-to-$\epsilon$ at several $B$, with a plateau check.
    * **Action.** The 9-task array is held. A zero-shot pilot runs as job 410960 (3 transfer tasks, `--budget 0 --steps-long 0`). Pre-committed rule:
        * kill 3.1 if the spread of $\epsilon_{\mathrm{tr0}}$ exceeds half the median, or if the sector-pick gate (S) fails;
        * otherwise apply fixes 3, 4, 5, 8 and 10 before any array runs.
* **P3.2 round-3 per-leaf critic (12 objections, 9 Major)**. The critic computed the oracle cases from exact ED snapshots.
    * **AT is killed.**
        * Even with exact snapshots it commits the wrong decision at $d=-0.0002$. The two diabats have unequal curvature ($d^2E/dt'^2pprox-14$ vs $-10$), which gives a differential transport bias of about $2.1\times10^{-3}$, above the $1.4\times10^{-3}$ gap.
        * It is also the variational principle plus a zero-step warm-start (hysteresis) protocol, and it falls under the leaf's own $2\times2$ lemma.
    * **SO-EC is conditional.**
        * It is exact on all 10 targets with ED snapshots, but its signal lives in an overlap eigendirection of about $4\times10^{-6}$, roughly $30\times$ below the sampling standard error at $2\times10^4$ draws.
        * Shared ansatz bias passes through the Ritz solve, so it can only beat the control where the control's branches cross over at the target.
        * Multi-vector EC near avoided crossings already exists (Frame et al. 1711.07090; review 2310.19419); the new element is only the split by measured order.
    * **Test defects.**
        * The R0P control is cold-started; a warm-started R0P-warm control is needed.
        * The snapshot-certification gate is inverted: it certifies branch collapse.
        * The sampled arm still trains on exact forces.
        * The aggregator does not check target lists.
    * **Instance verified, as a methods instance only.** It is a genuine same-irrep near-degeneracy with the correct 48-element group, but the two "orders" are spin textures forced by the odd $L_y=3$ (stripe at $(\pi/2,0)$ vs a Néel state at $(\pi,\pm2\pi/3)$, equal $P_d$). It is not stripe vs $d$-wave, and no physics transfer may be claimed from it.
    * **Action.** The 9-task array is held. A control-only pilot runs as job 410917: width 8, seed 1234, the 4 inner targets, about 5.4 h, output in `results_pilot/`. Pre-committed regimes:
        * **NO-TARGET**: drop width 8.
        * **NO-ROOM**: the R0P control is correct everywhere, so the array is a guaranteed FAIL and is not submitted.
        * **INTRINSIC-BIAS**: add R0P-warm and fix the certification gate first.
        * **GO**, only if the control's branches cross over: apply the fixes, then run the array.
* **P3.1 round-3 per-leaf critic (12 objections, 8 Major)**:
    * **Gate 0, both routes.**
        * ZR is the Assaraf–Caffarel zero-bias correction with a learned $\psi'$ (physics/0310035; Toulouse, Assaraf & Umrigar 0705.0721), and also a dual-weighted-residual estimate (Cancès, Dusson, Kemlin & Levitt 2111.01470). It needs one extra NQS per observable, which contradicts the leaf's "cheap information" premise. The round-2 kill "zero-variance, zero-bias estimators reduce variance, not bias" was factually wrong; that log entry is corrected here.
        * ECC's regressor $c=\langle(H-E)(A-a)\rangle$ is the first Lanczos moment, so it inherits the killed route's blindness ($\propto\Delta$) to low-energy competitors.
    * **The tangent-space lemma** is correct but standard (the Brillouin/stationarity condition).
    * **No real target exists.**
        * Projected converged states have near-zero pair error, so the scored population would be step-30/100 snapshots (high-$\omega$ error), where one Lanczos step wins trivially.
        * The score filter selects exactly the states where first-order expansions fail.
        * At scale, $\mathrm{Var}(A)\sim1/V$ while the infidelity tends to 1, so the expansion has no small parameter.
        * The test covers neither half of the Hard Core.
    * **ZR's sampling is unrealistic.** Its scale factor and training use exact sums. Sampling with $N_s/D\approx0.2$–$0.4$ i.i.d. nearly enumerates the support.
    * **Action.**
        * The 36-task job is held. A 6-task exact-quantities target pilot runs as job 410398: converged projected NNBF of widths 1, 2 and 4 plus the projected foil, 1500 steps, one seed per instance.
        * Go criterion: at least 2 instances with at least 3 converged finals at $\lvert\delta\rvert\ge0.02\lvert a_0\rvert$, on which ZRO tracks at least 70% and L1 tracks fewer than 90%.
        * Otherwise NO-GO, and P3.1 is re-scoped to larger sampled systems or a same-irrep near-degenerate instance.
        * Round-4 mutation of both routes waits on this, because it decides whether any target exists at ED size.
* **P1.2 round-3 per-leaf critic (12 objections, 6 Major)**:
    * **Premise gate likely fails.** What sets how close a projected NNBF gets to exact is the ratio $\dim\mathrm{range}(P_SP_\chi)/(\text{parameters})$, not $\lvert G\rvert$. The exact sector dimensions are:

| Cluster | Sector dimension | Ratio to control parameters |
| :--- | :--- | :--- |
| P1.1 torus $4+4$ | 1618 | 0.4 |
| P1.1 torus $5+5$ | 3752 | 2.9 |
| $4\times3$ OBC | 17905 | 4.4 |
| $4\times3$ cylinder | 4663 | 1.15 |
| $6\times2$ ladder | 11920 | 2.4 |

* At ratio 2.9 the projected control already sat about $1000\times$ below $0.01t$, so at most one instance is likely to be informative.
    * **FRD vs control is confounded.** FRD sits on a backflow-free Slater–Jastrow base while the control is NNBF. The decisive contrast should be NNBF×FRD vs NNBF.
    * **The adoption barrier is never tested.** That barrier is SR instability through zero crossings under *sampled* force; every arm trains with the exact force.
    * **The heavy-tail gate cannot fire.** The FRD $E_{\mathrm{loc}}$ kurtosis is $143\times$ the control's, yet the arm passes.
    * **Novelty is only partial.** In the zero-crossing regime the factor is locally a configuration-dependent Lanczos/Baeriswyl step. The real adoption barrier is the $\mathcal{O}((zV)^2N_e)$ per-sample local-energy cost, hidden by full-basis amplitudes in the test.
    * **Action.** The 108-task job is held; the critic's 3-task premise pilot (projected control, seed 1234, shift $10^{-2}$, one task per instance) runs as job 410216. Pre-committed rule:
        * if the projected control's $\Delta E_{\mathrm{sign}}<0.003t$ on at least 2 of 3 instances, the verdict is NO-TARGET, the array is cancelled, and the instances are redesigned for a larger sector-dimension/parameter ratio;
        * if $\Delta E_{\mathrm{sign}}\ge0.01t$ on at least 2 instances, the test is fixed (NNBF×FRD contrast, sampled-force arm, kurtosis gate) and then run;
        * otherwise 2 more seeds of the projected control run first.
    * **Broadcast.** The sector-dimension/parameter insight was sent to the P2.1, P3.2, P4.1 and P4.2 workers.
* **P1.3 round-3 per-leaf critic (12 objections, 9 Major)**:
    * **Premise.** It is close to certainly false at ED size. The projected trivial-irrep sector of $4+4$ on 12 sites has about $1.5\times10^3$ dimensions, against about $4\times10^3$ NNBF parameters.
    * **OCP fails Gate 0.** It is SCF metadynamics / deflation, the family already killed as FCD (Thom & Head-Gordon 2008; variational quantum deflation; arXiv:2605.30944 combines projection with an orthogonality penalty). It also cannot escape a diabatic (mixed) basin.
    * **The PW1 trap test is uninformative** (an exact competitor eigenstate is a saddle, not a minimum), and it wrongly vetoes cold-run evidence.
    * **The sampled-fidelity gate is cosmetic**: training uses the exact fidelity gradient.
    * **The coarse gap scan missed real crossings.** Same-symmetry degeneracies have codimension 2, so exact in-sector crossings are generic in the $(t',U)$ plane. A refined 2D search could reach the Hard-Core scale; this was forwarded to the running P3.2 worker.
    * **Action.** The full 120-task job is held. Only the critic's 9-task premise pilot runs (job 410199), with a pre-committed rule:
        * if all 9 runs end with $e_{\mathrm{var}}\le0.05\Delta$ and overlap with $\psi_0\ge0.9$, the premise is falsified at ED size and the P1.3 route search closes at these sizes;
        * if any run ends trapped at $e_{\mathrm{var}}\ge0.25\Delta$ with a flat energy, the full test is first fixed (items 2, 3, 4, 6, 7) and then run on a refined smaller-gap instance;
        * otherwise the pilot extends to 6 seeds.
### Pilot read-out (2026-11-12): pre-committed rules applied
Every number below was read from the result files by an agent; the orchestrator re-checked the P1.1-P aggregate and the P2.1 instance-0 numbers.

| Leaf | Key numbers | Rule outcome | Next step |
| :--- | :--- | :--- | :--- |
| **P1.1-P** (407525) | Width 128, medians over 3 seeds. $f_E$: U 0.61/0.77/0.66, P 0.008/0.006/0.005. $\Delta E_{\mathrm{sign}}(P)/\Delta E_{\mathrm{sign}}(U)\le3.5\times10^{-6}$ | **SYMMETRY-DOMINATED** on all three tori | Final. P1.1's "sign is binding" result is symmetry contamination on these clusters |
| **P1.2** (410216) | Projected-control $\Delta E_{\mathrm{sign}}$: $6.8\times10^{-6}$, $1.1\times10^{-6}$, $1.0\times10^{-5}$ | **NO-TARGET** (3 of 3 below $0.003t$) | Redesign onto the 4×4 OBC $t$–$t'$–$J$ instance below |
| **P1.3** (410199) | All 9 runs: $e_{\mathrm{var}}\le2.7\times10^{-5}$ ($\le1.2\times10^{-4}\Delta$), overlap with $\psi_0\ge0.999998$, none trapped | **Premise falsified at ED size** on the 4×3 crossing instance | Closed on 4×3; reopened only as a basin check on the 4×4 OBC instance |
| **P2.1** (410988) | Projected $\Delta E_{\mathrm{sign}}$: inst0 0.0125, inst1 0.0038, inst2 0.0028 | **Otherwise → 2 more seeds** | Job 451706 (inst0 and inst1, seeds 22 and 33) |
| **P2.2** (411919) | H4, $K=20$: median $\delta_{\mathrm{CGK}}=0.691$ mHa, above $\tau=0.2$ | **Kill R4.1 (CGK)** | Round-5 mutation |
| **P3.1** (410398) | Instances meeting the go criterion: 0 of 6 (2 needed). L1 never tracks; ZRO tracks 0–2 per instance | **NO-GO** | Re-scope onto the 4×4 OBC instance |
| **P3.2** (410917) | 2 hard targets; R0P wrong at $d=+0.0002$; no branch crossover; both snapshots `certified` with collapsed branches. Final-window energy swing about $5\times10^{-3}$, above the gap | **INTRINSIC-BIAS**, by elimination; the wrong call may be optimizer noise | Fix: R0P-warm, certification gate, multi-seed |
| **P4.1** (411494, 411996) | Exactly projected: $m(\epsilon_{\mathrm{che}})=8.4\times10^{-5}$ vs $m(\epsilon_{\mathrm{nnbf\_SG}})=4.0\times10^{-5}$, ratio 2.10, seed ranges disjoint. Tabular AGP job **timed out** with no fit | `sfb` has headroom (charge-only is $2.1\times$ worse); the KAGP premise is **unavailable** | Fix list plus a faster completeness test |
| **P4.2** (410960) | $\epsilon_{\mathrm{tr0}}$ = 0.66/0.76/0.62 (spread 0.143, below ½·median); correct irrep picked 2 of 3 times | **Route 3.1 not killed** | Correction (P4.2 round 5): the pilot ran with `--budget 0`, so the "scratch" comparator was the untrained mean-field start. Zero-shot transfer (median 0.66) is worse than that untrained start (0.29); Route 3.1 is expected CLOSED-FAIL (closing job 451914) |

* **P3.1 saturation result (round 6, verified in code).** Two no-go statements explain every P3.1 kill through one mechanism.
    * **S1:** every Hamiltonian moment $\langle H^k\rangle$ of $\psi=\sum_nc_n\phi_n$ depends only on $\lvert c_n\rvert^2$, while the first-order observable error depends on the signed cross terms $c_0c_n\langle\phi_0\rvert A\lvert\phi_n\rangle$. So no function of $H$-moments fixes the observable error, or even its sign; the code shows the moments agreeing to $7\times10^{-16}$ while $\delta$ changes sign.
    * **S2 (Markov inequality):** a degree-$K$ polynomial filter in $H$ changes a component at excitation energy $\omega$ by at most $2K^2M\omega/W$, where $W$ is the spectral width and $M$ bounds the polynomial on the spectrum.
    * **Consequences.** Lanczos-step extrapolation, zero-variance extrapolation, A-Ritz, ECC, ZR and OSG are all blind to low-energy error. The underlying fact is known (Eckart; Weinhold; Kaniel–Paige–Saad), so the novelty is only in the unification.
    * **The leaf is refined** to information that is not generated by $H$. Survivor CLX extrapolates along a per-seed width ladder; its injected low-energy check on the real instance is job 452272.
    * **Job 452272 result.** The low-$\omega$ dominance is confirmed on real states. For the converged width-6 projected NNBF (seed 11), the fraction of the observable error carried by in-irrep components with $\omega\le1t$ is $F_\delta(1t)$ = 1.41 (pair), 0.87 (spin), 0.83 (charge), against $F_E(1t)=0.31$ for the energy error.
    * **CLX: INJECT-FAIL.** It tracked 0 of 4 set-A items, abstaining almost everywhere because its jackknife SE is about $\lvert\delta\rvert$.
    * **The Lanczos variance-extrapolation controls give false certificates on every set-A item** ($G=1.00$), as S1/S2 predict. They track only the high-$\omega$ junk set.
    * **P3.1 goes to round-7 mutation**, aimed at estimating or bounding the low-$\omega$ in-irrep subspace directly.
    * **Round 7, ESR0 (excited-state subspace Ritz): INJECT-PASS (job 454376, grade R).**
        * Set A: 16 of 21 items tracked (0.76), per-seed median 0.83, **no false certificates**. The Lanczos control L1R tracks 1 of 21 with false-certificate rate 0.90.
        * Real items: 3 of 3 tracked, none false.
        * The observable error is concentrated in few in-irrep levels: $K_{1/2}$ = 2 (pair), 6 (spin), 11 (charge).
        * Even an oracle energy-window certificate is $9$–$81\times$ looser than $\lvert\delta\rvert$.
        * **Proposition S3:** junk at energy $\varepsilon$ in an excited state recovers only $\omega/(\omega+\varepsilon)$ of that level's error, so trained excited states must reach $\varepsilon\lesssim\omega$.
        * Step 2 (training width-6 excited states) waits on the per-leaf critic.
* **P4.1 decisive pilot (job 451886, 4×3 OBC $5+5$, $U=8$, $t'=-0.25$; about 2.6k parameters per arm; 3 seeds; medians of relative energy error $\epsilon$; all states exact $S$).**

| Arm | $m(\epsilon)$ | Seed range |
| :--- | :--- | :--- |
| tocchio (one singlet AGP per charge configuration, no projector) | $1.47\times10^{-3}$ | 1.47–1.58e-3 |
| kagp2 (two AGPs per charge configuration) | $1.30\times10^{-3}$ | 1.19–1.64e-3 |
| che (charge-only broken-symmetry network, exactly projected) | $3.4\times10^{-4}$ | 3.3–4.3e-4 |
| sfb (che plus one exact-$S$ spin feature) | $3.7\times10^{-4}$ | 3.5–4.5e-4 |
| nnbf_SG (spin-resolved network, exactly projected, exponential cost) | $2.1\times10^{-4}$ | 1.9–2.3e-4 |

  * **Verdicts:**
    * Route 4.2 (KAGP): **FAIL** on accuracy. It is only 11% better than tocchio, with overlapping seed ranges, although its completeness premise was MET.
    * Route 4.1 (sfb): **FAIL**. It recovers $-22\%$ of the che→nnbf_SG gap.
  * **Findings:**
    * Exact-$S$-by-construction ansätze (tocchio, kagp2) are $4$–$7\times$ less accurate than an exactly projected broken-symmetry network at matched parameters.
    * The tabular completeness gain of a second AGP does not survive training at this capacity.
    * Spin-resolved conditioning is worth about $1.7\times$ over charge-only, and the one-feature polynomial subclass recovers none of it.
  * P4.1 goes to round-5 mutation on the narrowed open core.
* **P4.1 completeness result (job 451839, 4×3 OBC, $U=8$, $t'=-0.25$).** This is the best tabular fit of one or two charge-conditioned singlet AGPs per charge configuration to $\psi_0$. All fits are exact singlets ($\lvert\langle S^2\rangle\rvert\le10^{-16}$).

| $N_e$ | Weight on configurations with $N_s\ge10$ | Infidelity, $K=1$ | Infidelity, $K=2$ |
| :--- | :--- | :--- | :--- |
| 8 | 0 | $5.8\times10^{-4}$ | $\sim10^{-15}$ |
| 10 | 0.72 | $6.8\times10^{-3}$ | $2.0\times10^{-8}$ |
| 12 | 0.92 | $2.4\times10^{-2}$ | $6.6\times10^{-4}$ |

  * The Tocchio class (one AGP per charge configuration) becomes less complete as the number of singly occupied sites grows, and two AGPs recover almost all of the gap. This is KAGP's structural premise.
  * Whether it matters against a trained network's capacity is the decisive question for job 451886, whose `report()` enforces $\epsilon_{\mathrm{tab}}(K{=}1)\ge0.5\,m(\epsilon_{\mathrm{tocchio}})$.
* **New key instance (from P2.1's pilot).** On the 4×4 OBC $t$–$t'$–$J$ cluster, $6+6$ electrons, $t'=-0.25$, $J=0.4$ (projected sector dimension 30220; NNBF width 6 with about 1.7k parameters, ratio about 17), the symmetry-projected NNBF keeps a measurable sign error:
    * $\Delta E_{\mathrm{sign}}=0.0125t$, $f_E=0.24$, overlap with $\psi_0$ 0.936;
    * with exact $P_S$ added, $\Delta E_{\mathrm{sign}}=0.0063t$.

  * **Observable consequence (P3.1 round 5, seed 11).** The converged projected NNBF has relative errors of **31% in the long-range pair correlation, 15% in spin and 22% in charge**, with overlap 0.936. One and two Lanczos steps cut $\Delta E$ from 0.052 to 0.023 and 0.015 but barely move these observables, so the observable error lies in low-energy directions. This is the frozen Unacceptable Outcome happening in practice: a much better energy with the order parameter still wrong. Three-seed confirmation is job 451837.

  This is the first instance in this study where a nodal error survives projection, consistent with the sector-dimension/parameter criterion.
  * **Three-seed observable errors (job 451837, P3.1 target probe; converged projected NNBF, width 6).** Relative errors of the long-range correlations against ED:

| Seed | $\Delta E$ | Overlap with $\psi_0$ | Pair | Spin | Charge |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 11 | 0.052 | 0.936 | 0.31 | 0.15 | 0.22 |
| 22 | 0.057 | 0.916 | 0.62 | 0.14 | 0.31 |
| 33 | 0.058 | 0.914 | 0.89 | 0.08 | 0.30 |

  * **The three seeds have nearly the same energy but disagree by $3\times$ on the pair correlation.** Seed 33 shows almost no long-range pairing ($a=1.0\times10^{-5}$ against the exact $9.2\times10^{-5}$). This is the "variational bias decides the physics" problem of the doped-Hubbard NQS literature, reproduced on an exactly solvable cluster.
  * **Exact spin projection** lowers $\Delta E$ by about 40% (to 0.030–0.035) **but leaves every observable error unchanged**.
  * **Two Lanczos steps** cut $\Delta E$ by $3$–$4\times$ but move the pair error by less than half.
  * Step-1 verdict: TARGET for pair, spin and charge.
  * **Seed confirmation (job 451706).** On the 4×4 OBC $6+6$ instance, the projected NNBF (width 6) has median $\Delta E_{\mathrm{sign}}=0.0125t$ over seeds 11/22/33 (range 0.0125–0.0136), $f_E=0.22$–$0.24$. With exact $P_S$ added the median is $0.0071t$. The target is confirmed for P1.2, P1.3 and P3.1.
  * **Comparison instance.** The 4×4 cylinder $6+6$ (sector 15268) gives median $0.0038t$, which is below the $0.01t$ bar. It becomes the shared target for P1.2, P1.3 (basin check) and P3.1 (error bars). Seed confirmation is running as job 451706.
### Round 2 — Mutation & Replenishment Synthesis (2026-11-11)
Under the Persistence Mandate, every leaf except P1.1 was re-run through the Stage 3.3 mutation and replenishment loop, with the round-1 critic objections as binding constraints. Every leaf now has at least two routes that pass all four gates in the worker's own audit; P2.2 has three. None has yet been through the round-2 critic, and no round-2 decisive test has been submitted.

| Node | Status | Surviving routes | Sharpest gap | Decisive test (instance; control arm) |
| :--- | :--- | :--- | :--- | :--- |
| **P1.2** | SURVIVED (via Mutation) | FRD-K: Jastrow–Slater times a product of local factors $1+\alpha_e g(R_e)$, driven by the reference's own hop-amplitude ratios $R_e$; FRD-X: the same on spin-exchange edges | Whether NNBF's wrong-sign weight is locally repairable or sits in extended regions; learnability through a zero crossing | Three P1.1 instances, $H=8,32$, exact-sum SR; NNBF×Jastrow at matched parameters |
| **P1.3** | SURVIVED (via Mutation) | SNO-FN: natural-orbital or frontier re-anchoring, accepted by guided fixed-node energy; FCD: fidelity-capped deflation, then release | Recovery of re-anchored candidates within 40 steps; whether the FN slack preserves the $E_{\mathrm{sign}}$ ordering | $4\times3$, $4+4$, $t'=-0.25$ (sign half) and $t'=-0.32$ (avoided crossing), 8 paired seeds; matched-restart null, energy-gated ablation, "dominated by projection" check |
| **P2.1** | SURVIVED (via Mutation) | Split-capacity HFDS, which contains Liu–Clark plus a wide Jastrow, $\mathcal{O}(N_e^2)$ per $E_{\mathrm{loc}}$; shared-reference low-rank multi-determinant | Rank 2 is 40–50% of full rank at these sizes, so a PASS does not certify a fixed rank at $8\times8$ | Three P1.1 instances, scored on $\Delta E_{\mathrm{sign}}$ and $\Delta E_{\mathrm{tot}}$; NNBF-128, 4-determinant NNBF and Slater×neural-Jastrow at matched parameters |
| **P2.2** | SURVIVED (via Mutation), leans AMBIGUOUS | R2.0: frozen-backflow control variate with measured per-term cost; R2.4: multilevel MC over surrogate fidelity; R2.5: cost-aware residual allocation | R2.0's advantage over the best control is flat at about $0.6$, not falling, from $M=24$ to $32$ | H4 ring and chain at $R=2.8$ bohr, $M=16$ to $48$; best of 10 controls (exact, screening, deterministic completions, heat-bath, semistochastic) |
| **P3.1** | SURVIVED (via Mutation) | A-Ritz with a first-order error estimate and a non-vacuous trust rule (off-diagonal pairing $A$ only); sign-defect Ritz using configurations with $\kappa=(H_{\mathrm{off}}\psi)/\psi>0$ | First-order regime on realistic states; a false certificate if the step removes amplitude error faster than sign error | 36 real SR-trained states (foil and NNBF, 3 widths, 3 snapshots) on the P1.1 instances; one- and two-step Lanczos and cross-width extrapolation |
| **P3.2** | SURVIVED (via Mutation) | 2R-SS: Néel and stripe determinants under one shared backflow, coefficients by exact $2\times2$ Ritz; 2R-SA: state-averaged version | Under the state-specific objective the second root may degrade into a corrector of the first | Same-sector avoided crossing on $4\times3$, $4+4$, $t'\in[-0.42,-0.25]$, 28 instances; R0+ at matched wall time |
| **P4.1** | SURVIVED (via Mutation) | PCG: $P_SP_K e^{J(c)}\det f_c$ with $f_c=B_1(c)f_0B_2(c)^{\mathsf T}$, exact $S$; SCF: spin–charge factorised SU(2)-MPS at $\mathcal{O}(LD^3)$ | Whether a projected charge-conditioned Gaussian replaces the spin Jastrow's correlations, and whether exact $S$ lowers $E_{\mathrm{sign}}$; PCG is not below projection cost | $4\times3$ at $t'=-0.25$ and $-0.33$; spin-resolved NNBF at matched parameters and an mVMC-like foil with doublon–holon correlator |
| **P4.2** | SURVIVED (via Mutation) | MFNG: pair kernels as size-independent momentum-space neural fields with a charge-only local backflow; active probing on the seed-disagreement set | The source cannot identify the backflow: 218 parameters fit the 560-state source exactly | $t$–$t'$–$J$ $4\times2\to4\times4$ (1.68M states); size-transferred foil and from-scratch training at the same budget |

* **Round-2 critic verdict (2026-11-11, two independent referees, 40 objections, 31 Major)**:
    * **Systemic flaw**: no round-2 test has a symmetry-projected control. On the $4+4$ instance projection removes 77% of the NNBF sign error, so the claimed gains may be cheap symmetry restoration. Several controls are also crippled:
        * the P1.3 null restarts from the same broken-symmetry UHF;
        * P3.2 trains unprojected branches, so its "same-sector" design collapses back onto the cross-sector triplet;
        * the P4.1 NNBF control lacks $P_S$.
    * **Stale Hard-Core claims**: P1.2, P2.1 and P3.1 still cite the retracted same-sector competitor, and their code keys the competitor clause to it.
    * **Gate 0 relabels**:
        * P2.1 split-capacity HFDS is HFDS times a Jastrow (Liu–Clark; HFPS; lrux).
        * P3.2 2R-SS/2R-SA is multi-determinant Slater–Jastrow–backflow (Brown, Trail, López Ríos & Needs, arXiv:1011.4343) plus state-averaged VMC.
        * P4.1 PCG is mVMC with Tocchio backflow and generalized parameters; it is also not below projection cost.
        * P1.3 FCD is SCF metadynamics with a cap.
        * P3.1 A-Ritz's $\hat\delta$ is algebraically the killed two-point Lanczos extrapolation (in $\sigma$ rather than $\sigma^2$).
    * **Mechanism failures**:
        * P3.1's $\kappa>0$ set is a set of local-energy outliers, triggered by amplitude error near nodes, so it is not a sign witness.
        * P3.2's cross-overlap terms are hidden by exact sums and carry the killed NOCI variance at the crossing. Its in-sector gap $0.273t$ is about $20\times$ the Hard-Core scale, so it is not near-degenerate.
        * P4.1 SCF stalls at $\epsilon=2.4\times10^{-2}$ on a 400-state cluster where every other arm is exact.
        * P4.2's source is fitted exactly, so the transferred state is an arbitrary point on a solution manifold, and the code lets one seed decide. P4.2's real Minimal Contrast Foil, a Jastrow–BCS optimized at the target, is absent. Prior art arXiv:2505.00233 may already do lattice pairing-NQS size transfer.
        * P2.2 has no legitimate survivor. R2.0 is a revived round-1 kill whose PASS rests on a hybrid cost model ($\kappa$ from $N=20$, variances from $N=4$); R2.4 and R2.5 meet FAIL on their own pre-checks.
    * **Routes the critics did not break**: P1.2 FRD's novelty (no prior art found), and P2.1 SK-LR (shared-reference low-rank multi-determinant). Both still need a projected control and a test on the competitor instance.
    * **Decision (orchestrator)**: every round-2 status above is downgraded to **Under Mutation (round 3 pending)**, except P1.2 FRD and P2.1 SK-LR, which are **CONDITIONAL (projected control required)**. Round 3 is gated on the critical-path test **P1.1-P**, a re-run of P1.1 with symmetry-projected NNBF. If projection removes most of the sign error on all three instances, the sign branch and most claimed gains are symmetry problems, and round 3 must start from projected baselines. If a large sign fraction survives projection, P1.1's conclusion stands on firmer ground.
    * **P1.1-P submitted (2026-11-11)** as Slurm array job 407525 (18 tasks, partition `kim`); leaf note `.subproblems/P1.1P_projected.md`. The orchestrator re-ran the correctness checks before submission, and all passed:
        * the projector is idempotent to $5\times10^{-16}$, commutes with $H$ to $7\times10^{-17}$, and gives overlap $1$ with $\psi_0$;
        * the projected gradient matches finite differences.

      Read the verdict with `python3 micro_test.py --aggregate results` in `tests/P1.1P_projected/`.
    * **New global anti-patterns**:
        * Any NQS comparison on a cluster whose mean-field start breaks symmetry, made without a projected control arm.
        * Any variational decision or witness validated only by exact sums, when it needs cross-terms or ratios that are heavy-tailed under sampling.
* **New finding (P1.3)**: on the $4+4$ instance, most of the NNBF sign error is symmetry contamination; see the qualification under the P1.1 result.
* **Correction (P3.2)**: P1.1's competitor is not same-sector; see the correction under the P1.1 result.
* **New anti-patterns (round 2)**:
    * A sign layer that only sees local occupations cannot locate the nodes (P1.2).
    * A linear dressing $(1+\sum_b\alpha_bK_b)\Phi$ is not size-consistent (P1.2, P4.1).
    * Sign-stratified or positivised local energies leave the magnitude dispersion $\mathrm{Var}(x/\lvert y\rvert)$, so they gain nothing (P2.2′ measured within 3% of single-proposal sampling).
    * Cross-width variance extrapolation under-states the error when the error is sign-borne (P3.1).
    * Real-space rescaling of NQS parameters shifts $k_F$ and the density (P4.2).
### P1.1 decisive-test result (Slurm job 329057, 2026-11-11)
Global verdict **PASS** on all three pre-committed $4\times3$ PBC clusters ($t'=-0.25$). The calibration error is at most $6\times10^{-13}\,\Delta E_{\mathrm{tot}}$ and the multi-start spread is at most $5\times10^{-13}$. Here $f_E^{\mathrm{up}}=\Delta E_{\mathrm{sign}}/\Delta E_{\mathrm{tot}}$ is the sign fraction of the energy error, $f_C$ the sign fraction of the $d$-wave pair-correlation error, and $r=\Delta E(H{=}8)/\Delta E(H{=}128)$ the reduction factor from the narrowest to the widest network.

| Instance | $\Delta E_{\mathrm{tot}}$ (NNBF $H{=}128$) | $f_E^{\mathrm{up}}$ | $f_C$ | $r_{\mathrm{sign}}$ | $r_{\mathrm{amp}}$ | $\Delta_1/V$ | Foil (SJ) $f_E^{\mathrm{up}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $N_\uparrow=N_\downarrow=5$, $U=8$ | 0.166 | 0.607 | 0.71 | 1.64 | 3.40 | $0.032t$ | 0.33 |
| $N_\uparrow=N_\downarrow=4$, $U=8$ | 0.099 | 0.772 | 0.93 | 1.18 | 4.92 | $0.007t$ (but $E_1$ is a triplet at another momentum; see correction) | 0.31 |
| $N_\uparrow=N_\downarrow=5$, $U=4$ | 0.048 | 0.652 | 0.76 | 1.22 | 3.78 | $0.033t$ | 0.24 |

* **Reading**: making the backflow network wider cuts the amplitude error by a factor of 3.4 to 4.9, but the sign error only by 1.2 to 1.6. As a result the sign fraction rises with width, from 0.38–0.45 at $H=8$ to 0.61–0.77 at $H=128$. The sign also carries 71–93% of the pairing-correlation error. The foil is the opposite: 67–76% of its error is amplitude.
* **Caveats**:
    * The rigorous lower bound $f_E^{\mathrm{lo}}$ is 0 on every instance, so the PASS rests on the multi-start upper estimate. All four starts, including the fixed-node one, agree to $10^{-13}$.
    * The first instance passes only narrowly (0.607).
    * **P1.1-P interim result (2026-11-11, job 407525; instance 1 complete, instances 0 and 2 running)**: on $N_\uparrow=N_\downarrow=4$, width 32, medians over 3 seeds:

| Arm | $\Delta E_{\mathrm{tot}}$ | $\Delta E_{\mathrm{sign}}$ | $f_E$ | $f_C$ | overlap with $\psi_0$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| U (unprojected, P1.1 protocol) | 0.119 | 0.082 | 0.69 | 0.91 | 0.28 |
| U→P (post-hoc space-group projection) | 0.072 | 0.016 | 0.23 | 0.30 | 0.92 |
| P (trained with projection) | $3\times10^{-5}$ | $<10^{-5}$ | 0.002 | 0.07 | 1.000 |
| P with exact $P_S$ | $1\times10^{-5}$ | $<10^{-5}$ | 0.001 | 0.04 | 1.000 |

  On this instance the pre-committed rule returns **SYMMETRY-DOMINATED**, and the width-32 U arm reproduces P1.1 to within 1%. Interim logs on the two $5+5$ instances show projected width-8 training ending within $3\times10^{-3}$ ($U=8$) and $3\times10^{-4}$ ($U=4$) of $E_0$; the final verdict awaits their budgets.
  * **Consequence (domino)**: on 4×3 PBC clusters the "sign is the binding error" conclusion is a symmetry-contamination effect. A symmetry-projected NNBF leaves essentially no sign error. Any route tested there against a projected control has nothing to improve.
  * **Broadcast**: sent to the running round-3 workers (P2.1, P3.1, P3.2, P4.1, P4.2). They are to add premise gates and move to instances where the projected control keeps measurable error, such as lower-symmetry clusters or larger sampled systems.
  * **New anti-pattern**: testing a sign layer, basin escape or error estimator on clusters where the symmetry-projected control is already near-exact.
  * **Remaining live question**: does any nodal error survive projection where projection is weaker (open or cylinder boundaries, ladders) or the system is larger (P1.1's size-transfer gap)?
    * **Qualification (2026-11-11, from the P1.3 round-2 pre-diagnostic)**: on the $N_\uparrow=N_\downarrow=4$ instance most of the NNBF sign error is symmetry contamination inherited from the broken-symmetry UHF start, not an intrinsic nodal error. P1.1's NNBF is unprojected.
        * Evidence: a width-32 NNBF after 400 plain-SR steps has $E_{\mathrm{sign}}-E_0=0.084$ and overlap $0.27$ with $\psi_0$. Projecting it onto $\psi_0$'s irrep (translations and mirrors) lowers $E_{\mathrm{sign}}-E_0$ to $0.019$, a 77% reduction (`tests/P1.3_wrong_sign_basins_r2/results_prediag_S400.log`).
        * Error character differs by instance. On this instance the wrong-sign weight is high ($w\approx0.20$ at every width, inherited from UHF's 0.32). The two $5+5$ instances have low-weight, near-node errors ($w=0.006$–$0.03$).
        * Consequence: the largest P1.1 sign fraction (0.772) is inflated by missing symmetry projection. The PASS should be re-read as resting mainly on the two $5+5$ instances (0.607 and 0.652), whose sign errors are near-node. Re-running P1.1 with irrep-projected NNBF is the clean check.
    * **Correction (2026-11-11, from the P3.2 round-2 ED sector analysis)**: none of the three instances has a same-sector competitor. P1.1's flag used the lowest excitation within the fixed-$(N_\uparrow,N_\downarrow)$ space. On the $N_\uparrow=N_\downarrow=4$ instance that excitation is a spin triplet ($\langle\hat{S}^2\rangle=2$) at momentum $(\pi,\pm2\pi/3)$, $0.086t$ above $E_0$. Inside the ground-state sector ($K=0$, both mirrors even, $S=0$) the next level is $0.515t$ up. P1.1 therefore tests the sign half of the Hard Core only. A genuine same-sector avoided crossing exists on the same cluster when $t'$ is swept: the in-sector gap reaches a minimum of $0.273t$ between $t'=-0.33$ and $-0.31$, where a $d$-wave/short-range-AF state and a period-4 stripe exchange character. P3.2 round 2 uses that crossing.
    * Transfer to $8\times8$ is untested; this remains P1.1's sharpest gap.
* **Consequence**: the sign-structure branch (P1.2, P1.3) targets the binding error of NQS on these clusters. Network width is not the lever for that error.
### Node-level structural results (independent of the routes)
* **P2.1 lower bound**: for unrestricted backflow, where every orbital row depends on every occupation, one hop rewrites $\Omega(N_e^2)$ matrix entries, and there are $\mathcal{O}(zN_e)$ hops, so $E_{\mathrm{loc}}$ costs $\Omega(N_e^3)$. Here $z$ is the coordination number. Any exact speedup must restrict the family to low-rank hop response.
* **P2.2 variance floor (corrected after critic)**: for *single-proposal* importance sampling of a local-energy row $\sum_ix_i$, the per-draw variance is at least $(\sum_i\lvert x_i\rvert)^2-(\sum_ix_i)^2$, which is nonzero whenever the row is non-stoquastic. This is a textbook importance-sampling result, not a new one. It is **not** a floor for all stochastic estimators: positivisation (Owen & Zhou, JASA 2000), with separate samplers for the positive and negative parts, removes it. Sign-stratification with a cheap surrogate sign reduces it to the sign-mismatch weight. Measured on H6 at $R=3.0$ near convergence, single-proposal oracle sampling costs $146\times$ and heat-bath $657\times$ exact enumeration.
* **P4.1 charge-only lemma (standard, not new; it is why Tahara & Imada's projection uses spin-commuting correlators)**: a diagonal correlator $f(\mathbf{n})$ commutes with $\hat{S}^+=\sum_i\hat{S}_i^+$ if and only if $f$ is invariant under every single-spin flip, i.e. it depends on the charge configuration only. This is why mVMC's projection is cheap and why spin-resolved backflow cannot be projected cheaply.
* **P1.2 preview, input to P1.1**: on $3\times4$ open $t$–$t'$–$J$ ($t'=-0.3$, 2 holes), the foil's Slater-AF sign costs $\Delta E_{\mathrm{FN}}=0.104t$ even with optimal amplitudes, against $4.5\times10^{-4}t$ for the exact sign. The foil's sign is wrong on 7.3% of the weight. For the foil, the sign is the binding error on this instance.

---

## Cross-Pollination Analysis
### Mechanism transfer
* **P1.1's fixed-sign energy is the common currency for P1.2 and P1.3.** P1.2 uses the ten Haaf fixed-node energy and P1.3 uses $\langle H\rangle$ with exact amplitudes and trial signs. Both are sign-only metrics, but only $E_{\mathrm{sign}}$ (best non-negative amplitudes under fixed signs, an exact variational quantity) cleanly separates sign from amplitude. Recommendation: report all three, and adopt $E_{\mathrm{sign}}$ as the primary metric in the P1.2 and P1.3 tests.
* **P4.1's charge-only lemma constrains P4.2 and P1.2.** P4.2's residual correlator uses $S^z$ features, so it breaks SU(2). Rebuilding it on charge-only features plus bond exchange would make the transferred state exactly spin-adapted, and translation-equivariant charge networks transfer across sizes without change. P1.2's Walsh sign layer acts on spin-resolved occupations and would break exact $S$ as well.
* **P2.1's LGB-BF structure fits P4.1's R1.** Charge-only backflow with finite range plus $K$ global channels keeps both the $\mathcal{O}(N_e^2)$ local energy and the exact spin. The two cost results compose for hopping moves, and spin-exchange moves are already $\mathcal{O}(N_e^2)$ in R1.
* **P3.1 and P3.2 are one procedure.** P3.2 decides which branch is the ground state; P3.1's A-Ritz gives the error bar on the order parameter within a branch, and its measured order character is the label P3.2 needs.
### Domino invalidation
* **If P1.1 FAILS (amplitudes binding for NNBF at width 128)**, P1.2 and P1.3 aim at the wrong target for NQS, though not for the foil (the P1.2 preview shows the sign binds for the foil). The P1 branch would then reduce to "NQS backflow already fixes the sign on ED sizes; the remaining issue is size transfer", which hands the problem to P4.2 and to P1.1's sharpest gap.
* **The variance floor generalizes, but only to single-proposal schemes.** P2.1-R4, P2.2-R1/R2/R3 and P4.1-R2 were killed partly on this basis. On re-audit, each still dies on other grounds: P2.2-R1 at Gate 0 (prior art), P2.2-R2 on noise amplification by $S^{-1}$, P2.1-R4 on the $\Omega(N_e^3)$ matrix-writing bound, and P4.1-R2 on the biased energy from noisy amplitude ratios. No kill rests on the floor alone. The floor does open one untested candidate, recorded as **P2.2′** below.
* **P4.2 and P1.2 both lean FAIL for the same reason**: they both try to carry a learned sign or pair-function correction that is local, but the correction they need appears to be non-local (power-law $d$-wave tails; residual sign error not low-degree). If both FAIL, that is cumulative evidence that the residual nodal correction to mean-field fermionic states is not local at these sizes. That statement is falsifiable and worth recording in its own right.
### Critical Path Recommendation (post-critic)
**Run P1.1 first, and only P1.1 for now.** Its test is the only one that survived the critic with a fixable defect; the optimizer-bias and calibration fixes are now applied. It still decides whether the sign-structure branch (P1.3, and any successor to the killed P1.2) targets the binding error. It also now reports whether each test cluster contains a near-degenerate competitor (critic item 23).
The two conditional leaves wait for redesign before they spend compute:
* **P1.3**:
    * Add a null arm: plain SR restarted with the same total step budget and the same number of trials, so that "basin escape" is distinguished from "extra restarts".
    * Replace the medians over three seeds with paired comparisons over at least 8 seeds.
    * Treat ranking basins by energy at $10^{-3}t$ per site as a primary risk, not a secondary one.
* **P3.1**:
    * Restrict the claim to off-diagonal order operators (pairing $\hat\Delta_r^\dagger\hat\Delta_0$). For diagonal $A$, the direction $A\psi$ is a Jastrow derivative that the linear method already spans.
    * Replace the contaminant family with states produced by an actual optimized ansatz, instead of $\psi_0$ contaminated along directions chosen for $\langle k\rvert A\lvert0\rangle\ne0$.
    * Lower the trust threshold so that the foil is not trivially abstained on.
### New candidate from the critic pass
* **P2.2′ — Sign-stratified local energy.** Split the connected configurations of $E_{\mathrm{loc}}$ by a cheap surrogate sign, e.g. the frozen-backflow sign $\mathrm{sgn}\,\tilde r$ that P2.2 already computes. Enumerate the large terms exactly and sample each sign stratum with its own importance distribution (positivisation). The variance then scales with the sign-mismatch weight between surrogate and true terms rather than with the full non-stoquastic floor.
    * Status: not yet gated.
    * Gate 0 check: has positivisation already been applied to VMC local energies?
    * Hard-Core risk: the mismatch weight may be large exactly in the multireference regime (P2.2 measured a residual fraction of 0.56 to 0.82).

---

## Problem Anti-Patterns & Killed Routes
### Global anti-patterns (killed in more than one leaf)
* **Single-proposal stochastic replacement of an exact non-stoquastic sum** (P2.1-R4, P2.2-R1/R2/R3, P4.1-R2): with one importance distribution, the variance floor $(\sum\lvert x\rvert)^2-(\sum x)^2$ does not vanish at convergence, and nonlinear use of the estimate (ratios, $S^{-1}$ in SR) biases the result. Multi-proposal (sign-stratified) schemes are not covered by this anti-pattern; see P2.2′.
* **Continuation from the sign-free or mean-field limit** (P1.3-R3, P4.2 momentum-orbital generator): a mean-field start already is the $U=0$ continuation, and the path follows the metastable branch through the near-degenerate level crossings that make up the Hard Core.
* **Re-labelled sign networks** (P1.2-R1, P1.2-R5, P1.3-R1): Jordan–Wigner sign times a learned sign network, neural Pfaffians, and power-method sign refitting already exist (arXiv:2108.08631; PRB 96, 205152; arXiv:2507.10705; arXiv:2405.14762; SWO / self-learning power method) and do not fix their generalization barrier.
* **Bounds that need the gap** (P3.1-R2, P3.2-R3): Temple, Eckart and susceptibility bounds all need a gap $\Delta$ that closes exactly where competing orders sit.
### Per-leaf kills
* **P1.1**: hybrid swap / Shapley split (Gate 0; order-dependent, the amplitude part can be negative); sign-mismatch weight (Gate 1; saturates exponentially in $V$); lattice fixed-node GFMC with the NQS as guide (Gate 0; Loehr & Clark 2510.26906, and its energy depends on the guide amplitudes).
* **P1.2**: pairwise antisymmetric product (fixed pair function gives a quadratic GF(2) sign that already fails free-fermion nodes on $3\times4$); holon determinant times Marshall sign (wrong on 34% of the weight at $J/t=0.4$).
* **P1.3**: sign-flip-aware Metropolis (Gate 1; SR fixed points do not depend on the sampler).
* **P2.1**: finite-range convolutional backflow plus Woodbury (Gate 0; SCALE arXiv:2604.25775 plus Lrux); readout-rank ratios for transformer backflow (Gate 1; attention costs $\Theta(N_e^3d)$, where $d$ is the embedding dimension).
* **P3.1**: Lanczos-step variance extrapolation of $C(r)$ (Gate 0; Sorella–Becca scheme, and $H$-Krylov reaches competing states only with weight $\propto\Delta$); pinning-field extrapolation (Gate 2; an ordered ansatz answers with its own order).
* **P3.2**: calibrated independent-branch extrapolation (Gates 0 and 2; reports the seeding reference); sector-resolved branches (Gate 3; stripe competitors share a sector); non-orthogonal CI of separately optimized branches (Gate 1; exponential sample cost).
* **P4.1**: analytic projection from polynomial roots (Gate 2; same family as the foil); spin-adapted CSF/GUGA sampling (Gates 0 and 3).
* **Post-critic kills of the leaf survivors** (critic item numbers in parentheses):
    * **P1.2, residual Walsh sign layer** (items 5 and 6): Gate 0.
        * A local, sign-capable diagonal correlator times a Slater/BCS reference is the correlator-product-state × Slater ansatz (Changlani et al., PRB 80, 245116; Neuscamman et al., PRB 84, 205132, tested on the 2D Hubbard model).
        * The test fits the layer to oracle labels from $\psi_0$, which is P1.3's anti-solution.
        * A single global threshold cannot express a product of independent local sign corrections, so the test was biased toward FAIL anyway.
    * **P2.1, LGB backflow** (items 10 and 11): Gates 0 and 2.
        * Finite-range convolutional backflow with low-rank updates is SCALE.
        * The $K$-channel global term is a strict subfamily of Liu–Clark's low-rank correction, which already underperforms.
        * The decisive test checks only the Woodbury cost that the foil also has, so it cannot fail.
    * **P2.2, frozen-backflow control variate** (item 12): killed on measurement.
        * The measured residual fraction is 0.56 to 0.82, and the pre-check gave $M=1.03$ to 1.07.
        * The construction is the standard semistochastic control variate.
        * The H8/STO-3G instance is too small to show $\mathcal{O}(M^4)$ scaling.
    * **P3.2, joint two-state rule** (items 16 to 18): Gates 0 and 3.
        * Gate 0: it is the NES determinant of states (Pfau et al. 2308.16848; Ma et al. 2506.08594), and relabelling by measured character is already R0+.
        * Gate 3: the test's competitors are mirror-even and mirror-odd, i.e. in different sectors, which is the killed sector-resolved route.
        * The test also replaces the route with exact block-Krylov.
    * **P4.1, charge-only singlet backflow** (items 19 and 20): Gate 0.
        * It is Tocchio, Becca, Parola & Sorella's doublon–holon backflow on a projected singlet BCS state (PRB 78, 041101, 2008), with an MLP in place of the hand-crafted form.
        * SA-NNBF (2604.06841) also uses spatial-only backflow input.
        * The test also lacked a spin-resolved NNBF arm at matched parameter count.
    * **P4.2, residual-geminal transfer** (items 21 and 22): Gate 3.
        * The test geometry is a spin-gapped ladder with a single dominant order.
        * With $R_{\mathrm{corr}}=1$, the transferred residual is a few Jastrow-like scalars, i.e. foil transfer.
        * The smoke pre-check gave $M=1.30$.
* **P4.2**: unit-cell gluing (Gate 1; zero pair correlation across the cut); twist-folded transfer (Gate 0; Gerard et al. arXiv:2405.07599).

---

## Revision & Critic Log
* 2026-11-10: Step 0 and Stage 1 written; Mode B dispatched over leaves P1.1–P4.2.
* 2026-11-10: Fan-in.
    * Nine leaf artifacts compiled into the Master Leaf Synthesis Matrix; cross-pollination and anti-patterns written.
    * Login-node pre-checks verified against the raw outputs: P1.2 $M_{\mathrm{local}}=1.0286$, P4.1 $\langle S^2\rangle$ exact to $10^{-16}$, and the P2.2 smoke JSON.
* 2026-11-10: Adversarial critic pass returned 25 objections (17 Major). Actions:
    * Accepted, routes killed: P1.2 (items 5 and 6), P2.1 (10 and 11), P2.2 (12), P3.2 (16 to 18), P4.1 (19 and 20), P4.2 (21 and 22). Details are under Post-critic kills.
    * Accepted, routes made conditional with a mandatory test redesign: P1.3 (items 7 to 9), P3.1 (items 13 to 15).
    * Accepted, P1.1 test patched (items 2 and 3):
        * multi-start including $\lvert\psi_\theta\rvert$ and the ten Haaf fixed-node state;
        * upper and lower bounds on $f_E$, with a spread guard;
        * the calibration→AMBIGUOUS and width-trend rules, now implemented in code.
    * Accepted, P1.1 is a diagnostic rather than a route (item 4). Its value is routing, which is why it is the critical path.
    * Accepted, master corrected (items 1 and 19):
        * the variance floor holds only for single-proposal sampling;
        * the kills that cited it were re-audited, and none rests on the floor alone;
        * new candidate P2.2′ recorded;
        * the charge-only lemma is relabelled as standard.
    * Accepted, not yet fixed: the competing-order half of the Hard Core is missing from most tests (item 23). P1.1 now reports the lowest gap in the fixed-$(N_\uparrow,N_\downarrow)$ space, which is not a same-sector gap (see the correction under the P1.1 result). Every future test must show a same-sector near-degenerate competitor, or say that it tests only the sign half.
    * Accepted, partly addressed: sign metrics are not commensurate across leaves (item 25). P1.1 reports both $E_{\mathrm{sign}}$ and $E_{\mathrm{FN}}$, so P1.3 can adopt either.
    * Critical-path test submitted: P1.1 as Slurm array job 329057 (partition `kim`, tasks 0–2, 24 h limit), after the orchestrator reproduced the patched smoke test ($f_E^{\mathrm{up}}=0.1407$, calibration error $4\times10^{-12}\,\Delta E_{\mathrm{tot}}$). Read the verdict with `python3 micro_test.py --aggregate results` in `tests/P1.1_sign_vs_amplitude/`.
    * 2026-11-11: P1.1 job 329057 finished. All three tasks completed with exit code 0; global verdict PASS (see P1.1 decisive-test result above).
    * 2026-11-11: The skill was updated with a Persistence Mandate (no early surrender) and a mutation and replenishment engine (Stage 3.3). The six critic-killed leaves (P1.2, P2.1, P2.2, P3.2, P4.1, P4.2) are reopened as **Under Mutation**, with their kill mechanisms as boundary conditions. P1.3 and P3.1 get their redesigned tests.
    * Saturation note: five of the nine routes died at Gate 0 after a critic search, so the field around NQS for fermions is saturated with relabelable constructions. The surviving value is in diagnostics (P1.1) and certification (P3.1), not in new architectures.
