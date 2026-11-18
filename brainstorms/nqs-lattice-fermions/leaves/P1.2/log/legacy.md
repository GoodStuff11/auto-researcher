# Log: P1.2 (legacy rounds, migrated)

### 2. Candidate Routes & Falsification Gates
* **Route 1**: Determinant-free second-quantized sign, i.e. the Jordan–Wigner sign times a learned sign or phase network over occupations, or the same network multiplying a fixed determinant.
  - Gate 0 (Novelty & Adoption): FAIL. This already exists as a determinant-free real-space exchange sign plus a feed-forward network (Inui, Kato & Motome, PRR 3, 043126 (2021), arXiv:2108.08631), as a complex RBM times a pair product (Nomura et al., PRB 96, 205152 (2017)), and in second-quantized chemistry (Choo et al., Nat. Commun. 11, 2368 (2020)). It is not standard practice because the Jordan–Wigner string in 2D is non-local and generic sign networks generalize poorly from samples. The route offers no mechanism against that specific barrier.
  - Gate 1 (Conservation & Asymptotics): PASS. It conserves $N_e$ and $S^z$ and is polynomial per amplitude.
  - Gate 2 (Baseline Gameability): not evaluated (already killed).
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED
* **Route 2**: Pairwise antisymmetric product (generalized Vandermonde) $\psi=\prod_{i<j}f(x_i,x_j)\times$ (symmetric), at cost $\mathcal{O}(N_e^2)$, below cubic.
  - Gate 0 (Novelty & Adoption): PASS as a lattice use. Pang et al. (arXiv:2205.13205) and Zweig & Bruna (arXiv:2208.03264) cover the continuum.
  - Gate 1 (Conservation & Asymptotics): FAIL. With a configuration-independent $f$, the sign in the site-ordered basis is $(-1)^{\sum_{i<j}b_{ij}n_in_j}$ with $b_{ij}\in\{0,1\}$, a quadratic polynomial over GF(2). It cannot even reproduce the *mean-field* nodes. Measured: on a $3\times4$ open lattice with $N=4$ free spinless fermions and $t'=-0.3$, the exact sign over the 396 nonzero configurations is not any quadratic GF(2) polynomial of the 79-term family (`pairwise_check` prints `sign is quadratic GF(2) polynomial: False`). With a configuration-dependent $f$ the product becomes universal, but then the route reduces to Route 1.
  - Gate 2 (Baseline Gameability): FAIL. The foil's determinant already contains signs that this family does not.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED
* **Route 3**: Slave-fermion anchor. Use $s(\mathbf{n})=\kappa(\mathbf{n})M(\mathbf{n})\,\mathrm{sgn}\det X_{[\text{holes},:]}$, where $\kappa=(-1)^{\sum_{\text{holes}}i}$ is the gauge that maps electron hops to holon hops with amplitude $+t$, $M=(-1)^{N_\downarrow(A)}$ is the Marshall sign ($N_\downarrow(A)$ counts down spins on sublattice $A$), and $X$ holds $N_h$ holon orbitals ($N_h$ is the hole number). The determinant has size $N_h\times N_h$, so it costs $\mathcal{O}(N_h^3)$, far below $\mathcal{O}(N_e^3)$. The anchor is not an electron mean field, and it is exact both in the Heisenberg limit and in the fully polarized sector (verified in smoke check (c), where $W<10^{-10}$).
  - Gate 0 (Novelty & Adoption): PASS. The slave-fermion and phase-string sign structure is known (Weng et al.; Lu et al., arXiv:2303.13498), but fermionic NQS for $t$–$J$ use electron determinants (G-HFDS, Lange et al., arXiv:2411.10430). No NQS uses a holon-determinant times Marshall anchor.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): FAIL. On a $3\times4$ open lattice with 2 holes and $t'=0$, the minimum over **all** hole-only functions $f$ of $W[M f]$ is 0.0001 at $J/t=100$, 0.0100 at $J/t=10$, 0.1954 at $J/t=2$, and 0.3415 at $J/t=0.4$. On the same cluster at $J/t=0.4$ the foil's optimized Slater sign has $W=0.073$. The bound covers every holon determinant with arbitrary hole-only backflow, so the whole family is beaten by the foil. The physical reason is that the nearest-neighbour phase string $(-1)^{N^{\downarrow}_{h}}$ entangles spin and hole positions.
  - Gate 3 (Hard-Core Shield): FAIL. The route is good only as $J/t\to\infty$, which is the strong-exchange limit where the Marshall sign cures the sign problem.
  - *Verdict*: KILLED
* **Route 4**: Residual Walsh sign layer on the foil determinant. The ansatz is $\psi(\mathbf{n})=s_{\mathrm{foil}}(\mathbf{n})\,c(\mathbf{n})\,A(\mathbf{n})$ with $c(\mathbf{n})=\mathrm{sgn}\big(b+\sum_{S}w_S\chi_S(\mathbf{n})\big)$. Here $\chi_S=\prod_{v\in S}(1-2x_v)$ are Walsh characters of the binary occupations $x_v\in\{n_{i\uparrow},n_{i\downarrow}\}$, and $S$ ranges over subsets of degree at most $k$ inside each $2\times2$ plaquette. In VMC the sign function is replaced by a smooth phase, $e^{i\pi\,\sigma(\cdot)}$, so that a sign can rotate through the complex plane without $\lvert\psi\rvert$ crossing zero, which removes barrier (b). The cost is $\mathcal{O}(N_e^3)$ for the reference plus $\mathcal{O}(V\,2^{8})$ for the layer, with no backflow and no determinant sum. The point of departure is that the foil's sign is already nearly right ($W$ of a few percent on small clusters). The hypothesis, motivated by the Boolean-Fourier structure of frustrated signs (Schurov et al., arXiv:2508.09870), is that the residual $r=\mathrm{sgn}\,\psi_0\cdot s_{\mathrm{foil}}$ is local and of low degree. Its family strictly contains the foil ($w=0$) and is not anchored. Every configuration's sign can move independently of the orbitals.
  - Gate 0 (Novelty & Adoption): PASS. The closest existing work is the complex RBM times pair product (Nomura et al., PRB 96, 205152 (2017)), whose RBM factor can change the pair-product sign, and NNBF (Luo & Clark, PRL 122, 226401 (2019)), which deforms the orbitals. Neither is standard as a *sign fixer*. The RBM phase is dense and global and has never been shown to fix nodes; backflow must drive a determinant through zero. No search turned up a degree- and locality-certified residual sign layer for fermions. The concrete mechanism against the adoption barrier is a pre-training diagnostic: locality and low degree of $r$ are measured on ED clusters against an equal-capacity non-local control, so the layer is adopted only if the structure is shown to exist.
  - Gate 1 (Conservation & Asymptotics): PASS. The layer is diagonal in $\mathbf{n}$, so it conserves $N_e$ and $S^z$, and it is translation-covariant when weights are shared. Its expressivity is restricted (bounded degree and locality), and there is no claim over QMA-hard spaces.
  - Gate 2 (Baseline Gameability): PASS. The metric is defined *relative* to the foil's optimized sign. A positive Jastrow cannot change $s$, and a mean-field state is the reference itself. An equal-capacity non-local layer is the built-in control against gains from capacity alone.
  - Gate 3 (Hard-Core Shield): PASS. The instances are doped ($\delta=1/8$), $t'/t=\pm0.3$, 2D, and $J/t=0.4$, so the route does not rely on half filling, 1D, or the non-interacting limit.
  - *Verdict*: SURVIVED
* **Route 5**: Pfaffian of a learned, configuration-dependent pairing matrix with non-mean-field structure.
  - Gate 0 (Novelty & Adoption): FAIL. This is a relabeling of neural-network-augmented Pfaffians (Chen et al., arXiv:2507.10705) and Neural Pfaffians (Gao & Günnemann, arXiv:2405.14762). It inherits the same orbital-anchored zero-crossing barrier and adds no new mechanism.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): not evaluated.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED
### 3. Top Surviving Route & Sharpest Gap
* **Mechanism**: Keep the foil determinant or Pfaffian as a cheap prior that is already close to the true sign. Put all nodal freedom into a local, low-degree Walsh layer $c(\mathbf{n})$ that multiplies it, implemented in VMC as a phase $e^{i\pi\sigma(b+w\cdot\chi)}$. A sign can then change through phase rotation without an amplitude zero, and without moving the orbitals.
* **Sharpest Gap**: It is unverified that the residual sign $r(\mathbf{n})=\mathrm{sgn}\,\psi_0\cdot s_{\mathrm{foil}}$ on doped $t'\ne0$ clusters is concentrated on plaquette-local Walsh characters of degree at most 3, in the energy-relevant sense (reducing $E_{\mathrm{FN}}$, not just $W$). A second gap follows: even if that holds, VMC must learn $w$ from energy gradients rather than oracle labels, which is P1.3's question.
### 4. 24-Hour Decisive Test & Compute Execution
* **Minimal Instance**: $t$–$t'$–$J$ model with $t=1$ and $J/t=0.4$ on a $4\times4$ periodic lattice with 2 holes ($N_\uparrow=N_\downarrow=7$, $S^z=0$, 411,840 states, $\delta=1/8$), at $t'/t=-0.3$ and $t'/t=+0.3$. A third instance is a $3\times4$ open lattice with 2 holes and $t'=0$ (16,632 states). The instances are 2D, doped, and non-stoquastic, and both signs of $t'$ are included.
* **Scalar Metric $M$**: The foil sign $s_{\mathrm{foil}}$ is the better of two optimized families: projected AF Slater $(t'_v,m,a)$ and projected $d$-BCS $(\Delta,\mu,t'_v)$. Each is optimized by Nelder–Mead on $E_{\mathrm{FN}}$. Here $t'_v$ is a variational next-nearest-neighbour hopping, $m$ a staggered field, $a$ an x-bond anisotropy, $\Delta$ the $d$-wave gap and $\mu$ the chemical potential. The layer $c$ is fitted to $r$ by logistic regression on 4000 samples drawn from $\lvert\psi_0\rvert^2$, with the L2 penalty and the trivial layer $c\equiv1$ chosen by validation. The metric is
$$M_{\mathrm{local}}=\frac{E_{\mathrm{FN}}[s_{\mathrm{foil}}c_{\mathrm{local}}]-E_0}{E_{\mathrm{FN}}[s_{\mathrm{foil}}]-E_0},\qquad M_{\mathrm{null}}=\frac{E_{\mathrm{FN}}[s_{\mathrm{foil}}c_{\mathrm{null}}]-E_0}{E_{\mathrm{FN}}[s_{\mathrm{foil}}]-E_0},$$
where $c_{\mathrm{null}}$ uses the same number of characters, with the same degrees, built from variables drawn across the whole lattice. $E_{\mathrm{FN}}$ uses 10 self-consistent guiding iterations. With the exact sign it is $4.5\times10^{-4}$ above $E_0$ on $3\times4$, which sets the resolution. The script also reports the absolute errors $\Delta E=E_{\mathrm{FN}}-E_0$, the sign disagreement $W$, and the errors of the correlations $C_s(r)$ and $g_{hh}(r)$ alongside $M$. Here $C_s(r)$ is the staggered spin correlation $\langle S^z_0S^z_r\rangle(-1)^r$ and $g_{hh}(r)$ is the hole–hole correlation.
* **Execution Plan**: The test script is `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry/micro_test.py` and the job script is `/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry/submit_test.sbatch` (partition `kim`, 8 CPUs, 24G, 20 h; runs smoke, the two kill-evidence checks, then the decisive run, and writes logs and JSON next to the script). The command, not run, is `cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry && sbatch submit_test.sbatch`. The estimated wall time is 3–5 h, about $10^3$ Lanczos solves on 411,840 states per $4\times4$ instance.
* **Thresholds**: **PASS** requires all three on every instance: $M_{\mathrm{local}}\le0.5$; $M_{\mathrm{null}}-M_{\mathrm{local}}\ge0.1$; and the $C_s$ and $g_{hh}$ errors no worse than the foil's. **FAIL** if, on any instance, $M_{\mathrm{local}}\ge0.9$ or $M_{\mathrm{local}}\ge M_{\mathrm{null}}$. **AMBIGUOUS** covers everything else. The pre-committed action is to rerun once with `--degree 4`. If that run is not PASS, the route is recorded as FAIL and the node is closed with the result "the foil's sign residual is not local and low-degree, so P1.2 needs a non-local mechanism".
* **Smoke-test evidence**: The command was `OMP_NUM_THREADS=4 /share/apps/software/anaconda3/bin/python3 micro_test.py --mode smoke` (output saved to `smoke_output.txt`). This was a reduced preview on the $3\times4$ open lattice at $t'=-0.3$, with 40 Nelder–Mead evaluations, 1 start and 1500 samples. It is not one of the decisive instances.

```
(a) 2-site 1 electron: E0=-1.0000000000 (expect -1)
(b) 4-site ring Heisenberg: E0=-3.0000000000 (expect -3)  E_FN[Marshall]=-3.0000000000
(c) polarised 2x3 N_h=1: E0-E_free=1.33e-15  W[holon det]=-2.22e-16  W[Slater]=0.00e+00
(c) polarised 2x3 N_h=2: E0-E_free=2.22e-15  W[holon det]=-8.88e-16  W[Slater]=0.00e+00
(c) polarised 3x4 N_h=1: E0-E_free=1.33e-15  W[holon det]=-2.22e-16  W[Slater]=0.00e+00
(c) polarised 3x4 N_h=2: E0-E_free=3.55e-15  W[holon det]=5.40e-32  W[Slater]=-2.22e-16
(d) doped 3x4 exact sign: E_FN-E0 at K=0,5,10: [0.29473, 0.00189, 0.00045]
[3x4o:2:-0.3] D=16632 E0=-7.28967405 deg=1 (0.1s)
  foil slater_af E_FN=-7.18526867 dE=1.0441e-01 params=[-0.3297, 0.295, 0.02] (17s)
  layer local P=460 val_err=0.0653 (const-layer weighted err 0.0733) nontrivial=True E_FN=-7.18227840 (3s)
  layer null  P=460 val_err=0.0667 (const-layer weighted err 0.0733) nontrivial=True E_FN=-7.16968315 (3s)
  exact_sign dE=4.4542e-04 W=0.0000 errCs=0.000e+00 errG=0.000e+00
  foil       dE=1.0441e-01 W=0.0733 errCs=8.161e-03 errG=3.079e-03
  local      dE=1.0740e-01 W=0.0717 errCs=8.175e-03 errG=3.183e-03
  null       dE=1.1999e-01 W=0.0636 errCs=8.123e-03 errG=3.273e-03
  foil=slater_af params=[-0.3297, 0.295, 0.02]  M_local=1.0286  M_null=1.1493  corr_ok=False
SMOKE: OK
```

The checks confirm three things. The ED is correct: the 2-site value $-t$, the 4-site ring value $-3J$, and the polarized sector, which equals the free-fermion sum to $10^{-15}$. The sign conventions are consistent: the holon-determinant and Slater signs are exact in the polarized sector, and Marshall is exact for the ring. The fixed-node bound decreases monotonically toward $E_0$ with the exact sign.

The preview's foil sign is wrong on 7.3% of the weight and costs $0.104t$ of fixed-node energy, while the exact sign leaves only a $4.5\times10^{-4}$ residual. That is direct evidence that wrong foil signs cost real energy, which is relevant to P1.1. The local layer lowers $W$ only from 0.073 to 0.072 and does not lower $E_{\mathrm{FN}}$ ($M_{\mathrm{local}}=1.03$). On this reduced preview the route would FAIL. The decisive run on $4\times4$ periodic with the full budget is still required.
### 5. Local Autopsy Log
* **Route 1 (Jordan–Wigner sign plus sign network)**: This is a relabel of Inui et al. (arXiv:2108.08631), RBM+PP (PRB 96, 205152) and Choo et al. (Nat. Commun. 11, 2368). The non-local Jordan–Wigner string and the poor generalization of generic sign networks remain unaddressed.
* **Route 2 (pairwise antisymmetric product)**: A fixed $f$ gives a quadratic GF(2) sign, which already fails to represent free-fermion nodes on $3\times4$ (`pairwise_check`: False). A configuration-dependent $f$ collapses the route into Route 1.
* **Route 3 (slave-fermion holon determinant times Marshall)**: The phase string ties the sign to joint spin–hole positions. At $J/t=0.4$, even the best arbitrary hole-only function times Marshall has $W=0.34$, against $W=0.07$ for the foil Slater sign. The anchor is good only when $J/t$ is at least about 10 (`holon_check`; output in `tests/P1.2_unanchored_antisymmetry/kill_evidence_output.txt`).
* **Route 5 (neural Pfaffian pairing)**: This is a relabel of Chen et al. (arXiv:2507.10705) and Gao & Günnemann (arXiv:2405.14762). The same orbital zero-crossing anchor remains.
### Round 2 — Mutation & Replenishment (2026-11-11)
* **Boundary conditions carried in**:
    * Critic objection 1: a single linear threshold over an extensive Walsh sum cannot represent a product of independent local sign corrections.
    * Critic objection 2: a local, sign-capable diagonal correlator times a Slater or BCS reference is the correlator-product-state (CPS) times Slater ansatz, and the round-1 test fitted the layer to oracle labels from $\lvert\psi_0\rvert^2$.
    * P1.1 (job 329057): at width $H=128$ (hidden units of the backflow MLP), NNBF keeps $61$–$77\%$ of its energy error in the sign, and its wrong-sign weight $w_{\mathrm{sign}}=\sum_{\mathbf{n}:\,s_\theta s_0<0}\lvert\psi_0(\mathbf{n})\rvert^2$ is about $0.20$ on the $N_\uparrow=N_\downarrow=4$ instance at every width ($0.210$, $0.205$, $0.197$ for $H=8,32,128$).
    * Global anti-patterns: relabelled sign networks; continuation from a mean-field limit; single-proposal stochastic replacement of exact non-stoquastic sums.
* **What the new layer must do**: change the sign of $\psi$ at individual configurations through a mechanism that is neither a wider backflow MLP nor a second determinant, using only information computable at $\mathcal{O}(N_e^3)$ per amplitude, and be trainable by the energy gradient alone.
* **Key identity behind the batch**: for any eigenstate, $\psi_0(\mathbf{n})=\sum_{\mathbf{m}\ne\mathbf{n}}H_{\mathbf{n}\mathbf{m}}\psi_0(\mathbf{m})/(E_0-H_{\mathbf{n}\mathbf{n}})$, with $H_{\mathbf{n}\mathbf{m}}$ the Hamiltonian matrix element between occupation configurations $\mathbf{n}$ and $\mathbf{m}$. The true sign at $\mathbf{n}$ is therefore a weighted vote of the amplitudes at its Hamiltonian neighbours. A single determinant can only imitate this vote collectively, by moving its whole nodal surface. The routes below put the vote into the ansatz explicitly.
#### Generation Batch 2
* **Route 2.1 (Mutated from 1.4 via Operation A.1, target: expressivity of $c$)**: Product of plaquette-local sign or phase factors times the foil, $\psi=\Phi(\mathbf{n})\prod_pc_p(\mathbf{n}_p)$. Here $\Phi$ is the Slater–Jastrow reference and $c_p$ a learned real or complex function of the occupations $\mathbf{n}_p$ inside plaquette $p$. Trained by SR from the energy. This replaces the single threshold by a product of independent local factors, as the critic required.
  - Gate 0 (Novelty & Adoption): FAIL. This is exactly CPS times Slater (Changlani et al., PRB 80, 245116 (2009); Neuscamman, Changlani, Kinder & Chan, PRB 84, 205132 (2011), which notes that correlators are diagonal operators identical to Jastrow factors), and with complex factors it is the complex RBM times pair product (Nomura et al., PRB 96, 205152 (2017)). These factors see only local occupations, so they cannot know where the reference's nodes are. A plaquette pattern that is wrong-signed in one region is right-signed in another. That is why CPS signs have not displaced backflow, and this route adds no fix for it.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): not evaluated.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED. *Mutation fuel*: a sign factor must be conditioned on the **reference's own nodal geometry**, not only on local occupations.
* **Route 2.3 (Mutated from 2.1 via Operation A.1, conditioning on the reference)**: Linear first-order dressing, $\psi(\mathbf{n})=\langle\mathbf{n}\rvert\big(1+\sum_b\alpha_b(\mathbf{n})\hat K_b\big)\lvert\Phi\rangle$. Here $\hat K_b$ is the hopping operator on bond $b$, and $\alpha_b(\mathbf{n})$ is produced by an MLP.
  - Gate 0 (Novelty & Adoption): FAIL. With uniform $\alpha$ this is the one-step Lanczos or power step (Sorella, PRB 64, 024512 (2001); "Lanczos steps to improve variational wave functions", arXiv:1412.2656), which is applied after optimization to Hubbard backflow states (arXiv:2308.11823, arXiv:2507.01856) and NNBF (Loehr & Clark, arXiv:2510.26906). A linear exchange dressing $1+\sum_b\alpha_b\hat X_b$ is already P4.1-R1. The adoption barrier is size consistency: arXiv:1412.2656 states that $p$ Lanczos steps are not size-consistent unless $p$ grows with $V$. A configuration-dependent $\alpha$ does not change this, because the correction is still a sum of $\mathcal{O}(V)$ terms against a fixed $1$.
  - Gate 1 (Conservation & Asymptotics): FAIL. The relative effect of an extensive linear correction on an intensive quantity vanishes as $V\to\infty$.
  - Gate 2 (Baseline Gameability): not evaluated.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED. *Mutation fuel*: the correction must be a **product** of local factors, as in Route 2.1, but each factor must be built from the reference amplitudes, as in Route 2.3.
* **Route 2.4 (Operation B.3, cross-field translation from fermionic tensor-network gate layers and the Baeriswyl projector)**: Apply a brick layer of two-site fermionic gates, $\psi=\langle\mathbf{n}\rvert\prod_b e^{\theta_b(\mathbf{n})\hat K_b}\lvert\Phi\rangle$, the lattice analogue of the Baeriswyl kinetic projector $e^{-\lambda\hat T}$ (Hetényi, Tanatar & Martelo, PRB 93, 174518 (2016), arXiv:1605.02698) and of Yanagisawa's off-diagonal wave function (arXiv:cond-mat/0206033, arXiv:1611.02391).
  - Gate 0 (Novelty & Adoption): FAIL. The uniform-$\lambda$ form is published.
  - Gate 1 (Conservation & Asymptotics): FAIL. With interacting gates the exact amplitude is a signed sum over $\mathcal{O}(2^{N_b})$ branches, where $N_b$ is the number of bonds. Sampling the branches as auxiliary configurations (the Hetényi or OWMC route) reintroduces a sign problem in the auxiliary sum; the bosonic application works because that sum is positive. With quadratic gates the layer is an orbital rotation and adds nothing.
  - Gate 2 (Baseline Gameability): not evaluated.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED. *Mutation fuel*: keep only the first-order, single-bond content of each gate and **factorize** it. This is Route 2.2.
* **Route 2.2 (Mutated from 2.1, 2.3 and 2.4 via Operation A.1; kinetic frame), "factorized-ratio dressing" (FRD-K)**:
$$\psi_\theta(\mathbf{n})=\Phi(\mathbf{n})\prod_{e}\Big[1+\alpha_e(\mathbf{n})\,g\big(R_e(\mathbf{n})\big)\Big],\qquad R_e(\mathbf{n})=\frac{c_e(\mathbf{n})\,\Phi(\mathbf{m}_e(\mathbf{n}))}{\Phi(\mathbf{n})},\qquad g(R)=\frac{R}{\sqrt{1+R^2}} .$$
    * The symbols:
        * $\Phi$ is a Slater determinant with trainable orbitals times a two-body Jastrow factor (the P1.1 foil).
        * $e=(b,\sigma)$ runs over every nearest- and next-nearest-neighbour bond $b$ and spin $\sigma$.
        * $\mathbf{m}_e(\mathbf{n})$ is the configuration reached by the single hop of spin $\sigma$ across $b$. When no hop is possible the factor is $1$.
        * $c_e(\mathbf{n})=\langle\mathbf{n}\rvert-t_b\hat c^\dagger\hat c\lvert\mathbf{m}_e\rangle$ is the signed Hamiltonian matrix element, including the Jordan–Wigner sign.
        * $\alpha_e(\mathbf{n})=[\tanh(\mathbf{x}W_1+b_1)W_2+b_2]_e$ is the same MLP as NNBF: the input $\mathbf{x}\in\{\pm1\}^{2V}$ is the spin-resolved occupation, with the same hidden width $H$. Only the read-out differs: one coefficient per edge instead of $V(N_\uparrow+N_\downarrow)$ orbital shifts.
    * A factor turns negative, flipping the sign that $\Phi$ assigns to $\mathbf{n}$, only when $\alpha_eg_e<-1$, i.e. when the Hamiltonian neighbour $\mathbf{m}_e$ votes against $\Phi$'s sign with the weight $\alpha_e$ that the network assigns. Since $\lvert g\rvert<1$, $\psi$ stays finite at $\Phi$'s nodes, where $g\to\pm1$.
    * *Why this is not the round-1 failure*: the layer is a **product** of edge-local factors, so it can express independent local corrections (critic objection 1). Each factor depends on the **reference amplitude ratio**, which carries the determinant's non-local fermionic sign and nodal proximity; a CPS sees only local occupations. And it is **trained by SR from the energy** from the same UHF start as NNBF, with no labels (critic objection 2).
    * *Cost*: with the inverse of the $N_\sigma\times N_\sigma$ occupied-orbital matrix (where $N_\sigma$ is the number of electrons of spin $\sigma$), each ratio $R_e$ costs $\mathcal{O}(N_\sigma)$ plus an $\mathcal{O}(1)$ Jastrow ratio from stored fields, so one amplitude costs $\mathcal{O}(N_e^3+zVN_e)$ for coordination number $z$. A local energy needs $\psi$ at $\mathcal{O}(zN_e)$ connected configurations, each requiring $\mathcal{O}(zV)$ ratios from a rank-1-updated inverse, giving $\mathcal{O}(z^2VN_e^2)$. That is cubic when $V\propto N_e$, the same order as NNBF. No second determinant and no sum over determinants is needed: every ratio is a ratio of the **same** determinant.
  - Gate 0 (Novelty & Adoption): PASS. Searches for products of determinant-ratio factors, configuration-dependent Lanczos coefficients, and operator-dressed NQS returned the one-step Lanczos and power step (PRB 64, 024512; arXiv:1412.2656; arXiv:2308.11823; arXiv:2510.26906), Baeriswyl and off-diagonal Gutzwiller states (arXiv:1605.02698; arXiv:1008.1272; arXiv:1611.02391), backflow and HFDS as Jastrow-like corrections to a determinant (Lange et al., arXiv:2411.10430; Liu & Clark, arXiv:2311.09450), and CPS. None of them uses a product over Hamiltonian edges of bounded functions of the reference's own neighbour ratios. The known methods are not standard for three reasons, and FRD answers each with a named mechanism:
    * The Lanczos step is not size-consistent. FRD's correction is a product of local factors.
    * The exact Baeriswyl projector needs auxiliary sampling with its own sign problem. FRD has a closed-form amplitude at $\mathcal{O}(zVN_e)$ extra cost.
    * CPS cannot see the nodes. FRD's factors are functions of $\Phi(\mathbf{m})/\Phi(\mathbf{n})$.
  - Gate 1 (Conservation & Asymptotics): PASS.
    * Every edge move conserves $N_\uparrow$ and $N_\downarrow$.
    * Shared MLP weights give translation covariance when the input is translation-covariant.
    * The product of edge-local factors is size-consistent.
    * There is no claim of polynomial convergence; the claim is a restricted family whose sign is set by a local Hamiltonian vote.
    * Exactness checks: summed over kinetic edges, $c_e\Phi(\mathbf{m}_e)$ reproduces the off-diagonal part of $H$ acting on a random vector to $3.6\times10^{-15}$, and the analytic SR gradient matches finite differences to $10^{-9}$ (smoke output below).
  - Gate 2 (Baseline Gameability): PASS.
    * The control is the strongest cheap baseline that also learns: NNBF times the same Jastrow, with the same MLP width, the same UHF start, the same SR force, sample count, step size and step budget, the same two-value SR-shift search, and a matched parameter count. On the $4\times3$ PBC clusters $P=1165$ at $H=8$ and $P=4069$ at $H=32$ for both arms, because the $96$ edges equal the $96$ orbital outputs.
    * The Jastrow–Slater foil alone cannot change the sign.
    * The diagnostic `flip_weight` (the $\lvert\psi\rvert^2$ weight on configurations where the dressing actually flips $\Phi$'s sign) reports whether any gain comes from the vote or only from re-optimized orbitals.
  - Gate 3 (Hard-Core Shield): PASS.
    * The instances are the three P1.1 clusters: doped $4\times3$ PBC with $t'=-0.25$, at $U=8$ and $U=4$.
    * *[Retracted in round 3]* This bullet claimed a same-sector competitor on the $N_\uparrow=N_\downarrow=4$ instance. That claim was wrong: the low level there is a spin triplet at another momentum (master-doc Correction), so the round-2 test covered the sign half only. See the round-3 autopsy.
    * Nothing uses half filling, 1D, or a mean-field continuation. The UHF start is shared with the control, so any anchor bias is identical in both arms.
  - *Verdict*: SURVIVED
* **Route 2.5 (Operation B.1, duality / frame transformation: Schrieffer–Wolff frame), "exchange-frame FRD" (FRD-X)**:
    * *Mechanism*: the same factorized-ratio engine, but the edges are the off-diagonal moves of the large-$U$ effective Hamiltonian obtained by the Schrieffer–Wolff transformation, not of the bare $H$. Each $e=(b,o)$ is the spin exchange $\hat S^+_i\hat S^-_j$ (orientation $o=0$) or $\hat S^-_i\hat S^+_j$ (orientation $o=1$) on bond $b=(i,j)$, with $c_e=\langle\mathbf{n}\rvert\hat S^+_i\hat S^-_j\lvert\mathbf{m}\rangle=-s_\uparrow s_\downarrow$, where $s_\sigma$ is the Jordan–Wigner sign of the spin-$\sigma$ hop.
    * *Why it is a distinct route*: at $U=8$ the kinetic edges of Route 2.2 lead from singly occupied configurations into doublon configurations, where $\lvert\Phi\rvert$ is small and poorly converged, so the kinetic vote is cast by the least reliable amplitudes. Exchange edges connect configurations within the same doublon sector, which is where the low-energy spin sign structure (Marshall-like, but frustrated by $t'$ and holes) is decided. The two frames make different predictions, testable on instances 0 and 1 ($U=8$) against instance 2 ($U=4$), and either can fail while the other passes.
    * *Exactness checks*: on the 2-site singlet the exchange ratio sum is $-1.0000$; on a 4-site ring at $U=400$ the exchange expectation computed through the edges is $-1.3333$, against $-1.3332$ from ED at leading order in $t/U$. Both are in the smoke output.
  - Gate 0 (Novelty & Adoption): PASS. Same audit as Route 2.2. The only exchange-dressing precedent is linear (P4.1-R1, $1+\sum_b\alpha_b\hat X_b$), which falls under the Route 2.3 kill and is not a product of ratio factors.
  - Gate 1 (Conservation & Asymptotics): PASS. Exchange conserves $N_\uparrow$, $N_\downarrow$ and the doublon number. The cost is identical to Route 2.2: one exchange ratio is a product of two single-spin ratios times a Jastrow ratio.
  - Gate 2 (Baseline Gameability): PASS. Same matched NNBF times Jastrow control, with an identical parameter count.
  - Gate 3 (Hard-Core Shield): PASS. The Schrieffer–Wolff frame does not remove the sign problem, since doped $t$–$t'$–$J$ is non-stoquastic. The verdict requires every instance, including $U=4$, where the frame is least natural.
  - *Verdict*: SURVIVED
* **Route 2.6 (Operation B.1, duality: momentum or natural-orbital frame for NNBF)**: Sample and run backflow in a learned orbital frame, $\hat c^\dagger_k=\sum_iU_{ik}\hat c^\dagger_i$, so that the nodes are defined in the rotated occupation basis.
  - Gate 0 (Novelty & Adoption): FAIL. This is published: variational orbital rotations co-optimized with NQS (Robledo Moreno, Cohn, Sels & Motta, arXiv:2302.11588), and single-particle basis optimization for HFDS and DMRG (Gauvin-Ndiaye et al., PRL 134, 076502 (2025)); a momentum-space reference also appears in arXiv:2412.12248. The adoption barrier is that the rotated Hubbard interaction has $\mathcal{O}(N_\uparrow N_\downarrow V^2)$ connected elements per sample, the P2.2 local-energy wall. No mechanism here bypasses it.
  - Gate 1 (Conservation & Asymptotics): not evaluated.
  - Gate 2 (Baseline Gameability): not evaluated.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED
### Top Surviving Route & Sharpest Gap (Round 2)
* **Mechanism (Route 2.2, with 2.5 as the sibling frame)**: keep one cheap determinant $\Phi$, and let each configuration's sign be corrected by a product of edge-local factors $1+\alpha_e(\mathbf{n})g(R_e)$. Each factor is driven by $\Phi$'s own amplitude ratio to a Hamiltonian neighbour. The nodes move configuration by configuration, through a learned local version of the eigen-equation vote $\psi_0(\mathbf{n})\propto\sum_{\mathbf{m}}H_{\mathbf{n}\mathbf{m}}\psi_0(\mathbf{m})$, instead of by deforming the orbitals of a single determinant, which is the only lever NNBF has.
* **Sharpest Gap**: a one-step vote can only flip configurations whose Hamiltonian neighbours are mostly right-signed (P1.3 records the same locality limit for one-step projector targets). It is unknown whether NNBF's wrong-sign weight of about $0.20$ consists of such isolated configurations, which are locally repairable, or of extended regions where neighbours agree on the wrong sign, which would require a global node move. A second, practical gap is learnability. A factor flips by passing through zero, where $\partial\log\psi/\partial\theta$ diverges. On the $3\times2$ probe, SR with shift $10^{-3}$ was unstable for FRD-K: the energy rose from $-5.31$ to $-3.52$ between iterations 150 and 250. A shift of $10^{-2}$ stabilized it. The test therefore gives every arm the same two-shift search, selected by energy alone.
### 24-Hour Decisive Test (Round 2)
* **Minimal Instance**: the three P1.1 clusters, all $4\times3$ PBC with $t'=-0.25$:
    * instance 0: $N_\uparrow=N_\downarrow=5$, $U=8$, $D=627264$ configurations;
    * instance 1: $N_\uparrow=N_\downarrow=4$, $U=8$, $D=245025$ (no same-sector competitor; round-3 correction);
    * instance 2: $N_\uparrow=N_\downarrow=5$, $U=4$, $D=627264$.
* **Training**: every arm is trained variationally by SR with the exact force and an exact-sample metric. Arms: FRD-K, FRD-X, and the control NNBF times Jastrow, each at $H=8$ and $H=32$ and each at SR shifts $10^{-3}$ and $10^{-2}$, for 1500 iterations from the same UHF orbitals. No reference to $\psi_0$ enters training, and the per-(arm, width) shift is chosen by the lowest variational energy.
* **Scalar Metric $M$**: everything is computed with P1.1's `budget()` on exact sums.
    * $\Delta E_{\mathrm{sign}}=E_{\mathrm{sign}}^{\mathrm{up}}-E_0$ is the rigorous upper bound on the error of the best non-negative amplitudes under $\mathrm{sgn}\,\psi_\theta$. P1.1's code takes it as the minimum over four L-BFGS-B starts and the ten Haaf fixed-node state.
    * $\Delta E_{\mathrm{tot}}=E_\theta-E_0$ is the total energy error.
    * $e_C$ is the maximum long-range error of the singlet bond-pair correlation.
    * The ratios against the matched control are
$$\rho_{\mathrm{sign}}(X,H)=\frac{\Delta E_{\mathrm{sign}}[X_H]}{\Delta E_{\mathrm{sign}}[\mathrm{NNBF{\cdot}J}_H]},\qquad\rho_{\mathrm{tot}}(X,H)=\frac{\Delta E_{\mathrm{tot}}[X_H]}{\Delta E_{\mathrm{tot}}[\mathrm{NNBF{\cdot}J}_H]},\qquad\rho_C(X,H)=\frac{e_C[X_H]}{e_C[\mathrm{NNBF{\cdot}J}_H]},$$
      with $X\in\{\text{FRD-K},\text{FRD-X}\}$.
    * Also reported: $w_{\mathrm{sign}}$, `flip_weight`, and whether $\Delta E_{\mathrm{sign}}[X_{32}]$ falls below the stored P1.1 NNBF value at $H=128$ ($0.0765$ on instance 1).
* **Execution Plan**: test folder `tests/P1.2_unanchored_antisymmetry_r2/`.
    * `micro_test.py` imports `p11_core.py`, a verbatim copy of P1.1's ED, NNBF, SR and `budget()`.
    * `submit_test.sbatch` is a Slurm array `0-35`, one task per (instance, arm, width, shift): partition `kim`, 8 CPUs, 32G, 24 h.
    * Measured cost on the login node (4 threads, instance 1, $H=32$): NNBF times Jastrow 2.47 s per iteration, FRD-X 5.78 s, FRD-K 8.46 s. The worst task (FRD-K on $D=627264$) is about 10 h including its budget.
    * Command (not run): `cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry_r2 && sbatch submit_test.sbatch`.
    * Verdict afterwards: `python3 micro_test.py --aggregate results`.
* **Thresholds** (pre-committed, per route, implemented in `aggregate()`):
    * **FAIL** if, on any instance, $\rho_{\mathrm{sign}}(X,32)\ge0.9$ or $\rho_{\mathrm{tot}}(X,32)>1.1$.
    * **PASS** if, on every instance, $\rho_{\mathrm{sign}}(X,32)\le0.7$, $\rho_{\mathrm{tot}}(X,32)\le1.0$ and $\rho_{\mathrm{sign}}(X,8)\le1.0$, and $\rho_C(X,32)\le1.1$ on the instance that round 2 wrongly flagged as having a competitor (retracted in round 3).
    * **AMBIGUOUS** otherwise. Pre-committed action: rerun with seeds 2 and 3; PASS if the per-instance medians satisfy PASS, otherwise FAIL.
    * Interpretation rule (not part of the verdict): a PASS with `flip_weight` near $0$ is credited to the re-optimized orbitals, not to the vote, and is logged as such.
* **Smoke-test evidence**: command `OMP_NUM_THREADS=4 /share/apps/software/anaconda3/bin/python3 micro_test.py --mode smoke` (1.7 s; full log in `smoke_output.txt`). The literal output, with the per-iteration lines removed:

```
[10:47:31] ALL CHECKS PASSED
[10:47:31] check kinetic edges 3x2 N=(2,2): max|K y - H_off y| = 3.55e-15
[10:47:31] check kinetic edges 2x3 N=(2,1): max|K y - H_off y| = 1.33e-15
[10:47:31] check exchange edges 2-site singlet: R = [-1.0000000000000004, -0.9999999999999996] (expect -1 each)
[10:47:31] check exchange edges 4-ring: sum<S+S-+h.c.>/2 = -1.3333  from ED sum<S.S>-<SzSz> = -1.3332 (leading order in t/U)
[10:47:31] check FRD-K gradient: analytic 7.8455276843e-01  finite-diff 7.8455276914e-01  |O-vs-grad| 2.2e-15  P=229
[10:47:31] check FRD-X gradient: analytic -4.5582141315e-02  finite-diff -4.5582141173e-02  |O-vs-grad| 2.7e-15  P=229
[10:47:31] ALL R2 CHECKS PASSED
[10:47:31] model V=6 N=(2,2) D=225  E0..2=[-5.626495825504258, -4.944174688230718, -4.944174688230716]  Delta1/V=0.1137
SMOKE nnbfj_H4: P=199 dE_tot=5.5277e-02 dE_sign=1.4313e-02 w_sign=0.0027
SMOKE frdK_H4: P=229 dE_tot=9.9778e-02 dE_sign=4.5485e-02 w_sign=0.0126
SMOKE frdX_H4: P=229 dE_tot=1.1779e-01 dE_sign=3.7435e-02 w_sign=0.0099
SMOKE: OK
```

* **Login-node probe (not decisive)**: the probe cluster is $3\times2$ with $N_\uparrow=N_\downarrow=2$, $U=8$ and $t'=-0.3$. It has no competitor and is too small for the Hard Core, and parameter counts are not matched there ($199$ against $229$). After 400 SR iterations at shift $10^{-2}$:
    * $\Delta E_{\mathrm{sign}}$: NNBF times Jastrow $0.00398$, FRD-X $0.00347$, FRD-K $0.00734$;
    * $\Delta E_{\mathrm{tot}}$: $0.0106$, $0.0207$, $0.0318$;
    * `flip_weight` of $0.007$–$0.009$, so the factors do flip signs.
  So FRD-X is slightly ahead on the sign and behind on the total energy, and FRD-K is behind on both. The aggregation path was exercised end to end on a 60-iteration run, where both routes returned FAIL, as expected on this tiny cluster where backflow is close to universal. This shows that the rule can return FAIL.
### Local Autopsy Log (Round 2 additions)
* **Round-1 Route 4, the residual Walsh sign layer** (killed by the critic):
    * One threshold cannot express a product of local corrections.
    * As a diagonal local correlator it is CPS times Slater.
    * Its test used oracle labels from $\lvert\psi_0\rvert^2$.
    * Mutated into Route 2.1, then 2.3, then 2.2.
* **Route 2.1, a product of plaquette sign factors**: Gate 0. It is CPS times Slater or complex RBM times pair product, and local-occupation factors cannot locate the reference's nodes.
* **Route 2.3, a linear dressing $1+\sum_b\alpha_b(\mathbf{n})\hat K_b$**: Gates 0 and 1. It is a Lanczos or power step with local coefficients and is not size-consistent (arXiv:1412.2656).
* **Route 2.4, an exact brick layer of fermionic gates or Baeriswyl projector**: Gate 1. An exponential signed branch sum, and sampling it brings back a sign problem.
* **Route 2.6, a rotated or momentum orbital frame for NNBF**: Gate 0 (arXiv:2302.11588; PRL 134, 076502). The rotated two-body Hamiltonian hits the P2.2 local-energy wall.
* **Anti-pattern recorded for the master doc**: *a sign layer that sees only local occupations cannot find the nodes*, whether it is CPS, Walsh or local RBM. Sign corrections must be conditioned on the reference's own amplitudes, which is what FRD does.
### Round 3 — Mutation & Replenishment (2026-11-11)
* **Boundary conditions carried in** (round-2 critic, binding):
    * [Major] The Gate 3 shield cited the retracted same-sector competitor, and the aggregate keyed a clause to it.
    * [Major] There was no symmetry-projected arm. FRD's edge factors act like a localized power step and might filter contaminant irreps, so a lower $\rho_{\mathrm{sign}}$ could be symmetry restoration rather than nodal repair; `flip_weight` cannot separate the two.
    * [Major] On the $4+4$ instance the shared UHF contamination ($w_{\mathrm{sign}}\approx0.20$) nearly decides the verdict.
    * [Minor] Parameters were matched only on $4+4$; on $5+5$ NNBF had 240–790 extra parameters.
    * [Minor] One seed decided the verdict.
    * [Minor] Size consistency was overstated: to first order FRD is a local-energy Jastrow, not a controlled $e^{-\lambda\hat T}$.
* **New evidence from P1.1-P (job 407525, interim logs, still running)**: projected-SR NNBF without a Jastrow reaches, at width $H=8$ after 1100–1500 steps, $\Delta E_{\mathrm{tot}}\approx3\times10^{-3}$ on $5+5$, $U=8$ (energies $-11.0276$ to $-11.0283$ against $E_0=-11.0304$), $8\times10^{-4}$ on $4+4$, $U=8$ (finished budgets: $\Delta E_{\mathrm{sign}}<10^{-4}$, 3 seeds) and $3\times10^{-4}$ on $5+5$, $U=4$ ($-13.8586$ against $-13.8589$). The same NNBF unprojected has $\Delta E_{\mathrm{tot}}=0.12$–$0.20$. Once the full $\lvert G\rvert=48$ space group is projected, **the three P1.1 clusters leave almost no sign error for any sign layer to repair**. A round-3 test on them could not discriminate FRD from the control, so it moves to lower-symmetry clusters and gates its verdict on a premise check.
#### Generation Batch 3
* **Route 3.0 (re-gating Route 2.2 against the corrected first-order analysis), "local-energy Jastrow" limit**: for $\lvert R_e\rvert\ll1$, $g(R_e)\approx R_e$ and $\ln(1+\alpha_eg)\approx\alpha_eR_e$, so
$$\psi_\theta(\mathbf{n})\approx\Phi(\mathbf{n})\exp\Big(\sum_e\alpha_e(\mathbf{n})R_e(\mathbf{n})\Big),\qquad \alpha_e\equiv-\lambda\ \Rightarrow\ \psi\approx\Phi(\mathbf{n})\,e^{-\lambda E^{\mathrm{kin}}_{\mathrm{loc}}(\mathbf{n})},\quad E^{\mathrm{kin}}_{\mathrm{loc}}(\mathbf{n})=\frac{(\hat T\Phi)(\mathbf{n})}{\Phi(\mathbf{n})}=\sum_{e\in K}R_e(\mathbf{n}),$$
  with $\hat T$ the kinetic operator and $K$ the kinetic edges. The exact Baeriswyl state is $e^{-\lambda\hat T}\Phi=\Phi\,[1-\lambda E^{\mathrm{kin}}_{\mathrm{loc}}+\tfrac{\lambda^2}{2}(\hat T^2\Phi)/\Phi-\dots]$, and $\Phi e^{-\lambda E^{\mathrm{kin}}_{\mathrm{loc}}}$ differs from it at order $\lambda^2$ by $\tfrac{\lambda^2}{2}\big[(\hat T^2\Phi)/\Phi-(E^{\mathrm{kin}}_{\mathrm{loc}})^2\big]$, the local variance of the hopping. So the critic is right: the higher orders are **not** a controlled $e^{-\lambda\hat T}$.
  - Gate 0 (Novelty & Adoption): PASS on searching, but irrelevant. Searches for local-energy Jastrow factors, $\Phi\exp(\alpha E_{\mathrm{loc}})$, exponentiated Lanczos or power steps, and NQS sign corrections from amplitude ratios of Hamiltonian-connected configurations returned only the Lanczos step (arXiv:1412.2656), Baeriswyl and Gutzwiller–Baeriswyl projectors (Hetényi, Tanatar & Martelo, arXiv:1605.02698; the projector hierarchy of PRL 126, 206402 (2021)), local-energy fluctuations used as a diagnostic (Williams & Wagner, arXiv:1506.02072), and separate sign networks (Szabó & Castelnovo, PRR 2, 033075 (2020)). No prior art for the exponentiated local form was found.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): FAIL. $\exp(\cdot)>0$, so this limit never changes a sign. It is an amplitude-only correlator, in the foil's sign class, and cannot touch the error that P1.1 found to be binding.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED. *Consequence*: **FRD's novelty and its sign capability live only in the nonlinear regime** $\alpha_eg(R_e)<-1$, which needs $\lvert R_e\rvert\gtrsim1$, i.e. a configuration within one hop of $\Phi$'s nodal surface. The round-2 argument that leaned on size consistency is withdrawn.
* **Route 3.1 (Route 2.2 re-audited), FRD-K, kinetic frame**: the same ansatz as Route 2.2, with every claim restated.
  - Gate 0 (Novelty & Adoption): PASS, on a corrected basis. The novel element is a product over Hamiltonian edges of bounded factors $1+\alpha_e(\mathbf{n})g(R_e)$ that **cross zero only near the reference's own nodes**, so a sign flip at $\mathbf{n}$ is decided by $\Phi$'s value at a Hamiltonian neighbour. That is a local vote of the eigen-equation $\psi_0(\mathbf{n})\propto\sum_{\mathbf{m}}H_{\mathbf{n}\mathbf{m}}\psi_0(\mathbf{m})$. Its first-order limit (Route 3.0) is new as far as the searches show, but sign-inert; the Lanczos step is linear (not a product) and the Baeriswyl projector needs a signed auxiliary sum. CPS factors cannot see $\Phi$'s nodes. Why it is not standard: nobody has built a sign layer whose zero crossings are tied to the reference's nodal proximity. The adoption risk that would explain the absence is exactly the round-2 learnability gap (SR instability through zero crossings at shift $10^{-3}$), which the test measures through the two-shift search and the sampled-estimator arm.
  - Gate 1 (Conservation & Asymptotics): PASS, with a corrected claim.
    * Every edge move conserves $N_\uparrow,N_\downarrow$.
    * **Withdrawn**: "the product is size-consistent" as an approximation to $e^{-\lambda\hat T}$.
    * **What holds**: multiplicative separability. If $\Phi=\Phi_A\Phi_B$ for decoupled subsystems and $\alpha_e$ depends only on occupations inside $e$'s subsystem, then $R_e$ for $e\in A$ depends only on $\Phi_A$, and $\psi=\psi_A\psi_B$ (log-additive, like a Jastrow). The global MLP input makes this approximate only. Nothing in the novelty argument now rests on it.
    * Spin: FRD acts on spin-resolved occupations, so like NNBF it conserves $S^z$ only. By P4.1's charge-only lemma, exact $P_S$ is an oracle at ED size for both FRD and the control. Both arms receive the same projector.
  - Gate 2 (Baseline Gameability): PASS.
    * The control is NNBF×Jastrow **trained with** $P_SP_\chi$, matched per instance (FRD width chosen so $P_{\mathrm{FRD}}\le P_{\mathrm{ctrl}}$; gap at most $25$ parameters, below $1\%$; printed by `--mode checks`).
    * A vote ablation (all $\alpha_e=0$ in the same trained state) separates the vote from re-optimized orbitals.
    * A premise check makes the test return NO-TARGET, not PASS, when the projected control leaves no sign error.
  - Gate 3 (Hard-Core Shield): PASS, for the **sign half only**.
    * The instances are doped, $t'=-0.25$, 2D $4\times3$ or a 2-leg ladder, with non-stoquastic sign structure.
    * The code computes each instance's in-sector gap (Lanczos inside $\mathrm{range}(P_SP_\chi)$): $0.345t$, $0.362t$ and $0.259t$, i.e. $0.022$–$0.030t$ per site, about $20$–$30\times$ the Hard-Core scale. **No competitor half is claimed.**
    * Nothing uses half filling, the 1D open chain, or a mean-field continuation. Every competitor clause and every citation of the retracted $4+4$ competitor are removed.
  - *Verdict*: SURVIVED (conditional on the round-3 test).
* **Route 3.2 (Route 2.5 re-audited), FRD-X, exchange frame**: the same corrections. Its first-order limit is $\Phi\exp(\sum_{e\in X}\alpha_eR_e)$, a local exchange-energy correlator, also sign-inert. The sign capability again lives only where $\lvert R_e\rvert\gtrsim1$, i.e. where a spin exchange takes $\Phi$ across its node.
  - Gate 0 (Novelty & Adoption): PASS, on the same basis as Route 3.1. The only exchange-dressing precedent is linear (P4.1-R1).
  - Gate 1 (Conservation & Asymptotics): PASS. It conserves $N_\uparrow,N_\downarrow$ and the doublon number. The same separability caveat applies.
  - Gate 2 (Baseline Gameability): PASS. Same projected control, same ablation and same premise check as Route 3.1.
  - Gate 3 (Hard-Core Shield): PASS, sign half only. The $6\times2$ ladder at $U=8$ is where the Schrieffer–Wolff frame is most natural, and the $4\times3$ open cluster is where it is least natural.
  - *Verdict*: SURVIVED (conditional on the round-3 test).
* **Route 3.3 (Operation A.2 on the critic's symmetry objection), "projected-reference FRD"**: build the ratios from the projected reference, $R_e=c_e(P_\chi\Phi)(\mathbf{m}_e)/(P_\chi\Phi)(\mathbf{n})$, so that the votes are cast by a symmetric state.
  - Gate 0 (Novelty & Adoption): not evaluated.
  - Gate 1 (Conservation & Asymptotics): FAIL. Each ratio needs $\lvert G\rvert$ determinants, so one amplitude costs $\mathcal{O}(\lvert G\rvert zV)$ determinant ratios. That is affordable, but the factors then depend on the whole group orbit of $\mathbf{n}$, and the product loses even approximate separability, because a translated image of a distant region enters every factor.
  - Gate 2 (Baseline Gameability): not evaluated.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: KILLED. Projecting the whole FRD state, $P_SP_\chi\psi_\theta$, achieves the symmetry goal at the same cost and keeps the factors local. That is what the test does.
### Top Surviving Route & Sharpest Gap (Round 3)
* **Mechanism**: FRD-K, with FRD-X as the distinct sibling frame. A Slater–Jastrow reference times $\prod_e[1+\alpha_e(\mathbf{n})g(R_e)]$, where the sign capability comes only from factors that cross zero when a Hamiltonian neighbour, reached across $\Phi$'s node, outvotes $\Phi$. In the round-3 test it is used inside the same $P_SP_\chi$ projection as the control.
* **Sharpest Gap**: whether any nodal error survives symmetry projection at all at sizes where the error can be measured exactly. P1.1-P interim data say almost none survives on the $\lvert G\rvert=48$ tori. If the low-symmetry clusters also come back NO-TARGET, the P1.2 question cannot be decided at ED size. It then has to move to sampled VMC on $6\times6$ or larger, where $E_{\mathrm{sign}}$ is no longer computable, which is P1.1's own transfer gap. A secondary gap: $P_S$ is an oracle here, and a production FRD would need SU(2) projection by rotation-angle quadrature on a spin-resolved ansatz, which has the same cost problem as for NNBF.
### 24-Hour Decisive Test (Round 3)
* **Minimal Instance** (all $t=1$, $t'=-0.25$, $U=8$; ED, irrep and in-sector gap from `--mode probe`, `probe_output.txt`):

| ID | Cluster | Filling | $D$ | $\lvert G\rvert$ | $\psi_0$ irrep, $S$ | In-sector gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | $4\times3$ open × open | $4+4$ | 245025 | 4 | $M_x=M_y=+1$, $S=0$ | $0.345t$ |
| 1 | $4\times3$ periodic × open (cylinder) | $4+4$ | 245025 | 16 | $T_x=M_x=M_y=+1$, $S=0$ | $0.362t$ |
| 2 | $6\times2$ periodic × open (2-leg ladder) | $5+5$ | 627264 | 24 | $T_x=M_y=+1$, $M_x=-1$, $S=1$ | $0.259t$ |

* **Why these instances**: projection averages over 4, 16 and 24 images instead of 48, so a residual sign error is more likely to survive. Whether it does is checked by the premise rule, not assumed. All three are doped, with $t'\ne0$, and 2D or a 2-leg ladder. The test covers the **sign half only**.
* **Arms** (one Slurm array task per instance × seed × arm × SR shift; 108 tasks):
    * Decisive, trained **with** projection $\psi=P_SP_\chi\psi_\theta$: `ctrl_PS` (NNBF×Jastrow, $H=32$), `frdK_PS`, `frdX_PS`. The force is exact. The SR metric is the projected-log-derivative metric of $P_\chi\psi_\theta$ from 2000 exact draws, identical for all arms. The code is copied and generalized from `tests/P1.1P_projected/` (open-boundary mirrors, generic ansatz).
    * Diagnostic, trained without projection: `ctrl_U`, `frdK_U`, `frdX_U`. Each is budgeted unprojected (U), after post-hoc $P_\chi$ (UG) and after post-hoc $P_SP_\chi$ (UGS).
    * Seeds 1234, 2345 and 3456; a seed fixes the UHF start, so seeds pair arms. SR shifts $10^{-3}$ and $10^{-2}$, with the shift chosen per (instance, arm, seed) by variational energy only. 1500 steps and $\eta=0.05$, as in P1.1 and P1.1-P.
    * Parameters, from `--mode checks`: $4073$ / $4064$ (instance 0), $4070$ / $4054$ (instance 1), $4887$ / $4862$ (instance 2), control / FRD.
* **Scalar Metric $M$** (exact sums, P1.1 `budget()`; medians over seeds):
$$\rho_{\mathrm{sign}}(X)=\frac{\mathrm{med}\,\Delta E_{\mathrm{sign}}[X_{PS}]}{\mathrm{med}\,\Delta E_{\mathrm{sign}}[\mathrm{ctrl}_{PS}]},\quad \rho_{\mathrm{tot}}(X)=\frac{\mathrm{med}\,\Delta E_{\mathrm{tot}}[X_{PS}]}{\mathrm{med}\,\Delta E_{\mathrm{tot}}[\mathrm{ctrl}_{PS}]},\quad \rho_{\mathrm{vote}}(X)=\mathrm{med}\,\frac{\Delta E_{\mathrm{sign}}[X_{PS}]}{\Delta E_{\mathrm{sign}}[P_SP_\chi\Phi_X]},$$
  with $X\in\{\text{FRD-K},\text{FRD-X}\}$. $\Phi_X$ is the trained reference of the same state with every $\alpha_e=0$ (the vote ablation). The paired win count is the number of seeds with $\Delta E_{\mathrm{sign}}[X_{PS}]<\Delta E_{\mathrm{sign}}[\mathrm{ctrl}_{PS}]$.
* **Sampled-estimator arm** (rule 4): every task draws $8\times1000$ genuine samples from $\lvert P_\chi\psi_\theta\rvert^2$ and evaluates the estimators a sampled VMC run would use: local energies and projected log-derivatives $O_\chi$, including FRD's $1/F$ terms near factor zeros. Gated quantities:
    * $s_E$: the exact share of $\mathrm{Var}(E_{\mathrm{loc}})$ carried by the most extreme $10^{-3}$ of the $\lvert\psi\rvert^2$ weight;
    * $s_O$: the share of $\sum\lVert O_\chi-\langle O_\chi\rangle\rVert^2$ carried by the top $10^{-3}$ of the samples.

  A share is **heavy** if it exceeds $0.5$ and exceeds twice the control's value. Kurtosis and the sampled-force noise are reported only, since the noise depends on the parameterization.
* **Thresholds** (pre-committed; implemented in `aggregate()`):
    * **Premise**: an instance is informative iff $\mathrm{med}\,\Delta E_{\mathrm{sign}}[\mathrm{ctrl}_{PS}]\ge0.01t$. With fewer than 2 informative instances the route verdict is **NO-TARGET**. That means the projected control leaves too little sign error at ED size to decide the route. It is neither a PASS nor a FAIL; see the Sharpest Gap.
    * **FAIL** if, on any informative instance, $\rho_{\mathrm{sign}}\ge0.9$, or $\rho_{\mathrm{tot}}>1.1$, or $s_E$ or $s_O$ is heavy.
    * **PASS** if, on every informative instance, all of the following hold:
        * the guards hold (calibration error at most $10^{-3}\Delta E_{\mathrm{tot}}$, multi-start spread below $0.05$);
        * $\rho_{\mathrm{sign}}\le0.7$ and $\rho_{\mathrm{tot}}\le1.0$;
        * at least $\lceil2n/3\rceil$ paired wins out of $n$ seeds;
        * $\rho_{\mathrm{vote}}\le0.8$;
        * no heavy share.
    * **AMBIGUOUS** otherwise. Pre-committed action: run seeds 4567 and 5678 for the same arms and re-aggregate with `--final 1`, which applies the same rules to medians over 5 seeds and turns AMBIGUOUS into FAIL.
    * Reported, not in the verdict: the unprojected arms' $\rho_{\mathrm{sign}}$ for U, UG and UGS. A FRD advantage that is present in U but vanishes in UG and UGS is symmetry restoration; one that survives in PS is nodal repair. Also reported: the seed-to-seed min and max of every $\Delta E_{\mathrm{sign}}$, $w_{\mathrm{sign}}$ (error character), `flip_weight` and the in-sector gap.
* **Execution Plan**: folder `tests/P1.2_unanchored_antisymmetry_r3/`.
    * `micro_test.py` imports `p11_core.py`, a verbatim copy of the P1.1 core. The FRD and edge code is copied verbatim from round 2. The symmetry, spin and projected-SR code is copied from P1.1-P and generalized.
    * `submit_test.sbatch` is an array `0-107` on partition `kim`, with 8 CPUs, 16G and 24 h per task.
    * Measured cost (`--mode timing`, 8 threads on a shared compute node, `timing_output.txt`), in seconds per SR step:

| Instance | `ctrl_PS` | `frdK_PS` | `frdX_PS` | `ctrl_U` | `frdK_U` | `frdX_U` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 ($D=245025$, $\lvert G\rvert=4$) | 4.0 | 9.0 | 6.2 | 5.5 | 5.7 | 5.0 |
| 2 ($D=627264$, $\lvert G\rvert=24$) | 10.4 | 30.2 | 15.6 | 9.7 | 15.6 | 11.6 |

* **Worst task**: `frdK_PS` on instance 2, $1500\times30.2$ s, about 12.6 h, plus budgets and the sampled arm (under 30 min). That leaves a $1.9\times$ margin to the 24 h limit.
* **Command** (not run): `cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry_r3 && sbatch submit_test.sbatch`.
* **Verdict afterwards**: `python3 micro_test.py --aggregate results`.
* **Scope of the code (rule 5)**: the test measures **accuracy only**. Amplitudes are evaluated over the full basis, with $R_e$ read from the stored $\Phi$ vector. That value is identical to the per-amplitude Sherman–Morrison ratio, but the $\mathcal{O}(N_e^3+zVN_e)$ per-amplitude cost stated in round 2 is an arithmetic estimate that this test does not check, and no verdict depends on it.
* **Smoke-test evidence**: `OMP_NUM_THREADS=4 /share/apps/software/anaconda3/bin/python3 micro_test.py --mode smoke` (19 s): all round-2 and round-3 checks, then all six arms × 3 seeds on the $3\times2$ smoke cluster (40 SR steps), then the aggregate. The literal check lines and aggregate are below; per-iteration lines are removed, and the full log is in `smoke_output.txt`.

```
[13:49:33] ALL CHECKS PASSED
[13:49:33] ALL R2 CHECKS PASSED
[13:49:33] check 3x2 obc/obc |G|=4 chi={'My': -1, 'Mx': -1} S=1: [H,U_g] 1.7e-16 idem 6.4e-17 sym 0.0e+00 [P,H] 9.7e-17 <psi0|P psi0> 1.000000000000 P_S idem 1.5e-16 [P_S,P_chi] 1.5e-16 ||P_S P_chi psi0 - psi0|| 1.5e-15
[13:49:33] check 3x2 pbc/obc |G|=12 chi={'Tx': 1, 'My': 1, 'Mx': 1} S=0: [H,U_g] 2.2e-16 idem 2.2e-16 sym 3.8e-18 [P,H] 9.1e-17 <psi0|P psi0> 1.000000000000 P_S idem 5.7e-17 [P_S,P_chi] 1.9e-16 ||P_S P_chi psi0 - psi0|| 7.9e-16
[13:49:33] check 4x2 pbc/obc |G|=16 chi={'Tx': 1, 'My': 1, 'Mx': 1} S=0: [H,U_g] 2.1e-16 idem 2.4e-16 sym 1.1e-17 [P,H] 8.6e-17 <psi0|P psi0> 1.000000000000 P_S idem 1.1e-16 [P_S,P_chi] 2.2e-16 ||P_S P_chi psi0 - psi0|| 1.4e-14
[13:49:33] check projected gradient ctrl mode G: analytic -4.0402149816e-01  finite-diff -4.0402149759e-01
[13:49:33] check projected gradient ctrl mode GS: analytic -9.8589095888e-01  finite-diff -9.8589095954e-01
[13:49:33] check projected O_chi ctrl: max rel err vs finite-diff 1.5e-10
[13:49:33] check projected gradient frdK mode G: analytic 1.5285102345e-01  finite-diff 1.5285102339e-01
[13:49:33] check projected gradient frdK mode GS: analytic 3.6382273822e-01  finite-diff 3.6382273816e-01
[13:49:33] check projected O_chi frdK: max rel err vs finite-diff 1.7e-10
[13:49:33] check projected gradient frdX mode G: analytic -9.6898651377e-02  finite-diff -9.6898651325e-02
[13:49:33] check projected gradient frdX mode GS: analytic 8.7065227230e-03  finite-diff 8.7065230581e-03
[13:49:33] check projected O_chi frdX: max rel err vs finite-diff 1.3e-10
[13:49:34] check parameter match instance 0: ctrl H=32 P=4073; frdK H=47 P=4064; frdX H=47 P=4064
[13:49:34] check parameter match instance 1: ctrl H=32 P=4070; frdK H=40 P=4054; frdX H=40 P=4054
[13:49:35] check parameter match instance 2: ctrl H=32 P=4887; frdK H=55 P=4862; frdX H=55 P=4862
[13:49:35] ALL R3 CHECKS PASSED
P1.2 r3 aggregate: SIGN HALF ONLY (no same-sector competitor at the Hard-Core scale is claimed)
pre-committed: informative iff med dE_sign(ctrl_PS) >= 0.01; per route X on informative instances: FAIL if rho_sign >= 0.9 or rho_tot > 1.1 or a sampled tail share (E_loc or O) of X is > 0.5 and > 2.0x the control's; PASS if guards & rho_sign <= 0.7 & rho_tot <= 1.0 & paired sign wins >= 2/3 of seeds & rho_vote <= 0.8 & sampling ok; else AMBIGUOUS

instance 3 {'Lx': 3, 'Ly': 2, 'bcx': 'pbc', 'bcy': 'obc', 'nup': 2, 'ndn': 2, 'U': 8.0, 'tp': -0.3}  |G|=12 chi={'Tx': 1, 'My': 1, 'Mx': 1} S=0 in-sector gap=2.8871 (0.4812/site)  seeds=[1234, 2345, 3456]
   ctrl_PS   PS: dE_sign med 0.00000 [min 0.00000, max 0.00000]  dE_tot med 0.00006  w_sign 1.286e-32  flip_weight nan  P=199
   frdK_PS   PS: dE_sign med 0.00000 [min 0.00000, max 0.00000]  dE_tot med 0.00000  w_sign 3.741e-32  flip_weight 0.0000  P=186
   frdK_PS  abl: dE_sign med 0.00000 [min 0.00000, max 0.00912]  dE_tot med 0.06892  w_sign 3.557e-32  flip_weight nan  P=186
   frdX_PS   PS: dE_sign med 0.00000 [min 0.00000, max 0.00000]  dE_tot med 0.00014  w_sign 2.259e-32  flip_weight 0.0000  P=186
   frdX_PS  abl: dE_sign med 0.00000 [min 0.00000, max 0.00000]  dE_tot med 0.00829  w_sign 1.142e-32  flip_weight nan  P=186
    ctrl_U    U: dE_sign med 0.01996 [min 0.01737, max 0.02810]  dE_tot med 0.09492  w_sign 5.363e-03  flip_weight nan  P=199
    ctrl_U   UG: dE_sign med 0.00209 [min 0.00000, max 0.00516]  dE_tot med 0.01028  w_sign 3.547e-04  flip_weight nan  P=199
    ctrl_U  UGS: dE_sign med 0.00000 [min 0.00000, max 0.00516]  dE_tot med 0.00935  w_sign 1.781e-32  flip_weight nan  P=199
    frdK_U    U: dE_sign med 0.04736 [min 0.04550, max 0.04969]  dE_tot med 0.13591  w_sign 1.228e-02  flip_weight 0.0000  P=186
    frdK_U   UG: dE_sign med 0.00476 [min 0.00000, max 0.00664]  dE_tot med 0.06663  w_sign 3.531e-04  flip_weight 0.0000  P=186
    frdK_U  UGS: dE_sign med 0.00000 [min 0.00000, max 0.00664]  dE_tot med 0.06285  w_sign 2.893e-32  flip_weight 0.0000  P=186
    frdX_U    U: dE_sign med 0.04473 [min 0.04351, max 0.04933]  dE_tot med 0.19051  w_sign 1.254e-02  flip_weight 0.0000  P=186
    frdX_U   UG: dE_sign med 0.00000 [min 0.00000, max 0.00000]  dE_tot med 0.04232  w_sign 1.815e-32  flip_weight 0.0000  P=186
    frdX_U  UGS: dE_sign med 0.00000 [min 0.00000, max 0.00000]  dE_tot med 0.04140  w_sign 2.301e-32  flip_weight 0.0000  P=186
  premise: med dE_sign(ctrl_PS) = 0.00000 -> NOT informative (projected control leaves no sign error to repair)
  [frdK] rho_sign=1.000 rho_tot=0.022 paired wins 1/3 (need 2) rho_vote=0.667 tail share E_loc 0.299 (ctrl 0.040) O 0.029 (ctrl 0.017) [kurt ratio 143.29, eta ratio 1.03] guards=OK -> FAIL (instance not informative: not counted)
           diag unprojected arms: rho_sign U=2.372  post-hoc P_chi=2.281  post-hoc P_S P_chi=1.000
  [frdX] rho_sign=0.500 rho_tot=2.393 paired wins 2/3 (need 2) rho_vote=0.500 tail share E_loc 0.202 (ctrl 0.040) O 0.030 (ctrl 0.017) [kurt ratio 7.85, eta ratio 1.28] guards=OK -> FAIL (instance not informative: not counted)
           diag unprojected arms: rho_sign U=2.241  post-hoc P_chi=0.000  post-hoc P_S P_chi=0.667
ROUTE frdK: NO-TARGET (0 informative instance(s) < 2: the projected control leaves < 0.01 t sign error, so there is nothing for a sign-repair layer to fix at ED size)
ROUTE frdX: NO-TARGET (0 informative instance(s) < 2: the projected control leaves < 0.01 t sign error, so there is nothing for a sign-repair layer to fix at ED size)
```

  The smoke cluster is NO-TARGET, as it should be. The projected control is exact there ($\Delta E_{\mathrm{sign}}<10^{-5}$), which shows that the premise rule fires and that a toy cluster cannot produce a PASS. The checks confirm that the open-boundary and cylinder group operators commute with $H$ to $2\times10^{-16}$, that $P_\chi$ and $P_S$ are idempotent and commute, and that $P_SP_\chi\psi_0=\psi_0$. They also confirm that the projected-SR gradient (modes $P_\chi$ and $P_SP_\chi$) matches finite differences for all three ansätze, and that the projected log-derivatives $O_\chi$ match finite differences to $2\times10^{-10}$.
### Local Autopsy Log (Round 3 additions)
* **How each critic objection was handled**:
    * *Gate 3 cited the retracted competitor; the aggregate keyed a clause to it* (Major): **accepted, fixed**.
        * The round-2 bullets that cited it are struck in place.
        * The round-3 aggregate has no competitor clause and prints "SIGN HALF ONLY".
        * The in-sector gap of every instance is computed in code and stored in every JSON ($0.26$–$0.36t$), so no competitor is claimed.
        * The round-2 code (`p11_core.py:973`, `micro_test.py:582`) is left as the historical record; nothing in round 3 imports it.
    * *No projected arm; FRD might filter contaminant irreps* (Major): **accepted, fixed**. The decisive arms are all trained with $P_SP_\chi$, so FRD can gain nothing from symmetry restoration that the control does not also get. Every unprojected arm is also reported after post-hoc $P_\chi$ and $P_SP_\chi$, to show whether any unprojected advantage is symmetry.
    * *UHF contamination nearly decides the 4+4 verdict* (Major): **accepted, fixed** by deciding on projected arms only and reporting $w_{\mathrm{sign}}$ (error character) per arm. The P1.1 tori are dropped altogether, because the interim P1.1-P data show that the projected control nearly solves them, which would make the comparison noise.
    * *Parameters matched only on 4+4* (Minor): **accepted, fixed**. The FRD width is derived per instance, the counts are checked in code (`check parameter match`), and the residual gap is 9–25 parameters in the control's favour.
    * *Single seed* (Minor): **accepted, fixed**. Three seeds, medians, a paired win count, and a pre-committed 5-seed final rule.
    * *Size consistency overstated* (Minor): **accepted**. The first-order limit is a local-energy Jastrow, which differs from Baeriswyl at $\mathcal{O}(\lambda^2)$ (Route 3.0). The size-consistency claim is withdrawn and replaced by the weaker multiplicative-separability statement, and the novelty argument now rests only on the sign-changing nonlinear regime. A WebSearch for local-energy Jastrow prior art found none.
* **Route 3.0, the local-energy Jastrow limit $\Phi e^{-\lambda E^{\mathrm{kin}}_{\mathrm{loc}}}$**: killed at Gate 2. It is positive, so it cannot change any sign.
* **Route 3.3, projected-reference FRD**: killed at Gate 1. Building ratios from $P_\chi\Phi$ couples every factor to the group orbit and destroys locality. Projecting the whole state is equivalent in cost and keeps the factors local.
* **New anti-pattern (for the master doc)**: *testing a sign layer on clusters where the symmetry-projected control is already near exact*. On the $4\times3$ tori with $\lvert G\rvert=48$, projected NNBF at width 8 reaches $\Delta E_{\mathrm{tot}}\lesssim3\times10^{-3}$ (P1.1-P interim). A sign-repair route needs a premise check that the projected control leaves sign error.
### Round 5 (2026-11-12)
* **Boundary conditions carried in**:
    * Pilot 410216 (pre-committed rule applied): projected-control $\Delta E_{\mathrm{sign}}=6.8\times10^{-6}$, $1.1\times10^{-6}$ and $1.0\times10^{-5}$ on the three 4×3 clusters, so the round-3 test is **NO-TARGET** and is cancelled. What sets the residual is the ratio of the projected sector dimension to the parameter count (round-3 critic), not $\lvert G\rvert$.
    * The new shared instance is the 4×4 OBC $t$–$t'$–$J$ cluster with $N_\uparrow=N_\downarrow=6$, $t'=-0.25$, $J=0.4$: $D=1681680$, $\lvert G\rvert=8$, $\dim\mathrm{range}(P_SP_\chi)=30220$. There P2.1's projected NNBF of width 6 (1734 parameters) keeps $\Delta E_{\mathrm{sign}}=0.0125t$, $f_E=0.24$, overlap 0.936 (seed 11; seeds 22 and 33 run as job 451706).
    * Round-3 critic items, all binding: (a) the contrast must be NNBF×FRD against NNBF, not FRD on a Slater–Jastrow base; (b) a sampled-force SR arm with a recorded divergence and stall rate; (c) a kurtosis-ratio gate on $E_{\mathrm{loc}}$; (d) the ablation must be a separately trained control, not the co-adapted $\alpha=0$ state; (e) the per-sample local-energy cost must be measured in code.
* **The ansatz on $t$–$J$**: $\psi_\theta(\mathbf{n})=\Phi(\mathbf{n})\prod_e[1+\alpha_e(\mathbf{n})g(R_e(\mathbf{n}))]$ with $R_e=c_e\Phi(\mathbf{m}_e)/\Phi(\mathbf{n})$ and $g(R)=R/\sqrt{1+R^2}$, where now $\Phi$ is an NNBF of width 4 and $\Phi(\mathbf{m}_e)$ is the NNBF amplitude of the neighbour with its own backflow. $\alpha(\mathbf{n})$ is a separate one-hidden-layer tanh network on $x=2\mathbf{n}-1$.
    * FRD-K edges: every hopping bond (NN and NNN) times spin. In $t$–$J$ an electron of spin $\sigma$ can cross the bond only into the empty site, so each edge is a hole hop with at most one active direction; $c_e=\langle\mathbf{n}\rvert H\lvert\mathbf{m}_e\rangle=-T_{ij}\times$(Jordan–Wigner sign). On the 4×4 OBC cluster $E=84$.
    * FRD-X edges: every NN bond and orientation (up at $b$ and down at $a$, swapped); $c_e=\langle\mathbf{n}\rvert H\lvert\mathbf{m}_e\rangle/(J/2)$. $E=48$.
    * Exactness in code: $\sum_ec_e\,y(\mathbf{m}_e)$ reproduces the kinetic part of $H$ (frame K) and the exchange part (frame X) to $9\times10^{-16}$ on two clusters.
#### Generation Batch 5
* **Route 5.0 (Gate 0 re-check for FRD on $t$–$J$)**: the prior art named by the critic, searched on 2026-11-12.
    * Lanczos-step VMC for $t$–$J$: Heeb & Rice (1994), Becca, Capriotti & Sorella (statistical Lanczos, d-wave $t$–$J$), Sorella PRB 64, 024512 (2001), Becca et al. arXiv:1412.2656. These apply $(1+\alpha\hat H)^p$ with global scalar $\alpha$ to a fixed state. They are linear, not size-consistent at fixed $p$, and use no per-edge factor.
    * Lanczos steps on NNBF or other NQS: Loehr & Clark arXiv:2510.26906; Chen et al. NeurIPS 2022 (arXiv:2206.14307); arXiv:2211.07749; supervised generalized Lanczos arXiv:2502.01264. All are the same global linear step, or a step distilled into a network by fitting.
    * Neural Gutzwiller–Baeriswyl states: no paper combining a neural network with the Baeriswyl factor $e^{-\lambda\hat T}$ was found. The closest NQS for doped $t$–$J$ is G-HFDS (Lange et al., arXiv:2411.10430), a Gutzwiller-projected hidden-fermion determinant; the closest Gutzwiller–Baeriswyl work is non-neural (Hetényi et al., arXiv:1605.02698). The exact Baeriswyl factor needs a signed auxiliary sum (round-2 Route 2.4).
    * "Local-energy Jastrow" $\Phi e^{\lambda E_{\mathrm{loc}}}$: no prior art found (also searched in round 3). It is FRD's sign-inert first-order limit (Route 3.0), already killed at Gate 2.
    * *Verdict*: Gate 0 PASS for a product over Hamiltonian edges of bounded factors driven by the base's own neighbour ratios. The adoption barrier is stated as two concrete walls, each measured in this round: instability of SR through zero crossings under a sampled force (critic b), and an $\mathcal{O}(V)$ larger per-sample local energy (critic e).
* **Route 5.1 (Route 3.1 ported, Operation A.2 on critic item a), NNBF×FRD-K**:
  - Gate 0 (Novelty & Adoption): PASS (Route 5.0). The concrete mechanism against the instability wall is that $\lvert g\rvert<1$ keeps every factor finite at $\Phi$'s nodes, and the test measures the residual instability directly in a sampled-force arm.
  - Gate 1 (Conservation & Asymptotics): PASS. Hole hops conserve $N_\uparrow$, $N_\downarrow$ and the no-double-occupancy constraint (inactive edges into occupied sites). The per-sample cost is measured, not assumed (see the cost table).
  - Gate 2 (Baseline Gameability): PASS. The decisive control is a separately trained projected NNBF of width 6 with 1734 parameters; FRD-K has 1719 (base width 4, $\alpha$ width 3). The same base without FRD (NNBF width 4, 1284 parameters) is a second, separately trained arm, so a gain from re-optimised orbitals alone is visible (critic d).
  - Gate 3 (Hard-Core Shield): PASS, **sign half only**. Doped ($\delta=1/4$), $t'\ne0$, 2D and non-stoquastic; no same-sector competitor is claimed.
  - *Verdict*: SURVIVED (conditional on the decisive test).
* **Route 5.2 (Route 3.2 ported), NNBF×FRD-X**: the same, with spin-exchange edges. In $t$–$J$ the Schrieffer–Wolff frame of round 2 is no longer an approximation: the exchange term is in $H$ itself.
  - Gate 0 (Novelty & Adoption): PASS (Route 5.0).
  - Gate 1 (Conservation & Asymptotics): PASS. Exchange conserves $N_\uparrow$, $N_\downarrow$ and the hole positions.
  - Gate 2 (Baseline Gameability): PASS. 1656 parameters (base width 4, $\alpha$ width 4) against 1734 for the control, a 78-parameter gap in the control's favour.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - *Verdict*: SURVIVED (conditional on the decisive test).
* **Route 5.3 (FRD on a Slater–Jastrow base against an NNBF control, the round-3 design)**: killed by critic item (a). It confounds the sign layer with the absence of backflow, so a FAIL says nothing about FRD and a PASS is implausible by construction. Not ported.
* **Route 5.4 (Operation A.1 on the cost wall), frozen-frame FRD (FRD-F)**: take each ratio from the base's backflowed orbitals frozen at $\mathbf{n}$, $R_e=c_e\det\Phi_{\mathbf{n}}[\mathrm{occ}(\mathbf{m}_e)]/\det\Phi_{\mathbf{n}}[\mathrm{occ}(\mathbf{n})]$, which is one Sherman–Morrison ratio, $\mathcal{O}(N_\sigma)$, from the inverse that NNBF already forms. An amplitude then costs one NNBF pass plus $\mathcal{O}(EN_\sigma)$, and a local energy costs the same number of NNBF passes as NNBF itself. On a Slater base it equals FRD exactly.
  - Gate 0 (Novelty & Adoption): PASS on the same search; it targets exactly the measured cost wall.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): not evaluated. The vote is cast by $\Phi_{\mathbf{n}}$'s mean-field picture of the neighbour, not by the neighbour's own amplitude, so it may lose the information that makes FRD a vote.
  - Gate 3 (Hard-Core Shield): not evaluated.
  - *Verdict*: HELD (pre-committed): it is built and tested only if Route 5.1 or 5.2 passes accuracy. If both fail accuracy, the cheaper variant cannot help, because it can only lose information.
### Top Surviving Route & Sharpest Gap (Round 5)
* **Mechanism**: NNBF×FRD-K, with NNBF×FRD-X as the sibling frame. The base NNBF moves the nodes by deforming orbitals; the FRD factors add a per-configuration sign flip wherever a Hamiltonian neighbour, reached across the base's node, outvotes the base with the weight the network assigns.
* **Sharpest Gap**: cost. Measured with every amplitude evaluated per sample (no shortcuts), one FRD local energy needs 16–47 times as many NNBF forward passes as an NNBF local energy at $L=4$–$7$, and the ratio grows as $N_e^{0.9}$ (slopes 1.83–1.98 against 0.96). The wall-time ratio grows from 1.2–1.5× at $L=4$ to 6.3–7.4× at $L=7$. So even an accuracy PASS on the 4×4 instance only buys a matched-*parameter* advantage. At matched *compute* the control could be about $V$ times larger, and FRD-F (Route 5.4) is then the required next step. A second gap: the instance covers the sign half only.
### 24-Hour Decisive Test (Round 5)
* **Minimal Instance**: instance 0 of P2.1: 4×4 open × open $t$–$t'$–$J$, $N_\uparrow=N_\downarrow=6$, $t=1$, $t'=-0.25$, $J=0.4$; $D=1681680$, $\lvert G\rvert=8$ ($M_x=M_y=+1$, $D=-1$), non-degenerate singlet, $\dim\mathrm{range}(P_SP_\chi)=30220$. The code is reused, not rewritten: `tj_core.py`, `p11_core.py` and `p21_harness.py` are verbatim copies of P2.1's round-3 files.
* **Arms** (all trained with $P_\chi$ projection by P2.1's exact-force projected SR: 1200 steps, step 0.05, 2000 metric samples):
    * `nnbf6`: NNBF width 6, 1734 parameters. Decisive control, identical to P2.1's projected `nnbf` arm.
    * `nnbf4`: NNBF width 4, 1284 parameters. The same base, separately trained (critic d).
    * `frdK`: NNBF-4 × FRD-K, 1719 parameters. `frdX`: NNBF-4 × FRD-X, 1656 parameters.
    * Seeds 11, 22, 33 and SR shifts $10^{-3}$, $10^{-2}$ per arm; per (arm, seed) the shift with the lower final variational energy is kept.
    * Sampled-force arms (critic b): `nnbf6_mc`, `frdK_mc`, `frdX_mc` on seed 11, shift $10^{-2}$. Force and metric come only from Metropolis samples of $\lvert P_\chi\psi_\theta\rvert^2$: 100 persistent chains, bond-swap proposals, $N_e$ proposals per measurement, and $f=\langle(E_{\mathrm{loc}}-\bar E_{\mathrm{loc}})(O_P-\bar O_P)\rangle$. The final state is the last iterate. Recorded: non-finite flag, divergence (final exact energy above the step-0 energy), jump rate (fraction of steps whose exact energy rises by more than $0.01t$), and stall ratio $\Delta E_{\mathrm{tot}}(X_{\mathrm{mc}})/\Delta E_{\mathrm{tot}}(X,\text{exact force, same seed and shift})$.
* **Scalar Metric $M$** (P1.1 `budget()` on exact sums, medians over seeds):
$$\rho_{\mathrm{sign}}(X)=\frac{\mathrm{med}\,\Delta E_{\mathrm{sign}}[X]}{\mathrm{med}\,\Delta E_{\mathrm{sign}}[\mathrm{nnbf6}]},\quad \rho_{\mathrm{tot}}(X)=\frac{\mathrm{med}\,\Delta E_{\mathrm{tot}}[X]}{\mathrm{med}\,\Delta E_{\mathrm{tot}}[\mathrm{nnbf6}]},\quad \rho_{\mathrm{abl}}(X)=\frac{\mathrm{med}\,\Delta E_{\mathrm{sign}}[X]}{\mathrm{med}\,\Delta E_{\mathrm{sign}}[\mathrm{nnbf4}]},\quad \kappa(X)=\frac{\mathrm{med}\,\mathrm{kurt}\,E_{\mathrm{loc}}[X]}{\mathrm{med}\,\mathrm{kurt}\,E_{\mathrm{loc}}[\mathrm{nnbf6}]},$$
  where $\mathrm{kurt}\,E_{\mathrm{loc}}=\sum p(E_{\mathrm{loc}}-E)^4/[\sum p(E_{\mathrm{loc}}-E)^2]^2$ is exact under $p=\lvert P_\chi\psi\rvert^2$.
* **Thresholds** (pre-committed; the logic of `aggregate()`):
    * **Premise**: the test counts only if $\mathrm{med}\,\Delta E_{\mathrm{sign}}[\mathrm{nnbf6}]\ge0.005t$ over 3 seeds. Otherwise both routes are **NO-TARGET**.
    * **Spread-aware margin**: the seed ranges must be disjoint, $\max_s\Delta E_{\mathrm{sign}}[X]<\min_s\Delta E_{\mathrm{sign}}[\mathrm{nnbf6}]$.
    * **Sampled-force gate**: MC_OK iff $X_{\mathrm{mc}}$ is finite and not diverged, its stall ratio is at most $2\times$ the control's and its jump rate at most $\max(2\times\text{control},0.02)$. MC_FAIL iff non-finite or diverged, stall above $4\times$ the control's, or jump rate above $\max(5\times\text{control},0.05)$.
    * **FAIL** if $\rho_{\mathrm{sign}}\ge0.9$, or $\rho_{\mathrm{tot}}>1.1$, or $\kappa>10$ (critic c), or MC_FAIL.
    * **PASS** if the guards hold (calibration at most $10^{-3}\Delta E_{\mathrm{tot}}$, budget spread below 0.05, 3 seeds per arm), $\rho_{\mathrm{sign}}\le0.7$, the ranges are disjoint, $\rho_{\mathrm{tot}}\le1.0$, $\rho_{\mathrm{abl}}\le0.8$, $\kappa\le10$ and MC_OK.
    * **AMBIGUOUS** otherwise. Pre-committed action: seeds 44 and 55 for `nnbf6` and $X$ (both shifts), then `--aggregate --final 1`: PASS iff $\rho_{\mathrm{sign}}\le0.7$ on 5-seed medians, at least 4 of 5 seeds below the control's median, $\rho_{\mathrm{tot}}\le1.0$, $\kappa\le10$ and MC_OK; otherwise FAIL.
    * Reported, not in the verdict: post-hoc $P_SP_\chi$ budgets, $w_{\mathrm{sign}}$, overlap, `flip_weight`, the co-adapted $\alpha=0$ energy (diagnostic only), and the cost table.
* **Cost measurement (critic e; `--mode cost`, one thread, `cost_output.txt`)**: every amplitude is a full NNBF forward pass on its own occupation, for both families. $L\times L$ periodic $t$–$J$, doping about $1/8$, random weights, 3 configurations per size.

| $L$ | $N_e$ | NNBF passes per $E_{\mathrm{loc}}$ | FRD-K passes | FRD-X passes | $T_{\mathrm{FRD\text{-}K}}/T_{\mathrm{NNBF}}$ | $T_{\mathrm{FRD\text{-}X}}/T_{\mathrm{NNBF}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4 | 14 | 29 | 477 | 387 | 1.5 | 1.2 |
| 5 | 22 | 44 | 1034 | 946 | 2.7 | 2.3 |
| 6 | 32 | 57 | 1552 | 1847 | 3.2 | 3.4 |
| 7 | 42 | 87 | 4105 | 3510 | 7.4 | 6.3 |

  Log-log slopes in $N_e$: passes 0.96 (NNBF), 1.83 (FRD-K), 1.98 (FRD-X); time 0.83, 2.17, 2.25. The NNBF times are overhead-dominated at these sizes, so the pass counts are the asymptotic statement: FRD's local energy costs $(1+E_{\mathrm{act}})$ times NNBF's, with $E_{\mathrm{act}}\propto V$ active edges. This confirms the critic's $\mathcal{O}((zV)^2N_e)$-type wall for the honest algorithm.
* **Execution Plan**: folder `tests/P1.2_unanchored_antisymmetry_r5/`.
    * `micro_test.py` (FRD on an NNBF base, edges, sampled-force SR, aggregate, cost), importing the three verbatim copies.
    * `submit_test.sbatch`: array `0-26`, partition `kim`, 16 CPUs, 48G, 24 h.
    * Measured cost on instance 0 (`--mode time`, 8 threads on a shared kim node, `timing_output.txt`), seconds per SR step: `nnbf6` 12.98, `nnbf4` 12.32, `frdK` 37.57, `frdX` 24.58, `frdK_mc` 20.2 including chain set-up, 15.7 per later step (the sampled-force arms skip the full-basis force). Edge tables take 12–16 s once. The worst task (`frdK`, $1200\times37.6$ s, about 12.5 h, plus two budgets of about 6 min each and the sampled diagnostics) leaves a $1.9\times$ margin to 24 h at 8 threads; the array asks for 16.
    * Command (not run): `cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry_r5 && sbatch submit_test.sbatch`. Verdict afterwards: `python3 micro_test.py --mode aggregate`.
    * Submission can proceed before job 451706 finishes: the premise gate is evaluated on this test's own `nnbf6` arm, which is identical to P2.1's.
* **Scope of the code (rule 5)**: training uses full-basis amplitudes with $\Phi(\mathbf{m}_e)$ read from the stored base vector. The per-amplitude path (`FRD.psi_bits`, a base forward pass at each configuration and each active neighbour) agrees with it to $2\times10^{-16}$ and is what the cost table times. The cost claim in this section is the measured one.
* **Smoke-test evidence**: `OMP_NUM_THREADS=4 /share/apps/software/anaconda3/bin/python3 micro_test.py --mode smoke` (32 s; full log in `smoke_output.txt`). It runs the checks, all 27 tasks on the 4×2 OBC $3+3$ cluster ($D=560$, base width 2, control width 4, 30 SR steps), the aggregate, and a two-size cost run. The literal check lines and aggregate:

```
[11:14:13] check inst 9 D=90 frame K: E=30 edges, max|sum_e c_e y(m_e) - H_K y| = 8.9e-16
[11:14:13] check inst 9 D=90 frame X: E=18 edges, max|sum_e c_e y(m_e) - H_X y| = 5.6e-17
[11:14:13] check inst 8 D=560 frame K: E=32 edges, max|sum_e c_e y(m_e) - H_K y| = 8.9e-16
[11:14:13] check inst 8 D=560 frame X: E=20 edges, max|sum_e c_e y(m_e) - H_X y| = 2.2e-16
[11:14:13] check FRD-K grad proj=none: analytic 1.1368320329e+00 fd 1.1368320324e+00 (rel 4.3e-10); O vs fd 1.5e-08
[11:14:13] check FRD-K grad proj=chi: analytic 4.5892863467e+00 fd 4.5892863456e+00 (rel 2.4e-10); per-amplitude psi_P 1.3e-16; O vs fd 5.9e-10
[11:14:13] check FRD-K per-sample E_loc == (H psi)/psi on all 90 configs: 1.8e-15 (NNBF per-sample path 4.6e-15); flip_weight 0.781; base evals per E_loc mean 102.6
[11:14:13] check FRD-X grad proj=none: analytic -9.9104166334e-02 fd -9.9104165985e-02 (rel 3.5e-10); O vs fd 2.2e-10
[11:14:13] check FRD-X grad proj=chi: analytic 5.5831513214e-01 fd 5.5831513190e-01 (rel 2.4e-10); per-amplitude psi_P 1.7e-16; O vs fd 1.7e-10
[11:14:14] check FRD-X per-sample E_loc == (H psi)/psi on all 90 configs: 9.5e-16 (NNBF per-sample path 4.6e-15); flip_weight 0.004; base evals per E_loc mean 39.4
[11:14:25] check sampled force (inst 8 D=560): exact expectation of the sampled estimator vs force 5.8e-16; 40000 Metropolis samples rel err 0.140 vs iid-predicted 8.798 (acceptance 0.159)
[11:14:25] check parameter match inst 0: nnbf6 1734, nnbf4 1284, frdK (base 4, alpha width 3, E=84) 1719
[11:14:25] check parameter match inst 0: nnbf6 1734, nnbf4 1284, frdX (base 4, alpha width 4, E=48) 1656
[11:14:25] ALL R5 CHECKS PASSED
P1.2 r5 aggregate: SIGN HALF ONLY (no same-sector competitor is claimed)
pre-committed: premise med dE_sign(nnbf6) >= 0.005; FAIL if rho_sign >= 0.9 or rho_tot > 1.1 or kurt ratio > 10.0 or MC_FAIL; PASS if guards & rho_sign <= 0.7 & seed ranges disjoint & rho_tot <= 1.0 & rho_abl <= 0.8 & kurt ratio <= 10.0 & MC_OK

instance 8 {'Lx': 4, 'Ly': 2, 'bcx': 'obc', 'bcy': 'obc', 'nup': 3, 'ndn': 3, 'tp': -0.25, 'J': 0.4}
  nnbf6  seeds [11, 22, 33] dE_sign med 0.00116 [0.00015,0.00584] dE_tot med 0.03908 w_sign 1.04e-03 ov0 0.889 kurt 84.8 | P_S: dE_sign 0.00208
  nnbf4  seeds [11, 22, 33] dE_sign med 0.00581 [0.00017,0.00679] dE_tot med 0.06103 w_sign 4.40e-03 ov0 0.931 kurt 524 | P_S: dE_sign 0.00308
  frdK   seeds [11, 22, 33] dE_sign med 0.00323 [0.00014,0.00453] dE_tot med 0.02089 w_sign 3.53e-03 ov0 0.938 kurt 322 | P_S: dE_sign 0.00101 flip_w 0.00e+00
  frdX   seeds [11, 22, 33] dE_sign med 0.00573 [0.00211,0.02564] dE_tot med 0.04385 w_sign 6.06e-03 ov0 0.872 kurt 2.15e+03 | P_S: dE_sign 0.00541 flip_w 0.00e+00
  premise: med dE_sign(nnbf6) = 0.00116 over 3 seeds -> NOT informative
  [frdK] rho_sign=2.783 rho_tot=0.535 rho_abl=0.556 disjoint=False wins 1/3 kurt ratio 3.79 guards=OK MC stall 6.41 (ctrl 5.28) jumps 0.000 (ctrl 0.000) diverged False -> OK -> FAIL (not counted)
ROUTE frdK (instance 8): NO-TARGET
  [frdX] rho_sign=4.934 rho_tot=1.122 rho_abl=0.985 disjoint=False wins 0/3 kurt ratio 25.36 guards=OK MC stall 4.60 (ctrl 5.28) jumps 0.000 (ctrl 0.000) diverged False -> OK -> FAIL (not counted)
ROUTE frdX (instance 8): NO-TARGET
[11:14:44] cost L=4 V=16 Ne=14: NNBF 30 evals 6.603e-03 s | FRD-K 494 evals 1.270e-02 s (x1.9) | FRD-X 431 evals 1.113e-02 s (x1.7)
[11:14:44] cost L=5 V=25 Ne=22: NNBF 43 evals 1.017e-02 s | FRD-K 993 evals 2.477e-02 s (x2.4) | FRD-X 905 evals 2.016e-02 s (x2.0)
[11:14:44] cost log-log slopes in N_e: T_nnbf=0.96 T_frdK=1.48 T_frdX=1.31 ev_nnbf=0.80 ev_frdK=1.54 ev_frdX=1.64
[11:14:44] smoke total 31.5 s
```

  The checks show three things. The edges reproduce the kinetic and exchange parts of $H$ exactly, and FRD's exact force, projected log-derivatives and per-amplitude $\psi_P$ match finite differences and the vector path. The honest per-sample local energy equals $(H\psi)/\psi$ on every configuration. The sampled-force estimator is unbiased: its exact expectation equals the force to $6\times10^{-16}$. Its error at 40000 samples (0.14) sits far below the iid prediction (8.8), because the predicted variance is dominated by rare near-node configurations: that heavy tail is the instability the `_mc` arms measure. The smoke cluster is NO-TARGET, as it should be: its projected sector has dimension 44.
### Local Autopsy Log (Round 5 additions)
* **How each round-3 critic objection was handled**:
    * *Premise likely fails; sector-dimension/parameter ratio matters* (Major): **confirmed** by pilot 410216 (NO-TARGET). The test moves to the instance with ratio $30220/1734\approx17$, and its premise gate ($0.005t$, the brief's value) is computed from its own control.
    * *FRD vs control confounded by the base* (Major): **fixed**. FRD now sits on an NNBF base; the decisive contrast is NNBF-4×FRD against NNBF-6 at matched parameters (gap 15 and 78 parameters, in the control's favour). Route 5.3 records the old design as killed.
    * *Adoption barrier (zero crossings under a sampled force) never tested* (Major): **fixed**. Three sampled-force SR arms with pre-committed stall and jump gates.
    * *Heavy-tail gate cannot fire; kurtosis $143\times$ the control's passed* (Major): **fixed**. The gate is now $\kappa>10\Rightarrow$ FAIL. Note that P2.1's projected control already has $\mathrm{kurt}\,E_{\mathrm{loc}}\approx4\times10^5$ on this instance, so the gate is relative.
    * *Novelty only partial; real barrier is the per-sample cost* (Major): **accepted and measured**. The cost wall is real (table above); FRD-F (Route 5.4) is the pre-committed mutation against it.
    * *Co-adapted $\alpha=0$ is not an ablation* (critic d): **fixed**. `nnbf4` is trained separately; the co-adapted $\alpha=0$ energy is reported as a diagnostic only.
* **Route 5.3, FRD on a Slater–Jastrow base against NNBF**: killed (confounded contrast).
* **New anti-pattern (for the master doc)**: *timing a ratio-based sign layer from a stored full-basis vector*. On ED instances every neighbour amplitude is a free lookup, which hides an $\mathcal{O}(V)$ factor in the per-sample local energy.
### Round 6 — Evidence Tiers & Route Portfolio (2026-11-16)
* **Boundary conditions carried in**:
    * Round-5 result (job 451916, 3 seeds): FRD-K is **AMBIGUOUS** with $\rho_{\mathrm{sign}}=0.84$, $\rho_{\mathrm{tot}}=0.92$, $\rho_{\mathrm{abl}}=0.56$, 3 of 3 paired wins, disjoint seed ranges, $\kappa=3.7$ and MC_OK. FRD-X is **FAIL** with $\rho_{\mathrm{sign}}=1.29$, $\rho_{\mathrm{tot}}=1.24$ and $\kappa=11.9$. The pre-committed AMBIGUOUS action (seeds 44 and 55 for `nnbf6` and `frdK`) runs as job 553724.
    * Read from the stored round-5 JSONs (no new computation):
        * FRD-K's `flip_weight` is small. The median over its six runs is $6\times10^{-4}$ and the maximum is $3.7\times10^{-3}$, under $\lvert P_\chi\psi\rvert^2$.
        * After post-hoc $P_S$ the advantage remains: median $\Delta E_{\mathrm{sign}}$ is $0.0052$ for FRD-K against $0.0068$ for `nnbf6` (ratio about $0.77$). So the gain is not spin restoration.
        * Whether the gain is carried by the few flips (the vote) or by amplitude off-loading under projection is the open mechanism question. Test T2-a below decides it.
    * The skill now uses tiers T0 to T3, Gate 4, a route portfolio, submit-and-switch scheduling and pre-flight dry runs. Test code for this round is in `tests/P1.2_unanchored_antisymmetry_r6/` and reuses the round-5 modules by import.
#### Generation Batch 6 (gated with Gates 0–4)
* **Route 6.1, FRD-F (Route 5.4 re-gated: frozen-frame ratios)**. Each ratio is taken from the base's backflowed orbitals frozen at $\mathbf{n}$, $R^F_e=c_e\det O_{\mathbf{n}}[\mathrm{rows}(\mathbf{m}_e)]/\det O_{\mathbf{n}}[\mathrm{rows}(\mathbf{n})]$. That is one Sherman–Morrison ratio per edge, with no NNBF pass at the neighbour.
  - Gate 0 (Novelty & Adoption): PASS. A WebSearch on 2026-11-16 for frozen backflow orbitals with Sherman–Morrison local-energy shortcuts found low-rank updates for local moves in SCALE (arXiv:2604.25775) and nothing that uses a frozen frame inside a sign factor. It targets FRD's measured adoption wall, the cost.
  - Gate 1 (Conservation & Asymptotics): PASS. Same moves and conservation laws as FRD-K.
  - Gate 2 (Baseline Gameability): PASS as a design. The control is `nnbf6`, and the read-out below compares it with FRD-K at the same parameters.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): PASS on paper. A local energy needs the same $\mathcal{O}(zN_e)$ NNBF passes as NNBF itself, plus $\mathcal{O}(E_{\mathrm{act}}N_\sigma)$ ratio work per amplitude, which is subleading next to an NNBF pass at $\mathcal{O}(VN_eH+N_e^3)$. Measured at T1 below.
  - *Verdict*: SURVIVED. It enters T1.
* **Route 6.2, FRD-H (Operation A on the FRD-X failure)**. The FRD-X failure mechanism to fix is that its votes come from the exchange term alone ($J/2=0.2$), which is the minority part of the eigen-equation $\psi_0(\mathbf{n})(E_0-H_{\mathbf{n}\mathbf{n}})=\sum_{\mathbf{m}}H_{\mathbf{n}\mathbf{m}}\psi_0(\mathbf{m})$. The mutation puts FRD factors on every off-diagonal move, $K\cup X$ (132 edges on 4×4), so the vote is cast by the full eigen-equation.
  - Gate 0 (Novelty & Adoption): PASS (the round-5 FRD audit, Route 5.0).
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS (same control design).
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): FAIL as-is. It has FRD-K's $\mathcal{O}(V)$ cost wall, with $1.6\times$ more edges. It is admitted to T1 only because its first question is whether the extra edges carry information at all, and a frozen version would repair the cost.
  - *Verdict*: SURVIVED, conditional. It enters T1 with pre-registered rule R2.
* **Route 6.3, AO-F (Operation A.3 on the round-3 Route 3.0 kill: amplitude off-loading)**. $\psi=\Phi_{\mathrm{NNBF}}(\mathbf{n})\exp\big(\sum_e\alpha_e(\mathbf{n})g(R^F_e(\mathbf{n}))\big)$, the sign-inert local-energy Jastrow on frozen ratios. Route 3.0 was killed at Gate 2 because a positive factor cannot change $\mathrm{sgn}\,\psi_\theta$. The higher-order witness that Operation A.3 asks for is the **projected** sign. Since $\mathrm{sgn}(P_\chi\psi)(\mathbf{n})=\mathrm{sgn}\sum_g\chi_g\psi(g\mathbf{n})$ depends on image amplitudes, a positive factor can move the projected nodes, and it can also free backflow capacity for the nodes.
  - Gate 0 (Novelty & Adoption): PASS. No local-energy Jastrow was found in the round-3 and round-5 searches, nor in a WebSearch on 2026-11-16 for Jastrow factors that depend on the local energy (the hits are variance minimization and cusp conditions).
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS only with a second control. The cheap baseline is NNBF-4 times a neural Jastrow $\exp(\mathrm{MLP}(\mathbf{x}))$ at matched parameters. It is written into the T3 design below.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): PASS on paper. It costs the same as FRD-F, and a log-additive local correlator gives a per-site, extensive change.
  - *Verdict*: SURVIVED. Blocked on T2-a (job 697584, task 0).
* **Route 6.4, complex detour (Operation B.1, real to complex frame)**. Train a complex backflow $\psi_c=A+iB$ and read out $\mathrm{Re}\,\psi_c$. A sign then changes by phase rotation, without $\lvert\psi_c\rvert$ crossing zero.
  - Gate 0 (Novelty & Adoption): unclear. A WebSearch on 2026-11-16 found complex networks used for Hubbard sign problems (arXiv:2203.00390), modulus–phase networks (Szabó & Castelnovo, arXiv:2002.04613), real-part arguments for real Hamiltonians (arXiv:2302.02523) and a real backflow choice justified by time reversal (arXiv:2601.21310). No complex backflow with real-part readout for $t$–$J$ was found.
  - **T0 kill (Gate 2)**. For a real symmetric $H$, $\langle A\rvert H\lvert B\rangle=\langle B\rvert H\lvert A\rangle$, so
$$E[\psi_c]=\frac{\langle A\rvert H\lvert A\rangle+\langle B\rvert H\lvert B\rangle}{\langle A\vert A\rangle+\langle B\vert B\rangle},$$
    a weighted average of $E[A]$ and $E[B]$. The objective never couples the two parts except through shared parameters, and nothing makes the readout $A$ the better part. A node of $A$ still has to cross zero in $A$'s own landscape. *Verdict*: KILLED at T0.
* **Route 6.5, holon-frame backflow (Operation B.1, electron to holon frame)**. $\psi=M(\mathbf{n})\det X_\theta(\mathbf{n})_{[\mathrm{holes}]}A(\mathbf{n})$, with spin-dependent holon backflow. This is the round-1 Route 3 with a configuration-dependent $X$.
  - **T0 kill (Gate 3)**. With full spin-dependent backflow the family is universal, so it reduces to "anchor quality". The round-1 measurement already decides that: the best Marshall-times-hole-function anchor has $W=0.34$, against $0.073$ for the foil Slater on $3\times4$ at $J/t=0.4$. The anchor is good only as $J/t\to\infty$ (Anti-Solution Space (3)). *Verdict*: KILLED at T0.
* **Route 6.6, node-targeted sampling (Operation B.2, active probing)**. Bias the sampler toward configurations near the nodes, where sign gradients live, then reweight.
  - **T0 kill (Gate 1)**. Every test in this leaf uses an exact-sum force, so the sampler cannot matter there. In sampled SR the fixed point does not depend on the sampler, which is the P1.3 kill (sign-flip-aware Metropolis). *Verdict*: KILLED at T0 (repeats a logged anti-pattern).
### Route Portfolio & Sharpest Gaps

*Superseded by the critic's portfolio table in "Round 6 per-leaf critic (2026-11-16)" below.* The worker's table, kept as the record:

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FRD-K (5.1) | T3 | IN-FLIGHT(553724); vote mechanism KILLED at T2-a (697584) | seeds 44, 55 | `--final 1`: PASS iff 5-seed $\rho_{\mathrm{sign}}\le0.7$, at least 4 of 5 seeds below the control median, $\rho_{\mathrm{tot}}\le1.0$, $\kappa\le10$, MC_OK; else FAIL | Decision tree D-K (VOTE-INERT branches) |
| AO-F (6.3) | T0 (T2-a evidence for its premise) | LIVE, lead route | 553724 (decides A-C or not) | A-C (T3): PASS iff 5-seed $\rho_{\mathrm{sign}}\le0.7$ against `nnbf6` AND below `nnbfJ` on at least 4 of 5 seeds, $\rho_{\mathrm{tot}}\le1.0$; else FAIL | Decision tree D-A below |
| FRD-F (6.1) | T2 | KILLED(T2-b, $u=1.50$) | — | — | Logged |
| FRD-H (6.2) | T1 | KILLED(T1, recall rule R2) | — | — | Logged; see Tier log |

* **Live routes after job 697584**: 2 (FRD-K as an accuracy data point only, and AO-F). The portfolio is **below 3**. The refill attempts made after the result (Routes 6.7 and 6.8 below) were both killed at T0, so one refill is still owed to the next round. Before 697584 returned, all three live routes (FRD-K, FRD-F, AO-F) were in flight or blocked on it, which made the wait legitimate (Stage 3.5.3, item 5).
* **T2 result (job 697584, both tasks valid: $E_0$ and identity energy reproduced; `t2_aggregate_output.txt`)**:

```
[09:49:27] abs: E=-11.34112278 dE_sign=0.009757 dE_tot=0.046888 (692 s)
[09:50:31] lift: lift_K_U=6.299 recall_K_U=0.347 lift_X_U=3.009 recall_X_U=0.425 lift_H_U=7.107 recall_H_U=0.380 w_b_U=0.118 lift_K_G=128.614 recall_K_G=0.478 lift_X_G=4.219 recall_X_G=0.434 lift_H_G=148.013 recall_H_G=0.474 w_b_G=0.006 (64 s)
[09:50:31] frozen: E=-11.28973375 dE_sign=0.013202 dE_tot=0.098277 (756 s)
task0: {"valid": true, "gap": 0.002294020110413797, "v_abs": 0.0025581980809512767, "T2a_abs": "VOTE-INERT", "gate4_lift": "EXTENSIVE-OK"}
task1: {"valid": true, "gap": 0.002294020110413797, "u_frozen": 1.504108664365367, "T2b": "KILLED"}
```

* **What the T2 result decides**:
    * **T2-a: VOTE-INERT ($v=0.003$)**. Removing every vote flip from the trained FRD-K state changes $\Delta E_{\mathrm{sign}}$ from $0.009750$ to $0.009757$. None of FRD-K's sign gain over the control comes from the flips. FRD-K's novelty claim (a sign-capable vote) is **falsified at T2**, whatever job 553724 returns. Its gain is an amplitude effect, from off-loading and a co-adapted base. Under the pre-registered tree, AO-F becomes the lead route.
    * **T2-b: FRD-F KILLED ($u=1.50$)**. The frozen-frame read-out loses the whole gain and more ($\Delta E_{\mathrm{sign}}=0.0132$, above the control's $0.0120$), and it doubles $\Delta E_{\mathrm{tot}}$. Frozen ratios are not a drop-in replacement. Since the vote is inert, FRD-F has no remaining rationale.
    * **Gate 4 lift at $V=16$: EXTENSIVE-OK ($\mathrm{lift}_K=6.3$ on the trained base, unprojected)**. The vote *information* grows with size: the lifts were 2.9, 1.3 and 2.0 for the UHF anchor at $V=8$ to $12$. On the projected trained base it reaches $129$, with recall $0.48$. So the vote is informative but **unused by training**: SR never pushes $\alpha_eg$ below $-1$. Barrier (b), the zero crossing, is what blocks it, not missing information.
#### FRD-K
* **Mechanism**: NNBF-4 times $\prod_e[1+\alpha_e(\mathbf{n})g(R_e)]$ on hole-hop edges. A factor flips the sign where a Hamiltonian neighbour, reached across the base's node, outvotes the base.
* **Sharpest Gap**: is the 3-seed gain ($\rho_{\mathrm{sign}}=0.84$) carried by the vote flips or by amplitude off-loading under projection? The trained states flip only $6\times10^{-4}$ (median) of the projected weight. **Answered by T2-a: amplitude off-loading.** Removing the flips leaves $\Delta E_{\mathrm{sign}}$ unchanged ($v=0.003$).
* **Gate 4 claim (retroactive, T0)**:
    * *Cost per sample*: a local energy needs $(1+n_{\mathrm{conn}})(1+E_{\mathrm{act}})$ NNBF passes, against $(1+n_{\mathrm{conn}})$ for NNBF, where $n_{\mathrm{conn}}\propto zN_e$ and $E_{\mathrm{act}}\propto V$. Measured again this round (T1, part B, one thread): the pass ratio is $522/32$ at $L=4$ and $1867/62$ at $L=6$, the wall-time ratio goes from $1.9\times$ to $3.5\times$, and its log-log slope in $N_e$ is $0.76$ (round 5: wall-time slope 2.17 against 0.83 for NNBF). **The cost claim FAILS Gate 4 as written**: the overhead grows as $\mathcal{O}(V)$. At matched compute the control could be about $V$ times larger, so a matched-parameter win cannot transfer. The only rescue is FRD-F.
    * *Extensivity of the gain*: possible on paper. The factors are multiplicatively separable, so independent local repairs add per site. Against it, the flip weight at $V=16$ is about $10^{-3}$, a fixed absolute weight rather than a per-site density. **T1 trend (`t1_vote.py`)**: for the UHF anchor, the hole-hop vote lift is $2.90$, $1.27$, $2.05$ at $V=8,10,12$, which is not monotone ($V=10$ has a near-degenerate gap of $0.004t$). Rule R1 gives UNDECIDED. The trained-base lift at $V=16$ comes from T2-a.
    * *Gate 4 verdict for FRD-K*: **FAIL on cost as-is, extensivity UNDECIDED**. FRD-K can at most be a parent: its accuracy result is useful only if FRD-F (or AO-F) inherits it at NNBF cost.
* **Decision tree D-K** (job 553724, then `micro_test.py --mode aggregate --results results --final 1` in `tests/P1.2_unanchored_antisymmetry_r5/`):
    * **PASS**:
        * If T2-a says VOTE-CARRIES and T2-b says DROP-IN-PASS: submit **T3 matched-compute test M-C**. The arms are `frdF` (FRD-F at the trained FRD-K widths, trained from scratch) and `nnbfC`, an NNBF whose width is chosen so that its measured per-sample $E_{\mathrm{loc}}$ time equals FRD-F's (about $2.2\times$ NNBF-4's, measured with `t1_frozen.py --parts B` at $L=4$ before submission). Each arm gets 5 seeds and both SR shifts, with the round-5 harness on instance 0. The rules reuse the round-5 constants with `nnbfC` as the control: PASS iff $\rho_{\mathrm{sign}}\le0.7$, at least 4 of 5 wins and $\rho_{\mathrm{tot}}\le1.0$. Power check first: the round-5 seed spread of `nnbf6` (0.0118–0.0136) against a gap of 0.0023 gives about $2.5\sigma$ at 5 seeds.
        * If T2-a says VOTE-CARRIES and T2-b says AMBIGUOUS: M-C as above, with `frdF` trained (its own T3).
        * If T2-a says VOTE-INERT: the PASS is credited to amplitude off-loading, not to the vote. FRD-K and FRD-F lose their novelty claim. Submit **T3 test A-C**: `aoF` (AO-F, base width 4, $\alpha$ width matched to 1734 parameters) against `nnbf6` and `nnbfJ` (NNBF-4 times a neural Jastrow of matched parameter count), 5 seeds, the same rules with `nnbfJ` as an extra must-beat control.
        * If T2-a says MIXED: run both M-C and A-C, in that order (at most two T3 submissions).
    * **FAIL**: FRD-K is KILLED. Log the anti-pattern "a one-hop vote layer on NNBF does not beat a wider NNBF over 5 seeds". FRD-F dies with it (it is only FRD-K's cost fix). AO-F stays live only if T2-a says VOTE-INERT, because the 3-seed gain was then amplitude-borne and A-C is still informative. Otherwise refill the portfolio by Operation A on the FAIL mechanism.
    * **INCOMPLETE** (a task missing): resubmit only the missing tasks, with the same settings.
#### FRD-F
* **Mechanism**: FRD-K with frame-frozen ratios. Every vote is one Sherman–Morrison ratio from the inverse NNBF already holds.
* **Sharpest Gap**: frozen ratios deviate from the true ones (median relative error $23\%$, and $1.8\%$ of the edges with $\lvert R\rvert>0.5$ change sign). Whether that keeps FRD-K's sign gain is exactly T2-b.
* **Gate 4 claim**: the per-sample local energy uses the same number of NNBF passes as NNBF (32, 44, 62 at $L=4,5,6$), and the wall-time ratio is flat at $2.30$, $2.22$, $2.21$, with slope $-0.05$ in $N_e$. The extra cost is a constant factor (a Python loop over edges), so the gain would be extensive if FRD-K's is. **Gate 4 PASS at T1.**
* **Decision tree D-F** (job 697584, task 1, `t2_readout.py --mode aggregate`): DROP-IN-PASS goes into D-K's PASS branch (M-C with the read-out parameters as a warm start). AMBIGUOUS means `frdF` is trained in M-C. KILLED removes FRD-F, and if FRD-K passes, the only cost-honest successor is AO-F or a refill.
#### AO-F
* **Mechanism**: a sign-inert local-energy Jastrow on frozen ratios, which acts on the nodes only through the projection and through capacity freed in the backflow.
* **T0 lemma (recorded)**: $\mathrm{sgn}\,\psi_{\mathrm{AO}}(\mathbf{n})=\mathrm{sgn}\,\Phi(\mathbf{n})$ at every $\mathbf{n}$, so any change in $\Delta E_{\mathrm{sign}}$ of $P_\chi\psi$ comes from the image amplitudes in $\sum_g\chi_g\psi(g\mathbf{n})$ or from a different trained base.
* **Sharpest Gap**: whether amplitude off-loading accounts for FRD-K's gain at all (T2-a), and whether it beats a plain neural Jastrow (A-C).
* **Gate 4 claim**: the same cost as FRD-F (flat ratio of about $2.2$). A log-additive correlator is extensive.
* **T1**: none possible. Every projected toy at Hilbert dimension up to about $10^4$ is NO-TARGET (round-3 pilot 410216; round-5 smoke). The trained-state read-out T2-a stands in for it.
* **Decision tree D-A**: VOTE-INERT makes AO-F the leaf's lead route, with test A-C as above. VOTE-CARRIES KILLS AO-F (the gain needs flips). MIXED keeps it live behind M-C.
#### Tier log
* **T0 (this round)**: FRD-K Gate 4 (cost FAIL, extensivity open); FRD-F and AO-F cost arguments; the complex-detour lemma (KILLED); the holon-frame anchor bound (KILLED); node-targeted sampling (KILLED). Also from stored round-5 data: the post-hoc $P_S$ ratio is about $0.77$, so the gain is not spin restoration.
* **T1-a, vote informativeness** (`t1_vote.py`, 9 s, 4 threads; rules R1–R3 written in its docstring before the first run). The first run was invalid: the positive control divided by symmetry zeros of $\psi_0$ on $5\times2$. The check was fixed to exclude configurations with $\lvert\psi_0\rvert<10^{-6}\max$, and the rules were unchanged. Output of the valid run (U rows, the unprojected UHF anchor; the full log, including G rows, is in `t1_vote_output.txt`):

```
size 0 4x2 3+3 D=560   valid=True  neg_ctrl_lift_H 1.009  [U] w_b=0.0455 dEs_base=0.03783
   K: recall=0.221 lift=2.900 | X: recall=0.282 lift=2.144 | H: recall=0.095 lift=1.929
size 1 5x2 4+4 D=3150  valid=True  neg_ctrl_lift_H 1.023  [U] w_b=0.1880 dEs_base=0.08836
   K: recall=0.255 lift=1.269 | X: recall=0.395 lift=2.570 | H: recall=0.221 lift=2.662
size 2 6x2 5+5 D=16632 valid=True  neg_ctrl_lift_H 1.062  [U] w_b=0.1112 dEs_base=0.10838
   K: recall=0.222 lift=2.046 | X: recall=0.370 lift=3.236 | H: recall=0.332 lift=3.992
VERDICT: R1_frdK_gate4 UNDECIDED; R2_frdH KILLED; prediction lift_X < lift_K: false
```

* **T1-a verdicts**:
    * R1: UNDECIDED for FRD-K's Gate 4 trend.
    * R2: **FRD-H KILLED**, because $\mathrm{recall}_H\le\mathrm{recall}_K$ at $V=8$ and $V=10$. A caveat is logged without resurrecting the route: $\mathrm{lift}_H>\mathrm{lift}_K$ at $V=10$ and $12$, and the max-edge detector's recall is higher for H at all three sizes.
    * Applying the raw vote unconditionally raises $\Delta E_{\mathrm{sign}}$ in every frame and at every size (for example $0.108\to0.166$ for K at $V=12$). The vote is informative (lift above 1) but full of false alarms, so it is useful only behind a learned gate $\alpha$.
    * The recorded prediction failed: exchange votes are *more* informative than hole-hop votes at $V=10$ and $12$. So FRD-X's failure is not uninformative votes. It is the training cost of using them (heavy local-energy tails, $\kappa=11.9$; anti-pattern below).
* **T1-b, FRD-F** (`t1_frozen.py`, 9 s, one thread; rules F1–F3 written before the run):

```
A: SM vs direct frozen ratio rel 2.17e-16; FRD-F vs FRD-K at zero backflow rel 9.59e-16; true_factors vs r5 FRD.amplitudes rel 7.53e-17
B: L=4 Ne=14: NNBF 32 passes 8.066e-03 s | FRD-K 522 passes x1.89 | FRD-F 32 passes x2.30
B: L=5 Ne=22: NNBF 44 passes 1.269e-02 s | FRD-K 1008 passes x2.80 | FRD-F 44 passes x2.22
B: L=6 Ne=32: NNBF 62 passes 1.853e-02 s | FRD-K 1867 passes x3.52 | FRD-F 62 passes x2.21
B: log-log slopes in N_e: ratio_frdK=0.76 ratio_frdF=-0.05 T_nnbf=1.01 T_frdK=1.76 T_frdF=0.96
C: nsamp=20000 flipped samples 0; w_flip=0.000e+00 w_dis=0.000e+00 weighted median |log ratio|=8.825e-02; edges |R|>0.5: 159774 sign-changed frac 1.809e-02; median rel |R^F - R|=2.272e-01
VERDICT FRD-F: KILLED (F3 fidelity)
```

* **T1-b verdicts**:
    * F1 is valid, and F2 (cost) PASSES.
    * F3 fired as written, but **only on $0\ge0$**. The 20000 draws from the *unprojected* $\lvert\psi\rvert^2$ contained no flipped configuration, so the flips sit where $\psi_\theta$ is small and $P_\chi\psi$ is not. The sign clause therefore had no power, and the amplitude clause passed ($0.088<0.1$).
    * This is a defect of the rule (a non-strict comparison of two zero estimates). The verdict is recorded as **T1-INCONCLUSIVE, not PASS**, and the decision moves to the exact projected read-out T2-b. This departure from the literal rule is flagged for the per-leaf critic.
* **Pre-flight dry run for T2** (`t2_readout.py --mode dryrun`, 4×2 OBC $3+3$, round-5 smoke FRD-K state; full log `dryrun_output.txt`). The dry run uses a synthetic control level $\Delta E_{\mathrm{sign}}^{C}=\Delta E_{\mathrm{sign}}^{K}+0.02$, because the toy FRD-K has no gain over its control (0.036 against 0.001):

```
[09:34:56] loaded inst 8 D=560 |G|=4 ED -5.1899410651 (stored -5.1899410651) (0 s)
[09:34:57] ident: E=-5.01201790 dE_sign=0.036175 dE_tot=0.177923 (0 s)
[09:34:57] abs: E=-5.01200915 dE_sign=0.036175 dE_tot=0.177932 (0 s)
[09:34:57] frozen: E=-4.96216282 dE_sign=0.036175 dE_tot=0.227778 (0 s)
[09:34:57] rflip: E=-4.75388941 dE_sign=0.077407 dE_tot=0.436052 (0 s)
DRYRUN verdicts: {"valid": true, "v_abs": 0.0, "T2a_abs": "VOTE-INERT", "v_rflip": 2.06, "T2a_rflip": "VOTE-CARRIES", "v_ident": 0.0, "T2a_ident": "VOTE-INERT", "u_frozen": 4.4e-14, "T2b": "DROP-IN-PASS", "gate4_lift": "EXTENSIVE-OK"}
DRYRUN controls give intended verdicts: True
```

* **Dry-run checks**:
    * The validity asserts pass: the ED $E_0$ equals the stored value, the identity read-out reproduces the stored energy and $\Delta E_{\mathrm{sign}}$, and the parameter count matches.
    * Both controls give their intended verdicts: identity gives VOTE-INERT ($v=0$), and the random 2% sign flip gives VOTE-CARRIES ($v=2.06$).
    * The synthetic aggregate prints every branch: INERT, MIXED and CARRIES; DROP-IN-PASS, AMBIGUOUS and KILLED; FINITE-SIZE, UNDECIDED and EXTENSIVE-OK; INVALID; and no gain.
* **T2 pilot, job 697584** (`tests/P1.2_unanchored_antisymmetry_r6/submit_t2.sbatch`, array 0–1, partition `kim`, 4 CPUs × 30 min, so each task is capped at 2 CPU-hours). It runs on the trained round-5 FRD-K state with seed 11 and shift $10^{-3}$, with no training. The rules are in the `t2_readout.py` docstring, with $\Delta E_{\mathrm{sign}}^{K}=0.00975$ and $\Delta E_{\mathrm{sign}}^{C}=0.012045$ (gap $0.0023$):
    * T2-a: $v=(\Delta E_{\mathrm{sign}}[\lvert F\rvert]-\Delta E_{\mathrm{sign}}^{K})/\mathrm{gap}$. VOTE-INERT if $v<0.25$, VOTE-CARRIES if $v\ge0.5$, MIXED otherwise.
    * T2-b: $u=(\Delta E_{\mathrm{sign}}[F^F]-\Delta E_{\mathrm{sign}}^{K})/\mathrm{gap}$. DROP-IN-PASS if $u\le0.25$, KILLED if $u\ge1$, AMBIGUOUS otherwise.
    * Gate 4 lift at $V=16$: FINITE-SIZE if $\mathrm{lift}_K<1.2$, EXTENSIVE-OK if $\mathrm{lift}_K\ge1.5$.
    * Validity: $E_0$ and the identity energy must match the stored values.
    * There are no training checkpoints, so there is no futility stop. Partial JSONs are written after each stage, and a timeout gives INCOMPLETE.
    * Verdict command: `python3 t2_readout.py --mode aggregate`.
### Local Autopsy Log (Round 6 additions)
* **FRD-H (Route 6.2)**: killed at T1 by R2, because adding the exchange edges lowers the recall of the summed vote at two of three sizes.
* **Complex detour (Route 6.4)**: killed at T0. $E[A+iB]$ is a weighted mean of $E[A]$ and $E[B]$, so the readout's nodes are not unanchored.
* **Holon-frame backflow (Route 6.5)**: killed at T0. Its anchor is worse than the foil's ($W=0.34$ against $0.073$) and is exact only as $J/t\to\infty$.
* **Node-targeted sampling (Route 6.6)**: killed at T0. It repeats the P1.3 sampler anti-pattern.
* **FRD-F (Route 6.1)**: killed at T2-b. The frozen-frame read-out of the trained state has $u=1.50$ (it loses the whole gain), and the vote it was meant to make cheap is inert anyway.
* **FRD-K's vote mechanism**: killed at T2-a ($v=0.003$). The accuracy test 553724 continues as evidence for amplitude off-loading only.
* **Route 6.7, vote-initialized FRD (refill, Operation A on the VOTE-INERT result)**. Start every $\alpha_e$ so that factors flip where the trained base's vote fires ($\mathrm{lift}_K=6.3$), then train by SR. **T0 kill**: T1-a showed that applying the raw vote raises $\Delta E_{\mathrm{sign}}$ at every size, because of false alarms. SR would have to switch the false alarms off through the same zero crossings (barrier (b) in reverse), so the start does not remove the barrier that left the vote unused.
* **Route 6.8, projected-vote sign read-out (refill, Operation B.2)**. Apply the vote one-shot to $P_\chi\psi$, where its lift is $129$. **T0 kill (Gate 0)**: a sign set by one exact local Hamiltonian application to a projected state is a selective one-step power or Lanczos sign (arXiv:1412.2656). It is not size-consistent at a fixed number of steps, and it is the round-2 Route 2.3 kill.
* **New anti-pattern (for the master doc)**: *a sign layer can win a matched-parameter test without using its sign capacity*. FRD-K's gain survived removing all of its flips (T2-a). Every sign-layer PASS needs a flip-off read-out before it is credited to the sign mechanism.
* **New anti-pattern (for the master doc)**: *an informative vote is not a trainable vote*. Exchange votes carry more wrong-sign information than hole-hop votes on the toy ladders, yet FRD-X failed with $\kappa=11.9$. A vote route must be judged by the local-energy tails it adds as well as by its information.
### Round 6 per-leaf critic (2026-11-16)
Scope: the Round 6 section above, `tests/P1.2_unanchored_antisymmetry_r6/` (`t2_readout.py`, `t1_vote.py`, `t1_frozen.py`, outputs) and the round-5 result JSONs. New critic code, all in the same folder: `critic_r6_checks.py` (claims checked from stored data), `t1_ao.py` (AO-family T1), `t1_drf.py` (refill T1), `t2c_readout.py` with `t2c_synthetic.py` and `submit_t2c.sbatch` (prepared T2 read-outs). Job 553724 was not touched.
#### Claims checked from stored data (`critic_r6_checks.py`, output `critic_r6_checks_output.txt`)

```
(1) nnbf6 selected per seed: [(11, 0.01246, 0.001), (22, 0.01205, 0.01), (33, 0.01176, 0.01), (44, 0.01292, 0.001), (55, 0.01319, 0.01)]
    frdK  selected per seed: [(11, 0.00975, 0.001), (22, 0.01011, 0.001), (33, 0.01038, 0.001)]
    nnbf6 seeds 5, med_5 dEs = 0.01246; frdK seeds 3; rho_sign lower bound = 0.782 (3-seed frdK median gives 0.811)
    553724 final verdict: FAIL is certain (bound > 0.7)
(2) T2-a: dEs_K 0.009751 dEs_abs 0.009757 gap 0.00229 v = 0.0026; E_abs - E_ident = 5.21e-05
(3) precision = lift * w_b (upper bound of lift is 1/w_b)
    toy V=8 U: w_b 0.0455 1/w_b 22.0 | K: lift 2.90 prec 0.132 X: lift 2.14 prec 0.098 H: lift 1.93 prec 0.088
    toy V=10 U: w_b 0.1880 1/w_b 5.3 | K: lift 1.27 prec 0.239 X: lift 2.57 prec 0.483 H: lift 2.66 prec 0.500
    toy V=12 U: w_b 0.1112 1/w_b 9.0 | K: lift 2.05 prec 0.228 X: lift 3.24 prec 0.360 H: lift 3.99 prec 0.444
    trained V=16 U: w_b 0.1183 1/w_b 8.5 | K: lift 6.30 prec 0.745 recall 0.347 X: lift 3.01 prec 0.356 recall 0.425 H: lift 7.11 prec 0.840 recall 0.380
    trained V=16 G: w_b 0.0062 1/w_b 161.5 | K: lift 128.61 prec 0.796 recall 0.478 X: lift 4.22 prec 0.026 recall 0.434 H: lift 148.01 prec 0.916 recall 0.474
```

The toy G rows are in the full output file.
#### Major objections
1. **The job-553724 decision is already made, so the tree D-K was planned on the wrong branch.** All ten `nnbf6` runs are on disk. With the per-seed lower-energy shift (the rule of `aggregate()`), the 5-seed control median is $0.01246$. A 5-seed median of `frdK` cannot fall below the smallest of its three stored values, $0.00975$. So $\rho_{\mathrm{sign}}\ge0.782>0.7$ and `--final 1` returns **FAIL** whatever seeds 44 and 55 give. Every PASS branch of D-K (M-C, and A-C after a PASS) is moot. The branch that will run is FAIL, and its clause "AO-F stays live if VOTE-INERT, because A-C is still informative" sends a route to T3 on the strength of a gain ($\rho_{\mathrm{sign}}\approx0.8$) that its own parent failed to establish. That clause is struck. AO-F had to earn its premise at T1 first (objection 4), and it failed there.
2. **VOTE-INERT is a sound reading of the trained state, but not of the architecture.** The read-out is clean: $v=0.0026$, an energy shift of $5\times10^{-5}$, a budget spread of $5\times10^{-13}$, and flip weights of at most $3.7\times10^{-3}$ in all six `frdK` runs, so seed 11 is representative. What it shows is "this SR-trained FRD-K does not use its flips". It does not show "the vote is irrelevant to the gain". The vote's factors are built from the true neighbour amplitudes $\Phi(\mathbf{m}_e)$, and the gain could need those features without needing sign capacity. The counterfactual that answers the architecture question is a sign-inert factor on the **same exact ratios**, retrained from scratch (`aoK`). The pre-planned test A-C changes two things at once: it removes the flips **and** swaps exact ratios for frozen ones. T2-b had just shown that frozen ratios lose the whole drop-in gain, so a FAIL of A-C could not be attributed. On the foil question: a matched NNBF × neural Jastrow (`nnbfJ`) is enough to show the vote irrelevant only in the branch where `nnbfJ` matches FRD-K. If `nnbfJ` loses, it cannot tell features from flips, and `aoK` is required. Both arms were run at T1 below.
3. **"The vote is informative but unused; barrier (b) blocks it" is not supported, and the lift is not a Gate 4 trend.** Lift is precision divided by the base rate, so it is bounded by $1/w_b$. The values compared across sizes come from different bases (UHF anchor at $V=8$ to $12$, trained NNBF-4 at $V=16$) with different $w_b$. The projected lift of 129 mostly reflects the small base rate $w_b=0.006$. In precision terms the hole-hop vote goes 0.13, 0.24 and 0.23 on the toys, and 0.75 (unprojected) or 0.80 (projected) on the trained base. That is a change of base, not of size, so "EXTENSIVE-OK" carries no Gate 4 information and is relabelled **no evidence**. On the barrier claim: T1-a found that applying the raw vote *raises* $\Delta E_{\mathrm{sign}}$ for the anchor, which supports the "net-harmful" reading, not the barrier reading. On the trained base the precision is 0.80, so the question is genuinely open. It is decided by the prepared read-out `gvote` (T2c, task 2) before any route that relies on trainable flips is reopened.
4. **AO-F failed Gate 4 on paper, and "T1 impossible" was wrong.** AO-F has two mechanisms. The first moves the projected nodes through the image amplitudes in $\sum_g\chi_g\psi(g\mathbf{n})$. That is a point-group symmetry-restoration effect with fixed $\lvert G\rvert=8$, and the standard finite-size argument of Gate 4 applies. The second frees backflow capacity, which is not specific to AO, since any positive factor offers it. "A log-additive correlator is extensive" is a statement about energy, not about a sign gain. Unprojected toys do have targets: matched-width NNBF keeps $\Delta E_{\mathrm{sign}}=0.006$ to $0.024$ (below). So mechanism 2 was testable at T1 on three sizes. The test was run (`t1_ao.py`) and **killed AO-F and its new sibling AO-L**.
5. **FRD-F's kill is right, but for a different reason, and the same reasoning had to apply to AO-F.** T2-b substitutes frozen ratios under an $\alpha$ trained against exact ratios. A drop-in failure ($u=1.50$) does not show that a retrained FRD-F fails, because the 1.8% of large edges that change sign under freezing can create spurious flips. The kill stands on VOTE-INERT instead: FRD-F exists only to make the vote cheap, and the vote is unused. The worker read the same kind of drop-in loss as fatal for FRD-F but not for AO-F, which uses the same frozen ratios. With objection 4, AO-F now dies at T1 on a retrained test.
#### Minor objections
6. **FRD-H: the R2 recall rule is contradicted at $V=16$, so the kill rests on Gate 4.** On the trained base, $\mathrm{recall}_H=0.380>\mathrm{recall}_K=0.347$, and the precision is 0.84 against 0.75. The toy rule used the UHF anchor only. The kill stands on two other grounds: FRD-H has 1.6 times FRD-K's $\mathcal{O}(V)$ cost wall, and its vote would be as unused as FRD-K's.
7. **The F3 zero-against-zero rule.** The worker's handling was correct: the result was recorded as T1-INCONCLUSIVE, the decision was deferred to T2-b, and the defect was flagged. In future, a sign-fidelity clause must sample the distribution where the flips live (the projected one), or be written as a strict inequality with a minimum count.
8. **The Gate 4 cost argument for FRD-K is correct, with one addition.** The pass counts are the asymptotic statement: $E_{\mathrm{act}}\propto V$ extra NNBF passes per amplitude. The measured wall-time ratios of 1.9 to 3.5 are still overhead-dominated. The extra factor applies to every amplitude, not only to the local energy, so sampling pays it too. The verdict, FAIL as-is, stands.
9. **The five T0 kills hold.**
    * *Complex detour (6.4)*: the lemma is right. For real symmetric $H$, $E[A+iB]$ is the norm-weighted mean of $E[A]$ and $E[B]$. The wording "nothing makes $A$ the better part" is beside the point, since one could read out whichever part is better. The real point is that the objective never couples the parts, so a node crossing in either part costs that part's own barrier.
    * *Holon frame (6.5)*: "universal, so it reduces to anchor quality" is loose, because the round-1 bound speaks only to the starting anchor. The kill stands as a Gate 3 statement about the start ($W=0.34$ against $0.073$).
    * *Node-targeted sampling (6.6)*: sound. It repeats the P1.3 anti-pattern.
    * *Vote-initialized FRD (6.7)*: sound, pending `gvote`. If T2c returns VOTE-USABLE, 6.7 is the first route to revisit.
    * *Projected-vote read-out (6.8)*: sound. The sign of a Lanczos-stepped state is this read-out (Loehr & Clark, arXiv:2510.26906).
10. **Tier discipline.** No T2 or T3 was requested without its cheaper tiers, and the dry runs had working controls. One lapse: FRD-K's cost failure was knowable at T0 in round 5 from an operation count, yet the route was carried as a "parent" through a 24-hour T3. Objection 1 now closes that pattern.
11. **Budget note (this critic).** `t1_ao.py` used 9.9 min of wall time and 34.8 CPU-minutes at 4 threads, which exceeds the 10 CPU-minute T1 cap. One seed would have fitted under the cap. The verdict uses two seeds as pre-registered and is not re-run.
#### Per-route verdicts
* **FRD-K (5.1)**: **KILLED**. Three grounds: T2-a VOTE-INERT, the Gate 4 cost FAIL as-is, and the 553724 final FAIL, which is already certain. Job 553724 runs to completion as data only, untouched.
* **FRD-F (6.1)**: **KILLED**. The kill is accepted, with its reason re-based on VOTE-INERT (objection 5).
* **FRD-H (6.2)**: **KILLED**. The kill is accepted, with its reason re-based on Gate 4 (objection 6).
* **AO-F (6.3)**: **KILLED at T1** (`t1_ao.py`, below). Test A-C is withdrawn.
* **Routes 6.4 to 6.8**: kills accepted (objection 9).
#### T1 run by the critic: AO-family premise (`t1_ao.py`, output `t1_ao_output.txt`)
* **Rules and setup** (written in the docstring before the run):
    * *Toys*: three unprojected 2-leg ladders with $V=8,10,12$. Every arm starts from UHF and is trained by exact-sum SR (300 steps, seeds 11 and 22).
    * *Arms*: `nnbfW` (backflow width 4; it sets the parameter target), `nnbfJ` (width 2 times an MLP Jastrow) and `aoK` (width 2 times $\exp\sum_e\alpha_e(\mathbf{n})g(R_e)$ on **exact** hole-hop ratios). `aoK` is AO-L's ansatz and the most informative member of the AO family.
    * *Metric*: $r^\ast=\max(r_J,r_W)$, where $r_J=\Delta E_{\mathrm{sign}}[\mathrm{aoK}]/\Delta E_{\mathrm{sign}}[\mathrm{nnbfJ}]$ and $r_W=\Delta E_{\mathrm{sign}}[\mathrm{aoK}]/\Delta E_{\mathrm{sign}}[\mathrm{nnbfW}]$.
    * *Kill rule*: $r^\ast\ge0.9$ at 2 or more counted sizes, including the largest.
    * *Checks*: the analytic force matches finite differences ($1.5\times10^{-8}$), and $\alpha=0$ reproduces the base exactly.

```
  size=0 V=8 dEs_W=0.006293 dEs_J=0.02285 dEs_aoK=0.01284 r_J=0.5619 r_W=2.04 r_star=2.04 r_JW=3.631 counted=True
  size=1 V=10 dEs_W=0.02351 dEs_J=0.03348 dEs_aoK=0.02245 r_J=0.6706 r_W=0.9548 r_star=0.9548 r_JW=1.424 counted=True
  size=2 V=12 dEs_W=0.01845 dEs_J=0.04562 dEs_aoK=0.02187 r_J=0.4793 r_W=1.185 r_star=1.185 r_JW=2.472 counted=True
VERDICT AO-family premise (AO-F, AO-L): KILL (valid True)
```

* **Reading**:
    * The exact-ratio factor beats a plain neural Jastrow at every size ($r_J=0.48$ to $0.67$), so the ratio features carry real information.
    * It still loses at every size to the same parameters spent on backflow width ($r_W=2.04$, $0.95$ and $1.19$).
    * Any positive factor loses to width even more clearly ($r_{JW}=1.4$ to $3.6$).
    * So mechanism 2 ("a positive factor frees backflow capacity for the nodes") is refuted at matched parameters on the toys. Mechanism 1 is a point-group finite-size effect (objection 4).
* **Caveats**: only 2 seeds, with large seed spreads ($\Delta E_{\mathrm{sign}}[\mathrm{nnbfW}]$ is 0.011 and 0.026 at $V=12$). Training is unprojected, with 300 SR steps. The kill follows the pre-registered rule and applies to AO-F and AO-L together.
* **Lesson carried forward**: ratio features help, but they must enter the orbitals, not a positive factor. That sets the refill.
### Round 6 refill (critic, 2026-11-16)
* **Gate 0 searches** (WebSearch, 2026-11-16): a ratio-feature or local-energy Jastrow; iterative backflow with Green's-function or determinant-inverse features.
* **Route 7.1, AO-L (Operation A on FRD-K: drop the inert flips, fix the cost wall)**. The ansatz is $\Phi\exp\sum_e\alpha_e g(R_e)$ on **exact** ratios. A finite-receptive-field backflow turns each neighbour amplitude into a rank-$\mathcal{O}(r^d)$ update. That update is SCALE's machinery (arXiv:2604.25775), so it is not new; the factor is.
  - Gate 0 (Novelty & Adoption): PASS for the factor. No ratio-feature Jastrow was found.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS as a design, with `nnbfW` and `nnbfJ` as foils.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): cost PASS on paper (a constant factor in $r$); extensivity open.
  - *Verdict*: **KILLED at T1** by `t1_ao.py` (it is the `aoK` arm).
* **Route 7.2, DRF-K: determinant-ratio-feature backflow on the hole-hop frame (Operation B.3, cross-field from continuum iterative backflow)**. The construction:
    * A first NNBF pass gives orbitals $O_{\mathbf{n}}$.
    * Its inverse gives the frozen hop ratios $R^F_e(\mathbf{n})$, which are Green's-function entries, each $\mathcal{O}(1)$ once $O_{\mathbf{n}}^{-1}$ exists.
    * The features $g(R^F_e)$, gathered per site, feed a second backflow pass that outputs the final orbitals, and $\psi=\det O^{(2)}_{\mathbf{n}}[\mathrm{rows}(\mathbf{n})]$.
    * The information that made the vote informative then moves the nodes by **continuous orbital deformation**. That avoids factor flips, which SR declines, and positive factors, which lose to width.
  - Gate 0 (Novelty & Adoption): PASS, narrowly. Iterative backflow exists in continuum QMC (Taddei et al., PRB 91, 115106 (2015); in network form, Ruggeri et al., arXiv:1711.01993). Lattice NNBF feeds occupations only (Luo & Clark, arXiv:1807.10770; symmetry-aware backflow, arXiv:2406.09077). No backflow fed by its own determinant's Green's function was found. *Why it is not standard*: a second pass doubles the cost, and generic depth already saturates (Loehr & Clark). The concrete mechanism against that wall is the new input itself: the inverse of the orbital matrix, which locates the first pass's nodes and which an MLP cannot form cheaply.
  - Gate 1 (Conservation & Asymptotics): PASS. Its features are diagonal in $\mathbf{n}$, so it conserves $N_\uparrow$, $N_\downarrow$ and the $t$–$J$ constraint.
  - Gate 2 (Baseline Gameability): PASS as a design. The foils are a wider single-pass NNBF and a two-pass NNBF fed occupations only, both at matched parameters.
  - Gate 3 (Hard-Core Shield): PASS, sign half only. The decisive instance is doped, has $t'\ne0$ and is 2D.
  - Gate 4 (Scaling & Relevance): cost PASS on paper. An amplitude costs two NNBF passes, plus $\mathcal{O}(VN_e^2)$ to form $G=O\,O_{\mathrm{occ}}^{-1}$, plus $\mathcal{O}(E_{\mathrm{act}})$ lookups. That is a constant factor of about 2 over NNBF with no $\mathcal{O}(V)$ wall, because the local energy needs the same $(1+n_{\mathrm{conn}})$ amplitudes as NNBF. The gain is extensive if the information premise grows with size, which T1 checks.
  - *Verdict*: SURVIVED. T0 is the cost count above. T1 is UNDECIDED with a rising trend (below).
* **Route 7.3, DRF-X (sibling frame: exchange-move ratios)**. The same mechanism with exchange-edge ratios, which are a rank-1 update in each spin determinant. It is motivated by T1-a, where exchange votes were more informative than hole-hop votes at $V=10$ and $12$. It inherits the gates of Route 7.2. FRD-X's failure ($\kappa=11.9$, heavy local-energy tails) came from multiplicative factors near nodes. DRF has no such factor, which is its Gate 2 and Gate 4 distinction from FRD-X. *Verdict*: SURVIVED. **T1 PASS.**
* **Route 7.4, DRF-H (sibling frame: all off-diagonal moves)**. Features from $K\cup X$. It inherits the gates of Route 7.2, with about 1.6 times DRF-K's feature count, each still $\mathcal{O}(1)$. *Verdict*: SURVIVED. **T1 PASS.**
* **Route 7.5, gated mixture of NNBF experts (Operation B.3, from machine-learning mixtures of experts)**. $\psi=\sum_k\pi_k(\mathbf{n})\Phi_k(\mathbf{n})$ with a softmax gate, so a sign changes by gate competition. **T0 kill (Gate 4)**: independent local sign repairs in $R$ distant regions need $2^R$ experts, or a gate that factorizes over regions. A factorized gate is a CPS sign correlator, which is the round-2 relabel kill. The route also falls under Anti-Solution Space (1).
* **Route 7.6, transcorrelated frame (Operation B.3, from transcorrelated FCIQMC: Dobrautz, Luo & Alavi, PRB 99, 075119 (2019))**. Train an NNBF $\Phi$ on $\tilde H=e^{-J}He^{J}$ with a pre-optimised diagonal $J$. **T0 kill (Gate 2)**: on a lattice with diagonal $J$, the represented state $e^J\Phi$ is exactly the `nnbfJ` family, which just lost to width at T1. The only difference is a non-variational objective, and that is P1.3's scope.
#### Refill tier log
* **T1 for Routes 7.2 to 7.4** (`t1_drf.py --frame {K,X,H}`, output `t1_drf_output.txt`, under 10 s each; rules in the docstring before the first run):
    * *Premise*: information only. Oracle labels are used **only** as a kill test, never as a training target.
    * *Models*: weighted $L_2$-logistic models of the anchor's wrong-sign label. Model A uses the occupations. Model B adds the frame's ratio features. Model C is a capacity control that adds the same number of random $\pm1$ features instead.
    * *Protocol*: each model is fitted on a random half of the configurations and scored by weighted AUC on the other half.
    * *Rule*: $\mathrm{dAUC}=\mathrm{AUC}_B-\max(\mathrm{AUC}_A,\mathrm{AUC}_C)$. KILL if $\mathrm{dAUC}<0.03$ at 2 or more sizes including $V=12$. PASS if $\mathrm{dAUC}\ge0.05$ at all sizes and $\mathrm{dAUC}(12)\ge0.5\,\mathrm{dAUC}(8)$.

```
size=0 V=8  K dAUC=0.0288 | X dAUC=0.1352 | H dAUC=0.1212
size=1 V=10 K dAUC=0.0820 | X dAUC=0.1014 | H dAUC=0.1467
size=2 V=12 K dAUC=0.1265 | X dAUC=0.2299 | H dAUC=0.2515
VERDICT DRF-K UNDECIDED; DRF-X PASS; DRF-H PASS   (AUC code check: true-label AUC = 1.0000 at every size)
```

* **Reading**: the block above is condensed from the three runs in `t1_drf_output.txt`. The ratio features add 0.10 to 0.25 of held-out AUC over the occupations and over an equal number of random features, and the gain grows with $V$ in every frame. That is the Gate 4 trend the AO and FRD routes never showed.
* **Sharpest gap (shared by the three DRF routes)**: the T1 premise used the UHF anchor, where frozen ratios equal the exact ones. On a trained backflow the frozen ratios deviate (median relative error 23%, `t1_frozen.py`). And information is not trainability. The next tier must show that a two-pass NNBF *trained by SR* beats a matched single-pass NNBF and a matched two-pass NNBF fed occupations only.
* **Next tier (T1b, not yet built)**: a two-pass DRF in the round-5 numpy framework.
    * *Gradient*: it needs the gradient through $G=O\,O_{\mathrm{occ}}^{-1}$, which is $\partial G=\partial O\,O_{\mathrm{occ}}^{-1}-G\,\partial O_{\mathrm{occ}}\,O_{\mathrm{occ}}^{-1}$.
    * *Design*: train unprojected on the three ladders against `nnbfW` and against the occupation-fed two-pass control, at matched parameters.
    * *Pre-registered kill*: $\Delta E_{\mathrm{sign}}[\mathrm{DRF}]\ge0.9\min(\text{controls})$ at 2 of 3 sizes including $V=12$.
#### T2 read-outs (T2c; submitted as job 698178)
* **Code and state**: `t2c_readout.py` reads out the same trained FRD-K state as job 697584. The rules are in its docstring.
* **Task 0, `base`**: $\Phi$ alone. It locates the 4×4 gain either in the co-adapted base's nodes or in the projection reshaping them (mechanism 1). The answer goes to the anti-pattern record.
* **Task 2, `gvote`**: the projected hole-hop vote, applied without a gate. It decides objection 3: VOTE-USABLE reopens Route 6.7, and VOTE-HARMFUL rejects the barrier reading.
* **Task 1, `absfroz`**: dropped, because AO-F is dead.
* **Pre-flight dry run** (`t2c_dryrun_output.txt`): the validity asserts pass, the base equals the identity read-out at $\alpha=0$ to $0$, and the synthetic aggregate prints every branch.
* **Command (submitted as job 698178 by the orchestrator)**: `cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P1.2_unanchored_antisymmetry_r6 && sbatch submit_t2c.sbatch` (array 0,2; partition `kim`; 4 CPUs for 30 min per task). For the verdict afterwards, run `python3 t2c_readout.py --mode aggregate`.
#### Route Portfolio (critic, 2026-11-16)

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DRF-K (7.2) | T1 | LIVE, UNDECIDED (dAUC rising from 0.03 to 0.13) | T1b code | T1b: KILL if the trained $\Delta E_{\mathrm{sign}}\ge0.9\min$(controls) at 2 of 3 sizes incl. $V=12$ | PASS: three-size trend, then T2 on instance 0 against `nnbf6`; KILL: log |
| DRF-X (7.3) | T1 | LIVE, T1 PASS | T1b code | same T1b rule | same |
| DRF-H (7.4) | T1 | LIVE, T1 PASS | T1b code | same T1b rule | same; if all three pass, keep H and drop the dominated frame |
| FRD-K (5.1) | T2 | KILLED(T2-a, Gate 4 cost; 553724 FAIL certain) | — | — | 553724 runs to completion as data only |
| AO-F (6.3), AO-L (7.1) | T1 | KILLED(T1, `t1_ao.py`: loses to matched width) | — | — | Logged |
| FRD-F (6.1), FRD-H (6.2) | T2, T1 | KILLED (reasons re-based) | — | — | Logged |
| Vote reopen (6.7) | T0 | KILLED, conditional | T2c `gvote` | VOTE-USABLE reopens 6.7 | Submit T2c |

#### Local Autopsy Log (critic additions)
* **AO-F and AO-L**: killed at T1. A sign-inert factor on exact Hamiltonian-neighbour ratios beats a plain neural Jastrow but loses to the same parameters spent on backflow width (`t1_ao.py`).
* **New anti-pattern (for the master doc)**: *off-loading amplitude to a positive factor does not buy sign accuracy at matched parameters*. On the toys every positive factor lost to backflow width ($r_{JW}=1.4$ to $3.6$).
* **New anti-pattern (for the master doc)**: *comparing vote lifts across bases or sizes*. Lift is bounded by $1/w_b$. Use precision and recall at a fixed base.
* **Process anti-pattern**: *keeping a route alive on a T3 whose final rule stored data has already decided*. Before planning branches on an outcome, check that it is still reachable.

#### T2c Read-out Result (job 698178, 2026-11-16)

Aggregate: `python3 t2c_readout.py --mode aggregate` (saved as `tests/P1.2_unanchored_antisymmetry_r6/t2c_698178_aggregate.txt`). Both tasks are valid; the gap is $2.29\times10^{-3}t$.

| read-out | statistic | verdict | pre-registered meaning |
| :--- | :--- | :--- | :--- |
| `base` ($\Phi$ alone) | $b=3.59$ | REFRAME-CARRIES ($b\ge1$) | the positive amplitudes $\lvert F\rvert$ reshape the projected nodes; this is a point-group projection effect, so any route that relies on it needs a three-size Gate 4 trend before T3 |
| `gvote` (projected vote, no gate) | $z=-1.33$ | VOTE-USABLE ($z\le-0.5$) | the projected vote, applied without any gate, removes 1.33 gaps of sign error that SR left in place |

* **Consequence (pre-planned decision tree):** VOTE-USABLE reopens route 6.7 (vote-initialized FRD) as a live route at T0. The critic's caution stands: the raw unprojected vote raised $\Delta E_{\mathrm{sign}}$ for the anchor at T1, so the reopened route must use the projected vote and must beat the matched-width backflow foil.

### Round 7 — Trained DRF T1, route 6.7 and refill (2026-11-16)
Scope: Gate 4 at T0 for the DRF family, the trained T1 (T1b) of DRF-K, DRF-X and DRF-H against the matched-width foil, the reopened route 6.7 (broadcast after job 698178), and the refill. New code, all in `tests/P1.2_unanchored_antisymmetry_r6/`: `t1b_drf.py` (the DRF ansatz, hand-written backprop, checks, training, verdict, cost) and `t1_vote67.py` (the vote premise on the foil). Jobs 553724 and 698178 were not touched. No job was submitted.
#### Gate 4 at T0: DRF family
* **Operation count per amplitude.** DRF costs two backflow passes, one $N_\sigma\times N_\sigma$ inverse per spin (the same $\mathcal{O}(N^3)$ as the determinant), $E_{\mathrm{act}}$ Green's-function entries $G_\sigma[j_e,k_e]=O_1[j_e,:]\cdot A^{-1}[:,k_e]$ at $\mathcal{O}(N)$ each, and an $\mathcal{O}(E)$ gather. With $E\propto V$ and $N\propto V$ the extra work is $\mathcal{O}(VN)+\mathcal{O}(N^3)$, the same order as one NNBF amplitude.
* **Local energy.** Each connected configuration $\mathbf{m}$ evaluates its own DRF amplitude, with features frozen at $\mathbf{m}$. So a local energy needs $(1+n_{\mathrm{conn}})$ DRF amplitudes, against $(1+n_{\mathrm{conn}})$ NNBF amplitudes. There are no $\mathcal{O}(V)$ extra passes, which is the trap that killed FRD-K ($522$ against $32$ passes at $L=4$).
* **Measured** (`t1b_drf.py --mode cost`, output `t1b_cost_output.txt`; batch 256, best of 20, one thread, DRF-H with $H_1=H_2=3$ against NNBF width 6):

```
cost L=4 V=16 Ne=14 E_H=132: NNBF-6 3.4 us/amp, DRF-H 33.5 us/amp, ratio 9.75
cost L=5 V=25 Ne=22 E_H=224: NNBF-6 7.3 us/amp, DRF-H 71.0 us/amp, ratio 9.66
cost L=6 V=36 Ne=32 E_H=340: NNBF-6 15.2 us/amp, DRF-H 139.1 us/amp, ratio 9.15
cost L=7 V=49 Ne=42 E_H=480: NNBF-6 27.6 us/amp, DRF-H 223.8 us/amp, ratio 8.10
cost: log-log slope of the DRF/NNBF per-amplitude ratio in N_e = -0.16
```

* **Verdict: Gate 4 cost PASS.** The ratio is a constant factor that falls slowly with size (slope $-0.16$). The constant, about 8 to 10 in this numpy code, is gather overhead. It matters for any later matched-compute test, where the foil could be about 9 times wider. Extensivity of the gain was left to the trained three-size T1 below.
#### T1b: trained DRF against matched width (`t1b_drf.py`)
* **Ansatz.**
    * Pass 1 is an NNBF of width $H_1=3$ with orbitals $O_1$.
    * Its frozen ratios are $R_e=(h_e/s_e)\prod_\sigma\pi_\sigma G_\sigma[j_\sigma,k_\sigma]$, where $h_e$ is the Hamiltonian element, $s_e$ the frame scale and $\pi_\sigma$ the row-sorting sign. They pass through $g(R)=R/\sqrt{1+R^2}$ and are summed per site and channel.
    * Pass 2 is a residual layer, $O_2=O_1+\tanh([\mathbf{x},\mathbf{f}]W_3+b_3)W_4$, and $\psi=\det O_{2\uparrow}\det O_{2\downarrow}$.
    * $H_2$ is the largest width with $n_{\mathrm{params}}\le T$, the parameter count of the foil `nnbfW` (NNBF width 6).
* **Checks** (`t1b_checks_output.txt`, all pass):
    * CHECK 1: at zero backflow, $R_e$ equals the exact $h_e\Phi(\mathbf{m}_e)/(s_e\Phi(\mathbf{n}))$ to $9\times10^{-16}$ on $V=8$ and $V=10$. The first version failed this check because it omitted the sorting sign $\pi_\sigma$; that was fixed before any training.
    * CHECK 2: the analytic force matches central finite differences to $3.3\times10^{-8}$ (feature G) and $2.0\times10^{-8}$ (control feature O).
    * CHECK 3: with $W_4=0$ the ansatz reproduces the pass-1 NNBF exactly.
    * CHECK 4: the per-sample log-derivatives match the force rows to $4\times10^{-16}$.
* **Protocol.** Three unprojected ladders ($V=8,10,12$) with UHF starts, 300 exact-sum SR steps, lr 0.05, shift $10^{-3}$, seeds 11 and 22.
* **Pre-registered rule** (docstring, written before training; the critic's T1b rule):
    * A size is counted iff the median $\Delta E_{\mathrm{sign}}[\mathrm{nnbfW}]\ge0.005$.
    * KILL if $r_W=\Delta E_{\mathrm{sign}}[\mathrm{drf}]/\Delta E_{\mathrm{sign}}[\mathrm{nnbfW}]\ge0.9$ at 2 or more counted sizes, including the largest.
    * PASS-W if $r_W\le0.7$ at every counted size, with no rise of more than 0.1 from the smallest to the largest. UNDECIDED otherwise.
    * An attribution control (feature O, the inverse removed) was to run only after a PASS-W or UNDECIDED.
* **Extension, written before it ran.** DRF-K came out UNDECIDED at 2 seeds ($r_W=0.76$ at $V=10$, $1.62$ at $V=12$). Seed 33 was added to `nnbfW` and `drfK` at all sizes, the 3-seed medians replace the 2-seed ones, and the result is final at T1.
* **Budget.** RLIMIT_CPU of 600 s per invocation, plus a projection from 3 timed SR steps that aborts before the heavy part. The seven training invocations used 61 to 395 CPU-s each, on one thread, and about 24 CPU-minutes in total. Every invocation met the 10 CPU-minute cap, but the T1b total did not, which is flagged for the critic.

```
  DRF-K size 0: dEs_W 0.00302 dEs_drf 0.00706 r_W 2.342 counted False
  DRF-K size 1: dEs_W 0.01558 dEs_drf 0.02375 r_W 1.525 counted True
  DRF-K size 2: dEs_W 0.00934 dEs_drf 0.01995 r_W 2.136 counted True
VERDICT DRF-K: KILL
  DRF-X size 0: dEs_W 0.00302 dEs_drf 0.01632 r_W 5.412 counted False
  DRF-X size 1: dEs_W 0.01558 dEs_drf 0.03657 r_W 2.347 counted True
  DRF-X size 2: dEs_W 0.00934 dEs_drf 0.01324 r_W 1.417 counted True
VERDICT DRF-X: KILL
  DRF-H size 0: dEs_W 0.00302 dEs_drf 0.02415 r_W 8.007 counted False
  DRF-H size 1: dEs_W 0.01558 dEs_drf 0.02373 r_W 1.523 counted True
  DRF-H size 2: dEs_W 0.00934 dEs_drf 0.01780 r_W 1.905 counted True
VERDICT DRF-H: KILL
```

* **Reading.**
    * In the block above, `nnbfW` and `drfK` are 3-seed medians and `drfX` and `drfH` are 2-seed medians. Against the 2-seed `nnbfW` medians the X and H verdicts are unchanged: $r_W$ is 1.57 and 1.20 for X, and 1.02 and 1.61 for H.
    * Size 0 is not counted, because width 6 nearly solves $V=8$ ($\Delta E_{\mathrm{sign}}=0.003$).
    * Every DRF arm loses to matched width at every size. The information premise (held-out AUC gain from 0.10 to 0.25) did not turn into a trained gain.
    * DRF also reaches a higher $\Delta E_{\mathrm{tot}}$, and its energies are still falling at step 300. Part of the loss is therefore optimization speed per step. That is a property of the route at matched steps, and the rule is applied as written.
#### Route 6.7 reopened (broadcast after job 698178: `gvote` VOTE-USABLE, $z=-1.33$)
* **Route 6.7 as specified.** FRD factors are initialized to flip where the **projected exact** vote fires, then trained by SR.
  - Gate 0 (Novelty & Adoption): PASS for the initialization. The one-shot projected read-out is route 6.8, killed as a Lanczos-step sign.
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS as a design, with matched width as the foil.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): **FAIL on cost.**
    * The exact projected vote at $\mathbf{n}$ needs $P_\chi\psi(\mathbf{m}_e)$ for every active edge. That is $(1+E_{\mathrm{act}})\lvert G\rvert$ base passes per amplitude, and $(1+n_{\mathrm{conn}})(1+E_{\mathrm{act}})\lvert G\rvert$ per local energy.
    * This is FRD-K's $\mathcal{O}(V)$ wall (522 against 32 passes at $L=4$, unprojected) multiplied by $\lvert G\rvert=8$.
    * Extensivity also fails. Job 698178 measured `gvote` on one projected 4×4 state, and the `base` read-out (REFRAME-CARRIES, $b=3.59$) puts the node change in the point-group projection, which is the finite-size mechanism of Gate 4.
  - *Verdict*: **KILLED at T0** (Gate 4 cost). The only cost-honest variant is the frozen frame, tested next.
* **Route 6.7F (Operation A on the cost failure: frozen-frame vote, one inverse per amplitude).** It inherits 6.7's Gates 0 to 3. Gate 4 cost PASS on paper (the DRF count above). It went to T1.
* **T1 for 6.7F** (`t1_vote67.py`, output `t1_vote67_output.txt`, 106 CPU-s, one thread).
    * *Design*: the matched-width foil `nnbfW` is retrained with the T1b recipe (its $\Delta E_{\mathrm{sign}}$ reproduces the T1b JSONs exactly: validity holds). Its sign is flipped wherever its own hole-hop vote fires, with no gate and zero added parameters. So the comparison with the foil is exact.
    * *Projection*: the toys are unprojected, and projected toys are NO-TARGET. The vote is therefore computed on the state being scored, which is the toy analogue of the projected vote on $P_\chi\psi$.
    * *Statistic*: $q=$ median of $\Delta E_{\mathrm{sign}}[\mathrm{flipped}]/\Delta E_{\mathrm{sign}}[\mathrm{foil}]$.
    * *Rule* (written first): PASS iff $q_{\mathrm{frozen}}\le0.9$ at 2 or more counted sizes including the largest. KILL iff $q_{\mathrm{frozen}}\ge1.0$ at 2 or more counted sizes including the largest. The exact-vote $q$ is reported but decides nothing.

```
  size 0: med dEs base 0.00401 q_frozen 5.220 q_exact 0.809 counted False
  size 1: med dEs base 0.02330 q_frozen 1.745 q_exact 0.897 counted True
  size 2: med dEs base 0.01107 q_frozen 1.305 q_exact 0.918 counted True
VERDICT route 6.7F vote premise on the matched-width foil: KILL (valid True); CPU 106 s
```

* **Reading.**
    * On the trained foil, the exact vote helps in **6 of 6** (size, seed) pairs, but only by 8 to 19%, and the effect weakens with size ($q_{\mathrm{exact}}$ goes 0.81, 0.90, 0.92).
    * The frozen-frame vote hurts at every size ($q_{\mathrm{frozen}}$ from 1.3 to 5.2). This agrees with T2-b ($u=1.50$) and gives the shared cause of the DRF and 6.7F kills: **frozen-frame ratios of a trained backflow carry wrong-sign information.** Their 23% median error is concentrated where it matters.
    * The critic's caution is confirmed in a new form. The vote's usefulness depends on the base (harmful for the UHF anchor, mildly useful for a trained foil), and it lives only in the exact frame, which costs $\mathcal{O}(V)$.
#### Refill (portfolio at 0 live routes; Operations A and B, all five gates)
* **Route 8.1, 6.7L (Operation A on the 6.7 cost kill: exact vote on a finite-receptive-field backflow).** If pass 1 has a receptive field of radius $r$, a hop changes only the orbital rows within $r$ of the hop. Each exact neighbour ratio is then a rank-$\mathcal{O}(r^d)$ determinant-lemma update, $\mathcal{O}(r^dN)$ given $A^{-1}$.
  - Gate 0 (Novelty & Adoption): PASS. The low-rank machinery is SCALE's (arXiv:2604.25775). The vote-initialized sign factor on it was not found (round-6 searches).
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS as a design. The foil is a local NNBF of the same radius at matched width. The vote adds no parameters at initialization.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): cost PASS on paper, a constant factor $r^d$ (all $E_{\mathrm{act}}\propto N$ ratios cost $\mathcal{O}(r^dN^2)\le\mathcal{O}(r^dN^3)$). Extensivity is open: $q_{\mathrm{exact}}$ weakens from 0.81 to 0.92 on the toys.
  - *Sharpest gap*: T2-a showed SR leaves trained flips unused, and the gain at initialization is at most 8 to 10% at the counted sizes. Does SR keep or extend it?
  - *Verdict*: SURVIVED (GATED).
* **Route 8.2, DRF-E (Operation A on the DRF kills: swap the failing sub-component, frozen ratios, for exact ratios).** The same two-pass architecture, fed **exact** neighbour ratios through 8.1's low-rank updates.
  - Gate 0 (Novelty & Adoption): PASS. No iterative backflow fed by exact neighbour amplitude ratios was found. Continuum iterative backflow uses coordinates (Taddei et al.; Ruggeri et al.).
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS as a design, with `nnbfW` as the foil and DRF-K as the one-change control (frozen against exact).
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): cost as for 8.1. Extensivity is checked at T1.
  - *Sharpest gap*: AO-L (exact ratios in a positive factor) lost to width too, and DRF lost by 1.4 to 2.3 times, so exact features have a large deficit to recover.
  - *Verdict*: SURVIVED (GATED).
* **Route 8.3, DRF-WS (Operation A on the DRF kill: the optimization sub-component).** Train pass 1 at the foil's width to convergence, then add pass 2 with $W_4=0$, which is an exact identity start. The foil is `nnbfW`, widened to the same final parameter count and trained for the same total number of steps.
  - Gate 0 (Novelty & Adoption): PASS (as DRF).
  - Gate 1 (Conservation & Asymptotics): PASS.
  - Gate 2 (Baseline Gameability): PASS as a design.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): cost PASS, the same constant factor as DRF (measured above).
  - *Sharpest gap*: 6.7F shows the frozen features point the wrong way, so a warm start may only make DRF tie the foil.
  - *Verdict*: SURVIVED (GATED). It is ordered last (rule 3 of 3.5.2: 8.1 is the cheapest next tier).
* **Rejected at T0 while refilling.**
    * A threshold-scanned exact vote (one scalar gate chosen by an energy scan) is route 6.8, the Lanczos-step sign (Gate 0).
    * An MPS sign factor along a snake path has bounded entanglement in 2D (Gate 3).
    * A hash-table of learned sign flips memorizes configurations and is not extensive (Gate 4).
#### Route Portfolio (Round 7)

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 8.1 6.7L (exact vote, local backflow) | T0 | GATED | T1 code: local NNBF and low-rank exact ratios | T1: vote-initialized FRD on the local NNBF, trained; KILL if $\Delta E_{\mathrm{sign}}\ge0.9\times$ local-`nnbfW` at 2 of 3 sizes incl. $V=12$ | PASS: T2 on the 4×4 OBC instance against `nnbf6`; KILL: log, refill |
| 8.2 DRF-E (exact-ratio DRF) | T0 | GATED | the same low-rank code | the same rule, with DRF-K as the attribution control | PASS: T2; KILL: close the ratio-feature family |
| 8.3 DRF-WS (warm-started DRF) | T0 | GATED | none (`t1b_drf.py` plus a warm-start flag) | the same rule at matched total steps | PASS: attribution run with feature O, then T2; KILL: log |
| DRF-K, DRF-X, DRF-H (7.2 to 7.4) | T1 | KILLED(T1b, lose to matched width: $r_W$ 1.4 to 2.3) | — | — | Logged |
| 6.7 vote-initialized FRD | T0 | KILLED(T0, Gate 4 cost $(1+E_{\mathrm{act}})\lvert G\rvert$ passes) | — | — | Mutated to 6.7F and 8.1 |
| 6.7F frozen-frame vote | T1 | KILLED(T1, $q_{\mathrm{frozen}}$ 1.75 and 1.31) | — | — | Logged |

* **Live routes: 3, all GATED. No route is at T2, so no T2 pilot and no sbatch file was prepared.** Stage 3.5.4 requires a T1 pass before any T2.
#### Tier log (Round 7)
* T0: the DRF Gate 4 count (PASS) and the 6.7 Gate 4 count (FAIL). Gates 0 to 4 for 8.1, 8.2 and 8.3 on paper.
* T1: the DRF cost measurement (ratio 9.75 to 8.10, slope $-0.16$). The T1b checks (all pass). T1b training: DRF-K, DRF-X and DRF-H KILLED. 6.7F KILLED.
#### Local Autopsy Log (Round 7 additions)
* **DRF-K, DRF-X, DRF-H**: killed at T1b. A second backflow pass fed frozen Green's-function ratios loses to the same parameters spent on width, by 1.4 to 2.3 times at $V=10$ and 12.
* **6.7 and 6.7F**: killed at T0 (exact projected vote, $\mathcal{O}(V)\lvert G\rvert$ passes) and at T1 (frozen vote, $q$ from 1.3 to 1.7).
* **New anti-pattern (for the master doc)**: *an information test on the starting determinant does not predict a trained gain*. DRF passed the held-out AUC premise (UHF anchor) and lost to width once trained, because the trained backflow's frozen ratios carry wrong-sign information that the anchor's did not.
* **New anti-pattern (for the master doc)**: *frozen-frame ratios are not a cheap stand-in for exact ones on a trained backflow*. This held three times: T2-b ($u=1.50$), 6.7F ($q$ from 1.3 to 5.2) and DRF. Any ratio route must pay for exact ratios (low-rank updates on a local backflow) or drop them.

### Round 7 per-leaf critic (2026-11-16)
Scope: the Round 7 section above, `t1b_drf.py`, `t1_vote67.py`, their outputs and all 30 `results/t1b/*.json` files. New critic code in `tests/P1.2_unanchored_antisymmetry_r6/`: `critic_r7_premise.py`, the shared T1 premise of routes 8.2 and 8.3 measured on the **trained** matched-width foil. Its rules are in the docstring, written before the run; the output is `critic_r7_premise_output.txt` (94 CPU-s, one thread). An ED gap check of the toy ladders was also run inline (numbers below). No job was submitted.
#### Claims checked against code
* **Rules written first.** Both docstrings carry the rules. The seed-33 extension of `t1b_drf.py` was in the file when seed 33 ran: the file's last modification (10:32:43) is no later than the first line of `t1b_train_seed33.txt` (10:32:43), and the run imports the file.
* **Foil matching.** `make()` sets $H_2$ as the largest width with $n_{\mathrm{params}}\le T$. That gives $H_2=2$ at every size, and the DRF arms use 89% to 97% of the foil's parameters (`drfX` has 989 of 1110 at $V=12$). The pass-1 conventions of `DRF` and `LRAnsatz` (full mode) are identical, so the frozen ratios of `t1_vote67.py` are the trained foil's own Green's-function entries. `critic_r7_premise.py` confirms this numerically: the copied pass 1 reproduces the foil's amplitudes to machine precision at every size and seed.
* **Sizes counted per rule.** Yes. Size 0 has a foil median $\Delta E_{\mathrm{sign}}$ of 0.003 and is not counted, so sizes 1 and 2 decide. Per-seed values from the JSONs:

| size | `nnbfW` $\Delta E_{\mathrm{sign}}$ (seeds 11, 22, 33) | `drfK` (11, 22, 33) | `drfX` (11, 22) | `drfH` (11, 22) |
| :--- | :--- | :--- | :--- | :--- |
| 1 ($V=10$) | 0.0156, 0.0310, 0.0147 | 0.0117, 0.0238, 0.0293 | 0.0469, 0.0262 | 0.0122, 0.0353 |
| 2 ($V=12$) | 0.0128, 0.0093, 0.0092 | 0.0219, 0.0140, 0.0200 | 0.0121, 0.0144 | 0.0131, 0.0225 |

* **ED gaps of the toys** ($E_1-E_0$ in units of $t$): 0.066 at $V=8$, **0.0041 at $V=10$** (5×2, 4+4) and 0.118 at $V=12$. Replacements with two holes: 4×3 5+5 has a gap of 0.085, and 7×2 6+6 ($D=84084$) has 0.037. The 4-hole 5×2 3+3 has 0.051.
#### Shared premise of 8.2 and 8.3 on the trained foil (`critic_r7_premise.py`)
* **Design.**
    * *Base*: `nnbfW`, retrained with the T1b recipe. Its $\Delta E_{\mathrm{sign}}$ reproduces the T1b JSONs to $10^{-9}$.
    * *Label*: the foil's own wrong sign against $\psi_0$, weighted by $\psi_0^2$. Weighted logistic models are scored by held-out AUC.
    * *Baseline*: the best of three models. A uses the occupations. A2 adds the foil's own six hidden activations, which is what width already computes. C adds random $\pm1$ features.
    * *Test model*: B adds $g(R_e)$ for every hole-hop and exchange edge, with exact ratios (8.2) or frozen ratios (8.3).
* **Rule (written first).** KILL if the median $\mathrm{dAUC}<0.03$ at 2 or more counted sizes including the largest. PASS if $\mathrm{dAUC}\ge0.05$ at every counted size and $\mathrm{dAUC}(\text{largest})\ge0.5\,\mathrm{dAUC}(\text{smallest})$. A PASS only promotes the route to its trained T1.

```
  size 0: med dEs base 0.00401 counted False | med dAUC_ex 0.1523 | med dAUC_fr 0.0214
  size 1: med dEs base 0.02330 counted True | med dAUC_ex 0.1212 | med dAUC_fr 0.1186
  size 2: med dEs base 0.01107 counted True | med dAUC_ex 0.3127 | med dAUC_fr 0.2838
VERDICT route 8.2 DRF-E premise on the trained matched-width foil: PASS (counted dAUC_ex [0.1212, 0.3127])
VERDICT route 8.3 DRF-WS premise on the trained matched-width foil: PASS (counted dAUC_fr [0.1186, 0.2838])
valid True; CPU 94 s
```

* **Reading.**
    * At $V=12$ the frozen-ratio features predict where the trained foil's signs are still wrong with held-out AUC 0.93 and 0.94 (two seeds). The occupations plus the foil's own hidden units reach 0.65.
    * Exact ratios add little on top (AUC 0.96 and 0.97).
    * The foil's hidden units add nothing to the occupations: A2 is within 0.008 of A.
    * At $V=10$, seed 22 has wrong-sign weight 0.45. That is the near-degenerate partner's sign pattern (gap 0.0041), so the $V=10$ labels are unreliable. The $V=12$ result stands on its own.
#### Major objections
1. **The round-7 shared cause is refuted as stated.** The autopsy says that frozen-frame ratios of a trained backflow "carry wrong-sign information" and that "any ratio route must pay for exact ratios or drop them". The premise run shows the opposite for *features*. On the trained foil the frozen ratios locate its residual wrong signs almost as well as the exact ones ($\mathrm{dAUC}$ 0.28 against 0.31 at $V=12$), and the information grows with size. What T2-b ($u=1.50$) and 6.7F ($q_{\mathrm{frozen}}$ from 1.3 to 1.7) show is narrower: a **fixed thresholded vote** built from frozen ratios is harmful, because its fixed sign rule is wrong where the ratios are distorted. A learned model reads the same ratios correctly. The two anti-patterns are restated in the autopsy additions below. This matters for the leaf, because the T1b loss of DRF is not explained by its features.
2. **The T1b kills test a narrow, cold-started design, not DRF at its best allocation.** At matched parameters the design runs pass 1 at width 3 (half the foil's) and gives pass 2 only $H_2=2$ hidden units, because most of pass 2's parameters sit in its output layer ($H_2VN$). The ratio features are also non-stationary: at the start they are the UHF ratios, and they change as pass 1 trains. The worker noted that DRF energies were still falling at step 300. These are the two remaining explanations of the loss, and neither is a statement about sign mechanisms. The kills stand as T1 kills of routes 7.2 to 7.4 *as specified*, but they cannot carry a Saturation Diagnostic.
3. **The $V=10$ counted size is near-degenerate and noise-dominated.**
    * The 5×2 4+4 ladder has $E_1-E_0=0.0041t$, below the counting threshold of 0.005. The foil's three seeds span a factor of 2.1 there (0.0147 to 0.0310).
    * `drfK` beats `nnbfW` in 2 of 3 seeds at $V=10$. The DRF-K result there was decided by seed 33 alone: $r_W$ is 0.76 with 2 seeds and 1.525 with 3.
    * At $V=12$ the kill is clean for K and H: every DRF seed lies above every foil seed. For X it is not clean, since `drfX` seed 11 (0.0121) beats foil seed 11 (0.0128), although $r_W\ge1.2$ in every median reading.
    * **The kills stand by the rule as written**, but they rest on one clean size. Every later T1 in this leaf must assert a gap: replace 5×2 4+4 with 4×3 5+5 (gap 0.085) or 7×2 6+6 (gap 0.037), and assert $E_1-E_0\ge0.02t$.
4. **Route 8.1 is route 6.8 at initialization, with flips that SR does not use.** See the per-route verdicts.
5. **The 8.1 and 8.2 portfolio rows name the wrong foil.** "Local-`nnbfW`" is a weaker foil than the full-width NNBF that has beaten every mechanism in this leaf. A local pass 1 is the *cost device* for exact ratios, not the comparison. The foil stays the full-width NNBF at matched parameters (Gate 2). The local NNBF may be reported, but it decides nothing.
#### Minor objections
6. **T1b budget.** The seven invocations each met the 10 CPU-minute cap, but the T1b total was about 24 CPU-minutes. That is over the T1 cap, with no written exception. The verdicts are kept. The next T1 runs one seed set per invocation and writes the exception before a third seed.
7. **Matched compute.** The measured DRF cost ratio of 8 to 10 is numpy gather overhead. The asymptotic count gives about 2 (two passes, one inverse, an $\mathcal{O}(EN)$ gather). Before any T2, DRF must be re-timed in compiled form, as P4.1 did for its Pfaffian, and the T2 foil must be matched in wall time as well as in parameters.
8. **6.7F reading.** "The exact vote helps in 6 of 6 pairs" includes the uncounted size 0, and the exact vote flips at most 0.1% of the weight. The supported statement is narrower: at the counted sizes the one-shot exact vote removes 8% to 10% of the foil's sign error.
9. **No T1 rule was written for 8.3's warm start.** It is now written below, before any code exists.
#### Per-route verdicts
* **Route 8.1, 6.7L (exact vote on a local backflow): KILLED at T0.**
    * *Gate 0 (relabel)*: at initialization the ansatz is the trained base times a sign flip wherever the exact local-energy vote fires. That is route 6.8's one-step power (Lanczos) sign, killed at Gate 0, made cheap by locality. Locality fixes 6.7's cost, not 6.8's novelty.
    * *Training adds nothing usable*: the only change from 6.8 is that the flips become trainable factors. T2-a measured that SR leaves trained flips unused ($v=0.003$), and 6.7's T0 kill showed that SR would have to switch false alarms off through zero crossings.
    * *Gate 4 (finite size)*: the gain at initialization is a fixed-step power-method gain. On the trained foil it is $q_{\mathrm{exact}}=0.90$ and 0.92 at the counted sizes (0.81 at $V=8$). So it already sits at the route's own kill threshold of 0.9 before training, and it weakens with size. The 4×4 `gvote` gain ($z=-1.33$) is the same one-step mechanism on one projected state and carries no size trend.
    * *Failure modes*: 8.1 escapes the cost failure (a constant factor $r^d$) but not the width failure.
* **Route 8.2, DRF-E (exact-ratio DRF): SURVIVES T0, premise PASS at T1.**
    * *Cost*: it escapes the exact-vote cost failure only with a finite-receptive-field pass 1. Each exact neighbour ratio is then a rank-$\mathcal{O}(r^d)$ update, costing $\mathcal{O}(r^dN)$ given $A^{-1}$. A local energy needs $(1+n_{\mathrm{conn}})$ amplitudes, as for NNBF. The gradient through the neighbour amplitudes costs a constant multiple of the forward pass (reverse mode). So the cost is a constant factor in $r^d$, and Gate 4 cost passes on paper.
    * *Width*: neither escaped nor refuted on paper. Exact features carry 0.31 of held-out AUC on the trained foil, rising with size, against 0.28 for frozen ones.
    * *Order*: second. Its T1 needs new code (a local pass 1, low-rank exact ratios and their gradient), and its one-change advantage over 8.3 is small ($\Delta\mathrm{dAUC}=0.03$ at $V=12$). If 8.3 fails because a second pass cannot convert information that is present, 8.2 inherits that failure.
* **Route 8.3, DRF-WS (warm-started DRF): SURVIVES T0, premise PASS at T1, first to its trained T1.**
    * *Cost*: the measured constant factor (Gate 4 cost PASS). It never pays for exact ratios.
    * *Width*: the worker's sharpest gap, "the frozen features point the wrong way", is refuted by Major 1. The warm start removes the cold-start confound of Major 2: pass 1 *is* the converged foil, so its frozen ratios are exactly the features shown to locate its residual sign errors.
    * *Why first*: it is the cheapest test of the premise 8.2 shares, namely that a second pass converts the ratio information on a trained base into a sign gain at matched parameters.
* **Route 8.4, DRF-T (tied re-entry; refill, Operation A on DRF's parameter-allocation sub-component, Major 2).** The ratio features re-enter pass 1's hidden layer, $h=\tanh(xW_1+b_1)+\tanh([x,f]W_3+b_3)$, and the orbitals are $O_2=P_0+\mathrm{reshape}(hW_2+b_2)$. Only $W_3$ and $b_3$ are new, about $3V\times H_1$ parameters. So the matched foil is barely wider, and DRF no longer has to buy a second $VN$ output layer. With $W_3=0$ and $b_3=0$ the extra term is $\tanh(0)=0$, so a warm start from the converged foil is an exact identity.
  - Gate 0 (Novelty & Adoption): PASS, narrowly. It is weight-tied iterative backflow (Taddei et al., PRB 91, 115106 (2015)) with a lattice Green's-function input, which the round-6 searches did not find. The adoption wall of iterative backflow (twice the cost for little gain) is met by the new input, as for DRF.
  - Gate 1 (Conservation & Asymptotics): PASS. The features are diagonal in $\mathbf{n}$.
  - Gate 2 (Baseline Gameability): PASS as a design. The foil is the full-width NNBF, widened to the same parameter count from the same warm start. The attribution control is feature O.
  - Gate 3 (Hard-Core Shield): PASS, sign half only.
  - Gate 4 (Scaling & Relevance): the cost is DRF's (two hidden-layer evaluations, one inverse, the gather), a constant factor. Extensivity is checked at T1 on gap-asserted sizes.
  - *Sharpest gap*: re-entry through a shared $W_2$ moves the orbitals only along directions the foil already learned. It may lack the new directions a sign fix needs.
  - *Verdict*: SURVIVED (GATED). T1 after 8.3, with the same rule.
* **Pre-registered T1 for 8.3 (and 8.4), written before any code exists.**
    * *Sizes*: 4×2 3+3, 6×2 5+5 and 4×3 5+5, with the validity assert $E_1-E_0\ge0.02t$ (Major 3). A size is counted only if the foil's median $\Delta E_{\mathrm{sign}}\ge0.005$.
    * *Start*: the foil `nnbfW` (width 6) trained for 300 steps (seeds 11 and 22).
    * *Arms*, each trained for 300 more SR steps from that state: `ws` (pass 2 with $W_4=0$ and the largest $H_2$ that fits), `wsO` (the feature-O attribution control), and `wide` (the foil widened to at least the parameter count of `ws`, with zero new output rows, so it starts from the same state).
    * *Rule*: $r=\Delta E_{\mathrm{sign}}[\mathrm{ws}]/\Delta E_{\mathrm{sign}}[\mathrm{wide}]$. KILL if $r\ge0.9$ at 2 or more counted sizes including the largest. PASS-W if $r\le0.7$ at every counted size, with no rise above 0.1 from the smallest to the largest. UNDECIDED otherwise. After a PASS-W, the gain is credited to the inverse iff $\Delta E_{\mathrm{sign}}[\mathrm{ws}]\le0.9\,\Delta E_{\mathrm{sign}}[\mathrm{wsO}]$ at the largest size.
    * *Budget*: one invocation per size, each under 10 CPU-minutes. A third seed needs a written exception first.
#### Close or keep open: KEEP OPEN
* **Why not close now.** A Saturation Diagnostic needs a mechanism-level argument as well as the evidence pattern.
    * The pattern is real: matched width has beaten every sign mechanism tried in this leaf (FRD-K, AO-F, AO-L, DRF-K, DRF-X, DRF-H, 6.7 and 6.7F).
    * The only mechanism offered for it was "the cheap information is wrong-signed, and the right information costs $\mathcal{O}(V)$". That is refuted at T1. The information that locates the trained foil's residual sign errors exists at constant cost (one inverse per amplitude), beats what width computes (AUC 0.93 against 0.65 at $V=12$) and grows with size.
    * The one test that tried to convert it (T1b) was confounded by a cold start, a narrow second pass and a near-degenerate counted size.
    * The Persistence Mandate forbids closing on that record, and two routes (8.3 and 8.4) escape both recorded failure modes on paper.
* **First to T1: 8.3 DRF-WS**, under the rule above. It needs only a warm-start flag and a gap-asserted size list in `t1b_drf.py`.
* **Conditional Saturation Diagnostic, to be adopted if 8.3, 8.4 and 8.2 are all killed under the rule above.**
    * *Statement*: at ED size, the residual sign error of a trained NNBF is *locatable* at constant cost, because its own Green's-function ratios predict where it is wrong. But neither a second backflow pass nor a re-entry into the first converts that location into a lower sign error than the same parameters spent on width. Discontinuous carriers (flips, votes) are either left untrained by SR or are one-step power signs. Positive factors cannot move nodes. Continuous carriers lose to width.
    * *Mechanism-level argument*: P1.3 (the sign error is capacity-limited, with no basins) plus Major 1 (the information is present). Together they place the limit of a second pass in the *expressivity of the orbital update*, not in the information, and width enlarges the same update space more cheaply per parameter.
    * *Minimal extra structure needed to beat it*: a sign carrier that is continuous in its parameters, takes the ratio information as input, and adds **new orbital directions at a parameter cost below one width unit**. One example is a low-rank update $O_2=O_1+u(\mathbf{n})v(\mathbf{n})^{\mathsf T}$ whose factors come from local ratio features through a weight-shared kernel, so that its parameter count does not depend on $V$.
    * *Falsifying experiment*: any gap-asserted three-size T1 in which a ratio-fed carrier reaches $r\le0.7$ against the matched-width foil from the same warm start, with the attribution control showing that the gain needs the inverse.
#### Route Portfolio (Round 7 critic)

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 8.3 DRF-WS (warm-started DRF) | T1 premise | LIVE, premise PASS (frozen $\mathrm{dAUC}$ 0.12 and 0.28 on the trained foil) | warm-start flag and gap-asserted sizes in `t1b_drf.py` | the 8.3 T1 rule above ($r\ge0.9$ at 2 counted sizes incl. the largest) | PASS-W: attribution with `wsO`, compiled re-timing, then T2 on the 4×4 instance against `nnbf6`; KILL: run 8.4 |
| 8.4 DRF-T (tied re-entry) | T0 | GATED | 8.3's code plus the tied layer | the same rule | PASS-W: as for 8.3; KILL: if 8.2 is also killed, adopt the conditional Saturation Diagnostic |
| 8.2 DRF-E (exact-ratio DRF) | T1 premise | LIVE, premise PASS (exact $\mathrm{dAUC}$ 0.12 and 0.31) | local pass 1, low-rank exact ratios and their gradient | the same rule, with the full-width NNBF as foil (Major 5) | run after 8.3; PASS-W: as for 8.3; KILL: close the ratio-feature family |
| 8.1 6.7L (exact vote, local backflow) | T0 | KILLED(T0: route 6.8 at initialization, trained flips inert, $q_{\mathrm{exact}}$ at the 0.9 threshold and rising) | — | — | Logged |
| DRF-K, DRF-X, DRF-H (7.2 to 7.4) | T1 | KILLED(T1b as specified: a narrow, cold-started second pass; the kills rest on one clean size) | — | — | Logged |
| 6.7, 6.7F | T0, T1 | KILLED (accepted) | — | — | Logged |

* **Live routes: 3.** Routes 8.3 and 8.2 have passed the premise, and 8.4 is GATED. No route is at T2, so nothing is submitted.
#### Local Autopsy Log (critic additions)
* **Route 8.1**: killed at T0. It is the Lanczos-step sign of route 6.8 at initialization, made cheap by locality, plus trainable flips that SR leaves unused.
* **Restated anti-pattern (replaces "frozen-frame ratios are not a cheap stand-in for exact ones")**: *a fixed thresholded vote on frozen ratios is harmful on a trained backflow, but the same ratios used as learned features are informative*. Frozen and exact ratio features locate the trained foil's residual wrong signs about equally well ($\mathrm{dAUC}$ 0.28 against 0.31 at $V=12$).
* **Restated anti-pattern (replaces the causal clause of "an information test on the starting determinant does not predict a trained gain")**: the anti-pattern stands, but its cause is not wrong-signed features. On the trained base the information is present. What remains unexplained is the conversion step, which T1b tested only with a narrow, cold-started second pass.
* **New process anti-pattern**: *a counted toy size must have a gap*. The 5×2 4+4 ladder has $E_1-E_0=0.0041t$, so wrong-sign labels there can follow the near-degenerate partner (weight 0.45 for one foil seed). Every T1 in this leaf asserts $E_1-E_0\ge0.02t$.

### Round 8 — Trained T1 of routes 8.3 and 8.4 (2026-11-16)
Scope: the critic's pre-registered T1 for 8.3 (DRF-WS), then 8.4 (DRF-T) after 8.3 was killed. Both ran against the full-width NNBF foil on gap-asserted sizes. New code in `tests/P1.2_unanchored_antisymmetry_r6/`: `t1c_warm.py` (the foil stage, the warm-started arms, checks, budget ledger and verdict). Outputs are `t1c_*_output.txt` and `t1c_train_*.txt`, results are in `results/t1c/`, and the CPU ledger is `results/t1c/ledger.json`. Job 553724 was not touched. No job was submitted.
#### Design (docstring of `t1c_warm.py`, written before any run)
* **Sizes** (the critic's list), each with the gap assert $E_1-E_0\ge0.02t$ (`t1c_gaps_output.txt`):
    * size 0 is 4×2 3+3 ($V=8$, gap 0.066);
    * size 1 is 6×2 5+5 ($V=12$, gap 0.118);
    * size 2 is 4×3 5+5 ($V=12$, gap 0.085).
* **Largest size.** Sizes 1 and 2 have the same $V$ and $D$. "The largest" in the rule is size 2, the last in the critic's list and the only one with a 3-leg direction. This was fixed before any run.
* **Start.** The foil `nnbfW` (width 6, UHF start, 300 SR steps, seeds 11 and 22) is trained once, saved, and loaded by every arm. At size 1 it reproduces the T1b `2_nnbfW` values (0.01280 and 0.00934).
* **Arms**, each trained for 300 more SR steps from the saved foil:
    * `ws` (8.3) is the DRF with frame H, pass 1 the trained foil, and pass 2 starting at $W_4=0$. "The largest $H_2$ that fits" is fixed as the largest $H_2$ with $n_{\mathrm{params}}\le2T$. That gives $H_2=6=H_1$ at $V=12$ and 2196 parameters, against $H_2=2$ in T1b (critic Major 2).
    * `wide` is the foil widened to width 14 (2270 parameters). Its new hidden units have zero output rows, so it starts from the foil's state.
    * `dt` (8.4) is the tied re-entry, which is DRF with $H_2=H_1$ and $W_4$ tied to $W_2$, starting at $W_3=b_3=0$ (1476 parameters).
    * `widet` is the foil widened to width 9 (1545 parameters).
* **Rule.** This is the critic's rule, with $r=\Delta E_{\mathrm{sign}}[\text{route}]/\Delta E_{\mathrm{sign}}[\text{its wide foil}]$ as a median over two seeds.
* **Checks** (`t1c_checks_output.txt`, `t1c_checks2_output.txt`; all pass):
    * CHECK A: every warm start reproduces the foil's amplitudes exactly (relative difference 0).
    * CHECK B: the force of `dt` matches finite differences to $3.4\times10^{-8}$ (feature G) and $8.7\times10^{-7}$ (feature O). Its per-sample log-derivatives match the force rows to $2\times10^{-16}$.
    * CHECK D: the faster forward pass (full $G=O_1A^{-1}$ followed by one lookup, plus reuse of the amplitude pass in the force) matches the T1b `DRF` to $6\times10^{-15}$ in amplitudes and force.
* **Budget.** RLIMIT_CPU is 540 s per invocation. Three SR steps per arm are timed before the heavy part, and a task ledger aborts any invocation whose projection would take the task over 1800 CPU-s. One invocation was aborted by this gate and split. Actual use was **1687 CPU-s (28.1 CPU-minutes)**, including 30 s of hand-logged diagnostics. The largest invocation used 343 CPU-s, and every invocation stayed under 10 CPU-minutes.
#### Route 8.3 DRF-WS: KILLED at T1

```
  8.3 size 0: dEs foil 0.00401 wide 0.00075 ws 0.00066 r 0.873 counted False
  8.3 size 1: dEs foil 0.01107 wide 0.00866 ws 0.00858 r 0.991 counted True
  8.3 size 2: dEs foil 0.04962 wide 0.03801 ws 0.03766 r 0.991 counted True
VERDICT route 8.3: KILL
```

* **Per seed** (seeds 11 and 22): at size 1, `ws` has 0.00868 and 0.00848, and `wide` has 0.00811 and 0.00920. At size 2, `ws` has 0.03596 and 0.03936, and `wide` has 0.03732 and 0.03870. The seed ranges overlap at both sizes, so this is a tie, not a loss.
* **Reading.**
    * The warm start removes both T1b confounds. The second pass trains normally: $\Delta E_{\mathrm{tot}}$ falls from 0.073 to 0.052 at size 1, below `wide` at 0.065, and from 0.162 to 0.124 at size 2.
    * But its sign error ties width ($r=0.99$ at both counted sizes). The gain of the second pass goes into amplitudes, not signs.
    * This happens although its input features locate the foil's wrong signs with held-out AUC 0.93 (critic premise).
    * The conversion step therefore fails when it is given a converged base, a pass as wide as pass 1, and gapped sizes.
#### Route 8.4 DRF-T: T0 kept, T1 UNDECIDED
* **T0 after the 8.3 kill.** The sharpest gap ("re-entry through the shared $W_2$ moves the orbitals only along directions the foil already learned") was not settled by 8.3, because 8.4 changes the parameter allocation and 8.3 does not. Gates 0 to 4 stand as the critic wrote them, so 8.4 went to T1.
* **As run, the arm did not train.** At $V=12$ the `dt` arm never left its warm start. The first SR step of the recipe (shift $10^{-3}$, lr 0.05) is clipped to norm 1 and raises $E$ by 0.68. The best-energy guard of `p21_harness.optimize` then returns the start, so `dt` at size 1 equals the foil to every printed digit.
* **Probes** (8 steps from the size-1 seed-11 start): lr 0.01 and 0.003 also diverge, a shift of $10^{-2}$ diverges, and a shift of $10^{-1}$ descends monotonically.
* **Amendment, written in the docstring before any `dts` run.** Arm `dts` is `dt` trained with an SR shift of 0.1, the only change. `widet` keeps the recipe, because it trained normally. This change can only favour the route, so a KILL of `dts` would be final. A PASS-W would also need `widet` retrained at shift 0.1. The `dt` rows decide nothing. Size 0 of `dts` was not run, to stay inside the task cap; it is uncounted (foil median 0.004) and decides nothing.

```
  8.4-as-run size 1: dEs foil 0.01107 widet 0.00821 dt 0.01107 r 1.348 counted True
  8.4 size 1: dEs foil 0.01107 widet 0.00821 dts 0.00979 r 1.191 counted True
  8.4 size 2: dEs foil 0.04962 widet 0.04450 dts 0.03985 r 0.895 counted True
VERDICT route 8.4: UNDECIDED
```

* **Per seed.** At size 1, `dts` has 0.01053 and 0.00905, and `widet` has 0.00744 and 0.00899, so `dts` loses for both seeds. At size 2, `dts` has 0.03917 and 0.04052, and `widet` has 0.04622 and 0.04278, so `dts` wins for both seeds.
* **Reading.**
    * At size 2, $r=0.895$ misses the KILL threshold of 0.9 by 0.005. That decides the verdict, since KILL needs $r\ge0.9$ at the largest size. No reading is near PASS-W ($r\le0.7$ everywhere).
    * The size-2 win is not attributable. `dts` and `widet` differ in optimizer shift as well as in architecture, and `widet` barely improved on the foil in energy at size 2 ($\Delta E_{\mathrm{tot}}$ from 0.162 to 0.150, against 0.114 for `dts`).
    * The wider foil of 8.3 (`wide`, width 14, 0.0380) beats `dts` (0.0399) at size 2.
* **Pre-registered tie-break (written now; not run, because the task cap leaves 113 CPU-s).** Retrain `widet` with SR shift 0.1 at sizes 1 and 2 (arm `widets`, about 140 CPU-s) and apply the same rule with $r=\mathrm{dts}/\mathrm{widets}$. KILL closes 8.4. PASS-W needs the `dtsO` attribution run. UNDECIDED again kills 8.4 at T1 for lack of a size trend, with no further seeds.
#### Route 8.2 DRF-E: KILLED at T0 (inheritance clause)
* **Inheritance.** The critic wrote this clause before 8.3 ran: "If 8.3 fails because a second pass cannot convert information that is present, 8.2 inherits that failure." 8.3 failed in exactly that way. The second pass trained, its frozen features carried AUC 0.93 of wrong-sign information, and the sign error tied width ($r=0.99$).
* **Size of the exact-ratio advantage.** Exact features add only $\Delta\mathrm{dAUC}=0.03$ over frozen ones at $V=12$ (AUC 0.96 against 0.93). For 8.2 to reach $r\le0.7$, that small increment would have to produce a 30% sign gain where the larger frozen information produced none.
* **Budget.** The T1 would need new code (a local pass 1, low-rank exact ratios and their gradient), and the task's T1 budget is spent.
* **Status.** The T0 kill is flagged for the per-leaf critic.
#### Gate 4 and T2
No route passed T1 (PASS-W), so no Gate 4 extensivity run, no compiled re-timing and no T2 job file were prepared (Stage 3.5.4 requires a T1 pass first).
#### Saturation Diagnostic: condition NOT met
The critic's conditional Saturation Diagnostic needs 8.2, 8.3 and 8.4 all killed. Routes 8.3 and 8.2 are killed, but 8.4 is UNDECIDED. So the diagnostic is **not adopted**, and the leaf is not yet ready to close. If the `widets` tie-break kills 8.4, the conditional text applies unchanged, with one addition to its evidence: the conversion fails even with a warm start, a pass-2 width equal to pass 1, and gapped sizes (8.3, $r=0.99$).
#### Route Portfolio (Round 8)

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 8.4 DRF-T (tied re-entry) | T1 | LIVE, T1 UNDECIDED ($r$ 1.19 at 6×2, 0.895 at 4×3; arm trained at SR shift 0.1) | CPU budget for the `widets` tie-break (about 140 CPU-s) | tie-break: $r=\mathrm{dts}/\mathrm{widets}$, the critic's rule; UNDECIDED again counts as KILL | KILL: adopt the conditional Saturation Diagnostic and close; PASS-W: `dtsO` attribution, Gate 4 extensivity, compiled re-timing, then T2 on the 4×4 OBC instance |
| 8.3 DRF-WS (warm-started DRF) | T1 | KILLED(T1: ties width, $r=0.99$ at both counted sizes; the pass trains, but the gain goes to amplitudes) | — | — | Logged |
| 8.2 DRF-E (exact-ratio DRF) | T0 | KILLED(T0: the critic's inheritance clause after 8.3; exact features add only 0.03 AUC) | — | — | Logged; the critic may reopen it |
| 8.1, DRF-K, DRF-X, DRF-H, 6.7, 6.7F | T0, T1 | KILLED (accepted) | — | — | Logged |

* **Live routes: 1** (8.4, UNDECIDED). The portfolio is below 3. Refill is deferred to the orchestrator, because the conditional Saturation Diagnostic decides whether the leaf closes after the 8.4 tie-break.
#### Tier log (Round 8)
* T0: the 8.4 sharpest gap was re-read after the 8.3 kill (kept). 8.2 was killed by the inheritance clause.
* T1, with the CPU-s for each step:
    * gap asserts: 3 sizes, all pass;
    * foil training: 159;
    * checks A, B and D: 38;
    * 8.3 at sizes 0 to 2: 678, plus 4 for an invocation the budget gate aborted, verdict KILL;
    * 8.4 as run: 343, invalid because the arm never trained;
    * stability probes: 30;
    * 8.4 `dts` at sizes 1 and 2 and `widet` at size 2: 435, verdict UNDECIDED.
* The T1 total was 1687 CPU-s, within the 1800 CPU-s cap.
#### Local Autopsy Log (Round 8 additions)
* **Route 8.3**: killed at T1. A warm-started second pass, as wide as the first and fed frozen ratios that locate the foil's wrong signs with AUC 0.93, lowers the total energy faster than width but leaves the sign error equal to width's ($r=0.99$ at 6×2 and 4×3).
* **New anti-pattern (for the master doc)**: *located sign errors are not corrected sign errors*. Even when a feature locates where a trained backflow's sign is wrong (AUC 0.93), SR on a ratio-fed orbital update spends the new capacity on amplitudes. The sign error moves only as much as width moves it.
* **New process anti-pattern**: *an exact-identity warm start through a shared, trained output layer can be SR-unstable*. With the tied re-entry and $W_3=0$, the first SR step at shift $10^{-3}$ is clipped and raises $E$ by 0.68, and the best-energy guard silently returns the start. Every warm-started arm must log that it left its start (check $\Delta E_{\mathrm{tot}}$ against the start) before its row is scored.

#### Route 8.4 Tie-break Result and Leaf Closure (2026-11-16)

* **Run.** The pre-registered tie-break (`widets`: `widet` retrained at SR shift 0.1 at sizes 1 and 2) was not implemented in `t1c_warm.py`, so the orchestrator added it with three one-line edits that change no other arm or rule: `widets` in `SHIFT_ARMS` (0.1), in the allowed arm list, and a `"8.4-tiebreak": ("dts", "dtsO", "widets")` row in `ROUTE_ARMS`. It then ran `python3 t1c_warm.py --mode train --sizes 1,2 --arms widets --task_cap 2000` and `--mode verdict` (saved as `t1c_tiebreak_verdict.txt`). The task cap was raised from 1800 to 2000 CPU-s for this one pre-registered run; the ledger ends at 1812 CPU-s.

| size | $\Delta E_{\mathrm{sign}}$ foil | `widets` | `dts` (route) | $r=\mathrm{dts}/\mathrm{widets}$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 (6×2, 5+5) | 0.01107 | 0.00993 | 0.00979 | 0.985 |
| 2 (4×3, 5+5) | 0.04962 | 0.04170 | 0.03985 | 0.956 |

* **Verdict: route 8.4 KILLED.** With the optimiser setting matched, the earlier $r=0.895$ at size 2 rises to 0.956. The apparent win came from the shift, not the tied re-entry.
* **All three refill routes (8.2, 8.3, 8.4) are now killed**, which is the condition of the Round 7 critic's conditional Saturation Diagnostic. **P1.2 closes with that diagnostic**, with the evidence the Round 8 worker added: the conversion fails even with a warm start, a second pass as wide as the first, gapped sizes, and a matched optimiser.
* **What the leaf established.** The trained backflow's own frozen ratios *locate* its remaining sign errors (held-out AUC 0.93 at $V=12$, rising with size). No tested mechanism converts that location into a lower sign error faster than spending the same parameters on backflow width. A fixed vote is harmful; learned re-entry ties width ($r$ 0.96–0.99); exact projected votes cost $(1+E_{\mathrm{act}})\lvert G\rvert$ passes per amplitude. The minimal missing ingredient and the falsifying experiment are those stated in the critic's conditional diagnostic.

### Closed-leaf falsifier (2026-11-16)
Scope: the idle-time queue item "falsifiers of closed leaves" (SKILL.md Stage 3.5.3 step 3), plus the cheapest projected-control re-check asked for by the cross-node critic (Majors 3 and 4: unprojected toys and a 300-step learning-speed statement). Code: `tests/P1.2_unanchored_antisymmetry_r6/t1e_closed.py`. Both rules are in its docstring and were written before any run. Outputs: `t1e_checks_output.txt`, the `t1e_trainA_size*.txt` and `t1e_foilP_size*.txt` logs, and `t1e_verdict_output.txt`; results are in `results/t1e/`. No job was submitted and no running job was touched.
#### Part A: the Saturation Diagnostic's own falsifier (route LRK)
* **Carrier.** LRK is the diagnostic's "minimal extra structure", built as the diagnostic describes. Pass 1 is the trained foil. Its frame-H ratio features (the DRF features, which locate the foil's wrong signs with AUC 0.93) feed a weight-shared per-site kernel. The kernel input is $\phi_v=(x_{\uparrow v},x_{\downarrow v},f_{0v},f_{1v},f_{2v})$, with $K=8$ hidden units. It outputs $U_s(\mathbf n)$ and $T_s(\mathbf n)$, both $V\times R$ with $R=2$, and the orbitals become $O_{2,s}=O_{1,s}+U_s\,(T_s^{\mathsf T}O_{1,s})$. Every orbital thus gains configuration-dependent site directions.
    * The carrier has 120 parameters at every $V$, below one width unit (145 at $V=12$).
    * Its warm start is exact ($U=0$).
    * The foil `wid` is the trained foil widened to 1255 parameters, so $P/D=0.075$ with $D=16632$ (unprojected).
    * Both arms use the same warm start, 300 more SR steps, the t1c recipe, and seeds 11 and 22.
* **Checks** (all pass). Every warm start reproduces the foil exactly. The LRK force matches finite differences to $3.4\times10^{-7}$ (feature G) and $3.3\times10^{-8}$ (feature O). The per-sample log-derivatives match the force rows to $3\times10^{-16}$.
* **Rule A** is the diagnostic's falsifier under t1c's rule, with $r=\Delta E_{\mathrm{sign}}[\mathrm{lrk}]/\Delta E_{\mathrm{sign}}[\mathrm{wid}]$. Only a PASS-W at the counted sizes plus an attribution PASS falsifies the diagnostic.

| size | foil $\Delta E_{\mathrm{sign}}$ | `wid` | `lrk` | $r$ | counted |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (4×2, 3+3) | 0.00401 | 0.00168 | 0.00204 | 1.21 | no |
| 1 (6×2, 5+5) | 0.01107 | 0.00850 | 0.00892 | 1.05 | yes |
| 2 (4×3, 5+5) | 0.04962 | 0.04267 | 0.04269 | 1.00 | yes |

* **Verdict A: KILL. The diagnostic's falsifier did not fire, and the diagnostic stands.** LRK lowers the total energy more than width does (at size 2, $\Delta E_{\mathrm{tot}}$ 0.120 and 0.123 against 0.134 and 0.137), but it leaves the sign error unchanged. That is the same pattern as 8.3: the ratio-fed update goes to amplitudes, not nodes. The attribution control `lrkO` was not run, because the rule calls for it only after a PASS-W.
#### Part B: projected-control re-check of the 8.3 kill
* **Design.** The space-group projection $P_\chi$ has $\lvert G\rvert=4$; ED puts $\chi$ in the fully symmetric irrep of $M_x$ and $M_y$. SR runs on $\psi_P$. The projected per-sample log-derivatives match $\mathrm{force}(P_\chi e_n)/\psi_P(n)$ to $2\times10^{-15}$ for LRK, DRF and NNBF. The projected cold foil is NNBF width 6 after 300 projected SR steps. It is followed by `wsP` (8.3) against `wideP` (width 14) for 600 steps. The comparison counts only if the projected foil's median $\Delta E_{\mathrm{sign}}\ge0.005$, tried at size 2 first and then at size 1.
* **Result: NO-TARGET-P at both toys.** The projected foil's median $\Delta E_{\mathrm{sign}}$ is 0.00448 at 4×3 (seeds 0.00410 and 0.00486) and 0.00166 at 6×2 (0.00111 and 0.00221). The unprojected foils had 0.0496 and 0.0111. So the 8.3 comparison cannot be re-run under projection on these toys, and `wsP` and `wideP` were not trained.
* **Reading.**
    * About 90% of the unprojected sign error on which the closing T1s were scored is symmetry contamination, which projection removes. This confirms the round-2 anti-pattern quantitatively for this leaf.
    * The closing kills compare two unprojected arms, so their ratios stay fair (cross-node critic Major 4). But "no mechanism converts location faster than width" is a statement about **unprojected** $V\le12$ toys at 300 steps only.
    * On projected toys of ED size, NNBF width 6 already leaves less than 0.005 of sign error, so no projected toy of this family can test the diagnostic. The projected evidence on the target instance remains P1.3 and the 553724 T2.
* **Budget.** The leaf ledger is 866 CPU-s (14.4 CPU-minutes) out of 1800. The largest invocation used 318 CPU-s, with one thread throughout. The budget gate aborted one invocation before any heavy work (projected 667 CPU-s). It was rerun after the LRK forward pass was switched to the FastDRF Green's-function gather, which halved the step cost.
* **Leaf status: stays CLOSED.** The diagnostic's falsifier was run and killed, and the projected re-check is NO-TARGET-P. Neither reopens the leaf. A scope note was added to the status line.

* **Job 553724 cancelled (2026-11-16).** Its pre-registered verdict (FAIL) was already decided from the stored seeds and the leaf is closed, so finishing it could not teach anything new. Five tasks had completed, three were cancelled.

### Reopening review (2026-11-16)
Scope: the user's question "is each subproblem unsolvable, or already solved?" A closed leaf is neither. This review turns the Saturation Diagnostic's named ingredient and untried regimes into new routes, gates them, and tests the survivors at T1 only. Code is in `tests/P1.2_unanchored_antisymmetry_r9/`: `t1f_phase.py` (route RPC), `t1f_curv.py` (the phase-family premise) and `t1g_projnarrow.py` (route LRK-PN). Each rule is in its docstring and was written before the run it governs. Outputs are the `t1f_*` and `t1g_*` text files, results are in `results/t1f/` and `results/t1g/`, and the CPU ledger is `results/t1f/ledger.json`. No job was submitted, no job was cancelled, and no earlier file was edited.
#### 1. What the diagnostic names
* **Named ingredients.**
    * I1: a sign carrier that is continuous in its parameters, is fed the ratio information, and adds new orbital directions for less than one width unit. LRK was built to this description and tied width ($r$ 1.05 and 1.00).
    * I2: an attribution control showing that a gain needs the inverse. It has never run, because no arm reached PASS-W.
    * I3 (implicit in "located sign errors are not corrected sign errors"): a carrier whose new capacity goes to the nodes rather than to the amplitudes.
* **Untried regimes.**
    * U1: projected training with a target. Width 6 leaves less than 0.005 of projected sign error (NO-TARGET-P), so the diagnostic was scoped to unprojected toys.
    * U2: horizons past 300 steps, and comparisons at equal wall time rather than equal parameters (cross-node critic Major 3).
    * U3: a carrier that changes the sign without passing through zero. Barrier (b) is cited as the mechanism, but the only carriers tried without a zero crossing were Route 4 (fitted to oracle labels) and the complex detour 6.4 (two independent parts).
    * U4: exact ratios. Route 8.2 was killed by inheritance and never run.
    * U5: sizes above $V=12$, and the $4\times4$ target instance.
#### 2. New candidate routes and gates
* **Route 9.1, RPC: ratio-fed phase carrier (Operation B.1, real frame to phase frame; built from I3 and U3).**
    * *Ansatz.* $\psi=\psi_F(\mathbf n)\,e^{i\varphi(\mathbf n)}$, where $\psi_F$ is the trained foil, held frozen. The phase is $\varphi(\mathbf n)=\pi\sum_v[\tanh(\phi_vW_k+b_k)\cdot a+c]$, where $\phi_v$ is LRK's per-site input (occupations plus frame-H frozen ratio features). The kernel is weight-shared, with $K=8$ hidden units and 57 parameters at every $V$. The sign is read out as $s=\mathrm{sgn}(\psi_F\cos\varphi)$.
    * *Why it addresses I3 and U3.* A sign changes by rotation, so $\lvert\psi\rvert$ never passes through zero. For real symmetric $H$, $E[\varphi]=\sum_{nm}\psi_n\psi_mH_{nm}\cos(\varphi_n-\varphi_m)/N$, where $N=\sum_n\psi_n^2$. The real and imaginary parts share one amplitude, so the energy couples them. In the killed route 6.4 they were independent networks.
    * Gate 0: PASS, narrowly. A WebSearch on 2026-11-16 found a neural sign-correcting factor of the occupations on a Slater determinant (arXiv:2008.00118), modulus–phase networks (Szabó & Castelnovo), RBM times pair product (PRB 96, 205152) and determinant-free signs (arXiv:2108.08631). It found no phase carrier fed by Green's-function ratios. The adoption wall of phase networks is poor generalization. The mechanism offered against it is that the inputs already separate the wrong signs (held-out AUC 0.93). The route is not Route 4: its inputs are non-local, so it is not a correlator product state, and it is trained variationally, not on oracle labels.
    * Gate 1: PASS. The carrier is diagonal in $\mathbf n$.
    * Gate 2: PASS as a design. The foil is t1e's matched `wid`, which has 145 added parameters against 57, so the comparison leans toward KILL. The attribution control is `rpcO`, and the premise ceiling is `free`, an unrestricted phase on each configuration.
    * Gate 3: PASS, sign half only.
    * Gate 4: the parameter count does not depend on $V$, and $\varphi$ is a sum of per-site terms, so it is extensive. The cost is the foil plus the DRF feature gather (a constant factor) plus $\mathcal O(VK)$.
    * *T0 lemma.* At $\varphi=0$ the gradient vanishes identically, because $\mathrm{Re}\,\langle ig\psi\rvert H-E\rvert\psi\rangle=0$ for real $\psi$. The second-order change is $\varepsilon^2q(g)$, with $q(g)=-\frac{1}{2N}\sum_{nm}\psi_n\psi_mH_{nm}(g_n-g_m)^2$, which is negative only along phase patterns whose differences sit on frustrated bonds. So the real foil is a saddle or a minimum in phase space, and that decides the family.
    * *Verdict*: SURVIVED to T1.
* **Route 9.2, LRK-PN: the diagnostic's carrier in the projected, capacity-limited regime (Operation A from U1).**
    * *Ansatz.* LRK unchanged, on a **narrow** projected foil (NNBF width 2, 530 parameters). The narrow foil restores a target on ED-size toys at a capacity-to-difficulty ratio closer to the $4\times4$ instance, where projected width 6 leaves 0.012.
    * Gate 0: as for LRK, the diagnostic's own carrier. Only the regime is new.
    * Gate 1: PASS.
    * Gate 2: PASS. The foil `widN` is the narrow foil widened to 675 parameters, against 650 for LRK. $P/d_{S\chi}\approx0.16$, which is outside the sector-fitting regime.
    * Gate 3: PASS, sign half only ($t'=-0.25$, doped, 6×2 and 4×3).
    * Gate 4: the carrier has 120 parameters at every $V$, and its cost is a constant factor.
    * *Verdict*: SURVIVED to T1.
* **Route 9.3, node optimization on a fixed-node objective (Operation B.3, self-healing DMC transferred to the lattice; built from I3).** KILLED at Gate 0. It is self-healing DMC (Reboredo, Hood & Kent, arXiv:0808.3417), and a 2026 assessment shows that it converges to a wrong node in any finite basis (arXiv:2609.01301). Its lattice form needs FN-GFMC with the network as guide, which is the P1.1 Gate 0 kill (Loehr & Clark, arXiv:2510.26906).
* **Route 9.4, converged-horizon or equal-wall-time LRK (Operation A from U2).** KILLED at T0 under the value-of-information rule.
    * Going past 300 steps at ED size moves the width arm toward sector fitting.
    * The equal-CPU arm `widNL` below already shows that extra steps give width most of LRK's gain.
    * A tie is predicted with probability at least 0.8, and a tie leaves the decision unchanged.
    * The evidence that would overturn this prediction is a compiled re-timing that puts the LRK step cost below twice the width step cost.
* **Not reopened.** U4 (exact ratios, Route 8.2) stays killed by the inheritance clause, because the frozen features it would improve on already carry AUC 0.93 and did not convert. U5 is T2 scope, so it comes only after a T1 PASS-W.
#### 3. T1 results
* **Validity** (`t1f_checks_output.txt`). The copied pass 1 reproduces the foil exactly (relative difference 0). The kernel gradient matches finite differences to $1\times10^{-10}$. $E[\varphi=0]$ equals the foil energy. The gradient at $\varphi=0$ is exactly 0, which confirms the T0 lemma. Every LRK-PN warm start reproduces the projected foil to below $10^{-10}$, and the t1c gap asserts pass ($\ge0.02t$).
* **RPC** (`t1f_verdict_output.txt`; medians of foil seeds 11 and 22; $r=\Delta E_{\mathrm{sign}}/\Delta E_{\mathrm{sign}}[\mathrm{wid}]$):

| size | foil | `wid` | `free` (ceiling) | $r_{\mathrm{free}}$ | `rpc` | $r$ | counted |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (4×2) | 0.00401 | 0.00168 | 0.00189 | 1.12 | 0.00401 | 2.38 | no |
| 1 (6×2) | 0.01107 | 0.00850 | 0.00999 | 1.17 | 0.01107 | 1.30 | yes |
| 2 (4×3) | 0.04962 | 0.04267 | 0.03272 | 0.77 | 0.04962 | 1.16 | yes |

* **RPC verdict: KILL (PREMISE-OK).** In every run the kernel returned to a phase with $\lvert\cos\varphi\rvert\ge0.996$ and **zero** flipped weight, so RPC equals the foil. The unrestricted phase does descend, flipping 1.7% of the weight at 4×3, and beats width there ($r_{\mathrm{free}}=0.77$). It does not beat width at 6×2.
* **Phase-family premise** (`t1f_curv.py`, written after the RPC kill and before its own run). The quantity is $\lambda(S)$, the minimum over the span $S$ of $q(g)/\mathrm{var}_{\psi^2}(g)$, with medians over seeds:

```
median size 0: ratio +0.1777 ratio2 +0.1251 occ +0.1966 rand +2.5428 free_dir -1.7873
median size 1: ratio +0.3522 ratio2 +0.2632 occ +0.3730 rand +4.4720 free_dir -1.7642
median size 2: ratio +0.3051 ratio2 +0.2173 occ +0.3341 rand +4.4861 free_dir -1.6969
VERDICT phase-family second-order premise: FAMILY-KILL; CPU 11 s
```

* **Reading.**
    * The trained foil is a saddle in phase space. Descending directions exist: the unrestricted direction gives about $-1.7$.
    * But every phase linear or quadratic in the per-site ratio features, without weight sharing ($15V$ columns), has $\lambda>0$ at every size.
    * The ratio features lower $\lambda$ only a little relative to occupations (0.22 against 0.33 at 4×3).
    * So the descent directions are configuration-specific and are not available to any extensive phase carrier built on these features. That is why RPC returned to $\varphi\approx0$.
    * **The phase family is KILLED at T1.**
* **LRK-PN** (`t1g_foilN_output.txt`, `t1g_verdict_output.txt`). The narrow projected foil has a target at both sizes ($\Delta E_{\mathrm{sign}}$ 0.0095 and 0.0159), so U1 is testable for the first time.

| size | foilN | `widN` (300 steps) | `lrkN` (300 steps) | $r$ | `widNL` (1150 steps, equal CPU) | $q=\mathrm{lrkN}/\mathrm{widNL}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 (6×2) | 0.00948 | 0.00503 | 0.00281 | 0.56 | 0.00313 | 0.90 |
| 2 (4×3) | 0.01591 | 0.00862 | 0.00882 | 1.02 | — | — |

* **LRK-PN verdict: UNDECIDED** under RULE N, and PASS-W is out of reach.
    * At 6×2, LRK wins clearly at equal steps, and both seeds are separated (0.0027 and 0.0029 against 0.0045 and 0.0056). Its $\Delta E_{\mathrm{tot}}$ is lower too (0.028 to 0.030 against 0.042 to 0.044), so the gain is general, not specific to the sign.
    * At 4×3 it ties ($r=1.02$, with both seeds within the foil's spread).
    * The equal-CPU arm (amendment written before it ran; LRK costs 3.8 times as much per numpy step) closes most of the 6×2 gap. $q=0.90$ sits on the COST-ARTEFACT edge, which is MIXED by the rule.
    * Since 4×3 already ties at equal steps, no seed extension or compiled re-timing can produce $r\le0.7$ at every counted size. The route stops at T1.
* **Budget.** The ledger total is 1488 CPU-s (24.8 CPU-minutes) of the 30-minute cap, all on one thread. The largest invocation used 416 CPU-s, which is under the 10-minute cap, and every heavy part was gated by a timed projection first.
#### 4. T2
No route passed T1, so there is no value-of-information note and no T2 design (Stage 3.5.4 needs a T1 pass first).
#### 5. Recommendation: STAY-CLOSED
The new evidence adds four things to the diagnostic:
* **Barrier (b) is not what binds.** A carrier that changes the sign without any zero crossing (a phase) fails for a different reason. At the trained foil, the descending phase directions exist but lie outside every extensive span of the ratio features ($\lambda$ from $+0.13$ to $+0.35$). The obstruction is the **non-locality of the needed sign correction in feature space**, not the zero-crossing barrier. The diagnostic's "minimal extra structure" (I1) should now read: *a carrier whose function class contains the configuration-specific descent directions*. Weight-shared per-site kernels on occupations and ratios do not.
* **Unrestricted phases are a finite-size device.** They beat width at 4×3 ($r_{\mathrm{free}}=0.77$) but use $D$ parameters, and they lose at 6×2. So that gain fails Gate 4.
* **The diagnostic now holds in the projected regime too (U1), up to a learning-speed effect on one geometry.** On narrow projected foils, LRK ties width at 4×3. At 6×2 it beats width per step ($r=0.56$), but only by 10% at equal CPU time ($q=0.90$).
* **Scope is unchanged.** All the evidence is at ED size ($V\le12$, $t'=-0.25$, $J=0.4$). The $4\times4$ target is untested by any carrier in this family.
* **What would reopen the leaf.** A carrier whose function class provably contains the free-phase descent direction while keeping a parameter count that does not depend on $V$. A cheap test is $\lambda(S)<0$ in `t1f_curv.py` for a candidate feature span $S$, at both counted sizes, before any training.

### Reopening-ledger row 4: curvature screen and trained carrier (2026-11-17)
Row 4 of the reopening ledger is the restated missing ingredient: a sign carrier whose function class contains the configuration-specific phase-descent directions, with a parameter count independent of $V$. Both steps below are T1, interactive (no Slurm job, no compute-gate entry), single-thread, with each rule written before its run in the test directory's `RULE.md`. Instances: the unprojected $6\times2$ and $4\times3$ $t$–$t'$–$J$ toys ($5+5$, $t'=-0.25$, $J=0.4$, gap at least $0.02t$), the frozen trained width-6 foils of t1c, seeds 11 and 22, exact sums.

#### Tier log (row 4)
* **r10, curvature screen (T1, 3.3 CPU-min; `tests/P1.2_unanchored_antisymmetry_r10/`).**
    * *Span S.* Not weight-shared: R9.1's per-site span (`ratio2`), plus cross-site products $f_a(i)f_b(j)$ of the five per-site features on every Hamiltonian bond (25 columns per bond), plus single edge ratios $g(R_e)$ and two-hop products $g(R_e)g(R_{e'})$ over edges sharing a site, with $g(R)=R/(1+\lvert R\rvert)$ on exact foil ratios. Rank 1464 at $6\times2$ and 1760 at $4\times3$. Control `rand`: random $\pm1$ columns of equal rank.
    * *Rule.* $\lambda(S)=\min_{g\in S}q(g)/\mathrm{var}_{\psi^2}(g)$ (R9.1's quantity and code). FAMILY-KILL if $\lambda(S)\ge0$ at both counted sizes; FAMILY-OPEN if $\lambda(S)<0$ and $\lambda(S)<\lambda(\mathrm{rand})$ at both, which licenses a training run of at most 1 CPU-h; AMBIGUOUS otherwise.
    * *Result.* $\lambda(S)=-9.73$ and $-10.22$ ($6\times2$, seeds 11 and 22) and $-4.48$ and $-4.43$ ($4\times3$), against $+1.27$, $+1.40$, $+1.16$, $+1.01$ for `rand`. The span captured a median 0.61 ($6\times2$) and 0.49 ($4\times3$) of the free-phase direction, against 0.16 and 0.12 for `ratio2` and 0.08 and 0.12 for `rand`. **Verdict: FAMILY-OPEN**, as pre-registered.
    * *Read after the verdict (no decision weight).* The unrestricted minimum of the same ratio is $-2.5\times10^4$ to $-6.5\times10^5$ and sits on a single configuration: the $\psi^2$-weighted variance in the denominator rewards directions concentrated where $\psi$ is nearly zero. The $\lambda(S)$ eigenvector lives on 12–86 configurations, against 170–636 for the free phase; its realisable gain $q/\max\lvert g\rvert^2$ is $2\times10^{-6}$ to $3\times10^{-5}$, 30–300 times below the free phase's; and the part of the free direction that $S$ captures ascends at $4\times3$. S has $\mathcal O(V)$ columns, so an OPEN would not by itself cover a $V$-independent carrier. The predicted outcome of the training run was recorded as KILL with $p\approx0.85$.
* **r11, trained carrier under the t1c rule (T1, about 3 CPU-min; `tests/P1.2_unanchored_antisymmetry_r11/`).**
    * *Arms.* `S`: $\psi=\psi_F e^{i\varphi}$ with $\varphi=Bc$ on the whitened basis $B$ of span S (1464 and 1760 free coefficients), started along the $\lambda(S)$ eigenvector, L-BFGS on the exact energy. `rand`: the same on the equal-rank random span. `Sb`: the same span with $\lvert\varphi\rvert<\pi/2$, so it can never flip a sign. `S-nn`: the trained `S` with every flip reverted on near-node configurations ($\max_e\lvert R_e\rvert\ge10$). Comparator: the stored `wid` arm (foil widened by 145 parameters, 300 more SR steps). S is not weight-shared, so the comparison leans toward the route, and a KILL covers every weight-shared linear carrier on these columns a fortiori.
    * *Rule (the leaf's t1c rule).* $r=\Delta E_{\mathrm{sign}}[S]/\Delta E_{\mathrm{sign}}[\mathrm{wid}]$, median over seeds. KILL if $r\ge0.9$ at both counted sizes; PASS-W if $r\le0.7$ at both and $r(4\times3)\le r(6\times2)+0.1$; UNDECIDED otherwise. A KILL closes row 4 as KILLED(T1), row 9 as GATE-REFUSED (no PASS-W), and moves the leaf to EXHAUSTED at this budget.
    * *Checks.* Foil reproduced exactly; $E[\varphi=0]$ equals the foil energy; gradient against finite differences $4\times10^{-8}$; $\lambda(S)$ reproduces r10's $-9.7317$.
    * *Result.*

| size | foil | `wid` | `S` | $r$ | `S-nn` | $r_{\mathrm{nn}}$ | `rand` | `Sb` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $6\times2$ | 0.01107 | 0.00850 | 0.00918 | 1.08 | 0.01070 | 1.26 | 0.01107 | 0.01107 |
| $4\times3$ | 0.04962 | 0.04267 | 0.04327 | 1.01 | 0.04822 | 1.13 | 0.04962 | 0.04962 |

* **r11 reading.** Unlike R9.1, `S` leaves $\varphi=0$ and flips 321–665 configurations ($\psi_0$ weight 0.002–0.009), lowering $\Delta E_{\mathrm{sign}}$ below the foil by 17% ($6\times2$) and 13% ($4\times3$), but less than width does with 10–12 times fewer added parameters (145 against 1464 and 1760). 83%, 90%, 84% and 75% of the flipped $\psi_0$ weight (by size and seed) is on near-node configurations, and reverting those flips returns $\Delta E_{\mathrm{sign}}$ to within 3.3% ($6\times2$) and 2.8% ($4\times3$) of the foil. `rand` never leaves the foil. `Sb` reaches nearly the same complex energy without any flip (0.0716 against 0.0714 at $6\times2$, 0.1614 against 0.1606 at $4\times3$), so the descent is mostly flip-free phase. A post-hoc retrain at $4\times3$ with tight tolerances converged to the same point ($\Delta E_{\mathrm{sign}}$ 0.04303 and 0.04355).
* **r11 verdict: KILL** (predicted, $p\approx0.85$). Row 4 is KILLED(T1) (card R10.1). Row 9 ($V>12$ and the $4\times4$ instance for continuous carriers) needs a T1 PASS-W first, which no carrier in this leaf ever reached; a tie at $4\times4$ is predicted with $p\ge0.8$, so it is GATE-REFUSED(G2). Every ledger row is now closed, and the leaf leaves DORMANT for EXHAUSTED at this budget (section 3b, exit 3).
* **Lesson.** A variance-normalised curvature screen is gamed by spikes on near-node configurations; normalise by $\max\lvert g\rvert^2$ or score the projection onto the free direction (library L54; the trained result also joins L21).

Written 2026-11-17 from the rounds above; one card per route ever generated in this leaf, in the order generated. Ids follow the leaf's own labels (Route 1 is R1, Route 2.2 is R2.2, Route 6.7F is R6.7F); routes the leaf left unnumbered get the next free number in their round and say so. Re-audits of an existing route (Routes 3.1, 3.2, 6.1) are folded into the card of the route they re-audit.

Definitions used by every card, so that each card can be read alone:
* **Instances.** The $t$–$t'$–$J$ model (electron hopping $t=1$ between nearest neighbours and $t'$ between next-nearest neighbours, spin exchange $J$, no double occupancy) or the Hubbard model (on-site repulsion $U$) on $V$ sites with $N_\uparrow+N_\downarrow$ electrons. "ED" is exact diagonalization. The **4×4 instance** is the open $4\times4$ $t$–$t'$–$J$ cluster with $6+6$ electrons, $t'=-0.25$, $J=0.4$ (1,681,680 configurations). The **toys** are 2-leg or 3-leg $t$–$t'$–$J$ ladders with $V=8$ to $12$ sites and two holes; a toy size is "counted" only if the foil leaves sign error at least $0.005t$, and from Round 8 on only if its gap $E_1-E_0\ge0.02t$.
* **Ansätze.** NNBF (neural-network backflow) is a Slater determinant whose orbital matrix is the output of a one-hidden-layer network fed the occupations $\mathbf n$; its hidden-layer size is its "width". UHF is the unrestricted Hartree–Fock start. SR (stochastic reconfiguration) is the natural-gradient optimiser; here it uses exact sums over all configurations unless a card says "sampled".
* **Metrics.** $\Delta E_{\mathrm{sign}}$ is an exact upper bound on the energy error of the best non-negative amplitudes placed under the ansatz's sign (P1.1's `budget()`); $\Delta E_{\mathrm{tot}}$ is the ansatz's total energy error; $W$ is the ground-state weight on configurations whose sign is wrong. The **foil** is an NNBF made wider until it has the route's parameter count and trained with the same recipe (`nnbf6`, `nnbfW`, `wide`, `wid` in the code). $r=\Delta E_{\mathrm{sign}}[\text{route}]/\Delta E_{\mathrm{sign}}[\text{foil}]$ (written $\rho_{\mathrm{sign}}$ in Rounds 2 to 5), a median over seeds; $r<1$ means the route has less sign error.
* **Ratios and votes.** For an edge $e$ (a single hop or a spin exchange) taking $\mathbf n$ to $\mathbf m_e$ with signed Hamiltonian element $c_e$, the ratio is $R_e=c_e\Phi(\mathbf m_e)/\Phi(\mathbf n)$ for a reference $\Phi$. Because $\psi_0(\mathbf n)(E_0-H_{\mathbf n\mathbf n})=\sum_{\mathbf m}H_{\mathbf n\mathbf m}\psi_0(\mathbf m)$, the sum of ratios is a "vote" on the sign at $\mathbf n$. "Exact" ratios use the neighbour's own amplitude; "frozen" ratios keep the backflowed orbitals of $\mathbf n$ and change only the occupied rows (one Green's-function entry each). $E_{\mathrm{act}}\propto V$ is the number of active edges at a configuration. $P_\chi$ projects onto a point-group irrep and $P_S$ onto total spin $S$.
* **Codes.** Gates G0 (novelty and adoption), G1 (conservation and asymptotics), G2 (a cheap baseline matches it), G3 (works only without the Hard Core), G4 (gain or cost does not survive growing $V$); tiers T0 (argument), T1 (toy computation under 10 CPU-minutes per invocation), T2 (one-seed pilot), T3 (multi-seed decisive test); CRITIC (a critic objection not tied to one gate).



## Compaction 2026-11-17: verbatim snapshot of summary.md before compaction

(The summary was rewritten to the 3a format and its cap; every line it dropped is preserved here.)

#### Subproblem: P1.2 Antisymmetric layers without a fixed nodal anchor
**Parent Chain**: P1 -> P1.2  **Status**: DORMANT (2026-11-16, unchanged by the 2026-11-17 documentation pass; Saturation Diagnostic adopted, Reopening review STAY-CLOSED; reopening-ledger rows 4 (the restated missing ingredient) and 9 (the $V>12$ regime) are open; see the Leaf Status Summary at the end). Earlier: closed 2026-11-16 with the conditional Saturation Diagnostic (route 8.4 killed by the pre-registered tie-break, r 0.985 and 0.956; all refill routes killed). **Scope note 2026-11-16 (cross-node critic; closed-leaf falsifier):** the closing kills ran on unprojected $V\le12$ toys at 300 steps. Under the space-group projection the same toys carry almost no sign error: the width-6 projected foil has $\Delta E_{\mathrm{sign}}$ 0.0045 at 4×3 and 0.0017 at 6×2, against 0.050 and 0.011 unprojected. The projected re-check of 8.3 is therefore NO-TARGET-P, and the diagnostic is scoped to unprojected toys. The diagnostic's own falsifier (low-rank kernel carrier LRK) was run and KILLED ($r$ 1.05 and 1.00). Verdict unchanged; see "Closed-leaf falsifier (2026-11-16)" at the end. Earlier, Round 8 (2026-11-16): route 8.3 DRF-WS KILLED at T1. Warm-started from the trained foil, with pass 2 as wide as pass 1 and on gap-asserted sizes, it ties matched width ($r=0.99$ at 6×2 and 4×3). Its second pass lowers the total energy, not the sign error. Route 8.4 DRF-T is UNDECIDED at T1 ($r$ 1.19 and 0.895, trained at SR shift 0.1 because the recipe's first step diverges). A symmetric-optimizer tie-break is pre-registered but not run (task CPU cap: 1687 of 1800 CPU-s used). Route 8.2 is KILLED at T0 by the critic's inheritance clause. The conditional Saturation Diagnostic is NOT yet adopted, because it needs 8.4 killed. No T2. See "Round 8" at the end. Earlier status, Round 7 per-leaf critic (2026-11-16): KEEP OPEN, no Saturation Diagnostic yet. The critic premise run (`critic_r7_premise.py`) shows that the trained matched-width foil's own frozen ratios locate its residual wrong signs (held-out AUC 0.93 against 0.65 at $V=12$, rising with size), so the round-7 cause "frozen ratios are wrong-signed" is refuted. The DRF T1b kills stand as specified (a narrow, cold-started second pass; one clean size, since $V=10$ has gap $0.0041t$). Route 8.1 KILLED at T0 (route 6.8 at initialization, trained flips inert). Live: 8.3 DRF-WS (premise PASS, first to the trained T1, rule pre-registered), 8.2 DRF-E (premise PASS) and the refill 8.4 DRF-T (GATED). A conditional Saturation Diagnostic is written, to be adopted only if all three are killed. See "Round 7 per-leaf critic" at the end. Earlier status, Round 7 (2026-11-16): the trained T1 (`t1b_drf.py`) KILLED all three DRF routes, which lose to matched backflow width at both counted sizes ($r_W$ from 1.4 to 2.3). Reopened route 6.7 KILLED at T0 (Gate 4 cost) and its frozen-frame mutation 6.7F KILLED at T1 (`t1_vote67.py`). DRF Gate 4 cost PASSES (a constant factor, slope $-0.16$), so the kills are on accuracy, not cost. Refill: three GATED routes (8.1 exact-vote FRD on a local backflow, 8.2 exact-ratio DRF, 8.3 warm-started DRF). No route reached T2, so nothing is submitted. See "Round 7" at the end. Earlier status, Round 6 per-leaf critic and refill (2026-11-16): FRD-K KILLED (vote inert at T2, Gate 4 cost FAIL, and the job-553724 final rule can no longer pass: $\rho_{\mathrm{sign}}\ge0.78>0.7$ from stored data). AO-F KILLED at T1 together with its sibling AO-L (`t1_ao.py`: a sign-inert exact-ratio factor loses to matched backflow width at 2 of 3 toy sizes). Three live routes from the refill: DRF-K (T1 UNDECIDED, rising), DRF-X and DRF-H (T1 PASS), all determinant-ratio-feature backflow. T2c read-outs prepared, not submitted. Earlier status: Round 6 (2026-11-16), route portfolio with 2 live routes, 1 refill owed; see "Route Portfolio & Sharpest Gaps" in the Round 6 section at the end. FRD-K's 5-seed final rule is IN-FLIGHT (job 553724). FRD-K fails Gate 4 on cost as-is. The T2 pilot (job 697584) found its vote inert ($v=0.003$): the gain is amplitude-borne, so AO-F is now the lead route. FRD-F was KILLED at T2, FRD-H at T1, and five replenishments at T0. The round-5 status below is kept as the record: Completed (Round 5), CONDITIONAL on the decisive test. The round-3 pilot (job 410216) returned NO-TARGET on the 4×3 clusters, so round 5 moves to the shared 4×4 OBC $t$–$t'$–$J$ instance ($6+6$). Two routes survive: FRD-K (hole-hop edges) and FRD-X (spin-exchange edges), both now built on an NNBF base and compared with a projected NNBF at matched parameters. Every round-3 critic fix is implemented. The measured per-sample local-energy cost of FRD is $\mathcal{O}(V)$ times NNBF's, which is the route's adoption wall; the frozen-frame mutation FRD-F is held for the case where accuracy passes. Sign half only.

##### Leaf Status Summary
* **State**: EXHAUSTED at this budget
* **Updated**: 2026-11-17, after the row-4 T1 pair: the curvature screen of the cross-site ratio-product span returned FAMILY-OPEN (r10), and the carrier it licensed was KILLED at T1 by the t1c rule (r11, card R10.1); row 9 is then GATE-REFUSED(G2), every reopening-ledger row is closed, and the leaf leaves DORMANT by exit 3. The per-leaf critic signed the closure certificate the same day, with the summary rewritten (row 4 scoped, row 9's step and cost named).
* **Question**: Is there an antisymmetric construction at $\mathcal{O}(N_e^3)$ cost (or slightly above) whose nodal surface is provably more expressive than a backflow single determinant or Pfaffian, without a sum over many determinants?
* **Answer**: Not known yet; we found none that helps in practice. A trained network can locate its own remaining sign errors, so the information exists. But no extra sign-fixing layer we tried, at about the same cost, cut the sign error more than spending the same parameters on a wider ordinary network. This held on ladder-shaped clusters up to 12 sites and one 16-site cluster, where the one gain came from the magnitudes. Nothing was proved about expressivity, and the larger-cluster test was judged predictable and not run.
* **Conclusion**: A trained neural backflow's own determinant ratios tell where its remaining wrong signs are (held-out AUC 0.93 at 12 sites), but no sign layer or phase carrier tested turned that knowledge into a lower sign error than the same parameters spent on backflow width: the warm-started layers and carriers tied or lost ($r$ between 0.96 and 1.08 at the counted sizes, the last carrier at 1.08 and 1.01), and the two wins were not sign wins (on the $4\times4$ instance FRD-K's 16% gain was amplitude-borne; on a narrow projected 6×2 toy LRK's per-step win shrank to 10% at equal CPU time). The last carrier, a linear phase on a span that couples neighbouring edges, did flip signs, but mostly on near-node configurations, and still lost to width despite 10–12 times more parameters. The step that would test continuous carriers on larger clusters was refused by the compute gate as predictable (no carrier ever passed at 12 sites), so the leaf is closed at this budget, not answered.
* **Scope**: exact-sum training and scoring on 2-leg and 3-leg $t$–$t'$–$J$ toys with $V\le12$ sites ($t'=-0.25$, $J=0.4$, two holes, gap $E_1-E_0\ge0.02t$ on counted sizes), mostly unprojected, 300 warm-started SR steps (the R10.1 phase trained to L-BFGS convergence on a frozen foil), two seeds, matched parameter count; one projected narrow-foil regime (R9.2); the FRD family also on the $4\times4$ open $6+6$ instance with space-group projection. Sign half only; no competing-order claim.
* **Compute spent**: 4,401 CPU-h on Slurm (master Compute Ledger: 410216, 451916, 553724, 697584, 698178; 451916 alone is 3,260), plus about 2.3 CPU-h of single-thread interactive T1 runs (rounds 6 to 9 and the row-4 pair, r10 3.3 CPU-min and r11 about 3 CPU-min). Rounds 2 and 3 left no job ids.

**The question.** A neural quantum state (NQS) writes the wave function $\psi(\mathbf n)$ on occupation configurations $\mathbf n$ of $V$ lattice sites; for doped, frustrated fermions its sign (the nodes) is the hard part. The standard fermionic NQS, neural-network backflow (NNBF), is a Slater determinant whose orbitals a small network computes from $\mathbf n$, so its nodes move only by deforming orbitals from a mean-field anchor. P1.2 asks whether an added sign-changing layer, at about one determinant's cost, reaches better nodes. The score is $\Delta E_{\mathrm{sign}}$, an exact upper bound on the energy error of the best non-negative amplitudes under the ansatz's sign, compared with a matched-width foil (an NNBF made wider to the same parameter count, trained by the same stochastic reconfiguration (SR) recipe) through $r=\Delta E_{\mathrm{sign}}[\text{route}]/\Delta E_{\mathrm{sign}}[\text{foil}]$; $r<1$ means the route has less sign error.

**The mechanism tried.** The ground state obeys $\psi_0(\mathbf n)(E_0-H_{\mathbf n\mathbf n})=\sum_{\mathbf m}H_{\mathbf n\mathbf m}\psi_0(\mathbf m)$, so the true sign at $\mathbf n$ is a weighted vote of its Hamiltonian neighbours. The determinant supplies it through ratios $R_e=c_e\Phi(\mathbf m_e)/\Phi(\mathbf n)$ over edges $e$ (hops or exchanges, $c_e$ the signed matrix element); "frozen" ratios are Green's-function entries at one inverse per amplitude. The routes fed this information through different carriers: discrete flips, positive factors, a second backflow pass, a low-rank orbital update, and a phase $e^{i\varphi(\mathbf n)}$ that can turn a sign without passing through zero.

**The chain of evidence.**
1. Vote factors $\Phi\prod_e[1+\alpha_eg(R_e)]$ on an NNBF base (FRD-K, R5.1), on the projected $4\times4$ instance (1,681,680 configurations), beat width 6 at matched parameters (1719 against 1734) with $r=0.84$ over 3 seeds (451916). Removing every flip moved $\Delta E_{\mathrm{sign}}$ from 0.009751 to 0.009757 (697584): the gain was amplitude, not vote. Its cost grows as $(1+E_{\mathrm{act}})$ NNBF passes per local energy, with $E_{\mathrm{act}}\propto V$ active edges.
2. Positive factors on exact ratios (R6.3, R7.1) beat a neural Jastrow but lost to width ($r$ 2.04, 0.95, 1.19 at $V=8,10,12$).
3. A second backflow pass fed frozen ratios lost cold-started (R7.2 to R7.4, $r$ 1.4 to 2.3). Yet on the trained foil those features locate its wrong signs (AUC 0.93 against 0.65 for occupations). Warm-started, as wide as pass 1, it lowered total energy faster but tied on sign ($r=0.991$ at 6×2 and 4×3; R8.3); a tied re-entry tied once the optimiser shift was matched ($r$ 0.985, 0.956; R8.4).
4. The diagnostic's own falsifier, a weight-shared low-rank orbital update with 120 parameters (below one width unit, 145), tied again ($r$ 1.05, 1.00; R9.0).
5. A per-site phase carrier returned to $\varphi\approx0$ ($r$ 1.30, 1.16; R9.1): its second-order curvature $\lambda$ is positive (0.22 to 0.35) for every per-site span of the ratio features, while an unrestricted per-configuration direction descends ($\approx-1.7$).
6. A span coupling neighbouring edges (cross-site and two-hop ratio products) passed the pre-registered curvature screen ($\lambda=-4.4$ to $-10.2$ against $+1.0$ to $+1.4$ for random spans), but the trained phase on it lost to width ($r=1.08$ at 6×2, 1.01 at 4×3) despite 10–12 times width's added parameters (1464 and 1760 free coefficients against 145), so the loss holds a fortiori; 75–90% of its flipped weight was on near-node configurations (83, 90, 84 and 75% by size and seed), and reverting those flips put it back within 3.3% (6×2) and 2.8% (4×3) of the foil (R10.1). The screen's negative curvature came from spikes where $\psi$ nearly vanishes (L54).

Fairness: same start, matched parameters (for R10.1, 10–12 times more parameters for the route, so not a matched-width comparison: its loss holds a fortiori), same SR recipe and steps, exact sums, rules written before each run.

**Why the alternatives fail.** Discrete flips are left unused by SR (R5.1, R6.7). The exact projected vote helped once but is a one-step Lanczos sign costing $(1+E_{\mathrm{act}})\lvert G\rvert$ passes per amplitude (R6.7, R6.8, R8.1). A fixed vote on frozen ratios raises the sign error 1.3 to 5.2 times (R6.7F, R5.4). Local-occupation layers cannot see the nodes (R4, R2.1); a complex detour decouples (R6.4); an unrestricted phase beats width only at 4×3 ($r=0.77$), with one parameter per configuration.

**What this leaf taught beyond itself** (`library.md`):
* L21: ratio features locate wrong signs but no trained use of them beat width; occupation-only layers do not even locate them (R2.1, R4, R5.2, R6.7F, R7.2, R7.4, R8.2–R8.4, R9.0, R10.1).
* L22: positive correlators lose to width on sign (R6.3, R7.1).
* L23: a phase carrier moves only along negative-curvature directions in its span (R6.4, R9.1).
* L54: a negative $\lambda(S)$ carried by near-node spikes does not train into a sign gain; screen with a sup-normalised $q$ ($q/\max\lvert g\rvert^2$) (R10.1).
* L38, L47, L50, L40, L33, L10, L24: flips-off read-outs (R5.1), hidden $\mathcal O(V)$ costs (R6.7), checking whether a pending rule can still pass (R5.1), gapped toys (R7.3), matched optimiser settings (R8.4), finite-size linear dressings (R2.3, R7.5), published sign networks (R1, R5).

**Strongest objection and answer.** "These are short runs on unprojected 12-site toys, where projection removes about 90% of the sign error, so the toys test contamination, not nodes." Partly answered: on a narrow projected foil LRK ties at 4×3 and its 6×2 win shrinks to 10% at equal CPU time (R9.2), and P1.3 finds the projected $4\times4$ sign error capacity-limited. Not answered for continuous carriers on the $4\times4$ instance, which is the refused step below.

**Limits.** Not covered: $V>12$ for continuous carriers, other $t'$ or $J$, sampled-force training, horizons beyond 300 SR steps, compiled cost constants, carriers non-linear in the R10.1 columns (message passing) and carriers trained jointly with the backflow; the row-4 kill covers linear phase spans on a frozen unprojected foil at 6×2 and 4×3 only.

**Closure certificate** (EXHAUSTED at this budget; signed by the per-leaf critic, 2026-11-17):
1. **Claim.** On exact-sum $t$–$t'$–$J$ toys up to 12 sites (and, for the vote family, the projected $4\times4$ instance), no sign carrier fed the trained backflow's own ratio information lowered the sign error faster than spending the same parameters on backflow width, under pre-registered matched-parameter rules; every carrier named by the Saturation Diagnostic and the Reopening review was killed at T1 or T0, or its next step was refused by the compute gate.
2. **Decisive evidence.** R9.0 (KILL, $r$ 1.05, 1.00), R9.1 (KILL, $r$ 1.30, 1.16; per-site spans $\lambda>0$), R10.1 (KILL, $r$ 1.08, 1.01, licensed by the r10 FAMILY-OPEN), R8.3 and R8.4 (ties), R5.1 with its flips-off read-out (697584), all under rules written before the runs; checked by the Round 7 critic and the critic sign-off of 2026-11-16 for rounds 6 to 9, and by the reviewer of the DORMANT triage (2026-11-17) for the row-4 pair.
3. **Why no doubt remains.** Unprojected toys (cross-node critic Major 4): the projected re-check found no target (NO-TARGET-P), and the narrow projected regime (R9.2) ties at 4×3. 300-step horizons (Major 3): the equal-CPU width arm closed most of LRK's 6×2 gap, and the converged-horizon arm (R9.4) is refused with a predicted tie. The FRD-K $4\times4$ win: the flips-off read-out shows it is amplitude. The R10.1 screen said OPEN: answered by the trained test it licensed. R10.1 had more parameters than width: that biases toward the route, so the KILL holds a fortiori; its L-BFGS stop: a tight-tolerance retrain converged to the same point.
4. **Falsifiers run.** The diagnostic's own falsifier R9.0: DIAGNOSTIC-HOLDS. Projected re-check of R8.3: DIAGNOSTIC-HOLDS (scope narrowed). Curvature screen of per-site spans: DIAGNOSTIC-HOLDS. Row-4 screen and training: KILLED(T1). Refused: R9.4 (G2, tie predicted with $p\ge0.8$, a few CPU-minutes per size), R9.2's decisive step (G2, PASS-W unreachable), row 9 (G2 and G4: the best continuous carrier, LRK (R9.0, the weight-shared low-rank orbital update, 120 parameters), warm-started on the stored projected width-6 NNBF foils of job 451916 on the $4\times4$ open $6+6$ instance (3 seeds), 300 SR steps against a width-plus-one arm from the same foils, with $\Delta E_{\mathrm{sign}}$ read out for both. Capped cost from existing timings: the width-6 projected step takes 13.1 s on 16 CPUs (451916 logs), and LRK's step costs about 3.8 times width's (R9.2: 300 LRK steps matched 1150 width steps in CPU time), so the LRK arm is about 200 CPU-h, the width arm about 60 CPU-h and the six sign budgets about 15 CPU-h (566 s each on 16 CPUs): about 275 CPU-h in all and about 350 CPU-h capped with the 1.3 wall-time margin, a T3-tier job. Prediction: a tie ($r\ge0.9$) with $p\ge0.8$, from R9.0's ties at 12 sites ($r$ 1.05, 1.00), R9.2's projected 4×3 tie, and P1.3's capacity-limited reading of the same instance. It changes no decision: a tie or loss leaves the leaf closed, and even a win would need the 12-site PASS-W that no carrier reached before it could reopen the leaf. Refused on G2, and on G4 (a T3-tier job needs a passed T2, which LRK lacks)).
5. **Missing ingredients exhausted.** I1 (continuous ratio-fed carrier below one width unit): R9.0, KILLED(T1). I2 (attribution control): CLOSED (precondition never met, no PASS-W). I3 (capacity to the nodes): R9.1, KILLED(T1). Restated ingredient (row 4, a span containing the phase-descent directions): R10.1, KILLED(T1). U1 (projected target): R9.2, GATE-REFUSED(G2). U2 (horizons, equal wall time): R9.4, GATE-REFUSED(G2). U3 (no zero crossing): R9.1, KILLED(T1). U4 (exact ratios): R8.2, KILLED(T0). U5 (row 9, $V>12$ and the $4\times4$ instance for continuous carriers): GATE-REFUSED(G2, G4): the refused step is LRK against a width-plus-one arm on the projected width-6 $4\times4$ foils, about 350 CPU-h capped, tie predicted with $p\ge0.8$ (item 4).
6. **What would reopen it.** A carrier with a $V$-independent parameter count whose span passes an amended curvature screen ($q/\max\lvert g\rvert^2$, or the captured projection of the free-phase direction, negative at both counted sizes) and then reaches PASS-W with attribution at 12 sites; only then would the refused $4\times4$ step be worth its cost. Capped cost of that step (LRK against width-plus-one on the stored projected width-6 foils of 451916, 3 seeds, 300 SR steps): about 350 CPU-h, a T3-tier job that P1.2's 300 CPU-h leaf allocation under the gate does not cover alone; it would need an auditor-approved move of about 50 CPU-h from another leaf, or a user budget increase, plus a passed T2. R9.4 would cost a few CPU-minutes per size and reopens only with a compiled re-timing that puts LRK's step cost below twice width's.

**Reopening ledger** (section 3b; every row closed, at least one GATE-REFUSED, so exit 3 applies):

| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| 1. I1: a continuous carrier fed the ratio information that adds orbital directions for less than one width unit | Saturation Diagnostic (Round 7 critic) | R9.0 LRK | KILLED(T1) |
| 2. I2: an attribution control showing a gain needs the inverse | Saturation Diagnostic | runs only after a PASS-W | CLOSED (precondition never met: no route reached PASS-W) |
| 3. I3: a carrier whose new capacity goes to the nodes, not the amplitudes | Reopening review | R9.1 RPC | KILLED(T1) |
| 4. Restated ingredient: a carrier whose function class contains the configuration-specific phase-descent directions, parameter count independent of $V$ | Reopening review | r10 curvature screen of the cross-site ratio-product span (FAMILY-OPEN), then the r11 t1c training of R10.1 | KILLED(T1) (scope: linear phase carriers on span S, frozen unprojected foil, 6×2 and 4×3; non-linear (message-passing) and jointly trained carriers untested) |
| 5. U1: projected training with a sign-error target | Reopening review; cross-node critic Major 4 | R9.2 LRK-PN | GATE-REFUSED(G2 predicted outcome: 4×3 ties, so PASS-W is unreachable) |
| 6. U2: horizons past 300 steps, or equal wall time | Reopening review; cross-node critic Major 3 | R9.4 | GATE-REFUSED(G2 predicted outcome: tie with probability at least 0.8) |
| 7. U3: a sign change without a zero crossing | Reopening review | R9.1 RPC, after R6.4; R10.1 | KILLED(T1) |
| 8. U4: exact instead of frozen ratios | Reopening review | R8.2 DRF-E | KILLED(T0) |
| 9. U5: sizes above $V=12$ and the $4\times4$ instance for continuous carriers | Reopening review | refused step: LRK (best continuous carrier) against a width-plus-one arm on the stored projected width-6 NNBF foils of 451916, $4\times4$ open $6+6$, 3 seeds, 300 SR steps; about 350 CPU-h capped (13.1 s per width step on 16 CPUs, LRK about 3.8 times that) | GATE-REFUSED(G2: tie predicted with $p\ge0.8$, and neither a tie nor a win without a 12-site PASS-W changes the decision; G4: T3 tier without a passed T2) |
| 10. Diagnostic falsifier: a ratio-fed carrier with $r\le0.7$ against width plus attribution | Saturation Diagnostic | R9.0 (`t1e_closed.py` Part A) | DIAGNOSTIC-HOLDS |
| 11. Projected re-check of the R8.3 kill | cross-node critic Major 4 | `t1e_closed.py` Part B: NO-TARGET-P (projected foil 0.0045 and 0.0017) | DIAGNOSTIC-HOLDS (scope narrowed to unprojected toys) |
| 12. Curvature screen of per-site linear and quadratic ratio spans | Reopening review | `t1f_curv.py` ($\lambda$ from 0.13 to 0.35) | DIAGNOSTIC-HOLDS |
| 13. The projected vote is usable (`gvote`, $z=-1.33$) | Round 6 critic objection 3 | R6.7, R6.7F, R8.1 | KILLED(G4) |
| 14. Compiled re-timing of DRF before any T2 | Round 7 critic minor 7 | runs only after a T1 pass | CLOSED (precondition never met) |
| 15. Remaining tasks of job 553724 (FRD-K seeds 44 and 55) | compute gate | cancelled | GATE-REFUSED(verdict already decided from stored seeds) |

Exits of section 3b checked on 2026-11-17 after the r11 KILL: no row is SURVIVED-T1 or DIAGNOSTIC-FALSIFIED; every row is closed, rows 5, 6, 9 and 15 are GATE-REFUSED and rows 2 and 14 are CLOSED (precondition never met), so exit 3 (EXHAUSTED at this budget) applies.

Critic sign-off (2026-11-16): approved as DORMANT, with numbers checked against the round 6 to 9 outputs and the Compute Ledger. Documentation pass (2026-11-17): Analogy Map, Route Register and reopening ledger added. Transition to EXHAUSTED at this budget (2026-11-17, editor pass applying the triage decision and the pre-registered r11 rule): per-leaf critic, 2026-11-17: EXHAUSTED at this budget approved; row 4 KILLED(T1) by r11 (r = 1.08 and 1.01 against width, at 10–12 times the width arm's added parameters, scoped to linear phase spans on the frozen foil); rows 5, 6 and 9 G2-refused, with row 9 naming its step, cost and p.


## Compaction 2026-11-17: verbatim snapshot of card R10.1 before compaction

#### R10.1 — Span-S phase carrier: a linear phase on cross-site ratio products (reopening-ledger row 4) (KILLED)
* **The idea**: Put a phase $e^{i\varphi(\mathbf n)}$ on the frozen trained backflow, with $\varphi$ linear in a span S that adds cross-site products of the per-site features on every bond and two-hop products of edge ratios to R9.1's per-site span. R9.1 failed because every per-site span had positive curvature; the descending phase directions vary across pairs of neighbouring edges, so a span that couples edges might contain one and let the carrier rotate wrong signs to right ones without any zero crossing.
* **Lineage**: reopening-ledger row 4 (the restated missing ingredient of the Reopening review), Operation A on R9.1 after its kill at T1; screened by r10 and trained by r11 (2026-11-17). No successor: its kill closes row 4.
* **Outcome**: KILLED at T1 (2026-11-17).
* **Kill rule**: the leaf's t1c rule, written in `tests/P1.2_unanchored_antisymmetry_r11/RULE.md` before the run: "$r=\Delta E_{\mathrm{sign}}[S]/\Delta E_{\mathrm{sign}}[\mathrm{wid}]$. KILL if $r\ge0.9$ at both counted sizes (≥ 2 counted sizes including the largest). PASS-W if $r\le0.7$ at both counted sizes and $r(2)\le r(1)+0.1$. UNDECIDED otherwise." It was licensed by the r10 screen's FAMILY-OPEN ($\lambda(S)<0$ and below the random control at both sizes).
* **Evidence**: unprojected $6\times2$ and $4\times3$ $t$–$t'$–$J$ toys ($5+5$, $t'=-0.25$, $J=0.4$), frozen width-6 foils, seeds 11 and 22, exact sums. Screen (r10, 3.3 CPU-min): $\lambda(S)=-9.73$, $-10.22$ ($6\times2$) and $-4.48$, $-4.43$ ($4\times3$) against $+1.27$, $+1.40$, $+1.16$, $+1.01$ for an equal-rank random span. Training (r11, about 3 CPU-min): $\Delta E_{\mathrm{sign}}$ of `S` 0.00918 against width's 0.00850 and the foil's 0.01107 at $6\times2$ ($r=1.08$), 0.04327 against 0.04267 and 0.04962 at $4\times3$ ($r=1.01$). `S` flipped 321–665 configurations, with 83%, 90%, 84% and 75% of the flipped $\psi_0$ weight (by size and seed) on near-node configurations ($\max_e\lvert R_e\rvert\ge10$); reverting those gives $r_{\mathrm{nn}}=1.26$ and 1.13, within 3.3% ($6\times2$) and 2.8% ($4\times3$) of the foil. A flip-free version of the same span reached nearly the same complex energy (0.0716 against 0.0714; 0.1614 against 0.1606). Tight-tolerance retraining at $4\times3$ converged to the same point.
* **Why this kills it**: $r\ge0.9$ at both counted sizes, so the rule's KILL fires, and it lost to width despite 10–12 times more parameters, so the loss holds a fortiori. Because S is not weight-shared (1464 and 1760 coefficients against width's 145 added parameters) and contains every weight-shared linear carrier on these columns, a matched-parameter carrier on these features can only do worse. What gain `S` has sits on near-node flips and on flip-free phase, not on the nodal errors the carrier was meant to fix.
* **What the kill does not rule out**: the kill covers linear phase carriers on span S on the frozen unprojected foil at $6\times2$ and $4\times3$ only. Untested: carriers that are non-linear in these columns, or built on other features (message passing, multi-hop beyond two hops); carriers trained jointly with the backflow instead of on a frozen foil; projected foils; sizes above 12 sites; spans screened with the amended curvature metric ($q/\max\lvert g\rvert^2$ or projection onto the free direction), which no span has passed.
* **Strongest case against the kill**: the screen said OPEN: $\lambda(S)$ was strongly negative at both sizes while every earlier span was positive, so the family looked like it contained the missing descent directions. Answered by the trained test that the screen licensed: the pre-registered training lost to width, the flips that did happen sat on near-node configurations, and the post-hoc screen diagnostics (eigenvector on 12–86 configurations, realisable gain 30–300 times below the free phase) show that the negative $\lambda$ was carried by near-node spikes.
* **Checked by**: pre-registered rule applied by the orchestrator's reviewer (DORMANT triage, 2026-11-17); per-leaf critic, 2026-11-17 (KILLED(T1) upheld, scoped to linear phase spans on the frozen unprojected foil).
* **Lesson**: L54 (`library.md`); the trained result also joins L21.

