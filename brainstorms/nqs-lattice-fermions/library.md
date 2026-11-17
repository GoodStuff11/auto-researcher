# Findings Library: Neural Quantum States for Discrete Fermion Systems

This library collects what the brainstorm `brainstorm-nqs-fermions.md` learned that reaches beyond a single route or leaf. It was compiled on 2026-11-17 from the route cards in `.subproblems/P*.md` (section `## Route Register` of each leaf) and from the master's compute ledger and former anti-pattern list. Every entry is meant to be read on its own. Status PROVISIONAL means the cross-node critic has not yet reviewed the entry; ESTABLISHED means the cross-node critic of 2026-11-16 proposed or signed off exactly this lesson.

## Index

* L1 (PROVISIONAL): At exact-diagonalization size, most of the sign error of a neural backflow trained from a broken-symmetry mean-field start is symmetry contamination, and only training inside the target symmetry sector removes it.
* L2 (PROVISIONAL): The sign error left in a symmetry-projected neural backflow at exact-diagonalization size is a capacity limit of the network, not a wrong-sign basin that an optimiser could escape.
* L3 (PROVISIONAL): No function of Hamiltonian moments or of low-degree polynomial filters of the trial state can determine an observable error that sits in low-energy directions.
* L4 (PROVISIONAL): An eigenstate window that must contain every level carrying the observable error grows with the level density, so excited-state subspace corrections are an exact-diagonalization diagnostic, not a scalable error bar.
* L5 (PROVISIONAL): A summed observable such as a structure factor can be accurate while its long-range tail carries the error, and rebuilding the tail from per-mode pieces amplifies their errors about a hundredfold.
* L6 (PROVISIONAL): Certificates and bounds that need a spectral gap or a susceptibility measured inside the variational family fail exactly at competing-order points, where the gap closes and the family is stiffest.
* L7 (PROVISIONAL): Two macroscopically distinct orders share neither overlap nor local structure, so cross-order estimators cost samples exponential in the system size and shared-parameter error cancellation does not happen.
* L8 (PROVISIONAL): Ansätze that make the chemistry local energy cheap do so with configuration-independent or window-restricted orbitals, and they lose the external pair correlation that grows with the basis.
* L9 (PROVISIONAL): The size-extensive saving of local-energy screening in molecules comes from the localized orbital frame, not from the estimator.
* L10 (PROVISIONAL): A correction with a fixed number of terms, components, projector steps or global vectors gives a gain that does not grow per site, so it is a finite-size device; size consistency needs a product form inside one object.
* L11 (PROVISIONAL): Exact non-Abelian symmetry at polynomial cost is available only when every symmetry-breaking ingredient sits inside an object on which the group acts in closed form.
* L12 (PROVISIONAL): Any single-proposal importance estimator of a non-stoquastic sum has a variance floor that does not vanish as the state converges.
* L13 (PROVISIONAL): Any error estimate built from the variational tangent space or the variational dynamics of a converged state vanishes identically, because the state is stationary there.
* L14 (PROVISIONAL): A per-observable response vector recovers the observable error exactly at first order, but its tolerance to training error shrinks as the density of low-lying levels grows.
* L15 (PROVISIONAL): A Ritz correction built from noisy generated vectors cannot resolve a competitor closer than a noise floor, and a gap witness built from the same vectors over-estimates the gap.
* L16 (PROVISIONAL): Ritz diagonalisation in the span of two trained variational states returns the state with the lower trained energy, so it cannot beat their differential bias.
* L17 (PROVISIONAL): Any probe or transport of a capacity-limited ansatz (field response, hysteresis, basin stability, continuation of snapshots) measures the biased energies, so it inherits the energy rule's decision boundary.
* L18 (PROVISIONAL): Parameter continuation from the non-interacting or sign-free limit follows the adiabatically connected branch and keeps the wrong state at near-degenerate level crossings.
* L19 (PROVISIONAL): On a discrete configuration space, sampler-only changes cannot change which basin an exact-expectation optimiser reaches.
* L20 (PROVISIONAL): A sign layer that sees only local occupations cannot locate a reference determinant's nodes; sign corrections must be conditioned on the reference's amplitudes.
* L21 (PROVISIONAL): Knowing where a trained backflow's signs are wrong does not lower its sign error: features that locate wrong signs feed amplitude capacity, not sign capacity, under energy training.
* L22 (PROVISIONAL): Off-loading amplitude into a positive correlator does not buy sign accuracy at matched parameters; the same parameters spent on backflow width win.
* L23 (PROVISIONAL): A complex phase carrier escapes the zero-crossing barrier of real nodes but not the span barrier: at a real stationary state it moves only if its feature span contains a negative-curvature direction.
* L24 (PROVISIONAL): Learned sign networks, neural Pfaffians and local sign correlators for lattice fermions are published constructions whose adoption barrier no proposal here removed.
* L25 (PROVISIONAL): Cheap local-energy surrogates have sample-dependent errors far larger than the local-energy variance on converged states, so cross-sample debiasing cannot pay.
* L26 (PROVISIONAL): Near convergence the deterministic work per local energy sets the cost, so a cheaper local energy helps only if its saving buys more fresh samples.
* L27 (PROVISIONAL): A rigorous certificate on a screened local energy that ignores sign cancellations is so loose that certified screening keeps almost every term.
* L28 (PROVISIONAL): Carrying fewer than one hidden row per site to shrink a hidden-fermion determinant's inverse loses sign capacity on 12- to 15-site clusters.
* L29 (PROVISIONAL): In linear-method or natural-gradient steps for over-parameterised states, truncate the parameter metric at a relative cutoff instead of shifting it, because the gain sits in near-redundant parameter combinations.
* L30 (PROVISIONAL): Before speeding up or transferring a size-independent family, measure its capacity plateau and effective tangent rank at the target; if the rank does not grow with width, the family cannot reach the target accuracy.
* L31 (PROVISIONAL): Swapping exact and approximate parts of jointly optimised factors is not an error budget, because each part is co-adapted to the other's errors.
* L32 (PROVISIONAL): Only the best energy at fixed signs is a sign-only error measure; configuration-weighted sign accuracy and lattice fixed-node energies mix in amplitude effects.
* L33 (PROVISIONAL): Every comparison between separately trained ansätze is also a comparison of how far each optimiser got, so rankings need polished arms or a check of the remaining optimisation gap.
* L34 (PROVISIONAL): In a cost-reduction study, accuracy ratios at matched parameters after a short fixed number of steps measure learning speed, so they may kill a route but never promote it.
* L35 (ESTABLISHED): An accuracy gate between arms with different cost per step or per sample must add an equal-wall-time arm, or state that it scores parameter efficiency only.
* L36 (ESTABLISHED): When the parameter count reaches about half the projected sector dimension, width comparisons measure sector fitting and carry no capacity or decision-rule information beyond exact-diagonalization size.
* L37 (ESTABLISHED): An exact-inverse or exact-matrix reference certifies the machinery, not the information source; a premise run needs a matched Hamiltonian-only foil before its information source is credited.
* L38 (PROVISIONAL): A mechanism gets credit for a gain only after an ablation that removes the mechanism, or a matched-compute capacity foil, fails to reproduce the gain.
* L39 (PROVISIONAL): Exact-sum clusters cannot test claims whose value appears only at large size, such as a cheaper projection or a local-versus-global split, because there projection is free and every patch covers most of the lattice.
* L40 (PROVISIONAL): A toy instance may decide a rule only if it is representative: it has a spectral gap, its sector is not nearly spanned by the method's subspace, and its noise and infidelity match the real states.
* L41 (PROVISIONAL): Ensembles of seeds, widths or fields of one architecture share their bias, so their disagreement, or an extrapolation through them, under-states the error.
* L42 (PROVISIONAL): An extrapolation through separately sampled states fails when the signal it extrapolates is smaller than each state's statistical error.
* L43 (PROVISIONAL): A warm-started or transferred arm must be scored against its own starting state and against training from scratch, not only against the foil.
* L44 (PROVISIONAL): A test bed for competing orders must put both competitors in the same symmetry sector, which needs a two-parameter scan because same-sector degeneracies have codimension 2.
* L45 (PROVISIONAL): A decision rule that may abstain must be scored on its wrong-commit rate and its decision rate together.
* L46 (PROVISIONAL): Fluctuation along the optimisation trajectory can overstate the noise of a final-state energy difference about a hundredfold; estimate decision noise from final-state sampling or seed replicas.
* L47 (PROVISIONAL): Cost gates must be predicted from per-sample operation counts of the real algorithm and from measured speeds of compiled kernels, not from multiply-add totals, stored full-basis vectors or interpreter timings.
* L48 (PROVISIONAL): Almost half of the compute spent before the compute gate changed no decision, and most of it could have been predicted by a scaling argument that cost no compute.
* L49 (PROVISIONAL): Caps per array task do not bound what an array job spends; compute caps must be set on the total of a job.
* L50 (PROVISIONAL): Before planning on, resuming or extending a job, check from stored data whether its pre-registered rule can still change the verdict.
* L51 (PROVISIONAL): Removing the reason a route was killed, or rescuing it after the result, is not a new mechanism; a revival needs its own pre-registered test with a cost model measured on one instance at one size.

## About the problem itself

### L1. At exact-diagonalization size, most of the sign error of a neural backflow trained from a broken-symmetry mean-field start is symmetry contamination, and only training inside the target symmetry sector removes it.
* **Status**: PROVISIONAL
* **Applies to**: any error budget, basin study or sign diagnostic of a neural quantum state (NQS) for lattice fermions whose training starts from a symmetry-broken mean-field state (for example unrestricted Hartree–Fock, UHF), and any method that restores symmetry after training.
* **The lesson**: A network trained from a broken-symmetry start converges to a state that mixes several symmetry sectors of the Hamiltonian. Its error against the exact ground state $\psi_0$ is then dominated by the wrong-sector admixture, and much of that admixture shows up as wrong signs. Measured "sign errors" and apparent "wrong-sign basins" of such a network are therefore properties of the contamination, not of the nodal structure inside the target sector. Projecting the trained network onto the sector afterwards does not repair this, because the part of it that lies in the sector is not the sector's best state; only variation after projection (training the projected network) removes the error.
* **Evidence**: P1.1-P (job 407525): neural-network backflow (NNBF) of width 128 on three $4\times3$ periodic Hubbard tori ($5+5$ electrons at $U=8t$, $4+4$ at $U=8t$, $5+5$ at $U=4t$), medians over 3 seeds. The sign fraction $f_E=\Delta E_{\mathrm{sign}}/\Delta E_{\mathrm{tot}}$ (share of the energy error due to signs, with $\Delta E_{\mathrm{sign}}$ the error of the best state with the network's signs) is 0.61–0.77 without projection and at most 0.023 with space-group-projected training. Training inside the sector removes more than 99.999% of the unprojected sign error (residual at most $3.5\times10^{-6}$ of it); projecting the unprojected network after training removes only 35%, 90% and 1%, and exact spin projection alone leaves $f_E=0.37$, 0.67 and 0.25 (card P1.1-P R2). P1.3 R2.1: on the $4\times3$ $4+4$ torus at $t'=-0.25t$, projected stochastic reconfiguration (SR) from the same UHF reaches $\Delta E_{\mathrm{sign}}\approx5\times10^{-6}t$ at width 8 against $0.082t$ for unprojected SR at width 32 (job 407525), so a basin route tested against an unprojected control would have "won" by symmetry restoration alone.
* **Scope and limits**: Three $4\times3$ tori and the P1.3 toy; exact sums (no sampling noise); widths 8 to 128. It does not say that projected networks have no sign error: on the $4\times4$ open $t$–$t'$–$J$ cluster ($6+6$) projected NNBF keeps $f_E=0.22$–$0.24$ (see L2).
* **Counter-evidence and open questions**: The original P1.1 PASS ("sign is the binding error", job 329057) was read before this was known and is superseded by it. Whether contamination dominates at sizes where projection is too costly to train with is untested.
* **How to use it**: Train every control and every arm inside the target symmetry sector before attributing error to signs or testing a basin-escape route; never use post-hoc projection of an unprojected network as a substitute.
* **Sources**: P1.1-P R1 and R2 (job 407525); P1.1 R1 (job 329057, superseded); P1.3 R2.1; P4.1 RG5 (train unprojected, project at the end, killed as standard practice).

### L2. The sign error left in a symmetry-projected neural backflow at exact-diagonalization size is a capacity limit of the network, not a wrong-sign basin that an optimiser could escape.
* **Status**: PROVISIONAL
* **Applies to**: basin-escape, tempering, re-anchoring and second-order-optimiser proposals for fermionic NQS; any claim that a better optimiser would fix the sign structure of a trained network.
* **The lesson**: If the residual sign error were a basin (a better-signed state reachable by the same network but separated by a barrier), then a fit aimed directly at the exact signs would reach a markedly better-signed state, and oracle-started runs would stay there under energy training. Neither happens. A second-order fit to a sign-oracle target (the trained state's own amplitudes with the exact signs) plateaus while its step has become pure Gauss–Newton, so the plateau belongs to the network family. States with better signs reached through an oracle sit higher in energy than the best cold run, so energy training correctly leaves them: a better sign at fixed width is a trade against amplitude accuracy, not a trap. The sign error falls when the network is widened.
* **Evidence**: P1.3 R3.1 (jobs 507179 and 625182): $4\times4$ open $t$–$t'$–$J$ cluster, $6+6$, $t'=-0.25t$, $J=0.4t$, three seeds per width; best cold $\Delta E_{\mathrm{sign}}$ 0.0192, 0.0117, 0.0070 at widths 4, 6, 10 (1284, 1734, 2634 parameters); the released oracle arm ended at 0.0189, 0.0115, 0.0070, inside margins of 0.003 to 0.005; verdict CAPACITY by the pre-registered width tie-break. P1.3 R7.1: projected $4\times3$ open $t$–$J$ toy ($5+5$, gap $0.085t$, 675 parameters, 3 seeds); the Levenberg–Marquardt (LM) sign-oracle fit reached a median sign error 22% below the cold start (0.00854 against cold 0.01098, 0.00930, 0.01090), far short of the 50% threshold, with damping down to $10^{-10}$–$10^{-15}$; released, the states returned to the cold level. P1.3 R7.3: same toy at in-sector gap $0.1217t$; 0 of 3 cold runs captured by the competitor; the oracle state kept a better sign (0.0052 against 0.0084–0.0158) but sat $0.018t$ above the best cold run in energy (one seed). Job 410199: on the $4\times3$ torus all 9 projected cold runs reached $\psi_0$ (energy error at most $2.7\times10^{-5}t$).
* **Scope and limits**: Exact-diagonalization sizes (12 and 16 sites), exact sums, one $4\times4$ instance and one $4\times3$ toy at width 3. The cross-node critic noted that on the $4\times4$ instance the oracle start was never better than the best cold start, so no better-signed state was ever placed there; the direct probe (P1.3 R7.7) is shelved.
* **Counter-evidence and open questions**: Basins may appear at larger sizes or in sampled training. Running P1.3 R7.7 (direct sign-oracle probe on the $4\times4$ instance, about 250 CPU-h) would test the statement where it matters most.
* **How to use it**: Spend effort on expressivity (width, better features) rather than on basin escape, tempering or alternative optimisers for the sign; before proposing a basin route, show a better-signed state that the same network reaches and that has lower energy.
* **Sources**: P1.3 R3.1, R7.1, R7.2, R7.3, R7.5, R2.4; jobs 410199, 507179, 625182; master Revision Log (cross-node critic, Minor 8).

### L3. No function of Hamiltonian moments or of low-degree polynomial filters of the trial state can determine an observable error that sits in low-energy directions.
* **Status**: PROVISIONAL
* **Applies to**: error bars on observables (correlation functions, order parameters) of any variational state, including variance extrapolation, Lanczos-step extrapolation, covariance regressions on energy moments and local imaginary-time steps.
* **The lesson**: Write the trial state as $\psi=\sum_nc_n\lvert n\rangle$ in eigenstates of $\hat H$ with excitation energies $\omega_n=E_n-E_0$. The error $\delta=\langle\psi\rvert\hat A\lvert\psi\rangle-\langle\psi_0\rvert\hat A\lvert\psi_0\rangle$ of an observable $\hat A$ depends on the signs of the $c_n$ and on the matrix elements of $\hat A$, while energy moments depend only on $\lvert c_n\rvert^2$ and $\omega_n$, so moments are blind to the sign of the error. A polynomial filter of degree $K$ in $\hat H$ with spectral width $W$ changes a component at excitation $\omega$ by at most about $2K^2\omega/W$, so it barely touches the low-energy components where a converged state's observable error sits (in a two-level model the Krylov vector $(\hat H-E)\psi$ reaches the competitor only with weight proportional to the gap $\Delta$).
* **Evidence**: P3.1 R1.1: on the real $4\times4$ state, one and two Lanczos steps cut the energy error from 0.052 to 0.023 and 0.015 while the pair-correlation error moved only from $2.9$ to $2.6$ and $2.8\times10^{-5}$. P3.1 R3.1: the ensemble covariance regressor reduces to the first Lanczos moment; pilot 410398 found 0 of 6 $4\times3$ instances with a target. P3.1 R10.3: a depth-2 Krylov two-mode window dropped from 1.00 tracking at $3\times3$ to 0.61 at $4\times3$ (0.27–0.39 drop). R3.7, R5.4, R6.3 and R5.3 reduce to the same moments or filters by derivation.
* **Scope and limits**: Proven for moment functions and polynomial filters (P3.1 statements S1, S2). It does not cover rational filters: a shift-inverted residual escapes it (see L14 and L37).
* **Counter-evidence and open questions**: None among the tested cases; the open question is whether shift-inverted, trained corrections keep their exact-arithmetic gain (P3.1 R10.1d, AMBIGUOUS).
* **How to use it**: Do not build observable error bars from energy variances, Lanczos steps or other polynomial filters; test any proposed estimator first on a two-level model with a near-degenerate competitor.
* **Sources**: P3.1 R1.1, R2.1, R3.1, R3.7, R5.3, R5.4, R6.3, R10.3; job 410398; Sorella–Becca extrapolation (arXiv:cond-mat/0009149).

### L4. An eigenstate window that must contain every level carrying the observable error grows with the level density, so excited-state subspace corrections are an exact-diagonalization diagnostic, not a scalable error bar.
* **Status**: PROVISIONAL
* **Applies to**: excited-state Ritz corrections, observable-ordered windows and window-size extrapolations for long-range correlators in many-body variational methods.
* **The lesson**: A Ritz correction in a window of the $K$ lowest eigenstates fixes the observable error only if the window contains the levels that carry it. Define $K_{1/2}$ as the number of lowest in-sector levels needed to hold half of the error $\delta$. Because the number of levels below a fixed excitation energy grows as $e^{s V}$ with the number of sites $V$ (entropy density $s$), $K_{1/2}$ grows with size, and ordering the window by observable weight instead of energy does not help. The error outside the window and the data inside it are different linear functionals, so no extrapolation in $K$ can recover the missing part.
* **Evidence**: P3.1 R7.1: median $K_{1/2}$ for the pair aggregate 2, 1, 2, 17.5 and for charge 1, 1, 2, 13 on toys of $3\times2$, $3\times3$, $6\times2$, $4\times3$; on the real $4\times4$ state $K_{1/2}$ is 2 (pair), 6 (spin), 11 (charge). P3.1 R8.4: observable-ordered windows moved median $K_{1/2}$ for spin from 1.5 to 9 and charge from 1 to 19. P3.1 R8.6: a counting argument shows that with more than $K+1$ out-of-window levels some perturbations change $\delta$ and leave every window datum unchanged.
* **Scope and limits**: Real-space long-range correlators on toys up to 12 sites and one $4\times4$ state. At exact-diagonalization size the method works (job 454376: 16 of 21 injected items, pilot 456094 GO), which is why it stays a diagnostic.
* **Counter-evidence and open questions**: Observables carried by one or two isolated low levels could escape the growth; none of the tested ones did.
* **How to use it**: Use excited-state windows only to diagnose where an error sits at exact-diagonalization size; check $K_{1/2}$ at two or more sizes before proposing a window method as an error bar.
* **Sources**: P3.1 R7.1, R8.4, R8.6; jobs 454376, 456094, 618646 (shelved extension).

### L5. A summed observable such as a structure factor can be accurate while its long-range tail carries the error, and rebuilding the tail from per-mode pieces amplifies their errors about a hundredfold.
* **Status**: PROVISIONAL
* **Applies to**: choosing target observables for NQS error estimation and toy tests; any method that reconstructs real-space correlations from momentum-resolved pieces.
* **The lesson**: A structure factor sums the correlation function over all distances, so short-range contributions that the network gets right dominate it, while the long-range tail that decides order is a small part of it. A method validated on structure factors can therefore look accurate where the error that matters is invisible, and assembling the tail from per-mode corrections requires per-mode precision about a hundred times finer than the tail error, because the modes cancel.
* **Evidence**: P3.1 R8.3 (job 698163, real $4\times4$ width-6 states, seeds 11, 22, 33): pair structure-factor errors 2.6%, 3.7%, 5.0% and charge 0.8%, 0.8%, 0.0%, while real-space long-range correlators carry the error (verdict NO-TARGET). P3.1 R8.9: the pair tail is about 1% of the structure-factor error ($2.75\,\delta$ against $1.0$–$2.0\times10^{-2}$), so factor-2 tracking of the tail needs per-mode precision of about 0.5%, a cancellation ratio of order 100.
* **Scope and limits**: One $4\times4$ open $t$–$t'$–$J$ instance, three seeds; pair, spin and charge channels.
* **Counter-evidence and open questions**: In the spin channel the structure-factor error was larger ($-4.3\%$, $-13\%$, $-0.9\%$), so the statement is per channel, not universal over observables.
* **How to use it**: Measure the real-space long-range tail directly; never accept a toy result on a summed observable as evidence for the tail.
* **Sources**: P3.1 R8.3, R8.9; job 698163.

### L6. Certificates and bounds that need a spectral gap or a susceptibility measured inside the variational family fail exactly at competing-order points, where the gap closes and the family is stiffest.
* **Status**: PROVISIONAL
* **Applies to**: rigorous or semi-rigorous error certificates for variational states (Temple, Weinstein and Eckart bounds, fidelity bounds, doublet certificates, Legendre or susceptibility bounds, pinning-field extrapolation) near phase competition.
* **The lesson**: Gap-based bounds divide by a gap $\Delta$ to the next level (Temple needs a lower bound $\beta$ with $E<\beta\le E_1$; Eckart-type fidelity bounds scale as error over gap), and at a competing-order point that gap is the quantity that closes. Weinstein windows of width $2\sigma$ grow as $\sqrt V$ and contain exponentially many levels at finite energy density. Susceptibility bounds need the true susceptibility $\chi_A$, but only the family's $\hat\chi_A$ is measurable, and a family that cannot represent the competing fluctuation is stiffer ($\hat\chi_A<\chi_A$), so it reports a small error exactly when it is wrong. A pinning field gives a state with built-in order a linear response that extrapolates to its own order.
* **Evidence**: P3.1 R7.3: even with oracle energies the energy-window certificate floor was $C^*/\lvert\delta\rvert=81$ (pair), 8.6 (spin), 29 (charge) times the actual error (job 454376). P3.2 R5.1: on the 12-site torus crossing the rigorous Eckart bound commits only if the trained state's error ratio $\eta$ is at most 0.03–0.08 (state error $\varepsilon\le4.6\times10^{-5}t$ at the inner targets), while the trained states had $\eta=0.19$–$0.55$ at width 8, giving 0 of 12 certified commits. P3.2 R5.2: the in-doublet rotation error scales as $\varepsilon/\Delta_1$. P3.1 R1.2, R1.3, R6.5, R7.2 and P3.2 R1.3, R3.3 are killed by the same argument.
* **Scope and limits**: Arguments are general; measurements are from one $4\times4$ state (P3.1) and one $4\times3$ torus crossing (P3.2).
* **Counter-evidence and open questions**: Away from competition, with a robust gap, the bounds are valid and may be useful; the lesson is about where the Hard Core sits.
* **How to use it**: Do not propose a certificate for competing orders that divides by a gap or uses an in-family susceptibility; first check what it gives at a same-sector crossing.
* **Sources**: P3.1 R1.2, R1.3, R6.5, R7.2, R7.3; P3.2 R1.3, R3.3, R5.1, R5.2; job 454376; Temple, Proc. R. Soc. A 119, 276 (1928); former master anti-pattern "Bounds that need the gap".

### L7. Two macroscopically distinct orders share neither overlap nor local structure, so cross-order estimators cost samples exponential in the system size and shared-parameter error cancellation does not happen.
* **Status**: PROVISIONAL
* **Applies to**: deciding between competing orders with variational states: non-orthogonal configuration interaction (NOCI) between branches, ratio, mixture or correlated (paired) sampling of energy differences, curvature or susceptibility terms that need cross-order matrix elements, and shared-backflow or state-averaged ansätze meant to cancel errors.
* **The lesson**: Two states whose orders differ on an extensive region have overlap $\langle\psi_A\vert\psi_B\rangle$ and Bhattacharyya overlap $B=\sum_n\lvert\psi_A(n)\psi_B(n)\rvert$ that decay as $e^{-cV}$. Every sampled estimator of a cross-order quantity (ratio estimator, mixture sampling, paired sampling) has variance that grows as an inverse power of that overlap. The chemistry idea of making errors cancel by sharing parameters between compared species (isodesmic reactions) needs the two states to share local structure; distinct orders do not, so the bias stays differential.
* **Evidence**: P3.2 R1.4a and R7.2 (reopening-review T1, 12-site torus): $B\approx0.20$ already at 12 sites, and paired sampling cut the variance only by a factor 1.6 ($R_{\mathrm{pair}}\approx0.64$). P3.1 R3.3: seed NOCI needs cross-overlaps whose sampling cost is exponential in $V$. P3.2 R2.1 and R2.2: shared-backflow superposition and state-averaged two-reference backflow need heavy-tailed cross terms and are known ansätze. P3.2 R3.2: the curvature of a branch lacks the competitor term, because the competitor is outside the tangent space.
* **Scope and limits**: The decay is an argument for macroscopically distinct states; the only measurement is at 12 sites.
* **Counter-evidence and open questions**: For competitors that differ only locally (domain walls, few defects) the overlap does not decay and these estimators could work.
* **How to use it**: Before proposing any estimator that couples two competing branches, estimate how its variance scales with their overlap; expect only finite-size gains for distinct orders.
* **Sources**: P3.2 R1.4a, R2.1, R2.2, R3.2, R7.2; P3.1 R3.3; former master anti-pattern on non-orthogonal CI.

### L8. Ansätze that make the chemistry local energy cheap do so with configuration-independent or window-restricted orbitals, and they lose the external pair correlation that grows with the basis.
* **Status**: PROVISIONAL
* **Applies to**: neural or Jastrow-type ansätze for second-quantized molecular Hamiltonians whose local energy is made contractible or cheap by freezing orbitals, restricting the backflow input to an orbital window of size $K$, or coarse-graining the input.
* **The lesson**: In a molecular basis, energy optimisation puts the multireference correlation into the configuration dependence of the orbitals. A surrogate with configuration-frozen orbitals discards it, so its residual stays of order one; a backflow that sees only an active window is a single determinant in the external orbitals and misses the external pair correlation, which dominates the correlation energy as the basis grows. Removing external–external Jastrow pairs costs nothing measurable, which locates the binding limit in the frozen orbitals, not the Jastrow.
* **Evidence**: P2.2 R1.4: frozen-backflow control variate residual fraction 0.56–0.82 on H$_4$ and 0.69–0.80 on H$_6$ (STO-3G, $R=3.0$ bohr); cost ratio $M=1.03$–$1.07$ against the best control. P2.2 R4.1 (pilot 411919, distorted H$_4$ at $R=4.4$ bohr, $K=20$, 3 seeds): median error 0.691 mHa against the 0.2 mHa target. P2.2 R5.1 (job 452238): WIBJ error 0.013, 0.064, 0.187 mHa at $K=12$, 16, 20 against NNBF-Jastrow's 0.028 mHa, extrapolating to 0.61–7.3 mHa at $K=40$. P2.2 R6.1: contractible CAS×Jastrow 0.3885 mHa with external pairs masked against 0.3899 unmasked (H$_4$, $K=12$). P2.2 R2.6 by argument.
* **Scope and limits**: Hydrogen chains and rings, H$_4$ to H$_8$, STO-3G and small bases up to $K=136$ for cost, $K\le20$ for accuracy.
* **Counter-evidence and open questions**: A window that grows with the basis at sub-cubic cost was not found; it would overturn the lesson.
* **How to use it**: Judge a cheap-local-energy ansatz by its own absolute error trend in basis size against a minimal-contrast foil, not by a paired difference against a noisy unrestricted control.
* **Sources**: P2.2 R1.4, R2.6, R4.1, R5.1, R6.1; jobs 411919, 452238.

### L9. The size-extensive saving of local-energy screening in molecules comes from the localized orbital frame, not from the estimator.
* **Status**: PROVISIONAL
* **Applies to**: screening, truncation and certified-screening schemes for local energies of second-quantized chemistry NQS.
* **The lesson**: In canonical (delocalized) orbitals almost every connected term of the local energy is non-negligible, so screening keeps most of them at any system size. In localized orbitals the Hamiltonian's connected terms become sparse in space, so the kept fraction falls with system size. The saving that grows with size is therefore a property of the basis; an estimator layered on top changes it by a constant at most.
* **Evidence**: P2.2 R7.3 (linear H$_N$ chains, STO-3G, $R=3.0$ bohr, exact ground states): plain screening keeps 0.306, 0.144, 0.079 of the terms at $N=6$, 8, 10 in Boys-localized orbitals against 0.85–0.91 in canonical orbitals. P2.2 R7.1: a certified variational screen in canonical orbitals keeps 0.917, 0.949, 0.976 at the same sizes.
* **Scope and limits**: Linear hydrogen chains up to 10 atoms, exact states; not tested on trained NQS states or on 2D or 3D molecules.
* **Counter-evidence and open questions**: Whether certified screening in localized orbitals (P2.2 R7.4, AMBIGUOUS) keeps the saving is open.
* **How to use it**: Compare screening estimators only within the same orbital frame, and attribute size-dependent savings to the frame unless a frame-matched foil says otherwise.
* **Sources**: P2.2 R7.1, R7.3, R7.4; Boys-localized NQS chemistry (arXiv:2305.12776).

## About methods

### L10. A correction with a fixed number of terms, components, projector steps or global vectors gives a gain that does not grow per site, so it is a finite-size device; size consistency needs a product form inside one object.
* **Status**: PROVISIONAL
* **Applies to**: any ansatz or correction for an extensive quantum system that adds a fixed number $K$ of terms: linear dressings $(1+\sum_b\alpha_b\hat K_b)\Phi$, fixed-rank sums of mean-field states or determinants, mixtures of experts, a fixed number of Lanczos or power steps, linear readouts of a global summary, linear corrections in a fixed set of global vectors, and symmetry restoration by projection.
* **The lesson**: The error of a variational state in a system of $V$ sites is extensive, of order $V$. Components that differ on an extensive region are exponentially orthogonal, so a fixed linear superposition of them gains $\mathcal O(1)$ energy, at most $\mathcal O(\sqrt V)$ in the configuration-interaction-like case; a fixed number of Lanczos steps gains about $\sqrt{\kappa_2}\propto\sqrt V$ (with $\kappa_2$ the extensive energy variance), and symmetry projection restores symmetry at $\mathcal O(1)$. So the relative improvement vanishes as $V$ grows. A finite projector step $(1+\alpha\hat H)\psi$ also changes signs only on configurations with $\alpha E_{\mathrm{loc}}<-1$, a local condition, while sign errors differ by global moves of the nodal surface. A sum of $K$ low-degree terms cannot represent a sign that is the parity of $M$ decoupled blocks (parity has polynomial-threshold degree $M$). The size-consistent fix, a product over regions, must live inside one object: multiplying normalised ratios that share one reference determinant divides by it once per factor and creates a pole of order $V-1$ at its nodes, and a product of components loses closed-form symmetry projection (see L11).
* **Evidence**: P4.1 R1 (job 508440): on a 12-site open cluster a rank-8 exact-spin mixture beat the single component (`che`) by a factor 2.5 ($1.35\times10^{-4}$ against $3.43\times10^{-4}$), but the T0 argument gives $R(K,N)=\epsilon_{K}/\epsilon_{\mathrm{che}}\to1$ with size, and the VMC pilot projected $R(8,48)=0.64$–$0.82$ (killed at Gate 4). P3.1 R10.1: in a toy of $M$ decoupled copies, a fixed-base response correction recovered 0.95 falling to 0.81 and 0.91 falling to 0.53 of the error as infidelity grew, against 0.99–1.01 with the base updated each pass. P3.2 R6.5 and R7.1: a sampled single Lanczos step, gain $\propto\sqrt V$ against bias $\propto V$. P2.1 R3.1 and R3.2: shared-reference low-rank sums, parity lemma; job 526080 found only finite-size gains (AMBIGUOUS). P2.1 R6.2: product of rank-1 ratio factors, $\lvert\psi\rvert\propto\varepsilon^{-35.11}$ near reference nodes at $M=V=36$ (predicted $\varepsilon^{-35}$), local energy near $-9\times10^{20}$ at $L=8$. P2.1 R6.1 and R7.4, P1.2 R2.3 and R7.5, P4.1 R2.1, R4, R5, R9, R12, RG1, RG2, P2.2 R7.2, P2.1 R6.3, P4.2 R9.7 and P1.3 R1 die on the same argument.
* **Scope and limits**: Mostly T0 arguments plus measurements at 8 to 48 sites; the scaling claims are asymptotic and say nothing against such corrections at a fixed small size.
* **Counter-evidence and open questions**: P4.1 R1's size ladder (job 686990) was cancelled, so the measured size trend for that route is a pilot projection, not a ladder.
* **How to use it**: Run Gate 4 by argument before any size ladder: ask whether the number of terms, steps or vectors grows with $V$, and if a product form is used, whether it lives inside one determinant or Gaussian.
* **Sources**: P1.2 R2.3, R7.5; P1.3 R1; P2.1 R3.1, R3.2, R6.1, R6.2, R6.3, R7.4; P2.2 R7.2; P3.1 R10.1, R10.2; P3.2 R6.5, R6.6, R7.1; P4.1 R1, R2.1, R4, R5, R9, R12, RG1, RG2; P4.2 R9.7; jobs 508440, 526080; Becca et al., arXiv:1412.2656 (fixed Lanczos steps are not size-consistent); former master anti-pattern "A summed or fixed-rank construction is a finite-size device".

### L11. Exact non-Abelian symmetry at polynomial cost is available only when every symmetry-breaking ingredient sits inside an object on which the group acts in closed form.
* **Status**: PROVISIONAL
* **Applies to**: exact SU(2) (total-spin) projection, or other non-Abelian projections, of neural or correlated fermion wave functions.
* **The lesson**: Projection onto total spin $S$ integrates the state over the group. If the spin-breaking part of the amplitude is a Gaussian (a determinant or pairing state) for each charge configuration, the rotated Gaussian has a closed form and the integral costs a polynomial number of quadrature nodes. Any diagonal factor that reads the broken quantity directly (a spin Jastrow, a spin-resolved network, spin features, plaquette exchange correlators) makes the rotated amplitude a sum over an exponential orbit of configurations. Sampling that orbit sum instead trades the exponential cost for an exponential variance: the rotation kernel's absolute sum grows as $(\lvert\cos\tfrac\beta2\rvert+\lvert\sin\tfrac\beta2\rvert)^{N_s}$, so the average sign decays as $e^{-a\sqrt{N_s}}$ with the number of spins $N_s$.
* **Evidence**: P4.1 R0.2: stochastic SU(2) projection of a spin-resolved NNBF sums $\binom{N_s}{N_s/2+M}$ signed terms and is biased through noisy amplitude ratios. P4.1 R2.4: numerator-only sampling, average sign $e^{-a\sqrt{N_s}}$. P4.1 R10: a size-extensive spin Jastrow needs a Hubbard–Stratonovich sum over $2^{\#\mathrm{bonds}}$ fields. P4.1 R3.2, R4.3, R5.1, R5.3, R5.7 and R13 fail the same way (Gate 1). The closed-form class (charge-backflowed Gaussians, `che`, `pcg`) projects exactly with $\langle S^2\rangle$ exact to $10^{-16}$.
* **Scope and limits**: Proven for SU(2) on Hubbard-type models; the closed-form class is not shown to be accurate enough (see P4.1's open question on R2).
* **Counter-evidence and open questions**: None within the tested constructions; a new closed-form group-covariant object beyond Gaussians would extend the class.
* **How to use it**: Before designing an exact-symmetry ansatz, check that every ingredient that breaks the symmetry is inside a group-covariant object with a closed-form rotation; otherwise count the orbit sum at T0 and stop.
* **Sources**: P4.1 R0.2, R2.4, R3.2, R4.3, R5.1, R5.3, R5.7, R10, R13 (Lemma B in the P4.1 leaf).

### L12. Any single-proposal importance estimator of a non-stoquastic sum has a variance floor that does not vanish as the state converges.
* **Status**: PROVISIONAL
* **Applies to**: stochastic local energies (heat-bath, semistochastic, factorized-Hamiltonian proposals), stochastic determinant-ratio estimators and doubly stochastic SR forces for fermionic NQS.
* **The lesson**: Estimating $\sum_jx_j$ with one importance distribution has per-draw variance at least $(\sum_j\lvert x_j\rvert)^2-(\sum_jx_j)^2$, which is non-zero whenever the terms have mixed signs, as in non-stoquastic fermionic Hamiltonians, and does not shrink as the state converges (unlike the exact local energy, whose variance vanishes at an eigenstate). Non-linear use of the estimate (ratios, the inverse SR metric) adds bias and amplifies the noise. The cost relative to exact enumeration therefore diverges exactly where accuracy matters. Multi-proposal or sign-stratified schemes are not covered, but they did not help where magnitude errors dominate (P2.2 R2.1).
* **Evidence**: P2.2 R1.1 (H$_6$ chain, STO-3G, $R=3.0$ bohr, state 0.17 mHa from the ground state): heat-bath costs $657\times$ exact enumeration; semistochastic $0.932\times$ against plain screening $0.911\times$. P2.2 R1.3: even the oracle proposal $p\propto\lvert x\rvert$ costs $145.6\times$ exact. P2.2 R1.2: the SR force noise is amplified by $S^{-1}$ along small-eigenvalue directions. P2.1 R4: an unbiased signed Russian-roulette ratio series has variance diverging near nodes.
* **Scope and limits**: Small molecules (H$_4$, H$_6$) for measurements; the floor itself is a general inequality.
* **Counter-evidence and open questions**: None measured.
* **How to use it**: Compare any stochastic local energy against exact enumeration with screening at a converged state, not at a poor one; prefer deterministic reductions of the number of terms.
* **Sources**: P2.2 R1.1, R1.2, R1.3; P2.1 R4; former master global anti-pattern "Single-proposal stochastic replacement of an exact non-stoquastic sum"; semistochastic NQS local energies (arXiv:2507.19276).

### L13. Any error estimate built from the variational tangent space or the variational dynamics of a converged state vanishes identically, because the state is stationary there.
* **Status**: PROVISIONAL
* **Applies to**: error bars from tangent-space Ritz (linear method at convergence), in-family zero-bias corrections, Laplace or Fisher posteriors, time-dependent variational dynamics and eigenvector continuation of nearby trained states.
* **The lesson**: At a converged variational minimum the residual $r=(\hat H-E)\psi$ is orthogonal to every tangent vector $\partial_k\psi$ (Brillouin condition), so any correction expanded in the tangent span gives zero, the tangent Ritz vector is $\psi$ itself, and the projected (TDVP) dynamics has $\psi$ as a fixed point. Corrections from nearby trained states (field ladders) differ from $\psi$ by tangent vectors to first order and share its bias at finite field. Curvature-based posteriors give a small error bar when the family is stiff along the observable, which is when it is wrong (L6).
* **Evidence**: P3.1 R3.4 (lemma, Round 3), R7.4, R7.5, R8.2 (derivations), R3.6 (stiff-foil mechanism).
* **Scope and limits**: Exact for converged states; an unconverged state leaves a non-zero but uninformative remainder.
* **Counter-evidence and open questions**: None.
* **How to use it**: Reject any error estimator that only uses the trained state's own tangent space or dynamics; information must come from outside the family (an exact or trained shift-invert, a different family).
* **Sources**: P3.1 R3.4, R3.6, R7.4, R7.5, R8.2; Frame et al., PRL 121, 032501 (2018).

### L14. A per-observable response vector recovers the observable error exactly at first order, but its tolerance to training error shrinks as the density of low-lying levels grows.
* **Status**: PROVISIONAL
* **Applies to**: zero-bias corrections with trained improved functions (Assaraf–Caffarel type), response-window Ritz methods and shift-inverted responses for observable error bars.
* **The lesson**: The response $\lvert\chi\rangle=(\hat H-E_0)^{-1}Q(\hat A-a)\psi$, with $Q$ the projector off $\psi$, gives the first-order error of $\langle\hat A\rangle$ exactly. Built with a trained network, it enters a Ritz problem whose denominator is its energy-weighted norm, which is dominated by low-lying levels; as their density grows, a fixed relative training error in $\chi$ moves the correction by more, so tracking falls with size.
* **Evidence**: P3.1 R8.1: response-window Ritz with noisy responses tracked (estimate within a factor 2 of the true error) 0.80 of items at $3\times2$ and 0.37 at $4\times3$. P3.1 R8.8: 0.86 at $3\times2$ and 0.33 at $4\times3$ on structure factors. P3.1 R9.1: Krylov-cleaned responses 0.91 at $3\times3$ and 0.56 at $4\times3$. P3.1 R3.2: the first-order oracle over-stated the error by 1.5–2.2 at infidelities 0.02–0.2. With exact responses the method recovered 8 of 8 real $4\times4$ items (699756), and trained responses tracked 9 of 11 items on a $4\times3$ T3 (705951).
* **Scope and limits**: Toys up to 12 sites, noise models for training error; the exact-response success is an inverse-iteration premise (L37).
* **Counter-evidence and open questions**: The self-consistent version (P3.1 R10.1d) is AMBIGUOUS: trained responses 9 of 11, a twice-wider network 7 of 10.
* **How to use it**: Test any response-based error bar at two or more toy sizes with the realistic training-error level, and compare with a matched capacity foil (L38).
* **Sources**: P3.1 R3.2, R8.1, R8.8, R9.1, R10.1d; jobs 699756, 705951.

### L15. A Ritz correction built from noisy generated vectors cannot resolve a competitor closer than a noise floor, and a gap witness built from the same vectors over-estimates the gap.
* **Status**: PROVISIONAL
* **Applies to**: Davidson-type or observable-sourced Ritz corrections and gap witnesses built from sampled or trained vectors.
* **The lesson**: Each generated vector carries a noise of relative size $\eta$ spread over levels at typical excitation $\bar\omega_w$. In the Ritz problem that noise acts like a level shift of order $\eta^2\bar\omega_w$ divided by the coherent weight $w_c$ the sources put on the competitor. A competitor closer than that floor is mixed away, and the Ritz gap between the two lowest Ritz values reports the floor, not the true gap, so a certificate that "the gap is large" is false exactly when the gap is small.
* **Evidence**: P3.1 R10.4 (observable-free generalised Davidson with grade-R noise): native gaps tracked 1.00, 1.00, 0.80 at $4\times2$, $5\times2$, $4\times3$, but with a competitor moved to $0.02t$ (stress) only 0.00, 0.12, 0.18; noise-free it tracked 1.00. P3.1 R12.1: at stress $\Delta=0.02t$ the witness gap was $0.41t$ and $0.47t$ (twenty times the true gap), with false-certificate rates 0.88 and 0.71; the Davidson-sourced witness fails identically.
* **Scope and limits**: Toys of 8 to 12 sites with a modelled noise grade.
* **Counter-evidence and open questions**: An interval readout (P3.1 R12.3d, live) is designed to stay honest below the floor; it is untested.
* **How to use it**: Report any Ritz-based error bar with its noise floor $\eta^2\bar\omega_w/w_c$, and test it with an artificial competitor moved below that floor.
* **Sources**: P3.1 R10.4, R12.1, R12.3d.

### L16. Ritz diagonalisation in the span of two trained variational states returns the state with the lower trained energy, so it cannot beat their differential bias.
* **Status**: PROVISIONAL
* **Applies to**: decisions between competing orders from trained variational states: excited-state Ritz with a penalty-trained partner, Ritz in the span of two order branches, jointly optimised two-state ensembles relabelled by character.
* **The lesson**: For a real symmetric $2\times2$ matrix the lowest eigenvector has weight at least $\tfrac12$ on the basis vector with the lower diagonal entry, whatever the coupling. So a Ritz rotation of two trained states decides like the energy rule between them: it picks the state whose trained energy $E_X+b_X$ (true energy plus variational bias $b_X$) is lower, and is wrong whenever the differential bias $\Delta b$ exceeds the true gap $\delta$. Off-diagonal information can change this only through the coupling $W$, which is bounded by about half the smallest gap.
* **Evidence**: P3.2 R6.1 (job 619080, 84 CPU-h, width 4, 9 instances on a $4\times3$ Hubbard torus with a same-sector crossing): the Ritz decision equalled the lower-diagonal prediction on 9 of 9 and was right on 4 of 9; diabatic coupling $W\approx5\times10^{-4}t$ against measured $\Delta b=3.6$–$5.4\times10^{-3}t$. P3.2 R6.2 and R6.3: the same lemma for the residual-augmented and two-branch spans. P3.2 R1.4: the jointly optimised ensemble relabelled by character reduces to the same rule. P3.1 R7.7: multi-seed Ritz inherits the seeds' shared bias.
* **Scope and limits**: One 12-site torus crossing between spin textures forced by $L_y=3$, widths 4 and 8; the lemma itself is general.
* **Counter-evidence and open questions**: A symmetric block-Lanczos step with exact matrices escapes it (9 of 9 in-sample), but that is the exact Hamiltonian at work (L37), and sampled, it abstains (L45).
* **How to use it**: Do not expect a Ritz rotation of trained states to decide better than comparing their energies; use the plain energy rule with honest abstention.
* **Sources**: P3.2 R1.4, R6.1, R6.2, R6.3; P3.1 R7.7; job 619080.

### L17. Any probe or transport of a capacity-limited ansatz (field response, hysteresis, basin stability, continuation of snapshots) measures the biased energies, so it inherits the energy rule's decision boundary.
* **Status**: PROVISIONAL
* **Applies to**: deciding competing orders with variational methods by conjugate fields, level-crossing kinks, Berry phases in parameter space, branch-stability kicks, eigenvector continuation or affine transport of snapshots.
* **The lesson**: A capacity-limited ansatz in branch $X$ has energy $E_X+b_X$, with $b_X$ its variational bias. Every functional of its landscape (where branches cross under a field, which basin survives a kick, where the lower envelope has a kink, where the Berry phase sees a seam) is a functional of those biased energies, so it puts the decision boundary at $\delta+\Delta b=0$ (true gap $\delta=E_A-E_B$, differential bias $\Delta b=b_A-b_B$), the same boundary as comparing energies. Transporting states from where training is easy to where the decision is hard removes optimisation failure at the target but carries the snapshots' differential bias with it.
* **Evidence**: P3.2 R3.5 (job 451852, 1,412 CPU-h): same-order eigenvector continuation at width 4 (676 parameters) committed on all 15 hard (target, seed) pairs and was wrong on 5 (wrong-commit rate 0.33), always toward order $A$ on the $B$ side, while oracle continuation from exact snapshots was right everywhere. P3.2 R3.6: even with exact snapshots, affine transport commits the wrong order at $d=-0.0002$ because the diabats' curvatures differ ($-14$ against $-10$), a transport bias of $2.1\times10^{-3}t$ above the $1.4\times10^{-3}t$ gap. P3.2 R2.3, R2.7, R3.1 and R7.4 by argument.
* **Scope and limits**: One $4\times3$ torus crossing; widths 4 and 8.
* **Counter-evidence and open questions**: None among tests; a probe that uses information outside the family (exact $\hat H$ action) is not covered.
* **How to use it**: Before proposing a landscape probe, write its decision boundary in terms of $E_X+b_X$; if it is $\delta+\Delta b=0$, it is the energy rule in disguise.
* **Sources**: P3.2 R2.3, R2.7, R3.1, R3.5, R3.6, R7.4; job 451852.

### L18. Parameter continuation from the non-interacting or sign-free limit follows the adiabatically connected branch and keeps the wrong state at near-degenerate level crossings.
* **Status**: PROVISIONAL
* **Applies to**: annealing, continuation or warm starting in a Hamiltonian parameter (interaction strength, seam coupling, hopping) for NQS and other variational methods near competing orders.
* **The lesson**: A mean-field start already is continuation from the $U=0$ point. Following the lowest variational root along a path tracks the adiabatically connected branch; at first-order or avoided level crossings with tiny gaps, finite-step training crosses diabatically and stays on the metastable branch. The Hard Core of competing orders consists of exactly such crossings, so continuation fails where it is needed.
* **Evidence**: P1.3 R4 (argument; the default start). P4.2 R2.1: ring-surgery continuation in a seam coupling breaks translation for $0<\lambda<1$, turning crossings into avoided crossings; a login smoke landed on $E_1$, not $E_0$. P1.3 R3.3: diabatic branch transport in $t'$ ends in the energy comparison that P3.2 showed fails (L17).
* **Scope and limits**: Mostly arguments and one smoke run; no ladder of sizes.
* **Counter-evidence and open questions**: Continuation can help when the path avoids crossings; the lesson applies at competing-order points.
* **How to use it**: Do not propose continuation or warm-start paths through a competing-order region as a fix for the wrong branch; test any path for crossings first.
* **Sources**: P1.3 R3.3, R4; P4.2 R2.1; former master global anti-pattern "Continuation from the sign-free or mean-field limit".

### L19. On a discrete configuration space, sampler-only changes cannot change which basin an exact-expectation optimiser reaches.
* **Status**: PROVISIONAL
* **Applies to**: sign-flip-aware Metropolis moves, node-targeted or importance-modified sampling, and other sampler changes proposed to fix the trained nodes of a lattice NQS.
* **The lesson**: On a lattice, Metropolis sampling of $\lvert\psi\rvert^2$ jumps between configurations and never visits the zero-amplitude points between them, so nodes do not confine it (unlike continuum diffusion). The SR update is a function of expectation values, so in the infinite-sample limit its fixed points do not depend on the sampler; a sampler change alters only the noise, not the flow.
* **Evidence**: P1.3 R3 and P1.2 R6.6 (arguments); every decisive test in P1.2 and P1.3 used exact sums, where no sampler enters, and still found the same trained nodes.
* **Scope and limits**: Exact-expectation optimisers; at finite samples a sampler changes noise and can change stochastic trajectories, but not the targeted fixed points.
* **Counter-evidence and open questions**: None.
* **How to use it**: Fix trained nodes through the ansatz or the objective, not the sampler.
* **Sources**: P1.3 R3; P1.2 R6.6.

### L20. A sign layer that sees only local occupations cannot locate a reference determinant's nodes; sign corrections must be conditioned on the reference's amplitudes.
* **Status**: PROVISIONAL
* **Applies to**: sign correlators, Walsh or plaquette sign factors and correlator-product states (CPS) multiplied onto a Slater, pairing or backflow reference.
* **The lesson**: The sign error of a reference sits near its nodal surface, which is a global function of the configuration. The same local occupation pattern can be right-signed in one region of configuration space and wrong-signed in another, so a factor that reads only local occupations cannot tell where to flip. A single threshold over an extensive sum also cannot express a product of independent local repairs.
* **Evidence**: P1.2 R4 (smoke on $3\times4$ open, $t'=-0.3t$, layer fitted to the exact residual sign on 1500 samples): the sign-mismatch weight $W$ moved only from 0.073 to 0.072 and the fixed-node energy rose, $M_{\mathrm{local}}=1.03$ against $M_{\mathrm{null}}=1.15$ for a non-local control (both above the FAIL threshold 0.9). P1.2 R2.1: the plaquette sign product is CPS times Slater, which has not replaced backflow for this reason.
* **Scope and limits**: One smoke measurement plus prior-art reasoning.
* **Counter-evidence and open questions**: Conditioning on the reference's amplitudes (ratio features) does locate wrong signs (AUC 0.93), but locating does not correct them (L21).
* **How to use it**: Feed sign layers with the reference's amplitude information, not occupations alone, and expect that to be necessary, not sufficient.
* **Sources**: P1.2 R2.1, R4; Changlani et al., PRB 80, 245116 (2009); Neuscamman et al., PRB 84, 205132 (2011).

### L21. Knowing where a trained backflow's signs are wrong does not lower its sign error: features that locate wrong signs feed amplitude capacity, not sign capacity, under energy training.
* **Status**: PROVISIONAL
* **Applies to**: sign layers, votes, ratio features and second passes added to neural backflow (NNBF) or other determinant NQS to repair their signs.
* **The lesson**: Determinant-ratio features of a trained backflow predict where its signs are wrong, but three steps separate that information from a lower sign error, and each failed. A fixed, thresholded vote on those ratios is harmful (it flips right signs as well), although the same ratios used as learned features are informative. An information premise measured on the starting determinant does not predict a gain after training. And when the features enter a trainable orbital update, SR spends the new capacity on amplitudes: total energy falls, the sign error moves only as much as extra width moves it.
* **Evidence**: Notation: $r$ is the sign error $\Delta E_{\mathrm{sign}}$ of the route divided by that of the foil widened to the same parameter count. P1.2 R8.3 (gapped toys, both arms warm-started from the same trained foil, 300 more SR steps, seeds 11 and 22): $r=0.991$ at $6\times2$ $5+5$ and 0.991 at $4\times3$ $5+5$, while features locate wrong signs with held-out AUC 0.93 against 0.65 for occupations. P1.2 R9.0 (low-rank carrier, the closed leaf's own falsifier): $r=1.05$ and 1.00, with lower total energy (0.120 against 0.134 at $4\times3$). P1.2 R7.2, R7.3, R7.4: trained second passes $r=1.42$–$2.35$ at $V=10$, 12 although untrained AUC gains were 0.03–0.25. P1.2 R6.7F: a frozen vote raised the foil's sign error, $q=1.75$ and 1.31 at $V=10$, 12. P1.2 R5.2: exchange votes, $\rho_{\mathrm{sign}}=1.29$ and local-energy tails 11.9 times heavier on the $4\times4$ instance. P1.2 R8.4: a tied re-entry's 4×3 win disappeared once the optimiser shift was matched ($r$ 0.895 to 0.956).
* **Scope and limits**: Unprojected toys up to 12 sites at 300 steps, plus the projected $4\times4$ instance for R5.1 and R5.2; the cross-node critic scoped the diagnostic to these horizons.
* **Counter-evidence and open questions**: Longer horizons or equal-wall-time comparisons (P1.2 R9.4) and a narrow projected foil (R9.2, UNDECIDED) were not run.
* **How to use it**: Before building a sign layer on located errors, show at T1 that a trained arm lowers the sign error faster than width, not only the total energy.
* **Sources**: P1.2 R5.2, R6.7F, R7.2, R7.4, R8.2, R8.3, R8.4, R9.0; job 451916.

### L22. Off-loading amplitude into a positive correlator does not buy sign accuracy at matched parameters; the same parameters spent on backflow width win.
* **Status**: PROVISIONAL
* **Applies to**: Jastrow-type positive factors (including ratio-fed Jastrows) proposed to free a backflow's capacity for signs.
* **The lesson**: A positive factor has the reference's sign by construction, so it can help signs only indirectly, by taking over amplitude work. At matched parameter count that indirect route is no better than spending the parameters on backflow width, which improves signs and amplitudes together, and is usually worse.
* **Evidence**: P1.2 R6.3 and R7.1 (unprojected ladders, 300 exact SR steps, seeds 11 and 22): the most informative ratio-Jastrow arm against matched width gave $r_W=2.04$, 0.95, 1.19 at $V=8$, 10, 12 (a 5% edge at $V=10$ only, far from the pass threshold $r\le0.7$), while beating a plain neural Jastrow ($r_J=0.56$, 0.67, 0.48); every positive factor lost to width ($r_{JW}$ 1.4 to 3.6).
* **Scope and limits**: Toys up to 12 sites, unprojected, 300 steps.
* **Counter-evidence and open questions**: None at the tested sizes.
* **How to use it**: Use width as the foil for any capacity-freeing factor.
* **Sources**: P1.2 R3.0, R6.3, R7.1.

### L23. A complex phase carrier escapes the zero-crossing barrier of real nodes but not the span barrier: at a real stationary state it moves only if its feature span contains a negative-curvature direction.
* **Status**: PROVISIONAL
* **Applies to**: complex or phase-carrying extensions of real fermionic NQS (complex backflow, phase networks) meant to move nodes without crossing zero.
* **The lesson**: At a real trained state the energy gradient with respect to any phase is zero, so SR moves the phase only through second order, and only along directions where the energy curvature $\lambda$ is negative. Taking the real part of a complex backflow does not couple its parts at all: $E[\psi_c]$ is a weighted mean of the energies of the two real components. So a phase carrier helps only if its span contains a negative-curvature direction, which a cheap second-order screen $\lambda(S)$ of the span can test before training.
* **Evidence**: P1.2 R9.1: the ratio-fed phase carrier returned to $\lvert\cos\varphi\rvert\ge0.996$ in every run, equal to the foil ($r=1.30$ at $6\times2$, 1.16 at $4\times3$); an unrestricted per-configuration phase did descend (flipping 1.7% of the weight at $4\times3$). Curvatures: $\lambda=0.35$, 0.31 (linear ratio span), 0.26, 0.22 (quadratic), 0.37, 0.33 (occupations) against $-1.76$, $-1.70$ for the unrestricted direction. P1.2 R6.4: the complex detour by derivation.
* **Scope and limits**: Two toys of 12 sites, one foil.
* **Counter-evidence and open questions**: A feature span with negative curvature was not found; non-extensive or non-local features might have one.
* **How to use it**: Compute $\lambda(S)$ of a proposed phase span at the trained foil before training it.
* **Sources**: P1.2 R6.4, R9.1.

### L24. Learned sign networks, neural Pfaffians and local sign correlators for lattice fermions are published constructions whose adoption barrier no proposal here removed.
* **Status**: PROVISIONAL
* **Applies to**: proposals of new fermionic NQS sign structures; novelty checks (Gate 0) in this field.
* **The lesson**: Jordan–Wigner sign times a learned sign network, complex RBM times a pair product, neural Pfaffians of a learned pairing matrix and power-method sign refits are all published. Their known barriers are that the 2D Jordan–Wigner string is non-local, that generic sign networks generalise poorly from samples, and that Pfaffian or determinant nodes move only by deforming a matrix through zero. Proposals that recombine these pieces without a mechanism against those barriers are relabellings; in round 1, five of nine leaf routes died at Gate 0 this way.
* **Evidence**: P1.2 R1 (Inui, Kato and Motome, arXiv:2108.08631; Nomura et al., PRB 96, 205152; Choo et al., Nat. Commun. 11, 2368); P1.2 R5 (arXiv:2507.10705; arXiv:2405.14762); P1.3 R1 (SWO, arXiv:1811.12423); master Revision Log, 2026-11-11 saturation note.
* **Scope and limits**: A literature finding as of November 2026.
* **Counter-evidence and open questions**: None.
* **How to use it**: State the adoption barrier of the nearest published construction and the mechanism against it before testing any new sign structure.
* **Sources**: P1.2 R1, R5; P1.3 R1; former master global anti-pattern "Re-labelled sign networks"; Szabó and Castelnovo, PRR 2, 033075.

### L25. Cheap local-energy surrogates have sample-dependent errors far larger than the local-energy variance on converged states, so cross-sample debiasing cannot pay.
* **Status**: PROVISIONAL
* **Applies to**: multifidelity, control-variate and learned-regressor corrections of the local energy $E_L$ in VMC.
* **The lesson**: A cross-sample correction estimates the mean of the error $\delta=E_L-\tilde E_L$ of a cheap surrogate $\tilde E_L$ on a subset of samples. It pays only if $\mathrm{Var}(\delta)$ is well below $\mathrm{Var}_\pi(E_L)$. On converged states $\mathrm{Var}_\pi(E_L)$ is small, while cheap surrogates miss the configuration-dependent correlation, so their errors vary much more from sample to sample than $E_L$ itself.
* **Evidence**: P2.2 R3.2′ (H$_4$, $M=24$ orbitals): $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)=466$ (frozen backflow), 19.5 (linearized), 2.3–48 (screened completions); best multifidelity cost ratio 0.96; regression $R^2$ of $E_L$ on all cheap features 0.13. P2.2 R5.6: a learned regressor cannot beat that measured information content.
* **Scope and limits**: H$_4$ and H$_6$, small bases, near-converged states.
* **Counter-evidence and open questions**: Far from convergence the ratio is smaller and debiasing could help, but that is not where cost matters.
* **How to use it**: Measure $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ on a converged state before building any surrogate correction.
* **Sources**: P2.2 R3.2′, R5.6.

### L26. Near convergence the deterministic work per local energy sets the cost, so a cheaper local energy helps only if its saving buys more fresh samples.
* **Status**: PROVISIONAL
* **Applies to**: cost comparisons of local-energy estimators for VMC near convergence, in chemistry and on lattices.
* **The lesson**: On realistically converged states the sampled (stochastic) part of a semistochastic local energy adds only a few per cent to the variance, and the deterministic work per key (per evaluated term class) is most of the cost. Better sampling of the residual then buys nothing; only fewer keys or fewer evaluations do. A saving matters only if it is spent on more independent samples, and a fair comparison lets the controls spend their own savings the same way.
* **Evidence**: P2.2 R3.2 (H$_6$ ring): the residual added only 1–2% to $\mathrm{Var}(E_L)$; deterministic floor about 0.33 of exact enumeration, while semistochastic costs 0.55–0.70; cost ratios $M=1.04$–$1.85$. P2.2 R7.5: the semistochastic tail variance was only 1–2%. P2.2 R5.2 (anchored local energy, five seeds, budget 100 exact steps): median error 17.0 mHa against semistochastic 7.0 and screening 9.0 mHa; every semistochastic seed (4.8–7.6) below every anchored seed (10.6–21.1).
* **Scope and limits**: Small molecules at convergence; R5.2 is formally AMBIGUOUS because the noise floor was widened by trapped exact seeds.
* **Counter-evidence and open questions**: Step-limited optimisation far from convergence may still benefit from anchored reuse.
* **How to use it**: Profile where the cost sits on a converged state before optimising an estimator; compare at equal wall time with controls that reinvest their savings.
* **Sources**: P2.2 R3.2, R5.2, R7.5.

### L27. A rigorous certificate on a screened local energy that ignores sign cancellations is so loose that certified screening keeps almost every term.
* **Status**: PROVISIONAL
* **Applies to**: certified or bounded truncation of local-energy sums (Hadamard bounds, diagonal positive-semidefinite certificates).
* **The lesson**: A bound that treats every dropped term at its worst case, without cancellation between terms of opposite sign, pays at least twice for every sign-favourable dropped term. The certified kept fraction therefore stays near one and rises with system size, while uncertified screening drops most terms with a tiny actual error.
* **Evidence**: P2.2 R5.3 (H$_4$, $K=12$, warm state): the Hadamard-certified bias was 140–1000 times the actual dropped sum; at threshold $10^{-4}$, 0.62 mHa certified against $2\times10^{-4}$ mHa actual. P2.2 R7.1 (H$_N$ chains, canonical orbitals): certified kept fraction 0.917, 0.949, 0.976 at $N=6$, 8, 10 against uncertified 0.891, 0.910, 0.846.
* **Scope and limits**: Hydrogen chains and rings up to 10 atoms; canonical orbitals. In localized orbitals (P2.2 R7.4) the outcome is AMBIGUOUS.
* **Counter-evidence and open questions**: A certificate that accounts for cancellations was not found.
* **How to use it**: Compare a certified screen's kept fraction with plain screening on one exact state before building it.
* **Sources**: P2.2 R5.3, R7.1, R7.4.

### L28. Carrying fewer than one hidden row per site to shrink a hidden-fermion determinant's inverse loses sign capacity on 12- to 15-site clusters.
* **Status**: PROVISIONAL
* **Applies to**: hidden-fermion determinant and pairing states (HFDS, HFPS) and other augmented-determinant NQS whose cost is reduced by thinning hidden rows (sublattice rows, hole-gated rows).
* **The lesson**: The sub-cubic local energy of patch-local hidden rows pays at reachable sizes only if the carried matrix is smaller than one hidden row per site. Every way of thinning the rows tested (a fixed checkerboard, a quarter lattice, rows only on holes) lost sign quality by a factor that grows with cluster size beyond 8 sites, more than the cost saving can justify.
* **Evidence**: Notation: $\rho$ is the route's sign error divided by that of the full-row arm at the same hidden width $F=4$; $R_D$ is the cost per decorrelated sample of dense HFPS divided by that of the route at $L=16$ (compiled). P2.1 R11.1 (`half`): $R_D=5.05\pm1.27$, $\rho=2.44$, 1.59, 2.07, 1.80 at $V=8$, 12, 14, 15. P2.1 R11.2 (`quarter`): $R_D=10.26\pm2.45$, $\rho=9.92$, 1.86, 6.15, 2.75. P2.1 R13.2 (hole-gated, partial job 707185): $R_D=4.45\pm1.04$, $\rho=0.83$, 1.37, 2.04 at $V=8$, 12, 15. All on open $t$–$t'$–$J$ toys, seed 11.
* **Scope and limits**: Single seed, $F=4$, toys of 8 to 15 sites. The `half` arm at larger $F$ (P2.1 R12.3) is live and partially measured ($\rho$ 0.04, 0.89, 1.11, not scored).
* **Counter-evidence and open questions**: If R12.3 passes at larger $F$, the lesson narrows to fixed small $F$.
* **How to use it**: Measure sign capacity at three or more sizes above 8 sites before claiming a cost gain from thinned hidden rows.
* **Sources**: P2.1 R11.1, R11.2, R12.3, R13.2, R13.7; jobs 704752, 706240, 707185.

### L29. In linear-method or natural-gradient steps for over-parameterised states, truncate the parameter metric at a relative cutoff instead of shifting it, because the gain sits in near-redundant parameter combinations.
* **Status**: PROVISIONAL
* **Applies to**: linear-method (Toulouse–Umrigar), block linear method and natural-gradient (SR) steps for NQS with many near-redundant parameters.
* **The lesson**: In an over-parameterised state many parameter combinations change the wave function almost not at all, so the overlap (metric) matrix $S$ has a long tail of tiny eigenvalues. The function-space gain the linear method finds sits partly along these directions, and mapping it back to parameters through an unregularised $S^{-1}$ gives huge steps; a level shift large enough to tame them also suppresses the gain. Dropping eigen-directions below a relative cutoff (about $10^{-4}$ to $10^{-5}$ of the largest) keeps the gain and gives stable steps.
* **Evidence**: P4.1 R18 (block linear method, unshifted): step norm 17.7, every step raised the energy by hundreds of gaps ($R=-244$). P4.1 R18′ (level shift): $R=10^{-4}$ with the shift at its ceiling ($\lambda\approx10^6$) and the step still 5.7. P4.1 R18″ (truncated metric, exact sums, live): $R=0.273$ after one step at cutoff $10^{-4}$, and 4 steps at cutoff $10^{-5}$ took the error $\epsilon$ from $2.86$ to $1.32\times10^{-3}$ with no step raising the energy. Here $R$ is the fraction of the predicted tangent-space gain that a step realises.
* **Scope and limits**: Exact sums on one 12-site exact-spin family; the sampled version (rung 1b) has not run.
* **Counter-evidence and open questions**: Under sampling the truncation threshold may need to grow with noise.
* **How to use it**: When a linear-method step blows up, truncate the metric before reaching for a shift.
* **Sources**: P4.1 R15d, R18, R18′, R18″.

### L30. Before speeding up or transferring a size-independent family, measure its capacity plateau and effective tangent rank at the target; if the rank does not grow with width, the family cannot reach the target accuracy.
* **Status**: PROVISIONAL
* **Applies to**: size-transferable NQS families (size-independent kernels or weight-shared generators), warm-start and pretraining schemes.
* **The lesson**: A size-independent family can only be as good as its own best state at the target, and transfer can at best reach that plateau faster. In the families built here, the plateau is a stationary point (one linear-method step improves it by almost nothing), and widening the network only re-parameterises a few dozen lookup tables of local hole counts: the effective tangent rank barely grows. Enlargement is then futile, and the transfer question is moot.
* **Evidence**: P4.2 R5.1 and R5.3 (jobs 452196, 604157, 659969; $4\times4$ open $t$–$t'$–$J$, $6+6$, 600 projected-SR steps): plateau relative energy error 5.97% at 426 parameters and 5.71% at 818 (apparent exponent $\alpha\approx0.07$ in $\epsilon\propto P^{-\alpha}$), effective tangent rank 34 of 426 and 48 of 818; the site-indexed projected NNBF reaches 0.56% with 1734 parameters on the same instance. At the plateau the best state in the span of the wave function and its derivatives improves the energy by at most 0.17% of the remaining error. P4.2 R9.4 and P4.1 R7 inherit this verdict.
* **Scope and limits**: One $4\times4$ open target; periodic targets (P4.2 R9.5, shelved) were never measured, and the optimiser falsifier 704844 was INCONCLUSIVE.
* **Counter-evidence and open questions**: A family whose tangent rank grows with width would escape it.
* **How to use it**: Run a one-rung capacity pilot with a tangent-rank count before investing in transfer or speed-ups; stop if the rank does not grow.
* **Sources**: P4.2 R5.1, R5.3, R9.4; P4.1 R7; jobs 452196, 604157, 659969, 704844.

## About testing and measurement

### L31. Swapping exact and approximate parts of jointly optimised factors is not an error budget, because each part is co-adapted to the other's errors.
* **Status**: PROVISIONAL
* **Applies to**: error attribution between factors of any jointly trained model (sign and amplitude of a wave function, orbitals and Jastrow, encoder and decoder), including Shapley-type splits.
* **The lesson**: Replacing one trained part by its exact counterpart, keeping the other trained part, does not remove "that part's error": the kept part was trained to compensate the replaced part's errors and is wrong for the exact partner. The resulting shares can be negative or exceed the total error, so they measure incompatibility between the trained halves, not error content. A budget must instead fix one part (for example the signs) and re-optimise the other to its best.
* **Evidence**: P1.1 R2 (job 329057, seed 1234; three $4\times3$ Hubbard tori, NNBF widths 8–128): on all 15 states both hybrids lie above the trained energy. On $5+5$, $U=8t$, width 128: total error $\Delta E_{\mathrm{tot}}=0.166t$, exact-sign hybrid error $0.352t$, exact-amplitude hybrid error $1.324t$, so the Shapley shares are sign $+0.569t$ and amplitude $-0.403t$; the amplitude share is negative for all 12 foil and NNBF states ($-0.075t$ to $-0.977t$).
* **Scope and limits**: One budget question on three small tori; the co-adaptation argument is general.
* **Counter-evidence and open questions**: For independently obtained parts (not jointly trained) swaps can be meaningful; only the 3 UHF states gave two positive shares.
* **How to use it**: Attribute error by constrained re-optimisation (fix one part, optimise the other), as P1.1's fixed-sign budget $E_{\mathrm{sign}}$ does.
* **Sources**: P1.1 R1, R2; job 329057; Westerhout et al., Nat. Commun. 11, 1593 (2020).

### L32. Only the best energy at fixed signs is a sign-only error measure; configuration-weighted sign accuracy and lattice fixed-node energies mix in amplitude effects.
* **Status**: PROVISIONAL
* **Applies to**: measuring how much of a fermionic variational state's error is due to its sign structure, on lattices.
* **The lesson**: The sign-mismatch weight $w_{\mathrm{sign}}$ (probability weight of configurations with the wrong sign) is not an energy: a sign error between configurations joined by small matrix elements costs almost nothing, and for any fixed per-region accuracy the chance of a fully right sign pattern decays as $e^{-cV}$, so $w_{\mathrm{sign}}$ saturates with size. The lattice fixed-node energy $E_{\mathrm{FN}}$ is not sign-only either: its sign-flip term $\sum_{\mathbf n'}H_{\mathbf n\mathbf n'}\psi(\mathbf n')/\psi(\mathbf n)$ over sign-violating pairs depends on the guide's amplitudes, so $E_{\mathrm{FN}}$ is only an upper bound on $E_{\mathrm{sign}}$, the lowest energy of any state with the given signs. The stoquastic lower bound on $E_{\mathrm{sign}}$ is too loose to certify a large sign fraction.
* **Evidence**: P1.1 R3 (job 329057, width 128): $w_{\mathrm{sign}}=0.197$, 0.029, 0.0063 on the three tori (31-fold range) while the sign fraction $f_E^{\mathrm{up}}$ is 0.77, 0.61, 0.65; widening from 8 to 128 cut $w_{\mathrm{sign}}$ 3.3 times but $\Delta E_{\mathrm{sign}}$ only 1.6 times. P1.1 R4: fixed-node sign fractions 0.665, 0.827, 0.756 against $f_E^{\mathrm{up}}$ 0.607, 0.772, 0.652, and 0.845 against 0.141 on a 6-site smoke. P1.1 R1: the stoquastic lower bound on $E_{\mathrm{sign}}$ lay $2.7$–$4.1t$ below $E_0$ on every instance.
* **Scope and limits**: Three $4\times3$ tori, unprojected networks.
* **Counter-evidence and open questions**: A non-vacuous lower bound on $E_{\mathrm{sign}}$ is an open row of P1.1's reopening ledger.
* **How to use it**: Report sign errors as $\Delta E_{\mathrm{sign}}$ (fixed-sign optimum) with its upper and lower brackets; never compare $w_{\mathrm{sign}}$ or $E_{\mathrm{FN}}$ across leaves as a sign fraction.
* **Sources**: P1.1 R1, R3, R4; job 329057; ten Haaf et al., PRB 51, 13039 (1995); Loehr and Clark, arXiv:2510.26906.

### L33. Every comparison between separately trained ansätze is also a comparison of how far each optimiser got, so rankings need polished arms or a check of the remaining optimisation gap.
* **Status**: PROVISIONAL
* **Applies to**: matched-parameter ablations, class-enlargement tests, basin-hopping acceptance and any ranking of variational ansätze trained by SR or similar first-order methods.
* **The lesson**: SR at a fixed step count leaves each arm somewhere above its own best state, and how far above differs between architectures, start points and optimiser settings. A larger class can then end worse than its subset, a ranking can flip once both arms are polished, and a short relaxation cannot rank candidates whose differences are below the amplitude error the relaxation leaves. One exact tangent-space (linear-method) step, or polishing both arms to stationarity, measures the remaining gap.
* **Evidence**: P4.1 Round 11 (R18″): four damped linear-method steps removed 0.539 of the 12-site SR-trained `che` state's remaining gap, and polished `che` matched unpolished `pcg` ($\rho=0.97$), so R2's earlier 12-site win ($\rho=0.449$) and its 24-site test were optimiser-confounded and cancelled. P4.1 R4.2: a charge-conditioned sum of two singlet AGPs gained a tabular completeness premise but trained only to a ratio 0.88 over its $K=1$ foil. P4.1 R3iii: rational hidden-fermion head $\rho=1.57$ at equal iterations. P4.2 R9.1 and R5.5: a strict superset (`fsym`, 3.62%) ended worse than its base (1.91%) after 300 SR steps from the same start, one seed. P1.2 R8.4: a 4×3 win disappeared once the optimiser shift was matched ($r$ 0.895 to 0.956). P1.3 R2: after short relaxations the amplitude error ($0.1t$ to $1t$) exceeds the in-sector gap ($0.273t$) the acceptance must rank.
* **Scope and limits**: 10- to 16-site clusters; mostly single-seed ablations.
* **Counter-evidence and open questions**: Whether sampled truncated-metric polishing (L29) is cheap enough to make polished comparisons routine is open.
* **How to use it**: Before reading a ranking, polish both arms or run one exact linear-method step on each and report the remaining gap; treat a single-seed SR ablation as an upper bound on capacity only.
* **Sources**: P4.1 R3iii, R4.2, R8, R18″; P4.2 R5.5, R9.1; P1.2 R8.4; P1.3 R2; cross-node critic Major 3 (master Revision Log).

### L34. In a cost-reduction study, accuracy ratios at matched parameters after a short fixed number of steps measure learning speed, so they may kill a route but never promote it.
* **Status**: PROVISIONAL
* **Applies to**: accuracy gates in studies whose claim is a cheaper ansatz or estimator at equal accuracy.
* **The lesson**: After a short fixed budget of optimisation steps, an arm that learns faster looks more accurate, even if it converges to a worse state or costs more per step. Such a ratio can show that a route is bad (if it loses even on learning speed), but a win says nothing about converged accuracy or cost at equal accuracy.
* **Evidence**: P2.1 R6.1″ (job 699528, $5\times3$ open $t$–$t'$–$J$, 75 SR steps): the route's sign error 0.0222 against 0.0376 for NNBF matched in operations ($\rho_N=0.28$, PROMOTE by rule), but NNBF with hidden width 48 was 3.4 times faster per step and 2.2 times lower in sign error, and NNBF's error fell from 0.088 at 75 steps to 0.0159 at 1200. P2.1 R7.3 (job 699305): PROMOTE at 75 steps ($\rho_N=0.594$) with the route 8 times slower per step. P2.1 R6.1′: a margin kill at 75 steps ($\rho_F=0.868$, single seed).
* **Scope and limits**: One $5\times3$ instance, single seeds, 75-step budgets.
* **Counter-evidence and open questions**: None.
* **How to use it**: Promote a cost route only on converged or equal-wall-time comparisons; let short-horizon ratios kill only.
* **Sources**: P2.1 R6.1′, R6.1″, R7.3; jobs 698229, 699305, 699528; former master cross-leaf anti-pattern.

### L35. An accuracy gate between arms with different cost per step or per sample must add an equal-wall-time arm, or state that it scores parameter efficiency only.
* **Status**: ESTABLISHED (cross-node critic, 2026-11-16, Major 3)
* **Applies to**: matched-parameter comparisons of variational ansätze, estimators or optimisers whose per-step or per-sample cost differs.
* **The lesson**: Matching parameter counts equalises model size, not compute. If one arm costs several times more per step, a matched-parameter, matched-step win is a statement about parameter efficiency; the practical question (more accuracy for the same compute) needs an arm that spends the cheaper method's savings on more steps, samples or width.
* **Evidence**: P2.1 routes cost up to 8 times NNBF per step (R7.3: 30.4 s against 3.7 s), and in R6.1″ the equal-wall-time comparison reversed a matched-parameter PROMOTE. P4.1 `pcg` costs 2.6 times `che` per step at 24 sites, so its T2b was given equal-wall-time `che` arms (job 703587) before it was cancelled for the optimiser confound (L33).
* **Scope and limits**: A methodological rule; it holds for every comparison in this run where costs differed.
* **Counter-evidence and open questions**: None.
* **How to use it**: Add an equal-wall-time arm whenever per-step cost differs by more than the claimed accuracy gain, or scope the claim to parameter efficiency.
* **Sources**: Master Revision Log, cross-node critic of 2026-11-16 (Major 3); P2.1 R6.1″, R7.3; P4.1 R2-T2b.

### L36. When the parameter count reaches about half the projected sector dimension, width comparisons measure sector fitting and carry no capacity or decision-rule information beyond exact-diagonalization size.
* **Status**: ESTABLISHED (cross-node critic, 2026-11-16, Major 2)
* **Applies to**: width and capacity comparisons of NQS on exact-diagonalization clusters, especially with symmetry projection.
* **The lesson**: A symmetry-projected network on a small cluster only needs to represent a vector in the projected sector, of dimension $d_{S\chi}$. When the parameter count $P$ is a sizable fraction of $d_{S\chi}$, any wide enough network fits that vector nearly exactly, so "width wins" or "the energy rule decides" says nothing about how the same networks behave on clusters whose sector dimension is astronomically larger.
* **Evidence**: P2.1's wide NNBF control `nnbf_w` had $P/d_{S\chi}=0.70$; P3.2's "easy" width-8 branch used 1160 parameters against $d=1618$ ($P/d\approx0.72$), where the naive energy rule decided 6 of 6; the P1.1-P tori sit in the same regime.
* **Scope and limits**: The threshold "about half" is a working rule from three cases, not a derived bound.
* **Counter-evidence and open questions**: None; a smaller safe ratio may be needed for highly structured sectors.
* **How to use it**: Report $P/d_{S\chi}$ beside every width comparison and treat results above about 0.5 as sector-fitting artefacts.
* **Sources**: Master Revision Log, cross-node critic of 2026-11-16 (Major 2); P2.1 (control `nnbf_w`); P3.2 width-8 branch; P1.1-P.

### L37. An exact-inverse or exact-matrix reference certifies the machinery, not the information source; a premise run needs a matched Hamiltonian-only foil before its information source is credited.
* **Status**: ESTABLISHED (cross-node critic, 2026-11-16, Major 5)
* **Applies to**: premise tests of correction methods (response Ritz, Lanczos or Davidson corrections, observable-sourced vectors) run with exact linear algebra.
* **The lesson**: With exact matrices, Hamiltonian-derived corrections alone (a shift-invert of the residual, a symmetric Lanczos step) already fix most errors at exact-diagonalization size. A premise run that uses exact algebra therefore shows that the algebra works, not that the method's extra information (an observable source, a trained partner) adds anything. Only a matched foil that uses the Hamiltonian alone, with the same algebra, can show that.
* **Evidence**: P3.1 T3 critic: an observable-free shift-invert of the $\hat H$ residual tracked 14 of 14 items in exact arithmetic, so the 8 of 8 exact-response PASS (job 699756) is an inverse-iteration premise. P3.2 R6.4: a symmetric block-Lanczos step with exact matrices decided 9 of 9 width-4 instances in-sample (job 619080), while sampled at $2\times10^4$ draws per branch it abstained almost everywhere (R6.5).
* **Scope and limits**: Exact-diagonalization sizes, P3.1 and P3.2.
* **Counter-evidence and open questions**: None.
* **How to use it**: Pair every exact-algebra premise run with a Davidson or Lanczos foil built from $\hat H$ only, and credit only the margin over it.
* **Sources**: Master Revision Log, cross-node critic of 2026-11-16 (Major 5); P3.1 R10.1d; P3.2 R6.4, R6.5; jobs 619080, 699756.

### L38. A mechanism gets credit for a gain only after an ablation that removes the mechanism, or a matched-compute capacity foil, fails to reproduce the gain.
* **Status**: PROVISIONAL
* **Applies to**: any architecture or correction method whose claimed mechanism (sign flips, an information source, an improved state) could be replaced by plain extra capacity.
* **The lesson**: An added layer or correction can win a matched-parameter test while its claimed mechanism is unused: the gain comes from extra amplitude capacity or extra optimisation. Removing the mechanism from the trained model (for a sign layer, switching every flip off) or giving a plain wider model the same compute tells which it is.
* **Evidence**: P1.2 R5.1 (job 451916, projected $4\times4$ open $t$–$t'$–$J$, 3 seeds): the vote-factor layer (1719 parameters) beat NNBF width 6 (1734) with sign-error ratio $\rho_{\mathrm{sign}}=0.84$, but removing every flip from the trained state moved $\Delta E_{\mathrm{sign}}$ from 0.009751 to 0.009757 (job 697584), so the 16% gain was amplitude-borne. P3.1 R10.1d (T3, job 705951, $4\times3$ $4+4$, $n=11$ items): trained self-consistent responses tracked 9 items, but a twice-wider network trained with the same CPU tracked 7 of 10, so the T3 is AMBIGUOUS and the correction cannot be read as an error bar.
* **Scope and limits**: Two cases at exact-diagonalization size.
* **Counter-evidence and open questions**: None.
* **How to use it**: Pre-register a mechanism-off read-out and a matched-compute capacity foil for every T2 or T3.
* **Sources**: P1.2 R5.1; P3.1 R10.1d; jobs 451916, 697584, 705951.

### L39. Exact-sum clusters cannot test claims whose value appears only at large size, such as a cheaper projection or a local-versus-global split, because there projection is free and every patch covers most of the lattice.
* **Status**: PROVISIONAL
* **Applies to**: T1 tests on clusters small enough for exact sums (up to about 16 sites) of routes whose claim is a cost advantage or a locality structure.
* **The lesson**: In exact-sum tests every amplitude of the sector is computed anyway, so symmetry projection costs nothing extra, and a route whose only advantage is cheaper or partial projection cannot win or lose on the measurable metric. On a $4\times4$ cluster a range-1 patch with diagonal bonds, or two stacked plus-shaped layers, already sees nearly every site, so "local" and "global" features coincide and a split between them cannot be measured.
* **Evidence**: P1.3 R3.4 and R7.4: projector annealing and projection escalation are dominated by full projection on accuracy, and their cost claim is unmeasurable with exact sums. P2.1 R2.6, R3.4, R6.4, R7.2: on the exact-sum clusters ($V\le16$) all sites lie in the near field; two plus-shaped layers reach 11 of the 15 sites of the $5\times3$ instance from its centre; Tree-HFDS has no coarse level on $5\times3$.
* **Scope and limits**: Exact-sum sizes; the routes themselves are not refuted at large size, only untestable here.
* **Counter-evidence and open questions**: Sampled tests at 30 to 100 sites could test them, at a cost the gate did not fund.
* **How to use it**: Before a T1, ask whether the route's advantage exists at the T1 size; if not, the route needs a sampled test at a size where it does, or it waits.
* **Sources**: P1.3 R3.4, R7.4; P2.1 R2.6, R3.4, R6.4, R7.2.

### L40. A toy instance may decide a rule only if it is representative: it has a spectral gap, its sector is not nearly spanned by the method's subspace, and its noise and infidelity match the real states.
* **Status**: PROVISIONAL
* **Applies to**: choosing toy instances and noise models for T1 tests of NQS methods and error estimators.
* **The lesson**: Three toy defects decided verdicts by accident in this run. On a near-degenerate toy, the wrong-sign labels follow the partner state and one seed decides. On a toy whose sector has barely more levels than the method's subspace has vectors, every arm passes. And an isotropic noise model of trained vectors can rank information sources in the wrong order, because real training errors are not isotropic.
* **Evidence**: P1.2 R7.3: on a $5\times2$ $4+4$ toy with $E_1-E_0=0.0041t$ the verdict hinged on one seed; every later T1 asserts $E_1-E_0\ge0.02t$ on counted sizes. P3.1 R9.3: at $3\times3$ the 50-vector span nearly fills the 60-level sector, so the method tracked 0.96 there and 0.43 at $6\times2$. P3.1 R10.1d T3 (job 705951): trained observable-sourced responses reached median in-window error 0.08 against 0.90 for trained Davidson vectors, the reverse of the isotropic-noise toy prediction.
* **Scope and limits**: Three cases at 10 to 12 sites.
* **Counter-evidence and open questions**: None.
* **How to use it**: Assert a gap on every counted size, compare the subspace dimension with the sector dimension, and calibrate toy noise to the real states' squared overlap and per-site infidelity.
* **Sources**: P1.2 R7.3; P3.1 R9.3, R10.1d; job 705951; former master cross-leaf anti-pattern "A counted toy size must have a gap".

### L41. Ensembles of seeds, widths or fields of one architecture share their bias, so their disagreement, or an extrapolation through them, under-states the error.
* **Status**: PROVISIONAL
* **Applies to**: error bars from seed spread, architecture consensus, capacity ladders, multi-seed Ritz and averaging of independently trained corrections.
* **The lesson**: Independently trained copies of one architecture make similar errors, because the error comes from what the family cannot represent, not from random optimisation noise. Their spread measures the noise part only, and an extrapolation along width keeps the shared part: widening removes amplitude error while the sign-borne observable error barely moves.
* **Evidence**: P3.1 R6.4, R7.7, R9.4 (job 698163): the real $4\times4$ seeds 11, 22, 33 have the same sign of the observable error in all three channels. P3.1 R2.3 (job 329057): width lowered the energy variance two- to threefold while the sign-borne largest bond-pair error moved only from 0.0273 (width 8) to 0.0255 (width 128); a cross-width extrapolation under-stated the error on 3 of 3 widest states. P3.1 R7.6: a Ritz-trained auxiliary is a two-member capacity ladder.
* **Scope and limits**: One $4\times4$ instance (3 seeds) and the $4\times3$ $4+4$ torus.
* **Counter-evidence and open questions**: Ensembles of different architectures with independent biases could do better; not tested.
* **How to use it**: Never report seed spread as an error bar for a systematic error; check whether ensemble members share the sign of their errors.
* **Sources**: P3.1 R2.3, R6.4, R7.6, R7.7, R9.4; jobs 329057, 698163; Viteritti et al., arXiv:2604.21978.

### L42. An extrapolation through separately sampled states fails when the signal it extrapolates is smaller than each state's statistical error.
* **Status**: PROVISIONAL
* **Applies to**: capacity-ladder, hybrid and zero-variance extrapolations of observables through several sampled variational states.
* **The lesson**: Each state on the ladder carries its own Monte Carlo error. If the differences between them (the signal the fit uses) are below those errors, the fit's slope and endpoint are noise, and a fit through one ensemble anchors neither.
* **Evidence**: P3.1 R5.2 ($4\times4$ open instance, $10^5$ draws): 0 of 7 injected items tracked; spin over-shot 2.6–2.8 times. P3.1 R6.1 (job 452272): 0 of 4 tracked; the jackknife standard error was about the size of the error itself, so the method abstained almost everywhere.
* **Scope and limits**: One instance, injected checks.
* **Counter-evidence and open questions**: None.
* **How to use it**: Compare the ladder's signal with each state's standard error before fitting; if it is smaller, sample more or drop the extrapolation.
* **Sources**: P3.1 R5.2, R6.1; jobs 452146, 452272.

### L43. A warm-started or transferred arm must be scored against its own starting state and against training from scratch, not only against the foil.
* **Status**: PROVISIONAL
* **Applies to**: warm-start, oracle-start and transfer tests of NQS (across sizes, Hamiltonians or architectures).
* **The lesson**: A warm-start arm tests a basin only if its start is measurably better than the cold runs; a transferred state can start worse than doing nothing, because an exactly fitted but under-identified source network carries arbitrary content in its null directions; and a head start that training uses up in a few dozen steps is worth a speed-up of about 1 on an offset-free clock. Each of these was invisible when the arm was compared with the foil alone.
* **Evidence**: P1.3 R3.1 and the cross-node critic (Minor 8): the oracle start was worse than the best cold start at every width (0.0213, 0.0195, 0.0138 against 0.0194, 0.0117, 0.0075), so no better-signed state was ever placed. P4.2 R3.1 ($4\times4$ torus, 4 holes, 3 seeds): zero-shot transferred error 0.663, 0.759, 0.615 per site against 0.318, 0.130, 0.285 for the untrained mean-field start; still behind scratch after 20 and 50 steps (0.0335 against 0.0240; 0.0248 against 0.0183). P4.2 R3.0: three seeds fit the source to $10^{-9}$ or better yet sit 1.05 to 1.75 apart in parameters, with 107 of 122 metric directions below $10^{-8}$. P4.2 R5.b: resetting the null-direction content destroyed the source fit. P4.2 R9.6: the target network passes the transferred level within its first tens of steps.
* **Scope and limits**: Exact-diagonalization sizes; one transfer family.
* **Counter-evidence and open questions**: None.
* **How to use it**: Score every warm or transferred start at step 0 against the untrained prior and the cold start, and report the offset-free speed-up.
* **Sources**: P1.3 R3.1; P4.2 R3.0, R3.1, R3.2, R5.a, R5.b, R9.6; jobs 410960, 451914, 507179, 625182.

### L44. A test bed for competing orders must put both competitors in the same symmetry sector, which needs a two-parameter scan because same-sector degeneracies have codimension 2.
* **Status**: PROVISIONAL
* **Applies to**: designing small-cluster tests of methods that decide between competing orders.
* **The lesson**: Competitors in different symmetry sectors can be separated by projection or level spectroscopy, which makes the decision trivial and the test uninformative. The physically relevant competitors (stripes against $d$-wave) share a sector on clusters commensurate with both. By the von Neumann–Wigner rule, two same-sector levels generically meet only on a codimension-2 set, so a one-parameter sweep finds avoided crossings, and an exact conical intersection needs a second parameter (here hopping anisotropy $t_y$ added to $t'$).
* **Evidence**: P3.2 R1.2: on the $4\times3$ cluster the earlier "same-sector" competitor was a spin triplet at momentum $(\pi,\pm2\pi/3)$, $0.086t$ above $E_0$, while the in-sector gap was $0.515t$. P3.2 R1.4: its test's competitors were mirror-even and mirror-odd. P3.2 R2.5: no twist separates same-sector orders. P1.3 R2.3: an irrep scan meets same-sector competitors. P3.2 then built its test bed as a same-irrep crossing in $(t',t_y)$.
* **Scope and limits**: Small Hubbard clusters.
* **Counter-evidence and open questions**: None.
* **How to use it**: Run an exact sector analysis of every proposed test instance and place targets on a two-parameter crossing.
* **Sources**: P3.2 R1.2, R1.4, R2.5; P1.3 R2.3; master Revision Log item 23 (round-1 critic).

### L45. A decision rule that may abstain must be scored on its wrong-commit rate and its decision rate together.
* **Status**: PROVISIONAL
* **Applies to**: decision rules with an abstention option (certificates, rules with a noise margin) in method comparisons.
* **The lesson**: A rule that abstains whenever it is unsure never commits wrong, so a low wrong-commit rate alone is uninformative; a rule is useful only if it decides often while rarely being wrong.
* **Evidence**: P3.2 R5.1: the Eckart-certified decision made 0 certified commits on the inner targets even with exact inputs. P3.2 R6.5: sampled one-step Lanczos was right on 1.3% and wrong on 0.6% of decisions (LOSES by abstention), although exact Lanczos fixed 4 of 6 wrong calls. P3.2 R3.5: projected controls abstained on all 15 hard pairs while the continuation rule committed on all and was wrong on 5.
* **Scope and limits**: One $4\times3$ torus crossing.
* **Counter-evidence and open questions**: None.
* **How to use it**: Pre-register both rates and a minimum decision rate in every rule.
* **Sources**: P3.2 R3.5, R5.1, R6.5.

### L46. Fluctuation along the optimisation trajectory can overstate the noise of a final-state energy difference about a hundredfold; estimate decision noise from final-state sampling or seed replicas.
* **Status**: PROVISIONAL
* **Applies to**: noise margins of decision rules and error bars computed from training traces (SR windows).
* **The lesson**: The energy along an SR trajectory still drifts and fluctuates from parameter updates, so its window variance measures optimisation motion, not the sampling noise of the final state's energy. Margins built from it are far too wide (or, used the other way, misleading).
* **Evidence**: P3.2 R5.3 (round-5 per-leaf critic, job 451852 data): the SR-window fluctuation over-stated the final-state noise of the energy difference by about 100 times; the margin also rested on a zero-variance estimate off by up to $+2.2\times10^{-3}t$ at width 4.
* **Scope and limits**: One crossing study, widths 4 and 8.
* **Counter-evidence and open questions**: None.
* **How to use it**: Compute decision noise by sampling the final states (or seed replicas), never from the training window.
* **Sources**: P3.2 R5.3; job 451852.

### L47. Cost gates must be predicted from per-sample operation counts of the real algorithm and from measured speeds of compiled kernels, not from multiply-add totals, stored full-basis vectors or interpreter timings.
* **Status**: PROVISIONAL
* **Applies to**: T0 and T1 cost gates of NQS routes and estimators.
* **The lesson**: Three cost errors appeared. Timing a route on a stored full-basis vector hides that it needs exact neighbour amplitudes, $(1+E_{\mathrm{act}})$ amplitude evaluations per amplitude (with $E_{\mathrm{act}}$ the number of active edges), an $\mathcal O(V)$ factor. A multiply-add saving that turns matrix–matrix work into matrix–vector work or scalar loops loses about a third of its advantage on the clock. And interpreter-bound timings measure the interpreter: a Python Pfaffian made a cost ratio look about eight times worse than compiled code.
* **Evidence**: P1.2 R6.7 and R5.1: 522 against 32 network passes per local energy at $L=4$ for the vote route against NNBF. P2.1 R13.1: operation-count ratio 5.4 against measured sweep-time ratio 3.54 at $L=16$ (3.89 at $L=24$); P2.1 R13.3: the gathered active-row loops of `half` were slower than the dense engine (0.056 s against 0.035 s per sweep). P4.1 R2: cost ratio 30.5 with a Python Pfaffian, 3.59 compiled. P2.1 Round 11: a cost lemma derived from one implementation's carried tables described that implementation, not the minimal algorithm.
* **Scope and limits**: Sizes up to $L=24$ for timing; CPU kernels.
* **Counter-evidence and open questions**: None.
* **How to use it**: Count operations of the per-sample algorithm by kernel class (GEMM, GEMV, LU, scalar loops) and convert with measured kernel speeds; time only compiled code; list the tables the algorithm strictly needs before closing on a bottleneck.
* **Sources**: P1.2 R5.1, R6.7; P2.1 R13.1, R13.1′, R13.3 and its Round 11 diagnostic; P4.1 R2 cost gate; former master anti-pattern "Interpreter-bound timings are not method costs".

## About process and compute

### L48. Almost half of the compute spent before the compute gate changed no decision, and most of it could have been predicted by a scaling argument that cost no compute.
* **Status**: PROVISIONAL
* **Applies to**: compute planning in research brainstorms and method studies on clusters.
* **The lesson**: Before the compute gate existed (2026-11-10 to 2026-11-16), jobs were submitted when a route reached a decisive-test design, without asking whether a cheaper argument could settle it. The largest no-change spends were decisive arrays on routes that a Gate 4 scaling argument later killed with no compute, size ladders whose outcome the same argument predicted, and pilots repeated until a tie-break rule fired.
* **Evidence**: Master compute ledger (`sacct`, 2026-11-16): 66 jobs, 28,162 CPU-h; 21 jobs worth 13,388 CPU-h (48%) changed no decision. P2.1's round-5 decisive array 526080 (85 tasks, 8,906 CPU-h, 32% of the total) ended AMBIGUOUS, and its routes were then killed at T0 by a Gate 4 argument (P2.1 R3.1, R3.2). P4.1's size ladder 686990 (about 13,000 CPU-h of work, cap 107,520) was cancelled before it started because the T0 argument already predicts its outcome (P4.1 R1-L); the generality job 637776 spent 1,424 CPU-h before cancellation (R1-G). P3.1's ESR0 extension 618646 spent 169 CPU-h before ESR0 failed Gate 4 (R7.1a). P1.3's ladder 451976, 507178, 507179 (1,527 CPU-h) returned UNCLEAR three times before the CAPACITY tie-break.
* **Scope and limits**: One brainstorm; the "changed a decision" column was read from the leaf texts.
* **Counter-evidence and open questions**: Some no-change jobs (wall-cap hits) were plain bad luck rather than predictable.
* **How to use it**: Run the T0 scaling and dominance arguments (Gate 4) before any job above about 10 CPU-h, and write the predicted verdict down; if the prediction is confident, record it instead of running the job.
* **Sources**: Master `## Compute Ledger` (historical ledger and observations); P2.1 R3.1, R3.2; P4.1 R1, R1-L, R1-G; P3.1 R7.1, R7.1a; P1.3 R3.1; jobs 526080, 686990, 637776, 618646, 451976, 507178, 507179.

### L49. Caps per array task do not bound what an array job spends; compute caps must be set on the total of a job.
* **Status**: PROVISIONAL
* **Applies to**: compute-budget rules for Slurm array jobs and any batch system with per-task limits.
* **The lesson**: A per-task cap (CPUs times wall limit per task) looks small, but an array multiplies it by the number of tasks, and resumes and extensions multiply it again. Without a cap on the job total, large spends pass every per-task check.
* **Evidence**: Master compute ledger: P2.1's 85-task array 526080 spent 8,906 CPU-h without breaking any per-task cap; eight jobs used more than 1,000 CPU-h each without the user's approval; the cancelled jobs' combined caps were about 285,700 CPU-h, almost all in four P4.1 size-ladder and generality jobs; the queued 24-site reference 687302 still held 3,840 CPU-h of wall limit at the pause.
* **Scope and limits**: One cluster and one brainstorm.
* **Counter-evidence and open questions**: None.
* **How to use it**: Gate every job on its total CPU-hours (tasks times CPUs times wall limit), including resumes, as `compute_gate.py` now does.
* **Sources**: Master `## Compute Ledger`; SKILL.md rule 6; job 526080.

### L50. Before planning on, resuming or extending a job, check from stored data whether its pre-registered rule can still change the verdict.
* **Status**: PROVISIONAL
* **Applies to**: pending, running or resumed jobs whose verdict depends on a pre-registered rule over seeds or items.
* **The lesson**: Once enough seeds or items are stored, a pre-registered rule may already be decided whatever the remaining tasks return. Running them, or branching the plan on their outcome, spends compute and time on a known answer. The same holds for resumes and extensions: if no outcome of the resumed run changes the decision, it should not run.
* **Evidence**: P1.2 R5.1: with all ten control runs stored, the 5-seed sign-error ratio was at least $0.782>0.7$ whatever seeds 44 and 55 gave, yet job 553724 spent 1,090 CPU-h before 3 of its 8 tasks were cancelled. P3.1 R10.1d: the T3 705951 printed INCOMPLETE, but seed 11 could lift the best foil to 8 items at most against a PASS threshold of 6 and a KILL threshold of 9, so the verdict was already AMBIGUOUS.
* **Scope and limits**: Three cases in one brainstorm.
* **Counter-evidence and open questions**: None.
* **How to use it**: Before every resume, extension or plan branch, run the aggregate on stored data with the missing items set to their best and worst values; if both give the same verdict, stop.
* **Sources**: P1.2 R5.1; P3.1 R10.1d; jobs 553724, 705951.

### L51. Removing the reason a route was killed, or rescuing it after the result, is not a new mechanism; a revival needs its own pre-registered test with a cost model measured on one instance at one size.
* **Status**: PROVISIONAL
* **Applies to**: reopening killed routes, and post-hoc combinations proposed after a route fails.
* **The lesson**: A revived route whose only change is to argue away its kill reason tends to pass by mixing measurements from different settings, and a rescue chosen after a result (adding a known trick to a failed route) is a constant-factor combination of known pieces. Both need the same gate as a new route: a mechanism, a pre-registered rule and a cost model measured consistently.
* **Evidence**: P2.2 R2.0: the revived frozen-backflow control variate showed a falling cost ratio (0.90, 0.59, 0.60 at $M=16$, 24, 32), but its cost constant came from $N=20$, $M=80$ while its variances came from $N=4$, a hybrid model; the round-1 kill stood. P2.2 R5.1s: screened WIBJ, proposed after WIBJ failed, was flat in operations (0.21 to 0.10) with accuracy already beyond the 0.2 mHa target (about 0.29 mHa at $K=20$).
* **Scope and limits**: Two routes in one leaf.
* **Counter-evidence and open questions**: None.
* **How to use it**: Treat every revival or rescue as a new route: card it, gate it and pre-register its rule before measuring.
* **Sources**: P2.2 R2.0, R5.1s.
