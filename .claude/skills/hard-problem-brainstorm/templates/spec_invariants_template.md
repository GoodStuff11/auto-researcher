<!-- SPEC_INVARIANTS: Immutable problem contract. Once filled, DO NOT modify during critic rounds or worker runs. -->

1. Mathematical Statement of the Problem
* Input Space: State the input space $\mathcal{X}$, Hamiltonian $\hat{H}$, instance distribution $\mathcal{D}$, or parameter regime.
* Target Output: State the exact object to compute, prove, or optimize (e.g., ground-state expectation value $\langle \hat{O} \rangle$, spectral gap $\Delta$, bounding certificate).
* Success Criterion: State the operational condition for a successful result: 
* $$\text{Target Condition (e.g., } |\langle \hat{O} \rangle_{\text{ansatz}} - \langle \hat{O} \rangle_{\text{exact}}| < \epsilon \text{ with cost } \mathcal{O}(\text{poly}(N))\text{)}$$

2. The Hard Core
* The single irreducible mathematical or physical phenomenon that makes this problem difficult and that no simplification or toy model may remove:
<!-- Example: Fermionic exchange antisymmetry leading to exponential cancellation of path-integral weights. -->

3. The Unacceptable Outcome (Failure Threshold)
* A result meeting any of the following conditions is considered completely worthless:
<!-- Example: Optimizes ground-state energy to high accuracy but yields vanishing overlap on off-diagonal order parameters. -->
<!-- Example: Requires computational runtime or sample complexity that scales exponentially with system volume $V$. -->
<!-- Example: Yields bounds that can be satisfied by a product state or uncorrelated mean-field ansatz. -->
<!-- Example: Regurgitates a known published technique without solving its primary adoption failure. -->

4. Minimal Contrast Foil & Known Baselines
* The Minimal Contrast Foil: The simplest existing trivial baseline or heuristic method:
<!-- Example: Unprojected Hartree-Fock / Single Slater Determinant. -->
* Known Partial Solutions in the Literature: Methods that address parts of this problem:
<!-- Example: Projected Entangled Pair States (PEPS) with bond dimension D=4. -->
* Why Known Solutions Are Not Universally Adopted: The fatal practical or fundamental bottleneck preventing existing methods from fully closing this problem:
<!-- Example: Contraction complexity scales as O(D^10), making gradient backpropagation numerically unstable on systems larger than 8x8. -->

5. Verified No-Go Theorems & Boundaries
* Known fundamental limits that rule out naive solution forms:
    * Theorem / Bound 1: <!-- State exact hypotheses and domain of validity -->
    * Theorem / Bound 2: <!-- State exact hypotheses and domain of validity -->
