# P4.1 r21: pre-registered rules

Copied verbatim from the Round 20 route cards (/home/jek354/.claude/jobs/91298e5c/tmp/P4.1_round20.md) on 2026-11-17, before any r21 computation. Run order: R20 step 1, R22 falsifier 1, R19 one-step probe. Batch budget: about 10 CPU-minutes, interactive.

## R20 `abl-rank`

* **The idea**: `pcg` and `che` differ in two ways at once: the frame (general pairing against a product of up and down orbitals) and the backflow parametrisation ($B_1f_0B_2^{\mathsf T}$ against `che`'s orbital backflow, at $H=15$ against 16). So $\rho$ cannot attribute any gain to pairing, even with perfect polishing (library L38: credit needs an ablation that removes the mechanism). Inside `pcg`'s own code, replace $f_0$ by its best rank-5 factorisation $f_0\approx U_0D_0^{\mathsf T}$ ($U_0,D_0\in\mathbb R^{12\times5}$). Then $\det f_c[U,D]=\det(B_1U_0)[U]\det(B_2D_0)[D]$: a charge-backflowed UHF in exactly `pcg`'s parametrisation, with the same backflow, Jastrow, projector and quadrature. It has 120 pairing parameters against 144, so the counts are matched to within 1% overall. Starting from `pcg`'s polished point puts the ablated arm in `pcg`'s basin, which favours the ablated arm and makes the test conservative for pairing.

* **Cheapest falsifier, step 1 (T1, under 1 CPU-minute)**: truncate $f_0$ at `pcg`'s r19 N8 parameters by SVD to rank 5, and evaluate the exact energy with no re-optimisation.
  - Pre-registered rule: if $\epsilon_{\mathrm{trunc}}\le1.10\,\epsilon_{\mathrm{pcg}}$, the trained `pcg` does not use pairing beyond UHF. R2's pairing mechanism is then KILLED at T1, and any `pcg` advantage over `che` is reassigned to the backflow form, which is a new route that needs its own pre-registered test (L51). If $\epsilon_{\mathrm{trunc}}>1.10\,\epsilon_{\mathrm{pcg}}$, the test goes to step 2. A large rise by itself proves nothing, because the parameters were not relaxed.

## R22 `spin-tree`

  - **Lemma (T0, Wigner–Eckart)**: the hopping $\sum_\sigma c^\dagger_{j\sigma}c_{i\sigma}$ is a rank-0 tensor supported in the subtree of the lowest common ancestor (LCA) of $i$ and $j$. It therefore leaves every node at or above the LCA unchanged, and changes each node on the two paths $i\to$ LCA and $j\to$ LCA by $\pm\tfrac12$ or 0. The number of connected states is at most $2^{d_i+d_j}$, where $d$ is the depth below the LCA.
  - For a balanced 2D tree, about $2^{h/2}$ bonds cross the split of each of the $L/2^h$ nodes of height $h$, each costing at most $2^{2h}$. That gives $\sum_hL\,2^{1.5h}=\mathcal O(L^{2.5})$ connected states per configuration in the worst case. This is polynomial, against $e^{\mathcal O(W)}$ for sequential coupling on a width-$W$ strip. Matrix elements are products of $6j$ symbols along the paths.

* **Cheapest falsifier 1 (T0/T1, seconds to minutes, no Hilbert space)**: count connected states for nearest- and next-nearest-neighbour hops, averaged over random valid $(c,T)$ at doping $1/8$, on balanced k-d trees for square clusters of $L=16$, 64, 256 and 1024 sites, with triangle rules enforced.
  - Pre-registered rule: KILL at G4 if the fitted exponent of the mean count against $L$ exceeds 3. PASS if it is at most 2.6. Otherwise UNCLEAR, and the worst-case count at $L=1024$ decides.

## R19 `ladder`

* **Cheapest falsifier (T1, about 6–8 CPU-minutes, interactive)**: one exact LM step from each N8 endpoint at cutoffs $\{10^{-5}\ (\text{control}),10^{-6},10^{-7},10^{-8}\}$, with $\lvert z\rvert\le0.05$ and a line search $s\in\{1,\tfrac12,\tfrac14\}$ on the exact energy. Also record the fraction of the energy gradient outside the kept subspace (the P4.2 check). At r19's cost (about 3.6 CPU-minutes per `pcg` step and 1.3 per `che` step) one metric build per arm is reused for all four cutoffs.
  - Pre-registered rule, per arm: FROZEN-ARTEFACT if some cutoff below $10^{-5}$ lowers $\epsilon$ by at least 3% in that one step and beats the control step by at least 2 percentage points. GENUINE if no cutoff lowers $\epsilon$ by more than 1%. UNCLEAR otherwise.
  - Next action: if `pcg` is FROZEN-ARTEFACT, run the full ladder polish of both arms to the stopping rule "predicted gain below $10^{-3}$ of the gap", with the r19 thresholds ($\le0.60$ KEEP-LIVE, $\ge0.85$ RETRACT) read on the ladder endpoints. This costs at most 1 CPU-hour and goes as a quick-note job. If both arms are GENUINE, the 12-site energy ratio is final at about $0.76$–$0.79$ (CONFOUNDED), and R2's decision passes to R20. If only `che` is FROZEN-ARTEFACT, ladder `che` alone, because RETRACT is then reachable.
