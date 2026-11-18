# Kill Index

One row per KILLED or SHELVED route, grouped by leaf. **Code** is where the route died (the kill codes of SKILL.md: `G0`–`G4` gates, `T0`–`T3` evidence tiers, `CRITIC`), or SHELVED when the compute gate refused its next step without refuting it; a shelved idea counts as untested. **Card** links to the full route card in the leaf's `## Route Register`; **Library** names the findings-library entry in `library.md` that the kill supports, or route-specific. Built 2026-11-17 from the leaves' Route Registers.

### P1.1 — Sign versus amplitude

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R2 | P1.1 | Route 2: hybrid sign/amplitude swap with a Shapley split | G0 | co-adapted halves: amplitude share negative on all 12 foil and NNBF states ($-0.40t$ against total $0.166t$) | [[leaves/P1.1/cards/R2]] | L31 |
| R3 | P1.1 | Route 3: sign-mismatch weight as the budget | G4 | saturates exponentially with size (a size-scaling argument; recoded from G1 by the per-leaf critic, 2026-11-17) and does not track energy (31-fold spread while $f_E$ stays 0.61–0.77) | [[leaves/P1.1/cards/R3]] | L32 |
| R4 | P1.1 | Route 4: lattice fixed-node GFMC with the network as guide, as a scalable budget | G0 | known (Loehr and Clark); the fixed-node energy depends on the guide's amplitudes | [[leaves/P1.1/cards/R4]] | L32 |

### P1.1-P — Projected sign budget

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R2 | P1.1-P | Arm U→P (and its spin copy U$_S$): projection after variation of the unprojected network | T3 | projecting after training removes only 1–90% of the sign error, against more than 99.999% for projected training | [[leaves/P1.1-P/cards/R2]] | L1 |

### P1.2 — Antisymmetry without a nodal anchor

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | P1.2 | Jordan–Wigner sign times a learned sign network (Route 1) | G0 | published (Inui et al.; Nomura et al.); non-local Jordan–Wigner string and poor sign generalisation unaddressed | [[leaves/P1.2/cards/R1]] | L24 |
| R2 | P1.2 | Pairwise antisymmetric product (Route 2) | G1 | a fixed pair function cannot reproduce free-fermion nodes on $3\times4$ | [[leaves/P1.2/cards/R2]] | route-specific |
| R3 | P1.2 | Slave-fermion holon determinant times the Marshall sign (Route 3) | G2 | the holon determinant times Marshall sign is wrong on about 4.7 times more weight than the foil at $J/t=0.4$ | [[leaves/P1.2/cards/R3]] | route-specific |
| R4 | P1.2 | Residual Walsh sign layer on the foil determinant (Route 4) | G0 | CPS times Slater; local occupations cannot see the reference's nodes; smoke $M_{\mathrm{local}}=1.03$ | [[leaves/P1.2/cards/R4]] | L21 |
| R5 | P1.2 | Neural Pfaffian of a learned pairing matrix (Route 5) | G0 | neural Pfaffians are published; nodes still move only by deforming a pairing matrix | [[leaves/P1.2/cards/R5]] | L24 |
| R2.1 | P1.2 | Product of plaquette sign factors times the foil (Route 2.1) | G0 | CPS times Slater; local factors cannot locate the reference's nodes | [[leaves/P1.2/cards/R2.1]] | L21 |
| R2.3 | P1.2 | Linear first-order bond dressing (Route 2.3) | G0 | a one-step Lanczos dressing; not size-consistent at fixed steps | [[leaves/P1.2/cards/R2.3]] | L10 |
| R2.4 | P1.2 | Brick layer of two-site fermionic gates, the lattice Baeriswyl projector (Route 2.4) | G1 | exponential cost, a new sign problem, or no new expressivity | [[leaves/P1.2/cards/R2.4]] | route-specific |
| R2.6 | P1.2 | Backflow in a learned rotated orbital frame (Route 2.6) | G0 | known; inherits the dense local-energy cost wall of rotated two-body terms | [[leaves/P1.2/cards/R2.6]] | route-specific |
| R3.0 | P1.2 | Local-energy Jastrow, the first-order limit of FRD (Route 3.0) | G2 | a positive factor cannot change any sign | [[leaves/P1.2/cards/R3.0]] | L22 |
| R3.3 | P1.2 | Projected-reference FRD (Route 3.3) | G1 | loses FRD's local separability; projecting the whole state does the same at equal cost | [[leaves/P1.2/cards/R3.3]] | route-specific |
| R5.1 | P1.2 | NNBF×FRD-K, hop-edge vote factors on a neural backflow base (Routes 5.1, ported from R2.2) | T2 | flips inert ($v=0.003$): the 16% gain is amplitude-borne; 5-seed rule unreachable; $\mathcal O(V)$ cost | [[leaves/P1.2/cards/R5.1]] | L38, L50 |
| R5.2 | P1.2 | NNBF×FRD-X, exchange-edge vote factors on a neural backflow base (Routes 5.2, ported from R2.5) | T3 | $\rho_{\mathrm{sign}}=1.29$ against matched width, local-energy tails $\kappa=11.9$ | [[leaves/P1.2/cards/R5.2]] | L21 |
| R5.3 | P1.2 | FRD on a Slater–Jastrow base against an NNBF control (Route 5.3) | CRITIC | confounded design: a FAIL says nothing, a PASS is implausible by construction | [[leaves/P1.2/cards/R5.3]] | route-specific |
| R5.4 | P1.2 | FRD-F, frozen-frame ratios to remove FRD's cost wall (Routes 5.4 and 6.1) | T2 | the read-out loses all of FRD-K's gain; the vote it cheapens is unused | [[leaves/P1.2/cards/R5.4]] | route-specific |
| R6.2 | P1.2 | FRD-H, vote factors on every off-diagonal move (Route 6.2) | T1 | 1.6 times R5.1's $\mathcal O(V)$ cost wall and an equally unused vote | [[leaves/P1.2/cards/R6.2]] | route-specific |
| R6.3 | P1.2 | AO-F, sign-inert ratio Jastrow on frozen ratios (Route 6.3) | T1 | no sign gain over matched width ($r_W$ 2.04, 0.95, 1.19; a pass needed $r\le0.7$) | [[leaves/P1.2/cards/R6.3]] | L22 |
| R6.4 | P1.2 | Complex detour, real part of a complex backflow (Route 6.4) | T0 | the energy is a weighted mean of the two real parts; no coupling | [[leaves/P1.2/cards/R6.4]] | L23 |
| R6.5 | P1.2 | Holon-frame backflow (Route 6.5) | G3 | universal with full backflow, so it reduces to anchor quality, good only at strong exchange | [[leaves/P1.2/cards/R6.5]] | route-specific |
| R6.6 | P1.2 | Node-targeted sampling (Route 6.6) | T0 | SR fixed points do not depend on the sampler | [[leaves/P1.2/cards/R6.6]] | L19 |
| R6.7 | P1.2 | Vote-initialized FRD (Route 6.7) | G4 | $(1+E_{\mathrm{act}})$ amplitude evaluations per amplitude; the raw vote raised $\Delta E_{\mathrm{sign}}$ | [[leaves/P1.2/cards/R6.7]] | L47 |
| R6.7F | P1.2 | Frozen-frame vote on the trained foil (Route 6.7F) | T1 | a frozen vote raises the foil's sign error by 30–75% ($q$ 1.75, 1.31) | [[leaves/P1.2/cards/R6.7F]] | L21 |
| R6.8 | P1.2 | One-shot projected-vote sign read-out (Route 6.8) | G0 | known one-step Lanczos read-out; not size-consistent | [[leaves/P1.2/cards/R6.8]] | route-specific |
| R7.1 | P1.2 | AO-L, sign-inert ratio Jastrow on exact ratios (Route 7.1) | T1 | no sign gain over matched width ($r_W$ 2.04, 0.95, 1.19; a pass needed $r\le0.7$) | [[leaves/P1.2/cards/R7.1]] | L22 |
| R7.2 | P1.2 | DRF-K, determinant-ratio-feature backflow on hop edges (Route 7.2) | T1 | loses to width ($r_W$ 1.52, 2.14) although its untrained premise passed | [[leaves/P1.2/cards/R7.2]] | L21 |
| R7.3 | P1.2 | DRF-X, determinant-ratio-feature backflow on exchange edges (Route 7.3) | T1 | loses to width ($r_W$ 2.35, 1.42); counted toy needed a gap | [[leaves/P1.2/cards/R7.3]] | L40 |
| R7.4 | P1.2 | DRF-H, determinant-ratio-feature backflow on all edges (Route 7.4) | T1 | loses to width ($r_W$ 1.52, 1.91) | [[leaves/P1.2/cards/R7.4]] | L21 |
| R7.5 | P1.2 | Gated mixture of NNBF experts (Route 7.5) | G4 | needs $2^R$ experts for $R$ independent repairs, or a gate that is a CPS | [[leaves/P1.2/cards/R7.5]] | L10 |
| R7.6 | P1.2 | Transcorrelated frame for NNBF (Route 7.6) | G2 | the losing baseline with a non-variational objective | [[leaves/P1.2/cards/R7.6]] | route-specific |
| R8.1 | P1.2 | 6.7L, exact vote on a finite-receptive-field backflow (Route 8.1) | G0 | a cheap form of the R6.8 relabel; starting gain already at threshold and falling with $V$ | [[leaves/P1.2/cards/R8.1]] | route-specific |
| R8.2 | P1.2 | DRF-E, two-pass backflow fed exact neighbour ratios (Route 8.2) | T0 | exact features add only 0.03 AUC over frozen ones, which gave no sign gain | [[leaves/P1.2/cards/R8.2]] | L21 |
| R8.3 | P1.2 | DRF-WS, warm-started second backflow pass (Route 8.3) | T1 | ties width ($r=0.991$ at both sizes); new capacity goes to amplitudes | [[leaves/P1.2/cards/R8.3]] | L21 |
| R8.4 | P1.2 | DRF-T, tied re-entry of ratio features into the first pass (Route 8.4) | T1 | ties once the optimiser shift is matched ($r$ 0.985, 0.956) | [[leaves/P1.2/cards/R8.4]] | L21 |
| R8.5 | P1.2 | Threshold-scanned exact vote (unnumbered in the Round 7 refill) | G0 | known and not size-consistent (R6.8 relabel) | [[leaves/P1.2/cards/R8.5]] | route-specific |
| R8.6 | P1.2 | Matrix-product-state sign factor along a snake path (unnumbered in the Round 7 refill) | G3 | efficient only in quasi-1D, which the Hard Core excludes | [[leaves/P1.2/cards/R8.6]] | route-specific |
| R8.7 | P1.2 | Hash table of learned sign flips (unnumbered in the Round 7 refill) | G4 | not extensive and not polynomial (memorisation) | [[leaves/P1.2/cards/R8.7]] | route-specific |
| R9.0 | P1.2 | LRK, weight-shared low-rank orbital update fed ratio features (closed-leaf falsifier) | T1 | ties width ($r$ 1.05, 1.00) while lowering total energy | [[leaves/P1.2/cards/R9.0]] | L21 |
| R9.1 | P1.2 | RPC, ratio-fed phase carrier on the frozen foil (Route 9.1) | T1 | returns to zero phase; every extensive feature span has positive curvature $\lambda$ | [[leaves/P1.2/cards/R9.1]] | L23 |
| R9.2 | P1.2 | LRK-PN, the low-rank carrier on a narrow projected foil (Route 9.2) | SHELVED | UNDECIDED at T1 (6×2 win, 4×3 tie); the decisive step was refused | [[leaves/P1.2/cards/R9.2]] | — |
| R9.3 | P1.2 | Fixed-node objective for the nodes, self-healing DMC on the lattice (Route 9.3) | G0 | known, with a documented convergence failure | [[leaves/P1.2/cards/R9.3]] | route-specific |
| R9.4 | P1.2 | Converged-horizon or equal-wall-time LRK (Route 9.4) | SHELVED | G2 prediction: a tie, since extra steps give width most of LRK's gain | [[leaves/P1.2/cards/R9.4]] | — |
| R10.1 | P1.2 | Span-S phase carrier: a linear phase on cross-site ratio products (reopening-ledger row 4) | T1 | the screen said OPEN ($\lambda(S)$ $-4.4$ to $-10.2$), but the trained carrier lost to width ($r$ 1.08, 1.01), 75–90% of its flips on near-node configurations | [[leaves/P1.2/cards/R10.1]] | L54 |

### P1.3 — Escaping wrong-sign basins

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | P1.3 | Lanczos-target sign refit (round-1 Route 1) | G0 | SWO and power method; a one-step refit flips signs only locally | [[leaves/P1.3/cards/R1]] | L21 |
| R2 | P1.3 | Orbital-swap basin hopping, OSBH (round-1 Route 2, re-audited as Route 2.0) | G3 | relaxed-energy acceptance error exceeds basin gaps; known moves, no null | [[leaves/P1.3/cards/R2]] | L33 |
| R3 | P1.3 | Sign-flip-aware Metropolis moves (round-1 Route 3) | G1 | SR fixed points do not depend on the sampler | [[leaves/P1.3/cards/R3]] | L19 |
| R4 | P1.3 | Hamiltonian continuation from the sign-free or non-interacting limit (round-1 Route 4) | G0 | the default start; follows the metastable branch through crossings | [[leaves/P1.3/cards/R4]] | L18 |
| R2.1 | P1.3 | Symmetrised natural-orbital re-anchoring with fixed-node acceptance, SNO-FN (round-2 Route 2.1, re-audited as Route 3.0) | G0 | acceptance is the killed lattice FN-GFMC; the target vanished under projection | [[leaves/P1.3/cards/R2.1]] | L1 |
| R2.2 | P1.3 | Fidelity-capped deflation excursions, FCD (round-2 Route 2.2) | G0 | SCF metadynamics with a cap | [[leaves/P1.3/cards/R2.2]] | route-specific |
| R2.3 | P1.3 | Irrep scan with fixed-node-ranked selection (round-2 Route 2.3) | G3 | competitors share the sector; otherwise standard projection | [[leaves/P1.3/cards/R2.3]] | L44 |
| R2.4 | P1.3 | Better local optimiser, MinSR or SPRING with a smaller rate (round-2 Route 2.4) | G0 | known optimisers reach the same basin faster | [[leaves/P1.3/cards/R2.4]] | L2 |
| R3.1 | P1.3 | Symmetric natural-orbital re-anchoring of a projected NNBF with sampled-energy acceptance, SNO-PE (round-3 Route 3.1) | T3 | no basin on any probed instance (410199; CAPACITY in 625182) | [[leaves/P1.3/cards/R3.1]] | L43 |
| R3.2 | P1.3 | In-sector orthogonal-complement probe, OCP (round-3 Route 3.2) | G0 | deflation and metadynamics are known | [[leaves/P1.3/cards/R3.2]] | route-specific |
| R3.3 | P1.3 | Diabatic branch transport in $t'$ (round-3 Route 3.3) | G0 | hysteresis sweeps are standard; ends in P3.2's killed energy ranking | [[leaves/P1.3/cards/R3.3]] | L18 |
| R3.4 | P1.3 | Projector annealing (round-3 Route 3.4) | G2 | the cost claim is unmeasurable with exact sums | [[leaves/P1.3/cards/R3.4]] | L39 |
| R7.1 | P1.3 | Second-order sign-oracle reachability probe (reopening Route R7.1) | T1 | second-order sign-oracle fit gains 22% against 50% needed | [[leaves/P1.3/cards/R7.1]] | L2 |
| R7.2 | P1.3 | Second-order energy optimiser to separate slow convergence from capacity (reopening Route R7.2) | G0 | dominated by R7.1 | [[leaves/P1.3/cards/R7.2]] | L2 |
| R7.3 | P1.3 | Same-sector competitor capture at a small in-sector gap (reopening Route R7.3) | T1 | 0 of 3 cold runs captured at $\Delta_{\mathrm{in}}=0.12t$ | [[leaves/P1.3/cards/R7.3]] | L2 |
| R7.4 | P1.3 | Projection escalation on demand (reopening Route R7.4) | G2 | as R3.4: dominated, and its cost claim is unmeasurable with exact sums | [[leaves/P1.3/cards/R7.4]] | L39 |
| R7.5 | P1.3 | Replica-exchange SR (reopening Route R7.5) | G0 | known; no basin to find | [[leaves/P1.3/cards/R7.5]] | L2 |
| R7.6 | P1.3 | Spin-projected training as a basin remedy (reopening Route R7.6) | G0 | standard; the question belongs to P4.1 | [[leaves/P1.3/cards/R7.6]] | route-specific |
| R7.7 | P1.3 | Direct sign-oracle basin probe on the 4×4 instance (reopening ledger item I1) | SHELVED | GATE-REFUSED (G2: ORACLE-MISS or RELAX predicted, $p\ge0.8$; G4: about 250 CPU-h without a passed T2); P1.3 now EXHAUSTED at this budget | [[leaves/P1.3/cards/R7.7]] | L2 |

### P2.1 — Sub-cubic antisymmetric amplitudes

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | P2.1 | Finite-range convolutional backflow with Woodbury updates (Route 1) | G0 | this is SCALE; its locality gap to global backflow is not fixed | [[leaves/P2.1/cards/R1]] | route-specific |
| R2 | P2.1 | Readout-rank exact ratios for transformer backflow (Route 2) | G1 | the attention forward pass costs $\Theta(N_e^3d)$ | [[leaves/P2.1/cards/R2]] | route-specific |
| R3 | P2.1 | Local plus global-bottleneck backflow, LGB-BF (Route 3) | G0 | subfamily of Liu–Clark's weaker correction; the test could not fail | [[leaves/P2.1/cards/R3]] | route-specific |
| R4 | P2.1 | Unbiased stochastic ratio estimator for unrestricted NNBF (Route 4) | G1 | F1 cost bound; variance diverges at nodes | [[leaves/P2.1/cards/R4]] | L12 |
| R2.1 | P2.1 | Split-capacity hidden-fermion backflow, SC-HFDS (Route 2.1) | G0 | relabel of HFDS with a Jastrow | [[leaves/P2.1/cards/R2.1]] | route-specific |
| R2.3 | P2.1 | Low-rank-response AGP (Route 2.3) | G0 | singlet restriction of HFPS | [[leaves/P2.1/cards/R2.3]] | route-specific |
| R2.4 | P2.1 | Thouless rotation into an active virtual window (Route 2.4) | G0 | frozen subfamily of R2.1 | [[leaves/P2.1/cards/R2.4]] | route-specific |
| R2.5 | P2.1 | Delayed-acceptance two-stage Metropolis (Route 2.5) | G1 | the local energy still needs full ratios | [[leaves/P2.1/cards/R2.5]] | route-specific |
| R2.6 | P2.1 | Fast-multipole hierarchical backflow (Route 2.6) | G2 | pooled far field is a weak subfamily; untestable at exact-sum size | [[leaves/P2.1/cards/R2.6]] | L39 |
| R3.1 | P2.1 | SK-LR, shared-reference low-rank sum, round-3 test (Route 3.1) | G4 | a sum of $K$ terms cannot carry a block-parity sign | [[leaves/P2.1/cards/R3.1]] | L53 |
| R3.2 | P2.1 | SK-AGP, shared pairing-reference low-rank sum (Route 3.2) | G4 | same parity argument as R3.1 | [[leaves/P2.1/cards/R3.2]] | L53 |
| R3.3 | P2.1 | Image-table projected SK-LR (Route 3.3) | G1 | per-image forwards dominate the cost | [[leaves/P2.1/cards/R3.3]] | route-specific |
| R3.4 | P2.1 | SCALE plus an unrestricted Liu–Clark block (Route 3.4) | G3 | the local part is global on $4\times4$ | [[leaves/P2.1/cards/R3.4]] | L39 |
| R6.1 | P2.1 | Global-kernel linear-readout backflow, GK-BF (Route 6.1) | G4 | one fixed $F\times F$ hidden-fermion factor | [[leaves/P2.1/cards/R6.1]] | L53 |
| R6.1′ | P2.1 | Local-hidden-row HFDS with a plus-shaped patch, LH-HFDS (Route 6.1′) | T2 | $\rho_F=0.868$, above 0.85 (75 steps, one seed) | [[leaves/P2.1/cards/R6.1p]] | L33 |
| R6.1″ | P2.1 | LH-HFDS with a $3\times3$ patch (Route 6.1″) | CRITIC | 75-step learning speed only; wide NNBF wins at equal wall time | [[leaves/P2.1/cards/R6.1pp]] | L33 |
| R6.2 | P2.1 | Product of local rank-1 determinant-ratio factors, PLR (Route 6.2) | G1 | pole of order $V-1$ at the reference's nodes | [[leaves/P2.1/cards/R6.2]] | L53 |
| R6.3 | P2.1 | Translation-equivariant trunk with shared image tables (Route 6.3) | G4 | cubic cost, or a finite-size projection bonus | [[leaves/P2.1/cards/R6.3]] | L10 |
| R6.4 | P2.1 | Hierarchical local-hidden-row HFDS, Tree-HFDS (Route 6.4) | G3 | no testable hierarchy at exact-sum size | [[leaves/P2.1/cards/R6.4]] | L39 |
| R7.1 | P2.1 | Local-feature hidden-fermion pairing state, LH-HFPS (Route 7.1) | SHELVED | 820 CPU-h T3 held; no cost case at reachable sizes | [[leaves/P2.1/cards/R7.1]] | — |
| R7.2 | P2.1 | Two-layer local trunk, LH2 (Route 7.2) | G0 | a patch-size variant; near-global at exact-sum size | [[leaves/P2.1/cards/R7.2]] | L39 |
| R7.3 | P2.1 | Sparse hidden–hidden HFDS, SH-HFDS (Route 7.3) | CRITIC | learning-speed PROMOTE only; 8 times slower per step, no cost case | [[leaves/P2.1/cards/R7.3]] | L33 |
| R7.4 | P2.1 | Linear-attention backflow with a global summary (Route 7.4) | G4 | fixed-rank factor, or hits the F1 cost bound | [[leaves/P2.1/cards/R7.4]] | L53 |
| R11.1 | P2.1 | Sublattice hidden rows, checkerboard `half` at $F=4$ (Round 11b, 11c, 12, 12b) | T1 | $\rho$ 1.59–2.44 at all four sizes | [[leaves/P2.1/cards/R11.1]] | L28 |
| R11.2 | P2.1 | Sublattice hidden rows, `quarter` at $F=4$ (Round 11b, 11c, 12, 12b) | T1 | $\rho$ 1.86–9.92 | [[leaves/P2.1/cards/R11.2]] | L28 |
| R13.1 | P2.1 | Schur-complement engine for the all-sites route, SCH-full (Round 13.1) | T1 | $R_D-\sigma=2.66$ at $L=16$, margin not resolved | [[leaves/P2.1/cards/R13.1]] | L47 |
| R13.1′ | P2.1 | Schur engine scored at $L=24$ (Round 13.1′) | T1 | $R_D=2.02\pm0.80$ at $L=24$ | [[leaves/P2.1/cards/R13.1p]] | L47 |
| R13.2 | P2.1 | Hole-gated hidden rows, `hg` (Round 13.2) | T2 | $\rho$ 1.37 and 2.04 at two counted sizes (partial data, KILL clause complete; upheld by the per-leaf critic, 2026-11-17) | [[leaves/P2.1/cards/R13.2]] | L28 |
| R13.3 | P2.1 | Checkerboard plus hole-gated rows, `cbh` (Round 13.3) | T1 | $R_D=1.71\pm0.43$, below the cost gate | [[leaves/P2.1/cards/R13.3]] | L47 |
| R13.4 | P2.1 | Shared low-rank hidden couplings, LR (Round 13.4) | G4 | at most 1.5 times saving over R13.1 | [[leaves/P2.1/cards/R13.4]] | route-specific |
| R13.5 | P2.1 | Local sign factor on the pairing reference, LSF (Round 13.5) | G0 | known: CPS times Slater | [[leaves/P2.1/cards/R13.5]] | route-specific |
| R13.6 | P2.1 | Delayed acceptance with `half` as surrogate, DA (Round 13.6) | G4 | saving at most 1.4 times | [[leaves/P2.1/cards/R13.6]] | route-specific |
| R13.7 | P2.1 | Hole-gated rows at $F=8$ (Round 13.7) | SHELVED | G2 predicted outcome: run condition $\rho\le1.5$ not met (an unmet run condition is not a refutation; was T0, overturned by the per-leaf critic 2026-11-17); unshelve condition not met (R12.3 KILLED at $V=15$, job 731409); was: unshelved if R12.3 passes at $V=15$ | [[leaves/P2.1/cards/R13.7]] | — |
| R12.3 | P2.1 | Checkerboard `half` at larger hidden width $F$ (Round 12 option 3; Rounds 12c and 12d) | T1 | trend clause: $\rho=0.038$, 0.886, 1.110, 1.915 at $V=8$, 12, 14, 15 against `full_F4` (job 731409), above 1.15 and 1.5 | [[leaves/P2.1/cards/R12.3]] | L28 |
| R15.2 | P2.1 | Checkerboard rows with a radius-2 receptive field (Round 15.2) | G4 | rows changed per hop rise from 4 to 9, so predicted $R_D(16)=3.3$–4.2, no better than R13.1's failed 3.50 | [[leaves/P2.1/cards/R15.2]] | L58 |
| R15.3 | P2.1 | Schur engine for the all-sites route scored at $L=32$ (Round 15.3) | G4 | measured ratio 3.54 → 3.89 from $L=16$ to 24 extrapolates to about 4.2 at $L=32$, below a scaled bar of 6 | [[leaves/P2.1/cards/R15.3]] | L47 |
| R15.4 | P2.1 | Domain-local hidden couplings (Round 15.4; from local correlation in quantum chemistry) | G0 | finite-range backflow in the pairing frame (SCALE, tensor backflow); the locality accuracy gap is unaddressed | [[leaves/P2.1/cards/R15.4]] | L57 |
| R15.5 | P2.1 | Near-field plus low-rank far-field hidden couplings (Round 15.5; from fast multipole methods) | G0 | the affordable far rank $V/8$ is the published low-rank backflow restriction with its expressivity loss | [[leaves/P2.1/cards/R15.5]] | L57 |
| R15.6 | P2.1 | Learned sparse gating of hidden rows (Round 15.6; from mixture-of-experts routing) | T1 | $\rho(12)=2.57$ at active fraction 0.495, worse than the plain checkerboard it contains (1.59) | [[leaves/P2.1/cards/R15.6]] | L33 |
| R15.7 | P2.1 | Multiple-try Metropolis on cheap candidate ratios (Round 15.7) | G4 | accepted updates dominate the sweep; extra candidates add ratios without more decorrelation per accepted hop | [[leaves/P2.1/cards/R15.7]] | route-specific |

### P2.2 — Cheap local energy for chemistry

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1.1 | P2.2 | Heat-bath / semistochastic stochastic local energy (Route 1) | G0 | prior art; single-proposal variance floor (heat-bath $657\times$ exact) | [[leaves/P2.2/cards/R1.1]] | L12 |
| R1.2 | P2.2 | Doubly stochastic SR force (Route 2) | G1 | noise floor amplified by the inverse SR metric | [[leaves/P2.2/cards/R1.2]] | L12 |
| R1.3 | P2.2 | Factorized-Hamiltonian (Cholesky) proposal (Route 3) | G2 | even the oracle proposal costs $146\times$ exact | [[leaves/P2.2/cards/R1.3]] | L12 |
| R1.4 | P2.2 | Frozen-backflow semistochastic control variate, SCV (Route 4) | CRITIC | residual fraction 0.56–0.82, cost ratio $M\approx1.03$ | [[leaves/P2.2/cards/R1.4]] | L8 |
| R2.0 | P2.2 | Revived frozen-backflow control variate at measured per-term cost | CRITIC | hybrid cost model; no new mechanism | [[leaves/P2.2/cards/R2.0]] | L51 |
| R2.1 | P2.2 | Positivised (sign-stratified) semistochastic local energy, P2.2′ | G2 | within 3% of single-proposal sampling | [[leaves/P2.2/cards/R2.1]] | route-specific |
| R2.2 | P2.2 | Epstein–Nesbet perturbative tail proposal, PT2IS | G3 | integral-only proposal ($\rho=0.88$, $M$ rises) | [[leaves/P2.2/cards/R2.2]] | route-specific |
| R2.3 | P2.2 | Linearized-backflow surrogate (LRX) summed over all terms | G2 | 0.21 forward passes per term | [[leaves/P2.2/cards/R2.3]] | route-specific |
| R2.4 | P2.2 | Multilevel Monte Carlo over surrogate fidelity, FB to LRX to exact | G2 | $M=1.03$–1.08 | [[leaves/P2.2/cards/R2.4]] | route-specific |
| R2.5 | P2.2 | Cost-aware cross-sample allocation of residual draws | G2 | $M\approx1$ | [[leaves/P2.2/cards/R2.5]] | route-specific |
| R2.6 | P2.2 | Active-space-restricted backflow input | G3 | no external pair correlation | [[leaves/P2.2/cards/R2.6]] | L8 |
| R2.7 | P2.2 | Row sketching by randomized linear algebra | G1 | needs every amplitude | [[leaves/P2.2/cards/R2.7]] | route-specific |
| R2.8 | P2.2 | Exact Cholesky/THC contraction of the frozen surrogate sum | G0 | known (AFQMC) | [[leaves/P2.2/cards/R2.8]] | route-specific |
| R2.9 | P2.2 | Multilevel Monte Carlo across basis-set levels | G2 | it is semistochastic sampling | [[leaves/P2.2/cards/R2.9]] | route-specific |
| R3.0 | P2.2 | Tangent-Slater surrogate | G1 | diverges at nodes | [[leaves/P2.2/cards/R3.0]] | route-specific |
| R3.1′ | P2.2 | Cholesky sum-of-squares energy estimator (Jackson–Feenberg form) | G1 | support condition fails | [[leaves/P2.2/cards/R3.1p]] | route-specific |
| R3.2′ | P2.2 | Cross-sample multifidelity correction | G2 | $\mathrm{Var}(\delta)/\mathrm{Var}(E_L)$ from 2 to 466 | [[leaves/P2.2/cards/R3.2p]] | L25 |
| R3.3′ | P2.2 | Row-calibrated surrogate | G2 | extrapolates worse than the head terms | [[leaves/P2.2/cards/R3.3p]] | route-specific |
| R3.4′ | P2.2 | Amplitude caching across the batch or Markov chain | G2 | overlap is $\mathcal O(NM)$ | [[leaves/P2.2/cards/R3.4p]] | route-specific |
| R3.5′ | P2.2 | Horvitz–Thompson (Poisson or priority) sampling of terms | G2 | same dispersion as plain sampling | [[leaves/P2.2/cards/R3.5p]] | route-specific |
| R3.6′ | P2.2 | Exact low-rank local energy | G1 | needs fixed orbitals | [[leaves/P2.2/cards/R3.6p]] | route-specific |
| R3.1 | P2.2 | Window-input backflow × all-orbital Jastrow, WIBJ, round 3 | G2 | $M=1.32$ on its own pre-check | [[leaves/P2.2/cards/R3.1]] | route-specific |
| R3.2 | P2.2 | Window-keyed partially-frozen control variate, PFCV | CRITIC | relabel of the revived control variate; $M>1$ | [[leaves/P2.2/cards/R3.2]] | L26 |
| R4.0 | P2.2 | Factorized-Jastrow keyed Cholesky contraction, FJKC | G1 | same order as enumeration (0.67–0.87) | [[leaves/P2.2/cards/R4.0]] | route-specific |
| R4.1 | P2.2 | Coarse-grained-key backflow, CGK | T2 | $\delta=0.691$ mHa against 0.2 | [[leaves/P2.2/cards/R4.1]] | L8 |
| R4.2 | P2.2 | Cross-spin-keyed backflow, XSK | G3 | measured on the pre-check: losses 4.0, 3.8 and 10.2 mHa (19–51 $\tau$) against a converged NNBF-J control; variance 64 times only at $K=8$ | [[leaves/P2.2/cards/R4.2]] | route-specific |
| R5.1 | P2.2 | WIBJ re-entered under F$_{A1}$ with an exact keyed local energy | T3 | $M_{\mathrm{wall}}=4.03$ at $K=136$ | [[leaves/P2.2/cards/R5.1]] | L8 |
| R5.1s | P2.2 | Screened WIBJ, WIBJ plus SCREEN | CRITIC | post-hoc constant-factor rescue | [[leaves/P2.2/cards/R5.1s]] | L51 |
| R5.2 | P2.2 | Anchored local energy, ALE | SHELVED | GATE-REFUSED (G2): semistochastic sampling beat ALE on 5 of 5 cold-start seeds (worst SS 7.6 against best ALE 10.6 mHa), FAIL on a new instance predicted $p\approx0.85$, changes no action | [[leaves/P2.2/cards/R5.2]] | L26 |
| R5.3 | P2.2 | Hadamard-certified screening | G2 | certificate 140–1000 times loose | [[leaves/P2.2/cards/R5.3]] | L27 |
| R5.4 | P2.2 | Statistical bias certificate | G0 | it is semistochastic sampling | [[leaves/P2.2/cards/R5.4]] | route-specific |
| R5.5 | P2.2 | Stale-plus-linear local energy without correction | G1 | uncertified bias | [[leaves/P2.2/cards/R5.5]] | route-specific |
| R5.6 | P2.2 | Learned local-energy regressor, debiased on a subset | G2 | $R^2=0.13$ | [[leaves/P2.2/cards/R5.6]] | L25 |
| R6.1 | P2.2 | Contractible CAS × Jastrow without external–external pairs (F$_{A2}$ closed-leaf falsifier) | T1 | 0.3885 mHa against 0.2 at the smallest basis | [[leaves/P2.2/cards/R6.1]] | L8 |
| R7.1 | P2.2 | Variational product-weighted screening in canonical orbitals, VPS (C1) | T1 | certified kept fraction 0.917–0.976 | [[leaves/P2.2/cards/R7.1]] | L27 |
| R7.2 | P2.2 | Sum of $R$ configuration-independent Slater–Jastrows (C2) | G0 | prior art; a fixed-rank finite-size device | [[leaves/P2.2/cards/R7.2]] | L10 |
| R7.3 | P2.2 | Plain screening in localized orbitals (C3) | G0 | prior art (localized-orbital screening) | [[leaves/P2.2/cards/R7.3]] | L9 |
| R7.5 | P2.2 | Anchored local energy on the semistochastic tail only (C4) | G2 | tail variance only 1–2% | [[leaves/P2.2/cards/R7.5]] | L26 |
| R7.4 | P2.2 | Variational product-weighted screening in localized orbitals (C3′) | T0 | cost: the kept-pattern compensator needs reference amplitudes on about 1352 of 4900 determinants per sample at H$_8$; frozen-NQS reference 3.8 times EXACT (about 1.5 times at the grid end), cheap reference estimated to break the bound 2–20 times (accuracy side PASS-T1b only at $\Delta E\le0.03$ mHa) | [[leaves/P2.2/cards/R7.4]] | L27, L9 |

### P3.1 — Order-parameter error from cheap information

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1.1 | P3.1 | Lanczos-step variance extrapolation of the correlation function (Route 1) | G0 | Sorella–Becca; Krylov reaches the competitor only with weight $\propto\Delta$ | [[leaves/P3.1/cards/R1.1]] | L3 |
| R1.2 | P3.1 | Susceptibility (Legendre) certificate (Route 2) | G2 | the in-family susceptibility is too stiff on the foil | [[leaves/P3.1/cards/R1.2]] | L6 |
| R1.3 | P3.1 | Pinning or pairing-field response extrapolation (Route 3) | G2 | an ordered ansatz returns its own order | [[leaves/P3.1/cards/R1.3]] | L6 |
| R2.1 | P3.1 | A-Ritz on real states with an abstention-penalised rule (Route 2.1) | CRITIC | two-point extrapolation and noisy subspace solve | [[leaves/P3.1/cards/R2.1]] | L3 |
| R2.2 | P3.1 | Sign-defect Ritz (Route 2.2) | CRITIC | the flagged set is local-energy outliers | [[leaves/P3.1/cards/R2.2]] | route-specific |
| R2.3 | P3.1 | Cross-width zero-variance extrapolation (Route 2.3) | G2 | sign error does not shrink with width | [[leaves/P3.1/cards/R2.3]] | L41 |
| R2.4 | P3.1 | Wholesale one-step sign replacement (Route 2.4) | G1 | no variational control | [[leaves/P3.1/cards/R2.4]] | route-specific |
| R2.5 | P3.1 | Fixed-sign amplitude correction as error bar (Route 2.5) | G2 | recovers only 7–29% of the error | [[leaves/P3.1/cards/R2.5]] | route-specific |
| R2.6 | P3.1 | Zero-variance zero-bias improved estimators (Route 2.6) | G0 | reduces variance, not bias | [[leaves/P3.1/cards/R2.6]] | route-specific |
| R3.1 | P3.1 | Ensemble-calibrated covariance extrapolation (ECC, Route 3.1) | CRITIC | the regressor is the first $\hat H$ moment; pilot NO-GO | [[leaves/P3.1/cards/R3.1]] | L3 |
| R3.2 | P3.1 | Hylleraas-trained response network (ZR, Route 3.2) | CRITIC | Assaraf–Caffarel; one network per observable | [[leaves/P3.1/cards/R3.2]] | L14 |
| R3.3 | P3.1 | Seed non-orthogonal CI (seed NOCI, Route 3.3) | G1 | cross-overlap cost exponential in $V$ | [[leaves/P3.1/cards/R3.3]] | L7 |
| R3.4 | P3.1 | In-family zero-bias correction (Route 3.4) | G1 | tangent-space blindness | [[leaves/P3.1/cards/R3.4]] | L13 |
| R3.5 | P3.1 | Provable sign witness on projected states (Route 3.5) | G2 | projected sign error about 0 | [[leaves/P3.1/cards/R3.5]] | route-specific |
| R3.6 | P3.1 | Laplace or Fisher posterior error bar (Route 3.6) | G2 | tangent-space; a stiff foil gets a small error bar | [[leaves/P3.1/cards/R3.6]] | L13 |
| R3.7 | P3.1 | Single-state covariance extrapolation along one Lanczos step (Route 3.7) | G0 | R1.1 in a new variable | [[leaves/P3.1/cards/R3.7]] | L3 |
| R5.1 | P3.1 | Observable-split Galerkin zero-bias correction (OSG, Route 5.1) | G0 | ZVZB with a fixed basis; wrong sign on 4 of 6 | [[leaves/P3.1/cards/R5.1]] | route-specific |
| R5.2 | P3.1 | Ensemble sign and amplitude hybrid extrapolation (EHX, Route 5.2) | T1 | injected check 0 of 7 | [[leaves/P3.1/cards/R5.2]] | L42 |
| R5.3 | P3.1 | Single-mode approximation ratio (SMA, Route 5.3) | G0 | an $\hat H$-moment ratio | [[leaves/P3.1/cards/R5.3]] | L3 |
| R5.4 | P3.1 | Local imaginary-time step (Route 5.4) | CRITIC | an $\hat H$-generated direction | [[leaves/P3.1/cards/R5.4]] | L3 |
| R6.1 | P3.1 | Capacity-ladder extrapolation with a Lanczos-cleaned regressor (CLX, Route 6.1) | T1 | injected 0 of 4; jackknife error about the error itself | [[leaves/P3.1/cards/R6.1]] | L42 |
| R6.2 | P3.1 | Twist or boundary-condition averaging as an error signal (Route 6.2) | G2 | targets finite-size, not variational, error; OBC | [[leaves/P3.1/cards/R6.2]] | route-specific |
| R6.3 | P3.1 | Field derivative of the ladder's extrapolated energy (Route 6.3) | T0 | reduces to the second $\hat H$ moment | [[leaves/P3.1/cards/R6.3]] | L3 |
| R6.4 | P3.1 | Architecture disagreement calibrated by an independent-error model (Route 6.4) | G0 | consensus or seed spread; shared bias | [[leaves/P3.1/cards/R6.4]] | L41 |
| R6.5 | P3.1 | Proxy-calibrated alignment (Route 6.5) | G2 | in-family susceptibility; transfer unproven | [[leaves/P3.1/cards/R6.5]] | L6 |
| R7.1 | P3.1 | Excited-state subspace Ritz (ESR0, Route 7.1) | G4 | window $K_{1/2}$ grows (2 to 17.5; real 11) | [[leaves/P3.1/cards/R7.1]] | L4 |
| R7.1a | P3.1 | ESR0 extension to six excited states (step 1.5b, job 618646) | SHELVED | the verdict could not change a decision | [[leaves/P3.1/cards/R7.1a]] | L48 |
| R7.1b | P3.1 | GO-6 step 2: excited states for seeds 22 and 33 | SHELVED | about 1280 CPU-h, above the leaf budget | [[leaves/P3.1/cards/R7.1b]] | — |
| R7.2 | P3.1 | Response of the projected network to a conjugate field (Route 7.2) | G2 | measures response, not offset | [[leaves/P3.1/cards/R7.2]] | L6 |
| R7.3 | P3.1 | Certificate from Weinstein or Temple bounds in the low window (Route 7.3) | G1 | needs lower bounds; floor 9 to 81 times the error | [[leaves/P3.1/cards/R7.3]] | L6 |
| R7.4 | P3.1 | Diagonal-ensemble (real-time dephasing) estimator (Route 7.4) | G1 | TDVP fixed point at the minimum | [[leaves/P3.1/cards/R7.4]] | L13 |
| R7.5 | P3.1 | Tangent-space Ritz (linear method at convergence, Route 7.5) | G1 | Brillouin condition: zero correction | [[leaves/P3.1/cards/R7.5]] | L13 |
| R7.6 | P3.1 | Ritz-trained single auxiliary network (RTA, Route 7.6) | G2 | a two-member capacity ladder | [[leaves/P3.1/cards/R7.6]] | L41 |
| R7.7 | P3.1 | Multi-seed Ritz (MSR, Route 7.7) | G2 | the seeds share the sign of the error | [[leaves/P3.1/cards/R7.7]] | L41 |
| R8.1 | P3.1 | Response-window Ritz (RWR, Route 8.1) | T1 | tracks 0.37 with noisy responses at $4\times3$ | [[leaves/P3.1/cards/R8.1]] | L14 |
| R8.2 | P3.1 | Field-ladder eigenvector continuation (Route 8.2) | T0 | Brillouin; shared bias | [[leaves/P3.1/cards/R8.2]] | L13 |
| R8.3 | P3.1 | Structure-factor targeting (Route 8.3) | T2 | NO-TARGET: structure factors already accurate | [[leaves/P3.1/cards/R8.3]] | L5 |
| R8.4 | P3.1 | Observable-ordered excited-state window (AOW, Route 8.4) | T1 | $K_{1/2}$ grows 1 to 19 | [[leaves/P3.1/cards/R8.4]] | L4 |
| R8.5 | P3.1 | Two-mode (Bijl–Feynman) window (TMW, Route 8.5) | T1 | operator vectors sit at high excitation (0.09, 0.47) | [[leaves/P3.1/cards/R8.5]] | route-specific |
| R8.6 | P3.1 | Extrapolation in the window size (Route 8.6) | T0 | out-of-window error unidentifiable | [[leaves/P3.1/cards/R8.6]] | L4 |
| R8.7 | P3.1 | Momentum-sector-resolved window (Route 8.7) | T0 | selection rule; no momentum sectors with open boundaries | [[leaves/P3.1/cards/R8.7]] | route-specific |
| R8.8 | P3.1 | Response-window Ritz on concentrated structure factors (RWR-SF, Route 8.8) | T1 | tracks 0.33 at $4\times3$ | [[leaves/P3.1/cards/R8.8]] | L14 |
| R8.9 | P3.1 | Mode-resolved windows (Route 8.9) | T0 | cancellation ratio of order 100 | [[leaves/P3.1/cards/R8.9]] | L5 |
| R9.1 | P3.1 | Krylov-cleaned response Ritz (RWR-K, Route 9.1) | T1 | tracking falls with level density; dominated by 10.1 | [[leaves/P3.1/cards/R9.1]] | L14 |
| R9.2 | P3.1 | Excited-state-deflated response (EDR, Route 9.2) | T1 | no gain over R9.1 | [[leaves/P3.1/cards/R9.2]] | route-specific |
| R9.3 | P3.1 | Krylov-cleaned two-mode window over all modes (TMW-K, Route 9.3) | T1 | 0.43 at $6\times2$; the passing instance was trivially spanned | [[leaves/P3.1/cards/R9.3]] | L40 |
| R9.4 | P3.1 | Response-ensemble averaging (Route 9.4) | G2 | coherent bias across trainings | [[leaves/P3.1/cards/R9.4]] | L41 |
| R10.1 | P3.1 | Self-consistent response Ritz with a fixed base (SCR, Route 10.1) | G4 | not size-consistent (recovered 0.91 to 0.53) | [[leaves/P3.1/cards/R10.1]] | L10 |
| R10.1d | P3.1 | Depth-free self-consistent response Ritz (Route 10.1d) | SHELVED | T3 AMBIGUOUS (9 of 11 against a capacity foil's 7 of 10, two of the foil's hits borderline at 1.975 and 0.528); the $4\times4$ test needs about 2400 CPU-h (G5) | [[leaves/P3.1/cards/R10.1d]] | L38 |
| R10.2 | P3.1 | Response seeded by the two-mode Ritz vector (TMW-SCR, Route 10.2) | G0 | subspace-expansion noise wall unaddressed; fixed base | [[leaves/P3.1/cards/R10.2]] | L10 |
| R10.3 | P3.1 | Depth-2 Krylov two-mode window (TMW-K2, Route 10.3) | T1 | tracking drops 0.27–0.39 by $4\times3$ | [[leaves/P3.1/cards/R10.3]] | L3 |
| R10.4 | P3.1 | Observable-free generalised-Davidson error bars (Route 10.4) | T1 | gap-stress tracking 0.00–0.18 | [[leaves/P3.1/cards/R10.4]] | L15 |
| R12.1 | P3.1 | Observable-sourced gap witness (OGW, Route 12.1) | T1 | false certificates 0.88 and 0.71 | [[leaves/P3.1/cards/R12.1]] | L15 |
| R12.2 | P3.1 | Shift-swept susceptibility pole (CPG, Route 12.2) | G0 | correction-vector spectroscopy; signal below training scatter | [[leaves/P3.1/cards/R12.2]] | route-specific |

### P3.2 — Decision rule for a biased ensemble

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1.1 | P3.2 | Calibrated independent-branch extrapolation (round-1 Route 1) | G0 | current practice; echoes the seed | [[leaves/P3.2/cards/R1.1]] | route-specific |
| R1.2 | P3.2 | Sector-resolved branches (round-1 Route 2) | G3 | competitors share a sector | [[leaves/P3.2/cards/R1.2]] | L44 |
| R1.3 | P3.2 | Temple and Weinstein eigenvalue brackets (round-1 Route 3) | G1 | needs a gap that closes | [[leaves/P3.2/cards/R1.3]] | L6 |
| R1.4 | P3.2 | Jointly optimised two-state ensemble with character-labelled Ritz extrapolation (round-1 Route 4) | G0 | natural-excited-state VMC; its test used different sectors | [[leaves/P3.2/cards/R1.4]] | L16 |
| R1.4a | P3.2 | Non-orthogonal CI of independently optimised branches (sub-variant of round-1 Route 4) | G1 | cross-overlap variance $e^{\mathcal O(V)}$ | [[leaves/P3.2/cards/R1.4a]] | L7 |
| R2.1 | P3.2 | Two-reference shared-backflow superposition, 2R-SS (round-2 Route 2.1) | G0 | multi-determinant backflow; heavy-tailed cross terms | [[leaves/P3.2/cards/R2.1]] | L7 |
| R2.2 | P3.2 | State-averaged two-reference shared backflow with order-operator diabatisation, 2R-SA (round-2 Route 2.2) | G0 | state-averaged VMC; distinct orders share no local structure | [[leaves/P3.2/cards/R2.2]] | L7 |
| R2.3 | P3.2 | Two-sided conjugate-field response (round-2 Route 2.3) | G2 | boundary $\delta+\Delta b=0$; basin-trappable | [[leaves/P3.2/cards/R2.3]] | L17 |
| R2.4 | P3.2 | Thermodynamic integration from a field-dominated anchor (round-2 Route 2.4) | G1 | first-order observable errors | [[leaves/P3.2/cards/R2.4]] | route-specific |
| R2.5 | P3.2 | Okamoto–Nomura level spectroscopy with twisted boundaries (round-2 Route 2.5) | G3 | no twist separates same-sector orders | [[leaves/P3.2/cards/R2.5]] | L44 |
| R2.6 | P3.2 | Bayesian evidence comparison of the two branches (round-2 Route 2.6) | G1 | no likelihood; it is R1.1 | [[leaves/P3.2/cards/R2.6]] | route-specific |
| R2.7 | P3.2 | Conical-intersection Berry-phase tracking in the $(t',h)$ plane (round-2 Route 2.7) | G2 | locates the biased seam | [[leaves/P3.2/cards/R2.7]] | L17 |
| R2.8 | P3.2 | Decision from the momentum-resolved structure factor of the winning branch (round-2 Route 2.8) | G0 | it is the foil (R0+) | [[leaves/P3.2/cards/R2.8]] | route-specific |
| R3.1 | P3.2 | Projected-NQS level-crossing detector (round-3 Route 3.1) | G2 | kink at $\delta+\Delta b=0$ | [[leaves/P3.2/cards/R3.1]] | L17 |
| R3.2 | P3.2 | Decision from the curvature or susceptibility of a projected branch (round-3 Route 3.2) | G1 | term absent; needs cross-order elements | [[leaves/P3.2/cards/R3.2]] | L7 |
| R3.3 | P3.2 | Conical-intersection localisation by the Berry phase on Eckart-certified loops (round-3 Route 3.3) | G2 | resolution about the full error | [[leaves/P3.2/cards/R3.3]] | L6 |
| R3.4 | P3.2 | In-irrep zero-variance extrapolation with a contamination-sign lemma (round-3 Route 3.4) | G0 | mVMC practice | [[leaves/P3.2/cards/R3.4]] | route-specific |
| R3.5 | P3.2 | Same-order eigenvector continuation, SO-EC (round-3 Route 3.5) | T3 | wrong-commit rate 0.33 at width 4 (job 451852) | [[leaves/P3.2/cards/R3.5]] | L17 |
| R3.6 | P3.2 | Affine transport of snapshot expectation vectors, AT (round-3 Route 3.6) | CRITIC | wrong even with exact snapshots | [[leaves/P3.2/cards/R3.6]] | L17 |
| R5.1 | P3.2 | Eckart-certified direct decision, ECD (round-5 Route 5.1) | T1 | inner decision rate 0 even with exact inputs | [[leaves/P3.2/cards/R5.1]] | L45 |
| R5.2 | P3.2 | Doublet certificate (round-5 Route 5.2) | G2 | in-doublet rotation error about $\varepsilon/\Delta_1$ | [[leaves/P3.2/cards/R5.2]] | L6 |
| R5.3 | P3.2 | Ladder-crossing certificate, LCC (round-5 Route 5.3) | CRITIC | R3.1 plus an anti-conservative margin; ladder centred on the crossing | [[leaves/P3.2/cards/R5.3]] | L46 |
| R6.1 | P3.2 | Excited-state Ritz with one penalty-trained state, K1 (round-6 ESR) | G2 | $2\times2$ lemma; 9 of 9 lower-diagonal (619080) | [[leaves/P3.2/cards/R6.1]] | L16 |
| R6.2 | P3.2 | Excited-state Ritz with the residual direction, ESR0 (round-6 ESR) | G2 | a one-sided Lanczos step under the same lemma | [[leaves/P3.2/cards/R6.2]] | L16 |
| R6.3 | P3.2 | Ritz in the span of the two order branches, PAIR (round-6 kill-check span) | G2 | the energy rule (lemma) | [[leaves/P3.2/cards/R6.3]] | L16 |
| R6.4 | P3.2 | Symmetric block-Lanczos step on both trained states, BL (round-6 alternative) | G0 | Sorella practice; exact matrices only | [[leaves/P3.2/cards/R6.4]] | L37 |
| R6.5 | P3.2 | Sampled single-branch Lanczos step, L1 (closed-leaf falsifier) | T1 | sampled: right 1.3%, wrong 0.6% (LOSES) | [[leaves/P3.2/cards/R6.5]] | L42 |
| R6.6 | P3.2 | Sampled pair block-Lanczos step, PBL (closed-leaf falsifier) | T1 | premise fails (2 of 6) | [[leaves/P3.2/cards/R6.6]] | route-specific |
| R7.1 | P3.2 | Lanczos step at the Hard-Core per-site scale (reopening candidate C1) | G4 | gain $\propto\sqrt V$ against bias $\propto V$ | [[leaves/P3.2/cards/R7.1]] | L10 |
| R7.2 | P3.2 | Mixture-paired difference estimator (reopening candidate C2) | G4 | overlap $e^{-cV}$ | [[leaves/P3.2/cards/R7.2]] | L7 |
| R7.3 | P3.2 | Symmetric finite-$\tau$ imaginary-time step (reopening candidate C3) | G1 | sign problem or constraint bias | [[leaves/P3.2/cards/R7.3]] | route-specific |
| R7.4 | P3.2 | Branch-stability probe (reopening candidate C4) | G2 | the landscape is biased | [[leaves/P3.2/cards/R7.4]] | L17 |
| R7.5 | P3.2 | Small-cluster bias calibration (reopening candidate C5) | G3 | ED clusters lack the competitors | [[leaves/P3.2/cards/R7.5]] | route-specific |
| R7.6 | P3.2 | Sequential (Wald) sample budget for the energy difference (ledger item I2b) | T0 | a stopping rule changes the sample count, not the sign of $\delta+\Delta b$; sampling is not binding at the Hard-Core scale | [[leaves/P3.2/cards/R7.6]] | route-specific |

### P4.1 — Exact SU(2) below projection cost

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R0.1 | P4.1 | round-1 charge-conditioned singlet geminal with exchange dressing (`r1`, `r1x`) | G0 | prior art (Tocchio et al.); exchange dressing gain vanishes per site | [[leaves/P4.1/cards/R0.1]] | route-specific |
| R0.2 | P4.1 | round-1 stochastic SU(2) projection of a spin-resolved NNBF | G1 | orbit sum: exponential relative variance, biased ratios | [[leaves/P4.1/cards/R0.2]] | L11 |
| R0.3 | P4.1 | round-1 analytic projection by polynomial roots | G2 | same accuracy as the foil, no cost gain | [[leaves/P4.1/cards/R0.3]] | route-specific |
| R0.4 | P4.1 | round-1 spin-adapted-basis NQS (CSF, GUGA or valence-bond basis) | G0 | known; sign simplifications only at half filling | [[leaves/P4.1/cards/R0.4]] | route-specific |
| R2.1 | P4.1 | local spin group-algebra dressing beyond first order | G1 | linear form not size-consistent; product form leaves the closed-form class | [[leaves/P4.1/cards/R2.1]] | L10 |
| R2.3 | P4.1 | Shiba (partial particle–hole) frame | G0 | the frame relabels the barrier | [[leaves/P4.1/cards/R2.3]] | route-specific |
| R2.4 | P4.1 | numerator-only stochastic projection of a spin-resolved NNBF | G1 | average sign decays as $e^{-a\sqrt{N_s}}$ | [[leaves/P4.1/cards/R2.4]] | L11 |
| R2.5 | P4.1 | spin–charge factorised SU(2)-symmetric MPS (SCF; also Route 3.5 and Route 4.0b) | G1 | cost exponential in width in 2D | [[leaves/P4.1/cards/R2.5]] | route-specific |
| R3.1 | P4.1 | closed-form per-amplitude projection of a charge-backflowed determinant (Che's method; the `che` control) | G0 | published (Che's method); kept as the control | [[leaves/P4.1/cards/R3.1]] | route-specific |
| R3.2 | P4.1 | spin-multilinear backflow projected at $N_q$ nodes | G1 | exponential orbit sum (Lemma B) | [[leaves/P4.1/cards/R3.2]] | L11 |
| R3.3 | P4.1 | SU(2)-invariant network on continuous spinors | G1 | exponential integrals or no determinant layer | [[leaves/P4.1/cards/R3.3]] | route-specific |
| R3.4 | P4.1 | spin–charge factorised RVB with charge-conditioned spinon pairing (SQ-RVB; also Route 4.0) | G0 | known class at 3.3 times its cost | [[leaves/P4.1/cards/R3.4]] | route-specific |
| R4.1 | P4.1 | polynomial spin-feature subclass of exact projection (`sfb`) | T2 | one spin feature added nothing | [[leaves/P4.1/cards/R4.1]] | route-specific |
| R4.2 | P4.1 | charge-conditioned sum of $K$ singlet AGPs (KAGP, `kagp2`) | T2 | trained gain ratio 0.88 despite the class premise | [[leaves/P4.1/cards/R4.2]] | L33 |
| R4.3 | P4.1 | exact projection of a spin Jastrow times a charge-conditioned determinant | G1 | exponential orbit sum (Lemma B) | [[leaves/P4.1/cards/R4.3]] | L11 |
| R4.4 | P4.1 | hidden spinful ancillas in the spinon AGP | G2 | Gaussian ancillas cannot leave the one-AGP class | [[leaves/P4.1/cards/R4.4]] | route-specific |
| R5.1 | P4.1 | few-point spin quadrature on a spin-resolved network | G1 | exponential orbit sum (Lemma B) | [[leaves/P4.1/cards/R5.1]] | L11 |
| R5.2 | P4.1 | truncated Löwdin filter trained in the loop | G1 | not exact at fixed $d$; cost exponent grows | [[leaves/P4.1/cards/R5.2]] | route-specific |
| R5.3 | P4.1 | sampled projection inside the amplitude, and its two mutations | G1 | sampled projection: exponential variance | [[leaves/P4.1/cards/R5.3]] | L11 |
| R5.4h | P4.1 | one backflow head per component (`che_kh4`, the containing foil) | T3 | worse than `che`; the gain needs the shared backflow | [[leaves/P4.1/cards/R5.4h]] | route-specific |
| R5.5 | P4.1 | projected UHF mixture from enumerated hidden units (`che_rbm`, ranks 2, 4, 8) | SHELVED | rank ladder AMBIGUOUS; extension (about 19,200 CPU-h cap) cancelled | [[leaves/P4.1/cards/R5.5]] | — |
| R5.6 | P4.1 | two to four integer collective spin features (the `sfb` extension) | G2 | redundant with the foil at higher cost | [[leaves/P4.1/cards/R5.6]] | route-specific |
| R5.7 | P4.1 | `che` plus a small spin-resolved correction trained on the residual | G1 | exponential orbit sum (Lemma B) | [[leaves/P4.1/cards/R5.7]] | L11 |
| R1 | P4.1 | fixed-rank exact-$S$ projected mixture on a shared charge backflow (`che_kK`) | G4 | fixed-rank mixture: finite-size gain, $R(8,48)=0.64$–0.82 projected | [[leaves/P4.1/cards/R1]] | L10 |
| R1-L | P4.1 | R1 size ladder at 12 to 48 sites (job 686990 with DMRG 687302) | SHELVED | about 13,000 CPU-h to measure an outcome predicted by Gate 4 | [[leaves/P4.1/cards/R1-L]] | L48 |
| R1-G | P4.1 | R1 generality on two more 12-site instances (job 637776) | SHELVED | a 12-site result predicted not to scale; changes no decision | [[leaves/P4.1/cards/R1-G]] | L48 |
| R2-T2b | P4.1 | R2 at 24 sites against `che`, `r1` and an equal-wall-time `che` (jobs 699824 and 703587) | SHELVED | optimiser-confounded after Round 11 (polished `che` matches `pcg`) | [[leaves/P4.1/cards/R2-T2b]] | L33 |
| R3 | P4.1 | extensive charge backflow under closed-form Che projection (`che_ext`: (i) local equivariant `che_loc`, (ii) width $H\propto N$) | SHELVED | premise W FAILS at 24 sites; 48-site arms lived only in the cancelled ladder | [[leaves/P4.1/cards/R3]] | — |
| R3iii | P4.1 | hidden-fermion rational head with a charge-only hidden network (`che_hf`; Route 6 merged) | T1 | $\rho_{\mathrm{hf}}=1.57$ at matched parameters | [[leaves/P4.1/cards/R3iii]] | L33 |
| R4 | P4.1 | $c$-local rank: a product over patches of per-patch component sums | T0 | collapses to one determinant, or an exponential sum | [[leaves/P4.1/cards/R4]] | L11 |
| R5 | P4.1 | rank growing polynomially with $N$ from translated local defects | G4 | CIS-like, gain at most $\mathcal O(\sqrt N)$ | [[leaves/P4.1/cards/R5]] | L10 |
| R7 | P4.1 | size-independent pairing with zero-shot size transfer (`pcg_tr`) | G0 | inherits P4.2's verdict (size-independent families plateau) | [[leaves/P4.1/cards/R7]] | L30 |
| R8 | P4.1 | R2 pairing frame with the hidden-fermion rational head (`pcg_hf`) | T1 | its rational lift lost (R3iii) | [[leaves/P4.1/cards/R8]] | L33 |
| R9 | P4.1 | R2 with translation projection on cylinders | G4 | translation projection gain vanishes per site | [[leaves/P4.1/cards/R9]] | L10 |
| R10 | P4.1 | R2 with a size-extensive spin Jastrow | G1 | exponential Hubbard–Stratonovich sum | [[leaves/P4.1/cards/R10]] | L11 |
| R11 | P4.1 | valence-bond amplitude-product states with mobile holes | G3 | needs the sign-free regime the Invariants exclude | [[leaves/P4.1/cards/R11]] | route-specific |
| R12 | P4.1 | one Lanczos step on an exact-$S$ state | G4 | one Lanczos step: finite-size gain | [[leaves/P4.1/cards/R12]] | L10 |
| R13 | P4.1 | SU(2)-invariant plaquette exchange correlators | G1 | exponential sum, or breaks $S$ | [[leaves/P4.1/cards/R13]] | L11 |
| R14 | P4.1 | Shiba-dual frame: exact $S$ as exact $\eta$-spin of a determinant | G0 | no new function class | [[leaves/P4.1/cards/R14]] | route-specific |
| R15 | P4.1 | SPRING momentum on sample-space SR for the exact-$S$ arms (`spring`) | T1 | SPRING 2.67 times worse at equal wall time | [[leaves/P4.1/cards/R15]] | route-specific |
| R15b | P4.1 | four times the samples per SR step at equal wall time (protocol premise O, `ns-rich`) | T2 | more samples did not lower the 24-site energy | [[leaves/P4.1/cards/R15b]] | route-specific |
| R15c | P4.1 | warm start of the 24-site state by tiling the 12-site one | T0 | architecture mismatch; the mechanism does not apply | [[leaves/P4.1/cards/R15c]] | route-specific |
| R15d | P4.1 | sampled linear method (Toulouse–Umrigar) for the exact-$S$ family | T0 | 2–3 times the local-energy cost; unstable when sample-starved | [[leaves/P4.1/cards/R15d]] | L29 |
| RG1 | P4.1 | several pairing references with exact $S$ (`pcg_kK`) | G4 | fixed-rank superposition, $o(N)$ gain | [[leaves/P4.1/cards/RG1]] | L10 |
| RG2 | P4.1 | R2 with a spin-projected Jastrow that stays Gaussian per charge configuration | G4 | finite-size gain (R1's argument) | [[leaves/P4.1/cards/RG2]] | L10 |
| RG3 | P4.1 | non-collinear (generalised HFB) frame under full SU(2) projection | G4 | far over the cost gate | [[leaves/P4.1/cards/RG3]] | route-specific |
| RG4 | P4.1 | best of $K$ seeds at equal compute | G4 | not a method; gain vanishes per site | [[leaves/P4.1/cards/RG4]] | route-specific |
| RG5 | P4.1 | train unprojected, project at the end | G0 | standard practice; does not address the limiter | [[leaves/P4.1/cards/RG5]] | L1 |
| R18 | P4.1 | block linear method on the slack blocks, unshifted (`blm`) | T1 | step blows up ($R=-244$) | [[leaves/P4.1/cards/R18]] | L29 |
| R18' | P4.1 | block linear method with a level shift | T1 | shift at its ceiling, $R=10^{-4}$ | [[leaves/P4.1/cards/R18p]] | L29 |
| R23 | P4.1 | frame averaging over a finite subgroup of SU(2) (`frame-avg`) | G1 | inexact ($S'\ge4$ admixtures) and, beyond spin flips, every group element is an exponential orbit sum | [[leaves/P4.1/cards/R23]] | L56 |
| R24 | P4.1 | spin-flip averaging plus a Löwdin filter for the residual spins | G1 | the filter needs about $N_s/4$ factors of cost $\mathcal O(N_s^2)$ each; truncated, it is R5.2 | [[leaves/P4.1/cards/R24]] | L56, L10 |
| R25 | P4.1 | projected coupled cluster via an SU(2)-scalar bond Hubbard–Stratonovich decoupling (`pcc-hs`) | G1 | exact per field, but the field integral is exponential (sign problem) or a fixed-rank Gaussian sum | [[leaves/P4.1/cards/R25]] | L11 |
| R26 | P4.1 | charge variables in a dimer-orbital basis (`bond-basis`) | G2 | the site Gutzwiller factor becomes non-diagonal, so the leading strong-coupling correlator is lost or exponential | [[leaves/P4.1/cards/R26]] | route-specific |
| R27 | P4.1 | full-sum L-BFGS with exact gradients (`lbfgs`) | T1 | 0.13-0.17x the ladder LM's gain per CPU-min at the polished endpoints | [[leaves/P4.1/cards/R27]] | route-specific (proposed entry, round 22) |
| R28 | P4.1 | metric-whitened L-BFGS with the exact metric refreshed every k gradient steps (`plbfgs` hybrid) | T0 | build costs about 14 gradient calls; every refresh schedule is bounded at 1.31-1.79x the ladder rate (3x needed) from round 22's own trajectory | [[leaves/P4.1/cards/R28]] | route-specific (proposed entry, round 24) |
| R29 | P4.1 | truncated-Newton trust region with exact-gradient Hessian-vector products (`ncg`) | T1 | 0.05x the ladder LM's rate on `che` | [[leaves/P4.1/cards/R29]] | route-specific (as R27) |
| R30 | P4.1 | Anderson/DIIS mixing over ladder-LM iterates (`anderson`) | T1 | 0.80x the ladder LM rate on `che`; every Anderson candidate lost to the plain step (non-smooth map: line-search factor jumps 1/4 to 1) | [[leaves/P4.1/cards/R30]] | route-specific (as R27) |
| R31 | P4.1 | Krylov-subspace or larger-block linear method (`krylov-lm`) | G2 | the build, not the solve, sets the step cost, and the model solved is the same | [[leaves/P4.1/cards/R31]] | route-specific |
| R32 | P4.1 | two-stage blocked SR step (`bsr`) | T1 | blocked Rbar -0.002 and -0.007 vs base 0.016, below the KILL line base + 0.02 | [[leaves/P4.1/cards/R32]] | L59 |
| R33 | P4.1 | minSR / kernel-trick SR (`minsr`) | G2 | push-through identity: same samples, same step; only the solve cost changes | [[leaves/P4.1/cards/R33]] | route-specific |
| R35 | P4.1 | RLS/Kalman running metric (`rls-S`) | T1 | a fourfold staleness-free metric average gave Rbar 0.003 vs 0.016 | [[leaves/P4.1/cards/R35]] | as R32 |
| R36 | P4.1 | Ledoit-Wolf / NERCOME / factor-model metric (`shrink`) | T1 | Rbar 0.016, 0.021, -0.011 vs base 0.016, below the KILL line base + 0.02 | [[leaves/P4.1/cards/R36]] | L59 |
| R37 | P4.1 | Hamiltonian-symmetry control variate for the force (`cv`) | T1 | variance down 3.9-6.7x against a 32.6x cost; steps worse at equal samples | [[leaves/P4.1/cards/R37]] | as R32 |
| R2 | P4.1 | exactly projected charge-backflowed pairing Gaussian (`pcg`) | SHELVED | 12-site pairing gain over its own rank-5 ablation 0.623 at the predicted-gain stop (credit line 0.60): mechanism unattributed (job 759446); no 24-site optimiser reaches polished states | [[leaves/P4.1/cards/R2]] | L33 |
| R16 | P4.1 | squeezed (spin–charge separated) frame term (`sq`) | T1 | gain at most 0.0097 of the gap at both polished parents, far below the 0.05 PASS bar (route vs foil 0.0064 vs 0.0067, 0.0097 vs 0.0127; foil had more directions) | [[leaves/P4.1/cards/R16]] | route-specific |
| R17 | P4.1 | plaquette charge-pattern Jastrow (`nj`) | SHELVED | three AMBIGUOUS tangent reads, at most 1.3% of the gap; training tier (about 2.5-3 CPU-h) G2-refused (p 0.9 stays below 0.05 of the gap) | [[leaves/P4.1/cards/R17]] | route-specific |
| R19 | P4.1 | cutoff-ladder re-polish of both arms (`ladder`) | SHELVED | unfroze both arms but did not converge them (tails 22-100% of error); SHELVED pending: the 13 CPU-h resume needs the compute auditor; a cheaper decisive version (rank-5 arm alone, ~1.4 CPU-h) is being costed | [[leaves/P4.1/cards/R19]] | L29 |
| R20 | P4.1 | rank-5 truncation of `pcg`'s pairing as a same-code ablation (`abl-rank`) | SHELVED | test complete: ABL-CONFOUNDED, ratio 0.623 between the 0.60 credit and 0.85 retraction lines (job 759446); no further step proposed | [[leaves/P4.1/cards/R20]] | L38 |
| R21 | P4.1 | sampled SR with truncated metric and held-out step length (`trSR`) | G2 | dominated by R38 at equal cost per step: R(15) 0.364 vs 0.195; truncation no better than the shift | [[leaves/P4.1/cards/R21]] | L59 |
| R22 | P4.1 | balanced coupling-tree spin basis with a charge determinant (`spin-tree`) | SHELVED | seed 0 at 52.6x the polished control's error; seeds 1-2 G2-refused (predicted KILL p 0.99); sign head did not learn the 48%-negative tree-basis sign | [[leaves/P4.1/cards/R22]] | route-specific |
| R38 | P4.1 | guided (flattened) sampling for the SR step (`guided`) | T1 | median R(60) 0.512 < 0.60 against 0.7475 for the exact polish; flat last steps (noise floor) | [[leaves/P4.1/cards/R38]] | proposed entry (guided sampling plateaus at half the gap) |
| R34 | P4.1 | importance-reweighted sample reuse across SR steps (`reuse`) | G4 | at 24 sites the reused row still needs its local energy, 85-90% of the step: (sampling + row)/row = 1.05-1.08 for `che`/`pcg`, rule needs 2 | [[leaves/P4.1/cards/R34]] | route-specific |
| R39 | P4.1 | James-Stein shrinkage of the sampled SR force (`js-force`, translation repair of R36) | T1 | mode-wise and grouped shrinkage at or below the plain sampled force (rule: base + 0.02); most force modes have SNR below one | [[leaves/P4.1/cards/R39]] | proposed entry (force shrinkage cannot replace samples) |
| R40 | P4.1 | same NNBF reading charges (`che`) vs spins (`nnbf_SG`), both exactly projected and ladder-polished, 12 sites (`che-vs-spin`) | SHELVED | compute auditor REFUSED (11.0 CPU-h): seed-robust LOSS p about 0.1, AMBIGUOUS likeliest, NO-LOSS would not carry to 24+ sites; pretraining cost unmeasured (~440 > 360 CPU-min cap); inconsistent seed margin | [[leaves/P4.1/cards/R40]] | route-specific |
| R41 | P4.1 | Anderson mixing on the raw fixed-cutoff linear-method direction (`anderson-raw`, translation repair of R30) | T1 | rate 0.83 of the plain linear method (rule: kill below 1.5); the secant model overshoots a step already cut back by the line search | [[leaves/P4.1/cards/R41]] | proposed entry (Anderson accelerates a contraction, not a curvature-limited step) |
| R44 | P4.1 | same NNBF reading charges vs spins, both exactly projected, converged at 16 sites (`conv-16`) | G4 | sampled Hubbard 4x4 costs ~18 CPU-h/step (spin-orbit sums); affordable form is exact-sum (t-J 4x4), says nothing beyond 16 sites (L39) | [[leaves/P4.1/cards/R44]] | proposed entry (exactly projected spin-reading comparisons are ED-bound) |
| R46 | P4.1 | S^2-penalised spin-reading network as a near-feasible point of the projected family, 24 sites (`pen-bound-24`) | G2 | one-sided test whose informative side (LOSS detected) is predicted at p ~0.1 from R42; reopenable on R43 | [[leaves/P4.1/cards/R46]] | route-specific |
| R48 | P4.1 | backflow fed SU(2)-scalar moments of its own charge-conditioned Gaussian (`moment-feat`) | G2 | features are functions of (c, theta): contained in a wider charge network, no new information | [[leaves/P4.1/cards/R48]] | L11 |
| R49 | P4.1 | angle-dependent Gaussians inside the spin-projection integral (`angle-gcm`) | G1 | the integral is S-pure only for an angle-independent state; exact sub-cases are `che` and R1 | [[leaves/P4.1/cards/R49]] | route-specific |

### P4.2 — Size-transferable fermionic NQS

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | P4.2 | Residual-geminal transfer (Route 1) | G3 | gapped-ladder test; residual collapses to Jastrow scalars ($M=1.30$) | [[leaves/P4.2/cards/R1]] | route-specific |
| R2 | P4.2 | Momentum-labelled equivariant orbital generator (Route 2) | G2 | Bloch: only the occupied set transfers, equal to the foil | [[leaves/P4.2/cards/R2]] | route-specific |
| R3 | P4.2 | Unit-cell gluing (Route 3) | G1 | zero pair correlation across the cut | [[leaves/P4.2/cards/R3]] | route-specific |
| R4 | P4.2 | Twist-folded transfer (Route 4) | G0 | Gerard et al. relabel | [[leaves/P4.2/cards/R4]] | route-specific |
| R2.1 | P4.2 | Ring-surgery continuation in a seam coupling λ (Route 2.1) | G0 | lattice GFMC (killed in P1.1) or a Hamiltonian warm start | [[leaves/P4.2/cards/R2.1]] | L18 |
| R2.3 | P4.2 | Ensemble-disagreement probing of the transferred sign (Route 2.3) | T1 | recall 0.345 at cost 0.394 | [[leaves/P4.2/cards/R2.3]] | route-specific |
| R2.4 | P4.2 | Multigrid or RG prolongation of NQS parameters (Route 2.4) | G1 | rescaling changes the density | [[leaves/P4.2/cards/R2.4]] | route-specific |
| R2.5 | P4.2 | MERA-style block-and-refine (Route 2.5) | G1 | 9 against 3 local states | [[leaves/P4.2/cards/R2.5]] | route-specific |
| R3.0 | P4.2 | MFNG as submitted: single 4x2 source, seed picked by source energy (Route 3.0) | CRITIC | the seed, not the source, decides the transferred state | [[leaves/P4.2/cards/R3.0]] | L43 |
| R3.1a | P4.2 | Twist-complete 4x2 source with y-phase 0 and π (Route 3.1a) | G1 | odd-per-leg configurations are zero | [[leaves/P4.2/cards/R3.1a]] | route-specific |
| R3.1 | P4.2 | Grid-complete, minimum-norm, irrep-projected transfer, TCMN (Route 3.1) | T2 | zero-shot state worse than the mean-field start; behind scratch | [[leaves/P4.2/cards/R3.1]] | L43 |
| R3.2 | P4.2 | Frozen-sign transfer, FST (Route 3.2) | CRITIC | relabel; transferred sign worse than mean field | [[leaves/P4.2/cards/R3.2]] | L43 |
| R3.3 | P4.2 | Direct lattice port of Gerard et al. (Route 3.3) | G0 | relabel, no sector fix | [[leaves/P4.2/cards/R3.3]] | route-specific |
| R3.4 | P4.2 | Averaging parameters over source seeds (Route 3.4) | G1 | seeds sit in different basins | [[leaves/P4.2/cards/R3.4]] | route-specific |
| R5.a | P4.2 | Transfer only the source-identified one- and two-body kernels (M5.a) | G2 | the identified part is the foil's refit | [[leaves/P4.2/cards/R5.a]] | L43 |
| R5.b | P4.2 | Linear source-null filter (M5.b) | G1 | destroys the source fit | [[leaves/P4.2/cards/R5.b]] | L43 |
| R5.c | P4.2 | Copy the projected NNBF of P2.1 through a site embedding (M5.c) | G1 | site-indexed, no size map | [[leaves/P4.2/cards/R5.c]] | route-specific |
| R5.1 | P4.2 | Prior-anchored transfer, PA (R5.1) | T2 | family plateau 5.7% against NNBF 0.56% | [[leaves/P4.2/cards/R5.1]] | L30 |
| R5.2 | P4.2 | Environment-library local transfer, ELT (R5.2) | G0 | routine VMC warm start | [[leaves/P4.2/cards/R5.2]] | route-specific |
| R5.3 | P4.2 | Enlarge the size-independent family (critic lever "enlarge") | T2 | 5.97% to 5.71%; tangent rank 34 to 48 | [[leaves/P4.2/cards/R5.3]] | L30 |
| R5.4 | P4.2 | Spin features and two-body Jastrows on the size-independent family (critic ablation) | T1 | at most 1.10 times gain | [[leaves/P4.2/cards/R5.4]] | route-specific |
| R5.5 | P4.2 | Boundary-class pair kernel bnd-f 3 (critic's size-independent substitute) | T1 | 4.41% against the 0.85% gate | [[leaves/P4.2/cards/R5.5]] | L33 |
| R5.6 | P4.2 | Adopt a weight-shared CNN or transformer orbital generator (critic §3) | G0 | published practice | [[leaves/P4.2/cards/R5.6]] | route-specific |
| R9.1 | P4.2 | Boundary-layer kernel with edge depth and orientation (R9.1) | T1 | `fsym` 3.62% against base 1.91% | [[leaves/P4.2/cards/R9.1]] | L33 |
| R9.2 | P4.2 | Spectral graph-filter pair kernel of the hopping matrix (R9.2) | T1 | 1.02 times gain | [[leaves/P4.2/cards/R9.2]] | route-specific |
| R9.3 | P4.2 | Global configuration view at constant parameter count (R9.3) | T0 | published, or vanishes per site | [[leaves/P4.2/cards/R9.3]] | route-specific |
| R9.4 | P4.2 | Environment-complete sources (R9.4) | T0 | dominated by the target's scratch plateau | [[leaves/P4.2/cards/R9.4]] | L30 |
| R9.5 | P4.2 | Bulk-only transfer on periodic targets (R9.5) | SHELVED | GATE-REFUSED (G2): 451914 bounds MFNG torus capacity (1.96–2.13%), not MFNG-O; the 9.6 CPU-h torus transfer test was predicted to close R9.5 (FAIL or AMBIGUOUS, $p=0.85$) | [[leaves/P4.2/cards/R9.5]] | L30 |
| R9.6 | P4.2 | Transferred state as initialiser of the site-indexed NNBF (R9.6) | T0 | head start worth a speed-up of about 1 | [[leaves/P4.2/cards/R9.6]] | L43 |
| R9.7 | P4.2 | Projected period-2 patterned kernel (R9.7) | T0 | finite-size gain (Gate 4) | [[leaves/P4.2/cards/R9.7]] | L10 |

### P5.1 — Optimising fermionic NQS to their representable minimum

| Route | Leaf | Idea | Code | Why | Card | Library |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R51.2 | P5.1 | randomised Nystrom / sketched low-rank metric for exact-sum polishing (`sketch`) | T1 | Jacobian rows alone cost 1.14 CPU-min against the 0.33 needed; needed rank over 1200 of 2637 saves nothing on the Gram | [[leaves/P5.1/cards/R51.2]] | proposed entry (metric cost is the Jacobian) |
| R51.4 | P5.1 | semistochastic metric: exact top-weight rows plus sampled tail (`semistoch`) | G4 | top-weight share vanishes exponentially with volume; finite-size bonus | [[leaves/P5.1/cards/R51.4]] | route-specific |
| R51.6 | P5.1 | stochastic-trust-region acceptance of held-out steps (`storm`) | T1 | held-out error (4.6e-4) exceeds one step's gain (2.3e-4): the 2-sigma test never fires | [[leaves/P5.1/cards/R51.6]] | route-specific (proposed entry "selection makes the floor" withdrawn by the per-leaf critic, 2026-11-17) |
| R51.7 | P5.1 | Robbins-Monro step decay with Polyak-Ruppert averaging (`rmavg`) | SHELVED | its T2 kill (halving the step removed 14% of the shortfall, 25% needed) tested it while every run was still descending, outside its premise; unshelve when a fixed-step run levels off (last-20 slope below 2e-4); trigger met at 10 sites by R51.8's exact arm (2026-11-17) | [[leaves/P5.1/cards/R51.7]] | — |
| R51.5 | P5.1 | median-of-means SR step: geometric median of SR steps solved on K sample blocks (`mom`) | T1 | premise arm (4096-sample fixed-step guided SR, 4x3, 240 steps, 769669): R(240) 0.495 passed (line 0.247), but the energy rose on 83 of 240 steps against the line of 80 (binomial sd about 7), so block steps are too often uphill for a median of 4 | [[leaves/P5.1/cards/R51.5]] | route-specific |
| R51.9 | P5.1 | one-step direction-fidelity proxy for the size scaling of the sample need (`phiscale`) | T0 | one-step quantities miss the multi-step need by 2-4x in both directions at the one size calibrated | [[leaves/P5.1/cards/R51.9]] | proposed entry (one-step rates overstate sustained rates) |
| R51.12 | P5.1 | sketched ridge-leverage row selection for the sampled metric (`sketchlev`) | T0 | every drawn row feeds the force, so only the Gram can be saved (about 10-15% of a step), and the leverage estimate costs as much | [[leaves/P5.1/cards/R51.12]] | proposed entry (metric row subsampling pays only where rows serve the metric alone) |
| R51.13 | P5.1 | SVRG/STORM anchor control variate on the SR direction (`svrg`) | T1 | at mixing weight a = 0.1 only: STORM at equal CPU R_plat 0.211 vs plain 0.274 (5x2); a = 0.5 untried (linear theory predicts a gain) | [[leaves/P5.1/cards/R51.13]] | route-specific (R51.18's rerun also KILLED, 2026-11-17; a = 0.5 untried) |
| R51.18 | P5.1 | STORM recursion on the metric and force, then solve (`storm-sf`) | T1 | rerun, length in the fresh sample metric: R_plat -3.47 vs plain equal-CPU 0.274 (5x2, N = 2048, a = 0.1), energy rose on 82 of 120 steps; recursed metric indefinite every step; exact/sample length ratio median 1.13, so the direction, not the length, failed (this length rule only) | [[leaves/P5.1/cards/R51.18]] | L70 (with R51.13: STORM recursion with this length rule fails at a = 0.1 below P unique draws) |
| R51.14 | P5.1 | importance-reweighted reuse of high-leverage configurations (`levreuse`) | T0 | weights of old near-node draws explode ($\chi^2$ 1.7e4 at 20 steps); reuse over 1-5 steps under slow Markov chains stays open | [[leaves/P5.1/cards/R51.14]] | route-specific |
| R51.17 | P5.1 | without-replacement pool reshuffling across steps (`reshuffle`) | T0 | accumulated error equals fresh draws at first order (no finite-sum target); weights add variance | [[leaves/P5.1/cards/R51.17]] | proposed entry (reshuffling gains only relative to a finite pool) |
| R51.15 | P5.1 | samples-per-step schedule grown on the measured noise scale (`adaptN`) | T1 | censoring artefact (rule on the card and portfolio row, written before 769213): fixed 16,384 continued 332 steps reached 0.90 / 0.95 R_pol at 4.26M / 6.67M draws, below the oracle schedule envelope (4.8M / 12.1M) built from 120-step arms | [[leaves/P5.1/cards/R51.15]] | proposed entry (continue the smallest-N arm before crediting a schedule) |

---

