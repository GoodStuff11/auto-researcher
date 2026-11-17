# R3iii — hidden-fermion rational head with a charge-only hidden network (`che_hf`; Route 6 merged) (KILLED)
* **The idea**: Add $M$ hidden rows that depend on $c$; the Schur complement gives a charge Jastrow $\det D(c)$ times an effective orbital matrix $P_a-P_bD(c)^{-1}C(c)$ that is rational, not affine, in the hidden vector. The set of orbital deformations then escapes the $H$-dimensional affine bound while staying Gaussian per $c$ and exactly projectable.
* **Lineage**: R6 (Operation B.3 from hidden-fermion determinant states), merged into R3 by the round-7 critic; Round 8 T1. Its pairing-frame copy is R8.
* **Outcome**: KILLED at T1 (Round 8, 2026-11-16).
* **Kill rule**: "KILL if $\epsilon_{\mathrm{hf}}\ge\epsilon_{\mathrm{che}}$ or $\epsilon_{\mathrm{hf}}\ge\epsilon_{\mathrm r1}$" (NO-TARGET if width does not bind), compared at the largest common logged iteration (Amendment 1).
* **Evidence**: Open $5\times2$ strip, $N_e=8$, exact-sum L-BFGS, seed 0. At iteration 100: `che` ($H=4$, 620 parameters) $1.12\times10^{-3}$, `che_hf` (620) $1.76\times10^{-3}$, `r1` (594) $3.27\times10^{-3}$, `che8` ($H=8$, 1024) $5.42\times10^{-4}$. So $\rho_{\mathrm{hf}}=1.57$ at equal iterations and 2.13 at equal CPU, while width binds ($\epsilon_{\mathrm{che8}}/\epsilon_{\mathrm{che}}=0.48$).
* **Why this kills it**: There is a target, and at matched parameters the rational head is worse than the affine one.
* **What the kill does not rule out**: A rational head added on top of a full-width affine head (more parameters).
* **Strongest case against the kill**: One seed, at most 219 iterations, capped runs on one toy; the rational head has a harder landscape, and Round 11 showed this leaf's comparisons are optimiser-sensitive, so a polished `che_hf` might not lose. Unresolved.
* **Checked by**: not yet reviewed by a critic (worker T1, 2026-11-16).
* **Lesson**: L33 (`library.md`)

