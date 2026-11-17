# RG3 — non-collinear (generalised HFB) frame under full SU(2) projection (KILLED)
* **The idea**: Allow pairing that also breaks $S^z$ (spiral textures), as projected generalised Hartree–Fock does in chemistry, under full SU(2) projection with charge backflow, to widen R2's frame.
* **Lineage**: Operation B.1, Round 10 (G3).
* **Outcome**: KILLED at G4 (T0 cost, Round 10).
* **Kill rule**: Gate 4 cost, with the leaf's cost gate of 5 times `che`.
* **Evidence**: The $S^z$ projection ($N_e+1$ nodes) sits inside the $\beta$ quadrature (about $N_e/2+1$ nodes): $41\times21=861$ Pfaffians of size 40 per amplitude at 48 sites, against 11 for `pcg`, about 78 times `pcg` and 280 times `che`.
* **Why this kills it**: Far over the cost gate.
* **What the kill does not rule out**: A non-collinear frame at a fixed small size as a diagnostic.
* **Strongest case against the kill**: None.
* **Checked by**: not yet reviewed by a critic (Round 10 worker gate, 2026-11-16).
* **Lesson**: route-specific — a node count.

