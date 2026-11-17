# RG2 — R2 with a spin-projected Jastrow that stays Gaussian per charge configuration (KILLED)
* **The idea**: Find a spin Jastrow that keeps each charge block Gaussian, so R2 gains spin correlations without losing closed-form projection; the only candidate is a Hubbard–Stratonovich sum over one collective coordinate, whose terms are one-body operators.
* **Lineage**: orchestrator suggestion (a), Round 10 (G2).
* **Outcome**: KILLED at G4 (T0, Round 10).
* **Kill rule**: Gate 4.
* **Evidence**: $e^{v(\sum_ia_iS^z_i)^2}=\sum_qw_qe^{\phi_q\sum_ia_iS^z_i}$ is a fixed-rank mixture over one collective mode, which gains $\mathcal O(1)$ (R1's argument).
* **Why this kills it**: Finite-size gain.
* **What the kill does not rule out**: A charge-only extensive Jastrow (R17).
* **Strongest case against the kill**: None.
* **Checked by**: not yet reviewed by a critic (Round 10 worker gate, 2026-11-16).
* **Lesson**: L10 (`library.md`)

