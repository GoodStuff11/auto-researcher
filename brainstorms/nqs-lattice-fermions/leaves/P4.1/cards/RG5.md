# RG5 — train unprojected, project at the end (KILLED)
* **The idea**: Save the projection cost during training by optimising the unprojected state and applying $P_S$ only at the end (projection after variation), then refining with the projection switched on.
* **Lineage**: Operation B, Round 10 (G5).
* **Outcome**: KILLED at G0 (T0, Round 10).
* **Kill rule**: Gate 0 and the measured evidence.
* **Evidence**: Starting projected HF from the unprojected solution is standard; the per-step saving is only the `che`/`nnbf` cost ratio (2.6 at 24 sites, 2.3 at 48), and more steps of the same optimiser bought little at 24 sites.
* **Why this kills it**: Known, and its saving does not address the limiter.
* **What the kill does not rule out**: Its use as an initialisation for R18''.
* **Strongest case against the kill**: None.
* **Checked by**: not yet reviewed by a critic (Round 10 worker gate, 2026-11-16).
* **Lesson**: L1 (`library.md`)

