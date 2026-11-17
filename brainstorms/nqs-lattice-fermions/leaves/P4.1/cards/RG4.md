# RG4 — best of $K$ seeds at equal compute (KILLED)
* **The idea**: Since 24-site runs scatter by about 2.5 times their quoted errors, train $K$ seeds and keep the best, spending the compute on seeds instead of steps to exploit the spread.
* **Lineage**: orchestrator suggestion (b), Round 10 (G4).
* **Outcome**: KILLED at G4 (T0, Round 10; also Gate 0).
* **Kill rule**: Gate 0 (already standard) and Gate 4.
* **Evidence**: Best of seeds is in every pre-registered rule of the leaf; its expected gain is $\sigma_{\mathrm{seed}}\sqrt{2\ln K}$, about 0.012 for $K=4$ at 24 sites, 7% of the `che` gap of about 0.17, and it falls per site as $N^{-1/2}$ if the spread comes from independent regions.
* **Why this kills it**: Not a method, and its gain vanishes per site.
* **What the kill does not rule out**: Nothing.
* **Strongest case against the kill**: None.
* **Checked by**: not yet reviewed by a critic (Round 10 worker gate, 2026-11-16).
* **Lesson**: route-specific — an extreme-value estimate.

