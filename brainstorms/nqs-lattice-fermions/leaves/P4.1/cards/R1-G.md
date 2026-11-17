# R1-G — R1 generality on two more 12-site instances (job 637776) (SHELVED)
* **The idea**: Check that R1's 12-site PASS is not specific to one instance by rerunning the rank ladder on the $4\times3$ cylinder (periodic along $x$, $5+5$) and on the open $4\times3$ at $4+4$, with a pre-registered PROMOTE or KILL-GENERALITY rule.
* **Lineage**: the pre-committed action of the 508440 PASS.
* **Outcome**: SHELVED (cancelled by the second queue review, 2026-11-16, after 6 of 32 tasks finished and about 1,424 CPU-hours).
* **Why shelved**: R1 had become a 12-site result whose failure to scale is predicted by argument, so whether its 12-site gain holds on two other 12-site instances changes no decision; the cap was about 41,000 CPU-hours. Descriptive read of the six finished tasks (cylinder, L-BFGS 2000 iterations): `che` $\epsilon=1.15$, 1.22 and $1.27\times10^{-5}$; `che_k2` 1.29 and $0.95\times10^{-5}$; `nnbf_SG` seed 0 $0.90\times10^{-5}$. The cylinder is near-exact for every arm, and its `che` to `nnbf_SG` ratio (about 1.35) is below the 1.5 premise, so it would probably have returned NO-TARGET.
* **What would unshelve it**: A reason to make a cluster-scale claim for R1 (a paper that uses `che_kK` as a sector-resolved benchmark tool), on an instance with a resolved `che` to `nnbf_SG` gap.

