# R1-L — R1 size ladder at 12 to 48 sites (job 686990 with DMRG 687302) (SHELVED)
* **The idea**: Measure whether the R1 rank gain survives from 12 to 48 sites on open $L_x\times3$ strips at filling $5/6$, with `che`, `che_k2`, `che_k4`, `che_k8` and a plain spin-resolved NNBF at equal compute, all trained by VMC-SR with $P_G$, against sector-resolved DMRG references, under a pre-registered SCALES or DOES-NOT-SCALE rule.
* **Lineage**: R1's pre-committed scaling step (2026-11-16); it superseded 685310 and 683137.
* **Outcome**: SHELVED (cancelled before start by the Compute Value Gate queue review, 2026-11-16).
* **Why shelved**: About 850 task-hours of 16 CPUs (roughly 13,000 CPU-hours; cap 107,520) to measure an outcome the T0 Gate 4 argument predicts (DOES-NOT-SCALE), with single-seed comparisons weaker than the measured 24-site run-to-run spread (about 2.3 times the per-run error). Later evidence makes it uninterpretable as well: it trained every arm by SR, which Round 11 found optimiser-limited.
* **What would unshelve it**: A sampled optimiser that realises the polished gain (rung 1b of Round 11 passing), plus a result that contradicts the T0 argument; even then the cheapest falsifier is a single 24-site rung (`che_k8` against `che`, two seeds, about 200 CPU-hours).

