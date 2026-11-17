# VOI note (refused, not submitted): P1.3 I1, direct sign-oracle basin probe on the 4×4 instance

This note records a step that the compute gate refused, so that its prediction counts in the G2 calibration (Stage 3.5.8). No script was written and nothing was submitted.

* leaf: P1.3
* route: R7.7 (direct sign-oracle basin probe on the 4×4 OBC $t$–$t'$–$J$ instance; reopening-ledger item I1)
* tier: T3
* decision: BASIN (an oracle-fitted, better-signed state released at widths 4, 6 and 10 stays below the paired cold continuation's sign error by more than the margin) reopens P1.3 on an optimiser route. ORACLE-MISS (the fidelity fit cannot reach a state with clearly better signs) or RELAX (the released state drifts back to the cold level) leaves the leaf EXHAUSTED at this budget with the CAPACITY reading.
* prediction: ORACLE-MISS or RELAX
* p: 0.8
* prediction-changes-decision: no (both predicted outcomes leave the leaf closed; only BASIN, at $p\le0.2$, would change it)
* cheapest-kill-tried: T1 sign-oracle probes on a projected 4×3 $t$–$J$ toy. The first-order SR fit gave ORACLE-MISS (0.0092 against cold 0.0109, threshold 0.0055), and the Levenberg–Marquardt fit with an exact Jacobian gave LM-MISS (median 0.00854 against threshold 0.00545). On the 4×4 instance the oracle arm W of jobs 507179 and 625182 ended at the cold level at every width (0.0189, 0.0115, 0.0070 against 0.0192, 0.0117, 0.0070).
* cost-cpuh: about 250 (9 width-seed pairs, an oracle arm of at most 600 SR steps and a 300-step cold arm, at 12–15 s per step on 8 threads, from the 507179 and 625182 timings)
* cheapest-version: none cheaper carries the question; the basin question lives only on the 4×4 instance, and the toy versions were already run (above).
* refused: G2 (the predicted outcome, $p\ge0.8$, does not change the decision) and G4 (a job of 100 to 1000 CPU-h needs a passed T2 on the route, and R7.7 has none). Recorded from the reviewer triage of 2026-11-17 and confirmed by the per-leaf critic of 2026-11-17.
* what would overturn the prediction cheaply: an LM sign-oracle fit that halves the error on a gapped exact-sum toy with cold error at least $0.005t$, or a projected cold run captured by a same-sector competitor.
