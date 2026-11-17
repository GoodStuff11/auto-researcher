# VOI note: P4.2 r10 optimiser falsifier, arm c8h32

* leaf: P4.2
* route: U3 (optimiser falsifier, reopening-ledger row)
* tier: T2
* decision: r* <= 0.77% -> attribution KILLED (DIAGNOSTIC-FALSIFIED): P4.2 reopens on U3, its "size-independent plateau is an ansatz limit" conclusion is withdrawn, and polished training becomes a required control in P4.1/P2.1 comparisons. r* in (0.77%, 1.02%] -> AMBIGUOUS: release the base arm next (cheapest, 0.95 CPU-h). r* > 1.02% -> c8h32 holds; release base and spin_jas2cs (holds needs every arm) to confirm DIAGNOSTIC-HOLDS and close U3.
* prediction: AMBIGUOUS (polish removes part but not 27%+ of c8h32's 1.057% error) — r* in (0.77%, 1.02%]
* p: 0.45
* prediction-changes-decision: yes (AMBIGUOUS releases the next arm; either other outcome closes or reopens U3; no outcome leaves the plan unchanged)
* cheapest-kill-tried: T1 smoke test of the same polish on the saved LM base state (1.778%): zero gain at cutoff 1e-5, i.e. that state is LM-stationary; that cannot decide U3 because the question is about the SR-3000 states, whose theta was never saved (critic_capacity.train writes only options). No T0 argument bounds how far an exact LM polish moves an SR-converged state.
* cost-cpuh: 3.0
* cheapest-version: one arm only, c8h32, the lowest-error arm (1.057%) and the only one where KILL needs under 30% error removal; KILL is decided by any single arm, so it is released first; base and spin_jas2cs follow only if needed. Same 4x3 OBC 5+5 instance as 704844; no smaller instance carries the U3 question.
* convergence: deterministic regeneration of a completed 704844 run (3000 SR steps finished in 6968 s on one thread, energy logged); polish is at most 8 exact LM steps with stop-on-no-gain (smoke test ran in ~10 CPU-s).
* wall-time-basis: 6968 s measured (704844 log, one thread) + ~60 s polish, x1.3 = 2.54 h -> --time 3:00:00
* power: exact sums, deterministic; no statistical error, one seed per the original rule; P(INCOMPLETE) only from a regen mismatch, which regen_sr.py checks against the 704844 energy.
