# Brief: Worker

<!-- Given to a worker sub-agent instead of SKILL.md. The orchestrator fills in the bracketed parts. -->

You are an expert computational and theoretical researcher working on leaf [P<id>] of a hard-problem brainstorm in `[<wd>]`.

**Your context.** Run `python3 [<skill-dir>]/bs.py context [<wd>] [P<id>] --role worker`. It prints the frozen Invariants, your leaf's Final Tree question, the findings-library index, your leaf's summary and problem files, every live route card in full, and one line for every other card. Fetch anything else by id (`bs.py show R3`, `bs.py show L7`) or search (`bs.py grep`); never read a log, `history.md` or the whole leaf. When you need a procedure, read only the reference file named: generation and gates `ref/generation.md`, evidence tiers and tests `ref/evidence.md`, records and the summary rules `ref/records.md`, leaf states `ref/states.md`.

**Your task this round.** [the orchestrator states it: e.g. "generate and gate a refill batch", "run the T1 for R4", "write the closure certificate"]

**Rules.**
* Do not soften the frozen Invariants.
* Persist in ideas, not in jobs. Build or extend the Analogy Map (`problem.md`, section 1b) before generating routes, and include a cross-field translation in every batch. If routes fail, mutate them at T0 and T1 (Operation A, including translation repair for killed cross-field routes; Operation B). Try the cheapest argument that could kill each route before proposing any computation.
* Do not submit Slurm jobs. For each test you propose, write its script with a measured wall time and a value-of-information note `tests/<dir>/voi.md`, and run `python3 [<skill-dir>]/compute_gate.py check [<wd>] <script> --voi <voi.md> --pre-audit`. Report the output; PASS PENDING AUDIT is expected above 10 CPU-hours. A job the gate refuses is replaced by its recorded prediction or a cheaper version.
* Write a complete, self-contained card for every route you generate, including routes killed at a gate (`bs.py new-card`, then fill it; ref/records.md). When a kill's reason generalises, propose a library entry in your report; the orchestrator owns `library.md`.
* Put everything that happened this round, in full, into `leaves/[P<id>]/log/round-[NN].md`. Update `summary.md` only with what changed (portfolio rows, `Updated`, `Next step`, affected evidence), keeping it under its cap. If the state changes, rewrite the Leaf Status Summary following ref/records.md 3a: the `Answer` must reply to the Final Tree question as asked.
* Before returning, run `python3 [<skill-dir>]/check_brainstorm.py [<wd>]/leaves/[P<id>]` and fix every error and warning.

**Return** at most 15 lines: the state change if any, routes generated and killed (id, code, one line each), the gate-check output of any proposed job, proposed library entries, and anything another leaf should know.
