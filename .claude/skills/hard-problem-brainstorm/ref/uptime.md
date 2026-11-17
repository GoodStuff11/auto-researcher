# Uptime

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

### 3.5.9 Uptime: Waiting, Resuming, Crashes and Finishing

The procedure must keep moving without supervision, survive the loss of its own context, and stop cleanly when there is nothing left worth doing.

* **Waiting on jobs.** Never sleep-poll in the foreground. Use one background watcher that wakes the agent when any brainstorm job changes state (for example a monitor on `squeue --me`, or a background loop that exits on a state change), with a poll interval matched to the job: minutes for T1 and T2, tens of minutes for T3. The thinking queue keeps running while the watcher waits.
* **Write state to disk as you go.** Every decision is written to the leaf or the master in the turn it is made, including the pre-planned decision tree of every job in flight (3.5.3, step 1). The front page's "Current leaf state" table lists every agent in flight (worker, critic, auditor) with its task, so it can be respawned. Nothing the run needs lives only in the conversation.
* **Resuming** (a new session, a context compaction, or a restarted orchestrator), in this order:
  1. `compute_gate.py reconcile`, then `refresh`; `record` every finished job and apply its pre-planned next action. An unledgered job is cancelled with `compute_gate.py cancel` and resubmitted through the gate if still wanted;
  2. re-arm the background watcher on the jobs still queued;
  3. run `bs.py state` and read the front page's "Current leaf state" table; open a leaf's `summary.md` (`bs.py show P<id>`) only when acting on it, never the whole leaf;
  4. respawn every agent the "Current leaf state" table shows in flight whose result is not on disk;
  5. rebuild the thinking queue from 3.5.3, step 3, and continue.
* **Crashes.** A job that ends FAILED, OUT_OF_MEMORY, NODE_FAIL or BOOT_FAIL gave no verdict: record it with `--changed crash`. Crashes do not count toward the no-change streak. One plain resubmission is allowed (for example after a node failure). After two crashes in a row, the gate refuses the script until the crash is reproduced at T1 with the pre-flight dry run, fixed, and the fix stated in a `fixed:` line. The fix must be new: a fix already tried by a run that crashed does not count. An array job in which only some tasks failed is recorded by its verdict, not as a crash. A TIMEOUT is not a crash but a wall-time miss: re-measure the time per step and resubmit only what the last checkpoint did not cover, as a gated resume.
* **Submission environment.** `compute_gate.py submit` strips inherited `SLURM_*` and `SBATCH_*` variables before calling `sbatch`, so a job never inherits the orchestrator's own allocation and environment overrides cannot change the cost the gate computed. Anything run with `srun` from inside an allocation needs the same care.
* **Finishing the run.** The run is over when every leaf is in a terminal state, or is DORMANT with no open reopening-ledger row the agent can work on (only `GATE-REFUSED` rows waiting on the user's budget or approval); by 3b, such a leaf has already taken exit 3. Then, in one pass: `refresh` and `report`, the cross-node critic (including a final review of every library entry's status and scope), a final master Summary, a whole-directory lint, and a report to the user. The final Summary opens with the problem's top-level question and, for every leaf, its Final Tree question and its plain-words `Answer` (as in "The master Summary" below), then covers what each leaf established, every leaf's state, the library's index (the lessons that reach beyond this problem), compute spent against budget, and what the user could fund next: the refused steps with their costs. Do not idle with an empty queue, and do not spend remaining budget just because it is there.

---
