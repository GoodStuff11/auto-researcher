# Review: Linting and Critics

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

## Evaluation Protocol (Adversarial Critic Loop)

After drafting, lint every Markdown file in the working directory (front page, leaf files, cards, library, kill index and notes):

```bash
python3 check_brainstorm.py <working-dir>
```

Resolve all syntax and formatting failures flagged by the linter until the summary line reports `0 with errors` and no file prints a warning. Re-run it after every edit pass, including after the critic loop and after appending test results. A single file can still be linted with `python3 check_brainstorm.py <file>.md`.

**Scheduling (Mode B)**: critics are pipelined too (Mode B step 0).
- Spawn a **per-leaf critic** as soon as that leaf's artifact returns and lints clean, or a critic for a small batch of leaves that return together. Do not wait for the whole fan-out.
- Point the per-leaf critic at that leaf's artifact and test code, plus the master Invariants, the Kill Index and the findings library. Have it read the test code itself, not only the markdown. The critic signs each kill it has checked in that card's **Checked by** field.
- Each critic's objections go straight to that leaf's fix or mutation step.
- A single **cross-node critic** closes each round, once the last leaf of the round has passed its own critic. It checks only claims that span leaves: shared baselines, shared instances, corrections that must propagate, and the findings library. For every new or changed library entry, it asks whether the evidence supports the generalisation and whether the scope is honest, and it alone can set an entry to ESTABLISHED. It never holds back per-leaf progress.

Spawn a fresh adversarial critic agent with this mandate:
> The full mandate is in `briefs/critic.md`; give the critic that brief and its `bs.py context`.

Address all major critiques in place. Log every action in the Revision & Critic Log in `history.md`.
