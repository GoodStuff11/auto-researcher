# Records: Documents, Route Cards, Kill Index, Findings Library and Leaf Summaries

<!-- Part of the hard-problem-brainstorm skill. Read when SKILL.md points here; never needed whole. -->

## Documents and context

A run produces far more text than any agent can afford to read: the first run of this procedure reached about 745,000 tokens, with single leaves over 100,000. Every file is therefore a **working file**, read whenever its subject is in play, or an **archive** that is only ever searched. Nothing an agent needs to act on lives only in an archive.

**Size policy: cap fixed-scope records, split growing documents.** A hard cap belongs only on a record whose scope is fixed by design (one route card, one library entry, one index line, a leaf's current-state summary): if such a record grows, something that belongs elsewhere has crept in. A document that legitimately grows as the run grows (the front page's Invariants and Final Tree as leaves are added, a leaf's Analogy Map as barriers and fields are added, the Kill Index) has **no hard cap**: keep it as short as its content allows, and **split it by reader**, so that each agent loads only the part it needs (`bs.py context` assembles that part). If a reader genuinely needs all of a document, its growth is unavoidable and is not a lint failure. Sizes of uncapped working files are reported by `bs.py sizes` as advice, not as warnings.

| File | What it holds | Cap (about tokens) | How it is read |
| :--- | :--- | ---: | :--- |
| `brainstorm-<slug>.md` (front page) | Summary (plain-language answers), Invariants, Final Tree, Stage 1.5 routing, current leaf state, current critical path | none (grows with the leaves; history moves to `history.md`) | by the orchestrator; its Invariants and the leaf's Final Tree bullet go to every worker and critic through `bs.py context` |
| `history.md` | everything historical about the front page: old matrices, round syntheses, the Revision & Critic Log | none (archive) | `bs.py grep` only |
| `kill_index.md` | one row per KILLED or SHELVED route, linking its card | none (table) | `bs.py grep`, or by leaf |
| `library.md` | findings library: an index, then the entries | 700 per entry, 80 per index line | the index by default; entries by id (`bs.py show L7`) |
| `leaves/<id>/summary.md` | title and status line, the live Route Portfolio, the open reopening-ledger rows (3b), the Leaf Status Summary | 2500 | by anyone working on or reading about the leaf |
| `leaves/<id>/problem.md` | formulation, barrier, Analogy Map | none (grows with the Analogy Map; superseded text moves to the leaf's log) | by the leaf's worker and critic |
| `leaves/<id>/cards/<R>.md` | one route card | 900 | live cards by the worker; any card by id |
| `leaves/<id>/log/*.md` | round logs, tier logs, critic reports, pasted outputs | none (archive) | `bs.py grep` only |
| `compute/` | budget and ledger (`compute_gate.py`) | — | `compute_gate.py report` |

In Mode A the same layout is used with a single leaf.

**Reading rules.**
* **Never read an archive whole**, and never read a whole leaf directory. Start from `bs.py state` (one line per leaf) or `bs.py context <leaf> --role <role>` (the minimal set for one agent), and fetch anything else by id: `bs.py show R2.3`, `show L7`, `show P2.1`, `show <job id>`. Consult history with `bs.py grep <pattern>`.
* **Sub-agents get context, not files.** The orchestrator pastes the output of `bs.py context` into a sub-agent's brief (`briefs/`), or tells it the exact command. It never says "read the leaf".
* **The orchestrator reads summaries, not leaves.** It acts on `bs.py state`, returned worker reports (at most 15 lines), tool outputs and the front page.

**Writing rules.**
* **Every fact has one home.** The current state of a leaf lives in its `summary.md`, what an idea is and why it died lives in its card, what a lesson is lives in `library.md`, and what happened when lives in a log. Other files point to the home by id instead of repeating it.
* **Working files are rewritten in place; archives are appended to.** A result goes into the leaf's current log (`log/round-NN.md`) in full, and into the working files only as the update it causes: a card's Outcome, a portfolio row, the summary's `Updated`, `Next step` and affected evidence sentence.
* **Caps are enforced only where they exist.** `check_brainstorm.py` warns when a capped record (leaf summary, route card, library entry or index line) exceeds its cap, and rule 9 requires zero warnings. Uncapped working files get no size warning; `bs.py sizes` reports them so the orchestrator can move history out or split them by reader. `bs.py sizes --over` lists only capped records over their cap.

**Compaction.** When a capped record is over its cap, and in any case at the end of each round for every working file, a compaction agent (`briefs/compactor.md`) rewrites it: a capped record to fit its cap, an uncapped file to remove what is historical or duplicated:
* superseded text moves to the leaf's log (or to `history.md`), with a one-line pointer left behind if a reader would look for it;
* duplicated facts are reduced to their one home plus pointers;
* nothing a working file needs is dropped: every id it mentions still resolves (`bs.py show`), every field the linter requires is still present, and the Question and Answer are unchanged unless the evidence changed.

Compaction never changes a verdict or a state. It is mechanical work for a cheaper model where one is available.

## Ideas, Kills and Lessons: Route Cards, the Kill Index and the Findings Library

The user must be able to read what each idea was, why it was killed, and what the run learned, and decide for themselves whether they agree. Two records serve this, and both are held to the same standard as the Leaf Status Summary (3a): **complete and self-contained.** A reader with domain background who has not followed the run must understand every entry without opening anything else. Every symbol and acronym is defined at first use, no route codename appears without saying what it is, and every number comes with its comparison (arm versus foil, instance, size) and its uncertainty. A pointer to a job id or a section may support a claim, but it never replaces the explanation.

### Route cards (one per idea, live or dead)

Every route ever generated gets a card, including routes killed at a gate within minutes of being proposed, so that every decision can be checked. Each card is its own file, `leaves/<id>/cards/<R>.md`; create it with `bs.py new-card <wd> <leaf> <R> --name "..."`. A card is written when the route is generated and completed in the same turn as its outcome. When the outcome changes, the card is rewritten, not appended to; the history of the route goes to the leaf's log.

```markdown
# R<id> — <short descriptive name, not just a code> (<LIVE | KILLED | SHELVED | MERGED>)
* **The idea**: what the route does, in plain terms; what it changes relative to the Minimal Contrast Foil; and why it was expected to work (the mechanism, not the name). At least 25 words.
* **Lineage**: how it arose (initial batch, Operation A from R<x> after its kill at <code>, Operation B, cross-field translation from <field>, translation repair of R<x>, a reopening-ledger ingredient) and what it later became (mutated into R<y>, merged into R<z>).
* **Source analogy** (cross-field translations and their repairs only): the source field and problem, what corresponds to what, why the technique worked there, the translation gap, and a citation.
* **Outcome**: LIVE (tier reached) | KILLED at <code> (date) | SHELVED (<compute-gate reason>) | MERGED into R<z>.
* **Kill rule**: the gate's question, or the pre-registered rule, quoted as written before the test.
* **Evidence**: the argument or the measurement — instance, size, arm against foil, the numbers side by side with their uncertainties, and the job id or derivation.
* **Why this kills it**: the inference from the evidence to the verdict, step by step.
* **What the kill does not rule out**: its scope (sizes, instances, regimes), and the nearest variant that this evidence would not kill.
* **Strongest case against the kill**: the best argument that the kill is wrong (a confound, an unfair comparison, an untested regime, an optimiser rather than an ansatz effect), and how it was answered, or "unresolved".
* **Checked by**: the critic (per-leaf or cross-node) or auditor who reviewed the kill, and the date.
* **Lesson**: the library entry this kill supports (L<n>), or "route-specific", with one sentence on why it does not generalise.
```

**Source analogy** is required whenever the Lineage says "cross-field" or "translation repair". The last seven fields are required for a KILLED card. A SHELVED card instead has **Why shelved** (which compute-gate rule refused the next step, its capped cost, and the prediction that made it not worth running) and **What would unshelve it**. A LIVE or MERGED card needs only the first three fields. The linter enforces this (R1–R4), and the 900-token cap keeps each card to the argument, with logs and raw numbers in the leaf's log.

**Where a route died: kill codes.** The `KILLED at <code>` in the Outcome uses the procedure's own vocabulary, so a reader knows at once what kind of evidence ended it:

| Code | Killed because | Evidence it rests on |
| :--- | :--- | :--- |
| `G0` | it is known, and the reason it is not standard practice is not fixed (Gate 0) | citations and the adoption barrier they show |
| `G1` | it violates a conservation law, a symmetry, or a complexity bound (Gate 1) | an argument |
| `G2` | a cheap baseline or the foil reaches the same metric (Gate 2) | an argument or a cheap computation |
| `G3` | it works only by removing the Hard Core (Gate 3) | an argument |
| `G4` | its gain or its cost does not survive growing system size (Gate 4) | a scaling argument or the three-size T1 trend |
| `T0` | an analytical argument at tier 0 (lemma, operation count, small effective model) | a derivation |
| `T1`, `T2`, `T3` | it failed its pre-registered rule at that evidence tier | a computation, with its rule |
| `CRITIC` | a critic objection not tied to one gate (for example, its test cannot tell it from the foil) | the objection and the check that confirmed it |

`SHELVED` is not a kill: the idea was not refuted, its next step was refused by the compute gate (G2 predicted outcome, G5 budget, the auditor, or awaiting the user). Readers should weigh a shelved idea as untested.

**The Kill Index.** `kill_index.md` holds one table row per KILLED or SHELVED route, grouped by leaf: route, leaf, the idea in one line, where it died (the code), why in one line, a link to its card (`[[leaves/<leaf>/cards/<R>]]`), and its library entry if any. It is updated in the same turn as each card. It is the reader's map; the cards are the explanations. The linter checks that every row's card exists.

### The findings library (`library.md`)

The library collects what the run learned that is worth knowing beyond the single route or leaf that taught it: mechanisms that explain several outcomes, results about the problem itself, and traps in method that a later project should avoid. It is the main thing the user carries away from a run, so it is organised for reading, not in the order things happened.

**What enters the library.** A result or a kill becomes an entry when at least one of these holds:
* it explains two or more outcomes (routes, leaves or tests) by one mechanism;
* it would change how someone attacks a different problem or uses a different method;
* it is a trap in testing or measurement that was caught (a confound, a control that does not control, a metric that a trivial state games).

A kill whose reason applies only to that route stays on its card ("route-specific").

**Organisation.** The file starts with `## Index`: one line per entry (at most about 80 tokens), giving its number, status and the lesson in one plain sentence. The index is what every worker reads; the entries are fetched by id. Entries are grouped under four headings:
1. `## About the problem itself`: properties of the target (for example, where the error of a method actually sits).
2. `## About methods`: what works or fails, and why. This is also where the general anti-patterns live, the failure mechanisms that killed routes in more than one place, so new routes are generated against them. It also holds what was learned about analogies with other fields: which ones carried over, which did not, and the translation gap that decided it.
3. `## About testing and measurement`: confounds, fair comparisons, controls, metrics.
4. `## About process and compute`: what wasted or saved compute and agent effort.

Each entry uses this form, within about 700 tokens; supporting detail stays on the cards and in the logs it cites:

```markdown
### L<n>. <the lesson, as one complete sentence>
* **Status**: ESTABLISHED | PROVISIONAL | REFUTED (date)
* **Applies to**: the class of problems, methods or tests this lesson reaches beyond this brainstorm.
* **The lesson**: a self-contained explanation — what holds, and the mechanism that makes it hold — written for a reader who has not seen this run.
* **Evidence**: the chain of results that supports it: each comparison, its instance and size, the numbers with uncertainties, and the route cards or leaves it comes from.
* **Scope and limits**: exactly where it was tested and what it does not show.
* **Counter-evidence and open questions**: results that cut against it, and what would overturn it.
* **How to use it**: the concrete consequence for future work (for example, "test X before investing in Y", "always add control Z").
* **Sources**: route cards (leaf and R<id>), job ids, derivations, and literature.
```

**Status rules.**
* **ESTABLISHED** only if the lesson holds on every case it was tested on, within its stated scope, and the cross-node critic has signed off. A lesson that holds on some cases and not others is not hedged as "not universal": its scope is narrowed until it holds everywhere stated, or it stays PROVISIONAL.
* **PROVISIONAL** while the evidence is one instance, one size or one seed, or has not yet been reviewed.
* **REFUTED** entries are kept, with what refuted them under Counter-evidence, so the reader can see how understanding changed.

**Maintenance.** An entry is written or revised in the same turn as the result or kill that teaches it, and the route card's **Lesson** field links to it. Entries are edited in place, never appended to. The cross-node critic reviews every new or changed entry each round: does the evidence support the generalisation, and is the scope honest? Generation of new routes (Stage 3.1, Operation B) reads the library index first. At the end of a run, the final master Summary points to the library's index.

---

#### 3. Leaf Files (Written by Workers under `leaves/<id>/`)

Create a leaf with `bs.py new-leaf <wd> <id> --title "..." --parent P<n> --question "<the Final Tree question, word for word>"`.

`leaves/<id>/summary.md` (cap 2500 tokens):

```markdown
# Subproblem: P{i} <Title>
**Parent Chain**: P{parent} -> P{i}  **Status**: OPEN | DORMANT | EXHAUSTED at this scale | EXHAUSTED at this budget | ANSWERED | PROVEN-IMPOSSIBLE (must match State below)

## Route Portfolio
| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | T1 | IN-FLIGHT(job id) | — | ... | PASS → ...; FAIL → ... |

## Reopening Ledger (DORMANT only; section 3b)

## Leaf Status Summary
* **State**: ...
* **Updated**: YYYY-MM-DD (after which result, in one line)
* **Question**: the leaf's question, word for word from its Final Tree bullet.
* **Answer**: the direct reply to the question as asked (rules below).
* **What we showed**: the strongest findings, each with its scope.
* **What it would take**: what would settle the question as asked.
* **Conclusion**: the technical version, agreeing with Answer.
* (remaining fields and the prose argument: rules below)
```

`leaves/<id>/problem.md` (no cap; keep it short and move superseded text to the log):

```markdown
# Problem: P{i} <Title>

### 1. Mathematical Abstraction & Barriers
* **Formulation**: explicit inputs, outputs, and what "solved" means.
* **The Node Barrier**: where existing methods halt against the Hard Core.

### 1b. Analogy Map
* Each barrier restated in field-neutral terms; for each, at least three source fields with their problem, correspondence table, technique, why it worked there, translation gap and citation (Stage 2, item 3). Mark each entry untried, tried (R<id>) or repaired (R<id>).
```

`leaves/<id>/cards/<R>.md`: one route card each (above). For every live route the card also states its **Sharpest Gap** (the missing lemma, unproven scaling, or unverified assumption) and its **Gate 4 claim** (cost versus $N$; extensive gain or finite-size bonus), and, once it reaches T3, its decisive test: minimal instance, scalar metric $M$, execution plan, and PASS / FAIL / AMBIGUOUS thresholds.

`leaves/<id>/log/round-NN.md`: everything that happened in round NN, in full: generation batches (one line per route with its code), T0 arguments, T1 results and pre-flight outputs, value-of-information notes, critic reports, pasted logs. Append only.

#### 3a. Leaf Status Summary Rules (Mandatory)

The `## Leaf Status Summary` is the last section of `summary.md`. It is rewritten (not appended to) when the leaf's state changes. When a result lands without changing the state, update only the fields it touches (`Updated`, `Next step`, the affected evidence sentence, `Compute spent`); a full rewrite on every result costs agent time and changes nothing a reader needs. It is written for a reader who reads nothing else about the leaf.

**The answer comes first, and it answers the question as asked (mandatory).** A summary exists to answer the question the leaf was created for, not to narrate the run, and not to answer a narrower question the run happened to test. The user reads `Question`, `Answer`, `What we showed` and `What it would take` first, and often only those, so they must make sense on their own:
* **`Question`** is copied word for word from the leaf's bullet in the front page's `### Final Tree` (the text after the bold title). It is never paraphrased, narrowed or replaced by the question of the current round. The linter compares it with the Final Tree.
* **`Answer`** replies to that question in the generality in which it is asked. For a yes/no question it opens with exactly one of **Yes.**, **No.**, **Partly.** or **Not known.**; for a "how much" or "which" question it opens with the value or the choice. Then at most two short sentences say why, in plain words.
  - **Yes** or **No** only if the evidence settles the question as asked, not just the cases tried. A proof within a stated model settles it within that model; say so in the reply ("No, within the black-box model: ...").
  - **Partly.** when the question as asked splits into parts and the evidence settles some of them; name which.
  - **Not known.** when the evidence does not settle the question as asked, even if every case tried pointed one way. "No, for the networks we built" is not an answer to "Can a network of this kind do X?"; the answer is "Not known.", and the cases tried go under `What we showed`.
* **`What we showed`** gives the strongest findings that bear on the question, in at most four short sentences, each with its scope (models, sizes, design families) and at most one key number with what it is compared to.
* **`What it would take`** says what would settle the question as asked: the next decisive test and its cost, the larger instance or new method it needs, or why it is out of reach. For a settled Yes or No, it says what would overturn the answer.
* **Plain words in all four fields.** A physicist outside this subfield must understand them on first reading:
  - no route codes, job ids, tier or gate codes, arm names (`pcg`, `che`), verdict labels (CONFOUNDED, NO-TARGET), library numbers or rule thresholds;
  - no acronym or symbol that is not in the question itself, unless it is explained in a few words in the same sentence;
  - say what was compared with what in ordinary words ("the network with a pairing layer was 24% more accurate than the same network without it"), not as a ratio name.
* **Example.** Question: "Can a translation-equivariant fermionic NQS trained at $L$ initialize $2L$ with an energy error per site that is already close to converged?"
  - Not acceptable: "No, for the size-independent networks we built, going from 8 to 16 sites." This is a "No" to a narrower question.
  - Acceptable: Answer: "Not known. None of the transferable networks we built got close, but they cover one design family on clusters of at most 16 sites." What we showed: "Every size-independent network we built stalled at about 6% energy error on the 16-site target even when trained there directly, while a network with separate parameters per site reached 0.6%, so copying parameters could not help these designs." What it would take: "A transferable design that reaches converged accuracy when trained on the target itself; none is known."
* **`Conclusion`** may be more technical, but it must agree with `Answer` and add only what a specialist needs. Every other part of the summary supports the answer; none of it may contradict it or change its subject.
* **Self-check before saving.** Read only the four fields. If `Answer` responds to a narrower question than `Question`, if a Yes or No rests on cases rather than on the question as asked, or if a term is not defined, rewrite them. The per-leaf critic applies the same check, and the front page copies `Answer`, not `Conclusion`, into its tables and Summary.

**Self-contained argument (the standard every summary must meet).** After reading only the summary, the user must be able to explain the idea or the closure to a colleague and defend it against obvious objections. Field lists are not enough. The summary therefore contains, in prose (about 400–900 words; longer only if the argument needs it, and always within the 2500-token cap of `summary.md`, with details on the cards):
* **The question in plain terms**: what problem this leaf attacks and why it matters for the domain at hand, with every symbol and acronym defined at first use. No route labels ("7.1", "R2", "frdK") without saying what they are.
* **The idea or mechanism**: what the approach does and *why it should work*, explained physically or mathematically, not by name.
* **The chain of evidence**: premise → test → result → inference, step by step. For each step: what was compared with what (the arm and its foil), on which instance and size, the key numbers side by side with their uncertainties, and why that comparison is fair (matched parameters, matched wall time, pre-registered rule).
* **Why the alternatives fail or are worse**: the strongest competing routes or baselines and the evidence against them, summarised from their route cards (cite R<id>).
* **What this leaf taught beyond itself**: the findings-library entries it produced or changed (L<n>, one line each).
* **The strongest objection and the answer to it**: the best counterargument a skeptical referee would raise (from the critics, the cross-node critic, or a confound found later), and how the evidence answers it, or an honest statement that it is not answered.
* **Limits**: what the conclusion does not cover (sizes, instances, sectors, optimisation regime, sampled versus exact arithmetic).

Write it so that each claim is traceable: cite the card or job id where the number comes from in parentheses, but make the argument readable without following those pointers.

**Common fields (all states):** `State`, `Updated` (date and the result that triggered the update, in one line), `Question`, `Answer`, `What we showed`, `What it would take`, `Conclusion`, `Scope` (instances, sizes, sectors and compute regime the conclusion covers), and `Compute spent` (CPU-hours, from the compute ledger).

**Leaf still live: `State: OPEN` or `DORMANT`.** Add:
* `Most promising direction`: the single best live route, with its mechanism, the strongest evidence for it so far and the evidence against it.
* `Next step`: the concrete next test (tier, cost, pre-registered rule in one line), or the job in flight and what each verdict will trigger.
* `What would close it`: the result that would move the leaf to EXHAUSTED, ANSWERED or PROVEN-IMPOSSIBLE.
* DORMANT only: `Entered because` (which trigger of section 3b); the `## Reopening Ledger` section above holds the ledger table.

**Leaf truly complete: `State: ANSWERED`, `PROVEN-IMPOSSIBLE`, `EXHAUSTED at this scale` or `EXHAUSTED at this budget`.** These states are allowed only with a **closure certificate** containing all of:
1. `Claim`: the exact statement that is now settled, with its scope.
2. `Decisive evidence`: the results that settle it (job ids or derivations, verdicts under pre-registered rules, and the critic that reviewed them). For PROVEN-IMPOSSIBLE, the proof or its location and the model it holds in.
3. `Why no doubt remains`: every critic objection and confound raised in the leaf, each with how it was resolved, plus the cross-node critic's objections that touch this leaf.
4. `Falsifiers run`: every falsifier named by the leaf's diagnostics or critics, with its result. A named falsifier that has not run blocks these states, except EXHAUSTED at this budget, where a falsifier refused by the compute gate is listed with its refusal (section 3b, exit 3).
5. `Missing ingredients exhausted` (both EXHAUSTED states): every ingredient named by the Saturation Diagnostic, the route generated from it, and the tier at which it was killed, or, for EXHAUSTED at this budget only, the gate refusal that stopped it. An ingredient with neither blocks EXHAUSTED; the leaf stays DORMANT.
6. `What would reopen it`: the specific new evidence, scale or method that would invalidate the claim. For EXHAUSTED at this budget, also each refused step's capped cost and the budget that would fund it.

Within the summary's cap, the certificate gives each item in one or two sentences and points by id to the cards and log entries that hold the detail. If any certificate item cannot be filled honestly, the leaf is DORMANT, not closed. The per-leaf critic signs off on every certificate before the state changes, and the front page's "Current leaf state" table copies each leaf's `State`, `Question` and `Answer` verbatim.
