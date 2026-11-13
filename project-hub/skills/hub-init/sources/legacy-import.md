# Legacy import (any document)
Brings one existing document into the hub as proposals. Used by hub-init step 5, and any time the user names an old document to import. The importer assumes no format.
## Read
Any document, any format: Markdown, LaTeX, plain text, exported docs, tracker exports. Read it whole, or section by section if long. Never assume a structure; work out what the document is from its content. Make a scratch directory outside the vault (`SCRATCH=$(mktemp -d)`).
## Classify each passage
- **open question**: something still unresolved.
- **decision taken**: with rationale, date and who decided.
- **action**: with owner and whether done.
- **question already answered or dropped**: with the answer.
- **stable knowledge**: how the code works, conventions, what a result supports, claims tried and rejected.
- **vision material**: hand it to the hub-vision skill.
- **history narrative**: at most a few entry proposals (see Batch file).
- **noise or duplicates**: skip.

When unsure whether something was decided, make it an open question. Never invent facts, owners or dates. Use the document's own dates; otherwise today.
## People and source ids
Map names to people keys from `project.yaml`. A person with no key stays in the text and never becomes an `owner` (the batch would be rejected).
Source id: `doc:<file-stem-slug>#<section-slug>`, lowercase, every run of non-alphanumeric characters replaced by `-` (`notes.md`, section "Open Items" gives `doc:notes#open-items`). No whitespace.
## Batch file
Write it in the scratch directory. An import is all judgement, so everything goes under `proposals` (full format: the hub package's `.agent_docs/ingest.md` and `pending-format.md`). Quote dates. Verified recipe:
- **Dates.** An item's `raised` and `closed` fields hold the document's dates: set them through `fields` (`raised: "<date>"`, and `closed: "<date>"` when closed). Entry dates are always the day of the import and cannot be set, so cite the document's date and item number in the entry text. Use today only when the document has none (omit `raised`, which then defaults to today).
- **Statuses.** question: `open|answered|dropped|pivoted`; decision: `proposed|made|rejected|deferred`; action: `open|done|dropped`. A decision also takes `rationale`, `as_decided`, `no_action` in `fields`.
- **Who.** `owner` only for a known people key (a decision's owner is who decided). Otherwise omit it and name the person in the text or description.
- **A decision taken**: ONE proposal holding its question, the decision and its actions. The question is a `new` question with `ref: $q`, `status: answered` and the dates, plus a closing `entry` with `id: $q` stating the answer. If the question already exists in the vault (`hub list --type question`), skip that op, use its Q-id in `links: {answers: [Q-…]}`, and close it with a `set` op (`status: answered`, `closed`) and an `entry` op. The decision is a `new` decision with `ref: $d`, `status: made`, `rationale`, and `links: {answers: ["$q"]}`, then an `entry` with `id: $d` saying who decided, when, and why. Each action is a `new` action with `ref: $a` and `links: {spawned_by: ["$d"]}`; if done, `status: done` and `closed`, plus an `entry` with `id: $a` giving the outcome. A decision with no action needs `no_action: "<reason>"` in its `fields`. An open action with no known owner is allowed in the batch; it is fixed in "close the gaps".
- **One proposal per standalone open question** (`new` question, `raised`).
- **Closed questions** not part of a decision: `new` question with `ref: $q`, `status: answered` (or `dropped`), `closed`, and an `entry` with `id: $q` giving the answer.
- **One proposal per Knowledge note**: a `note` op, `path: Knowledge/<Topic>.md`, `mode: replace`.
- **History narrative**: at most a few proposals, each one `entry` op on the existing item it concerns; skip it when there is no such item (proposals cannot hold `log` lines).

Keep the document's own numbering in the description for traceability, e.g. "Imported from notes.md, item 12". Accepted by `hub ingest batch` and applied to the intended statuses:
```yaml
proposals:
  - title: "Import: Use framing B"
    reason: "Imported from notes.md, item 12 (decided by alice, 2026-03-02)."
    source: "doc:notes#decisions"
    ops:
      - {op: new, type: question, ref: $q, title: "Which framing?", source: "doc:notes#decisions",
         description: "Imported from notes.md, item 12",
         fields: {status: answered, raised: "2026-02-20", closed: "2026-03-02"}}
      - {op: entry, id: $q, text: "Answered: framing B (notes.md, item 12, 2026-03-02).", source: "doc:notes#decisions"}
      - {op: new, type: decision, ref: $d, title: "Use framing B", owner: alice, source: "doc:notes#decisions",
         description: "Imported from notes.md, item 12",
         fields: {status: made, raised: "2026-03-02", closed: "2026-03-02", rationale: "Simpler to compare.", as_decided: "Use framing B"},
         links: {answers: ["$q"]}}
      - {op: entry, id: $d, text: "Decided by alice on 2026-03-02 (notes.md, item 12): simpler to compare.", source: "doc:notes#decisions"}
      - {op: new, type: action, ref: $a, title: "Rewrite the intro", owner: bob, source: "doc:notes#decisions",
         description: "Imported from notes.md, item 13",
         fields: {status: done, raised: "2026-03-02", closed: "2026-03-09"}, links: {spawned_by: ["$d"]}}
      - {op: entry, id: $a, text: "Done (notes.md, item 13): intro rewritten.", source: "doc:notes#decisions"}
      - {op: new, type: action, title: "Rerun baseline", source: "doc:notes#decisions",
         description: "Imported from notes.md, item 14; owner named in the text only (Carol)",
         fields: {raised: "2026-03-02"}, links: {spawned_by: ["$d"]}}
  - title: "Import: Drop method C"
    reason: "Imported from notes.md, item 20; its question already exists."
    source: "doc:notes#methods"
    ops:
      - {op: set, id: Q-20261113-jpd, fields: {status: answered, closed: "2026-03-05"}}
      - {op: entry, id: Q-20261113-jpd, text: "Answered: drop method C (notes.md, item 20, 2026-03-05).", source: "doc:notes#methods"}
      - {op: new, type: decision, title: "Drop method C", source: "doc:notes#methods",
         fields: {status: made, raised: "2026-03-05", closed: "2026-03-05", no_action: "Nothing to do"},
         links: {answers: [Q-20261113-jpd]}}
  - title: "Knowledge: Solver conventions"
    reason: "Stable knowledge from notes.md."
    source: "doc:notes#conventions"
    ops:
      - op: note
        path: Knowledge/Solver conventions.md
        mode: replace
        text: |
          # Solver conventions
          ...
```
## Show a summary
Counts by kind and by section, plus everything skipped and why. In a dry run, stop here (the batch file stays in the scratch directory).
## Otherwise
1. `hub ingest batch <file>`; if it is rejected, fix every listed problem and re-run (nothing was written).
2. Approval batch: `hub pending list`; ask the user, by section and at most about 30 proposals at a time, which to apply or reject; then `hub pending apply <P-ids>` / `hub pending reject <P-ids>`.
3. Close the gaps as in hub-status update step 7 (for example `hub set <A-id> owner=<key>` for an action left without an owner).
4. `hub validate`.
5. `hub commit "import: <doc> (<date>)"`. Exit code 3 is a real conflict: stop and show the message.
6. `rm -rf "$SCRATCH"`.
## Privacy
Same rules as hub-status: the vault is read by everyone on the project. Leave out personal assessments of people, private notes and career material, even if the document contains them; tell the user you skipped them.
