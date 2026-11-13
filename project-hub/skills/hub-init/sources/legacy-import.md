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
- **history narrative**: at most a few `log` line proposals.
- **noise or duplicates**: skip.

When unsure whether something was decided, make it an open question. Never invent facts, owners or dates. Use the document's own dates; otherwise today.
## People and source ids
Map names to people keys from `project.yaml`. A person with no key stays in the text and never becomes an `owner` (the batch would be rejected).
Source id: `doc:<file-stem-slug>#<section-slug>`, lowercase, every run of non-alphanumeric characters replaced by `-` (`notes.md`, section "Open Items" gives `doc:notes#open-items`). No whitespace.
## Batch file
Write it in the scratch directory. An import is all judgement, so everything goes under `proposals` (shapes: hub-status `sources/circleback.md`; full format: the hub package's `.agent_docs/ingest.md`). Quote dates.
- One proposal per decision: the decision (`ref: $d`; its question first with `ref: $q` when new, linked by `answers`) and its actions (`links: {spawned_by: ["$d"]}`), all in the same proposal.
- One proposal per standalone open question.
- Closed questions: a `new` question op with `fields: {status: answered, closed: "<date>"}` (or `dropped`) plus a closing `entry` op stating the answer.
- Done actions likewise: a `new` action op with `fields: {status: done, closed: "<date>"}` plus a closing `entry`.
- One proposal per Knowledge note: a `note` op, `path: Knowledge/<Topic>.md`, `mode: replace`.

Keep the document's own numbering in the description for traceability, e.g. "Imported from notes.md, item 12".
```yaml
proposals:
  - title: "Import: Which framing?"
    reason: "Imported from notes.md, item 12; judged to be an open question."
    source: "doc:notes#open-items"
    ops:
      - {op: new, type: question, title: "Which framing?", source: "doc:notes#open-items",
         description: "Imported from notes.md, item 12"}
  - title: "Knowledge: Solver conventions"
    reason: "Stable knowledge from notes.md, section Conventions."
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
3. Close the gaps as in hub-status update step 7.
4. `hub validate`.
5. `hub commit "import: <doc> (<date>)"`. Exit code 3 is a real conflict: stop and show the message.
6. `rm -rf "$SCRATCH"`.
## Privacy
Same rules as hub-status: the vault is read by everyone on the project. Leave out personal assessments of people, private notes and career material, even if the document contains them; tell the user you skipped them.
