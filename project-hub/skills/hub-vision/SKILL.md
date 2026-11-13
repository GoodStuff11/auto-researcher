---
name: hub-vision
description: Use when asked to write, draft or revise a project hub's vision ("write the vision", "update Vision.md", "we pivoted, revise the vision") in a vault that has project.yaml and a Vision.md.
---
# Hub vision
Writes or revises `Vision.md`: what the project is for, why it matters and why this team. `Vision.md` changes only through an approved `note` proposal, never by direct edit. `hub <command> --help` documents every option.
## Before starting
1. Work from the vault root (the directory with `project.yaml`); `git pull --rebase` (skip in a dry run). If it fails, stop and show the error.
2. `SCRATCH=$(mktemp -d)` outside the vault. Every temporary file goes there, because `hub commit` stages everything. Delete it at the end.
3. **Dry run.** If the user says "dry run", only read: `hub validate`, `hub pending list`, reading `Vision.md`. Run the interview, write the draft and ops file to the scratch directory and show them, and skip `hub pending add|apply|reject` and `hub commit`. Say which steps were skipped.
## Interview
One question at a time. Push for specifics and evidence; do not accept slogans.
1. The vision.
2. The problem as the community poses it, with key references.
3. Our approach.
4. How we answer the problem.
5. Our unfair advantage: why us, why now, what competitors lack.
6. The difference we will make: who changes what.
7. Risks and early signals.
## Check claims of novelty or advantage
If a literature-search skill is available (for example research-question), offer to run it on each such claim. Otherwise ask the user for evidence. Mark every claim still unsupported "(to verify)" in the draft.
## Draft
Write the draft to the scratch directory in the sections of the current `Vision.md`: Vision; The problem, as the community poses it; Our approach; How we answer the problem; Our unfair advantage; The difference we will make; Risks and how we will know early; Revisions. Under `## Revisions` append one line:
`- <YYYY-MM-DD> — <what changed> — <initial | after [[D-…]]>`
Leave no blank line directly after a heading.
## Stage and apply
1. Ops file in the scratch directory, one `note` op:
```yaml
- op: note
  path: Vision.md
  mode: replace
  text: |
    # Vision
    ...
```
2. Show the user the diff between the current `Vision.md` and the draft (`diff -u Vision.md <draft>`).
3. `hub pending add --title "Vision: <what changed>" --ops-file <file> --source manual --reason "<why>"`.
4. After the user's yes, `hub pending apply <P-id>` (on a no, `hub pending reject <P-id>`).
5. `hub validate`, then `hub commit "Vision: <what changed>"`.
## Revisions later
Triggered by a pivot (a question pivoted, or a decision that changes direction) or by the user. Read the current `Vision.md`, re-interview only the affected sections, keep unchanged sections verbatim, and add a Revisions line (for a decision, `after [[D-…]]`). Stage and apply as above.
## Privacy
The vault is read by everyone on the project: no personal assessments, private notes or career material in `Vision.md`.
