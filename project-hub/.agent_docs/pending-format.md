# Pending proposals (`Status/Pending/<P-id>.md`)
Code: `hub/pending.py` (`add_proposal`, `load_proposals`, `apply_proposals`, `reject_proposals`, `apply_ops`, `PendingError`). Ids: `new_pending_id` in `hub/ids.py`. CLI: `hub pending list | apply PID… | reject PID… | add --title T --ops-file F.yaml --source S --reason R`.
A pending proposal is a judgement call staged for one yes/no from whoever runs the agent. Rule: any new item an agent infers from free text (a Circleback transcript, Slack, a handoff, an agenda cell) goes here instead of straight into the store; so does closing a question and recording a decision as made outside the agenda read-back.
## File format
One file per proposal, so concurrent proposals never touch the same file (no merge conflicts, no union driver needed). There is no `Pending.md`. P-ids are random: `P-` plus 4 characters of `a-z0-9`, e.g. `P-k3x9`; the filename is `<P-id>.md`.
````
## P-k3x9 · Actions from group/2026-11-19: Ship it
```yaml
proposer: alice
date: 2026-11-19
source: agenda:group/2026-11-19
reason: Free-text actions written in the meeting.
ops:
- op: new
  type: action
  title: Write docs
  owner: bob
  series: [group]
  links: {spawned_by: [D-20261112-igo]}
  source: agenda:group/2026-11-19
```
````
The heading is `## <P-id> · <title>` (title forced onto one line); then a fenced yaml block (regex `BLOCK_RE`, `hub/pending.py:18`). Files that do not match are silently skipped by `load_proposals`.
Block keys:
- `proposer` (string) — people key of whoever staged it
- `date` (date) — day staged
- `source` (string) — provenance id, same conventions as item entries (item-format.md)
- `reason` (string) — why a human must decide
- `ops` (list of mappings) — applied in order on approval
## Ops (each mapping has `op:` plus the keys below)
- `op: new` — create an item (`Store.create`)
  - `type` (required): `question`, `decision` or `action`
  - `title` (required)
  - `owner` (default null; set it, actions need an owner to pass validate)
  - `series` (list, default `[]`), `priority` (default 2), `description` (default empty)
  - `links` (mapping rel to list of ids or `$refs`), `fields` (mapping of extra frontmatter keys, e.g. `status`, `meeting`, `as_decided`; not `id`/`type`/`links`)
  - `source` (default `pending`) — used as the creation entry and in `sources`
  - `ref` (optional, a string starting with `$`, e.g. `$d`) — names the new item so later ops can refer to it
- `op: entry` — `id` (required), `text` (required), `source` (default `pending`); `Store.entry`
- `op: set` — `id` (required), `fields` (mapping, same rules as `hub set`); `Store.set_fields`
- `op: link` — `id`, `rel`, `target` (all required); `Store.link`
`$ref` resolution (`_resolve`): in `new.links` values, and in `id`, `target` of other ops, a string starting with `$` is replaced by the id of the item created earlier by the op with that `ref`; an unknown `$name` raises `PendingError`. Refs only live within one proposal. The entry author on apply is the person running `hub pending apply` (`me`), not the proposer; entry day is today.
## Commands and behaviour
- `list`: prints `P-xxxx  title  (by proposer, date): reason`, or `No pending proposals.`
- `apply PID…`: for each PID in order, run its ops, write a log line `Applied <P-id>: <title>` (source `pending:<P-id>`), delete the file. Prints `created <id>` for each new item. Author is validated before any write.
- `reject PID…`: log `Rejected <P-id>: <title>` and delete the file; ops are not run.
- `add`: stages a proposal from a YAML list file and prints the new P-id.
- Every failure of an op (unknown op kind, missing key, unknown ref, store error, bad type) raises `PendingError`; CLI prints `error: ...`, exit 1. An unknown PID raises `PendingError` before anything is applied.
## Non-transactional behaviour and recovery
Ops of a proposal are not rolled back: if op 3 of 5 fails, ops 1-2 are already written and the proposal file is still present (it is deleted only after all ops succeed). Recover: `git diff` (and `git status` for new item files) to see what landed; fix or delete those partial changes (or edit the proposal's ops to the remaining ones), run `hub validate`, then re-apply or reject.
## Who stages proposals automatically
`hub agenda read` (agenda-workbook.md) stages: free-text actions, decisions raised in the meeting, outcomes for already-resolved decisions, unrecognised outcome text, and notes without an outcome.
