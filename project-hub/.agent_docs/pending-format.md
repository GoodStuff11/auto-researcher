# Pending proposals (`Status/Pending/<P-id>.md`)
Code: `hub/pending.py` (`add_proposal`, `check_ops`, `load_proposals`, `load_unreadable`, `apply_proposals`, `reject_proposals`, `apply_ops`, `PendingError`). Ids: `new_pending_id` in `hub/ids.py`. CLI: `hub pending list | apply PID… | reject PID… | add --title T --ops-file F.yaml --source S --reason R`.
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
The heading is `## <P-id> · <title>` (title forced onto one line); then a fenced yaml block (regex `BLOCK_RE`, `hub/pending.py:18`). Files that do not match (or whose YAML does not parse to a mapping) are skipped by `load_proposals` and returned by `load_unreadable(root)`; `hub pending list` prints each as `UNREADABLE <path>` — fix or delete it by hand.
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
Ids of items created by a proposal are deterministic: the `new` op at index i of proposal P draws its id from `random.Random("P:i")` (`new_id` in `hub/ids.py`, still skipping ids that already exist). Two clones applying the same proposal therefore create the same item ids, and the `hub-item` merge driver merges the two copies (git-sync.md) instead of leaving duplicates.
## Checks (`check_ops(store, ops) -> list[str]`)
Run by `add_proposal` (raises `PendingError` listing every problem; nothing is staged) and again by `apply_proposals` before anything is written:
- `ops` is a list and each op is a mapping whose `op` is `new`, `entry`, `set` or `link`;
- required keys present: `new` → `type`, `title`; `entry` → `id`, `text`; `set` → `id`, `fields`; `link` → `id`, `rel`, `target`;
- `type` is a known item type; `rel` and every `links` key is `answers`, `spawned_by` or `pivoted_to`;
- each `$ref` used is defined by an earlier `new` op's `ref`; every non-ref `id`, `target` and `links` target exists in the store;
- `fields` is a mapping; `series` and each `links` value are lists;
- `source` (and `author`, if given) are non-empty with no whitespace.
## Commands and behaviour
- `list`: prints `P-xxxx  title  (by proposer, date): reason` per readable proposal, then `UNREADABLE <path>` per unreadable file, or `No pending proposals.`
- `apply PID…`: first a best-effort `git pull --rebase` (git-sync.md; a failure is a stderr warning). Author and every chosen proposal's ops are checked before any write. Then for each PID in order: run its ops, write a log line `Applied <P-id>: <title>` (source `pending:<P-id>`), delete the file, and print `applied <P-id>` followed by `created <id>` for each new item.
- `reject PID…`: log `Rejected <P-id>: <title>` and delete the file; ops are not run.
- `add`: stages a proposal from a YAML list file and prints the new P-id.
- Every failure of an op (unknown op kind, missing key, unknown ref, store error, bad type) raises `PendingError`; CLI prints `error: ...`, exit 1. An unknown PID raises `PendingError` before anything is applied.
## All-or-nothing per proposal
While a proposal applies, the prior bytes of every item file it changes are recorded before its first change, and every file it creates is recorded. If any op fails, those files are restored byte-for-byte, created files are deleted, nothing is logged, the proposal file stays in `Status/Pending/`, and `PendingError` is raised (naming any proposals earlier in the same command that were already applied — those stay applied). Fix the cause (or the proposal's ops) and re-apply, or reject it. `apply_ops` (library use) is not transactional.
## Who stages proposals automatically
`hub agenda read` (agenda-workbook.md) stages: free-text actions, decisions raised in the meeting, outcomes for already-resolved decisions, unrecognised outcome text, and notes without an outcome.
