# Pending proposals (`Status/Pending/<P-id>.md`)
Code: `hub/pending.py` (`add_proposal`, `check_ops`, `load_proposals`, `load_unreadable`, `apply_proposals`, `reject_proposals`, `apply_ops`, `note_path`, `PendingError`). Ids: `new_pending_id` in `hub/ids.py`. CLI: `hub pending list | apply PID… | reject PID… | add --title T --ops-file F.yaml --source S --reason R`.
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
The heading is `## <P-id> · <title>` (title forced onto one line); then a fenced yaml block (regex `BLOCK_RE` in `hub/pending.py`). Files that do not match (or whose YAML does not parse to a mapping) are skipped by `load_proposals` and returned by `load_unreadable(root)`; `hub pending list` prints each as `UNREADABLE <path>` — fix or delete it by hand.
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
- `op: note` — writes a human-owned note file (`path`, `text` required)
  - `path`: exactly `Vision.md`, `Locator.md`, or `Knowledge/<name>.md` (one level under `Knowledge/`; the name starts with a letter or digit and holds letters, digits, space, `_ . , ( )` or `-`). Anything else (`..`, other folders, `.txt`, lowercase `vision.md`, absolute paths, a trailing newline) is rejected by `check_ops` and by `note_path(root, rel)` (raises `PendingError`). The target's parent must resolve inside the vault and the target must not be a symbolic link, so a symlinked `Knowledge/` or `Vision.md` is rejected both at staging and at apply.
  - `text` (string) — the whole note when `mode` is `replace`, or the text added at the end when `mode` is `append`; a trailing newline is added if missing. `Knowledge/` is created if absent.
  - `mode`: `replace` (default) or `append`. Append joins a newline to the existing text when it does not already end with one.
  - `base` (replace mode only; written by `add_proposal`, i.e. by `hub pending add`, `hub ingest batch` and every other staging path): the sha256 hex digest of the target file's bytes when the proposal was staged, or `absent` if the file did not exist. An op that already carries `base` keeps it; `check_ops` rejects any value other than `absent` or 64 lowercase hex digits. At apply, the file as it was before this proposal touched it must match `base`; otherwise the op raises `PendingError` `<path> changed since this proposal was drafted; re-draft it (hub pending reject <P-id>)` and the whole proposal rolls back. So a hand edit after staging, a second replace proposal for the same path (staged before the first was applied), or a note created meanwhile where `absent` was recorded are all refused instead of silently overwritten; two replace ops for one path inside one proposal still apply. Ops with no `base` (written into a proposal file by hand) and append-mode ops are not checked.
  - Writes go through `atomic_write_text`. The op is rolled back with the rest of its proposal (see All-or-nothing below): the note's prior bytes are restored, or the file is removed if it did not exist.
  - `note` is never a safe op: `hub ingest batch` rejects `note` under `safe:` (only `entry`, `log` and `new` are safe). It may appear only in a staged `proposals` entry, so a human approves each change.
`$ref` resolution (`_resolve`): in `new.links` values, and in `id`, `target` of other ops, a string starting with `$` is replaced by the id of the item created earlier by the op with that `ref`; an unknown `$name` raises `PendingError`. Refs only live within one proposal. The entry author on apply is the person running `hub pending apply` (`me`), not the proposer; entry day is today.
Ids of items created by a proposal are deterministic: the `new` op at index i of proposal P draws its id from `random.Random("P:i")` (`new_id` in `hub/ids.py`, still skipping ids that already exist). Two clones applying the same proposal therefore create the same item ids, and the `hub-item` merge driver merges the two copies (git-sync.md) instead of leaving duplicates.
## Checks (`check_ops(store, ops) -> list[str]`)
Run by `add_proposal` (raises `PendingError` listing every problem; nothing is staged), by `hub ingest batch` and `hub clickup apply` for the proposals they stage, by `apply_proposals` for every chosen proposal before anything is written, and again for each proposal just before it is applied (so a proposal applied earlier in the same call, e.g. one that linked the same ClickUp task, is taken into account):
- `ops` is a non-empty list (`[]` → `proposal has no ops`) and each op is a mapping whose `op` is `new`, `entry`, `set`, `link` or `note`;
- required keys present: `new` → `type`, `title`; `entry` → `id`, `text`; `set` → `id`, `fields`; `link` → `id`, `rel`, `target`; `note` → `path`, `text`;
- `note`: `path` is an allowed note path, `text` is a string, `mode` is `replace` or `append`, `base` (if present) is `absent` or a sha256 hex digest;
- `type` is a known item type; `rel` and every `links` key is `answers`, `spawned_by` or `pivoted_to`;
- each `$ref` used is defined by an earlier `new` op's `ref`; every non-ref `id`, `target` and `links` target exists in the store;
- `fields` is a mapping; `series` and each `links` value are lists;
- `source` (and `author`, if given) are non-empty with no whitespace;
- values are dry-run as the store would apply them (`fields` of `new` and `set` ops): `coerce_value` and `check_field` (dates `YYYY-MM-DD`, integer `priority`, list `series`/`sources`, non-empty `title`; `invalid value for 'due': 'Friday'`), `id`/`type`/`links` cannot be set through fields, and `status` must be valid for the item's type (`set`: the existing item's type, or the type of the `new` op a `$ref` names; `new`: its `type`);
- a `new` op whose `fields.clickup_id` is already the `clickup_id` of an existing action is rejected: `ClickUp task <id> already linked to <A-id>`.
Owner checks against `people` are not done here; `hub ingest batch` does them (`check_batch`).
## Commands and behaviour
- `list`: prints `P-xxxx  title  (by proposer, date): reason` per readable proposal, then `UNREADABLE <path>` per unreadable file, or `No pending proposals.`
- `apply PID…`: first a best-effort `git pull --rebase` (git-sync.md; a failure is a stderr warning). Author and every chosen proposal's ops are checked before any write. Then for each PID in order: run its ops, write a log line `Applied <P-id>: <title>` (source `pending:<P-id>`), delete the file, and print `applied <P-id>` followed by `created <id>` for each new item.
- `reject PID…`: log `Rejected <P-id>: <title>` and delete the file; ops are not run.
- `add`: stages a proposal from a YAML list file and prints the new P-id.
- Every failure of an op (unknown op kind, missing key, unknown ref, store error, bad type) raises `PendingError`; CLI prints `error: ...`, exit 1. An unknown PID raises `PendingError` before anything is applied.
## All-or-nothing per proposal
While a proposal applies, the prior bytes of every item file and every `note` file it changes are recorded before its first change, and every file it creates is recorded. If any op fails, those files are restored byte-for-byte, created files are deleted, nothing is logged, the proposal file stays in `Status/Pending/`, and `PendingError` is raised (naming any proposals earlier in the same command that were already applied — those stay applied). Fix the cause (or the proposal's ops) and re-apply, or reject it. `apply_ops` (library use) is not transactional.
## Who stages proposals automatically
`hub agenda read` (agenda-workbook.md) stages: free-text actions, decisions raised in the meeting, outcomes for already-resolved decisions, unrecognised outcome text, and notes without an outcome.
