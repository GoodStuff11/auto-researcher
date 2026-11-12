# Circleback
## Configuration
`project.yaml` `sources.circleback`: `folder` (the Circleback folder the project's meetings are shared into) and/or `terms` (search words). `people.<key>.email` helps map attendees to people keys. Watermark `circleback` = date (YYYY-MM-DD) of the newest meeting ingested.
## Find new meetings
1. `ListFolders` → id of the configured folder (if missing: report "Circleback folder not visible to <me>" — the meeting may not be shared with this person — and treat the source as not reachable).
2. `SearchMeetings` (it requires an `intent` parameter: say what you are doing, e.g. "find new project meetings to ingest") with `folderId` (or `searchTerm` = the terms), `startDate` = watermark (first run: today minus 28 days), `pageIndex` 0, 1, … while a page has 20 results.
3. Drop meetings already recorded: `hub seen circleback:<meeting id>` prints `seen …`.
4. Series de-dup. A meeting belongs to a configured series when its title matches `series.<name>.calendar_match` in `project.yaml`.
   - Update mode: skip a series meeting whose agenda has not been read back yet (`Agenda/<series>.xlsx` still unread); post-meeting mode handles it. If a series meeting's agenda is still not read back a week after the meeting, tell the user (it holds the Circleback watermark back) and offer to run post-meeting mode, or to ingest it as a non-series meeting.
   - A series meeting that was already read back: compare with `hub meeting decided <series> <date> --json` and propose only what is missing.
5. `ReadMeetings` (also requires `intent`; up to 50 ids) for notes, action items, attendees. Ignore its "private notes" field. Use `GetTranscriptsForMeetings` only when the notes are ambiguous about a decision or an owner.
## Extract → batch file
Write the batch file in the scratch directory (format in `.agent_docs/ingest.md`), source `circleback:<meeting id>`:
- `safe` `log`: "<meeting name> (<date>, <attendee keys>): <one or two sentences of what moved>".
- `safe` `entry` on each existing item the meeting added a fact to (find it with `hub list` / `hub show`).
- `safe` `new` question for each open question raised (owner = who raised or will chase it).
- `proposals`: ONE proposal per decision, containing that decision and its actions; one proposal per closure.
- `watermark: {name: circleback, value: "<date of the newest meeting>"}`. Advance it to the newest meeting only when every older new meeting is in the same batch.
Rules for every `owner` in the batch (here and in the notes and mail/chat sources, safe ops and proposal ops alike): it must be a key of `people` in `project.yaml`, or the file is rejected. A person with no key (an outside attendee, someone unmapped) goes in the entry or title text, not in `owner`. Quote dates (`"2026-11-19"`) so YAML keeps them as strings. Actions found in Circleback are always proposals, never safe.
### A decision with its actions (one proposal)
The first op is the decision with `ref: $d`; its actions follow in the SAME proposal and link back with `"$d"` (a `$ref` only works inside one proposal).
```yaml
proposals:
  - title: "Decision: use framing B"
    reason: "Decided in the meeting, from the notes."
    source: "circleback:abc123"
    ops:
      - {op: new, type: decision, ref: $d, title: "Use framing B", owner: alice, source: "circleback:abc123",
         fields: {status: made, meeting: "group/2026-11-12"},   # meeting only for a configured series
         links: {answers: [Q-20261101-ab1]}}
      - {op: new, type: action, title: "Rewrite the intro", owner: bob, source: "circleback:abc123",
         links: {spawned_by: ["$d"]}}
```
- With no actions, put `no_action: "<reason>"` in the decision's `fields` instead.
- With no existing question to answer, add a question first in the same proposal: `{op: new, type: question, ref: $q, title: "...", source: ...}` and use `links: {answers: ["$q"]}`.
### Closing a question (one proposal)
Every closure sets `status` and `closed` (`"<YYYY-MM-DD>"`) and adds a closing `entry`:
```yaml
  - title: "Close: Which framing?"
    reason: "Answered in the meeting."
    source: "circleback:abc123"
    ops:
      - {op: set, id: Q-20261101-ab1, fields: {status: answered, closed: "2026-11-12"}}   # or dropped
      - {op: entry, id: Q-20261101-ab1, text: "Answered: framing B.", source: "circleback:abc123"}
```
A pivot uses `status: pivoted` and `closed`, plus a link to the replacing question: `{op: link, id: Q-old, rel: pivoted_to, target: Q-new}` (or `"$q"` for a question created earlier in the same proposal).
## Cross-check (post-meeting mode)
For the meeting just read back with `hub agenda read`: get `hub meeting decided <series> <date> --json`; propose only what the transcript/notes show was decided or assigned and is not in that list (title the proposal "Said at <series>/<date>: …"). Do not re-propose decisions already recorded. ALWAYS write a batch even when nothing needs proposing: a safe `log` line sourced `circleback:<meeting id>` plus the `circleback` watermark, so the meeting counts as ingested. Set the watermark only if the meeting's date is newer than the current one (`hub sync-state get circleback`); otherwise omit `watermark` from the batch, so it never moves back.
