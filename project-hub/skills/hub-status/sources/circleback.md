# Circleback
## Configuration
`project.yaml` `sources.circleback`: `folder` (the Circleback folder the project's meetings are shared into) and/or `terms` (search words). `people.<key>.email` helps map attendees to people keys. Watermark `circleback` = date (YYYY-MM-DD) of the newest meeting ingested.
## Find new meetings
1. `ListFolders` → id of the configured folder (if missing: report "Circleback folder not visible to <me>" — the meeting may not be shared with this person — and treat the source as not reachable).
2. `SearchMeetings` with `folderId` (or `searchTerm` = the terms), `startDate` = watermark (omit on first run, then ingest at most the last 4 weeks), `pageIndex` 0, 1, … while a page has 20 results.
3. Drop meetings already recorded: `hub seen circleback:<meeting id>` prints `seen …`.
4. `ReadMeetings` (up to 50 ids) for notes, action items, attendees. Use `GetTranscriptsForMeetings` only when the notes are ambiguous about a decision or an owner.
## Extract → batch file
For each meeting write into one batch file (`hub ingest batch`, format in `.agent_docs/ingest.md`), source `circleback:<meeting id>`:
- `safe` `log`: "<meeting name> (<date>, <attendee keys>): <one or two sentences of what moved>".
- `safe` `entry` on each existing item the meeting added a fact to (find it with `hub list` / `hub show`).
- `safe` `new` question for each open question raised (owner = who raised or will chase it).
- `proposals`: each decision taken (op `new` decision, `fields: {status: made, meeting: "<series>/<date>" when the meeting is a configured series}`, `links: {answers: [<question ids>]}`), each action item (op `new` action with `links: {spawned_by: [<decision id or $ref>]}`, owner mapped to a people key), each question closed or pivoted (`set` + closing `entry`).
- `watermark: {name: circleback, value: <date of the newest meeting in the file>}`.
Every `owner` in the batch (here and in the notes and mail/chat sources) must be a key of `people` in `project.yaml`; the file is rejected otherwise. A person with no key (an outside attendee, someone unmapped) goes in the entry or title text, not in `owner`. Quote dates (`"2026-11-19"`) so YAML keeps them as strings where a string is required.
## Cross-check (post-meeting mode)
For the meeting just read back with `hub agenda read`: get `hub meeting decided <series> <date> --json`; propose only what the transcript/notes show was decided or assigned and is not in that list (title the proposal "Said at <series>/<date>: …"). Do not re-propose decisions already recorded.
