# project-hub
**People: start with the [user guide](USER-GUIDE.md)** (setup, joining, meetings, troubleshooting).
A Markdown status store (questions, decisions, actions), staged proposals, an Excel meeting agenda and git sync, driven by the `hub` command line so people and agents can share one project vault. Design: [docs/superpowers/specs/2026-11-12-project-hub-design.md](../docs/superpowers/specs/2026-11-12-project-hub-design.md). Agents: start at [.agent_docs/_INDEX.md](.agent_docs/_INDEX.md).
## Install
`python -m venv .venv && . .venv/bin/activate && pip install -e '.[test]'`, or `uv pip install -e .`
## First use in a vault
The vault is a git repository whose root holds `project.yaml` (committed):
```
name: my-project
people:
  alice: {name: Alice}
  bob: {name: Bob}
series:
  group: {attendees: [alice, bob]}
```
and `hub.local.yaml` (gitignored) naming you: `me: alice`. Run `hub` from anywhere inside the vault. Full format details: `.agent_docs/vault-layout.md`.
## Everyday commands
- `hub new q|d|a "Title" --source manual --series group` creates an item and prints its id.
- `hub list --type action --owner me --status open`
- `hub show ID --chain`, `hub entry ID "text" --source manual`, `hub set ID status=answered closed=2026-11-12`, `hub link ID answers Q-...`
- `hub agenda scaffold group --date 2026-11-19`, fill `summary` in `Agenda/group.draft.yaml`, then `hub agenda build group`; after the meeting `hub agenda read group`
- `hub pending list`, `hub pending apply P-xxxx`, `hub pending reject P-xxxx`
- `hub validate`, `hub snapshot`, `hub commit "message"`
`hub --help` and `hub <command> --help` document every option; exit codes are 0 ok, 1 error or violations, 2 bad arguments, 3 sync conflict.
## Tests
`pytest -q`
