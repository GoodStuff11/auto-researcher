# Project hub skills and ingest — Implementation Plan (Plan 2 of 3)
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hub usable day to day.
- Agent skills `hub-status` (update / todos / post-meeting) and `hub-agenda` (meeting prep).
- Deterministic ingest helpers in the `hub` package, which the skills drive: git and Overleaf commits, ClickUp reconciliation, a generic ingest-batch file (Circleback, notes, mail and chat), idempotency checks, meeting dates, and a source overview.

**Architecture:** The skills do what needs an LLM or a connector: calling the ClickUp, Circleback and Google Calendar MCP tools, reading meetings, and deciding what a fact means. Everything mechanical is a `hub` subcommand with tests:
- diffing ClickUp against the store;
- de-duplicating re-ingested sources;
- listing commits since a watermark;
- applying "safe" changes and staging judgement calls.

The skills write only through `hub`. New commands live in `hub/cli_ext.py`, hooked into the existing `hub/cli.py` in two places, so the large CLI file barely changes.

**Tech Stack:** Python ≥ 3.10, PyYAML, openpyxl (unchanged dependencies), git, pytest. Claude Code skills (Markdown with YAML frontmatter). MCP connectors: ClickUp, Circleback, Google Calendar (plus optional Gmail and Slack).

**Spec:** `docs/superpowers/specs/2026-11-12-project-hub-design.md` (sections *Skills*, *Ingest sources*, *ClickUp mapping*, *Privacy and sharing*). Plan 1 and its follow-ups: `docs/superpowers/plans/2026-11-12-project-hub-core.md`, `docs/superpowers/plans/2026-11-12-project-hub-core-followups.md`.

## Global Constraints
- Runtime dependencies stay **only** PyYAML and openpyxl. Run tests with `project-hub/.venv/bin/python -m pytest -q` from `project-hub/`. Create the venv first if it is missing: `python3 -m venv .venv && .venv/bin/pip install -e '.[test]'`.
- **Everything Plan 1 established still holds:**
  - item and entry formats;
  - the source token, which is non-empty with no whitespace;
  - `Status/Pending/<P-id>.md` holds one proposal per file;
  - sync-state lives under `Status/sync-state/`, and key names must match `[A-Za-z0-9_-][A-Za-z0-9_.-]*`;
  - atomic writes via `hub/fsutil.atomic_write_text`;
  - CLI exit codes 0 / 1 / 2 / 3;
  - every subparser and argument has non-empty `help`, and every subparser a `description` (enforced by `tests/test_cli.py`).
- **Source id conventions** (use them exactly):

  | Source | Format |
  |---|---|
  | Circleback meeting | `circleback:<meeting id>` |
  | ClickUp task | `clickup:<task id>` |
  | Commit (code repo or Overleaf) | `commit:<ROOT>@<7-char sha>` |
  | Personal notes | `notes:<people key>@<YYYY-MM-DD>` |
  | Gmail thread | `gmail:<thread id>` |
  | Slack message | `slack:<channel id>-<ts>` |
  | Meeting read-back | `agenda:<series>/<date>` |
- **Watermark names:**
  - `clickup`: ClickUp time of the last sync;
  - `circleback`: date of the last ingested meeting, YYYY-MM-DD;
  - `git.<ROOT>`: full sha of the newest ingested commit;
  - `notes.<people key>`, `gmail`, `slack`: dates.
- **Safe vs. judgement (spec, *Pending changes*):**
  - Applied directly: entries and log lines with a source; new questions and actions with a source; mirroring ClickUp status, assignee and due date.
  - Always staged: closing a question; recording a decision as made from anything except the agenda read-back; a pivot; any Vision or Knowledge change; new actions parsed from free text; tasks created directly in ClickUp.
- **Privacy (spec):** record decisions, stated positions and actions with their source. Never record personal assessments of people, career or 1:1 material, or verbatim text from someone's private notes.
- Markdown written for Obsidian: no blank line after a heading (`.agents/rules/markdown-documentation.md`).
- Keep `project-hub/.agent_docs/` exhaustive and current (`.agents/rules/memory.md`). A new domain gets a new file and an `_INDEX.md` entry.

---
## File structure
```
project-hub/
  hub/
    items.py         # MODIFY: Item.links tolerant of wrong shapes
    validate.py      # MODIFY: `shape` rule
    merge_item.py    # MODIFY: neutral A=/B= conflict labels
    gitsync.py       # MODIFY: guidance says which side A/B is
    agenda/lifecycle.py  # MODIFY: untitled-row whitespace, unmatched action owner stays in the title
    sources.py       # NEW: source_seen (idempotency)
    ingest_git.py    # NEW: commits since watermark git.<ROOT>
    batch.py         # NEW: `hub ingest batch` file format: safe ops + proposals + flags + watermark
    clickup.py       # NEW: load_tasks, plan_clickup, apply_clickup
    cli_ext.py       # NEW: seen | ingest git|batch | clickup plan|apply | meeting next|decided | sources
    cli.py           # MODIFY: import + 2 hook lines + docstring
  tests/
    test_followups.py  test_sources.py  test_ingest_git.py  test_batch.py  test_clickup.py
    test_cli_ext.py    test_skills.py
    (MODIFY test_merge_item.py, test_agenda_lifecycle.py for changed messages/behaviour)
  skills/
    hub-status/SKILL.md
    hub-status/sources/{clickup,circleback,git,calendar,notes,mail-chat}.md
    hub-agenda/SKILL.md
  .agent_docs/
    ingest.md        # NEW: every new command and file format
    _INDEX.md, item-format.md, git-sync.md, agenda-workbook.md, vault-layout.md   # MODIFY
```
Paths below are relative to the repository root (the worktree).

In code blocks, `# EDIT <path>` blocks give an exact `# OLD` snippet to find (it occurs once) and the `# NEW` text to replace it with. Blocks headed `# <path>` are whole new files.

---
### Task 1: Plan 1 follow-ups (wrong-shaped items, merge labels, meeting text)
These fix the parked items from `2026-11-12-project-hub-core-followups.md` that later tasks rely on.

**Files:**
- Modify: `project-hub/hub/items.py`, `project-hub/hub/validate.py`, `project-hub/hub/merge_item.py`, `project-hub/hub/gitsync.py`, `project-hub/hub/agenda/lifecycle.py`
- Modify tests: `project-hub/tests/test_merge_item.py`, `project-hub/tests/test_agenda_lifecycle.py`
- Create: `project-hub/tests/test_followups.py`

**Interfaces:**
- Produces:
  - `Item.links(key)` returns `[]` for any non-mapping `links` or non-list value;
  - validate rule name `shape`;
  - merge-driver stderr line `hub-item conflict in <id>: <key>: A=<value> B=<value>`;
  - `_action_ops` puts an unmatched owner back into the title with owner `None`.

- [ ] **Step 1: Write the failing tests**
```python
# project-hub/tests/test_followups.py
from datetime import date

import pytest
from openpyxl import load_workbook

from hub.agenda.build import SHEET_D
from hub.agenda.draft import save_draft, scaffold
from hub.agenda.lifecycle import build_agenda, read_agenda
from hub.config import load_config
from hub.pending import load_proposals
from hub.snapshot import render_snapshot
from hub.store import Store
from hub.validate import validate

D = date(2026, 11, 12)


@pytest.fixture
def env(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    return Store(hub_root), load_config(hub_root), hub_root


def test_wrong_shaped_links_are_reported_not_fatal(env):
    store, cfg, root = env
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    p = store.path(q.id)
    text = p.read_text()
    text = text.replace("links:\n  answers: []\n  spawned_by: []\n  pivoted_to: []", "links:\n- a")
    text = text.replace("series: []", "series: group")
    p.write_text(text)
    item = store.get(q.id)
    assert item.links("answers") == []
    rules = sorted(v.rule for v in validate(store, cfg))
    assert rules == ["shape", "shape"]
    assert "# Snapshot — demo" in render_snapshot(store, cfg, D)


def _meeting(store, root, rows):
    q = store.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m",
                     series=["group"])
    d = store.create("decision", "Adopt Y", owner="alice", author="alice", day=date(2026, 11, 2), source="m",
                     series=["group"], links={"answers": [q.id]})
    draft = scaffold(store, root, "group", date(2026, 11, 19))
    draft["summary"] = "Status."
    save_draft(root, draft)
    path = build_agenda(root, "group")
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    for ref, val in rows(d.id, ws).items():
        ws[ref] = val
    wb.save(path)
    return d


def test_untitled_row_title_is_single_line(env):
    store, cfg, root = env
    _meeting(store, root, lambda did, ws: {"H6": "approve", "J6": "Rerun HVA (bob)\nPlot it (alice)"})
    read_agenda(store, cfg, "group", author="bob", today=date(2026, 11, 19))
    titles = [p.title for p in load_proposals(root)]
    assert titles == ["Raised at group/2026-11-19: (untitled: Rerun HVA (bob) Plot it (alice))"]


def test_unmatched_owner_stays_in_title(env):
    store, cfg, root = env
    _meeting(store, root, lambda did, ws: {"H5": "approve", "J5": "Review draft (due Friday); Plot (Bob)"})
    read_agenda(store, cfg, "group", author="bob", today=date(2026, 11, 19))
    p = next(x for x in load_proposals(root) if x.title.startswith("Actions"))
    assert [(o["title"], o["owner"]) for o in p.body["ops"]] == [("Review draft (due Friday)", None),
                                                                  ("Plot", "bob")]
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_followups.py -v`
Expected: 3 FAIL:
- `test_wrong_shaped_links_...` with `AttributeError: 'list' object has no attribute 'get'`;
- the untitled test, because its title contains a newline;
- the owner test, which gets `('Review draft', 'due Friday')`.

- [ ] **Step 2: Make `Item.links` tolerant**
```python
# EDIT project-hub/hub/items.py
# OLD
    def links(self, key: str) -> list[str]:
        return list((self.meta.get("links") or {}).get(key) or [])
# NEW
    def links(self, key: str) -> list[str]:
        links = self.meta.get("links")
        value = links.get(key) if isinstance(links, dict) else None
        return [str(v) for v in value] if isinstance(value, list) else []
```

- [ ] **Step 3: Add the `shape` rule**
```python
# EDIT project-hub/hub/validate.py
# OLD
def validate(store: Store, config: HubConfig) -> list[Violation]:
# NEW
def _shape_problems(it) -> list[str]:
    """Frontmatter values whose type is wrong (e.g. hand-edited in Obsidian): reported, never fatal."""
    probs = []
    title = it.meta.get("title")
    if not isinstance(title, str) or not title.strip():
        probs.append("title must be a non-empty string")
    links = it.meta.get("links")
    if links is not None and not (isinstance(links, dict)
                                  and all(v is None or isinstance(v, list) for v in links.values())):
        probs.append("links must be a mapping of relation -> list of ids")
    for key in ("series", "sources"):
        value = it.meta.get(key)
        if value is not None and not isinstance(value, list):
            probs.append(f"{key} must be a list")
    return probs


def validate(store: Store, config: HubConfig) -> list[Violation]:
```
```python
# EDIT project-hub/hub/validate.py
# OLD
        if it.status not in STATUSES[it.type]:
            bad(it, "status", f"status '{it.status}' invalid for {it.type}")
# NEW
        if it.status not in STATUSES[it.type]:
            bad(it, "status", f"status '{it.status}' invalid for {it.type}")
        for msg in _shape_problems(it):
            bad(it, "shape", msg)
```

- [ ] **Step 4: Neutral merge-conflict labels**
During `hub commit` (`git pull --rebase`), git's side A is the version already on the remote and B is the local commit. That is the reverse of what "ours" suggests, so the labels become A/B and the guidance says which is which.
```python
# EDIT project-hub/hub/merge_item.py
# OLD
                print(f"hub-item conflict in {item_id}: {key}: ours={_show(av)} theirs={_show(bv)}",
# NEW
                print(f"hub-item conflict in {item_id}: {key}: A={_show(av)} B={_show(bv)}",
```
Also replace the module docstring's mention `ours=<A> theirs=<B>` with `A=<A> B=<B>`. That line begins with ``hub-item conflict in <id>: <key>: ours=<A> theirs=<B>`` near the top of `merge_item.py`.
```python
# EDIT project-hub/hub/gitsync.py
# OLD
CONFLICT_GUIDANCE = ("re-run after fixing: `git pull --rebase`, fix the file, `git add` it, "
                     "`git rebase --continue`, then `hub validate` and `hub commit`")
# NEW
CONFLICT_GUIDANCE = ("in `hub-item conflict` lines, A is the version already on the remote and B is yours. "
                     "Re-run after fixing: `git pull --rebase`, fix the file, `git add` it, "
                     "`git rebase --continue`, then `hub validate` and `hub commit`")
```
In `project-hub/tests/test_merge_item.py`, change the two assertions:
```python
# EDIT project-hub/tests/test_merge_item.py
# OLD
    assert "hub-item conflict in Q-20261112-abc: title: ours=Title A theirs=Title B" in err
    assert "description: ours=Desc A theirs=Desc B" in err
# NEW
    assert "hub-item conflict in Q-20261112-abc: title: A=Title A B=Title B" in err
    assert "description: A=Desc A B=Desc B" in err
```
Search the tests for any other `ours=` or `theirs=` (e.g. `grep -rn "ours=" tests`) and update them the same way.

- [ ] **Step 5: Single-line untitled titles; unmatched owners stay in the title**
```python
# EDIT project-hub/hub/agenda/lifecycle.py
# OLD
                snippet = (row.as_decided or row.new_actions or row.outcome_text)[:UNTITLED_CHARS]
# NEW
                snippet = " ".join((row.as_decided or row.new_actions or row.outcome_text).split())[:UNTITLED_CHARS]
```
```python
# EDIT project-hub/hub/agenda/lifecycle.py
# OLD
def _action_ops(text, parent, series, source, people):
    return [{"op": "new", "type": "action", "title": t, "owner": owner_key(o, people), "series": [series],
             "links": {"spawned_by": [parent]}, "source": source} for t, o in parse_actions(text)]
# NEW
def _action_ops(text, parent, series, source, people):
    ops = []
    for title, raw in parse_actions(text):
        owner = owner_key(raw, people) if raw else None
        if raw and owner not in people:
            # "(due Friday)" is not a person: keep the parenthetical in the title, leave the owner empty.
            title, owner = f"{title} ({raw})", None
        ops.append({"op": "new", "type": "action", "title": title, "owner": owner, "series": [series],
                    "links": {"spawned_by": [parent]}, "source": source})
    return ops
```
In `project-hub/tests/test_agenda_lifecycle.py::test_action_owner_mapped_to_people_key`, change the expectation:
```python
# EDIT project-hub/tests/test_agenda_lifecycle.py
# OLD
        ("Rerun", "bob"), ("Plot", "alice"), ("Ask", "Bob Smith"), ("Mail", "carol")]
# NEW
        ("Rerun", "bob"), ("Plot", "alice"), ("Ask (Bob Smith)", None), ("Mail (carol)", None)]
```

- [ ] **Step 6: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest -q`
Expected: every test passes, including the 3 new ones.

- [ ] **Step 7: Update the agent docs and commit**
Update `.agent_docs` to match:
- `item-format.md`: add the `shape` rule to the validate rule list (condition: title not a non-empty string; links not a mapping of lists; series or sources not a list), and note that `Item.links` returns `[]` for wrong shapes;
- `git-sync.md`: the conflict line format is `A=… B=…`, where A is the remote version and B yours during `hub commit`;
- `agenda-workbook.md`: untitled snippets are whitespace-collapsed, and an unmatched `(owner)` stays in the action title with an empty owner.
```bash
git add project-hub/hub project-hub/tests project-hub/.agent_docs
git commit -m "fix(hub): tolerate wrong-shaped items, neutral merge labels, single-line meeting titles, unmatched owners stay in titles"
```

---
### Task 2: `source_seen` (idempotency)
**Files:**
- Create: `project-hub/hub/sources.py`
- Test: `project-hub/tests/test_sources.py`

**Interfaces:**
- Consumes: `Store` (Plan 1), `load_proposals` (Plan 1).
- Produces: `source_seen(root, store, source) -> list[str]`. It returns the item ids whose `sources` contain `source`, then `log:<YYYY-MM>` for each monthly log with an entry citing it, then the pending proposal ids whose `source` or op `source` is it. An empty list means the source is new.

- [ ] **Step 1: Write the failing test**
```python
# project-hub/tests/test_sources.py
from datetime import date

from hub.pending import add_proposal
from hub.sources import source_seen
from hub.store import Store

D = date(2026, 11, 12)


def test_source_seen(hub_root):
    s = Store(hub_root)
    q = s.create("question", "Q", owner="alice", author="alice", day=D, source="circleback:m1")
    s.log("Ran it.", author="alice", source="commit:ML@abc1234", day=D)
    pid = add_proposal(hub_root, "P", [{"op": "entry", "id": q.id, "text": "x", "source": "clickup:t9"}],
                       proposer="alice", source="clickup:t9", reason="r", day=D)
    assert source_seen(hub_root, s, "circleback:m1") == [q.id]
    assert source_seen(hub_root, s, "commit:ML@abc1234") == ["log:2026-11"]
    assert source_seen(hub_root, s, "clickup:t9") == [pid]
    assert source_seen(hub_root, s, "circleback:zzz") == []


def test_source_seen_does_not_match_prefixes(hub_root):
    s = Store(hub_root)
    s.log("Ran it.", author="alice", source="commit:ML@abc12345", day=D)
    assert source_seen(hub_root, s, "commit:ML@abc1234") == []
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_sources.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.sources'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/sources.py
"""Idempotency: has a source id (circleback:<id>, clickup:<id>, commit:<ROOT>@<sha>, …) already been recorded?

Ingest runs by different people overlap; every skill checks a source id here (CLI: `hub seen`) before
writing anything derived from it.
"""
from __future__ import annotations

from pathlib import Path

from .pending import load_proposals
from .store import Store


def source_seen(root, store: Store, source: str) -> list[str]:
    """Where `source` already appears: item ids listing it in `sources`, `log:<YYYY-MM>` for monthly logs
    with an entry citing it, and pending proposal ids whose body or ops cite it. Empty list = new."""
    where = [it.id for it in store.all() if source in (it.meta.get("sources") or [])]
    log_dir = Path(root) / "Status" / "Log"
    if log_dir.is_dir():
        needle = f" · {source} — "
        for p in sorted(log_dir.glob("*.md")):
            if needle in p.read_text(encoding="utf-8"):
                where.append(f"log:{p.stem}")
    for pr in load_proposals(root):
        cited = {pr.body.get("source")}
        cited |= {op.get("source") for op in pr.body.get("ops") or [] if isinstance(op, dict)}
        if source in cited:
            where.append(pr.pid)
    return where
```

- [ ] **Step 3: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_sources.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/sources.py project-hub/tests/test_sources.py
git commit -m "feat(hub): source_seen idempotency check"
```

---
### Task 3: Git and Overleaf ingest (`ingest_git`)
**Files:**
- Create: `project-hub/hub/ingest_git.py`
- Test: `project-hub/tests/test_ingest_git.py`

**Interfaces:**
- Consumes: `HubConfig` (`roots`, `sources`, `root`); `get_watermark` and `set_watermark` (Plan 1).
- Produces:
  - `IngestError(ValueError)`;
  - `watermark_name(root_name) -> str` (`"git.<ROOT>"`);
  - `resolve_root(config, name) -> Path`;
  - `configured_branch(config, name) -> str | None`;
  - `git_commits(path, branch, *, since_sha=None, since_date=None, limit=500) -> (list[dict], str | None)`, with commits oldest first, each `{sha, short, date, author, subject, files}`;
  - `ingest_git(config, name, *, today, since=None, advance=False) -> dict` with keys `root, path, branch, watermark_before, commits, warning, advanced`;
  - `DEFAULT_LOOKBACK_DAYS = 30`.

Configuration: `project.yaml` `sources.repos: [{root: <ROOT>, branch: <name>}]` and `sources.overleaf: {root: <ROOT>, branch: <name>}`. `branch` is optional and defaults to `HEAD`. `<ROOT>` is a key of `roots`, and the local path comes from `hub.local.yaml` `roots`.

- [ ] **Step 1: Write the failing tests**
```python
# project-hub/tests/test_ingest_git.py
import subprocess
import textwrap
from datetime import date

import pytest

from hub.config import load_config
from hub.ingest_git import IngestError, ingest_git
from hub.sync_state import get_watermark, set_watermark


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def commit_file(repo, name, msg):
    (repo / name).write_text(msg)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", msg)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    repo = tmp_path / "code"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Tester")
    for i in range(3):
        commit_file(repo, f"f{i}.txt", f"c{i}")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text(textwrap.dedent(f"""\
        name: demo
        people:
          alice: {{name: Alice}}
        series:
          group: {{attendees: [alice]}}
        roots:
          CODE: /nowhere
          NOTGIT: {tmp_path}
        sources:
          repos:
            - {{root: CODE, branch: main}}
        """))
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {repo}\n")
    return load_config(vault), repo, vault


def test_first_run_lists_all_oldest_first_and_advances(setup):
    cfg, repo, vault = setup
    res = ingest_git(cfg, "CODE", today=date.today(), advance=True)
    assert [c["subject"] for c in res["commits"]] == ["c0", "c1", "c2"]
    assert res["commits"][0]["files"] == ["f0.txt"]
    assert res["commits"][0]["author"] == "Tester" and len(res["commits"][0]["short"]) >= 7
    assert res["branch"] == "main" and res["watermark_before"] is None and res["advanced"] is True
    assert get_watermark(vault, "git.CODE") == res["commits"][-1]["sha"]


def test_second_run_lists_only_new_commits(setup):
    cfg, repo, vault = setup
    ingest_git(cfg, "CODE", today=date.today(), advance=True)
    commit_file(repo, "f3.txt", "c3")
    res = ingest_git(cfg, "CODE", today=date.today())
    assert [c["subject"] for c in res["commits"]] == ["c3"]
    assert res["advanced"] is False
    again = ingest_git(cfg, "CODE", today=date.today(), advance=True)
    assert [c["subject"] for c in again["commits"]] == ["c3"]
    assert ingest_git(cfg, "CODE", today=date.today())["commits"] == []


def test_unknown_watermark_falls_back_to_lookback(setup):
    cfg, repo, vault = setup
    set_watermark(vault, "git.CODE", "deadbeef" * 5)
    res = ingest_git(cfg, "CODE", today=date.today())
    assert "not found" in res["warning"]
    assert [c["subject"] for c in res["commits"]] == ["c0", "c1", "c2"]


def test_explicit_since_ignores_watermark(setup):
    cfg, repo, vault = setup
    ingest_git(cfg, "CODE", today=date.today(), advance=True)
    res = ingest_git(cfg, "CODE", today=date.today(), since=date(2000, 1, 1))
    assert len(res["commits"]) == 3


def test_errors(setup):
    cfg, repo, vault = setup
    with pytest.raises(IngestError, match="unknown root"):
        ingest_git(cfg, "NOPE", today=date.today())
    with pytest.raises(IngestError, match="not a git repository"):
        ingest_git(cfg, "NOTGIT", today=date.today())
    cfg.sources["repos"][0]["branch"] = "nosuch"
    with pytest.raises(IngestError, match="branch 'nosuch'"):
        ingest_git(cfg, "CODE", today=date.today())
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_ingest_git.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.ingest_git'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/ingest_git.py
"""Commits of a configured repository (code repo or Overleaf clone) since the last ingest.

The watermark `git.<ROOT>` (Status/sync-state/watermark.git.<ROOT>.yaml) holds the full sha of the newest
commit already ingested. Without a usable watermark the last DEFAULT_LOOKBACK_DAYS days are listed.
Configuration (project.yaml):
  roots: {<ROOT>: <path>}                       # local path overridden per user in hub.local.yaml
  sources:
    repos:    [{root: <ROOT>, branch: <name>}]  # branch optional, default HEAD
    overleaf: {root: <ROOT>, branch: <name>}
"""
from __future__ import annotations

import subprocess
from datetime import date, timedelta
from pathlib import Path

from .config import HubConfig
from .sync_state import get_watermark, set_watermark

DEFAULT_LOOKBACK_DAYS = 30
_FIELD = "\x1f"
_RECORD = "\x1e"


class IngestError(ValueError):
    """A source cannot be read as configured (unknown root, not a repository, missing branch, git failure)."""


def watermark_name(root_name: str) -> str:
    return f"git.{root_name}"


def resolve_root(config: HubConfig, name: str) -> Path:
    if name not in config.roots:
        raise IngestError(f"unknown root '{name}' (roots: {', '.join(sorted(config.roots)) or 'none'})")
    path = Path(str(config.roots[name])).expanduser()
    if not (path / ".git").exists():
        raise IngestError(f"root '{name}' ({path}) is not a git repository on this machine; "
                          f"set its local path under `roots` in hub.local.yaml")
    return path


def configured_branch(config: HubConfig, name: str) -> str | None:
    sources = config.sources or {}
    entries = list(sources.get("repos") or [])
    if isinstance(sources.get("overleaf"), dict):
        entries.append(sources["overleaf"])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("root") == name and entry.get("branch"):
            return str(entry["branch"])
    return None


def _git(path, *args):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True)


def git_commits(path: Path, branch: str | None, *, since_sha=None, since_date=None, limit=500):
    """Non-merge commits on `branch` (default HEAD) after `since_sha`, or since `since_date` when the sha is
    absent or unknown. Returns (commits oldest first, warning or None)."""
    ref = branch or "HEAD"
    if _git(path, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        raise IngestError(f"branch '{ref}' not found in {path}")
    warning = None
    if since_sha and _git(path, "cat-file", "-e", f"{since_sha}^{{commit}}").returncode != 0:
        warning = (f"watermark {since_sha} not found in {path} (history rewritten?); "
                   f"listed commits since {since_date} instead")
        since_sha = None
    args = ["log", "--no-merges", f"--max-count={limit}", "--date=short",
            f"--format={_RECORD}%H{_FIELD}%h{_FIELD}%ad{_FIELD}%an{_FIELD}%s", "--name-only"]
    if since_sha:
        args += [f"{since_sha}..{ref}"]
    else:
        if since_date:
            args.append(f"--since={since_date.isoformat()}")
        args.append(ref)
    res = _git(path, *args, "--")
    if res.returncode != 0:
        raise IngestError(f"git log failed in {path}: {res.stderr.strip()}")
    commits = []
    for block in res.stdout.split(_RECORD):
        block = block.strip("\n")
        if not block:
            continue
        head, *files = block.split("\n")
        sha, short, day, author, subject = head.split(_FIELD)
        commits.append({"sha": sha, "short": short, "date": day, "author": author, "subject": subject,
                        "files": [f for f in files if f.strip()]})
    commits.reverse()
    return commits, warning


def ingest_git(config: HubConfig, name: str, *, today: date, since: date | None = None,
               advance: bool = False) -> dict:
    """List commits since the watermark (or since `since`, which ignores the watermark). With `advance`,
    move the watermark to the newest listed commit; run it only after the commits have been recorded."""
    path = resolve_root(config, name)
    branch = configured_branch(config, name)
    wm = watermark_name(name)
    before = get_watermark(config.root, wm)
    since_date = since or (today - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    commits, warning = git_commits(path, branch, since_sha=None if since else before, since_date=since_date)
    advanced = bool(advance and commits)
    if advanced:
        set_watermark(config.root, wm, commits[-1]["sha"])
    return {"root": name, "path": str(path), "branch": branch or "HEAD", "watermark_before": before,
            "commits": commits, "warning": warning, "advanced": advanced}
```

- [ ] **Step 3: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_ingest_git.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/ingest_git.py project-hub/tests/test_ingest_git.py
git commit -m "feat(hub): list repository commits since the git.<ROOT> watermark"
```

---
### Task 4: Ingest batch files (`run_batch`)
A batch file is what an ingest skill writes after reading a source such as Circleback, notes, Gmail or Slack. The script decides nothing. It enforces the safe-vs-judgement split, de-duplicates, and writes.

**Files:**
- Create: `project-hub/hub/batch.py`
- Test: `project-hub/tests/test_batch.py`

**Interfaces:**
- Consumes: `Store`; `add_proposal`, `check_ops` and `load_proposals`; `load_state`, `save_state` and `set_watermark`.
- Produces:
  - `BatchError(ValueError)`;
  - `check_batch(store, batch) -> list[str]`;
  - `run_batch(root, store, batch, *, author, day) -> dict` with keys `applied, skipped, staged (list of P-ids), skipped_proposals, flags_added, watermark`.

Batch file schema (YAML):
```yaml
watermark: {name: circleback, value: "2026-11-19"}   # optional; set only after everything succeeded
safe:                                                # applied directly; each needs a source id
  - {op: entry, id: Q-20261112-abc, text: "…", source: circleback:abc123}
  - {op: log, text: "…", source: commit:ML@1a2b3c4}
  - {op: new, type: question, title: "…", source: circleback:abc123,
     owner: alice, series: [group], priority: 2, links: {spawned_by: [D-…]}, description: "…"}
proposals:                                           # staged as Status/Pending/<P-id>.md
  - {title: "…", reason: "…", source: circleback:abc123, ops: [ …pending ops… ]}
flags: ["…"]                                         # added to Status/sync-state/flags.md
```
Rules:
- A safe `new` may create only questions and actions, and its `fields` may not set `status` or `closed`. Decisions and closures are proposals.
- Links in safe ops must point at existing items; `$refs` are allowed only inside proposals.
- Re-running a batch is a no-op. These are all skipped:
  - an entry whose item already has an entry with the same source and the same whitespace-normalised text;
  - a log line already present with the same source and text;
  - a new item when an item of the same type and title already lists that source;
  - a proposal whose normalised title is already pending, or was already applied or rejected according to the log.

- [ ] **Step 1: Write the failing tests**
```python
# project-hub/tests/test_batch.py
from datetime import date

import pytest

from hub.batch import BatchError, check_batch, run_batch
from hub.pending import apply_proposals, load_proposals
from hub.store import Store
from hub.sync_state import get_watermark, load_state

D = date(2026, 11, 19)


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def base(store):
    q = store.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m")
    return q, {
        "watermark": {"name": "circleback", "value": "2026-11-19"},
        "safe": [
            {"op": "entry", "id": q.id, "text": "Discussed;  HVA plateaus at L=12.", "source": "circleback:m1"},
            {"op": "log", "text": "Group meeting: framing settled.", "source": "circleback:m1"},
            {"op": "new", "type": "action", "title": "Rerun HVA", "owner": "bob", "source": "circleback:m1",
             "series": ["group"]},
        ],
        "proposals": [{"title": "Close: Why?", "reason": "Answered in meeting.", "source": "circleback:m1",
                       "ops": [{"op": "set", "id": q.id, "fields": {"status": "answered", "closed": "2026-11-19"}},
                               {"op": "entry", "id": q.id, "text": "Answered.", "source": "circleback:m1"}]}],
        "flags": ["circleback: meeting m2 has no notes yet"],
    }


def test_run_batch_applies_safe_stages_proposals(hub_root, store):
    q, batch = base(store)
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["applied"] == 3 and rep["skipped"] == 0 and len(rep["staged"]) == 1
    assert rep["flags_added"] == 1 and rep["watermark"] == "circleback=2026-11-19"
    assert store.get(q.id).entries[-1].text == "Discussed; HVA plateaus at L=12."
    assert [a.title for a in store.query(type="action")] == ["Rerun HVA"]
    assert "Group meeting: framing settled." in (hub_root / "Status" / "Log" / "2026-11.md").read_text()
    assert [p.title for p in load_proposals(hub_root)] == ["Close: Why?"]
    assert get_watermark(hub_root, "circleback") == "2026-11-19"
    assert load_state(hub_root)["flags"] == ["circleback: meeting m2 has no notes yet"]


def test_rerun_is_a_no_op(hub_root, store):
    q, batch = base(store)
    run_batch(hub_root, store, batch, author="alice", day=D)
    rep = run_batch(hub_root, store, batch, author="bob", day=D)
    assert rep["applied"] == 0 and rep["skipped"] == 3 and rep["staged"] == [] and rep["skipped_proposals"] == 1
    assert rep["flags_added"] == 0
    assert len(store.get(q.id).entries) == 2 and len(store.query(type="action")) == 1


def test_applied_proposal_is_not_staged_again(hub_root, store):
    q, batch = base(store)
    pid = run_batch(hub_root, store, batch, author="alice", day=D)["staged"][0]
    apply_proposals(hub_root, store, [pid], author="alice", day=D)
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["staged"] == [] and rep["skipped_proposals"] == 1


@pytest.mark.parametrize("mutate,needle", [
    (lambda b, q: b["safe"].append({"op": "new", "type": "decision", "title": "X", "source": "s"}), "decisions are proposals"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "fields": {"status": "done"}}), "without status"),
    (lambda b, q: b["safe"].append({"op": "entry", "id": "Q-20990101-zzz", "text": "x", "source": "s"}), "does not exist"),
    (lambda b, q: b["safe"].append({"op": "log", "text": "x", "source": "has space"}), "no whitespace"),
    (lambda b, q: b["safe"].append({"op": "new", "type": "action", "title": "X", "source": "s",
                                    "links": {"spawned_by": ["$d"]}}), "does not exist"),
    (lambda b, q: b["proposals"].append({"title": "T", "reason": "r", "source": "s",
                                         "ops": [{"op": "set", "id": "Q-20990101-zzz", "fields": {}}]}), "does not exist"),
    (lambda b, q: b.update(extra=1), "unknown top-level key"),
    (lambda b, q: b.update(safe={"op": "log"}), "safe must be a list"),
    (lambda b, q: b.update(watermark={"name": "circleback"}), "watermark must be"),
])
def test_invalid_batches_write_nothing(hub_root, store, mutate, needle):
    q, batch = base(store)
    mutate(batch, q)
    assert any(needle in e for e in check_batch(store, batch))
    with pytest.raises(BatchError, match="nothing written"):
        run_batch(hub_root, store, batch, author="alice", day=D)
    assert len(store.get(q.id).entries) == 1 and load_proposals(hub_root) == []
    assert not (hub_root / "Status" / "Log").exists()
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_batch.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.batch'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/batch.py
"""`hub ingest batch FILE`: apply an agent-written ingest file — safe changes directly, judgement calls staged.

File schema (YAML), every op carrying a source id (no whitespace):
  watermark: {name: <sync-state key>, value: <string>}   # optional; set only after everything succeeded
  safe:                                                  # applied directly
    - {op: entry, id: <item id>, text: <text>, source: <id>}
    - {op: log, text: <text>, source: <id>}
    - {op: new, type: question|action, title: <t>, source: <id>,
       owner?, series? (list), priority? (1-3), links? ({rel: [ids]}), description?, fields? (no status/closed)}
  proposals:                                             # staged as Status/Pending/<P-id>.md
    - {title: <t>, reason: <why a human decides>, source: <id>, ops: [<pending ops, see pending-format.md>]}
  flags: [<text>, ...]                                   # added to Status/sync-state/flags.md (duplicates ignored)
The whole file is checked before anything is written. Re-running the same file is a no-op.
"""
from __future__ import annotations

import re
from pathlib import Path

from .pending import add_proposal, check_ops, load_proposals
from .store import Store
from .sync_state import load_state, save_state, set_watermark

SAFE_REQUIRED = {"entry": ("id", "text", "source"), "log": ("text", "source"), "new": ("type", "title", "source")}
SAFE_NEW_TYPES = ("question", "action")
FORBIDDEN_NEW_FIELDS = ("status", "closed")
TOP_KEYS = ("watermark", "safe", "proposals", "flags")
DECIDED_RE = re.compile(r" — (?:Applied|Rejected) P-[a-z0-9]{4}: (.*)$")


class BatchError(ValueError):
    """The ingest file is malformed; nothing was written."""


def _norm(text) -> str:
    return " ".join(str(text).split())


def _token(value) -> bool:
    return isinstance(value, str) and value != "" and not any(c.isspace() for c in value)


def _check_safe(store: Store, safe: list, errs: list[str]) -> None:
    existing = store.ids()
    for i, op in enumerate(safe):
        where = f"safe {i + 1}"
        if not isinstance(op, dict) or op.get("op") not in SAFE_REQUIRED:
            errs.append(f"{where}: must be a mapping with op entry|log|new")
            continue
        missing = [k for k in SAFE_REQUIRED[op["op"]] if not op.get(k)]
        if missing:
            errs.append(f"{where} ({op['op']}): missing {', '.join(missing)}")
            continue
        if not _token(op["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {op['source']!r}")
        if op["op"] == "entry" and op["id"] not in existing:
            errs.append(f"{where}: item {op['id']} does not exist")
        if op["op"] != "new":
            continue
        if op["type"] not in SAFE_NEW_TYPES:
            errs.append(f"{where}: safe `new` may only create {' or '.join(SAFE_NEW_TYPES)}; decisions are proposals")
        fields = op.get("fields") or {}
        if not isinstance(fields, dict) or any(k in fields for k in FORBIDDEN_NEW_FIELDS):
            errs.append(f"{where}: fields must be a mapping without status or closed (closures are proposals)")
        if "series" in op and not isinstance(op["series"], list):
            errs.append(f"{where}: series must be a list")
        links = op.get("links") or {}
        if not isinstance(links, dict):
            errs.append(f"{where}: links must be a mapping of relation -> list of ids")
            continue
        for rel, targets in links.items():
            if not isinstance(targets, list):
                errs.append(f"{where}: links.{rel} must be a list")
                continue
            for t in targets:
                if t not in existing:
                    errs.append(f"{where}: links.{rel} -> {t} does not exist")


def check_batch(store: Store, batch) -> list[str]:
    """Every problem with a batch, as readable lines (empty = fine). Nothing is written."""
    if not isinstance(batch, dict):
        return ["batch file must be a mapping"]
    errs = [f"unknown top-level key {k!r} (expected {', '.join(TOP_KEYS)})" for k in batch if k not in TOP_KEYS]
    for key in ("safe", "proposals", "flags"):
        if batch.get(key) is not None and not isinstance(batch[key], list):
            errs.append(f"{key} must be a list")
    if errs:
        return errs
    _check_safe(store, batch.get("safe") or [], errs)
    for i, pr in enumerate(batch.get("proposals") or []):
        where = f"proposal {i + 1}"
        if not isinstance(pr, dict) or not all(pr.get(k) for k in ("title", "reason", "source", "ops")):
            errs.append(f"{where}: needs title, reason, source and ops")
            continue
        if not _token(pr["source"]):
            errs.append(f"{where}: source must be non-empty with no whitespace, got {pr['source']!r}")
        errs += [f"{where}: {e}" for e in check_ops(store, pr["ops"])]
    if not all(isinstance(f, str) and f.strip() for f in batch.get("flags") or []):
        errs.append("flags must be non-empty strings")
    wm = batch.get("watermark")
    if wm is not None and not (isinstance(wm, dict) and _token(str(wm.get("name") or ""))
                               and wm.get("value") not in (None, "")):
        errs.append("watermark must be {name: <sync-state key>, value: <non-empty>}")
    return errs


def _log_text(root) -> str:
    d = Path(root) / "Status" / "Log"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))) if d.is_dir() else ""


def run_batch(root, store: Store, batch, *, author, day) -> dict:
    errs = check_batch(store, batch)
    if errs:
        raise BatchError("nothing written; fix the batch file:\n  " + "\n  ".join(errs))
    report = {"applied": 0, "skipped": 0, "staged": [], "skipped_proposals": 0, "flags_added": 0,
              "watermark": None}
    logs = _log_text(root)
    items = store.all()
    for op in batch.get("safe") or []:
        kind, source = op["op"], op["source"]
        if kind == "entry":
            if any(e.source == source and _norm(e.text) == _norm(op["text"]) for e in store.get(op["id"]).entries):
                report["skipped"] += 1
                continue
            store.entry(op["id"], op["text"], author=author, source=source, day=day)
        elif kind == "log":
            line = f" · {source} — {_norm(op['text'])}"
            if line in logs:
                report["skipped"] += 1
                continue
            store.log(op["text"], author=author, source=source, day=day)
            logs += "\n" + line
        else:
            if any(i.type == op["type"] and i.title == op["title"] and source in (i.meta.get("sources") or [])
                   for i in items):
                report["skipped"] += 1
                continue
            items.append(store.create(
                op["type"], op["title"], owner=op.get("owner"), author=author, day=day, source=source,
                series=list(op.get("series") or []), priority=op.get("priority", 2),
                links={rel: list(t) for rel, t in (op.get("links") or {}).items()},
                description=op.get("description", ""), fields=op.get("fields")))
        report["applied"] += 1
    pending = {_norm(p.title) for p in load_proposals(root)}
    decided = {_norm(m[1]) for line in _log_text(root).splitlines() if (m := DECIDED_RE.search(line))}
    for pr in batch.get("proposals") or []:
        title = _norm(pr["title"])
        if title in pending or title in decided:
            report["skipped_proposals"] += 1
            continue
        report["staged"].append(add_proposal(root, title, pr["ops"], proposer=author, source=pr["source"],
                                             reason=pr["reason"], day=day))
        pending.add(title)
    flags = [_norm(f) for f in batch.get("flags") or []]
    if flags:
        state = load_state(root)
        for f in flags:
            if f not in state["flags"]:
                state["flags"].append(f)
                report["flags_added"] += 1
        save_state(root, state)
    wm = batch.get("watermark")
    if wm:
        set_watermark(root, str(wm["name"]), str(wm["value"]))
        report["watermark"] = f"{wm['name']}={wm['value']}"
    return report
```

- [ ] **Step 3: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_batch.py -v`
Expected: 12 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/batch.py project-hub/tests/test_batch.py
git commit -m "feat(hub): ingest batch files (safe ops, proposals, flags, watermark), idempotent"
```

---
### Task 5: ClickUp reconciliation
**Files:**
- Create: `project-hub/hub/clickup.py`
- Test: `project-hub/tests/test_clickup.py`

**Interfaces:**
- Consumes: `Store`, `HubConfig`; `add_proposal` and `load_proposals`; `load_state` and `save_state`.
- Produces:
  - `ClickUpError(ValueError)`;
  - `FLAG_PREFIX = "clickup: "`;
  - `DEFAULT_DONE = ("complete", "completed", "closed", "done")`;
  - `load_tasks(path) -> list[dict]`, normalised to `{id, name, status, assignees: [[alias, …], …], due: "YYYY-MM-DD" | None}`;
  - `plan_clickup(store, config, tasks, *, today) -> dict` with keys `set` (`[{id, fields, entry, source}]`), `proposals` (`[{title, reason, source, ops}]`), `flags` (`[str]`) and `push` (`[{id, title, owner, assignee, due}]`);
  - `apply_clickup(root, store, plan, *, author, day) -> dict` with keys `updated, staged, flags, push`.

Configuration:
- `project.yaml` `sources.clickup: {list_id: "<id>", done_statuses: [<status names>]}`. `done_statuses` is optional and defaults to `DEFAULT_DONE`.
- `people.<key>.clickup` holds that person's ClickUp user id. An assignee is matched case-insensitively to a people key by `clickup`, `email`, `name` or the key itself.

Rules, for each hub action with a `clickup_id`:

| Condition | Result |
|---|---|
| Task missing from the export | flag |
| Task done (status in `done_statuses`), action open | set `status: done`, `closed: today`, plus the entry "Closed in ClickUp (status '<s>')." with source `clickup:<id>` |
| Task open, action done or dropped | flag (the hub does not reopen) |
| First assignee maps to a people key different from the owner | set `owner` |
| First assignee does not map | flag |
| Task `due` differs | set `due` (empty clears it) |

Tasks in the export with no linked action and not done become a proposal "New ClickUp task: <name>", a new action with `clickup_id` (and `due`), unless a proposal with that title is already pending. Open actions with an owner and no `clickup_id` are listed in `push`; the skill creates their tasks. `apply_clickup` replaces every existing flag that starts with `clickup: ` with the plan's flags.

- [ ] **Step 1: Write the failing tests**
```python
# project-hub/tests/test_clickup.py
import json
import textwrap
from datetime import date

import pytest

from hub.clickup import ClickUpError, apply_clickup, load_tasks, plan_clickup
from hub.config import load_config
from hub.pending import load_proposals
from hub.store import Store
from hub.sync_state import load_state, save_state
from hub.validate import validate

D = date(2026, 11, 19)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    (tmp_path / "project.yaml").write_text(textwrap.dedent("""\
        name: demo
        people:
          alice: {name: Alice, clickup: "111", email: alice@x.edu}
          bob: {name: Bob, clickup: "222"}
        series:
          group: {attendees: [alice, bob]}
        sources:
          clickup: {list_id: "900", done_statuses: [complete, closed]}
        """))
    (tmp_path / "hub.local.yaml").write_text("me: alice\n")
    store, cfg = Store(tmp_path), load_config(tmp_path)

    def action(title, owner="alice", **fields):
        return store.create("action", title, owner=owner, author="alice", day=date(2026, 11, 1), source="m",
                            fields=fields)
    return store, cfg, tmp_path, action


def write_tasks(root, tasks):
    p = root / "tasks.json"
    p.write_text(json.dumps(tasks))
    return load_tasks(p)


def test_load_tasks_normalises(env):
    store, cfg, root, action = env
    tasks = write_tasks(root, [{"id": "t1", "name": "Do", "status": {"status": "in progress"},
                                "assignees": [{"id": 111, "username": "Alice A", "email": "a@x"}, "bob"],
                                "due": "2026-11-30T00:00:00"}])
    assert tasks == [{"id": "t1", "name": "Do", "status": "in progress",
                      "assignees": [["111", "Alice A", "a@x"], ["bob"]], "due": "2026-11-30"}]
    for bad in ({"name": "x"}, [{"id": "t", "name": "x", "due": "soon"}]):
        (root / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ClickUpError):
            load_tasks(root / "bad.json")


def test_mirror_status_owner_due(env):
    store, cfg, root, action = env
    a1 = action("Rerun", clickup_id="t1", due="2026-11-20")
    a2 = action("Plot", clickup_id="t2")
    tasks = write_tasks(root, [
        {"id": "t1", "name": "Rerun", "status": "complete", "assignees": [{"id": "222"}], "due": None},
        {"id": "t2", "name": "Plot", "status": "open", "assignees": [], "due": "2026-12-01"}])
    plan = plan_clickup(store, cfg, tasks, today=D)
    # item ids carry a random suffix, so store order is arbitrary: compare order-independently
    assert sorted(plan["set"], key=lambda s: s["source"]) == [
        {"id": a1.id, "fields": {"status": "done", "closed": "2026-11-19", "owner": "bob", "due": ""},
         "entry": "Closed in ClickUp (status 'complete').", "source": "clickup:t1"},
        {"id": a2.id, "fields": {"due": "2026-12-01"}, "entry": None, "source": "clickup:t2"}]
    assert plan["flags"] == [] and plan["proposals"] == [] and plan["push"] == []
    rep = apply_clickup(root, store, plan, author="alice", day=D)
    assert sorted(rep["updated"]) == sorted([a1.id, a2.id])
    got = store.get(a1.id)
    assert got.status == "done" and got.owner == "bob" and got.meta["due"] is None
    assert store.get(a2.id).meta["due"] == date(2026, 12, 1)
    assert validate(store, cfg) == []


def test_flags_proposals_push(env):
    store, cfg, root, action = env
    gone = action("Gone", clickup_id="t9")
    done = action("Finished", clickup_id="t3")
    store.entry(done.id, "Done.", author="alice", source="m", day=D)
    store.set_fields(done.id, {"status": "done", "closed": "2026-11-19"})
    stranger = action("Mine", clickup_id="t4")
    local = action("Local only", owner="bob", due="2026-11-25")
    tasks = write_tasks(root, [
        {"id": "t3", "name": "Finished", "status": "open", "assignees": []},
        {"id": "t4", "name": "Mine", "status": "open", "assignees": [{"username": "zed"}]},
        {"id": "t5", "name": "New  from ClickUp", "status": "open", "assignees": [{"email": "ALICE@x.edu"}],
         "due": "2026-12-05"},
        {"id": "t6", "name": "Old closed", "status": "closed", "assignees": []}])
    plan = plan_clickup(store, cfg, tasks, today=D)
    assert sorted(plan["flags"]) == sorted([
        f"clickup: task t9 for {gone.id} (Gone) is not in the export (deleted or moved?)",
        f"clickup: {done.id} is done in the hub but task t3 is 'open' in ClickUp",
        "clickup: assignee 'zed' of task t4 is not in project.yaml people (add their ClickUp id as `clickup:`)"])
    assert plan["proposals"] == [{
        "title": "New ClickUp task: New from ClickUp", "source": "clickup:t5",
        "reason": "Task created directly in ClickUp; link it (spawned_by) to the decision it serves, then apply.",
        "ops": [{"op": "new", "type": "action", "title": "New  from ClickUp", "owner": "alice",
                 "fields": {"clickup_id": "t5", "due": "2026-12-05"}, "source": "clickup:t5"}]}]
    assert plan["push"] == [{"id": local.id, "title": "Local only", "owner": "bob", "assignee": "222",
                             "due": "2026-11-25"}]
    state = load_state(root)
    state["flags"] = ["clickup: stale", "other: keep"]
    save_state(root, state)
    rep = apply_clickup(root, store, plan, author="alice", day=D)
    assert len(rep["staged"]) == 1 and rep["push"] == plan["push"]
    assert load_state(root)["flags"] == ["other: keep"] + plan["flags"]
    assert plan_clickup(store, cfg, tasks, today=D)["proposals"] == []   # already pending
    assert [p.title for p in load_proposals(root)] == ["New ClickUp task: New from ClickUp"]
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_clickup.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.clickup'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/clickup.py
"""Reconcile a ClickUp list export with the hub's actions (ClickUp owns task status, assignee and due date).

Input: a JSON (or YAML) list the agent exported from the list with the ClickUp connector:
  [{id, name, status: <str | {status: str}>, assignees: [<id | username | email | {id, username, email}>],
    due: YYYY-MM-DD | null}]
Configuration: project.yaml `sources.clickup: {list_id, done_statuses?}`; `people.<key>.clickup` = user id.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from .pending import add_proposal, load_proposals
from .store import Store
from .sync_state import load_state, save_state

DEFAULT_DONE = ("complete", "completed", "closed", "done")
FLAG_PREFIX = "clickup: "


class ClickUpError(ValueError):
    """The task export is malformed."""


def _due(task_id, value):
    if value in (None, ""):
        return None
    text = value.isoformat() if isinstance(value, date) else str(value)
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ClickUpError(f"task {task_id}: due must be YYYY-MM-DD or null, got {value!r}") from exc


def load_tasks(path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ClickUpError("the task export must be a list of tasks")
    tasks = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict) or not t.get("id") or not t.get("name"):
            raise ClickUpError(f"task {i + 1}: needs id and name")
        status = t.get("status")
        if isinstance(status, dict):
            status = status.get("status")
        assignees = []
        for a in t.get("assignees") or []:
            if isinstance(a, dict):
                assignees.append([str(a[k]) for k in ("id", "username", "email") if a.get(k) not in (None, "")])
            else:
                assignees.append([str(a)])
        tasks.append({"id": str(t["id"]), "name": str(t["name"]), "status": str(status or ""),
                      "assignees": [al for al in assignees if al], "due": _due(t["id"], t.get("due"))})
    return tasks


def _people_index(people: dict) -> dict[str, str]:
    idx = {}
    for key, person in people.items():
        idx[str(key).casefold()] = key
        if isinstance(person, dict):
            for field in ("clickup", "email", "name"):
                if person.get(field) not in (None, ""):
                    idx[str(person[field]).casefold()] = key
    return idx


def _owner_of(aliases, idx):
    for alias in aliases:
        if alias.casefold() in idx:
            return idx[alias.casefold()]
    return None


def _iso(value):
    return value.isoformat() if isinstance(value, date) else None


def plan_clickup(store: Store, config, tasks: list[dict], *, today: date) -> dict:
    conf = (config.sources or {}).get("clickup") or {}
    done = tuple(str(s).casefold() for s in (conf.get("done_statuses") or DEFAULT_DONE))
    idx = _people_index(config.people)
    by_id = {t["id"]: t for t in tasks}
    plan: dict = {"set": [], "proposals": [], "flags": [], "push": []}
    linked: set[str] = set()
    for a in store.query(type="action"):
        cid = a.meta.get("clickup_id")
        if not cid:
            if a.status == "open" and a.owner:
                person = config.people.get(a.owner)
                plan["push"].append({"id": a.id, "title": a.title, "owner": a.owner,
                                     "assignee": person.get("clickup") if isinstance(person, dict) else None,
                                     "due": _iso(a.meta.get("due"))})
            continue
        cid = str(cid)
        linked.add(cid)
        t = by_id.get(cid)
        if t is None:
            plan["flags"].append(f"{FLAG_PREFIX}task {cid} for {a.id} ({a.title}) is not in the export "
                                 f"(deleted or moved?)")
            continue
        fields: dict = {}
        entry = None
        task_done = t["status"].casefold() in done
        if task_done and a.status == "open":
            fields.update(status="done", closed=today.isoformat())
            entry = f"Closed in ClickUp (status '{t['status']}')."
        elif not task_done and a.status in ("done", "dropped"):
            plan["flags"].append(f"{FLAG_PREFIX}{a.id} is {a.status} in the hub but task {cid} is "
                                 f"'{t['status']}' in ClickUp")
        if t["assignees"]:
            owner = _owner_of(t["assignees"][0], idx)
            if owner is None:
                plan["flags"].append(f"{FLAG_PREFIX}assignee '{t['assignees'][0][0]}' of task {cid} is not in "
                                     f"project.yaml people (add their ClickUp id as `clickup:`)")
            elif owner != a.owner:
                fields["owner"] = owner
        if t["due"] != _iso(a.meta.get("due")):
            fields["due"] = t["due"] or ""
        if fields:
            plan["set"].append({"id": a.id, "fields": fields, "entry": entry, "source": f"clickup:{cid}"})
    pending = {p.title for p in load_proposals(config.root)}
    for t in tasks:
        if t["id"] in linked or t["status"].casefold() in done:
            continue
        title = f"New ClickUp task: {' '.join(t['name'].split())}"
        if title in pending:
            continue
        fields = {"clickup_id": t["id"]}
        if t["due"]:
            fields["due"] = t["due"]
        plan["proposals"].append({
            "title": title, "source": f"clickup:{t['id']}",
            "reason": "Task created directly in ClickUp; link it (spawned_by) to the decision it serves, then apply.",
            "ops": [{"op": "new", "type": "action", "title": t["name"],
                     "owner": _owner_of(t["assignees"][0], idx) if t["assignees"] else None,
                     "fields": fields, "source": f"clickup:{t['id']}"}]})
    return plan


def apply_clickup(root, store: Store, plan: dict, *, author, day) -> dict:
    for s in plan["set"]:
        store.set_fields(s["id"], s["fields"])
        if s.get("entry"):
            store.entry(s["id"], s["entry"], author=author, source=s["source"], day=day)
    staged = [add_proposal(root, p["title"], p["ops"], proposer=author, source=p["source"], reason=p["reason"],
                           day=day) for p in plan["proposals"]]
    state = load_state(root)
    state["flags"] = [f for f in state["flags"] if not f.startswith(FLAG_PREFIX)] + list(plan["flags"])
    save_state(root, state)
    return {"updated": [s["id"] for s in plan["set"]], "staged": staged, "flags": plan["flags"],
            "push": plan["push"]}
```

- [ ] **Step 3: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_clickup.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/clickup.py project-hub/tests/test_clickup.py
git commit -m "feat(hub): ClickUp list reconciliation (mirror, flags, proposals, push list)"
```

---
### Task 6: CLI commands and agent docs
**Files:**
- Create: `project-hub/hub/cli_ext.py`, `project-hub/tests/test_cli_ext.py`, `project-hub/.agent_docs/ingest.md`
- Modify: `project-hub/hub/cli.py` (import, two hooks, docstring), `project-hub/.agent_docs/_INDEX.md`, `project-hub/.agent_docs/vault-layout.md`

**Interfaces:**
- Consumes: everything from Tasks 2–5; `load_state` and `save_state`.
- Produces:
  - `add_ext_parsers(sub)`;
  - `dispatch_ext(args, cfg, store, *, today, me) -> int | None`, which returns None for commands it does not own;
  - the commands `hub seen SOURCE`, `hub ingest git ROOT [--since D] [--advance]`, `hub ingest batch FILE`, `hub clickup plan|apply --tasks FILE`, `hub meeting next SERIES DATE`, `hub meeting decided SERIES DATE [--json]` and `hub sources [--json]`.

- [ ] **Step 1: Write the failing tests**
```python
# project-hub/tests/test_cli_ext.py
import json
import subprocess

import pytest
import yaml

from hub.cli import main
from hub.store import Store
from hub.sync_state import load_state


@pytest.fixture
def cli(hub_root, monkeypatch):
    monkeypatch.chdir(hub_root)
    monkeypatch.delenv("HUB_ME", raising=False)
    monkeypatch.setenv("HUB_TODAY", "2026-11-19")
    return hub_root


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_seen(cli, capsys):
    run(capsys, "log", "Ran.", "--source", "commit:CODE@abc1234")
    assert run(capsys, "seen", "commit:CODE@abc1234")[1].strip() == "seen log:2026-11"
    assert run(capsys, "seen", "circleback:none")[1].strip() == "new"


def test_ingest_batch(cli, capsys):
    q = Store(cli).create("question", "Why?", owner="alice", author="alice",
                          day=__import__("datetime").date(2026, 11, 1), source="m")
    f = cli / "batch.yaml"
    f.write_text(yaml.safe_dump({"safe": [{"op": "entry", "id": q.id, "text": "Noted.", "source": "circleback:m1"}],
                                 "watermark": {"name": "circleback", "value": "2026-11-19"}}))
    code, out, _ = run(capsys, "ingest", "batch", str(f))
    assert code == 0 and "applied 1, skipped 0, staged 0" in out
    code, out, _ = run(capsys, "ingest", "batch", str(f))
    assert "applied 0, skipped 1" in out
    f.write_text(yaml.safe_dump({"safe": [{"op": "new", "type": "decision", "title": "X", "source": "s"}]}))
    code, _, err = run(capsys, "ingest", "batch", str(f))
    assert code == 1 and "decisions are proposals" in err


def test_ingest_git(cli, capsys, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@x"], ["config", "user.name", "T"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    (repo / "a.txt").write_text("a")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=repo, check=True)
    (cli / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {repo}\n")
    code, out, _ = run(capsys, "ingest", "git", "CODE", "--advance")
    res = json.loads(out)
    assert code == 0 and [c["subject"] for c in res["commits"]] == ["first"] and res["advanced"] is True
    code, _, err = run(capsys, "ingest", "git", "NOPE")
    assert code == 1 and "unknown root" in err


def test_clickup_plan_and_apply(cli, capsys):
    a = Store(cli).create("action", "Rerun", owner="alice", author="alice",
                          day=__import__("datetime").date(2026, 11, 1), source="m", fields={"clickup_id": "t1"})
    f = cli / "tasks.json"
    f.write_text(json.dumps([{"id": "t1", "name": "Rerun", "status": "complete", "assignees": []}]))
    code, out, _ = run(capsys, "clickup", "plan", "--tasks", str(f))
    assert code == 0 and json.loads(out)["set"][0]["id"] == a.id
    code, out, _ = run(capsys, "clickup", "apply", "--tasks", str(f))
    assert code == 0 and json.loads(out)["updated"] == [a.id]
    assert Store(cli).get(a.id).status == "done"


def test_meeting_next_and_decided(cli, capsys):
    assert run(capsys, "meeting", "next", "group", "2026-11-26")[0] == 0
    assert str(load_state(cli)["next_meeting"]["group"]) == "2026-11-26"
    code, _, err = run(capsys, "meeting", "next", "nosuch", "2026-11-26")
    assert code == 1 and "unknown series" in err
    s = Store(cli)
    d = s.create("decision", "Adopt Y", owner="alice", author="alice",
                 day=__import__("datetime").date(2026, 11, 1), source="m",
                 fields={"status": "made", "meeting": "group/2026-11-19", "as_decided": "Only 4x3"})
    s.create("action", "Do it", owner="bob", author="alice", day=__import__("datetime").date(2026, 11, 19),
             source="m", links={"spawned_by": [d.id]})
    code, out, _ = run(capsys, "meeting", "decided", "group", "2026-11-19", "--json")
    rows = json.loads(out)
    assert rows == [{"id": d.id, "title": "Adopt Y", "status": "made", "as_decided": "Only 4x3",
                     "actions": [s.query(type="action")[0].id]}]


def test_sources(cli, capsys):
    (cli / "project.yaml").write_text((cli / "project.yaml").read_text() + (
        "sources:\n  clickup: {list_id: '900'}\n  circleback: {folder: Proj}\n"
        "  repos: [{root: CODE, branch: main}]\n  calendar: {}\n"))
    run(capsys, "sync-state", "set", "clickup", "2026-11-18T10:00")
    code, out, _ = run(capsys, "sources", "--json")
    rows = json.loads(out)
    assert [r["source"] for r in rows] == ["clickup", "circleback", "git.CODE", "calendar"]
    assert rows[0]["watermark"] == "2026-11-18T10:00" and rows[1]["watermark"] is None
    code, out, _ = run(capsys, "sources")
    assert code == 0 and "git.CODE" in out
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_cli_ext.py -v`
Expected: FAIL. The new subcommands are unknown, so argparse exits with code 2 (`SystemExit: 2`).

- [ ] **Step 2: Implement `cli_ext.py`**
```python
# project-hub/hub/cli_ext.py
"""Plan 2 commands (ingest helpers and meeting/source bookkeeping), hooked into hub/cli.py."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

from .batch import run_batch
from .clickup import apply_clickup, load_tasks, plan_clickup
from .ingest_git import ingest_git
from .sources import source_seen
from .sync_state import load_state, save_state


def _sp(sub, name, help, description):
    return sub.add_parser(name, help=help, description=description,
                          formatter_class=argparse.RawDescriptionHelpFormatter)


def add_ext_parsers(sub) -> None:
    s = _sp(sub, "seen", "check whether a source id was already recorded",
            "Print `new` if SOURCE appears nowhere in the vault, else `seen` followed by where it appears: item "
            "ids that list it in `sources`, log:<YYYY-MM> for monthly logs citing it, and pending proposal ids "
            "citing it. Always exits 0. Use before writing anything derived from a source.")
    s.add_argument("source", help="source id exactly as written in entries, e.g. circleback:<meeting id>, "
                                  "clickup:<task id>, commit:<ROOT>@<7-char sha>")

    ing = _sp(sub, "ingest", "read a source into the vault",
              "Ingest helpers used by the hub-status skill: `git` lists new commits of a repository, `batch` "
              "applies an agent-written ingest file.")
    isub = ing.add_subparsers(dest="icmd", required=True, metavar="SUBCOMMAND")
    g = _sp(isub, "git", "list commits of a configured repository since the last ingest (JSON)",
            "Print JSON {root, path, branch, watermark_before, commits: [{sha, short, date, author, subject, "
            "files}] oldest first, warning, advanced}. ROOT is a key of `roots` (local path from "
            "hub.local.yaml); the branch comes from sources.repos[]/sources.overleaf `branch` (default HEAD). "
            "Without a usable watermark git.<ROOT>, the last 30 days are listed.")
    g.add_argument("root_name", metavar="ROOT", help="a key of `roots` in project.yaml/hub.local.yaml")
    g.add_argument("--since", type=date.fromisoformat, metavar="YYYY-MM-DD",
                   help="list commits since this date instead of since the watermark")
    g.add_argument("--advance", action="store_true",
                   help="after listing, move the watermark git.<ROOT> to the newest listed commit; use only "
                        "once the commits have been recorded (e.g. via `hub ingest batch`)")
    b = _sp(isub, "batch", "apply an ingest batch file: safe changes directly, judgement calls as proposals",
            "Apply a YAML batch file with keys `safe` (entry/log/new ops applied directly), `proposals` "
            "(staged in Status/Pending/), `flags` (added to Status/sync-state/flags.md) and `watermark` "
            "({name, value}, set last). The whole file is checked first; on any problem nothing is written "
            "(exit 1). Re-running the same file is a no-op. Format: .agent_docs/ingest.md. Prints "
            "`applied N, skipped N, staged N (P-ids), skipped proposals N, flags added N`.")
    b.add_argument("file", type=Path, help="path to the batch YAML file")

    c = _sp(sub, "clickup", "reconcile a ClickUp list export with the hub's actions",
            "ClickUp owns task status, assignee and due date. `plan` prints what would change (JSON keys "
            "set, proposals, flags, push); `apply` applies it: mirrors fields onto actions with clickup_id, "
            "stages tasks created in ClickUp as proposals, replaces `clickup: ` flags, and prints JSON "
            "{updated, staged, flags, push}. `push` lists open owned actions with no ClickUp task yet.")
    csub = c.add_subparsers(dest="ccmd", required=True, metavar="SUBCOMMAND")
    for name, helptext in (("plan", "show the reconciliation without writing"),
                           ("apply", "apply the reconciliation")):
        cp = _sp(csub, name, helptext, f"{helptext.capitalize()}. Input: --tasks, the list export as JSON "
                                       "[{id, name, status, assignees, due}] (see .agent_docs/ingest.md).")
        cp.add_argument("--tasks", type=Path, required=True, help="path to the task export (JSON or YAML list)")

    m = _sp(sub, "meeting", "meeting bookkeeping (next date, what a meeting decided)",
            "`next` records the next meeting date of a series (shown in the Snapshot); `decided` lists the "
            "decisions recorded for one meeting (by its agenda read-back), for cross-checking a transcript.")
    msub = m.add_subparsers(dest="mcmd", required=True, metavar="SUBCOMMAND")
    mn = _sp(msub, "next", "record the next meeting date of a series",
             "Write Status/sync-state/next_meeting.<SERIES>.yaml.")
    mn.add_argument("series", help="a key of `series` in project.yaml")
    mn.add_argument("date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="date of the next meeting")
    md = _sp(msub, "decided", "list decisions recorded for one meeting",
             "List decisions whose `meeting` is SERIES/DATE or that carry an entry with source "
             "agenda:SERIES/DATE, with their status, as-decided text and spawned actions.")
    md.add_argument("series", help="a key of `series` in project.yaml")
    md.add_argument("date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="date of the meeting")
    md.add_argument("--json", action="store_true",
                    help="print a JSON list of {id, title, status, as_decided, actions} instead of text")

    so = _sp(sub, "sources", "list the configured ingest sources and their watermarks",
             "One row per source configured under `sources` in project.yaml: clickup, circleback, "
             "git.<ROOT> per repository and for overleaf, calendar, gmail and slack (when enabled), with its "
             "watermark (empty = never ingested).")
    so.add_argument("--json", action="store_true",
                    help="print a JSON list of {source, detail, watermark} instead of a table")


def _source_rows(cfg) -> list[dict]:
    src = cfg.sources or {}
    wm = load_state(cfg.root)["watermarks"]
    rows = []
    if src.get("clickup"):
        rows.append({"source": "clickup", "detail": f"list {src['clickup'].get('list_id')}", "watermark": wm.get("clickup")})
    if src.get("circleback"):
        cb = src["circleback"]
        rows.append({"source": "circleback", "detail": f"folder {cb.get('folder')} terms {cb.get('terms') or []}",
                     "watermark": wm.get("circleback")})
    repos = [("repo", r) for r in src.get("repos") or []]
    if isinstance(src.get("overleaf"), dict):
        repos.append(("overleaf", src["overleaf"]))
    for kind, r in repos:
        name = f"git.{r.get('root')}"
        rows.append({"source": name, "detail": f"{kind} {r.get('root')} ({r.get('branch') or 'HEAD'})",
                     "watermark": wm.get(name)})
    if "calendar" in src:
        rows.append({"source": "calendar", "detail": "next meeting per series", "watermark": None})
    for key in ("gmail", "slack"):
        if (src.get(key) or {}).get("enabled"):
            rows.append({"source": key, "detail": "opt-in", "watermark": wm.get(key)})
    return rows


def _decided(store, series: str, day: date) -> list[dict]:
    tag = f"{series}/{day.isoformat()}"
    actions = store.query(type="action")
    rows = []
    for d in store.query(type="decision"):
        if d.meta.get("meeting") == tag or any(e.source == f"agenda:{tag}" for e in d.entries):
            rows.append({"id": d.id, "title": d.title, "status": d.status,
                         "as_decided": d.meta.get("as_decided") or "",
                         "actions": [a.id for a in actions if d.id in a.links("spawned_by")]})
    return rows


def dispatch_ext(args, cfg, store, *, today, me) -> int | None:
    cmd = args.cmd
    if cmd == "seen":
        where = source_seen(cfg.root, store, args.source)
        print("seen " + " ".join(where) if where else "new")
    elif cmd == "ingest" and args.icmd == "git":
        res = ingest_git(cfg, args.root_name, today=today, since=args.since, advance=args.advance)
        print(json.dumps(res, indent=2))
        if res["warning"]:
            print(f"warning: {res['warning']}", file=sys.stderr)
    elif cmd == "ingest":
        author = me()
        batch = yaml.safe_load(args.file.read_text(encoding="utf-8"))
        rep = run_batch(cfg.root, store, batch, author=author, day=today)
        print(f"applied {rep['applied']}, skipped {rep['skipped']}, staged {len(rep['staged'])} "
              f"({' '.join(rep['staged'])}), skipped proposals {rep['skipped_proposals']}, "
              f"flags added {rep['flags_added']}" + (f", watermark {rep['watermark']}" if rep["watermark"] else ""))
    elif cmd == "clickup":
        author = me() if args.ccmd == "apply" else None
        plan = plan_clickup(store, cfg, load_tasks(args.tasks), today=today)
        if args.ccmd == "plan":
            print(json.dumps(plan, indent=2))
        else:
            print(json.dumps(apply_clickup(cfg.root, store, plan, author=author, day=today), indent=2))
    elif cmd == "meeting":
        if args.series not in cfg.series:
            raise ValueError(f"unknown series '{args.series}' (project.yaml series: {', '.join(cfg.series)})")
        if args.mcmd == "next":
            state = load_state(cfg.root)
            state["next_meeting"][args.series] = args.date
            save_state(cfg.root, state)
            print(f"next {args.series} meeting: {args.date.isoformat()}")
        else:
            rows = _decided(store, args.series, args.date)
            if args.json:
                print(json.dumps(rows, indent=2))
            for r in [] if args.json else rows:
                print(f"{r['id']}  {r['status']:<9} {r['title']} — {r['as_decided'] or 'as proposed'}"
                      f"  actions: {' '.join(r['actions']) or 'none'}")
    elif cmd == "sources":
        rows = _source_rows(cfg)
        if args.json:
            print(json.dumps(rows, indent=2))
        for r in [] if args.json else rows:
            print(f"{r['source']:<20} {r['watermark'] or '—':<28} {r['detail']}")
    else:
        return None
    return 0
```

- [ ] **Step 3: Hook it into `cli.py`**
```python
# EDIT project-hub/hub/cli.py
# OLD
from .agenda.lifecycle import AgendaError, build_agenda, read_agenda
# NEW
from .agenda.lifecycle import AgendaError, build_agenda, read_agenda
from .cli_ext import add_ext_parsers, dispatch_ext
```
```python
# EDIT project-hub/hub/cli.py
# OLD
def _dispatch(args, cfg, store) -> int:
    today = _today()
    cmd = args.cmd
# NEW
def _dispatch(args, cfg, store) -> int:
    today = _today()
    cmd = args.cmd
    rc = dispatch_ext(args, cfg, store, today=today, me=lambda: _me(cfg))
    if rc is not None:
        return rc
```
In `build_parser`, add `add_ext_parsers(sub)` on the line immediately before its final `return p` (currently `cli.py:247`, the only line that is exactly `    return p`):
```python
# EDIT project-hub/hub/cli.py
# OLD
    return p
# NEW
    add_ext_parsers(sub)
    return p
```
In the module docstring, insert these lines after the `sync-state get SOURCE | set SOURCE VALUE` entry and its continuation line:
```text
  seen SOURCE                print `new`, or `seen` and where SOURCE (circleback:<id>, clickup:<id>, …) is recorded
  ingest git ROOT [--since YYYY-MM-DD] [--advance]
                             JSON list of commits of a configured repository since watermark git.<ROOT>
  ingest batch FILE          apply an agent-written ingest file (safe ops directly, proposals staged; idempotent)
  clickup plan|apply --tasks FILE
                             reconcile a ClickUp list export: mirror status/assignee/due, stage new tasks, flags
  meeting next SERIES YYYY-MM-DD | meeting decided SERIES YYYY-MM-DD [--json]
                             record the next meeting date; list what a meeting decided
  sources [--json]           configured ingest sources and their watermarks
```

- [ ] **Step 4: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest -q`
Expected: every test passes. That includes `tests/test_cli.py`'s check that every subparser and argument is documented, which now walks the new commands too.

- [ ] **Step 5: Agent docs**
Create `project-hub/.agent_docs/ingest.md`. It must cover:
- the source id table and the watermark names from the Global Constraints;
- the batch file schema and every rule from Task 4, including each idempotency check;
- the ClickUp export format and the plan/apply rule table from Task 5, with `people.<key>.clickup` and `sources.clickup`;
- `ingest git` (configuration, watermark, fallback, `--advance` semantics);
- `seen`, `meeting next|decided` and `sources`, each with its exact output format;
- code references as file:line.

Add `ingest.md` to `_INDEX.md`. In `vault-layout.md`, document the new `project.yaml` keys:
- `people.<key>.clickup` and `people.<key>.email`;
- `sources.clickup`, `sources.circleback` (`folder`, `terms`), `sources.repos[]` (`root`, `branch`), `sources.overleaf`, `sources.calendar` and `sources.gmail` / `sources.slack` (`enabled`, `terms` / `channels`);
- `series.<name>.calendar_match`.

Also document the `hub.local.yaml` key `notes` (a list of private-note folders, opt-in, read only by the skill).

- [ ] **Step 6: Commit**
```bash
git add project-hub/hub/cli_ext.py project-hub/hub/cli.py project-hub/tests/test_cli_ext.py project-hub/.agent_docs
git commit -m "feat(hub): CLI for seen, ingest git|batch, clickup plan|apply, meeting next|decided, sources"
```

---
### Task 7: Skill `hub-status` with per-source references
**Files:**
- Create: `project-hub/skills/hub-status/SKILL.md`, `project-hub/skills/hub-status/sources/clickup.md`, `circleback.md`, `git.md`, `calendar.md`, `notes.md`, `mail-chat.md`
- Create: `project-hub/tests/test_skills.py`

**Interfaces:**
- Consumes: every `hub` command (Plan 1 plus Task 6).
- Produces: the skill text. `test_skills.py` fails if a skill cites a `hub` command or subcommand that does not exist, or if a SKILL.md's frontmatter is malformed.

- [ ] **Step 1: Write the failing test**
```python
# project-hub/tests/test_skills.py
import argparse
import re
from pathlib import Path

from hub.cli import build_parser

SKILLS = Path(__file__).resolve().parents[1] / "skills"
CMD_RE = re.compile(r"`hub ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?")


def _choices(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def test_every_cited_hub_command_exists():
    files = sorted(SKILLS.rglob("*.md"))
    assert files, "no skill files found"
    top = _choices(build_parser())
    bad = []
    for f in files:
        for m in CMD_RE.finditer(f.read_text(encoding="utf-8")):
            cmd, sub = m[1], m[2]
            if cmd not in top:
                bad.append(f"{f.relative_to(SKILLS)}: hub {cmd}")
                continue
            nested = _choices(top[cmd])
            if nested and sub is not None and sub not in nested:
                bad.append(f"{f.relative_to(SKILLS)}: hub {cmd} {sub}")
    assert bad == []


def test_skill_frontmatter():
    skills = sorted(SKILLS.glob("*/SKILL.md"))
    assert skills
    for f in skills:
        text = f.read_text(encoding="utf-8")
        head = text.split("\n---\n", 1)[0]
        assert text.startswith(f"---\nname: {f.parent.name}\n"), f
        assert "\ndescription: Use when" in head, f
```
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_skills.py -v`
Expected: FAIL with `AssertionError: no skill files found`.

- [ ] **Step 2: Write `skills/hub-status/SKILL.md`**
````markdown
---
name: hub-status
description: Use when asked to update a project hub's status, list someone's todos, or process a meeting that just happened ("update the status", "what are my todos", "we just met", "after the meeting") in a vault that has project.yaml and a Status/ folder.
---
# Hub status
Maintains the shared status store of one project hub vault (an Obsidian vault that is a git repo shared by the whole project). Three modes: **todos**, **update**, **post-meeting**. Every write goes through the `hub` command; never edit files under `Status/` by hand. `hub <command> --help` documents every option.
## Before any mode
1. Work from the vault root (the directory with `project.yaml`). Check `hub.local.yaml` has `me: <your people key>`; if not, ask the user which key in `project.yaml` people is them and add it.
2. Run `git pull --rebase`. If it fails, stop and show the error.
3. Run `hub validate` and note existing violations (do not fix unrelated ones silently).
4. If the user says **dry run**: never call a write tool of a connector (ClickUp create/update), never run `hub` commands that write (`new`, `entry`, `set`, `link`, `log`, `ingest batch`, `clickup apply`, `pending apply`, `commit`, …). Write batch files to a scratch directory and show them instead. `hub clickup plan`, `hub ingest git` without `--advance`, `hub seen`, `hub sources`, `hub list` and `hub meeting decided` are read-only and allowed.
## Mode: todos
`hub list --type action --status open --owner me` (or `--owner <key>` for someone else). Sort by due date, then priority. Present a short list: title, id, due date, the decision it came from (`hub show <id> --chain` on request). Mention `hub pending list` if proposals are waiting for approval. No writes.
## Mode: update
1. `hub sources` lists what is configured and when each source was last ingested.
2. For each configured source, follow its reference file in `sources/` (next to this file): ClickUp → `sources/clickup.md`, Circleback → `sources/circleback.md`, each repository and Overleaf → `sources/git.md`, calendar → `sources/calendar.md`, opted-in personal notes → `sources/notes.md`, Gmail/Slack → `sources/mail-chat.md`. Each ends with either `hub ingest batch <file>` or `hub clickup apply`.
3. Classify every fact with this table — the batch format enforces the left column, the right column always goes in `proposals`:

| Apply directly (`safe`) | Stage as a proposal |
|---|---|
| an entry on an existing item, with its source | closing a question (answered/dropped) |
| a log line, with its source | recording a decision as made (except via `hub agenda read`) |
| a new open question raised, with its source | a pivot (question replaced by another) |
| a new action that is unambiguous (named owner, from a tracker or explicit notes) | an action parsed from free text or a transcript |
| mirroring ClickUp status/assignee/due (`hub clickup apply`) | any change to Vision.md or Knowledge/ |

4. Before writing anything derived from a source id, `hub seen <source>`; skip what is already recorded unless it adds a new fact.
5. When all sources are done: `hub validate`, `hub snapshot`, then `hub commit "status: ingest <sources> (<YYYY-MM-DD>)"`. Exit code 3 means a real conflict: stop, show the message, and do not resolve it by guessing.
6. **Report** to the user: a table with one row per source — reached / not reachable (say why: connector missing, folder not shared, path not on this machine) / disabled / not configured — and what changed (entries, new items, flags). Never present a partial run as complete.
7. **Approval batch:** if proposals were staged, list them (`hub pending list`) with one line each on what applying does, and ask once: apply all, apply some, or reject some. Then `hub pending apply <P-ids>` / `hub pending reject <P-ids>` and `hub commit "status: approvals (<date>)"`.
## Mode: post-meeting
1. Identify the series and date (the user, or the series whose `Agenda/<series>.xlsx` has not been read back; ask if ambiguous).
2. `hub agenda read <series>` applies the decisions recorded in the workbook and stages free-text items. If it refuses, show its message (it explains what to do).
3. **Circleback cross-check** (`sources/circleback.md`, section *Cross-check*): compare what was said with `hub meeting decided <series> <date> --json`; anything decided or assigned aloud but not recorded becomes a proposal.
4. Run **update** for the other sources (ClickUp especially: new actions need tasks — see the *Push* step in `sources/clickup.md`, done after their proposals are applied).
5. Approval batch and commit as in update.
## Writing entries
One factual sentence each, with numbers when there are numbers ("4×3 open-boundary run finished: F = 0.97 at 1092 parameters"). Name the item the fact serves. Cite the source id in `--source` (formats in each source file).
## Privacy (the vault is read by everyone on the project)
Record decisions, stated positions ("Eun-Ah prefers framing B") and actions with their source. Never record personal assessments of people, career or 1:1 material, health, or anything from someone's private notes verbatim. When unsure whether something belongs, leave it out and mention it to the user.
````

- [ ] **Step 3: Write the source references**
````markdown
<!-- project-hub/skills/hub-status/sources/clickup.md -->
# ClickUp
ClickUp owns task status, assignee and due date. The hub mirrors them onto actions that carry `clickup_id`, stages tasks created directly in ClickUp, and gets a ClickUp task created for each new owned action.
## Configuration
`project.yaml`: `sources.clickup.list_id` (and optional `done_statuses`, default complete/completed/closed/done); `people.<key>.clickup` = that person's ClickUp user id.
## Export
1. `clickup_get_user` → note `current_time` (used for the watermark and for any relative date).
2. `clickup_filter_tasks` with `list_ids: [<list_id>]`, `include_closed: true`, `fields: ["status", "assignees", "due_date"]`, `page: 0`; repeat with `page: next_page` while `has_more` is true.
3. Write every task to a scratch file `tasks.json` as `{"id", "name", "status": <status name>, "assignees": [{"id", "username", "email"}], "due": "YYYY-MM-DD" or null}` (convert a millisecond timestamp to a date in the user's time zone).
## Reconcile
1. `hub clickup plan --tasks tasks.json` — read `set`, `proposals`, `flags`, `push` and sanity-check (a mass of unexpected changes means a wrong list id: stop and ask).
2. `hub clickup apply --tasks tasks.json` (skip in dry run).
3. `hub sync-state set clickup <current_time>`.
## Push (new actions → tasks)
For each entry of `push` (open actions with an owner and no ClickUp task), unless dry run: `clickup_create_task` with `list_id`, `name` = the action title, `assignees: [<assignee>]` when present, `due_date` when present, `markdown_description` = "Hub item <id>: <title>". Then record the link: `hub set <id> clickup_id=<task id>` and `hub entry <id> "ClickUp task created." --source clickup:<task id>`. Do this after the approval batch, so actions approved in this run get tasks too.
## Never
Change a task's status, assignee or due date from the hub (ClickUp owns them); delete tasks. Mismatches are flags for a human.
````
````markdown
<!-- project-hub/skills/hub-status/sources/circleback.md -->
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
## Cross-check (post-meeting mode)
For the meeting just read back with `hub agenda read`: get `hub meeting decided <series> <date> --json`; propose only what the transcript/notes show was decided or assigned and is not in that list (title the proposal "Said at <series>/<date>: …"). Do not re-propose decisions already recorded.
````
````markdown
<!-- project-hub/skills/hub-status/sources/git.md -->
# Code repositories and Overleaf
## Configuration
`project.yaml` `sources.repos: [{root: <ROOT>, branch: <name>}]`, `sources.overleaf: {root: <ROOT>, branch: <name>}`; each `<ROOT>` is a key of `roots`, with this machine's path in `hub.local.yaml` `roots`. Watermark `git.<ROOT>` = newest ingested commit.
## Steps (per root)
1. `git -C <path> fetch` then fast-forward the configured branch if it is a clean clone you own (for Overleaf: `git -C <path> pull`). If the path is missing on this machine, `hub ingest git` says so: report the root as not reachable.
2. `hub ingest git <ROOT>` → JSON commits since the watermark (a `warning` means history was rewritten; mention it).
3. Group commits by topic (same files / same subject prefix). Per group write one `safe` `log` line, source `commit:<ROOT>@<short sha of the newest commit in the group>`: what changed and why it matters ("Trotter optimizer: warm starts resume in their own gate order (3 commits)"). For Overleaf, name who edited which section ("Tamra: rewrote ¶2 and Fig. 1 caption").
4. Where a commit clearly serves an item (its message names the item id, or it finishes the work an action describes), add a `safe` `entry` on that item. A commit that appears to finish an action or answer a question → a proposal to close it, never a direct close.
5. `hub ingest batch <file>`, then, only if it succeeded, `hub ingest git <ROOT> --advance`.
````
````markdown
<!-- project-hub/skills/hub-status/sources/calendar.md -->
# Calendar (next meeting per series)
`project.yaml` `series.<name>.calendar_match` = words in the meeting's calendar title. For each series with a match: `list_events` (Google Calendar) with `fullText` = the match, `orderBy: startTime`, `pageSize: 5` (default window: the next 7 days; pass `startTime`/`endTime` for up to 5 weeks ahead if nothing is found). Take the first upcoming event's date and run `hub meeting next <series> <YYYY-MM-DD>`. If the calendar connector is not available, report the source as not reachable; never invent a date.
````
````markdown
<!-- project-hub/skills/hub-status/sources/notes.md -->
# Personal notes (opt-in)
Only the folders listed under `notes:` in **your own** `hub.local.yaml` (e.g. `notes: [/home/me/obsidian/personal/research]`) — never another person's, never anything not listed. Watermark `notes.<me>` = last date ingested.
1. List files in those folders modified after the watermark.
2. Keep only facts about this project: results, decisions you took, questions you opened. Summarise in your own words; never copy text verbatim; skip anything personal, career-related or about other people's performance.
3. Batch file: `safe` entries/log lines with source `notes:<me>@<YYYY-MM-DD of the note>`; anything that closes or decides → `proposals`. `watermark: {name: notes.<me>, value: <today>}`. Then `hub ingest batch <file>`.
````
````markdown
<!-- project-hub/skills/hub-status/sources/mail-chat.md -->
# Gmail and Slack (opt-in per project)
Only when `project.yaml` `sources.gmail.enabled` / `sources.slack.enabled` is true. Watermarks `gmail` / `slack` = last date ingested.
- **Gmail:** `search_threads` with the configured `terms` and `after:<watermark>`; read matching threads (`get_thread`). Source `gmail:<thread id>`.
- **Slack:** read the configured `channels` since the watermark (if the Slack connector needs authentication, report the source as not reachable and stop). Source `slack:<channel id>-<message ts>`.
Same extraction rules as `sources/circleback.md` (decisions and actions → proposals; facts → entries/log lines), same privacy rules (no personal matters, no verbatim private messages). One batch file; `watermark: {name: gmail|slack, value: <today>}`; `hub ingest batch <file>`.
````
Write each block to the path in its first-line comment, and remove that comment line from the file.

- [ ] **Step 4: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_skills.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**
```bash
git add project-hub/skills/hub-status project-hub/tests/test_skills.py
git commit -m "feat(skills): hub-status (todos, update, post-meeting) with per-source references"
```

---
### Task 8: Skill `hub-agenda`
**Files:**
- Create: `project-hub/skills/hub-agenda/SKILL.md`

**Interfaces:**
- Consumes: the `hub agenda scaffold|build`, `hub meeting next`, `hub sources`, `hub new` and `hub link` commands, and the `hub-status` skill.

- [ ] **Step 1: Write the skill**
````markdown
---
name: hub-agenda
description: Use when asked to prepare, prep or build the agenda for an upcoming meeting of a project hub ("prep the agenda for group", "get ready for Monday's meeting", "build the meeting sheet") in a vault that has project.yaml.
---
# Hub agenda
Prepares `Agenda/<series>.xlsx` for one meeting: Sheet 1 recaps last meeting's decisions and their follow-up, Sheet 2 lists open questions, Sheet 3 holds at most 7 decisions to take. The script owns layout; you write content into `Agenda/<series>.draft.yaml` only.
## Steps
1. **Series and date.** From the user; otherwise the Snapshot's *Next meeting* (`Status/Snapshot.md`, or run `sources/calendar.md` of the hub-status skill, which sets `hub meeting next`). Confirm both with the user if they were not stated.
2. **Last meeting read back?** If `Agenda/<series>.xlsx` exists for an earlier date and was not read back, run the hub-status skill's post-meeting mode first (ask the user — it applies that meeting's decisions).
3. **Fresh status.** `hub sources`: if any watermark is older than the last meeting of this series, run the hub-status update mode first.
4. **Scaffold.** `hub agenda scaffold <series> --date <YYYY-MM-DD>`. If it refuses because a draft exists, ask before passing `--force` (it discards the old draft's text).
5. **Fill the draft** (`Agenda/<series>.draft.yaml`). Edit only:
   - `summary` — 3–5 sentences, findings first, framed against last meeting's decisions (what was decided → what happened → what that means now).
   - `since_last[].follow_up` — what was done for each decision, with evidence (commit short shas, figure paths, ClickUp links). Correct `outcome` (closed question / new question / pivot / in progress) only if the store-derived value is wrong, and say why in `follow_up`.
   - `decisions[]` — keep at most 7, ordered by what unblocks most work first. For each: `proposed_action` (concrete, with an owner), `reason` (one sentence, cite evidence), `weigh_in` (people keys whose input is needed), `minutes` (realistic), `link` (figure or document to look at).
   - A decision that is needed but not in the store yet: create it first — `hub new d "<what to decide>" --link answers:<question id> --series <series> --source manual` — then re-scaffold with `--force` (after saving your text elsewhere) so it gets an id the read-back can match.
   - Do not invent questions; do not delete open questions from Sheet 2 (reorder only by editing priorities in the store with `hub set`).
6. **Build.** `hub agenda build <series>`. If it refuses (previous meeting unread, or the live workbook already holds meeting input), show the message; never pass `--force` without the user's explicit yes.
7. **Show the user** the decisions chosen and why, the total minutes against the meeting length, and open questions left off; invite pruning and rebuild after edits.
8. `hub commit "agenda: <series> <date>"`.
## Tell the meeting (once, when sharing the file)
One person edits the sheet. In **Outcome** type Approve, Modify, Defer or Reject (Office Viewer in VS Code shows no dropdown — typing works). Put the decision as actually taken in **As decided**; new actions in **New actions** as `title (owner); title (owner)`. Unplanned decisions go in the yellow rows. After the meeting, run the hub-status skill's post-meeting mode.
````
Write it to `project-hub/skills/hub-agenda/SKILL.md`.

- [ ] **Step 2: Run the tests**
Run: `cd project-hub && .venv/bin/python -m pytest tests/test_skills.py -v`
Expected: 2 passed. Every `hub …` command cited in the new skill exists.

- [ ] **Step 3: Commit**
```bash
git add project-hub/skills/hub-agenda
git commit -m "feat(skills): hub-agenda meeting preparation"
```

---
### Task 9: Acceptance dry run (controller-run; needs the MCP connectors)
This can't be delegated to an implementer subagent without connector access. It checks the skills against real services without writing to them.

- [ ] **Step 1: Make a scratch vault**
In `/home/jek354/.claude/jobs/<job>/tmp/accept/` (or any scratch directory), `git init` and write a `project.yaml` with:
- the real people of the UCC project (`jonathon`, `tamra`, `eunah`; `clickup` ids looked up with `clickup_find_member_by_name`);
- `series: {group: {attendees: [jonathon, tamra, eunah], calendar_match: <title of the group meeting>}}`;
- `sources.clickup.list_id: "901307841677"`, `sources.circleback.terms: [UCC, HVA]`;
- `sources.repos: [{root: ML_REPO, branch: main}]`.

Then write `hub.local.yaml` with `me: jonathon` and `roots: {ML_REPO: /home/jek354/research/ML-signproblem}`.
- [ ] **Step 2: Run the hub-status skill in dry run.** Ask a fresh agent, with the skill text and connector access: "dry run: update the status". Check:
  - only read-only tools were called;
  - `tasks.json` was produced and `hub clickup plan` output is plausible (actions are empty in a fresh vault, so every open task becomes a proposal and nothing is mirrored);
  - a Circleback batch file was written with sources `circleback:<id>` and decisions only under `proposals`;
  - `hub ingest git ML_REPO` listed commits;
  - the report table names every source.
- [ ] **Step 3: Run hub-agenda** on the scratch vault after a real (non-dry) `hub ingest batch` of the Circleback file and `hub pending apply` of a few proposals. Check the draft and the built workbook open in Office Viewer.
- [ ] **Step 4:** Record the problems found as fixes to the skill texts. Commit them with `fix(skills): …` messages, re-running `tests/test_skills.py`. Delete the scratch vault.

---
## Self-review notes
- **Spec coverage:**
  - *Skills*: `hub-status` modes update, todos and post-meeting (Task 7); `hub-agenda` (Task 8).
  - *Ingest sources*: ClickUp (Task 5 and `sources/clickup.md`); Circleback via batch (Task 4 and `sources/circleback.md`); repos and Overleaf (Task 3 and `sources/git.md`); calendar (`hub meeting next` and `sources/calendar.md`); notes and Gmail/Slack (`sources/notes.md`, `sources/mail-chat.md`).
  - Watermarks and idempotency: Tasks 2–4. The source report: `hub sources` and SKILL step 6.
  - *ClickUp mapping*: Task 5 rule table, plus push via the skill.
  - *Privacy*: Global Constraints and SKILL privacy section.
  - Pending approval batch: SKILL step 7.
  - Plan 1 follow-ups: Task 1 (the CLI-relevant ones; the rest stay in the follow-ups file).
- **Deferred to Plan 3:** installing the skills into a vault's `.claude/skills/` (`hub-init`), `hub-vision`, `hub-figures`, the Locator, `import-handoff`, and putting `hub` on PATH inside a vault.
- **Names used across tasks:**
  - `source_seen(root, store, source)`;
  - `ingest_git(config, name, *, today, since, advance)`;
  - `run_batch(root, store, batch, *, author, day)`;
  - `plan_clickup(store, config, tasks, *, today)`, `apply_clickup(root, store, plan, *, author, day)`;
  - `dispatch_ext(args, cfg, store, *, today, me)`.
