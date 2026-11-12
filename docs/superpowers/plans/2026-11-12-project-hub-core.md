# Project hub core — Implementation Plan (Plan 1 of 3)
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `hub` Python package and CLI: the Markdown status store, validation, snapshot, staged changes, multi-writer git sync, and the three-sheet Excel agenda (scaffold → build → read-back).

**Architecture:** A pure-Python package `hub` under `project-hub/` in the research repo. That directory later becomes the template copied into every `<project>-hub` vault (Plan 3). The Markdown files in a vault are the store of record. Every module takes the vault root as a `Path` and never caches state between calls. The CLI (`hub.cli:main`) is a thin argparse layer over the modules, so agents and people use the same commands.

**Tech Stack:** Python ≥ 3.10, PyYAML ≥ 6.0, openpyxl ≥ 3.1, pytest ≥ 7, git.

**Spec:** `docs/superpowers/specs/2026-11-12-project-hub-design.md`

**Later plans:**
- Plan 2: skills `hub-status` and `hub-agenda`, plus the ClickUp, Circleback, repo and Overleaf ingesters.
- Plan 3: figures, Locator, `check-links`, `hub-vision`, `hub-init`, the vault skeleton, `import-handoff`, and the UCC migration.

## Global Constraints
- Python ≥ 3.10; runtime dependencies are **only** PyYAML and openpyxl. Install with `pip install -e 'project-hub[test]'`; `uv` works on the same `pyproject.toml` but is not required (not installed on the cluster).
- Item IDs: `<T>-<YYYYMMDD>-<rand3>`, `T ∈ {Q, D, A}`, `rand3` = 3 chars from `[a-z0-9]`.
- Pending proposal IDs: `P-<rand4>` from `[a-z0-9]`.
- Entry line format, exactly: `- YYYY-MM-DD · <author> · <source> — <text>` (middle dot U+00B7, em dash U+2014), single line.
- Item statuses: question `open|answered|dropped|pivoted`; decision `proposed|made|rejected|deferred`; action `open|done|dropped`. Closing statuses: `answered, dropped, pivoted, done, rejected`.
- Link keys: `answers`, `spawned_by`, `pivoted_to`. Links are stored in one direction only.
- Vault paths: `Status/Items/<ID>.md`, `Status/Log/<YYYY-MM>.md`, `Status/Meetings/<series>/<YYYY-MM-DD>.md`, `Status/Snapshot.md`, `Status/Pending.md`, `Status/sync-state.yaml`, `Agenda/<series>.xlsx`, `Agenda/<series>.draft.yaml`, `Agenda/archive/<series>/<YYYY-MM-DD>.xlsx`, `project.yaml`, `hub.local.yaml` (gitignored).
- Agenda: at most 7 planned decisions; 5 "Raised in meeting" rows; Outcome choices `Approve, Modify, Defer, Reject`.
- Markdown written for Obsidian: no blank line after a heading; blank lines only between paragraphs, before tables and around `---`.
- Every CLI command has exhaustive `--help` text (repo rule `.agents/rules/command-line.md`); argument parsing lives in one function, `build_parser()`.
- Every function that needs "today" takes it as a parameter; the CLI reads `HUB_TODAY` (ISO date) if set, else `date.today()`.
- The package keeps `project-hub/.agent_docs/` current (repo rule `.agents/rules/memory.md`).

### Deviations from the spec, decided while planning
1. **Merging append-only files.** Two writers appending to the end of the same file conflict in git. So `Status/Log/*.md` and `Status/Pending.md` use git's built-in `merge=union`, and `Status/Snapshot.md` uses `merge=ours` and is regenerated after every pull. Item files and the `.xlsx` keep normal merging, so a real conflict is reported rather than hidden.
2. **Pending IDs.** Proposals are `P-<rand4>`, not sequential numbers, because sequential numbers collide when several people write.
3. **Decisions with no action.** A made decision with no action records the reason in a frontmatter field `no_action: <reason>`, not in a free-text entry, so `validate` can check it mechanically.
4. **Re-styling the archived agenda.** This is deferred until the Task 1 spike shows whether Office Viewer drops styling.

---
## File structure
```
project-hub/
  pyproject.toml
  README.md                     # install + command overview for humans
  .agent_docs/
    _INDEX.md
    item-format.md
    pending-format.md
    agenda-workbook.md
    git-sync.md
    office-viewer-spike.md      # Task 1 result
  spike/
    make_fixture.py             # Task 1, throwaway
    check_fixture.py            # Task 1, throwaway
  hub/
    __init__.py
    config.py                   # project.yaml + hub.local.yaml → HubConfig
    ids.py                      # new_id, new_pending_id
    items.py                    # Item/Entry model, parse_item, render_item, constants
    store.py                    # Store: CRUD, entries, links, queries, log
    validate.py                 # validate(store, config) -> list[Violation]
    sync_state.py               # Status/sync-state.yaml
    snapshot.py                 # render_snapshot, write_snapshot
    pending.py                  # Pending.md proposals and op application
    gitsync.py                  # ensure_repo_files, commit with rebase/retry
    agenda/
      __init__.py
      draft.py                  # scaffold, validate_draft, load/save draft
      build.py                  # build_workbook
      read.py                   # read_workbook, parse_outcome, parse_actions
      lifecycle.py              # build_agenda, read_agenda (refusal rules, archive, meeting note)
    cli.py                      # build_parser, main
  tests/
    conftest.py
    test_config.py
    test_ids.py
    test_items.py
    test_store.py
    test_validate.py
    test_snapshot.py
    test_pending.py
    test_gitsync.py
    test_agenda_draft.py
    test_agenda_build_read.py
    test_agenda_lifecycle.py
    test_cli.py
```
All paths below are relative to `/home/jek354/research/.claude/worktrees/project-hub-spec/` unless stated.

---
### Task 1: Office Viewer spike (throwaway, needs the user)
This tests what VS Code *Office Viewer* (`cweijan.vscode-office`) preserves when it saves an xlsx. The result decides whether the `_meta` sheet and the hidden ID column are safe to rely on.

**Files:**
- Create: `project-hub/spike/make_fixture.py`, `project-hub/spike/check_fixture.py`
- Create: `project-hub/.agent_docs/office-viewer-spike.md`

- [ ] **Step 1: Write the fixture generator**
```python
# project-hub/spike/make_fixture.py
"""Throwaway: write spike.xlsx exercising every workbook feature the agenda relies on.

Usage: python make_fixture.py OUT.xlsx
  OUT.xlsx  path of the workbook to create (overwritten).
"""
import sys
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


def main(argv):
    out = argv[0]
    wb = Workbook()
    ws = wb.active
    ws.title = "Decisions"
    ws["B1"] = "group — 2026-11-19 · Decisions"
    ws["B1"].font = Font(bold=True, size=14)
    ws.merge_cells("B2:E2")
    ws["B2"] = "Merged, wrapped summary cell. " * 6
    ws["B2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 60
    headers = ["id", "What to decide", "Outcome", "As decided"]
    for col, name in enumerate(headers, 1):
        c = ws.cell(4, col, name)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="404040")
    ws.cell(5, 1, "D-20261112-abc")
    ws.cell(5, 2, "Row 5: pick Approve from the dropdown")
    ws.cell(6, 1, "D-20261112-def")
    ws.cell(6, 2, "Row 6: TYPE the word modify by hand")
    ws.cell(5, 2).fill = PatternFill("solid", fgColor="E2F0D9")
    ws.column_dimensions["A"].hidden = True
    ws.column_dimensions["B"].width = 45
    ws.freeze_panes = "B5"
    dv = DataValidation(type="list", formula1='"Approve,Modify,Defer,Reject"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add("C5:C10")
    meta = wb.create_sheet("_meta")
    meta["A1"], meta["B1"] = "series", "group"
    meta["A2"], meta["B2"] = "read_back", "false"
    meta.sheet_state = "hidden"
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 2: Write the checker**
```python
# project-hub/spike/check_fixture.py
"""Throwaway: report which features survived an edit+save in Office Viewer.

Usage: python check_fixture.py EDITED.xlsx
"""
import sys
from openpyxl import load_workbook


def main(argv):
    wb = load_workbook(argv[0])
    print("sheets:", wb.sheetnames)
    print("_meta present:", "_meta" in wb.sheetnames,
          "state:", wb["_meta"].sheet_state if "_meta" in wb.sheetnames else None)
    ws = wb["Decisions"]
    print("col A hidden:", ws.column_dimensions["A"].hidden)
    print("A5 id:", ws["A5"].value, "| A6 id:", ws["A6"].value)
    print("C5 outcome:", ws["C5"].value, "| C6 outcome:", ws["C6"].value)
    print("data validations:", len(ws.data_validations.dataValidation))
    print("freeze:", ws.freeze_panes, "| merged:", ws.merged_cells.ranges)
    print("B5 fill:", ws["B5"].fill.fgColor.rgb, "| header font bold:", ws["A4"].font.bold)
    print("B2 wrap:", ws["B2"].alignment.wrap_text, "| row2 height:", ws.row_dimensions[2].height)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 3: Generate the fixture and hand it to the user**
Run: `python project-hub/spike/make_fixture.py /home/jek354/research/spike.xlsx`
Expected: `wrote /home/jek354/research/spike.xlsx`
Ask the user to:
1. Open `spike.xlsx` in VS Code with Office Viewer.
2. In C5, choose **Approve** from the dropdown.
3. In C6, type `modify`.
4. In D5, type `as proposed`.
5. Save and close.

Then report in one line whether the dropdown appeared and whether column A and the `_meta` tab were visible.

- [ ] **Step 4: Check what survived**
Run: `python project-hub/spike/check_fixture.py /home/jek354/research/spike.xlsx`
Record the full output plus the user's observations in `project-hub/.agent_docs/office-viewer-spike.md`, using these headings: `## Setup`, `## Observed in Office Viewer`, `## Survived save`, `## Consequences for the agenda`. Under the last heading, state for each of the following whether it is kept as designed or changed:
- the `_meta` sheet: if it was lost, meeting metadata moves to the visible title row of each sheet, which `read_meta` already falls back to;
- the hidden ID column: if it was lost, read-back falls back to title matching, which is already designed;
- the dropdown: if it was lost, typed outcomes still work;
- styling: if it was lost, Plan 1 gains a follow-up task to re-render the archived copy.

Delete `spike.xlsx` afterwards.

- [ ] **Step 5: Commit**
```bash
git add project-hub/spike project-hub/.agent_docs/office-viewer-spike.md
git commit -m "spike: Office Viewer xlsx round-trip findings"
```

---
### Task 2: Package scaffold, config and IDs
**Files:**
- Create: `project-hub/pyproject.toml`, `project-hub/hub/__init__.py`, `project-hub/hub/config.py`, `project-hub/hub/ids.py`
- Test: `project-hub/tests/conftest.py`, `project-hub/tests/test_config.py`, `project-hub/tests/test_ids.py`

**Interfaces:**
- Produces:
  - `HubConfig(root: Path, name: str, people: dict[str, dict], series: dict[str, dict], sources: dict, roots: dict[str, str], me: str | None)`
  - `find_root(start: Path) -> Path`
  - `load_config(root: Path) -> HubConfig`
  - `ConfigError`
  - `TYPE_PREFIX: dict[str, str]`
  - `new_id(item_type: str, day: date, existing: set[str], rng=None) -> str`
  - `new_pending_id(existing: set[str], rng=None) -> str`
  - pytest fixture `hub_root` (a tmp vault with `project.yaml` people `alice`, `bob`, series `group`, and `hub.local.yaml` `me: alice`)

- [ ] **Step 1: pyproject and package marker**
```toml
# project-hub/pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "project-hub"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["PyYAML>=6.0", "openpyxl>=3.1"]

[project.optional-dependencies]
test = ["pytest>=7"]

[project.scripts]
hub = "hub.cli:main"

[tool.setuptools.packages.find]
include = ["hub*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
```python
# project-hub/hub/__init__.py
"""Project hub: Markdown status store, staged changes, git sync and Excel agenda."""
__version__ = "0.1.0"
```
Run: `cd project-hub && python -m venv .venv && . .venv/bin/activate && pip install -e '.[test]'`
Expected: `Successfully installed ... project-hub-0.1.0`. Add `project-hub/.gitignore` containing `.venv/`, `__pycache__/`, `*.egg-info/`.

- [ ] **Step 2: Write failing tests**
```python
# project-hub/tests/conftest.py
import textwrap
import pytest


@pytest.fixture
def hub_root(tmp_path):
    (tmp_path / "project.yaml").write_text(textwrap.dedent("""\
        name: demo
        people:
          alice: {name: Alice}
          bob: {name: Bob}
        series:
          group: {attendees: [alice, bob]}
        roots:
          CODE: /nowhere/code
        """))
    (tmp_path / "hub.local.yaml").write_text("me: alice\nroots:\n  CODE: /here/code\n")
    return tmp_path
```
```python
# project-hub/tests/test_config.py
import pytest
from hub.config import ConfigError, find_root, load_config


def test_load_config_merges_local(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    cfg = load_config(hub_root)
    assert cfg.name == "demo"
    assert set(cfg.people) == {"alice", "bob"}
    assert cfg.me == "alice"
    assert cfg.roots["CODE"] == "/here/code"


def test_env_overrides_me(hub_root, monkeypatch):
    monkeypatch.setenv("HUB_ME", "bob")
    assert load_config(hub_root).me == "bob"


def test_unknown_me_rejected(hub_root, monkeypatch):
    monkeypatch.setenv("HUB_ME", "carol")
    with pytest.raises(ConfigError, match="carol"):
        load_config(hub_root)


def test_missing_key_rejected(tmp_path):
    (tmp_path / "project.yaml").write_text("name: x\npeople: {}\n")
    with pytest.raises(ConfigError, match="series"):
        load_config(tmp_path)


def test_find_root_walks_up(hub_root):
    deep = hub_root / "Status" / "Items"
    deep.mkdir(parents=True)
    assert find_root(deep) == hub_root.resolve()


def test_find_root_fails(tmp_path):
    with pytest.raises(ConfigError):
        find_root(tmp_path)
```
```python
# project-hub/tests/test_ids.py
import re
from datetime import date
from hub.ids import new_id, new_pending_id


class FixedRng:
    def __init__(self, seq):
        self.seq = list(seq)

    def choice(self, _alphabet):
        return self.seq.pop(0)


def test_format():
    i = new_id("question", date(2026, 11, 12), set())
    assert re.fullmatch(r"Q-20261112-[a-z0-9]{3}", i)


def test_prefixes():
    d = date(2026, 1, 2)
    assert new_id("decision", d, set()).startswith("D-20260102-")
    assert new_id("action", d, set()).startswith("A-20260102-")


def test_avoids_existing():
    rng = FixedRng("aaabbb")
    assert new_id("question", date(2026, 11, 12), {"Q-20261112-aaa"}, rng) == "Q-20261112-bbb"


def test_pending_id():
    assert re.fullmatch(r"P-[a-z0-9]{4}", new_pending_id(set()))
    assert new_pending_id({"P-aaaa"}, FixedRng("aaaabbbb")) == "P-bbbb"
```
Run: `cd project-hub && pytest tests/test_config.py tests/test_ids.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.config'`.

- [ ] **Step 3: Implement**
```python
# project-hub/hub/config.py
"""Load a hub vault's configuration: project.yaml (shared) merged with hub.local.yaml (per user)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """project.yaml / hub.local.yaml is missing or inconsistent."""


@dataclass
class HubConfig:
    root: Path
    name: str
    people: dict[str, dict]
    series: dict[str, dict]
    sources: dict
    roots: dict[str, str]
    me: str | None


def find_root(start: Path) -> Path:
    """Return the nearest directory at or above `start` that contains project.yaml."""
    p = Path(start).resolve()
    for d in [p, *p.parents]:
        if (d / "project.yaml").is_file():
            return d
    raise ConfigError(f"no project.yaml found at or above {start}")


def load_config(root: Path) -> HubConfig:
    root = Path(root)
    data = yaml.safe_load((root / "project.yaml").read_text()) or {}
    local_path = root / "hub.local.yaml"
    local = (yaml.safe_load(local_path.read_text()) if local_path.is_file() else None) or {}
    for key in ("name", "people", "series"):
        if key not in data:
            raise ConfigError(f"project.yaml is missing required key '{key}'")
    roots = dict(data.get("roots") or {})
    roots.update(local.get("roots") or {})
    people = data["people"] or {}
    me = os.environ.get("HUB_ME") or local.get("me")
    if me is not None and me not in people:
        raise ConfigError(f"me='{me}' is not a key of project.yaml people")
    return HubConfig(root=root, name=data["name"], people=people, series=data["series"] or {},
                     sources=data.get("sources") or {}, roots=roots, me=me)
```
```python
# project-hub/hub/ids.py
"""Collision-safe identifiers for items and pending proposals (safe across independent writers)."""
from __future__ import annotations

import random
import string
from datetime import date

TYPE_PREFIX = {"question": "Q", "decision": "D", "action": "A"}
ALPHABET = string.ascii_lowercase + string.digits


def _suffix(rng, n: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def new_id(item_type: str, day: date, existing: set[str], rng=None) -> str:
    """Return `<T>-<YYYYMMDD>-<rand3>` not present in `existing`."""
    prefix = TYPE_PREFIX[item_type]
    rng = rng or random.Random()
    for _ in range(100):
        candidate = f"{prefix}-{day:%Y%m%d}-{_suffix(rng, 3)}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("could not find a free item id after 100 attempts")


def new_pending_id(existing: set[str], rng=None) -> str:
    """Return `P-<rand4>` not present in `existing`."""
    rng = rng or random.Random()
    for _ in range(100):
        candidate = f"P-{_suffix(rng, 4)}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("could not find a free pending id after 100 attempts")
```

- [ ] **Step 4: Run tests**
Run: `cd project-hub && pytest tests/test_config.py tests/test_ids.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**
```bash
git add project-hub/pyproject.toml project-hub/.gitignore project-hub/hub/__init__.py project-hub/hub/config.py project-hub/hub/ids.py project-hub/tests/conftest.py project-hub/tests/test_config.py project-hub/tests/test_ids.py
git commit -m "feat(hub): package scaffold, vault config loading, collision-safe ids"
```

---
### Task 3: Item model — parse and render
**Files:**
- Create: `project-hub/hub/items.py`
- Test: `project-hub/tests/test_items.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - constants `TYPES`, `STATUSES: dict[str, tuple]`, `CLOSING: set[str]`, `LINK_KEYS`, `ENTRIES_HEADING = "## Entries"`
  - `Entry(day: date, author: str, source: str, text: str)` with `.render() -> str`
  - `Item(meta: dict, description: str, entries: list[Entry])` with properties `id, type, status, title, owner` and methods `.links(key) -> list[str]`, `.last_activity() -> date`
  - `default_meta(item_id, item_type, title, owner, raised, series, priority, sources) -> dict`
  - `parse_item(text: str) -> Item`
  - `render_item(item: Item) -> str`
  - `ItemFormatError`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_items.py
from datetime import date
import pytest
from hub.items import (Entry, Item, ItemFormatError, default_meta, parse_item, render_item)

CANONICAL = """---
id: Q-20261112-a3f
type: question
title: Does the HVA beat pruned UCC?
status: open
owner: alice
series:
- group
raised: 2026-11-12
closed:
priority: 2
links:
  answers: []
  spawned_by: []
  pivoted_to: []
sources:
- circleback:123
custom_key: kept
---
Why this matters.
## Entries
- 2026-11-12 · alice · circleback:123 — Raised at group meeting.
- 2026-11-15 · bob · commit:ML@ab12cd3 — Run finished: F=0.97.
"""


def test_round_trip_is_exact():
    assert render_item(parse_item(CANONICAL)) == CANONICAL


def test_parsed_fields():
    it = parse_item(CANONICAL)
    assert it.id == "Q-20261112-a3f"
    assert it.type == "question" and it.status == "open" and it.owner == "alice"
    assert it.meta["raised"] == date(2026, 11, 12)
    assert it.meta["closed"] is None
    assert it.meta["custom_key"] == "kept"
    assert it.description == "Why this matters."
    assert it.entries[1] == Entry(date(2026, 11, 15), "bob", "commit:ML@ab12cd3", "Run finished: F=0.97.")
    assert it.links("answers") == []


def test_last_activity():
    assert parse_item(CANONICAL).last_activity() == date(2026, 11, 15)


def test_entry_render_collapses_newlines():
    e = Entry(date(2026, 1, 1), "alice", "manual", "two\nlines")
    assert e.render() == "- 2026-01-01 · alice · manual — two lines"


def test_malformed_entry_rejected():
    bad = CANONICAL.replace("— Raised", "- Raised")
    with pytest.raises(ItemFormatError, match="malformed entry"):
        parse_item(bad)


def test_missing_frontmatter_rejected():
    with pytest.raises(ItemFormatError):
        parse_item("no frontmatter\n")


def test_default_meta_by_type():
    d = default_meta("D-20261112-aaa", "decision", "Pick X", "alice", date(2026, 11, 12), ["group"], 1, ["manual"])
    assert d["status"] == "proposed" and d["rationale"] == "" and d["meeting"] is None and d["as_decided"] == ""
    a = default_meta("A-20261112-aaa", "action", "Do X", "bob", date(2026, 11, 12), [], 2, [])
    assert a["status"] == "open" and a["clickup_id"] is None and a["due"] is None
    q = default_meta("Q-20261112-aaa", "question", "Why?", "bob", date(2026, 11, 12), [], 3, [])
    assert q["status"] == "open" and "clickup_id" not in q
    assert q["links"] == {"answers": [], "spawned_by": [], "pivoted_to": []}


def test_no_entries_section():
    text = CANONICAL.split("## Entries")[0]
    it = parse_item(text)
    assert it.entries == [] and it.description == "Why this matters."
```
Run: `cd project-hub && pytest tests/test_items.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.items'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/items.py
"""Item files: YAML frontmatter (state) + description + append-only '## Entries' (history)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import yaml

TYPES = ("question", "decision", "action")
STATUSES = {
    "question": ("open", "answered", "dropped", "pivoted"),
    "decision": ("proposed", "made", "rejected", "deferred"),
    "action": ("open", "done", "dropped"),
}
CLOSING = {"answered", "dropped", "pivoted", "done", "rejected"}
LINK_KEYS = ("answers", "spawned_by", "pivoted_to")
ENTRIES_HEADING = "## Entries"
ENTRY_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) · (\S+) · (\S+) — (.*)$")


class ItemFormatError(Exception):
    """An item file does not follow the documented format."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", ""))


@dataclass
class Entry:
    day: date
    author: str
    source: str
    text: str

    def render(self) -> str:
        text = " ".join(self.text.split())
        return f"- {self.day.isoformat()} · {self.author} · {self.source} — {text}"


@dataclass
class Item:
    meta: dict
    description: str = ""
    entries: list[Entry] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def type(self) -> str:
        return self.meta["type"]

    @property
    def status(self) -> str:
        return self.meta["status"]

    @property
    def title(self) -> str:
        return self.meta["title"]

    @property
    def owner(self) -> str | None:
        return self.meta.get("owner")

    def links(self, key: str) -> list[str]:
        return list((self.meta.get("links") or {}).get(key) or [])

    def last_activity(self) -> date:
        days = [d for d in (self.meta.get("raised"), self.meta.get("closed")) if isinstance(d, date)]
        days += [e.day for e in self.entries]
        return max(days) if days else date.min


def default_meta(item_id, item_type, title, owner, raised, series, priority, sources) -> dict:
    meta = {
        "id": item_id, "type": item_type, "title": title, "status": STATUSES[item_type][0],
        "owner": owner, "series": list(series), "raised": raised, "closed": None,
        "priority": priority, "links": {k: [] for k in LINK_KEYS}, "sources": list(sources),
    }
    if item_type == "decision":
        meta.update({"rationale": "", "meeting": None, "as_decided": ""})
    if item_type == "action":
        meta.update({"clickup_id": None, "due": None})
    return meta


def parse_item(text: str) -> Item:
    if not text.startswith("---\n"):
        raise ItemFormatError("missing frontmatter opening '---'")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ItemFormatError("missing frontmatter closing '---'")
    meta = yaml.safe_load(text[4:end]) or {}
    for key in ("id", "type", "title", "status"):
        if key not in meta:
            raise ItemFormatError(f"frontmatter missing '{key}'")
    lines = text[end + 5:].splitlines()
    entries: list[Entry] = []
    if ENTRIES_HEADING in lines:
        idx = lines.index(ENTRIES_HEADING)
        description = "\n".join(lines[:idx]).strip()
        for line in lines[idx + 1:]:
            if not line.strip():
                continue
            m = ENTRY_RE.match(line)
            if not m:
                raise ItemFormatError(f"malformed entry line: {line!r}")
            entries.append(Entry(date.fromisoformat(m[1]), m[2], m[3], m[4]))
    else:
        description = "\n".join(lines).strip()
    return Item(meta, description, entries)


def render_item(item: Item) -> str:
    front = yaml.dump(item.meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True,
                      default_flow_style=False)
    out = ["---", front.rstrip("\n"), "---", item.description.strip(), ENTRIES_HEADING]
    out += [e.render() for e in item.entries]
    return "\n".join(out) + "\n"
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_items.py -v`
Expected: 8 passed. If `test_round_trip_is_exact` fails on `closed:` rendering as `closed: ''` or `closed: null`, check that `_Dumper` is passed to `yaml.dump` and represents `None` as an empty plain scalar.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/items.py project-hub/tests/test_items.py
git commit -m "feat(hub): item file model with exact frontmatter/entries round-trip"
```

---
### Task 4: Store — create, entries, fields, links, queries, log
**Files:**
- Create: `project-hub/hub/store.py`
- Test: `project-hub/tests/test_store.py`

**Interfaces:**
- Consumes: `new_id` (Task 2); `Item, Entry, default_meta, parse_item, render_item, TYPES, STATUSES, LINK_KEYS` (Task 3).
- Produces: `Store(root: Path)` with:
  - `.all() -> list[Item]`
  - `.ids() -> set[str]`
  - `.get(item_id) -> Item` (raises `KeyError`)
  - `.save(item)`
  - `.create(item_type, title, *, owner, author, day, source, series=(), priority=2, links=None, description="", fields=None, rng=None) -> Item`
  - `.entry(item_id, text, *, author, source, day) -> Item`
  - `.set_fields(item_id, fields: dict) -> Item`
  - `.link(item_id, rel, target) -> Item`
  - `.query(type=None, status=None, owner=None, series=None, since=None) -> list[Item]`
  - `.children(item_id) -> list[Item]`
  - `.log(text, *, author, source, day) -> Path`
  - `StoreError`
  - `coerce_value(key, value)`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_store.py
from datetime import date
import pytest
from hub.store import Store, StoreError

D = date(2026, 11, 12)


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def mk(store, t, title, **kw):
    kw.setdefault("owner", "alice")
    return store.create(t, title, author="alice", day=D, source="manual", **kw)


def test_create_writes_file_with_raise_entry(store, hub_root):
    q = mk(store, "question", "Why?", series=["group"], description="Because.")
    path = hub_root / "Status" / "Items" / f"{q.id}.md"
    assert path.is_file()
    got = store.get(q.id)
    assert got.title == "Why?" and got.description == "Because."
    assert got.entries[0].text == "Raised." and got.entries[0].source == "manual"
    assert got.meta["sources"] == ["manual"]


def test_create_rejects_bad_type_and_link(store):
    with pytest.raises(StoreError):
        mk(store, "idea", "x")
    with pytest.raises(StoreError, match="unknown link"):
        mk(store, "action", "x", links={"blocks": ["Q-1"]})
    with pytest.raises(StoreError, match="does not exist"):
        mk(store, "action", "x", links={"spawned_by": ["D-20990101-zzz"]})


def test_entry_appends(store):
    q = mk(store, "question", "Why?")
    store.entry(q.id, "More.", author="bob", source="commit:x@1", day=date(2026, 11, 13))
    assert [e.text for e in store.get(q.id).entries] == ["Raised.", "More."]


def test_set_fields_validates_and_coerces(store):
    q = mk(store, "question", "Why?")
    store.set_fields(q.id, {"status": "answered", "closed": "2026-11-14", "priority": "1"})
    got = store.get(q.id)
    assert got.status == "answered" and got.meta["closed"] == date(2026, 11, 14) and got.meta["priority"] == 1
    with pytest.raises(StoreError, match="status"):
        store.set_fields(q.id, {"status": "done"})
    with pytest.raises(StoreError, match="cannot set"):
        store.set_fields(q.id, {"id": "X"})


def test_link_and_children(store):
    q = mk(store, "question", "Why?")
    d = mk(store, "decision", "Do X")
    store.link(d.id, "answers", q.id)
    a = mk(store, "action", "Run X", links={"spawned_by": [d.id]})
    store.link(d.id, "answers", q.id)  # idempotent
    assert store.get(d.id).links("answers") == [q.id]
    assert [c.id for c in store.children(q.id)] == [d.id]
    assert [c.id for c in store.children(d.id)] == [a.id]


def test_query_filters(store):
    q = mk(store, "question", "Q1", series=["group"])
    mk(store, "action", "A1", owner="bob")
    store.entry(q.id, "late", author="alice", source="manual", day=date(2026, 12, 1))
    assert [i.title for i in store.query(type="action")] == ["A1"]
    assert [i.title for i in store.query(owner="bob")] == ["A1"]
    assert [i.title for i in store.query(series="group")] == ["Q1"]
    assert [i.title for i in store.query(since=date(2026, 11, 20))] == ["Q1"]
    assert {i.title for i in store.query(status="open")} == {"Q1", "A1"}


def test_log_appends_monthly(store, hub_root):
    store.log("Run finished.", author="alice", source="commit:x@1", day=D)
    store.log("Draft pushed.", author="bob", source="overleaf:2", day=date(2026, 11, 20))
    text = (hub_root / "Status" / "Log" / "2026-11.md").read_text()
    assert text == ("# Log 2026-11\n"
                    "- 2026-11-12 · alice · commit:x@1 — Run finished.\n"
                    "- 2026-11-20 · bob · overleaf:2 — Draft pushed.\n")
```
Run: `cd project-hub && pytest tests/test_store.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.store'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/store.py
"""The status store: one Markdown file per item under Status/Items, monthly logs under Status/Log."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .ids import new_id
from .items import (LINK_KEYS, STATUSES, TYPES, Entry, Item, default_meta, parse_item, render_item)

DATE_KEYS = {"raised", "closed", "due"}
INT_KEYS = {"priority"}
PROTECTED_KEYS = {"id", "type", "links"}


class StoreError(Exception):
    """A store operation would break the item format or reference a missing item."""


def coerce_value(key: str, value):
    """Convert CLI strings into the types stored in frontmatter."""
    if isinstance(value, str):
        if value == "":
            return None
        if key in DATE_KEYS:
            return date.fromisoformat(value)
        if key in INT_KEYS:
            return int(value)
    return value


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.items_dir = self.root / "Status" / "Items"
        self.log_dir = self.root / "Status" / "Log"

    def path(self, item_id: str) -> Path:
        return self.items_dir / f"{item_id}.md"

    def all(self) -> list[Item]:
        if not self.items_dir.is_dir():
            return []
        return [parse_item(p.read_text()) for p in sorted(self.items_dir.glob("*.md"))]

    def ids(self) -> set[str]:
        if not self.items_dir.is_dir():
            return set()
        return {p.stem for p in self.items_dir.glob("*.md")}

    def get(self, item_id: str) -> Item:
        p = self.path(item_id)
        if not p.is_file():
            raise KeyError(item_id)
        return parse_item(p.read_text())

    def save(self, item: Item) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.path(item.id).write_text(render_item(item))

    def create(self, item_type, title, *, owner, author, day, source, series=(), priority=2,
               links=None, description="", fields=None, rng=None) -> Item:
        if item_type not in TYPES:
            raise StoreError(f"unknown item type '{item_type}' (expected one of {TYPES})")
        existing = self.ids()
        for rel, targets in (links or {}).items():
            if rel not in LINK_KEYS:
                raise StoreError(f"unknown link '{rel}' (expected one of {LINK_KEYS})")
            for t in targets:
                if t not in existing:
                    raise StoreError(f"link target {t} does not exist")
        item_id = new_id(item_type, day, existing, rng)
        meta = default_meta(item_id, item_type, title, owner, day, series, priority, [source])
        for rel, targets in (links or {}).items():
            meta["links"][rel] = list(targets)
        for k, v in (fields or {}).items():
            if k in PROTECTED_KEYS:
                raise StoreError(f"cannot set '{k}' through fields")
            meta[k] = coerce_value(k, v)
        if meta["status"] not in STATUSES[item_type]:
            raise StoreError(f"status '{meta['status']}' invalid for {item_type}")
        item = Item(meta, description, [Entry(day, author, source, "Raised.")])
        self.save(item)
        return item

    def entry(self, item_id, text, *, author, source, day) -> Item:
        item = self.get(item_id)
        item.entries.append(Entry(day, author, source, text))
        src = item.meta.setdefault("sources", [])
        if source not in src:
            src.append(source)
        self.save(item)
        return item

    def set_fields(self, item_id, fields: dict) -> Item:
        item = self.get(item_id)
        for k, v in fields.items():
            if k in PROTECTED_KEYS:
                raise StoreError(f"cannot set '{k}' (use `hub link` for links)")
            v = coerce_value(k, v)
            if k == "status" and v not in STATUSES[item.type]:
                raise StoreError(f"status '{v}' invalid for {item.type}; expected one of {STATUSES[item.type]}")
            item.meta[k] = v
        self.save(item)
        return item

    def link(self, item_id, rel, target) -> Item:
        if rel not in LINK_KEYS:
            raise StoreError(f"unknown link '{rel}' (expected one of {LINK_KEYS})")
        if target not in self.ids():
            raise StoreError(f"link target {target} does not exist")
        item = self.get(item_id)
        links = item.meta.setdefault("links", {k: [] for k in LINK_KEYS})
        lst = links.setdefault(rel, [])
        if target not in lst:
            lst.append(target)
        self.save(item)
        return item

    def query(self, type=None, status=None, owner=None, series=None, since=None) -> list[Item]:
        out = []
        for it in self.all():
            if type and it.type != type:
                continue
            if status and it.status != status:
                continue
            if owner and it.owner != owner:
                continue
            if series and series not in (it.meta.get("series") or []):
                continue
            if since and it.last_activity() < since:
                continue
            out.append(it)
        return out

    def children(self, item_id) -> list[Item]:
        """Items that point at item_id: decisions answering it, items spawned by it, its pivot targets."""
        out = []
        for it in self.all():
            if item_id in it.links("answers") or item_id in it.links("spawned_by"):
                out.append(it)
        try:
            for t in self.get(item_id).links("pivoted_to"):
                out.append(self.get(t))
        except KeyError:
            pass
        return out

    def log(self, text, *, author, source, day) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{day:%Y-%m}.md"
        if not path.is_file():
            path.write_text(f"# Log {day:%Y-%m}\n")
        with path.open("a") as fh:
            fh.write(Entry(day, author, source, text).render() + "\n")
        return path
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_store.py -v`
Expected: 7 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/store.py project-hub/tests/test_store.py
git commit -m "feat(hub): status store with entries, validated fields, links, queries, monthly log"
```

---
### Task 5: Validation
**Files:**
- Create: `project-hub/hub/validate.py`
- Test: `project-hub/tests/test_validate.py`

**Interfaces:**
- Consumes: `Store` (Task 4), `HubConfig` (Task 2), constants (Task 3), `TYPE_PREFIX` (Task 2).
- Produces:
  - `Violation(item_id: str, rule: str, message: str)`
  - `validate(store: Store, config: HubConfig) -> list[Violation]`
  - rule names: `type`, `status`, `id-prefix`, `link-target`, `entry-author`, `unknown-owner`, `closed-date`, `closing-entry`, `pivot-target`, `decision-answers`, `decision-actions`, `action-owner`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_validate.py
from datetime import date
import pytest
from hub.config import load_config
from hub.store import Store
from hub.validate import validate

D = date(2026, 11, 12)


@pytest.fixture
def env(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    return Store(hub_root), load_config(hub_root)


def rules(store, cfg):
    return sorted(v.rule for v in validate(store, cfg))


def mk(store, t, title, **kw):
    kw.setdefault("owner", "alice")
    return store.create(t, title, author="alice", day=D, source="manual", **kw)


def test_clean_store(env):
    store, cfg = env
    q = mk(store, "question", "Q")
    d = mk(store, "decision", "D", links={"answers": [q.id]})
    mk(store, "action", "A", links={"spawned_by": [d.id]})
    store.set_fields(d.id, {"status": "made"})
    assert rules(store, cfg) == []


def test_closing_rules(env):
    store, cfg = env
    q = mk(store, "question", "Q")
    store.set_fields(q.id, {"status": "answered"})
    assert rules(store, cfg) == ["closed-date"]
    store.set_fields(q.id, {"closed": "2026-11-20"})
    assert rules(store, cfg) == ["closing-entry"]
    store.entry(q.id, "Answered by D-x.", author="bob", source="manual", day=date(2026, 11, 20))
    assert rules(store, cfg) == []


def test_pivot_needs_target(env):
    store, cfg = env
    q = mk(store, "question", "Q")
    store.set_fields(q.id, {"status": "pivoted", "closed": "2026-11-12"})
    assert rules(store, cfg) == ["pivot-target"]


def test_made_decision_needs_answers_and_actions(env):
    store, cfg = env
    d = mk(store, "decision", "D")
    store.set_fields(d.id, {"status": "made"})
    assert rules(store, cfg) == ["decision-actions", "decision-answers"]
    store.set_fields(d.id, {"no_action": "Pure framing choice."})
    assert rules(store, cfg) == ["decision-answers"]


def test_owner_and_author_rules(env):
    store, cfg = env
    a = mk(store, "action", "A", owner=None)
    assert rules(store, cfg) == ["action-owner"]
    store.set_fields(a.id, {"owner": "carol"})
    assert rules(store, cfg) == ["action-owner", "unknown-owner"]
    store.set_fields(a.id, {"owner": "bob"})
    store.entry(a.id, "x", author="mallory", source="manual", day=D)
    assert rules(store, cfg) == ["entry-author"]


def test_broken_link_and_status(env, hub_root):
    store, cfg = env
    q = mk(store, "question", "Q")
    p = hub_root / "Status" / "Items" / f"{q.id}.md"
    p.write_text(p.read_text().replace("pivoted_to: []", "pivoted_to:\n  - Q-20990101-zzz")
                              .replace("status: open", "status: weird"))
    assert rules(store, cfg) == ["link-target", "status"]
```
Run: `cd project-hub && pytest tests/test_validate.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'hub.validate'`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/validate.py
"""Structural rules every status store must satisfy. `hub validate` exits non-zero on any violation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .config import HubConfig
from .ids import TYPE_PREFIX
from .items import CLOSING, LINK_KEYS, STATUSES, TYPES
from .store import Store


@dataclass
class Violation:
    item_id: str
    rule: str
    message: str


def validate(store: Store, config: HubConfig) -> list[Violation]:
    items = {i.id: i for i in store.all()}
    out: list[Violation] = []
    spawned_from: dict[str, list[str]] = defaultdict(list)
    for it in items.values():
        for t in it.links("spawned_by"):
            spawned_from[t].append(it.id)

    def bad(it, rule, msg):
        out.append(Violation(it.id, rule, msg))

    for it in items.values():
        if it.type not in TYPES:
            bad(it, "type", f"unknown type '{it.type}'")
            continue
        if it.status not in STATUSES[it.type]:
            bad(it, "status", f"status '{it.status}' invalid for {it.type}")
        if not it.id.startswith(TYPE_PREFIX[it.type] + "-"):
            bad(it, "id-prefix", f"id does not start with {TYPE_PREFIX[it.type]}-")
        for k in LINK_KEYS:
            for t in it.links(k):
                if t not in items:
                    bad(it, "link-target", f"{k} -> {t} does not exist")
        for e in it.entries:
            if e.author not in config.people:
                bad(it, "entry-author", f"entry author '{e.author}' not in project.yaml people")
                break
        if it.owner is not None and it.owner not in config.people:
            bad(it, "unknown-owner", f"owner '{it.owner}' not in project.yaml people")
        if it.status in CLOSING:
            closed = it.meta.get("closed")
            if not isinstance(closed, date):
                bad(it, "closed-date", f"status '{it.status}' requires a 'closed' date")
            elif not any(e.day >= closed for e in it.entries):
                bad(it, "closing-entry", "no entry dated on/after 'closed' explaining the closure")
        if it.status == "pivoted" and not it.links("pivoted_to"):
            bad(it, "pivot-target", "pivoted question needs a pivoted_to link")
        if it.type == "decision" and it.status == "made":
            if not it.links("answers"):
                bad(it, "decision-answers", "made decision answers no question")
            if not spawned_from[it.id] and not it.meta.get("no_action"):
                bad(it, "decision-actions", "made decision spawned no action and has no 'no_action' reason")
        if it.type == "action" and it.status == "open" and (it.owner is None or it.owner not in config.people):
            bad(it, "action-owner", "open action needs an owner from project.yaml people")
    return out
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_validate.py -v`
Expected: 6 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/validate.py project-hub/tests/test_validate.py
git commit -m "feat(hub): store validation rules"
```

---
### Task 6: Sync state and snapshot
**Files:**
- Create: `project-hub/hub/sync_state.py`, `project-hub/hub/snapshot.py`
- Test: `project-hub/tests/test_snapshot.py`

**Interfaces:**
- Consumes: `Store`, `validate`, `HubConfig`.
- Produces:
  - `load_state(root) -> dict`, `save_state(root, state)`, `get_watermark(root, source) -> str | None`, `set_watermark(root, source, value)`. State keys: `watermarks: {source: str}`, `next_meeting: {series: date}`, `flags: [str]`.
  - `last_meeting(root, series, before: date | None = None) -> date | None` (reads `Status/Meetings/<series>/*.md` stems)
  - `render_snapshot(store, config, today) -> str`, `write_snapshot(store, config, today) -> Path`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_snapshot.py
from datetime import date
from hub.config import load_config
from hub.snapshot import last_meeting, render_snapshot, write_snapshot
from hub.store import Store
from hub.sync_state import get_watermark, load_state, save_state, set_watermark

D = date(2026, 11, 12)


def test_watermarks_round_trip(hub_root):
    assert get_watermark(hub_root, "clickup") is None
    set_watermark(hub_root, "clickup", "2026-11-12T10:00")
    assert get_watermark(hub_root, "clickup") == "2026-11-12T10:00"
    assert load_state(hub_root)["watermarks"] == {"clickup": "2026-11-12T10:00"}


def test_last_meeting(hub_root):
    d = hub_root / "Status" / "Meetings" / "group"
    d.mkdir(parents=True)
    for s in ("2026-11-03", "2026-11-10"):
        (d / f"{s}.md").write_text("x")
    assert last_meeting(hub_root, "group") == date(2026, 11, 10)
    assert last_meeting(hub_root, "group", before=date(2026, 11, 10)) == date(2026, 11, 3)
    assert last_meeting(hub_root, "other") is None


def test_snapshot_content(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    store, cfg = Store(hub_root), load_config(hub_root)
    q = store.create("question", "Why?", owner="alice", author="alice", day=D, source="m", priority=1)
    d = store.create("decision", "Do X", owner="bob", author="alice", day=D, source="m",
                     links={"answers": [q.id]})
    store.create("action", "Run X", owner="bob", author="alice", day=D, source="m",
                 links={"spawned_by": [d.id]}, fields={"due": "2026-11-20"})
    state = load_state(hub_root)
    state["next_meeting"] = {"group": date(2026, 11, 19)}
    state["flags"] = ["ClickUp task 86x deleted but A-... open"]
    save_state(hub_root, state)
    text = render_snapshot(store, cfg, date(2026, 11, 15))
    assert text.startswith("# Snapshot — demo\n*Generated 2026-11-15 by `hub snapshot`.")
    assert "| group | — | 2026-11-19 |" in text
    assert f"| [[{q.id}]] | Why? | alice | 3 | 1 |" in text
    assert f"| [[{d.id}]] | Do X | {q.id} | bob |" in text
    assert "### bob\n" in text and "Run X (due 2026-11-20)" in text
    assert "- ClickUp task 86x deleted but A-... open" in text
    assert "## Validation\n0 issues." in text
    path = write_snapshot(store, cfg, date(2026, 11, 15))
    assert path == hub_root / "Status" / "Snapshot.md" and path.read_text() == text
```
Run: `cd project-hub && pytest tests/test_snapshot.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement sync_state**
```python
# project-hub/hub/sync_state.py
"""Status/sync-state.yaml: per-source watermarks, next meeting per series, unresolved sync flags."""
from __future__ import annotations

from pathlib import Path

import yaml


def _path(root) -> Path:
    return Path(root) / "Status" / "sync-state.yaml"


def load_state(root) -> dict:
    p = _path(root)
    data = (yaml.safe_load(p.read_text()) if p.is_file() else None) or {}
    data.setdefault("watermarks", {})
    data.setdefault("next_meeting", {})
    data.setdefault("flags", [])
    return data


def save_state(root, state: dict) -> None:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(state, sort_keys=True, allow_unicode=True))


def get_watermark(root, source: str):
    return load_state(root)["watermarks"].get(source)


def set_watermark(root, source: str, value: str) -> None:
    state = load_state(root)
    state["watermarks"][source] = value
    save_state(root, state)
```

- [ ] **Step 3: Implement snapshot**
```python
# project-hub/hub/snapshot.py
"""Generate Status/Snapshot.md — a one-page, never hand-edited view of where the project stands."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .config import HubConfig
from .store import Store
from .sync_state import load_state
from .validate import validate


def last_meeting(root, series: str, before: date | None = None) -> date | None:
    d = Path(root) / "Status" / "Meetings" / series
    if not d.is_dir():
        return None
    days = []
    for p in d.glob("*.md"):
        try:
            day = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if before is None or day < before:
            days.append(day)
    return max(days) if days else None


def render_snapshot(store: Store, config: HubConfig, today: date) -> str:
    state = load_state(config.root)
    items = store.all()
    out = [f"# Snapshot — {config.name}",
           f"*Generated {today.isoformat()} by `hub snapshot`. Do not edit; regenerate.*", ""]
    out += ["## Meetings", "| Series | Last meeting | Next meeting |", "|---|---|---|"]
    for s in config.series:
        last = last_meeting(config.root, s)
        nxt = state["next_meeting"].get(s)
        out.append(f"| {s} | {last.isoformat() if last else '—'} | {nxt.isoformat() if nxt else '—'} |")
    qs = sorted((i for i in items if i.type == "question" and i.status == "open"),
                key=lambda i: (i.meta.get("priority") or 9, i.meta["raised"]))
    out += ["", f"## Open questions ({len(qs)})", "| ID | Title | Owner | Age (d) | Priority |", "|---|---|---|---|---|"]
    for q in qs:
        out.append(f"| [[{q.id}]] | {q.title} | {q.owner or '—'} | {(today - q.meta['raised']).days} | {q.meta.get('priority')} |")
    ds = [i for i in items if i.type == "decision" and i.status == "proposed"]
    out += ["", f"## Proposed decisions ({len(ds)})", "| ID | Title | Answers | Owner |", "|---|---|---|---|"]
    for d in ds:
        out.append(f"| [[{d.id}]] | {d.title} | {', '.join(d.links('answers')) or '—'} | {d.owner or '—'} |")
    acts = [i for i in items if i.type == "action" and i.status == "open"]
    out += ["", f"## Open actions ({len(acts)})"]
    for owner in sorted({a.owner or "unassigned" for a in acts}):
        out.append(f"### {owner}")
        for a in sorted((a for a in acts if (a.owner or "unassigned") == owner),
                        key=lambda a: (a.meta.get("due") or date.max)):
            due = f" (due {a.meta['due'].isoformat()})" if a.meta.get("due") else ""
            out.append(f"- [[{a.id}]] {a.title}{due}")
    out += ["", "## Flags"]
    out += [f"- {f}" for f in state["flags"]] or ["None."]
    n = len(validate(store, config))
    out += ["", "## Validation", f"{n} issues." + (" Run `hub validate`." if n else "")]
    return "\n".join(out) + "\n"


def write_snapshot(store: Store, config: HubConfig, today: date) -> Path:
    path = Path(config.root) / "Status" / "Snapshot.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_snapshot(store, config, today))
    return path
```

- [ ] **Step 4: Run tests**
Run: `cd project-hub && pytest tests/test_snapshot.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**
```bash
git add project-hub/hub/sync_state.py project-hub/hub/snapshot.py project-hub/tests/test_snapshot.py
git commit -m "feat(hub): sync-state watermarks and generated Snapshot.md"
```

---
### Task 7: Pending proposals
**Files:**
- Create: `project-hub/hub/pending.py`
- Test: `project-hub/tests/test_pending.py`

**Interfaces:**
- Consumes: `Store` (Task 4), `new_pending_id` (Task 2).
- Produces:
  - `Proposal(pid: str, title: str, body: dict)`; body keys `proposer, date, source, reason, ops`.
  - `load_proposals(root) -> list[Proposal]`
  - `save_proposals(root, proposals)`
  - `add_proposal(root, title, ops, *, proposer, source, reason, day, rng=None) -> str`
  - `apply_ops(store, ops, *, author, day) -> dict[str, str]` (ref → created id)
  - `apply_proposals(root, store, pids, *, author, day) -> list[str]` (created ids)
  - `reject_proposals(root, store, pids, *, author, day) -> None`
  - `PendingError`

Op schema (each op a dict):
- `{op: new, ref: "$x" (optional), type, title, owner, series: [...], priority, links: {rel: [ids or $refs]}, description, fields: {...}, source}`
- `{op: entry, id, text, source}`
- `{op: set, id, fields: {...}}`
- `{op: link, id, rel, target}`

Any `id`, `target` or link list value may be a `$ref` from an earlier `new` in the same proposal.

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_pending.py
from datetime import date
import pytest
from hub.pending import (PendingError, add_proposal, apply_proposals, load_proposals, reject_proposals)
from hub.store import Store

D = date(2026, 11, 12)


@pytest.fixture
def store(hub_root):
    return Store(hub_root)


def test_add_and_load(hub_root):
    pid = add_proposal(hub_root, "Close Q", [{"op": "set", "id": "Q-1", "fields": {"status": "answered"}}],
                       proposer="alice", source="circleback:9", reason="Answered in meeting.", day=D)
    props = load_proposals(hub_root)
    assert [p.pid for p in props] == [pid]
    assert props[0].title == "Close Q" and props[0].body["reason"] == "Answered in meeting."
    text = (hub_root / "Status" / "Pending.md").read_text()
    assert text.startswith("# Pending changes\n")
    assert f"## {pid} · Close Q\n" + "`" * 3 + "yaml\n" in text


def test_apply_with_refs(hub_root, store):
    q = store.create("question", "Q", owner="alice", author="alice", day=D, source="m")
    ops = [
        {"op": "new", "ref": "$d", "type": "decision", "title": "Do X", "owner": "alice",
         "links": {"answers": [q.id]}, "fields": {"status": "made"}, "source": "agenda:group/2026-11-12"},
        {"op": "new", "type": "action", "title": "Run X", "owner": "bob",
         "links": {"spawned_by": ["$d"]}, "source": "agenda:group/2026-11-12"},
        {"op": "entry", "id": q.id, "text": "Decision taken.", "source": "agenda:group/2026-11-12"},
        {"op": "link", "id": "$d", "rel": "answers", "target": q.id},
    ]
    pid = add_proposal(hub_root, "Record decision", ops, proposer="alice", source="agenda", reason="r", day=D)
    created = apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert len(created) == 2
    d = store.get(created[0])
    assert d.status == "made" and d.links("answers") == [q.id]
    assert store.get(created[1]).links("spawned_by") == [d.id]
    assert store.get(q.id).entries[-1].text == "Decision taken."
    assert load_proposals(hub_root) == []
    log = (hub_root / "Status" / "Log" / "2026-11.md").read_text()
    assert f"Applied {pid}: Record decision" in log


def test_reject_logs_and_removes(hub_root, store):
    pid = add_proposal(hub_root, "X", [], proposer="alice", source="s", reason="r", day=D)
    reject_proposals(hub_root, store, [pid], author="bob", day=D)
    assert load_proposals(hub_root) == []
    assert f"Rejected {pid}: X" in (hub_root / "Status" / "Log" / "2026-11.md").read_text()


def test_unknown_pid(hub_root, store):
    with pytest.raises(PendingError, match="P-zzzz"):
        apply_proposals(hub_root, store, ["P-zzzz"], author="bob", day=D)


def test_failed_apply_keeps_proposal(hub_root, store):
    pid = add_proposal(hub_root, "Bad", [{"op": "set", "id": "Q-20990101-zzz", "fields": {}}],
                       proposer="alice", source="s", reason="r", day=D)
    with pytest.raises(PendingError):
        apply_proposals(hub_root, store, [pid], author="bob", day=D)
    assert [p.pid for p in load_proposals(hub_root)] == [pid]
```
Run: `cd project-hub && pytest tests/test_pending.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/pending.py
"""Status/Pending.md: staged judgement calls, each a list of store ops applied only on approval."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .ids import new_pending_id
from .store import Store, StoreError

FENCE = "`" * 3  # built, not literal, so this file nests inside Markdown fences
HEADER = "# Pending changes\nApply with `hub pending apply P-xxxx …`; reject with `hub pending reject P-xxxx …`.\n"
BLOCK_RE = re.compile(r"^## (P-[a-z0-9]{4}) · (.*?)\n" + FENCE + r"yaml\n(.*?)\n" + FENCE + r"\n", re.M | re.S)


class PendingError(Exception):
    """A proposal is unknown or its ops cannot be applied."""


@dataclass
class Proposal:
    pid: str
    title: str
    body: dict


def _path(root) -> Path:
    return Path(root) / "Status" / "Pending.md"


def load_proposals(root) -> list[Proposal]:
    p = _path(root)
    if not p.is_file():
        return []
    return [Proposal(m[1], m[2], yaml.safe_load(m[3]) or {}) for m in BLOCK_RE.finditer(p.read_text())]


def save_proposals(root, proposals: list[Proposal]) -> None:
    parts = [HEADER]
    for pr in proposals:
        body = yaml.safe_dump(pr.body, sort_keys=False, allow_unicode=True).rstrip("\n")
        parts.append(f"## {pr.pid} · {pr.title}\n{FENCE}yaml\n{body}\n{FENCE}\n")
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(parts))


def add_proposal(root, title, ops, *, proposer, source, reason, day, rng=None) -> str:
    proposals = load_proposals(root)
    pid = new_pending_id({p.pid for p in proposals}, rng)
    proposals.append(Proposal(pid, title, {"proposer": proposer, "date": day, "source": source,
                                           "reason": reason, "ops": list(ops)}))
    save_proposals(root, proposals)
    return pid


def _resolve(value, refs):
    if isinstance(value, str) and value.startswith("$"):
        if value not in refs:
            raise PendingError(f"unknown ref {value}")
        return refs[value]
    if isinstance(value, list):
        return [_resolve(v, refs) for v in value]
    return value


def apply_ops(store: Store, ops, *, author, day) -> dict[str, str]:
    refs: dict[str, str] = {}
    created: list[str] = []
    try:
        for op in ops:
            kind = op.get("op")
            if kind == "new":
                links = {k: _resolve(v, refs) for k, v in (op.get("links") or {}).items()}
                item = store.create(op["type"], op["title"], owner=op.get("owner"), author=author, day=day,
                                    source=op.get("source", "pending"), series=op.get("series") or (),
                                    priority=op.get("priority", 2), links=links,
                                    description=op.get("description", ""), fields=op.get("fields"))
                created.append(item.id)
                if op.get("ref"):
                    refs[op["ref"]] = item.id
            elif kind == "entry":
                store.entry(_resolve(op["id"], refs), op["text"], author=author,
                            source=op.get("source", "pending"), day=day)
            elif kind == "set":
                store.set_fields(_resolve(op["id"], refs), op.get("fields") or {})
            elif kind == "link":
                store.link(_resolve(op["id"], refs), op["rel"], _resolve(op["target"], refs))
            else:
                raise PendingError(f"unknown op '{kind}'")
    except (StoreError, KeyError) as exc:
        raise PendingError(f"op failed: {exc}") from exc
    refs["_created"] = created  # type: ignore[assignment]
    return refs


def _take(root, pids) -> tuple[list[Proposal], list[Proposal]]:
    proposals = load_proposals(root)
    by_id = {p.pid: p for p in proposals}
    missing = [pid for pid in pids if pid not in by_id]
    if missing:
        raise PendingError(f"unknown proposal(s): {', '.join(missing)}")
    chosen = [by_id[pid] for pid in pids]
    rest = [p for p in proposals if p.pid not in set(pids)]
    return chosen, rest


def apply_proposals(root, store: Store, pids, *, author, day) -> list[str]:
    chosen, _ = _take(root, pids)
    created: list[str] = []
    for pr in chosen:
        refs = apply_ops(store, pr.body.get("ops") or [], author=author, day=day)
        created += refs["_created"]
        store.log(f"Applied {pr.pid}: {pr.title}", author=author, source=f"pending:{pr.pid}", day=day)
        save_proposals(root, [p for p in load_proposals(root) if p.pid != pr.pid])
    return created


def reject_proposals(root, store: Store, pids, *, author, day) -> None:
    chosen, rest = _take(root, pids)
    for pr in chosen:
        store.log(f"Rejected {pr.pid}: {pr.title}", author=author, source=f"pending:{pr.pid}", day=day)
    save_proposals(root, rest)
```
Note: ops inside one proposal are not transactional. If op 3 fails, ops 1–2 have already been written, and the proposal stays in `Pending.md`. `hub validate` plus `git diff` show the partial state. This is acceptable because every write is a reviewable file change. Document it in `.agent_docs/pending-format.md`.

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_pending.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/pending.py project-hub/tests/test_pending.py
git commit -m "feat(hub): staged proposals in Pending.md with ref-resolving ops"
```

---
### Task 8: Multi-writer git sync
**Files:**
- Create: `project-hub/hub/gitsync.py`
- Test: `project-hub/tests/test_gitsync.py`

**Interfaces:**
- Consumes: `Store` (in tests only).
- Produces:
  - `GITATTRIBUTES_LINES`, `GITIGNORE_LINES`
  - `ensure_repo_files(root) -> None` (idempotently appends missing lines to `.gitattributes`/`.gitignore`, sets `git config merge.ours.driver true`)
  - `commit(root, message, *, regenerate=None, retries=3) -> str` (returns HEAD sha)
  - `SyncConflict(files: list[str], detail: str)`
  - `SyncError`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_gitsync.py
import shutil
import subprocess
from datetime import date
from pathlib import Path
import pytest
from hub.gitsync import SyncConflict, commit, ensure_repo_files
from hub.store import Store

D = date(2026, 11, 12)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def two_clones(tmp_path, hub_root):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    shutil.copytree(hub_root, seed)
    (seed / "hub.local.yaml").unlink()
    git(seed, "init", "-b", "main")
    for c in (seed,):
        git(c, "config", "user.email", "t@example.com")
        git(c, "config", "user.name", "T")
    ensure_repo_files(seed)
    git(seed, "add", "-A")
    git(seed, "commit", "-m", "seed")
    git(seed, "remote", "add", "origin", str(bare))
    git(seed, "push", "-u", "origin", "main")
    clones = []
    for name in ("a", "b"):
        c = tmp_path / name
        git(tmp_path, "clone", str(bare), str(c))
        git(c, "config", "user.email", f"{name}@example.com")
        git(c, "config", "user.name", name)
        (c / "hub.local.yaml").write_text("me: alice\n")
        clones.append(c)
    return clones


def test_ensure_repo_files_idempotent(tmp_path):
    git(tmp_path, "init", "-b", "main")
    ensure_repo_files(tmp_path)
    ensure_repo_files(tmp_path)
    attrs = (tmp_path / ".gitattributes").read_text().splitlines()
    assert attrs.count("Status/Log/*.md merge=union") == 1
    assert "hub.local.yaml" in (tmp_path / ".gitignore").read_text().splitlines()
    assert git(tmp_path, "config", "merge.ours.driver").strip() == "true"


def test_concurrent_items_and_logs_merge(two_clones):
    a, b = two_clones
    Store(a).create("question", "From A", owner="alice", author="alice", day=D, source="m")
    Store(a).log("A ran.", author="alice", source="m", day=D)
    Store(b).create("question", "From B", owner="bob", author="bob", day=D, source="m")
    Store(b).log("B ran.", author="bob", source="m", day=D)
    commit(a, "a")
    commit(b, "b")  # must rebase over a's push, union-merge the log
    git(a, "pull", "--rebase")
    titles = {i.title for i in Store(a).all()}
    assert titles == {"From A", "From B"}
    log = (a / "Status" / "Log" / "2026-11.md").read_text()
    assert "A ran." in log and "B ran." in log and "<<<<<<<" not in log
    assert "hub.local.yaml" not in git(a, "ls-files")


def test_regenerate_runs_after_pull(two_clones):
    a, b = two_clones
    Store(a).create("question", "From A", owner="alice", author="alice", day=D, source="m")
    commit(a, "a")
    Store(b).create("question", "From B", owner="bob", author="bob", day=D, source="m")
    seen = []

    def regen():
        seen.append(len(Store(b).all()))
        (b / "Status" / "Snapshot.md").write_text(f"{len(Store(b).all())} items\n")

    commit(b, "b", regenerate=regen)
    assert seen == [2]
    assert git(b, "status", "--porcelain").strip() == ""


def test_same_item_conflict_reported(two_clones):
    a, b = two_clones
    q = Store(a).create("question", "Shared", owner="alice", author="alice", day=D, source="m")
    commit(a, "a")
    git(b, "pull", "--rebase")
    Store(a).set_fields(q.id, {"title": "From A"})
    Store(b).set_fields(q.id, {"title": "From B"})
    commit(a, "a2")
    with pytest.raises(SyncConflict) as exc:
        commit(b, "b2")
    assert f"Status/Items/{q.id}.md" in exc.value.files
    assert not (b / ".git" / "rebase-merge").exists()
```
Run: `cd project-hub && pytest tests/test_gitsync.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/gitsync.py
"""Commit and push a hub vault safely when several people (and their agents) write to it.

Strategy: commit locally → `git pull --rebase` → regenerate generated files → push; retry on rejection.
Append-only files merge with git's built-in `union` driver; Snapshot.md keeps upstream (`ours` driver)
and is regenerated; item files and .xlsx merge normally so genuine conflicts surface as SyncConflict.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

GITATTRIBUTES_LINES = [
    "Status/Log/*.md merge=union",
    "Status/Pending.md merge=union",
    "Status/Snapshot.md merge=ours",
    "*.xlsx binary",
]
GITIGNORE_LINES = ["hub.local.yaml", ".venv/", "__pycache__/", "*.egg-info/", ".obsidian/workspace*.json"]


class SyncError(Exception):
    """Push kept being rejected, or git failed unexpectedly."""


class SyncConflict(SyncError):
    def __init__(self, files: list[str], detail: str):
        super().__init__(f"merge conflict in: {', '.join(files) or '(unknown)'}\n{detail}")
        self.files = files
        self.detail = detail


def _git(root, *args, check=True):
    res = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise SyncError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res


def _ensure_lines(path: Path, lines: list[str]) -> None:
    existing = path.read_text().splitlines() if path.is_file() else []
    missing = [l for l in lines if l not in existing]
    if missing:
        path.write_text("\n".join(existing + missing) + "\n")


def ensure_repo_files(root) -> None:
    root = Path(root)
    _ensure_lines(root / ".gitattributes", GITATTRIBUTES_LINES)
    _ensure_lines(root / ".gitignore", GITIGNORE_LINES)
    _git(root, "config", "merge.ours.driver", "true")


def _commit_all(root, message) -> bool:
    _git(root, "add", "-A")
    if _git(root, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(root, "commit", "-m", message)
    return True


def commit(root, message, *, regenerate=None, retries=3) -> str:
    root = Path(root)
    ensure_repo_files(root)
    _commit_all(root, message)
    if not _git(root, "remote").stdout.strip():
        if regenerate:
            regenerate()
            _commit_all(root, "Regenerate generated files")
        return _git(root, "rev-parse", "HEAD").stdout.strip()
    for _ in range(retries):
        pull = _git(root, "pull", "--rebase", check=False)
        if pull.returncode != 0:
            files = _git(root, "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
            _git(root, "rebase", "--abort", check=False)
            raise SyncConflict(files, pull.stdout + pull.stderr)
        if regenerate:
            regenerate()
            _commit_all(root, "Regenerate generated files")
        if _git(root, "push", check=False).returncode == 0:
            return _git(root, "rev-parse", "HEAD").stdout.strip()
    raise SyncError(f"push rejected {retries} times; someone is pushing continuously — retry later")
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_gitsync.py -v`
Expected: 4 passed. If `test_concurrent_items_and_logs_merge` shows conflict markers in the log, check that `.gitattributes` was committed in the seed (it is created by `ensure_repo_files(seed)` before the seed commit).

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/gitsync.py project-hub/tests/test_gitsync.py
git commit -m "feat(hub): multi-writer commit with rebase/retry, union-merged logs, conflict reporting"
```

---
### Task 9: Agenda draft — scaffold and validation
**Files:**
- Create: `project-hub/hub/agenda/__init__.py` (empty docstring), `project-hub/hub/agenda/draft.py`
- Test: `project-hub/tests/test_agenda_draft.py`

**Interfaces:**
- Consumes: `Store`, `last_meeting` (Task 6).
- Produces:
  - `SINCE_OUTCOMES = ("closed question", "new question", "pivot", "in progress")`
  - `MAX_DECISIONS = 7`
  - `DraftError`
  - `scaffold(store, root, series, meeting_date: date) -> dict`
  - `validate_draft(draft: dict) -> list[str]` (error messages)
  - `draft_path(root, series) -> Path`
  - `save_draft(root, draft) -> Path`, `load_draft(root, series) -> dict`

Draft dict keys:
- `series: str`, `meeting_date: date`, `previous_meeting: date | None`, `summary: str`
- `since_last: [{id, decision, actions, follow_up, outcome, affected}]`
- `questions: [{id, title, description, owner, age_days, priority, proposed_decision}]`
- `decisions: [{id, decide, proposed_action, reason, weigh_in, minutes, link}]`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_agenda_draft.py
from datetime import date
import pytest
from hub.agenda.draft import load_draft, save_draft, scaffold, validate_draft
from hub.store import Store

D = date(2026, 11, 5)


@pytest.fixture
def setup(hub_root):
    s = Store(hub_root)
    q1 = s.create("question", "Old high", owner="alice", author="alice", day=date(2026, 10, 1), source="m",
                  series=["group"], priority=1)
    q2 = s.create("question", "New low", owner="bob", author="alice", day=D, source="m", series=["group"], priority=3)
    s.create("question", "Other series", owner="bob", author="alice", day=D, source="m", series=["sub"])
    dprop = s.create("decision", "Adopt Y", owner="alice", author="alice", day=D, source="m", series=["group"],
                     links={"answers": [q2.id]}, fields={"rationale": "Cheaper."})
    dmade = s.create("decision", "Do X", owner="alice", author="alice", day=D, source="m", series=["group"],
                     links={"answers": [q1.id]},
                     fields={"status": "made", "meeting": "group/2026-11-05", "as_decided": "Do X at 4x4"})
    s.create("action", "Run X", owner="bob", author="alice", day=D, source="m", links={"spawned_by": [dmade.id]})
    mdir = hub_root / "Status" / "Meetings" / "group"
    mdir.mkdir(parents=True)
    (mdir / "2026-11-05.md").write_text("x")
    return s, hub_root, dict(q1=q1, q2=q2, dprop=dprop, dmade=dmade)


def test_scaffold(setup):
    s, root, ids = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    assert d["previous_meeting"] == date(2026, 11, 5) and d["summary"] == ""
    assert d["since_last"] == [{"id": ids["dmade"].id, "decision": "Do X at 4x4", "actions": "Run X (bob, open)",
                                "follow_up": "", "outcome": "in progress", "affected": ids["q1"].id}]
    assert [q["title"] for q in d["questions"]] == ["New low", "Old high"]  # has proposed decision first
    assert d["questions"][1]["age_days"] == 42 and d["questions"][0]["proposed_decision"] == ids["dprop"].id
    assert d["decisions"] == [{"id": ids["dprop"].id, "decide": "Adopt Y", "proposed_action": "",
                               "reason": "Cheaper.", "weigh_in": "", "minutes": 10, "link": ""}]


def test_validate_draft(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    assert validate_draft(d) == ["summary is empty"]
    d["summary"] = "Where we stand."
    d["since_last"][0]["outcome"] = "done"
    d["decisions"] = d["decisions"] * 8
    errs = validate_draft(d)
    assert any("outcome 'done'" in e for e in errs) and any("at most 7" in e for e in errs)


def test_save_load_round_trip(setup):
    s, root, _ = setup
    d = scaffold(s, root, "group", date(2026, 11, 12))
    path = save_draft(root, d)
    assert path == root / "Agenda" / "group.draft.yaml"
    assert load_draft(root, "group") == d
```
Run: `cd project-hub && pytest tests/test_agenda_draft.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/agenda/__init__.py
"""Meeting agenda: draft YAML (agent-written content) → styled xlsx → read-back into the store."""
```
```python
# project-hub/hub/agenda/draft.py
"""Agenda draft: the content of one meeting's workbook, pre-filled from the store for the agent to finish."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from ..snapshot import last_meeting
from ..store import Store

SINCE_OUTCOMES = ("closed question", "new question", "pivot", "in progress")
MAX_DECISIONS = 7
REQUIRED = ("series", "meeting_date", "previous_meeting", "summary", "since_last", "questions", "decisions")


class DraftError(Exception):
    """The draft is missing or invalid."""


def _in_series(item, series) -> bool:
    s = item.meta.get("series") or []
    return not s or series in s


def scaffold(store: Store, root, series: str, meeting_date: date) -> dict:
    items = store.all()
    prev = last_meeting(root, series, before=meeting_date)
    since_last = []
    if prev:
        tag = f"{series}/{prev.isoformat()}"
        for d in items:
            if d.type == "decision" and d.status == "made" and d.meta.get("meeting") == tag:
                acts = [a for a in items if a.type == "action" and d.id in a.links("spawned_by")]
                since_last.append({
                    "id": d.id, "decision": d.meta.get("as_decided") or d.title,
                    "actions": "; ".join(f"{a.title} ({a.owner}, {a.status})" for a in acts),
                    "follow_up": "", "outcome": "in progress", "affected": ", ".join(d.links("answers")),
                })
    proposed = [d for d in items if d.type == "decision" and d.status == "proposed" and _in_series(d, series)]
    proposed_for = {}
    for d in proposed:
        for q in d.links("answers"):
            proposed_for.setdefault(q, d.id)
    qs = [q for q in items if q.type == "question" and q.status == "open" and _in_series(q, series)]
    qs.sort(key=lambda q: (q.id not in proposed_for, q.meta.get("priority") or 9, q.meta["raised"]))
    questions = [{"id": q.id, "title": q.title, "description": q.description, "owner": q.owner,
                  "age_days": (meeting_date - q.meta["raised"]).days, "priority": q.meta.get("priority"),
                  "proposed_decision": proposed_for.get(q.id, "")} for q in qs]
    proposed.sort(key=lambda d: (d.meta.get("priority") or 9, d.meta["raised"]))
    decisions = [{"id": d.id, "decide": d.title, "proposed_action": d.meta.get("proposed_action", ""),
                  "reason": d.meta.get("rationale", ""), "weigh_in": "", "minutes": 10, "link": ""}
                 for d in proposed[:MAX_DECISIONS]]
    return {"series": series, "meeting_date": meeting_date, "previous_meeting": prev, "summary": "",
            "since_last": since_last, "questions": questions, "decisions": decisions}


def validate_draft(draft: dict) -> list[str]:
    errs = [f"missing key '{k}'" for k in REQUIRED if k not in draft]
    if errs:
        return errs
    if not str(draft["summary"]).strip():
        errs.append("summary is empty")
    for row in draft["since_last"]:
        if row.get("outcome") not in SINCE_OUTCOMES:
            errs.append(f"since_last {row.get('id')}: outcome '{row.get('outcome')}' not in {SINCE_OUTCOMES}")
    if len(draft["decisions"]) > MAX_DECISIONS:
        errs.append(f"decisions: at most {MAX_DECISIONS} allowed, got {len(draft['decisions'])}")
    return errs


def draft_path(root, series: str) -> Path:
    return Path(root) / "Agenda" / f"{series}.draft.yaml"


def save_draft(root, draft: dict) -> Path:
    p = draft_path(root, draft["series"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True))
    return p


def load_draft(root, series: str) -> dict:
    p = draft_path(root, series)
    if not p.is_file():
        raise DraftError(f"no draft at {p}; run `hub agenda scaffold {series} --date YYYY-MM-DD` first")
    return yaml.safe_load(p.read_text())
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_agenda_draft.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/agenda/__init__.py project-hub/hub/agenda/draft.py project-hub/tests/test_agenda_draft.py
git commit -m "feat(hub): agenda draft scaffold from the store, with validation"
```

---
### Task 10: Workbook build and read
**Files:**
- Create: `project-hub/hub/agenda/build.py`, `project-hub/hub/agenda/read.py`
- Test: `project-hub/tests/test_agenda_build_read.py`

**Interfaces:**
- Consumes: draft dict (Task 9), `validate_draft`, `DraftError`.
- Produces (build.py):
  - constants `SHEET_SINCE = "Since last meeting"`, `SHEET_Q = "Open questions"`, `SHEET_D = "Decisions"`, `SHEET_META = "_meta"`, `HEADER_ROW = 4`, `OUTCOME_CHOICES = ("Approve", "Modify", "Defer", "Reject")`, `UNPLANNED_ROWS = 5`, `D_HEADERS` (list of header names)
  - `build_workbook(draft: dict, out_path: Path) -> None`
  - `set_read_back(path: Path, value: bool) -> None`
- Produces (read.py):
  - `DecisionRow(id: str | None, decide: str, outcome: str | None, as_decided: str, new_actions: str, planned: bool)`
  - `ReadBack(series: str | None, meeting_date: date | None, read_back: bool, rows: list[DecisionRow])`
  - `parse_outcome(text) -> str | None` (returns one of `approve`/`modify`/`defer`/`reject`/None)
  - `parse_actions(text) -> list[tuple[str, str | None]]`
  - `read_workbook(path) -> ReadBack`

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_agenda_build_read.py
from datetime import date
import pytest
from openpyxl import load_workbook
from hub.agenda.build import (D_HEADERS, SHEET_D, SHEET_META, SHEET_Q, SHEET_SINCE, UNPLANNED_ROWS,
                              build_workbook, set_read_back)
from hub.agenda.draft import DraftError
from hub.agenda.read import parse_actions, parse_outcome, read_workbook

DRAFT = {
    "series": "group", "meeting_date": date(2026, 11, 19), "previous_meeting": date(2026, 11, 12),
    "summary": "We locked the framing.",
    "since_last": [{"id": "D-20261112-aaa", "decision": "Do X", "actions": "Run X (bob, done)",
                    "follow_up": "Ran at 4x4", "outcome": "closed question", "affected": "Q-20261101-qqq"}],
    "questions": [{"id": "Q-20261101-qqq", "title": "Why?", "description": "Because.", "owner": "alice",
                   "age_days": 18, "priority": 1, "proposed_decision": "D-20261115-bbb"}],
    "decisions": [{"id": "D-20261115-bbb", "decide": "Adopt Y", "proposed_action": "Rewrite ¶2",
                   "reason": "Cheaper", "weigh_in": "eunah", "minutes": 10, "link": "https://example.com/fig"}],
}


@pytest.fixture
def xlsx(tmp_path):
    p = tmp_path / "group.xlsx"
    build_workbook(DRAFT, p)
    return p


def test_structure(xlsx):
    wb = load_workbook(xlsx)
    assert wb.sheetnames == [SHEET_SINCE, SHEET_Q, SHEET_D, SHEET_META]
    assert wb[SHEET_META].sheet_state == "hidden"
    ws = wb[SHEET_D]
    assert [c.value for c in ws[4]] == D_HEADERS
    assert ws.column_dimensions["A"].hidden
    assert ws["A5"].value == "D-20261115-bbb" and ws["B5"].value == "Adopt Y"
    assert ws["A6"].value == "NEW-1" and ws[f"A{5 + UNPLANNED_ROWS}"].value == f"NEW-{UNPLANNED_ROWS}"
    assert len(ws.data_validations.dataValidation) == 1
    assert ws["G5"].hyperlink.target == "https://example.com/fig"
    assert wb[SHEET_SINCE]["B2"].value == "We locked the framing."
    assert wb[SHEET_SINCE]["B1"].value == "group — 2026-11-19 · Since last meeting"


def test_build_rejects_invalid(tmp_path):
    with pytest.raises(DraftError, match="summary"):
        build_workbook({**DRAFT, "summary": ""}, tmp_path / "x.xlsx")


def test_read_back_round_trip(xlsx):
    wb = load_workbook(xlsx)
    ws = wb[SHEET_D]
    ws["H5"], ws["I5"], ws["J5"] = "modify", "Adopt Y but only for 4x3", "Rewrite ¶2 (alice); Rerun fig (bob)"
    ws["B6"], ws["H6"], ws["I6"] = "Drop UCJ", "✓", "Dropped"
    wb.save(xlsx)
    rb = read_workbook(xlsx)
    assert rb.series == "group" and rb.meeting_date == date(2026, 11, 19) and rb.read_back is False
    assert len(rb.rows) == 2
    r0, r1 = rb.rows
    assert (r0.id, r0.outcome, r0.planned, r0.as_decided) == ("D-20261115-bbb", "modify", True, "Adopt Y but only for 4x3")
    assert (r1.id, r1.decide, r1.outcome, r1.planned) == ("NEW-1", "Drop UCJ", "approve", False)


def test_read_without_meta_or_id_column(xlsx):
    wb = load_workbook(xlsx)
    del wb[SHEET_META]
    ws = wb[SHEET_D]
    ws.delete_cols(1)
    ws["G4"] = "Outcome"
    ws["G5"] = "Approve"
    wb.save(xlsx)
    rb = read_workbook(xlsx)
    assert rb.series == "group" and rb.meeting_date == date(2026, 11, 19)
    assert rb.rows[0].id is None and rb.rows[0].decide == "Adopt Y" and rb.rows[0].outcome == "approve"


def test_set_read_back(xlsx):
    set_read_back(xlsx, True)
    assert read_workbook(xlsx).read_back is True


@pytest.mark.parametrize("text,expected", [
    ("Approve", "approve"), ("  a ", "approve"), ("✓", "approve"), ("yes", "approve"),
    ("Modify", "modify"), ("m", "modify"), ("Defer", "defer"), ("d", "defer"),
    ("Reject", "reject"), ("✗", "reject"), ("x", "reject"), ("no", "reject"), ("", None), (None, None)])
def test_parse_outcome(text, expected):
    assert parse_outcome(text) == expected


def test_parse_actions():
    assert parse_actions("Rewrite ¶2 (alice); Rerun fig (bob)\nThink more") == [
        ("Rewrite ¶2", "alice"), ("Rerun fig", "bob"), ("Think more", None)]
    assert parse_actions("") == []
```
Run: `cd project-hub && pytest tests/test_agenda_build_read.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement build.py**
```python
# project-hub/hub/agenda/build.py
"""Render an agenda draft into a fixed-format xlsx (the agent never styles cells itself)."""
from __future__ import annotations

import math
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .draft import DraftError, validate_draft

SHEET_SINCE, SHEET_Q, SHEET_D, SHEET_META = "Since last meeting", "Open questions", "Decisions", "_meta"
HEADER_ROW = 4
OUTCOME_CHOICES = ("Approve", "Modify", "Defer", "Reject")
UNPLANNED_ROWS = 5
SINCE_COLS = [("id", 16), ("Decision", 40), ("Actions · owners", 40), ("Follow-up", 50), ("Outcome", 16), ("Affected", 20)]
Q_COLS = [("id", 16), ("Title", 35), ("Description", 60), ("Owner", 12), ("Age (d)", 9), ("Priority", 9), ("Proposed decision", 18)]
D_COLS = [("id", 16), ("What to decide", 40), ("Proposed action", 40), ("Reason", 40), ("Who weighs in", 16),
          ("Min", 6), ("Figure / pre-read", 25), ("Outcome", 12), ("As decided", 40), ("New actions · owners", 40)]
D_HEADERS = [n for n, _ in D_COLS]

HEADER_FILL = PatternFill("solid", fgColor="404040")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=14)
WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Border(bottom=Side(style="thin", color="BFBFBF"))
OUTCOME_FILLS = {"closed question": "E2F0D9", "new question": "DDEBF7", "pivot": "FCE4D6"}
UNPLANNED_FILL = PatternFill("solid", fgColor="FFF2CC")
DECIDE_FILL = PatternFill("solid", fgColor="F2F2F2")


def _sheet(wb, title, cols, draft, first=False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    ws["B1"] = f"{draft['series']} — {draft['meeting_date'].isoformat()} · {title}"  # B: column A is hidden
    ws["B1"].font = TITLE_FONT
    for i, (name, width) in enumerate(cols, 1):
        c = ws.cell(HEADER_ROW, i, name)
        c.fill, c.font, c.alignment = HEADER_FILL, HEADER_FONT, WRAP
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.column_dimensions["A"].hidden = True
    ws.freeze_panes = f"B{HEADER_ROW + 1}"
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    return ws


def _row(ws, r, values, fill=None):
    for i, v in enumerate(values, 1):
        c = ws.cell(r, i, v)
        c.alignment, c.border = WRAP, THIN
        if fill:
            c.fill = fill
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
            c.hyperlink, c.font = v, Font(color="0563C1", underline="single")


def build_workbook(draft: dict, out_path: Path) -> None:
    errs = validate_draft(draft)
    if errs:
        raise DraftError("; ".join(errs))
    wb = Workbook()
    ws = _sheet(wb, SHEET_SINCE, SINCE_COLS, draft, first=True)
    ws.merge_cells(start_row=2, start_column=2, end_row=2, end_column=len(SINCE_COLS))
    ws["B2"] = draft["summary"]
    ws["B2"].alignment = WRAP
    ws.row_dimensions[2].height = 15 * max(2, math.ceil(len(draft["summary"]) / 150) + 1)
    if draft["previous_meeting"] is None:
        _row(ws, HEADER_ROW + 1, ["", "First meeting of this series — nothing to follow up.", "", "", "", ""])
    for r, s in enumerate(draft["since_last"], HEADER_ROW + 1):
        fill = PatternFill("solid", fgColor=OUTCOME_FILLS[s["outcome"]]) if s["outcome"] in OUTCOME_FILLS else None
        _row(ws, r, [s["id"], s["decision"], s["actions"], s["follow_up"], s["outcome"], s["affected"]])
        if fill:
            ws.cell(r, 5).fill = fill
    ws = _sheet(wb, SHEET_Q, Q_COLS, draft)
    for r, q in enumerate(draft["questions"], HEADER_ROW + 1):
        _row(ws, r, [q["id"], q["title"], q["description"], q["owner"], q["age_days"], q["priority"],
                     q["proposed_decision"]])
    ws = _sheet(wb, SHEET_D, D_COLS, draft)
    r = HEADER_ROW + 1
    for d in draft["decisions"]:
        _row(ws, r, [d["id"], d["decide"], d["proposed_action"], d["reason"], d["weigh_in"], d["minutes"],
                     d["link"], "", "", ""])
        for col in (8, 9, 10):
            ws.cell(r, col).fill = DECIDE_FILL
        r += 1
    for k in range(1, UNPLANNED_ROWS + 1):
        _row(ws, r, [f"NEW-{k}", "", "", "", "", "", "", "", "", ""], fill=UNPLANNED_FILL)
        r += 1
    ws.cell(2, 2, f"Yellow rows: decisions raised in the meeting. Outcome: {' / '.join(OUTCOME_CHOICES)}.")
    dv = DataValidation(type="list", formula1='"' + ",".join(OUTCOME_CHOICES) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"H{HEADER_ROW + 1}:H{r - 1}")
    meta = wb.create_sheet(SHEET_META)
    for i, (k, v) in enumerate([("series", draft["series"]), ("meeting_date", draft["meeting_date"].isoformat()),
                                ("read_back", "false"), ("format", "1")], 1):
        meta.cell(i, 1, k)
        meta.cell(i, 2, v)
    meta.sheet_state = "hidden"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def set_read_back(path: Path, value: bool) -> None:
    wb = load_workbook(path)
    if SHEET_META not in wb.sheetnames:
        meta = wb.create_sheet(SHEET_META)
        meta.sheet_state = "hidden"
    meta = wb[SHEET_META]
    for row in meta.iter_rows(min_row=1, max_row=meta.max_row):
        if row[0].value == "read_back":
            row[1].value = "true" if value else "false"
            break
    else:
        meta.append(["read_back", "true" if value else "false"])
    wb.save(path)
```

- [ ] **Step 3: Implement read.py**
```python
# project-hub/hub/agenda/read.py
"""Read a meeting-edited agenda workbook back: matched by hidden id, tolerant of viewers that drop features."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from openpyxl import load_workbook

from .build import SHEET_D, SHEET_META

TITLE_RE = re.compile(r"^(\S+) — (\d{4}-\d{2}-\d{2}) · ")
ACTION_RE = re.compile(r"^(.*?)\s*\(([\w-]+)\)\s*$")


@dataclass
class DecisionRow:
    id: str | None
    decide: str
    outcome: str | None
    as_decided: str
    new_actions: str
    planned: bool


@dataclass
class ReadBack:
    series: str | None
    meeting_date: date | None
    read_back: bool
    rows: list[DecisionRow]


def parse_outcome(text) -> str | None:
    t = str(text or "").strip().lower()
    if not t:
        return None
    if t[0] in "✓✔" or t.startswith(("a", "y")):
        return "approve"
    if t.startswith("m"):
        return "modify"
    if t.startswith("d"):
        return "defer"
    if t[0] in "✗✘x" or t.startswith(("r", "n")):
        return "reject"
    return None


def parse_actions(text) -> list[tuple[str, str | None]]:
    out = []
    for part in re.split(r"[;\n]", str(text or "")):
        part = part.strip()
        if not part:
            continue
        m = ACTION_RE.match(part)
        out.append((m[1].strip(), m[2]) if m else (part, None))
    return out


def _cell(v) -> str:
    return "" if v is None else str(v).strip()


def read_workbook(path) -> ReadBack:
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    series = meeting = None
    read_back = False
    if SHEET_META in wb.sheetnames:
        kv = {_cell(r[0].value): _cell(r[1].value) for r in wb[SHEET_META].iter_rows() if len(r) >= 2}
        series = kv.get("series") or None
        meeting = date.fromisoformat(kv["meeting_date"]) if kv.get("meeting_date") else None
        read_back = kv.get("read_back") == "true"
    if series is None or meeting is None:
        for ref in ("B1", "A1"):  # A1 if a viewer dropped the hidden id column
            m = TITLE_RE.match(_cell(ws[ref].value))
            if m:
                series, meeting = m[1], date.fromisoformat(m[2])
                break
    header_row, cols = None, {}
    for r in range(1, 11):
        values = [_cell(c.value) for c in ws[r]]
        if "What to decide" in values:
            header_row = r
            cols = {v: i + 1 for i, v in enumerate(values) if v}
            break
    if header_row is None:
        raise ValueError(f"{path}: no 'What to decide' header in the first 10 rows of '{SHEET_D}'")

    def get(r, name):
        return _cell(ws.cell(r, cols[name]).value) if name in cols else ""

    rows = []
    for r in range(header_row + 1, ws.max_row + 1):
        rid = get(r, "id") or None
        decide = get(r, "What to decide")
        if not decide:
            continue
        planned = rid is not None and not rid.startswith("NEW")
        if rid is None:
            planned = False
        rows.append(DecisionRow(rid, decide, parse_outcome(get(r, "Outcome")), get(r, "As decided"),
                                get(r, "New actions · owners"), planned))
    return ReadBack(series, meeting, read_back, rows)
```
`planned` is False whenever the ID column is missing. `read_agenda` (Task 11) then matches those rows to proposed decisions by exact title.

- [ ] **Step 4: Run tests**
Run: `cd project-hub && pytest tests/test_agenda_build_read.py -v`
Expected: 20 passed (6 tests + 14 parametrized outcome cases). In `test_read_without_meta_or_id_column`, deleting column A shifts Outcome from H to G. The test re-labels G4 and fills G5 by hand, which simulates a viewer that drops the hidden column.

- [ ] **Step 5: Commit**
```bash
git add project-hub/hub/agenda/build.py project-hub/hub/agenda/read.py project-hub/tests/test_agenda_build_read.py
git commit -m "feat(hub): styled agenda workbook build and tolerant read-back"
```

---
### Task 11: Agenda lifecycle — refusal, apply read-back, meeting note, archive
**Files:**
- Create: `project-hub/hub/agenda/lifecycle.py`
- Test: `project-hub/tests/test_agenda_lifecycle.py`

**Interfaces:**
- Consumes:
  - `build_workbook`, `set_read_back`, `read_workbook`, `parse_actions` (Task 10)
  - `load_draft` (Task 9)
  - `add_proposal` (Task 7)
  - `Store` (Task 4)
  - `HubConfig` (Task 2)
- Produces:
  - `AgendaError`
  - `live_path(root, series) -> Path`
  - `build_agenda(root, series, *, force=False) -> Path`
  - `read_agenda(store, config, series, *, author, today) -> dict` with keys `applied: list[str]`, `staged: list[str]` (pending IDs), `not_reached: list[str]`, `meeting_note: Path`, `archive: Path`

Rules:
- `build_agenda` raises `AgendaError` if the live xlsx exists, its `read_back` is false, and its meeting date is earlier than the draft's (unless `force`). It also raises if the live file's series differs.
- `read_agenda` raises `AgendaError` if the live file is missing, already read back, or for another series.
- Row → store mapping, with `tag = f"{series}/{date}"` and `source = f"agenda:{tag}"`:
  - A planned row with a known ID, or an unplanned row whose `decide` exactly matches a `proposed` decision's title, resolves to that decision.
  - **approve/modify:** set `status: made`, `meeting: tag`, and `as_decided` (if non-empty). Add the entry `"Approved at {tag}: {as_decided or 'as proposed'}"` (or "Modified at"). If `new_actions` is non-empty, stage a proposal "Actions from {tag}: {decide}" whose ops are `new action` × N with `spawned_by: [decision id]`, owner from `parse_actions`, and `series: [series]`.
  - **defer:** set `status: deferred` and add the entry `"Deferred at {tag}."`
  - **reject:** set `status: rejected` and `closed: today`, and add the entry `"Rejected at {tag}: {as_decided or 'no reason recorded'}"`.
  - **None:** add the entry `"Not reached at {tag}."` and list the ID in `not_reached`.
  - An unresolved row (raised in the meeting) with an outcome, or with "as decided" text: stage a proposal "Raised at {tag}: {decide}". Its ops are a `new decision` (ref `$d`) with fields `status` (made, deferred or rejected from the outcome, defaulting to made), `meeting: tag` and `as_decided`, plus the parsed actions spawned by `$d`. The proposal's reason is `"Decision raised in meeting; link it to the question(s) it answers before applying."`
- Meeting note `Status/Meetings/{series}/{date}.md`:
```
# {series} — {date}
- attendees: alice, bob
- source: agenda:{tag}
## Decisions
- [[D-…]] approve — Adopt Y but only for 4x3
## Not reached
- [[D-…]]
## Staged for approval
- P-xxxx · Raised at group/2026-11-19: Drop UCJ
```
  Empty sections print `None.`
- Then: `set_read_back(live, True)`, copy the live file to `Agenda/archive/{series}/{date}.xlsx`, and append a log entry `"Agenda read back: {n} applied, {m} staged"`.

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_agenda_lifecycle.py
from datetime import date
import pytest
from openpyxl import load_workbook
from hub.agenda.build import SHEET_D
from hub.agenda.draft import save_draft, scaffold
from hub.agenda.lifecycle import AgendaError, build_agenda, live_path, read_agenda
from hub.config import load_config
from hub.pending import load_proposals
from hub.store import Store

MEET = date(2026, 11, 19)


@pytest.fixture
def env(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    s, cfg = Store(hub_root), load_config(hub_root)
    q = s.create("question", "Why?", owner="alice", author="alice", day=date(2026, 11, 1), source="m", series=["group"])
    d1 = s.create("decision", "Adopt Y", owner="alice", author="alice", day=date(2026, 11, 2), source="m",
                  series=["group"], links={"answers": [q.id]})
    d2 = s.create("decision", "Drop Z", owner="bob", author="alice", day=date(2026, 11, 2), source="m",
                  series=["group"], links={"answers": [q.id]})
    d3 = s.create("decision", "Try W", owner="bob", author="alice", day=date(2026, 11, 2), source="m",
                  series=["group"], links={"answers": [q.id]})
    draft = scaffold(s, hub_root, "group", MEET)
    draft["summary"] = "Status."
    save_draft(hub_root, draft)
    return s, cfg, hub_root, dict(q=q, d1=d1, d2=d2, d3=d3)


def fill(path, rows):
    wb = load_workbook(path)
    ws = wb[SHEET_D]
    for ref, val in rows.items():
        ws[ref] = val
    wb.save(path)


def row_of(path, item_id):
    ws = load_workbook(path)[SHEET_D]
    for r in range(5, ws.max_row + 1):
        if ws.cell(r, 1).value == item_id:
            return r
    raise AssertionError(item_id)


def test_build_and_read_back(env):
    s, cfg, root, ids = env
    path = build_agenda(root, "group")
    assert path == live_path(root, "group")
    r1, r2 = row_of(path, ids["d1"].id), row_of(path, ids["d2"].id)
    fill(path, {f"H{r1}": "Modify", f"I{r1}": "Only 4x3", f"J{r1}": "Rewrite ¶2 (alice)",
                f"H{r2}": "Reject", f"I{r2}": "Not worth it",
                "B8": "Drop UCJ", "H8": "approve", "I8": "Dropped", "J8": "Tell Tamra (bob)"})
    res = read_agenda(s, cfg, "group", author="bob", today=date(2026, 11, 19))
    d1 = s.get(ids["d1"].id)
    assert d1.status == "made" and d1.meta["meeting"] == "group/2026-11-19" and d1.meta["as_decided"] == "Only 4x3"
    assert d1.entries[-1].text == "Modified at group/2026-11-19: Only 4x3"
    d2 = s.get(ids["d2"].id)
    assert d2.status == "rejected" and d2.meta["closed"] == date(2026, 11, 19)
    assert s.get(ids["d3"].id).entries[-1].text == "Not reached at group/2026-11-19."
    assert res["not_reached"] == [ids["d3"].id]
    props = load_proposals(root)
    titles = sorted(p.title for p in props)
    assert titles == ["Actions from group/2026-11-19: Adopt Y", "Raised at group/2026-11-19: Drop UCJ"]
    raised = next(p for p in props if p.title.startswith("Raised"))
    assert raised.body["ops"][0]["fields"]["status"] == "made"
    assert raised.body["ops"][1]["owner"] == "bob"
    note = (root / "Status" / "Meetings" / "group" / "2026-11-19.md").read_text()
    assert note.startswith("# group — 2026-11-19\n- attendees: alice, bob\n")
    assert f"- [[{ids['d1'].id}]] modify — Only 4x3" in note
    assert (root / "Agenda" / "archive" / "group" / "2026-11-19.xlsx").is_file()
    with pytest.raises(AgendaError, match="already read back"):
        read_agenda(s, cfg, "group", author="bob", today=date(2026, 11, 19))


def test_build_refuses_unread_previous(env):
    s, cfg, root, _ = env
    build_agenda(root, "group")
    draft = scaffold(s, root, "group", date(2026, 11, 26))
    draft["summary"] = "Next."
    save_draft(root, draft)
    with pytest.raises(AgendaError, match="not been read back"):
        build_agenda(root, "group")
    assert build_agenda(root, "group", force=True).is_file()


def test_read_requires_live_file(env):
    s, cfg, root, _ = env
    with pytest.raises(AgendaError, match="no live agenda"):
        read_agenda(s, cfg, "group", author="bob", today=MEET)
```
Run: `cd project-hub && pytest tests/test_agenda_lifecycle.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/agenda/lifecycle.py
"""Agenda lifecycle: build (with the 'previous meeting not read back' guard) and read back into the store."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from ..config import HubConfig
from ..pending import add_proposal
from ..store import Store
from .build import build_workbook, set_read_back
from .draft import load_draft
from .read import parse_actions, read_workbook

VERB = {"approve": "Approved", "modify": "Modified"}
RAISED_STATUS = {"approve": "made", "modify": "made", "defer": "deferred", "reject": "rejected", None: "made"}


class AgendaError(Exception):
    """The agenda lifecycle would lose meeting edits or act on the wrong file."""


def live_path(root, series: str) -> Path:
    return Path(root) / "Agenda" / f"{series}.xlsx"


def build_agenda(root, series: str, *, force: bool = False) -> Path:
    draft = load_draft(root, series)
    live = live_path(root, series)
    if live.is_file():
        rb = read_workbook(live)
        if rb.series and rb.series != series:
            raise AgendaError(f"{live} belongs to series '{rb.series}', not '{series}'")
        if (not force and not rb.read_back and rb.meeting_date and rb.meeting_date < draft["meeting_date"]):
            raise AgendaError(f"the {rb.meeting_date} agenda has not been read back; run "
                              f"`hub agenda read {series}` first (or pass --force to discard its edits)")
    build_workbook(draft, live)
    return live


def _action_ops(text, parent, series, source):
    return [{"op": "new", "type": "action", "title": t, "owner": o, "series": [series],
             "links": {"spawned_by": [parent]}, "source": source} for t, o in parse_actions(text)]


def read_agenda(store: Store, config: HubConfig, series: str, *, author: str, today: date) -> dict:
    root = Path(config.root)
    live = live_path(root, series)
    if not live.is_file():
        raise AgendaError(f"no live agenda at {live}")
    rb = read_workbook(live)
    if rb.read_back:
        raise AgendaError(f"{live} was already read back")
    if rb.series and rb.series != series:
        raise AgendaError(f"{live} belongs to series '{rb.series}', not '{series}'")
    if rb.meeting_date is None:
        raise AgendaError(f"{live} has no meeting date (neither _meta nor sheet title)")
    tag = f"{series}/{rb.meeting_date.isoformat()}"
    source = f"agenda:{tag}"
    decisions = {i.id: i for i in store.all() if i.type == "decision"}
    by_title = {d.title: d.id for d in decisions.values() if d.status == "proposed"}
    applied, staged, not_reached, lines = [], [], [], []
    for row in rb.rows:
        did = row.id if row.planned and row.id in decisions else by_title.get(row.decide)
        if did is None:
            if row.outcome or row.as_decided:
                ops = [{"op": "new", "ref": "$d", "type": "decision", "title": row.decide, "owner": author,
                        "series": [series], "source": source,
                        "fields": {"status": RAISED_STATUS[row.outcome], "meeting": tag,
                                   "as_decided": row.as_decided}}]
                ops += _action_ops(row.new_actions, "$d", series, source)
                pid = add_proposal(root, f"Raised at {tag}: {row.decide}", ops, proposer=author, source=source,
                                   reason="Decision raised in meeting; link it to the question(s) it answers before applying.",
                                   day=today)
                staged.append((pid, f"Raised at {tag}: {row.decide}"))
            continue
        if row.outcome in ("approve", "modify"):
            fields = {"status": "made", "meeting": tag}
            if row.as_decided:
                fields["as_decided"] = row.as_decided
            store.set_fields(did, fields)
            store.entry(did, f"{VERB[row.outcome]} at {tag}: {row.as_decided or 'as proposed'}",
                        author=author, source=source, day=today)
            if row.new_actions:
                title = f"Actions from {tag}: {decisions[did].title}"
                pid = add_proposal(root, title, _action_ops(row.new_actions, did, series, source),
                                   proposer=author, source=source, reason="Free-text actions written in the meeting.",
                                   day=today)
                staged.append((pid, title))
        elif row.outcome == "defer":
            store.set_fields(did, {"status": "deferred"})
            store.entry(did, f"Deferred at {tag}.", author=author, source=source, day=today)
        elif row.outcome == "reject":
            store.set_fields(did, {"status": "rejected", "closed": today})
            store.entry(did, f"Rejected at {tag}: {row.as_decided or 'no reason recorded'}",
                        author=author, source=source, day=today)
        else:
            store.entry(did, f"Not reached at {tag}.", author=author, source=source, day=today)
            not_reached.append(did)
            continue
        applied.append(did)
        lines.append(f"- [[{did}]] {row.outcome} — {row.as_decided or decisions[did].title}")
    attendees = ", ".join((config.series.get(series) or {}).get("attendees") or [])
    note = [f"# {series} — {rb.meeting_date.isoformat()}", f"- attendees: {attendees}", f"- source: {source}",
            "## Decisions", *(lines or ["None."]),
            "## Not reached", *([f"- [[{i}]]" for i in not_reached] or ["None."]),
            "## Staged for approval", *([f"- {p} · {t}" for p, t in staged] or ["None."])]
    note_path = root / "Status" / "Meetings" / series / f"{rb.meeting_date.isoformat()}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(note) + "\n")
    set_read_back(live, True)
    archive = root / "Agenda" / "archive" / series / f"{rb.meeting_date.isoformat()}.xlsx"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live, archive)
    store.log(f"Agenda read back: {len(applied)} applied, {len(staged)} staged", author=author,
              source=source, day=today)
    return {"applied": applied, "staged": [p for p, _ in staged], "not_reached": not_reached,
            "meeting_note": note_path, "archive": archive}
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_agenda_lifecycle.py -v`
Expected: 3 passed. In `test_build_and_read_back`, row 8 is the first "NEW-" row: three decisions occupy rows 5–7.

- [ ] **Step 4: Commit**
```bash
git add project-hub/hub/agenda/lifecycle.py project-hub/tests/test_agenda_lifecycle.py
git commit -m "feat(hub): agenda read-back into the store, meeting note, archive, unread-guard"
```

---
### Task 12: CLI
**Files:**
- Create: `project-hub/hub/cli.py`
- Test: `project-hub/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv: list[str] | None = None) -> int`, and the console script `hub`. Exit codes: 0 ok, 1 validation violations or a user error (message on stderr), 2 argparse error, 3 sync conflict.

- [ ] **Step 1: Write failing tests**
```python
# project-hub/tests/test_cli.py
import json
import pytest
from hub.cli import main
from hub.store import Store


@pytest.fixture
def cli(hub_root, monkeypatch):
    monkeypatch.chdir(hub_root)
    monkeypatch.delenv("HUB_ME", raising=False)
    monkeypatch.setenv("HUB_TODAY", "2026-11-12")
    return hub_root


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_new_list_show_set(cli, capsys):
    code, qid, _ = run(capsys, "new", "q", "Why?", "--source", "manual", "--series", "group", "--priority", "1")
    qid = qid.strip()
    assert code == 0 and qid.startswith("Q-20261112-")
    code, did, _ = run(capsys, "new", "d", "Do X", "--source", "manual", "--link", f"answers:{qid}")
    did = did.strip()
    code, out, _ = run(capsys, "list", "--type", "question", "--json")
    assert json.loads(out)[0]["id"] == qid
    code, out, _ = run(capsys, "show", qid, "--chain")
    assert qid in out and f"  {did}" in out
    assert run(capsys, "entry", qid, "More info.", "--source", "commit:x@1")[0] == 0
    assert run(capsys, "set", qid, "status=answered", "closed=2026-11-12")[0] == 0
    assert Store(cli).get(qid).status == "answered"


def test_validate_exit_codes(cli, capsys):
    assert run(capsys, "validate")[0] == 0
    run(capsys, "new", "d", "Do X", "--source", "manual")
    did = Store(cli).query(type="decision")[0].id
    run(capsys, "set", did, "status=made")
    code, out, _ = run(capsys, "validate")
    assert code == 1 and "decision-answers" in out


def test_user_error_is_clean(cli, capsys):
    code, _, err = run(capsys, "set", "Q-20990101-zzz", "status=open")
    assert code == 1 and "Q-20990101-zzz" in err


def test_writes_need_me(cli, capsys):
    (cli / "hub.local.yaml").unlink()
    code, _, err = run(capsys, "new", "q", "Why?", "--source", "m")
    assert code == 1 and "hub.local.yaml" in err


def test_agenda_flow_and_pending(cli, capsys):
    run(capsys, "new", "q", "Why?", "--source", "m", "--series", "group")
    code, out, _ = run(capsys, "agenda", "scaffold", "group", "--date", "2026-11-19")
    assert code == 0 and "group.draft.yaml" in out
    code, _, err = run(capsys, "agenda", "build", "group")
    assert code == 1 and "summary is empty" in err
    draft = cli / "Agenda" / "group.draft.yaml"
    draft.write_text(draft.read_text().replace("summary: ''", "summary: Status."))
    assert run(capsys, "agenda", "build", "group")[0] == 0
    code, out, _ = run(capsys, "agenda", "read", "group")
    assert code == 0 and "applied 0" in out
    code, out, _ = run(capsys, "pending", "list")
    assert code == 0 and "No pending" in out


def test_snapshot_and_sync_state(cli, capsys):
    assert run(capsys, "snapshot")[0] == 0
    assert (cli / "Status" / "Snapshot.md").is_file()
    assert run(capsys, "sync-state", "set", "clickup", "2026-11-12T09:00")[0] == 0
    assert run(capsys, "sync-state", "get", "clickup")[1].strip() == "2026-11-12T09:00"


def test_help_documents_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("new", "entry", "set", "link", "list", "show", "since", "log", "validate", "snapshot",
                "pending", "agenda", "sync-state", "commit"):
        assert cmd in out
```
Run: `cd project-hub && pytest tests/test_cli.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/cli.py
"""hub — command line for a project hub vault (status store, staged changes, agenda, git sync).

Run from anywhere inside a vault (the nearest parent directory containing project.yaml is used), or pass
--root. Writes need your people key: set `me: <key>` in hub.local.yaml (gitignored) or HUB_ME=<key>.
"Today" is the system date unless HUB_TODAY=YYYY-MM-DD is set (used for tests and back-dating).

Commands (each has its own --help):
  new {q,d,a} TITLE        create a question / decision / action; prints the new id
      --source S             (required) provenance: circleback:<id>, commit:<repo>@<sha>, clickup:<id>,
                             agenda:<series>/<date>, manual, …
      --owner KEY            people key from project.yaml (default: you)
      --series NAME          meeting series this item may appear on; repeatable (default: none = all)
      --priority {1,2,3}     1 = highest (default 2)
      --link REL:ID          REL ∈ answers | spawned_by | pivoted_to; repeatable
      --description TEXT     one-paragraph description
  entry ID TEXT --source S   append a dated history entry authored by you
  set ID KEY=VALUE …         set frontmatter fields (not id/type/links). Dates as YYYY-MM-DD; empty VALUE
                             clears. status must be valid for the type:
                               question: open|answered|dropped|pivoted
                               decision: proposed|made|rejected|deferred
                               action:   open|done|dropped
  link ID REL TARGET         add a link; REL ∈ answers | spawned_by | pivoted_to
  list [filters] [--json]    --type {question,decision,action} --status S --owner KEY|me --series NAME
                             --since YYYY-MM-DD (last activity on/after)
  show ID [--chain]          print one item; --chain also prints everything descending from it
  since DATE                 items with activity on/after DATE (same as list --since)
  log TEXT --source S        append to Status/Log/<YYYY-MM>.md
  validate                   check every rule; exit 1 and list violations if any
  snapshot                   regenerate Status/Snapshot.md
  pending list               show staged proposals
  pending apply PID …        apply proposals (P-xxxx) in order; each is removed once applied
  pending reject PID …       drop proposals, logging the rejection
  pending add --title T --ops-file F.yaml --source S --reason R
                             stage a proposal whose ops are read from a YAML list (see .agent_docs/pending-format.md)
  agenda scaffold SERIES --date YYYY-MM-DD
                             write Agenda/<series>.draft.yaml pre-filled from the store (summary/follow-ups empty)
  agenda build SERIES [--force]
                             render the draft into Agenda/<series>.xlsx; refuses while the previous meeting's
                             workbook is unread, unless --force (which discards its meeting edits)
  agenda read SERIES         apply the meeting's edits to the store, stage free-text items in Pending.md,
                             write the meeting note, archive the workbook
  sync-state get SOURCE | set SOURCE VALUE
                             read/write ingest watermarks in Status/sync-state.yaml
  commit MESSAGE             git add -A, commit, pull --rebase, regenerate Snapshot.md, push (3 retries);
                             exit 3 on a genuine merge conflict (rebase is aborted, nothing lost)

Exit codes: 0 ok · 1 user error or validation violations · 2 bad arguments · 3 sync conflict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import yaml

from .agenda.draft import DraftError, save_draft, scaffold
from .agenda.lifecycle import AgendaError, build_agenda, read_agenda
from .config import ConfigError, find_root, load_config
from .gitsync import SyncConflict, SyncError, commit
from .items import LINK_KEYS, render_item
from .pending import PendingError, add_proposal, apply_proposals, load_proposals, reject_proposals
from .snapshot import write_snapshot
from .store import Store, StoreError
from .sync_state import get_watermark, set_watermark
from .validate import validate

TYPE_ALIASES = {"q": "question", "d": "decision", "a": "action"}


class UserError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="vault root (default: nearest parent containing project.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    n = sub.add_parser("new", help="create an item")
    n.add_argument("type", choices=sorted(TYPE_ALIASES))
    n.add_argument("title")
    n.add_argument("--source", required=True)
    n.add_argument("--owner")
    n.add_argument("--series", action="append", default=[])
    n.add_argument("--priority", type=int, choices=(1, 2, 3), default=2)
    n.add_argument("--link", action="append", default=[], metavar="REL:ID")
    n.add_argument("--description", default="")

    e = sub.add_parser("entry", help="append a history entry")
    e.add_argument("id")
    e.add_argument("text")
    e.add_argument("--source", required=True)

    s = sub.add_parser("set", help="set frontmatter fields")
    s.add_argument("id")
    s.add_argument("pairs", nargs="+", metavar="KEY=VALUE")

    l = sub.add_parser("link", help="add a link")
    l.add_argument("id")
    l.add_argument("rel", choices=LINK_KEYS)
    l.add_argument("target")

    ls = sub.add_parser("list", help="query items")
    ls.add_argument("--type", choices=("question", "decision", "action"))
    ls.add_argument("--status")
    ls.add_argument("--owner")
    ls.add_argument("--series")
    ls.add_argument("--since", type=date.fromisoformat)
    ls.add_argument("--json", action="store_true")

    sh = sub.add_parser("show", help="print an item")
    sh.add_argument("id")
    sh.add_argument("--chain", action="store_true")

    si = sub.add_parser("since", help="items with activity since a date")
    si.add_argument("date", type=date.fromisoformat)

    lg = sub.add_parser("log", help="append to the monthly log")
    lg.add_argument("text")
    lg.add_argument("--source", required=True)

    sub.add_parser("validate", help="check store rules")
    sub.add_parser("snapshot", help="regenerate Status/Snapshot.md")

    pe = sub.add_parser("pending", help="staged proposals")
    pes = pe.add_subparsers(dest="pcmd", required=True)
    pes.add_parser("list")
    pa = pes.add_parser("apply")
    pa.add_argument("pids", nargs="+")
    pr = pes.add_parser("reject")
    pr.add_argument("pids", nargs="+")
    pad = pes.add_parser("add")
    pad.add_argument("--title", required=True)
    pad.add_argument("--ops-file", type=Path, required=True)
    pad.add_argument("--source", required=True)
    pad.add_argument("--reason", required=True)

    ag = sub.add_parser("agenda", help="meeting agenda workbook")
    ags = ag.add_subparsers(dest="acmd", required=True)
    asc = ags.add_parser("scaffold")
    asc.add_argument("series")
    asc.add_argument("--date", type=date.fromisoformat, required=True)
    ab = ags.add_parser("build")
    ab.add_argument("series")
    ab.add_argument("--force", action="store_true")
    ar = ags.add_parser("read")
    ar.add_argument("series")

    ss = sub.add_parser("sync-state", help="ingest watermarks")
    sss = ss.add_subparsers(dest="scmd", required=True)
    sg = sss.add_parser("get")
    sg.add_argument("source")
    sst = sss.add_parser("set")
    sst.add_argument("source")
    sst.add_argument("value")

    c = sub.add_parser("commit", help="commit and push with rebase/retry")
    c.add_argument("message")
    return p


def _today() -> date:
    return date.fromisoformat(os.environ["HUB_TODAY"]) if os.environ.get("HUB_TODAY") else date.today()


def _me(cfg) -> str:
    if not cfg.me:
        raise UserError("writes need your people key: add `me: <key>` to hub.local.yaml or set HUB_ME")
    return cfg.me


def _row(it) -> dict:
    return {"id": it.id, "type": it.type, "status": it.status, "title": it.title, "owner": it.owner,
            "priority": it.meta.get("priority"), "last_activity": it.last_activity().isoformat()}


def _print_items(items, as_json):
    if as_json:
        print(json.dumps([_row(i) for i in items], indent=2))
        return
    for i in items:
        print(f"{i.id}  {i.status:<9} {i.owner or '-':<10} {i.title}")


def _dispatch(args, cfg, store) -> int:
    today = _today()
    cmd = args.cmd
    if cmd == "new":
        links: dict[str, list[str]] = {}
        for spec in args.link:
            rel, _, target = spec.partition(":")
            links.setdefault(rel, []).append(target)
        item = store.create(TYPE_ALIASES[args.type], args.title, owner=args.owner or _me(cfg), author=_me(cfg),
                            day=today, source=args.source, series=args.series, priority=args.priority,
                            links=links, description=args.description)
        print(item.id)
    elif cmd == "entry":
        store.entry(args.id, args.text, author=_me(cfg), source=args.source, day=today)
    elif cmd == "set":
        fields = {}
        for pair in args.pairs:
            k, sep, v = pair.partition("=")
            if not sep:
                raise UserError(f"expected KEY=VALUE, got '{pair}'")
            fields[k] = v
        store.set_fields(args.id, fields)
    elif cmd == "link":
        store.link(args.id, args.rel, args.target)
    elif cmd == "list":
        owner = _me(cfg) if args.owner == "me" else args.owner
        _print_items(store.query(type=args.type, status=args.status, owner=owner, series=args.series,
                                 since=args.since), args.json)
    elif cmd == "show":
        print(render_item(store.get(args.id)), end="")
        if args.chain:
            seen = {args.id}

            def walk(item_id, depth):
                for ch in store.children(item_id):
                    if ch.id in seen:
                        continue
                    seen.add(ch.id)
                    print(f"{'  ' * depth}{ch.id}  [{ch.type}/{ch.status}] {ch.title}")
                    walk(ch.id, depth + 1)

            print("## Chain")
            walk(args.id, 1)
    elif cmd == "since":
        _print_items(store.query(since=args.date), False)
    elif cmd == "log":
        store.log(args.text, author=_me(cfg), source=args.source, day=today)
    elif cmd == "validate":
        vs = validate(store, cfg)
        for v in vs:
            print(f"{v.item_id}  {v.rule}: {v.message}")
        print(f"{len(vs)} violation(s)")
        return 1 if vs else 0
    elif cmd == "snapshot":
        print(write_snapshot(store, cfg, today))
    elif cmd == "pending":
        if args.pcmd == "list":
            props = load_proposals(cfg.root)
            if not props:
                print("No pending proposals.")
            for pr in props:
                print(f"{pr.pid}  {pr.title}  (by {pr.body.get('proposer')}, {pr.body.get('date')}): "
                      f"{pr.body.get('reason')}")
        elif args.pcmd == "apply":
            for i in apply_proposals(cfg.root, store, args.pids, author=_me(cfg), day=today):
                print(f"created {i}")
        elif args.pcmd == "reject":
            reject_proposals(cfg.root, store, args.pids, author=_me(cfg), day=today)
        else:
            ops = yaml.safe_load(args.ops_file.read_text()) or []
            print(add_proposal(cfg.root, args.title, ops, proposer=_me(cfg), source=args.source,
                               reason=args.reason, day=today))
    elif cmd == "agenda":
        if args.acmd == "scaffold":
            if args.series not in cfg.series:
                raise UserError(f"unknown series '{args.series}' (project.yaml series: {', '.join(cfg.series)})")
            print(save_draft(cfg.root, scaffold(store, cfg.root, args.series, args.date)))
        elif args.acmd == "build":
            print(build_agenda(cfg.root, args.series, force=args.force))
        else:
            res = read_agenda(store, cfg, args.series, author=_me(cfg), today=today)
            print(f"applied {len(res['applied'])}, staged {len(res['staged'])} "
                  f"({' '.join(res['staged'])}), not reached {len(res['not_reached'])}")
            print(f"meeting note: {res['meeting_note']}\narchive: {res['archive']}")
    elif cmd == "sync-state":
        if args.scmd == "get":
            print(get_watermark(cfg.root, args.source) or "")
        else:
            set_watermark(cfg.root, args.source, args.value)
    elif cmd == "commit":
        print(commit(cfg.root, args.message, regenerate=lambda: write_snapshot(store, cfg, today)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.root or find_root(Path.cwd())
        cfg = load_config(root)
        return _dispatch(args, cfg, Store(root))
    except SyncConflict as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyError as exc:
        print(f"error: no such item {exc}", file=sys.stderr)
        return 1
    except (UserError, ConfigError, StoreError, PendingError, DraftError, AgendaError, SyncError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run tests**
Run: `cd project-hub && pytest tests/test_cli.py -v`
Expected: 7 passed. `test_agenda_flow_and_pending` relies on `save_draft` writing `summary: ''`. That is PyYAML's rendering of an empty string. If PyYAML renders it differently, change the test's `replace` to match the rendering.

- [ ] **Step 4: Run the full suite**
Run: `cd project-hub && pytest -q`
Expected: `76 passed` (verified 2026-11-12 by extracting this plan's code blocks and running them).

- [ ] **Step 5: Commit**
```bash
git add project-hub/hub/cli.py project-hub/tests/test_cli.py
git commit -m "feat(hub): documented CLI over store, pending, agenda, sync-state and commit"
```

---
### Task 13: Agent docs, README, end-to-end smoke test
**Files:**
- Create:
  - `project-hub/README.md`
  - `project-hub/.agent_docs/_INDEX.md`
  - `project-hub/.agent_docs/item-format.md`
  - `project-hub/.agent_docs/pending-format.md`
  - `project-hub/.agent_docs/agenda-workbook.md`
  - `project-hub/.agent_docs/git-sync.md`

- [ ] **Step 1: Write `.agent_docs/_INDEX.md`**
```markdown
# project-hub agent docs
Read this first. Each file below is the complete truth for its domain; do not read the code to learn a format.
- [item-format.md](item-format.md) — item files under `Status/Items/`: every frontmatter key, statuses, links, entries, IDs, validation rules (`hub/items.py`, `hub/store.py`, `hub/validate.py`)
- [pending-format.md](pending-format.md) — `Status/Pending.md` proposals and the op schema (`hub/pending.py`)
- [agenda-workbook.md](agenda-workbook.md) — draft YAML keys, the three sheets and their columns, `_meta`, read-back mapping (`hub/agenda/`)
- [git-sync.md](git-sync.md) — `.gitattributes` merge drivers, `hub commit` sequence, conflict handling (`hub/gitsync.py`)
- [office-viewer-spike.md](office-viewer-spike.md) — what VS Code Office Viewer preserves on save
Also: `Status/Log/<YYYY-MM>.md` is `# Log YYYY-MM` followed by entry lines in the item entry format; `Status/sync-state.yaml` has keys `watermarks` (source → opaque string), `next_meeting` (series → date), `flags` (list of strings) — `hub/sync_state.py`.
```

- [ ] **Step 2: Write `.agent_docs/item-format.md`**
It must contain:
- the full annotated example file from Task 3's `CANONICAL`;
- a table of every frontmatter key: `id, type, title, status, owner, series, raised, closed, priority, links.answers, links.spawned_by, links.pivoted_to, sources`; decision-only `rationale, meeting, as_decided, no_action, proposed_action`; action-only `clickup_id, due`. For each key: type, who sets it, meaning, allowed values;
- what each link direction means (decision `answers` question; action or question `spawned_by` decision or item; question `pivoted_to` question) and the fact that links are stored one way and reverse lookups come from `Store.children`;
- the exact entry regex `^- (\d{4}-\d{2}-\d{2}) · (\S+) · (\S+) — (.*)$` and the source-ID conventions (`circleback:<meeting id>`, `commit:<root>@<short sha>`, `clickup:<task id>`, `agenda:<series>/<date>`, `pending:<P-id>`, `manual`);
- the ID scheme;
- each `validate` rule name with its exact condition (copy the conditions from Task 5).

- [ ] **Step 3: Write `.agent_docs/pending-format.md`, `agenda-workbook.md` and `git-sync.md`**
- `pending-format.md`: the block format (`## P-xxxx · title` + fenced yaml with keys `proposer, date, source, reason, ops`), each op kind with every key (copied from the Task 7 interface block), `$ref` resolution, the non-transactional behaviour and how to recover (`git diff`, `hub validate`), and the rule that any new item an agent infers from free text goes here.
- `agenda-workbook.md`:
  - the draft keys (from the Task 9 interface block);
  - the sheets in order, with the column lists `SINCE_COLS`, `Q_COLS`, `D_COLS` copied verbatim, the header on row 4, data from row 5, and the hidden column A holding the ID (`NEW-k` for raised-in-meeting rows);
  - the `_meta` keys `series, meeting_date, read_back, format`, with the title-row fallback `"<series> — <YYYY-MM-DD> · <sheet>"` in B1 (A1 if the id column was dropped);
  - the outcome parsing table (from `parse_outcome`) and the action syntax `title (owner); …`;
  - every read-back mapping rule (copied from the Task 11 Rules block);
  - the build refusal rule.
- `git-sync.md`: the four `.gitattributes` lines and why each exists, the `.gitignore` lines, the `commit` sequence step by step, what `SyncConflict` means and how a human resolves it (`git pull --rebase`, fix the item file, `hub validate`, `hub commit`), and the warning that the live `.xlsx` must have one editor per meeting.

- [ ] **Step 4: Write `project-hub/README.md`**
Sections: *What this is* (two sentences plus a link to the spec); *Install* (`python -m venv .venv && . .venv/bin/activate && pip install -e '.[test]'`, or `uv pip install -e .`); *First use in a vault* (`project.yaml` minimal example from the spec, `hub.local.yaml` with `me:`); *Everyday commands* (`hub list --type action --owner me --status open`, `hub agenda scaffold/build/read`, `hub pending list/apply`, `hub commit`); *Tests* (`pytest -q`).

- [ ] **Step 5: End-to-end smoke test in a throwaway vault**
```bash
cd /home/jek354/.claude/jobs/08d18e11/tmp && rm -rf smoke && mkdir smoke && cd smoke && git init -q -b main
printf 'name: smoke\npeople:\n  alice: {name: A}\n  bob: {name: B}\nseries:\n  group: {attendees: [alice, bob]}\n' > project.yaml
printf 'me: alice\n' > hub.local.yaml
export HUB_TODAY=2026-11-12
Q=$(hub new q "Does it work?" --source manual --series group --priority 1)
D=$(hub new d "Ship it" --source manual --series group --link answers:$Q)
hub agenda scaffold group --date 2026-11-19
sed -i "s/summary: ''/summary: Smoke test./" Agenda/group.draft.yaml
hub agenda build group
python - <<'EOF'
from openpyxl import load_workbook
wb = load_workbook("Agenda/group.xlsx"); ws = wb["Decisions"]; ws["H5"] = "Approve"; ws["J5"] = "Write docs (bob)"; wb.save("Agenda/group.xlsx")
EOF
HUB_TODAY=2026-11-19 hub agenda read group
hub pending list
hub pending apply $(hub pending list | awk '/^P-/{print $1}')
hub validate && hub snapshot && hub commit "smoke" && cat Status/Snapshot.md
```
Expected:
- `agenda read` prints `applied 1, staged 1`.
- `pending apply` prints `created A-2026…`.
- `validate` prints `0 violation(s)`.
- The snapshot lists one open action for `bob`.
- `git log` shows the commit.

Delete `smoke/` afterwards.

- [ ] **Step 6: Commit**
```bash
git add project-hub/README.md project-hub/.agent_docs
git commit -m "docs(hub): agent docs for every file format and the CLI README"
```

---
## Self-review notes (completed while writing)
- **Spec coverage for Plan 1:** status store (Tasks 3–5), Snapshot (6), Pending (7), concurrency and commit (8), agenda draft/build/read/lifecycle with refusal and archive (9–11), CLI table (12; `figures`, `check-links` and `import-handoff` are Plan 3), the Office Viewer spike (1).
- **Deferred to Plan 2:** the Circleback cross-check of the read-back, all ingesters and watermark use, and ClickUp mapping and flags (the flag display is already in Snapshot).
- **Deferred to Plan 3:** stale figures in the Snapshot.
- **Names used consistently across tasks:**
  - `Store.create(..., author=, day=, source=)`
  - `add_proposal(root, title, ops, *, proposer, source, reason, day)`
  - `read_agenda(store, config, series, *, author, today)`
  - `live_path`, `last_meeting(root, series, before=)`
