# Project hub init, vision, figures — Implementation Plan (Plan 3 of 3)
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the project hub. It gains:
- vault creation (`hub init` and the `hub-init` skill), which vendors the tools and skills into each vault so collaborators need nothing else;
- a generic importer for legacy project documents;
- the vision document (`hub-vision`);
- the figures registry and gallery (`hub figures`, `hub-figures`);
- the Locator link checker (`hub check-links`);
- a fix for the one data-loss path found in Plan 2: a merge conflict dropping your side of an item file.

**Architecture:** Same split as Plan 2. Deterministic file work is a tested `hub` subcommand. Interviewing, reading documents and classifying content is skill text. **Nothing in the code or skills knows about any particular project.** Legacy documents are read by the agent in whatever format they come, and everything extracted from them goes through the normal proposal and approval flow. New writable files (`Vision.md`, `Locator.md`, `Knowledge/*.md`) are changed only through a new `note` proposal op, so they get the same approval and rollback as item changes.

**Tech Stack:** Python ≥ 3.10, PyYAML, openpyxl (unchanged runtime dependencies), git, bash, pytest; Claude Code skills.

**Spec:** `docs/superpowers/specs/2026-11-12-project-hub-design.md` (*Vault layout*, *Skills* → hub-init / hub-vision / hub-figures, *CLI* → figures / check-links / import, *Ownership*, *Testing*). Earlier plans and follow-ups: `docs/superpowers/plans/2026-11-12-project-hub-core*.md`, `docs/superpowers/plans/2026-11-12-project-hub-skills-ingest*.md`. User guide: `project-hub/USER-GUIDE.md`.

## Global Constraints
- Runtime dependencies stay PyYAML and openpyxl only. Tests: `project-hub/.venv/bin/python -m pytest -q` from `project-hub/` (create the venv with `python3 -m venv .venv && .venv/bin/pip install -e '.[test]'` if missing).
- **No project-specific content** in code, skills, templates or tests. Test fixtures use an invented project (`demo`, people `alice`/`bob`, roots `SRC`/`CODE`). The legacy importer must not parse any particular document format.
- Everything from Plans 1–2 holds:
  - item and entry formats;
  - source tokens with no whitespace;
  - one file per proposal in `Status/Pending/`;
  - sync-state key regex;
  - atomic writes (`hub/fsutil.atomic_write_text`);
  - exit codes 0 / 1 / 2 / 3;
  - every subparser and argument has help, and every subparser a description (`tests/test_cli.py` walks them);
  - every `hub …` cited in a skill exists (`tests/test_skills.py`).
- New ownership rules:
  - `Vision.md`, `Locator.md` and `Knowledge/*.md` change only through a `note` proposal op, or by a human by hand;
  - `Figures/<id>/caption.md` is written once, at registration, and never again by the tools;
  - `Figures/Gallery.md` and `Figures/<id>/figure.*` are generated.
- Path references use `$ROOT/relative/path`, where ROOT is a key of `roots`. A `$` NOT followed by `NAME/` is never a path, so LaTeX `$x$` is safe.
- Markdown for Obsidian: no blank line directly after a heading.
- Keep `project-hub/.agent_docs/` exhaustive and current, with a new file per new domain listed in `_INDEX.md`.

---
## File structure
```
project-hub/
  hub/
    merge_item.py   # MODIFY: on a real conflict write the merged item (both sides' entries/lists), still exit 1
    pending.py      # MODIFY: `note` op (Vision.md, Locator.md, Knowledge/<name>.md), rollback-safe
    paths.py        # NEW: resolve("$ROOT/rel" | vault-relative | absolute)
    figures.py      # NEW: add_figure, sync_figures, stale_figures, render_gallery, write_gallery
    links.py        # NEW: check_links (paths offline; URLs optionally online)
    init.py         # NEW: init_vault (skeleton, idempotent), update_tools (vendor package + skills), templates
    snapshot.py     # MODIFY: stale/missing figures listed under ## Flags
    cli.py          # MODIFY: `init` handled before the vault lookup
    cli_ext.py      # MODIFY: init | figures add|sync|gallery | check-links
  skills/
    hub-init/SKILL.md, hub-init/sources/legacy-import.md   # NEW
    hub-vision/SKILL.md                                    # NEW
    hub-figures/SKILL.md                                   # NEW
    hub-status/SKILL.md                                    # MODIFY: figures sync source, pivot → vision check
  tests/
    test_plan3_merge.py test_note_op.py test_figures.py test_links.py test_init.py test_cli_plan3.py
  .agent_docs/   figures.md init.md (NEW); pending-format.md git-sync.md vault-layout.md _INDEX.md (MODIFY)
  USER-GUIDE.md  # MODIFY
```
`# EDIT <path>` blocks give an exact `# OLD` snippet (it occurs once) and its `# NEW` replacement. Blocks headed `# <path>` are whole new files.

---
### Task 1: Merge conflicts keep both sides
On a real conflict, the hub-item merge driver currently leaves the file holding only side A (the remote side during `pull --rebase`). Every other change from side B is then lost unless recovered by hand (USER-GUIDE, conflict recipe).

The fix: write the merged item anyway, as entries and list unions plus every non-conflicting field merged, keeping A's value only for the conflicting keys, and still exit 1, so git marks the file conflicted. The human then only has to set the conflicting field.

**Files:** Modify `project-hub/hub/merge_item.py`, `project-hub/tests/test_merge_item.py`; create `project-hub/tests/test_plan3_merge.py`.

- [ ] **Step 1: Failing test**
```python
# project-hub/tests/test_plan3_merge.py
from datetime import date

from hub.items import Entry, parse_item, render_item
from hub.merge_item import main
from hub.store import Store

D = date(2026, 11, 12)


def test_conflict_writes_merged_item_keeping_both_sides(hub_root, tmp_path, capsys):
    q = Store(hub_root).create("question", "Base title", owner="alice", author="alice", day=D, source="m")
    base = Store(hub_root).get(q.id)
    a = parse_item(render_item(base))
    b = parse_item(render_item(base))
    a.meta["title"] = "Title A"
    b.meta["title"] = "Title B"
    a.entries.append(Entry(date(2026, 11, 13), "alice", "manual", "From A."))
    b.entries.append(Entry(date(2026, 11, 14), "bob", "manual", "From B."))
    b.meta["sources"] = b.meta["sources"] + ["circleback:x1"]
    paths = []
    for name, item in (("o", base), ("a", a), ("b", b)):
        p = tmp_path / f"{name}.md"
        p.write_text(render_item(item))
        paths.append(str(p))
    code = main(paths)
    err = capsys.readouterr().err
    merged = parse_item((tmp_path / "a.md").read_text())
    assert code == 1
    assert "title: A=Title A B=Title B" in err
    assert merged.title == "Title A"                       # conflicting key: A kept for the human to fix
    assert [e.text for e in merged.entries] == ["Raised.", "From A.", "From B."]
    assert "circleback:x1" in merged.meta["sources"]
```
In `project-hub/tests/test_merge_item.py`, the existing conflict test asserts the file is left exactly as A (`out == a_text`). Change that assertion to check that the title and description keep A's values, and that every entry of both sides is present. Read the test to see which entries it builds.

Run: `cd project-hub && .venv/bin/python -m pytest tests/test_plan3_merge.py -v`
Expected: FAIL on the entries/sources assertion, because the file is still A only.

- [ ] **Step 2: Implement**
```python
# EDIT project-hub/hub/merge_item.py
# OLD
        merged, conflicts = merge_items(o, a, b)
        if conflicts:
# NEW
        merged, conflicts = merge_items(o, a, b)
        # Always write the merge: on a real conflict it keeps A's value for each conflicting key and
        # everything else from both sides, so nothing is lost while git marks the file conflicted.
        atomic_write_text(a_path, render_item(merged))
        if conflicts:
```
Remove the now-duplicate `atomic_write_text(a_path, render_item(merged))` that followed the `if conflicts:` block, and update the module docstring: "On a real conflict the merged item (A's value for each conflicting key) is written to A and the exit code is 1".

- [ ] **Step 3: Run** `.venv/bin/python -m pytest -q`. Expected: all pass.

- [ ] **Step 4: Docs.**
  - `.agent_docs/git-sync.md`, *The hub-item merge driver* and *Resolving a SyncConflict*: on a real conflict the file holds the merged item with A's value for the conflicting key(s). The human sets those keys, then runs `git add` and `git rebase --continue`.
  - In `USER-GUIDE.md`, replace the recipe under *If `hub commit` exits with code 3*. Its first lines must read "When you pull again, the item file already holds both sides' changes; only the field(s) named in the message show the GitHub value." Then:
    ```bash
    git pull --rebase
    # edit the named field(s) in Status/Items/<ID>.md
    git add Status/Items/<ID>.md
    GIT_EDITOR=true git rebase --continue
    hub validate
    hub commit "Resolve conflict on <ID>"
    ```
    Keep the notes about several files, sync-state files and backing out. Remove the "Conflict resolution drops your side" item from Known limitations.
- [ ] **Step 5: Commit** `fix(hub): merge driver keeps both sides on a real conflict`.

---
### Task 2: `note` proposal op (Vision, Locator, Knowledge)
**Files:** Modify `project-hub/hub/pending.py`; create `project-hub/tests/test_note_op.py`; modify `.agent_docs/pending-format.md`.

**Interfaces:**
- Produces:
  - op `{op: note, path: <Vision.md | Locator.md | Knowledge/<name>.md>, text: <str>, mode: replace|append (default replace), source?: <token>}`;
  - `NOTE_PATH_RE`;
  - `note_path(root, rel) -> Path`, which raises PendingError on a bad path.
- Usable in `hub pending add --ops-file` and in batch `proposals` (`check_batch` already runs `check_ops`).

- [ ] **Step 1: Failing tests**
```python
# project-hub/tests/test_note_op.py
from datetime import date

import pytest

from hub.batch import BatchError, run_batch
from hub.pending import PendingError, add_proposal, apply_proposals, check_ops, load_proposals
from hub.store import Store

D = date(2026, 11, 13)


def test_note_replace_and_append(hub_root):
    s = Store(hub_root)
    pid = add_proposal(hub_root, "Vision draft", [{"op": "note", "path": "Vision.md", "text": "# Vision\nFirst."}],
                       proposer="alice", source="manual", reason="r", day=D)
    apply_proposals(hub_root, s, [pid], author="alice", day=D)
    assert (hub_root / "Vision.md").read_text() == "# Vision\nFirst.\n"
    pid = add_proposal(hub_root, "Add a fact", [{"op": "note", "path": "Knowledge/Code map.md",
                                                 "text": "- solver lives in src/", "mode": "append"},
                                                {"op": "note", "path": "Knowledge/Code map.md",
                                                 "text": "- tests in tests/", "mode": "append"}],
                       proposer="alice", source="manual", reason="r", day=D)
    apply_proposals(hub_root, s, [pid], author="alice", day=D)
    assert (hub_root / "Knowledge" / "Code map.md").read_text() == "- solver lives in src/\n- tests in tests/\n"


@pytest.mark.parametrize("path", ["../x.md", "Status/Items/Q-20261113-abc.md", "Knowledge/sub/x.md",
                                  "Knowledge/x.txt", "vision.md", "/etc/passwd", "Knowledge/.md"])
def test_note_paths_are_restricted(hub_root, path):
    errs = check_ops(Store(hub_root), [{"op": "note", "path": path, "text": "x"}])
    assert any("note path" in e for e in errs)


def test_note_rejects_bad_text_and_mode(hub_root):
    s = Store(hub_root)
    assert any("text" in e for e in check_ops(s, [{"op": "note", "path": "Vision.md", "text": 3}]))
    assert any("mode" in e for e in check_ops(s, [{"op": "note", "path": "Vision.md", "text": "x", "mode": "x"}]))


def test_failed_proposal_restores_the_note(hub_root, monkeypatch):
    (hub_root / "Locator.md").write_text("original\n")
    s = Store(hub_root)
    pid = add_proposal(hub_root, "Locator and a bad link",
                       [{"op": "note", "path": "Locator.md", "text": "new"},
                        {"op": "new", "type": "question", "title": "Q", "source": "manual"}],
                       proposer="alice", source="manual", reason="r", day=D)
    monkeypatch.setattr(Store, "create", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(PendingError):
        apply_proposals(hub_root, s, [pid], author="alice", day=D)
    assert (hub_root / "Locator.md").read_text() == "original\n"
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_note_is_never_a_safe_batch_op(hub_root):
    with pytest.raises(BatchError):
        run_batch(hub_root, Store(hub_root), {"safe": [{"op": "note", "path": "Vision.md", "text": "x",
                                                        "source": "manual"}]}, author="alice", day=D)
    rep = run_batch(hub_root, Store(hub_root),
                    {"proposals": [{"title": "Knowledge: setup", "reason": "from doc", "source": "doc:readme#setup",
                                    "ops": [{"op": "note", "path": "Knowledge/Setup.md", "text": "Run make."}]}]},
                    author="alice", day=D)
    assert len(rep["staged"]) == 1
```
Run: `.venv/bin/python -m pytest tests/test_note_op.py -v`. Expected: FAIL with "unknown op 'note'".

- [ ] **Step 2: Implement in `hub/pending.py`**
```python
# EDIT project-hub/hub/pending.py
# OLD
OP_REQUIRED = {"new": ("type", "title"), "entry": ("id", "text"), "set": ("id", "fields"),
               "link": ("id", "rel", "target")}
# NEW
OP_REQUIRED = {"new": ("type", "title"), "entry": ("id", "text"), "set": ("id", "fields"),
               "link": ("id", "rel", "target"), "note": ("path", "text")}
# Human-owned notes a proposal may write: the vision, the locator, one level of Knowledge/ notes.
NOTE_PATH_RE = re.compile(r"^(Vision\.md|Locator\.md|Knowledge/[A-Za-z0-9][A-Za-z0-9 _.,()-]*\.md)$")
NOTE_MODES = ("replace", "append")


def note_path(root, rel) -> Path:
    """Vault path of a note op target; PendingError unless `rel` is an allowed note path."""
    if not isinstance(rel, str) or not NOTE_PATH_RE.match(rel) or ".." in rel:
        raise PendingError(f"note path must be Vision.md, Locator.md or Knowledge/<name>.md, got {rel!r}")
    return Path(root) / rel
```
In `check_ops`, add a branch for `note` before the final `else:` that handles id-based ops:
```python
# EDIT project-hub/hub/pending.py
# OLD
        else:
            target_ok(f"{where} id", op["id"])
# NEW
        elif kind == "note":
            if not isinstance(op["path"], str) or not NOTE_PATH_RE.match(op["path"]) or ".." in op["path"]:
                errs.append(f"{where}: note path must be Vision.md, Locator.md or Knowledge/<name>.md, "
                            f"got {op['path']!r}")
            if not isinstance(op["text"], str):
                errs.append(f"{where}: text must be a string")
            if op.get("mode", "replace") not in NOTE_MODES:
                errs.append(f"{where}: mode must be one of {', '.join(NOTE_MODES)}")
        else:
            target_ok(f"{where} id", op["id"])
```
Give `_Undo` a path-level hook:
```python
# EDIT project-hub/hub/pending.py
# OLD
    def created(self, item_id) -> None:
# NEW
    def touch_path(self, path: Path) -> None:
        if path not in self.before:
            self.before[path] = path.read_bytes() if path.is_file() else None

    def created(self, item_id) -> None:
```
In `_run_ops`, add the op:
```python
# EDIT project-hub/hub/pending.py
# OLD
            else:
                raise PendingError(f"unknown op '{kind}'")
# NEW
            elif kind == "note":
                path = note_path(store.root, _req(op, "path"))
                text = _req(op, "text")
                if not isinstance(text, str):
                    raise PendingError("note text must be a string")
                undo.touch_path(path)
                if op.get("mode", "replace") == "append" and path.is_file():
                    old = path.read_text(encoding="utf-8")
                    text = old + ("" if not old or old.endswith("\n") else "\n") + text
                atomic_write_text(path, text if text.endswith("\n") else text + "\n")
            else:
                raise PendingError(f"unknown op '{kind}'")
```
`atomic_write_text` must create missing parent directories (`Knowledge/`). Check `hub/fsutil.py`, and add `Path(path).parent.mkdir(parents=True, exist_ok=True)` if it does not already do so.

- [ ] **Step 3: Run** the new tests and the full suite. Expected: all pass.
- [ ] **Step 4: Docs.** In `.agent_docs/pending-format.md`, document the `note` op: its keys, the allowed paths, `mode`, rollback, and that it is never a safe batch op.
- [ ] **Step 5: Commit** `feat(hub): note proposal op for Vision, Locator and Knowledge notes`.

---
### Task 3: Paths and the figures registry
**Files:** Create `project-hub/hub/paths.py`, `project-hub/hub/figures.py`, `project-hub/tests/test_figures.py`; modify `project-hub/hub/snapshot.py`.

**Interfaces:**
- `paths.PathError(ValueError)`;
- `paths.resolve(config, spec) -> Path`;
- `paths.ROOT_REF_RE`, which matches `$NAME/rest` only;
- `figures.FigureError(ValueError)`;
- `figures.add_figure(config, fig_id, source, *, title="", script="", data=(), licenses="", caption="") -> Path`;
- `figures.sync_figures(config, *, today) -> list[dict]`, one row per figure: `{id, copied, stale, problem}`;
- `figures.stale_figures(root) -> list[str]`, flag texts;
- `figures.render_gallery(root) -> str`, `figures.write_gallery(root) -> Path`.

Figure metadata `Figures/<id>/meta.yaml` has the keys:
- `id`, `title`;
- `source`: path spec of the figure file the project's scripts produce;
- `script`: spec or "";
- `data`: list of specs, files or directories;
- `status`: `draft|final`;
- `licenses`: one sentence on what claim the figure supports;
- `figure_file`: e.g. `figure.png`;
- `figure_hash`, `data_hash`: sha256 prefixes;
- `synced`: date;
- `stale`: bool;
- `problem`: str or null.

- [ ] **Step 1: Failing tests**
```python
# project-hub/tests/test_figures.py
import textwrap
from datetime import date

import pytest

from hub.config import load_config
from hub.figures import FigureError, add_figure, render_gallery, stale_figures, sync_figures, write_gallery
from hub.paths import PathError, resolve
from hub.snapshot import render_snapshot
from hub.store import Store

D = date(2026, 11, 13)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    src = tmp_path / "src"
    (src / "data").mkdir(parents=True)
    (src / "fig.png").write_bytes(b"png-1")
    (src / "data" / "a.csv").write_text("1\n")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text(textwrap.dedent("""\
        name: demo
        people:
          alice: {name: Alice}
        series: {}
        roots:
          SRC: /nowhere
        """))
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  SRC: {src}\n")
    return load_config(vault), vault, src


def test_resolve_forms(env):
    cfg, vault, src = env
    assert resolve(cfg, "$SRC/fig.png") == src / "fig.png"
    assert resolve(cfg, "$SRC") == src
    assert resolve(cfg, "Figures/x") == vault / "Figures" / "x"
    assert resolve(cfg, str(src / "fig.png")) == src / "fig.png"
    with pytest.raises(PathError, match="unknown root"):
        resolve(cfg, "$NOPE/x")


def test_add_sync_stale_gallery(env):
    cfg, vault, src = env
    d = add_figure(cfg, "energy-vs-u", "$SRC/fig.png", title="Energy vs U", data=["$SRC/data"],
                   licenses="Energy error falls with U.", caption="Energy error against U.")
    assert (d / "figure.png").read_bytes() == b"png-1"
    assert (d / "caption.md").read_text() == "Energy error against U.\n"
    assert sync_figures(cfg, today=D) == [{"id": "energy-vs-u", "copied": False, "stale": False, "problem": None}]
    (d / "caption.md").write_text("Edited by a human.\n")
    (src / "data" / "a.csv").write_text("2\n")                 # data changed, figure not regenerated
    assert sync_figures(cfg, today=D)[0]["stale"] is True
    assert stale_figures(vault) == ["figure energy-vs-u: its data changed after the figure was made — "
                                    "regenerate it, then run `hub figures sync`"]
    (src / "fig.png").write_bytes(b"png-2")                   # figure regenerated
    row = sync_figures(cfg, today=D)[0]
    assert row["copied"] is True and row["stale"] is False
    assert (d / "figure.png").read_bytes() == b"png-2"
    assert (d / "caption.md").read_text() == "Edited by a human.\n"
    assert stale_figures(vault) == []
    text = render_gallery(vault)
    assert "![[Figures/energy-vs-u/figure.png]]" in text and "![[Figures/energy-vs-u/caption]]" in text
    assert "Energy error falls with U." in text
    assert write_gallery(vault) == vault / "Figures" / "Gallery.md"


def test_missing_source_is_a_problem_and_a_flag(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    (src / "fig.png").unlink()
    row = sync_figures(cfg, today=D)[0]
    assert row["problem"].startswith("source missing")
    assert any("f1" in f for f in stale_figures(vault))
    assert "figure f1" in render_snapshot(Store(vault), cfg, D)


@pytest.mark.parametrize("fig_id,source,msg", [
    ("Bad Id", "$SRC/fig.png", "figure id"),
    ("f2", "$SRC/none.png", "does not exist"),
    ("f3", "$SRC/data", "not a file"),
])
def test_add_errors(env, fig_id, source, msg):
    cfg, vault, src = env
    with pytest.raises(FigureError, match=msg):
        add_figure(cfg, fig_id, source)


def test_duplicate_and_bad_data(env):
    cfg, vault, src = env
    add_figure(cfg, "f1", "$SRC/fig.png")
    with pytest.raises(FigureError, match="already registered"):
        add_figure(cfg, "f1", "$SRC/fig.png")
    with pytest.raises(FigureError, match="does not exist"):
        add_figure(cfg, "f2", "$SRC/fig.png", data=["$SRC/missing.csv"])
```
Run: `.venv/bin/python -m pytest tests/test_figures.py -v`. Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 2: `hub/paths.py`**
```python
# project-hub/hub/paths.py
"""Resolve the path forms used by figures, the Locator and the CLI.

  $ROOT/rel/path   a path under a named root (`roots` in project.yaml, overridden in hub.local.yaml)
  rel/path         relative to the vault root
  /abs/path, ~/x   absolute
A `$` that is not followed by NAME/ is not a path (so LaTeX like $x$ is never taken for one).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)/([^\s)\]>`'\"|*]*)")
_FULL_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)(/.*)?")


class PathError(ValueError):
    """A path refers to an unknown root."""


def resolve(config, spec) -> Path:
    spec = str(spec).strip()
    m = _FULL_RE.fullmatch(spec)
    if m:
        name = m[1]
        if name not in config.roots:
            raise PathError(f"unknown root ${name} (roots: {', '.join(sorted(config.roots)) or 'none'})")
        base = Path(str(config.roots[name])).expanduser()
        return base / m[2].lstrip("/") if m[2] and m[2] != "/" else base
    p = Path(spec).expanduser()
    return p if p.is_absolute() else Path(config.root) / p
```

- [ ] **Step 3: `hub/figures.py`**
```python
# project-hub/hub/figures.py
"""Figures registry: Figures/<id>/{figure.<ext>, caption.md, meta.yaml} and the generated Figures/Gallery.md.

A figure is produced by the project's own scripts somewhere under a root; the vault keeps a copy so it can
be read in Obsidian. `sync_figures` re-copies a figure whose source changed and marks it stale when its
data changed after it was made. caption.md is human-owned: written once at registration, never again.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from datetime import date
from pathlib import Path

import yaml

from .fsutil import atomic_write_text
from .paths import resolve

FIG_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FIG_STATUSES = ("draft", "final")
CAPTION_PLACEHOLDER = "<!-- caption: what the figure shows, findings first; edit freely, the tools never overwrite it -->"


class FigureError(ValueError):
    """A figure cannot be registered or synced as asked."""


def figures_dir(root) -> Path:
    return Path(root) / "Figures"


def _fig_dir(root, fig_id) -> Path:
    return figures_dir(root) / fig_id


def _sha_bytes(h, path: Path) -> None:
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    _sha_bytes(h, path)
    return h.hexdigest()[:16]


def _hash_data(config, specs) -> str | None:
    if not specs:
        return None
    h = hashlib.sha256()
    for spec in sorted(specs):
        path = resolve(config, spec)
        if not path.exists():
            raise FigureError(f"data path {spec} ({path}) does not exist")
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for f in files:
            h.update(str(f.relative_to(path) if path.is_dir() else f.name).encode())
            _sha_bytes(h, f)
    return h.hexdigest()[:16]


def load_meta(root, fig_id) -> dict:
    p = _fig_dir(root, fig_id) / "meta.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None
    if not isinstance(data, dict):
        raise FigureError(f"Figures/{fig_id}/meta.yaml is missing or not a mapping")
    return data


def _save_meta(root, fig_id, meta: dict) -> None:
    atomic_write_text(_fig_dir(root, fig_id) / "meta.yaml", yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))


def figure_ids(root) -> list[str]:
    d = figures_dir(root)
    return sorted(p.parent.name for p in d.glob("*/meta.yaml")) if d.is_dir() else []


def add_figure(config, fig_id, source, *, title="", script="", data=(), licenses="", caption="") -> Path:
    root = Path(config.root)
    if not isinstance(fig_id, str) or not FIG_ID_RE.match(fig_id):
        raise FigureError(f"figure id must be lowercase letters, digits and '-', got {fig_id!r}")
    d = _fig_dir(root, fig_id)
    if (d / "meta.yaml").exists():
        raise FigureError(f"figure {fig_id} is already registered (Figures/{fig_id})")
    src = resolve(config, source)
    if not src.exists():
        raise FigureError(f"source {source} ({src}) does not exist")
    if not src.is_file():
        raise FigureError(f"source {source} ({src}) is not a file")
    data = list(data or [])
    data_hash = _hash_data(config, data)
    d.mkdir(parents=True, exist_ok=True)
    fig_file = f"figure{src.suffix.lower()}"
    shutil.copyfile(src, d / fig_file)
    atomic_write_text(d / "caption.md", (caption.strip() or CAPTION_PLACEHOLDER) + "\n")
    _save_meta(root, fig_id, {
        "id": fig_id, "title": title, "source": str(source), "script": script, "data": data,
        "status": "draft", "licenses": licenses, "figure_file": fig_file, "figure_hash": _hash_file(src),
        "data_hash": data_hash, "synced": None, "stale": False, "problem": None})
    return d


def sync_figures(config, *, today: date) -> list[dict]:
    root = Path(config.root)
    rows = []
    for fig_id in figure_ids(root):
        meta = load_meta(root, fig_id)
        row = {"id": fig_id, "copied": False, "stale": bool(meta.get("stale")), "problem": None}
        try:
            src = resolve(config, meta.get("source", ""))
            if not src.is_file():
                raise FigureError(f"source missing: {meta.get('source')} ({src})")
            data_hash = _hash_data(config, meta.get("data") or [])
            src_hash = _hash_file(src)
            if src_hash != meta.get("figure_hash"):
                fig_file = f"figure{src.suffix.lower()}"
                old = meta.get("figure_file")
                if old and old != fig_file:
                    (_fig_dir(root, fig_id) / old).unlink(missing_ok=True)
                shutil.copyfile(src, _fig_dir(root, fig_id) / fig_file)
                meta.update(figure_file=fig_file, figure_hash=src_hash, data_hash=data_hash, synced=today)
                row["copied"], row["stale"] = True, False
            elif data_hash != meta.get("data_hash"):
                row["stale"] = True
        except (FigureError, ValueError, OSError) as exc:
            row["problem"] = str(exc)
        meta["stale"], meta["problem"] = row["stale"], row["problem"]
        _save_meta(root, fig_id, meta)
        rows.append(row)
    return rows


def stale_figures(root) -> list[str]:
    out = []
    for fig_id in figure_ids(root):
        try:
            meta = load_meta(root, fig_id)
        except FigureError as exc:
            out.append(f"figure {fig_id}: {exc}")
            continue
        if meta.get("problem"):
            out.append(f"figure {fig_id}: {meta['problem']}")
        elif meta.get("stale"):
            out.append(f"figure {fig_id}: its data changed after the figure was made — regenerate it, "
                       f"then run `hub figures sync`")
    return out


def render_gallery(root) -> str:
    out = ["# Figures", "*Generated by `hub figures gallery`; captions are edited in each figure's caption.md.*"]
    for fig_id in figure_ids(root):
        meta = load_meta(root, fig_id)
        title = meta.get("title") or fig_id
        out += ["", f"## {fig_id} — {title} ({meta.get('status') or 'draft'})",
                f"![[Figures/{fig_id}/{meta.get('figure_file')}]]", f"![[Figures/{fig_id}/caption]]"]
        if meta.get("licenses"):
            out.append(f"*Supports:* {meta['licenses']}")
        out.append(f"*Source:* `{meta.get('source')}`" + (f" · *script:* `{meta['script']}`" if meta.get("script") else ""))
        if meta.get("problem"):
            out.append(f"**Problem:** {meta['problem']}")
        elif meta.get("stale"):
            out.append("**Stale:** its data changed after the figure was made.")
    if len(out) == 2:
        out += ["", "No figures registered yet (`hub figures add`)."]
    return "\n".join(out) + "\n"


def write_gallery(root) -> Path:
    path = figures_dir(root) / "Gallery.md"
    atomic_write_text(path, render_gallery(root))
    return path
```

- [ ] **Step 4: Snapshot flags.** In `hub/snapshot.py`, import `stale_figures` and add figure flags to the Flags section:
```python
# EDIT project-hub/hub/snapshot.py
# OLD
from .config import HubConfig
# NEW
from .config import HubConfig
from .figures import stale_figures
```
```python
# EDIT project-hub/hub/snapshot.py
# OLD
              for p, err in store.load_errors]
# NEW
              for p, err in store.load_errors]
    flags += [f"- {f}" for f in stale_figures(config.root)]
```
- [ ] **Step 5: Run** `.venv/bin/python -m pytest tests/test_figures.py -v`, then the full suite. Expected: all pass.
- [ ] **Step 6: Commit** `feat(hub): path resolution and figures registry with staleness and gallery`.

---
### Task 4: Link checker
**Files:** Create `project-hub/hub/links.py`, `project-hub/tests/test_links.py`.

**Interface:** `check_links(config, *, online=False, timeout=5.0, opener=None) -> list[dict]`, with rows `{file, ref, kind: "path"|"url", ok: True|False|None, problem: str|None}`. It scans `Locator.md` and every `Figures/*/meta.yaml` (`source`, `script`, `data`). A URL's `ok` is None (not checked) unless `online`.

- [ ] **Step 1: Failing tests**
```python
# project-hub/tests/test_links.py
import textwrap

from hub.config import load_config
from hub.links import check_links


def _vault(tmp_path):
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text("name: demo\npeople: {alice: {name: Alice}}\nseries: {}\nroots: {CODE: /x}\n")
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {code}\n")
    (vault / "Locator.md").write_text(textwrap.dedent("""\
        # Locator
        - Solver: `$CODE/src`
        - Old script: `$CODE/old.py`
        - Elsewhere: $NOPE/thing
        - Energy $E_0$ and $x$ are maths, not paths.
        - Repo: https://github.com/example/demo
        """))
    return load_config(vault)


def test_offline(tmp_path):
    rows = check_links(_vault(tmp_path))
    by_ref = {r["ref"]: r for r in rows}
    assert set(by_ref) == {"$CODE/src", "$CODE/old.py", "$NOPE/thing", "https://github.com/example/demo"}
    assert by_ref["$CODE/src"]["ok"] is True
    assert by_ref["$CODE/old.py"]["ok"] is False and "does not exist" in by_ref["$CODE/old.py"]["problem"]
    assert by_ref["$NOPE/thing"]["ok"] is False and "unknown root" in by_ref["$NOPE/thing"]["problem"]
    assert by_ref["https://github.com/example/demo"]["ok"] is None


def test_online_uses_opener(tmp_path):
    calls = []

    class Resp:
        status = 404

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        calls.append(req.full_url)
        return Resp()

    rows = check_links(_vault(tmp_path), online=True, opener=opener)
    url = next(r for r in rows if r["kind"] == "url")
    assert calls == ["https://github.com/example/demo"]
    assert url["ok"] is False and "404" in url["problem"]


def test_figure_meta_paths(tmp_path):
    cfg = _vault(tmp_path)
    d = cfg.root / "Figures" / "f1"
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text("id: f1\nsource: $CODE/fig.png\nscript: ''\ndata: [$CODE/src]\n")
    refs = {r["ref"]: r for r in check_links(cfg) if r["file"] == "Figures/f1/meta.yaml"}
    assert refs["$CODE/fig.png"]["ok"] is False and refs["$CODE/src"]["ok"] is True
```
Run: expected FAIL with ModuleNotFoundError.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/links.py
"""`hub check-links`: every $ROOT/path and URL in Locator.md and figure meta.yaml files.

Paths are checked on this machine (roots come from hub.local.yaml). URLs are only listed unless
online=True, which sends a HEAD request (falling back to GET) with a short timeout.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from .paths import ROOT_REF_RE, PathError, resolve

URL_RE = re.compile(r"https?://[^\s)\]>`'\"|]+")


def _refs_in_text(text: str):
    for m in ROOT_REF_RE.finditer(text):
        yield "path", f"${m[1]}/{m[2]}".rstrip(".,;:")
    for m in URL_RE.finditer(text):
        yield "url", m[0].rstrip(".,;:")


def _meta_refs(path: Path):
    try:
        meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return
    specs = [meta.get("source"), meta.get("script")] + list(meta.get("data") or [])
    for spec in specs:
        if isinstance(spec, str) and spec.strip():
            yield ("url" if URL_RE.fullmatch(spec.strip()) else "path"), spec.strip()


def _check_url(url, timeout, opener):
    opener = opener or urllib.request.urlopen
    for method in ("HEAD", "GET"):
        try:
            with opener(urllib.request.Request(url, method=method), timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
            if status < 400:
                return True, None
            if method == "GET" or status not in (403, 405):
                return False, f"HTTP {status}"
        except urllib.error.HTTPError as exc:
            if method == "GET" or exc.code not in (403, 405):
                return False, f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, f"unreachable: {exc}"
    return False, "unreachable"


def check_links(config, *, online=False, timeout=5.0, opener=None) -> list[dict]:
    root = Path(config.root)
    sources = []
    if (root / "Locator.md").is_file():
        sources.append(("Locator.md", list(_refs_in_text((root / "Locator.md").read_text(encoding="utf-8")))))
    for meta in sorted((root / "Figures").glob("*/meta.yaml")) if (root / "Figures").is_dir() else []:
        sources.append((str(meta.relative_to(root)), list(_meta_refs(meta))))
    rows = []
    for file, refs in sources:
        seen = set()
        for kind, ref in refs:
            if (kind, ref) in seen:
                continue
            seen.add((kind, ref))
            row = {"file": file, "ref": ref, "kind": kind, "ok": None, "problem": None}
            if kind == "path":
                try:
                    path = resolve(config, ref)
                    row["ok"] = path.exists()
                    if not row["ok"]:
                        row["problem"] = f"does not exist on this machine: {path}"
                except PathError as exc:
                    row["ok"], row["problem"] = False, str(exc)
            elif online:
                row["ok"], row["problem"] = _check_url(ref, timeout, opener)
            rows.append(row)
    return rows
```
- [ ] **Step 3: Run** the tests and the suite. Expected: all pass.
- [ ] **Step 4: Commit** `feat(hub): link checker for Locator and figure paths`.

---
### Task 5: `init_vault` — skeleton and vendored tools
A vault carries its own copy of the tools (`tools/project-hub/`) and skills (`.claude/skills/hub-*`). A collaborator who clones the vault then has everything: `bash tools/setup.sh` makes a venv, puts `hub` on PATH and registers the merge drivers. `init_vault` is idempotent: it creates what is missing, never overwrites human files, and always refreshes the vendored tools and skills from the running package.

**Files:** Create `project-hub/hub/init.py`, `project-hub/tests/test_init.py`.

**Interfaces:**
- `init_vault(dest, *, name) -> list[str]`: vault-relative paths created or refreshed;
- `update_tools(dest) -> list[str]`;
- `PACKAGE_DIR`, `VENDOR_ITEMS`, `TEMPLATES` (dict of vault-relative path → text).

- [ ] **Step 1: Failing tests**
```python
# project-hub/tests/test_init.py
import os
import subprocess

from hub.config import load_config
from hub.init import TEMPLATES, init_vault
from hub.store import Store
from hub.validate import validate


def test_init_creates_skeleton_and_vendors(tmp_path):
    v = tmp_path / "vault"
    created = init_vault(v, name="demo")
    for rel in ["project.yaml", "README.md", "Vision.md", "Locator.md", "Knowledge/README.md", "Figures/Gallery.md",
                "Status/Items.base", ".gitignore", ".gitattributes", "tools/setup.sh",
                "tools/project-hub/pyproject.toml", "tools/project-hub/hub/cli.py",
                "tools/project-hub/USER-GUIDE.md", ".claude/skills/hub-status/SKILL.md",
                ".claude/skills/hub-agenda/SKILL.md"]:
        assert (v / rel).exists(), rel
    assert "project.yaml" in created
    assert not (v / "tools" / "project-hub" / "tests").exists()
    assert not list((v / "tools").rglob("__pycache__"))
    assert os.access(v / "tools" / "setup.sh", os.X_OK)
    assert subprocess.run(["bash", "-n", str(v / "tools" / "setup.sh")]).returncode == 0
    cfg = load_config(v)
    assert cfg.name == "demo" and cfg.people == {} and cfg.series == {}
    assert validate(Store(v), cfg) == []
    assert "tools/.venv/" in (v / ".gitignore").read_text() or ".venv/" in (v / ".gitignore").read_text()


def test_init_is_idempotent_and_keeps_human_files(tmp_path):
    v = tmp_path / "vault"
    init_vault(v, name="demo")
    (v / "Vision.md").write_text("# Our vision\nEdited.\n")
    (v / "project.yaml").write_text((v / "project.yaml").read_text().replace("name: demo", "name: renamed"))
    (v / "tools" / "project-hub" / "hub" / "cli.py").write_text("broken\n")
    created = init_vault(v, name="demo")
    assert (v / "Vision.md").read_text() == "# Our vision\nEdited.\n"
    assert load_config(v).name == "renamed"
    assert "broken" not in (v / "tools" / "project-hub" / "hub" / "cli.py").read_text()
    assert "Vision.md" not in created and "project.yaml" not in created


def test_init_registers_merge_drivers_in_a_git_repo(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=v, check=True)
    init_vault(v, name="demo")
    out = subprocess.run(["git", "config", "merge.hub-item.driver"], cwd=v, capture_output=True, text=True).stdout
    assert "hub.merge_item" in out


def test_templates_are_generic():
    text = "\n".join(TEMPLATES.values()).lower()
    for word in ("ucc", "hubbard", "fermion", "hva", "clickup list 9"):
        assert word not in text
```
Run: expected FAIL with ModuleNotFoundError.

- [ ] **Step 2: Implement**
```python
# project-hub/hub/init.py
"""Create or refresh a project hub vault: the skeleton files and a vendored copy of the tools and skills.

init_vault never overwrites a file that exists (project.yaml, Vision.md, Locator.md, README.md, …), so it
is safe to re-run; it always refreshes tools/project-hub/ and .claude/skills/hub-* from the package that
is running, which is how a vault's tools are updated.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .fsutil import atomic_write_text
from .gitsync import GITATTRIBUTES_LINES, GITIGNORE_LINES, _ensure_lines, ensure_repo_files

PACKAGE_DIR = Path(__file__).resolve().parents[1]
VENDOR_ITEMS = ("pyproject.toml", "README.md", "USER-GUIDE.md", "hub", "skills", ".agent_docs")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".venv", ".pytest_cache")

PROJECT_TEMPLATE = """\
name: {name}
# Every key is explained in tools/project-hub/USER-GUIDE.md (Step 3) and tools/project-hub/.agent_docs/vault-layout.md.
people: {{}}
#  alice: {{name: Alice Example, clickup: "1234567", email: alice@example.edu}}
series: {{}}
#  group: {{attendees: [alice], calendar_match: "Group meeting"}}
roots: {{}}
#  CODE: /path/to/code/repository
sources: {{}}
#  clickup: {{list_id: "123"}}
#  circleback: {{folder: "Project name", terms: []}}
#  repos: [{{root: CODE, branch: main}}]
#  overleaf: {{root: PAPER, branch: master}}
#  calendar: {{}}
"""

README_TEMPLATE = """\
# Project hub
This folder is the project's shared hub: an Obsidian vault and a git repository. Open questions, decisions and
actions live in `Status/`; the overview is `Status/Snapshot.md`; meetings run from `Agenda/<series>.xlsx`.
## Set up on your machine
1. Ask the owner to add you to `people` in `project.yaml`.
2. Clone this repository, then from its root run `bash tools/setup.sh` (needs Python 3.10+; set `PYTHON=python3.12`
   if `python3` is older). It installs the `hub` command and registers the git merge drivers.
3. Create `hub.local.yaml` (never committed) with `me: <your people key>` and your local `roots`.
4. Start Claude Code in this folder; the hub skills are in `.claude/skills/`.
Full guide: `tools/project-hub/USER-GUIDE.md`.
"""

VISION_TEMPLATE = """\
# Vision
<!-- Written with the hub-vision skill; changed only through approved proposals. -->
## Vision
## The problem, as the community poses it
## Our approach
## How we answer the problem
## Our unfair advantage
## The difference we will make
## Risks and how we will know early
## Revisions
"""

LOCATOR_TEMPLATE = """\
# Locator
Where everything is. Paths use `$ROOT/relative/path` (ROOT from `roots` in project.yaml); check with `hub check-links`.
## Code and data
## Documents
## External links
"""

KNOWLEDGE_README = """\
# Knowledge
Stable project knowledge, one note per topic: what the code does, conventions, what each result supports,
claims that were tried and rejected. Notes change only through approved proposals (`note` op).
"""

ITEMS_BASE = """\
filters:
  and:
    - file.inFolder("Status/Items")
views:
  - type: table
    name: Open questions
    filters:
      and:
        - type == "question"
        - status == "open"
    order: [file.name, title, owner, priority, raised]
  - type: table
    name: Open actions
    filters:
      and:
        - type == "action"
        - status == "open"
    order: [file.name, title, owner, due]
  - type: table
    name: Decisions
    filters:
      and:
        - type == "decision"
    order: [file.name, title, status, meeting, owner]
"""

GALLERY_TEMPLATE = "# Figures\nNo figures registered yet (`hub figures add`).\n"

SETUP_SH = r"""#!/usr/bin/env bash
# Set up this vault's hub tools on this machine: a venv, `hub` on PATH, and the git merge drivers.
# Usage: bash tools/setup.sh        (PYTHON=python3.12 bash tools/setup.sh if python3 is older than 3.10)
set -euo pipefail
VAULT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is needed: re-run as PYTHON=python3.12 bash tools/setup.sh")'
"$PY" -m venv "$VAULT/tools/.venv"
BIN="$VAULT/tools/.venv/bin"
[ -d "$BIN" ] || BIN="$VAULT/tools/.venv/Scripts"
"$BIN/python" -m pip install -q -e "$VAULT/tools/project-hub"
mkdir -p "$HOME/.local/bin"
ln -sf "$BIN/hub" "$HOME/.local/bin/hub"
(cd "$VAULT" && "$BIN/python" -c 'from hub.gitsync import ensure_repo_files; ensure_repo_files(".")')
echo "hub installed: $HOME/.local/bin/hub -> $BIN/hub"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "Add ~/.local/bin to your PATH (see tools/project-hub/USER-GUIDE.md, Step 1)." ;; esac
"""

TEMPLATES = {
    "README.md": README_TEMPLATE, "Vision.md": VISION_TEMPLATE, "Locator.md": LOCATOR_TEMPLATE,
    "Knowledge/README.md": KNOWLEDGE_README, "Status/Items.base": ITEMS_BASE, "Figures/Gallery.md": GALLERY_TEMPLATE,
}


def _is_git_repo(path: Path) -> bool:
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


def update_tools(dest) -> list[str]:
    dest = Path(dest)
    done = []
    vendored = dest / "tools" / "project-hub"
    if PACKAGE_DIR.resolve() != vendored.resolve():
        vendored.mkdir(parents=True, exist_ok=True)
        for name in VENDOR_ITEMS:
            src = PACKAGE_DIR / name
            if not src.exists():
                continue
            target = vendored / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            if src.is_dir():
                shutil.copytree(src, target, ignore=_IGNORE)
            else:
                shutil.copyfile(src, target)
            done.append(f"tools/project-hub/{name}")
    skills_src = vendored / "skills" if (vendored / "skills").is_dir() else PACKAGE_DIR / "skills"
    for skill in sorted(p for p in skills_src.iterdir() if p.is_dir()):
        target = dest / ".claude" / "skills" / skill.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(skill, target, ignore=_IGNORE)
        done.append(f".claude/skills/{skill.name}")
    setup = dest / "tools" / "setup.sh"
    atomic_write_text(setup, SETUP_SH)
    os.chmod(setup, 0o755)
    done.append("tools/setup.sh")
    return done


def init_vault(dest, *, name: str) -> list[str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    created = []
    files = {"project.yaml": PROJECT_TEMPLATE.format(name=name), **TEMPLATES}
    for rel, text in files.items():
        path = dest / rel
        if not path.exists():
            atomic_write_text(path, text)
            created.append(rel)
    if _is_git_repo(dest):
        ensure_repo_files(dest)
    else:
        _ensure_lines(dest / ".gitattributes", GITATTRIBUTES_LINES)
        _ensure_lines(dest / ".gitignore", GITIGNORE_LINES)
    return created + update_tools(dest)
```
`atomic_write_text` must create parent directories (Task 2 Step 2 ensures this). `GITIGNORE_LINES` already contains `.venv/`, which ignores `tools/.venv/`.

- [ ] **Step 3: Run** the tests and the full suite. Expected: all pass.
- [ ] **Step 4: Commit** `feat(hub): init_vault — vault skeleton, vendored tools and skills, setup.sh`.

---
### Task 6: CLI wiring and agent docs
**Files:** Modify `project-hub/hub/cli.py` and `project-hub/hub/cli_ext.py`; create `project-hub/tests/test_cli_plan3.py`, `project-hub/.agent_docs/figures.md` and `project-hub/.agent_docs/init.md`; modify `.agent_docs/_INDEX.md` and `.agent_docs/vault-layout.md`.

**Commands:**
- `hub init DIR --name NAME`: works outside any vault. Prints each created or refreshed path, then `next: cd DIR && bash tools/setup.sh`.
- `hub figures add ID SOURCE [--title T] [--script S] [--data D]… [--licenses L] [--caption C]`.
- `hub figures sync`: prints one line per figure (`<id>  copied|unchanged|STALE|PROBLEM: …`), writes the gallery, exits 1 if any problem.
- `hub figures gallery`.
- `hub check-links [--online] [--json]`: exits 1 if any `ok is False`.

- [ ] **Step 1: Failing tests**
```python
# project-hub/tests/test_cli_plan3.py
import json
import textwrap

import pytest

from hub.cli import main


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_init_runs_outside_a_vault(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(capsys, "init", str(tmp_path / "v"), "--name", "demo")
    assert code == 0 and "project.yaml" in out and "bash tools/setup.sh" in out
    assert (tmp_path / "v" / ".claude" / "skills" / "hub-status" / "SKILL.md").is_file()


@pytest.fixture
def vault(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.png").write_bytes(b"x")
    v = tmp_path / "v"
    v.mkdir()
    (v / "project.yaml").write_text("name: demo\npeople: {alice: {name: Alice}}\nseries: {}\nroots: {SRC: /x}\n")
    (v / "hub.local.yaml").write_text(f"me: alice\nroots:\n  SRC: {src}\n")
    monkeypatch.chdir(v)
    monkeypatch.delenv("HUB_ME", raising=False)
    return v, src


def test_figures_cli(vault, capsys):
    v, src = vault
    code, out, _ = run(capsys, "figures", "add", "f1", "$SRC/f.png", "--title", "T", "--licenses", "L")
    assert code == 0 and (v / "Figures" / "f1" / "figure.png").is_file()
    code, out, _ = run(capsys, "figures", "sync")
    assert code == 0 and "f1  unchanged" in out and (v / "Figures" / "Gallery.md").is_file()
    (src / "f.png").unlink()
    code, out, _ = run(capsys, "figures", "sync")
    assert code == 1 and "f1  PROBLEM" in out
    code, _, err = run(capsys, "figures", "add", "Bad Id", "$SRC/f.png")
    assert code == 1 and "figure id" in err


def test_check_links_cli(vault, capsys):
    v, src = vault
    (v / "Locator.md").write_text("- `$SRC/f.png`\n- `$SRC/gone.txt`\n")
    code, out, _ = run(capsys, "check-links", "--json")
    rows = {r["ref"]: r for r in json.loads(out)}
    assert code == 1 and rows["$SRC/f.png"]["ok"] is True and rows["$SRC/gone.txt"]["ok"] is False
```
Run: expected FAIL (argparse exits 2 for the unknown commands).

- [ ] **Step 2: Implement.** In `cli_ext.py`:
  - add the parsers with full help and descriptions, as for the existing commands;
  - add the dispatch branches;
  - add `run_init(args) -> int` (init_vault, print, return 0).

  Errors (`FigureError`, `PathError`) are ValueErrors, so the existing handler prints `error:` with exit 1. In `cli.py` `main`, handle `init` before the vault lookup:
```python
# EDIT project-hub/hub/cli.py
# OLD
        root = args.root or find_root(Path.cwd())
# NEW
        if args.cmd == "init":
            from .cli_ext import run_init
            return run_init(args)
        root = args.root or find_root(Path.cwd())
```
  Add the new commands to the `cli.py` module docstring overview.
- [ ] **Step 3: Run** the tests plus `tests/test_cli.py`, which checks that every argument is documented, and then the full suite.
- [ ] **Step 4: Docs.**
  - `figures.md`: meta keys, the sync rules (copied/stale/problem), the gallery format, caption ownership, the CLI.
  - `init.md`: what `init_vault` creates and never overwrites, the vendoring layout, `setup.sh` steps, how to update a vault's tools (`<newer project-hub>/.venv/bin/hub init <vault> --name <name>`, then commit), check-links.
  - `vault-layout.md`: `Vision.md`, `Locator.md`, `Knowledge/`, `Figures/`, `tools/`, `.claude/skills/`, `Status/Items.base`.
  - `_INDEX.md`: list both new files.
- [ ] **Step 5: Commit** `feat(hub): CLI for init, figures and check-links`.

---
### Task 7: Skills — hub-init (with a generic legacy importer), hub-vision, hub-figures; hub-status additions
**Files:**
- Create `project-hub/skills/hub-init/SKILL.md`, `project-hub/skills/hub-init/sources/legacy-import.md`, `project-hub/skills/hub-vision/SKILL.md`, `project-hub/skills/hub-figures/SKILL.md`.
- Modify `project-hub/skills/hub-status/SKILL.md`.

**Requirements:**
- Every `hub …` cited exists; `tests/test_skills.py` enforces this.
- Frontmatter `name` equals the folder name, and `description` starts with "Use when".
- **No project-specific content.**
- Follow the conventions of the existing skills: scratch files in `mktemp -d` outside the vault, a dry-run allow-list, commit after each logical step, `hub validate` before committing, privacy rules, and asking the user before outward-facing actions.

- [ ] **Step 1: `skills/hub-init/SKILL.md`** must cover, in order:
  0. **Before starting.** Ask for the project name, where the vault folder should live, and whether a remote repository exists. Creating a GitHub repository is outward-facing: ask the user to create it, or get an explicit yes before running `gh repo create --private`. Never make it public. In a dry run, only describe each step.
  1. **Vault.** Clone the empty remote, or `git init`. Then `hub init <path> --name <name>`, using the user's installed `hub`. If `hub` is not installed, point to USER-GUIDE Step 1. Then `bash <path>/tools/setup.sh`.
  2. **`project.yaml` interview.** Ask about each key in turn and fill it in:
     - **people:** keys, names and emails; ClickUp ids from `clickup_get_workspace_members`.
     - **series:** attendees, and `calendar_match` checked against real titles with `list_events`.
     - **sources:**
       - ClickUp list via `clickup_get_list` by name;
       - Circleback folder via `ListFolders` (recommend creating and sharing a project folder);
       - repositories (roots plus branch);
       - Overleaf git clone;
       - calendar;
       - Gmail/Slack, opt-in.
     - **roots.**

     Write `project.yaml` and the user's `hub.local.yaml`. Then run `hub validate` and `hub sources`.
  3. `hub commit "Set up project hub"`. If it reports no upstream: `git push -u origin <branch>` once.
  4. **Vision:** run the hub-vision skill.
  5. **Existing material:** for each document the user names, follow `sources/legacy-import.md`.
  6. **Figures:** run the hub-figures skill.
  7. **Locator:**
     - draft `Locator.md` content (code and data roots as `$ROOT/...`, documents, external links);
     - show it;
     - stage it as a proposal with a `note` op (`hub pending add --ops-file`) and apply it after the user's yes;
     - `hub check-links`.
  8. **First ingest:** the hub-status skill, first in dry run, then for real.
  9. **Collaborators:** tell the user what to send each one: the repo invitation, add them to `people` first, they run `bash tools/setup.sh` and write `hub.local.yaml`, share the Circleback folder, connect the services.
  - Also cover **updating an existing vault's tools:** `<newer project-hub>/.venv/bin/hub init <vault> --name <name>`, review the diff, commit.
- [ ] **Step 2: `skills/hub-init/sources/legacy-import.md`**, the generic importer. It must say:
  - **Any document, any format.** Markdown, LaTeX, plain text, exported docs, tracker exports. Read it whole, or section by section if long. Never assume a structure.
  - **Classify each passage:**
    - open question;
    - decision taken (with rationale, date, who);
    - action (owner, done or not);
    - question already answered or dropped (with the answer);
    - stable knowledge (how the code works, conventions, what a result supports, claims tried and rejected);
    - vision material (hand it to hub-vision);
    - history narrative (at most a few log-line proposals);
    - noise or duplicates (skip).
  - **When unsure** whether something was decided, make it an open question. Never invent facts, owners or dates. Use the document's dates; otherwise today.
  - **People:** map names to people keys. Unknown people stay in the text and never become an `owner`.
  - **Source ids:** `doc:<file-stem-slug>#<section-slug>` (lowercase, non-alphanumerics → `-`).
  - **Batch file** (scratch dir, outside the vault). An import is all judgement, so everything goes in `proposals`:
    - one proposal per decision, with its question (`ref: $q` when new) and its actions (`ref: $d`, `spawned_by`);
    - one proposal per standalone open question;
    - closed questions as `new` question with `fields: {status: answered|dropped, closed: <date>}` plus a closing `entry`;
    - done actions likewise (`status: done`, `closed`, entry);
    - one proposal per Knowledge note: a `note` op, path `Knowledge/<Topic>.md`, mode `replace`.

    Keep the document's own numbering in the description for traceability, e.g. "Imported from <doc>, item 12".
  - **Show a summary:** counts by kind and section, plus anything skipped and why. In a dry run, stop here.
  - **Otherwise:**
    1. `hub ingest batch <file>`;
    2. the approval batch (approve by section, at most about 30 at a time);
    3. `hub pending apply`;
    4. close the gaps as in hub-status update step 7;
    5. `hub validate`;
    6. `hub commit "import: <doc> (<date>)"`.
  - **Privacy:** the same rules as hub-status. Leave out personal assessments, private notes, and career material.
- [ ] **Step 3: `skills/hub-vision/SKILL.md`**:
  - **Interview** (one question at a time; push for specifics and evidence):
    - the vision;
    - the problem as the community poses it, with key references;
    - our approach;
    - how we answer the problem;
    - our unfair advantage (why us, why now; what competitors lack);
    - the difference we will make (who changes what);
    - risks and early signals.
  - **Check claims of novelty or advantage:**
    - if a literature-search skill is available, offer to run it;
    - otherwise ask for evidence;
    - mark unsupported claims "(to verify)".
  - **Draft** `Vision.md` in the template's sections. Under `## Revisions` append `- <YYYY-MM-DD> — <what changed> — <initial | after [[D-…]]>`.
  - **Stage** it as a proposal:
    - an ops file with one `note` op, path `Vision.md`, mode `replace`;
    - `hub pending add --title "Vision: <what changed>" --ops-file <file> --source manual --reason "<why>"`;
    - show the diff against the current `Vision.md`;
    - `hub pending apply` after the user's yes;
    - `hub commit`.
  - **Revisions later:** triggered by a pivot (a question pivoted, or a decision that changes direction) or by the user. Keep unchanged sections verbatim. Every revision adds a Revisions line.
  - Include the dry-run rule.
- [ ] **Step 4: `skills/hub-figures/SKILL.md`**:
  - **Add.** For each figure the user names, ask for:
    - a short id (lowercase, `-`);
    - the figure file under a root (`$ROOT/...`);
    - the script that makes it;
    - its data files or folders;
    - a title;
    - `licenses`: one sentence on the claim it supports.

    Then `hub figures add …` with `--caption` (draft one for the user to accept; captions are theirs afterwards) and `hub commit`.
  - **Sync:** after figures are regenerated, `hub figures sync`. Explain STALE (the data changed after the figure was made: regenerate it with its script) and PROBLEM (source missing). `hub figures gallery`. `hub commit`.
  - **Editing:** `title`, `status` (draft/final), `licenses`, `script` and `data` may be edited by hand in `Figures/<id>/meta.yaml`; then run `hub figures sync`. Never edit `caption.md` for the user unless asked; it is theirs.
  - `hub check-links` reports paths that are missing on this machine.
- [ ] **Step 5: `skills/hub-status/SKILL.md` additions** (minimal):
  - in update step 2's source list: if `Figures/` has registered figures, run `hub figures sync`. It is safe and committed like a source, and STALE or PROBLEM lines are reported;
  - in step 7 (close the gaps): if an applied proposal pivoted a question, or the user says the direction changed, suggest running hub-vision to revise the vision;
  - add `hub figures sync` to the dry-run skip list, since it writes.
- [ ] **Step 6: Run** `tests/test_skills.py` and the full suite.
- [ ] **Step 7: Commit** `feat(skills): hub-init with generic legacy import, hub-vision, hub-figures`.

---
### Task 8: User guide
**Files:** Modify `project-hub/USER-GUIDE.md`.
- **One-time setup:** the recommended path is now the hub-init skill (*"set up a project hub"*) or `hub init <dir> --name <name>` followed by `bash tools/setup.sh`. Keep the manual steps as the reference. Each step should say which part `hub init`/`setup.sh` does for you.
- **Joining:** clone the vault, then `bash tools/setup.sh`. No research-repository access is needed any more.
- **New sections:**
  - *The vision* (hub-vision; changed only through proposals);
  - *Figures* (add, sync, stale, gallery, captions are yours);
  - *The Locator and check-links*;
  - *Importing existing notes* (any document; everything is proposed; approve by section).
- Add the new commands to the command tables and the quick reference card. Add the new messages to Troubleshooting:
  - figure errors;
  - check-links lines;
  - `note path must be …`.
- **What is not built yet:** only Bases views remain uncertain. `Status/Items.base` is created, but its view syntax is untested in Obsidian; say so. Update Known limitations: remove the conflict data-loss item (fixed in Task 1).
- Keep the Obsidian style rule.
- **Commit:** `docs(hub): user guide for init, vision, figures, locator, import`.

---
### Task 9: Live check (controller-run, with the user)
Not delegable without connectors and without the user's choice of document.
1. **Scratch vault.** `hub init` a scratch vault inside the job's temp area (not git-pushed), then `bash tools/setup.sh` with `HOME` pointed at a scratch home, so the user's real `~/.local/bin` is untouched.
2. **Legacy import, dry run.** A fresh agent runs the hub-init skill's legacy import in **dry run** on a document the user picks (default: an existing project notes file such as the UCC `HANDOFF.md`, read-only). Check that:
   - the batch passes `hub ingest batch` in a throwaway copy of the vault;
   - nothing in the skill text was project-specific;
   - its friction notes are recorded.
3. **Figures and links:** `hub figures add` on two real figure files from that project, then `hub figures sync`, `hub check-links`, `hub snapshot`.
4. **Fix and delete.** Fix skill friction (commits `fix(skills): …`), then delete the scratch vault.

---
## Self-review notes
- **Spec coverage:**
  - Vault layout with `Vision.md`, `Locator.md`, `Figures/`, `Knowledge/`, `tools/`, `.claude/skills/`, `Items.base`: Tasks 5 and 6.
  - *Ownership*: Vision, Locator and Knowledge are written only through note proposals (Task 2); captions are never overwritten (Task 3).
  - Snapshot "stale figures": Task 3.
  - *Skills*: hub-init, hub-vision and hub-figures (Task 7).
  - *CLI*: figures, check-links and import (Tasks 3–6). The import is a skill procedure over `hub ingest batch` rather than a format-specific `import-handoff` command, so no project's format is baked in.
  - Collaborator setup without the research repository: vendoring in Task 5.
  - The conflict data loss found while writing the guide: Task 1.
- **Not covered:** Bases views are created but cannot be tested from here (flagged in the guide).
- **Names used across tasks:**
  - `resolve(config, spec)`, `ROOT_REF_RE`, `PathError`;
  - `add_figure` / `sync_figures` / `stale_figures` / `render_gallery` / `write_gallery`, `FigureError`;
  - `check_links`;
  - `init_vault` / `update_tools`, `TEMPLATES`;
  - `note_path`, `NOTE_PATH_RE`;
  - `run_init(args)`.
