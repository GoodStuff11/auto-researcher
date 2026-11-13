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
