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
    _git(root, "add", "-A", "--", ".")
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
    if _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False).returncode != 0:
        raise SyncError("no upstream branch configured; run `git push -u origin <branch>` once")
    for _ in range(retries):
        pull = _git(root, "pull", "--rebase", check=False)
        if pull.returncode != 0:
            output = pull.stdout + pull.stderr
            files = _git(root, "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
            if files:
                _git(root, "rebase", "--abort", check=False)
                raise SyncConflict(files, output)
            if (root / ".git" / "rebase-merge").exists() or (root / ".git" / "rebase-apply").exists():
                _git(root, "rebase", "--abort", check=False)
            raise SyncError(f"git pull --rebase failed: {output}")
        if regenerate:
            regenerate()
            _commit_all(root, "Regenerate generated files")
        push = _git(root, "push", check=False)
        if push.returncode == 0:
            return _git(root, "rev-parse", "HEAD").stdout.strip()
        if not any(s in push.stderr for s in ("non-fast-forward", "fetch first", "rejected")):
            raise SyncError(f"git push failed: {push.stderr.strip()}")
    raise SyncError(f"push rejected {retries} times; someone is pushing continuously — retry later")
