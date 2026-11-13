# project-hub/hub/gitsync.py
"""Commit and push a hub vault safely when several people (and their agents) write to it.

Strategy: commit locally → `git pull --rebase` → regenerate generated files → push; retry on rejection.
Append-only files merge with git's built-in `union` driver; Snapshot.md keeps upstream (`ours` driver)
and is regenerated; item files merge with the `hub-item` driver (hub.merge_item: union of entries and
lists, three-way on scalars), so only a genuine clash on one field surfaces as SyncConflict.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

GITATTRIBUTES_LINES = [
    "Status/Log/*.md merge=union",
    "Status/Snapshot.md merge=ours",
    "Figures/Gallery.md merge=ours",
    "Status/Items/*.md merge=hub-item",
    "Status/sync-state/flags.md merge=union",
    "Status/sync-state/watermark.*.yaml merge=hub-watermark",
    "Status/sync-state/next_meeting.*.yaml merge=hub-watermark",
    "*.xlsx binary",
    "*.md text eol=lf",
    "*.yaml text eol=lf",
]
GITIGNORE_LINES = ["hub.local.yaml", ".venv/", "__pycache__/", "*.egg-info/", ".obsidian/workspace*.json",
                   "~$*", ".~lock.*#", ".*.tmp"]
CONFLICT_GUIDANCE = ("in `hub-item conflict` lines, A is the version already on the remote and B is yours. "
                     "The item file already holds both sides' changes except the field(s) named in those "
                     "lines, which show A's value; B's value appears only in those lines, so copy it, then "
                     "set each named field to the right value before `git add` (a plain `git add` silently "
                     "keeps A). "
                     "A conflicted Status/sync-state/*.yaml file (watermark or next meeting) means the "
                     "hub-watermark driver could not read one side: keep the side with the newer `updated:`. "
                     "Plain-text files (Vision.md, Locator.md, Knowledge/*.md, Figures/<id>/meta.yaml, "
                     "Figures/<id>/caption.md) hold normal git conflict markers: open the file, keep both sides' "
                     "intent, remove the <<<<<<< ======= >>>>>>> markers, then `git add` it. "
                     "Status/Snapshot.md and Figures/Gallery.md are regenerated automatically: never edit them. "
                     "Re-run after fixing: `git pull --rebase`, fix the file, `git add` it, "
                     "`git rebase --continue`, then `hub validate` and `hub commit`")


class SyncError(Exception):
    """Push kept being rejected, or git failed unexpectedly."""


class SyncConflict(SyncError):
    def __init__(self, files: list[str], detail: str):
        super().__init__(f"merge conflict in: {', '.join(files) or '(unknown)'}\n{detail.rstrip()}\n"
                         f"{CONFLICT_GUIDANCE}")
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


def _driver_python(root) -> str:
    """The vault's own tools/.venv python when it exists (it survives other venvs being moved), else ours."""
    vault_py = Path(root).resolve() / "tools" / ".venv" / "bin" / "python"
    if vault_py.is_file() and os.access(vault_py, os.X_OK):
        return str(vault_py)
    return sys.executable


def _configure_drivers(root) -> None:
    """Register the merge drivers named in .gitattributes in this clone's .git/config."""
    py = shlex.quote(_driver_python(root))
    _git(root, "config", "merge.ours.driver", "true")
    _git(root, "config", "merge.hub-item.name", "hub item merge")
    _git(root, "config", "merge.hub-item.driver", f"{py} -m hub.merge_item %O %A %B")
    _git(root, "config", "merge.hub-watermark.name", "hub watermark merge (newer updated wins)")
    _git(root, "config", "merge.hub-watermark.driver", f"{py} -m hub.merge_watermark %O %A %B")


def ensure_repo_files(root) -> None:
    root = Path(root)
    _ensure_lines(root / ".gitattributes", GITATTRIBUTES_LINES)
    _ensure_lines(root / ".gitignore", GITIGNORE_LINES)
    _configure_drivers(root)


def _has_upstream(root) -> bool:
    return _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False).returncode == 0


def _abort_rebase_if_any(root) -> None:
    git_dir = _git(root, "rev-parse", "--git-dir", check=False).stdout.strip()
    if git_dir and any((Path(root) / git_dir / d).exists() for d in ("rebase-merge", "rebase-apply")):
        _git(root, "rebase", "--abort", check=False)


def pull_if_possible(root) -> str | None:
    """Best-effort `git pull --rebase` before acting on shared state (pending apply, agenda read).

    Does nothing (returns None) when `root` is not a git work tree or the branch has no upstream.
    On failure the rebase is aborted and a one-line warning is returned for the caller to print;
    the caller carries on with the local state.
    """
    root = Path(root)
    inside = _git(root, "rev-parse", "--is-inside-work-tree", check=False)
    if inside.returncode != 0 or inside.stdout.strip() != "true" or not _has_upstream(root):
        return None
    _configure_drivers(root)
    pull = _git(root, "pull", "--rebase", check=False)
    if pull.returncode == 0:
        return None
    _abort_rebase_if_any(root)
    detail = " ".join((pull.stderr or pull.stdout).split())
    return f"warning: could not pull before acting (continuing with local state): {detail}"


def _regenerate(root, regenerate) -> None:
    try:
        regenerate()
    except Exception as exc:  # a broken generator must never block pushing the user's own changes
        print(f"warning: could not regenerate generated files: {exc}", file=sys.stderr)
        return
    _commit_all(root, "Regenerate generated files")


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
            _regenerate(root, regenerate)
        return _git(root, "rev-parse", "HEAD").stdout.strip()
    if not _has_upstream(root):
        raise SyncError("no upstream branch configured; run `git push -u origin <branch>` once")
    for _ in range(retries):
        pull = _git(root, "pull", "--rebase", check=False)
        if pull.returncode != 0:
            output = pull.stdout + pull.stderr
            files = _git(root, "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
            if files:
                _git(root, "rebase", "--abort", check=False)
                raise SyncConflict(files, output)
            _abort_rebase_if_any(root)
            raise SyncError(f"git pull --rebase failed: {output}")
        if regenerate:
            _regenerate(root, regenerate)
        push = _git(root, "push", check=False)
        if push.returncode == 0:
            return _git(root, "rev-parse", "HEAD").stdout.strip()
        if not any(s in push.stderr for s in ("non-fast-forward", "fetch first", "rejected")):
            raise SyncError(f"git push failed: {push.stderr.strip()}")
    raise SyncError(f"push rejected {retries} times; someone is pushing continuously — retry later")
