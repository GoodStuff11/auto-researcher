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
_RECORD = "\x00"


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
    elif since_sha and _git(path, "merge-base", "--is-ancestor", since_sha, ref).returncode != 0:
        warning = (f"watermark {since_sha} is not an ancestor of {ref} (branch reset?); "
                   f"listed commits since {since_date} instead")
        since_sha = None
    args = ["-c", "core.quotePath=false", "log", "--no-merges", "--date=short",
            f"--format=%x00%H{_FIELD}%h{_FIELD}%ad{_FIELD}%an{_FIELD}%s", "--name-only"]
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
        sha, short, day, author, subject = head.split(_FIELD, 4)
        commits.append({"sha": sha, "short": short, "date": day, "author": author, "subject": subject,
                        "files": [f for f in files if f.strip()]})
    commits.reverse()
    parts = [warning] if warning else []
    if len(commits) > limit:
        dropped = len(commits) - limit
        commits = commits[:limit]
        parts.append(f"{dropped} newer commit(s) not listed yet (limit {limit}); run again after --advance")
    return commits, "; ".join(parts) or None


def ingest_git(config: HubConfig, name: str, *, today: date, since: date | None = None,
               advance: bool = False, limit: int = 500) -> dict:
    """List commits since the watermark (or since `since`, which ignores the watermark). With `advance`,
    move the watermark to the newest listed commit; run it only after the commits have been recorded."""
    path = resolve_root(config, name)
    branch = configured_branch(config, name)
    wm = watermark_name(name)
    before = get_watermark(config.root, wm)
    since_date = since or (today - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    commits, warning = git_commits(path, branch, since_sha=None if since else before,
                                   since_date=since_date, limit=limit)
    advanced = bool(advance and commits)
    if advanced:
        set_watermark(config.root, wm, commits[-1]["sha"])
    return {"root": name, "path": str(path), "branch": branch or "HEAD", "watermark_before": before,
            "commits": commits, "warning": warning, "advanced": advanced}
