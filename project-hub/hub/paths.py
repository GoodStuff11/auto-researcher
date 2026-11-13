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

ROOT_REF_RE = re.compile(r"(?<![\w$])\$([A-Za-z_][A-Za-z0-9_]*)/((?:[^\s)\]>`'\"|*]*[^\s)\]>`'\"|*.,;:])?)")
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
