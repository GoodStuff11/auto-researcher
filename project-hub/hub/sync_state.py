"""Status/sync-state/: per-source ingest watermarks, next meeting per series, unresolved sync flags.

One small file per key, so clones updating different sources never touch the same file:
  Status/sync-state/watermark.<source>.yaml      `value: <watermark>` and `updated: <ISO datetime, UTC>`
  Status/sync-state/next_meeting.<series>.yaml   `value: <YYYY-MM-DD>` and `updated: <ISO datetime, UTC>`
  Status/sync-state/flags.md                     one `- <text>` line per flag (git `merge=union`)

`updated` is the time the value last changed; it lets the `hub-watermark` merge driver (hub.merge_watermark)
keep the newer side when two clones change the same key. A file without `updated` (older vaults) counts
as oldest. A write that does not change the value leaves the file (and its `updated`) untouched.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .fsutil import atomic_write_text

KINDS = {"watermarks": "watermark", "next_meeting": "next_meeting"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")


def state_dir(root) -> Path:
    return Path(root) / "Status" / "sync-state"


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"sync-state key must be letters, digits, '_', '-' or '.', got {name!r}")
    return name


def _key_path(root, kind: str, name: str) -> Path:
    return state_dir(root) / f"{kind}.{_check_name(name)}.yaml"


def _read_value(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("value") if isinstance(data, dict) else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _write_value(path: Path, value) -> None:
    """Write `value` with a fresh `updated` stamp, unless the file already holds this value."""
    if path.is_file():
        try:
            if _norm(_read_value(path)) == _norm(value):
                return
        except yaml.YAMLError:
            pass
    atomic_write_text(path, yaml.safe_dump({"value": value, "updated": _now()}, sort_keys=False,
                                           allow_unicode=True))


def load_state(root) -> dict:
    d = state_dir(root)
    state: dict = {"watermarks": {}, "next_meeting": {}, "flags": []}
    if not d.is_dir():
        return state
    for key, kind in KINDS.items():
        for p in sorted(d.glob(f"{kind}.*.yaml")):
            state[key][p.name[len(kind) + 1:-len(".yaml")]] = _read_value(p)
    flags = d / "flags.md"
    if flags.is_file():
        state["flags"] = [ln[2:].strip() for ln in flags.read_text(encoding="utf-8").splitlines()
                          if ln.startswith("- ") and ln[2:].strip()]
    return state


def save_state(root, state: dict) -> None:
    """Write the whole state; keys absent from `state` are removed."""
    d = state_dir(root)
    for key, kind in KINDS.items():
        wanted = {_check_name(n): v for n, v in (state.get(key) or {}).items()}
        for p in d.glob(f"{kind}.*.yaml") if d.is_dir() else []:
            if p.name[len(kind) + 1:-len(".yaml")] not in wanted:
                p.unlink()
        for name, value in wanted.items():
            _write_value(_key_path(root, kind, name), value)
    flags = [" ".join(str(f).split()) for f in state.get("flags") or []]
    atomic_write_text(d / "flags.md", "".join(f"- {f}\n" for f in flags if f))


def get_watermark(root, source: str):
    p = _key_path(root, "watermark", source)
    return _read_value(p) if p.is_file() else None


def set_watermark(root, source: str, value: str) -> None:
    _write_value(_key_path(root, "watermark", source), value)
