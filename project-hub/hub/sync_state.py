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
