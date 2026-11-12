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
