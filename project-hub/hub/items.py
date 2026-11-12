# project-hub/hub/items.py
"""Item files: YAML frontmatter (state) + description + append-only '## Entries' (history)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

import yaml

TYPES = ("question", "decision", "action")
STATUSES = {
    "question": ("open", "answered", "dropped", "pivoted"),
    "decision": ("proposed", "made", "rejected", "deferred"),
    "action": ("open", "done", "dropped"),
}
CLOSING = {"answered", "dropped", "pivoted", "done", "rejected"}
LINK_KEYS = ("answers", "spawned_by", "pivoted_to")
DATE_FIELDS = ("raised", "closed", "due")
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
        for name, value in (("author", self.author), ("source", self.source)):
            if not value or any(ch.isspace() for ch in value):
                raise ValueError(
                    f"entry author/source must be non-empty with no whitespace: "
                    f"{self.author!r} / {self.source!r}")
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
        links = self.meta.get("links")
        value = links.get(key) if isinstance(links, dict) else None
        return [str(v) for v in value] if isinstance(value, list) else []

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
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ItemFormatError("missing frontmatter opening '---'")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ItemFormatError("missing frontmatter closing '---'")
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as exc:
        raise ItemFormatError(f"invalid frontmatter YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise ItemFormatError("frontmatter must be a mapping")
    for key in ("id", "type", "title", "status"):
        if key not in meta:
            raise ItemFormatError(f"frontmatter missing '{key}'")
    for key in DATE_FIELDS:  # YAML reads `2026-11-12 10:00` as a datetime; the store works in dates
        if isinstance(meta.get(key), datetime):
            meta[key] = meta[key].date()
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
            try:
                day = date.fromisoformat(m[1])
            except ValueError as exc:
                raise ItemFormatError(f"malformed entry date: {line!r}") from exc
            entries.append(Entry(day, m[2], m[3], m[4]))
    else:
        description = "\n".join(lines).strip()
    return Item(meta, description, entries)


def render_item(item: Item) -> str:
    front = yaml.dump(item.meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True,
                      default_flow_style=False, width=float("inf"))
    out = ["---", front.rstrip("\n"), "---", item.description.strip(), ENTRIES_HEADING]
    out += [e.render() for e in item.entries]
    return "\n".join(out) + "\n"
