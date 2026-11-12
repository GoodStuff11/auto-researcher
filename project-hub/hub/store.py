# project-hub/hub/store.py
"""The status store: one Markdown file per item under Status/Items, monthly logs under Status/Log."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .ids import new_id
from .items import (LINK_KEYS, STATUSES, TYPES, Entry, Item, default_meta, parse_item, render_item)

DATE_KEYS = {"raised", "closed", "due"}
INT_KEYS = {"priority"}
LIST_KEYS = {"series", "sources"}
PROTECTED_KEYS = {"id", "type", "links"}
ID_RE = re.compile(r"^[QDA]-\d{8}-[a-z0-9]{3}$")


class StoreError(Exception):
    """A store operation would break the item format or reference a missing item."""


def coerce_value(key: str, value):
    """Convert CLI strings into the types stored in frontmatter."""
    if isinstance(value, str):
        if value == "":
            return None
        try:
            if key in DATE_KEYS:
                return date.fromisoformat(value)
            if key in INT_KEYS:
                return int(value)
        except ValueError as exc:
            raise StoreError(f"invalid value for '{key}': {value!r}") from exc
    return value


def check_field(key: str, value):
    """Reject values whose shape would break the item format or the queries built on it."""
    if key in LIST_KEYS and not isinstance(value, list):
        raise StoreError(f"'{key}' must be a list, got {value!r}")
    if key == "title" and (not isinstance(value, str) or not value.strip()):
        raise StoreError("'title' must be a non-empty string")
    return value


def _valid_ref(target, existing: set[str]) -> bool:
    return isinstance(target, str) and bool(ID_RE.match(target)) and target in existing


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.items_dir = self.root / "Status" / "Items"
        self.log_dir = self.root / "Status" / "Log"

    def path(self, item_id: str) -> Path:
        if not isinstance(item_id, str) or not ID_RE.match(item_id):
            raise KeyError(item_id)
        return self.items_dir / f"{item_id}.md"

    def all(self) -> list[Item]:
        if not self.items_dir.is_dir():
            return []
        return [parse_item(p.read_text()) for p in sorted(self.items_dir.glob("*.md"))]

    def ids(self) -> set[str]:
        if not self.items_dir.is_dir():
            return set()
        return {p.stem for p in self.items_dir.glob("*.md")}

    def get(self, item_id: str) -> Item:
        p = self.path(item_id)
        if not p.is_file():
            raise KeyError(item_id)
        return parse_item(p.read_text())

    def save(self, item: Item) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.path(item.id).write_text(render_item(item))

    def create(self, item_type, title, *, owner, author, day, source, series=(), priority=2,
               links=None, description="", fields=None, rng=None) -> Item:
        if item_type not in TYPES:
            raise StoreError(f"unknown item type '{item_type}' (expected one of {TYPES})")
        existing = self.ids()
        for rel, targets in (links or {}).items():
            if rel not in LINK_KEYS:
                raise StoreError(f"unknown link '{rel}' (expected one of {LINK_KEYS})")
            for t in targets:
                if not _valid_ref(t, existing):
                    raise StoreError(f"link target {t} does not exist")
        item_id = new_id(item_type, day, existing, rng)
        meta = default_meta(item_id, item_type, title, owner, day, series, priority, [source])
        for rel, targets in (links or {}).items():
            meta["links"][rel] = list(targets)
        for k, v in (fields or {}).items():
            if k in PROTECTED_KEYS:
                raise StoreError(f"cannot set '{k}' through fields")
            meta[k] = check_field(k, coerce_value(k, v))
        for k in ("title", "series", "sources"):
            check_field(k, meta[k])
        if meta["status"] not in STATUSES[item_type]:
            raise StoreError(f"status '{meta['status']}' invalid for {item_type}")
        item = Item(meta, description, [Entry(day, author, source, "Raised.")])
        self.save(item)
        return item

    def entry(self, item_id, text, *, author, source, day) -> Item:
        item = self.get(item_id)
        item.entries.append(Entry(day, author, source, text))
        src = item.meta.setdefault("sources", [])
        if source not in src:
            src.append(source)
        self.save(item)
        return item

    def set_fields(self, item_id, fields: dict) -> Item:
        item = self.get(item_id)
        for k, v in fields.items():
            if k in PROTECTED_KEYS:
                raise StoreError(f"cannot set '{k}' (use `hub link` for links)")
            v = check_field(k, coerce_value(k, v))
            if k == "status" and v not in STATUSES[item.type]:
                raise StoreError(f"status '{v}' invalid for {item.type}; expected one of {STATUSES[item.type]}")
            item.meta[k] = v
        self.save(item)
        return item

    def link(self, item_id, rel, target) -> Item:
        if rel not in LINK_KEYS:
            raise StoreError(f"unknown link '{rel}' (expected one of {LINK_KEYS})")
        if not _valid_ref(target, self.ids()):
            raise StoreError(f"link target {target} does not exist")
        item = self.get(item_id)
        links = item.meta.setdefault("links", {k: [] for k in LINK_KEYS})
        lst = links.setdefault(rel, [])
        if target not in lst:
            lst.append(target)
        self.save(item)
        return item

    def query(self, type=None, status=None, owner=None, series=None, since=None) -> list[Item]:
        out = []
        for it in self.all():
            if type and it.type != type:
                continue
            if status and it.status != status:
                continue
            if owner and it.owner != owner:
                continue
            if series and series not in (it.meta.get("series") or []):
                continue
            if since and it.last_activity() < since:
                continue
            out.append(it)
        return out

    def children(self, item_id) -> list[Item]:
        """Items that point at item_id: decisions answering it, items spawned by it, its pivot targets.

        Raises KeyError if item_id does not exist. Pivot targets that do not exist are skipped;
        each child appears once, in first-seen order.
        """
        me = self.get(item_id)
        found: dict[str, Item] = {}
        for it in self.all():
            if item_id in it.links("answers") or item_id in it.links("spawned_by"):
                found.setdefault(it.id, it)
        for t in me.links("pivoted_to"):
            if t in found:
                continue
            try:
                found[t] = self.get(t)
            except KeyError:
                continue
        return list(found.values())

    def log(self, text, *, author, source, day) -> Path:
        line = Entry(day, author, source, text).render() + "\n"  # validates before touching disk
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{day:%Y-%m}.md"
        if not path.is_file():
            path.write_text(f"# Log {day:%Y-%m}\n")
        with path.open("a") as fh:
            fh.write(line)
        return path
