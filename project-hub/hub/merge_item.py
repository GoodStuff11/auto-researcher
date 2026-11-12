"""Git merge driver for item files (Status/Items/*.md): `python -m hub.merge_item %O %A %B`.

Two people adding entries to (or linking) the same item is normal, and a line-based text merge
turns it into a conflict every time. This driver merges the parsed items instead:

- entries: union of ours (A) and theirs (B), de-duplicated by rendered line, stable-sorted by date
  (ties keep A's order, then B's);
- list fields (`sources`, `series`, every `links.*`): union, order preserved (A first);
- every other frontmatter key and the description: three-way merge against the base (O). One side
  changed it → that side wins; both changed it to the same value → that value; both changed it
  differently → a real conflict.

Exit 0 after writing the merged item to %A. On a real conflict, %A keeps A's version, one line per key
`hub-item conflict in <id>: <key>: ours=<A> theirs=<B>` goes to stderr, and the exit code is 1 (git
then reports the file as conflicted). An input that does not parse also exits 1. Registered by
`gitsync.ensure_repo_files` (`.gitattributes`: `Status/Items/*.md merge=hub-item`).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .fsutil import atomic_write_text
from .items import LINK_KEYS, Item, ItemFormatError, parse_item, render_item

LIST_KEYS = ("sources", "series")
_MISSING = object()


def _union(a, b) -> list:
    out = list(a or [])
    for v in b or []:
        if v not in out:
            out.append(v)
    return out


def _three_way(o, a, b):
    """Return (value, conflict?) for one scalar; _MISSING means the key is absent."""
    if a == b:
        return a, False
    if a == o:
        return b, False
    if b == o:
        return a, False
    return a, True


def _show(v) -> str:
    if v is _MISSING:
        return "(absent)"
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return repr(v) if isinstance(v, str) and (not v or v != v.strip()) else str(v)


def merge_items(o: Item | None, a: Item, b: Item) -> tuple[Item, list[tuple[str, object, object]]]:
    """Merge three parsed items; return the merged item and the list of (key, ours, theirs) conflicts."""
    o_meta = o.meta if o else {}
    conflicts: list[tuple[str, object, object]] = []
    meta: dict = {}
    for key in list(a.meta) + [k for k in b.meta if k not in a.meta]:
        av, bv = a.meta.get(key, _MISSING), b.meta.get(key, _MISSING)
        if key in LIST_KEYS and (isinstance(av, list) or isinstance(bv, list)):
            meta[key] = _union(av if isinstance(av, list) else [], bv if isinstance(bv, list) else [])
            continue
        if key == "links" and (isinstance(av, dict) or isinstance(bv, dict)):
            al = av if isinstance(av, dict) else {}
            bl = bv if isinstance(bv, dict) else {}
            meta[key] = {rel: _union(al.get(rel), bl.get(rel))
                         for rel in list(al) + [r for r in bl if r not in al]}
            for rel in LINK_KEYS:
                meta[key].setdefault(rel, [])
            continue
        value, bad = _three_way(o_meta.get(key, _MISSING), av, bv)
        if bad:
            conflicts.append((key, av, bv))
        if value is not _MISSING:
            meta[key] = value
    description, bad = _three_way(o.description if o else "", a.description, b.description)
    if bad:
        conflicts.append(("description", a.description, b.description))
    seen: set[str] = set()
    entries = []
    for e in a.entries + b.entries:
        line = e.render()
        if line not in seen:
            seen.add(line)
            entries.append(e)
    entries.sort(key=lambda e: e.day)  # stable: ties keep A's order, then B's
    return Item(meta, description, entries), conflicts


def _load(path: str, allow_empty: bool) -> Item | None:
    text = Path(path).read_text(encoding="utf-8")
    if allow_empty and not text.strip():
        return None  # add/add: git passes an empty base
    return parse_item(text)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("usage: python -m hub.merge_item %O %A %B", file=sys.stderr)
        return 2
    o_path, a_path, b_path = argv
    try:
        o = _load(o_path, allow_empty=True)
        a = _load(a_path, allow_empty=False)
        b = _load(b_path, allow_empty=False)
        merged, conflicts = merge_items(o, a, b)
        if conflicts:
            item_id = a.meta.get("id", "?")
            for key, av, bv in conflicts:
                print(f"hub-item conflict in {item_id}: {key}: ours={_show(av)} theirs={_show(bv)}",
                      file=sys.stderr)
            return 1
        atomic_write_text(a_path, render_item(merged))
    except (ItemFormatError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"hub-item: cannot merge: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
