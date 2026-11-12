"""Git merge driver for sync-state key files: `python -m hub.merge_watermark %O %A %B`.

Status/sync-state/watermark.<source>.yaml and next_meeting.<series>.yaml hold `value:` plus `updated:`
(ISO datetime, UTC, written by hub.sync_state). Two people ingesting the same source on the same day
both change the file; a text merge would conflict every time. This driver keeps the side whose
`updated` is newer (a missing or unreadable `updated` counts as oldest); on a tie, the side whose
`value` is larger as a string. The base (%O) is ignored.

Exit 0 after writing the winning side's text to %A. A side that is not a YAML mapping exits 1 (git then
reports the file as conflicted). Registered by `gitsync.ensure_repo_files` (`.gitattributes`:
`Status/sync-state/watermark.*.yaml merge=hub-watermark`, same for `next_meeting.*.yaml`).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .fsutil import atomic_write_text

_OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def _updated(data: dict) -> datetime:
    raw = data.get("updated")
    if isinstance(raw, datetime):
        stamp = raw
    else:
        try:
            stamp = datetime.fromisoformat(str(raw))
        except ValueError:
            return _OLDEST
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _value(data: dict) -> str:
    v = data.get("value")
    return v.isoformat() if hasattr(v, "isoformat") else ("" if v is None else str(v))


def _load(text: str) -> dict:
    data = yaml.safe_load(text) if text.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("not a YAML mapping")
    return data


def pick(a_text: str, b_text: str) -> str:
    """Return the text of the side to keep (A on a full tie)."""
    a, b = _load(a_text), _load(b_text)
    if (_updated(b), _value(b)) > (_updated(a), _value(a)):
        return b_text
    return a_text


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("usage: python -m hub.merge_watermark %O %A %B", file=sys.stderr)
        return 2
    _, a_path, b_path = argv
    try:
        a_text = Path(a_path).read_text(encoding="utf-8")
        b_text = Path(b_path).read_text(encoding="utf-8")
        keep = pick(a_text, b_text)
        if keep is not a_text:
            atomic_write_text(a_path, keep)
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        print(f"hub-watermark: cannot merge {a_path}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
