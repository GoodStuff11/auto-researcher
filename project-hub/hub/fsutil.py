"""Filesystem helpers shared by every writer of vault files."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path, text: str) -> None:
    """Write `text` to `path` so readers see either the old or the new content, never a partial file.

    The text is written to a temporary file in the same directory, flushed to disk, then moved over
    `path` with os.replace (atomic on POSIX and Windows when both are on one filesystem).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
