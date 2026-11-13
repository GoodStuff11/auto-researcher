# project-hub/hub/links.py
"""`hub check-links`: every $ROOT/path and URL in Locator.md and figure meta.yaml files.

Paths are checked on this machine (roots come from hub.local.yaml). URLs are only listed unless
online=True, which sends a HEAD request (falling back to GET) with a short timeout.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from .paths import ROOT_REF_RE, PathError, resolve

URL_RE = re.compile(r"https?://[^\s)\]>`'\"|]+")


def _refs_in_text(text: str):
    for m in ROOT_REF_RE.finditer(text):
        yield "path", f"${m[1]}/{m[2]}".rstrip(".,;:")
    for m in URL_RE.finditer(text):
        yield "url", m[0].rstrip(".,;:")


def _meta_refs(path: Path):
    try:
        meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return
    if not isinstance(meta, dict):
        return
    data = meta.get("data") or []
    if isinstance(data, str):
        data = [data]
    elif not isinstance(data, list):
        data = []
    specs = [meta.get("source"), meta.get("script"), *data]
    for spec in specs:
        if isinstance(spec, str) and spec.strip():
            yield ("url" if URL_RE.fullmatch(spec.strip()) else "path"), spec.strip()


def _check_url(url, timeout, opener):
    opener = opener or urllib.request.urlopen
    for method in ("HEAD", "GET"):
        try:
            with opener(urllib.request.Request(url, method=method), timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
            if status < 400:
                return True, None
            if method == "GET" or status not in (403, 405):
                return False, f"HTTP {status}"
        except urllib.error.HTTPError as exc:
            if method == "GET" or exc.code not in (403, 405):
                return False, f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, f"unreachable: {exc}"
    return False, "unreachable"


def check_links(config, *, online=False, timeout=5.0, opener=None) -> list[dict]:
    root = Path(config.root)
    sources = []
    if (root / "Locator.md").is_file():
        sources.append(("Locator.md", list(_refs_in_text((root / "Locator.md").read_text(encoding="utf-8")))))
    for meta in sorted((root / "Figures").glob("*/meta.yaml")) if (root / "Figures").is_dir() else []:
        sources.append((str(meta.relative_to(root)), list(_meta_refs(meta))))
    rows = []
    for file, refs in sources:
        seen = set()
        for kind, ref in refs:
            if (kind, ref) in seen:
                continue
            seen.add((kind, ref))
            row = {"file": file, "ref": ref, "kind": kind, "ok": None, "problem": None}
            if kind == "path":
                try:
                    path = resolve(config, ref)
                    row["ok"] = path.exists()
                    if not row["ok"]:
                        row["problem"] = f"does not exist on this machine: {path}"
                except PathError as exc:
                    row["ok"], row["problem"] = False, str(exc)
            elif online:
                row["ok"], row["problem"] = _check_url(ref, timeout, opener)
            rows.append(row)
    return rows
