# project-hub/hub/links.py
"""`hub check-links`: every $ROOT/path and URL in Locator.md and figure meta.yaml files.

Paths are checked on this machine (roots come from hub.local.yaml). URLs are only listed unless
online=True, which sends a HEAD request (falling back to GET) with a short timeout. The check never
aborts: a bad file or a failing request becomes a row with ok False and a problem.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from .paths import ROOT_REF_RE, PathError, resolve

URL_RE = re.compile(r"https?://[^\s\]>`'\"|]+")
SPACED_SPAN_RE = re.compile(r"`(\$[A-Za-z_][A-Za-z0-9_]*/[^`\n]*)`")


def _clean_url(raw: str) -> str:
    """Drop trailing sentence punctuation, and a closing ')' that has no matching '('."""
    url = raw
    while True:
        trimmed = url.rstrip(".,;:")
        if trimmed.endswith(")") and trimmed.count(")") > trimmed.count("("):
            trimmed = trimmed[:-1]
        if trimmed == url:
            return url
        url = trimmed


def _refs_in_text(text: str):
    # Backtick spans that start with $NAME/ are one ref, spaces included.
    for m in SPACED_SPAN_RE.finditer(text):
        ref = m[1].strip()
        if not ref.endswith("$"):  # `$x/y$` is maths
            yield "path", ref
    rest = SPACED_SPAN_RE.sub(lambda m: " " * len(m[0]), text)
    for m in ROOT_REF_RE.finditer(rest):
        # A match that runs into a closing `$` is LaTeX maths, not a path.
        if m[2].endswith("$") or rest[m.end():m.end() + 1] == "$":
            continue
        yield "path", f"${m[1]}/{m[2]}".rstrip(".,;:")
    for m in URL_RE.finditer(text):
        yield "url", _clean_url(m[0])


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
            s = spec.strip()
            if URL_RE.fullmatch(s):
                yield "url", _clean_url(s)
            else:
                yield "path", s


def _locator_refs(path: Path):
    """Return (refs, None), or (None, problem) when Locator.md cannot be read."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"unreadable: {exc}"
    return list(_refs_in_text(text)), None


def _check_url(url, timeout, opener):
    opener = opener or urllib.request.urlopen
    problem = "unreachable"
    for method in ("HEAD", "GET"):
        try:
            with opener(urllib.request.Request(url, method=method), timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
            if status < 400:
                return True, None
            if method == "GET" or status not in (403, 405):
                return False, f"HTTP {status}"
            # HEAD refused with 403/405: fall through to GET.
        except urllib.error.HTTPError as exc:
            if method == "GET" or exc.code not in (403, 405):
                return False, f"HTTP {exc.code}"
        except Exception as exc:  # transport failure: retry with GET, then give up
            problem = f"unreachable: {type(exc).__name__}: {exc}"
    return False, problem


def check_links(config, *, online=False, timeout=5.0, opener=None) -> list[dict]:
    root = Path(config.root)
    rows = []
    sources = []
    locator = root / "Locator.md"
    if locator.exists():
        refs, problem = _locator_refs(locator)
        if problem:
            rows.append({"file": "Locator.md", "ref": "Locator.md", "kind": "path", "ok": False, "problem": problem})
        else:
            sources.append(("Locator.md", refs))
    for meta in sorted((root / "Figures").glob("*/meta.yaml")) if (root / "Figures").is_dir() else []:
        sources.append((str(meta.relative_to(root)), list(_meta_refs(meta))))
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
                except (OSError, ValueError, RuntimeError) as exc:
                    row["ok"], row["problem"] = False, f"cannot check: {exc}"
            elif online:
                row["ok"], row["problem"] = _check_url(ref, timeout, opener)
            rows.append(row)
    return rows
