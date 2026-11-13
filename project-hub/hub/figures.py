# project-hub/hub/figures.py
"""Figures registry: Figures/<id>/{figure.<ext>, caption.md, meta.yaml} and the generated Figures/Gallery.md.

A figure is produced by the project's own scripts somewhere under a root; the vault keeps a copy so it can
be read in Obsidian. `sync_figures` re-copies a figure whose source changed and marks it stale when its
data changed after it was made. caption.md is human-owned: written once at registration, never again.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from datetime import date
from pathlib import Path

import yaml

from .fsutil import atomic_write_text
from .paths import resolve

FIG_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FIG_FILE_RE = re.compile(r"figure(\.[a-z0-9]+)?")
FIG_STATUSES = ("draft", "final")
CAPTION_PLACEHOLDER = "<!-- caption: what the figure shows, findings first; edit freely, the tools never overwrite it -->"


class FigureError(ValueError):
    """A figure cannot be registered or synced as asked."""


def figures_dir(root) -> Path:
    return Path(root) / "Figures"


def _fig_dir(root, fig_id) -> Path:
    return figures_dir(root) / fig_id


def _sha_bytes(h, path: Path) -> None:
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    _sha_bytes(h, path)
    return h.hexdigest()[:16]


def _hash_data(config, specs) -> str | None:
    if not specs:
        return None
    h = hashlib.sha256()
    for spec in sorted(specs):
        path = resolve(config, spec)
        if not path.exists():
            raise FigureError(f"data path {spec} ({path}) does not exist")
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for f in files:
            h.update(str(f.relative_to(path) if path.is_dir() else f.name).encode())
            _sha_bytes(h, f)
    return h.hexdigest()[:16]


def _check_specs(source, data) -> tuple[str, list[str]]:
    if not isinstance(source, str) or not source.strip():
        raise FigureError(f"source must be a non-empty path string, got {source!r}")
    if data is None:
        data = []
    if isinstance(data, tuple):
        data = list(data)
    if not isinstance(data, list) or not all(isinstance(x, str) and x.strip() for x in data):
        raise FigureError(f"data must be a list of non-empty path strings, got {data!r}")
    return source, list(data)


def load_meta(root, fig_id) -> dict:
    p = _fig_dir(root, fig_id) / "meta.yaml"
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise FigureError(f"Figures/{fig_id}/meta.yaml cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise FigureError(f"Figures/{fig_id}/meta.yaml is missing or not a mapping")
    return data


def _save_meta(root, fig_id, meta: dict) -> None:
    atomic_write_text(_fig_dir(root, fig_id) / "meta.yaml", yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))


def _replace_old_figure(fig_dir: Path, old, new_name: str) -> None:
    """Remove the previous figure file, but only if it is a plain figure(.ext) inside fig_dir."""
    if not isinstance(old, str) or old == new_name or not FIG_FILE_RE.fullmatch(old):
        return
    path = fig_dir / old
    try:
        if path.resolve().parent == fig_dir.resolve():
            path.unlink(missing_ok=True)
    except (OSError, RuntimeError):
        return


def figure_ids(root) -> list[str]:
    d = figures_dir(root)
    return sorted(p.parent.name for p in d.glob("*/meta.yaml")) if d.is_dir() else []


def add_figure(config, fig_id, source, *, title="", script="", data=(), licenses="", caption="") -> Path:
    root = Path(config.root)
    if not isinstance(fig_id, str) or not FIG_ID_RE.match(fig_id):
        raise FigureError(f"figure id must be lowercase letters, digits and '-', got {fig_id!r}")
    d = _fig_dir(root, fig_id)
    if (d / "meta.yaml").exists():
        raise FigureError(f"figure {fig_id} is already registered (Figures/{fig_id})")
    source, data = _check_specs(source, data)
    src = resolve(config, source)
    if not src.exists():
        raise FigureError(f"source {source} ({src}) does not exist")
    if not src.is_file():
        raise FigureError(f"source {source} ({src}) is not a file")
    data_hash = _hash_data(config, data)
    created = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    fig_file = f"figure{src.suffix.lower()}"
    try:
        shutil.copyfile(src, d / fig_file)
        if not (d / "caption.md").exists():
            atomic_write_text(d / "caption.md", (caption.strip() or CAPTION_PLACEHOLDER) + "\n")
        _save_meta(root, fig_id, {
            "id": fig_id, "title": title, "source": source, "script": script, "data": data,
            "data_baseline": list(data), "status": "draft", "licenses": licenses, "figure_file": fig_file,
            "figure_hash": _hash_file(src), "data_hash": data_hash, "synced": None, "stale": False,
            "problem": None})
    except BaseException:
        if created:
            shutil.rmtree(d, ignore_errors=True)
        raise
    return d


def sync_figures(config, *, today: date) -> list[dict]:
    root = Path(config.root)
    rows = []
    for fig_id in figure_ids(root):
        try:
            meta = load_meta(root, fig_id)
        except FigureError as exc:
            rows.append({"id": fig_id, "copied": False, "stale": False, "problem": str(exc)})
            continue
        row = {"id": fig_id, "copied": False, "stale": False, "problem": None}
        try:
            source, data = _check_specs(meta.get("source"), meta.get("data"))
            baseline = _check_specs(source, data if meta.get("data_baseline") is None else meta["data_baseline"])[1]
            src = resolve(config, source)
            if not src.is_file():
                raise FigureError(f"source missing: {source} ({src})")
            data_hash = _hash_data(config, data)
            src_hash = _hash_file(src)
            if src_hash != meta.get("figure_hash"):
                fig_file = f"figure{src.suffix.lower()}"
                fig_dir = _fig_dir(root, fig_id)
                _replace_old_figure(fig_dir, meta.get("figure_file"), fig_file)
                shutil.copyfile(src, fig_dir / fig_file)
                meta.update(figure_file=fig_file, figure_hash=src_hash, data_hash=data_hash, synced=today)
                row["copied"] = True
            elif data != baseline:
                meta["data_hash"] = data_hash          # data list edited by hand: re-baseline
            else:
                row["stale"] = data_hash != meta.get("data_hash")
            meta["data_baseline"] = data
            meta["stale"], meta["problem"] = row["stale"], None
        except (FigureError, ValueError, OSError) as exc:
            row = {"id": fig_id, "copied": False, "stale": False, "problem": str(exc)}
            meta["stale"], meta["problem"] = False, str(exc)
        try:
            _save_meta(root, fig_id, meta)
        except OSError as exc:
            row = {"id": fig_id, "copied": False, "stale": False, "problem": str(exc)}
        rows.append(row)
    return rows


def stale_figures(root) -> list[str]:
    out = []
    for fig_id in figure_ids(root):
        try:
            meta = load_meta(root, fig_id)
        except FigureError as exc:
            out.append(f"figure {fig_id}: {exc}")
            continue
        if meta.get("problem"):
            out.append(f"figure {fig_id}: {meta['problem']}")
        elif meta.get("stale"):
            out.append(f"figure {fig_id}: its data changed after the figure was made — regenerate it, "
                       f"then run `hub figures sync`")
    return out


def render_gallery(root) -> str:
    out = ["# Figures", "*Generated by `hub figures gallery`; captions are edited in each figure's caption.md.*"]
    for fig_id in figure_ids(root):
        try:
            meta = load_meta(root, fig_id)
        except FigureError as exc:
            out += ["", f"## {fig_id}", f"**Problem:** {exc}"]
            continue
        title = meta.get("title") or fig_id
        out += ["", f"## {fig_id} — {title} ({meta.get('status') or 'draft'})",
                f"![[Figures/{fig_id}/{meta.get('figure_file')}]]", f"![[Figures/{fig_id}/caption]]"]
        if meta.get("licenses"):
            out.append(f"*Supports:* {meta['licenses']}")
        out.append(f"*Source:* `{meta.get('source')}`" + (f" · *script:* `{meta['script']}`" if meta.get("script") else ""))
        if meta.get("problem"):
            out.append(f"**Problem:** {meta['problem']}")
        elif meta.get("stale"):
            out.append("**Stale:** its data changed after the figure was made.")
    if len(out) == 2:
        out += ["", "No figures registered yet (`hub figures add`)."]
    return "\n".join(out) + "\n"


def write_gallery(root) -> Path:
    path = figures_dir(root) / "Gallery.md"
    atomic_write_text(path, render_gallery(root))
    return path
