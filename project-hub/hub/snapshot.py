# project-hub/hub/snapshot.py
"""Generate Status/Snapshot.md — a one-page, never hand-edited view of where the project stands."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .config import HubConfig
from .figures import stale_figures
from .fsutil import atomic_write_text
from .store import Store
from .sync_state import load_state
from .validate import validate


def last_meeting(root, series: str, before: date | None = None) -> date | None:
    d = Path(root) / "Status" / "Meetings" / series
    if not d.is_dir():
        return None
    days = []
    for p in d.glob("*.md"):
        try:
            day = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if before is None or day < before:
            days.append(day)
    return max(days) if days else None


def _raised(item, today: date) -> date:
    r = item.meta.get("raised")
    return r if isinstance(r, date) else today


def _or_dash(v) -> str:
    return "—" if v is None else str(v)


def render_snapshot(store: Store, config: HubConfig, today: date) -> str:
    state = load_state(config.root)
    items = store.all()
    out = [f"# Snapshot — {config.name}",
           f"*Generated {today.isoformat()} by `hub snapshot`. Do not edit; regenerate.*", ""]
    out += ["## Meetings", "| Series | Last meeting | Next meeting |", "|---|---|---|"]
    for s in config.series:
        last = last_meeting(config.root, s)
        nxt = state["next_meeting"].get(s)
        out.append(f"| {s} | {last.isoformat() if last else '—'} | {nxt.isoformat() if nxt else '—'} |")
    qs = sorted((i for i in items if i.type == "question" and i.status == "open"),
                key=lambda i: (i.meta.get("priority") or 9, _raised(i, today)))
    out += ["", f"## Open questions ({len(qs)})", "| ID | Title | Owner | Age (d) | Priority |", "|---|---|---|---|---|"]
    for q in qs:
        out.append(f"| [[{q.id}]] | {q.title} | {q.owner or '—'} | {(today - _raised(q, today)).days} | "
                   f"{_or_dash(q.meta.get('priority'))} |")
    for status, heading in (("proposed", "Proposed decisions"), ("deferred", "Deferred decisions")):
        ds = [i for i in items if i.type == "decision" and i.status == status]
        out += ["", f"## {heading} ({len(ds)})", "| ID | Title | Answers | Owner |", "|---|---|---|---|"]
        for d in ds:
            out.append(f"| [[{d.id}]] | {d.title} | {', '.join(d.links('answers')) or '—'} | {d.owner or '—'} |")
    acts = [i for i in items if i.type == "action" and i.status == "open"]
    out += ["", f"## Open actions ({len(acts)})"]
    for owner in sorted({a.owner or "unassigned" for a in acts}):
        out.append(f"### {owner}")
        for a in sorted((a for a in acts if (a.owner or "unassigned") == owner),
                        key=lambda a: (a.meta.get("due") or date.max)):
            due = f" (due {a.meta['due'].isoformat()})" if a.meta.get("due") else ""
            out.append(f"- [[{a.id}]] {a.title}{due}")
    out += ["", "## Flags"]
    flags = [f"- {f}" for f in state["flags"]]
    flags += [f"- cannot load {p.relative_to(config.root) if p.is_relative_to(config.root) else p}: {err}"
              for p, err in store.load_errors]
    flags += [f"- {f}" for f in stale_figures(config.root)]
    out += flags or ["None."]
    n = len(validate(store, config))
    out += ["", "## Validation", f"{n} issues." + (" Run `hub validate`." if n else "")]
    return "\n".join(out) + "\n"


def write_snapshot(store: Store, config: HubConfig, today: date) -> Path:
    path = Path(config.root) / "Status" / "Snapshot.md"
    atomic_write_text(path, render_snapshot(store, config, today))
    return path
