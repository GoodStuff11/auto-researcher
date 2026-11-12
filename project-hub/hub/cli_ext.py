"""Plan 2 commands (ingest helpers and meeting/source bookkeeping), hooked into hub/cli.py."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

from .batch import run_batch
from .clickup import apply_clickup, load_tasks, plan_clickup
from .ingest_git import ingest_git
from .sources import source_seen
from .sync_state import load_state, save_state


def _positive(text):
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _parse_error(path, exc):
    first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
    return ValueError(f"cannot parse {path}: {first}")


def _sp(sub, name, help, description):
    return sub.add_parser(name, help=help, description=description,
                          formatter_class=argparse.RawDescriptionHelpFormatter)


def add_ext_parsers(sub) -> None:
    s = _sp(sub, "seen", "check whether a source id was already recorded",
            "Print `new` if SOURCE appears nowhere in the vault, else `seen` followed by where it appears: item "
            "ids that list it in `sources`, log:<YYYY-MM> for monthly logs citing it, and pending proposal ids "
            "citing it. An unreadable:<path> entry means the answer may be incomplete (that file could not be "
            "read; fix it). Exits 0 unless a log file cannot be read. Use before writing anything derived from a source.")
    s.add_argument("source", help="source id exactly as written in entries, e.g. circleback:<meeting id>, "
                                  "clickup:<task id>, commit:<ROOT>@<7-char sha>")

    ing = _sp(sub, "ingest", "read a source into the vault",
              "Ingest helpers used by the hub-status skill: `git` lists new commits of a repository, `batch` "
              "applies an agent-written ingest file.")
    isub = ing.add_subparsers(dest="icmd", required=True, metavar="SUBCOMMAND")
    g = _sp(isub, "git", "list commits of a configured repository since the last ingest (JSON)",
            "Print JSON {root, path, branch, watermark_before, commits: [{sha, short, date, author, subject, "
            "files}] oldest first, warning, advanced}. ROOT is a key of `roots` (local path from "
            "hub.local.yaml); the branch comes from sources.repos[]/sources.overleaf `branch` (default HEAD). "
            "Without a usable watermark git.<ROOT>, the last 30 days are listed. A warning is also printed "
            "on stderr.")
    g.add_argument("root_name", metavar="ROOT", help="a key of `roots` in project.yaml/hub.local.yaml")
    g.add_argument("--since", type=date.fromisoformat, metavar="YYYY-MM-DD",
                   help="list commits since this date instead of since the watermark")
    g.add_argument("--advance", action="store_true",
                   help="after listing, move the watermark git.<ROOT> to the newest listed commit; use only "
                        "once the commits have been recorded (e.g. via `hub ingest batch`)")
    g.add_argument("--limit", type=_positive, default=500, metavar="N",
                   help="at most N oldest unlisted commits per run; a warning says when more remain "
                        "(default 500)")
    b = _sp(isub, "batch", "apply an ingest batch file: safe changes directly, judgement calls as proposals",
            "Apply a YAML batch file with keys `safe` (entry/log/new ops applied directly), `proposals` "
            "(staged in Status/Pending/), `flags` (added to Status/sync-state/flags.md) and `watermark` "
            "({name, value}, set last). The whole file is checked first; on any problem nothing is written "
            "(exit 1). Re-running the same file is a no-op. Format: .agent_docs/ingest.md. Prints "
            "`applied N, skipped N, staged N (P-ids), skipped proposals N, flags added N`.")
    b.add_argument("file", type=Path, help="path to the batch YAML file")

    c = _sp(sub, "clickup", "reconcile a ClickUp list export with the hub's actions",
            "ClickUp owns task status, assignee and due date. `plan` prints what would change (JSON keys "
            "set, proposals, flags, push); `apply` applies it: mirrors fields onto actions with clickup_id, "
            "stages tasks created in ClickUp as proposals, replaces `clickup: ` flags, and prints JSON "
            "{updated, staged, flags, push}. `push` lists open owned actions with no ClickUp task yet.")
    csub = c.add_subparsers(dest="ccmd", required=True, metavar="SUBCOMMAND")
    for name, helptext in (("plan", "show the reconciliation without writing"),
                           ("apply", "apply the reconciliation")):
        cp = _sp(csub, name, helptext, f"{helptext.capitalize()}. Input: --tasks, the list export as JSON "
                                       "[{id, name, status, assignees, due}] (see .agent_docs/ingest.md).")
        cp.add_argument("--tasks", type=Path, required=True, help="path to the task export (JSON or YAML list)")

    m = _sp(sub, "meeting", "meeting bookkeeping (next date, what a meeting decided)",
            "`next` records the next meeting date of a series (shown in the Snapshot); `decided` lists the "
            "decisions recorded for one meeting (by its agenda read-back), for cross-checking a transcript.")
    msub = m.add_subparsers(dest="mcmd", required=True, metavar="SUBCOMMAND")
    mn = _sp(msub, "next", "record the next meeting date of a series",
             "Write Status/sync-state/next_meeting.<SERIES>.yaml.")
    mn.add_argument("series", help="a key of `series` in project.yaml")
    mn.add_argument("date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="date of the next meeting")
    md = _sp(msub, "decided", "list decisions recorded for one meeting",
             "List decisions whose `meeting` is SERIES/DATE or that carry an entry with source "
             "agenda:SERIES/DATE, with their status, as-decided text and spawned actions.")
    md.add_argument("series", help="a key of `series` in project.yaml")
    md.add_argument("date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="date of the meeting")
    md.add_argument("--json", action="store_true",
                    help="print a JSON list of {id, title, status, as_decided, actions} instead of text")

    so = _sp(sub, "sources", "list the configured ingest sources and their watermarks",
             "One row per source configured under `sources` in project.yaml: clickup, circleback, "
             "git.<ROOT> per repository and for overleaf, calendar, gmail and slack (when enabled), with its "
             "watermark (empty = never ingested).")
    so.add_argument("--json", action="store_true",
                    help="print a JSON list of {source, detail, watermark} instead of a table")


def _source_rows(cfg) -> list[dict]:
    src = cfg.sources or {}
    wm = load_state(cfg.root)["watermarks"]
    rows = []
    if src.get("clickup"):
        rows.append({"source": "clickup", "detail": f"list {src['clickup'].get('list_id')}", "watermark": wm.get("clickup")})
    if src.get("circleback"):
        cb = src["circleback"]
        rows.append({"source": "circleback", "detail": f"folder {cb.get('folder')} terms {cb.get('terms') or []}",
                     "watermark": wm.get("circleback")})
    repos = [("repo", r) for r in src.get("repos") or []]
    if isinstance(src.get("overleaf"), dict):
        repos.append(("overleaf", src["overleaf"]))
    for kind, r in repos:
        name = f"git.{r.get('root')}"
        rows.append({"source": name, "detail": f"{kind} {r.get('root')} ({r.get('branch') or 'HEAD'})",
                     "watermark": wm.get(name)})
    if "calendar" in src:
        rows.append({"source": "calendar", "detail": "next meeting per series", "watermark": None})
    for key in ("gmail", "slack"):
        if (src.get(key) or {}).get("enabled"):
            rows.append({"source": key, "detail": "opt-in", "watermark": wm.get(key)})
    return rows


def _decided(store, series: str, day: date) -> list[dict]:
    tag = f"{series}/{day.isoformat()}"
    actions = store.query(type="action")
    rows = []
    for d in store.query(type="decision"):
        if d.meta.get("meeting") == tag or any(e.source == f"agenda:{tag}" for e in d.entries):
            rows.append({"id": d.id, "title": d.title, "status": d.status,
                         "as_decided": d.meta.get("as_decided") or "",
                         "actions": [a.id for a in actions if d.id in a.links("spawned_by")]})
    return rows


def dispatch_ext(args, cfg, store, *, today, me) -> int | None:
    cmd = args.cmd
    if cmd == "seen":
        where = source_seen(cfg.root, store, args.source)
        print("seen " + " ".join(where) if where else "new")
    elif cmd == "ingest" and args.icmd == "git":
        res = ingest_git(cfg, args.root_name, today=today, since=args.since, advance=args.advance,
                         limit=args.limit)
        print(json.dumps(res, indent=2, default=str))
        if res["warning"]:
            print(f"warning: {res['warning']}", file=sys.stderr)
    elif cmd == "ingest":
        author = me()
        try:
            batch = yaml.safe_load(args.file.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise _parse_error(args.file, exc) from exc
        rep = run_batch(cfg.root, store, batch, author=author, day=today, people=cfg.people)
        print(f"applied {rep['applied']}, skipped {rep['skipped']}, staged {len(rep['staged'])} "
              f"({' '.join(rep['staged'])}), skipped proposals {rep['skipped_proposals']}, "
              f"flags added {rep['flags_added']}" + (f", watermark {rep['watermark']}" if rep["watermark"] else ""))
    elif cmd == "clickup":
        author = me() if args.ccmd == "apply" else None
        try:
            tasks = load_tasks(args.tasks)
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise _parse_error(args.tasks, exc) from exc
        plan = plan_clickup(store, cfg, tasks, today=today)
        if args.ccmd == "plan":
            print(json.dumps(plan, indent=2, default=str))
        else:
            print(json.dumps(apply_clickup(cfg.root, store, plan, author=author, day=today), indent=2, default=str))
    elif cmd == "meeting":
        if args.series not in cfg.series:
            raise ValueError(f"unknown series '{args.series}' (project.yaml series: {', '.join(cfg.series)})")
        if args.mcmd == "next":
            state = load_state(cfg.root)
            state["next_meeting"][args.series] = args.date
            save_state(cfg.root, state)
            print(f"next {args.series} meeting: {args.date.isoformat()}")
        else:
            rows = _decided(store, args.series, args.date)
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
            for r in [] if args.json else rows:
                print(f"{r['id']}  {r['status']:<9} {r['title']} — {r['as_decided'] or 'as proposed'}"
                      f"  actions: {' '.join(r['actions']) or 'none'}")
    elif cmd == "sources":
        rows = _source_rows(cfg)
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        for r in [] if args.json else rows:
            print(f"{r['source']:<20} {r['watermark'] or '—':<28} {r['detail']}")
    else:
        return None
    return 0
