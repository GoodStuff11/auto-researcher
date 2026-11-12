# project-hub/hub/cli.py
"""hub — command line for a project hub vault (status store, staged changes, agenda, git sync).

Run from anywhere inside a vault (the nearest parent directory containing project.yaml is used), or pass
--root. Writes need your people key: set `me: <key>` in hub.local.yaml (gitignored) or HUB_ME=<key>.
"Today" is the system date unless HUB_TODAY=YYYY-MM-DD is set (used for tests and back-dating).

Commands (each has its own --help):
  new {q,d,a} TITLE        create a question / decision / action; prints the new id
      --source S             (required) provenance: circleback:<id>, commit:<repo>@<sha>, clickup:<id>,
                             agenda:<series>/<date>, manual, …
      --owner KEY            people key from project.yaml (default: you)
      --series NAME          meeting series this item may appear on; repeatable (default: none = all)
      --priority {1,2,3}     1 = highest (default 2)
      --link REL:ID          REL ∈ answers | spawned_by | pivoted_to; repeatable
      --description TEXT     one-paragraph description
  entry ID TEXT --source S   append a dated history entry authored by you
  set ID KEY=VALUE …         set frontmatter fields (not id/type/links). Dates as YYYY-MM-DD; empty VALUE
                             clears. status must be valid for the type:
                               question: open|answered|dropped|pivoted
                               decision: proposed|made|rejected|deferred
                               action:   open|done|dropped
  link ID REL TARGET         add a link; REL ∈ answers | spawned_by | pivoted_to
  list [filters] [--json]    --type {question,decision,action} --status S --owner KEY|me --series NAME
                             --since YYYY-MM-DD (last activity on/after)
  show ID [--chain]          print one item; --chain also prints everything descending from it
  since DATE                 items with activity on/after DATE (same as list --since)
  log TEXT --source S        append to Status/Log/<YYYY-MM>.md
  validate                   check every rule; exit 1 and list violations if any
  snapshot                   regenerate Status/Snapshot.md
  pending list               show staged proposals
  pending apply PID …        apply proposals (P-xxxx) in order; each is removed once applied
  pending reject PID …       drop proposals, logging the rejection
  pending add --title T --ops-file F.yaml --source S --reason R
                             stage a proposal whose ops are read from a YAML list (see .agent_docs/pending-format.md)
  agenda scaffold SERIES --date YYYY-MM-DD [--force]
                             write Agenda/<series>.draft.yaml pre-filled from the store (summary/follow-ups
                             empty); refuses if the draft already exists unless --force (overwrites it)
  agenda build SERIES [--force]
                             render the draft into Agenda/<series>.xlsx; refuses while the previous meeting's
                             workbook is unread, unless --force (which discards its meeting edits)
  agenda read SERIES         apply the meeting's edits to the store, stage free-text items as
                             Status/Pending/<P-id>.md (one file per proposal),
                             write the meeting note, archive the workbook
  sync-state get SOURCE | set SOURCE VALUE
                             read/write ingest watermarks in Status/sync-state.yaml
  commit MESSAGE             git add -A, commit, pull --rebase, regenerate Snapshot.md, push (3 retries);
                             exit 3 on a genuine merge conflict (rebase is aborted, nothing lost)

Exit codes: 0 ok · 1 user error or validation violations · 2 bad arguments · 3 sync conflict.

Pending proposals live one per file in Status/Pending/<P-id>.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import yaml

from .agenda.draft import DraftError, draft_path, save_draft, scaffold
from .agenda.lifecycle import AgendaError, build_agenda, read_agenda
from .config import ConfigError, find_root, load_config
from .gitsync import SyncConflict, SyncError, commit
from .items import LINK_KEYS, render_item
from .pending import PendingError, add_proposal, apply_proposals, load_proposals, reject_proposals
from .snapshot import write_snapshot
from .store import Store, StoreError
from .sync_state import get_watermark, set_watermark
from .validate import validate

TYPE_ALIASES = {"q": "question", "d": "decision", "a": "action"}


class UserError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="vault root (default: nearest parent containing project.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    n = sub.add_parser("new", help="create an item")
    n.add_argument("type", choices=sorted(TYPE_ALIASES))
    n.add_argument("title")
    n.add_argument("--source", required=True)
    n.add_argument("--owner")
    n.add_argument("--series", action="append", default=[])
    n.add_argument("--priority", type=int, choices=(1, 2, 3), default=2)
    n.add_argument("--link", action="append", default=[], metavar="REL:ID")
    n.add_argument("--description", default="")

    e = sub.add_parser("entry", help="append a history entry")
    e.add_argument("id")
    e.add_argument("text")
    e.add_argument("--source", required=True)

    s = sub.add_parser("set", help="set frontmatter fields")
    s.add_argument("id")
    s.add_argument("pairs", nargs="+", metavar="KEY=VALUE")

    l = sub.add_parser("link", help="add a link")
    l.add_argument("id")
    l.add_argument("rel", choices=LINK_KEYS)
    l.add_argument("target")

    ls = sub.add_parser("list", help="query items")
    ls.add_argument("--type", choices=("question", "decision", "action"))
    ls.add_argument("--status")
    ls.add_argument("--owner")
    ls.add_argument("--series")
    ls.add_argument("--since", type=date.fromisoformat)
    ls.add_argument("--json", action="store_true")

    sh = sub.add_parser("show", help="print an item")
    sh.add_argument("id")
    sh.add_argument("--chain", action="store_true")

    si = sub.add_parser("since", help="items with activity since a date")
    si.add_argument("date", type=date.fromisoformat)

    lg = sub.add_parser("log", help="append to the monthly log")
    lg.add_argument("text")
    lg.add_argument("--source", required=True)

    sub.add_parser("validate", help="check store rules")
    sub.add_parser("snapshot", help="regenerate Status/Snapshot.md")

    pe = sub.add_parser("pending", help="staged proposals")
    pes = pe.add_subparsers(dest="pcmd", required=True)
    pes.add_parser("list")
    pa = pes.add_parser("apply")
    pa.add_argument("pids", nargs="+")
    pr = pes.add_parser("reject")
    pr.add_argument("pids", nargs="+")
    pad = pes.add_parser("add")
    pad.add_argument("--title", required=True)
    pad.add_argument("--ops-file", type=Path, required=True)
    pad.add_argument("--source", required=True)
    pad.add_argument("--reason", required=True)

    ag = sub.add_parser("agenda", help="meeting agenda workbook")
    ags = ag.add_subparsers(dest="acmd", required=True)
    asc = ags.add_parser("scaffold")
    asc.add_argument("series")
    asc.add_argument("--date", type=date.fromisoformat, required=True)
    asc.add_argument("--force", action="store_true")
    ab = ags.add_parser("build")
    ab.add_argument("series")
    ab.add_argument("--force", action="store_true")
    ar = ags.add_parser("read")
    ar.add_argument("series")

    ss = sub.add_parser("sync-state", help="ingest watermarks")
    sss = ss.add_subparsers(dest="scmd", required=True)
    sg = sss.add_parser("get")
    sg.add_argument("source")
    sst = sss.add_parser("set")
    sst.add_argument("source")
    sst.add_argument("value")

    c = sub.add_parser("commit", help="commit and push with rebase/retry")
    c.add_argument("message")
    return p


def _today() -> date:
    return date.fromisoformat(os.environ["HUB_TODAY"]) if os.environ.get("HUB_TODAY") else date.today()


def _me(cfg) -> str:
    if not cfg.me:
        raise UserError("writes need your people key: add `me: <key>` to hub.local.yaml or set HUB_ME")
    return cfg.me


def _row(it) -> dict:
    return {"id": it.id, "type": it.type, "status": it.status, "title": it.title, "owner": it.owner,
            "priority": it.meta.get("priority"), "last_activity": it.last_activity().isoformat()}


def _print_items(items, as_json):
    if as_json:
        print(json.dumps([_row(i) for i in items], indent=2))
        return
    for i in items:
        print(f"{i.id}  {i.status:<9} {i.owner or '-':<10} {i.title}")


def _dispatch(args, cfg, store) -> int:
    today = _today()
    cmd = args.cmd
    if cmd == "new":
        links: dict[str, list[str]] = {}
        for spec in args.link:
            rel, _, target = spec.partition(":")
            links.setdefault(rel, []).append(target)
        item = store.create(TYPE_ALIASES[args.type], args.title, owner=args.owner or _me(cfg), author=_me(cfg),
                            day=today, source=args.source, series=args.series, priority=args.priority,
                            links=links, description=args.description)
        print(item.id)
    elif cmd == "entry":
        store.entry(args.id, args.text, author=_me(cfg), source=args.source, day=today)
    elif cmd == "set":
        fields = {}
        for pair in args.pairs:
            k, sep, v = pair.partition("=")
            if not sep:
                raise UserError(f"expected KEY=VALUE, got '{pair}'")
            fields[k] = v
        store.set_fields(args.id, fields)
    elif cmd == "link":
        store.link(args.id, args.rel, args.target)
    elif cmd == "list":
        owner = _me(cfg) if args.owner == "me" else args.owner
        _print_items(store.query(type=args.type, status=args.status, owner=owner, series=args.series,
                                 since=args.since), args.json)
    elif cmd == "show":
        print(render_item(store.get(args.id)), end="")
        if args.chain:
            seen = {args.id}

            def walk(item_id, depth):
                for ch in store.children(item_id):
                    if ch.id in seen:
                        continue
                    seen.add(ch.id)
                    print(f"{'  ' * depth}{ch.id}  [{ch.type}/{ch.status}] {ch.title}")
                    walk(ch.id, depth + 1)

            print("## Chain")
            walk(args.id, 1)
    elif cmd == "since":
        _print_items(store.query(since=args.date), False)
    elif cmd == "log":
        store.log(args.text, author=_me(cfg), source=args.source, day=today)
    elif cmd == "validate":
        vs = validate(store, cfg)
        for v in vs:
            print(f"{v.item_id}  {v.rule}: {v.message}")
        print(f"{len(vs)} violation(s)")
        return 1 if vs else 0
    elif cmd == "snapshot":
        print(write_snapshot(store, cfg, today))
    elif cmd == "pending":
        if args.pcmd == "list":
            props = load_proposals(cfg.root)
            if not props:
                print("No pending proposals.")
            for pr in props:
                print(f"{pr.pid}  {pr.title}  (by {pr.body.get('proposer')}, {pr.body.get('date')}): "
                      f"{pr.body.get('reason')}")
        elif args.pcmd == "apply":
            for i in apply_proposals(cfg.root, store, args.pids, author=_me(cfg), day=today):
                print(f"created {i}")
        elif args.pcmd == "reject":
            reject_proposals(cfg.root, store, args.pids, author=_me(cfg), day=today)
        else:
            ops = yaml.safe_load(args.ops_file.read_text()) or []
            print(add_proposal(cfg.root, args.title, ops, proposer=_me(cfg), source=args.source,
                               reason=args.reason, day=today))
    elif cmd == "agenda":
        if args.acmd == "scaffold":
            if args.series not in cfg.series:
                raise UserError(f"unknown series '{args.series}' (project.yaml series: {', '.join(cfg.series)})")
            if draft_path(cfg.root, args.series).exists() and not args.force:
                raise UserError(f"draft Agenda/{args.series}.draft.yaml already exists; pass --force to overwrite")
            print(save_draft(cfg.root, scaffold(store, cfg.root, args.series, args.date)))
        elif args.acmd == "build":
            print(build_agenda(cfg.root, args.series, force=args.force))
        else:
            res = read_agenda(store, cfg, args.series, author=_me(cfg), today=today)
            print(f"applied {len(res['applied'])}, staged {len(res['staged'])} "
                  f"({' '.join(res['staged'])}), not reached {len(res['not_reached'])}")
            print(f"meeting note: {res['meeting_note']}\narchive: {res['archive']}")
    elif cmd == "sync-state":
        if args.scmd == "get":
            print(get_watermark(cfg.root, args.source) or "")
        else:
            set_watermark(cfg.root, args.source, args.value)
    elif cmd == "commit":
        print(commit(cfg.root, args.message, regenerate=lambda: write_snapshot(store, cfg, today)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.root or find_root(Path.cwd())
        cfg = load_config(root)
        return _dispatch(args, cfg, Store(root))
    except SyncConflict as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyError as exc:
        print(f"error: no such item {exc}", file=sys.stderr)
        return 1
    except (UserError, ConfigError, StoreError, PendingError, DraftError, AgendaError, SyncError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
