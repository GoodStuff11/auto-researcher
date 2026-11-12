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
  since DATE [--series NAME] items with activity on/after DATE (same as list --since [--series])
  log TEXT --source S        append to Status/Log/<YYYY-MM>.md
  validate                   check every rule; exit 1 and list violations if any
  snapshot                   regenerate Status/Snapshot.md
  pending list               show staged proposals; files that cannot be read print as UNREADABLE <path>
  pending apply PID …        apply proposals (P-xxxx) in order, each all-or-nothing; prints `applied <P-id>`
                             then `created <id>` per new item as each completes. Pulls first (best effort)
  pending reject PID …       drop proposals, logging the rejection
  pending add --title T --ops-file F.yaml --source S --reason R
                             stage a proposal whose ops are read from a YAML list (see .agent_docs/pending-format.md)
  agenda scaffold SERIES --date YYYY-MM-DD [--force]
                             write Agenda/<series>.draft.yaml pre-filled from the store (summary/follow-ups
                             empty); refuses if the draft already exists unless --force (overwrites it)
  agenda build SERIES [--force]
                             render the draft into Agenda/<series>.xlsx; refuses while the previous meeting's
                             workbook is unread or the live workbook already holds meeting input, unless
                             --force (which discards its meeting edits)
  agenda read SERIES         pull (best effort), apply the meeting's edits to the store, stage free-text items as
                             Status/Pending/<P-id>.md (one file per proposal),
                             write the meeting note, archive the workbook
  sync-state get SOURCE | set SOURCE VALUE
                             read/write ingest watermarks (Status/sync-state/watermark.<SOURCE>.yaml)
  seen SOURCE                print `new`, or `seen` and where SOURCE (circleback:<id>, clickup:<id>, …) is recorded
  ingest git ROOT [--since YYYY-MM-DD] [--advance] [--limit N]
                             JSON list of commits of a configured repository since watermark git.<ROOT>
  ingest batch FILE          apply an agent-written ingest file (safe ops directly, proposals staged; idempotent)
  clickup plan|apply --tasks FILE
                             reconcile a ClickUp list export: mirror status/assignee/due, stage new tasks, flags
  meeting next SERIES YYYY-MM-DD | meeting decided SERIES YYYY-MM-DD [--json]
                             record the next meeting date; list what a meeting decided
  sources [--json]           configured ingest sources and their watermarks
  commit MESSAGE             git add -A, commit, pull --rebase, regenerate Snapshot.md, push (3 retries);
                             item files merge field by field (hub-item driver); exit 3 only when both sides
                             changed one field differently (rebase is aborted, nothing lost)

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
from .cli_ext import add_ext_parsers, dispatch_ext
from .config import ConfigError, find_root, load_config
from .gitsync import SyncConflict, SyncError, commit, pull_if_possible
from .items import LINK_KEYS, ItemFormatError, render_item
from .pending import (PendingError, add_proposal, apply_proposals, load_proposals, load_unreadable,
                      reject_proposals)
from .snapshot import write_snapshot
from .store import Store, StoreError, UnknownItem
from .sync_state import get_watermark, set_watermark
from .validate import validate

TYPE_ALIASES = {"q": "question", "d": "decision", "a": "action"}


class UserError(Exception):
    pass


def _sp(sub, name, help, description):
    return sub.add_parser(name, help=help, description=description, formatter_class=argparse.RawDescriptionHelpFormatter)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="vault root directory (the one containing project.yaml). "
                   "Default: the nearest parent of the current directory that contains project.yaml.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    n = _sp(sub, "new", "create a question, decision or action; prints its id",
            "Create a new item in Status/Items and print its id. Needs `me` (hub.local.yaml or HUB_ME) "
            "as the entry author.")
    n.add_argument("type", choices=sorted(TYPE_ALIASES),
                   help="item type: q = question, d = decision, a = action")
    n.add_argument("title", help="one-line title of the item")
    n.add_argument("--source", required=True,
                   help="provenance of the item, e.g. circleback:<id>, commit:<repo>@<sha>, clickup:<id>, "
                        "agenda:<series>/<date>, or manual")
    n.add_argument("--owner", help="people key from project.yaml who owns the item (default: you)")
    n.add_argument("--series", action="append", default=[],
                   help="meeting series (a key under `series` in project.yaml) the item may appear on; "
                        "repeatable. Default: none given, meaning the item may appear on every series")
    n.add_argument("--priority", type=int, choices=(1, 2, 3), default=2,
                   help="1 = highest, 2 = normal (default), 3 = lowest")
    n.add_argument("--link", action="append", default=[], metavar="REL:ID",
                   help="relation to an existing item; repeatable. REL is answers (this decision answers that "
                        "question), spawned_by (this item came out of that item) or pivoted_to (this item was "
                        "replaced by that item). Example: answers:Q-20261112-abc")
    n.add_argument("--description", default="", help="one-paragraph description (default: empty)")

    e = _sp(sub, "entry", "append a dated history entry to an item",
            "Append a dated history entry, authored by you, to an item's history.")
    e.add_argument("id", help="id of the item (Q-/D-/A-YYYYMMDD-xxx)")
    e.add_argument("text", help="the entry text")
    e.add_argument("--source", required=True, help="provenance of the entry (same forms as `new --source`)")

    s = _sp(sub, "set", "set frontmatter fields of an item",
            "Set frontmatter fields of an item. Not allowed for id, type or links (use `link` for links). "
            "Dates are YYYY-MM-DD; an empty value (KEY=) clears the field. Valid status per type:\n"
            "  question: open | answered | dropped | pivoted\n"
            "  decision: proposed | made | rejected | deferred\n"
            "  action:   open | done | dropped")
    s.add_argument("id", help="id of the item")
    s.add_argument("pairs", nargs="+", metavar="KEY=VALUE",
                   help="one or more assignments such as status=answered owner=bob closed=2026-11-12; "
                        "an empty VALUE clears the field")

    l = _sp(sub, "link", "add a link from one item to another",
            "Add a link of kind REL from item ID to TARGET.")
    l.add_argument("id", help="id of the item the link is added to")
    l.add_argument("rel", choices=LINK_KEYS,
                   help="answers = ID is a decision answering the question TARGET; spawned_by = ID came out of "
                        "TARGET; pivoted_to = ID was replaced by TARGET")
    l.add_argument("target", help="id of the item linked to")

    ls = _sp(sub, "list", "query items",
             "List items matching all given filters, one per line (id, status, owner, title), or as JSON.")
    ls.add_argument("--type", choices=("question", "decision", "action"), help="only items of this type")
    ls.add_argument("--status",
                    help="only items with this status. question: open|answered|dropped|pivoted; "
                         "decision: proposed|made|rejected|deferred; action: open|done|dropped")
    ls.add_argument("--owner", help="only items owned by this people key; `me` means your key from hub.local.yaml "
                                    "(or HUB_ME)")
    ls.add_argument("--series", help="only items that may appear on this meeting series")
    ls.add_argument("--since", type=date.fromisoformat, metavar="YYYY-MM-DD",
                    help="only items whose last activity is on or after this date")
    ls.add_argument("--json", action="store_true",
                    help="print a JSON list of objects with keys id, type, status, title, owner, priority "
                         "(1-3 or null) and last_activity (YYYY-MM-DD) instead of text lines")

    sh = _sp(sub, "show", "print one item",
             "Print an item as its Markdown file, optionally followed by everything descending from it.")
    sh.add_argument("id", help="id of the item")
    sh.add_argument("--chain", action="store_true",
                    help="also print an indented tree of items that answer, were spawned by, or were pivoted "
                         "from this item (recursively)")

    si = _sp(sub, "since", "items with activity since a date",
             "List items whose last activity is on or after DATE (same as `list --since DATE`).")
    si.add_argument("date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="earliest activity date to include")
    si.add_argument("--series", help="only items that may appear on this meeting series (same as `list --series`)")

    lg = _sp(sub, "log", "append to the monthly log",
             "Append a line to Status/Log/<YYYY-MM>.md for the current month, authored by you.")
    lg.add_argument("text", help="the log text")
    lg.add_argument("--source", required=True, help="provenance of the log line (same forms as `new --source`)")

    _sp(sub, "validate", "check every store rule",
        "Check every rule over all items. Prints one line per violation and a count; exit 1 if any, else 0.")
    _sp(sub, "snapshot", "regenerate Status/Snapshot.md",
        "Regenerate Status/Snapshot.md from the store and print its path.")

    pe = _sp(sub, "pending", "staged proposals (Status/Pending/<P-id>.md)",
             "Staged proposals, one file per proposal at Status/Pending/<P-id>.md. Subcommands: list, apply, "
             "reject, add.")
    pes = pe.add_subparsers(dest="pcmd", required=True, metavar="SUBCOMMAND")
    _sp(pes, "list", "show staged proposals",
        "Show every staged proposal (id, title, proposer, date, reason), or `No pending proposals.` Proposal "
        "files that cannot be parsed are listed as `UNREADABLE <path>`; fix or delete them by hand.")
    pa = _sp(pes, "apply", "apply proposals in order",
             "First runs a best-effort `git pull --rebase` (a failure is printed as a warning). Then applies "
             "proposals in the order given. Each proposal is all-or-nothing: its ops are checked first and, if "
             "any op fails, every file it touched is restored and it stays pending. As each completes it is "
             "logged and removed, and `applied <P-id>` is printed followed by `created <id>` per new item.")
    pa.add_argument("pids", nargs="+", metavar="PID", help="one or more proposal ids (P-xxxx) to apply")
    pr = _sp(pes, "reject", "drop proposals",
             "Drop proposals without applying them, recording the rejection in the log.")
    pr.add_argument("pids", nargs="+", metavar="PID", help="one or more proposal ids (P-xxxx) to reject")
    pad = _sp(pes, "add", "stage a new proposal",
              "Stage a proposal as Status/Pending/<P-id>.md and print its id. Format of the ops file: "
              ".agent_docs/pending-format.md.")
    pad.add_argument("--title", required=True, help="short title of the proposal")
    pad.add_argument("--ops-file", type=Path, required=True, metavar="F.yaml",
                     help="YAML file holding the list of operations the proposal performs when applied")
    pad.add_argument("--source", required=True, help="provenance of the proposal (same forms as `new --source`)")
    pad.add_argument("--reason", required=True, help="why this change is proposed")

    ag = _sp(sub, "agenda", "meeting agenda workbook",
             "Meeting agenda lifecycle for a series: scaffold a draft, build the workbook, read it back after "
             "the meeting. Subcommands: scaffold, build, read.")
    ags = ag.add_subparsers(dest="acmd", required=True, metavar="SUBCOMMAND")
    asc = _sp(ags, "scaffold", "write a draft agenda from the store",
              "Write Agenda/<series>.draft.yaml pre-filled from the store (summary and follow-ups left empty) "
              "and print its path. Refuses if the draft already exists unless --force.")
    asc.add_argument("series", help="meeting series key from project.yaml")
    asc.add_argument("--date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD",
                     help="date of the meeting")
    asc.add_argument("--force", action="store_true",
                     help="overwrite an existing draft (losing its edits); without it an existing draft is an error")
    ab = _sp(ags, "build", "render the draft into the xlsx workbook",
             "Render Agenda/<series>.draft.yaml into Agenda/<series>.xlsx and print its path. Fails if the draft "
             "is incomplete (e.g. empty summary).")
    ab.add_argument("series", help="meeting series key")
    ab.add_argument("--force", action="store_true",
                    help="build even if the live workbook has not been read back (an earlier meeting, or meeting "
                         "input already typed into it); this discards that workbook's meeting edits")
    ar = _sp(ags, "read", "apply the meeting's edits to the store",
             "First runs a best-effort `git pull --rebase` (a failure is printed as a warning). Then reads "
             "Agenda/<series>.xlsx after the meeting: apply edits to the store, stage free-text items as "
             "Status/Pending/<P-id>.md, write the meeting note and archive the workbook.")
    ar.add_argument("series", help="meeting series key")

    ss = _sp(sub, "sync-state", "ingest watermarks",
             "Read or write ingest watermarks, one file per source at Status/sync-state/watermark.<SOURCE>.yaml. "
             "SOURCE is letters, digits, _, - or . only. Subcommands: get, set.")
    sss = ss.add_subparsers(dest="scmd", required=True, metavar="SUBCOMMAND")
    sg = _sp(sss, "get", "print a watermark", "Print the watermark for SOURCE (empty line if none).")
    sg.add_argument("source", help="ingest source name, e.g. clickup or circleback")
    sst = _sp(sss, "set", "store a watermark", "Store VALUE as the watermark for SOURCE.")
    sst.add_argument("source", help="ingest source name, e.g. clickup or circleback")
    sst.add_argument("value", help="watermark value, typically an ISO timestamp such as 2026-11-12T09:00")

    c = _sp(sub, "commit", "commit and push with rebase/retry",
            "git add -A, commit, pull --rebase, regenerate Snapshot.md, push (3 retries). Item files merge field "
            "by field through the hub-item git merge driver (entries and lists are unioned). Exit 3 on a genuine "
            "conflict, i.e. both sides changed one field of an item differently; the message names the field and "
            "both values (the rebase is aborted, nothing is lost). A failure to regenerate Snapshot.md is only a "
            "warning.")
    c.add_argument("message", help="git commit message")
    add_ext_parsers(sub)
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


def _pull_first(root) -> None:
    """Best-effort `git pull --rebase` so we act on the latest shared state; a failure is only a warning."""
    warning = pull_if_possible(root)
    if warning:
        print(warning, file=sys.stderr)


def _warn_load_errors(store) -> None:
    for path, err in store.load_errors:
        print(f"warning: cannot load {path}: {err}", file=sys.stderr)


def _print_items(items, as_json):
    if as_json:
        print(json.dumps([_row(i) for i in items], indent=2))
        return
    for i in items:
        print(f"{i.id}  {i.status:<9} {i.owner or '-':<10} {i.title}")


def _dispatch(args, cfg, store) -> int:
    today = _today()
    cmd = args.cmd
    rc = dispatch_ext(args, cfg, store, today=today, me=lambda: _me(cfg))
    if rc is not None:
        return rc
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
        _warn_load_errors(store)
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
        _print_items(store.query(series=args.series, since=args.date), False)
        _warn_load_errors(store)
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
            unreadable = load_unreadable(cfg.root)
            if not props and not unreadable:
                print("No pending proposals.")
            for pr in props:
                print(f"{pr.pid}  {pr.title}  (by {pr.body.get('proposer')}, {pr.body.get('date')}): "
                      f"{pr.body.get('reason')}")
            for path in unreadable:
                print(f"UNREADABLE {path}")
        elif args.pcmd == "apply":
            author = _me(cfg)
            _pull_first(cfg.root)

            def report(pid, ids):
                print(f"applied {pid}")
                for i in ids:
                    print(f"created {i}")
                sys.stdout.flush()

            apply_proposals(cfg.root, store, args.pids, author=author, day=today, on_applied=report)
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
            author = _me(cfg)
            _pull_first(cfg.root)
            res = read_agenda(store, cfg, args.series, author=author, today=today)
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
    except UnknownItem as exc:
        print(f"error: no such item {exc}", file=sys.stderr)
        return 1
    except (UserError, ConfigError, StoreError, PendingError, DraftError, AgendaError, SyncError, ItemFormatError, ValueError,
            OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
