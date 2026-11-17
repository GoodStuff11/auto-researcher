#!/usr/bin/env python3
"""
bs.py — read and write a hard-problem-brainstorm working directory by id, so that no agent ever
has to load a whole (arbitrarily long) document.

LAYOUT (ref/records.md, "Documents and context"; <wd> is the brainstorm working directory)
    brainstorm-<slug>.md      front page: Summary, Invariants, Final Tree, current leaf state, critical
                              path. Capped; the only master file read by default.
    history.md                everything historical about the master (old matrices, round syntheses,
                              Revision & Critic Log). Append-only; searched, never read whole.
    kill_index.md             one row per KILLED or SHELVED route, linking its card.
    library.md                findings library: an index (one line per entry), then the entries.
    compute/                  compute_gate.py budget and ledger.
    leaves/<id>/summary.md    the leaf's current state: title, Parent Chain/Status line, Leaf Status
                              Summary, live Route Portfolio, open reopening-ledger rows. Capped.
    leaves/<id>/problem.md    formulation, barrier, Analogy Map. Capped; changes rarely.
    leaves/<id>/cards/<R>.md  one route card per file (idea, outcome, kill reasoning). Capped.
    leaves/<id>/log/*.md      round logs, tier logs, critic reports, original legacy files.
                              Append-only; searched, never read whole.
    tests/                    test code and results (unchanged).

SIZE CAPS (approximate tokens = characters / 4; check_brainstorm.py warns above them)
    Only fixed-scope records are capped: leaf summary 2500, route card 900, library entry 700,
    library index line 80. Documents that grow with the run (front page, problem.md, kill index)
    have no cap: keep them short, split them by reader, and read their sizes here as advice.

USAGE
    python3 bs.py state    <wd>
    python3 bs.py context  <wd> <leaf> [--role worker|critic|compactor] [--cards R1,R2] [--lessons L3,L7]
    python3 bs.py show     <wd> <id>                 (R<id>, P<leaf>, L<n>, or a job id)
    python3 bs.py grep     <wd> <regex> [--leaf P2.1] [--max 40]
    python3 bs.py sizes    <wd> [--over]
    python3 bs.py new-leaf <wd> <leaf> --title TEXT --parent P2 --question TEXT
    python3 bs.py new-card <wd> <leaf> <route> --name TEXT
    python3 bs.py migrate  <legacy-wd> [--dry-run]

SUBCOMMANDS
    state     One line per leaf: id, State, and the first sentence of its Answer. This is what the
              orchestrator reads instead of the leaves.
    context   Print the minimal context for one agent working on one leaf, with a size header:
                worker     the front page's Invariants, the leaf's Final Tree bullet, the library
                           index, the leaf's summary and problem, every LIVE card in full, and one
                           line per other card;
                critic     the Invariants, the leaf's Final Tree bullet, the leaf's summary and
                           problem, the cards named by --cards in full (default: every card whose
                           outcome changed since the leaf summary's Updated date cannot be known,
                           so pass them), the library entries named by --lessons, one line per other card;
                compactor  the leaf's summary and problem, one line per card, and the list of log
                           files with their sizes.
    show      Print one item: a route card (R<id>, searched in every leaf; prefix with the leaf,
              "P2.1:R3", if ids repeat), a leaf summary (P<id>), a library entry (L<n>), or, for a
              number of 5 or more digits, the compute-ledger rows and up to 20 log lines naming it.
    grep      Search every Markdown file (logs and history included) for a regular expression and
              print file:line: text, truncated, at most --max lines. The way to consult history.
    sizes     Approximate tokens of every file, against its cap where it has one (uncapped growing
              documents are listed as advice); --over lists only capped records over their cap,
              which are the compaction targets.
    new-leaf  Create leaves/<leaf>/ with summary.md, problem.md, cards/ and log/ from templates. The
              question must be the leaf's Final Tree question, word for word.
    new-card  Create leaves/<leaf>/cards/<route>.md from the route-card template.
    migrate   Convert a working directory in the legacy layout (one long .subproblems/P*.md per leaf,
              one long master) into this layout, without losing anything:
                * each leaf file is split into summary.md (title, status line, Leaf Status Summary),
                  problem.md ("### 1." and "### 1b." sections), one card per "### R" block of its
                  Route Register, and log/legacy.md (everything else, in order); the original file
                  is kept unchanged as log/original.md;
                * the master is split into the front page (Summary, Invariants, the Final Tree,
                  Stage 1.5, the "Current ..." subsections of the synthesis matrix and the critical
                  path), kill_index.md (its Kill Index) and history.md (everything else, in order);
                  the original is kept as log/master-original.md;
                * links of the form [[.subproblems/<file>#R<id> ...]] are rewritten to
                  [[leaves/<leaf>/cards/<R>]] in every new file;
                * a report lists sizes before and after and what still needs a compactor (for
                  example a summary without a live Route Portfolio).
              --dry-run prints the report without writing. Refuses if leaves/ already exists.

EXIT CODES
    0 success; 1 not found, refused or failed; 2 bad arguments.
"""

import argparse
import csv
import os
import re
import shutil
import signal
import sys

CAPS = {"summary": 2500, "card": 900, "library-entry": 700,  # fixed-scope records only
        "library-index-line": 80}
CARD_RE = re.compile(r"^#{1,3}\s+(R\S+)\s+[—–-]\s+(.*)$")  # id up to the spaced dash, so R1-L stays whole


# --------------------------------------------------------------------------- helpers

def tokens(text):
    return len(text) // 4


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def write(p, text):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def front_page(wd):
    for n in sorted(os.listdir(wd)):
        if n.startswith("brainstorm-") and n.endswith(".md"):
            return os.path.join(wd, n)
    return None


def leaves(wd):
    d = os.path.join(wd, "leaves")
    if not os.path.isdir(d):
        return []
    return sorted((x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))), key=natkey)


def leaf_dir(wd, leaf):
    return os.path.join(wd, "leaves", leaf)


def field(text, name):
    m = re.search(r"\*\*" + re.escape(name) + r"\*\*\s*:\s*(.*)", text)
    return m.group(1).strip() if m else ""


def first_sentence(s, words=45):
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    out = m.group(1) if m else s
    w = out.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def sections(text, level):
    """Split text at headings of exactly `level` '#'. Returns [(heading_line or '', body)]."""
    pat = re.compile(r"^" + "#" * level + r"\s")
    out, head, buf = [], "", []
    for line in text.splitlines(keepends=True):
        if pat.match(line):
            out.append((head, "".join(buf)))
            head, buf = line.rstrip("\n"), []
        else:
            buf.append(line)
    out.append((head, "".join(buf)))
    return out


def section(text, title_re, level=2):
    for head, body in sections(text, level):
        if head and re.match(r"^#{%d}\s+%s" % (level, title_re), head):
            return head + "\n" + body
    return ""


def card_file_name(rid):
    s = rid.replace("″", "pp").replace("′", "p").replace("'", "p")
    return re.sub(r"[^A-Za-z0-9.\-_]", "_", s)


def card_files(wd, leaf):
    d = os.path.join(leaf_dir(wd, leaf), "cards")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, n) for n in sorted(os.listdir(d), key=natkey) if n.endswith(".md")]


def card_line(p):
    t = read(p)
    title = t.splitlines()[0].lstrip("# ").strip() if t.strip() else os.path.basename(p)
    out = field(t, "Outcome")
    why = field(t, "Why this kills it") or field(t, "Why shelved")
    s = f"* {title} — {first_sentence(out, 20)}"
    if why:
        s += f" — {first_sentence(why, 30)}"
    return s


def final_tree_bullet(wd, leaf):
    fp = front_page(wd)
    if not fp:
        return ""
    for line in read(fp).splitlines():
        if re.match(r"^\s*\*\s+\*\*" + re.escape(leaf) + r"\s+\[leaf\]", line):
            return line.strip()
    return ""


def library_entry(wd, lid):
    p = os.path.join(wd, "library.md")
    if not os.path.exists(p):
        return ""
    for head, body in sections(read(p), 3):
        if re.match(r"^###\s+" + re.escape(lid) + r"\.", head):
            return head + "\n" + body
    return ""


def library_index(wd):
    p = os.path.join(wd, "library.md")
    return section(read(p), r"Index") if os.path.exists(p) else ""


# --------------------------------------------------------------------------- commands

def cmd_state(a):
    ls = leaves(a.wd)
    if not ls:
        print("no leaves/ directory (legacy layout? run `bs.py migrate`)")
        return 1
    print("| Leaf | State | Answer (first sentence) |")
    print("| :--- | :--- | :--- |")
    for lf in ls:
        p = os.path.join(leaf_dir(a.wd, lf), "summary.md")
        t = read(p) if os.path.exists(p) else ""
        print(f"| {lf} | {field(t, 'State') or '?'} | {first_sentence(field(t, 'Answer'), 40) or '?'} |")
    return 0


def emit(parts):
    total = sum(tokens(b) for _, b in parts)
    print(f"<!-- context: {len(parts)} parts, about {total} tokens -->")
    for title, body in parts:
        print(f"\n<!-- {title} ({tokens(body)} tokens) -->\n")
        print(body.rstrip())


def cmd_context(a):
    d = leaf_dir(a.wd, a.leaf)
    if not os.path.isdir(d):
        print(f"no leaf {a.leaf}")
        return 1
    fp = front_page(a.wd)
    inv = section(read(fp), r"Invariants") if fp else ""
    summ = read(os.path.join(d, "summary.md")) if os.path.exists(os.path.join(d, "summary.md")) else ""
    prob = read(os.path.join(d, "problem.md")) if os.path.exists(os.path.join(d, "problem.md")) else ""
    cards = card_files(a.wd, a.leaf)
    want = {c.strip() for c in (a.cards or "").split(",") if c.strip()}
    parts = []
    if a.role in ("worker", "critic"):
        parts.append(("Invariants (front page)", inv))
        parts.append(("Final Tree bullet", final_tree_bullet(a.wd, a.leaf)))
    if a.role == "worker":
        parts.append(("library index", library_index(a.wd)))
    parts.append((f"leaves/{a.leaf}/summary.md", summ))
    parts.append((f"leaves/{a.leaf}/problem.md", prob))
    full, lines = [], []
    for c in cards:
        rid = os.path.basename(c)[:-3]
        t = read(c)
        live = "LIVE" in field(t, "Outcome")[:20] or "(LIVE)" in t.splitlines()[0]
        if (a.role == "worker" and live) or (a.role == "critic" and (rid in want or card_file_name(rid) in want)):
            full.append((f"card {rid}", t))
        else:
            lines.append(card_line(c))
    parts += full
    parts.append(("other cards (one line each; `bs.py show` for any)", "\n".join(lines)))
    if a.role == "critic":
        for lid in (a.lessons or "").split(","):
            if lid.strip():
                parts.append((f"library {lid.strip()}", library_entry(a.wd, lid.strip())))
    if a.role == "compactor":
        logd = os.path.join(d, "log")
        logs = [f"* log/{n} ({tokens(read(os.path.join(logd, n)))} tokens)" for n in sorted(os.listdir(logd))] \
            if os.path.isdir(logd) else []
        parts.append(("log files (search with `bs.py grep`, never read whole)", "\n".join(logs)))
    emit(parts)
    return 0


def find_cards(wd, rid):
    leaf = None
    if ":" in rid:
        leaf, rid = rid.split(":", 1)
    hits = []
    for lf in ([leaf] if leaf else leaves(wd)):
        for c in card_files(wd, lf):
            if os.path.basename(c)[:-3] in (rid, card_file_name(rid)):
                hits.append(c)
    return hits


def cmd_show(a):
    i = a.id
    if re.fullmatch(r"(P[\w.\-]+:)?R.*", i):
        hits = find_cards(a.wd, i)
        if not hits:
            print(f"no card {i}")
            return 1
        for h in hits:
            print(f"<!-- {os.path.relpath(h, a.wd)} -->\n{read(h).rstrip()}\n")
        return 0
    if re.fullmatch(r"L\d+", i):
        e = library_entry(a.wd, i)
        print(e.rstrip() if e else f"no library entry {i}")
        return 0 if e else 1
    if i.startswith("P"):
        p = os.path.join(leaf_dir(a.wd, i), "summary.md")
        if not os.path.exists(p):
            print(f"no leaf {i}")
            return 1
        print(read(p).rstrip())
        return 0
    if re.fullmatch(r"\d{5,}", i):
        led = os.path.join(a.wd, "compute", "ledger.tsv")
        found = False
        if os.path.exists(led):
            for r in csv.DictReader(open(led, newline=""), delimiter="\t"):
                if r.get("job_id") == i:
                    found = True
                    print("ledger: " + "; ".join(f"{k}={v}" for k, v in r.items() if v))
        a.regex, a.leaf, a.max = re.escape(i), None, 20
        return 0 if (cmd_grep(a) == 0 or found) else 1
    print(f"unrecognised id {i}")
    return 1


def md_files(wd):
    for root, dirs, names in os.walk(wd):
        dirs[:] = sorted(d for d in dirs if d not in ("tests", ".git", "__pycache__"))
        for n in sorted(names):
            if n.endswith(".md"):
                yield os.path.join(root, n)


def cmd_grep(a):
    rx = re.compile(a.regex)
    n = 0
    for p in md_files(a.wd):
        rel = os.path.relpath(p, a.wd)
        if a.leaf and not rel.startswith(os.path.join("leaves", a.leaf) + os.sep):
            continue
        for k, line in enumerate(read(p).splitlines(), 1):
            if rx.search(line):
                print(f"{rel}:{k}: {line.strip()[:220]}")
                n += 1
                if n >= a.max:
                    print(f"... stopped at --max {a.max}")
                    return 0
    return 0 if n else 1


def kind_of(wd, p):
    rel = os.path.relpath(p, wd).split(os.sep)
    if len(rel) == 1 and rel[0].startswith("brainstorm-"):
        return "front"
    if rel[0] == "leaves" and len(rel) == 3 and rel[2] == "summary.md":
        return "summary"
    if rel[0] == "leaves" and len(rel) == 3 and rel[2] == "problem.md":
        return "problem"
    if rel[0] == "leaves" and len(rel) == 4 and rel[2] == "cards":
        return "card"
    return None


def cmd_sizes(a):
    rows = []
    for p in md_files(a.wd):
        k = kind_of(a.wd, p)
        t = tokens(read(p))
        cap = CAPS.get(k)
        rows.append((os.path.relpath(p, a.wd), t, cap))
    lib = os.path.join(a.wd, "library.md")
    if os.path.exists(lib):
        for head, body in sections(read(lib), 3):
            if re.match(r"^###\s+L\d+", head) and tokens(head + body) > CAPS["library-entry"]:
                rows.append((f"library.md {head[4:40]}...", tokens(head + body), CAPS["library-entry"]))
    over = 0
    for rel, t, cap in sorted(rows, key=lambda r: -r[1]):
        flag = cap is not None and t > cap
        over += flag
        if a.over and not flag:
            continue
        print(f"{t:8d} {('/ ' + str(cap)) if cap else '':>8} {'OVER ' if flag else '     '}{rel}")
    print(f"{over} file(s) or entries over cap")
    return 0


SUMMARY_T = """# Subproblem: {leaf} {title}
**Parent Chain**: {parent} -> {leaf}  **Status**: OPEN

## Route Portfolio

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |

## Leaf Status Summary
* **State**: OPEN
* **Updated**: (date, after which result)
* **Question**: {question}
* **Answer**: Not known. (Replace with the direct reply to the question as asked; ref/records.md, 3a.)
* **What we showed**: (nothing yet)
* **What it would take**: (the next decisive step)
* **Conclusion**: (technical version, agreeing with Answer)
* **Scope**: (instances, sizes, sectors, compute regime)
* **Compute spent**: 0 CPU-h
"""

PROBLEM_T = """# Problem: {leaf} {title}

### 1. Mathematical Abstraction & Barriers
* **Formulation**: (inputs, outputs, what "solved" means)
* **The Node Barrier**: (where existing methods halt against the Hard Core)

### 1b. Analogy Map
* (each barrier restated in field-neutral terms; at least three source fields each)
"""

CARD_T = """# {rid} — {name} (LIVE)
* **The idea**: (what it does in plain terms, what it changes relative to the foil, and why it should work; at least 25 words)
* **Lineage**: (initial batch / Operation A from R<x> / Operation B / cross-field translation from <field> / translation repair of R<x>)
* **Outcome**: LIVE (GATED)
"""


def cmd_new_leaf(a):
    d = leaf_dir(a.wd, a.leaf)
    if os.path.exists(d):
        print(f"{d} exists")
        return 1
    kw = dict(leaf=a.leaf, title=a.title, parent=a.parent, question=a.question)
    write(os.path.join(d, "summary.md"), SUMMARY_T.format(**kw))
    write(os.path.join(d, "problem.md"), PROBLEM_T.format(**kw))
    os.makedirs(os.path.join(d, "cards"), exist_ok=True)
    os.makedirs(os.path.join(d, "log"), exist_ok=True)
    print(f"created {os.path.relpath(d, a.wd)}/ (summary.md, problem.md, cards/, log/)")
    return 0


def cmd_new_card(a):
    p = os.path.join(leaf_dir(a.wd, a.leaf), "cards", card_file_name(a.route) + ".md")
    if os.path.exists(p):
        print(f"{p} exists")
        return 1
    write(p, CARD_T.format(rid=a.route, name=a.name))
    print(f"created {os.path.relpath(p, a.wd)}")
    return 0


# --------------------------------------------------------------------------- migrate

LINK_RE = re.compile(r"\[\[\.subproblems/([^#\]|]+?)(?:\.md)?#\s*(R[^\s\]|—–]*)[^\]]*\]\]")


def migrate_leaf(src, wd, file_to_leaf, report):
    text = read(src)
    lines = text.splitlines(keepends=True)
    m = re.match(r"^#\s+Subproblem:\s*(\S+)\s*(.*)$", lines[0])
    leaf, title = m.group(1), m.group(2).strip()
    # header: lines before the first heading after line 0
    k = 1
    while k < len(lines) and not re.match(r"^#{2,3}\s", lines[k]):
        k += 1
    header = "".join(lines[:k])
    body = "".join(lines[k:])
    problem, legacy, summary, cards = [], [], "", []
    for head2, body2 in sections(body, 2):
        if re.match(r"^##\s+Leaf Status Summary", head2):
            summary = head2 + "\n" + body2
        elif re.match(r"^##\s+Route Register", head2):
            for head3, body3 in sections(body2, 3):
                cm = CARD_RE.match(head3) if head3 else None
                if cm:
                    cards.append((cm.group(1), "# " + head3.lstrip("#").strip() + "\n" + body3))
                elif (head3 + body3).strip():
                    legacy.append((head3 + "\n" if head3 else "") + body3)
        else:
            for head3, body3 in sections((head2 + "\n" if head2 else "") + body2, 3):
                block = (head3 + "\n" if head3 else "") + body3
                if head3 and re.match(r"^###\s+1b?\.\s", head3):
                    problem.append(block)
                elif block.strip():
                    legacy.append(block)
    d = leaf_dir(wd, leaf)
    out = {}
    out[os.path.join(d, "summary.md")] = header.rstrip() + "\n\n" + (summary or "## Leaf Status Summary\n* **State**: ?\n")
    out[os.path.join(d, "problem.md")] = f"# Problem: {leaf} {title}\n\n" + "".join(problem)
    seen = {}
    for rid, card in cards:
        fn = card_file_name(rid)
        seen[fn] = seen.get(fn, 0) + 1
        if seen[fn] > 1:
            fn = f"{fn}-{seen[fn]}"
        out[os.path.join(d, "cards", fn + ".md")] = card
    out[os.path.join(d, "log", "legacy.md")] = f"# Log: {leaf} (legacy rounds, migrated)\n\n" + "".join(legacy)
    out[os.path.join(d, "log", "original.md")] = text
    file_to_leaf[os.path.basename(src)[:-3]] = leaf
    todo = []
    if not summary:
        todo.append("no Leaf Status Summary")
    if "Route Portfolio" not in summary:
        todo.append("summary lacks the live Route Portfolio")
    if not problem:
        todo.append("no '### 1.' problem sections")
    report.append((leaf, tokens(text), {k: tokens(v) for k, v in out.items()}, len(cards), todo))
    return out


FRONT_KEEP2 = (r"Summary", r"Invariants", r"Stage 1\.5")


def migrate_master(src, wd):
    text = read(src)
    lines = text.splitlines(keepends=True)
    title = lines[0] if lines and lines[0].startswith("# ") else "# Brainstorm\n"
    rest = "".join(lines[1:]) if lines and lines[0].startswith("# ") else text
    front, hist, kill = [title], ["# History (migrated from the master; append-only, never read whole)\n\n"], ""
    for head2, body2 in sections(rest, 2):
        block = (head2 + "\n" if head2 else "") + body2
        if not head2:
            if block.strip():
                hist.append(block)
            continue
        if any(re.match(r"^##\s+" + t, head2) for t in FRONT_KEEP2):
            front.append(block)
        elif re.match(r"^##\s+Kill Index", head2):
            kill = "# Kill Index\n" + body2
        elif re.match(r"^##\s+Stage 1\b", head2):
            subs = sections(body2, 3)
            ft = [(h, b) for h, b in subs if h and re.match(r"^###\s+Final Tree", h)]
            front.append(head2 + "\n" + "".join(h + "\n" + b for h, b in ft))
            other = "".join((h + "\n" if h else "") + b for h, b in subs if not (h and re.match(r"^###\s+Final Tree", h)))
            if other.strip():
                hist.append(head2 + " (non-tree parts)\n" + other)
        elif re.match(r"^##\s+(Master Leaf Synthesis Matrix|Cross-Pollination)", head2):
            subs = sections(body2, 3)
            cur = [(h, b) for h, b in subs if h and re.match(r"^###\s+Current", h)]
            if cur:
                front.append(head2 + "\n" + "".join(h + "\n" + b for h, b in cur))
            other = "".join((h + "\n" if h else "") + b for h, b in subs if not (h and re.match(r"^###\s+Current", h)))
            if other.strip():
                hist.append(head2 + " (history)\n" + other)
        else:
            hist.append(block)
    front.append("\n## Where everything else is\n\n* History of this page: `history.md` (search with `bs.py grep`).\n"
                 "* Killed and shelved routes: `kill_index.md`; cards in `leaves/<leaf>/cards/`.\n"
                 "* Findings library: `library.md`. Compute: `compute_gate.py report`.\n")
    return "".join(front), "".join(hist), kill


def rewrite_links(text, file_to_leaf):
    def sub(m):
        leaf = file_to_leaf.get(m.group(1))
        if not leaf:
            return m.group(0)
        return f"[[leaves/{leaf}/cards/{card_file_name(m.group(2))}]]"
    return LINK_RE.sub(sub, text)


def cmd_migrate(a):
    wd = a.wd
    sp = os.path.join(wd, ".subproblems")
    if os.path.exists(os.path.join(wd, "leaves")):
        print("leaves/ already exists: refusing to migrate twice")
        return 1
    if not os.path.isdir(sp):
        print("no .subproblems/: nothing to migrate")
        return 1
    file_to_leaf, report, out = {}, [], {}
    for n in sorted(os.listdir(sp), key=natkey):
        p = os.path.join(sp, n)
        if n.endswith(".md") and read(p).startswith("# Subproblem:"):
            out.update(migrate_leaf(p, wd, file_to_leaf, report))
        elif os.path.isfile(p):
            out[os.path.join(wd, "log", "subproblems", n)] = read(p)
    fp = front_page(wd)
    old_master = tokens(read(fp)) if fp else 0
    if fp:
        front, hist, kill = migrate_master(fp, wd)
        out[fp] = front
        out[os.path.join(wd, "history.md")] = hist
        if kill:
            out[os.path.join(wd, "kill_index.md")] = kill
        out[os.path.join(wd, "log", "master-original.md")] = read(fp)
    for p in list(out):
        if not p.endswith("original.md"):
            out[p] = rewrite_links(out[p], file_to_leaf)
    lib = os.path.join(wd, "library.md")
    if os.path.exists(lib):
        out[lib] = rewrite_links(read(lib), file_to_leaf)

    print(f"master: {old_master} tokens -> front page {tokens(out.get(fp, ''))}, "
          f"history {tokens(out.get(os.path.join(wd, 'history.md'), ''))}, "
          f"kill index {tokens(out.get(os.path.join(wd, 'kill_index.md'), ''))}")
    print("| Leaf | Before | summary | problem | cards (n, largest) | log/legacy | Still to do |")
    print("| :--- | ---: | ---: | ---: | :--- | ---: | :--- |")
    for leaf, before, sizes, ncards, todo in report:
        g = lambda name: next((v for k, v in sizes.items() if k.endswith(os.path.join(leaf, name))), 0)
        cs = [v for k, v in sizes.items() if os.sep + "cards" + os.sep in k]
        print(f"| {leaf} | {before} | {g('summary.md')} | {g('problem.md')} | {ncards}, {max(cs) if cs else 0} | "
              f"{g(os.path.join('log', 'legacy.md'))} | {'; '.join(todo) or '—'} |")
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    for p, t in out.items():
        write(p, t)
    shutil.move(sp, os.path.join(wd, "log", "subproblems-original"))
    print(f"wrote {len(out)} files; the old .subproblems/ is now log/subproblems-original/")
    return 0


# --------------------------------------------------------------------------- CLI

def parse_args(argv):
    ap = argparse.ArgumentParser(prog="bs.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("state"); p.add_argument("wd")
    p = sp.add_parser("context"); p.add_argument("wd"); p.add_argument("leaf")
    p.add_argument("--role", choices=["worker", "critic", "compactor"], default="worker")
    p.add_argument("--cards", default=""); p.add_argument("--lessons", default="")
    p = sp.add_parser("show"); p.add_argument("wd"); p.add_argument("id")
    p = sp.add_parser("grep"); p.add_argument("wd"); p.add_argument("regex")
    p.add_argument("--leaf"); p.add_argument("--max", type=int, default=40)
    p = sp.add_parser("sizes"); p.add_argument("wd"); p.add_argument("--over", action="store_true")
    p = sp.add_parser("new-leaf"); p.add_argument("wd"); p.add_argument("leaf")
    p.add_argument("--title", required=True); p.add_argument("--parent", required=True)
    p.add_argument("--question", required=True)
    p = sp.add_parser("new-card"); p.add_argument("wd"); p.add_argument("leaf"); p.add_argument("route")
    p.add_argument("--name", required=True)
    p = sp.add_parser("migrate"); p.add_argument("wd"); p.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    a = parse_args(argv)
    h = {"state": cmd_state, "context": cmd_context, "show": cmd_show, "grep": cmd_grep, "sizes": cmd_sizes,
         "new-leaf": cmd_new_leaf, "new-card": cmd_new_card, "migrate": cmd_migrate}
    return h[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
