#!/usr/bin/env python3
"""
Mechanical linter for brainstorm working documents.
Validates structural integrity, frozen invariant tags, LaTeX syntax, banned subjective ratings,
and Markdown structure (code fences, indented code blocks, tables, inline code and math delimiters).
Supports Mode A single documents, Mode B fan-out master matrices, and Mode B leaf artifacts.
Ensures zero early surrender by checking for surrender phrases.

Usage:
    python3 check_brainstorm.py <path> [<path> ...]

Each <path> is either
    * a Markdown file, or
    * a directory: every *.md file under it is linted, including hidden directories such as
      .subproblems/, skipping tests/, __pycache__/ and .git/.

The document role is detected per file, by path in the leaves/ layout (ref/records.md, "Documents and
context") and by content for legacy files:
    * master       : brainstorm-*.md (the front page), or a file with "## Invariants". Requires "## Summary",
                     "## Invariants" and a "Final Tree" heading (legacy layout: also "## Revision & Critic
                     Log"). Warns K0 if there is no kill_index.md (or legacy "## Kill Index"), K1 if there is
                     no library.md, L9 if the directory has no leaves/ (legacy layout), C0 over 5000 tokens.
    * leaf-summary : leaves/<id>/summary.md. Errors: S0 it starts with "# Subproblem: <id> <title>", has
                     **Parent Chain** and **Status**, and ends with "## Leaf Status Summary"; S1 the summary
                     has **State**, **Question**, **Answer**, **What we showed** and **What it would take**;
                     S2 the Status line matches State; S3 Question equals the leaf's Final Tree question in
                     the front page word for word (warning if the bullet is not found); S4 a yes/no Question
                     ("Can", "Is", "Does", ...) has an Answer opening with Yes, No, Partly or Not known.
                     Warning C0 over 2500 tokens.
    * leaf-problem : leaves/<id>/problem.md. Warning A0 without an "Analogy Map" heading; C0 over 4000.
    * card         : leaves/<id>/cards/<R>.md. Starts with "# R<id> — <name> (<STATE>)"; route-card fields
                     R1-R4 as below; warning C0 over 900 tokens.
    * kill-index   : kill_index.md. Error K2 for a [[leaves/...]] link with no card file; warning K3 for a
                     legacy [[.subproblems/...]] link.
    * library      : library.md. Entry fields L0-L2 as below; warnings C1 for an entry over 700 tokens and
                     C2 for an index line over 80 tokens.
    * archive      : history.md and anything under a log/ directory. Hygiene checks only, no size cap.
    * leaf         : legacy single-file leaf (first heading "# Subproblem:" outside leaves/). The legacy
                     "## Route Register" checks below, plus warning L9 (run `bs.py migrate`).
    * other        : any other Markdown file (briefs, notes). Hygiene checks only.

Route-card fields (card files, and legacy Route Registers):
    R0 (legacy only) a "## Route Register" section exists (warning) and comes before the final
       "## Leaf Status Summary" (error);
    R1 every card has **The idea**, **Lineage** and **Outcome**; a KILLED card also has **Kill rule**,
       **Evidence**, **Why this kills it**, **What the kill does not rule out**, **Strongest case against
       the kill**, **Checked by** and **Lesson**; a SHELVED card has **Why shelved** and **What would
       unshelve it** (errors);
    R2 a KILLED outcome names where it died: "KILLED at G0".."G4", "T0".."T3" or "CRITIC" (error);
    R3 **The idea** has at least 25 words (warning);
    R4 a card whose **Lineage** says "cross-field" or "translation repair" has **Source analogy** (error).
Library entries ("### L<n>. ..."): L1 every field (**Status**, **Applies to**, **The lesson**, **Evidence**,
    **Scope and limits**, **Counter-evidence and open questions**, **How to use it**, **Sources**); L2 Status
    is ESTABLISHED, PROVISIONAL or REFUTED (errors); L0 no entries at all (warning).

Approximate tokens are characters / 4.

Hygiene checks (all roles):
    Markdown structure (errors, because they change how the document renders in Obsidian/GFM):
        M1 a fenced code block (``` or ~~~) that is never closed;
        M2 an opening fence whose previous line is non-blank (unless it is a heading). Obsidian
           then absorbs the fence into the preceding list item or paragraph and pairs every later
           fence off by one, leaving one extra, never-closed code block;
        M3 a closing fence whose next line is non-blank;
        M4 an accidental indented code block: text indented by 4 or more spaces that CommonMark
           renders as code, typically list items continued after a fence or after a table;
        M5 a table that is not separated by a blank line from the text before or after it
           (without the blank line after, GFM swallows the next line into the table).
    Markdown inline (warnings):
        M6 a line outside code fences with an odd number of unescaped backticks;
        M7 a paragraph outside code fences with an odd number of unescaped single '$' signs.
    Content:
        banned subjective ratings (errors), surrender phrases (errors), unescaped '|' inside
        table cells (warnings), raw math outside LaTeX (warnings).

Exit status: 0 if no file has errors, 1 otherwise. Warnings do not change the exit status, but
the procedure requires them to be fixed as well.

M4 uses the markdown-it-py parser when it is installed, and a conservative heuristic otherwise
(a line indented by 4 or more spaces after a blank line, a closing fence or a table, outside any
list context).
"""

import sys
import os
import re

try:
    from markdown_it import MarkdownIt
    _MD = MarkdownIt("commonmark").enable("table")
except Exception:  # pragma: no cover - fallback path when markdown-it-py is missing
    _MD = None

REQUIRED_CORE_SECTIONS = [
    r"^##\s+Summary",
    r"^##\s+Invariants",
    r"^##\s+Stage 1",
    r"^##\s+Revision & Critic Log",
]

REQUIRED_LEAF_FIELDS = [
    r"\*\*Parent Chain\*\*",
    r"\*\*Status\*\*",
]

BANNED_RATING_PATTERNS = [
    r"\bscore:\s*\d+\b",
    r"\brating:\s*\d+\b",
    r"\bfeasibility:\s*[1-9]/10\b",
    r"\bimpact:\s*[1-9]/10\b",
    r"\b\d+\s*/\s*10\b",
]

RAW_MATH_PATTERNS = [
    (r"(?<![\\a-zA-Z])sqrt\(", "Bare 'sqrt(' detected. Use $\\sqrt{...}$."),
    (r"(?<![\\a-zA-Z])psi\b", "Bare 'psi' detected. Use $\\psi$."),
    (r"(?<![\\a-zA-Z])chi\b", "Bare 'chi' detected. Use $\\chi$."),
    (r"(?<!\$)<=|(?<!\$)>=|(?<!\$)!=|(?<!\$)≈", "Raw inequality detected outside math delimiter. Use $\\le$, $\\ge$, $\\ne$, or $\\approx$."),
]

SURRENDER_PHRASES = [
    "no viable routes found",
    "all routes were killed and no alternatives exist",
    "problem appears intractable with no remaining ideas",
    "giving up on this direction",
]

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")  # any indent: fences nested in list items are valid
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
LIST_RE = re.compile(r"^\s*([*+-]|\d+[.)])\s")
SKIP_DIRS = {"tests", "__pycache__", ".git"}


ROUTE_COMMON = ["The idea", "Lineage", "Outcome"]
ROUTE_KILLED = ["Kill rule", "Evidence", "Why this kills it", "What the kill does not rule out",
                "Strongest case against the kill", "Checked by", "Lesson"]
ROUTE_SHELVED = ["Why shelved", "What would unshelve it"]
KILL_CODE_RE = re.compile(r"KILLED at (G[0-4]|T[0-3]|CRITIC)\b")
LIBRARY_FIELDS = ["Status", "Applies to", "The lesson", "Evidence", "Scope and limits",
                  "Counter-evidence and open questions", "How to use it", "Sources"]
LIBRARY_STATUS_RE = re.compile(r"\*\*Status\*\*\s*:\s*\**\s*(ESTABLISHED|PROVISIONAL|REFUTED)\b")
MIN_IDEA_WORDS = 25


def _field(block, name):
    """Value text of a '**name**:' field in a block of lines, or None."""
    m = re.search(r"\*\*" + re.escape(name) + r"\*\*\s*:\s*(.*)", block)
    return m.group(1).strip() if m else None


def _section(lines, title_re):
    """(start, end) line indices of the '## ' section whose title matches, or None."""
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^##\s", line):
            if start is not None:
                return start, i
            if re.match(title_re, line):
                start = i
    return (start, len(lines)) if start is not None else None


def _cards(lines, start, end, head_re):
    """Split lines[start:end] into (title_line_no, title, text) blocks at '### ' headings matching head_re."""
    out, cur = [], None
    for i in range(start, end):
        if re.match(r"^###\s", lines[i]):
            if cur:
                out.append(cur)
            cur = (i + 1, lines[i].strip(), "") if re.match(head_re, lines[i]) else None
        elif cur:
            cur = (cur[0], cur[1], cur[2] + lines[i])
    if cur:
        out.append(cur)
    return out


def check_route_register(lines):
    """Leaf: a '## Route Register' of route cards, above the final Leaf Status Summary. (level, line, msg)."""
    out = []
    sec = _section(lines, r"^##\s+Route Register")
    if sec is None:
        return [("WARNING", 1, "R0 leaf has no '## Route Register' section: every route ever generated needs a route card (ref/records.md).")]
    summ = _section(lines, r"^##\s+Leaf Status Summary")
    if summ is not None and summ[0] < sec[0]:
        out.append(("ERROR", sec[0] + 1, "R0 '## Route Register' must come before '## Leaf Status Summary', which is always the last section."))
    cards = _cards(lines, sec[0] + 1, sec[1], r"^###\s+R")
    if not cards:
        out.append(("WARNING", sec[0] + 1, "R0 Route Register has no route cards ('### R<id> — <name> (<state>)')."))
    for ln, title, text in cards:
        need = list(ROUTE_COMMON)
        outcome = _field(text, "Outcome") or ""
        if "KILLED" in outcome:
            need += ROUTE_KILLED
            if not KILL_CODE_RE.search(outcome):
                out.append(("ERROR", ln, f"R2 {title}: a KILLED outcome must name where it died, e.g. 'KILLED at G4' or 'KILLED at T1' or 'KILLED at CRITIC'."))
        elif "SHELVED" in outcome:
            need += ROUTE_SHELVED
        for f in need:
            v = _field(text, f)
            if v is None or v == "":
                out.append(("ERROR", ln, f"R1 {title}: route card is missing '**{f}**:'."))
        lineage = (_field(text, "Lineage") or "").lower()
        if ("cross-field" in lineage or "translation repair" in lineage) and not _field(text, "Source analogy"):
            out.append(("ERROR", ln, f"R4 {title}: a cross-field translation (or its repair) needs '**Source analogy**:' (source field and problem, correspondence, why it worked there, translation gap, citation)."))
        idea = _field(text, "The idea") or ""
        if idea and len(idea.split()) < MIN_IDEA_WORDS:
            out.append(("WARNING", ln, f"R3 {title}: 'The idea' has {len(idea.split())} words; explain the mechanism and why it should work in plain terms (at least {MIN_IDEA_WORDS} words)."))
    return out


def check_library(lines):
    """library.md: '### L<n>. <lesson>' entries, each with every field. (level, line, msg)."""
    out = []
    entries = _cards(lines, 0, len(lines), r"^###\s+L\d+")
    if not entries:
        out.append(("WARNING", 1, "L0 library has no entries ('### L<n>. <one-sentence lesson>')."))
    for ln, title, text in entries:
        for f in LIBRARY_FIELDS:
            v = _field(text, f)
            if v is None or v == "":
                out.append(("ERROR", ln, f"L1 {title}: library entry is missing '**{f}**:'."))
        if _field(text, "Status") is not None and not LIBRARY_STATUS_RE.search(text):
            out.append(("ERROR", ln, f"L2 {title}: Status must be ESTABLISHED, PROVISIONAL or REFUTED."))
    return out


CAPS = {"master": 5000, "leaf-summary": 2500, "leaf-problem": 4000, "card": 900}
LIB_ENTRY_CAP, LIB_INDEX_LINE_CAP = 700, 80
SUMMARY_FIELDS = ["State", "Question", "Answer", "What we showed", "What it would take"]
YESNO_START = re.compile(r"^(Can|Is|Are|Does|Do|Did|Will|Would|Should|Could|Has|Have|May|Must)\b")
ANSWER_OPEN = re.compile(r"^\**\s*(Yes|No|Partly|Not known)\b")


def approx_tokens(text):
    return len(text) // 4


def _norm(s):
    return " ".join(s.replace("**", "").split())


def final_tree_question(master_text, leaf):
    """Text of the leaf's Final Tree bullet after its bold title, or None."""
    m = re.search(r"^\s*\*\s+\*\*" + re.escape(leaf) + r"\s+\[leaf\][^*]*\*\*\s*(.*)$", master_text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _find_master(start_dir):
    d = os.path.abspath(start_dir)
    for _ in range(4):
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if n.startswith("brainstorm-") and n.endswith(".md"):
                return os.path.join(d, n)
        d = os.path.dirname(d)
    return None


def detect_role(filepath, content):
    """Classify a Markdown file by its path (leaves/ layout) or, for legacy files, by its content."""
    parts = os.path.abspath(filepath).split(os.sep)
    name = parts[-1]
    if "leaves" in parts:
        rel = parts[parts.index("leaves") + 1:]
        if len(rel) == 2 and rel[1] == "summary.md":
            return "leaf-summary"
        if len(rel) == 2 and rel[1] == "problem.md":
            return "leaf-problem"
        if len(rel) == 3 and rel[1] == "cards":
            return "card"
        return "archive"
    if "log" in parts[:-1] or name == "history.md":
        return "archive"
    if name == "library.md":
        return "library"
    if name == "kill_index.md":
        return "kill-index"
    if name.startswith("brainstorm-") or re.search(r"^##\s+Invariants", content, re.MULTILINE):
        return "master"
    for line in content.splitlines():
        if line.startswith("#"):
            return "leaf" if re.match(r"^#\s+Subproblem:", line) else "other"
    return "other"


def check_card_file(lines):
    """A card file: '# R<id> — name (STATE)' and the fields of a route card (R1-R4)."""
    if not lines or not re.match(r"^#\s+R", lines[0]):
        return [("ERROR", 1, "R1 a card file must start with '# R<id> — <name> (<STATE>)'.")]
    return [x for x in check_route_register(["## Route Register\n", "### " + lines[0][2:]] + lines[1:])
            if not x[2].startswith("R0")]


def check_summary_file(filepath, lines, content):
    out = []
    if not lines or not re.match(r"^#\s+Subproblem:\s*(\S+)", lines[0]):
        return [("ERROR", 1, "S0 summary.md must start with '# Subproblem: <id> <title>'.")]
    leaf = re.match(r"^#\s+Subproblem:\s*(\S+)", lines[0]).group(1)
    for pat in REQUIRED_LEAF_FIELDS:
        if not re.search(pat, content):
            out.append(("ERROR", 1, f"S0 missing the field matching '{pat}'."))
    sec = _section(lines, r"^##\s+Leaf Status Summary")
    if sec is None:
        return out + [("ERROR", 1, "S0 no '## Leaf Status Summary' section.")]
    later = [i for i, l in enumerate(lines) if re.match(r"^##\s", l) and i > sec[0]]
    if later:
        out.append(("ERROR", later[0] + 1, "S0 '## Leaf Status Summary' must be the last section of summary.md."))
    block = "".join(lines[sec[0]:sec[1]])
    for f in SUMMARY_FIELDS:
        if not _field(block, f):
            out.append(("ERROR", sec[0] + 1, f"S1 Leaf Status Summary is missing '**{f}**:'."))
    state = _field(block, "State")
    status_line = next((l for l in lines[:6] if "**Status**" in l), "")
    if state and status_line and _norm(state).split("(")[0].strip().split(";")[0][:20] not in _norm(status_line):
        out.append(("ERROR", sec[0] + 1, f"S2 the Status line does not match State '{state[:40]}'."))
    q, ans = _field(block, "Question"), _field(block, "Answer")
    master = _find_master(os.path.dirname(filepath))
    if q and master:
        ft = final_tree_question(open(master, encoding="utf-8").read(), leaf)
        if ft is None:
            out.append(("WARNING", sec[0] + 1, f"S3 no Final Tree bullet '**{leaf} [leaf] ...**' found in {os.path.basename(master)} to check the Question against."))
        elif _norm(ft) != _norm(q):
            out.append(("ERROR", sec[0] + 1, "S3 Question is not the leaf's Final Tree question word for word."))
    if q and ans and YESNO_START.match(_norm(q)) and not ANSWER_OPEN.match(_norm(ans)):
        out.append(("ERROR", sec[0] + 1, "S4 a yes/no Question needs an Answer that opens with Yes, No, Partly or Not known (the reply to the question as asked)."))
    return out


def check_kill_index(filepath, content):
    out = []
    base = os.path.dirname(os.path.abspath(filepath))
    for i, line in enumerate(content.splitlines(), 1):
        for m in re.finditer(r"\[\[(leaves/[^\]|#]+)\]\]", line):
            if not os.path.exists(os.path.join(base, m.group(1) + ".md")):
                out.append(("ERROR", i, f"K2 link [[{m.group(1)}]] has no card file."))
        if re.search(r"\[\[\.subproblems/", line):
            out.append(("WARNING", i, "K3 legacy link to .subproblems/: run `bs.py migrate`."))
    return out


def size_check(role, content, lines):
    out = []
    cap = CAPS.get(role)
    if cap and approx_tokens(content) > cap:
        out.append(("WARNING", 1, f"C0 about {approx_tokens(content)} tokens, over the {cap}-token cap for a {role}: compact it (briefs/compactor.md)."))
    if role == "library":
        for ln, title, text in _cards(lines, 0, len(lines), r"^###\s+L\d+"):
            if approx_tokens(title + text) > LIB_ENTRY_CAP:
                out.append(("WARNING", ln, f"C1 {title[:40]}...: about {approx_tokens(title + text)} tokens, over the {LIB_ENTRY_CAP}-token entry cap."))
        idx = _section(lines, r"^##\s+Index")
        if idx:
            for i in range(idx[0] + 1, idx[1]):
                if approx_tokens(lines[i]) > LIB_INDEX_LINE_CAP:
                    out.append(("WARNING", i + 1, f"C2 index line over {LIB_INDEX_LINE_CAP} tokens: one plain sentence per entry."))
    return out


def role_checks(filepath, lines, content, role):
    """Role-specific (level, line, message) findings."""
    out = []
    is_mode_b = "Master Leaf Synthesis Matrix" in content or "Stage 1.5" in content
    base = os.path.dirname(os.path.abspath(filepath))
    new_layout = os.path.isdir(os.path.join(base, "leaves"))
    if role == "master":
        req = [r"^##\s+Summary", r"^##\s+Invariants", r"^#{2,3}\s+Final Tree"]
        if not new_layout:
            req += [r"^##\s+Revision & Critic Log"]
        for pattern in req:
            if not re.search(pattern, content, re.MULTILINE):
                out.append(("ERROR", 1, f"Missing required core section matching: '{pattern}'"))
        if not re.search(r"^##\s+Kill Index", content, re.MULTILINE) and not os.path.exists(os.path.join(base, "kill_index.md")):
            out.append(("WARNING", 1, "K0 no kill_index.md (one row per killed or shelved route, linking its card)."))
        if not os.path.exists(os.path.join(base, "library.md")):
            out.append(("WARNING", 1, "K1 no library.md next to the front page: the findings library is required."))
        if not new_layout:
            out.append(("WARNING", 1, "L9 legacy layout (no leaves/): run `bs.py migrate` so documents stay within context."))
        elif not is_mode_b and not re.search(r"Analogy Map", content):
            pass
    elif role == "library":
        out += check_library(lines)
    elif role == "kill-index":
        out += check_kill_index(filepath, content)
    elif role == "leaf-summary":
        out += check_summary_file(filepath, lines, content)
    elif role == "leaf-problem":
        if not re.search(r"^#{2,4}\s+(1b\.\s+)?Analogy Map", content, re.MULTILINE):
            out.append(("WARNING", 1, "A0 no 'Analogy Map' heading: map the barrier to how other fields attacked the same structure before generating routes (ref/setup.md, Stage 2 item 3)."))
    elif role == "card":
        out += check_card_file(lines)
    elif role == "leaf":
        for pattern in REQUIRED_LEAF_FIELDS:
            if not re.search(pattern, content):
                out.append(("ERROR", 1, f"Leaf artifact is missing the field matching: '{pattern}'"))
        out += check_route_register(lines)
        if not re.search(r"^#{2,4}\s+(1b\.\s+)?Analogy Map", content, re.MULTILINE):
            out.append(("WARNING", 1, "A0 no 'Analogy Map' heading (ref/setup.md, Stage 2 item 3)."))
        out.append(("WARNING", 1, "L9 legacy single-file leaf: run `bs.py migrate` so documents stay within context."))
    out += size_check(role, content, lines)
    return out


def detect_role_legacy(filepath, content):
    """Legacy content-based role detection (kept for reference)."""
    if os.path.basename(filepath) == "library.md":
        return "library"
    if os.path.basename(filepath).startswith("brainstorm-") or re.search(r"^##\s+Invariants", content, re.MULTILINE):
        return "master"
    for line in content.splitlines():
        if line.startswith("#"):
            return "leaf" if re.match(r"^#\s+Subproblem:", line) else "other"
    return "other"


def fence_mask(lines):
    """Return (inside, opens, closes, unclosed): inside[i] is True for fence lines and their content."""
    inside = [False] * len(lines)
    opens, closes = [], []
    open_char, open_len, open_idx = None, 0, None
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if open_char is None:
            if m:
                open_char, open_len, open_idx = m.group(1)[0], len(m.group(1)), i
                opens.append(i)
                inside[i] = True
        else:
            inside[i] = True
            if m and m.group(1)[0] == open_char and len(m.group(1)) >= open_len and line.strip() == m.group(1):
                closes.append(i)
                open_char = None
    return inside, opens, closes, (open_idx if open_char is not None else None)


def check_markdown_structure(lines):
    """Return a list of (level, line_no, message) for Markdown-structure problems."""
    out = []
    inside, opens, closes, unclosed = fence_mask(lines)

    def blank(i):
        return i < 0 or i >= len(lines) or lines[i].strip() == ""

    if unclosed is not None:
        out.append(("ERROR", unclosed + 1, "M1 code fence opened here is never closed."))
    for i in opens:
        if not blank(i - 1) and not HEADING_RE.match(lines[i - 1]):
            out.append(("ERROR", i + 1, "M2 opening code fence must be preceded by a blank line (otherwise Obsidian merges it into the previous list item or paragraph and mis-pairs every later fence)."))
    for i in closes:
        if not blank(i + 1):
            out.append(("ERROR", i + 1, "M3 closing code fence must be followed by a blank line."))
    # M4 accidental indented code blocks
    if _MD is not None:
        for tok in _MD.parse("".join(lines)):
            if tok.type == "code_block" and tok.map:
                out.append(("ERROR", tok.map[0] + 1, "M4 text indented by 4+ spaces renders as a code block (dedent it, or attach it to a list item that is still open)."))
    else:
        list_ctx = False
        for i, line in enumerate(lines):
            if inside[i]:
                list_ctx = False
                continue
            if line.strip() == "":
                continue
            if LIST_RE.match(line) and not line.startswith("    "):
                list_ctx = True
            elif HEADING_RE.match(line) or TABLE_RE.match(line):
                list_ctx = False
            elif line.startswith("    ") and not list_ctx:
                out.append(("ERROR", i + 1, "M4 (heuristic) text indented by 4+ spaces outside a list renders as a code block."))
    # M5 tables need blank lines around them
    for i, line in enumerate(lines):
        if inside[i] or not TABLE_RE.match(line):
            continue
        prev_is_table = i > 0 and not inside[i - 1] and TABLE_RE.match(lines[i - 1])
        next_is_table = i + 1 < len(lines) and not inside[i + 1] and TABLE_RE.match(lines[i + 1])
        if not prev_is_table and not blank(i - 1) and not HEADING_RE.match(lines[i - 1]):
            out.append(("ERROR", i + 1, "M5 table must be preceded by a blank line."))
        if not next_is_table and not blank(i + 1):
            out.append(("ERROR", i + 1, "M5 table must be followed by a blank line (GFM otherwise swallows the next line into the table)."))
    # M6 inline code backticks
    for i, line in enumerate(lines):
        if inside[i]:
            continue
        if len(re.findall(r"(?<!\\)`", line)) % 2:
            out.append(("WARNING", i + 1, "M6 odd number of backticks: an inline code span is not closed."))
    # M7 inline math dollars, per paragraph (blank-line separated, outside fences)
    para_start, count = None, 0
    for i in range(len(lines) + 1):
        if i == len(lines) or inside[i] or lines[i].strip() == "":
            if para_start is not None and count % 2:
                out.append(("WARNING", para_start + 1, "M7 odd number of single '$' in this paragraph: an inline math span is not closed."))
            para_start, count = None, 0
            continue
        text = re.sub(r"`[^`]*`", "", lines[i])
        text = text.replace("\\$", "").replace("$$", "")
        count += text.count("$")
        if para_start is None:
            para_start = i
    return out


def check_file(filepath):
    """Lint one Markdown file; return 1 if it has errors, else 0."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return 1

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = "".join(lines)
    role = detect_role(filepath, content)
    errors = 0
    warnings = 0

    print(f"--- Linting: {filepath} (role: {role}) ---")

    # 1. Required structure per role
    for level, ln, msg in role_checks(filepath, lines, content, role):
        print(f"[{level}] Line {ln}: {msg}")
        errors += level == "ERROR"
        warnings += level == "WARNING"
    is_mode_b = "Master Leaf Synthesis Matrix" in content or "Stage 1.5" in content

    # 2. Zero Early Surrender
    for phrase in SURRENDER_PHRASES:
        if phrase in content.lower():
            print(f"[ERROR] Early surrender detected ('{phrase}'). Trigger Stage 3.3 Mutation/Replenishment.")
            errors += 1

    inside, _, _, _ = fence_mask(lines)

    # 3. Banned Subjective Scoring
    for idx, line in enumerate(lines, 1):
        for pat in BANNED_RATING_PATTERNS:
            if re.search(pat, line, re.IGNORECASE):
                print(f"[ERROR] Line {idx}: Subjective rating metric detected ('{line.strip()}'). Replace with binary falsification gates.")
                errors += 1

    # 4. Table cell pipes
    for idx, line in enumerate(lines, 1):
        if inside[idx - 1]:
            continue
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            for segment in stripped[1:-1].split("|"):
                if re.search(r"(?<!\\)\|", segment):
                    print(f"[WARNING] Line {idx}: Possible unescaped pipe '|' in table cell. Use '\\|' or enclose in backticks.")
                    warnings += 1

    # 5. LaTeX Syntax Hygiene (inline code spans are exempt: they hold literal code)
    for idx, line in enumerate(lines, 1):
        if inside[idx - 1]:
            continue
        prose = re.sub(r"`[^`]*`", "", line)
        for pat, msg in RAW_MATH_PATTERNS:
            if re.search(pat, prose):
                print(f"[WARNING] Line {idx}: {msg}")
                warnings += 1

    # 6. Markdown structure
    for level, line_no, msg in check_markdown_structure(lines):
        print(f"[{level}] Line {line_no}: {msg}")
        if level == "ERROR":
            errors += 1
        else:
            warnings += 1

    # 7. Invariants Completeness Check
    if role == "master" and "## Invariants" in content:
        if "The Hard Core" not in content or "The Unacceptable Outcome" not in content:
            print("[ERROR] '## Invariants' section exists but is missing required fields (Hard Core or Unacceptable Outcome).")
            errors += 1

    # 8. Subproblem Directory if Mode B is active
    subproblems_dir = os.path.join(os.path.dirname(filepath) or ".", ".subproblems")
    if role == "master" and is_mode_b and os.path.exists(subproblems_dir):
        sub_files = [f for f in os.listdir(subproblems_dir) if f.endswith(".md")]
        print(f"[INFO] Mode B detected. Found {len(sub_files)} subproblem artifacts in .subproblems/")

    print(f"Linting complete: {errors} error(s), {warnings} warning(s).")
    print("---------------------------------")
    return 1 if errors > 0 else 0


def collect(paths):
    """Expand directories into the Markdown files they contain."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".md")]
        else:
            files.append(p)
    return files


def parse_args(argv):
    """Return the list of paths to lint; print the docstring for -h/--help or no arguments."""
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    return argv[1:]


def main(argv):
    files = collect(parse_args(argv))
    if not files:
        print("[ERROR] No Markdown files found.")
        return 1
    failed = [f for f in files if check_file(f)]
    if len(files) > 1:
        print(f"SUMMARY: {len(files)} file(s) linted, {len(failed)} with errors.")
        for f in failed:
            print(f"  FAILED: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
