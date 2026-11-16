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

The document role is detected per file:
    * master  : file name starts with "brainstorm-", or the file contains "## Invariants".
                Full structural checks (required core sections, Mode A stages / Mode B matrix,
                invariants completeness) plus all hygiene checks.
    * leaf    : the first heading starts with "# Subproblem:".
                Leaf schema checks ("**Parent Chain**" and "**Status**" present) plus all hygiene
                checks. The master-only "missing core section" errors are not raised.
    * other   : any other Markdown file (briefs, notes). Hygiene checks only.

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


def detect_role(filepath, content):
    """Classify a Markdown file as 'master', 'leaf' or 'other'."""
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
    is_mode_b = "Master Leaf Synthesis Matrix" in content or "Stage 1.5" in content
    if role == "master":
        for pattern in REQUIRED_CORE_SECTIONS:
            if not re.search(pattern, content, re.MULTILINE):
                print(f"[ERROR] Missing required core section matching: '{pattern}'")
                errors += 1
        if not is_mode_b:
            for stage in [r"^##\s+Stage 2", r"^##\s+Stage 3", r"^##\s+Stage 4"]:
                if not re.search(stage, content, re.MULTILINE):
                    print(f"[ERROR] Mode A active, but missing stage: '{stage}'")
                    errors += 1
        elif "Master Leaf Synthesis Matrix" not in content and "Leaf Comparison Matrix" not in content:
            print("[WARNING] Mode B indicated, but no Master Leaf Matrix found in document.")
            warnings += 1
    elif role == "leaf":
        for pattern in REQUIRED_LEAF_FIELDS:
            if not re.search(pattern, content):
                print(f"[ERROR] Leaf artifact is missing the field matching: '{pattern}'")
                errors += 1

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
