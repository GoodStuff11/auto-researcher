#!/usr/bin/env python3
"""
Tests for bs.py: scaffolding (new-leaf, new-card), reading by id (state, context, show, grep, sizes), and
the lossless migration of a legacy working directory.

    python3 tests/test_bs.py          (from the skill directory)

Uses temporary directories only. Prints one line per test; exits non-zero on any failure.
"""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
sys.path.insert(0, SKILL)
import bs  # noqa: E402

Q = "Can a network trained at $L$ start $2L$ near convergence?"
IDEA = ("Rebuild the pair function on each lattice from a few transferred parameters and learn only a "
        "short-range residual, so that the transferred part is exactly size independent and the residual is "
        "small enough to carry over unchanged between cluster sizes.")

LEGACY_MASTER = f"""# Brainstorm: legacy
## Summary
The answers.
## Invariants
### 2. The Hard Core
Sign structure.
### 3. The Unacceptable Outcome (Failure Threshold)
x
## Stage 1 — Survey
### Pass 1: Pool
old pool
### Final Tree
* **P4.2 [leaf] — Size transfer.** {Q}
## Stage 1.5 — Routing
Mode B.
## Master Leaf Synthesis Matrix
### Current leaf state (today)
| Leaf | State |
| :--- | :--- |
| P4.2 | DORMANT |

### Round-1 matrix (historical snapshot)
old matrix text
## Kill Index
| Route | Card |
| :--- | :--- |
| R1 | [[.subproblems/P4.2_size_transfer#R1 — Residual geminal (KILLED)\\|R1]] |

## Revision & Critic Log
* 2026-11-10: things happened.
"""

LEGACY_LEAF = f"""# Subproblem: P4.2 Size transfer
**Parent Chain**: P4 -> P4.2  **Status**: DORMANT

### 1. Mathematical Abstraction & Barriers
* **Formulation**: transfer.

### 1b. Analogy Map
* multigrid, transfer learning, renormalisation.

### Round 2 — Mutation (2026-11-11)
round two narrative UNIQUE-ROUND-TEXT

## Route Register

### R1 — Residual geminal (KILLED)
* **The idea**: {IDEA}
* **Lineage**: initial batch.
* **Outcome**: KILLED at G3 (2026-11-11)

### R2.1″ — Seam continuation (KILLED)
* **The idea**: x

### R1-L — Hyphenated id (KILLED)
* **The idea**: {IDEA}
* **Lineage**: Operation B.
* **Outcome**: KILLED at T1

## Leaf Status Summary
* **State**: DORMANT
* **Question**: {Q}
* **Answer**: Not known. Nothing transferable converged.
"""


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bs.main(argv)
    return rc, buf.getvalue()


class Scaffold(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wd = self.tmp.name
        open(os.path.join(self.wd, "brainstorm-x.md"), "w").write(
            "# B\n## Summary\nx\n## Invariants\nThe Hard Core. The Unacceptable Outcome.\n### Final Tree\n"
            f"* **P4.2 [leaf] — Size transfer.** {Q}\n")
        open(os.path.join(self.wd, "library.md"), "w").write("# L\n\n## Index\n\n* L1 (PROVISIONAL): a lesson.\n\n## About methods\n\n### L1. A lesson.\n* **Status**: PROVISIONAL\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_leaf_new_card_state_context_show(self):
        rc, out = run(["new-leaf", self.wd, "P4.2", "--title", "Size transfer", "--parent", "P4", "--question", Q])
        self.assertEqual(rc, 0, out)
        rc, out = run(["new-card", self.wd, "P4.2", "R2.1″", "--name", "Seam continuation"])
        self.assertEqual(rc, 0, out)
        self.assertTrue(os.path.exists(os.path.join(self.wd, "leaves", "P4.2", "cards", "R2.1pp.md")))
        rc, out = run(["state", self.wd])
        self.assertIn("| P4.2 | OPEN | Not known.", out)
        rc, out = run(["context", self.wd, "P4.2", "--role", "worker"])
        for piece in ("Invariants", "Final Tree bullet", "library index", "summary.md", "problem.md", "card R2.1pp"):
            self.assertIn(piece, out)
        rc, out = run(["show", self.wd, "R2.1″"])
        self.assertIn("Seam continuation", out)
        rc, out = run(["show", self.wd, "L1"])
        self.assertIn("A lesson", out)
        lint = subprocess.run([sys.executable, os.path.join(SKILL, "check_brainstorm.py"),
                               os.path.join(self.wd, "leaves", "P4.2", "summary.md")], capture_output=True, text=True)
        self.assertIn("0 error(s)", lint.stdout, lint.stdout)                      # template passes S0-S4


class Migrate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wd = self.tmp.name
        os.makedirs(os.path.join(self.wd, ".subproblems"))
        open(os.path.join(self.wd, "brainstorm-legacy.md"), "w").write(LEGACY_MASTER)
        open(os.path.join(self.wd, ".subproblems", "P4.2_size_transfer.md"), "w").write(LEGACY_LEAF)
        open(os.path.join(self.wd, ".subproblems", ".round2_brief.md"), "w").write("brief\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_migrate_is_lossless_and_rewrites_links(self):
        rc, out = run(["migrate", self.wd])
        self.assertEqual(rc, 0, out)
        L = os.path.join(self.wd, "leaves", "P4.2")
        summ = open(os.path.join(L, "summary.md")).read()
        self.assertTrue(summ.startswith("# Subproblem: P4.2"))
        self.assertIn("## Leaf Status Summary", summ)
        self.assertNotIn("UNIQUE-ROUND-TEXT", summ)
        self.assertIn("Analogy Map", open(os.path.join(L, "problem.md")).read())
        self.assertTrue(open(os.path.join(L, "cards", "R1.md")).read().startswith("# R1 — Residual geminal (KILLED)"))
        self.assertTrue(os.path.exists(os.path.join(L, "cards", "R2.1pp.md")))
        self.assertTrue(os.path.exists(os.path.join(L, "cards", "R1-L.md")))  # hyphenated ids stay whole
        self.assertIn("UNIQUE-ROUND-TEXT", open(os.path.join(L, "log", "legacy.md")).read())
        self.assertEqual(open(os.path.join(L, "log", "original.md")).read(), LEGACY_LEAF)  # nothing lost
        front = open(os.path.join(self.wd, "brainstorm-legacy.md")).read()
        for keep in ("## Summary", "## Invariants", "### Final Tree", "### Current leaf state", "## Stage 1.5"):
            self.assertIn(keep, front)
        for gone in ("old pool", "old matrix text", "things happened", "## Kill Index"):
            self.assertNotIn(gone, front)
        hist = open(os.path.join(self.wd, "history.md")).read()
        for kept in ("old pool", "old matrix text", "things happened"):
            self.assertIn(kept, hist)
        self.assertIn("[[leaves/P4.2/cards/R1]]", open(os.path.join(self.wd, "kill_index.md")).read())
        self.assertEqual(open(os.path.join(self.wd, "log", "master-original.md")).read(), LEGACY_MASTER)
        self.assertTrue(os.path.exists(os.path.join(self.wd, "log", "subproblems", ".round2_brief.md")))
        self.assertFalse(os.path.exists(os.path.join(self.wd, ".subproblems")))
        rc, out = run(["grep", self.wd, "UNIQUE-ROUND-TEXT"])
        self.assertIn("leaves/P4.2/log/legacy.md", out)
        rc, out = run(["migrate", self.wd])
        self.assertEqual(rc, 1)                                                   # refuses a second run

    def test_dry_run_writes_nothing(self):
        rc, out = run(["migrate", self.wd, "--dry-run"])
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(os.path.join(self.wd, "leaves")))
        self.assertIn("| P4.2 |", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
