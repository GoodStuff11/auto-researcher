#!/usr/bin/env python3
"""
Tests for check_brainstorm.py on the leaves/ layout: route-card files, leaf summaries (Question against the
Final Tree, Answer opening, Status/State agreement), problem files, the Kill Index, the findings library,
size caps, and the legacy-layout warning.

    python3 tests/test_check_brainstorm.py          (from the skill directory)

Builds small working directories in a temporary directory, lints them, and checks the reported errors
and warnings. Prints one line per test; exits non-zero on any failure.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
LINTER = os.path.join(os.path.dirname(HERE), "check_brainstorm.py")

IDEA = ("Replace the single backflow determinant by a product of local sign factors driven by the "
        "reference state's own hopping ratios, so that wrong signs near holes can flip without "
        "changing the amplitude network; it should help because the residual sign error sits next to holes.")

CARD = f"""# R1.0 — Local hop-ratio sign factors (KILLED)
* **The idea**: {IDEA}
* **Lineage**: initial batch.
* **Outcome**: KILLED at T1 (2026-11-16)
* **Kill rule**: KILL if the sign error ratio against matched-width backflow exceeds 1.0 at two sizes.
* **Evidence**: ratio 1.4 +- 0.1 at 12 sites and 2.3 +- 0.2 at 16 sites against matched-width backflow (job 1).
* **Why this kills it**: both counted sizes exceed the threshold by more than three standard errors.
* **What the kill does not rule out**: projected states, and runs longer than 300 steps.
* **Strongest case against the kill**: the toys are unprojected; answered by the projected re-check, which had no target.
* **Checked by**: per-leaf critic, 2026-11-16.
* **Lesson**: L1.
"""

QUESTION = "Can a translation-equivariant network trained at $L$ initialize $2L$ with a converged energy error?"

MASTER = f"""# Brainstorm: test
## Summary
x
## Invariants
The Hard Core. The Unacceptable Outcome.
## Stage 1 — Decomposition
### Final Tree
* **P4.2 [leaf] — Size-transferable networks.** {QUESTION}
## Stage 1.5 — Routing
Mode B. Master Leaf Synthesis Matrix lives in the current leaf state table.
"""


def summary(question=QUESTION, answer="Not known. None of the transferable designs got close.", state="DORMANT",
            status="DORMANT", tail=""):
    return f"""# Subproblem: P4.2 Size-transferable networks
**Parent Chain**: P4 -> P4.2  **Status**: {status}

## Route Portfolio

| Route | Tier reached | State | Blocked on | Pending kill rule | Next action on each verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
{tail}
## Leaf Status Summary
* **State**: {state}
* **Updated**: 2026-11-17 (after the capacity test)
* **Question**: {question}
* **Answer**: {answer}
* **What we showed**: every size-independent design stalled at about 6% error on the 16-site target.
* **What it would take**: a transferable design that converges when trained on the target itself.
"""


LIBRARY = """# Findings Library: test

## Index

* L1 (PROVISIONAL): Width beats sign layers on unprojected toys.

## About methods

### L1. On unprojected toys, spending parameters on width beats every tested sign layer.
* **Status**: PROVISIONAL (2026-11-16)
* **Applies to**: sign-correcting layers on top of determinant-based variational states.
* **The lesson**: a sign layer competes for the same parameters as width; on these toys width wins.
* **Evidence**: three routes killed at T1, ratios 1.4 to 2.3 against matched width.
* **Scope and limits**: unprojected toys up to 12 sites, 300 optimisation steps.
* **Counter-evidence and open questions**: one projected 16-site instance gave ratio 0.84.
* **How to use it**: compare any new sign layer against matched width first.
* **Sources**: P1.2 R1.0, R2.1, R3.4.
"""

PROBLEM = """# Problem: P4.2 Size-transferable networks

### 1. Mathematical Abstraction & Barriers
* **Formulation**: x

### 1b. Analogy Map
* Barrier restated: reuse a fitted model on a larger domain. Sources: multigrid, transfer learning, renormalisation.
"""


def lint(path):
    r = subprocess.run([sys.executable, LINTER, path], capture_output=True, text=True)
    return r.returncode, r.stdout


class Layout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wd = self.tmp.name
        self.leaf = os.path.join(self.wd, "leaves", "P4.2")
        os.makedirs(os.path.join(self.leaf, "cards"))
        os.makedirs(os.path.join(self.leaf, "log"))
        self.w("brainstorm-test.md", MASTER)
        self.w("library.md", LIBRARY)
        self.w("kill_index.md", "# Kill Index\n\n| Route | Card |\n| :--- | :--- |\n| R1.0 | [[leaves/P4.2/cards/R1.0]] |\n")
        self.w("leaves/P4.2/summary.md", summary())
        self.w("leaves/P4.2/problem.md", PROBLEM)
        self.w("leaves/P4.2/cards/R1.0.md", CARD)

    def tearDown(self):
        self.tmp.cleanup()

    def w(self, rel, text):
        with open(os.path.join(self.wd, rel), "w") as fh:
            fh.write(text)

    def test_clean_directory_passes(self):
        rc, out = lint(self.wd)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("[WARNING]", out, out)
        for role in ("master", "library", "kill-index", "leaf-summary", "leaf-problem", "card"):
            self.assertIn(f"(role: {role})", out)

    def test_question_must_match_final_tree(self):
        self.w("leaves/P4.2/summary.md", summary(question="Can a network trained on 8 sites start well on 16 sites?"))
        rc, out = lint(os.path.join(self.leaf, "summary.md"))
        self.assertIn("S3 Question is not the leaf's Final Tree question", out)

    def test_answer_must_open_with_direct_reply(self):
        self.w("leaves/P4.2/summary.md", summary(answer="For the networks we built, going from 8 to 16 sites did not work."))
        rc, out = lint(os.path.join(self.leaf, "summary.md"))
        self.assertIn("S4", out)

    def test_status_must_match_state(self):
        self.w("leaves/P4.2/summary.md", summary(status="OPEN"))
        rc, out = lint(os.path.join(self.leaf, "summary.md"))
        self.assertIn("S2", out)

    def test_summary_fields_and_last_section(self):
        s = summary().replace("* **What it would take**: a transferable design that converges when trained on the target itself.\n", "")
        self.w("leaves/P4.2/summary.md", s + "\n## Notes\nx\n")
        rc, out = lint(os.path.join(self.leaf, "summary.md"))
        self.assertIn("missing '**What it would take**:'", out)
        self.assertIn("must be the last section", out)

    def test_summary_cap(self):
        self.w("leaves/P4.2/summary.md", summary(tail="".join(f"| R{i} | T1 | LIVE | — | {'x' * 120} | y |\n" for i in range(80))))
        rc, out = lint(os.path.join(self.leaf, "summary.md"))
        self.assertIn("C0", out)

    def test_card_file_checks(self):
        self.w("leaves/P4.2/cards/R1.0.md", CARD.replace("KILLED at T1", "KILLED by a toy run"))
        rc, out = lint(os.path.join(self.leaf, "cards", "R1.0.md"))
        self.assertIn("R2", out)
        self.w("leaves/P4.2/cards/R1.0.md", CARD.replace("* **Lineage**: initial batch.", "* **Lineage**: cross-field translation from multigrid."))
        rc, out = lint(os.path.join(self.leaf, "cards", "R1.0.md"))
        self.assertIn("R4", out)
        self.w("leaves/P4.2/cards/R1.0.md", "* **The idea**: no title line\n")
        rc, out = lint(os.path.join(self.leaf, "cards", "R1.0.md"))
        self.assertIn("must start with '# R<id>", out)

    def test_kill_index_links_resolve(self):
        self.w("kill_index.md", "# Kill Index\n\n| Route | Card |\n| :--- | :--- |\n| R9 | [[leaves/P4.2/cards/R9]] |\n")
        rc, out = lint(os.path.join(self.wd, "kill_index.md"))
        self.assertIn("K2 link [[leaves/P4.2/cards/R9]] has no card file", out)

    def test_problem_needs_analogy_map(self):
        self.w("leaves/P4.2/problem.md", PROBLEM.replace("### 1b. Analogy Map", "### 1b. Notes"))
        rc, out = lint(os.path.join(self.leaf, "problem.md"))
        self.assertIn("A0", out)

    def test_library_fields_and_caps(self):
        bad = LIBRARY.replace("* **How to use it**: compare any new sign layer against matched width first.\n", "")
        bad = bad.replace("PROVISIONAL (2026", "LIKELY (2026")
        bad = bad.replace("* **Sources**: P1.2 R1.0, R2.1, R3.4.", "* **Sources**: " + "long " * 700)
        self.w("library.md", bad)
        rc, out = lint(os.path.join(self.wd, "library.md"))
        self.assertIn("missing '**How to use it**:'", out)
        self.assertIn("L2", out)
        self.assertIn("C1", out)

    def test_logs_are_archives(self):
        self.w("leaves/P4.2/log/round-01.md", "# Log\n" + "long text " * 5000 + "\n")
        rc, out = lint(os.path.join(self.leaf, "log", "round-01.md"))
        self.assertIn("(role: archive)", out)
        self.assertNotIn("C0", out)

    def test_legacy_leaf_warns(self):
        p = os.path.join(self.wd, "old.md")
        with open(p, "w") as fh:
            fh.write("# Subproblem: P1.2 Old\n**Parent Chain**: P1 -> P1.2  **Status**: OPEN\n\n### 1b. Analogy Map\nx\n\n## Route Register\n\n### "
                     + CARD[2:] + "\n## Leaf Status Summary\n* **State**: OPEN\n")
        rc, out = lint(p)
        self.assertIn("L9 legacy single-file leaf", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
