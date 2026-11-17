#!/usr/bin/env python3
"""
Tests for the route-card, Kill Index and findings-library checks of check_brainstorm.py.

    python3 tests/test_check_brainstorm.py          (from the skill directory)

Writes small leaf, library and master files to a temporary directory, lints them, and checks the
reported errors and warnings. Prints one line per test; exits non-zero on any failure.
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

KILLED_CARD = f"""### R1.0 — Local hop-ratio sign factors (KILLED)
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

LEAF_HEAD = """# Subproblem: P1.2 Test leaf
**Parent Chain**: P1 -> P1.2  **Status**: DORMANT

"""

SUMMARY = """## Leaf Status Summary
* **State**: DORMANT
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


def lint(path):
    r = subprocess.run([sys.executable, LINTER, path], capture_output=True, text=True)
    return r.returncode, r.stdout


class RouteCards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = os.path.join(self.tmp.name, "P1.2_test.md")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, body):
        with open(self.p, "w") as fh:
            fh.write(body)

    def test_complete_killed_card_passes(self):
        self.write(LEAF_HEAD + "## Route Register\n\n" + KILLED_CARD + "\n" + SUMMARY)
        rc, out = lint(self.p)
        self.assertEqual(rc, 0, out)
        self.assertIn("0 error(s), 0 warning(s)", out)

    def test_missing_kill_fields(self):
        card = KILLED_CARD.replace("* **Strongest case against the kill**: the toys are unprojected; answered by the projected re-check, which had no target.\n", "")
        self.write(LEAF_HEAD + "## Route Register\n\n" + card + "\n" + SUMMARY)
        rc, out = lint(self.p)
        self.assertEqual(rc, 1)
        self.assertIn("missing '**Strongest case against the kill**:'", out)

    def test_kill_needs_code(self):
        card = KILLED_CARD.replace("KILLED at T1", "KILLED by the toy run")
        self.write(LEAF_HEAD + "## Route Register\n\n" + card + "\n" + SUMMARY)
        rc, out = lint(self.p)
        self.assertIn("R2", out)

    def test_short_idea_warns(self):
        card = KILLED_CARD.replace(IDEA, "FRD-K on hole hops.")
        self.write(LEAF_HEAD + "## Route Register\n\n" + card + "\n" + SUMMARY)
        rc, out = lint(self.p)
        self.assertIn("R3", out)

    def test_shelved_and_live_cards(self):
        shelved = ("### R2.0 — Bigger ladder (SHELVED)\n* **The idea**: " + IDEA + "\n* **Lineage**: Operation B.\n"
                   "* **Outcome**: SHELVED (G5 budget)\n* **Why shelved**: 400 CPU-h against 120 left.\n"
                   "* **What would unshelve it**: a leaf budget of 500 CPU-h.\n")
        live = "### R3.0 — Something live (LIVE)\n* **The idea**: " + IDEA + "\n* **Lineage**: Operation A from R1.0.\n* **Outcome**: LIVE (T1)\n"
        self.write(LEAF_HEAD + "## Route Register\n\n" + shelved + "\n" + live + "\n" + SUMMARY)
        rc, out = lint(self.p)
        self.assertEqual(rc, 0, out)

    def test_register_must_precede_summary(self):
        self.write(LEAF_HEAD + SUMMARY + "\n## Route Register\n\n" + KILLED_CARD)
        rc, out = lint(self.p)
        self.assertIn("must come before '## Leaf Status Summary'", out)

    def test_missing_register_warns(self):
        self.write(LEAF_HEAD + SUMMARY)
        rc, out = lint(self.p)
        self.assertIn("R0 leaf has no '## Route Register'", out)


class Library(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = os.path.join(self.tmp.name, "library.md")

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_library_passes(self):
        open(self.p, "w").write(LIBRARY)
        rc, out = lint(self.p)
        self.assertIn("(role: library)", out)
        self.assertIn("0 error(s), 0 warning(s)", out)

    def test_missing_field_and_bad_status(self):
        bad = LIBRARY.replace("* **How to use it**: compare any new sign layer against matched width first.\n", "")
        bad = bad.replace("PROVISIONAL (2026", "LIKELY (2026")
        open(self.p, "w").write(bad)
        rc, out = lint(self.p)
        self.assertEqual(rc, 1)
        self.assertIn("missing '**How to use it**:'", out)
        self.assertIn("L2", out)


class Master(unittest.TestCase):
    def test_master_needs_kill_index_and_library(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "brainstorm-x.md")
            open(p, "w").write("# B\n## Summary\nx\n## Invariants\nThe Hard Core. The Unacceptable Outcome.\n"
                               "## Stage 1\nx\n## Stage 1.5\nx\n## Master Leaf Synthesis Matrix\nx\n"
                               "## Revision & Critic Log\nx\n")
            rc, out = lint(p)
            self.assertIn("K0", out)
            self.assertIn("K1", out)
            open(p, "a").write("## Kill Index\nx\n")
            open(os.path.join(d, "library.md"), "w").write(LIBRARY)
            rc, out = lint(p)
            self.assertNotIn("K0", out)
            self.assertNotIn("K1", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
