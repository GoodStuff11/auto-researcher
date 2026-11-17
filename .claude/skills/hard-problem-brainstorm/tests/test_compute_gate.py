#!/usr/bin/env python3
"""
Tests for compute_gate.py. Run from anywhere:

    python3 tests/test_compute_gate.py            (from the skill directory)

Uses fake sbatch / sacct / scancel scripts in a temporary directory; no cluster access is needed
and nothing is submitted. Prints one line per check and exits non-zero on any failure.
"""
import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
sys.path.insert(0, SKILL)
import compute_gate as cg  # noqa: E402

FULL_VOI = """# VOI note
* **leaf**: P2.1
* **route**: R2
* **tier**: T2
* **decision**: PASS -> promote to T3; FAIL -> kill R2 and refill
* **prediction**: FAIL
* **p**: {p}
* **prediction-changes-decision**: {pcd}
* **cheapest-kill-tried**: Gate 4 op count, passes (O(N_e^2))
* **cost-cpuh**: {cost}
* **cheapest-version**: one seed, one instance, smallest size with the competitor
* **convergence**: pilot energy flat to 1e-4 by step 600 of 1000
* **wall-time-basis**: 1.8 s/step x 1000 steps x 1.3
* **power**: P(AMBIGUOUS) about 0.1 from the 3-seed spread
{extra}"""


def sbatch_text(time="01:00:00", cpus=4, array=None, extra=""):
    lines = ["#!/bin/bash", f"#SBATCH --time={time}", f"#SBATCH --cpus-per-task={cpus}", "#SBATCH --ntasks=1"]
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines.append(extra)
    lines.append("python3 micro_test.py")
    return "\n".join(lines) + "\n"


def make_bin(d, name, body):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.write("#!/bin/bash\n" + body + "\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return p


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cg.main(argv)
    return rc, buf.getvalue()


class Parsers(unittest.TestCase):
    def test_walltime(self):
        self.assertAlmostEqual(cg.parse_walltime_hours("30"), 0.5)
        self.assertAlmostEqual(cg.parse_walltime_hours("90:00"), 1.5)
        self.assertAlmostEqual(cg.parse_walltime_hours("02:30:00"), 2.5)
        self.assertAlmostEqual(cg.parse_walltime_hours("1-00"), 24)
        self.assertAlmostEqual(cg.parse_walltime_hours("1-02:30"), 26.5)
        self.assertAlmostEqual(cg.parse_walltime_hours("1-00:00:36"), 24.01)

    def test_array(self):
        self.assertEqual(cg.count_array_tasks("0-9"), 10)
        self.assertEqual(cg.count_array_tasks("0-9%4"), 10)
        self.assertEqual(cg.count_array_tasks("1,3,5-7"), 5)
        self.assertEqual(cg.count_array_tasks("0-15:2"), 8)

    def test_sbatch_forms(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.sbatch")
            with open(p, "w") as fh:
                fh.write("#!/bin/bash\n#SBATCH --time 02:00:00\n#SBATCH -c 8\n#SBATCH -a 0-84\n#SBATCH -n2\n")
            sb = cg.parse_sbatch(p)
            self.assertEqual((sb["cpus"], sb["tasks"], sb["array_tasks"]), (8, 2, 85))
            self.assertAlmostEqual(sb["cap_cpuh"], 85 * 2 * 8 * 2)  # the 526080-sized array: 2720 CPU-h

    def test_no_time_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.sbatch")
            with open(p, "w") as fh:
                fh.write("#!/bin/bash\n#SBATCH -c 8\n")
            with self.assertRaises(ValueError):
                cg.parse_sbatch(p)


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.wd = os.path.join(self.d, "work")
        self.td = os.path.join(self.wd, "tests", "P2.1_r1")
        os.makedirs(self.td)
        rc, _ = run(["init", self.wd, "--slug", "demo", "--total-cpuh", "500", "--per-leaf-cpuh", "150",
                     "--leaf", "P4.1=300"])
        self.assertEqual(rc, 0)
        self.log = os.path.join(self.d, "calls.log")
        self.sbatch = make_bin(self.d, "sbatch", f'echo "$@" >> {self.log}; echo $((700000 + RANDOM % 1000))')
        self.scancel = make_bin(self.d, "scancel", f'echo "scancel $@" >> {self.log}')

    def tearDown(self):
        self.tmp.cleanup()

    def files(self, p=0.6, pcd="yes", cost=5, extra="", audit=False, **sb):
        """Write submit.sbatch and voi.md. audit=True appends the auditor line with the note's real digest."""
        s = os.path.join(self.td, "submit.sbatch")
        v = os.path.join(self.td, "voi.md")
        with open(s, "w") as fh:
            fh.write(sbatch_text(**sb))
        with open(v, "w") as fh:
            fh.write(FULL_VOI.format(p=p, pcd=pcd, cost=cost, extra=extra))
        if audit:
            dg = cg.voi_digest(v)
            with open(v, "a") as fh:
                fh.write(f"\n* **auditor**: APPROVED 2026-11-16 {dg}\n")
        return s, v

    def check(self, s, v):
        return run(["check", self.wd, s, "--voi", v])

    def test_pass_small(self):
        s, v = self.files()                            # 1 h x 4 cpus = 4 CPU-h
        rc, out = self.check(s, v)
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE: PASS", out)

    def test_g1_missing(self):
        s, v = self.files()
        txt = open(v).read().replace("* **convergence**: pilot energy flat to 1e-4 by step 600 of 1000", "* **convergence**: TODO")
        open(v, "w").write(txt)
        rc, out = self.check(s, v)
        self.assertEqual(rc, 1)
        self.assertIn("G1 VOI field 'convergence'", out)

    def test_g2_predicted(self):
        s, v = self.files(p=0.85, pcd="no")
        rc, out = self.check(s, v)
        self.assertEqual(rc, 1)
        self.assertIn("G2 predicted, don't run", out)
        s, v = self.files(p=0.85, pcd="yes")           # confident but decision-relevant: allowed
        self.assertEqual(self.check(s, v)[0], 0)

    def test_g3_declared(self):
        s, v = self.files(cost=2)                      # cap 4 > 1.05 x 2
        rc, out = self.check(s, v)
        self.assertIn("G3 computed cap", out)

    def test_g4_tiers(self):
        s, v = self.files(cost=50, time="12:30:00")    # 50 CPU-h
        rc, out = self.check(s, v)
        self.assertIn("G4 cap 50.0 > 10", out)
        s, v = self.files(cost=50, time="12:30:00", audit=True)
        self.assertEqual(self.check(s, v)[0], 0)
        s, v = self.files(cost=120, time="30:00:00", audit=True)
        rc, out = self.check(s, v)
        self.assertIn("needs 'passed-t2", out)
        s, v = self.files(cost=1200, time="300:00:00", extra="* **passed-t2**: job 1", audit=True)
        rc, out = self.check(s, v)
        self.assertIn("needs 'user-approval", out)

    def test_g5_budget(self):
        s, v = self.files(cost=160, time="40:00:00",
                          extra="* **passed-t2**: job 1", audit=True)  # 160 > leaf 150
        rc, out = self.check(s, v)
        self.assertIn("G5 leaf P2.1 budget", out)

    def test_g6_streak(self):
        rows = cg.load_ledger(self.wd)
        for i in range(3):
            rows.append({"leaf": "P2.1", "job_id": str(i), "job_name": f"x{i}", "state": "done",
                         "cap_cpuh": "1", "used_cpuh": "1", "changed_decision": "no", "p": "0.5",
                         "prediction_hit": "na", "notes": "", "script": "", "voi": ""})
        cg.save_ledger(self.wd, rows)
        s, v = self.files()
        rc, out = self.check(s, v)
        self.assertIn("G6 leaf P2.1: last 3 finished jobs changed no decision", out)
        s, v = self.files(extra="* **reopen-justification**: new ingredient X", audit=True)
        self.assertEqual(self.check(s, v)[0], 0)

    def test_calibration_threshold(self):
        rows = cg.load_ledger(self.wd)
        for i, hit in enumerate(["yes", "yes", "no", "no", "no"]):  # overconfident: 2/5
            rows.append({"leaf": "P9", "job_id": str(i), "job_name": f"x{i}", "state": "done", "cap_cpuh": "1",
                         "used_cpuh": "1", "changed_decision": "yes", "p": "0.85", "prediction_hit": hit,
                         "notes": "", "script": "", "voi": ""})
        cg.save_ledger(self.wd, rows)
        s, v = self.files(p=0.85, pcd="no")            # base 0.8 would refuse; raised 0.9 lets it run
        rc, out = self.check(s, v)
        self.assertEqual(rc, 0, out)
        self.assertIn("raised to 0.9", out)

    def test_submit_dup_refresh_cancel_report(self):
        s, v = self.files()
        rc, out = run(["submit", self.wd, s, "--voi", v, "--sbatch-bin", self.sbatch])
        self.assertEqual(rc, 0, out)
        row = cg.load_ledger(self.wd)[0]
        self.assertEqual(row["job_name"], "hpb-demo-P2.1-R2-001")
        self.assertIn("--job-name=hpb-demo-P2.1-R2-001", open(self.log).read())
        # G7: same script + VOI while pending
        rc, out = run(["submit", self.wd, s, "--voi", v, "--sbatch-bin", self.sbatch])
        self.assertEqual(rc, 1)
        self.assertIn("G7 duplicate", out)
        # refresh: one matching TIMEOUT task, one reused-id row from another project (must be ignored)
        jid = row["job_id"]
        sacct = make_bin(self.d, "sacct",
                         f'echo "{jid}_0|hpb-demo-P2.1-R2-001|TIMEOUT|4|3600"\n'
                         f'echo "{jid}|pysr_chunk_0|COMPLETED|64|360000"')
        rc, out = run(["refresh", self.wd, "--sacct-bin", sacct])
        row = cg.load_ledger(self.wd)[0]
        self.assertEqual(row["used_cpuh"], "4.00")          # 4 cpus x 1 h; the 6400 CPU-h impostor excluded
        self.assertEqual(row["state"], "done")
        self.assertIn("wall-time miss", row["notes"])
        rc, out = run(["record", self.wd, jid, "--verdict", "FAIL", "--changed", "yes", "--hit", "yes"])
        self.assertEqual(rc, 0)
        # cancel a second job
        s2 = os.path.join(self.td, "submit2.sbatch")
        open(s2, "w").write(sbatch_text(time="00:30:00"))
        rc, out = run(["submit", self.wd, s2, "--voi", v, "--sbatch-bin", self.sbatch])
        self.assertEqual(rc, 0, out)
        jid2 = cg.load_ledger(self.wd)[1]["job_id"]
        self.assertEqual(cg.load_ledger(self.wd)[1]["job_name"], "hpb-demo-P2.1-R2-002")
        rc, out = run(["cancel", self.wd, jid2, "--reason", "verdict decided from data on disk",
                       "--scancel-bin", self.scancel])
        self.assertEqual(rc, 0)
        row2 = cg.load_ledger(self.wd)[1]
        self.assertEqual((row2["state"], row2["changed_decision"]), ("cancelled", "cancelled"))  # kept out of G6
        # report is lint-clean Markdown
        rc, out = run(["report", self.wd])
        self.assertEqual(rc, 0)
        md = os.path.join(self.d, "ledger_report.md")
        open(md, "w").write(out)
        lint = subprocess.run([sys.executable, os.path.join(SKILL, "check_brainstorm.py"), md],
                              capture_output=True, text=True)
        self.assertIn("0 error(s), 0 warning(s)", lint.stdout, lint.stdout)
        self.assertIn("Wall-time misses", out)

    def test_no_budget_refused(self):
        wd = os.path.join(self.d, "nobudget")
        os.makedirs(wd)
        s, v = self.files()
        rc, out = run(["check", wd, s, "--voi", v])
        self.assertEqual(rc, 1)
        self.assertIn("run `compute_gate.py init`", out)


class SecondPass(Gate):
    """Quick notes, digest-bound approvals, GPU cost, crash handling and reconcile."""

    def test_quick_note_for_tiny_jobs(self):
        s = os.path.join(self.td, "tiny.sbatch")
        v = os.path.join(self.td, "tiny_voi.md")
        open(s, "w").write(sbatch_text(time="00:10:00", cpus=2))       # 1/3 CPU-h
        open(v, "w").write("* leaf: P2.1\n* route: R9\n* tier: T1\n* decision: KILL or keep\n"
                           "* prediction: KILL\n* p: 0.6\n* prediction-changes-decision: yes\n")
        rc, out = self.check(s, v)
        self.assertEqual(rc, 0, out)
        self.assertIn("quick note", out)
        open(s, "w").write(sbatch_text(time="00:40:00", cpus=2))       # 1.33 CPU-h: full note needed
        rc, out = self.check(s, v)
        self.assertIn("G1 VOI field 'convergence'", out)

    def test_digest_voided_by_edit(self):
        s, v = self.files(cost=50, time="12:30:00", audit=True)
        self.assertEqual(self.check(s, v)[0], 0)
        txt = open(v).read().replace("* **p**: 0.6", "* **p**: 0.5")    # edit after approval
        open(v, "w").write(txt)
        rc, out = self.check(s, v)
        self.assertEqual(rc, 1)
        self.assertIn("G4 cap 50.0 > 10", out)
        rc, out = run(["digest", v])
        self.assertEqual(len(out.strip()), 12)

    def test_gpu_cost(self):
        self.assertEqual(cg.gres_gpus("gpu:t4:2"), 2)
        self.assertEqual(cg.gres_gpus("gpu"), 1)
        self.assertEqual(cg.gres_gpus("shard:1"), 0)
        self.assertEqual(cg.gpus_opt("a100:1,t4:1"), 2)
        s = os.path.join(self.td, "g.sbatch")
        open(s, "w").write("#!/bin/bash\n#SBATCH --time=02:00:00\n#SBATCH --cpus-per-task=1\n#SBATCH --gres=gpu:1\n")
        sb = cg.parse_sbatch(s, 20)
        self.assertAlmostEqual(sb["cap_cpuh"], 2 * (1 + 20))           # 42 CPU-h equivalents, not 2

    def test_crash_excluded_from_streak_and_crash_loop(self):
        s, v = self.files()
        s_abs = os.path.abspath(s)
        rows = cg.load_ledger(self.wd)
        for i, ch in enumerate(["no", "no", "crash", "crash"]):
            rows.append({"leaf": "P2.1", "job_id": str(i), "job_name": f"x{i}", "state": "done", "cap_cpuh": "1",
                         "used_cpuh": "1", "changed_decision": ch, "p": "0.5", "prediction_hit": "na",
                         "notes": "crash" if ch == "crash" else "", "script": s_abs, "voi": ""})
        cg.save_ledger(self.wd, rows)
        rc, out = self.check(s, v)
        self.assertIn("leaf no-change streak: 2", out)                  # crashes do not count toward G6
        self.assertIn("G7 crash loop", out)
        s, v = self.files(extra="* **fixed**: off-by-one in the sector index")
        self.assertEqual(self.check(s, v)[0], 0)

    def test_refresh_marks_crash_and_record_crash(self):
        s, v = self.files()
        rc, out = run(["submit", self.wd, s, "--voi", v, "--sbatch-bin", self.sbatch])
        jid = cg.load_ledger(self.wd)[0]["job_id"]
        sacct = make_bin(self.d, "sacct", f'echo "{jid}|hpb-demo-P2.1-R2-001|OUT_OF_MEMORY|4|600"')
        run(["refresh", self.wd, "--sacct-bin", sacct])
        row = cg.load_ledger(self.wd)[0]
        self.assertIn("crash: OUT_OF_MEMORY", row["notes"])
        run(["record", self.wd, jid, "--verdict", "OOM at step 3", "--changed", "crash"])
        row = cg.load_ledger(self.wd)[0]
        self.assertEqual((row["changed_decision"], row["prediction_hit"]), ("crash", "na"))
        rc, out = run(["report", self.wd])
        self.assertIn("**Crashes**", out)

    def test_submit_strips_slurm_env(self):
        envlog = os.path.join(self.d, "env.log")
        sb = make_bin(self.d, "sbatch_env", f'env | grep -E "^(SLURM|SBATCH)_" > {envlog}; echo 700123')
        s, v = self.files()
        os.environ["SBATCH_TIMELIMIT"] = "99:00:00"
        os.environ["SLURM_JOB_ID"] = "4242"
        try:
            rc, out = run(["submit", self.wd, s, "--voi", v, "--sbatch-bin", sb])
        finally:
            del os.environ["SBATCH_TIMELIMIT"], os.environ["SLURM_JOB_ID"]
        self.assertEqual(rc, 0, out)
        self.assertEqual(open(envlog).read().strip(), "")              # nothing leaked into sbatch

    def test_reconcile_finds_unledgered(self):
        s, v = self.files()
        run(["submit", self.wd, s, "--voi", v, "--sbatch-bin", self.sbatch])
        jid = cg.load_ledger(self.wd)[0]["job_id"]
        squeue = make_bin(self.d, "squeue",
                          f'echo "{jid}|hpb-demo-P2.1-R2-001|RUNNING|{self.td}"\n'
                          f'echo "999001|sneaky|PENDING|{self.td}"\n'
                          f'echo "999002|other_project|RUNNING|/elsewhere"')
        rc, out = run(["reconcile", self.wd, "--squeue-bin", squeue])
        self.assertEqual(rc, 1)
        self.assertIn("UNLEDGERED 999001", out)
        self.assertNotIn("999002", out)                                 # other projects are ignored
        squeue2 = make_bin(self.d, "squeue2", "true")
        rc, out = run(["reconcile", self.wd, "--squeue-bin", squeue2])
        self.assertEqual(rc, 0)
        self.assertIn(f"finished {jid}", out)


class ReviewFixes(unittest.TestCase):
    """Defects found by the independent review of the second pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.wd = os.path.join(self.d, "work")
        self.td = os.path.join(self.wd, "tests", "t")
        os.makedirs(self.td)
        rc, out = run(["init", self.wd, "--slug", "demo", "--total-cpuh", "500", "--per-leaf-cpuh", "150",
                       "--leaves", "P2.1,P4.1"])
        self.assertEqual(rc, 0, out)
        self.scancel = make_bin(self.d, "scancel", "true")
        self.sbatch = make_bin(self.d, "sbatch", "echo $((700000 + RANDOM % 1000))")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, time="01:00:00", leaf="P2.1", extra="", audit=False, name="s"):
        s, v = os.path.join(self.td, name + ".sbatch"), os.path.join(self.td, name + "_voi.md")
        open(s, "w").write(sbatch_text(time=time))
        open(v, "w").write(FULL_VOI.replace("P2.1", leaf).format(p=0.6, pcd="yes", cost=200, extra=extra))
        if audit:
            open(v, "a").write(f"\n* auditor: APPROVED 2026-11-16 {cg.voi_digest(v)}\n")
        return s, v

    def seed(self, outcomes, script=""):
        rows = cg.load_ledger(self.wd)
        for i, (ch, fixed) in enumerate(outcomes):
            rows.append({"leaf": "P2.1", "job_id": f"9{i}", "job_name": f"x{i}", "state": "done", "cap_cpuh": "1",
                         "used_cpuh": "1", "changed_decision": ch, "p": "0.5", "prediction_hit": "na",
                         "notes": "", "script": script, "voi": "", "fixed": fixed})
        cg.save_ledger(self.wd, rows)

    def test_unknown_leaf_refused(self):                               # defect 12
        s, v = self.write(leaf="P2.1x")
        rc, out = run(["check", self.wd, s, "--voi", v])
        self.assertIn("G1 unknown leaf", out)
        run(["set-budget", self.wd, "--add-leaf", "P2.1x"])
        self.assertNotIn("unknown leaf", run(["check", self.wd, s, "--voi", v])[1])

    def test_g6_quick_exempt_and_no_auditor_below_10(self):            # defect 9
        self.seed([("no", "")] * 3)
        s, v = self.write(time="00:10:00")                              # 0.67 CPU-h, quick
        self.assertEqual(run(["check", self.wd, s, "--voi", v])[0], 0)
        s, v = self.write(time="02:00:00")                              # 8 CPU-h, no justification
        self.assertIn("G6", run(["check", self.wd, s, "--voi", v])[1])
        s, v = self.write(time="02:00:00", extra="* reopen-justification: ingredient X")
        self.assertEqual(run(["check", self.wd, s, "--voi", v])[0], 0)  # no auditor needed at 8 CPU-h

    def test_cancel_is_kept_out_of_streak(self):                       # defect 5
        self.seed([("no", ""), ("no", "")])
        s, v = self.write()
        run(["submit", self.wd, s, "--voi", v, "--sbatch-bin", self.sbatch])
        jid = cg.load_ledger(self.wd)[-1]["job_id"]
        run(["cancel", self.wd, jid, "--reason", "early-stop kill", "--scancel-bin", self.scancel])
        self.assertEqual(cg.no_change_streak(cg.load_ledger(self.wd), "P2.1"), 2)

    def test_cancel_unledgered(self):                                  # defect 4
        rc, out = run(["cancel", self.wd, "123456", "--reason", "found by reconcile", "--scancel-bin", self.scancel])
        self.assertEqual(rc, 0, out)
        self.assertIn("unledgered", out)

    def test_crash_loop_needs_a_new_fix(self):                         # defect 6
        s, v = self.write(extra="* fixed: bug A")
        sa = os.path.abspath(s)
        self.seed([("crash", ""), ("crash", ""), ("crash", "bug A")], script=sa)
        out = run(["check", self.wd, s, "--voi", v])[1]
        self.assertIn("G7 crash loop", out)                             # bug A already tried and crashed
        s, v = self.write(extra="* fixed: bug B")
        self.assertNotIn("G7 crash loop", run(["check", self.wd, s, "--voi", v])[1])
        r = cg.load_ledger(self.wd)[0]
        r["changed_decision"], r["notes"] = "yes", "crash: FAILED"      # one failed array task, recorded yes
        self.assertFalse(cg.is_crash(r))

    def test_pre_audit(self):                                          # defect 8
        s, v = self.write(time="05:00:00")                              # 20 CPU-h
        rc, out = run(["check", self.wd, s, "--voi", v, "--pre-audit"])
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS PENDING AUDIT", out)
        self.assertEqual(run(["check", self.wd, s, "--voi", v])[0], 1)

    def test_user_approval_keeps_audit(self):                          # defect 3c
        s, v = self.write(time="05:00:00", audit=True)
        d1 = cg.voi_digest(v)
        open(v, "a").write("* user-approval: 2026-11-16 'go ahead'\n")
        self.assertEqual(cg.voi_digest(v), d1)

    def test_set_budget(self):                                         # defects 10, 11
        rc, out = run(["set-budget", self.wd, "--leaf", "P2.1=400"])
        self.assertEqual(rc, 1)
        self.assertIn("needs --approved-by", out)
        rc, out = run(["set-budget", self.wd, "--leaf", "P2.1=400", "--approved-by", "auditor 2026-11-16"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(cg.leaf_budget(cg.load_budget(self.wd), "P2.1"), 400)
        self.assertEqual(run(["set-budget", self.wd, "--total", "900"])[0], 1)
        self.assertEqual(run(["set-budget", self.wd, "--total", "900", "--user-approval", "'ok' 11-16"])[0], 0)
        self.assertEqual(cg.load_budget(self.wd)["total_cpuh"], 900)
        log = open(os.path.join(self.wd, "compute", "budget_log.tsv")).read()
        self.assertIn("auditor 2026-11-16", log)
        self.assertIn("user: 'ok' 11-16", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
