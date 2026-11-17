#!/usr/bin/env python3
"""
compute_gate.py — the only sanctioned way to spend cluster compute in a hard-problem-brainstorm run.

It enforces the Compute Value Gate (SKILL.md, Stage 3.5.6) mechanically: a job is submitted only
if its value-of-information (VOI) note is complete, its cost fits its tier and the brainstorm's
budget, it is not predicted to leave the decision unchanged, its leaf is not in a run of jobs
that changed nothing, and it is not a resubmission of a script that keeps crashing. Every
submission is written to a machine-owned ledger, which `report` renders as the master
document's Compute Ledger and `reconcile` checks against the live Slurm queue.

FILES (all under <workdir>/compute/, where <workdir> is the brainstorm working directory)
    budget.json   Written once by `init` at Step 0. Keys:
                    slug             short name used in job names (e.g. "nqs-fermions")
                    total_cpuh       CPU-hour budget for the whole brainstorm
                    per_leaf_cpuh    default CPU-hour budget per leaf
                    leaf_overrides   {"P2.1": 800, ...} optional per-leaf budgets
                    skip_threshold   base probability above which a decision-neutral job is not run (0.8)
                    gpu_weight       CPU-hour equivalents charged per GPU-hour (default 20)
                    leaves           registered leaf ids; a note naming any other leaf is refused (G1),
                                     so a typo cannot open a fresh budget or reset a streak
    budget_log.tsv  Every change made by `set-budget`, with who approved it.
    ledger.tsv    One row per submitted job (tab-separated, header line). Written only by the
                  orchestrator through submit/record/refresh/cancel; workers only run `check`.

COST
    The cap of a job, in CPU-hour equivalents, is computed from its #SBATCH lines:
        array_tasks x walltime_hours x (tasks x cpus_per_task + gpus x gpu_weight)
    GPUs are read from --gres=gpu[:type]:N, --gpus / -G (per job), --gpus-per-task (x tasks) and
    --gpus-per-node (x nodes). A missing --time is refused: the gate needs a wall-time limit.

VOI NOTE FORMAT (a Markdown file, usually tests/<test-dir>/voi.md, linked from the leaf)
    One "key: value" line per field. Bullets ("* ", "- ") and bold ("**key**:") are allowed.
    Keys are case-insensitive. A value that is empty, "TODO", "TBD", "...", or starts with "<"
    counts as missing.
    Required for every job (the QUICK set; enough for a job whose cap is at most 1 CPU-hour):
        leaf                 leaf id, e.g. P2.1
        route                route id, e.g. R2 or 13.1
        tier                 T1 | T2 | T3
        decision             each verdict and the concrete action it triggers
        prediction           the expected verdict
        p                    probability of that verdict, a number in [0, 1]
        prediction-changes-decision   yes | no  (does the PREDICTED verdict change what you do next?)
    Also required when the cap exceeds 1 CPU-hour:
        cheapest-kill-tried  the T0 argument (scaling, op count, lemma) most likely to kill the route,
                             and its outcome; "none exists" must say why
        cost-cpuh            the capped CPU-hours you declare (must cover the computed cap)
        cheapest-version     why no smaller instance / fewer seeds / fewer arms / one rung decides it;
                             seeds, instances and arms are released one at a time
        convergence          evidence (pilot trajectory) that the run ends converged within its budget
        wall-time-basis      measured time per step x steps (+~30%) that sets --time
        power                P(AMBIGUOUS or INCOMPLETE) from measured run-to-run spread, and whether the
                             expected margin is so large that 1 seed suffices
    Required by the computed cap C:
        C > 10      auditor: APPROVED <date> <digest>, where <digest> is printed by `digest` for the
                    note as the auditor saw it. Editing any line of the note other than `auditor` and
                    `user-approval` afterwards changes the digest and voids the approval.
        C > 100     passed-t2: <job id or section> — the route already passed a T2 pilot
        C > 1000    user-approval: <date and the user's words> — never automatic
    Optional:
        resume-of            ledger job id this job continues (a resume is a new submission, fully gated)
        reopen-justification required (with auditor approval) when the leaf is in a no-change streak
        fixed                what was fixed, required to resubmit a script whose last 2 runs crashed; must
                             differ from any fix already tried by one of those crashed runs

GATE RULES (check and submit both apply them; submit refuses on any failure)
    G1 VOI complete            every required field present and not a placeholder; leaf registered
    G2 predicted, don't run    p >= effective threshold and prediction-changes-decision = no -> refuse
                               (effective threshold: skip_threshold, raised to 0.9 when past p>=0.8
                               predictions missed more than 25% of >=5 scored cases, lowered to 0.7
                               when they hit at least 90% of >=5 scored cases)
    G3 declared cost           computed cap <= 1.05 x cost-cpuh (when cost-cpuh is required)
    G4 tier                    auditor (with matching digest) / passed-t2 / user-approval as above
    G5 budget                  leaf committed + C <= leaf budget, and total committed + C <= total_cpuh
                               (committed = used CPU-h of finished jobs + cap of unfinished jobs)
    G6 no-change streak        the leaf's last 3 recorded jobs (crashes and deliberate cancels
                               excluded) all changed no decision -> a job above 1 CPU-h needs a
                               reopen-justification (above 10 CPU-h the auditor's approval, G4, covers
                               it); quick jobs are exempt
    G7 duplicate or crash loop the same script with the same VOI note is already pending or running;
                               or the script's last 2 finished runs crashed and the note has no new `fixed`

USAGE
    python3 compute_gate.py init      <workdir> --slug S --total-cpuh X --per-leaf-cpuh Y
                                      [--leaves P1.1,P1.2,...] [--leaf P2.1=800 ...]
                                      [--skip-threshold 0.8] [--gpu-weight 20]
    python3 compute_gate.py set-budget <workdir> [--leaf P2.1=600 ...] [--add-leaf P5.1 ...]
                                      [--total X] [--approved-by TEXT] [--user-approval TEXT]
    python3 compute_gate.py digest    <voi.md>
    python3 compute_gate.py check     <workdir> <script.sbatch> --voi <voi.md> [--pre-audit]
    python3 compute_gate.py submit    <workdir> <script.sbatch> --voi <voi.md> [--sbatch-bin PATH]
    python3 compute_gate.py record    <workdir> <job_id> --verdict TEXT --changed yes|no|crash
                                      [--hit yes|no|na] [--used-cpuh X]
    python3 compute_gate.py refresh   <workdir> [--sacct-bin PATH]
    python3 compute_gate.py cancel    <workdir> <job_id> --reason TEXT [--changed yes|no] [--scancel-bin PATH]
    python3 compute_gate.py reconcile <workdir> [--squeue-bin PATH]
    python3 compute_gate.py report    <workdir>

SUBCOMMANDS
    init      Create compute/budget.json and an empty ledger, registering the leaves. Run it once the
              leaves and the budget are known (after Stage 1.5). Refuses if budget.json exists.
    set-budget  Change the budget, logged in budget_log.tsv. --add-leaf registers a leaf (no approval
              needed). --leaf K=X sets a leaf cap: lowering is free (never below what is committed);
              raising needs --approved-by (the auditor moving budget between leaves within the
              unchanged total, which still binds) or --user-approval. --total needs --user-approval.
    digest    Print the 12-character digest of a VOI note, ignoring its `auditor:` and `user-approval:`
              lines. The auditor's approval line must quote it.
    check     Apply G1-G7 and print every failure; submits nothing. Exit 0 if the job would pass.
              Read-only, so workers may run it. With --pre-audit a missing auditor approval is
              reported as "PASS PENDING AUDIT" instead of a failure.
    submit    Apply G1-G7; on success run `sbatch --parsable --job-name=<unique>` from the script's
              directory and append a ledger row. The unique name is hpb-<slug>-<leaf>-<route>-<NNN>,
              so `sacct` rows can be matched even when the cluster reuses job ids. sbatch runs with
              every SLURM_* and SBATCH_* environment variable removed, so environment overrides cannot
              change the cost the gate computed and the job never inherits the caller's allocation.
    record    Fill in the outcome of a finished job: verdict text; whether it changed a decision
              (yes / no / crash — crash means it died before producing a verdict: code error, OOM,
              node failure; crashes do not count toward the G6 streak); whether the VOI prediction
              hit (yes/no/na); optionally the CPU-hours used.
    refresh   Query sacct for every unfinished ledger job (rows matched by job name), update used
              CPU-hours and state (done / cancelled), flag TIMEOUT as a wall-time miss and
              FAILED / OUT_OF_MEMORY / NODE_FAIL / BOOT_FAIL as a crash.
    cancel    scancel a job and mark it cancelled with a reason (queue review, early stop). Its outcome
              becomes "cancelled", which does not count toward G6, unless --changed is given. Also
              cancels a queued job that is not in the ledger (one found by reconcile).
    reconcile Compare the live queue (squeue) with the ledger, for resuming a run and for catching
              direct sbatch calls. Prints: ledger jobs that are no longer queued (run `refresh`),
              queued jobs whose working directory is inside <workdir> but that are not in the ledger
              (submitted around the gate: a procedure violation), and each queued job's state.
              Exit 1 if any unledgered job is found.
    report    Print the Compute Ledger as lint-clean Markdown: per-job table, per-leaf totals against
              budget, no-change streaks, crashes, wall-time misses, and prediction calibration.

EXIT CODES
    0 success / gate passed; 1 gate refused, violation found, or command failed; 2 bad arguments.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys

LEDGER_FIELDS = [
    "submitted_at", "leaf", "route", "job_id", "job_name", "tier", "cap_cpuh", "script", "voi",
    "prediction", "p", "pred_changes_decision", "auditor", "resume_of", "state", "used_cpuh",
    "verdict", "changed_decision", "prediction_hit", "notes", "fixed",
]
QUICK_VOI = ["leaf", "route", "tier", "decision", "prediction", "p", "prediction-changes-decision"]
FULL_VOI = QUICK_VOI + ["cheapest-kill-tried", "cost-cpuh", "cheapest-version", "convergence",
                        "wall-time-basis", "power"]
QUICK_CAP = 1.0
PLACEHOLDERS = {"", "todo", "tbd", "...", "n/a?", "?"}
STREAK = 3
CRASH_LOOP = 2
TERMINAL_SACCT = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
                  "PREEMPTED", "BOOT_FAIL", "DEADLINE")
CRASH_SACCT = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL")


# --------------------------------------------------------------------------- parsing helpers

def parse_walltime_hours(s):
    """Slurm --time formats: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S. Returns hours (float)."""
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        parts = [int(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, sec = parts
    else:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        else:
            h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


def count_array_tasks(spec):
    """Slurm --array: '0-9', '0-9%4', '1,3,5-7', '0-15:2'. Returns the number of tasks."""
    spec = spec.split("%", 1)[0]
    n = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, st = part.split(":", 1)
            step = int(st)
        if "-" in part:
            a, b = part.split("-", 1)
            n += len(range(int(a), int(b) + 1, step))
        else:
            n += 1
    return n


def gres_gpus(spec):
    """--gres value: 'gpu:2' -> 2, 'gpu:t4:2' -> 2, 'gpu' -> 1, 'gpu:2,shard:1' -> 2, 'shard:1' -> 0."""
    total = 0
    for item in spec.split(","):
        f = item.strip().split(":")
        if f[0] == "gpu":
            total += int(f[-1]) if len(f) > 1 and f[-1].isdigit() else 1
    return total


def gpus_opt(spec):
    """--gpus / --gpus-per-task / --gpus-per-node value: '2' -> 2, 't4:2' -> 2, 'a100:1,t4:1' -> 2."""
    total = 0
    for item in spec.split(","):
        f = item.strip().split(":")
        total += int(f[-1]) if f[-1].isdigit() else 1
    return total


def parse_sbatch(path, gpu_weight=20.0):
    """Read #SBATCH directives. Returns dict with time_h, cpus, tasks, gpus, array_tasks, job_name, cap_cpuh."""
    opts = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"^#SBATCH\s+(.*)$", line.strip())
            if not m:
                continue
            toks = m.group(1).split("#", 1)[0].split()
            i = 0
            while i < len(toks):
                tok = toks[i]
                nxt = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("-") else None
                if tok.startswith("--"):
                    k, eq, v = tok[2:].partition("=")
                    if not eq and nxt is not None:  # "--time 01:00:00"
                        v, i = nxt, i + 1
                    opts[k] = v
                elif re.fullmatch(r"-[a-zA-Z]", tok):  # "-c 4"
                    if nxt is not None:
                        opts["_short_" + tok[1]] = nxt
                        i += 1
                elif re.fullmatch(r"-[a-zA-Z].+", tok):  # "-c4"
                    opts["_short_" + tok[1]] = tok[2:]
                i += 1

    def get(long, short=None):
        if long in opts and opts[long] != "":
            return opts[long]
        if short and opts.get("_short_" + short):
            return opts["_short_" + short]
        return None

    t = get("time", "t")
    if t is None:
        raise ValueError(f"{path}: no #SBATCH --time; the gate needs a wall-time limit")
    time_h = parse_walltime_hours(t)
    cpus = int(get("cpus-per-task", "c") or 1)
    nodes = int(str(get("nodes", "N") or 1).split("-")[-1])
    ntasks = get("ntasks", "n")
    if ntasks is None and get("ntasks-per-node"):
        ntasks = int(get("ntasks-per-node")) * nodes
    tasks = int(ntasks or 1)
    gpus = 0
    if get("gres"):
        gpus += gres_gpus(get("gres")) * nodes
    if get("gpus", "G"):
        gpus += gpus_opt(get("gpus", "G"))
    if get("gpus-per-task"):
        gpus += gpus_opt(get("gpus-per-task")) * tasks
    if get("gpus-per-node"):
        gpus += gpus_opt(get("gpus-per-node")) * nodes
    arr = get("array", "a")
    array_tasks = count_array_tasks(arr) if arr else 1
    cap = array_tasks * time_h * (tasks * cpus + gpus * gpu_weight)
    return {"time_h": time_h, "cpus": cpus, "tasks": tasks, "gpus": gpus, "array_tasks": array_tasks,
            "job_name": get("job-name", "J"), "cap_cpuh": cap, "gpu_weight": gpu_weight}


VOI_LINE = re.compile(r"^\s*(?:[*\-+]\s+)?\**([A-Za-z][A-Za-z0-9 _\-]*?)\**\s*:\s*\**\s*(.*?)\s*$")


def _voi_key(raw):
    return raw.strip().lower().replace(" ", "-").replace("_", "-")


def parse_voi(path):
    """Read 'key: value' lines from a VOI Markdown note. Keys are lower-cased."""
    voi = {}
    with open(path) as fh:
        for line in fh:
            m = VOI_LINE.match(line)
            if m:
                key = _voi_key(m.group(1))
                if key not in voi:
                    voi[key] = m.group(2).strip()
    return voi


def voi_digest(path):
    """12-hex digest of the note with its auditor and user-approval lines removed (whitespace-normalised)."""
    h = hashlib.sha256()
    with open(path) as fh:
        for line in fh:
            m = VOI_LINE.match(line)
            if m and _voi_key(m.group(1)) in ("auditor", "user-approval"):
                continue
            s = " ".join(line.split())
            if s:
                h.update((s + "\n").encode())
    return h.hexdigest()[:12]


def missing(v):
    return v is None or v.strip().lower() in PLACEHOLDERS or v.strip().startswith("<")


# --------------------------------------------------------------------------- state

def compute_dir(workdir):
    return os.path.join(workdir, "compute")


def load_budget(workdir):
    p = os.path.join(compute_dir(workdir), "budget.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run `compute_gate.py init` at Step 0 to set the budget")
    with open(p) as fh:
        return json.load(fh)


def load_ledger(workdir):
    p = os.path.join(compute_dir(workdir), "ledger.tsv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        for k in LEDGER_FIELDS:
            if r.get(k) is None:
                r[k] = ""
    return rows


def save_ledger(workdir, rows):
    p = os.path.join(compute_dir(workdir), "ledger.tsv")
    tmp = p + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEDGER_FIELDS})
    os.replace(tmp, p)


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def committed(rows, leaf=None):
    tot = 0.0
    for r in rows:
        if leaf is not None and r["leaf"] != leaf:
            continue
        if r["state"] in ("done", "cancelled"):
            tot += fnum(r["used_cpuh"], fnum(r["cap_cpuh"]))
        else:
            tot += fnum(r["cap_cpuh"])
    return tot


def leaf_budget(budget, leaf):
    return float(budget.get("leaf_overrides", {}).get(leaf, budget["per_leaf_cpuh"]))


def no_change_streak(rows, leaf):
    """Most-recent finished jobs of this leaf, in a row, that changed no decision (crashes skipped)."""
    fin = [r for r in rows if r["leaf"] == leaf and r["changed_decision"] in ("yes", "no")]
    n = 0
    for r in reversed(fin):
        if r["changed_decision"] == "no":
            n += 1
        else:
            break
    return n


def is_crash(r):
    """Recorded as a crash, or flagged as one by refresh and not yet recorded otherwise."""
    if r["changed_decision"] == "crash":
        return True
    return r["changed_decision"] in ("pending", "") and "crash" in r["notes"]


def crash_run(rows, script):
    """(count, fixes) for the most-recent finished runs of this script that crashed, in a row.
    fixes is the set of `fixed` texts those crashed runs were submitted with."""
    fin = [r for r in rows if r["script"] == script and r["state"] in ("done", "cancelled")]
    n, fixes = 0, set()
    for r in reversed(fin):
        if is_crash(r):
            n += 1
            if r.get("fixed"):
                fixes.add(" ".join(r["fixed"].split()).lower())
        else:
            break
    return n, fixes


def calibration(rows):
    """(scored, hits) over predictions with p >= 0.8 whose hit was recorded yes/no."""
    sc = [r for r in rows if fnum(r["p"], -1) >= 0.8 and r["prediction_hit"] in ("yes", "no")]
    return len(sc), sum(1 for r in sc if r["prediction_hit"] == "yes")


def effective_threshold(budget, rows):
    base = float(budget.get("skip_threshold", 0.8))
    n, h = calibration(rows)
    if n >= 5 and h / n < 0.75:
        return 0.9, f"raised to 0.9: past p>=0.8 predictions hit {h}/{n}"
    if n >= 5 and h / n >= 0.9:
        return min(base, 0.7), f"lowered to 0.7: past p>=0.8 predictions hit {h}/{n}"
    return base, f"base {base} ({n} scored predictions)"


def approved(voi_value, digest):
    v = (voi_value or "").strip()
    return v.upper().startswith("APPROVED") and digest in v


# --------------------------------------------------------------------------- the gate

def gate(workdir, script, voi_path, pre_audit=False):
    """Apply G1-G7. Returns (failures, info). pre_audit=True reports a missing auditor approval as
    pending instead of failing (for workers, who check before the orchestrator gets the audit)."""
    fails = []
    budget = load_budget(workdir)
    rows = load_ledger(workdir)
    sb = parse_sbatch(script, float(budget.get("gpu_weight", 20.0)))
    voi = parse_voi(voi_path)
    digest = voi_digest(voi_path)
    cap = sb["cap_cpuh"]
    quick = cap <= QUICK_CAP
    info = {"cap": cap, "sb": sb, "voi": voi, "budget": budget, "rows": rows, "digest": digest,
            "quick": quick}

    # G1
    for k in (QUICK_VOI if quick else FULL_VOI):
        if missing(voi.get(k)):
            fails.append(f"G1 VOI field '{k}' is missing or a placeholder")
    p = fnum(voi.get("p"), -1)
    if not missing(voi.get("p")) and not (0.0 <= p <= 1.0):
        fails.append(f"G1 'p' must be a probability in [0,1], got {voi.get('p')!r}")
    pcd = (voi.get("prediction-changes-decision") or "").strip().lower()
    if pcd and not (pcd.startswith("y") or pcd.startswith("n")):
        fails.append("G1 'prediction-changes-decision' must be yes or no")
    if (voi.get("tier") or "").upper() not in ("T1", "T2", "T3"):
        fails.append("G1 'tier' must be T1, T2 or T3")
    known = budget.get("leaves")
    if known and (voi.get("leaf") or "").strip() not in known:
        fails.append(f"G1 unknown leaf {voi.get('leaf')!r}; registered leaves: {', '.join(known)} "
                     f"(add a new leaf with `set-budget --add-leaf`)")

    # G2
    thr, why = effective_threshold(budget, rows)
    info["threshold"] = (thr, why)
    if p >= thr and pcd.startswith("n"):
        fails.append(f"G2 predicted, don't run: p={p} >= {thr} ({why}) and the predicted verdict does "
                     f"not change the decision. Record the prediction and act on it.")

    # G3
    declared = fnum(voi.get("cost-cpuh"), -1)
    if not quick and declared >= 0 and cap > 1.05 * declared:
        fails.append(f"G3 computed cap {cap:.1f} CPU-h exceeds declared cost-cpuh {declared:g}")

    # G4
    info["audit_pending"] = False
    if cap > 10 and not approved(voi.get("auditor"), digest):
        if pre_audit:
            info["audit_pending"] = True
        else:
            fails.append(f"G4 cap {cap:.1f} > 10 CPU-h needs 'auditor: APPROVED <date> {digest}' from the "
                         f"compute auditor (digest of this exact note; any edit voids an earlier approval)")
    if cap > 100 and missing(voi.get("passed-t2")):
        fails.append(f"G4 cap {cap:.1f} > 100 CPU-h needs 'passed-t2: <job id or section>'")
    if cap > 1000 and missing(voi.get("user-approval")):
        fails.append(f"G4 cap {cap:.1f} > 1000 CPU-h needs 'user-approval: <date, user's words>'; ask the "
                     f"user and keep working while the request is pending")

    # G5
    leaf = (voi.get("leaf") or "").strip()
    lb = leaf_budget(budget, leaf)
    lc = committed(rows, leaf)
    tc = committed(rows)
    info.update(leaf=leaf, leaf_budget=lb, leaf_committed=lc, total_committed=tc)
    if lc + cap > lb:
        fails.append(f"G5 leaf {leaf} budget: committed {lc:.1f} + {cap:.1f} > {lb:g} CPU-h. "
                     f"Find a cheaper version or record GATE-REFUSED(budget) (SKILL.md section 3b)")
    if tc + cap > float(budget["total_cpuh"]):
        fails.append(f"G5 brainstorm budget: committed {tc:.1f} + {cap:.1f} > {budget['total_cpuh']} CPU-h")

    # G6
    streak = no_change_streak(rows, leaf)
    info["streak"] = streak
    if streak >= STREAK and not quick and missing(voi.get("reopen-justification")):
        fails.append(f"G6 leaf {leaf}: last {streak} finished jobs changed no decision. The leaf is "
                     f"DORMANT; give a 'reopen-justification' (above 10 CPU-h the auditor's approval, G4, "
                     f"covers it). Quick jobs of at most 1 CPU-h are exempt")

    # G7
    s_abs, v_abs = os.path.abspath(script), os.path.abspath(voi_path)
    for r in rows:
        if r["state"] not in ("done", "cancelled") and r["script"] == s_abs and r["voi"] == v_abs:
            fails.append(f"G7 duplicate: job {r['job_id']} with this script and VOI is still {r['state']}")
    crashes, fixes = crash_run(rows, s_abs)
    info["crashes"] = crashes
    fixed_now = " ".join((voi.get("fixed") or "").split()).lower()
    if crashes >= CRASH_LOOP and (missing(voi.get("fixed")) or fixed_now in fixes):
        fails.append(f"G7 crash loop: the last {crashes} runs of this script crashed. Reproduce the crash "
                     f"at T1 (pre-flight dry run), fix it, and state the new fix in a 'fixed:' line "
                     f"(a fix already tried by a crashed run does not count)")
    return fails, info


def print_gate(fails, info):
    sb = info["sb"]
    gpu = f" + {sb['gpus']} gpus x {sb['gpu_weight']:g}" if sb["gpus"] else ""
    print(f"cap = {sb['array_tasks']} array x {sb['time_h']:.2f} h x ({sb['tasks']} tasks x {sb['cpus']} cpus"
          f"{gpu}) = {info['cap']:.1f} CPU-h" + ("  [quick note: cap <= 1 CPU-h]" if info["quick"] else ""))
    print(f"leaf {info['leaf']}: committed {info['leaf_committed']:.1f} / {info['leaf_budget']:g} CPU-h; "
          f"brainstorm committed {info['total_committed']:.1f} / {info['budget']['total_cpuh']} CPU-h")
    print(f"skip threshold: {info['threshold'][1]}; leaf no-change streak: {info['streak']}; "
          f"script crash run: {info['crashes']}; VOI digest: {info['digest']}")
    if fails:
        print("GATE: REFUSED")
        for f in fails:
            print("  - " + f)
    elif info.get("audit_pending"):
        print(f"GATE: PASS PENDING AUDIT (cap > 10 CPU-h; the orchestrator gets the auditor's approval "
              f"for digest {info['digest']} before submitting)")
    else:
        print("GATE: PASS")


# --------------------------------------------------------------------------- subcommands

def cmd_init(a):
    d = compute_dir(a.workdir)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "budget.json")
    if os.path.exists(p):
        print(f"{p} already exists; edit it deliberately (a budget change is a user decision)")
        return 1
    overrides = {}
    for kv in a.leaf or []:
        k, _, v = kv.partition("=")
        overrides[k] = float(v)
    leaves = [x.strip() for x in (a.leaves or "").split(",") if x.strip()]
    b = {"slug": a.slug, "total_cpuh": a.total_cpuh, "per_leaf_cpuh": a.per_leaf_cpuh, "leaves": leaves,
         "leaf_overrides": overrides, "skip_threshold": a.skip_threshold, "gpu_weight": a.gpu_weight,
         "created": _dt.datetime.now().isoformat(timespec="seconds")}
    with open(p, "w") as fh:
        json.dump(b, fh, indent=2)
    save_ledger(a.workdir, load_ledger(a.workdir))
    print(f"wrote {p} and {os.path.join(d, 'ledger.tsv')}")
    return 0


def cmd_digest(a):
    print(voi_digest(a.voi))
    return 0


def cmd_check(a):
    fails, info = gate(a.workdir, a.script, a.voi, pre_audit=getattr(a, "pre_audit", False))
    print_gate(fails, info)
    return 1 if fails else 0


def unique_name(budget, rows, voi):
    base = "hpb-{}-{}-{}".format(budget.get("slug", "run"), voi.get("leaf", "P"), voi.get("route", "R"))
    base = re.sub(r"[^A-Za-z0-9_.\-]", "_", base)
    n = sum(1 for r in rows if r["job_name"].startswith(base)) + 1
    return f"{base}-{n:03d}"


def cmd_submit(a):
    fails, info = gate(a.workdir, a.script, a.voi)
    print_gate(fails, info)
    if fails:
        return 1
    rows, voi = info["rows"], info["voi"]
    name = unique_name(info["budget"], rows, voi)
    sdir = os.path.dirname(os.path.abspath(a.script))
    cmd = [a.sbatch_bin, "--parsable", f"--job-name={name}", os.path.abspath(a.script)]
    # SBATCH_* variables override #SBATCH lines (and so the cap the gate computed); SLURM_* variables
    # inherited from an orchestrator running inside an allocation can attach the job to that allocation.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    res = subprocess.run(cmd, cwd=sdir, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print("sbatch failed:", res.stderr.strip())
        return 1
    job_id = res.stdout.strip().split(";")[0]
    rows.append({
        "submitted_at": _dt.datetime.now().isoformat(timespec="seconds"), "leaf": voi.get("leaf", ""),
        "route": voi.get("route", ""), "job_id": job_id, "job_name": name, "tier": voi.get("tier", "").upper(),
        "cap_cpuh": f"{info['cap']:.2f}", "script": os.path.abspath(a.script), "voi": os.path.abspath(a.voi),
        "prediction": voi.get("prediction", ""), "p": voi.get("p", ""),
        "pred_changes_decision": voi.get("prediction-changes-decision", ""),
        "auditor": voi.get("auditor", ""), "resume_of": voi.get("resume-of", ""), "state": "submitted",
        "used_cpuh": "", "verdict": "", "changed_decision": "pending", "prediction_hit": "", "notes": "",
        "fixed": voi.get("fixed", ""),
    })
    save_ledger(a.workdir, rows)
    print(f"SUBMITTED job {job_id} as {name}; ledger updated")
    return 0


def find_row(rows, job_id):
    for r in rows:
        if r["job_id"] == str(job_id):
            return r
    raise KeyError(f"job {job_id} is not in the ledger")


def add_note(r, note):
    if note not in r["notes"]:
        r["notes"] = (r["notes"] + "; " + note).strip("; ")


def cmd_record(a):
    rows = load_ledger(a.workdir)
    r = find_row(rows, a.job_id)
    r["verdict"] = a.verdict
    r["changed_decision"] = a.changed
    r["prediction_hit"] = "na" if a.changed == "crash" else a.hit
    if a.changed == "crash":
        add_note(r, "crash")
    if a.used_cpuh is not None:
        r["used_cpuh"] = f"{a.used_cpuh:.2f}"
    if r["state"] != "cancelled":
        r["state"] = "done"
    save_ledger(a.workdir, rows)
    print(f"recorded job {a.job_id}: {a.verdict} (changed decision: {a.changed}, prediction hit: {r['prediction_hit']})")
    return 0


def cmd_refresh(a):
    rows = load_ledger(a.workdir)
    changed = 0
    for r in rows:
        if r["state"] in ("done", "cancelled") and r["used_cpuh"]:
            continue
        res = subprocess.run([a.sacct_bin, "-j", r["job_id"], "-X", "-n", "-P",
                              "-o", "JobID,JobName,State,AllocCPUS,ElapsedRaw"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print(f"sacct failed for {r['job_id']}: {res.stderr.strip()}")
            continue
        used, states = 0.0, []
        for line in res.stdout.splitlines():
            f = line.split("|")
            if len(f) < 5 or f[1] != r["job_name"]:
                continue  # job-id reuse by an unrelated job: skip
            states.append(f[2].split()[0])
            used += fnum(f[3]) * fnum(f[4]) / 3600
        if not states:
            continue
        r["used_cpuh"] = f"{used:.2f}"
        if all(s.startswith(TERMINAL_SACCT) for s in states):
            if r["state"] != "cancelled":
                r["state"] = "cancelled" if all(s.startswith("CANCELLED") for s in states) else "done"
            if any(s.startswith("TIMEOUT") for s in states):
                add_note(r, "wall-time miss")
            crashed = sorted({s for s in states if s.startswith(CRASH_SACCT)})
            if crashed:
                add_note(r, "crash: " + ",".join(crashed))
        else:
            r["state"] = "running" if any(s.startswith("RUNNING") for s in states) else "pending"
        changed += 1
    save_ledger(a.workdir, rows)
    print(f"refreshed {changed} job(s)")
    return 0


def cmd_cancel(a):
    rows = load_ledger(a.workdir)
    r = next((x for x in rows if x["job_id"] == str(a.job_id)), None)
    res = subprocess.run([a.scancel_bin, str(a.job_id)], capture_output=True, text=True)
    if res.returncode != 0:
        print("scancel failed:", res.stderr.strip())
        return 1
    if r is None:  # e.g. found by reconcile: submitted around the gate, or lost between sbatch and ledger
        print(f"cancelled unledgered job {a.job_id}: {a.reason} (resubmit through `submit` if still wanted)")
        return 0
    r["state"] = "cancelled"
    if r["changed_decision"] == "pending":
        # A deliberate cancel (early-stop kill, queue review) is housekeeping, not a job that failed to
        # inform: it is kept out of the G6 streak unless --changed says otherwise.
        r["changed_decision"] = a.changed or "cancelled"
    add_note(r, "cancelled: " + a.reason)
    save_ledger(a.workdir, rows)
    print(f"cancelled job {a.job_id}: {a.reason} (changed decision: {r['changed_decision']})")
    return 0


def cmd_set_budget(a):
    """Change leaf caps, register leaves, or (with the user's words) change the total. Logged."""
    p = os.path.join(compute_dir(a.workdir), "budget.json")
    budget = load_budget(a.workdir)
    rows = load_ledger(a.workdir)
    changes = []
    if a.total is not None:
        if not a.user_approval:
            print("REFUSED: changing total_cpuh needs --user-approval '<date, the user's words>'")
            return 1
        changes.append(f"total_cpuh {budget['total_cpuh']} -> {a.total}")
        budget["total_cpuh"] = a.total
    for name in a.add_leaf or []:
        lv = budget.setdefault("leaves", [])
        if name not in lv:
            lv.append(name)
            changes.append(f"registered leaf {name}")
    for kv in a.leaf or []:
        k, _, v = kv.partition("=")
        new = float(v)
        old = leaf_budget(budget, k)
        if new > old and not a.user_approval:
            # Raising a leaf cap within the unchanged total moves budget between leaves: the auditor may
            # approve it, but only while the total still covers everything already committed.
            if not a.approved_by:
                print(f"REFUSED: raising {k} from {old:g} to {new:g} needs --approved-by '<auditor verdict>'")
                return 1
        if new < committed(rows, k):
            print(f"REFUSED: {k} has {committed(rows, k):.1f} CPU-h committed, more than {new:g}")
            return 1
        budget.setdefault("leaf_overrides", {})[k] = new
        changes.append(f"{k} cap {old:g} -> {new:g}")
    if not changes:
        print("nothing to change")
        return 1
    with open(p, "w") as fh:
        json.dump(budget, fh, indent=2)
    log = os.path.join(compute_dir(a.workdir), "budget_log.tsv")
    new_log = not os.path.exists(log)
    with open(log, "a") as fh:
        if new_log:
            fh.write("time\tchange\tapproved_by\n")
        who = a.user_approval and f"user: {a.user_approval}" or a.approved_by or "registration"
        for c in changes:
            fh.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{c}\t{who}\n")
    print("budget updated: " + "; ".join(changes))
    return 0


def cmd_reconcile(a):
    rows = load_ledger(a.workdir)
    res = subprocess.run([a.squeue_bin, "--me", "-h", "-o", "%i|%j|%T|%Z"], capture_output=True, text=True)
    if res.returncode != 0:
        print("squeue failed:", res.stderr.strip())
        return 1
    wd = os.path.realpath(a.workdir)
    queued = {}
    for line in res.stdout.splitlines():
        f = line.split("|")
        if len(f) < 4:
            continue
        base_id = f[0].split("_")[0].split("[")[0]
        queued.setdefault(base_id, (f[1], f[2], f[3]))
    ledger_ids = {r["job_id"] for r in rows}
    bad = []
    for jid, (name, state, wdir) in sorted(queued.items()):
        inside = os.path.realpath(wdir).startswith(wd + os.sep) or os.path.realpath(wdir) == wd
        if jid in ledger_ids:
            print(f"queued   {jid} {name} {state}")
        elif inside or name.startswith("hpb-"):
            bad.append(jid)
            print(f"UNLEDGERED {jid} {name} {state} in {wdir}: submitted around the gate; cancel it or "
                  f"write its VOI note and record it")
    open_rows = [r for r in rows if r["state"] not in ("done", "cancelled")]
    for r in open_rows:
        if r["job_id"] not in queued:
            print(f"finished {r['job_id']} {r['job_name']}: no longer queued; run `refresh` then `record`")
    print(f"reconcile: {len(queued)} queued, {len(open_rows)} open in ledger, {len(bad)} unledgered")
    return 1 if bad else 0


def cmd_report(a):
    budget = load_budget(a.workdir)
    rows = load_ledger(a.workdir)
    cell = lambda s: (s or "").replace("|", "\\|")
    out = ["## Compute Ledger", "",
           f"Generated by `compute_gate.py report` on {_dt.date.today().isoformat()}. "
           f"Budget: {budget['total_cpuh']} CPU-h in all, {budget['per_leaf_cpuh']} per leaf by default; "
           f"GPU-hours are charged at {budget.get('gpu_weight', 20)} CPU-h each.", "",
           "| Leaf | Route | Job | Tier | Cap CPU-h | Used CPU-h | State | Prediction (p) | Verdict | Changed a decision | Prediction hit | Notes |",
           "| :--- | :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- | :--- | :--- |"]
    for r in rows:
        out.append(f"| {r['leaf']} | {cell(r['route'])} | {r['job_id']} | {r['tier']} | {fnum(r['cap_cpuh']):.1f} | "
                   f"{r['used_cpuh'] or '—'} | {r['state']} | {cell(r['prediction'])} ({r['p']}) | "
                   f"{cell(r['verdict']) or '—'} | {r['changed_decision']} | {r['prediction_hit'] or '—'} | "
                   f"{cell(r['notes']) or '—'} |")
    out += ["", "| Leaf | Jobs | Used CPU-h | Committed CPU-h | Leaf budget | No-change CPU-h | No-change streak |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    leaves = sorted({r["leaf"] for r in rows})
    flags = []
    for lf in leaves:
        lr = [r for r in rows if r["leaf"] == lf]
        used = sum(fnum(r["used_cpuh"]) for r in lr)
        nc = sum(fnum(r["used_cpuh"], fnum(r["cap_cpuh"])) for r in lr if r["changed_decision"] == "no")
        st = no_change_streak(rows, lf)
        lc, lb = committed(rows, lf), leaf_budget(budget, lf)
        out.append(f"| {lf} | {len(lr)} | {used:.1f} | {lc:.1f} | {lb:g} | {nc:.1f} | {st} |")
        if st >= STREAK:
            flags.append(f"* **{lf}**: last {st} finished jobs changed no decision; G6 blocks further jobs. "
                         f"The leaf is DORMANT (SKILL.md section 3b) unless the auditor approves a reopen justification.")
        if lc >= 0.9 * lb:
            flags.append(f"* **{lf}**: {lc:.1f} of {lb:g} CPU-h committed; plan the leaf's exit (section 3b) now.")
    tot_used = sum(fnum(r["used_cpuh"]) for r in rows)
    out += ["", f"**Total**: {len(rows)} jobs, {tot_used:.1f} CPU-h used, {committed(rows):.1f} committed "
                f"of {budget['total_cpuh']}.", ""]
    misses = [r["job_id"] for r in rows if "wall-time miss" in r["notes"]]
    if misses:
        flags.append(f"* **Wall-time misses** (TIMEOUT): {', '.join(misses)}. Re-measure time per step before resuming.")
    crashes = [r["job_id"] for r in rows if is_crash(r)]
    if crashes:
        flags.append(f"* **Crashes**: {', '.join(crashes)}. Reproduce at T1 and fix before resubmitting (G7 blocks a third crash).")
    n, h = calibration(rows)
    thr, why = effective_threshold(budget, rows)
    flags.append(f"* **Prediction calibration**: {h} of {n} scored predictions with $p\\ge0.8$ hit; skip threshold {why}.")
    out += ["### Flags", ""] + flags + [""]
    print("\n".join(out))
    return 0


# --------------------------------------------------------------------------- CLI

def parse_args(argv):
    ap = argparse.ArgumentParser(prog="compute_gate.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init", help="create compute/budget.json and the ledger (Step 0)")
    p.add_argument("workdir")
    p.add_argument("--slug", required=True)
    p.add_argument("--total-cpuh", type=float, required=True)
    p.add_argument("--per-leaf-cpuh", type=float, required=True)
    p.add_argument("--leaf", action="append", help="per-leaf override, e.g. P2.1=800 (repeatable)")
    p.add_argument("--skip-threshold", type=float, default=0.8)
    p.add_argument("--gpu-weight", type=float, default=20.0, help="CPU-h equivalents per GPU-hour")
    p.add_argument("--leaves", help="comma-separated leaf ids to register, e.g. P1.1,P1.2,P2.1")
    p = sp.add_parser("set-budget", help="register leaves, move budget between leaves, or change the total")
    p.add_argument("workdir")
    p.add_argument("--leaf", action="append", help="new cap for a leaf, e.g. P2.1=600 (repeatable)")
    p.add_argument("--add-leaf", action="append", help="register a new leaf id (repeatable)")
    p.add_argument("--total", type=float, help="new total_cpuh (needs --user-approval)")
    p.add_argument("--approved-by", help="auditor verdict authorising a leaf-cap raise within the total")
    p.add_argument("--user-approval", help="date and the user's words authorising the change")
    p = sp.add_parser("digest", help="print the digest of a VOI note (for the auditor's approval line)")
    p.add_argument("voi")
    for name in ("check", "submit"):
        p = sp.add_parser(name, help=f"{name} a job against gates G1-G7")
        p.add_argument("workdir")
        p.add_argument("script")
        p.add_argument("--voi", required=True)
        if name == "check":
            p.add_argument("--pre-audit", action="store_true",
                           help="report a missing auditor approval as pending, not a failure (for workers)")
        if name == "submit":
            p.add_argument("--sbatch-bin", default="sbatch")
    p = sp.add_parser("record", help="record a finished job's outcome")
    p.add_argument("workdir")
    p.add_argument("job_id")
    p.add_argument("--verdict", required=True)
    p.add_argument("--changed", choices=["yes", "no", "crash"], required=True)
    p.add_argument("--hit", choices=["yes", "no", "na"], default="na")
    p.add_argument("--used-cpuh", type=float)
    p = sp.add_parser("refresh", help="update used CPU-h and states from sacct")
    p.add_argument("workdir")
    p.add_argument("--sacct-bin", default="sacct")
    p = sp.add_parser("cancel", help="scancel a job and record why")
    p.add_argument("workdir")
    p.add_argument("job_id")
    p.add_argument("--reason", required=True)
    p.add_argument("--changed", choices=["yes", "no"],
                   help="only if the cancel itself settled something (default: 'cancelled', kept out of G6)")
    p.add_argument("--scancel-bin", default="scancel")
    p = sp.add_parser("reconcile", help="compare the live queue with the ledger (resume; catch direct sbatch)")
    p.add_argument("workdir")
    p.add_argument("--squeue-bin", default="squeue")
    p = sp.add_parser("report", help="print the Compute Ledger as Markdown")
    p.add_argument("workdir")
    return ap.parse_args(argv)


def main(argv):
    a = parse_args(argv)
    handlers = {"init": cmd_init, "set-budget": cmd_set_budget, "digest": cmd_digest, "check": cmd_check,
                "submit": cmd_submit,
                "record": cmd_record, "refresh": cmd_refresh, "cancel": cmd_cancel,
                "reconcile": cmd_reconcile, "report": cmd_report}
    try:
        return handlers[a.cmd](a)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
