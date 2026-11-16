#!/usr/bin/env python3
"""
P1.2 reopening review (2026-11-16): T1 of route LRK-PN, the diagnostic's carrier in the PROJECTED, capacity-limited
regime. Rules written here BEFORE any run.
====================================================================================================================
Why. The closure's kills ran on unprojected toys, where ~90% of the sign error is symmetry contamination; on
projected toys the width-6 foil leaves < 0.005 of sign error (NO-TARGET-P), so the diagnostic was never tested
where it matters (projected training, sign error capacity-limited as on the 4x4 target, where projected width 6
leaves 0.012). A NARROW projected foil restores a target on ED-size toys at a capacity-to-difficulty ratio closer
to the target instance.
Stage foilN: NNBF of width WN = 2 (LRAnsatz, UHF start), 300 projected SR steps (t1c recipe, P_chi with |G| = 4
  as t1e Part B), seeds 11 and 22, sizes 1 (6x2 5+5) and 2 (4x3 5+5), gap asserts of t1c. A size is counted iff
  the median projected dEs of foilN >= 0.005. If no size is counted at WN = 2, WN = 1 is tried once under the
  same rule (--wn 1); if still none: NO-TARGET-N, the route is untestable at T1 and is recorded as such.
Arms, each 300 more projected SR steps from the saved foilN (same seeds):
  lrkN  LRK of t1e (ratio-fed weight-shared low-rank kernel, 120 parameters, exact warm start) on the narrow foil;
  widN  the narrow foil widened to the smallest width with n_params >= n_params(lrkN), zero new output rows.
RULE N (medians over seeds; r = dEs[lrkN] / dEs[widN], projected dEs as t1e score):
  two counted sizes: KILL if r >= 0.9 at both; PASS-W if r <= 0.7 at both and r(2) <= r(1) + 0.1; UNDECIDED else.
  one counted size: KILL if r >= 0.9 there; otherwise UNDECIDED-1 (no PASS-W possible from one size).
  A PASS-W reopens P1.2 with LRK-PN only; its first next step is the lrkO attribution arm, then a T2 design.
AMENDMENT (written 2026-11-16 after RULE N returned UNDECIDED with r = 0.56 at size 1 and 1.02 at size 2, BEFORE
  the arm below ran). LRK costs 0.535 CPU-s per projected step at size 1 against 0.140 for widN (ratio 3.8,
  numpy). Arm widNL is widN trained for 1150 steps (300 x 3.8, equal CPU time) from the same foilN, seeds 11 and
  22, size 1 only. Reported reading (it cannot change RULE N, which has no PASS-W left because size 2 ties):
  COST-ROBUST if dEs[lrkN] <= 0.7 dEs[widNL] at size 1; COST-ARTEFACT if dEs[lrkN] >= 0.9 dEs[widNL]; MIXED else.
Validity (else INVALID): t1c gap asserts; every warm start reproduces the foilN projected amplitudes (rel < 1e-10).
Budget: RLIMIT_CPU per invocation (--cpu_cap, default 580 s); 3 timed projected SR steps per arm give a
  projection (t1e gate: steps x 1.05 + 8 s per run); abort if > 90% of the invocation cap or if the review's
  ledger (results/t1f/ledger.json, shared with t1f_phase.py) plus the projection exceeds --task_cap (1800 CPU-s).

Usage
-----
  python3 t1g_projnarrow.py --mode foilN --sizes 1,2 [--wn 2]
  python3 t1g_projnarrow.py --mode train --sizes 2 --arms lrkN,widN [--wn 2]
  python3 t1g_projnarrow.py --mode verdict [--wn 2]
Options: --mode {foilN,train,verdict}; --sizes LIST (subset of 1,2); --arms LIST (subset of lrkN,widN,widNL; widNL = widN at 1150 steps);
         --seeds LIST (default 11,22); --wn INT narrow width (1 or 2, default 2); --threads INT (1..4, default 1);
         --cpu_cap FLOAT (default 580); --task_cap FLOAT (default 1800).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r6")
sys.path.insert(0, HERE)
sys.path.insert(0, R6)
import t1e_closed as E  # noqa: E402  (sets thread env from --threads, patches projected O for DRF types)
import t1f_phase as P  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

C, B, H21 = E.C, E.B, E.H21
LRAnsatz = E.LRAnsatz
OUT = os.path.join(HERE, "results", "t1g")


def foilN_fn(k, seed, wn):
    return os.path.join(OUT, f"foilN{wn}_{k}_{seed}.npz")


def load_foilN(k, seed, wn):
    m, psi0, E0, ref, gap = C.model(k)
    nb = LRAnsatz(m, ref, "full", K=1, Hs=wn, seed=seed)
    z = np.load(foilN_fn(k, seed, wn))
    for key in nb.keys:
        nb.p[key] = z[key].copy()
    return nb


def make(arm, k, seed, wn):
    m, psi0, E0, ref, gap = C.model(k)
    nb = load_foilN(k, seed, wn)
    if arm == "widNL":
        return make("widN", k, seed, wn)
    if arm == "lrkN":
        a = E.LRK(m, ref, wn, seed + 3000, "G")
        C.copy_pass1(a, nb)
    else:
        target = make("lrkN", k, seed, wn)[0].n_params()
        w = wn
        while LRAnsatz(m, ref, "full", K=1, Hs=w, seed=0).n_params() < target:
            w += 1
        a = C.widen(nb, m, ref, w, seed)
    return a, nb


def gate(per_runs_steps, args, label):
    proj = sum(p * n * 1.05 + 8.0 for p, n in per_runs_steps)
    used, prior = time.process_time(), P.ledger_used()
    P.log(f"budget {label}: projected {proj:.0f} CPU-s + {used:.0f} used; invocation cap {args.cpu_cap:.0f}; "
          f"review ledger {prior:.0f} of {args.task_cap:.0f}")
    if proj + used > 0.9 * args.cpu_cap or prior + proj + used > args.task_cap:
        P.log("BUDGET-ABORT")
        return False
    return True


def run_foilN(args):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, s) for k in args.sizes for s in args.seeds if not os.path.exists(foilN_fn(k, s, args.wn))]
    per = {}
    for k in sorted({j[0] for j in jobs}):
        m, psi0, E0, ref, gap = C.model(k)
        per[k] = E.time_steps(LRAnsatz(m, ref, "full", K=1, Hs=args.wn, seed=11), m, E.group(k)[0], "chi")
    if not gate([(per[k], 300) for k, _ in jobs], args, "foilN"):
        P.ledger_add("foilN-abort", time.process_time(), f"{args.sizes}")
        return 3
    for k, seed in jobs:
        m, psi0, E0, ref, gap = C.model(k)
        sg, gen = E.group(k)
        c0 = time.process_time()
        nb = LRAnsatz(m, ref, "full", K=1, Hs=args.wn, seed=seed)
        H21.optimize(nb, m, sg, "chi", 0, C.sra(300), np.random.default_rng(seed + 5), f"s{k} foilN {seed}")
        np.savez(foilN_fn(k, seed, args.wn), **{key: nb.p[key] for key in nb.keys})
        dEt, dEs = E.score(nb, m, psi0, E0, sg)
        row = dict(size=k, arm=f"foilN{args.wn}", seed=seed, n_params=nb.n_params(), dEt=dEt, dEs=dEs,
                   cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"{k}_foilN{args.wn}_{seed}.json"), "w"), indent=1)
        P.log(f"size {k} foilN{args.wn} seed {seed}: n_params {row['n_params']} dE_tot {dEt:.5f} "
              f"dE_sign {dEs:.5f} ({row['cpu']:.0f} CPU-s)")
    P.ledger_add("foilN", time.process_time(), f"sizes {args.sizes} wn {args.wn}")
    return 0


def run_train(args):
    jobs = [(k, a, s) for k in args.sizes for a in args.arms for s in args.seeds
            if not os.path.exists(os.path.join(OUT, f"{k}_{a}{args.wn}_{s}.json"))]
    per = {}
    for k, arm, s in jobs:
        if (k, arm) not in per:
            a, nb = make(arm, k, s, args.wn)
            m = C.model(k)[0]
            sg = E.group(k)[0]
            pa, pn = sg.project(a.amplitudes(8192)), sg.project(nb.amplitudes(8192))
            chk = float(np.max(np.abs(pa - pn)) / np.max(np.abs(pn)))
            assert chk < 1e-10, f"warm start check failed {chk}"
            per[(k, arm)] = E.time_steps(a, m, sg, "chi")
    nst = {a: (1150 if a == "widNL" else 300) for a in args.arms}
    P.log("per-step CPU-s: " + ", ".join(f"{a}@{k}={v:.3f}" for (k, a), v in per.items()))
    if not gate([(per[(k, a)], nst[a]) for k, a, _ in jobs], args, "trainN"):
        P.ledger_add("trainN-abort", time.process_time(), f"{args.sizes} {args.arms}")
        return 3
    for k, arm, seed in jobs:
        m, psi0, E0, ref, gap = C.model(k)
        sg, gen = E.group(k)
        c0 = time.process_time()
        a, nb = make(arm, k, seed, args.wn)
        H21.optimize(a, m, sg, "chi", 0, C.sra(nst[arm]), np.random.default_rng(seed + 107), f"N s{k} {arm} {seed}")
        dEt, dEs = E.score(a, m, psi0, E0, sg)
        row = dict(size=k, arm=arm, seed=seed, wn=args.wn, n_params=a.n_params(), T=nb.n_params(), dEt=dEt,
                   dEs=dEs, cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"{k}_{arm}{args.wn}_{seed}.json"), "w"), indent=1)
        P.log(f"size {k} {arm} seed {seed}: n_params {row['n_params']} (foil {row['T']}) dE_tot {dEt:.5f} "
              f"dE_sign {dEs:.5f} ({row['cpu']:.0f} CPU-s)")
    P.ledger_add("trainN", time.process_time(), f"sizes {args.sizes} arms {args.arms} wn {args.wn}")
    return 0


def medv(k, name, seeds):
    fs = [os.path.join(OUT, f"{k}_{name}_{s}.json") for s in seeds]
    if not all(os.path.exists(f) for f in fs):
        return None
    return float(np.median([json.load(open(f))["dEs"] for f in fs]))


def run_verdict(args):
    rows = {}
    for k in (1, 2):
        fo = medv(k, f"foilN{args.wn}", args.seeds)
        if fo is None:
            continue
        r = dict(foil=fo, counted=fo >= 0.005, lrk=medv(k, f"lrkN{args.wn}", args.seeds),
                 wid=medv(k, f"widN{args.wn}", args.seeds))
        r["r"] = r["lrk"] / r["wid"] if r["lrk"] is not None and r["wid"] is not None else None
        rows[k] = r
        wl = medv(k, f"widNL{args.wn}", args.seeds)
        if wl is not None and r["lrk"] is not None:
            q = r["lrk"] / wl
            r.update(widNL=wl, q_cost=q, cost_reading="COST-ROBUST" if q <= 0.7 else
                     ("COST-ARTEFACT" if q >= 0.9 else "MIXED"))
        P.log(f"size {k}: {r}")
    cnt = [k for k in rows if rows[k]["counted"]]
    if not cnt:
        v = "NO-TARGET-N"
    elif any(rows[k]["r"] is None for k in cnt):
        v = "INCOMPLETE"
    elif len(cnt) == 2:
        rr = [rows[k]["r"] for k in cnt]
        if all(x >= 0.9 for x in rr):
            v = "KILL"
        elif all(x <= 0.7 for x in rr) and rows[2]["r"] <= rows[1]["r"] + 0.1:
            v = "PASS-W"
        else:
            v = "UNDECIDED"
    else:
        v = "KILL" if rows[cnt[0]]["r"] >= 0.9 else "UNDECIDED-1"
    P.log(f"VERDICT LRK-PN (wn {args.wn}): {v}; counted sizes {cnt}")
    json.dump(dict(rows=rows, verdict=v), open(os.path.join(OUT, f"verdict_wn{args.wn}.json"), "w"), indent=1)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["foilN", "train", "verdict"])
    p.add_argument("--sizes", default="1,2")
    p.add_argument("--arms", default="lrkN,widN")
    p.add_argument("--seeds", default="11,22")
    p.add_argument("--wn", type=int, default=2, choices=[1, 2])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cpu_cap", type=float, default=580.0)
    p.add_argument("--task_cap", type=float, default=1800.0)
    a = p.parse_args(argv)
    a.sizes = [int(x) for x in a.sizes.split(",")]
    a.arms = a.arms.split(",")
    a.seeds = [int(x) for x in a.seeds.split(",")]
    assert set(a.sizes) <= {1, 2} and set(a.arms) <= {"lrkN", "widN", "widNL"}
    return a


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 10))
    return {"foilN": run_foilN, "train": run_train, "verdict": run_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
