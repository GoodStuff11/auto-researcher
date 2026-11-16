#!/usr/bin/env python3
"""
P1.3 closed-leaf falsifier (2026-11-16): a DIRECT wrong-sign-basin probe at T1, rule written before any run.
==============================================================================================================

Why. P1.3 closed CAPACITY through the width-trend tie-break. The cross-node critic noted that the basin was
never directly probed: the oracle arm W started WORSE in sign than the cold runs (W0 0.0195 against cold
0.0117 at width 6), so "W relaxes to the cold level" says nothing about a better-sign state. This probe first
builds, at fixed NNBF width, a state whose sign error is well BELOW the cold run's, then releases it under
energy SR and asks whether it stays there.

Instance (toy, gapped, space-group projected, as the round-2 projected-control rule requires): t-J, t = 1,
t' = -0.25, J = 0.4, open 4x3 cluster, 5+5 (D = 16632, gap 0.085; P1.2 t1c size 2). Trained state
psi_P = P_chi psi_theta, |G| = 4 (identity, Mx, My, MxMy), chi from the ED ground state. Ansatz: NNBF
(LRAnsatz "full", K = 1) width 6 (1110 parameters). Exact-sum projected SR (p21_harness: lr 0.05, shift 1e-3,
600 metric samples). Fallbacks, in order, if the cold state has no target (median dEs(cold start) < 0.005):
size 1 (6x2 5+5, gap 0.118) at width 6, then size 2 at width 3; else NO-TARGET.
Cold start S: NNBF from the UHF start after 300 projected energy SR steps. For width 6, seeds 11 and 22 reuse
P1.2's foilP files (tests/P1.2_unanchored_antisymmetry_r6/results/t1e/foilP_<size>_<seed>.npz: the same
model, recipe and seed); every other (size, width, seed) is trained here. Seeds 11, 22, 33.
Arms, per seed, both from S:
  C (cold continuation): 300 more projected energy SR steps; scored after 100 and 300.
  O (oracle): fidelity SR on -log F, F = <t|psi_P>^2 / (<t|t><psi_P|psi_P>), towards the SIGN-ORACLE target
     t = sgn(psi_0) |psi_P(S)| (S's own amplitudes with the exact signs; t lies in the chi sector), half-gradient
     coefficient psi_P/<psi_P|psi_P> - t/<t|psi_P>, same SR metric. dEs is evaluated every 25 fidelity steps;
     the phase stops as soon as dEs <= 0.5 dEs(S) (cap 300 steps) and keeps the lowest-dEs checkpoint as O0.
     Then 300 projected energy SR steps from O0 (equal energy steps to C), scored after 100 and 300.
Metric: dEs = min_{a>=0} <a|S H S|a>/<a|a> - E_0 with S = sgn(psi_P) (P1.2 t1_vote.dE_sign); dEt = E(psi_P) - E_0.

PRE-REGISTERED RULE (medians over the 3 seeds; c0 = med dEs(S), O0 = med dEs(O0), Om = med dEs(O after 300),
Cm = med dEs(C after 300), dE = med(dEt(O) - dEt(C)) per seed, drift = med(dEs(O,300) - dEs(O,100))):
  NO-TARGET    c0 < 0.005 after the fallbacks (no residual sign error to probe).
  ORACLE-MISS  O0 > 0.5 c0: even a sign-targeted fit at this width cannot hold half the sign error, i.e. no
               better-sign state is reachable at this width (capacity reading supported). P1.3 stays closed.
  Otherwise the oracle is valid and the release decides:
  BASIN        Om <= 0.75 Cm, and o <= 0.75 c on >= 2 of 3 seeds, and dE <= -1e-3 (strictly lower energy, so
               energy SR would prefer it but cold starts do not find it), and drift <= 0.25 (Cm - Om).
               REOPENS P1.3 (with the direct probe as its route).
  SIGN-DEGENERATE  the sign and drift conditions of BASIN hold but |dE| < 1e-3: SR has nothing to prefer, no
               target for an optimizer fix; stays closed.
  TRADE (K3)   the sign condition holds but dE > +1e-3: the better sign costs energy, CAPACITY; stays closed.
  RELAX (K2)   Om >= 0.9 Cm: the better-sign state relaxes back to the cold level, CAPACITY now directly probed;
               stays closed.
  UNCLEAR      otherwise; stays closed, recorded as a scope note.
Validity: the ED gap assert of t1c; CHECK F the fidelity force matches central finite differences of -log F / 2
  (rel < 1e-5) on size 0; the target t is in the chi sector (|P t - t| < 1e-10 |t|).
Budget (enforced): RLIMIT_CPU per invocation (--cpu_cap, default 580 s); 3 timed SR steps project the cost of
  each invocation before the heavy part; abort if over 90% of the invocation cap or if the leaf ledger
  (results/ledger.json) plus the projection exceeds --task_cap (default 1800 CPU-s). One thread by default.

Usage
-----
  python3 t1_basin_probe.py --mode checks
  python3 t1_basin_probe.py --mode run --size 2 --width 6 --seeds 11
  python3 t1_basin_probe.py --mode verdict --size 2 --width 6
Options: --mode {checks,run,verdict}; --size {0,1,2}; --width INT (default 6); --seeds LIST (default 11,22,33);
         --threads INT (1..4); --cpu_cap FLOAT (default 580); --task_cap FLOAT (default 1800).
"""
import os
import sys


def _pre_threads():
    n = 1
    for i, a in enumerate(sys.argv):
        if a.startswith("--threads="):
            n = int(a.split("=")[1])
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    n = max(1, min(4, n))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)
    return n


NTHREADS = _pre_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P12 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r6")
sys.path.insert(0, P12)
import t1c_warm as C  # noqa: E402
import t1e_closed as E  # noqa: E402
from tj_core import LRAnsatz  # noqa: E402
import p21_harness as H21  # noqa: E402

OUT = os.path.join(HERE, "results")
LEDGER = os.path.join(OUT, "ledger.json")
log = C.log


def ledger_used():
    return float(sum(r["cpu"] for r in json.load(open(LEDGER)))) if os.path.exists(LEDGER) else 0.0


def ledger_add(mode, cpu, note):
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    rows.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, cpu=cpu, note=note))
    json.dump(rows, open(LEDGER, "w"), indent=1)


def start_fn(k, w, seed):
    p12 = E.foilP_fn(k, seed)
    if w == 6 and os.path.exists(p12):
        return p12, True
    return os.path.join(OUT, f"S_{k}_w{w}_{seed}.npz"), False


def load(fn, k, w, seed):
    m, psi0, E0, ref, gap = C.model(k)
    nb = LRAnsatz(m, ref, "full", K=1, Hs=w, seed=seed)
    z = np.load(fn)
    for key in nb.keys:
        nb.p[key] = z[key].copy()
    return nb


def fid_step(ans, m, sg, tgt, args, rng):
    psi_t = ans.amplitudes(args.chunk)
    psi = sg.project(psi_t)
    nn, ov = psi @ psi, tgt @ psi
    F = ov * ov / (nn * (tgt @ tgt))
    f = ans.force(psi / nn - tgt / ov, args.chunk)
    pc = psi ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = ans.per_sample_O_proj(sg, idx, psi, args.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    dx = (f - Oc.T @ np.linalg.solve(Oc @ Oc.T + lam * np.eye(len(idx)), Oc @ f)) / lam
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return float(F), step


def run_checks(args):
    os.makedirs(OUT, exist_ok=True)
    m, psi0, E0, ref, gap = C.model(0)
    sg, gen = E.group(0)
    a = LRAnsatz(m, ref, "full", K=1, Hs=4, seed=3)
    x = a.flat() + 0.05 * np.random.default_rng(1).standard_normal(a.n_params())
    a.set_flat(x)
    psiP = sg.project(a.amplitudes(8192))
    tgt = np.where(psi0 >= 0, 1.0, -1.0) * np.abs(psiP)
    tin = float(np.linalg.norm(sg.project(tgt) - tgt) / np.linalg.norm(tgt))

    def nlogF(xx):
        a.set_flat(xx)
        p = sg.project(a.amplitudes(8192))
        return -np.log((tgt @ p) ** 2 / ((p @ p) * (tgt @ tgt)))
    a.set_flat(x)
    p = sg.project(a.amplitudes(8192))
    f = a.force(p / (p @ p) - tgt / (tgt @ p), 8192)
    worst = 0.0
    for kk in np.random.default_rng(4).choice(len(x), 12, replace=False):
        e = np.zeros(len(x)); e[kk] = 1e-6
        fd = (nlogF(x + e) - nlogF(x - e)) / 2e-6 / 2.0
        worst = max(worst, abs(fd - f[kk]) / max(abs(fd), 1e-8))
    a.set_flat(x)
    ok = dict(F=worst, target_in_sector=tin, valid=bool(worst < 1e-5 and tin < 1e-10))
    log(f"size 0 group {gen}: CHECK F fidelity force vs FD worst rel {worst:.2e}; target in sector {tin:.1e}; "
        f"valid {ok['valid']}")
    json.dump(ok, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    ledger_add("checks", time.process_time(), "")
    return 0 if ok["valid"] else 1


def run(args):
    os.makedirs(OUT, exist_ok=True)
    k, w = args.size, args.width
    m, psi0, E0, ref, gap = C.model(k)
    sg, gen = E.group(k)
    jobs = [s for s in args.seed_list if not os.path.exists(os.path.join(OUT, f"R_{k}_w{w}_{s}.json"))]
    if not jobs:
        log("nothing to do")
        return 0
    per = E.time_steps(LRAnsatz(m, ref, "full", K=1, Hs=w, seed=11), m, sg, "chi")
    nsteps = sum((0 if start_fn(k, w, s)[0] and os.path.exists(start_fn(k, w, s)[0]) else 300) + 900 for s in jobs)
    proj = per * nsteps * 1.05 + 20.0 * len(jobs)          # + about 14 dEs evaluations of ~1 s per seed
    used, prior = time.process_time(), ledger_used()
    log(f"per-step {per:.3f} CPU-s; budget: projected {proj:.0f} CPU-s + {used:.0f} used; invocation cap "
        f"{args.cpu_cap:.0f}; leaf ledger {prior:.0f} of {args.task_cap:.0f}")
    if proj + used > 0.9 * args.cpu_cap or prior + proj + used > args.task_cap:
        log("BUDGET-ABORT")
        ledger_add("run-abort", time.process_time(), f"size {k} w {w} seeds {jobs}")
        return 3
    rec = C.sra(300)
    for seed in jobs:
        c0 = time.process_time()
        fn, shared = start_fn(k, w, seed)
        if not os.path.exists(fn):
            nb = LRAnsatz(m, ref, "full", K=1, Hs=w, seed=seed)
            H21.optimize(nb, m, sg, "chi", 0, C.sra(300), np.random.default_rng(seed + 5), f"S s{k} w{w} {seed}")
            np.savez(fn, **{key: nb.p[key] for key in nb.keys})
        S = load(fn, k, w, seed)
        dEtS, dEsS = E.score(S, m, psi0, E0, sg)
        row = dict(size=k, width=w, seed=seed, chi=gen, n_params=S.n_params(), start_shared_with_P12=shared,
                   S=dict(dEt=dEtS, dEs=dEsS))
        log(f"seed {seed}: start S dEt {dEtS:.5f} dEs {dEsS:.5f}")
        # C: cold continuation
        a = load(fn, k, w, seed)
        H21.optimize(a, m, sg, "chi", 0, C.sra(100), np.random.default_rng(seed + 11), f"C {seed} a")
        row["C100"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)))
        H21.optimize(a, m, sg, "chi", 0, C.sra(200), np.random.default_rng(seed + 12), f"C {seed} b")
        row["C300"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)))
        # O: sign-oracle fidelity phase
        a = load(fn, k, w, seed)
        tgt = np.where(psi0 >= 0, 1.0, -1.0) * np.abs(sg.project(a.amplitudes(8192)))
        rng = np.random.default_rng(seed + 13)
        best = (dEsS, a.flat().copy(), 0)
        trace = []
        for it in range(1, 301):
            F, step = fid_step(a, m, sg, tgt, rec, rng)
            a.set_flat(a.flat() - step)
            if it % 25 == 0:
                dEt_i, dEs_i = E.score(a, m, psi0, E0, sg)
                trace.append((it, F, dEt_i, dEs_i))
                log(f"O {seed} fid it {it}: F {F:.6f} dEt {dEt_i:.5f} dEs {dEs_i:.5f}")
                if dEs_i < best[0]:
                    best = (dEs_i, a.flat().copy(), it)
                if dEs_i <= 0.5 * dEsS:
                    break
        a.set_flat(best[1])
        row["O0"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)), fid_steps=best[2], trace=trace)
        H21.optimize(a, m, sg, "chi", 0, C.sra(100), np.random.default_rng(seed + 21), f"O {seed} a")
        row["O100"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)))
        H21.optimize(a, m, sg, "chi", 0, C.sra(200), np.random.default_rng(seed + 22), f"O {seed} b")
        row["O300"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)))
        row["cpu"] = time.process_time() - c0
        json.dump(row, open(os.path.join(OUT, f"R_{k}_w{w}_{seed}.json"), "w"), indent=1)
        log(f"seed {seed}: S {dEsS:.5f} | C100 {row['C100']['dEs']:.5f} C300 {row['C300']['dEs']:.5f} "
            f"(dEt {row['C300']['dEt']:.5f}) | O0 {row['O0']['dEs']:.5f} after {best[2]} fid steps "
            f"(dEt {row['O0']['dEt']:.5f}) O100 {row['O100']['dEs']:.5f} O300 {row['O300']['dEs']:.5f} "
            f"(dEt {row['O300']['dEt']:.5f}) ({row['cpu']:.0f} CPU-s)")
    ledger_add("run", time.process_time(), f"size {k} w {w} seeds {jobs}")
    log(f"leaf ledger now {ledger_used():.0f} s")
    return 0


def decide(rows):
    med = lambda f: float(np.median([f(r) for r in rows]))  # noqa: E731
    c0 = med(lambda r: r["S"]["dEs"])
    O0 = med(lambda r: r["O0"]["dEs"])
    Om, Cm = med(lambda r: r["O300"]["dEs"]), med(lambda r: r["C300"]["dEs"])
    dE = med(lambda r: r["O300"]["dEt"] - r["C300"]["dEt"])
    drift = med(lambda r: r["O300"]["dEs"] - r["O100"]["dEs"])
    nwin = sum(r["O300"]["dEs"] <= 0.75 * r["C300"]["dEs"] for r in rows)
    st = dict(c0=c0, O0=O0, Om=Om, Cm=Cm, dE=dE, drift=drift, nwin=nwin)
    if c0 < 0.005:
        return "NO-TARGET", st
    if O0 > 0.5 * c0:
        return "ORACLE-MISS", st
    sign = Om <= 0.75 * Cm and nwin >= 2
    if sign and drift <= 0.25 * (Cm - Om):
        if dE <= -1e-3:
            return "BASIN", st
        if abs(dE) < 1e-3:
            return "SIGN-DEGENERATE", st
    if sign and dE > 1e-3:
        return "TRADE (K3, capacity)", st
    if Om >= 0.9 * Cm:
        return "RELAX (K2, capacity)", st
    return "UNCLEAR", st


def run_verdict(args):
    def syn(S, O0, O1, O3, C3, eO, eC):
        return dict(S=dict(dEs=S), O0=dict(dEs=O0), O100=dict(dEs=O1), O300=dict(dEs=O3, dEt=eO),
                    C300=dict(dEs=C3, dEt=eC))
    for want, r in (("NO-TARGET", syn(0.003, 0.001, 0.001, 0.001, 0.003, 0.1, 0.1)),
                    ("ORACLE-MISS", syn(0.02, 0.015, 0.02, 0.02, 0.02, 0.1, 0.1)),
                    ("BASIN", syn(0.02, 0.005, 0.006, 0.006, 0.02, 0.09, 0.1)),
                    ("SIGN-DEGENERATE", syn(0.02, 0.005, 0.006, 0.006, 0.02, 0.1, 0.1)),
                    ("TRADE (K3, capacity)", syn(0.02, 0.005, 0.006, 0.006, 0.02, 0.11, 0.1)),
                    ("RELAX (K2, capacity)", syn(0.02, 0.005, 0.019, 0.019, 0.02, 0.1, 0.1)),
                    ("UNCLEAR", syn(0.02, 0.005, 0.017, 0.017, 0.02, 0.1, 0.1))):
        got = decide([r, r, r])[0]
        print(f"synthetic {want}: {got} {'OK' if got == want else 'MISMATCH'}")
    chk = json.load(open(os.path.join(OUT, "checks.json"))) if os.path.exists(os.path.join(OUT, "checks.json")) else {}
    rows = [json.load(open(os.path.join(OUT, f"R_{args.size}_w{args.width}_{s}.json"))) for s in args.seed_list
            if os.path.exists(os.path.join(OUT, f"R_{args.size}_w{args.width}_{s}.json"))]
    for r in rows:
        print(f"  seed {r['seed']}: S {r['S']['dEs']:.5f} C100 {r['C100']['dEs']:.5f} C300 {r['C300']['dEs']:.5f} "
              f"| O0 {r['O0']['dEs']:.5f} ({r['O0']['fid_steps']} fid steps) O100 {r['O100']['dEs']:.5f} "
              f"O300 {r['O300']['dEs']:.5f} | dEt O300 {r['O300']['dEt']:.5f} C300 {r['C300']['dEt']:.5f}")
    if len(rows) < len(args.seed_list):
        print("VERDICT: INCOMPLETE")
    elif not chk.get("valid"):
        print("VERDICT: INVALID")
    else:
        v, st = decide(rows)
        print("  " + " ".join(f"{a} {b:.5f}" if isinstance(b, float) else f"{a} {b}" for a, b in st.items()))
        print(f"VERDICT (size {args.size}, width {args.width}): {v}")
    print(f"leaf CPU ledger: {ledger_used():.0f} s")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.3 direct basin probe (see docstring)")
    ap.add_argument("--mode", choices=["checks", "run", "verdict"], required=True)
    ap.add_argument("--size", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu_cap", type=float, default=580.0)
    ap.add_argument("--task_cap", type=float, default=1800.0)
    a = ap.parse_args(argv)
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    return a


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 30))
    log(f"threads {NTHREADS}")
    return {"checks": run_checks, "run": run, "verdict": run_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
