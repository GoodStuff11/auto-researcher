#!/usr/bin/env python3
"""
P3.1 Round 12: tier-1 test of Route 12.3d (Davidson-block bracket) on unseen gap-surgery instances.
===================================================================================================

Pre-registered rule: leaf, "Route 12.3d", written before this script was run.
For each trial state psi (t1_scru.trial_state, the 8 fixed perturbations), on an instance built by
t1_davidson.build with gap surgery to --delta:
  DAV block   10.4 chain (sources Q(H-E)^p phi, p = 1..3; T = 4; grade-R noise; shift E_L1R - 0.1 t), then the
              witness pass span{phi_4, (H-E)phi_4, y~_p, K y~_p}; interval = eigenvalue range of A compressed onto
              the two lowest Ritz vectors (t1_r12.chain / t1_r12.bracket)                                (deciding)
  L1R block   A compressed onto span{psi, (H-E)psi} (no trained vector)                                  (Gate 2 foil)
Scored item |delta| >= 0.1 |a_0|; covered if a_0 in [lo - 0.1|delta|, hi + 0.1|delta|]; width ratio (hi - lo)/w_ex
with w_ex the eigenvalue range of A on span{psi_0, psi_1}.
Rule: KILL if coverage < 0.5 or median width ratio > 3 on any instance; KILL (Gate 2) if the L1R block has
coverage >= 0.8 and median width ratio <= 1.5 on every instance; PASS if the DAV block has coverage >= 0.8 and
median width ratio <= 1.5 on every instance; UNCLEAR otherwise; INCOMPLETE if an instance has < 6 scored items.

Usage
  python3 t1_r12d.py --mode main --size 4x3 --delta 0.05     # JSON results/r12d_<size>_d<delta>.json
  python3 t1_r12d.py --mode verdict                           # reads the six JSONs (deltas 0.03 and 0.05)
Options
  --mode {main, verdict}
  --size {4x2, 5x2, 4x3}
  --delta FLOAT   gap surgery target (required for main; the pre-registered values are 0.03 and 0.05)
  --states INT    trial states (default 8), soft CPU guard as in t1_r12.py
  --cpu-limit INT hard RLIMIT_CPU seconds (default 600)
  --threads {1,2,3,4}  BLAS threads (default 1)
"""
import os
import sys
_nt = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _nt
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r12", os.path.join(HERE, "t1_r12.py"))
r12 = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(r12)
sys.argv = _argv
np, t1, t1s, dav, log, REAL = r12.np, r12.t1, r12.t1s, r12.dav, r12.log, r12.REAL
DELTAS = (0.03, 0.05)


def run(I, nstates, cap, dump):
    perts = t1.perturbations(I)
    rng = np.random.default_rng(7)
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
    wex = {k: (lambda b: b[1] - b[0])(r12.bracket(I, k, [I.psi0, I.U[:, 0]])) for k in REAL}
    rows, worst = [], 0.0
    for si, lab in enumerate(list(perts)[:nstates]):
        c0 = r12.cpu()
        if r12.cpu() + 1.2 * worst > cap - 15:
            log(f"soft CPU guard: stop before state {si}")
            break
        psi = t1s.trial_state(I, perts[lab], rng)
        ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
        dl = {k: a0[k] - ap[k] for k in REAL}
        row = dict(size=I.name, state=lab, a0=a0, delta=dl, wex=wex,
                   scored={k: bool(abs(dl[k]) >= t1.SCORE_REL * abs(a0[k])) for k in REAL})
        L = [r12.unit(psi), r12.unit(t1s.resid(I, psi))]
        L[1] = r12.unit(L[1] - L[0] * (L[0] @ L[1]))
        row["br_L1R"] = {k: r12.bracket(I, k, L) for k in REAL}
        rs = np.random.default_rng(40_000 + 97 * I.m.V + si)
        phi, info = r12.chain(I, psi, r12.make_sources(I, "DAV", rs), rs, 4, True)
        row["br_DAV"] = {k: r12.bracket(I, k, info["block"]) for k in REAL}
        row["ghat_DAV"] = info["ghat"]
        rows.append(row)
        worst = max(worst, r12.cpu() - c0)
        json.dump(dict(rows=rows), open(dump, "w"), indent=1)
        log(f"{I.name} {lab:7s} cpu {r12.cpu() - c0:.0f}s | " + " | ".join(
            f"{k}{'' if row['scored'][k] else '(ns)'} cov DAV {r12.covered(row, k, 'DAV')} L1R "
            f"{r12.covered(row, k, 'L1R')} w/wex DAV {(row['br_DAV'][k][1] - row['br_DAV'][k][0]) / wex[k]:.2f} "
            f"L1R {(row['br_L1R'][k][1] - row['br_L1R'][k][0]) / wex[k]:.2f}" for k in REAL))
    return rows


def score(rows, arm):
    it = r12.items(rows)
    if not it:
        return 0.0, float("nan"), float("nan"), 0
    cov = sum(1 for r, k in it if r12.covered(r, k, arm)) / len(it)
    wr = [(r[f"br_{arm}"][k][1] - r[f"br_{arm}"][k][0]) / r["wex"][k] for r, k in it]
    wd = [(r[f"br_{arm}"][k][1] - r[f"br_{arm}"][k][0]) / abs(r["delta"][k]) for r, k in it]
    return cov, float(np.median(wr)), float(np.median(wd)), len(it)


def verdict():
    res = {}
    for d in DELTAS:
        for s in r12.SIZES:
            f = os.path.join(r12.RES, f"r12d_{s}_d{d}.json")
            if not os.path.exists(f):
                print(f"  missing {f}"); continue
            rows = json.load(open(f))["rows"]
            for arm in ("DAV", "L1R"):
                res[(arm, s, d)] = c = score(rows, arm)
                print(f"  {arm} {s} delta {d}: n={c[3]:2d} states={len(rows)} coverage {c[0]:.2f} "
                      f"median width/w_ex {c[1]:.2f} median width/|delta| {c[2]:.2f}")
    if len(res) < 12 or any(x[3] < 6 for x in res.values()):
        print("VERDICT 12.3d: INCOMPLETE"); return

    def ok(arm):
        return all(res[(arm, s, d)][0] >= 0.8 and res[(arm, s, d)][1] <= 1.5 for s in r12.SIZES for d in DELTAS)
    D = [res[("DAV", s, d)] for s in r12.SIZES for d in DELTAS]
    if any(x[0] < 0.5 or x[1] > 3 for x in D):
        v = "KILL"
    elif ok("L1R"):
        v = "KILL (Gate 2)"
    elif ok("DAV"):
        v = "PASS"
    else:
        v = "UNCLEAR"
    print(f"VERDICT 12.3d: {v}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("main", "verdict"), required=True)
    p.add_argument("--size", choices=r12.SIZES, default="4x2")
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=1)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    if a.mode == "verdict":
        return verdict()
    assert a.delta is not None, "--delta is required for main"
    I, info = dav.build(a.size, 15, a.delta)
    out = os.path.join(r12.RES, f"r12d_{a.size}_d{a.delta}.json")
    rows = run(I, a.states, a.cpu_limit, out)
    json.dump(dict(info=info, rows=rows), open(out, "w"), indent=1, default=str)
    log(f"total CPU {r12.cpu():.0f}s")


if __name__ == "__main__":
    main()
