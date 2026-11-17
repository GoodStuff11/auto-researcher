#!/usr/bin/env python3
r"""
P4.2 r10 (2026-11-17): truncated-metric linear-method polish of the 704844 SR-3000 states (4x3 OBC 5+5).

Rule: RULE.md (written before any run). Recipe = P4.1 R11 (tests/P4.1_cheap_su2_r11/realise.py, Tan class):
tangent T orthogonalised against Psi (lm_retrain.tangent, exact sums over the projected basis), metric
eigenvalues below thr * max dropped (thr 1e-5), lowest LM root via the secular equation with a level shift
raised until |z| <= tau (0.05), dtheta = Pm z, line search s in {1, 1/2, 1/4} on the exact projected energy,
stop at the first step with no lowering trial. Exact energies: no statistical error.

Usage: python3 polish.py --variant base|spin_jas2cs|c8h32 --theta-json results/sr3000_<v>.json [--steps 8] [--thr 1e-5] [--tau 0.05]
       python3 polish.py --aggregate
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.join(HERE, "..", "P4.2_closed_falsifier")
sys.path.insert(0, CF)
import lm_retrain as lr  # noqa: E402  (imports r3_core, critic_capacity read-only)
cc, rc = lr.cc, lr.rc

FALS, CONF = 0.0077, 0.0102


def lm_z(mu, gq, E, lam):
    mu = mu + lam
    f = lambda eps: E - eps - np.sum(gq ** 2 / (mu - eps))  # noqa: E731
    hi = mu[0] - 1e-14 * max(1.0, abs(mu[0]))
    lo = min(E, mu[0]) - 1.0 - np.linalg.norm(gq)
    while f(lo) < 0:
        lo -= 10 * (1 + abs(lo))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    eps = 0.5 * (lo + hi)
    return -gq / (mu - eps)


def step_dir(E, g, HT, ST, thr, tau):
    w, U = np.linalg.eigh(0.5 * (ST + ST.T))
    k = w > thr * w.max()
    Pm = U[:, k] / np.sqrt(w[k])
    gt = Pm.T @ g
    Ht = Pm.T @ (0.5 * (HT + HT.T)) @ Pm
    mu, Q = np.linalg.eigh(Ht)
    gq = Q.T @ gt
    zq = lm_z(mu, gq, E, 0.0)
    lam = 0.0
    if tau and np.linalg.norm(zq) > tau:
        lo, hi = 0.0, 1.0
        while np.linalg.norm(lm_z(mu, gq, E, hi)) > tau:
            hi *= 4.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(lm_z(mu, gq, E, mid)) > tau:
                lo = mid
            else:
                hi = mid
        lam = hi
        zq = lm_z(mu, gq, E, lam)
    z = Q @ zq
    pred = (E + 2 * z @ gt + z @ Ht @ z) / (1 + z @ z)
    return Pm @ z, pred, int(k.sum()), lam, float(np.linalg.norm(z))


def run(a):
    if not a.theta_json:
        sys.exit("the 704844 SR JSONs hold no parameters ('params' is the CLI namespace); "
                 "regenerate with regen_sr.py and pass --theta-json results/sr3000_<variant>.json")
    src = json.load(open(a.theta_json))
    args = cc.parse_args(["--mode", "train"] + lr.VARIANTS[a.variant] + ["--out", "x"])
    T, tgt = cc.setup_target(args)
    net = cc.make_net(T["lat"], T["nup"], args)
    th = np.asarray(src["theta"], dtype=float)
    assert th.size == net.npar, (th.size, net.npar)
    cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
    E0, aE0 = T["E0"], abs(T["E0"])
    c0, t0 = time.process_time(), time.time()
    E = lr.energy(cl, th, args.chunk)
    chk = abs(E - src["E"])
    rc.log(f"{a.variant}: npar={net.npar} E(theta_SR)={E:.10f} stored {src['E']:.10f} |diff|={chk:.1e} "
           f"relerr={(E - E0) / aE0:.4%}")
    assert chk < 1e-8, "INVALID: reloaded energy does not match stored SR energy"
    Ein = E
    steps = []
    for it in range(1, a.steps + 1):
        Et, g, HT, ST = lr.tangent(cl, net, th, args.chunk)
        d, pred, rank, lam, zn = step_dir(Et, g, HT, ST, a.thr, a.tau)
        acc = None
        trials = []
        for s in (1.0, 0.5, 0.25):
            En = lr.energy(cl, th + s * d, args.chunk)
            trials.append((s, (En - E0) / aE0))
            if np.isfinite(En) and En < E:
                acc = (s, En)
                break
        rec = dict(step=it, E_before=E, pred_relerr=(pred - E0) / aE0, rank=rank, lam=lam, zn=zn,
                   dth=float(np.linalg.norm(d)), trials=trials)
        if acc is None:
            rec["accepted"] = False
            steps.append(rec)
            rc.log(f"  step {it}: no lowering trial {trials}; stop")
            break
        th = th + acc[0] * d
        E = acc[1]
        rec.update(accepted=True, s=acc[0], E=E, relerr=(E - E0) / aE0)
        steps.append(rec)
        rc.log(f"  step {it}: relerr {(E - E0) / aE0:.4%} (pred {(pred - E0) / aE0:.4%}) s={acc[0]} rank={rank} "
               f"|z|={zn:.3f} |dth|={acc[0] * rec['dth']:.3f} [{time.process_time() - c0:.0f} CPU-s]")
    res = dict(variant=a.variant, npar=net.npar, target=tgt, E0=E0, E_in=Ein, relerr_in=(Ein - E0) / aE0,
               E_out=E, relerr_out=(E - E0) / aE0, steps=steps, thr=a.thr, tau=a.tau,
               cpu_s=time.process_time() - c0, wall_s=time.time() - t0, theta=th.tolist())
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"polish_{a.variant}.json"), "w"), default=float)
    rc.log(f"RESULT {a.variant}: relerr {res['relerr_in']:.4%} -> {res['relerr_out']:.4%} "
           f"({len([s for s in steps if s['accepted']])} accepted steps, {res['cpu_s']:.0f} CPU-s)")


def aggregate():
    R = {}
    for v in ("base", "spin_jas2cs", "c8h32"):
        f = os.path.join(HERE, "results", f"polish_{v}.json")
        if os.path.exists(f):
            R[v] = json.load(open(f))
    cpu = 0.0
    for v, x in R.items():
        cpu += x["cpu_s"]
        na = len([s for s in x["steps"] if s["accepted"]])
        print(f"{v:12s} npar={x['npar']:4d} E {x['E_in']:.8f} -> {x['E_out']:.8f}  relerr {x['relerr_in']:.4%} -> "
              f"{x['relerr_out']:.4%}  removed {1 - x['relerr_out'] / x['relerr_in']:.3f} of relerr  "
              f"steps {na}  cpu {x['cpu_s']:.0f}s")
    if len(R) < 3:
        print("VERDICT: INCOMPLETE")
        return
    rin = sorted(R, key=lambda v: R[v]["relerr_in"])
    rout = sorted(R, key=lambda v: R[v]["relerr_out"])
    print(f"ranking before {rin}; after {rout}")
    print(f"best/base gain ratio before {R['base']['relerr_in'] / R[rin[0]]['relerr_in']:.3f}; "
          f"after {R['base']['relerr_out'] / R[rout[0]]['relerr_out']:.3f}")
    rstar = R[rout[0]]["relerr_out"]
    v = ("KILL (DIAGNOSTIC-FALSIFIED)" if rstar <= FALS else
         ("ATTRIBUTION HOLDS (DIAGNOSTIC-HOLDS)" if rstar > CONF else "AMBIGUOUS"))
    print(f"r* = {rstar:.4%} ({rout[0]}); total CPU {cpu / 60:.2f} min")
    print(f"VERDICT: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(lr.VARIANTS), default="base")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--thr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--theta-json", default="")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    aggregate() if a.aggregate else run(a)
