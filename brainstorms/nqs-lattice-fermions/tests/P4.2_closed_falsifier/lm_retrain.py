#!/usr/bin/env python3
r"""
P4.2 closed-leaf falsifier (2026-11-16): is the size-independent family's 4x3 error optimiser-limited?
=====================================================================================================

The P4.2 Saturation Diagnostic says that no size-independent MFNG-O component gives even a 1.1x gain
on 4x3 OBC 5+5 (projected exact-sum SR, 600 steps; base 1.54%), that a 2.4x gain needs a
site-resolved, symmetry-breaking pair matrix, and that larger families containing the base ended
WORSE (c8h32 2.02%), i.e. single-seed SR results are upper bounds (basin dependence). This script
retrains size-independent families with a stronger optimiser: the exact-sum linear method (LM) on
the projected energy, with an energy-checked stabilisation scan, from the same initial state.

LM step (Toulouse & Umrigar 2007, exact sums). At theta, b = U^T psi (projected amplitudes), B = db/dtheta
(irrep x P); tangent T = (B - b (b^T B)/|b|^2)/|b|; p0 = b/|b|. Solve the generalised eigenproblem of
  H = [[E, T^T H p0], [T^T H p0, T^T H T + a I]],  S = [[1, 0], [0, T^T T]]
(overlap eigenvalues below 1e-10 of the largest dropped), take the eigenvector with the largest
|v_0| among the lowest few, dtheta = v[1:]/v[0], cap |dtheta| <= --max-step, and accept the a in
--shifts (scanned from large to small) with the lowest exact projected energy, only if it lowers the
energy; if no a lowers it, fall back to one projected-SR step (r5_common settings) with backtracking.

Pre-registered rule (also in the leaf file, written before any run)
  r* = the lowest final relative error over the size-independent LM runs (base, spin_jas2cs, c8h32).
  INCONCLUSIVE if the LM base run does not end at or below the SR-600 base value (1.536%): the
    optimiser is not better, so the test says nothing.
  FALSIFIED (reopens P4.2 with the route "the size-independent plateau is optimiser-limited") if
    r* <= 0.77% (a 2x gain over SR base, the critic's bottleneck threshold).
  CONFIRMED if r* > 1.02% (no size-independent run reaches the free-F oracle's SR-600 value).
  AMBIGUOUS otherwise (a T2 LM retrain on 4x4 is then prepared, not run).
  Reported beside it: whether the LM supersets end below LM base (the basin caveat).

Command-line options
--------------------
  --variant STR     base | spin_jas2cs | c8h32 (base: channels 4, bf-hidden 16; spin_jas2cs: base +
                    --spin-feat + --jas2 cs; c8h32: channels 8, bf-hidden 32). All size independent.
  --iters INT       maximum LM iterations (default 400).
  --wall FLOAT      wall-clock limit in seconds (default 480; the 10 CPU-minute cap, 1 thread).
  --shifts STR      comma list of LM diagonal shifts a (default "1,1e-1,1e-2,1e-3").
  --max-step FLOAT  cap on |dtheta| (default 0.5, the SR trust region of r3_core).
  --seed INT        init seed (default 0; init rng 2000 + seed as critic_capacity.py).
  --out PATH        result JSON (default results/lm_<variant>.json).
  --aggregate DIR   apply the rule to DIR/lm_*.json and exit.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "..", "P4.2_size_transfer_r5")
sys.path.insert(0, R5)
import r3_core as rc  # noqa: E402
import critic_capacity as cc  # noqa: E402

SR_BASE, SR_FREEF = 0.015363, 0.010189
VARIANTS = {"base": [], "spin_jas2cs": ["--spin-feat", "--jas2", "cs"],
            "c8h32": ["--channels", "8", "--bf-hidden", "32"]}


def tangent(cl, net, th, chunk):
    psi = rc.state_vector(net, th, cl.basis, chunk)
    b = cl.U.T @ psi
    n2 = float(b @ b)
    cache = net.precompute(th, True)
    B = np.zeros((cl.U.shape[1], net.npar))
    for s in range(0, len(cl.basis), chunk):
        sl = slice(s, s + chunk)
        _, _, O = net.evaluate(th, cl.basis[sl], True, cache)
        B += cl.U[sl].T @ (psi[sl, None] * O)
    p0 = b / np.sqrt(n2)
    Hp0 = cl.HG @ p0
    E = float(p0 @ Hp0)
    T = (B - np.outer(b, b @ B) / n2) / np.sqrt(n2)
    HT = np.asarray(cl.HG @ T)
    return E, T.T @ Hp0, T.T @ HT, T.T @ T


def energy(cl, th, chunk):
    return rc.reduced(cl, th, chunk, False)[0]


def lm_dirs(E, h0, Hkk, Skk, shifts, max_step):
    se, sv = np.linalg.eigh(Skk)
    keep = se > 1e-10 * se.max()
    X = sv[:, keep] / np.sqrt(se[keep])
    out = []
    Hs = (Hkk + Hkk.T) / 2
    for a in shifts:
        Hr = np.zeros((keep.sum() + 1,) * 2)
        Hr[0, 0] = E
        Hr[0, 1:] = Hr[1:, 0] = X.T @ h0
        Hr[1:, 1:] = X.T @ (Hs + a * np.eye(len(Hs))) @ X
        w, V = eigh(Hr)
        k = int(np.argmax(np.abs(V[0, :min(5, len(w))])))
        v = V[:, k]
        d = X @ (v[1:] / v[0])
        nd = np.linalg.norm(d)
        if nd > max_step:
            d *= max_step / nd
        out.append((a, d))
    return out


def run(a):
    args = cc.parse_args(["--mode", "train"] + VARIANTS[a.variant] + ["--out", "x"])
    T, tgt = cc.setup_target(args)
    lat, nup = T["lat"], T["nup"]
    net = cc.make_net(lat, nup, args)
    th = net.init(np.random.default_rng(2000 + a.seed), 2 * nup / lat.N, args.t, args.tp, args.delta0, args.eta)
    cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
    E0, absE0 = T["E0"], abs(T["E0"])
    shifts = [float(x) for x in a.shifts.split(",")]
    t0 = time.time()
    E = energy(cl, th, args.chunk)
    hist = [(E - E0) / absE0]
    acc = {"lm": 0, "sr": 0, "none": 0}
    rc.log(f"LM retrain {a.variant}: npar={net.npar} start relerr={hist[0]:.4%}")
    for it in range(1, a.iters + 1):
        if time.time() - t0 > a.wall:
            rc.log(f"wall limit reached after {it - 1} iterations")
            break
        Et, h0, Hkk, Skk = tangent(cl, net, th, args.chunk)
        best = (E, None, None)
        for sh, d in lm_dirs(Et, h0, Hkk, Skk, shifts, a.max_step):
            En = energy(cl, th + d, args.chunk)
            if np.isfinite(En) and En < best[0] - 1e-13:
                best = (En, d, sh)
        if best[1] is not None:
            th = th + best[1]
            E = best[0]
            acc["lm"] += 1
        else:
            Eg, n2, g, S = rc.reduced(cl, th, args.chunk, True)
            Sr = S + args.diag_shift * np.diag(np.diag(S)) + 1e-6 * np.diag(S).max() * np.eye(len(S))
            d = np.linalg.solve(Sr, g)
            moved = False
            lr = args.lr
            for _ in range(12):
                En = energy(cl, th - lr * d, args.chunk)
                if np.isfinite(En) and En < E - 1e-13:
                    th, E, moved = th - lr * d, En, True
                    break
                lr *= 0.5
            acc["sr" if moved else "none"] += 1
        hist.append((E - E0) / absE0)
        if it % 10 == 0:
            rc.log(f"  it {it} relerr={hist[-1]:.4%} accepted lm/sr/none={acc} ({time.time() - t0:.0f}s)")
    res = dict(variant=a.variant, npar=net.npar, target=tgt, E=E, relerr=hist[-1], hist=hist, accepted=acc,
               iters=len(hist) - 1, wall_s=time.time() - t0, theta=th.tolist())
    rc.log(f"RESULT {a.variant}: npar={net.npar} iters={len(hist) - 1} relerr={hist[-1]:.4%} "
           f"(SR-600 base 1.536%) accepted={acc} wall={res['wall_s']:.0f}s")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), default=float)


def aggregate(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "lm_*.json"))):
        x = json.load(open(f))
        R[x["variant"]] = x
        print(f"{x['variant']:12s} npar={x['npar']:4d} iters={x['iters']:4d} relerr={x['relerr']:.4%} "
              f"accepted={x['accepted']} wall={x['wall_s']:.0f}s")
    if "base" not in R:
        print("VERDICT: INCOMPLETE")
        return
    if R["base"]["relerr"] > SR_BASE:
        print("VERDICT: INCONCLUSIVE (LM base not below SR-600 base)")
        return
    rstar = min(x["relerr"] for x in R.values())
    v = "FALSIFIED" if rstar <= SR_BASE / 2 else ("CONFIRMED" if rstar > SR_FREEF else "AMBIGUOUS")
    sup = {k: x["relerr"] < R["base"]["relerr"] for k, x in R.items() if k != "base"}
    print(f"r* = {rstar:.4%}; supersets below LM base: {sup}")
    print(f"VERDICT P4.2 optimiser falsifier: {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="P4.2 LM retrain falsifier (see docstring)")
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--wall", type=float, default=480.0)
    ap.add_argument("--shifts", default="1,1e-1,1e-2,1e-3")
    ap.add_argument("--max-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--aggregate", default="")
    a = ap.parse_args(argv)
    if a.aggregate:
        aggregate(a.aggregate if os.path.isabs(a.aggregate) else os.path.join(HERE, a.aggregate))
        return
    if not a.out:
        a.out = os.path.join(HERE, "results", f"lm_{a.variant}.json")
    run(a)


if __name__ == "__main__":
    main()
