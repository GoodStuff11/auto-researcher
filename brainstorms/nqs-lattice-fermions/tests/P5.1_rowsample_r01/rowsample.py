#!/usr/bin/env python3
"""
P5.1 r01 T1 for R2 (sketched metric: timing) and R3 (leverage-score row sampling of the metric).  Rule: RULE.md.
===============================================================================================================

Run from this directory, thread variables exported before python and a hard CPU limit set:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python rowsample.py [--part {timing, all}]
Options
  --part timing   Part 1 only: time the Jacobian rows (all 157,216 representatives), the Gram, one eigh and one
                  exact energy, and print the R2 verdict.
  --part all      (default) Part 1, then Part 2: exact S_T and f at the polished `che`, and the 9-candidate step
                  for the exact metric and for row-sampled metrics (Born, guided, leverage; m = 3P, 8P; 2 draws).
Output: results/rowsample.json; log to stdout.
Reuses read only: ../P4.1_cheap_su2_r19/polish.py (ctx_for, logder, cpu_min, E0) and the polished theta of
../P4.1_cheap_su2_r22/results/theta_che.npy.
"""
import argparse
import json
import os
import sys
import time

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402  (read only)
import torch  # noqa: E402

CUTS, TAUS = (1e-5, 1e-6, 1e-7), (0.0005, 0.001, 0.002, 0.004)  # amendment 2 (RULE.md)
LAM_REL = 1e-7


def jac(ctx, th, CH=2048):
    """Y = [psi, d psi/d theta] * sqrt(orbit size) over all representatives (polish.build without HY)."""
    E, pr, phi0 = ctx.energy(th)
    Pn, nrep = ctx.m.nparam, len(ctx.reps)
    Y = np.empty((nrep, 1 + Pn))
    Y[:, 0] = pr * ctx.sw
    for r0 in range(0, nrep, CH):
        rr = ctx.reps[r0:r0 + CH]
        acc = np.zeros((len(rr), Pn))
        for g in range(4):
            ix = ctx.idx[g, rr]
            w = 0.25 * ctx.sg[g, rr] * phi0[ix]
            acc += w[:, None] * P.logder(ctx.m, th, ctx.nU[ix], ctx.nD[ix])
        Y[r0:r0 + CH, 1:] = acc * ctx.sw[r0:r0 + CH, None]
    return E, pr, Y


def centred(S, Hc0, E):
    """S (uncentred, psi column first, S_00 normalised to 1) -> S_T and f = H_c0 - E s."""
    S = S / S[0, 0]
    s = S[0, 1:]
    ST = S[1:, 1:] - np.outer(s, s)
    return 0.5 * (ST + ST.T), s


def step_gain(ctx, th, E, ST, f):
    w, U = np.linalg.eigh(ST)
    best, cands = 0.0, []
    for c in CUTS:
        k = w > c * w.max()
        Pm = U[:, k] / np.sqrt(w[k])
        gt = Pm.T @ f
        dir_ = -(Pm @ gt) / np.linalg.norm(gt)
        for tau in TAUS:
            En = ctx.energy(th + torch.tensor(tau * dir_))[0]
            cands.append(dict(cut=c, tau=tau, rank=int(k.sum()), dE=E - En))
            best = max(best, E - En)
    return best, cands


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--part", default="all", choices=["timing", "all"])
    a = pa.parse_args()
    out = {}
    t0 = time.time()
    ctx = P.ctx_for("che")
    th = torch.tensor(np.load(os.path.join(TESTS, "P4.1_cheap_su2_r22", "results", "theta_che.npy")))
    c = P.cpu_min()
    E = ctx.energy(th)[0]
    out["t_energy"] = P.cpu_min() - c
    c = P.cpu_min()
    E, pr, Y = jac(ctx, th)
    out["t_jacobian"] = P.cpu_min() - c - out["t_energy"]
    c = P.cpu_min()
    S = Y.T @ Y
    out["t_gram"] = P.cpu_min() - c
    c = P.cpu_min()
    w, U = np.linalg.eigh(S)
    out["t_eigh"] = P.cpu_min() - c
    out["eps"] = (E - P.E0) / abs(P.E0)
    out["R2"] = "KILLED" if out["t_jacobian"] >= 0.33 else "GO-ON"
    P.log(f"timing (CPU-min): energy {out['t_energy']:.3f}, jacobian {out['t_jacobian']:.3f}, gram "
          f"{out['t_gram']:.3f}, eigh {out['t_eigh']:.3f}; eps {out['eps']:.4e}; R2 {out['R2']}")
    if a.part == "all":
        HY0 = (ctx.Hr @ pr) * ctx.sw
        Hc0 = Y.T @ HY0
        nrm = S[0, 0]
        STx, s = centred(S, Hc0, E)
        f = Hc0[1:] / nrm - E * s
        gx, cx = step_gain(ctx, th, E, STx, f)
        out["gain_exact"], out["cands_exact"] = gx, cx
        P.log(f"exact metric: best gain {gx:.3e}; " + "; ".join(f"c {q['cut']:.0e} tau {q['tau']}: {q['dE']:+.2e}" for q in cx))
        assert gx > 0, "exact metric step gains nothing on the tau grid: amend the grid before Part 2"
        lam = LAM_REL * w.max()
        Z = Y @ (U / np.sqrt(w + lam))
        lev = np.einsum("ij,ij->i", Z, Z)
        del Z
        born = Y[:, 0] ** 2 / np.sum(Y[:, 0] ** 2)
        fl = np.abs(Y[:, 0]) * ctx.sw
        fl = fl / fl.sum()
        qs = dict(born=born, guided=0.8 * born + 0.2 * fl, leverage=0.5 * born + 0.5 * lev / lev.sum())
        lo = born < np.median(born)
        out["sum_lev"], out["lev_share_low_born"] = float(lev.sum()), float(lev[lo].sum() / lev.sum())
        P.log(f"effective dimension sum(lev) {lev.sum():.1f} of {Y.shape[1]}; leverage share on below-median Born "
              f"rows {out['lev_share_low_born']:.3f}")
        Pn = Y.shape[1] - 1
        res = {}
        for name, q in qs.items():
            for m in (8 * Pn,):  # RULE amend (before Part 2 ran): reported-only 3P dropped for the 10-CPU-min cap
                phis = []
                for seed in (1, 2):
                    r = np.random.default_rng(seed).choice(len(q), size=m, p=q)
                    Ys = Y[r] / np.sqrt(m * q[r])[:, None]
                    St, _ = centred(Ys.T @ Ys, None, E)
                    g, _ = step_gain(ctx, th, E, St, f)
                    phis.append(g / gx)
                res[f"{name}_{m}"] = phis
                P.log(f"{name} m={m}: phi {np.round(phis, 3)} [{P.cpu_min():.2f} CPU-min]")
        out["phi"] = res
        pl, pb = np.mean(res[f"leverage_{8 * Pn}"]), np.mean(res[f"born_{8 * Pn}"])
        out["R3"] = ("PASS" if pl >= 0.8 and pl >= pb + 0.15 else
                     "KILLED" if pl < 0.5 or pl <= pb + 0.05 else "MARGINAL")
        P.log(f"R3: phi(leverage, 8P) {pl:.3f}, phi(Born, 8P) {pb:.3f} -> {out['R3']}")
    out["cpu_min"], out["wall"] = P.cpu_min(), time.time() - t0
    json.dump(out, open(os.path.join(HERE, "results", "rowsample.json"), "w"), indent=1)
    P.log(f"done {P.cpu_min():.2f} CPU-min")


if __name__ == "__main__":
    main()
