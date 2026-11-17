#!/usr/bin/env python3
"""
P5.1 r01 T1b for R51.3 `levrow`: leverage scores computed at a stale parameter point.  Rule: RULE_stale.md.
==========================================================================================================

Run from this directory, thread variables exported before python and a hard CPU limit set:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python stale.py
No options.  Target: the ladder-polished `che` (../P4.1_cheap_su2_r22/results/theta_che.npy).  Stale points:
near = ../P4.1_cheap_su2_r25/results/theta_che_lbfgs.npy, far = the SR-trained `che` start.  Reuses rowsample.py
(jac, centred, step_gain) in this directory.  Output: results/stale.json.
"""
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rowsample as RS  # noqa: E402

P, torch = RS.P, RS.torch


def lev_at(ctx, th):
    _, _, Y = RS.jac(ctx, th)
    S = Y.T @ Y
    w, U = np.linalg.eigh(S)
    Z = Y @ (U / np.sqrt(w + RS.LAM_REL * w.max()))
    return np.einsum("ij,ij->i", Z, Z)


def main():
    out = {}
    ctx = P.ctx_for("che")
    th = torch.tensor(np.load(os.path.join(TESTS, "P4.1_cheap_su2_r22", "results", "theta_che.npy")))
    E, pr, Y = RS.jac(ctx, th)
    S = Y.T @ Y
    nrm = S[0, 0]
    Hc0 = Y.T @ ((ctx.Hr @ pr) * ctx.sw)
    STx, s = RS.centred(S, None, E)
    f = Hc0[1:] / nrm - E * s
    gx, _ = RS.step_gain(ctx, th, E, STx, f)
    w, U = np.linalg.eigh(S)
    Z = Y @ (U / np.sqrt(w + RS.LAM_REL * w.max()))
    lev_t = np.einsum("ij,ij->i", Z, Z)
    del Z, S
    born = Y[:, 0] ** 2 / np.sum(Y[:, 0] ** 2)
    out["gain_exact"] = gx
    P.log(f"target: exact gain {gx:.3e} [{P.cpu_min():.2f} CPU-min]")
    top = lambda v: set(np.argsort(v)[-len(v) // 10:])  # noqa: E731
    stale = dict(near=torch.tensor(np.load(os.path.join(TESTS, "P4.1_cheap_su2_r25", "results",
                                                        "theta_che_lbfgs.npy"))), far=ctx.th0.clone())
    m = 8 * (Y.shape[1] - 1)
    for name, ths in stale.items():
        lev = lev_at(ctx, ths)
        q = 0.5 * born + 0.5 * lev / lev.sum()
        phis = []
        for seed in ((1, 2) if name == "near" else (1,)):  # far is reported only: 1 draw (RULE_stale)
            r = np.random.default_rng(seed).choice(len(q), size=m, p=q)
            Ys = Y[r] / np.sqrt(m * q[r])[:, None]
            St, _ = RS.centred(Ys.T @ Ys, None, E)
            phis.append(RS.step_gain(ctx, th, E, St, f)[0] / gx)
        ov = len(top(lev) & top(lev_t)) / len(top(lev_t))
        out[name] = dict(phi=phis, top10_overlap=ov)
        P.log(f"{name}: phi {np.round(phis, 3)}; top-10% leverage overlap with target {ov:.3f} [{P.cpu_min():.2f} CPU-min]")
    pn = float(np.mean(out["near"]["phi"]))
    out["verdict"] = "PASS" if pn >= 0.8 else "KILLED" if pn < 0.5 else "MARGINAL"
    out["cpu_min"] = P.cpu_min()
    P.log(f"R51.3 T1b: phi(near) {pn:.3f}, phi(far) {np.mean(out['far']['phi']):.3f} -> {out['verdict']}")
    json.dump(out, open(os.path.join(HERE, "results", "stale.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
