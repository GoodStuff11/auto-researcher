#!/usr/bin/env python3
"""
lenratio_R51.18.py: per-leaf critic re-check (round 7, Major 1) for R51.18 (storm-sf rerun).
    ~/.conda/envs/lit_env/bin/python lenratio_R51.18.py [--steps 40]
No options other than --steps (default 40). Replays storm_v2.py --mode sf --nsamp 2048 --a 0.1 --tau 0.005 --seed0 9000
step for step (same draws, same arm; asserts the energies match results/storm_N2048_a0.1_sf_v2/run_diag.json) and at each
step logs r = v^T S_exact v / v^T S_n v, where S_exact is the exact metric (full sum over every orbit representative,
weights |psi|^2) at the current parameters and S_n the fresh 2048-draw sample metric that sets the step length.  Also
logs the same ratio for the fresh plain-SR direction xhat_n = (S_n + lam)^-1 f_n as a reference, the exact energy change
of the step, and the exact first-order predicted decrease lin_dec = -2 f_exact.d (f_exact = exact force; dE/dtheta = 2 f_exact,
checked by central finite difference at the start: 0.0207753 vs 2 x 0.0103876).  lin_dec > 0 means the step points downhill.  Writes results/storm_N2048_a0.1_sf_v2/lenratio.json.
Diagnostic only; changes nothing in the arm.
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import storm_v2 as SV  # noqa: E402

NS = SV.NS
import torch  # noqa: E402


def exact_parts(ctx, th, phi0, pr):
    p = ctx.size * pr * pr
    p = p / p.sum()
    uq = np.nonzero(p > 0)[0]
    w = p[uq] / p[uq].sum()
    O = NS.T.orows(ctx, th, phi0, pr, uq)
    el = (ctx.Hr @ pr)[uq] / pr[uq]
    Oc = O - w @ O
    f = (Oc * w[:, None]).T @ (el - w @ el)
    return Oc, w, f


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--steps", type=int, default=40)
    a = pa.parse_args()
    N, A, TAU, SEED0, LAM = 2048, 0.1, 0.005, 9000, SV.LAM
    ref = json.load(open(os.path.join(HERE, "results", "storm_N2048_a0.1_sf_v2", "run_diag.json")))["steps"]
    c0 = NS.cpu_min()
    ctx = NS.GCtx(5, 2, 8, SV.START10)
    th = ctx.th0.clone()
    E, pr, phi0 = ctx.energy(th)
    prev, out = None, []
    for n in range(1, a.steps + 1):
        p = ctx.size * pr * pr
        p = p / p.sum()
        pf = ctx.size * np.abs(pr)
        q = 0.8 * p + 0.2 * pf / pf.sum()
        rg = np.random.default_rng(SEED0 + n).choice(len(q), size=N, p=q)
        uq, inv = np.unique(rg, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uq)).astype(float)
        w = cnt * p[uq] / q[uq]
        w = w / w.sum()
        S, xh, fh = SV.solve(ctx, th, phi0, pr, uq, w)
        if prev is None:
            v = xh
            Sr, fr = S, fh
        else:
            thp, prp, phip = prev
            pp = ctx.size * prp * prp
            pp = pp / pp.sum()
            wp = cnt * pp[uq] / q[uq]
            wp = wp / wp.sum()
            Sp, xp, fp = SV.solve(ctx, thp, phip, prp, uq, wp)
            Sr = S + (1 - A) * (Sr - Sp)
            fr = fh + (1 - A) * (fr - fp)
            Sr = 0.5 * (Sr + Sr.T)
            v = np.linalg.solve(Sr + LAM * np.eye(len(fr)), fr)
        Oc, we, fe = exact_parts(ctx, th, phi0, pr)
        vSn = float(v @ S @ v)
        vSe = float(we @ (Oc @ v) ** 2)
        xSn = float(xh @ S @ xh)
        xSe = float(we @ (Oc @ xh) ** 2)
        d = -TAU * v / np.sqrt(max(vSn, 1e-300))
        prev = (th, pr, phi0)
        th = th + torch.tensor(d)
        En, pr, phi0 = ctx.energy(th)
        assert abs(En - ref[n - 1]["E"]) < 1e-10, (n, En, ref[n - 1]["E"])
        out.append(dict(step=n, n_unique=int(len(uq)), r_v=vSe / vSn, r_xhat=xSe / xSn,
                        exact_len=TAU * np.sqrt(vSe / vSn), dE=En - E, lin_dec=float(-fe @ d) * 2.0))
        E = En
        NS.log(f"step {n}: uniq {len(uq)} r_v {vSe / vSn:.3g} r_xhat {xSe / xSn:.3g} dE {out[-1]['dE']:.3e} "
               f"lin_dec {out[-1]['lin_dec']:.3e} [{NS.cpu_min() - c0:.2f} CPU-min]")
    rv = np.array([o["r_v"] for o in out])
    rx = np.array([o["r_xhat"] for o in out])
    summ = dict(r_v_median=float(np.median(rv)), r_v_min=float(rv.min()), r_v_max=float(rv.max()),
                r_v_median_2_40=float(np.median(rv[1:])), r_xhat_median=float(np.median(rx)),
                r_xhat_min=float(rx.min()), r_xhat_max=float(rx.max()),
                frac_downhill=float(np.mean([o['lin_dec'] > 0 for o in out])),
                frac_rose=float(np.mean([o['dE'] > 0 for o in out])),
                frac_rose_given_downhill=float(np.mean([o['dE'] > 0 for o in out if o['lin_dec'] > 0])),
                median_abs_dE=float(np.median([abs(o['dE']) for o in out])),
                median_curv=float(np.median([o['dE'] + o['lin_dec'] for o in out])), cpu_min=NS.cpu_min() - c0)
    json.dump(dict(summary=summ, steps=out), open(os.path.join(HERE, "results", "storm_N2048_a0.1_sf_v2", "lenratio.json"),
                                                  "w"), indent=1)
    NS.log("summary " + json.dumps(summ))


if __name__ == "__main__":
    main()
