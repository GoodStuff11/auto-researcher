#!/usr/bin/env python3
"""
R51.19 per-component c_k check (critic round 15, Minor 9; rule RULE_R51.19_ck.md, written first).
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python vr_ck.py --Lx 5 --Ly 2 --Ne 8 --state start|end
Same states, baseline g = O_c r, D_k = d_k E_loc, guided weights ww and metric M = A S A as vr_t1.py.
f(c) = sum_x ww (g + c.D)^T M (g + c.D) = A + 2 b.c + c^T Q c, b_k = sum ww (gM)_k D_k, Q = M o (D^T W D).
Reports: scalar c* (reproduces T1), per-component regression c_k and rho_k = 1 - R_k^2, f(c_reg)/A, and the oracle
joint diagonal optimum rho_diag = min_c f(c)/A. Output: results/ck_S<Lx>x<Ly>_<state>.json.
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SS = os.path.join(HERE, "..", "P5.1_size_scaling_r04")
sys.path.insert(0, SS)
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

LAM = 1e-3
RB = 16384  # row block


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--Lx", type=int, required=True)
    pa.add_argument("--Ly", type=int, required=True)
    pa.add_argument("--Ne", type=int, required=True)
    pa.add_argument("--state", choices=["start", "end"], required=True)
    a = pa.parse_args()
    tag = f"S{a.Lx}x{a.Ly}"
    start = os.path.join(SS, "starts", f"{tag}_N{a.Ne}_U8_tp-0.25_pg11_che_seed0_theta.npy")
    c0 = NS.cpu_min()
    ctx = NS.GCtx(a.Lx, a.Ly, a.Ne, start)
    if a.state == "start":
        th = ctx.th0
    else:
        th = torch.tensor(np.load(os.path.join(SS, "results", f"{tag}_N{a.Ne}_exact_tau0.005_s9000", "theta.npy")))
    E, pr, phi0 = ctx.energy(th)
    p = ctx.size * pr * pr
    p = p / p.sum()
    uq = np.nonzero(p > 0)[0]
    n = len(uq)
    pu = p[uq] / p[uq].sum()
    el_all = (ctx.Hr @ pr) / np.where(pr == 0, 1, pr)
    el = el_all[uq]
    pf = ctx.size[uq] * np.abs(pr[uq])
    pf = pf / pf.sum()
    q = 0.8 * pu + 0.2 * pf
    ww = pu * pu / q
    Oc = NS.T.orows(ctx, th, phi0, pr, uq)
    P = Oc.shape[1]
    NS.log(f"rows {Oc.shape} ({NS.cpu_min() - c0:.2f} CPU-min)")
    Ob = pu @ Oc
    Eb = float(pu @ el)
    Oc -= Ob
    S = (Oc * pu[:, None]).T @ Oc
    S = 0.5 * (S + S.T)
    f = Oc.T @ (pu * (el - Eb))
    w_, U = np.linalg.eigh(S)
    x = U @ ((U.T @ f) / (w_ + LAM))
    xSx = float(x @ S @ x)
    Mh = U * (np.sqrt(np.maximum(w_, 0)) / (w_ + LAM))
    M = Mh @ Mh.T
    r = (el - Eb) - Oc @ x
    Gb = (pu * r) @ Oc  # p-mean of g
    GM = np.empty((n, P))
    A_ = 0.0
    sgg = np.zeros(P)
    for i in range(0, n, RB):
        s = slice(i, min(i + RB, n))
        G = Oc[s] * r[s, None] - Gb
        GM[s] = G @ M
        A_ += float(ww[s] @ np.einsum("ij,ij->i", GM[s], G))
        sgg += ww[s] @ (G * G)
    kg = A_ / xSx
    NS.log(f"kappa_guided {kg:.4g} ({NS.cpu_min() - c0:.2f} CPU-min)")
    # D_k = (H d_k psi)/psi - E_loc O_k, d_k psi = pr O_k (O = Oc + Ob), p-centred
    D = np.empty((n, P))
    sgd = np.zeros(P)
    b = np.zeros(P)
    mrel = 0.0
    for c_ in range(0, P, 256):
        cs = slice(c_, min(c_ + 256, P))
        Ocs = Oc[:, cs] + Ob[cs]
        dpsi = np.zeros((len(pr), cs.stop - cs.start))
        dpsi[uq] = pr[uq][:, None] * Ocs
        Dc = (ctx.Hr @ dpsi)[uq] / pr[uq][:, None] - el[:, None] * Ocs
        Dm = pu @ Dc
        mrel = max(mrel, float(np.abs(Dm).max() / np.sqrt((pu @ (Dc * Dc)).max())))
        Dc -= Dm
        D[:, cs] = Dc
        Gc = Oc[:, cs] * r[:, None] - Gb[cs]
        sgd[cs] = ww @ (Gc * Dc)
        b[cs] = ww @ (GM[:, cs] * Dc)
    del Oc, dpsi, Dc, Gc, Ocs, GM
    NS.log(f"D built, mean_rel {mrel:.2e} ({NS.cpu_min() - c0:.2f} CPU-min)")
    DWD = np.zeros((P, P))
    for i in range(0, n, RB):
        s = slice(i, min(i + RB, n))
        DWD += D[s].T @ (ww[s, None] * D[s])
    del D
    sdd = np.diag(DWD).copy()
    Q = M * DWD
    Q = 0.5 * (Q + Q.T)
    NS.log(f"Q built ({NS.cpu_min() - c0:.2f} CPU-min)")

    def rho(c):
        return float(1.0 + (2 * b @ c + c @ Q @ c) / A_)

    cst = -b.sum() / Q.sum()
    rho_scalar = rho(np.full(P, cst))
    live = (sdd > sdd.max() * 1e-14) & (sgg > sgg.max() * 1e-14)  # parameters with no signal are left out
    creg = np.where(live, -sgd / np.where(live, sdd, 1), 0.0)
    rk_all = np.where(live, 1.0 - sgd * sgd / np.where(live, sgg * sdd, 1), 1.0)
    rk = rk_all[live]
    qw, qU = np.linalg.eigh(Q)
    keep = qw > qw.max() * 1e-12
    cd = -qU[:, keep] @ ((qU[:, keep].T @ b) / qw[keep])
    rho_diag = rho(cd)
    mdiag = np.diag(M)
    out = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, state=a.state, E=E, P=P, reps=n, xSx=xSx, kappa_guided=kg,
               zv_mean_rel=mrel, cstar_scalar=float(cst), rho_cstar_scalar=rho_scalar,
               rho_k_mean=float(rk.mean()), rho_k_median=float(np.median(rk)), rho_k_min=float(rk.min()),
               n_live=int(live.sum()),
               rho_k_gvarweighted=float(sgg @ rk_all / sgg.sum()),
               rho_k_metricdiag_weighted=float((mdiag * sgg) @ rk_all / (mdiag * sgg).sum()),
               creg_median_abs=float(np.median(np.abs(creg))),
               rho_metric_creg=rho(creg), rho_diag=rho_diag, Q_rank=int(keep.sum()),
               cdiag_median_abs=float(np.median(np.abs(cd))), cdiag_max_abs=float(np.abs(cd).max()))
    out["cpu_min"] = NS.cpu_min() - c0
    NS.log(json.dumps(out))
    json.dump(out, open(os.path.join(HERE, "results", f"ck_{tag}_{a.state}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
