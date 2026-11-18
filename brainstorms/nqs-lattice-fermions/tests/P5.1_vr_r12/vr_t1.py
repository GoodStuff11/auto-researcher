#!/usr/bin/env python3
"""
P5.1 round 12, T1 screen of force variance-reduction routes (rule: RULE.md, written first).
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -t 600
    ~/.conda/envs/lit_env/bin/python vr_t1.py --Lx 5 --Ly 2 --Ne 8 --state start|end [--flipcheck]
At one state, exact sums over orbit representatives: kappa (first-order SR step-error constant under guided draws)
for the baseline g = O_c r and for ZV (g + c dE_loc), Stein (g - C Z), Neyman strata over log p, and Metropolis
Rao-Blackwellisation K g. Output: results/vr_S<Lx>x<Ly>_<state>.json.
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SS = os.path.join(HERE, "..", "P5.1_size_scaling_r04")
sys.path.insert(0, SS)
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

LAM = 1e-3


def features(ctx, Lx, Ly):
    """orbit-symmetrised occupation features at every representative (columns may be linearly dependent)."""
    nU = ctx.nU.numpy()
    nD = ctx.nD.numpy()
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                bonds.append((i, (x + 1) * Ly + y))
            if y + 1 < Ly:
                bonds.append((i, x * Ly + y + 1))
    F = 0.0
    for g in range(4):
        ix = ctx.idx[g, ctx.reps]
        u, d = nU[ix], nD[ix]
        n, sz = u + d, u - d
        cols = [n, u * d]
        cols.append(np.stack([n[:, i] * n[:, j] for i, j in bonds], 1))
        cols.append(np.stack([sz[:, i] * sz[:, j] for i, j in bonds], 1))
        F = F + 0.25 * np.concatenate(cols, 1)
    return F


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--Lx", type=int, required=True)
    pa.add_argument("--Ly", type=int, required=True)
    pa.add_argument("--Ne", type=int, required=True)
    pa.add_argument("--state", choices=["start", "end"], required=True)
    pa.add_argument("--flipcheck", action="store_true")
    a = pa.parse_args()
    tag = f"S{a.Lx}x{a.Ly}"
    start = os.path.join(SS, "starts", f"{tag}_N{a.Ne}_U8_tp-0.25_pg11_che_seed0_theta.npy")
    c0 = NS.cpu_min()
    ctx = NS.GCtx(a.Lx, a.Ly, a.Ne, start)
    if a.state == "start":
        th = ctx.th0
    else:
        th = torch.tensor(np.load(os.path.join(SS, "results", f"{tag}_N{a.Ne}_exact_tau0.005_s9000", "theta.npy")))
    out = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, state=a.state)
    if a.flipcheck:  # R51.22: spin-flip invariance of |phi0| and O on random configurations
        rg = np.random.default_rng(1)
        s = rg.choice(ctx.sec.dim, 256, replace=False)
        l1, s1 = ctx.m.logpsi0(th, ctx.nU[s], ctx.nD[s])
        l2, s2 = ctx.m.logpsi0(th, ctx.nD[s], ctx.nU[s])
        o1 = NS.P.logder(ctx.m, th, ctx.nU[s], ctx.nD[s])
        o2 = NS.P.logder(ctx.m, th, ctx.nD[s], ctx.nU[s])
        out["flip_max_dlog"] = float((l1 - l2).abs().max())
        out["flip_sign_ratio_unique"] = sorted(set((s1 * s2).numpy().round(6).tolist()))
        out["flip_max_dO_rel"] = float(np.abs(o1 - o2).max() / np.abs(o1).max())
        NS.log(f"flipcheck {out['flip_max_dlog']:.2e} {out['flip_sign_ratio_unique']} {out['flip_max_dO_rel']:.2e}")
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
    ww = pu * pu / q  # sum_x ww h = first-order guided variance
    O = NS.T.orows(ctx, th, phi0, pr, uq)
    NS.log(f"rows {O.shape} ({NS.cpu_min() - c0:.2f} CPU-min)")
    Ob = pu @ O
    Eb = float(pu @ el)
    Oc = O - Ob
    S = (Oc * pu[:, None]).T @ Oc
    S = 0.5 * (S + S.T)
    f = Oc.T @ (pu * (el - Eb))
    w_, U = np.linalg.eigh(S)
    x = U @ ((U.T @ f) / (w_ + LAM))
    xSx = float(x @ S @ x)
    Mh = U * (np.sqrt(np.maximum(w_, 0)) / (w_ + LAM))
    r = (el - Eb) - Oc @ x
    G = Oc * r[:, None]
    G -= pu @ G
    del Oc
    zu = G @ Mh
    hu = np.einsum("ij,ij->i", zu, zu)
    kg = float(ww @ hu) / xSx
    out.update(E=E, P=int(ctx.m.nparam), reps=n, xSx=xSx, kappa_guided=kg, kappa_born=float(pu @ hu) / xSx,
               expH=float(np.exp(-np.sum(pu * np.log(pu)))))
    NS.log(f"kappa_guided {kg:.4g} ({NS.cpu_min() - c0:.2f} CPU-min)")
    # ---- Stein regression numerator first (needs G), then free G
    F = features(ctx, a.Lx, a.Ly)
    pv = pr[:, None] * F
    Z = (ctx.Hr @ pv)[uq] / pr[uq][:, None] - el[:, None] * F[uq]
    Zm = pu @ Z
    out["stein_mean_rel"] = float(np.abs(Zm).max() / np.sqrt((pu @ (Z * Z)).max()))
    Z -= Zm
    Zw = Z * ww[:, None]
    ZwG = Zw.T @ G
    del G
    # ---- ZV: D_k = (H d_k psi)/psi - E_loc O_k at reps (d_k psi = pr O_k, zero outside uq), column blocks
    zv = np.zeros_like(zu)
    mrel = 0.0
    for c0_ in range(0, O.shape[1], 256):
        cs = slice(c0_, min(c0_ + 256, O.shape[1]))
        dpsi = np.zeros((len(pr), cs.stop - cs.start))
        dpsi[uq] = pr[uq][:, None] * O[:, cs]
        D = (ctx.Hr @ dpsi)[uq] / pr[uq][:, None] - el[:, None] * O[:, cs]
        Dm = pu @ D
        mrel = max(mrel, float(np.abs(Dm).max() / np.sqrt((pu @ (D * D)).max())))
        zv += (D - Dm) @ Mh[cs]
    del dpsi, D, O
    out["zv_mean_rel"] = mrel
    A_ = float(ww @ hu)
    B_ = float(ww @ np.einsum("ij,ij->i", zu, zv))
    C_ = float(ww @ np.einsum("ij,ij->i", zv, zv))
    cst = -B_ / C_
    zvk = {f"c{c:g}": (A_ + 2 * c * B_ + c * c * C_) / A_ for c in (0.5, 1.0)}
    zvk["cstar"] = cst
    zvk["rho_cstar"] = (A_ + 2 * cst * B_ + cst * cst * C_) / A_
    # tail diagnostics: share of guided variance from the 1% smallest-p configurations
    lo = pu < np.quantile(pu, 0.01)
    zvk["tail1pct_share_base"] = float(ww[lo] @ hu[lo] / A_)
    hz = np.einsum("ij,ij->i", zu + zv, zu + zv)
    zvk["tail1pct_share_c1"] = float(ww[lo] @ hz[lo] / (ww @ hz))
    out["zv"] = zvk
    del zv, hz
    NS.log(f"ZV {zvk} ({NS.cpu_min() - c0:.2f} CPU-min)")
    # ---- Stein: Z_j = (H(psi v_j))/psi - E_loc v_j
    C = np.linalg.lstsq(Zw.T @ Z, ZwG, rcond=1e-10)[0]  # J x P, metric-independent optimum
    zs = zu - Z @ (C @ Mh)
    hs = np.einsum("ij,ij->i", zs, zs)
    out["stein"] = dict(J=int(F.shape[1]), rho=float(ww @ hs) / A_)
    NS.log(f"Stein {out['stein']} ({NS.cpu_min() - c0:.2f} CPU-min)")
    del zs, Z, Zw
    # ---- Strat (Neyman, 40 log p bins) vs R51.11 amp-only binned IS
    lp = np.log(pu)
    nb = 40
    edges = np.quantile(lp, np.linspace(0, 1, nb + 1))
    b_ = np.clip(np.searchsorted(edges, lp, side="right") - 1, 0, nb - 1)
    Pb = np.bincount(b_, weights=pu, minlength=nb)
    Hb = np.bincount(b_, weights=pu * hu, minlength=nb)
    k_amp = float(np.sum(np.sqrt(Pb * Hb)) ** 2 / xSx)
    Wb = np.zeros(nb)
    for b in range(nb):
        m = b_ == b
        if Pb[b] <= 0:
            continue
        mu = (pu[m] @ zu[m]) / Pb[b]
        dz = zu[m] - mu
        Wb[b] = float(pu[m] @ np.einsum("ij,ij->i", dz, dz))
    k_str = float(np.sum(np.sqrt(Pb * Wb)) ** 2 / xSx)
    out["strat"] = dict(kappa_amp40=k_amp, kappa_strat=k_str, rho=k_str / kg, rho_amp40=k_amp / kg,
                        strat_over_amp=k_str / k_amp)
    NS.log(f"strat {out['strat']}")
    # ---- RB: Metropolis kernel on the symmetrised hop graph of uq
    Hu = ctx.Hr[uq][:, uq].tocsr()
    Ad = ((abs(Hu) + abs(Hu).T) > 0).astype(float).tolil()
    Ad.setdiag(0)
    Ad = Ad.tocsr()
    Ad.eliminate_zeros()
    deg = np.asarray(Ad.sum(1)).ravel()
    Dmax = deg.max()
    Ac = Ad.tocoo()
    acc = np.minimum(1.0, pu[Ac.col] / pu[Ac.row]) / Dmax
    K = sp.csr_matrix((acc, (Ac.row, Ac.col)), shape=(n, n))
    stay = 1.0 - np.asarray(K.sum(1)).ravel()
    K = K + sp.diags(stay)
    rev = abs((pu[:, None] * K.toarray()) - (pu[:, None] * K.toarray()).T).max() if n < 20000 else -1.0
    zk = K @ zu  # K acts on rows; projection by Mh commutes
    mk = pu @ zk
    out_rb = dict(mean_shift_rel=float(np.abs(mk).max() / np.sqrt((pu @ (zu * zu)).max())), rev_err=float(rev))
    zk -= mk
    hk = np.einsum("ij,ij->i", zk, zk)
    out_rb.update(rho=float(ww @ hk) / A_, deg_mean_q=float(q @ deg), deg_max=float(Dmax),
                  cost=1.0 + float(q @ deg))
    out["rb"] = out_rb
    NS.log(f"RB {out_rb}")
    out["cpu_min"] = NS.cpu_min() - c0
    json.dump(out, open(os.path.join(HERE, "results", f"vr_{tag}_{a.state}.json"), "w"), indent=1)
    NS.log(f"done {out['cpu_min']:.2f} CPU-min")


if __name__ == "__main__":
    main()
