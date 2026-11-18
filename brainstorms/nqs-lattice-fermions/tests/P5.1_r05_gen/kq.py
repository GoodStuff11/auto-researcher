#!/usr/bin/env python3
"""
P5.1 round 5, T1 for R51.11 `qopt` (kappa-optimal sampling distribution) and R51.13 `svrg` (anchor control variate).
Rules: RULE_qopt.md, RULE_svrg.md (written before this ran).  Run with the lit_env interpreter from this directory:
    OMP_NUM_THREADS=1 ... ~/.conda/envs/lit_env/bin/python kq.py --Lx LX --Ly LY --Ne NE [--start F] [--anchors 1,5]

At theta (the size's SR-trained start), exact sums over orbit representatives, lambda = 1e-3 (R51.1's shift):
  S, f, x = (S + lambda)^-1 f, residual r = e - O_c^T x, g = O_c r, M = A S A (A = (S + lambda)^-1),
  h(x) = (g - <g>)^T M (g - <g>)   (per-configuration contribution to the first-order step error in the S norm).
For any sampling distribution q (weights w = p/q) kappa_q = sum_x p w h / x^T S x (identical to kappa.py's C_q
formula); so one pass gives kappa for every q:
  born (q = p), guided (0.8 p + 0.2 p_flat), |psi|^{2 alpha} (alpha 0.5..0.9), mixtures, residual-only
  q ~ p (a + |e|) [needs E_loc per proposal], the oracle q* ~ p sqrt(h) [needs rows and E_loc per proposal], and the
  best amplitude-only q ~ p m(log p) (piecewise constant in 40 quantile bins of log p; exact optimum
  (sum_b sqrt(P_b H_b))^2), which costs amplitudes only, like guided.
SVRG (R51.13): anchor theta~ = theta - d(s), d(s) the exact step of metric length s = k tau (tau = 0.005), k in
--anchors.  Control variate v = (p~/p) M^{1/2}(g~ - <g~>_p~) (mean zero under p for an exact anchor), u = M^{1/2}(g -
<g>); under guided draws rho(k) = min_c E_q[w^2 |u - c v|^2] / E_q[w^2 |u|^2] = 1 - b^2/(a c).
Output: results_kq_<tag>.json.
"""
import argparse
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P5.1_size_scaling_r04"))
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

LAM, TAU = 1e-3, 0.005


def pass1(ctx, th, chunks, uq=None):
    E, pr, phi0 = ctx.energy(th)
    p = ctx.size * pr * pr
    p = p / p.sum()
    if uq is None:
        uq = np.nonzero(p > 0)[0]
    pu = p[uq] / p[uq].sum()
    el = (ctx.Hr @ pr)[uq] / pr[uq]
    P_ = ctx.m.nparam
    Ob, M2, OE = np.zeros(P_), np.zeros((P_, P_)), np.zeros(P_)
    for c in chunks:
        O = NS.T.orows(ctx, th, phi0, pr, uq[c])
        Ob += pu[c] @ O
        M2 += (O * pu[c][:, None]).T @ O
        OE += O.T @ (pu[c] * el[c])
    Eb = float(pu @ el)
    S = M2 - np.outer(Ob, Ob)
    S = 0.5 * (S + S.T)
    f = OE - Ob * Eb
    w_, U = np.linalg.eigh(S)
    x = U @ ((U.T @ f) / (w_ + LAM))
    return dict(E=E, pr=pr, phi0=phi0, uq=uq, pu=pu, el=el, Ob=Ob, Eb=Eb, S=S, f=f, w=w_, U=U, x=x,
                psi=np.abs(pr[uq]) * np.sqrt(ctx.size[uq]))


def gcent(ctx, th, st, c):
    Oc = NS.T.orows(ctx, th, st["phi0"], st["pr"], st["uq"][c]) - st["Ob"]
    r = (st["el"][c] - st["Eb"]) - Oc @ st["x"]
    return Oc * r[:, None]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--Lx", type=int, required=True)
    pa.add_argument("--Ly", type=int, required=True)
    pa.add_argument("--Ne", type=int, required=True)
    pa.add_argument("--start", default=None)
    pa.add_argument("--anchors", default="")
    pa.add_argument("--skip-q", action="store_true")
    pa.add_argument("--tag", default=None)
    pa.add_argument("--chunk", type=int, default=16384)
    a = pa.parse_args()
    if a.start is None:
        a.start = NS.START12
    c0 = NS.cpu_min()
    ctx = NS.GCtx(a.Lx, a.Ly, a.Ne, a.start)
    th = ctx.th0
    st = None
    st = pass1(ctx, th, [slice(0, 0)])  # sizes only
    n = len(st["uq"])
    chunks = [slice(i, min(i + a.chunk, n)) for i in range(0, n, a.chunk)]
    st = pass1(ctx, th, chunks)
    w_, U, pu = st["w"], st["U"], st["pu"]
    Mh = U * (np.sqrt(np.maximum(w_, 0)) / (w_ + LAM))  # M = Mh Mh^T
    xSx = float(st["x"] @ st["S"] @ st["x"])
    NS.log(f"pass 1 done, {n} reps ({NS.cpu_min() - c0:.2f} CPU-min)")
    pf = st["psi"] / st["psi"].sum()
    wg = pu / (0.8 * pu + 0.2 * pf)
    out = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, P=int(ctx.m.nparam), reps=int(n), start=a.start, E=st["E"], xSx=xSx)
    # pass 2 at theta: g mean, then h
    gb = np.zeros(ctx.m.nparam)
    if not a.skip_q or a.anchors:
        for c in chunks:
            gb += pu[c] @ gcent(ctx, th, st, c)
    if not a.skip_q:
        h = np.empty(n)
        for c in chunks:
            z = (gcent(ctx, th, st, c) - gb) @ Mh
            h[c] = np.einsum("ij,ij->i", z, z)
        e = np.abs(st["el"] - st["Eb"])
        kap = lambda q: float(np.sum(pu * (pu / (q / q.sum())) * h) / xSx)  # noqa: E731
        res = {"born": kap(pu), "guided": kap(0.8 * pu + 0.2 * pf)}
        for al in (0.5, 0.6, 0.7, 0.8, 0.9):
            res[f"psi2^{al}"] = kap(pu ** al)
        for b in (0.5, 0.35):
            res[f"mix{b}"] = kap((1 - b) * pu + b * pf)
        for aa in (0.1, 1.0):
            res[f"resid_a{aa}"] = kap(pu * (aa * float(pu @ e) + e))
        res["oracle"] = float(np.sum(pu * np.sqrt(h)) ** 2 / xSx)
        lp = np.log(pu)
        for nb in (10, 40):
            edges = np.quantile(lp, np.linspace(0, 1, nb + 1))
            b_ = np.clip(np.searchsorted(edges, lp, side="right") - 1, 0, nb - 1)
            Pb = np.bincount(b_, weights=pu, minlength=nb)
            Hb = np.bincount(b_, weights=pu * h, minlength=nb)
            res[f"amp_only_{nb}bins"] = float(np.sum(np.sqrt(Pb * Hb)) ** 2 / xSx)
        srt = np.sort(pu * h)[::-1]
        res["born_share_top0.1pct"] = float(srt[:max(1, n // 1000)].sum() / srt.sum())
        res["ratio_guided_over_best_amp"] = res["guided"] / min(res["amp_only_40bins"], res["amp_only_10bins"])
        res["ratio_guided_over_oracle"] = res["guided"] / res["oracle"]
        out["kappa"] = res
        NS.log(json.dumps(res))
        json.dump(out, open(os.path.join(HERE, f"results_kq_{a.tag or 'S%dx%d_N%d' % (a.Lx, a.Ly, a.Ne)}.json"), "w"),
                  indent=1)
    # SVRG anchors
    out["svrg"] = {}
    for k in [int(s) for s in a.anchors.split(",") if s]:
        d = NS.T.d_shift(w_, U, st["f"], k * TAU)[0]
        tht = th - torch.tensor(d)  # anchor k steps behind along the exact direction
        sa = pass1(ctx, tht, chunks)
        if not np.array_equal(sa["uq"], st["uq"]):  # a representative with exactly zero amplitude at one theta
            com = np.intersect1d(sa["uq"], st["uq"])
            NS.log(f"uq differ ({len(st['uq'])} vs {len(sa['uq'])}); recomputing both on {len(com)} common reps")
            st = pass1(ctx, th, chunks, com)
            sa = pass1(ctx, tht, chunks, com)
            pu = st["pu"]
            Mh = st["U"] * (np.sqrt(np.maximum(st["w"], 0)) / (st["w"] + LAM))
            pf = st["psi"] / st["psi"].sum()
            wg = pu / (0.8 * pu + 0.2 * pf)
            gb = np.zeros(ctx.m.nparam)
            for c in chunks:
                gb += pu[c] @ gcent(ctx, th, st, c)
        gbt = np.zeros(ctx.m.nparam)
        for c in chunks:
            gbt += sa["pu"][c] @ gcent(ctx, tht, sa, c)
        wt = sa["pu"] / pu
        A_, B_, C_ = 0.0, 0.0, 0.0
        for c in chunks:
            zu = (gcent(ctx, th, st, c) - gb) @ Mh
            zv = ((gcent(ctx, tht, sa, c) - gbt) * wt[c][:, None]) @ Mh
            ww = pu[c] * wg[c]
            A_ += float(ww @ np.einsum("ij,ij->i", zu, zu))
            B_ += float(ww @ np.einsum("ij,ij->i", zu, zv))
            C_ += float(ww @ np.einsum("ij,ij->i", zv, zv))
        rho = 1 - B_ * B_ / (A_ * C_)
        out["svrg"][k] = dict(rho=rho, c_opt=B_ / C_, gain_per_cpu=1 / (2 * rho), w_max=float(wt.max()),
                              chi2=float(pu @ (wt * wt) - 1))
        NS.log(f"anchor k={k}: {out['svrg'][k]} ({NS.cpu_min() - c0:.2f} CPU-min)")
    out["cpu_min"] = NS.cpu_min() - c0
    tag = a.tag or f"S{a.Lx}x{a.Ly}_N{a.Ne}"
    json.dump(out, open(os.path.join(HERE, f"results_kq_{tag}.json"), "w"), indent=1)
    NS.log(f"done {out['cpu_min']:.2f} CPU-min")


if __name__ == "__main__":
    main()
