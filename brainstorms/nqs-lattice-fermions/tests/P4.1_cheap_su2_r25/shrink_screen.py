"""Round 24, R39 `js-force` (RULE_E.md): one-step force-shrinkage screen at R21's che theta0, exact energies.
Same protocol and draw order as noise_split.py (RULE_B.md). Run with OMP/MKL/OPENBLAS threads = 1 under ulimit -t 600."""
import json
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402

torch.set_num_threads(1)
log = P.log
SH, N, SEEDS, TAUS, NG = 1e-3, 16384, (3001, 3002, 3003), (0.01, 0.02, 0.03), 32
OUT = os.path.join(HERE, "results/shrink_screen.json")
ctx = P.ctx_for("che")
th0 = ctx.th0.clone()
E00, Y, HY, pr = P.build(ctx, th0)
gap = E00 - P.E0
log(f"build done, E {E00:.8f}, gap {gap:.4e}, {P.cpu_min():.1f} CPU-min")
Pn = Y.shape[1] - 1
p = ctx.size * pr * pr
p /= p.sum()
pf = ctx.size * np.abs(pr)
pf /= pf.sum()


def stats(r, q):
    """weighted per-sample O, E_loc -> S_T, f, and the per-sample force contributions with their weights."""
    wt = 1.0 / np.sqrt(len(r) * q[r])
    Ys, HYs = Y[r] * wt[:, None], HY[r] * wt[:, None]
    om = Ys[:, 0] ** 2
    om = om / om.sum()
    O = Ys[:, 1:] / Ys[:, :1]
    e = HYs[:, 0] / Ys[:, 0]
    Oc = O - om @ O
    ec = e - om @ e
    ST = (Oc * om[:, None]).T @ Oc
    f = Oc.T @ (om * ec)
    return 0.5 * (ST + ST.T), f, Oc * ec[:, None], om


def shrink_dirs(ST, f, Z, om):
    w, U = np.linalg.eigh(ST)
    ft = U.T @ f
    A = (Z - f) @ U                                   # per-sample force contributions in the eigenbasis
    var = (om[:, None] ** 2 * (A - 0.0) ** 2).sum(0)  # Var(ft_k), (z_i - f) already centred
    del A
    den = w + SH
    D = {"V1_base": -(U @ (ft / den))}
    s1 = np.clip(1.0 - var / np.maximum(ft ** 2, 1e-300), 0.0, 1.0)
    D["J1_modeJS"] = -(U @ (s1 * ft / den))
    g2, v2 = ft ** 2 / den, var / den
    s2 = np.ones_like(ft)
    for b in np.array_split(np.arange(Pn), NG):       # eigh sorts w ascending: contiguous eigenvalue groups
        nb = len(b)
        s2[b] = max(0.0, 1.0 - (nb - 2) / nb * v2[b].sum() / g2[b].sum())
    D["J2_groupJS"] = -(U @ (s2 * ft / den))
    info = dict(frac_kept_J1=float((s1 > 0).mean()), mean_s1=float(s1.mean()), s2_groups=[float(x) for x in s2[[b[0] for b in np.array_split(np.arange(Pn), NG)]]],
                snr_med=float(np.median(ft ** 2 / var)))
    return D, info


def score(d):
    ml = np.sqrt(max(d @ STx @ d, 1e-300))
    return {tau: float((E00 - ctx.energy(th0 + torch.tensor(tau * d / ml))[0]) / gap) for tau in TAUS}


c = slice(1, None)
S = Y.T @ Y
s = S[0, c]
STx = S[c, c] - np.outer(s, s)
STx = 0.5 * (STx + STx.T)
res = {"rule": "RULE_E.md", "runs": {}}
guided_draws = {}
for sd in SEEDS:
    rng = np.random.default_rng(sd)
    r = rng.choice(len(p), size=N, p=p)
    rng.choice(len(p), size=3 * N, p=p)              # noise_split's V4/V5 draw, consumed to keep the stream
    qg = 0.8 * p + 0.2 * pf
    guided_draws[sd] = (rng.choice(len(p), size=N, p=qg), qg)
    ST, f, Z, om = stats(r, p)
    D, info = shrink_dirs(ST, f, Z, om)
    del Z
    for nm, d in D.items():
        sc = score(d)
        res["runs"].setdefault(nm, []).append(dict(seed=sd, R=sc, Rbest=max(sc.values())))
        log(f"seed {sd} {nm}: R {['%+.4f' % v for v in sc.values()]} [{P.cpu_min():.1f} CPU-min]")
    res["runs"].setdefault("info", []).append(info)
    log(f"seed {sd} info {info}")
    json.dump(res, open(OUT, "w"), indent=1)

if P.cpu_min() < 6.5:                                 # reported-only: shrinkage on top of guided draws
    for sd in SEEDS:
        if P.cpu_min() > 8.0:
            log("stopping guided extras (CPU)")
            break
        rg, qg = guided_draws[sd]
        ST, f, Z, om = stats(rg, qg)
        D, info = shrink_dirs(ST, f, Z, om)
        del Z
        for nm, d in D.items():
            sc = score(d)
            res["runs"].setdefault("G_" + nm, []).append(dict(seed=sd, R=sc, Rbest=max(sc.values())))
            log(f"seed {sd} guided+{nm}: R {['%+.4f' % v for v in sc.values()]} [{P.cpu_min():.1f} CPU-min]")
        json.dump(res, open(OUT, "w"), indent=1)

summ = {}
for nm, v in res["runs"].items():
    if nm == "info":
        continue
    rb = [x["Rbest"] for x in v]
    summ[nm] = dict(Rbar=float(np.mean(rb)), sd=float(np.std(rb)), n=len(rb))
base = summ["V1_base"]["Rbar"]
best = max(summ["J1_modeJS"]["Rbar"], summ["J2_groupJS"]["Rbar"])
verdict = "PASS-T1" if best >= base + 0.05 else ("KILL" if best <= base + 0.02 else "AMBIGUOUS")
for nm, v in summ.items():
    log(f"{nm:16s} Rbar {v['Rbar']:+.4f} (sd {v['sd']:.4f}, n {v['n']})")
log(f"RULE_E verdict: {verdict} (best shrink {best:+.4f} vs base {base:+.4f})")
res.update(summary=summ, verdict=verdict, cpu_min=P.cpu_min())
json.dump(res, open(OUT, "w"), indent=1)
log(f"done {P.cpu_min():.1f} CPU-min")
