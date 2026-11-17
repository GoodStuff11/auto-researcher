"""Round 22 sub-question B (RULE_B.md): one-step estimator comparison at R21's che theta0, exact energies."""
import json
import os
import sys
import time

THR = sys.argv[1] if len(sys.argv) > 1 else "1"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = THR
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402

torch.set_num_threads(int(THR))
log = P.log
SH, N, SEEDS, TAUS = 1e-3, 16384, (3001, 3002, 3003), (0.01, 0.02, 0.03)
ctx = P.ctx_for("che")
th0 = ctx.th0.clone()
E00, Y, HY, pr = P.build(ctx, th0)
gap = E00 - P.E0
log(f"build done, E {E00:.8f}, gap {gap:.4e}, {P.cpu_min():.1f} CPU-min")
Pn = Y.shape[1] - 1
c = slice(1, None)
S = Y.T @ Y
s = S[0, c]
STx = S[c, c] - np.outer(s, s)
STx = 0.5 * (STx + STx.T)
Hx = Y[:, 0] @ HY
fx = 0.5 * (Y[:, 1:].T @ HY[:, 0] + Hx[1:]) - E00 * s
p = ctx.size * pr * pr
p /= p.sum()
pf = ctx.size * np.abs(pr)
pf /= pf.sum()
nnz_row = np.diff(ctx.Hr.indptr)
sizes = [120, 77, 384, 16] + [120] * 16 + [120]   # P0, jv, W1, b1, W2 per hidden unit, b2 (che shapes order)
starts = np.cumsum([0, 120, 77, 384, 16, 1920])
blocks = [np.arange(0, 120), np.arange(120, 197), np.arange(197, 581), np.arange(581, 597)]
blocks += [np.arange(597 + 120 * h, 597 + 120 * (h + 1)) for h in range(16)] + [np.arange(2517, 2637)]
assert sum(len(b) for b in blocks) == Pn == 2637


def stats(r, q):
    wt = 1.0 / np.sqrt(len(r) * q[r])
    Ys, HYs = Y[r] * wt[:, None], HY[r] * wt[:, None]
    s00 = Ys[:, 0] @ Ys[:, 0]
    Ss = Ys.T @ Ys / s00
    H0c = Ys[:, 1:].T @ HYs[:, 0] / s00          # <O E_loc>
    Hc0 = HYs[:, 1:].T @ Ys[:, 0] / s00          # <(H O psi)/psi>
    E = (Ys[:, 0] @ HYs[:, 0]) / s00
    ss = Ss[0, c]
    ST = Ss[c, c] - np.outer(ss, ss)
    return 0.5 * (ST + ST.T), H0c - E * ss, Hc0 - E * ss, Ys


def solve(Sm, f):
    return -np.linalg.solve(Sm + SH * np.eye(Pn), f)


def score(d):
    ml = np.sqrt(max(d @ STx @ d, 1e-300))
    out = {}
    for tau in TAUS:
        Ex = ctx.energy(th0 + torch.tensor(tau * d / ml))[0]
        out[tau] = (E00 - Ex) / gap
    return out


dx = solve(STx, fx)
cos_ref = dx / np.sqrt(dx @ STx @ dx)


def cosx(d):
    return float((d @ STx @ cos_ref) / np.sqrt(d @ STx @ d))


res = {"exact": score(dx)}
log(f"exact shift direction: {res['exact']}")
rng0 = np.random.default_rng
for sd in SEEDS:
    rng = rng0(sd)
    r = rng.choice(len(p), size=N, p=p)
    ST, f, fa, Ys = stats(r, p)
    D = {}
    D["V1_base"] = solve(ST, f)
    w, U = np.linalg.eigh(ST)
    k = w > 1e-4 * w.max()
    D["V1t_trunc"] = -(U[:, k] / w[k]) @ (U[:, k].T @ f)
    D["V2_exactS"] = solve(STx, f)
    D["V3_exactf"] = solve(ST, fx)
    r4 = rng.choice(len(p), size=3 * N, p=p)
    ST4, f4, _, _ = stats(np.concatenate([r, r4]), p)
    D["V4_S4N_fN"] = solve(ST4, f)
    D["V5_plain4N"] = solve(ST4, f4)
    O = Ys[:, 1:] / Ys[:, :1]
    x = O - O.mean(0)
    a2 = (x * x).sum(1)
    sum_all = np.mean(a2 * a2) - np.sum(ST * ST)
    sum_diag = np.sum(np.mean(x ** 4, 0) - np.diag(ST) ** 2)
    off2 = np.sum(ST * ST) - np.sum(np.diag(ST) ** 2)
    alpha = float(np.clip((sum_all - sum_diag) / N / off2, 0, 1))
    D["V6_LW"] = solve((1 - alpha) * ST + alpha * np.diag(np.diag(ST)), f)
    h = N // 2
    STa, _, _, _ = stats(r[:h], p)
    STb, _, _, _ = stats(r[h:], p)
    Sn = 0
    for A, B in ((STa, STb), (STb, STa)):
        _, Ua = np.linalg.eigh(A)
        lam = np.einsum("ij,ij->j", Ua, B @ Ua)
        Sn = Sn + 0.5 * (Ua * lam) @ Ua.T
    D["V7_NERCOME"] = solve(Sn, f)
    lr = (U[:, -256:] * w[-256:]) @ U[:, -256:].T
    D["V8_LRD256"] = solve(lr + np.diag(np.clip(np.diag(ST - lr), 0, None)), f)
    Sbd = np.zeros_like(ST)
    for b in blocks:
        Sbd[np.ix_(b, b)] = ST[np.ix_(b, b)]
    D["V9a_blockdiag"] = solve(Sbd, f)
    Dm = np.zeros((Pn, len(blocks)))
    for j, b in enumerate(blocks):
        Dm[b, j] = np.linalg.solve(ST[np.ix_(b, b)] + SH * np.eye(len(b)), f[b])
    A = Dm.T @ ST @ Dm
    cc = np.linalg.solve(A + 1e-3 * np.trace(A) / len(blocks) * np.eye(len(blocks)), Dm.T @ f)
    D["V9b_blocked2stage"] = -Dm @ cc
    D["V10_cv0.5"] = solve(ST, 0.5 * (f + fa))
    D["V10_cv1"] = solve(ST, fa)
    qg = 0.8 * p + 0.2 * pf
    rg = rng.choice(len(p), size=N, p=qg)
    STg, fg_, _, _ = stats(rg, qg)
    D["V11_guided"] = solve(STg, fg_)
    D["V2t_exactS_trunc"] = None
    for name, d in D.items():
        if d is None:
            continue
        sc = score(d)
        res.setdefault(name, []).append(dict(seed=sd, R=sc, Rbest=max(sc.values()), cos=cosx(d)))
        log(f"seed {sd} {name}: R {['%+.4f' % v for v in sc.values()]} cos {cosx(d):+.3f}"
            + (f" alpha {alpha:.3f}" if name == "V6_LW" else "") + f" [{P.cpu_min():.1f} CPU-min]")

# force variance reduction (V10), 16 seeds, no energies
Ainv = np.linalg.inv(STx + SH * np.eye(Pn))
err = {"plain": [], "cv0.5": [], "cv1": []}
Kc = []
for sd in range(4001, 4017):
    r = rng0(sd).choice(len(p), size=N, p=p)
    _, f, fa, _ = stats(r, p)
    Kc.append(nnz_row[r].mean())
    for nm, ff in (("plain", f), ("cv0.5", 0.5 * (f + fa)), ("cv1", fa)):
        e = ff - fx
        err[nm].append(e @ Ainv @ e)
VR = {k: float(np.mean(err["plain"]) / np.mean(v)) for k, v in err.items()}
res["V10_VR"] = VR
res["K_conn"] = float(np.mean(Kc))
log(f"force VR {VR}; K {np.mean(Kc):.1f}")
base = np.mean([x["Rbest"] for x in res["V1_base"]])
summ = {}
for nm, v in res.items():
    if isinstance(v, list):
        rb = float(np.mean([x["Rbest"] for x in v]))
        verdict = "PASS-T1" if rb >= base + 0.05 else ("KILL" if rb <= base + 0.02 else "AMBIGUOUS")
        summ[nm] = dict(Rbar=rb, sd=float(np.std([x["Rbest"] for x in v])),
                        cos=float(np.mean([x["cos"] for x in v])), verdict=verdict)
        log(f"{nm:20s} Rbar {rb:+.4f} (sd {summ[nm]['sd']:.4f}) cos {summ[nm]['cos']:+.3f} -> {verdict}")
res["summary"] = summ
res["cpu_min"] = P.cpu_min()
json.dump(res, open(os.path.join(HERE, "results/noise_split.json"), "w"), indent=1, default=float)
log(f"done {P.cpu_min():.1f} CPU-min")
