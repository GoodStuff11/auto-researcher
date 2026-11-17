#!/usr/bin/env python3
"""
P4.1 r20: can a sampled truncated-metric linear method realise the exact LM gain at 12 sites?  Rule: RULE.md.
=============================================================================================================

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python lm_viability.py --mode MODE [options]

Reuses (read only) tests/P4.1_cheap_su2_r19/polish.py (ctx_for, build, exact_SH, step_candidate, sampled) and,
through it, tests/P4.1_cheap_su2_r11/realise.py.

Modes
  tu        Toulouse-Umrigar stochastic LM on `che`: non-symmetric H estimator, a_diag x I (raw-parameter space)
            added to the tangent H, a_diag chosen per step by correlated-sampling dE on an independent validation
            set (N_v = N, seed 1000 + step); accepted only if dE < -2 sigma.  Same cutoff 1e-5 and |z| <= 0.05
            cap as r19.  Writes results/tu_N<nsamp>.json (+ theta_tu_N<nsamp>.npy); resumes from them.
  naive     r19's sampled protocol unchanged (symmetrised H, in-sample line search) at --nsamp; calls
            polish.sampled with results in this directory (results/sampled_che_N<nsamp>.json).
  verdict   apply RULE.md to results/*.json; print and write results/verdict.json.
Options
  --nsamp INT (8192)   --steps INT (4)   --threads INT (1)   --cpu-cap FLOAT (CPU-min, 60)   --res DIR
"""
import argparse
import json
import os
import sys
import time

THR = "1"
for i, s_ in enumerate(sys.argv):
    if s_ == "--threads" and i + 1 < len(sys.argv):
        THR = sys.argv[i + 1]
for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = THR

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r19"))
import polish as P  # noqa: E402  (read only; sets thread env vars from --threads)
import torch  # noqa: E402

log = P.log
E0 = P.E0
THRCUT, TAU = P.THRCUT, P.TAU
ADIAG = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)   # x w_max(S_T); 1e-1, 1 added by amendment 1
R_EXACT4 = 0.5391510169766579                  # r19 results/seq_che.json, step 4
F_VIABLE, F_DEAD = 0.5, 0.2
RES = os.path.join(HERE, "results")


def draw(ctx, pr, N, seed):
    p = ctx.size * pr * pr
    p = p / p.sum()
    r = np.random.default_rng(seed).choice(len(p), size=N, p=p)
    return r, 1.0 / np.sqrt(N * p[r])


def sampled_SH_ns(Y, HY, r, wt):
    """non-symmetric estimator (strong zero-variance principle), normalised by the sampled S_00."""
    Ys, HYs = Y[r] * wt[:, None], HY[r] * wt[:, None]
    S = Ys.T @ Ys
    S = 0.5 * (S + S.T)
    Hm = Ys.T @ HYs
    return S / S[0, 0], Hm / S[0, 0]


class TU:
    """tangent model with a possibly non-symmetric H, truncated metric, and a_diag x I in raw-parameter space."""

    def __init__(self, S, Hm, E, P_):
        c = np.arange(1, P_ + 1)
        s, bR, bL = S[0, c], Hm[0, c], Hm[c, 0]
        ST = S[np.ix_(c, c)] - np.outer(s, s)
        HT = Hm[np.ix_(c, c)] - np.outer(s, bR) - np.outer(bL, s) + E * np.outer(s, s)
        w, U = np.linalg.eigh(0.5 * (ST + ST.T))
        k = w > THRCUT * w.max()
        self.Pm = U[:, k] / np.sqrt(w[k])
        self.wmax, self.rank, self.E = float(w.max()), int(k.sum()), E
        self.Ht0 = self.Pm.T @ HT @ self.Pm
        self.D = self.Pm.T @ self.Pm
        self.gr, self.gl = self.Pm.T @ (bR - E * s), self.Pm.T @ (bL - E * s)

    def candidate(self, a_rel):
        """lowest root of [[E, gr^T], [gl, Ht + a I_raw + lam]] by the secular equation; lam >= 0 until |z| <= TAU."""
        Ht = self.Ht0 + a_rel * self.wmax * self.D
        Lam, V = np.linalg.eig(Ht)
        qr, ql = self.gr @ V, np.linalg.solve(V, self.gl)
        E = self.E

        def z_of(lam):
            mu = Lam + lam
            f = lambda eps: (E - eps - np.sum(qr * ql / (mu - eps))).real  # noqa: E731
            hi = mu.real.min() - 1e-12 * max(1.0, abs(mu.real.min()))
            lo = min(E, mu.real.min()) - 1.0 - np.linalg.norm(self.gr) - np.linalg.norm(self.gl)
            while f(lo) < 0:
                lo -= 10 * (1 + abs(lo))
            if f(hi) > 0:
                return None
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            eps = 0.5 * (lo + hi)
            return (V @ (-ql / (mu - eps))).real

        lam = 0.0
        z = z_of(0.0)
        if z is None:
            return None
        if np.linalg.norm(z) > TAU:
            lo, hi = 0.0, 1.0
            while (zz := z_of(hi)) is None or np.linalg.norm(zz) > TAU:
                hi *= 4.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                zz = z_of(mid)
                if zz is not None and np.linalg.norm(zz) <= TAU:
                    hi = mid
                else:
                    lo = mid
            lam = hi
            z = z_of(lam)
        return self.Pm @ z, float(np.linalg.norm(z)), lam, float(np.max(np.abs(Lam.imag)))


def ztrue(Y, d):
    v = Y[:, 1:] @ d
    return float(np.sqrt(max(v @ v - (Y[:, 0] @ v) ** 2, 0.0)))


def tu(a):
    t0 = time.time()
    ctx = P.ctx_for("che")
    fn, fth = os.path.join(RES, f"tu_N{a.nsamp}.json"), os.path.join(RES, f"theta_tu_N{a.nsamp}.npy")
    th = ctx.th0.clone()
    E00 = ctx.energy(th)[0]
    gap = E00 - E0
    valid = abs(E00 - ctx.stored) < 1e-8
    Pn, hist, chk, per = ctx.m.nparam, [], {}, None
    if os.path.exists(fn):
        d0 = json.load(open(fn))
        hist, chk, valid = d0["steps"], d0["check"], d0["valid"]
        th = torch.tensor(np.load(fth))
    while len(hist) < a.steps:
        c0 = P.cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({c0:.1f} + {per:.1f} > {a.cpu_cap}); TRUNCATED")
            break
        n = len(hist) + 1
        E, Y, HY, pr = P.build(ctx, th)
        if n == 1 and a.nsamp == 8192 and "solver" not in chk:
            S, Hm = P.exact_SH(Y, HY)
            d_ref = P.step_candidate(S, Hm, E, Pn)[0]
            d_new = TU(S, Hm, E, Pn).candidate(0.0)[0]
            chk["solver"] = float(np.linalg.norm(d_new - d_ref) / np.linalg.norm(d_ref))
            valid = valid and chk["solver"] < 1e-6
            log(f"INVALID check: non-symmetric solver vs r19 exact-LM step-1 dtheta: rel {chk['solver']:.1e}")
            del S, Hm
        r, wt = draw(ctx, pr, a.nsamp, n)
        Ss, Hs = sampled_SH_ns(Y, HY, r, wt)
        del HY
        el_tr = (ctx.Hr @ pr)[r] / pr[r]
        E_tr, se_tr = float(el_tr.mean()), float(el_tr.std() / np.sqrt(a.nsamp))
        if n == 1:
            chk.update(E_samp=E_tr, se=se_tr, z=abs(E_tr - E) / se_tr)
            valid = valid and chk["z"] <= 6.0
            log(f"INVALID check: sampled E {E_tr:.6f} +- {se_tr:.1e} vs exact {E:.6f} ({chk['z']:.1f} se)")
        rv, _ = draw(ctx, pr, a.nsamp, 1000 + n)
        el_v = (ctx.Hr @ pr)[rv] / pr[rv]
        model = TU(Ss, Hs, float(Hs[0, 0]), Pn)
        cands = []
        for ar in ADIAG:
            out = model.candidate(ar)
            if out is None:
                cands.append(dict(a_rel=ar, ok=False))
                continue
            d, zn, lam, imag = out
            Ex, prn, _ = ctx.energy(th + torch.tensor(d))
            w = (prn[rv] / pr[rv]) ** 2
            eln = (ctx.Hr @ prn)[rv] / prn[rv]
            En = float(np.sum(w * eln) / np.sum(w))
            x = (w / w.mean()) * (eln - En) - (el_v - el_v.mean())
            cands.append(dict(a_rel=ar, ok=True, d=d, zn=zn, lam=lam, imag=imag, dth=float(np.linalg.norm(d)),
                              ztrue=ztrue(Y, d), dE_val=En - float(el_v.mean()),
                              sig_val=float(x.std() / np.sqrt(len(x))), E_exact=Ex, R_oracle=(E00 - Ex) / gap))
        del Y
        ok = [c for c in cands if c["ok"]]
        best = min(ok, key=lambda c: c["dE_val"]) if ok else None
        orac = min(ok, key=lambda c: c["E_exact"]) if ok else None
        accept = best is not None and best["dE_val"] < -2.0 * best["sig_val"]
        rec = dict(E=E, E_samp=E_tr, se=se_tr, rank=model.rank, accepted=bool(accept),
                   a_chosen=None if best is None else best["a_rel"],
                   a_oracle=None if orac is None else orac["a_rel"],
                   cands=[{k: v for k, v in c.items() if k != "d"} for c in cands])
        if accept:
            th = th + torch.tensor(best["d"])
            rec.update(E_new=best["E_exact"])
        else:
            rec.update(E_new=E)
        rec.update(R=(E00 - rec["E_new"]) / gap, eps=(rec["E_new"] - E0) / abs(E0))
        hist.append(rec)
        per = P.cpu_min() - c0
        cs = "; ".join(f"a {c['a_rel']:.0e}: val dE {c['dE_val']:+.2e}+-{c['sig_val']:.1e} exact R "
                       f"{c['R_oracle']:+.3f} |z| {c['zn']:.3f} |z|true {c['ztrue']:.3f}" if c["ok"] else
                       f"a {c['a_rel']:.0e}: no root" for c in cands)
        log(f"step {n}: {'ACCEPT a ' + format(best['a_rel'], '.0e') if accept else 'REJECT'}; exact R "
            f"{rec['R']:.4f}; oracle a {rec['a_oracle']}; rank {model.rank} [{P.cpu_min():.1f} CPU-min, step "
            f"{per:.1f}, wall {time.time() - t0:.0f} s]\n    {cs}")
        np.save(fth, th.numpy())
        P.dump(fn, dict(arm=f"tu_N{a.nsamp}", nsamp=a.nsamp, E_theta0=E00, gap=gap, valid=valid, check=chk,
                        steps=hist, cpu_min=P.cpu_min(), wall=time.time() - t0))
    P.dump(fn, dict(arm=f"tu_N{a.nsamp}", nsamp=a.nsamp, E_theta0=E00, gap=gap, valid=valid, check=chk,
                    steps=hist, cpu_min=P.cpu_min(), wall=time.time() - t0, done=len(hist) >= a.steps))
    log(f"wrote {os.path.basename(fn)} ({time.time() - t0:.0f} s, {P.cpu_min():.1f} CPU-min)")


def naive(a):
    P.RES = RES
    P.sampled(argparse.Namespace(arm="che", nsamp=a.nsamp, steps=a.steps, cpu_cap=a.cpu_cap))


def summarise(d, nsteps):
    acc = [s for s in d["steps"][:nsteps] if s.get("accepted")]
    E = acc[-1]["E_new"] if acc else d["E_theta0"]
    rise = any(s["E_new"] > s["E"] + 1e-10 for s in acc)
    done = len(d["steps"]) >= nsteps or any(not s.get("accepted") for s in d["steps"]) and "trials" in d["steps"][0]
    Rr = (d["E_theta0"] - E) / d["gap"]
    return dict(R=Rr, f=Rr / R_EXACT4, rise=rise, accepted=len(acc), done=bool(done), valid=bool(d.get("valid")))


def verdict(a):
    arms = {"tu8k": "tu_N8192.json", "tu64k": "tu_N65536.json", "naive64k": "sampled_che_N65536.json"}
    out, S = {}, {}
    for k, f in arms.items():
        p = os.path.join(RES, f)
        if os.path.exists(p):
            S[k] = summarise(json.load(open(p)), 4)
            print(f"{k}: {S[k]}")
        else:
            print(f"{k}: missing")
    valid = all(s["valid"] for s in S.values())
    if len(S) < 3 or not all(s["done"] for s in S.values()):
        v = "INCOMPLETE"
    elif not valid:
        v = "INVALID"
    elif any(s["f"] >= F_VIABLE and not s["rise"] for s in S.values()):
        v = "VIABLE"
    elif all(s["rise"] or s["f"] < F_DEAD for s in S.values()):
        v = "NOT-VIABLE"
    else:
        v = "MARGINAL"
    out.update(arms=S, valid=valid, verdict=v, R_exact4=R_EXACT4)
    print(f"VALID {valid}; VERDICT: {v}")
    P.dump(os.path.join(RES, "verdict.json"), out)


def main():
    global RES
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["tu", "naive", "verdict"])
    pa.add_argument("--nsamp", type=int, default=8192)
    pa.add_argument("--steps", type=int, default=4)
    pa.add_argument("--threads", default="1")
    pa.add_argument("--cpu-cap", type=float, default=60.0)
    pa.add_argument("--res", default=os.path.join(HERE, "results"))
    a = pa.parse_args()
    RES = a.res
    os.makedirs(RES, exist_ok=True)
    {"tu": tu, "naive": naive, "verdict": verdict}[a.mode](a)


if __name__ == "__main__":
    main()
