#!/usr/bin/env python3
"""
P4.1 R45 T2 (`sup-fit`): capacity of charge-only vs spin-reading backflow by supervised infidelity fitting, 12 sites.
Rule: RULE_T2.md (pre-registered, this folder). Card: leaves/P4.1/cards/R45.md.

System: open 4x3, N_e = 10 (5+5), U = 8, t' = -0.25, (+,+) via P_G (r11 orbit reduction), exact S = 0.
Arms (identical NNBF, H = 16, P = 2637; only the backflow input differs; as R40):
  che   V.Model('che', 16): reads doublon/holon occupations; exact P_S in closed form inside logpsi0.
  nnbf  V.Model('nnbf', 16): reads n_up, n_dn; exact P_S applied to the full 627,264-dim vector (block diagonal in c).
Both from V.Model(arm).init(seed) (common recipe), trained by L-BFGS-B (maxcor 30) on -log F,
F = <g|P phi>^2 / <P phi|P phi>, g = exact (+,+) S = 0 ground state; every evaluation logs (1-F, eps).
Training stops at --cpu-min (process_time) or --maxfun.  SIGUSR1: checkpoint, exit 4; rerun resumes from the best theta
(L-BFGS memory restarts).
Modes: train (--arm, --seed), check (gradient FD for both arms, prints per-evaluation cost), verdict.
"""
import argparse
import glob
import json
import os
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["train", "check", "verdict"])
ap.add_argument("--arm", choices=["che", "nnbf"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cpu-min", type=float, default=300.0, dest="cpu_min")
ap.add_argument("--maxfun", type=int, default=100000)
ap.add_argument("--threads", default="1")
ap.add_argument("--res", default=None)
A = ap.parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = A.threads  # L63
sys.argv = [sys.argv[0], "--threads", A.threads]

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r11"))
import realise as R  # noqa: E402  (read only)

V = R.V
torch.set_num_threads(int(A.threads))
E0 = -7.827899034271753
RES = A.res or os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
c0 = time.process_time()
NO_LOSS, LOSS = 1.1, 1.5


def cpu():
    return (time.process_time() - c0) / 60


class SpinProj:
    """exact S = 0 projector on the full S^z = 0 vector, block diagonal in the charge configuration (copy of r40)."""

    def __init__(self, sec):
        dim = sec.dim
        bu = sec.up.bits[sec.iu].astype(float)
        bd = sec.dn.bits[sec.id].astype(float)
        diag = (bu * (1 - bd)).sum(1) + (sec.Sz ** 2 - sec.Sz)
        M = sp.diags(diag).tocsr()
        for Eu, Ed in sec.spsm_pairs():
            M = M - sp.kron(Eu, Ed, format="csr")
        cid = sec.cidx
        order = np.argsort(cid, kind="stable")
        cnt = np.bincount(cid)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        pos = np.empty(dim, dtype=np.int64)
        pos[order] = np.arange(dim) - start[cid[order]]
        coo = M.tocoo()
        assert np.all(cid[coo.row] == cid[coo.col])
        self.groups = []
        for d in np.unique(cnt):
            cs = np.nonzero(cnt == d)[0]
            gi = np.full(len(cnt), -1)
            gi[cs] = np.arange(len(cs))
            idxs = order[start[cs][:, None] + np.arange(d)[None, :]]
            B = np.zeros((len(cs), d, d))
            sel = cnt[cid[coo.row]] == d
            np.add.at(B, (gi[cid[coo.row[sel]]], pos[coo.row[sel]], pos[coo.col[sel]]), coo.data[sel])
            w, U = np.linalg.eigh(B)
            keep = (np.abs(w) < 0.5).astype(float)
            self.groups.append((idxs, np.einsum("bik,bk,bjk->bij", U, keep, U)))

    def apply(self, X):
        Y = np.empty_like(X)
        for idxs, Pb in self.groups:
            Y[idxs] = np.einsum("bij,bj->bi", Pb, X[idxs])
        return Y


class Fit:
    def __init__(self, arm):
        self.ctx = R.Ctx()
        self.arm = arm
        self.m = V.Model(arm, 4, 3, 10, 16, -0.25, 8.0, (1, 1))
        self.PS = SpinProj(self.ctx.sec) if arm == "nnbf" else None
        self.g = self.ctx.gs / np.linalg.norm(self.ctx.gs)
        self.dim = self.ctx.sec.dim
        self.hist = []

    def phi(self, th, CH=8192):
        ls, ss = [], []
        with torch.no_grad():
            for s0 in range(0, self.dim, CH):
                l, s = self.m.logpsi0(th, self.ctx.nU[s0:s0 + CH], self.ctx.nD[s0:s0 + CH])
                ls.append(l)
                ss.append(s)
        l, s = torch.cat(ls), torch.cat(ss)
        return (s * torch.exp(l - l.max())).numpy()

    def proj(self, v):
        c = self.ctx
        if self.PS is not None:
            v = self.PS.apply(v)
        pr = 0.25 * (c.sg[:, c.reps] * v[c.idx[:, c.reps]]).sum(0)
        return c.E @ pr, pr

    def loss_grad(self, th_np, CH=8192):
        c = self.ctx
        th = torch.tensor(th_np)
        phi = self.phi(th)
        Pphi, pr = self.proj(phi)
        a = float(self.g @ Pphi)
        n2 = float(Pphi @ Pphi)  # = <phi|P|phi>, P = P_G P_S an orthogonal projector
        F = a * a / n2
        prn = pr / np.sqrt(np.sum(c.size * pr * pr))
        E = float(np.sum(c.size * prn * (c.Hr @ prn)))
        w = torch.tensor(-(2.0 * self.g / a - 2.0 * Pphi / n2) * phi)  # g = P g, so <g|P phi> = <g|phi>
        thg = th.clone().requires_grad_(True)
        for s0 in range(0, self.dim, CH):
            l, _ = self.m.logpsi0(thg, c.nU[s0:s0 + CH], c.nD[s0:s0 + CH])
            (w[s0:s0 + CH] * l).sum().backward()
        self.hist.append(dict(nfev=len(self.hist) + 1, infid=1 - F, eps=(E - E0) / abs(E0), cpu=cpu()))
        return -np.log(F), thg.grad.numpy().copy()


def check(a):
    out = {}
    for arm in ("che", "nnbf"):
        f = Fit(arm)
        t = cpu()
        th = f.m.init(0).numpy()
        l0, g0 = f.loss_grad(th)
        t1 = cpu() - t
        rng = np.random.default_rng(1)
        fd = []
        for j in rng.choice(len(th), 3, replace=False):
            e = np.zeros(len(th))
            e[j] = 1e-5
            fd.append(abs((f.loss_grad(th + e)[0] - f.loss_grad(th - e)[0]) / 2e-5 - g0[j]) / np.abs(g0).max())
        t = cpu()
        f.loss_grad(th)
        out[arm] = dict(P=f.m.nparam, infid_init=f.hist[0]["infid"], grad_fd_rel=float(max(fd)),
                        cpu_min_per_eval=cpu() - t, first_eval_cpu_min=t1,
                        gsym=float(np.abs(f.proj(f.g)[0] - f.g).max()))
        print(arm, json.dumps(out[arm]), flush=True)
    out["valid"] = all(out[k]["grad_fd_rel"] < 1e-6 and out[k]["gsym"] < 1e-10 for k in ("che", "nnbf"))
    out["cpu_min_total"] = cpu()
    json.dump(out, open(os.path.join(RES, "checks.json"), "w"), indent=1)
    print("CHECKS", json.dumps(out), flush=True)


class Stop(Exception):
    pass


def train(a):
    tag = f"{a.arm}_s{a.seed}"
    fst, fth = os.path.join(RES, f"state_{tag}.json"), os.path.join(RES, f"theta_{tag}.npy")
    f = Fit(a.arm)
    th0 = f.m.init(a.seed).numpy()
    prior, used = [], 0.0
    if os.path.exists(fst):
        st = json.load(open(fst))
        prior, used = st["hist"], st["cpu_min_train"]
        th0 = np.load(fth)
    best = dict(loss=np.inf, th=th0.copy())
    flag = dict(sig=False)
    signal.signal(signal.SIGUSR1, lambda *_: flag.update(sig=True))
    t_start = cpu()

    def save(done):
        hist = prior + [dict(h, cpu=h["cpu"] - t_start + used) for h in f.hist]
        np.save(fth, best["th"])
        json.dump(dict(arm=a.arm, seed=a.seed, done=done, cpu_min_train=used + cpu() - t_start, hist=hist),
                  open(fst, "w"))

    def fun(th):
        if flag["sig"] or used + cpu() - t_start >= a.cpu_min:
            raise Stop
        l, g = f.loss_grad(th)
        if l < best["loss"]:
            best.update(loss=l, th=th.copy())
        if len(f.hist) % 50 == 0:
            save(False)
            print(json.dumps(f.hist[-1]), flush=True)
        return l, g

    try:
        minimize(fun, th0, jac=True, method="L-BFGS-B",
                 options=dict(maxcor=30, maxfun=a.maxfun, maxiter=a.maxfun, ftol=1e-15, gtol=1e-12))
    except Stop:
        pass
    save(not flag["sig"])
    if flag["sig"]:
        sys.exit(4)


def verdict(a):
    S = {}
    for fn in glob.glob(os.path.join(RES, "state_*.json")):
        st = json.load(open(fn))
        h = st["hist"]
        T = st["cpu_min_train"]

        def best_by(tcut):
            hh = [x for x in h if x["cpu"] <= tcut] or h[:1]
            return min(hh, key=lambda x: x["infid"])
        full, half = best_by(T), best_by(T / 2)
        q = [x for x in h if x["cpu"] <= 0.75 * T]
        late = (1 - full["infid"] / min(x["infid"] for x in q)) if q else None
        S[(st["arm"], st["seed"])] = dict(done=st["done"], T=T, full=full, half=half, late_drop=late)
    out = dict(arms={f"{k[0]}_s{k[1]}": v for k, v in S.items()})

    seeds = sorted({s for (_, s) in S if ("che", s) in S and ("nnbf", s) in S})
    lo, hi = (NO_LOSS, LOSS) if len(seeds) > 1 else (1.0, 1.65)  # RULE_T2: single-seed margins

    def cls(k):
        return "NO-LOSS" if k <= lo else ("LOSS" if k >= hi else "INTERMEDIATE")
    for when in ("full", "half"):
        ks = [S[("che", s)][when]["infid"] / S[("nnbf", s)][when]["infid"] for s in seeds]
        ke = [(S[("che", s)][when]["eps"]) / (S[("nnbf", s)][when]["eps"]) for s in seeds]
        out[when] = dict(kappa=ks, kappa_mean=float(np.mean(ks)) if ks else None, eps_ratio=ke,
                         cls=[cls(k) for k in ks], cls_mean=cls(float(np.mean(ks))) if ks else None)
    c = out["full"]
    if not seeds or not all(S[k]["done"] for k in S):
        v = "INCOMPLETE"
    elif len(set(c["cls"])) > 1 or c["cls_mean"] != out["half"]["cls_mean"]:
        v = "UNSETTLED"
    else:
        v = c["cls_mean"]
    if v in ("NO-LOSS", "LOSS", "INTERMEDIATE") and any(S[k]["late_drop"] is not None and S[k]["late_drop"] > 0.10
                                                       for k in S):
        v += " (NOT-PLATEAUED)"
    out["verdict"] = v
    json.dump(out, open(os.path.join(RES, "verdict.json"), "w"), indent=1)
    print("VERDICT", json.dumps(out), flush=True)


dict(check=check, train=train, verdict=verdict)[A.mode](A)
