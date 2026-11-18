#!/usr/bin/env python3
"""
P4.1 R47 T1 (`distill-24` at 12 sites): can SAMPLED, UNPROJECTED spin-reading students reproduce at 12 sites the
capacity loss that exact projection shows (R45: kappa = 2.23)?  Rule: RULE_T1.md (with its 2026-11-18 amendment, this
folder).  Card: leaves/P4.1/cards/R47.md.

System (as R45): open 4x3, N_e = 10 (5+5), U = 8, t' = -0.25, (+,+) via P_G; teacher g = exact (+,+) S = 0 ground state.
Data (RULE_T1 Amendment 2, online distillation): every L-BFGS run of --seg = 50 evaluations fits a FRESH batch of
N_tr = 32,768 i.i.d. draws from |g|^2 (batch k seed 101 + 7919 k; batch 0 = seed 101), restarting from the carried
theta (best batch loss of the previous run); held-out N_ho = 131,072 (seed 202, scored through the full vector).
The overfitting clause uses infid_fresh = 1-F_hat of the logged theta on the NEXT batch (unseen); the in-batch
1-F_hat (infid_tr) is reported only (it is biased low by construction within a run).
Loss on a set X: -log F_hat, F_hat = |mean_X r|^2 / mean_X r^2, r = psi/g (real amplitudes).
Arms (all V.Model(...).init(0), H = 16, P = 2637, L-BFGS-B maxcor 30):
  che        charge-only backflow, exact S by Che's closed form (inside logpsi0), P_G.
  nnbf_u     spin-reading nnbf, P_G only (no P_S).                                psi = P_G phi
  nnbf_even  nnbf_u symmetrised under global spin flip (r43 FlipModel, here before P_G): phi(nU,nD) + s phi(nD,nU),
             s = the teacher's flip character (checked to be +-1 exactly).
  nnbf_p     nnbf with P_G and EXACT P_S (amendment 3b): P_S is block diagonal in the charge configuration, so
             psi(x) needs phi on the charge blocks of x's 4 images only.                psi = P_G P_S phi
Every arm: psi(X) = A phi(U) with A a sparse (|X| x |U|) matrix (images, signs, 1/g, flip or P_S folded in) and U the
configurations whose amplitudes are needed; gradient by one weighted backward pass over U (as R45).
Logging: every 50 evaluations and at the end, for the best-train-loss theta so far: exact 1-F (full 627,264 sum),
held-out and train 1-F_hat, eps, exact <S^2>, <P_S> (exact quantities are reported only, never in the loss).

Run with ~/.conda/envs/lit_env/bin/python, ONE thread (set below before numpy/torch import; L63).
Modes (--mode):
  check     (a) estimator: sampled 1-F_hat vs exact 1-F for R45's trained states (../P4.1_cheap_su2_r45/results/
                theta_{che,nnbf}_s0.npy; nnbf scored both with and without P_S), on the actual train/held-out sets and
                on 20 further independent 16,384-sets (relative spread);
            (b) teacher checks: P_G g = g, S^2 g = 0, flip character of g is +-1;
            (c) per arm at init(0): gradient vs central differences (3 random components; pass < 1e-6 relative to
                max|grad|), CPU-min per evaluation (2nd evaluation), |U|.  Writes results/checks.json.
  train     one arm (--arm) to --maxfun evaluations or --cpu-min CPU minutes; state results/state_<arm>.json and
            results/theta_<arm>.npy; resumes from them (best theta; L-BFGS memory restarts).  SIGUSR1: checkpoint,
            exit 4.
  dmrgcheck reads the Julia rehearsal output (--dmrg-dir, written by r47_dmrg.jl) and checks: MPO <S^2> vs dense
            <S^2> of the MPS mapped to the python basis (|diff| <= 1e-6), sampled amplitudes vs dense entries
            (relative <= 1e-8), reports |<MPS|g>|^2, w_T <= <S^2>/2.  Writes results/dmrg_check.json.
  verdict   RULE_T1 (+ amendment) scoring from the four state files and dmrg_check.json -> results/verdict.json.
Options: --arm {che,nnbf_u,nnbf_even,nnbf_p} (train), --maxfun INT (default 4000), --cpu-min FLOAT (default 1e9),
         --threads (default 1), --res DIR (default ./results), --dmrg-dir DIR (default ./dmrg_out),
         --ntr INT / --nho INT (default 32768 / 131072), --log-every INT (default 50; keep = --seg so every log
         point ends an L-BFGS run), --resample {restart (default; Amendment 2: fresh batch per L-BFGS run of --seg
         evaluations), eval (fresh batch every evaluation; rejected in the pre-flight: unstable), fixed (one batch;
         the original design, pre-flight comparison only)}, --seg INT (default 50).
"""
import argparse
import glob
import json
import os
import signal
import sys
import time

ARMS = ("che", "nnbf_u", "nnbf_even", "nnbf_p")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["check", "train", "dmrgcheck", "verdict"])
ap.add_argument("--arm", choices=ARMS)
ap.add_argument("--maxfun", type=int, default=4000)
ap.add_argument("--cpu-min", type=float, default=1e9, dest="cpu_min")
ap.add_argument("--threads", default="1")
ap.add_argument("--res", default=None)
ap.add_argument("--dmrg-dir", default=None, dest="dmrg_dir")
ap.add_argument("--ntr", type=int, default=32768)
ap.add_argument("--nho", type=int, default=131072)
ap.add_argument("--log-every", type=int, default=50, dest="log_every")
ap.add_argument("--resample", default="restart", choices=["restart", "eval", "fixed"])
ap.add_argument("--seg", type=int, default=50)
A = ap.parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = A.threads
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
DMRG_DIR = A.dmrg_dir or os.path.join(HERE, "dmrg_out")
os.makedirs(RES, exist_ok=True)
c0 = time.process_time()
MARGIN = 1.65
PLATEAU = 0.10


def cpu():
    return (time.process_time() - c0) / 60


def spin_ops(sec):
    """S^2 (sparse, full S^z = 0 sector) and exact P_S (sparse, block diagonal in the charge configuration).
    Same construction as r45.py SpinProj (copied; r45.py parses argv at import)."""
    dim = sec.dim
    bu = sec.up.bits[sec.iu].astype(float)
    bd = sec.dn.bits[sec.id].astype(float)
    diag = (bu * (1 - bd)).sum(1) + (sec.Sz ** 2 - sec.Sz)
    M = sp.diags(diag).tocsr()
    for Eu, Ed in sec.spsm_pairs():
        M = M - sp.kron(Eu, Ed, format="csr")
    M = M.tocsr()
    cid = sec.cidx
    order = np.argsort(cid, kind="stable")
    cnt = np.bincount(cid)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.empty(dim, dtype=np.int64)
    pos[order] = np.arange(dim) - start[cid[order]]
    coo = M.tocoo()
    assert np.all(cid[coo.row] == cid[coo.col])
    rows, cols, vals = [], [], []
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
        Pb = np.einsum("bik,bk,bjk->bij", U, keep, U)
        rows.append(np.repeat(idxs, d, axis=1).ravel())
        cols.append(np.tile(idxs, (1, d)).ravel())
        vals.append(Pb.ravel())
    PS = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    PS.eliminate_zeros()
    return M, PS


class Problem:
    """teacher, data sets, operators (built once per process)."""

    def __init__(self, ntr, nho, need_ps=True):
        self.ctx = c = R.Ctx()
        sec = c.sec
        self.dim = sec.dim
        nd = sec.dn.dim
        assert np.all(sec.iu * nd + sec.id == np.arange(self.dim))
        assert np.array_equal(sec.up.bits, sec.dn.bits)
        self.flip = sec.id * nd + sec.iu
        g = c.gs / np.linalg.norm(c.gs)
        self.g = g
        gf = g[self.flip]
        self.sflip = 1.0 if np.abs(gf - g).max() < 1e-10 else (-1.0 if np.abs(gf + g).max() < 1e-10 else 0.0)
        p = g * g / np.sum(g * g)
        self.p = p
        self.Xtr = np.random.default_rng(101).choice(self.dim, ntr, p=p)
        self.Xho = np.random.default_rng(202).choice(self.dim, nho, p=p)
        self.M, self.PS = spin_ops(sec) if need_ps else (None, None)
        I = sp.identity(self.dim, format="csr")
        Fl = sp.csr_matrix((np.ones(self.dim), (np.arange(self.dim), self.flip)), shape=(self.dim, self.dim))
        self.T = dict(che=I, nnbf_u=I, nnbf_even=(I + self.sflip * Fl).tocsr(), nnbf_p=self.PS)

    def gmap(self, X):
        """sparse G (|X| x dim): psi(X) = G @ v for v the pre-P_G vector, rows divided by g(X)."""
        c = self.ctx
        rows = np.repeat(np.arange(len(X)), 4)
        cols = c.idx[:, X].T.ravel()
        vals = (0.25 * c.sg[:, X] / self.g[X][None, :]).T.ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(X), self.dim))

    def batch(self, n, k):
        """fit batch k (k = 0: the rule's original seed-101 set; k >= 1: fresh i.i.d. draws, seed 101 + 7919 k)."""
        return np.random.default_rng(101 if k == 0 else 101 + 7919 * k).choice(self.dim, n, p=self.p)

    def amap(self, arm, X):
        Af = (self.gmap(X) @ self.T[arm]).tocsc()
        U = np.nonzero(np.diff(Af.indptr))[0]
        return Af[:, U].tocsr(), U

    def full_psi(self, arm, phi):
        """P_G T phi on the full basis."""
        c = self.ctx
        v = self.T[arm] @ phi
        pr = 0.25 * (c.sg[:, c.reps] * v[c.idx[:, c.reps]]).sum(0)
        return c.E @ pr, pr

    def exact(self, arm, phi):
        c = self.ctx
        psi, pr = self.full_psi(arm, phi)
        n2 = float(psi @ psi)
        F = float(self.g @ psi) ** 2 / n2
        prn = pr / np.sqrt(np.sum(c.size * pr * pr))
        E = float(np.sum(c.size * prn * (c.Hr @ prn)))
        out = dict(infid=1 - F, eps=(E - E0) / abs(E0))
        if self.M is not None:
            out["S2"] = float(psi @ (self.M @ psi)) / n2
            out["PS"] = float(psi @ (self.PS @ psi)) / n2
            out["infid_PSpsi"] = 1 - F / out["PS"]  # F(P_S psi) = F(psi)/<P_S> (g exactly S = 0)
        return out, psi


def fhat(r):
    return float(np.mean(r) ** 2 / np.mean(r * r))


class Student:
    def __init__(self, P, arm, seed=0):
        self.P, self.arm = P, arm
        self.m = V.Model("che" if arm == "che" else "nnbf", 4, 3, 10, 16, -0.25, 8.0, (1, 1))
        self.th0 = self.m.init(seed).numpy()
        self.set_batch(P.Xtr)
        self.Aho, self.Uho = P.amap(arm, P.Xho)
        self.hist = []

    def set_batch(self, X):
        self.Xcur = X
        self.Atr, self.Utr = self.P.amap(self.arm, X)

    def phi(self, th, U=None, CH=8192):
        c = self.P.ctx
        nU, nD = (c.nU, c.nD) if U is None else (c.nU[U], c.nD[U])
        ls, ss = [], []
        with torch.no_grad():
            for s0 in range(0, nU.shape[0], CH):
                l, s = self.m.logpsi0(torch.as_tensor(th), nU[s0:s0 + CH], nD[s0:s0 + CH])
                ls.append(l)
                ss.append(s)
        l, s = torch.cat(ls), torch.cat(ss)
        return (s * torch.exp(l - l.max())).numpy()

    def loss_grad(self, th_np, CH=8192):
        c = self.P.ctx
        th = torch.tensor(th_np)
        U = self.Utr
        ph = self.phi(th, U)
        r = self.Atr @ ph
        m1, m2 = float(r.mean()), float((r * r).mean())
        loss = -np.log(m1 * m1 / m2)
        dr = (-2.0 / m1 + 2.0 * r / m2) / len(r)
        w = torch.tensor((self.Atr.T @ dr) * ph)
        thg = th.clone().requires_grad_(True)
        nU, nD = c.nU[U], c.nD[U]
        for s0 in range(0, len(U), CH):
            l, _ = self.m.logpsi0(thg, nU[s0:s0 + CH], nD[s0:s0 + CH])
            (w[s0:s0 + CH] * l).sum().backward()
        self.hist.append(loss)
        return loss, thg.grad.numpy().copy()

    def metrics(self, th, Xfresh=None):
        ph = self.phi(th)
        out, psi = self.P.exact(self.arm, ph)
        # sampled estimators from the same theta, via the full vector (identical to A phi(U) up to scale)
        out["infid_tr"] = 1 - fhat(psi[self.Xcur] / self.P.g[self.Xcur])
        out["infid_ho"] = 1 - fhat(psi[self.P.Xho] / self.P.g[self.P.Xho])
        if Xfresh is not None:  # the next fit batch, not yet seen by this theta: out-of-sample 1-F_hat at N_tr
            out["infid_fresh"] = 1 - fhat(psi[Xfresh] / self.P.g[Xfresh])
        return out


# ------------------------------------------------------------------------------------------------ check
def check(a):
    P = Problem(a.ntr, a.nho)
    out = dict(teacher=dict(gsym=float(np.abs(P.full_psi("che", P.g)[0] - P.g).max()),
                            S2_g=float(P.g @ (P.M @ P.g)), PS_g=float(P.g @ (P.PS @ P.g)), sflip=P.sflip,
                            setup_cpu_min=cpu()))
    print("TEACHER", json.dumps(out["teacher"]), flush=True)
    # (a) estimator on R45's trained states
    r45 = os.path.join(TESTS, "P4.1_cheap_su2_r45", "results")
    est = {}
    for name, arm, thf in (("che_R45", "che", "theta_che_s0.npy"), ("nnbf_R45_projected", "nnbf_p", "theta_nnbf_s0.npy"),
                           ("nnbf_R45_unprojected", "nnbf_u", "theta_nnbf_s0.npy")):
        st = Student.__new__(Student)
        st.P, st.arm = P, arm
        st.m = V.Model("che" if arm == "che" else "nnbf", 4, 3, 10, 16, -0.25, 8.0, (1, 1))
        th = np.load(os.path.join(r45, thf))
        ph = st.phi(th)
        ex, psi = P.exact(arm, ph)
        r = psi / np.where(P.g == 0, 1, P.g)
        tr, ho = 1 - fhat(r[P.Xtr]), 1 - fhat(r[P.Xho])
        rng = np.random.default_rng(999)
        p = P.g ** 2
        reps = [1 - fhat(r[rng.choice(P.dim, a.nho, p=p)]) for _ in range(20)]
        e = dict(exact=ex["infid"], train=tr, heldout=ho, rel_ho=ho / ex["infid"] - 1, rel_tr=tr / ex["infid"] - 1,
                 rep_rel_mean=float(np.mean(reps) / ex["infid"] - 1), rep_rel_std=float(np.std(reps) / ex["infid"]),
                 rep_rel_maxabs=float(np.max(np.abs(np.array(reps) / ex["infid"] - 1))), S2=ex.get("S2"), PS=ex.get("PS"))
        est[name] = e
        print("EST", name, json.dumps(e), flush=True)
    out["estimator"] = est
    # (c) gradients and timing per arm
    for arm in ARMS:
        t = cpu()
        st = Student(P, arm)
        tb = cpu() - t
        th = st.th0
        l0, g0 = st.loss_grad(th)
        t = cpu()
        st.loss_grad(th)
        tev = cpu() - t
        rng = np.random.default_rng(1)
        fd = []
        for j in rng.choice(len(th), 3, replace=False):
            e = np.zeros(len(th))
            e[j] = 1e-5
            fd.append(abs((st.loss_grad(th + e)[0] - st.loss_grad(th - e)[0]) / 2e-5 - g0[j]) / np.abs(g0).max())
        t = cpu()
        mt = st.metrics(th)
        tm = cpu() - t
        out[arm] = dict(U_tr=int(len(st.Utr)), U_ho=int(len(st.Uho)), build_cpu_min=tb, cpu_min_per_eval=tev,
                        metrics_cpu_min=tm, grad_fd_rel=float(max(fd)), loss_init=float(l0), init=mt)
        print("ARM", arm, json.dumps(out[arm]), flush=True)
    out["valid"] = (all(out[k]["grad_fd_rel"] < 1e-6 for k in ARMS) and out["teacher"]["gsym"] < 1e-10
                    and abs(out["teacher"]["S2_g"]) < 1e-8 and P.sflip != 0.0)
    out["cpu_min_total"] = cpu()
    json.dump(out, open(os.path.join(RES, "checks.json"), "w"), indent=1)
    print("CHECKS valid", out["valid"], "cpu_min_total", out["cpu_min_total"], flush=True)


# ------------------------------------------------------------------------------------------------ train
class Stop(Exception):
    pass


def train(a):
    """online distillation (RULE_T1 amendment 2): --resample restart = a fresh batch of N_tr teacher samples for
    every L-BFGS run of --seg evaluations (restart from the carried theta, memory lost); eval = a fresh batch at every
    evaluation inside one L-BFGS run; fixed = the original fixed-data fit (batch 0 only; pre-flight comparison)."""
    tag = a.arm
    fst, fth = os.path.join(RES, f"state_{tag}.json"), os.path.join(RES, f"theta_{tag}.npy")
    P = Problem(a.ntr, a.nho, need_ps=True)
    st = Student(P, a.arm)
    th = st.th0
    logs, used, nfev, nb = [], 0.0, 0, 0
    if os.path.exists(fst):
        s = json.load(open(fst))
        if s.get("done"):
            print("already done", flush=True)
            return
        logs, used, nfev, nb = s["log"], s["cpu_min_train"], s["nfev"], s["nbatch"]
        th = np.load(fth)
    cur = dict(loss=np.inf, th=th.copy(), fresh=None)
    flag = dict(sig=False)
    signal.signal(signal.SIGUSR1, lambda *_: flag.update(sig=True))
    t_start = cpu()
    cnt = dict(n=nfev, nb=nb, seg=0)

    def now():
        return used + cpu() - t_start

    def log_point():
        mt = st.metrics(cur["th"], P.batch(a.ntr, cnt["nb"]) if a.resample == "restart" else None)
        logs.append(dict(nfev=cnt["n"], cpu=now(), nbatch=cnt["nb"], loss_batch=float(cur["loss"]),
                         loss_fresh=cur["fresh"], **mt))
        print(json.dumps(logs[-1]), flush=True)

    def save(done):
        np.save(fth, cur["th"])
        json.dump(dict(arm=a.arm, done=done, nfev=cnt["n"], nbatch=cnt["nb"], maxfun=a.maxfun, resample=a.resample,
                       seg=a.seg, cpu_min_train=now(), ntr=a.ntr, nho=a.nho, log=logs), open(fst, "w"))

    def newbatch():
        st.set_batch(P.batch(a.ntr, cnt["nb"]))
        cnt["nb"] += 1

    def fun(x):
        if flag["sig"] or now() >= a.cpu_min or cnt["n"] >= a.maxfun:
            raise Stop
        if a.resample == "eval":
            newbatch()
        l, g = st.loss_grad(x)
        cnt["n"] += 1
        cnt["seg"] += 1
        if cnt["seg"] == 1:
            cur["fresh"] = float(l)  # first evaluation on an unseen batch: unbiased loss of the carried theta
        if a.resample == "eval" or l < cur["loss"]:
            cur.update(loss=float(l), th=x.copy())
        if cnt["n"] % a.log_every == 0:
            log_point()
            save(False)
        return l, g

    stopped = False
    while not stopped:
        if a.resample == "restart" or (a.resample == "fixed" and cnt["nb"] == 0) or a.resample == "eval":
            if a.resample != "eval":
                newbatch()
            elif cnt["nb"] == 0:
                newbatch()
        cur["loss"] = np.inf
        cnt["seg"] = 0
        mf = a.seg if a.resample == "restart" else 10 ** 7
        try:
            minimize(fun, cur["th"], jac=True, method="L-BFGS-B",
                     options=dict(maxcor=30, maxfun=mf, maxiter=10 ** 7, ftol=1e-15, gtol=1e-12))
        except Stop:
            stopped = True
        if a.resample != "restart" and not stopped:  # L-BFGS terminated on its own: restart (same rule)
            if cnt["seg"] <= 1:
                break
    if not flag["sig"] and (not logs or logs[-1]["nfev"] != cnt["n"]):
        log_point()
    save(not flag["sig"])
    if flag["sig"]:
        sys.exit(4)


# ------------------------------------------------------------------------------------------------ dmrgcheck
def dmrgcheck(a):
    P = Problem(1, 1, need_ps=True)
    sec = P.ctx.sec
    meta = json.load(open(os.path.join(DMRG_DIR, "dmrg_meta.json")))
    ub = np.fromfile(os.path.join(DMRG_DIR, "sector_ubits.bin"), dtype="<i8")
    db = np.fromfile(os.path.join(DMRG_DIR, "sector_dbits.bin"), dtype="<i8")
    amp = np.fromfile(os.path.join(DMRG_DIR, "sector_amp.bin"), dtype="<f8")
    L = sec.lat.L if hasattr(sec.lat, "L") else 12
    w = 1 << np.arange(L)
    code_u = sec.up.bits.astype(np.int64) @ w
    code_d = sec.dn.bits.astype(np.int64) @ w
    lu = {int(k): i for i, k in enumerate(code_u)}
    ld = {int(k): i for i, k in enumerate(code_d)}
    nd = sec.dn.dim
    k = np.array([lu[int(u)] * nd + ld[int(d)] for u, d in zip(ub, db)])
    assert len(np.unique(k)) == len(k) == P.dim
    # ITensor order (1up,1dn,2up,2dn,...) -> python (all up in site order, then all dn): sign (-1)^{sum_j nu_j #dn(i<j)}
    bu = ((ub[:, None] >> np.arange(L)) & 1)
    bd = ((db[:, None] >> np.arange(L)) & 1)
    dn_before = np.cumsum(bd, 1) - bd
    s1 = 1 - 2 * ((bu * dn_before).sum(1) % 2)
    s2 = s1 * (1 - 2 * ((bu * bd).sum(1) % 2))  # alternative on-site convention (UpDn = c+dn c+up)
    best = None
    for name, s in (("updn_up_first", s1), ("updn_dn_first", s2)):
        v = np.zeros(P.dim)
        v[k] = s * amp
        n = np.linalg.norm(v)
        ov = float(P.g @ v / n) ** 2
        if best is None or ov > best[1]:
            best = (name, ov, v / n)
    name, ov, v = best
    S2_dense = float(v @ (P.M @ v))
    smp = json.load(open(os.path.join(DMRG_DIR, "samples.json")))
    dense = {}
    key = (ub << 32) | db
    order = np.argsort(key)
    rel = []
    for su, sd, sa in zip(smp["ubits"], smp["dbits"], smp["amp"]):
        j = order[np.searchsorted(key[order], (su << 32) | sd)]
        rel.append(abs(sa - amp[j]) / abs(amp[j]))
    out = dict(sign_convention=name, overlap_g=ov, S2_mpo=meta["S2_mpo"], S2_dense=S2_dense,
               dS2=abs(meta["S2_mpo"] - S2_dense), wT_bound=meta["S2_mpo"] / 2, sample_rel_max=float(max(rel)),
               nsamples=len(rel), E_mps=meta.get("E_last"), maxdims=meta.get("maxdims"), dmrg_seconds=meta.get("seconds"))
    out["valid"] = out["dS2"] <= 1e-6 and out["sample_rel_max"] <= 1e-8
    json.dump(out, open(os.path.join(RES, "dmrg_check.json"), "w"), indent=1)
    print("DMRGCHECK", json.dumps(out), flush=True)


# ------------------------------------------------------------------------------------------------ verdict
def verdict(a):
    S = {}
    for arm in ARMS:
        fn = os.path.join(RES, f"state_{arm}.json")
        if not os.path.exists(fn):
            continue
        st = json.load(open(fn))
        lg, T = st["log"], st["cpu_min_train"]

        def best_by(cond):
            hh = [x for x in lg if cond(x)] or lg[:1]
            return min(hh, key=lambda x: x["infid"])
        full, half = best_by(lambda x: x["cpu"] <= T + 1e-9), best_by(lambda x: x["cpu"] <= T / 2)
        e2000 = best_by(lambda x: x["nfev"] <= 2000)
        q = [x for x in lg if x["cpu"] <= 0.75 * T]
        late = (1 - full["infid"] / min(x["infid"] for x in q)) if q else None
        fin = lg[-1]
        S[arm] = dict(done=st["done"], T=T, nfev=st["nfev"], full=full, half=half, at2000=e2000, late_drop=late,
                      plateaued=(late is not None and late <= PLATEAU),
                      ho_rel=fin["infid_ho"] / fin["infid"] - 1,
                      tr_over_ho=fin.get("infid_fresh", fin["infid_tr"]) / fin["infid_ho"],
                      inbatch_over_ho=fin["infid_tr"] / fin["infid_ho"])
    out = dict(arms=S)
    inval = []
    for arm, s in S.items():
        if abs(s["ho_rel"]) > 0.20:
            inval.append(f"{arm}: held-out 1-F_hat off exact by {s['ho_rel']:+.3f}")
        if s["tr_over_ho"] < 0.8:
            inval.append(f"{arm}: fresh-batch/held-out 1-F_hat = {s['tr_over_ho']:.3f} < 0.8 (overfit; Amendment 2)")
    dfn = os.path.join(RES, "dmrg_check.json")
    if not os.path.exists(dfn) or not json.load(open(dfn))["valid"]:
        inval.append("DMRG pipeline check missing or failed")
    out["invalid_reasons"] = inval

    def kap(when):
        b = min(("nnbf_u", "nnbf_even"), key=lambda k: S[k][when]["infid"])
        return S["che"][when]["infid"] / S[b][when]["infid"], b
    if any(k not in S or not S[k]["done"] for k in ARMS):
        v = "INCOMPLETE"
    else:
        kf, bf = kap("full")
        kh, bh = kap("half")
        k2, _ = kap("at2000")
        out.update(kappa_u_full=kf, best_full=bf, kappa_u_half=kh, best_half=bh, kappa_u_2000=k2,
                   kappa_p_full=S["che"]["full"]["infid"] / S["nnbf_p"]["full"]["infid"],
                   kappa_p_half=S["che"]["half"]["infid"] / S["nnbf_p"]["half"]["infid"])
        if inval:
            v = "INVALID"
        elif kf >= MARGIN and kh >= MARGIN and S["che"]["plateaued"]:
            v = "PASS"
        elif kf < MARGIN and kh < MARGIN and S[bf]["plateaued"]:
            v = "KILL (" + ("missing projection: nnbf_p >= 1.65" if out["kappa_p_full"] >= MARGIN
                            else "sampled fit: nnbf_p also < 1.65") + ")"
        else:
            why = []
            if (kf >= MARGIN) != (kh >= MARGIN):
                why.append("full/half classes disagree")
            if kf >= MARGIN and not S["che"]["plateaued"]:
                why.append("che NOT-PLATEAUED")
            if kf < MARGIN and not S[bf]["plateaued"]:
                why.append(f"{bf} NOT-PLATEAUED")
            v = "UNSETTLED (" + "; ".join(why) + ")"
    out["verdict"] = v
    json.dump(out, open(os.path.join(RES, "verdict.json"), "w"), indent=1)
    print("VERDICT", v, json.dumps({k: out[k] for k in out if k.startswith("kappa")}), flush=True)


dict(check=check, train=train, dmrgcheck=dmrgcheck, verdict=verdict)[A.mode](A)
