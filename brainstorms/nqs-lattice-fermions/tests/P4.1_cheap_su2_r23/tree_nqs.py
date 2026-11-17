#!/usr/bin/env python3
"""
P4.1 r23 (R22 `spin-tree`, falsifier 2): tree-basis NQS psi = P_G[D(c) F_theta(c,T)] on the 12-site strip, exact sums.
=====================================================================================================================

Rule: RULE.md (pre-registered). Needs results/sector12.npz from `spin_tree.py --mode validate` (basis, recoupling H,
point-group maps), and that run's results/validate.json with ok = true.

Run with the lit_env interpreter from this directory:
    ~/.conda/envs/lit_env/bin/python tree_nqs.py --mode MODE [options]

Modes (--mode, required)
  check     gradient check (analytic d psi/d theta against autograd on 64 states), parameter count, orbit-reduction
            checks (irrep dimension against validate.json, expansion and norm identities); prints, writes nothing.
  time      time one stage-A Adam step (average over --nt steps) and one stage-B ladder step (S, H build, tangent
            eigendecomposition, all 5 rungs); writes results_probe/time.json. Energies are not read for any decision.
  pretrain  stage A for --seed: --nA Adam steps, cosine learning rate 3e-3 -> 1e-5, exact energy; writes
            results/stageA_seed<seed>.json and results/thetaA_seed<seed>.npy. Resumes nothing (cheap); skips if the
            json exists (rerunning-jobs rule).
  polish    stage B: picks the seed with the lowest stage-A energy among results/stageA_seed*.json (all --seeds must
            exist), runs the r22 ladder (RULE.md) to STOP-RULE / STOP-NOGAIN / STOP-CAP (--steps), resuming from
            results/state_tree.json and results/theta_tree.npy; --cpu-cap stops before a step that would cross it.
  sign      Gate-2 sign diagnostic at the tree endpoint (RULE.md); writes results/sign.json.
  verdict   rho = eps_tree / eps_che (r22 `che` state), convergence guard, classes; prints, writes results/verdict.json.
Options
  --seed INT        pretrain only (default 0)
  --seeds INT       number of stage-A seeds that polish requires (default 3; seeds 0..seeds-1)
  --nA INT          stage-A Adam steps (default: the value fixed in RULE.md's erratum, 3000)
  --nt INT          time mode: Adam steps to average (default 20)
  --steps INT       stage-B step cap (default 12)
  --cpu-cap FLOAT   CPU-minutes for polish (default 150)
  --threads INT     torch/BLAS threads (default 1)
  --res DIR         results directory (default ./results)
"""
import argparse
import glob
import json
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "time", "pretrain", "polish", "sign", "verdict"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nA", type=int, default=3000)
    ap.add_argument("--nt", type=int, default=20)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cpu-cap", type=float, default=150.0)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--res", default=None)
    return ap.parse_args()


A = parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = A.threads
sys.argv = [sys.argv[0], "--threads", A.threads]  # realise.py reads --threads at import

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r11"))
import realise as R  # noqa: E402  (read only: Tan, the truncated-metric LM used by r19/r22)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(A.threads))
E0 = -7.827899034271753
TAU, MIN_GAIN, STOP_FRAC = 0.05, 1e-7, 1e-3
LADDER = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
PASS, KILL = 0.60, 1.00
H_HID = 30
RES = A.res or os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
c0 = time.process_time()
cpu = lambda: (time.process_time() - c0) / 60  # noqa: E731


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump(fn, d):
    json.dump(d, open(fn, "w"), indent=1)


# ============================================================================ context: basis, H, orbit reduction
class Ctx:
    def __init__(self):
        v = json.load(open(os.path.join(HERE, "results/validate.json")))
        assert v["ok"], "basis validation failed (results/validate.json); no run"
        z = np.load(os.path.join(HERE, "results/sector12.npz"))
        self.dim = int(z["dim"])
        H = sp.csr_matrix((z["Hdata"], z["Hind"], z["Hptr"]), shape=(self.dim, self.dim))
        C, T = z["C"].astype(np.int64), z["T"].astype(np.int64)
        order, internal, nsize = z["order"], z["internal"], z["node_size"]
        idx, sg = z["pg_idx"], z["pg_sgn"]
        # (+,+): a representative is valid unless some g fixes it with sign -1
        gstar = np.argmin(idx, 0)
        rep_of = idx[gstar, np.arange(self.dim)]
        reps = np.unique(rep_of)
        bad = np.zeros(self.dim, bool)
        for g in range(4):
            bad |= (idx[g] == np.arange(self.dim)) & (sg[g] < 0)
        valid = ~bad[reps]
        self.reps_all = reps
        self.reps = reps[valid]
        pos = np.full(self.dim, -1)
        pos[self.reps] = np.arange(len(self.reps))
        keep = pos[rep_of] >= 0
        self.sigma = sg[gstar, np.arange(self.dim)]
        self.E = sp.csr_matrix((self.sigma[keep], (np.arange(self.dim)[keep], pos[rep_of][keep])),
                               shape=(self.dim, len(self.reps)))
        srt = np.sort(idx[:, self.reps], 0)
        self.size = 1.0 + (np.diff(srt, axis=0) != 0).sum(0)
        self.sw = np.sqrt(self.size)
        self.Hr = (H[self.reps, :] @ self.E).tocsr()
        self.H = H
        self.idx, self.sg = idx, sg
        self.irrep_dim_validate = v["irreps"]["(1, 1)"]["dim"]
        # features: one-hot occupation per site (36) + one-hot doubled spin of every non-root internal node
        L = C.shape[1]
        cols = [np.eye(3)[C[:, s]] for s in range(L)]
        for vtx in internal:
            if vtx == 0:
                continue
            cols.append(np.eye(int(nsize[vtx]) + 1)[T[:, vtx]])
        self.X = torch.tensor(np.concatenate(cols, 1))
        self.nin = self.X.shape[1]
        # determinant slots: slot 2p = first electron on the site at fermion position p, 2p + 1 = second
        Cp = C[:, order]  # occupations in fermion order
        sl = []
        for p in range(L):
            sl.append(np.where(Cp[:, p] >= 1, 2 * p, -1))
            sl.append(np.where(Cp[:, p] == 2, 2 * p + 1, -1))
        sl = np.stack(sl, 1)
        self.slots = torch.tensor(np.sort(np.where(sl < 0, 10 ** 6, sl), 1)[:, :10])
        assert (self.slots < 2 * L).all()
        self.nslot = 2 * L
        self.shapes = [("Phi", (2 * L, 10)), ("W1", (self.nin, H_HID)), ("b1", (H_HID,)), ("W2", (H_HID, 2)),
                       ("b2", (2,))]
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.P = sum(self.sizes)
        Hs = self.Hr.tocoo()
        self.Hr_t = torch.sparse_coo_tensor(np.vstack([Hs.row, Hs.col]), Hs.data, Hs.shape).coalesce()
        self.size_t = torch.tensor(self.size)
        self.idx_t = torch.tensor(idx[:, self.reps])
        self.sg_t = torch.tensor(sg[:, self.reps])
        log(f"sector: dim {self.dim}; (+,+) representatives {len(self.reps)} (validate.json irrep dim "
            f"{self.irrep_dim_validate}); Hr nnz {self.Hr.nnz}; inputs {self.nin}; P {self.P}")

    def unpack(self, th):
        out, k = {}, 0
        for (n, s), z in zip(self.shapes, self.sizes):
            out[n] = th[k:k + z].reshape(s)
            k += z
        return out

    def init(self, seed):
        g = torch.Generator().manual_seed(seed)
        p = dict(Phi=torch.randn(self.nslot, 10, generator=g) / np.sqrt(10),
                 W1=torch.randn(self.nin, H_HID, generator=g) / np.sqrt(self.nin),
                 b1=0.1 * torch.randn(H_HID, generator=g),
                 W2=0.01 * torch.randn(H_HID, 2, generator=g),
                 b2=torch.tensor([0.0, 1.0]))
        return torch.cat([p[n].reshape(-1) for n, _ in self.shapes])

    # -------------------------------------------------------------- amplitudes
    def parts(self, th, ix=None):
        p = self.unpack(th)
        X = self.X if ix is None else self.X[ix]
        sl = self.slots if ix is None else self.slots[ix]
        Amat = p["Phi"][sl]  # (B, 10, 10)
        sD, lD = torch.linalg.slogdet(Amat)
        h = torch.tanh(X @ p["W1"] + p["b1"])
        o = h @ p["W2"] + p["b2"]
        return sD, lD, o[:, 0], torch.tanh(o[:, 1]), h, Amat, X

    def psi_full(self, th, shift=None):
        sD, lD, a, tb, *_ = self.parts(th)
        u = lD + a
        if shift is None:
            shift = u.max().detach()
        return sD * torch.exp(u - shift) * tb, shift

    def project(self, phi):
        """(+,+) projection evaluated at the representatives, P_G phi (rep), torch."""
        return 0.25 * (self.sg_t * phi[self.idx_t]).sum(0)

    def energy_t(self, th):
        phi, _ = self.psi_full(th)
        pr = self.project(phi)
        nrm = (self.size_t * pr * pr).sum()
        Hp = torch.sparse.mm(self.Hr_t, pr.unsqueeze(1)).squeeze(1)
        return (self.size_t * pr * Hp).sum() / nrm

    def energy(self, th):
        with torch.no_grad():
            phi, _ = self.psi_full(th)
            pr = self.project(phi).numpy()
        nrm = np.sqrt(np.sum(self.size * pr * pr))
        pr = pr / nrm
        return float(np.sum(self.size * pr * (self.Hr @ pr))), pr, phi.numpy() / nrm

    def dpsi(self, th, ix, shift):
        """analytic d psi / d theta (psi = sD exp(lD + a - shift) tanh(b)) on states ix, (B, P) numpy."""
        p = self.unpack(th)
        with torch.no_grad():
            sD, lD, a, tb, h, Amat, X = self.parts(th, ix)
            psi = sD * torch.exp(lD + a - shift) * tb
            Ainv = torch.linalg.inv(Amat)  # (B, 10, 10); d log|det| / d A[r, k] = Ainv[k, r]
            B = len(ix)
            dPhi = torch.zeros(B, self.nslot, 10)
            dPhi.scatter_add_(1, self.slots[ix].unsqueeze(2).expand(B, 10, 10), Ainv.transpose(1, 2))
            dPhi = dPhi * psi[:, None, None]
            da = psi
            db = sD * torch.exp(lD + a - shift) * (1 - tb * tb)
            dW2 = torch.stack([h * da[:, None], h * db[:, None]], 2)
            dpre = (da[:, None] * p["W2"][:, 0] + db[:, None] * p["W2"][:, 1]) * (1 - h * h)
            dW1 = X[:, :, None] * dpre[:, None, :]
            out = torch.cat([dPhi.reshape(B, -1), dW1.reshape(B, -1), dpre, dW2.reshape(B, -1),
                             torch.stack([da, db], 1)], 1)
        return out.numpy()

    def build(self, th, CH=2048):
        """Y = orbit-weighted [Psi, d Psi/d theta] at representatives, HY; as r19 polish.build."""
        E, pr, phi = self.energy(th)
        with torch.no_grad():
            _, shift = self.psi_full(th)
            nrm = np.sqrt(np.sum(self.size * self.project(self.psi_full(th, shift)[0]).numpy() ** 2))
        nrep, P = len(self.reps), self.P
        Y = np.empty((nrep, 1 + P))
        Y[:, 0] = pr * self.sw
        for r0 in range(0, nrep, CH):
            rr = slice(r0, min(r0 + CH, nrep))
            acc = np.zeros((rr.stop - r0, P))
            for g in range(4):
                ix = self.idx[g, self.reps[rr]]
                acc += 0.25 * self.sg[g, self.reps[rr]][:, None] * self.dpsi(th, torch.tensor(ix), shift)
            Y[rr, 1:] = acc / nrm * self.sw[rr, None]
        HY = np.empty_like(Y)
        for cc in range(0, 1 + P, 512):
            cb = slice(cc, min(cc + 512, 1 + P))
            HY[:, cb] = (self.Hr @ (Y[:, cb] / self.sw[:, None])) * self.sw[:, None]
        return E, Y, HY


# ============================================================================ modes
def check(ctx):
    th = ctx.init(0)
    ix = torch.arange(0, ctx.dim, ctx.dim // 64)[:64]
    _, shift = ctx.psi_full(th)
    Ga = ctx.dpsi(th, ix, shift)
    thr = th.clone().requires_grad_(True)
    rows = []
    for k in ix.tolist():
        sD, lD, a, tb, *_ = ctx.parts(thr, torch.tensor([k]))
        val = (sD * torch.exp(lD + a - shift) * tb).sum()
        g, = torch.autograd.grad(val, thr)
        rows.append(g.numpy())
    Gb = np.array(rows)
    rel = float(np.abs(Ga - Gb).max() / np.abs(Gb).max())
    # orbit reduction identities on a random vector
    with torch.no_grad():
        phi = ctx.psi_full(th)[0]
        pr = ctx.project(phi).numpy()
    full = ctx.E @ pr
    direct = np.zeros(ctx.dim)
    ph = phi.numpy()
    for g in range(4):  # P_G phi on the full basis: (U_g phi)[idx[g,b]] = sg[g,b] phi[b]
        tmp = np.zeros(ctx.dim)
        tmp[ctx.idx[g]] = ctx.sg[g] * ph
        direct += 0.25 * tmp
    chk = dict(P=ctx.P, P_vs_che=ctx.P / 2637 - 1, grad_rel=rel, nrep=len(ctx.reps),
               irrep_dim_validate=ctx.irrep_dim_validate,
               expand=float(np.abs(full - direct).max() / np.abs(direct).max()),
               norm=float(abs(np.sum(ctx.size * pr * pr) / np.sum(direct * direct) - 1)),
               Hr=float(np.abs(ctx.Hr @ pr - (ctx.H @ direct)[ctx.reps]).max()))
    chk["ok"] = (abs(chk["P_vs_che"]) <= 0.02 and rel < 1e-10 and chk["nrep"] == chk["irrep_dim_validate"]
                 and chk["expand"] < 1e-12 and chk["norm"] < 1e-12 and chk["Hr"] < 1e-10)
    log(f"check: {chk}")
    return chk


def adam_run(ctx, seed, nA, t_report=None):
    th = ctx.init(seed).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=3e-3)
    hist = []
    for it in range(nA):
        lr = 1e-5 + 0.5 * (3e-3 - 1e-5) * (1 + np.cos(np.pi * it / max(nA - 1, 1)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.zero_grad()
        E = ctx.energy_t(th)
        E.backward()
        opt.step()
        if it % 100 == 0 or it == nA - 1:
            hist.append((it, float(E)))
        if t_report and it % t_report == 0:
            log(f"seed {seed} it {it}: E {float(E):.8f} lr {lr:.2e} [{cpu():.2f} CPU-min]")
    return th.detach(), hist


def time_mode(ctx):
    t0 = cpu()
    adam_run(ctx, 0, A.nt)
    t_adam = (cpu() - t0) / A.nt
    th = ctx.init(0)
    s0 = cpu()
    E, Y, HY = ctx.build(th)
    S, Hm = 0.5 * (Y.T @ Y + (Y.T @ Y).T), Y.T @ HY
    Hm = 0.5 * (Hm + Hm.T)
    del Y, HY
    t_build = cpu() - s0
    tan = R.Tan(S, Hm, E, list(range(ctx.P)))
    t_eig = cpu() - s0 - t_build
    for c in LADDER:
        r = tan.reduce(c)
        z, lam = tan.candidate(r, "lm", TAU)
        d = tan.dtheta(r, z, ctx.P)
        for s in (1.0, 0.5, 0.25):
            ctx.energy(th + s * torch.tensor(d))
    t_step = cpu() - s0
    out = dict(adam_step_cpu_min=t_adam, ladder_step_cpu_min=t_step, build=t_build, eig=t_eig, threads=A.threads)
    log(f"time: {out}")
    os.makedirs(os.path.join(HERE, "results_probe"), exist_ok=True)
    dump(os.path.join(HERE, "results_probe/time.json"), out)


def pretrain(ctx):
    fn = os.path.join(RES, f"stageA_seed{A.seed}.json")
    if os.path.exists(fn):
        log(f"{fn} exists; skip")
        return
    t0 = cpu()
    th, hist = adam_run(ctx, A.seed, A.nA, t_report=500)
    E = ctx.energy(th)[0]
    np.save(os.path.join(RES, f"thetaA_seed{A.seed}.npy"), th.numpy())
    dump(fn, dict(seed=A.seed, nA=A.nA, E=E, eps=(E - E0) / abs(E0), hist=hist, cpu_min=cpu() - t0))
    log(f"stage A seed {A.seed}: E {E:.10f} eps {(E - E0) / abs(E0):.5e} [{cpu() - t0:.1f} CPU-min]")


def polish(ctx):
    t0 = time.time()
    fn, fth = os.path.join(RES, "state_tree.json"), os.path.join(RES, "theta_tree.npy")
    seeds = []
    for s in range(A.seeds):
        f = os.path.join(RES, f"stageA_seed{s}.json")
        assert os.path.exists(f), f"stage A seed {s} missing"
        seeds.append(json.load(open(f)))
    best = min(seeds, key=lambda d: d["E"])
    th = torch.tensor(np.load(os.path.join(RES, f"thetaA_seed{best['seed']}.npy")))
    E = ctx.energy(th)[0]
    g = E - E0
    st = dict(arm="tree", seed=best["seed"], stageA=[dict(seed=d["seed"], eps=d["eps"]) for d in seeds], g=g,
              P=ctx.P, ladder=LADDER, tau=TAU, min_gain=MIN_GAIN, stop_frac=STOP_FRAC, cap=A.steps, steps=[],
              stop=None, E_start=E)
    if os.path.exists(fn):
        old = json.load(open(fn))
        st.update(steps=old["steps"], stop=old["stop"], E_start=old["E_start"], g=old["g"])
        g = st["g"]
        if st["stop"]:
            log(f"tree: already finished ({st['stop']})")
            return
        if old["steps"]:
            th = torch.tensor(np.load(fth))
            E = ctx.energy(th)[0]
    log(f"tree polish from seed {st['seed']}: eps {(E - E0) / abs(E0):.5e}, g {g:.5e} [{cpu():.2f} CPU-min]")
    Pn, per = ctx.P, None
    while st["stop"] is None:
        if len(st["steps"]) >= A.steps:
            st["stop"] = "STOP-CAP"
            break
        s0 = cpu()
        if per is not None and s0 + per > A.cpu_cap:
            log(f"stop: next step would exceed the CPU cap ({s0:.1f} + {per:.1f} > {A.cpu_cap}); resumable")
            break
        Eb, Y, HY = ctx.build(th)
        assert abs(Eb - E) < 1e-9, (Eb, E)
        S = Y.T @ Y
        Hm = Y.T @ HY
        del Y, HY
        S, Hm = 0.5 * (S + S.T), 0.5 * (Hm + Hm.T)
        tan = R.Tan(S, Hm, E, list(range(Pn)))
        del S, Hm
        rungs, best, prev = [], None, None
        for c in LADDER:
            r = tan.reduce(c)
            if r["rank"] == prev:
                rungs.append(dict(cut=c, rank=r["rank"], skipped=True))
                continue
            prev = r["rank"]
            z, lam = tan.candidate(r, "lm", TAU)
            d = tan.dtheta(r, z, Pn)
            acc, trials = None, []
            for s in (1.0, 0.5, 0.25):
                Es = ctx.energy(th + s * torch.tensor(d))[0]
                trials.append((s, (Es - E0) / abs(E0)))
                if np.isfinite(Es) and (E - Es) / abs(E0) > MIN_GAIN:
                    acc = (s, Es)
                    break
            row = dict(cut=c, rank=r["rank"], zn=float(np.linalg.norm(z)), lam=lam, pred_full=E - tan.pred(r, z),
                       trials=trials, s=None)
            if acc:
                row.update(s=acc[0], E_new=acc[1], pred=E - tan.pred(r, acc[0] * z))
                if best is None or acc[1] < best["E_new"]:
                    best = dict(row, d=d)
            rungs.append(row)
        rec = dict(E=E, eps=(E - E0) / abs(E0), rungs=rungs, cpu_min=None)
        if best is None:
            rec["accepted"] = False
            st["steps"].append(rec)
            st["stop"] = "STOP-NOGAIN"
            log(f"step {len(st['steps'])}: no rung lowers eps by > {MIN_GAIN:g}; STOP-NOGAIN")
            break
        th = th + best["s"] * torch.tensor(best["d"])
        E = best["E_new"]
        per = cpu() - s0
        rec.update(accepted=True, cut=best["cut"], s=best["s"], E_new=E, eps_new=(E - E0) / abs(E0),
                   gain=rec["E"] - E, pred=best["pred"], pred_over_g=best["pred"] / g, cpu_min=per)
        st["steps"].append(rec)
        np.save(fth, th.numpy())
        if best["pred"] < STOP_FRAC * g:
            st["stop"] = "STOP-RULE"
        log(f"step {len(st['steps'])}: cut {best['cut']:.0e} s {best['s']} eps {rec['eps']:.5e} -> "
            f"{rec['eps_new']:.5e} (drop {100 * (1 - rec['eps_new'] / rec['eps']):.2f}%); predicted gain "
            f"{rec['pred_over_g']:.2e} g; rungs " +
            " ".join(f"{x['cut']:.0e}:{'skip' if x.get('skipped') else x['s']}" for x in rungs) +
            f" [{cpu():.1f} CPU-min, step {per:.2f}, wall {time.time() - t0:.0f} s]{' ' + st['stop'] if st['stop'] else ''}")
        st["cpu_min_total"] = cpu()
        dump(fn, st)
    if not st["steps"] or st["steps"][-1].get("accepted") is not False:
        if not os.path.exists(fth):
            np.save(fth, th.numpy())
    st["cpu_min_total"] = cpu()
    dump(fn, st)
    log(f"tree: stop {st['stop']}; {len(st['steps'])} steps [{cpu():.1f} CPU-min, {time.time() - t0:.0f} s wall]")


def ground(ctx):
    """exact (+,+) ground state at the representatives (normalised in the orbit-weighted norm)."""
    from scipy.sparse.linalg import eigsh
    W = sp.diags(ctx.sw)
    Wi = sp.diags(1 / ctx.sw)
    Hs = (W @ ctx.Hr @ Wi).tocsr()
    Hs = 0.5 * (Hs + Hs.T)
    w, v = eigsh(Hs, k=1, which="SA", tol=1e-14)
    g0 = v[:, 0] / ctx.sw
    return float(w[0]), g0 / np.sqrt(np.sum(ctx.size * g0 * g0))


def eps_of(ctx, v):
    v = v / np.sqrt(np.sum(ctx.size * v * v))
    return (float(np.sum(ctx.size * v * (ctx.Hr @ v))) - E0) / abs(E0)


def sign(ctx):
    th = torch.tensor(np.load(os.path.join(RES, "theta_tree.npy")))
    E, pr, _ = ctx.energy(th)
    Eg, g0 = ground(ctx)
    ov = np.sum(ctx.size * g0 * pr)
    if ov < 0:
        g0 = -g0
    eps_t = (E - E0) / abs(E0)
    w0 = ctx.size * g0 * g0
    wrong = np.sign(pr) != np.sign(g0)
    # charge-only sign: P_G[D] with the endpoint D
    p = ctx.unpack(th)
    with torch.no_grad():
        sD, lD = torch.linalg.slogdet(p["Phi"][ctx.slots])
        prD = ctx.project(sD * torch.exp(lD - lD.max())).numpy()
    out = dict(E_ground=Eg, ground_vs_E0=abs(Eg - E0), overlap=float(abs(ov)), eps_tree=eps_t,
               eps_signfix=eps_of(ctx, np.abs(pr) * np.sign(g0)),
               eps_ampfix=eps_of(ctx, np.abs(g0) * np.sign(pr)),
               w_wrong=float(w0[wrong].sum()),
               eps_allplus=eps_of(ctx, np.abs(g0)),
               eps_chargesign=eps_of(ctx, np.abs(g0) * np.sign(prD)),
               w_wrong_chargesign=float(w0[np.sign(prD) != np.sign(g0)].sum()))
    r = out["eps_signfix"] / eps_t
    out["signfix_over_tree"] = r
    out["class"] = "SIGN-DOMINATED" if r <= 0.5 else "NOT-SIGN-LIMITED" if r >= 0.8 else "MIXED"
    for k, v in out.items():
        log(f"sign: {k} = {v}")
    dump(os.path.join(RES, "sign.json"), out)


def endpoint(st):
    acc = [s for s in st["steps"] if s.get("accepted")]
    E = acc[-1]["E_new"] if acc else st["E_start"]
    eps = (E - E0) / abs(E0)
    tail = 0.0
    if st["stop"] == "STOP-CAP":
        gains = [s["gain"] for s in acc]
        r = gains[-1] / gains[-2] if len(gains) >= 2 and gains[-2] > 0 else np.inf
        tail = gains[-1] * r / (1 - r) if r < 1 else np.inf
    return eps, max(eps - tail / abs(E0), 0.0), len(acc)


def cls(rho):
    return "PASS-T1" if rho <= PASS else "KILL (G2)" if rho >= KILL else "AMBIGUOUS"


def verdict(_ctx=None):
    ft = os.path.join(RES, "state_tree.json")
    fc = os.path.join(TESTS, "P4.1_cheap_su2_r22/results/state_che.json")
    out = {}
    if not (os.path.exists(ft) and os.path.exists(fc)):
        out["verdict"] = "INCOMPLETE (missing state file)"
    else:
        st, sc = json.load(open(ft)), json.load(open(fc))
        if st["stop"] is None or sc["stop"] is None:
            out["verdict"] = "INCOMPLETE"
        elif sc["stop"] == "INVALID" or not sc["checks"].get("valid"):
            out["verdict"] = "INVALID (r22 che)"
        else:
            et, eti, nt = endpoint(st)
            ec, eci, nc = endpoint(sc)
            rho = et / ec
            rho_i = eti / eci if eci > 0 else float("inf")
            c1, c2 = cls(rho), cls(rho_i)
            out.update(eps_tree=et, eps_tree_inf=eti, steps_tree=nt, stop_tree=st["stop"], eps_che=ec,
                       eps_che_inf=eci, steps_che=nc, stop_che=sc["stop"], rho=rho, rho_inf=rho_i, cls_end=c1,
                       cls_inf=c2, rho_vs_r19_che=et / 1.217e-3, info_default_0p85=("KILL" if rho >= 0.85 else "-"),
                       verdict=c1 if c1 == c2 else "UNSETTLED")
    sf = os.path.join(RES, "sign.json")
    if os.path.exists(sf):
        out["sign_class"] = json.load(open(sf))["class"]
    for k, v in out.items():
        print(f"{k}: {v}")
    dump(os.path.join(RES, "verdict.json"), out)


if __name__ == "__main__":
    if A.mode == "verdict":
        verdict()
    else:
        ctx = Ctx()
        {"check": check, "time": time_mode, "pretrain": pretrain, "polish": polish, "sign": sign}[A.mode](ctx)
