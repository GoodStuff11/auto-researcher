#!/usr/bin/env python3
"""
P5.1 round 4, route R51.8 `nscale`: how do the samples per step that fixed-length guided SR needs grow with size?
=================================================================================================================
Rule: RULE.md (pre-registered).  Run from this directory with the lit_env interpreter:
    ~/.conda/envs/lit_env/bin/python ns_run.py --Lx LX --Ly LY --Ne NE [options]

Same step as ../P5.1_noise_floor_r01/nf_run.py (R51.1), generalised to any open Lx x Ly strip with the (+,+)
point-group projection and the `che` network (width 16): draw N configurations from q = 0.8 p + 0.2 p_flat
(p ~ |psi|^2, p_flat ~ |psi|; or q = p with --sampler born), form the self-normalised S_T and f from the unique
draws weighted by their counts (the exact `--dedup` rewrite of R51.1), and take d = -tau x / sqrt(x^T S_T x),
x = (S_T + 1e-3 I)^{-1} f, always.  --exact replaces the draws by the exact sums over every orbit representative
(the N = infinity reference arm).  The exact energy is computed once per step (it is also the sampler's |psi|^2)
and only scores.

Options
  --Lx --Ly --Ne     lattice and electron number (U = 8, t' = -0.25, open, irrep (+,+), exact S = 0 by `che`).
  --start FILE       theta of the SR-trained start (vmc.py --mode train output, *_theta.npy).  For 4x3 the default
                     is R51.1's start (P4.1_cheap_su2_scaling/logs/pilot/pg_che/..._theta.npy).
  --nsamp INT        samples per step (ignored with --exact).
  --exact            exact-sum step (reference arm).
  --sampler {guided, born}
  --tau FLOAT        metric length (default 0.005, R51.1).    --steps INT (default 120).
  --seed0 INT        step n draws with seed seed0 + n (default 9000).
  --cpu-cap FLOAT    CPU-minutes; stop before a step that would exceed it.
  --res DIR          results dir (default results/S<Lx>x<Ly>_N<Ne>_<sampler_N<nsamp>|exact>_tau<tau>_s<seed0>).
  --info             print sector size, orbit representatives, P, lowest (+,+) energy, E(start), the effective
                     support exp(H[p]) of |psi|^2 over representatives, unique-draw counts and timings; then exit.
Output: <res>/run.json (per step E, E - E_start, n_unique, CPU; status DONE | TRUNCATED | RUNNING) and
<res>/theta.npy.  Theta and history are also saved together, atomically, in <res>/state.npz after every step;
rerunning the same command resumes from it (same seeds: step n always draws with seed0 + n).  The run stops
cleanly, saving, before a step that would exceed --cpu-cap, or after the current step on SIGUSR1 / SIGTERM
(sbatch scripts use `#SBATCH --signal=B:USR1@600` and run python in the foreground via `exec` or a trap).
Reuses read only: ../P4.1_cheap_su2_scaling/vmc.py (Model, exact_sector), ../P4.1_cheap_su2_r10/tangent.py
(pg_maps), ../P4.1_cheap_su2_r19/polish.py (logder), ../P4.1_cheap_su2_r24/trsr.py (orows, d_shift).
"""
import argparse
import json
import os
import signal
import sys
import time

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r24"))
import trsr as T  # noqa: E402  (read only; imports polish as T.P, realise as T.P.R)
import torch  # noqa: E402

P = T.P
TG = P.R.TG
V = TG.V
torch.set_num_threads(1)
START12 = os.path.join(TESTS, "P4.1_cheap_su2_scaling/logs/pilot/pg_che/S4x3_N10_U8_tp-0.25_pg11_che_seed0_theta.npy")


def cpu_min():
    return time.process_time() / 60.0


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


class GCtx:
    """orbit-reduced (+,+) sector of an open Lx x Ly strip (generalises P4.1 r11 realise.Ctx)."""

    def __init__(self, Lx, Ly, Ne, start):
        self.m = V.Model("che", Lx, Ly, Ne, 16, -0.25, 8.0, (1, 1))
        self.sec = V.exact_sector(self.m)
        self.nU, self.nD = V.exact_configs(self.sec)
        maps = TG.pg_maps(self.m, self.sec)
        self.idx = np.stack([mp[0] for mp in maps])
        self.sg = np.stack([mp[1] for mp in maps])
        dim = self.sec.dim
        gstar = np.argmin(self.idx, 0)
        rep_of = self.idx[gstar, np.arange(dim)]
        self.sigma = self.sg[gstar, np.arange(dim)]
        self.reps = np.unique(rep_of)
        pos = np.full(dim, -1)
        pos[self.reps] = np.arange(len(self.reps))
        self.pos_of = pos[rep_of]
        srt = np.sort(self.idx[:, self.reps], 0)
        self.size = 1.0 + (np.diff(srt, axis=0) != 0).sum(0)
        self.sw = np.sqrt(self.size)
        E = sp.csr_matrix((self.sigma, (np.arange(dim), self.pos_of)), shape=(dim, len(self.reps)))
        self.Hr = (self.sec.Hsparse()[self.reps, :] @ E).tocsr()
        self.th0 = torch.tensor(np.load(start))
        assert self.th0.numel() == self.m.nparam, (self.th0.numel(), self.m.nparam)
        log(f"{Lx}x{Ly} N_e {Ne}: sector {dim}, {len(self.reps)} representatives, P {self.m.nparam}, "
            f"Hr nnz {self.Hr.nnz}")

    def e0(self):
        """lowest (+,+) eigenvalue (any total spin; reported only, the rule does not use it)."""
        from scipy.sparse.linalg import eigsh
        K = sp.diags(self.sw) @ self.Hr @ sp.diags(1.0 / self.sw)
        K = 0.5 * (K + K.T)
        return float(eigsh(K, k=1, which="SA", tol=1e-10)[0][0])

    def energy(self, th):
        ls, ss = [], []
        with torch.no_grad():
            for s0 in range(0, self.sec.dim, 8192):
                l, s = self.m.logpsi0(th, self.nU[s0:s0 + 8192], self.nD[s0:s0 + 8192])
                ls.append(l)
                ss.append(s)
        l, s = torch.cat(ls), torch.cat(ss)
        phi0 = (s * torch.exp(l - l.max())).numpy()
        pr = 0.25 * (self.sg[:, self.reps] * phi0[self.idx[:, self.reps]]).sum(0)
        nrm = np.sqrt(np.sum(self.size * pr * pr))
        pr = pr / nrm
        return float(np.sum(self.size * pr * (self.Hr @ pr))), pr, phi0 / nrm


def stats(ctx, th, phi0, pr, N, seed, sampler, exact):
    p = ctx.size * pr * pr
    p = p / p.sum()
    if exact:
        uq = np.nonzero(p > 0)[0]  # symmetry-forbidden representatives have zero amplitude and no row
        w = p[uq] / p[uq].sum()
    else:
        if sampler == "guided":
            pf = ctx.size * np.abs(pr)
            q = 0.8 * p + 0.2 * pf / pf.sum()
        else:
            q = p
        rg = np.random.default_rng(seed).choice(len(q), size=N, p=q)
        uq, inv = np.unique(rg, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uq)).astype(float)
        w = cnt * p[uq] / q[uq]
        w = w / w.sum()
    O = T.orows(ctx, th, phi0, pr, uq)
    elg = (ctx.Hr @ pr)[uq] / pr[uq]
    Oc = O - w @ O
    ST = (Oc * w[:, None]).T @ Oc
    f = (Oc * w[:, None]).T @ (elg - w @ elg)
    return 0.5 * (ST + ST.T), f, len(uq)


def parse():
    pa = argparse.ArgumentParser()
    pa.add_argument("--Lx", type=int, required=True)
    pa.add_argument("--Ly", type=int, required=True)
    pa.add_argument("--Ne", type=int, required=True)
    pa.add_argument("--start", default=None)
    pa.add_argument("--nsamp", type=int, default=16384)
    pa.add_argument("--exact", action="store_true")
    pa.add_argument("--sampler", default="guided", choices=["guided", "born"])
    pa.add_argument("--tau", type=float, default=0.005)
    pa.add_argument("--steps", type=int, default=120)
    pa.add_argument("--seed0", type=int, default=9000)
    pa.add_argument("--cpu-cap", type=float, default=600.0)
    pa.add_argument("--res", default=None)
    pa.add_argument("--info", action="store_true")
    a = pa.parse_args()
    if a.start is None:
        if (a.Lx, a.Ly, a.Ne) != (4, 3, 10):
            pa.error("--start is required except for 4x3")
        a.start = START12
    if a.res is None:
        kind = "exact" if a.exact else f"{a.sampler}_N{a.nsamp}"
        a.res = os.path.join(HERE, "results", f"S{a.Lx}x{a.Ly}_N{a.Ne}_{kind}_tau{a.tau}_s{a.seed0}")
    return a


def info(a, ctx):
    c0 = cpu_min()
    E, pr, phi0 = ctx.energy(ctx.th0)
    t_e = cpu_min() - c0
    p = ctx.size * pr * pr
    p = p / p.sum()
    H = -np.sum(p[p > 0] * np.log(p[p > 0]))
    pf = ctx.size * np.abs(pr)
    q = 0.8 * p + 0.2 * pf / pf.sum()
    uniq = {}
    for N in (16384, 65536, 262144, 1048576):
        uniq[N] = [int(len(np.unique(np.random.default_rng(1).choice(len(x), size=N, p=x)))) for x in (q, p)]
    nr = min(4096, len(p))
    c0 = cpu_min()
    T.orows(ctx, ctx.th0, phi0, pr, np.arange(nr))
    t_row = (cpu_min() - c0) / nr
    E0 = ctx.e0() if len(p) <= 400_000 else float('nan')  # eigsh skipped above 4e5 representatives
    out = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, dim=int(ctx.sec.dim), reps=int(len(p)), P=int(ctx.m.nparam), E_start=E,
               E0_pg=E0, eps_start=(E - E0) / abs(E0), expH=float(np.exp(H)), uniq_guided_born=uniq,
               cpu_min_energy=t_e, cpu_min_per_row=t_row, cpu_min_exact_rows=t_row * len(p))
    log(json.dumps(out))
    return out


def main():
    a = parse()
    c00 = cpu_min()
    ctx = GCtx(a.Lx, a.Ly, a.Ne, a.start)
    log(f"setup {cpu_min() - c00:.2f} CPU-min")
    if a.info:
        out = info(a, ctx)
        json.dump(out, open(os.path.join(HERE, "results", f"info_S{a.Lx}x{a.Ly}_N{a.Ne}.json"), "w"), indent=1)
        return
    os.makedirs(a.res, exist_ok=True)
    t0 = time.time()
    E00, pr, phi0 = ctx.energy(ctx.th0)
    th = ctx.th0.clone()
    fn, fst = os.path.join(a.res, "run.json"), os.path.join(a.res, "state.npz")
    hist, cpu_prev = [], 0.0
    if os.path.exists(fst):  # resume (round 4 audit fix): theta and history saved together, atomically
        z = np.load(fst, allow_pickle=False)
        th = torch.tensor(z["theta"])
        hist = json.loads(str(z["hist"]))
        cpu_prev = float(z["cpu_prev"])
        log(f"resumed from {fst} at step {len(hist)} (previous runs {cpu_prev:.2f} CPU-min)")
    stop = {"why": None}

    def on_signal(signum, frame):  # sbatch --signal=B:USR1@<s> or SIGTERM: finish the current step, save, exit
        stop["why"] = f"signal {signum}"
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def save(status):
        rec = dict(Lx=a.Lx, Ly=a.Ly, Ne=a.Ne, nsamp=None if a.exact else a.nsamp, exact=a.exact,
                   sampler=a.sampler, tau=a.tau, seed0=a.seed0, start=a.start, E_start=E00, steps=hist,
                   status=status, steps_target=a.steps, cpu_min=cpu_prev + cpu_min(), wall=time.time() - t0)
        tmp = fst[:-4] + ".tmp.npz"
        np.savez(tmp, theta=th.numpy(), hist=json.dumps(hist), cpu_prev=cpu_prev + cpu_min())
        os.replace(tmp, fst)
        np.save(os.path.join(a.res, "theta.npy"), th.numpy())
        json.dump(rec, open(fn + ".tmp", "w"), indent=1)
        os.replace(fn + ".tmp", fn)

    E, pr, phi0 = ctx.energy(th)
    per = None
    status = "RUNNING"
    while len(hist) < a.steps:
        c0 = cpu_min()
        if per is not None and c0 + per > a.cpu_cap:
            stop["why"] = f"CPU cap ({c0:.1f} + {per:.2f} > {a.cpu_cap})"
        if stop["why"]:
            status = "TRUNCATED"
            log(f"stop before step {len(hist) + 1}: {stop['why']}; state saved, resume by rerunning the same command")
            break
        n = len(hist) + 1
        ST, f, nu = stats(ctx, th, phi0, pr, a.nsamp, a.seed0 + n, a.sampler, a.exact)
        w, U = np.linalg.eigh(ST)
        d = T.d_shift(w, U, f, a.tau)[0]
        th = th + torch.tensor(d)
        En, pr, phi0 = ctx.energy(th)
        rec = dict(E=En, dE=E - En, E_minus_start=En - E00, n_unique=nu, dth=float(np.linalg.norm(d)))
        E = En
        hist.append(rec)
        per = cpu_min() - c0
        rec["cpu_step"] = per
        log(f"S{a.Lx}x{a.Ly} {'exact' if a.exact else a.sampler + ' N ' + str(a.nsamp)} tau {a.tau} step {n}: "
            f"E {En:.6f} (E-E_start {En - E00:+.3e}); unique {nu} [{cpu_min():.2f} CPU-min, step {per:.3f}]")
        save("RUNNING")
    if len(hist) >= a.steps:
        status = "DONE"
    save(status)
    log(f"{status}: {len(hist)} of {a.steps} steps, {cpu_prev + cpu_min():.2f} CPU-min in all, "
        f"{time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
