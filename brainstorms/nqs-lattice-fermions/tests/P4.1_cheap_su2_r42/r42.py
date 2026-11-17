#!/usr/bin/env python3
"""
P4.1 R42 `practitioner-24`: charge-only exactly projected `che` vs spin-reading unprojected `nnbf`, 24 sites,
same sampled optimiser (guided SR, the R38 sampling ported to Metropolis), equal CPU time per arm.  Rule: RULE.md.
==================================================================================================================

Run with the lit_env interpreter, ONE thread (set below, before numpy/torch are imported; L63):
    ~/.conda/envs/lit_env/bin/python r42.py --mode MODE [options]

Imports, read only, ../P4.1_cheap_su2_scaling/vmc.py (Model: geometry, P_G wrapper, che closed-form P_S, nnbf
spin-resolved backflow, Metropolis proposals, local energy, log-derivatives).  vmc.py is not modified.

Arms (both vmc.Model at H = 16 on the open 8x3 strip, N_e = 20, U = 8, t' = -0.25, P_G (+,+) projection, P = 9711):
  che   reads charges (doublon, holon); exact S = 0 by Che's closed form.
  nnbf  reads spins (n_up, n_dn); S^z = 0 only, NO spin projection (exact P_S costs the spin-orbit sum at 24 sites);
        same P_G (+,+) projection as che; its <S^2> is measured at the end.

Optimiser (identical for both arms):
  guided sample-space SR.  Chains run Metropolis on q(x) = |psi(x)|^2 + lam |psi(x)| (vmc.py proposals, symmetric),
  i.e. the R38 mixture 0.8 p + 0.2 p_flat without needing normalisations: lam is adapted after every step so that the
  self-normalised estimate of the |psi|^2 mass fraction, mean_q[ |psi|^2 / q ], is 0.8 (log-lam += 2 (F - 0.8)).
  Weights w = (|psi|^2/q) / sum; weighted centred O and E_loc rows go into vmc.py's Woodbury SR step
  (diag shift 1e-3, max step 0.5).  Step length: lr 0.05 for the first half of the CPU budget, then cosine to 0.005
  at the budget's end (vmc.py's schedule, written in CPU fraction instead of step count, so both arms get the same
  schedule in CPU time).  Polyak average of the parameters over the last 25% of the CPU budget.
  --guided 0 gives plain |psi|^2 sampling (vmc.py SR), used only for the probe.

Final scoring: the Polyak theta, plain |psi|^2 Metropolis, --n-eval samples, chain-blocked error; <S^2> by the local
estimator of S^- S^+ (S^z = 0) on --n-s2 samples (che: a code check, must be 0 per sample to 1e-8).

Modes
  check   (a) check-lx x 3 (default 3x3, N_e 6, closed shell): guided-weighted energy estimate of the init state equals the exact energy within 4 sigma;
          (b) same cluster: the S^2 local estimator averaged exactly over |psi|^2 equals sec.S2 for nnbf at a random theta;
          (c) 8x3 che: S^2_loc = 0 per sample to 1e-8.
  probe   timing: --probe-steps guided and plain steps per arm at 8x3 (1 thread), CPU-min per step.
  train   one (--arm, --seed) to --cpu-min CPU minutes; writes results/<arm>_seed<s>.json (+ theta .npy);
          checkpoint every 25 steps, resumes.  Fail-fast at --early-cpu (20) CPU-min of training: median F outside
          [0.7, 0.9], median ESS < 512 or a non-finite step energy writes <tag>_INVALID-EARLY.json and exits 3.
          The JSON keeps the guided energy of every step (energy_per_step) for the energy-vs-step comparison.
"""
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k_] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
SCAL = os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_scaling")
sys.path.insert(0, SCAL)
import vmc as V  # noqa: E402  (read only)

torch.set_default_dtype(torch.float64)
F_TARGET = 0.8


def cpu_min():
    return time.process_time() / 60.0


def model(arm, Lx, H=16, Ne=None):
    Ne = V.default_ne(Lx * 3) if Ne is None else Ne
    return V.Model(arm, Lx, 3, Ne, H, -0.25, 8.0, pg=(1, 1))


# ------------------------------------------------------------------ guided Metropolis
def logq(l, loglam):
    # log(|psi|^2 + lam |psi|) = 2 l + softplus(loglam - l)
    return 2.0 * l + torch.nn.functional.softplus(loglam - l)


def mcmc(m, th, nU, nD, la, sa, nprop, gen, chunk, loglam):
    """vmc.Model.mcmc with the acceptance on q; loglam = None gives |psi|^2."""
    B = nU.shape[0]
    nb = len(m.bi)
    acc = tried = 0
    for _ in range(nprop):
        k = torch.randint(nb, (B,), generator=gen)
        typ = torch.rand(B, generator=gen)
        i, j = m.bi[k], m.bj[k]
        r = torch.arange(B)
        ui, uj, di, dj = nU[r, i], nU[r, j], nD[r, i], nD[r, j]
        hu = (typ < 0.4) & (ui != uj)
        hd = (typ >= 0.4) & (typ < 0.8) & (di != dj)
        ex = (typ >= 0.8) & (((ui == 1) & (di == 0) & (dj == 1) & (uj == 0)) |
                             ((di == 1) & (ui == 0) & (uj == 1) & (dj == 0)))
        mu, md = hu | ex, hd | ex
        idx = torch.nonzero(mu | md)[:, 0]
        if len(idx) == 0:
            continue
        nU2, nD2 = nU[idx].clone(), nD[idx].clone()
        ri = torch.arange(len(idx))
        ii, jj = i[idx], j[idx]
        su, sd = mu[idx], md[idx]
        tu = nU2[ri, ii].clone()
        nU2[ri[su], ii[su]] = nU2[ri[su], jj[su]]
        nU2[ri[su], jj[su]] = tu[su]
        td = nD2[ri, ii].clone()
        nD2[ri[sd], ii[sd]] = nD2[ri[sd], jj[sd]]
        nD2[ri[sd], jj[sd]] = td[sd]
        l2, s2 = m.logpsi_batched(th, nU2, nD2, chunk)
        if loglam is None:
            lr_ = 2.0 * (l2 - la[idx])
        else:
            lr_ = logq(l2, loglam) - logq(la[idx], loglam)
        ok = torch.rand(len(idx), generator=gen) < torch.exp(lr_)
        sel = idx[ok]
        nU[sel], nD[sel], la[sel], sa[sel] = nU2[ok], nD2[ok], l2[ok], s2[ok]
        acc += int(ok.sum())
        tried += len(idx)
    return acc / max(tried, 1)


def pweights(la, loglam):
    if loglam is None:
        return torch.full_like(la, 1.0 / len(la)), 1.0
    r = torch.exp(2.0 * la - logq(la, loglam))       # |psi|^2 / q in (0, 1]
    return r / r.sum(), float(r.mean())


# ------------------------------------------------------------------ S^2 local estimator (S^z = 0: S^2 = S^- S^+)
def jw_sign(occ_modes, p):
    """(-1)^(# occupied modes with index < p); occ_modes (B, 2L) 0/1, p (B,) long."""
    cs = torch.cumsum(occ_modes, 1) - occ_modes
    return 1.0 - 2.0 * torch.remainder(cs.gather(1, p[:, None])[:, 0], 2)


def s2_local(m, th, nU, nD, la, sa, chunk):
    """sum_x' <x|S^- S^+|x'> psi(x')/psi(x); mode order (up sites, then down sites), as vmc.py's basis."""
    L = m.L
    B = nU.shape[0]
    out = (nD * (1.0 - nU)).sum(1)                    # i = j term: n_i,dn (1 - n_i,up)
    sup = (nU * (1 - nD)) > 0.5                        # singly occupied up (site i of c_{i up})
    sdn = (nD * (1 - nU)) > 0.5                        # singly occupied down (site j of c_{j dn})
    R = torch.nonzero(sup[:, :, None] & sdn[:, None, :])   # (b, i, j)
    if len(R) == 0:
        return out
    b, i, j = R[:, 0], R[:, 1], R[:, 2]
    occ = torch.cat([nU[b], nD[b]], 1).clone()
    sgn = torch.ones(len(b))
    # S^- S^+ = sum_ij c+_{i dn} c_{i up} c+_{j up} c_{j dn}: acting on |x>, rightmost first.
    # Hermitian, so the local estimator sums psi(x')/psi(x) over x' = (term)|x> with the matrix element.
    for op, mode in (("a", L + j), ("c", j), ("a", i), ("c", L + i)):
        sgn = sgn * jw_sign(occ, mode)
        r = torch.arange(len(b))
        occ[r, mode] = 0.0 if op == "a" else 1.0
    l2, s2 = m.logpsi_batched(th, occ[:, :L].contiguous(), occ[:, L:].contiguous(), chunk)
    val = sgn * s2 * sa[b] * torch.exp(l2 - la[b])
    val = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
    return out.index_add(0, b, val)


# ------------------------------------------------------------------ driver
class Run:
    def __init__(self, m, a, seed, guided):
        self.m, self.a, self.guided = m, a, guided
        self.gen = torch.Generator().manual_seed(1000 + seed)
        self.nprop = int(math.ceil(a.sweeps_per_step * m.Ne))
        self.loglam = None

    def start(self, th):
        m, a = self.m, self.a
        self.nU, self.nD = m.random_configs(a.n_chains, self.gen)
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        bad = ~torch.isfinite(self.la)
        while bad.any():
            u, d = m.random_configs(int(bad.sum()), self.gen)
            self.nU[bad], self.nD[bad] = u, d
            self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
            bad = ~torch.isfinite(self.la)
        mcmc(m, th, self.nU, self.nD, self.la, self.sa, a.therm_sweeps * m.Ne, self.gen, a.chunk, None)
        if self.guided:
            self.loglam = torch.tensor(float(torch.median(self.la)) + math.log(0.25))

    def step(self, th, lr):
        m, a = self.m, self.a
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        acc = mcmc(m, th, self.nU, self.nD, self.la, self.sa, self.nprop, self.gen, a.chunk, self.loglam)
        El = m.local_energy(th, self.nU, self.nD, self.la, self.sa, a.chunk)
        O = m.logderiv(th, self.nU, self.nD, a.grad_chunk)
        w, F = pweights(self.la, self.loglam)
        sw = torch.sqrt(w)
        Ob = sw[:, None] * (O - w @ O)
        eb = sw * (El - w @ El)
        Tm = Ob @ Ob.T
        Tm.diagonal().add_(a.diag_shift)
        upd = lr * (Ob.T @ torch.linalg.solve(Tm, eb))
        nrm = float(torch.linalg.norm(upd))
        if nrm > a.max_step:
            upd = upd * (a.max_step / nrm)
        if self.guided:
            self.loglam = self.loglam + 2.0 * (F - F_TARGET)
        ess = float(1.0 / (w * w).sum())
        return th - upd, dict(E=float(w @ El), acc=acc, upd=nrm, F=F, ess=ess)

    def evaluate(self, th, n_eval, n_s2):
        m, a = self.m, self.a
        self.la, self.sa = m.logpsi_batched(th, self.nU, self.nD, a.chunk)
        mcmc(m, th, self.nU, self.nD, self.la, self.sa, 10 * m.Ne, self.gen, a.chunk, None)
        Es, S2s = [], []
        for k in range(max(1, n_eval // a.n_chains)):
            mcmc(m, th, self.nU, self.nD, self.la, self.sa, 2 * m.Ne, self.gen, a.chunk, None)
            Es.append(m.local_energy(th, self.nU, self.nD, self.la, self.sa, a.chunk))
            if k < max(1, n_s2 // a.n_chains):
                S2s.append(s2_local(m, th, self.nU, self.nD, self.la, self.sa, a.chunk))
        Es, S2s = torch.stack(Es), torch.stack(S2s)
        ce, cs = Es.mean(0), S2s.mean(0)
        return dict(E=float(Es.mean()), dE=float(ce.std() / math.sqrt(len(ce))), n_eval=int(Es.numel()),
                    S2=float(S2s.mean()), dS2=float(cs.std() / math.sqrt(len(cs))), n_s2=int(S2s.numel()),
                    S2_absmax=float(S2s.abs().max()))


def lr_at(a, frac):
    if frac <= 0.5:
        return a.lr
    x = min(1.0, (frac - 0.5) / 0.5)
    return a.lr * (a.lr_decay + (1 - a.lr_decay) * 0.5 * (1 + math.cos(math.pi * x)))


def train(a):
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.arm}_seed{a.seed}"
    out = os.path.join(a.outdir, tag + ".json")
    if os.path.exists(out):
        print("done already:", out)
        return
    m = model(a.arm, a.Lx)
    th = m.init(a.seed)
    run = Run(m, a, a.seed, bool(a.guided))
    ck = os.path.join(a.outdir, tag + "_ckpt.npz")
    cpu_used, it, traj, avg, navg = 0.0, 0, [], None, 0
    hist = dict(E=[], F=[], ess=[])          # every step (fail-fast check, RULE.md INVALID-EARLY)
    early_done = False
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        th = torch.tensor(z["th"])
        cpu_used, it = float(z["cpu"]), int(z["it"])
        traj = json.loads(str(z["traj"]))
        navg = int(z["navg"])
        if "hist" in z.files:
            hist = json.loads(str(z["hist"]))
            early_done = bool(z["early_done"])
        avg = torch.tensor(z["avg"]) if navg else None
    c0 = cpu_min()
    run.start(th)
    if os.path.exists(ck) and "loglam" in z.files and run.guided:
        run.loglam = torch.tensor(float(z["loglam"]))
    while True:
        used = cpu_used + cpu_min() - c0
        frac = used / a.cpu_min
        if frac >= 1.0:
            break
        half = os.path.join(a.outdir, tag + "_theta_half.npy")
        if frac >= 0.5 and not os.path.exists(half):
            np.save(half, th.numpy())          # RULE.md half-budget read (raw theta, scored after training)
        if not early_done and used >= a.early_cpu:
            early_done = True
            bad = []
            if hist["F"] and not (0.7 <= float(np.median(hist["F"])) <= 0.9):
                bad.append(f"median F {np.median(hist['F']):.3f} outside [0.7, 0.9]")
            if hist["ess"] and float(np.median(hist["ess"])) < 512:
                bad.append(f"median ESS {np.median(hist['ess']):.0f} < 512")
            if not all(math.isfinite(e) for e in hist["E"]):
                bad.append("non-finite energy")
            if bad:
                res = dict(tag=tag, verdict="INVALID-EARLY", reasons=bad, steps=it, cpu=used, traj=traj)
                json.dump(res, open(os.path.join(a.outdir, tag + "_INVALID-EARLY.json"), "w"), indent=1)
                json.dump(res, open(out, "w"), indent=1)
                print("INVALID-EARLY", bad, flush=True)
                sys.exit(3)
            print(f"fail-fast check passed at {used:.1f} CPU-min, step {it}", flush=True)
        th, info = run.step(th, lr_at(a, frac))
        it += 1
        hist["E"].append(info["E"])
        hist["F"].append(info["F"])
        hist["ess"].append(info["ess"])
        if frac >= 1.0 - a.avg_frac:
            avg = th.clone() if avg is None else avg + th
            navg += 1
        if it % 10 == 0:
            info.update(it=it, cpu=cpu_used + cpu_min() - c0)
            traj.append(info)
            print(json.dumps(info), flush=True)
        if it % 25 == 0:
            np.savez(ck, th=th.numpy(), cpu=cpu_used + cpu_min() - c0, it=it, traj=json.dumps(traj), navg=navg,
                     avg=(avg.numpy() if avg is not None else np.zeros(1)),
                     loglam=float(run.loglam) if run.loglam is not None else 0.0,
                     hist=json.dumps(hist), early_done=early_done)
    th_f = avg / navg if navg else th
    t_train = cpu_used + cpu_min() - c0
    evh = run.evaluate(torch.tensor(np.load(os.path.join(a.outdir, tag + "_theta_half.npy"))), a.n_half, 1024)
    ev = run.evaluate(th_f, a.n_eval, a.n_s2)
    ev["half"] = evh
    res = dict(tag=tag, energy_per_step=hist["E"], arm=a.arm, seed=a.seed, Lx=a.Lx, nparam=m.nparam, guided=a.guided, steps=it,
               cpu_train=t_train, cpu_total=cpu_used + cpu_min() - c0, cpu_budget=a.cpu_min, navg=navg,
               traj=traj, **ev)
    np.save(os.path.join(a.outdir, tag + "_theta.npy"), th_f.numpy())
    json.dump(res, open(out, "w"), indent=1)
    print("FINAL", json.dumps({k: v for k, v in res.items() if k != "traj"}), flush=True)


def check(a):
    import r2_micro_test as R  # noqa: F401  (through vmc.exact_sector)
    # (a) guided-weighted energy at 4x3 init vs exact
    for arm in ("che", "nnbf"):
        m = model(arm, a.check_lx, Ne=6)
        th = m.init(0)
        th = th + 0.05 * torch.randn(th.shape, generator=torch.Generator().manual_seed(3))
        sec = V.exact_sector(m)
        Eex, psi = V.exact_energy(m, th, sec)
        run = Run(m, a, 0, True)
        run.start(th)
        Es = []
        for _ in range(8):
            run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, a.chunk)
            mcmc(m, th, run.nU, run.nD, run.la, run.sa, 2 * m.Ne, run.gen, a.chunk, run.loglam)
            El = m.local_energy(th, run.nU, run.nD, run.la, run.sa, a.chunk)
            w, F = pweights(run.la, run.loglam)
            Es.append(float(w @ El))
            run.loglam = run.loglam + 2.0 * (F - F_TARGET)
        Es = np.array(Es[2:])
        z = (Es.mean() - Eex) / (Es.std(ddof=1) / math.sqrt(len(Es)))
        print(f"(a) {arm}: guided E {Es.mean():.5f} +- {Es.std(ddof=1)/math.sqrt(len(Es)):.5f}, exact {Eex:.5f}, "
              f"z {z:+.2f}, F last {F:.3f} [{'PASS' if abs(z) < 4 else 'FAIL'}]", flush=True)
        # (b) S^2 estimator, exact average over |psi|^2
        nU, nD = V.exact_configs(sec)
        l, s = m.logpsi_batched(th, nU, nD, a.chunk)
        p = torch.exp(2 * (l - l.max()))
        p = p / p.sum()
        keep = p > 0
        s2l = s2_local(m, th, nU[keep], nD[keep], l[keep], s[keep], a.chunk)
        S2est = float(p[keep] @ s2l)
        S2ex = float(np.sum(psi * sec.S2(psi)))
        print(f"(b) {arm}: S2 estimator {S2est:.10f}, exact {S2ex:.10f} "
              f"[{'PASS' if abs(S2est - S2ex) < 1e-8 else 'FAIL'}]", flush=True)
    # (c) 8x3 che: S2_loc = 0 per sample
    m = model("che", 8)
    th = m.init(0)
    th = th + 0.02 * torch.randn(th.shape, generator=torch.Generator().manual_seed(4))
    a2 = argparse.Namespace(**vars(a))
    a2.n_chains = 64
    run = Run(m, a2, 0, False)
    run.start(th)
    s2l = s2_local(m, th, run.nU, run.nD, run.la, run.sa, a.chunk)
    print(f"(c) che 8x3: max |S2_loc| over 64 samples {float(s2l.abs().max()):.2e} "
          f"[{'PASS' if float(s2l.abs().max()) < 1e-8 else 'FAIL'}]", flush=True)


def probe(a):
    res = {}
    for arm in a.probe_arms.split(","):
        m = model(arm, a.Lx)
        th = m.init(0)
        for guided in (1, 0):
            run = Run(m, a, 0, bool(guided))
            c0 = cpu_min()
            run.start(th)
            c1 = cpu_min()
            t = th.clone()
            ts = []
            for _ in range(a.probe_steps):
                s0 = cpu_min()
                t, info = run.step(t, a.lr)
                ts.append(cpu_min() - s0)
            ev0 = cpu_min()
            run.la, run.sa = m.logpsi_batched(t, run.nU, run.nD, a.chunk)   # amplitudes at the updated theta
            s2 = s2_local(m, t, run.nU[:128], run.nD[:128], run.la[:128], run.sa[:128], a.chunk)
            el = m.local_energy(t, run.nU[:128], run.nD[:128], run.la[:128], run.sa[:128], a.chunk)
            ev1 = cpu_min()
            res[f"{arm}_g{guided}"] = dict(start_cpu=c1 - c0, step_cpu=ts, step_cpu_med=float(np.median(ts)),
                                           last=info, s2_plus_eloc_128=ev1 - ev0,
                                           s2_mean_128=float(s2.mean()), el_mean_128=float(el.mean()))
            print(arm, guided, json.dumps(res[f"{arm}_g{guided}"]), flush=True)
            if guided == 1 and not a.probe_plain:
                break
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "logs", "probe.json"), "w"), indent=1)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["check", "probe", "train"])
    p.add_argument("--arm", choices=["che", "nnbf"], default="che")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--Lx", type=int, default=8)
    p.add_argument("--guided", type=int, default=1)
    p.add_argument("--cpu-min", type=float, default=300.0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-decay", type=float, default=0.1)
    p.add_argument("--avg-frac", type=float, default=0.25)
    p.add_argument("--diag-shift", type=float, default=1e-3)
    p.add_argument("--max-step", type=float, default=0.5)
    p.add_argument("--n-chains", type=int, default=1024)
    p.add_argument("--sweeps-per-step", type=float, default=0.5)
    p.add_argument("--therm-sweeps", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=131072)
    p.add_argument("--n-s2", type=int, default=32768)
    p.add_argument("--n-half", type=int, default=32768)
    p.add_argument("--early-cpu", type=float, default=20.0)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--grad-chunk", type=int, default=256)
    p.add_argument("--check-lx", type=int, default=3)
    p.add_argument("--probe-steps", type=int, default=3)
    p.add_argument("--probe-arms", default="che,nnbf")
    p.add_argument("--probe-plain", type=int, default=0)
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse()
    {"check": check, "probe": probe, "train": train}[a.mode](a)
