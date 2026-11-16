#!/usr/bin/env python3
"""
P4.1 R2 `pcg-scal` compiled-Pfaffian re-timing (2026-11-16): verification and per-step cost against `che`.
=========================================================================================================

Run with the lit_env interpreter, on a compute node, with at most 4 threads:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 ~/.conda/envs/lit_env/bin/python retime.py --mode MODE

The script IMPORTS ../P4.1_cheap_su2_t2/vmc_t2.py (and through it the frozen ../P4.1_cheap_su2_scaling/vmc.py)
read-only; neither file is modified (jobs 686990, 698223 and 698224 use them). The only change of code path is that,
inside this process, vmc_t2.slogpf (torch Parlett-Reid) is replaced by pf_fast.slogpf_fast (the same algorithm,
compiled with numba, parallel over the batch, analytic backward). Everything else (DetModel, P_G, Metropolis,
local energy, log-derivatives, SR solve) is the vmc_t2 / vmc code path unchanged.

Modes (--mode)
--------------
  verify  (1) random antisymmetric matrices, m in {2, 4, 10, 20, 30, 40, 60}, 200 each: relative error of Pf of
              slogpf_fast against vmc_t2.slogpf (torch) and against pfapack.ctypes.pfaffian (compiled C reference);
          (2) actual R2 amplitudes: pcg (P_G (1,1)) at Lx = 4, 8, 12, 16 (Ly = 3), 256 random configurations, init
              seed 0 and seed 1: relative amplitude error |1 - psi_fast / psi_torch| (and sign agreement);
          (3) per-sample log-derivatives (the SR O matrix) of pcg at Lx = 4 and 8 with P_G: max relative difference
              between the two Pfaffians (autograd through torch elimination vs analytic backward).
          (1) and (2) also compare both float Pfaffians with an exact-arithmetic reference (mpmath, 40 digits, the
              Pfaffian of the same float matrix): random matrices (first 40 per size, 10 at m = 60) and the 3 configs of
              largest fast-vs-torch disagreement per (L, seed), whose amplitude is recomputed with exact Pfaffians.
          PASS if, in (1) and (2), the fast error against the exact reference is <= max(1e-12, 2 x the existing
          Pfaffian's error), signs agree, and (3) <= 1e-9. The raw fast-vs-torch disagreement is also printed against
          the 1e-12 target (it can exceed it only where both are rounding-limited by conditioning).
  timing  Per-SR-step wall time of che (H = 16) and pcg (H matched to che's parameter count, as vmc_t2.build) at
          Lx in --sizes, Ly = 3, N_e = vmc.default_ne(L), P_G (1,1), --n-chains chains, --threads threads (torch and
          numba). Per size: both models are built and their chains started, one warm-up step each (excluded), then
          --repeats steps alternating che, pcg (so load drift hits both arms equally); t_step = median of repeats;
          the ratio is also given as the median of the per-pair ratios. --old-sizes additionally times pcg with the
          old torch Pfaffian (consistency with logs/timing_L12_24_output.txt in ../P4.1_cheap_su2_t2).
          Also a component profile on 2048 random configurations (no P_G): che logpsi0; pcg f_c = B1 f0 B2^T;
          X(u) assembly; compiled Pfaffian. Writes <outdir>/retime_<host>_t<threads>.json.

Options
-------
  --mode {verify, timing}  (required)
  --sizes STR ("4,8,12,16")   Lx values for timing (L = 3 Lx sites)
  --old-sizes STR ("4,8")     Lx values at which pcg is also timed with the old torch Pfaffian ("" = none)
  --repeats INT (5)           timed steps per arm per size (after one warm-up step)
  --n-chains INT (32)         Markov chains (samples per step), as in the round-7 timing
  --threads INT (4)
  --outdir PATH (<script dir>/results)
"""
import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P4.1_cheap_su2_t2"))
sys.path.insert(0, HERE)

import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vmc_t2 as T  # noqa: E402  (read only)
from pf_fast import slogpf_fast  # noqa: E402

V = T.V
log = V.log
SLOGPF_TORCH = T.slogpf


def use(fast):
    T.slogpf = {True: slogpf_fast, False: SLOGPF_TORCH, "mp": slogpf_mp}[fast]


def pf_mp(A, dps=40):
    """exact-arithmetic reference (mpmath, dps digits) Pfaffian of one float antisymmetric matrix (Parlett-Reid)."""
    import mpmath as mpm
    mpm.mp.dps = dps
    m = A.shape[0]
    M = [[mpm.mpf(float(x)) for x in row] for row in A]
    pf = mpm.mpf(1)
    for k in range(0, m - 1, 2):
        kp = max(range(k + 1, m), key=lambda j: abs(M[k][j]))
        if kp != k + 1:
            M[k + 1], M[kp] = M[kp], M[k + 1]
            for row in M:
                row[k + 1], row[kp] = row[kp], row[k + 1]
            pf = -pf
        piv = M[k][k + 1]
        if piv == 0:
            return mpm.mpf(0)
        pf *= piv
        tau = [M[k][i] / piv for i in range(m)]
        c1 = [M[i][k + 1] for i in range(m)]
        for i in range(k + 2, m):
            Mi = M[i]
            for j in range(k + 2, m):
                Mi[j] += tau[i] * c1[j] - c1[i] * tau[j]
    return pf


def slogpf_mp(A):
    import mpmath as mpm
    sh = A.shape[:-2]
    X = A.detach().reshape(-1, A.shape[-1], A.shape[-1]).numpy()
    s, l = [], []
    for x in X:
        p = pf_mp(x)
        s.append(float(mpm.sign(p)))
        l.append(float(mpm.log(abs(p))) if p != 0 else -np.inf)
    return torch.tensor(s).reshape(sh), torch.tensor(l).reshape(sh)


def base_args(a, Lx):
    b = T.parse_args(["--mode", "timing", "--pg=1,1", "--n-chains", str(a.n_chains), "--threads", str(a.threads),
                      "--Lx", str(Lx)])
    return b


# ============================================================================ verify
def verify(a):
    ok = True
    g = np.random.default_rng(7)
    import pfapack.ctypes as pc
    worst1 = 0.0
    exact1 = []
    for m in (2, 4, 10, 20, 30, 40, 60):
        A = g.standard_normal((200, m, m))
        A = A - A.transpose(0, 2, 1)
        At = torch.tensor(A)
        sf, lf = slogpf_fast(At)
        st, lt = SLOGPF_TORCH(At)
        ref = np.array([pc.pfaffian(np.asfortranarray(x)) for x in A])
        e_t = float((1 - (st * sf * torch.exp(lt - lf))).abs().max())
        e_c = float(np.abs(1 - np.sign(ref) * sf.numpy() * np.exp(np.log(np.abs(ref)) - lf.numpy())).max())
        worst1 = max(worst1, e_t, e_c)
        nx = 40 if m <= 40 else 10
        ex = np.array([float(pf_mp(x)) for x in A[:nx]])
        rel = lambda v: np.abs(v[:nx] - ex) / np.abs(ex)
        x_f, x_t, x_c = rel(sf.numpy() * np.exp(lf.numpy())), rel(st.numpy() * np.exp(lt.numpy())), rel(ref)
        exact1.append((m, x_f.max(), x_t.max(), x_c.max()))
        log(f"(1) m={m:2d}: max rel |Pf_fast - Pf_torch| = {e_t:.2e}   max rel |Pf_fast - Pf_pfapack| = {e_c:.2e}   "
            f"| vs exact (mpmath, first {nx}): fast {x_f.max():.2e} (median {np.median(x_f):.1e}), "
            f"torch {x_t.max():.2e}, pfapack {x_c.max():.2e}")
    log(f"(1) worst fast-vs-existing disagreement = {worst1:.2e} (target <= 1e-12: {'met' if worst1 <= 1e-12 else 'NOT met'})")
    wf = max(r[1] for r in exact1)
    wo = max(max(r[2], r[3]) for r in exact1)
    ok1 = wf <= max(1e-12, 2 * wo)
    ok &= ok1
    log(f"(1) vs exact: worst fast {wf:.2e}, worst torch/pfapack {wo:.2e} -> fast is as accurate as the existing and the "
        f"C reference: {'PASS' if ok1 else 'FAIL'} (fast <= max(1e-12, 2 x existing))")
    worst2 = 0.0
    exact2 = []
    for Lx in (4, 8, 12, 16):
        b = base_args(a, Lx)
        m = T.build(b, "pcg", Lx)
        gen = torch.Generator().manual_seed(11)
        nU, nD = m.random_configs(256, gen)
        for seed in (0, 1):
            th = m.init(seed)
            use(False)
            with torch.no_grad():
                l0, s0 = m.logpsi(th, nU, nD)
            use(True)
            with torch.no_grad():
                l1, s1 = m.logpsi(th, nU, nD)
            fin = torch.isfinite(l0)
            assert bool((fin == torch.isfinite(l1)).all())
            e = float((1 - s0[fin] * s1[fin] * torch.exp(l1[fin] - l0[fin])).abs().max())
            sg = bool((s0[fin] == s1[fin]).all())
            worst2 = max(worst2, e)
            ev = (1 - s0 * s1 * torch.exp(l1 - l0)).abs()
            ev[~fin] = -1
            wi = torch.argsort(ev, descending=True)[:3]
            use("mp")
            with torch.no_grad():
                lx, sx = m.logpsi(th, nU[wi], nD[wi])
            use(True)
            xf = (1 - sx * s1[wi] * torch.exp(l1[wi] - lx)).abs()
            xt = (1 - sx * s0[wi] * torch.exp(l0[wi] - lx)).abs()
            exact2.append((float(xf.max()), float(xt.max())))
            log(f"(2) pcg L={m.L:2d} Ne={m.Ne} seed {seed}: {int(fin.sum())} nonzero amplitudes, "
                f"max |1 - psi_fast/psi_torch| = {e:.2e}, signs agree: {sg} | 3 worst configs vs exact-Pfaffian "
                f"amplitude: fast {xf.max():.2e}, torch {xt.max():.2e}")
            ok &= sg
    log(f"(2) worst fast-vs-existing amplitude disagreement = {worst2:.2e} "
        f"(target <= 1e-12: {'met' if worst2 <= 1e-12 else 'NOT met'})")
    wf = max(r[0] for r in exact2)
    wo = max(r[1] for r in exact2)
    ok2 = wf <= max(1e-12, 2 * wo)
    ok &= ok2
    log(f"(2) vs exact-Pfaffian amplitudes: worst fast {wf:.2e}, worst torch {wo:.2e}: {'PASS' if ok2 else 'FAIL'} "
        f"(fast <= max(1e-12, 2 x existing))")
    worst3 = 0.0
    for Lx in (4, 8):
        b = base_args(a, Lx)
        m = T.build(b, "pcg", Lx)
        gen = torch.Generator().manual_seed(12)
        nU, nD = m.random_configs(64, gen)
        th = m.init(1)
        use(False)
        l0, _ = m.logpsi_batched(th, nU, nD, 4096)
        keep = torch.isfinite(l0)
        nU, nD = nU[keep], nD[keep]
        O0 = m.logderiv(th, nU, nD, 256)
        use(True)
        O1 = m.logderiv(th, nU, nD, 256)
        e = float((O1 - O0).abs().max() / O0.abs().max())
        worst3 = max(worst3, e)
        log(f"(3) pcg L={m.L:2d}: log-derivatives ({O0.shape[0]} x {O0.shape[1]}), max |dO| / max |O| = {e:.2e}")
    ok &= worst3 <= 1e-9
    log(f"(3) worst = {worst3:.2e}  {'PASS' if worst3 <= 1e-9 else 'FAIL'} (<= 1e-9)")
    log(f"VERIFY {'PASS' if ok else 'FAIL'}")


# ============================================================================ timing
def start(m, b, th):
    run = V.Run(m, b, 0)
    run.nU, run.nD = m.random_configs(b.n_chains, run.gen)
    run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, b.chunk)
    bad = ~torch.isfinite(run.la)
    while bad.any():
        u, d = m.random_configs(int(bad.sum()), run.gen)
        run.nU[bad], run.nD[bad] = u, d
        run.la, run.sa = m.logpsi_batched(th, run.nU, run.nD, b.chunk)
        bad = ~torch.isfinite(run.la)
    return run


def one_step(run, th):
    t = time.time()
    th2, info = run.step(th)
    return time.time() - t, info


def profile(mc, mp, thc, thp, nB=2048):
    gen = torch.Generator().manual_seed(5)
    nU, nD = mc.random_configs(nB, gen)
    out = {}
    with torch.no_grad():
        for rep in range(3):
            t0 = time.time()
            mc.logpsi0(thc, nU, nD)
            t1 = time.time()
            p = mp.unpack(thp)
            f = mp.fc(p, nU, nD, False)
            t2 = time.time()
            # X(u) assembly, copied from DetModel.logpsi0
            B, n = nU.shape[0], mp.Nu
            U = torch.nonzero(nU > 0.5)[:, 1].reshape(B, n)
            D = torch.nonzero(nD > 0.5)[:, 1].reshape(B, n)
            bi = torch.arange(B)[:, None, None]
            fUD = f[bi, U[:, :, None], D[:, None, :]]
            fTUD = f[bi, D[:, None, :], U[:, :, None]]
            fa = f - f.transpose(1, 2)
            aUU = fa[bi, U[:, :, None], U[:, None, :]]
            aDD = fa[bi, D[:, :, None], D[:, None, :]]
            c2, s2, cs = (t[None, :, None, None] for t in (mp.c2, mp.s2, mp.cs))
            G = c2 * fUD[:, None] + s2 * fTUD[:, None]
            top = torch.cat([-cs * aUU[:, None].expand_as(G), G], 3)
            bot = torch.cat([-G.transpose(2, 3), cs * aDD[:, None].expand_as(G)], 3)
            X = torch.cat([top, bot], 2)
            t3 = time.time()
            slogpf_fast(X)
            t4 = time.time()
            mp.logpsi0(thp, nU, nD)
            t5 = time.time()
            r = dict(che_logpsi0=t1 - t0, pcg_fc=t2 - t1, pcg_X=t3 - t2, pcg_pf=t4 - t3, pcg_logpsi0=t5 - t4)
            for k, v in r.items():
                out.setdefault(k, []).append(v)
    return {k: float(np.median(v)) for k, v in out.items()}


def timing(a):
    torch.set_num_threads(a.threads)
    numba.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    old = [int(s) for s in a.old_sizes.split(",") if s.strip()]
    for Lx in [int(s) for s in a.sizes.split(",")]:
        t0 = time.time()
        b = base_args(a, Lx)
        mc = T.build(b, "che", Lx)
        mp = T.build(b, "pcg", Lx)
        thc, thp = mc.init(0), mp.init(0)
        use(True)
        rc, rp = start(mc, b, thc), start(mp, b, thp)
        one_step(rc, thc)
        one_step(rp, thp)  # warm-up (numba JIT / caches), excluded
        tc, tp, ic, ip = [], [], [], []
        for _ in range(a.repeats):
            t, i = one_step(rc, thc)
            tc.append(t)
            ic.append(i)
            t, i = one_step(rp, thp)
            tp.append(t)
            ip.append(i)
        row = dict(Lx=Lx, L=mc.L, Ne=mc.Ne, n_q=int(len(mp.c2)), H_che=mc.H, P_che=mc.nparam, H_pcg=mp.H,
                   P_pcg=mp.nparam, t_che=float(np.median(tc)), t_pcg=float(np.median(tp)), t_che_all=tc, t_pcg_all=tp,
                   ratio=float(np.median(tp) / np.median(tc)),
                   ratio_pairs=float(np.median(np.array(tp) / np.array(tc))),
                   split_che={k: float(np.median([i[k] for i in ic])) for k in ("t_mcmc", "t_eloc", "t_grad", "t_sr")},
                   split_pcg={k: float(np.median([i[k] for i in ip])) for k in ("t_mcmc", "t_eloc", "t_grad", "t_sr")})
        if Lx in old:
            use(False)
            ro = start(mp, b, thp)
            one_step(ro, thp)
            to = [one_step(ro, thp)[0] for _ in range(max(2, a.repeats // 2))]
            row["t_pcg_torchpf"] = float(np.median(to))
            row["ratio_torchpf"] = row["t_pcg_torchpf"] / row["t_che"]
            use(True)
        row["profile_2048"] = profile(mc, mp, thc, thp)
        row["wall"] = time.time() - t0
        rows.append(row)
        sc, sp = row["split_che"], row["split_pcg"]
        log(f"L={mc.L:2d} Ne={mc.Ne} n_q={row['n_q']:2d}: che H={mc.H} P={mc.nparam} t={row['t_che']:.3f}s "
            f"[mcmc {sc['t_mcmc']:.2f} eloc {sc['t_eloc']:.2f} grad {sc['t_grad']:.2f}] | pcg H={mp.H} P={mp.nparam} "
            f"t={row['t_pcg']:.3f}s [mcmc {sp['t_mcmc']:.2f} eloc {sp['t_eloc']:.2f} grad {sp['t_grad']:.2f}] | "
            f"ratio {row['ratio']:.2f} (median of pairs {row['ratio_pairs']:.2f})"
            + (f" | old torch Pf: pcg {row['t_pcg_torchpf']:.3f}s, ratio {row['ratio_torchpf']:.2f}" if Lx in old else "")
            + f" | wall {row['wall']:.0f}s")
        pr = row["profile_2048"]
        log("    profile (2048 configs, no P_G): " + ", ".join(f"{k} {v * 1e3:.1f} ms" for k, v in pr.items()))
    if len(rows) >= 2:
        N = np.array([r["L"] for r in rows], float)
        for k in ("t_che", "t_pcg", "ratio"):
            y = np.array([r[k] for r in rows])
            sl = np.polyfit(np.log(N), np.log(y), 1)[0]
            log(f"fitted slope d log({k}) / d log(N) over N = {N.astype(int).tolist()}: {sl:.2f}")
    fn = os.path.join(a.outdir, f"retime_{socket.gethostname()}_t{a.threads}.json")
    with open(fn, "w") as f:
        json.dump(dict(host=socket.gethostname(), threads=a.threads, n_chains=a.n_chains, repeats=a.repeats,
                       loadavg=os.getloadavg(), rows=rows), f, indent=1)
    log(f"wrote {fn}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 R2 compiled-Pfaffian re-timing (see module docstring)")
    p.add_argument("--mode", required=True, choices=["verify", "timing"])
    p.add_argument("--sizes", default="4,8,12,16")
    p.add_argument("--old-sizes", default="4,8")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n-chains", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(a.threads)
    numba.set_num_threads(a.threads)
    {"verify": verify, "timing": timing}[a.mode](a)


if __name__ == "__main__":
    main()
