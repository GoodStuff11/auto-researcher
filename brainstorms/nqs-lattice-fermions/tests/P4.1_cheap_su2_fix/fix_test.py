#!/usr/bin/env python3
"""
P4.1 round-4 FIX: fast per-configuration AGP completeness, common-reference initialisation,
premise-enforcing spread-aware report.
=================================================================================================

Local copies only (nothing imported across folders): r4_test.py, r3_micro_test.py, r2_micro_test.py are
verbatim copies of tests/P4.1_cheap_su2_r4/. Cached ED ground states are READ (data, not code) from
--psi0-dir (default ../P4.1_cheap_su2_r4/results); if absent, ED is recomputed and cached in --outdir.

What changed against round 4 (critic fixes)
-------------------------------------------
  * Common reference start. Every arm starts from the SAME state: the closed-shell tight-binding Fermi
    sea Phi (lowest N_e/2 orbitals of the hopping matrix, both spins) times the Gutzwiller factor jv = -1.
    NNBF-type arms (che, sfb, nnbf_SG): P0 = (Phi, Phi), output weights W2 ~ 1e-3 (seed-dependent).
    KAGP arms: F0[0] = Phi Phi^T (det F0[0][U, D] = det Phi[U] det Phi[D], the same state), V ~ 1e-3.
    The round-4 asymmetry (UHF start for NNBF arms, noisy Fermi sea for KAGP) is removed; E_init is
    logged and stored for every run.
  * KAGP second term. F0[1] = c * Phi2 Phi2^T with Phi2 an INDEPENDENT Haar-random orthonormal L x n
    matrix (seed-dependent), c set so that |term 2| = --kagp-t2 (0.05) x |term 1| on the sector at
    start. After training, run stores the overlap cos(term1, term2) (raw and P_G-projected) and the norm
    ratio, so a collapse of the two terms onto one AGP is visible.
  * Fast completeness (--mode complete) replaces the timed-out tabular fit (job 411996). For a charge
    configuration c with N_s singles, any symmetric pairing F_c gives
        det F_c[U_s u D_b, D_s u D_b] = sign(c) det F_c[D_b, D_b] det S_c[U_s, D_s],
    S_c the (symmetric) Schur complement on the singles and sign(c) independent of the spins (checked in
    smoke). So the per-c K-AGP family is {sum_k beta_k det S_k[U_s, D_s]}: K symmetric N_s x N_s
    matrices (only the N_s(N_s-1)/2 off-diagonal entries enter) plus K-1 free coefficients. Each
    configuration is an independent fit of <= 2 * 66 + 1 parameters to chi_c = psi_0 restricted to the
    spin orbit of c (<= 924 entries), by L-BFGS with an iteration cap, distributed over a process pool,
    configurations taken in decreasing psi_0 weight p_c, in waves, until a wall-clock budget runs out.
    K = 2 starts from the K = 1 optimum (beta = 0.1) plus random restarts; since K = 1 is the beta = 0
    face of K = 2, F_c(K=2) is reported as max(F_c(K=2), F_c(K=1)).
    Reported per N_s group and in total:
      eps_tab(K)        = sum_c p_c (1 - F_c(K)) over fitted configurations (lower bound; unfitted
                          configurations counted as exact); the upper bound adds the unfitted weight;
      eps_E_tab(K)      = relative energy error of the assembled tabular state (fitted c: best K-AGP
                          approximant at its optimal amplitude; unfitted c: exact chi_c), with its <S^2>
                          (must be 0: each per-c piece is a singlet; this validates the sign bookkeeping);
      w(N_s >= 10)      = psi_0 weight of configurations with N_s >= 10.
    N_s <= 2 is exactly complete (singlet dimension 1) and not fitted.
  * report enforces every premise, each with a hard NO-TARGET path, and uses spread-aware verdicts
    (separation beyond the seed min-max range, plus best-of-seeds). See the leaf artifact, "Round 4 fix".

Usage
-----
  python fix_test.py --mode MODE [options]

--mode {smoke, complete, run, report, check}   (required)
  smoke    : <= 2 min login-node code check.
             (a) 4x2 open cluster, N_e = 6, t' = -0.3, U = 6: every arm (tocchio, kagp2, che, sfb, nnbf_SG)
                 built with the common reference: <S^2> = 0, analytic vs finite-difference gradient,
                 E_init of every arm equal to the reference energy (kagp2 within its term-2 admixture);
             (b) Schur-reduction identity on random symmetric F with doublons present (4x2, N_e = 6):
                 det F[U, D] / det S[U_s, D_s] constant over the spin orbit of every c;
             (c) completeness fitter on synthetic N_s = 8 targets: a K = 1 target is reached by K = 1
                 (F > 1 - 1e-8); a K = 2 target is reached by K = 2 but not by K = 1;
             (d) --mode complete on the 4x2 open cluster, N_e = 6 (all N_s <= 6: every F_c = 1, eps ~ 0,
                 assembled <S^2> = 0).
  complete : the fast completeness check on one cluster (--Lx --Ly --Ne --U --tp --bc-x --bc-y). Options:
             --budget-min (60) wall-clock minutes for the fits, --procs (16) pool size, --restarts (3),
             --fit-maxiter (400) L-BFGS iteration cap per restart, --min-Ns (4) smallest N_s fitted,
             --wave (256) configurations per wave. Writes <outdir>/complete_<tag>.json.
  run      : train ONE arm (--ansatz in tocchio, kagp2, che, sfb, nnbf_SG) with P_G (and exact P_S for che,
             sfb, nnbf_SG) from the common reference; L-BFGS with exact gradient, --maxiter (2000),
             --max-hours (15), --seed. Writes <outdir>/<tag>_w<width>_<ansatz>_seed<seed>.json with eps,
             E_init, overlap^2 with psi_0, <S^2>, capped, and (kagp2) the term overlaps.
  report   : premises and verdicts from the run JSONs, <outdir>/complete_<tag>.json and
             <outdir>/cost_r4.json (copied from round 4).
  check    : runs report on synthetic result sets (one per verdict path) in temporary directories and
             prints the verdict lines; no options used.

Options
  --Lx (4) --Ly (3) --Ne (10) --U (8) --tp (-0.25) --bc-x {obc,pbc} (obc) --bc-y {obc,pbc} (obc)
  --ansatz {tocchio, kagp2, che, sfb, nnbf_SG} (kagp2)   --nnbf-hidden (16)   --seed (0)
  --maxiter (2000) --max-hours (15) --chunk (16384) --kagp-t2 (0.05)
  --outdir (<script dir>/results) --psi0-dir (<script dir>/../P4.1_cheap_su2_r4/results) --tag (auto)
  --overwrite
"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r4_test as T  # noqa: E402  (local verbatim copy of the round-4 script)

M3, R = T.M3, T.R
log = R.log
RUN_ARMS = ["tocchio", "kagp2", "che", "sfb", "nnbf_SG"]
SPROJ = T.SPROJ

# pre-committed thresholds (artifact, "Round 4 fix")
FLOOR_NNBF_SG = 1e-5      # projected spin-resolved control must keep measurable error
CHE_HEADROOM = 1.5        # m(che) >= 1.5 m(nnbf_SG), and seed ranges disjoint
SFB_PASS_GAIN = 0.5       # fraction of the che -> nnbf_SG gap recovered (median AND best seed)
SFB_FAIL_GAIN = 0.2
KAGP_PASS = 0.7
KAGP_FAIL = 0.9
TAB_VS_NET = 0.5          # eps_tab(K=1) >= 0.5 x trained tocchio error
TAB_K2_CLOSE = 0.5        # eps_tab(K=2) <= 0.5 eps_tab(K=1)


# ============================================================================ common reference init
def fermi_sea(sec):
    e, Uh = np.linalg.eigh(-sec.lat.T)
    n = sec.Nu
    assert sec.Nu == sec.Nd
    gap = e[n] - e[n - 1]
    if gap < 1e-8:
        raise RuntimeError(f"open-shell Fermi sea (gap {gap:.1e}); common reference undefined")
    return Uh[:, :n], gap


def gutz(arm, L):
    jv = np.zeros(arm.Fj.shape[1])
    jv[L * (L - 1) // 2] = -1.0
    return jv


def kagp_terms(arm, th, chunk):
    """per-term vectors e^J det F^(k)_c[U, D] (K, dim)."""
    p = arm.unpack(th)
    sec = arm.sec
    J = arm.jastrow(p)
    expJ = np.exp(J - J.max())[sec.cidx]
    hh = np.tanh(arm.feat @ p["W"].T + p["b"])
    out = np.zeros((arm.K, sec.dim))
    for k in range(arm.K):
        F = arm._F(p, hh, k)
        for s in range(0, sec.dim, chunk):
            sl = slice(s, min(s + chunk, sec.dim))
            c = sec.cidx[sl]
            out[k, sl] = np.linalg.det(F[c[:, None, None], arm.ru[sl][:, :, None], arm.rd[sl][:, None, :]])
    return out * expJ[None]


def build_arm_fix(sec, kind, width, seed, t2=0.05, chunk=16384):
    arm = T.build_arm(sec, kind, width, seed)
    L = sec.lat.L
    Phi, _ = fermi_sea(sec)
    rng = np.random.default_rng(10_000 + seed)
    p = {k: v.copy() for k, v in arm.unpack(arm.theta0).items()}
    p["jv"] = gutz(arm, L)
    if kind in ("tocchio", "kagp2"):
        p["F0"][0] = Phi @ Phi.T
        p["V"] = 1e-3 * rng.standard_normal(p["V"].shape)
        p["W"] = rng.standard_normal(p["W"].shape) / np.sqrt(2 * L)
        p["b"] = np.zeros_like(p["b"])
        if arm.K > 1:
            Q, _ = np.linalg.qr(rng.standard_normal((L, sec.Nu)))
            p["F0"][1] = Q @ Q.T
            arm.theta0 = arm.pack(p)
            tv = kagp_terms(arm, arm.theta0, chunk)
            n1, n2 = np.linalg.norm(tv[0]), np.linalg.norm(tv[1])
            p["F0"][1] *= (t2 * n1 / n2) ** (1.0 / sec.Nu)
    else:
        P0 = np.concatenate([Phi.ravel(), Phi.ravel()])
        p["P0"] = P0
        p["W2"] = 1e-3 * rng.standard_normal(p["W2"].shape)
        p["b2"] = np.zeros_like(p["b2"])
        p["W1"] = rng.standard_normal(p["W1"].shape) / np.sqrt(p["W1"].shape[0])
        p["b1"] = np.zeros_like(p["b1"])
    arm.theta0 = arm.pack(p)
    return arm


def energy_of(sec, psi):
    nrm = np.sum(psi * psi)
    return float(np.sum(psi * sec.H(psi)) / nrm), float(np.sum(psi * sec.S2(psi)) / nrm)


# ============================================================================ fast completeness
def singles_tables(Ns):
    pats, _ = T.kernel(Ns)
    U = np.array([list(q) for q in pats], dtype=np.int64)
    D = np.array([[i for i in range(Ns) if i not in q] for q in pats], dtype=np.int64)
    iu, ju = np.triu_indices(Ns, 1)
    tri = -np.ones((Ns, Ns), dtype=np.int64)
    tri[iu, ju] = np.arange(len(iu))
    tri[ju, iu] = np.arange(len(iu))
    Tm = tri[U[:, :, None], D[:, None, :]]
    assert (Tm >= 0).all()
    return Tm, len(iu)


def kagp_vec(x, Tm, nt, K):
    w = 0.0
    ds, cofs, betas = [], [], []
    for k in range(K):
        S = x[k * nt:(k + 1) * nt][Tm]
        d = np.linalg.det(S)
        beta = 1.0 if k == 0 else x[K * nt + k - 1]
        w = w + beta * d
        ds.append(d); betas.append(beta); cofs.append(S)
    return w, ds, betas, cofs


def fid_obj(x, chi, Tm, nt, K):
    w, ds, betas, Ss = kagp_vec(x, Tm, nt, K)
    a = float(chi @ w); nw = float(w @ w); nc = float(chi @ chi)
    if nw < 1e-300:
        return 1.0, np.zeros_like(x)
    F = a * a / (nw * nc)
    dF = 2 * a * chi / (nw * nc) - 2 * a * a * w / (nw * nw * nc)
    g = np.zeros_like(x)
    for k in range(K):
        cof = R.cofactor(Ss[k])
        g[k * nt:(k + 1) * nt] = betas[k] * np.bincount(Tm.ravel(), weights=(cof * dF[:, None, None]).ravel(),
                                                        minlength=nt)
        if k > 0:
            g[K * nt + k - 1] = float(dF @ ds[k])
    return 1.0 - F, -g


def fit_K(chi, Tm, nt, K, restarts, maxiter, rng, x_start=None):
    best_F, best_w, best_x = -1.0, None, None
    starts = []
    if x_start is not None:
        starts.append(x_start)
    while len(starts) < restarts + (x_start is not None):
        starts.append(np.concatenate([rng.standard_normal(K * nt), 0.5 * rng.standard_normal(K - 1)]))
    nfev = 0
    for x0 in starts:
        res = minimize(fid_obj, x0, args=(chi, Tm, nt, K), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, maxcor=20, ftol=1e-15, gtol=1e-11))
        nfev += res.nfev
        F = 1.0 - float(res.fun)
        if F > best_F:
            best_F, best_x = F, res.x
            best_w = kagp_vec(res.x, Tm, nt, K)[0]
        if best_F > 1 - 1e-12:
            break
    return best_F, best_w, best_x, nfev


_TABLES = {}


def fit_config(task):
    """task = (key, Ns, chi, seed, restarts, maxiter) -> (key, F1, F2, w1, w2, nfev)."""
    key, Ns, chi, seed, restarts, maxiter = task
    if Ns not in _TABLES:
        _TABLES[Ns] = singles_tables(Ns)
    Tm, nt = _TABLES[Ns]
    rng = np.random.default_rng(seed)
    chi = chi / np.linalg.norm(chi)
    F1, w1, x1, n1 = fit_K(chi, Tm, nt, 1, restarts, maxiter, rng)
    x2 = np.concatenate([x1, 0.3 * rng.standard_normal(nt), [0.1]])
    F2, w2, _, n2 = fit_K(chi, Tm, nt, 2, restarts - 1, maxiter, rng, x_start=x2)
    if F1 > F2:
        F2, w2 = F1, w1
    return key, F1, F2, w1, w2, n1 + n2


def load_psi0(a, sec, sg):
    tag = tag_of(a)
    for d in (a.psi0_dir, a.outdir):
        fn = os.path.join(d, f"psi0_{tag}.npz")
        if os.path.exists(fn):
            z = np.load(fn)
            sg.set_irrep(z["P0"])
            return z["P0"], float(z["E0"])
    t = time.time()
    P0, E0, s20, lam, nxt = M3.ground_state(sec, sg)
    log(f"ED: E0={E0:.10f} <S^2>={s20:.1e} irrep={lam} ({time.time() - t:.0f}s)")
    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, f"psi0_{tag}.npz"), P0=P0, E0=E0)
    return P0, E0


def complete(a):
    t0 = time.time()
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    P0, E0 = load_psi0(a, sec, sg)
    psi0 = P0.ravel() / np.linalg.norm(P0)
    groups = T.charge_groups(sec)
    log(f"complete: {tag_of(a)} dim={sec.dim} nC={sec.nC} E0={E0:.10f} groups N_s={sorted(groups)} "
        f"(setup {time.time() - t0:.0f}s)")
    tasks, info = [], {}
    for Ns, g in groups.items():
        chi = psi0[g["state"]]
        pc = np.sum(chi * chi, 1)
        info[Ns] = dict(pc=pc, state=g["state"], F1=np.ones(len(pc)), F2=np.ones(len(pc)),
                        fitted=np.zeros(len(pc), bool), orbit=g["state"].shape[1])
        if Ns >= max(a.min_Ns, 4) and g["state"].shape[1] > 1:
            for ci in range(len(pc)):
                if pc[ci] > 1e-14:
                    tasks.append((pc[ci], Ns, ci))
                else:
                    info[Ns]["fitted"][ci] = True
        else:
            info[Ns]["fitted"][:] = True  # N_s <= 2 (or below --min-Ns): exactly complete, F_c = 1
    tasks.sort(key=lambda z: -z[0])
    log(f"  {len(tasks)} configurations to fit, total weight {sum(z[0] for z in tasks):.4f}; "
        f"budget {a.budget_min} min on {a.procs} processes")
    fit1 = psi0.copy(); fit2 = psi0.copy()
    tf = time.time()
    done = 0
    nfev = 0
    with Pool(a.procs) as pool:
        while done < len(tasks) and time.time() - tf < 60 * a.budget_min:
            wave = tasks[done:done + a.wave]
            jobs = [((Ns, ci), Ns, psi0[info[Ns]["state"][ci]], 1000 * Ns + ci, a.restarts, a.fit_maxiter)
                    for _, Ns, ci in wave]
            for (Ns, ci), F1, F2, w1, w2, nf in pool.imap_unordered(fit_config, jobs, chunksize=2):
                I = info[Ns]
                I["F1"][ci], I["F2"][ci], I["fitted"][ci] = F1, F2, True
                st = I["state"][ci]
                chi = psi0[st]
                for vec, w in ((fit1, w1), (fit2, w2)):
                    vec[st] = w * (chi @ w) / (w @ w)
                nfev += nf
            done += len(wave)
            log(f"  fitted {done}/{len(tasks)} (weight down to p_c={wave[-1][0]:.2e}), "
                f"{(time.time() - tf) / 60:.1f} min")
    res = dict(tag=tag_of(a), E0=E0, dim=sec.dim, nC=sec.nC, budget_min=a.budget_min, procs=a.procs,
               restarts=a.restarts, fit_maxiter=a.fit_maxiter, n_tasks=len(tasks), n_fitted=done,
               fit_seconds=time.time() - tf, nfev=nfev, groups=[])
    tot = dict(e1=0.0, e2=0.0, unfit=0.0)
    print(f"{'N_s':>4} {'#c':>6} {'weight':>9} {'singlet':>7} {'fitted':>7} {'w_unfit':>9} "
          f"{'wInfid K1':>10} {'minF K1':>9} {'wInfid K2':>10} {'minF K2':>9}")
    for Ns in sorted(info):
        I = info[Ns]
        f = I["fitted"]
        e1 = float(np.sum(I["pc"][f] * (1 - I["F1"][f])))
        e2 = float(np.sum(I["pc"][f] * (1 - I["F2"][f])))
        wu = float(np.sum(I["pc"][~f]))
        tot["e1"] += e1; tot["e2"] += e2; tot["unfit"] += wu
        row = dict(Ns=int(Ns), nc=int(len(f)), weight=float(I["pc"].sum()), singlet_dim=T.catalan(Ns // 2),
                   n_fitted=int(f.sum()), w_unfit=wu, wInfid_K1=e1, wInfid_K2=e2,
                   minF_K1=float(I["F1"][f].min()) if f.any() else 1.0,
                   minF_K2=float(I["F2"][f].min()) if f.any() else 1.0)
        res["groups"].append(row)
        print(f"{Ns:4d} {row['nc']:6d} {row['weight']:9.2e} {row['singlet_dim']:7d} {row['n_fitted']:7d} "
              f"{wu:9.2e} {e1:10.2e} {row['minF_K1']:9.6f} {e2:10.2e} {row['minF_K2']:9.6f}")
    res["w_Ns_ge10"] = float(sum(info[Ns]["pc"].sum() for Ns in info if Ns >= 10))
    res["w_unfit"] = tot["unfit"]
    for K, vec, e in ((1, fit1, tot["e1"]), (2, fit2, tot["e2"])):
        v = vec.reshape(sec.shape)
        E, s2 = energy_of(sec, v)
        ov = float(np.sum(v * P0) ** 2 / (np.sum(v * v) * np.sum(P0 * P0)))
        res[f"K{K}"] = dict(eps_tab_lo=e, eps_tab_hi=e + tot["unfit"], infid_assembled=1 - ov,
                            eps_E_tab=(E - E0) / abs(E0), S2=s2)
        log(f"K={K}: eps_tab (weighted infidelity) in [{e:.3e}, {e + tot['unfit']:.3e}]  assembled infidelity "
            f"{1 - ov:.3e}  eps_E_tab={(E - E0) / abs(E0):.3e}  <S^2>={s2:.1e}")
    log(f"weight of configurations with N_s >= 10: {res['w_Ns_ge10']:.4e}; unfitted weight {tot['unfit']:.2e}; "
        f"total {time.time() - t0:.0f}s")
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, f"complete_{tag_of(a)}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


# ============================================================================ smoke
def smoke(a):
    t0 = time.time()
    ok = True
    # (a) arms with the common reference
    sec = M3.make_sector(4, 2, 6, 6.0, -0.3, "obc", "obc")
    sg = M3.SG(sec)
    P0, E0, *_ = M3.ground_state(sec, sg)
    Phi, gap = fermi_sea(sec)
    log(f"(a) 4x2 open Ne=6 dim={sec.dim} E0={E0:.6f}; Fermi-sea shell gap {gap:.3f}")
    Eref = None
    for kind in RUN_ARMS:
        arm = build_arm_fix(sec, kind, 4, 0, t2=a.kagp_t2, chunk=4096)
        st = M3.State(sec, arm, SPROJ[kind], sg, 4096)
        phi, _ = arm.forward(arm.theta0, 4096)
        Ei, s2i = energy_of(sec, st.psi(phi))
        th = arm.theta0 + 0.1 * np.random.default_rng(3).standard_normal(arm.nparam)
        phi, _ = arm.forward(th, 4096)
        _, s2 = energy_of(sec, st.psi(phi))
        rel = M3.fd_check(st, th)
        if kind == "tocchio":
            Eref = Ei
        tolE = 0.05 if kind == "kagp2" else 2e-3
        good = abs(s2) < 1e-10 and rel < 1e-5 and abs(Ei - Eref) < tolE * abs(Eref)
        ok &= good
        log(f"  {kind:8s} P={arm.nparam:4d} E_init={Ei:.6f} <S^2>={s2:+.1e} grad rel.err={rel:.1e} "
            f"[{'PASS' if good else 'FAIL'}]")
    arm = build_arm_fix(sec, "kagp2", 4, 0, t2=a.kagp_t2, chunk=4096)
    tv = kagp_terms(arm, arm.theta0, 4096)
    log(f"  kagp2 init: |term2|/|term1| = {np.linalg.norm(tv[1]) / np.linalg.norm(tv[0]):.3f}, "
        f"cos(term1, term2) = {tv[0] @ tv[1] / np.linalg.norm(tv[0]) / np.linalg.norm(tv[1]):+.3f}")
    # (b) Schur reduction with doublons present
    rng = np.random.default_rng(1)
    G = rng.standard_normal((sec.lat.L, sec.lat.L)); F = G + G.T
    groups = T.charge_groups(sec)
    worst = 0.0; nchk = 0
    for Ns, g in groups.items():
        if Ns < 2:
            continue
        Tm, nt = singles_tables(Ns)
        for ci in range(0, g["state"].shape[0], max(1, g["state"].shape[0] // 15)):
            st = g["state"][ci]
            nU = sec.up.bits[sec.iu[st[0]]].astype(int); nD = sec.dn.bits[sec.id[st[0]]].astype(int)
            n = nU + nD
            sing = np.nonzero(n == 1)[0]; dbl = np.nonzero(n == 2)[0]
            Fs = F[np.ix_(sing, sing)]
            if len(dbl):
                Fs = Fs - F[np.ix_(sing, dbl)] @ np.linalg.solve(F[np.ix_(dbl, dbl)], F[np.ix_(dbl, sing)])
            full = np.array([np.linalg.det(F[np.ix_(np.nonzero(sec.up.bits[sec.iu[s]])[0],
                                                     np.nonzero(sec.dn.bits[sec.id[s]])[0])]) for s in st])
            iu, ju = np.triu_indices(Ns, 1)
            red = np.linalg.det(Fs[iu, ju][Tm])
            ratio = full / red
            worst = max(worst, float(np.ptp(ratio) / np.abs(ratio).max())); nchk += 1
    good = worst < 1e-10
    ok &= good
    log(f"(b) Schur reduction det F[U,D] / det S[U_s,D_s] constant over the orbit: {nchk} configs, "
        f"max rel. spread {worst:.1e} [{'PASS' if good else 'FAIL'}]")
    # (c) synthetic targets at N_s = 8
    Tm, nt = singles_tables(8)
    rng = np.random.default_rng(2)
    x1 = rng.standard_normal(nt)
    t1 = kagp_vec(x1, Tm, nt, 1)[0]
    x2 = np.concatenate([rng.standard_normal(2 * nt), [0.8]])
    t2 = kagp_vec(x2, Tm, nt, 2)[0]
    r1 = fit_config(("a", 8, t1, 5, 3, 2000))
    r2 = fit_config(("b", 8, t2, 6, 3, 2000))
    good = r1[1] > 1 - 1e-8 and r2[2] > 1 - 1e-8 and r2[1] < 1 - 1e-6
    ok &= good
    log(f"(c) N_s=8 synthetic: K=1 target F1={r1[1]:.12f}; K=2 target F1={r2[1]:.8f} F2={r2[2]:.12f} "
        f"[{'PASS' if good else 'FAIL'}]")
    # (d) complete on 4x2, N_e = 6
    with tempfile.TemporaryDirectory() as d:
        b = parse_args(["--mode", "complete", "--Lx", "4", "--Ly", "2", "--Ne", "6", "--U", "8", "--tp", "-0.25",
                        "--outdir", d, "--psi0-dir", d, "--procs", "2", "--budget-min", "2"])
        res = complete(b)
    good = all(res[f"K{K}"]["eps_tab_hi"] < 1e-10 and abs(res[f"K{K}"]["S2"]) < 1e-10
               and abs(res[f"K{K}"]["eps_E_tab"]) < 1e-10 for K in (1, 2))
    ok &= good
    log(f"(d) 4x2 open Ne=6 complete: eps_tab K1 {res['K1']['eps_tab_hi']:.1e}, K2 {res['K2']['eps_tab_hi']:.1e}, "
        f"assembled <S^2> {res['K1']['S2']:.1e} [{'PASS' if good else 'FAIL'}]")
    log(f"SMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    return ok


# ============================================================================ run
def tag_of(a):
    return a.tag or f"L{a.Lx}x{a.Ly}x{a.bc_x}{a.bc_y}_N{a.Ne}_U{a.U:g}_tp{a.tp:g}"


def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"{tag_of(a)}_w{a.nnbf_hidden}_{a.ansatz}_seed{a.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"exists: {fn}")
        return
    sec = M3.make_sector(a.Lx, a.Ly, a.Ne, a.U, a.tp, a.bc_y, a.bc_x)
    sg = M3.SG(sec)
    P0, E0 = load_psi0(a, sec, sg)
    arm = build_arm_fix(sec, a.ansatz, a.nnbf_hidden, a.seed, t2=a.kagp_t2, chunk=a.chunk)
    st = M3.State(sec, arm, SPROJ[a.ansatz], sg, a.chunk)
    phi, _ = arm.forward(arm.theta0, a.chunk)
    E_init, _ = energy_of(sec, st.psi(phi))
    log(f"{a.ansatz}: P={arm.nparam} (target NNBF width {a.nnbf_hidden}) E_init={E_init:.8f} E0={E0:.8f}")
    th, E, it, sec_ = R.optimise(st, arm.theta0, a.maxiter, a.max_hours)
    phi, _ = arm.forward(th, a.chunk)
    psi = st.psi(phi)
    nrm = np.sum(psi * psi)
    out = dict(ansatz=a.ansatz, seed=a.seed, P=arm.nparam, E=E, E0=E0, E_init=E_init, eps=(E - E0) / abs(E0),
               iters=it, seconds=sec_, capped=bool(sec_ > 3600 * a.max_hours * 0.999),
               S2=float(np.sum(psi * sec.S2(psi)) / nrm), overlap2=float(np.sum(psi * P0) ** 2 / (nrm * np.sum(P0 * P0))))
    if a.ansatz == "kagp2":
        tv = kagp_terms(arm, th, a.chunk)
        pv = [sg.project(v.reshape(sec.shape)).ravel() for v in tv]
        cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))  # noqa: E731
        out.update(term_cos=cs(tv[0], tv[1]), term_cos_PG=cs(pv[0], pv[1]),
                   term_norm_ratio=float(np.linalg.norm(tv[1]) / np.linalg.norm(tv[0])))
    log(json.dumps(out))
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)


# ============================================================================ report
def report(a):
    pat = os.path.join(a.outdir, f"{tag_of(a)}_w{a.nnbf_hidden}_*_seed*.json")
    rs = {}
    for f in glob.glob(pat):
        r = json.load(open(f))
        rs.setdefault(r["ansatz"], []).append(r)
    eps = {k: np.array([r["eps"] for r in v]) for k, v in rs.items()}
    m = {k: float(np.median(v)) for k, v in eps.items()}
    lo = {k: float(v.min()) for k, v in eps.items()}
    hi = {k: float(v.max()) for k, v in eps.items()}
    for k in RUN_ARMS:
        if k in rs:
            extra = ""
            if k == "kagp2":
                extra = " term cos " + ",".join(f"{r.get('term_cos', np.nan):+.2f}" for r in rs[k])
            print(f"{k:8s} n={len(eps[k])} P={rs[k][0]['P']:5d} m(eps)={m[k]:.3e} [min {lo[k]:.2e}, max {hi[k]:.2e}] "
                  f"E_init {np.median([r.get('E_init', np.nan) for r in rs[k]]):.5f} "
                  f"max|S2|={max(abs(r['S2']) for r in rs[k]):.1e} capped={any(r['capped'] for r in rs[k])}{extra}")
    full = all(len(rs.get(k, [])) >= 3 for k in RUN_ARMS)
    exact = all(max(abs(r["S2"]) for r in rs[k]) < 1e-10 for k in rs)
    capped = {k for k in rs if any(r["capped"] for r in rs[k])}
    if not full or not exact:
        why = "fewer than 3 seeds for some arm" if not full else "exactness violated"
        print(f"premise P_run: NOT MET ({why})")
        print(f"Route 4.2 kagp2: INCOMPLETE ({why})")
        print(f"Route 4.1 sfb  : INCOMPLETE ({why})")
        return
    # cluster premise: the projected spin-resolved control keeps measurable error
    floor_ok = m["nnbf_SG"] >= FLOOR_NNBF_SG
    print(f"premise P_floor: m(eps_nnbf_SG) = {m['nnbf_SG']:.2e} >= {FLOOR_NNBF_SG:.0e}: {'MET' if floor_ok else 'NOT MET'}")
    # ---------------- Route 4.1 (sfb)
    head_ok = m["che"] >= CHE_HEADROOM * m["nnbf_SG"] and lo["che"] > hi["nnbf_SG"]
    print(f"premise P_che (sfb): m(che)/m(nnbf_SG) = {m['che'] / m['nnbf_SG']:.2f} (>= {CHE_HEADROOM}), "
          f"min(che) {lo['che']:.2e} > max(nnbf_SG) {hi['nnbf_SG']:.2e}: {'MET' if head_ok else 'NOT MET'}")
    if not floor_ok:
        v1 = "NO-TARGET (projected control at floor)"
    elif not head_ok:
        v1 = "NO-TARGET (no resolved che -> nnbf_SG gap)"
    else:
        g_med = (m["che"] - m["sfb"]) / (m["che"] - m["nnbf_SG"])
        g_best = (lo["che"] - lo["sfb"]) / (lo["che"] - lo["nnbf_SG"])
        print(f"sfb gain fraction of the che -> nnbf_SG gap: median {g_med:.2f}, best-of-seeds {g_best:.2f}")
        if {"sfb", "che", "nnbf_SG"} & capped:
            v1 = "AMBIGUOUS (an arm hit the wall-clock cap)"
        elif hi["sfb"] < lo["che"] and g_med >= SFB_PASS_GAIN and g_best >= SFB_PASS_GAIN:
            v1 = "PASS"
        elif g_med < SFB_FAIL_GAIN and g_best < SFB_FAIL_GAIN and lo["sfb"] > hi["nnbf_SG"]:
            v1 = "FAIL (the spin feature recovers < 20% of the gap)"
        else:
            v1 = "AMBIGUOUS"
    # ---------------- Route 4.2 (kagp2)
    cf = os.path.join(a.outdir, f"complete_{tag_of(a)}.json")
    infid_toc = float(np.median([1 - r["overlap2"] for r in rs["tocchio"]]))
    if not os.path.exists(cf):
        prem2, why2 = False, "completeness json missing"
    else:
        cj = json.load(open(cf))
        e1, e2 = cj["K1"]["eps_tab_lo"], cj["K2"]["eps_tab_hi"]
        eE1 = cj["K1"]["eps_E_tab"]
        c1 = e1 >= TAB_VS_NET * infid_toc
        c2 = e1 >= TAB_VS_NET * m["tocchio"]
        c3 = e2 <= TAB_K2_CLOSE * e1
        print(f"premise P_tab (kagp2): eps_tab(K=1, lower) = {e1:.2e} vs 0.5 x m(infid_tocchio) = "
              f"{TAB_VS_NET * infid_toc:.2e} [{c1}] and vs 0.5 x m(eps_tocchio) = {TAB_VS_NET * m['tocchio']:.2e} "
              f"[{c2}]; eps_tab(K=2, upper) = {e2:.2e} <= 0.5 eps_tab(K=1) [{c3}]; w(N_s>=10) = "
              f"{cj['w_Ns_ge10']:.3e}; (not gating: eps_E_tab(K=1) = {eE1:.2e}, an upper bound on the family floor)")
        prem2 = c1 and c2 and c3
        why2 = "" if prem2 else ("tabular ceiling below half the trained tocchio error" if not (c1 and c2)
                                 else "K=2 does not close the tabular gap")
    if not floor_ok:
        v2 = "NO-TARGET (projected control at floor)"
    elif not prem2:
        v2 = f"NO-TARGET ({why2})"
    else:
        cost = json.load(open(os.path.join(a.outdir, "cost_r4.json")))["rows"][-1]
        if cost["kagp2"] >= cost["che_closed"]:
            v2 = "FAIL (cost)"
        elif {"kagp2", "tocchio", "che"} & capped:
            v2 = "AMBIGUOUS (an arm hit the wall-clock cap)"
        elif (hi["kagp2"] < lo["tocchio"] and m["kagp2"] <= KAGP_PASS * m["tocchio"]
              and lo["kagp2"] <= KAGP_PASS * lo["tocchio"] and m["kagp2"] <= m["che"]):
            v2 = "PASS"
        elif ((m["kagp2"] >= KAGP_FAIL * m["tocchio"] and lo["kagp2"] >= KAGP_FAIL * lo["tocchio"])
              or lo["kagp2"] > hi["che"]):
            v2 = "FAIL (accuracy)"
        else:
            v2 = "AMBIGUOUS"
    print(f"Route 4.2 kagp2: {v2}")
    print(f"Route 4.1 sfb  : {v1}")


def check(_a):
    def scen(name, eps, spread=None, comp=(1e-2, 1e-3, 1e-2), over=0.99, drop=None, capped=None):
        spread = dict({k: 0.05 for k in eps}, **(spread or {}))
        d = tempfile.mkdtemp()
        b = parse_args(["--mode", "report", "--Ne", "10", "--outdir", d])
        tag = tag_of(b)
        for k, e in eps.items():
            for s in range(3):
                if drop == (k, s):
                    continue
                json.dump(dict(ansatz=k, seed=s, P=2600, eps=e * (1 + spread[k] * (s - 1)), S2=1e-16,
                               capped=bool(capped == k), overlap2=over, E_init=-6.285, term_cos=0.1),
                          open(os.path.join(d, f"{tag}_w16_{k}_seed{s}.json"), "w"))
        shutil.copy(os.path.join(HERE, "results", "cost_r4.json"), d)
        if comp is not None:
            json.dump(dict(K1=dict(eps_tab_lo=comp[0], eps_E_tab=comp[2]), K2=dict(eps_tab_hi=comp[1]),
                           w_Ns_ge10=0.1), open(os.path.join(d, f"complete_{tag}.json"), "w"))
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            report(b)
        print(f"--- {name}\n" + "\n".join("    " + ln for ln in f.getvalue().splitlines() if "Route" in ln))
        shutil.rmtree(d)
    base = dict(tocchio=1e-3, kagp2=5e-4, che=8e-4, sfb=4e-4, nnbf_SG=3e-4)
    scen("expect 4.2 PASS, 4.1 PASS", base)
    scen("expect 4.2 FAIL(accuracy), 4.1 FAIL", dict(base, kagp2=9.6e-4, sfb=7.8e-4))
    scen("expect 4.1 AMBIGUOUS (median gain but ranges overlap)", dict(base, sfb=5e-4), spread=dict(sfb=0.6))
    scen("expect 4.2 NO-TARGET (tabular ceiling below trained error)", base, comp=(1e-5, 1e-6, 1e-5))
    scen("expect 4.2 NO-TARGET (completeness json missing)", base, comp=None)
    scen("expect 4.2 NO-TARGET (K=2 does not close)", base, comp=(1e-2, 8e-3, 1e-2))
    scen("expect 4.1 NO-TARGET (no che headroom); 4.2 FAIL since kagp2 is worse than che", dict(base, che=3.3e-4))
    scen("expect both NO-TARGET (control at floor)", dict(base, nnbf_SG=5e-6))
    scen("expect both INCOMPLETE (missing seed)", base, drop=("sfb", 2))
    scen("expect 4.1 AMBIGUOUS (capped)", base, capped="sfb")


# ============================================================================ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 round-4 fix tests (see module docstring)")
    p.add_argument("--mode", required=True, choices=["smoke", "complete", "run", "report", "check"])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--Ne", type=int, default=10)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--bc-x", default="obc", choices=["obc", "pbc"], dest="bc_x")
    p.add_argument("--bc-y", default="obc", choices=["obc", "pbc"], dest="bc_y")
    p.add_argument("--ansatz", default="kagp2", choices=RUN_ARMS)
    p.add_argument("--nnbf-hidden", type=int, default=16, dest="nnbf_hidden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--max-hours", type=float, default=15.0, dest="max_hours")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--kagp-t2", type=float, default=0.05, dest="kagp_t2")
    p.add_argument("--budget-min", type=float, default=60.0, dest="budget_min")
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--fit-maxiter", type=int, default=400, dest="fit_maxiter")
    p.add_argument("--min-Ns", type=int, default=4, dest="min_Ns")
    p.add_argument("--wave", type=int, default=256)
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    p.add_argument("--psi0-dir", default=os.path.join(HERE, "..", "P4.1_cheap_su2_r4", "results"), dest="psi0_dir")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dict(smoke=smoke, complete=complete, run=run, report=report, check=check)[a.mode](a)


if __name__ == "__main__":
    main()
