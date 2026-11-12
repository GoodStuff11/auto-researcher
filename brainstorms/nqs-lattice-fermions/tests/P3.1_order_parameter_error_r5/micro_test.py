#!/usr/bin/env python3
"""
P3.1 round-5 decisive test (after the per-leaf critic's fixes): error of long-range correlations of
CONVERGED symmetry-projected NNBF states on the shared 4x4 OBC t-t'-J instance, from cheap information
=====================================================================================================

Instance (r5_core.INST0): 4x4 OBC, N_up = N_dn = 6, t' = -0.25, J = 0.4, D = 1681680, psi_0 a
non-degenerate singlet (Mx = My = +1, D = -1, |G| = 8). Sign half of the Hard Core only.
Observables (r5_core.Observables): long-range d-wave pair correlation (bond-centre distance >= 3),
spin correlation S_i.S_j and charge correlation (n_i - nbar)(n_j - nbar) at site distance >= 3.5.

Population (CONVERGED finals only; no snapshots)
------------------------------------------------
NNBF (P2.1 r3 arm "nnbf": full backflow, K = 1, Ha = 0), WIDTH 6 ONLY (critic: no widths 3/12).
Seeds 11, 22, 33 are read from the P2.1 round-3 parameter files (jobs 410988, 451706); further
width-6 seeds are trained here with --mode train and read from <results>/params. ONE label per
network is scored: "P" (P_chi psi_theta). The P_S label is not scored, and P_S hybrids are never
formed (|psi_k| sgn psi_j of two spin-projected states is not a spin eigenstate). The unit is the
seed; the per-seed EHX/ZV estimates SHARE partner data (each uses every other seed), so the
per-seed numbers are not independent.

Route (P3.1 artifact, round 5)
------------------------------
  EHX  ensemble sign/amplitude hybrid extrapolation. For state k take every other seed j (same
       width, label P) and form h_kj = |psi_k| sgn psi_j (sampled from |psi_k|^2) and
       h_jk = |psi_j| sgn psi_k (sampled from |psi_j|^2). Both lie in the chi irrep because
       sgn psi_j(g n) = chi(g) xi_g(n) sgn psi_j(n) (xi_g the fermionic reordering sign) while
       |psi_k(g n)| = |psi_k(n)|. Over the point set {psi_k} U {psi_j-hybrids}, three fit forms are
       pre-registered; THE DECIDING FORM IS "S":
         S (decides)  a = alpha + gamma * sigma            -> a_hat = alpha (sigma -> 0)
         R            a = alpha + gamma * sqrt(max(E - E0_hat, 0)), E0_hat from E = E0_hat + s sigma^2
         E            a = alpha + beta * E,  a_hat = alpha + beta * E0_hat   (the original linear chord)
       Reason for S: delta a is first order in the wavefunction error e, sigma is first order, and
       E - E0 is second order, so a linear chord in E understates delta; S needs no E0_hat, whereas
       R puts the error of E0_hat (dE = 0.03-0.05 against sigma^2 ~ 0.14) under a square root.
       delta_hat = a_hat - a_k (a_k sampled). Nothing is fitted to the truth.
  OSG  KILLED by the critic (it is the Assaraf-Caffarel / Toulouse-Assaraf-Umrigar zero-bias
       correction with a fixed linear-basis psi'; its exact-element ablation OSGX has the wrong sign
       or zero on 4 of 6 smoke items, and the sampled version was biased 3.7x versus exact Galerkin).
       Not computed in --mode analyze. OSGX and ZRO are kept as DIAGNOSTICS only (never scored
       against a threshold). The sampled osg() function remains only for the historic --mode checks.
Controls
  ZV   the same three fits restricted to the plain ensemble members {psi_k} U {psi_j} (no hybrids).
       THE DECISIVE COMPARISON IS EHX (form S) vs ZV (form S): ZV isolates what the hybrids add.
  L1, L2  one / two exact Lanczos steps (variance extrapolation). Reported only: they fail here by a
       known mechanism (the residual error is low-omega; Lanczos lowers dE 2-3.5x and leaves delta).
  SD   seed spread at the same width (unsigned). Reported only.
Diagnostics: OSGX (Galerkin in the observable's pieces, exact elements), ZRO (exact first-order
oracle -2 <r|Kt^-1|w>, MINRES).

Uncertainty and trust
---------------------
SE(delta_hat) = sqrt(SE_boot^2 + SE_jk^2): SE_boot over Monte Carlo draws (n_boot replicas,
every point resampled), SE_jk the leave-one-PARTNER-out jackknife (each replicate drops one other
seed, i.e. its plain point and both its hybrids; SE_jk^2 = (n-1)/n sum (theta_i - theta_bar)^2).
A replicate needs >= 2 points (an exactly determined line is allowed).

Injected-sign-flip correctness check (EHX must recover it; part of the gate)
---------------------------------------------------------------------------
Three synthetic members psi_i = psi_0 * exp(eps_amp * xi(orbit)) * (-1)^{[n in S_i]}, with S_i a
union of whole G-orbits drawn with probability proportional to orbit weight until the flipped
weight reaches w_i in (0.01, 0.02, 0.04), xi a G-invariant Gaussian per orbit, eps_amp = 0.10 (so
the hybrids differ from the members). Every psi_i lies in the chi irrep (checked) and its delta
= a_0 - a(psi_i) is known exactly. EHX and ZV are run on this ensemble exactly as on the networks.
INJECT-PASS iff EHX (form S) tracks >= 50 % of the scored injected items, with >= 3 scored.

Metric
------
Item = (seed, observable), label P. delta = a_0 - a_psi (exact). Scored iff |delta| >= 0.02 |a_0|.
Trusted: SE <= 0.5 |delta_hat| (exact arms always trusted). Tracked: trusted and
1/2 <= delta_hat/delta <= 2 (|.| for SD). False: trusted and delta_hat/delta < 1/2 (includes the
wrong sign). T_X(u) = tracked fraction of seed u (abstentions and untrusted count as misses);
M_X = median_u T_X(u). G_X = false / TRUSTED items, defined only with >= MIN_TRUSTED = 3 trusted
items (otherwise G is undefined and X cannot PASS).

Pre-committed rules (--mode aggregate --rule gate|full)
------------------------------------------------------
PREMISE (both rules, NO-TARGET otherwise): P2.1 width-6 median dE_sign >= 0.005 t over the seeds
found, and >= 6 scored items over >= 3 seeds.
  gate (cheap run: width 6, seeds 11/22/33, label P):
    GATE-PASS iff INJECT-PASS and M_EHX >= 0.5 and median_u [T_EHX(u) - T_ZV(u)] > 0 and
    G_EHX <= 0.10 (defined). Action: train >= 8 more width-6 seeds (submit_train.sbatch) and run
    the full rule. Otherwise GATE-FAIL: EHX is "Route 2.3 restated" (variance extrapolation of a)
    and the leaf is marked for mutation; no training compute is spent.
  full (width 6, >= 11 seeds, label P):
    PASS:  M_EHX >= 0.5, Delta = median_u [T_EHX - T_ZV] >= 0.2, T_EHX(u) - T_ZV(u) > 0 on every
           seed, G_EHX <= 0.10 (defined), INJECT-PASS.
    FAIL:  M_EHX < 0.3 or Delta < 0.05 or G_EHX > 0.20 or INJECT fails.
    AMBIGUOUS otherwise: rerun --mode analyze with --mc-samples 500000; still not PASS -> killed.

Usage
-----
  python3 micro_test.py --mode checks                  small-cluster correctness checks (~3 min)
  python3 micro_test.py --mode inject [--results DIR]  injected-sign-flip check alone (~5 min)
  python3 micro_test.py --mode train --width 6 --seed S   one projected-SR training (hours)
  python3 micro_test.py --mode analyze --widths 6 --seeds 11,22,33   all arms + inject check
  python3 micro_test.py --mode aggregate --rule gate [--results DIR]   verdict (seconds)
Options
  --mode {checks,inject,train,analyze,aggregate}  (default checks)
      checks    small 4x4 2+2 cluster: first-order oracle, historic OSG sampled vs exact, local energy.
      inject    builds the instance, runs only the injected-sign-flip check; writes <results>/inject.json.
      train     trains one network (--width, --seed); writes <params-dir>/inst0_nnbf_w{W}_s{S}.npz.
      analyze   all arms on every (width, seed, label) found, plus the injected check;
                writes <results>/analysis.json.
      aggregate reads <results>/analysis.json, prints the table and verdict, writes aggregate.json.
  --rule {gate,full}  (aggregate only; default gate) which pre-committed rule above is applied.
  --width INT         NNBF hidden width for --mode train (default 6).
  --seed INT          Seed for --mode train.
  --widths LIST       Widths analysed (default 6). Width 6 is read from --p21-params first, then
                      from <params-dir>/inst0_nnbf_w6_s{S}.npz; other widths only from the latter.
  --seeds LIST        Seeds analysed (default 11,22,33). Missing parameter files are skipped (logged).
  --labels LIST       Labels analysed, subset of {P, P_S} (default P). P_S states get EHX = ZV = NaN
                      (no hybrids are formed); only P is scored by aggregate.
  --sr-iters INT (1200)  --sr-lr FLOAT (0.05)  --sr-shift FLOAT (1e-3)  --sr-samples INT (2000)
  --chunk INT (32768)  --o-chunk INT (8)        training settings (P2.1 values).
  --mc-samples INT    i.i.d. draws from |psi|^2 per sampled point (default 20000).
  --n-boot INT        bootstrap replicas (default 100).
  --minres-iters INT  ZRO iteration cap (default 3000).
  --no-inject         analyze: skip the injected check (aggregate then reports INJECT = missing,
                      which counts as INJECT not passed).
  --results DIR       output folder (default results/ next to this script).
  --params-dir DIR    trained parameter files (default results/params next to this script, independent
                      of --results, so a gate run with its own --results still finds trained seeds).
  --p21-params DIR    default ../P2.1_subcubic_amplitudes_r3/results/params
  --p21-results DIR   default ../P2.1_subcubic_amplitudes_r3/results (dE_sign premise)
Outputs
  train:     <params-dir>/inst0_nnbf_w{W}_s{S}.npz and <results>/train_w{W}_s{S}.json
  analyze:   <results>/analysis.json (rows: truth, every arm's estimate, SE_boot, SE_jk; inject rows)
  inject:    <results>/inject.json
  aggregate: printed table and verdict; <results>/aggregate.json
Sampling note: local estimators are read from exact vectors indexed at the sampled
configurations (identical to the per-sample local estimator; checked in --mode checks).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_core as rc  # noqa: E402
from p11_core import energy  # noqa: E402
from tj_core import TJ  # noqa: E402

TAU_M = 1e-3
TRUST = 0.5
SCORE_REL = 0.02
PREMISE_SIGN = 0.005
MIN_TRUSTED = 3
FORMS = ("S", "R", "E")            # S decides (pre-registered); R and E are reported
DECIDING = "S"
ROUTE = "EHX"                      # arm EHX = form S; EHX_R, EHX_E the other forms
COMPARATOR = "ZV"                  # decisive comparison: EHX vs ZV (both form S)
REPORTED = ("EHX_R", "EHX_E", "ZV_R", "ZV_E", "L1", "L2", "SD")
DIAGNOSTICS = ("OSGX", "ZRO")
ARMS = (ROUTE, COMPARATOR) + REPORTED + DIAGNOSTICS
INJ_WEIGHTS = (0.01, 0.02, 0.04)
INJ_EPS_AMP = 0.10
INJ_SEED = 777


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ training (copied from P2.1 r3)
def state_of(ans, m, sg, chunk):
    psi_t = ans.amplitudes(chunk)
    return psi_t, sg.project(psi_t)


def sr_step(ans, m, sg, args, rng):
    psi_t, psi_c = state_of(ans, m, sg, args.chunk)
    E, Hp, nn = energy(m.H, psi_c)
    f = ans.force((Hp - E * psi_c) / nn, args.chunk)
    pc = psi_c ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = ans.per_sample_O_proj(sg, idx, psi_c, args.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Km = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return E, f, step


def train(args):
    m, W, psi0, sg, S0, ob = rc.setup(4, log)
    ans = rc.nnbf(m, args.width, args.seed)
    rng = np.random.default_rng(args.seed + 1000 * 0)   # P2.1: ARM_LIST.index("nnbf") = 0
    best_E, best_x = np.inf, ans.flat().copy()
    t0 = time.time()
    for it in range(args.sr_iters):
        x0 = ans.flat().copy()
        E, f, step = sr_step(ans, m, sg, args, rng)
        if not np.isfinite(E):
            log(f"non-finite energy at it {it}; restoring best"); ans.set_flat(best_x); break
        if E < best_E:
            best_E, best_x = E, x0
        ans.set_flat(x0 - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"w{args.width} s{args.seed}: it {it:5d} E = {E:.8f} |f| = {np.linalg.norm(f):.2e} "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    E = energy(m.H, state_of(ans, m, sg, args.chunk)[1])[0]
    if not (E <= best_E):
        ans.set_flat(best_x)
    psiP = state_of(ans, m, sg, args.chunk)[1]
    psiP /= np.linalg.norm(psiP)
    Ef = float(psiP @ (m.H @ psiP))
    pd = args.results_params
    os.makedirs(pd, exist_ok=True)
    np.savez(os.path.join(pd, f"inst0_nnbf_w{args.width}_s{args.seed}.npz"), flat=ans.flat(),
             layout=json.dumps([[k, list(ans.p[k].shape)] for k in ans.keys]))
    out = dict(width=args.width, seed=args.seed, n_params=ans.n_params(), dE=Ef - W[0],
               ov0=float((psi0 @ psiP) ** 2), t=time.time() - t0)
    json.dump(out, open(os.path.join(args.results, f"train_w{args.width}_s{args.seed}.json"), "w"))
    log(f"trained: {out}")


# ============================================================================ estimators
def lanczos_vex(m, ob, psi, steps=2):
    """Exact Lanczos Ritz vectors phi_1, phi_2; returns (sigma^2, a) for psi, phi_1, phi_2."""
    basis = [psi]
    pts = []
    def pt(v):
        Hv = m.H @ v
        E = v @ Hv
        return float(Hv @ Hv - E * E), {k: ob.expect(k, v) for k in rc.OBS}
    pts.append(pt(psi))
    for p in range(steps):
        w = m.H @ basis[-1]
        for _ in range(2):
            for b in basis:
                w -= (b @ w) * b
        basis.append(w / np.linalg.norm(w))
        B = np.column_stack(basis)
        e, U = np.linalg.eigh(B.T @ (m.H @ B))
        phi = B @ U[:, 0]
        pts.append(pt(phi / np.linalg.norm(phi)))
    out = {}
    (s0, a0_), (s1, a1), (s2, a2) = pts
    for k in rc.OBS:
        L1 = a0_[k] + (a1[k] - a0_[k]) * s0 / (s0 - s1) - a0_[k]
        slope = (a2[k] - a1[k]) / (s2 - s1)
        L2 = a1[k] - slope * s1 - a0_[k]
        out[k] = dict(L1=float(L1), L2=float(L2))
    return out, pts


def sample_idx(p, n, rng):
    p = p / p.sum()
    return rng.choice(len(p), size=n, p=p)


def boot_fn(fn, n, n_boot, rng):
    vals = []
    for _ in range(n_boot):
        vals.append(fn(rng.integers(0, n, n)))
    vals = np.array(vals)
    return float(np.nanstd(vals, ddof=1))


def pinv_sym(M):
    M = 0.5 * (M + M.T)
    lam, U = np.linalg.eigh(M)
    keep = np.abs(lam) >= TAU_M * np.max(np.abs(lam))
    return (U[:, keep] / lam[keep]) @ U[:, keep].T, int(keep.sum())


def osg(m, ob, psi, obs, idx, n_boot, rng, single=False):
    """Route 5.1 (or SMA if single). Returns (delta_hat, SE, n_kept). One-sided estimators:
    c_p = mean[(1/2)((H-E)y_p + (A_p-a_p)r)(n)/psi(n)], M_pq = mean[((A_p-a_p)(H-E)y_q)(n)/psi(n)]
    (symmetrised after averaging), b_p = mean[((A_p-a_p)w)(n)/psi(n)]."""
    ps = psi[idx]
    Hpsi = m.H @ psi
    E = np.mean(Hpsi[idx] / ps)
    Apsi = ob.apply(obs, psi)
    a_t = np.mean(Apsi[idx] / ps)
    w = Apsi - a_t * psi
    r = Hpsi - E * psi
    if single:
        def dirs(v):
            return [ob.apply(obs, v)]
        Ppsi = [Apsi]
    else:
        def dirs(v):
            return ob.apply_pieces_all(obs, v)
        Ppsi = dirs(psi)
    ap = [np.mean(P[idx] / ps) for P in Ppsi]
    ys = [P - aj * psi for P, aj in zip(Ppsi, ap)]
    n = len(ys)
    Kys = [m.H @ y - E * y for y in ys]
    Ar = dirs(r)
    cS = np.array([0.5 * (Kys[p][idx] + (Ar[p] - ap[p] * r)[idx]) / ps for p in range(n)])
    MS = np.zeros((n, n, len(idx)))
    for q in range(n):
        AK = dirs(Kys[q])
        for p in range(n):
            MS[p, q] = (AK[p] - ap[p] * Kys[q])[idx] / ps
    Aw = dirs(w)
    bS = np.array([(Aw[p] - ap[p] * w)[idx] / ps for p in range(n)])

    # noise-aware truncation (fixed in advance): eigen-directions of the full-sample M are kept
    # iff |lambda| >= TAU_M max|lambda| and |lambda| >= 3 x its bootstrap SE (exact linear
    # dependences among the pieces, e.g. sum of all charge pieces = const, become pure noise
    # under sampling and must not be inverted).
    Mfull = MS.mean(2); Mfull = 0.5 * (Mfull + Mfull.T)
    lam, U = np.linalg.eigh(Mfull)
    lb = []
    for _ in range(max(n_boot, 20)):
        s_ = rng.integers(0, len(idx), len(idx))
        Mb = MS[:, :, s_].mean(2); Mb = 0.5 * (Mb + Mb.T)
        lb.append(np.einsum("ip,ij,jp->p", U, Mb, U))
    lse = np.std(np.array(lb), axis=0, ddof=1)
    keep = (np.abs(lam) >= TAU_M * np.max(np.abs(lam))) & (np.abs(lam) >= 3 * lse)
    Uk = U[:, keep]

    def est(sel):
        c = cS[:, sel].mean(1); M = MS[:, :, sel].mean(2); b = bS[:, sel].mean(1)
        M = 0.5 * (M + M.T)
        Mk = Uk.T @ M @ Uk
        return float(-2.0 * (Uk.T @ c) @ np.linalg.solve(Mk, Uk.T @ b)) if Uk.shape[1] else 0.0
    d = est(np.arange(len(idx)))
    se = boot_fn(est, len(idx), n_boot, rng)
    return d, se, int(keep.sum())


def osg_exact(m, ob, psi, obs):
    """Diagnostic OSGX: the same Galerkin estimate with exact matrix elements (pinv, TAU_M)."""
    E = psi @ (m.H @ psi)
    w = ob.apply(obs, psi) - ob.expect(obs, psi) * psi
    Pp = ob.apply_pieces_all(obs, psi)
    Y = np.array([P - (psi @ P) * psi for P in Pp])
    M = Y @ np.array([m.H @ y - E * y for y in Y]).T
    Mi, _ = pinv_sym(M)
    r = m.H @ psi - E * psi
    return float(-2.0 * (Y @ r) @ Mi @ (Y @ w))


def point_stats(m, ob, h, idx):
    """Sampled E, sigma^2, a for state h with samples idx ~ |h|^2: arrays per sample."""
    hs = h[idx]
    El = (m.H @ h)[idx] / hs
    Al = {k: ob.apply(k, h)[idx] / hs for k in rc.OBS}
    return El, Al


def fit_forms(points, sels):
    """points: list of (El, Al) per-sample arrays; sels: per-point index arrays.
    Returns {form: {obs: a_hat}} for the pre-registered forms S, R, E, or None if < 2 points."""
    E = []; S = []; A = {k: [] for k in rc.OBS}
    for (El, Al), sel in zip(points, sels):
        e = El[sel]
        E.append(e.mean()); S.append(e.var())
        for k in rc.OBS:
            A[k].append(Al[k][sel].mean())
    E, S = np.array(E), np.array(S)
    if len(E) < 2:
        return None
    sE, E0 = np.polyfit(S, E, 1)
    sig = np.sqrt(S)
    rt = np.sqrt(np.maximum(E - E0, 0.0))
    out = {f: {} for f in FORMS}
    for k in rc.OBS:
        a = np.array(A[k])
        out["S"][k] = float(np.polyfit(sig, a, 1)[1]) if np.ptp(sig) > 0 else float("nan")
        out["R"][k] = float(np.polyfit(rt, a, 1)[1]) if np.ptp(rt) > 0 else float("nan")
        beta, alpha = np.polyfit(E, a, 1)
        out["E"][k] = float(alpha + beta * E0)
    return out


def ensemble_extrap(base, partners, n_boot, rng):
    """base: points always present (psi_k's own); partners: one point-list per partner seed (its
    plain point, or its two hybrids). Returns {form: {obs: (a_hat, se_boot, se_jk)}}.
    se_boot: MC bootstrap (every point resampled); se_jk: leave-one-partner-out jackknife."""
    pts = base + [p for grp in partners for p in grp]
    nan = {f: {k: (float("nan"),) * 3 for k in rc.OBS} for f in FORMS}
    if len(partners) == 0:
        return nan
    full = fit_forms(pts, [np.arange(len(p[0])) for p in pts])
    reps = [fit_forms(pts, [rng.integers(0, len(p[0]), len(p[0])) for p in pts]) for _ in range(n_boot)]
    jk = []
    if len(partners) >= 2:
        for i in range(len(partners)):
            sub = base + [p for j, grp in enumerate(partners) if j != i for p in grp]
            jk.append(fit_forms(sub, [np.arange(len(p[0])) for p in sub]))
    out = {}
    for f in FORMS:
        out[f] = {}
        for k in rc.OBS:
            sb = float(np.nanstd([r[f][k] for r in reps], ddof=1))
            if jk:
                th = np.array([r[f][k] for r in jk]); n = len(th)
                sj = float(np.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2)))
            else:
                sj = float("nan")
            out[f][k] = (full[f][k], sb, sj)
    return out


def ehx_zv(m, ob, psi, idx, plain_k, partners, n_boot, rng):
    """EHX and ZV for state psi (samples idx, plain point plain_k). partners: list of
    (psi_j, idx_j, plain_j) from the other seeds. Returns {arm: {obs: dict(a, se_boot, se_jk)}}."""
    hyb = []
    for psj, idj, plj in partners:
        h1 = np.abs(psi) * np.sign(psj); h1 /= np.linalg.norm(h1)
        h2 = np.abs(psj) * np.sign(psi); h2 /= np.linalg.norm(h2)
        hyb.append([point_stats(m, ob, h1, idx), point_stats(m, ob, h2, idj)])
    res = {"EHX": ensemble_extrap([plain_k], hyb, n_boot, rng),
           "ZV": ensemble_extrap([plain_k], [[plj] for _, _, plj in partners], n_boot, rng)}
    out = {}
    for base, r in res.items():
        for f in FORMS:
            arm = base if f == DECIDING else f"{base}_{f}"
            out[arm] = {k: dict(a=r[f][k][0], se_boot=r[f][k][1], se_jk=r[f][k][2]) for k in rc.OBS}
    return out


def est_entry(a_hat, a_s, se_boot, se_jk):
    """delta_hat = a_hat - a_s; SE = sqrt(SE_boot^2 + SE_jk^2) (SE_jk = 0 when undefined)."""
    se = float(np.sqrt(se_boot ** 2 + (se_jk if np.isfinite(se_jk) else 0.0) ** 2))
    return dict(d=float(a_hat - a_s), se=se, se_boot=float(se_boot), se_jk=float(se_jk))


def injected_states(psi0, sg, rng):
    """Synthetic ensemble: psi_0 with sign flips on whole G-orbits (flipped weight INJ_WEIGHTS,
    orbits drawn with probability proportional to orbit weight) and a G-invariant amplitude
    distortion exp(INJ_EPS_AMP xi(orbit)). Returns list of (psi_i, w_flipped, |P_chi psi_i - psi_i|)."""
    rep = sg.inv.min(axis=0)            # orbit representative of every configuration
    D = len(psi0)
    ow = np.bincount(rep, weights=psi0 ** 2, minlength=D)
    reps = np.nonzero(ow > 0)[0]
    out = []
    for wt in INJ_WEIGHTS:
        keys = np.log(ow[reps]) + rng.gumbel(size=len(reps))
        order = reps[np.argsort(-keys)]
        nsel = int(np.searchsorted(np.cumsum(ow[order]), wt)) + 1
        fr = np.zeros(D, bool); fr[order[:nsel]] = True
        flip = fr[rep]
        xi = rng.standard_normal(D)[rep]
        v = psi0 * np.exp(INJ_EPS_AMP * xi) * np.where(flip, -1.0, 1.0)
        v /= np.linalg.norm(v)
        dev = float(np.linalg.norm(sg.project(v) - v))
        assert dev < 1e-8, f"injected state left the chi irrep: |P v - v| = {dev:.2e}"
        out.append((v, float(np.sum(psi0[flip] ** 2)), dev))
    return out


def inject_check(m, ob, psi0, sg, a0, args, rng):
    """Injected-sign-flip correctness check (module docstring). Returns rows like analyze's."""
    inj = injected_states(psi0, sg, np.random.default_rng(INJ_SEED))
    vs = [v for v, _, _ in inj]
    smp = [sample_idx(v ** 2, args.mc_samples, rng) for v in vs]
    plain = [point_stats(m, ob, v, s) for v, s in zip(vs, smp)]
    rows = []
    for i, (v, wf, dev) in enumerate(inj):
        partners = [(vs[j], smp[j], plain[j]) for j in range(len(vs)) if j != i]
        ez = ehx_zv(m, ob, v, smp[i], plain[i], partners, args.n_boot, rng)
        a = ob.all(v)
        ov = float((psi0 @ v) ** 2)
        for obs in rc.OBS:
            a_s = float(np.mean(plain[i][1][obs]))
            est = {arm: est_entry(ez[arm][obs]["a"], a_s, ez[arm][obs]["se_boot"], ez[arm][obs]["se_jk"])
                   for arm in ez}
            d = a0[obs] - a[obs]
            rows.append(dict(inj=i, w_flip=wf, irrep_dev=dev, obs=obs, a0=a0[obs], a=a[obs],
                             delta=d, ov0=ov, est=est))
            log(f"inject {i} (w_flip={wf:.4f}, ov0={ov:.4f}, irrep dev {dev:.1e}) {obs}: delta={d:+.3e} "
                f"(rel {abs(d) / abs(a0[obs]):.3f}) | "
                + " ".join(f"{k}={e['d']:+.2e}+-{e['se']:.1e}" for k, e in est.items()))
    return rows


def inject_verdict(rows):
    sc = [r for r in rows if abs(r["delta"]) >= SCORE_REL * abs(r["a0"])]
    tr = {arm: sum(classify(arm, r) == "tracked" for r in sc) for arm in ("EHX", "ZV", "EHX_R", "EHX_E")}
    ok = len(sc) >= 3 and tr["EHX"] >= 0.5 * len(sc)
    return dict(scored=len(sc), tracked=tr, passed=bool(ok),
                classes={arm: [classify(arm, r) for r in sc] for arm in ("EHX", "ZV")})


def inject_mode(args):
    m, W, psi0, sg, S0, ob = rc.setup(4, log)
    a0 = ob.all(psi0)
    rows = inject_check(m, ob, psi0, sg, a0, args, np.random.default_rng(2027))
    os.makedirs(args.results, exist_ok=True)
    v = inject_verdict(rows)
    json.dump(dict(a0=a0, inject_rows=rows, verdict=v, args=vars(args)),
              open(os.path.join(args.results, "inject.json"), "w"), indent=1)
    print(f"INJECT: scored {v['scored']}; tracked {v['tracked']}; EHX classes {v['classes']['EHX']}; "
          f"ZV classes {v['classes']['ZV']} -> {'PASS' if v['passed'] else 'FAIL'}")


def zro(m, ob, psi, obs, maxiter):
    E = psi @ (m.H @ psi)
    w = ob.apply(obs, psi); w -= (psi @ w) * psi
    r = m.H @ psi - E * psi

    def Kt(v):
        v = v - (psi @ v) * psi
        y = m.H @ v - E * v
        return y - (psi @ y) * psi
    L = sla.LinearOperator((m.D, m.D), matvec=Kt, dtype=float)
    x, info = sla.minres(L, w, maxiter=maxiter)
    return float(-2.0 * r @ x), int(info)


def param_file(args, wdt, seed):
    """Width 6: P2.1 file first, then this test's trained file; other widths: this test only."""
    cands = []
    if wdt == 6:
        cands.append(os.path.join(args.p21_params, f"inst0_nnbf_s{seed}.npz"))
    cands.append(os.path.join(args.results_params, f"inst0_nnbf_w{wdt}_s{seed}.npz"))
    for f in cands:
        if os.path.exists(f):
            return f
    return None


def analyze(args):
    m, W, psi0, sg, S0, ob = rc.setup(4, log)
    a0 = ob.all(psi0)
    log("a0 " + json.dumps(a0))
    rng = np.random.default_rng(2026)
    labels = [x for x in args.labels.split(",") if x]
    assert set(labels) <= {"P", "P_S"}, labels
    states = {}
    for wdt in [int(x) for x in args.widths.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            f = param_file(args, wdt, seed)
            if f is None:
                log(f"missing parameters for width {wdt} seed {seed}; skipped"); continue
            ans = rc.load_params(rc.nnbf(m, wdt, seed), f)
            psiP = sg.project(ans.amplitudes(args.chunk)); psiP /= np.linalg.norm(psiP)
            vs = {"P": psiP}
            if "P_S" in labels:
                psiPS = m.spin_project(psiP, S0); vs["P_S"] = psiPS / np.linalg.norm(psiPS)
            for lab in labels:
                v = vs[lab]
                if v @ psi0 < 0:
                    v = -v
                states[(wdt, seed, lab)] = v
            log(f"loaded {f}")
    log(f"{len(states)} states: {sorted(states)}")
    dsign = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        jf = os.path.join(args.p21_results, f"inst0_nnbf_s{seed}.json")
        if os.path.exists(jf):
            dsign.append(json.load(open(jf))["budgets"]["P"]["dE_sign"])
    samples = {key: sample_idx(v ** 2, args.mc_samples, rng) for key, v in states.items()}
    plain = {key: point_stats(m, ob, v, samples[key]) for key, v in states.items()}
    rows = []
    for key, psi in states.items():
        t0 = time.time()
        wdt, seed, lab = key
        E = float(psi @ (m.H @ psi))
        a = ob.all(psi)
        lz, _ = lanczos_vex(m, ob, psi)
        # EHX / ZV partners: other seeds, same width and label (leave-own-seed-out). No P_S hybrids.
        others = [k2 for k2 in states if k2[0] == wdt and k2[2] == lab and k2[1] != seed]
        if lab == "P":
            ez = ehx_zv(m, ob, psi, samples[key], plain[key],
                        [(states[k2], samples[k2], plain[k2]) for k2 in others], args.n_boot, rng)
        else:
            ez = None
        for obs in rc.OBS:
            d_true = a0[obs] - a[obs]
            sd_vals = [ob.expect(obs, states[k2]) for k2 in states if k2[0] == wdt and k2[2] == lab]
            est = {}
            a_s = float(np.mean(plain[key][1][obs]))
            for arm in ("EHX", "ZV", "EHX_R", "EHX_E", "ZV_R", "ZV_E"):
                if ez is None:
                    est[arm] = dict(d=float("nan"), se=float("nan"), se_boot=float("nan"), se_jk=float("nan"))
                else:
                    e = ez[arm][obs]
                    est[arm] = est_entry(e["a"], a_s, e["se_boot"], e["se_jk"])
            est["L1"] = dict(d=lz[obs]["L1"], se=0.0)
            est["L2"] = dict(d=lz[obs]["L2"], se=0.0)
            est["SD"] = dict(d=float(np.std(sd_vals, ddof=1)) if len(sd_vals) > 1 else float("nan"), se=0.0)
            est["OSGX"] = dict(d=osg_exact(m, ob, psi, obs), se=0.0)
            d, info = zro(m, ob, psi, obs, args.minres_iters)
            est["ZRO"] = dict(d=d, se=0.0, info=info)
            rows.append(dict(width=wdt, seed=seed, label=lab, obs=obs, a0=a0[obs], a=a[obs], delta=d_true,
                             dE=E - W[0], ov0=float((psi0 @ psi) ** 2), n_partners=len(others), est=est))
            log(f"w{wdt} s{seed} {lab} {obs}: delta={d_true:+.3e} (rel {abs(d_true) / abs(a0[obs]):.3f}) | "
                + " ".join(f"{k}={v['d']:+.2e}" for k, v in est.items())
                + f" | SE(EHX) boot {est['EHX'].get('se_boot', float('nan')):.1e} jk {est['EHX'].get('se_jk', float('nan')):.1e}")
        log(f"state {key} done ({time.time() - t0:.0f} s)")
    inj = [] if args.no_inject else inject_check(m, ob, psi0, sg, a0, args, rng)
    out = dict(a0=a0, ed=W.tolist(), dE_sign_w6=dsign, rows=rows, inject_rows=inj, args=vars(args))
    os.makedirs(args.results, exist_ok=True)
    json.dump(out, open(os.path.join(args.results, "analysis.json"), "w"), indent=1)


# ============================================================================ aggregate
def classify(arm, row):
    e = row["est"].get(arm)
    if e is None:
        return "abstain"
    d, dh = row["delta"], e["d"]
    if not np.isfinite(dh):
        return "abstain"
    se = e.get("se", 0.0)
    if arm != "SD" and (not np.isfinite(se) or se > TRUST * abs(dh)):
        return "untrusted"
    ratio = abs(dh) / abs(d) if arm == "SD" else dh / d
    if 0.5 <= ratio <= 2.0:
        return "tracked"
    if ratio < 0.5:
        return "false"
    return "over"


def g_false(cl):
    """G = false / trusted items; None (undefined) if fewer than MIN_TRUSTED trusted items."""
    nt = sum(c in ("tracked", "false", "over") for c in cl)
    return (sum(c == "false" for c in cl) / nt if nt >= MIN_TRUSTED else None), nt


def aggregate(args):
    D = json.load(open(os.path.join(args.results, "analysis.json")))
    allP = [r for r in D["rows"] if r["label"] == "P"]          # one label per network is scored
    rows = [r for r in allP if abs(r["delta"]) >= SCORE_REL * abs(r["a0"])]
    seeds = sorted({r["seed"] for r in rows})
    med_sign = float(np.median(D["dE_sign_w6"])) if D["dE_sign_w6"] else float("nan")
    print(f"rule {args.rule}; label P only; scored items {len(rows)} of {len(allP)} over seeds {seeds}; "
          f"width-6 control median dE_sign = {med_sign:.4f}")
    print("note: per-seed EHX/ZV estimates share partner data (each seed's fit uses every other seed).")
    for obs in rc.OBS:
        rr = [r for r in allP if r["obs"] == obs]
        if rr:
            print(f"  {obs}: a0={rr[0]['a0']:+.3e} median |delta|/|a0| = "
                  f"{np.median([abs(r['delta']) / abs(r['a0']) for r in rr]):.3f}, scored {sum(1 for r in rows if r['obs'] == obs)}")
    inj = inject_verdict(D["inject_rows"]) if D.get("inject_rows") else None
    print("INJECT: " + ("missing (counts as not passed)" if inj is None else
                        f"scored {inj['scored']}, tracked {inj['tracked']}, EHX {inj['classes']['EHX']} -> "
                        f"{'PASS' if inj['passed'] else 'FAIL'}"))
    inj_ok = bool(inj and inj["passed"])
    if not (med_sign >= PREMISE_SIGN and len(rows) >= 6 and len(seeds) >= 3):
        print("VERDICT: NO-TARGET (premise gate)")
        json.dump(dict(verdict="NO-TARGET", inject=inj), open(os.path.join(args.results, "aggregate.json"), "w"), indent=1)
        return
    T, G = {}, {}
    print("arm    role        M      G (false/trusted, n_trusted)   per-seed tracked fractions")
    for arm in ARMS:
        cl = [classify(arm, r) for r in rows]
        T[arm] = {s: sum(c == "tracked" for c, r in zip(cl, rows) if r["seed"] == s)
                  / sum(1 for r in rows if r["seed"] == s) for s in seeds}
        G[arm] = g_false(cl)
        role = ("decides" if arm == ROUTE else "comparator" if arm == COMPARATOR
                else "diagnostic" if arm in DIAGNOSTICS else "reported")
        gs = "undef" if G[arm][0] is None else f"{G[arm][0]:.3f}"
        print(f"{arm:6s} {role:10s} {np.median(list(T[arm].values())):.3f}  {gs:>5s} (n={G[arm][1]:2d})          "
              + " ".join(f"{s}:{T[arm][s]:.2f}" for s in seeds))
    gains = [T[ROUTE][s] - T[COMPARATOR][s] for s in seeds]
    M = float(np.median(list(T[ROUTE].values()))); Dl = float(np.median(gains))
    g, nt = G[ROUTE]
    g_ok = g is not None and g <= 0.10
    if args.rule == "gate":
        ok = inj_ok and M >= 0.5 and Dl > 0 and g_ok
        v = ("GATE-PASS: train >= 8 more width-6 seeds (submit_train.sbatch), then --rule full" if ok else
             "GATE-FAIL: EHX = Route 2.3 restated; mark the leaf for mutation; no training compute")
    else:
        if inj_ok and M >= 0.5 and Dl >= 0.2 and min(gains) > 0 and g_ok:
            v = "PASS"
        elif (not inj_ok) or M < 0.3 or Dl < 0.05 or (g is not None and g > 0.20):
            v = "FAIL"
        else:
            v = "AMBIGUOUS (rerun --mc-samples 500000; still not PASS -> killed)"
    res = dict(rule=args.rule, M=M, Delta=Dl, gains=gains, G=g, n_trusted=nt, inject=inj,
               comparator=COMPARATOR, verdict=v, T={k: {str(s): x for s, x in t.items()} for k, t in T.items()})
    print(f"EHX (form S): M={M:.3f} vs ZV M={np.median(list(T[COMPARATOR].values())):.3f}; paired median gain "
          f"{Dl:+.3f} (per-seed {['%+.2f' % x for x in gains]}); G={'undef' if g is None else '%.3f' % g} "
          f"(n_trusted={nt}); INJECT {'PASS' if inj_ok else 'not passed'}")
    print("VERDICT: " + v)
    json.dump(res, open(os.path.join(args.results, "aggregate.json"), "w"), indent=1)


# ============================================================================ checks
def checks(args):
    """Small-cluster checks: (1) Galerkin OSG with exact-vector estimators equals the dense
    formula; (2) with the full-space basis limit the Galerkin estimate equals ZRO; (3) the
    first-order formula approaches delta for a slightly perturbed ground state; (4) indexed
    local estimators equal the direct per-sample sum over the sparse row."""
    import scipy.sparse as sp
    m = TJ(4, 4, "obc", "obc", 2, 2, 1.0, -0.25, 0.4)
    m.spin_setup()
    W, Vv, _ = rc.solve_ed(m, 2, 1e-8, 1)
    psi0 = Vv[:, 0] / np.linalg.norm(Vv[:, 0])
    sg, _ = rc.build_symmetry(m, psi0)
    ob = rc.Observables(m)
    rng = np.random.default_rng(1)
    for eps in (1e-2, 1e-3):
        pert = sg.project(psi0 * (1 + eps * rng.standard_normal(m.D)))
        psi = pert / np.linalg.norm(pert)
        for obs in rc.OBS:
            d_true = ob.expect(obs, psi0) - ob.expect(obs, psi)
            dz, info = zro(m, ob, psi, obs, 5000)
            log(f"check first-order oracle eps={eps} {obs}: delta {d_true:+.6e} ZRO {dz:+.6e} (minres info {info})")
    # OSG with all samples = full enumeration weighting (use idx = all configs, weights |psi|^2)
    psi = sg.project(psi0 * (1 + 0.05 * rng.standard_normal(m.D))); psi /= np.linalg.norm(psi)
    idx = sample_idx(psi ** 2, 400000, rng)
    for obs in rc.OBS:
        d_mc, se, nk = osg(m, ob, psi, obs, idx, 20, rng)
        d_ex = osg_exact(m, ob, psi, obs)
        d_z, _ = zro(m, ob, psi, obs, 5000)
        d_t = ob.expect(obs, psi0) - ob.expect(obs, psi)
        log(f"check OSG {obs}: pieces {len(ob.pieces[obs])} kept {nk} sampled {d_mc:+.4e} +- {se:.1e} "
            f"vs exact-Galerkin {d_ex:+.4e} | ZRO {d_z:+.4e} | true delta {d_t:+.4e}")
    # indexed local estimator vs direct sparse-row sum
    k = int(idx[0])
    row = m.H.getrow(k)
    direct = float((row @ psi)[0]) / psi[k]
    log(f"check local energy indexed vs sparse row: {(m.H @ psi)[k] / psi[k]:.12e} vs {direct:.12e}")
    # injected-sign-flip machinery on the small cluster (irrep closure, known delta, EHX/ZV fits)
    a0s = ob.all(psi0)
    v = inject_verdict(inject_check(m, ob, psi0, sg, a0s, args, rng))
    log(f"check inject (small cluster): scored {v['scored']} tracked {v['tracked']} EHX {v['classes']['EHX']} "
        f"-> {'PASS' if v['passed'] else 'FAIL'}")
    log("P3.1-r5 CHECKS DONE")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="checks", choices=["checks", "inject", "train", "analyze", "aggregate"])
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--widths", default="6")
    ap.add_argument("--labels", default="P")
    ap.add_argument("--rule", default="gate", choices=["gate", "full"])
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--sr-iters", type=int, default=1200)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--mc-samples", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--minres-iters", type=int, default=3000)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--params-dir", dest="results_params", default=os.path.join(HERE, "results", "params"))
    ap.add_argument("--p21-params", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params"))
    ap.add_argument("--p21-results", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"checks": checks, "inject": inject_mode, "train": train, "analyze": analyze, "aggregate": aggregate}[args.mode](args)


if __name__ == "__main__":
    main()
