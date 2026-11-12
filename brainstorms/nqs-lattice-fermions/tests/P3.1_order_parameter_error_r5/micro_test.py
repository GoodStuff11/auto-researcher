#!/usr/bin/env python3
"""
P3.1 round-5 decisive test: error of long-range correlations of CONVERGED symmetry-projected NNBF
states on the shared 4x4 OBC t-t'-J instance, from cheap information
==================================================================================================

Instance (r5_core.INST0): 4x4 OBC, N_up = N_dn = 6, t' = -0.25, J = 0.4, D = 1681680, psi_0 a
non-degenerate singlet (Mx = My = +1, D = -1, |G| = 8). Sign half of the Hard Core only (see the
artifact for the in-irrep spectrum measured by target_probe.py).
Observables (r5_core.Observables): long-range d-wave pair correlation (bond-centre distance >= 3),
spin correlation S_i.S_j and charge correlation (n_i - nbar)(n_j - nbar) at site distance >= 3.5.

Population (all CONVERGED finals; no snapshots)
-----------------------------------------------
NNBF (P2.1 r3 arm "nnbf": full backflow, K = 1, Ha = 0) of hidden width W in {3, 6, 12}, seeds
{11, 22, 33}, trained with P_chi-projected SR (exact force, sampled metric; P2.1 settings: 1200
steps, step 0.05, shift 1e-3, 2000 samples, best-energy iterate). Width 6 is read from the P2.1
round-3 parameter files (jobs 410988 and 451706); widths 3 and 12 are trained here (--mode train).
Each trained state gives two verdict states: "P" (P_chi psi_theta) and "P_S" (its post-hoc
total-spin projection). 9 networks x 2 labels = 18 states x 3 observables.

Routes (P3.1 artifact, round 5)
-------------------------------
  OSG  Route 5.1 (Operation A on ZR: the trained response network is replaced by a closed-form
       Galerkin solve). Zero-bias / first-order correction delta ~ -2 <r|Kt^-1|w>, r = (H-E)psi,
       w = (A-a)psi, Kt = Q(H-E)Q, solved in the span of the observable's OWN resolved pieces
       y_c = (A_c - a_c) psi, one per symmetry-invariant distance class c of the same correlator
       (the full measured correlation function C(r), not powers of H):
           delta_hat = -2 c^T M^+ b,  c_p = <r|y_p>,  M_pq = <y_p|Kt|y_q>,  b_p = <y_p|w>.
       All three are sampled from |psi|^2 with ONE-SIDED local estimators X(n)/psi(n)
       (X = symmetrised (H-E)(A_p-a_p)psi, (A_p-a_p)(H-E)(A_q-a_q)psi, (A_p-a_p)(A-a)psi), cost
       O(N_A^2 N_H) connections per sample; no extra network, no training, nothing fitted.
       M is inverted on the eigen-directions of the full-sample M with |lambda| >= 1e-3 max|lambda|
       AND |lambda| >= 3 x its bootstrap SE (fixed in advance; exact linear dependences among the
       pieces, e.g. sum of all charge pieces = const, are otherwise inverted noise).
  EHX  Route 5.2 (Operation B, ensemble sign/amplitude hybrids). For state k (label L), take every
       ensemble member j of the same label from a DIFFERENT seed (leave-own-seed-out), and form the
       hybrids |psi_k| sgn psi_j (sampled from |psi_k|^2) and |psi_j| sgn psi_k (sampled from
       |psi_j|^2). These need only the two networks' amplitudes at the sampled and connected
       configurations: no cross-overlap, no heavy-tailed ratio psi_j/psi_k. Over the point set
       {psi_k} U hybrids, fit E = E0_hat + s sigma^2 (zero-variance energy) and a = alpha + beta E;
       delta_hat = alpha + beta E0_hat - a_k. Nothing is fitted to the truth.
Controls (the PASS must beat the best of them)
  L1   one Lanczos step: two-point linear extrapolation of a in sigma^2 through psi and phi_1
       (exact matrix elements, generous to the control).
  L2   two Lanczos steps: linear fit in sigma^2 through phi_1 and phi_2, extrapolated to 0 (exact).
  SD   seed spread of a at the same (width, label): unsigned |delta_hat| = std over seeds.
  ZV   the EHX fit restricted to the plain ensemble members (no hybrids), leave-own-seed-out
       (isolates what the hybrids add); sampled.
Ablations / premise
  SMA  OSG with the single basis vector w (single-mode Galerkin = covariance / single-mode scale).
  OSGX OSG with exact matrix elements (diagnostic: separates sampling noise from mechanism).
  ZRO  exact first-order oracle -2 <r|Kt^-1|w> (MINRES). Premise of OSG: if ZRO tracks < 70 % of
       scored items, OSG is "FAIL (premise)".

Metric and pre-committed rule (implemented in --mode aggregate)
--------------------------------------------------------------
Item = (state, observable). delta = a_0 - a_psi (exact). Scored iff |delta| >= 0.02 |a_0|.
Trusted: SE_boot(delta_hat) <= 0.5 |delta_hat| (sampled arms; exact arms always trusted).
Tracked: trusted and 1/2 <= delta_hat/delta <= 2 (|.| for SD). False: trusted and
delta_hat/delta < 1/2 (includes the wrong sign). Unit = seed (3 units; each unit holds its
widths x labels x observables). T_X(u) = tracked fraction of unit u (abstentions = misses),
M_X = median_u T_X(u); B = control with the largest M among L1, L2, SD, ZV;
Delta_X = median_u [T_X(u) - T_B(u)], spread = max-min over units of T_X(u) - T_B(u).
  PASS:  M_X >= 0.5, Delta_X >= 0.2, min_u [T_X(u) - T_B(u)] > 0 (positive on every seed),
         G_X (pooled false-certificate fraction) <= 0.10, and for OSG the ZRO premise holds.
  FAIL:  M_X < 0.3 or Delta_X < 0.05 or G_X > 0.20.
  AMBIGUOUS otherwise; pre-committed action: rerun --mode analyze with --mc-samples 200000;
  a route still not PASS is killed.
PREMISE GATE (NO-TARGET): the run counts only if the width-6 projected control's median dE_sign
over its seeds is >= 0.005 t (P2.1 JSONs) AND >= 6 items are scored over >= 3 seeds. Otherwise
both routes are NO-TARGET.

Usage
-----
  python3 micro_test.py --mode checks                  correctness checks (small clusters; ~1 min)
  python3 micro_test.py --mode train --width W --seed S   one projected-SR training (hours)
  python3 micro_test.py --mode analyze [--widths 3,6,12] [--seeds 11,22,33]   all arms (hours)
  python3 micro_test.py --mode aggregate [--results DIR]  verdict (seconds)
Options
  --mode {checks,train,analyze,aggregate}  (default checks)
  --width INT         NNBF hidden width for --mode train (3 or 12 in the decisive run).
  --seed INT          Seed for --mode train.
  --widths LIST       Widths analysed (default 3,6,12). Width 6 is read from --p21-params.
  --seeds LIST        Seeds analysed (default 11,22,33). Missing parameter files are skipped.
  --sr-iters INT (1200)  --sr-lr FLOAT (0.05)  --sr-shift FLOAT (1e-3)  --sr-samples INT (2000)
  --chunk INT (32768)  --o-chunk INT (8)        training settings (P2.1 values).
  --mc-samples INT    i.i.d. draws from |psi|^2 per sampled point (default 20000).
  --n-boot INT        bootstrap replicas (default 100).
  --minres-iters INT  ZRO iteration cap (default 3000).
  --results DIR       default results/ next to this script.
  --p21-params DIR    default ../P2.1_subcubic_amplitudes_r3/results/params
  --p21-results DIR   default ../P2.1_subcubic_amplitudes_r3/results (dE_sign premise)
Outputs
  train:     results/params/inst0_nnbf_w{W}_s{S}.npz and results/train_w{W}_s{S}.json
  analyze:   results/analysis.json (per item: truth, every arm's estimate, SE, diagnostics)
  aggregate: printed table and verdict; results/aggregate.json
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
CONTROLS = ("L1", "L2", "SD", "ZV")
ROUTES = ("OSG", "EHX")
ARMS = ROUTES + CONTROLS + ("SMA", "OSGX", "ZRO")


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
    pd = os.path.join(args.results, "params")
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


def zv_fit(points, sel_fn):
    """points: list of (El, Al) per-sample arrays. Returns {obs: delta-free extrapolated a}."""
    E = []; S = []; A = {k: [] for k in rc.OBS}
    for (El, Al), sel in zip(points, sel_fn):
        e = El[sel]
        E.append(e.mean()); S.append(e.var())
        for k in rc.OBS:
            A[k].append(Al[k][sel].mean())
    E, S = np.array(E), np.array(S)
    if len(E) < 3:
        return None
    sE, E0 = np.polyfit(S, E, 1)
    out = {}
    for k in rc.OBS:
        beta, alpha = np.polyfit(E, np.array(A[k]), 1)
        out[k] = float(alpha + beta * E0)
    return out


def ensemble_extrap(points, n_boot, rng):
    ns = [len(p[0]) for p in points]
    full = zv_fit(points, [np.arange(n) for n in ns])
    if full is None:
        nan = {k: float("nan") for k in rc.OBS}
        return nan, nan
    reps = []
    for _ in range(n_boot):
        r = zv_fit(points, [rng.integers(0, n, n) for n in ns])
        reps.append(r)
    se = {k: float(np.std([r[k] for r in reps], ddof=1)) for k in rc.OBS}
    return full, se


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


def analyze(args):
    m, W, psi0, sg, S0, ob = rc.setup(4, log)
    a0 = ob.all(psi0)
    log("a0 " + json.dumps(a0))
    rng = np.random.default_rng(2026)
    states = {}
    for wdt in [int(x) for x in args.widths.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            f = (os.path.join(args.p21_params, f"inst0_nnbf_s{seed}.npz") if wdt == 6
                 else os.path.join(args.results, "params", f"inst0_nnbf_w{wdt}_s{seed}.npz"))
            if not os.path.exists(f):
                log(f"missing {f}; skipped"); continue
            ans = rc.load_params(rc.nnbf(m, wdt, seed), f)
            psiP = sg.project(ans.amplitudes(args.chunk)); psiP /= np.linalg.norm(psiP)
            psiPS = m.spin_project(psiP, S0); psiPS /= np.linalg.norm(psiPS)
            for lab, v in (("P", psiP), ("P_S", psiPS)):
                if v @ psi0 < 0:
                    v = -v
                states[(wdt, seed, lab)] = v
    log(f"{len(states)} states")
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
        idx = samples[key]
        # EHX hybrids and ZV ensemble (other seeds, same label)
        others = [k2 for k2 in states if k2[2] == lab and k2[1] != seed]
        hyb = [plain[key]]
        for k2 in others:
            psj = states[k2]
            h1 = np.abs(psi) * np.sign(psj)
            h1 /= np.linalg.norm(h1)
            hyb.append(point_stats(m, ob, h1, idx))
            h2 = np.abs(psj) * np.sign(psi)
            h2 /= np.linalg.norm(h2)
            hyb.append(point_stats(m, ob, h2, samples[k2]))
        ehx, ehx_se = ensemble_extrap(hyb, args.n_boot, rng)
        zv, zv_se = ensemble_extrap([plain[key]] + [plain[k2] for k2 in others], args.n_boot, rng)
        for obs in rc.OBS:
            d_true = a0[obs] - a[obs]
            sd_vals = [ob.expect(obs, states[k2]) for k2 in states if k2[0] == wdt and k2[2] == lab]
            est = {}
            d, se, nk = osg(m, ob, psi, obs, idx, args.n_boot, rng)
            est["OSG"] = dict(d=d, se=se, nk=nk)
            d, se, _ = osg(m, ob, psi, obs, idx, args.n_boot, rng, single=True)
            est["SMA"] = dict(d=d, se=se)
            a_s = float(np.mean(plain[key][1][obs]))
            est["EHX"] = dict(d=ehx[obs] - a_s, se=ehx_se[obs])
            est["ZV"] = dict(d=zv[obs] - a_s, se=zv_se[obs])
            est["L1"] = dict(d=lz[obs]["L1"], se=0.0)
            est["L2"] = dict(d=lz[obs]["L2"], se=0.0)
            est["SD"] = dict(d=float(np.std(sd_vals, ddof=1)) if len(sd_vals) > 1 else float("nan"), se=0.0)
            est["OSGX"] = dict(d=osg_exact(m, ob, psi, obs), se=0.0)
            d, info = zro(m, ob, psi, obs, args.minres_iters)
            est["ZRO"] = dict(d=d, se=0.0, info=info)
            rows.append(dict(width=wdt, seed=seed, label=lab, obs=obs, a0=a0[obs], a=a[obs], delta=d_true,
                             dE=E - W[0], ov0=float((psi0 @ psi) ** 2), est=est))
            log(f"w{wdt} s{seed} {lab} {obs}: delta={d_true:+.3e} (rel {abs(d_true) / abs(a0[obs]):.3f}) | "
                + " ".join(f"{k}={v['d']:+.2e}" for k, v in est.items()))
        log(f"state {key} done ({time.time() - t0:.0f} s)")
    out = dict(a0=a0, ed=W.tolist(), dE_sign_w6=dsign, rows=rows, args=vars(args))
    os.makedirs(args.results, exist_ok=True)
    json.dump(out, open(os.path.join(args.results, "analysis.json"), "w"), indent=1)


# ============================================================================ aggregate
def classify(arm, row):
    e = row["est"][arm]
    d, dh = row["delta"], e["d"]
    if not np.isfinite(dh):
        return "abstain"
    if arm != "SD" and e.get("se", 0.0) > TRUST * abs(dh):
        return "untrusted"
    ratio = abs(dh) / abs(d) if arm == "SD" else dh / d
    if 0.5 <= ratio <= 2.0:
        return "tracked"
    if ratio < 0.5:
        return "false"
    return "over"


def aggregate(args):
    D = json.load(open(os.path.join(args.results, "analysis.json")))
    rows = [r for r in D["rows"] if abs(r["delta"]) >= SCORE_REL * abs(r["a0"])]
    seeds = sorted({r["seed"] for r in rows})
    med_sign = float(np.median(D["dE_sign_w6"])) if D["dE_sign_w6"] else float("nan")
    print(f"scored items {len(rows)} of {len(D['rows'])} over seeds {seeds}; "
          f"width-6 control median dE_sign = {med_sign:.4f}")
    for obs in rc.OBS:
        rr = [r for r in D["rows"] if r["obs"] == obs]
        print(f"  {obs}: a0={rr[0]['a0']:+.3e} median |delta|/|a0| = "
              f"{np.median([abs(r['delta']) / abs(r['a0']) for r in rr]):.3f}, scored {sum(1 for r in rows if r['obs'] == obs)}")
    if not (med_sign >= PREMISE_SIGN and len(rows) >= 6 and len(seeds) >= 3):
        print("VERDICT: NO-TARGET (premise gate)")
        json.dump(dict(verdict="NO-TARGET"), open(os.path.join(args.results, "aggregate.json"), "w"))
        return
    T = {arm: {} for arm in ARMS}
    G = {}
    print("arm   M      G      per-seed tracked fractions")
    for arm in ARMS:
        cl = [classify(arm, r) for r in rows]
        for s in seeds:
            sel = [c for c, r in zip(cl, rows) if r["seed"] == s]
            T[arm][s] = sum(c == "tracked" for c in sel) / len(sel)
        G[arm] = sum(c == "false" for c in cl) / len(cl)
        print(f"{arm:4s} {np.median(list(T[arm].values())):.3f}  {G[arm]:.3f}  "
              + " ".join(f"{s}:{T[arm][s]:.2f}" for s in seeds))
    B = max(CONTROLS, key=lambda c: np.median(list(T[c].values())))
    zro_ok = np.median(list(T["ZRO"].values())) >= 0.7
    verdict = {}
    for arm in ROUTES:
        gains = [T[arm][s] - T[B][s] for s in seeds]
        M = float(np.median(list(T[arm].values()))); Dl = float(np.median(gains))
        if arm == "OSG" and not zro_ok:
            v = "FAIL (premise: ZRO tracks < 70%)"
        elif M >= 0.5 and Dl >= 0.2 and min(gains) > 0 and G[arm] <= 0.10:
            v = "PASS"
        elif M < 0.3 or Dl < 0.05 or G[arm] > 0.20:
            v = "FAIL"
        else:
            v = "AMBIGUOUS"
        verdict[arm] = dict(M=M, Delta=Dl, spread=float(max(gains) - min(gains)), G=G[arm], best_control=B, verdict=v)
        print(f"{arm}: M={M:.3f} vs best control {B} M={np.median(list(T[B].values())):.3f}; "
              f"paired median Delta={Dl:+.3f} (per-seed {['%+.2f' % g for g in gains]}); G={G[arm]:.3f} -> {v}")
    print("VERDICT: " + "; ".join(f"{k} = {v['verdict']}" for k, v in verdict.items()))
    json.dump(verdict, open(os.path.join(args.results, "aggregate.json"), "w"), indent=1)


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
    log("P3.1-r5 CHECKS DONE")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="checks", choices=["checks", "train", "analyze", "aggregate"])
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--widths", default="3,6,12")
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
    ap.add_argument("--p21-params", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params"))
    ap.add_argument("--p21-results", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"checks": checks, "train": train, "analyze": analyze, "aggregate": aggregate}[args.mode](args)


if __name__ == "__main__":
    main()
