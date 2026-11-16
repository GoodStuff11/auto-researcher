#!/usr/bin/env python3
"""
P3.1 round 6: Saturation Diagnostic (numerical part) and the injected low-energy check for route CLX
=====================================================================================================

Instance: the shared 4x4 OBC t-t'-J cluster (6+6, t'=-0.25, J=0.4, D=1681680), built by
../P3.1_order_parameter_error_r5/r5_core.py (imported, not copied). Observables: the r5 long-range
pair, spin and charge correlators (r5_core.Observables). "--instance small" swaps in the 4x4 OBC 2+2
cluster for smoke runs only (no physics claim).

Part 1: Saturation measurement on the real converged states (seeds 11/22/33 of P2.1 width 6, those
present at run time). For psi = alpha psi_0 + e (e orthogonal to psi_0), the first-order error is
  delta_1 = -2 alpha <e|(A - a_0) psi_0>   (delta = a_0 - a_psi = delta_1 + O(|e|^2)),
and the energy error is dE = <e|(H - E_0)|e> exactly. Lanczos quadrature (full reorthogonalisation,
--lanczos-steps) from w = (A - a_0) psi_0 resolves delta_1 into omega = E - E_0 shells; a second
Lanczos run from e resolves dE. Printed: cumulative fractions F_delta(Omega) and F_E(Omega) for
Omega in OMEGAS. Pre-registered reading: LOW-OMEGA-CONFIRMED for an observable iff
F_delta(1.0 t) >= 0.5 and F_delta(1.0 t) >= 2 F_E(1.0 t) on the median state.
Also printed: the spectral width W (E_max of H minus E_0, from eigsh "LA") and the Markov-inequality
cap 2 K^2 Omega / W on what a degree-K H-polynomial filter can change in an omega <= Omega component.

Part 2: Injected low-energy check (pre-registered; route CLX must pass it before any decisive test).
Exact in-irrep (chi, S = S_0) eigenvectors |n>, n = 1..N_LOW, are computed (eigsh on P_S P_chi H P_chi P_S).
Synthetic "capacity ladders": for synthetic seed j and ladder level i = 0..3 (low to high capacity)
  psi_ij = N[ psi_0 + lam_i u_ij + mu_ij jnk_ij ],   lam = LAM, mu_ij = MU_i (1 + 0.3 U(-1,1)),
  u_ij  = low-energy direction (combination of the |n>), jnk_ij = high-omega junk:
          (H - E_0) P_chi[psi_0 * xi] with the psi_0 and |n> components removed, normalised, fresh xi.
  Set A ("fixed"): u_ij = N[u_j + 0.1 nu_ij], u_j a seed-fixed random low combination  -> MUST PASS.
  Set B ("rotating"): u_ij = cos(th_i) u_j + sin(th_i) u'_j, th_i = i pi/6        -> stress, reported.
  Set N ("null"): lam = 0 (junk only)                                                  -> reported.
Every member is checked to lie in the chi irrep; delta = a_0 - a(psi_ij) is exact.

Route CLX (capacity-ladder first-order extrapolation, Lanczos-cleaned regressor). Per seed, over its
ladder members i: take the one-step Lanczos (Ritz) state phi_i of psi_i, the sampled
sigma_i = std of E_loc(phi_i) and the sampled a(phi_i); OLS fit a = alpha + gamma sigma; a_hat = alpha;
delta_hat_i = a_hat - a_s(psi_i) (a_s sampled). Nothing is fitted to the truth.
  Reported arms: CLX2 (two-step Ritz state, sampled), CLXraw (no clean: sigma and a of psi_i itself,
  i.e. the ladder version of ZV), CLX_R (regressor sqrt(max(E_L1 - E0_hat, 0)), E0_hat from the
  ladder fit E_L1 = E0_hat + s sigma_L1^2), L1VEX / L2VEX (single-state Lanczos variance
  extrapolation, exact; the H-Krylov arms the Saturation Diagnostic says are blind here).
SE = sqrt(SE_boot^2 + SE_jk^2): MC bootstrap over every point's draws (--n-boot) and a
leave-one-ladder-level-out jackknife. Trusted iff SE <= 0.5 |delta_hat|.
Item = (seed, observable) at the TOP ladder level (the best state, whose error is wanted).
Scored iff |delta| >= 0.10 |a_0| (raised from 0.02 after the small-cluster smoke, before
any main-instance run: 3% items are noise-limited at 1e5 draws; the real target errors are 15-31%). Tracked: trusted and 1/2 <= delta_hat/delta <= 2. False: trusted and
delta_hat/delta < 1/2. Over: trusted and > 2. G = false / trusted (defined with >= 3 trusted).
INJECT-PASS (Set A, arm CLX) iff >= 9 scored items, tracked fraction >= 0.7, median over seeds of the
per-seed tracked fraction >= 2/3, and G <= 0.10. Otherwise INJECT-FAIL and CLX is killed.

Part 3 (--mode ladder): the same CLX on REAL capacity ladders (per-seed NNBF widths, e.g. 2,3,4,6),
read from parameter files; this is the decisive test and needs trained ladders (see
submit_ladder_train.sbatch, not submitted). Rule (seeds as units, >= 5 seeds with >= 3 widths):
  PREMISE: median over seeds of the top-width relative error >= 0.02 on >= 1 observable, and the
  Lanczos-cleaned sigma decreases monotonically with width on >= 80% of seeds (else NO-LADDER).
  PASS: median_u T_CLX(u) >= 2/3, lower quartile >= 1/3, T_CLX(u) - T_CLXraw(u) > 0 on >= 80% of
        seeds, G_CLX <= 0.10 (defined), and INJECT-PASS on record.
  FAIL: median_u T_CLX(u) <= 1/3, or G_CLX > 0.20, or INJECT-FAIL.
  AMBIGUOUS otherwise: add 5 seeds; still not PASS -> killed.

Usage
-----
  python3 micro_test.py --mode checks                       small-cluster checks (< 1 min)
  python3 micro_test.py --mode inject [--instance main]     Parts 1 and 2 (Slurm for main)
  python3 micro_test.py --mode ladder --widths 2,3,4,6 --seeds 11,22,33,44,55   Part 3
Options
  --mode {checks,inject,ladder}   default checks
  --instance {main,small}         default main (small = 4x4 OBC 2+2, smoke only)
  --n-low INT          in-irrep excited levels used for injection (default 9)
  --n-seeds INT        synthetic seeds per injected set (default 5)
  --mc-samples INT     draws per sampled point (default 100000)
  --n-boot INT         bootstrap replicas (default 100)
  --lanczos-steps INT  Lanczos quadrature steps for Part 1 (default 120)
  --widths LIST, --seeds LIST     ladder mode only
  --params-dir DIR     trained ladder files inst0_nnbf_w{W}_s{S}.npz (default results/params here)
  --p21-params DIR     width-6 P2.1 files (default ../P2.1_subcubic_amplitudes_r3/results/params)
  --results DIR        output folder (default results/ here)
Outputs: <results>/inject.json (Parts 1-2), <results>/ladder.json (Part 3); verdict lines on stdout.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r5")
sys.path.insert(0, R5)
import r5_core as rc  # noqa: E402
from p11_core import solve_ed  # noqa: E402
from tj_core import TJ  # noqa: E402

OBS = rc.OBS
LAM = (0.35, 0.28, 0.21, 0.14)
MU = (0.30, 0.26, 0.22, 0.18)
OMEGAS = (0.25, 0.5, 1.0, 1.2, 2.0, 4.0)
SCORE_REL, TRUST = 0.10, 0.5   # scored iff |delta| >= 0.10 |a0| (set after the small-cluster smoke; real errors are 15-31%)
ARMS = ("CLX", "CLX2", "CLXraw", "CLX_R", "L1VEX", "L2VEX")
SMALL = dict(Lx=4, Ly=4, bcx="obc", bcy="obc", nup=2, ndn=2, tp=-0.25, J=0.4)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ instance
def setup(instance):
    t0 = time.time()
    cfg = rc.INST0 if instance == "main" else SMALL
    m = TJ(cfg["Lx"], cfg["Ly"], cfg["bcx"], cfg["bcy"], cfg["nup"], cfg["ndn"], 1.0, cfg["tp"], cfg["J"])
    m.spin_setup()
    W, Vv, deg = solve_ed(m, 4, 1e-8, rc.UHF_SEED)
    assert deg == 1, f"degenerate ground level ({W})"
    psi0 = Vv[:, 0] / np.linalg.norm(Vv[:, 0])
    sg, gen = rc.build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    ob = rc.Observables(m)
    log(f"{instance}: D={m.D} ED {W.tolist()} |G|={sg.G} chi={gen} S0={S0:g} ({time.time() - t0:.0f} s)")
    return m, float(W[0]), psi0, sg, S0, ob


def sector_eigs(m, sg, S0, k, seed=3):
    """Lowest k levels of P_S P_chi H P_chi P_S inside the sector (out-of-sector null space at 0)."""
    def PH(v):
        x = m.spin_project(sg.project(v), S0)
        return m.spin_project(sg.project(m.H @ x), S0)
    L = sla.LinearOperator((m.D, m.D), matvec=PH, dtype=float)
    v0 = m.spin_project(sg.project(np.random.default_rng(seed).standard_normal(m.D)), S0)
    e, U = sla.eigsh(L, k=k, which="SA", v0=v0, tol=1e-10)
    o = np.argsort(e); e, U = e[o], U[:, o]
    keep = e < -1e-6
    e, U = e[keep], U[:, keep]
    for n in range(len(e)):
        u = U[:, n] / np.linalg.norm(U[:, n]); U[:, n] = u
        res = np.linalg.norm(m.H @ u - e[n] * u)
        dev = np.linalg.norm(sg.project(u) - u)
        assert res < 1e-6 and dev < 1e-8, (n, res, dev)
    return e, U


# ============================================================================ spectral split (Part 1)
def lanczos_basis(H, v, steps):
    q = v / np.linalg.norm(v)
    Q = np.empty((steps, len(v)))
    al, be = [], []
    for k in range(steps):
        Q[k] = q
        w = H @ q
        al.append(q @ w)
        w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        b = np.linalg.norm(w)
        if k == steps - 1 or b < 1e-12:
            Q = Q[:k + 1]; break
        be.append(b); q = w / b
    T = np.diag(al) + np.diag(be[:len(al) - 1], 1) + np.diag(be[:len(al) - 1], -1)
    th, S = np.linalg.eigh(T)
    return Q, th, S


def spectral_split(m, E0, psi0, psi, ob, a0, steps):
    """Cumulative fractions of delta_1 (per observable) and of dE in omega shells."""
    psi = psi / np.linalg.norm(psi)
    alpha = psi0 @ psi
    e = psi - alpha * psi0
    dE = float(e @ (m.H @ e) - E0 * (e @ e))
    Qe, th_e, S_e = lanczos_basis(m.H, e, steps)
    we = (e @ e) * S_e[0] ** 2 * (th_e - E0)
    out = dict(alpha=float(alpha), dE=dE, dE_quad=float(we.sum()),
               F_E={str(O): float(we[th_e - E0 <= O].sum() / we.sum()) for O in OMEGAS}, obs={})
    del Qe
    for k in OBS:
        w = ob.apply(k, psi0) - a0[k] * psi0
        Q, th, S = lanczos_basis(m.H, w, steps)
        contrib = -2 * alpha * (S.T @ (Q @ e)) * np.linalg.norm(w) * S[0]
        d1 = float(-2 * alpha * (e @ w))
        d_true = a0[k] - ob.expect(k, psi)
        out["obs"][k] = dict(delta=d_true, delta1=d1, delta1_quad=float(contrib.sum()),
                             F_delta={str(O): float(contrib[th - E0 <= O].sum() / contrib.sum()) for O in OMEGAS})
        del Q
    return out


# ============================================================================ states and points
def ritz_states(H, psi, steps=2):
    basis = [psi / np.linalg.norm(psi)]
    out = []
    for _ in range(steps):
        w = H @ basis[-1]
        for _r in range(2):
            for b in basis:
                w -= (b @ w) * b
        basis.append(w / np.linalg.norm(w))
        B = np.column_stack(basis)
        ev, U = np.linalg.eigh(B.T @ (H @ B))
        phi = B @ U[:, 0]
        phi /= np.linalg.norm(phi)
        if phi @ psi < 0:
            phi = -phi
        out.append(phi)
    return out


def exact_stats(m, ob, v):
    Hv = m.H @ v
    E = float(v @ Hv)
    return dict(E=E, s2=float(Hv @ Hv - E * E), a={k: ob.expect(k, v) for k in OBS})


def sampled_point(m, ob, v, n, rng):
    """Per-sample local E and local A for samples ~ |v|^2 (exact vectors indexed at the draws)."""
    p = v ** 2
    idx = rng.choice(len(v), size=n, p=p / p.sum())
    vs = v[idx]
    El = (m.H @ v)[idx] / vs
    Al = {k: ob.apply(k, v)[idx] / vs for k in OBS}
    return El, Al


def member(m, ob, v, n, rng):
    """All quantities of one ladder member: raw, one-step and two-step Ritz states."""
    phi1, phi2 = ritz_states(m.H, v, 2)
    ex = {"raw": exact_stats(m, ob, v), "L1": exact_stats(m, ob, phi1), "L2": exact_stats(m, ob, phi2)}
    smp = {"raw": sampled_point(m, ob, v, n, rng), "L1": sampled_point(m, ob, phi1, n, rng),
           "L2": sampled_point(m, ob, phi2, n, rng)}
    s0, s1, s2 = ex["raw"]["s2"], ex["L1"]["s2"], ex["L2"]["s2"]
    vex = {}
    for k in OBS:
        a0_, a1, a2 = ex["raw"]["a"][k], ex["L1"]["a"][k], ex["L2"]["a"][k]
        L1 = a0_ + (a1 - a0_) * s0 / (s0 - s1)
        L2 = a1 - (a2 - a1) / (s2 - s1) * s1
        vex[k] = dict(L1VEX=float(L1), L2VEX=float(L2))
    return ex, smp, vex


# ============================================================================ CLX fits
def _stats(smp, sel):
    El, Al = smp
    e = El[sel]
    return e.mean(), e.std(), {k: Al[k][sel].mean() for k in OBS}


def clx_fit(members, sels):
    """members: list of member-sample dicts {'raw','L1','L2'}; sels: per member per tag index arrays.
    Returns {arm: {obs: a_hat}} for CLX, CLX2, CLXraw, CLX_R."""
    out = {a: {} for a in ("CLX", "CLX2", "CLXraw", "CLX_R")}
    cols = {}
    for tag in ("raw", "L1", "L2"):
        st = [_stats(mb[tag], sl[tag]) for mb, sl in zip(members, sels)]
        cols[tag] = (np.array([s[0] for s in st]), np.array([s[1] for s in st]),
                     {k: np.array([s[2][k] for s in st]) for k in OBS})
    E1, sg1, _ = cols["L1"]
    E0h = np.polyfit(sg1 ** 2, E1, 1)[1]
    rt = np.sqrt(np.maximum(E1 - E0h, 0.0))
    for arm, tag, x in (("CLX", "L1", cols["L1"][1]), ("CLX2", "L2", cols["L2"][1]),
                        ("CLXraw", "raw", cols["raw"][1]), ("CLX_R", "L1", rt)):
        for k in OBS:
            a = cols[tag][2][k]
            out[arm][k] = float(np.polyfit(x, a, 1)[1]) if np.ptp(x) > 0 else float("nan")
    return out


def clx_estimates(members, n_boot, rng):
    """Returns {arm: {obs: (a_hat, se_boot, se_jk)}} for the four fitted arms."""
    full_sel = [{t: np.arange(len(mb[t][0])) for t in mb} for mb in members]
    full = clx_fit(members, full_sel)
    reps = []
    for _ in range(n_boot):
        sel = [{t: rng.integers(0, len(mb[t][0]), len(mb[t][0])) for t in mb} for mb in members]
        reps.append(clx_fit(members, sel))
    jk = []
    if len(members) >= 4:
        for i in range(len(members)):
            sub = [mb for j, mb in enumerate(members) if j != i]
            jk.append(clx_fit(sub, [{t: np.arange(len(mb[t][0])) for t in mb} for mb in sub]))
    out = {}
    for arm in full:
        out[arm] = {}
        for k in OBS:
            sb = float(np.nanstd([r[arm][k] for r in reps], ddof=1))
            if jk:
                th = np.array([r[arm][k] for r in jk]); n = len(th)
                sj = float(np.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2)))
            else:
                sj = 0.0
            out[arm][k] = (full[arm][k], sb, sj)
    return out


def classify(dh, se, d):
    if not np.isfinite(dh):
        return "abstain"
    if se > TRUST * abs(dh):
        return "untrusted"
    r = dh / d
    if r < 0.5:
        return "false"
    return "tracked" if r <= 2 else "over"


def score_ladder(ladder_rows, n_boot, rng, a0):
    """ladder_rows: list of dict(ex, smp, vex) for levels 0..L-1 of one seed. Returns item rows."""
    est = clx_estimates([r["smp"] for r in ladder_rows], n_boot, rng)
    items = []
    for lev, r in enumerate(ladder_rows):
        for k in OBS:
            d = a0[k] - r["ex"]["raw"]["a"][k]
            a_s = float(r["smp"]["raw"][1][k].mean())
            row = dict(level=lev, obs=k, delta=d, a0=a0[k], scored=bool(abs(d) >= SCORE_REL * abs(a0[k])),
                       est={})
            for arm in ("CLX", "CLX2", "CLXraw", "CLX_R"):
                ah, sb, sj = est[arm][k]
                dh = ah - a_s
                se = float(np.sqrt(sb ** 2 + sj ** 2))
                row["est"][arm] = dict(d=float(dh), se=se, cls=classify(dh, se, d))
            for arm in ("L1VEX", "L2VEX"):
                dh = r["vex"][k][arm] - r["ex"]["raw"]["a"][k]
                row["est"][arm] = dict(d=float(dh), se=0.0, cls=classify(dh, 0.0, d))
            items.append(row)
    return items


def summarize(seed_items, arm, top_only=True):
    """seed_items: {seed: items}. Returns tracked fraction, per-seed fractions, G, counts."""
    per_seed, n_sc, n_tr, n_false, n_trust = {}, 0, 0, 0, 0
    for s, items in seed_items.items():
        top = max(r["level"] for r in items)
        sc = [r for r in items if r["scored"] and (r["level"] == top or not top_only)]
        cl = [r["est"][arm]["cls"] for r in sc]
        n_sc += len(sc); n_tr += cl.count("tracked"); n_false += cl.count("false")
        n_trust += sum(c in ("tracked", "false", "over") for c in cl)
        if sc:
            per_seed[s] = cl.count("tracked") / len(sc)
    G = n_false / n_trust if n_trust >= 3 else float("nan")
    return dict(scored=n_sc, tracked=n_tr, frac=n_tr / n_sc if n_sc else float("nan"),
                per_seed=per_seed, median_seed=float(np.median(list(per_seed.values()))) if per_seed else float("nan"),
                G=G, trusted=n_trust)


# ============================================================================ injection (Part 2)
def junk(m, sg, psi0, E0, U, rng):
    x = sg.project(psi0 * rng.standard_normal(m.D))
    j = m.H @ x - E0 * x
    j -= U @ (U.T @ j)
    j -= psi0 * (psi0 @ j)
    return j / np.linalg.norm(j)


def build_set(name, m, sg, psi0, E0, U, n_seeds, rng):
    nl = U.shape[1]
    ladders = []
    for s in range(n_seeds):
        u = U @ rng.standard_normal(nl); u /= np.linalg.norm(u)
        up = U @ rng.standard_normal(nl); up -= u * (u @ up); up /= np.linalg.norm(up)
        lad = []
        for i, (lam, mu) in enumerate(zip(LAM, MU)):
            if name == "A":
                nu = U @ rng.standard_normal(nl); nu /= np.linalg.norm(nu)
                ui = u + 0.1 * nu; ui /= np.linalg.norm(ui)
            elif name == "B":
                th = i * np.pi / 6
                ui = np.cos(th) * u + np.sin(th) * up
            else:
                ui, lam = 0.0 * u, 0.0
            mu_ij = mu * (1 + 0.3 * rng.uniform(-1, 1))
            v = psi0 + lam * ui + mu_ij * junk(m, sg, psi0, E0, U, rng)
            v /= np.linalg.norm(v)
            dev = float(np.linalg.norm(sg.project(v) - v))
            assert dev < 1e-8, dev
            lad.append(v)
        ladders.append(lad)
    return ladders


def run_set(name, ladders, m, ob, a0, E0, psi0, args, rng):
    seed_items = {}
    for s, lad in enumerate(ladders):
        rows = []
        for v in lad:
            ex, smp, vex = member(m, ob, v, args.mc_samples, rng)
            rows.append(dict(ex=ex, smp=smp, vex=vex, ov0=float((psi0 @ v) ** 2),
                             dE=ex["raw"]["E"] - E0, dE_L1=ex["L1"]["E"] - E0))
        items = score_ladder(rows, args.n_boot, rng, a0)
        for r in items:
            r["dE"] = rows[r["level"]]["dE"]; r["dE_L1"] = rows[r["level"]]["dE_L1"]; r["ov0"] = rows[r["level"]]["ov0"]
        seed_items[s] = items
        for r in items:
            if r["level"] == len(lad) - 1:
                log(f"set {name} seed {s} top {r['obs']}: delta={r['delta']:+.3e} (rel {abs(r['delta']) / abs(a0[r['obs']]):.3f}, "
                    f"ov0 {r['ov0']:.3f}, dE {r['dE']:.4f}, dE_L1 {r['dE_L1']:.4f}) | "
                    + " ".join(f"{a}={e['d']:+.2e}+-{e['se']:.1e}[{e['cls']}]" for a, e in r["est"].items()))
    return seed_items


def inject(args):
    t0 = time.time()
    m, E0, psi0, sg, S0, ob = setup(args.instance)
    a0 = ob.all(psi0)
    log("a0 " + json.dumps(a0))
    out = dict(instance=args.instance, a0=a0, E0=E0, args=vars(args))
    # ---- spectral width and Markov cap
    Emax = float(sla.eigsh(m.H, k=1, which="LA", tol=1e-6)[0][0])
    Wd = Emax - E0
    out["W"] = Wd
    log(f"spectral width W = E_max - E_0 = {Wd:.3f}; Markov cap 2K^2 Omega/W at Omega=0.5: "
        f"K=1 {2 * 0.5 / Wd:.3f}, K=2 {8 * 0.5 / Wd:.3f}; degree for 1/omega resolution sqrt(W/Omega) = {np.sqrt(Wd / 0.5):.1f}")
    # ---- Part 1 on real states
    states = {}
    if args.instance == "main":
        for seed in (11, 22, 33):
            f = os.path.join(args.p21_params, f"inst0_nnbf_s{seed}.npz")
            if not os.path.exists(f):
                log(f"seed {seed}: no parameter file; skipped"); continue
            ans = rc.load_params(rc.nnbf(m, 6, seed), f)
            v = sg.project(ans.amplitudes(32768)); v /= np.linalg.norm(v)
            states[f"w6_s{seed}"] = v if v @ psi0 > 0 else -v
    else:
        rng0 = np.random.default_rng(5)
        v = sg.project(psi0 * np.exp(0.3 * rng0.standard_normal(m.D))); v /= np.linalg.norm(v)
        states["small_synthetic"] = v
    out["split"] = {}
    for key, v in states.items():
        sp_ = spectral_split(m, E0, psi0, v, ob, a0, args.lanczos_steps)
        out["split"][key] = sp_
        log(f"split {key}: dE={sp_['dE']:.5f} (quad {sp_['dE_quad']:.5f}) F_E " + json.dumps(sp_["F_E"]))
        for k, r in sp_["obs"].items():
            log(f"   {k}: delta={r['delta']:+.3e} delta1={r['delta1']:+.3e} (quad {r['delta1_quad']:+.3e}) F_delta "
                + json.dumps({o: round(x, 3) for o, x in r["F_delta"].items()}))
    if out["split"]:
        verdict = {}
        for k in OBS:
            fd = np.median([s["obs"][k]["F_delta"]["1.0"] for s in out["split"].values()])
            fe = np.median([s["F_E"]["1.0"] for s in out["split"].values()])
            verdict[k] = bool(fd >= 0.5 and fd >= 2 * fe)
        out["low_omega_confirmed"] = verdict
        log("SATURATION reading (F_delta(1t) >= 0.5 and >= 2 F_E(1t)): " + json.dumps(verdict))
    # ---- Part 2
    e, U = sector_eigs(m, sg, S0, args.n_low + 1)
    assert abs(e[0] - E0) < 1e-6, (e[0], E0)
    U = U[:, 1:]; om = e[1:] - E0
    out["omega_low"] = om.tolist()
    out["A_n0"] = {k: (U.T @ ob.apply(k, psi0)).tolist() for k in OBS}
    log(f"injection levels omega = {np.round(om, 4).tolist()} ({time.time() - t0:.0f} s)")
    rng = np.random.default_rng(2028)
    out["sets"] = {}
    for name in ("A", "B", "N"):
        lads = build_set(name, m, sg, psi0, E0, U, args.n_seeds, rng)
        si = run_set(name, lads, m, ob, a0, E0, psi0, args, rng)
        summ = {arm: summarize(si, arm) for arm in ARMS}
        out["sets"][name] = dict(items={str(s): v for s, v in si.items()}, summary=summ)
        log(f"SET {name} (top level): " + "; ".join(
            f"{arm} {s['tracked']}/{s['scored']} med_seed {s['median_seed']:.2f} G {s['G']:.2f}" for arm, s in summ.items()))
        if name == "A":
            vex = [abs(r["est"][arm]["d"] / r["delta"]) for its in si.values() for r in its
                   if r["scored"] for arm in ("L1VEX",)]
            log(f"SET A saturation: median |L1VEX/delta| over all scored levels = {np.median(vex):.3f}")
            out["setA_L1VEX_median_ratio"] = float(np.median(vex))
    s = out["sets"]["A"]["summary"]["CLX"]
    ok = s["scored"] >= 9 and s["frac"] >= 0.7 and s["median_seed"] >= 2 / 3 and np.isfinite(s["G"]) and s["G"] <= 0.10
    out["inject_pass"] = bool(ok)
    os.makedirs(args.results, exist_ok=True)
    json.dump(out, open(os.path.join(args.results, "inject.json"), "w"), indent=1, default=float)
    log(f"INJECT (set A, CLX): scored {s['scored']} tracked {s['tracked']} frac {s['frac']:.2f} "
        f"median_seed {s['median_seed']:.2f} G {s['G']:.2f} -> {'INJECT-PASS' if ok else 'INJECT-FAIL'} "
        f"({time.time() - t0:.0f} s)")


# ============================================================================ real ladders (Part 3)
def ladder(args):
    m, E0, psi0, sg, S0, ob = setup("main")
    a0 = ob.all(psi0)
    rng = np.random.default_rng(2029)
    widths = [int(x) for x in args.widths.split(",")]
    seed_items, mono, rel_top = {}, {}, {}
    for seed in [int(x) for x in args.seeds.split(",")]:
        rows = []
        for w in widths:
            cands = [os.path.join(args.params_dir, f"inst0_nnbf_w{w}_s{seed}.npz")]
            if w == 6:
                cands.insert(0, os.path.join(args.p21_params, f"inst0_nnbf_s{seed}.npz"))
            f = next((c for c in cands if os.path.exists(c)), None)
            if f is None:
                log(f"seed {seed} width {w}: missing; skipped"); continue
            ans = rc.load_params(rc.nnbf(m, w, seed), f)
            v = sg.project(ans.amplitudes(32768)); v /= np.linalg.norm(v)
            v = v if v @ psi0 > 0 else -v
            ex, smp, vex = member(m, ob, v, args.mc_samples, rng)
            rows.append(dict(ex=ex, smp=smp, vex=vex, width=w))
        if len(rows) < 3:
            log(f"seed {seed}: {len(rows)} widths (< 3); abstains"); continue
        s1 = [r["ex"]["L1"]["s2"] for r in rows]
        mono[seed] = bool(all(np.diff(s1) < 0))
        items = score_ladder(rows, args.n_boot, rng, a0)
        seed_items[seed] = items
        top = max(r["level"] for r in items)
        rel_top[seed] = {r["obs"]: abs(r["delta"]) / abs(a0[r["obs"]]) for r in items if r["level"] == top}
        for r in items:
            log(f"seed {seed} width {rows[r['level']]['width']} {r['obs']}: delta={r['delta']:+.3e} | "
                + " ".join(f"{a}={e['d']:+.2e}[{e['cls']}]" for a, e in r["est"].items()))
    out = dict(a0=a0, items={str(s): v for s, v in seed_items.items()}, monotone=mono, rel_top=rel_top)
    verdict = "NO-LADDER"
    if len(seed_items) >= 5:
        prem = any(np.median([rel_top[s][k] for s in rel_top]) >= 0.02 for k in OBS) and \
            np.mean(list(mono.values())) >= 0.8
        if prem:
            c, r_ = summarize(seed_items, "CLX"), summarize(seed_items, "CLXraw")
            T = np.array([c["per_seed"].get(s, 0.0) for s in seed_items])
            gain = np.array([c["per_seed"].get(s, 0.0) - r_["per_seed"].get(s, 0.0) for s in seed_items])
            Gd = np.isfinite(c["G"])
            inj = json.load(open(os.path.join(args.results, "inject.json"))).get("inject_pass", False) \
                if os.path.exists(os.path.join(args.results, "inject.json")) else False
            if np.median(T) >= 2 / 3 and np.percentile(T, 25) >= 1 / 3 and np.mean(gain > 0) >= 0.8 \
                    and Gd and c["G"] <= 0.10 and inj:
                verdict = "PASS"
            elif np.median(T) <= 1 / 3 or (Gd and c["G"] > 0.20) or not inj:
                verdict = "FAIL"
            else:
                verdict = "AMBIGUOUS"
            out.update(summary_CLX=c, summary_CLXraw=r_)
    out["verdict"] = verdict
    os.makedirs(args.results, exist_ok=True)
    json.dump(out, open(os.path.join(args.results, "ladder.json"), "w"), indent=1, default=float)
    log(f"LADDER VERDICT: {verdict}")


# ============================================================================ checks
def checks(args):
    m, E0, psi0, sg, S0, ob = setup("small")
    a0 = ob.all(psi0)
    e, U = sector_eigs(m, sg, S0, 8)
    U = U[:, 1:]; om = e[1:] - E0
    rng = np.random.default_rng(0)
    # (1) Proposition S1: flipping the sign of low components leaves every H moment unchanged and
    #     flips the first-order delta contribution.
    c = 0.2 * rng.standard_normal(U.shape[1])
    v1 = psi0 + U @ c; v2 = psi0 + U @ (-c)
    v1 /= np.linalg.norm(v1); v2 /= np.linalg.norm(v2)
    mom = []
    for v in (v1, v2):
        x, ms = v.copy(), []
        for _ in range(4):
            x = m.H @ x; ms.append(v @ x)
        mom.append(np.array(ms))
    dmom = float(np.max(np.abs(mom[0] - mom[1]) / np.abs(mom[0])))
    for k in OBS:
        d1, d2 = a0[k] - ob.expect(k, v1), a0[k] - ob.expect(k, v2)
        log(f"check S1 {k}: moments rel diff {dmom:.1e}; delta(+c) {d1:+.3e} delta(-c) {d2:+.3e}")
    assert dmom < 1e-12
    # (2) Lanczos quadrature reproduces delta_1 and dE; the low-omega split agrees with eigenvectors
    v = psi0 + U @ c + 0.2 * junk(m, sg, psi0, E0, U, rng); v /= np.linalg.norm(v)
    sp_ = spectral_split(m, E0, psi0, v, ob, a0, 60)
    log(f"check quad: dE {sp_['dE']:.6e} quad {sp_['dE_quad']:.6e}")
    assert abs(sp_["dE"] - sp_["dE_quad"]) < 1e-8 * max(1, sp_["dE"])
    alpha = psi0 @ v
    Om = 1.2
    for k in OBS:
        r = sp_["obs"][k]
        w = ob.apply(k, psi0) - a0[k] * psi0
        ev_low = float(-2 * alpha * sum((U[:, n] @ v) * (U[:, n] @ w) for n in range(U.shape[1]) if om[n] <= Om))
        log(f"check quad {k}: delta1 {r['delta1']:+.4e} quad {r['delta1_quad']:+.4e} | F_delta({Om}) "
            f"{r['F_delta'][str(Om)]:.4f} vs eigvec {ev_low / r['delta1']:.4f} (omega_low {np.round(om, 3).tolist()})")
        assert abs(r["delta1"] - r["delta1_quad"]) < 1e-8 * abs(r["delta1"]) + 1e-14
    # (3) CLX with EXACT regressor and observable recovers delta on a junk-free fixed-direction ladder
    #     (first-order regime), while L1VEX does not need to; sampled CLX is printed alongside.
    u = U @ rng.standard_normal(U.shape[1]); u /= np.linalg.norm(u)
    rows, xs = [], []
    for lam in LAM:
        x = psi0 + 0.3 * lam * u; x /= np.linalg.norm(x)
        ex, smp, vex = member(m, ob, x, 200000, rng)
        rows.append(dict(ex=ex, smp=smp, vex=vex)); xs.append(x)
    items = score_ladder(rows, 20, rng, a0)
    sig = np.sqrt([r["ex"]["L1"]["s2"] for r in rows])
    for r in items:
        if r["level"] == 3:
            k = r["obs"]
            a_ex = np.array([q["ex"]["L1"]["a"][k] for q in rows])
            d_ex = np.polyfit(sig, a_ex, 1)[1] - rows[3]["ex"]["raw"]["a"][k]
            log(f"check CLX junk-free ladder {k}: delta {r['delta']:+.4e} CLX-exact {d_ex:+.4e} "
                f"(ratio {d_ex / r['delta']:.3f}) CLX-sampled {r['est']['CLX']['d']:+.3e}+-{r['est']['CLX']['se']:.1e} "
                f"L1VEX {r['est']['L1VEX']['d']:+.3e}")
            assert abs(d_ex / r["delta"] - 1) < 0.15, "CLX-exact misses a first-order fixed-direction ladder"
    log("P3.1-r6 CHECKS DONE")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="checks", choices=["checks", "inject", "ladder"])
    ap.add_argument("--instance", default="main", choices=["main", "small"])
    ap.add_argument("--n-low", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--mc-samples", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--lanczos-steps", type=int, default=120)
    ap.add_argument("--widths", default="2,3,4,6")
    ap.add_argument("--seeds", default="11,22,33,44,55")
    ap.add_argument("--params-dir", default=os.path.join(HERE, "results", "params"))
    ap.add_argument("--p21-params", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"checks": checks, "inject": inject, "ladder": ladder}[args.mode](args)


if __name__ == "__main__":
    main()
