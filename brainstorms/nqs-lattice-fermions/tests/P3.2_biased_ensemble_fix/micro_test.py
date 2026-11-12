#!/usr/bin/env python3
"""
P3.2 round-4 FIX of the decisive test (pilot 410917 read-out: INTRINSIC-BIAS by elimination)
=========================================================================================

Copied from tests/P3.2_biased_ensemble_r3/micro_test.py and modified. Same instance, same
projected NNBF, same projected SR. What changed (each item is enforced in code):

  F1  R0P-warm control. At each target, the branch of order X is initialised from the
      parameters of the certified snapshot NQS of order X nearest in t', and trained with
      the SAME projected-SR budget (--sr-iters) as the cold control. Variant R0P-warm-abs
      abstains when a branch's measured c ends with a sign other than its snapshot's (the
      branch crossed over during training). Fallback when no certified snapshot of order X
      exists: the nearest snapshot BRANCH whose measured order is X (flagged "uncertified";
      --aggregate prints how many hard pairs used one).
  F2  Certification gate (replaces the inverted "both branches have the same sign of c"):
        Eckart  : the snapshot winner's energy error is below half the in-irrep snapshot gap
                  (computed in the run; ED truth is available for this methods instance);
        seed    : the winner's sign of c agrees with the majority over the seeds of the
                  width, and that majority holds in >= 2 of 3 seeds (computed in --aggregate).
      A snapshot is certified only if BOTH hold. EC (and the warm-start sources) use only
      Eckart-certified snapshots in the run; --aggregate turns a seed's EC decisions into
      abstentions if any of its snapshots fails the seed gate. ED at snapshots and targets
      uses check_global=True, and a run aborts if the irrep E0 differs from the global E0.
      --aggregate ASSERTS identical target lists, snapshot lists, t'_c and training settings
      across the runs of a width, and distinct seeds.
  F3  Optimizer noise. Every training returns its best-energy iterate (unchanged) and now
      also the energy history. The noise of a branch is sigma = std of its energy over the
      last --window steps. A control decision (lower-energy branch -> sign of its c) counts
      only if |E_A - E_B| > 2 * hypot(sigma_A, sigma_B); otherwise it abstains ("U").
      The final-window mean energies are recorded as well. At least 3 seeds per width.
  F4  Regime logging. For every target and control (cold, warm) the run records
        crossover : both branches end with the same sign of c;
        noise     : branches distinct, |dE| <= 2 sigma;
        correct   : branches distinct, committed, right;
        intrinsic : branches distinct, committed, wrong (differential bias Delta b).
      PRE-REGISTERED: SO-EC can win only in the crossover case; its verdict is computed on
      the hard (target, seed) pairs whose WARM control is in the crossover regime, and SO-EC
      FAILs (no room) if there are fewer than 3 such pairs at a width.
  F5  Rule 4, sampled force. On --sf-targets the warm control is also trained with a
      SAMPLED force: f = mean[(O_P - <O_P>)(E_loc - <E_loc>)] over the same --sr-samples i.i.d.
      draws from |psi_P|^2 that give the SR metric; best iterate and window noise use the
      sampled energy estimate only; the final decision uses VMC energies (--n-samples draws)
      with sigma = hypot(window std, VMC standard error). The local energy E_loc(n) is read
      from the vector H psi_P; it equals the local-estimator numerator sum_m H_nm psi_P(m)
      (#connections amplitude evaluations per sample in a real run). Report-only arm.
  AT is KILLED (round-3 critic) and kept only as a diagnostic (exact and oracle); it is not
  in any verdict.

Instance (unchanged; found by ed_scan.py, see ../P3.2_biased_ensemble_r3/scan_logs/)
  4x3 PBC Hubbard, N_up = N_dn = 4, U = 8, t_x = 1, t_y = 1.11394, t' swept through the cone
  of an exact same-irrep conical intersection. Irrep K = 0, M_x = M_y = +1 (S = 0 ground
  state; projected sector dim 1618). In-irrep gap minimum 1.0e-3 t at t'_c ~ -0.470482
  (recomputed at run time). Order character c > 0: period-4 stripe (B); c < 0: near-Neel (A).
  This is a METHODS instance (spin textures forced by L_y = 3); no physics transfer.

Arms (one task = one (width, seed); every state is P_chi psi_theta)
  R0P            cold: Neel-UHF (A) and stripe-UHF (B) seeded branches at the target.
  R0P-warm       F1.  R0P-warm-abs: F1 with sign-change abstention.
  R0P-warm-sf    F5 (only on --sf-targets).
  EC             Route 3.5 SO-EC on Eckart-certified snapshots (block per measured order).
  EC sampled     block matrices from mixture sampling, --n-samples draws, --n-rep replicas.
  ECor / AT / ATor  diagnostics.

Pre-committed decision rule (--aggregate; per width, >= 3 seeds)
  eps(t)  = median over seeds of min over the four control branches (cold A/B, warm A/B) of
            E - E0 (the best available control error).
  hard(t) = eps(t) >= gap(t)/2 and |c(E0)| >= tau.
  Valid width: >= 3 hard targets; >= 4 of 6 snapshot points certified (Eckart and seed) in
            the median seed; both order blocks non-empty in every seed.
  Controls C in {R0P, R0P-warm, R0P-warm-abs}: CW (committed-wrong fraction), DEC (commit
            fraction) over hard (target, seed); regime counts for cold and warm.
  Control verdict: NO-ROOM if every C has CW = 0 on the hard targets (no control error
            to beat; the array answers "projected R0+ decides this instance").
  SO-EC on the crossover subset X (hard pairs whose warm control is in crossover):
    FAIL      |X| < 3 (pre-registered: SO-EC cannot win outside crossover), or CW(EC) > 0.30,
              or some C dominates EC on X (CW(C) <= CW(EC) and DEC(C) >= DEC(EC));
    PASS      CW(EC) <= 0.15, DEC(EC) >= 0.6, for every C: CW(C) - CW(EC) >= 0.15 or
              (CW(EC) <= CW(C) and DEC(EC) >= DEC(C) + 0.2), sampled-EC flip rate <= 0.10
              and CW(EC sampled) <= 0.15;
    AMBIGUOUS otherwise (pre-committed: 6 seeds at the same widths; still AMBIGUOUS -> KILLED).
  Global: PASS at >= 1 valid width and FAIL at none.

Usage:  python3 micro_test.py [options]
  --width INT (4)  --seed INT (1234)
  --task-id INT    >= 0 overrides: width = (4, 8)[t // 3], seed = (1234, 2345, 3456)[t % 3]
                   (array 0-5). Widths 4/8 have 676/1160 parameters (sector-dim ratios 2.4/1.4).
  --ty FLOAT (1.11394)  --U FLOAT (8)
  --targets LIST   default "-0.002,-0.0005,-0.0002,0.0002,0.0005,0.002"
  --snaps LIST     default "-0.08,-0.05,-0.03,0.03,0.05,0.08"
  --sf-targets LIST  targets with the sampled-force warm arm (default "-0.0002,0.0002")
  --sr-iters INT (800)  --sr-lr (0.05)  --sr-shift (1e-3)  --sr-samples (2000)
  --window INT     noise window in SR steps (default 100)
  --z FLOAT        abstention multiple of the noise (default 2)
  --chunk (65536) --o-chunk (256) --n-samples (20000) --n-rep (20) --tau (0.02)
  --out PATH       JSON (default results/w<W>_s<seed>.json; snapshot parameters in .npz)
  --smoke INT      1: real instance, 1 target (also an sf target), 2 snapshots, 3 SR steps,
                   width 4, n-samples 2000, n-rep 3 (results in results_smoke/).
  --checks-only INT  run the correctness checks and exit.
  --aggregate DIR  apply the decision rule to every w*_s*.json in DIR.
Correctness checks (exit code 2 on failure): [H, U_g] = 0; projector idempotent and
commuting with H; projected gradient vs finite differences; MC mixture S_01 vs exact
(< 5 standard errors); sampled force vs exact force (|fs-f|^2 below 3x its summed
sampling variance at 20000 draws, i.e. unbiased);
warm start reproduces the snapshot amplitudes exactly.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p11_core as P  # noqa: E402
import proj_core as PC  # noqa: E402
import ed_scan as ES  # noqa: E402

log = P.log
PARTS = ("tx", "ty", "tp", "U")
CONTROLS = ("R0P", "R0P-warm", "R0P-warm-abs")


# ============================================================================ instance
class Instance:
    def __init__(self, ty, U):
        self.cl = cl = ES.Cluster("4x3pp", 4, 4)
        self.m = m = cl.m
        self.ty, self.U = ty, U
        chi = np.ones(cl.G)
        cl.build_irrep(chi, 0, 7000)
        self.sg = PC.SpaceGroup(m)
        self.sg.set_irrep(dict(Tx=1, Ty=1, Mx=1, My=1))
        Lx, Ly = m.Lx, m.Ly
        iu, idd = m.iu_id()
        sz = (m.occ_u[iu].astype(float) - m.occ_d[idd].astype(float)) / 2

        def sf(q):
            return np.abs(sz @ np.exp(1j * m.coords @ np.array(q))) ** 2 / m.V
        qB = (2 * np.pi / Lx, 0.0)
        qA = (np.pi, 2 * np.pi / Ly)
        self.cn = 0.5 * (sf(qB) + sf((-qB[0], 0.0))) - 0.5 * (sf(qA) + sf((np.pi, -qA[1])))
        self.spin = PC.Spin(m)
        r = minimize_scalar(lambda tp: np.diff(cl.sector_eigs(self.coup(tp), 2)[0])[0],
                            bounds=(-0.6, -0.35), method="bounded", options=dict(xatol=1e-8))
        self.tpc, self.gapmin = float(r.x), float(r.fun)

    def coup(self, tp):
        return {"tx": 1.0, "ty": self.ty, "tp": tp, "U": self.U}

    def H(self, tp):
        return self.cl.H(self.coup(tp))

    def set_T(self, tp):
        T = self.cl.Tm
        self.m.T = T["tx"] + self.ty * T["ty"] + tp * T["tp"]
        self.m.U = self.U

    def char(self, psi):
        return float(np.sum(psi ** 2 * self.cn) / (psi @ psi))

    def pvec(self, psi):
        nn = psi @ psi
        return np.array([psi @ (self.cl.parts_full[k] @ psi) / nn for k in PARTS])

    def ed(self, tp, check_global=True):
        w, v = self.cl.sector_eigs(self.coup(tp), 2)
        p0, p1 = self.cl.full_vec(v[:, 0]), self.cl.full_vec(v[:, 1])
        out = dict(tp=tp, E0=float(w[0]), E1=float(w[1]), gap=float(w[1] - w[0]),
                   c0=self.char(p0), c1=self.char(p1))
        if check_global:
            wg = eigsh(self.H(tp), k=1, which="SA", tol=1e-10,
                       v0=np.random.default_rng(0).standard_normal(self.cl.D))[0]
            out["E0_global"] = float(wg[0])
            out["irrep_is_global"] = bool(abs(out["E0"] - out["E0_global"]) < 1e-7)
            if not out["irrep_is_global"]:
                log(f"ABORT: irrep E0 {out['E0']:.10f} != global E0 {out['E0_global']:.10f} at t'={tp}")
                sys.exit(3)
        return out, p0


def uhf_pattern(m, pattern, seed, iters=3000, mix=0.3):
    """UHF orbitals from a 'neel' or period-4 'stripe' spin pattern (copied from round 2)."""
    rng = np.random.default_rng(seed)
    V, T, U = m.V, m.T, m.U
    if pattern == "neel":
        s = np.array([(-1) ** int(x + y) for x, y in m.coords], float)
    else:
        s = np.array([[1, -1, -1, 1][int(x) % 4] for x, y in m.coords], float)
    nu = (m.nup / V) * (1 + 0.4 * s) + 0.01 * rng.standard_normal(V)
    nd = (m.ndn / V) * (1 - 0.4 * s) + 0.01 * rng.standard_normal(V)
    for _ in range(iters):
        _, vu = np.linalg.eigh(-T + np.diag(U * nd))
        _, vd = np.linalg.eigh(-T + np.diag(U * nu))
        Pu, Pd = vu[:, :m.nup], vd[:, :m.ndn]
        nun, ndn = np.sum(Pu ** 2, 1), np.sum(Pd ** 2, 1)
        if max(np.abs(nun - nu).max(), np.abs(ndn - nd).max()) < 1e-10:
            break
        nu, nd = (1 - mix) * nu + mix * nun, (1 - mix) * nd + mix * ndn
    return Pu.copy(), Pd.copy()


# ============================================================================ training
def sr_step_proj_sampled(ans, H, sg, args, rng):
    """Projected SR step with a SAMPLED force (F5). Same draws give force and metric.
    Returns (sampled energy estimate, exact energy [log only], force)."""
    psi_t = ans.amplitudes(args.chunk)
    psi = sg.project(psi_t)
    Hp = H @ psi
    nn = psi @ psi
    E_ex = float(psi @ Hp / nn)
    idx = rng.choice(len(psi), size=args.sr_samples, p=psi ** 2 / nn)
    El = Hp[idx] / psi[idx]
    O = PC.per_sample_O_proj(ans, sg, idx, psi_t, psi, args.o_chunk)
    n = len(idx)
    Oc = (O - O.mean(0)) / np.sqrt(n)
    f = Oc.T @ ((El - El.mean()) / np.sqrt(n))
    lam = args.sr_shift
    Kmat = Oc @ Oc.T + lam * np.eye(n)
    dx = (f - Oc.T @ np.linalg.solve(Kmat, Oc @ f)) / lam
    nrm = np.linalg.norm(dx)
    step = args.sr_lr * dx
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    ans.set_flat(ans.flat() - step)
    return float(El.mean()), E_ex, f


def optimize(ans, H, sg, args, rng, label, sampled_force=False):
    """Projected SR (exact or sampled force). Keeps the best iterate by the selection energy
    (exact energy, or the sampled estimate when sampled_force) and returns noise statistics."""
    best_E, best_x = np.inf, ans.flat().copy()
    hist, hist_ex = [], []
    t0 = time.time()
    for it in range(args.sr_iters):
        x_before = ans.flat().copy()
        if sampled_force:
            Esel, Eex, f = sr_step_proj_sampled(ans, H, sg, args, rng)
        else:
            Eex, f = PC.sr_step_proj(ans, H, sg, args, rng)
            Esel = Eex
        if not np.isfinite(Esel):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x)
            break
        hist.append(float(Esel))
        hist_ex.append(float(Eex))
        if Esel < best_E:
            best_E, best_x = Esel, x_before
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {Esel:.8f}  |f| = {np.linalg.norm(f):.2e}  "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    psi = sg.project(ans.amplitudes(args.chunk))
    if sampled_force:
        E_fin = mc_energy_vec(H, psi, args.sr_samples, rng)[0]
    else:
        E_fin = P.energy(H, psi)[0]
    if E_fin > best_E:
        ans.set_flat(best_x)
    w = hist[-args.window:] if hist else [np.nan]
    st = dict(sigma=float(np.std(w)), win_mean=float(np.mean(w)), win_min=float(np.min(w)),
              win_max=float(np.max(w)), best_sel=float(best_E), n_iter=len(hist),
              hist_ex_tail=[float(x) for x in hist_ex[-args.window:]])
    return ans, st


def new_ansatz(inst, lab, width, seed):
    Pu, Pd = uhf_pattern(inst.m, "neel" if lab == "A" else "stripe", seed)
    return P.Ansatz(inst.m, Pu, Pd, "nnbf", width=width, jastrow=False,
                    seed=seed + width + (0 if lab == "A" else 7))


def train_branch(inst, tp, lab, width, seed, args, theta0=None, sampled_force=False, tag=""):
    """Projected-SR NNBF at t'. Cold: from the A (Neel) or B (stripe) UHF reference.
    Warm (theta0 given): from the snapshot parameters theta0. Returns (psi_P, theta, stats)."""
    inst.set_T(tp)
    ans = new_ansatz(inst, lab, width, seed)
    if theta0 is not None:
        ans.set_flat(np.array(theta0, float).copy())
    off = (0 if lab == "A" else 7) + (0 if theta0 is None else 101) + (1009 if sampled_force else 0)
    rng = np.random.default_rng(seed + off)
    H = inst.H(tp)
    ans, st = optimize(ans, H, inst.sg, args, rng,
                       f"{tag}{lab} t'={tp:+.5f} W={width} s={seed}", sampled_force)
    psi = inst.sg.project(ans.amplitudes(args.chunk))
    return psi / np.linalg.norm(psi), ans.flat().copy(), st


def state_info(inst, psi, tp, E0):
    H = inst.H(tp)
    E = float(psi @ (H @ psi))
    var = float(np.sum((H @ psi) ** 2) - E ** 2)
    ps = inst.spin.project(psi, 0)
    nps = np.linalg.norm(ps)
    Es = float(ps @ (H @ ps)) / nps ** 2 if nps > 1e-10 else float("nan")
    return dict(E=E, dE=E - E0, var=var, c=inst.char(psi), S2=inst.spin.expect(psi),
                E_spinproj=Es, dE_spinproj=Es - E0)


# ============================================================================ decisions
def order_of(c, tau):
    return "U" if abs(c) < tau else ("B" if c > 0 else "A")


def control_decision(br, truth, tau, z, init_order=None):
    """br: {X: dict(E, sigma, c)}. Lower-energy branch -> order of its c, abstaining when
    |E_A - E_B| <= z * hypot(sigma_A, sigma_B) (F3). Returns (dec, dec_abs, regime, info).
    dec_abs additionally abstains when a branch ends with an order other than init_order[X]."""
    dE = br["A"]["E"] - br["B"]["E"]
    noise = float(np.hypot(br["A"]["sigma"], br["B"]["sigma"]))
    win = "A" if dE < 0 else "B"
    raw = order_of(br[win]["c"], tau)
    dec = raw if abs(dE) > z * noise else "U"
    oA, oB = order_of(br["A"]["c"], tau), order_of(br["B"]["c"], tau)
    if oA == oB:
        regime = "crossover"
    elif dec == "U":
        regime = "noise"
    else:
        regime = "correct" if dec == truth else "intrinsic"
    flipped = None
    dec_abs = dec
    if init_order is not None:
        flipped = {X: order_of(br[X]["c"], tau) != init_order[X] for X in "AB"}
        if any(flipped.values()):
            dec_abs = "U"
    return dec, dec_abs, regime, dict(dE=float(dE), noise=noise, raw=raw, winner=win,
                                      ends=oA + oB, flipped=flipped)


def ec_block(inst, vecs, tp, mats=None):
    """Lowest generalized Ritz root of H(x*) in span(vecs). mats: optional sampled (S, {k: H_k})."""
    if mats is None:
        X = np.stack(vecs, 1)
        S = X.T @ X
        Hk = {k: X.T @ (inst.cl.parts_full[k] @ X) for k in PARTS}
    else:
        S, Hk = mats
    c = inst.coup(tp)
    Hm = sum(c[k] * Hk[k] for k in PARTS)
    S, Hm = 0.5 * (S + S.T), 0.5 * (Hm + Hm.T)
    w, U = np.linalg.eigh(S)
    keep = w > 1e-8 * w.max()
    T = U[:, keep] / np.sqrt(w[keep])
    return float(np.linalg.eigvalsh(T.T @ Hm @ T)[0])


def at_block(inst, pvecs, tp):
    c = inst.coup(tp)
    x = np.array([c[k] for k in PARTS])
    return float(min(x @ p for p in pvecs))


def decide_pair(thA, thB, seA=0.0, seB=0.0, z=2.0):
    if not np.isfinite(thA) and not np.isfinite(thB):
        return "U"
    if not np.isfinite(thB):
        return "A"
    if not np.isfinite(thA):
        return "B"
    if abs(thA - thB) < z * np.hypot(seA, seB):
        return "U"
    return "A" if thA < thB else "B"


# ============================================================================ sampling
def mc_block_mats(inst, vecs, n, rng):
    """Mixture-sampled S_ij and H_k,ij of a same-order block; returns mats and per-sample
    relative standard errors of the off-diagonal entries."""
    X = np.stack([v / np.linalg.norm(v) for v in vecs], 1)
    rho = np.mean(X ** 2, 1)
    idx = rng.choice(len(rho), size=n, p=rho / rho.sum())
    Xs = X[idx] / rho[idx, None]
    S = (X[idx].T @ Xs) / n
    Hk, rel = {}, []
    for k in PARTS:
        OX = (inst.cl.parts_full[k] @ X)[idx]
        Hk[k] = (Xs.T @ OX) / n
    K = X.shape[1]
    if K > 1:
        for i in range(K):
            for j in range(i + 1, K):
                terms = X[idx, i] * Xs[:, j]
                rel.append(float(np.std(terms) / max(abs(np.mean(terms)), 1e-300)))
    return (S, Hk), rel


def mc_energy_vec(H, psi, n, rng):
    p = psi ** 2 / (psi @ psi)
    idx = rng.choice(len(p), size=n, p=p)
    el = (H @ psi)[idx] / psi[idx]
    return float(el.mean()), float(el.std() / np.sqrt(n))


# ============================================================================ checks
def run_checks(inst, seed):
    ok = True
    m, sg = inst.m, inst.sg
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(m.D)
    H = inst.H(inst.tpc)
    hv = np.linalg.norm(H @ v)
    comm = max(np.linalg.norm(sg.apply(H @ v, k) - H @ sg.apply(v, k)) / hv for k in range(sg.G))
    Pv = sg.project(v)
    idem = np.linalg.norm(sg.project(Pv) - Pv) / np.linalg.norm(Pv)
    hc = np.linalg.norm(sg.project(H @ v) - H @ Pv) / hv
    log(f"check [H(t_y={inst.ty}),U_g] max {comm:.1e}; projector idempotent {idem:.1e}; [P,H] {hc:.1e}")
    ok &= comm < 1e-12 and idem < 1e-12 and hc < 1e-12
    inst.set_T(inst.tpc)
    a = new_ansatz(inst, "A", 4, seed)
    a.p["W2"] *= 100
    x0 = a.flat().copy()
    E, f, psi_t, psi, nn = PC.proj_energy_grad(a, H, sg, 65536)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    Ef = lambda x: (a.set_flat(x), P.energy(H, sg.project(a.amplitudes(65536)))[0])[1]
    fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
    a.set_flat(x0)
    log(f"check projected gradient (anisotropic H): analytic {2 * f @ d:.10e} FD {fd:.10e}")
    ok &= abs(2 * f @ d - fd) < 1e-6 * max(1.0, abs(fd))
    # F5: sampled force (the estimator used by sr_step_proj_sampled) vs exact force
    ns = 20000
    idx = rng.choice(len(psi), size=ns, p=psi ** 2 / nn)
    El = (H @ psi)[idx] / psi[idx]
    O = PC.per_sample_O_proj(a, sg, idx, psi_t, psi, 256)
    terms = (O - O.mean(0)) * (El - El.mean())[:, None]
    fs = terms.mean(0)
    se2 = float(np.sum(terms.var(0)) / ns)          # expected |fs - f|^2 if unbiased
    chi = float(np.sum((fs - f) ** 2) / se2)
    cos = float(fs @ f / (np.linalg.norm(fs) * np.linalg.norm(f)))
    log(f"check sampled force vs exact ({ns} draws): |fs-f|^2 / sum(SE^2) = {chi:.2f} (unbiased -> ~1), "
        f"cosine {cos:.4f}, |fs-f|/|f| {np.linalg.norm(fs - f) / np.linalg.norm(f):.3f}")
    ok &= chi < 3.0
    # F1: warm start reproduces the stored state
    b = new_ansatz(inst, "B", 4, seed + 1)
    b.set_flat(x0.copy())
    wdiff = np.linalg.norm(sg.project(b.amplitudes(65536)) - psi) / np.linalg.norm(psi)
    log(f"check warm start from stored parameters: relative amplitude difference {wdiff:.1e}")
    ok &= wdiff < 1e-12
    # MC mixture estimator vs exact
    _, p0 = inst.ed(inst.tpc + 0.05, check_global=False)
    _, p1 = inst.ed(inst.tpc + 0.08, check_global=False)
    n = 20000
    (Sm, _), rel = mc_block_mats(inst, [p0, p1], n, rng)
    ex = (p0 / np.linalg.norm(p0)) @ (p1 / np.linalg.norm(p1))
    se = rel[0] * abs(Sm[0, 1]) / np.sqrt(n)
    err = abs(Sm[0, 1] - ex) / max(se, 1e-14)
    log(f"check MC mixture S_01 {Sm[0, 1]:.6f} vs exact {ex:.6f} ({err:.1f} standard errors)")
    ok &= err < 5
    # F3: abstention logic on synthetic branches
    t1 = control_decision({"A": dict(E=-1.0, sigma=1e-3, c=-0.2), "B": dict(E=-1.001, sigma=1e-3, c=0.2)},
                          "A", 0.02, 2.0)
    t2 = control_decision({"A": dict(E=-1.0, sigma=1e-4, c=0.2), "B": dict(E=-1.01, sigma=1e-4, c=0.2)},
                          "A", 0.02, 2.0, dict(A="A", B="B"))
    t3 = control_decision({"A": dict(E=-1.0, sigma=1e-4, c=-0.2), "B": dict(E=-1.01, sigma=1e-4, c=0.2)},
                          "A", 0.02, 2.0, dict(A="A", B="B"))
    log(f"check decision logic: noise->{t1[0]}/{t1[2]} crossover->{t2[0]},{t2[1]}/{t2[2]} "
        f"intrinsic->{t3[0]}/{t3[2]}")
    ok &= (t1[0] == "U" and t1[2] == "noise" and t2[0] == "B" and t2[1] == "U"
           and t2[2] == "crossover" and t3[0] == "B" and t3[2] == "intrinsic")
    log(f"P3.2-fix CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ run
def run(args):
    t0 = time.time()
    inst = Instance(args.ty, args.U)
    log(f"instance: 4x3 PBC 4+4 U={args.U} t_y={args.ty}; t'_c={inst.tpc:+.6f} min gap "
        f"{inst.gapmin:.3e} ({time.time() - t0:.0f}s); irrep dim {inst.cl.dS}")
    if not run_checks(inst, args.seed):
        sys.exit(2)
    if args.checks_only:
        return
    npar = new_ansatz(inst, "A", args.width, args.seed).n_params()
    res = dict(args={k: v for k, v in vars(args).items()}, tpc=inst.tpc, gapmin=inst.gapmin,
               targets=[], snaps=[], sector_dim=inst.cl.dS, spacegroup_irrep_dim=inst.cl.dB,
               n_params=npar, dim_over_params=inst.cl.dS / npar)
    log(f"projected sector dim range(P_S P_chi) = {inst.cl.dS} (space-group irrep {inst.cl.dB}); "
        f"NNBF width {args.width}: {npar} parameters; ratio {inst.cl.dS / npar:.2f}")
    dump = lambda: json.dump(res, open(args.out, "w"), indent=1)
    rngmc = np.random.default_rng(args.seed + 999)
    # ---------------------------------------------------------------- snapshots (F2)
    snap = []          # certified (Eckart) snapshots: dict(d, label, psi, theta, c)
    allbr = []         # all snapshot branches with their measured order
    oracle_vecs = {"A": [], "B": []}
    for d in args.snap_list:
        tp = inst.tpc + d
        e, p0 = inst.ed(tp, check_global=True)
        oracle_vecs["A" if e["c0"] < 0 else "B"].append(p0)
        br = {lab: train_branch(inst, tp, lab, args.width, args.seed, args, tag="snap ") for lab in "AB"}
        info = {lab: dict(state_info(inst, br[lab][0], tp, e["E0"]), **{k: v for k, v in br[lab][2].items()
                                                                       if k != "hist_ex_tail"})
                for lab in "AB"}
        win = min("AB", key=lambda L: info[L]["E"])
        lab = order_of(info[win]["c"], args.tau)
        eckart = bool(info[win]["dE"] < e["gap"] / 2) and lab != "U"
        if eckart:
            snap.append(dict(d=d, label=lab, psi=br[win][0], theta=br[win][1], c=info[win]["c"]))
        for L in "AB":   # every snapshot branch, by measured order (warm-source fallback only)
            oL = order_of(info[L]["c"], args.tau)
            if oL != "U":
                allbr.append(dict(d=d, label=oL, theta=br[L][1], E=info[L]["E"]))
        res["snaps"].append(dict(d=d, ed=e, branches=info, winner_seed=win, label=lab,
                                 eckart=eckart, collapsed=bool(order_of(info["A"]["c"], args.tau)
                                                               == order_of(info["B"]["c"], args.tau)),
                                 p=inst.pvec(br[win][0]).tolist()))
        log(f"snap d={d:+.4f}: gap={e['gap']:.4f} c0={e['c0']:+.3f} E0-E0glob={e['E0'] - e['E0_global']:.1e} "
            f"| A dE={info['A']['dE']:.2e} c={info['A']['c']:+.3f} | B dE={info['B']['dE']:.2e} "
            f"c={info['B']['c']:+.3f} -> {lab} eckart={eckart}")
        dump()
    if snap:
        np.savez(args.out[:-5] + "_snap_theta.npz", **{f"d{s['d']:+.4f}_{s['label']}": s["theta"] for s in snap})
    blocks = {X: [s for s in snap if s["label"] == X] for X in "AB"}
    res["blocks"] = {X: [s["d"] for s in blocks[X]] for X in "AB"}
    log(f"certified (Eckart) snapshot blocks: A at {res['blocks']['A']}, B at {res['blocks']['B']}")
    # ---------------------------------------------------------------- targets
    for d in args.target_list:
        tp = inst.tpc + d
        e, _ = inst.ed(tp, check_global=True)
        truth = order_of(e["c0"], args.tau)
        row = dict(d=d, ed=e, truth=truth, dec={}, regime={}, ctrl={})
        # cold control
        cold = {lab: train_branch(inst, tp, lab, args.width, args.seed, args, tag="cold ") for lab in "AB"}
        # warm control (F1): nearest certified snapshot of each order
        src, warm = {}, {}
        for X in "AB":
            pool, kind = blocks[X], "certified"
            if not pool:     # no certified snapshot of order X: nearest branch of measured order X
                pool, kind = [b for b in allbr if b["label"] == X], "uncertified"
            if pool:
                s = min(pool, key=lambda s_: (abs(s_["d"] - d), s_.get("E", 0.0)))
                src[X] = dict(d=s["d"], kind=kind, theta=s["theta"])
                warm[X] = train_branch(inst, tp, X, args.width, args.seed, args, theta0=s["theta"], tag="warm ")
            else:
                src[X] = None
                warm[X] = cold[X]
        arms = {"R0P": (cold, None), "R0P-warm": (warm, {X: X for X in "AB"})}
        if d in args.sf_list:
            sf = {}
            for X in "AB":
                th = src[X]["theta"] if src[X] is not None else None
                sf[X] = train_branch(inst, tp, X, args.width, args.seed, args, theta0=th,
                                     sampled_force=True, tag="warm-sf ")
            arms["R0P-warm-sf"] = (sf, {X: X for X in "AB"})
        H = inst.H(tp)
        for nm, (brs, init) in arms.items():
            inf = {}
            for X in "AB":
                inf[X] = dict(state_info(inst, brs[X][0], tp, e["E0"]),
                              **{k: v for k, v in brs[X][2].items() if k != "hist_ex_tail"})
                if nm == "R0P-warm-sf":   # decision energies from VMC only (F5)
                    Em, se = mc_energy_vec(H, brs[X][0], args.n_samples, rngmc)
                    inf[X]["E_exact"], inf[X]["E"] = inf[X]["E"], Em
                    inf[X]["vmc_se"] = se
                    inf[X]["sigma"] = float(np.hypot(inf[X]["sigma"], se))
            dec, dec_abs, reg, di = control_decision(inf, truth, args.tau, args.z, init)
            row["ctrl"][nm] = dict(branches=inf, **di)
            row["dec"][nm] = dec
            row["regime"][nm] = reg
            if nm == "R0P-warm":
                row["dec"]["R0P-warm-abs"] = dec_abs
                row["ctrl"][nm]["sources"] = {X: ({"d": src[X]["d"], "kind": src[X]["kind"]} if src[X] else None)
                                              for X in "AB"}
            if nm == "R0P-warm-sf":
                row["dec"]["R0P-warm-sf-abs"] = dec_abs
        # EC on certified snapshots, AT and oracles (diagnostics)
        th = {}
        vv = {X: [s["psi"] for s in blocks[X]] for X in "AB"}
        for nm, vecs in (("EC", vv), ("ECor", oracle_vecs)):
            th[nm] = {X: (ec_block(inst, vecs[X], tp) if vecs[X] else np.inf) for X in "AB"}
            row["dec"][nm] = decide_pair(th[nm]["A"], th[nm]["B"])
        for nm, vecs in (("AT", vv), ("ATor", oracle_vecs)):
            th[nm] = {X: (at_block(inst, [inst.pvec(v) for v in vecs[X]], tp) if vecs[X] else np.inf)
                      for X in "AB"}
            row["dec"][nm] = decide_pair(th[nm]["A"], th[nm]["B"])
        # sampled EC (rule 4 for the route)
        relS, ecs = [], []
        for _ in range(args.n_rep):
            thE = {}
            for X in "AB":
                if not vv[X]:
                    thE[X] = np.inf
                    continue
                mats, rel = mc_block_mats(inst, vv[X], args.n_samples, rngmc)
                relS += rel
                thE[X] = ec_block(inst, vv[X], tp, mats)
            ecs.append((thE["A"], thE["B"]))
        ecA = np.array([t[0] for t in ecs])
        ecB = np.array([t[1] for t in ecs])
        seA = float(np.std(ecA)) if np.all(np.isfinite(ecA)) else 0.0
        seB = float(np.std(ecB)) if np.all(np.isfinite(ecB)) else 0.0
        row["sampled"] = {"EC": [decide_pair(a_, b_, seA, seB) for a_, b_ in zip(ecA, ecB)]}
        row["theta"] = {k: {X: float(v[X]) for X in "AB"} for k, v in th.items()}
        row["ec_offdiag_rel_se_per_sample"] = float(np.median(relS)) if relS else None
        row["ec_root_se"] = dict(A=seA, B=seB)
        res["targets"].append(row)
        c = row["ctrl"]
        log(f"target d={d:+.4f} gap={e['gap']:.4f} truth={truth} | "
            + " | ".join(f"{nm}: dE_AB={c[nm]['dE']:+.2e} noise={c[nm]['noise']:.1e} ends={c[nm]['ends']} "
                         f"dec={row['dec'][nm]} [{row['regime'][nm]}]" for nm in c)
            + f" | warm-abs={row['dec']['R0P-warm-abs']} EC={row['dec']['EC']} ECor={row['dec']['ECor']} "
              f"AT(diag)={row['dec']['AT']} | sampled EC {''.join(row['sampled']['EC'])}")
        dump()
    res["wall_s"] = time.time() - t0
    dump()
    log(f"done ({res['wall_s']:.0f}s) -> {args.out}")


# ============================================================================ aggregate
SAME_KEYS = ("ty", "U", "targets", "snaps", "sf_targets", "sr_iters", "sr_lr", "sr_shift",
             "sr_samples", "window", "z", "n_samples", "n_rep", "tau")


def check_consistent(runs, W):
    """F2: identical targets, snapshots, t'_c and settings across the runs of a width."""
    r0 = runs[0]
    seeds = [r["args"]["seed"] for r in runs]
    assert len(set(seeds)) == len(seeds), f"width {W}: duplicate seeds {seeds}"
    for r in runs[1:]:
        for k in SAME_KEYS:
            assert r["args"][k] == r0["args"][k], f"width {W}: arg {k} differs ({r['args'][k]} vs {r0['args'][k]})"
        assert [t["d"] for t in r["targets"]] == [t["d"] for t in r0["targets"]], f"width {W}: target lists differ"
        assert [s["d"] for s in r["snaps"]] == [s["d"] for s in r0["snaps"]], f"width {W}: snapshot lists differ"
        assert abs(r["tpc"] - r0["tpc"]) < 1e-7, f"width {W}: t'_c differs"
    for r in runs:
        assert len(r["targets"]) == len(r["args"]["target_list"]), f"width {W} seed {r['args']['seed']}: incomplete run"
        assert len(r["snaps"]) == len(r["args"]["snap_list"]), f"width {W} seed {r['args']['seed']}: incomplete snapshots"
        for t in r["targets"] + r["snaps"]:
            assert t["ed"].get("irrep_is_global") is True, f"width {W}: irrep E0 not checked against global E0"


def rates(pairs, arm, sampled=False):
    cw = dec = n = 0
    for r, t in pairs:
        ds_ = t["sampled"][arm] if sampled else [t["dec"][arm]]
        for x in ds_:
            n += 1
            dec += x != "U"
            cw += x not in ("U", t["truth"])
    return (cw / n, dec / n, n) if n else (float("nan"), float("nan"), 0)


def aggregate(d):
    files = sorted(glob.glob(os.path.join(d, "w*_s*.json")))
    by_w = {}
    for f in files:
        r = json.load(open(f))
        by_w.setdefault(r["args"]["width"], []).append(r)
    verdict = []
    for W, runs in sorted(by_w.items()):
        print(f"=== width {W}: seeds {[r['args']['seed'] for r in runs]}  sector dim "
              f"{runs[0].get('sector_dim')}  n_params {runs[0].get('n_params')}  ratio "
              f"{runs[0].get('dim_over_params', float('nan')):.2f}")
        if len(runs) < 3:
            print("  INCOMPLETE (< 3 seeds)")
            verdict.append("INCOMPLETE")
            continue
        check_consistent(runs, W)
        print("  consistency asserts passed (targets, snapshots, t'_c, settings, distinct seeds, irrep=global E0)")
        # ---- F2 seed gate on snapshots
        nsn = len(runs[0]["snaps"])
        seed_ok = {id(r): True for r in runs}
        ncert_per_run = {id(r): 0 for r in runs}
        for j in range(nsn):
            labs = [r["snaps"][j]["label"] for r in runs]
            maj = max("AB", key=labs.count)
            agree = labs.count(maj) >= max(2, int(np.ceil(2 * len(runs) / 3)))
            for r in runs:
                s = r["snaps"][j]
                good = agree and s["label"] == maj and s["eckart"]
                ncert_per_run[id(r)] += good
                if s["eckart"] and not good:
                    seed_ok[id(r)] = False   # an EC block of this run contains a seed-gate failure
            print(f"  snap d={runs[0]['snaps'][j]['d']:+.4f} labels {''.join(labs)} majority {maj} "
                  f"seed-agree {agree} eckart {''.join('y' if r['snaps'][j]['eckart'] else 'n' for r in runs)} "
                  f"collapsed {''.join('y' if r['snaps'][j]['collapsed'] else 'n' for r in runs)}")
        ncert = int(np.median(list(ncert_per_run.values())))
        blocks = all(r["blocks"][X] for r in runs for X in "AB")
        for r in runs:   # EC decisions of a seed with a seed-gate failure become abstentions
            if not seed_ok[id(r)]:
                for t in r["targets"]:
                    t["dec"]["EC"] = "U"
                    t["sampled"]["EC"] = ["U"] * len(t["sampled"]["EC"])
        # ---- premise gate
        ds = [t["d"] for t in runs[0]["targets"]]
        hard = []
        for i, dd in enumerate(ds):
            eps = np.median([min(r["targets"][i]["ctrl"][c]["branches"][X]["dE"]
                                 for c in ("R0P", "R0P-warm") for X in "AB") for r in runs])
            gap = runs[0]["targets"][i]["ed"]["gap"]
            h = bool(eps >= gap / 2 and runs[0]["targets"][i]["truth"] != "U")
            hard.append(h)
            print(f"  target d={dd:+.4f} gap={gap:.4f} eps_med={eps:.2e} hard={h} truth={runs[0]['targets'][i]['truth']} "
                  + " ".join(f"{k}:{''.join(r['targets'][i]['dec'][k] for r in runs)}"
                             for k in CONTROLS + ("EC", "ECor", "AT"))
                  + " | regime cold:" + ",".join(r["targets"][i]["regime"]["R0P"] for r in runs)
                  + " warm:" + ",".join(r["targets"][i]["regime"]["R0P-warm"] for r in runs))
        pairs = [(r, r["targets"][i]) for r in runs for i, h in enumerate(hard) if h]
        nh = int(sum(hard))
        print(f"  hard targets {nh}/{len(ds)}; certified snapshots (Eckart+seed, median seed) {ncert}/{nsn}; "
              f"both blocks {blocks}")
        # ---- F4 regime table (always printed)
        for c in ("R0P", "R0P-warm"):
            cnt = {k: sum(t["regime"][c] == k for _, t in pairs) for k in ("correct", "noise", "intrinsic", "crossover")}
            print(f"  regime {c:9s} on hard pairs: {cnt}")
        nunc = sum(1 for _, t in pairs for X in "AB"
                   if (t["ctrl"]["R0P-warm"].get("sources") or {}).get(X) is None
                   or t["ctrl"]["R0P-warm"]["sources"][X]["kind"] != "certified")
        print(f"  warm-start sources not certified (branch-count over hard pairs): {nunc}")
        sfp = [(r, t) for r, t in pairs if "R0P-warm-sf" in t["dec"]]
        if sfp:
            cws, dcs, nsf = rates(sfp, "R0P-warm-sf")
            cwe, dce, _ = rates(sfp, "R0P-warm")
            agree_sf = np.mean([t["dec"]["R0P-warm-sf"] == t["dec"]["R0P-warm"] for _, t in sfp])
            print(f"  rule 4 (sampled-force warm control, {nsf} hard pairs): CW={cws:.3f} DEC={dcs:.3f} vs "
                  f"exact-force warm CW={cwe:.3f} DEC={dce:.3f}; decision agreement {agree_sf:.2f}")
        if nh < 3 or ncert < 4 or not blocks:
            why = ("easy: fewer than 3 targets with best control error >= gap/2" if nh < 3 else
                   "snapshots: fewer than 4 certified or an order block empty")
            print(f"  PREMISE-FAIL at this width ({why})")
            verdict.append("PREMISE-FAIL:" + ("easy" if nh < 3 else "snapshots"))
            continue
        ctrl = {c: rates(pairs, c) for c in CONTROLS}
        for c, (cw, dc, n) in ctrl.items():
            print(f"  {c:13s} CW={cw:.3f} DEC={dc:.3f} (n={n})")
        if all(ctrl[c][0] == 0 for c in CONTROLS):
            print("  CONTROL: NO-ROOM (every control has CW = 0 on the hard targets) -> SO-EC FAIL (no room)")
            verdict.append("FAIL:no-room")
            continue
        # ---- SO-EC on the warm-crossover subset (pre-registered)
        X_ = [(r, t) for r, t in pairs if t["regime"]["R0P-warm"] == "crossover"]
        cwA, dcA, _ = rates(pairs, "EC")
        print(f"  EC on all hard pairs (report only): CW={cwA:.3f} DEC={dcA:.3f}")
        if len(X_) < 3:
            print(f"  SO-EC: FAIL (only {len(X_)} warm-crossover hard pairs; pre-registered: no win possible)")
            verdict.append("FAIL:no-crossover")
            continue
        cw, dc, _ = rates(X_, "EC")
        cws, _, _ = rates(X_, "EC", True)
        fl = float(np.mean([x != t["dec"]["EC"] for _, t in X_ for x in t["sampled"]["EC"]]))
        cx = {c: rates(X_, c) for c in CONTROLS}
        dominated = any(cx[c][0] <= cw and cx[c][1] >= dc for c in CONTROLS)
        beats = all(cx[c][0] - cw >= 0.15 or (cw <= cx[c][0] and dc >= cx[c][1] + 0.2) for c in CONTROLS)
        if cw > 0.30 or dominated:
            v = "FAIL"
        elif cw <= 0.15 and dc >= 0.6 and beats and fl <= 0.10 and cws <= 0.15:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        print(f"  crossover subset n={len(X_)}: " + " ".join(f"{c} CW={cx[c][0]:.3f}/DEC={cx[c][1]:.3f}" for c in CONTROLS)
              + f" | EC CW={cw:.3f} DEC={dc:.3f} sampled CW={cws:.3f} flip={fl:.3f} -> {v}")
        verdict.append(v)
    if verdict and all(v.startswith("PREMISE-FAIL") for v in verdict):
        g = "PREMISE-FAIL (route untested)"
    elif "PASS" in verdict and not any(v.startswith("FAIL") for v in verdict):
        g = "PASS"
    elif any(v.startswith("FAIL") for v in verdict):
        g = "FAIL"
    else:
        g = "AMBIGUOUS" if verdict else "NO DATA"
    print(f"GLOBAL VERDICT SO-EC: {g}  (per width: {verdict})")


# ============================================================================ cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--ty", type=float, default=1.11394)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--targets", default="-0.002,-0.0005,-0.0002,0.0002,0.0005,0.002")
    ap.add_argument("--snaps", default="-0.08,-0.05,-0.03,0.03,0.05,0.08")
    ap.add_argument("--sf-targets", default="-0.0002,0.0002")
    ap.add_argument("--sr-iters", type=int, default=800)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--o-chunk", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--checks-only", type=int, default=0)
    ap.add_argument("--aggregate", default="")
    a = ap.parse_args(argv)
    if a.task_id >= 0:
        a.width = (4, 8)[a.task_id // 3]
        a.seed = (1234, 2345, 3456)[a.task_id % 3]
    sub = "results"
    if a.smoke:
        a.targets, a.snaps, a.sf_targets, a.sr_iters, a.width = "0.002", "-0.05,0.05", "0.002", 3, 4
        a.n_samples, a.n_rep, a.sr_samples = 2000, 3, 200
        sub = "results_smoke"
    a.target_list = [float(s) for s in a.targets.split(",")]
    a.snap_list = [float(s) for s in a.snaps.split(",")]
    a.sf_list = [float(s) for s in a.sf_targets.split(",") if s.strip()]
    if not a.out:
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
        a.out = os.path.join(HERE, sub, f"w{a.width}_s{a.seed}.json")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.aggregate:
        d = a.aggregate if os.path.isabs(a.aggregate) else os.path.join(HERE, a.aggregate)
        aggregate(d)
        return
    run(a)


if __name__ == "__main__":
    main()
