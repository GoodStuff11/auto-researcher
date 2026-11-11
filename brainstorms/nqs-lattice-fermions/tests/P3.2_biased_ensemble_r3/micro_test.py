#!/usr/bin/env python3
"""
P3.2 round-3 decisive test: deciding a same-irrep competing order at a near-conical
intersection, with EVERY arm symmetry-projected
=====================================================================================

Question
--------
At a genuine same-irrep, near-degenerate crossing of two orders, where the projected NNBF
energy error is comparable to or larger than the in-irrep gap, does a decision taken from
projected-NQS SNAPSHOTS trained only far from the crossing, and continued to the target
point, beat the projected two-branch control (projected R0+)? The continuation is either
same-order eigenvector continuation (SO-EC, Route 3.5) or the affine transport envelope
(AT, Route 3.6).

Instance (found by ed_scan.py; see scan_logs/)
----------------------------------------------
4x3 PBC Hubbard, N_up = N_dn = 4, U = 8, t_x = 1, anisotropic NN hopping t_y = 1.11394, and
next-nearest t' on a path through the cone of an exact same-irrep conical intersection
(CI at t' = -0.515243, t_y = 1.143944). Irrep: K = 0, M_x = M_y = +1, S = 0, which is also
the global ground-state sector. On the path the in-irrep gap reaches its minimum 1.0e-3 t
(8e-5 t per site) at t'_c ~ -0.4705; the two lowest irrep states are a period-4 spin stripe
(c > 0, label B) and a near-Neel state (c < 0, label A), and they swap at t'_c. The gap to
the third irrep level is ~0.67 t. t'_c is recomputed at run time (bounded minimisation of
the ED gap) and stored.
  Targets: t' = t'_c + d, d in --targets (default +-0.0005, +-0.002, +-0.005, +-0.02;
           in-irrep gaps ~0.003 to 0.1 t).
  Snapshot points: t' = t'_c + d, d in --snaps (default +-0.03, +-0.05, +-0.08; gaps ~0.15
           to 0.4 t, i.e. far from the crossing).

Order character (diagonal operator, as in round 2)
  c = [S_s(q_B) + S_s(-q_B)]/2 - [S_s(q_A) + S_s(q_A')]/2,  q_B = (pi/2, 0),
  q_A, q_A' = (pi, +-2pi/3),  S_s(q) = <|sum_i e^{iq.r_i} S^z_i|^2>/V.
  c > 0 -> B (stripe), c < 0 -> A (near-Neel). Truth = sign c(E0) of the ED irrep ground
  state; "mixed" if |c(E0)| < --tau (none of the default targets is mixed: |c(E0)| > 0.2).

Arms (one Slurm task = one (width, seed)); all states are projected onto the irrep
--------------------------------------------------------------------------------
Every variational state is psi_P = P_chi psi_theta, trained by projected SR (exact force,
sampled metric) with the verified code of tests/P1.1P_projected (copied in proj_core.py):
P_chi = (1/|G|) sum_g chi(g) U_g over the 48-element space group of the 4x3 torus
(translations x {1, M_x, M_y, M_xM_y}; t_y != t_x keeps all of them). Spin: the projected
states are not spin-projected in training; <S^2> is recorded and the post-hoc spin-projected
energy P_S psi_P is reported for every arm (rule 1: a post-hoc space-group projection of a
projected state is the identity, so the post-hoc numbers reported are the spin ones).
  R0P  control (projected R0+). At each target, two projected NNBF branches seeded by a
       Neel-UHF (A) and a stripe-UHF (B) reference, --sr-iters steps each. Decision = sign of
       c of the lower-energy branch.
  SNAP at each snapshot point the same two branches are trained; the lower-energy one is the
       snapshot; its order label is the sign of its MEASURED c (never the side or the seed).
       "Certified" if both seeds end with the same sign of c (seed agreement).
  EC   Route 3.5, same-order eigenvector continuation. For order X, the snapshots labelled X
       span a block; at the target x* the generalized eigenproblem
       H^X(x*) a = theta S^X a, H^X_ij(x*) = sum_k x*_k <psi_i|H_k|psi_j>, S^X_ij = <psi_i|psi_j>
       (H_k = H_tx, H_ty, H_t', H_U) gives theta_X = lowest root. Only SAME-ORDER cross terms
       enter. Decision = order with the lower theta_X.
  AT   Route 3.6, affine transport. Each snapshot i has the expectation vector
       p_i = (<H_tx>, <H_ty>, <H_t'>, <H_U>), so f_i(x) = x . p_i is a rigorous variational upper
       bound at every x. theta_X = min over snapshots of order X of f_i(x*). Decision = lower.
  Oracle diagnostics (not arms): EC/AT built from the EXACT ED irrep ground states at the
       snapshot points (is continuation itself sound?).
  Sampled arm (rule 4): for EC the block matrices are re-estimated by Monte Carlo from the
       mixture rho(n) = (1/K) sum_i psi_i(n)^2 / ||psi_i||^2 with --n-samples exact draws,
       M_ij = mean[ psi_i(n) (O psi_j)(n) / rho(n) ]; for R0P and AT the energies and p_i are
       re-estimated by standard VMC (draws from |psi|^2, local energies). --n-rep replicas.
       Reported: relative standard error of every same-order S_ij and H_ij (per sample), the
       sampled decision, and the flip rate against the exact-sum decision. (The local value
       (O psi_j)(n) is read from the vector O psi_j; it equals the VMC local-estimator
       numerator sum_m O_nm psi_j(m), whose cost in a real run is #connections amplitude
       evaluations per sample. No claim in the markdown relies on cheaper evaluation.)
       A sampled arm abstains when its decision gap is below 2 standard errors.

Pre-committed decision rule (--aggregate; per width W, over seeds S_W, needs >= 3 seeds)
------------------------------------------------------------------------------------
  eps(t)  = median over seeds of min_X [E(branch X at t) - E0(t)]   (projected control error)
  hard(t) = eps(t) >= gap(t)/2                                      (PREMISE gate)
  Valid width: >= 3 hard targets, and >= 4 of 6 snapshot points certified in the median seed,
               and each order block non-empty in every seed. Otherwise the width reports
               PREMISE-FAIL (projected R0+ decides by construction, or no snapshots).
  On hard targets, over (target, seed): CW(R) = committed-wrong fraction, DEC(R) = commit
  fraction, for R in {R0P, EC, AT} (exact sums) and their sampled versions.
  Per route X in {EC, AT}:
    PASS      CW(X) <= 0.15 and DEC(X) >= 0.6 and CW(R0P) - CW(X) >= 0.15 and the sampled
              X has flip rate <= 0.10 and CW(X_sampled) <= CW(R0P_sampled) - 0.10;
    FAIL      CW(X) >= CW(R0P), or CW(X) > 0.30, or CW(R0P) == 0 (no room: the projected
              control is already perfect on the hard targets);
    AMBIGUOUS otherwise. Pre-committed action: rerun with --seeds 6 and widths 8,16; still
              AMBIGUOUS -> KILLED.
  Global: a route PASSes only if it PASSes at >= 1 valid width and FAILs at none.
  PREMISE-FAIL at every width -> the leaf reports "projected NQS resolves this instance; the
  competitor half needs a larger cluster" and both routes are recorded as untested.

Usage
-----
  python3 micro_test.py [options]
Options
  --width INT        NNBF hidden width (default 32).
  --seed INT         Seed (default 1234): UHF noise seed, NNBF init seed + width (+7 for B),
                     SR sampling default_rng(seed); MC replicas default_rng(seed + 999).
  --task-id INT      If >= 0 overrides --width/--seed: width = (8, 32)[t // 3],
                     seed = (1234, 2345, 3456)[t % 3] (array 0-5). Default -1.
  --ty FLOAT         t_y of the path (default 1.11394).  --U FLOAT (default 8).
  --targets LIST     Offsets d from t'_c (default "-0.02,-0.005,-0.002,-0.0005,0.0005,0.002,0.005,0.02").
  --snaps LIST       Snapshot offsets (default "-0.08,-0.05,-0.03,0.03,0.05,0.08").
  --sr-iters INT     Projected SR steps per branch (default 1000).
  --sr-lr FLOAT      (default 0.05)  --sr-shift FLOAT (default 1e-3)
  --sr-samples INT   Metric samples (default 2000).  --chunk INT (65536)  --o-chunk INT (256)
  --n-samples INT    MC draws per sampled estimate (default 20000).  --n-rep INT (default 20).
  --tau FLOAT        Character abstention threshold (default 0.02).
  --out PATH         JSON (default results/w<W>_s<seed>.json next to this script).
  --smoke INT        1: plumbing run on the real instance with 1 target, 2 snapshots, 3 SR
                     steps, width 4, n-samples 2000, n-rep 3 (results in results_smoke/).
  --checks-only INT  1: run the correctness checks and exit.
  --aggregate DIR    Apply the decision rule to every w*_s*.json in DIR and print verdicts.
Correctness checks (run first; exit code 2 on failure): [H(x), U_g] = 0 for all g with
t_y != t_x; projector idempotent and commuting with H; ED irrep E0 equals the global E0 at
every target; the projected gradient matches finite differences; the MC mixture estimator of
S_ij agrees with the exact sum within 5 standard errors.
"""
import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p11_core as P  # noqa: E402
import proj_core as PC  # noqa: E402
import ed_scan as ES  # noqa: E402

log = P.log
PARTS = ("tx", "ty", "tp", "U")
TP_CI, TY_CI = -0.515243, 1.143944


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
def train_branch(inst, tp, lab, width, seed, args):
    """Projected-SR NNBF from the A (Neel) or B (stripe) UHF reference at t'. Returns psi_P."""
    inst.set_T(tp)
    Pu, Pd = uhf_pattern(inst.m, "neel" if lab == "A" else "stripe", seed)
    ans = P.Ansatz(inst.m, Pu, Pd, "nnbf", width=width, jastrow=False,
                   seed=seed + width + (0 if lab == "A" else 7))
    rng = np.random.default_rng(seed + (0 if lab == "A" else 7))
    H = inst.H(tp)
    PC.optimize_sr_proj(ans, H, inst.sg, args, rng, f"{lab} t'={tp:+.5f} W={width} s={seed}")
    psi = inst.sg.project(ans.amplitudes(args.chunk))
    return psi / np.linalg.norm(psi)


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


def mc_energy(inst, psi, tp, n, rng):
    H = inst.H(tp)
    p = psi ** 2 / (psi @ psi)
    idx = rng.choice(len(p), size=n, p=p)
    el = (H @ psi)[idx] / psi[idx]
    return float(el.mean()), float(el.std() / np.sqrt(n))


def mc_pvec(inst, psi, n, rng):
    p = psi ** 2 / (psi @ psi)
    idx = rng.choice(len(p), size=n, p=p)
    out, se = [], []
    for k in PARTS:
        el = (inst.cl.parts_full[k] @ psi)[idx] / psi[idx]
        out.append(el.mean())
        se.append(el.std() / np.sqrt(n))
    return np.array(out), np.array(se)


# ============================================================================ checks
def run_checks(inst, seed):
    ok = True
    m, sg = inst.m, inst.sg
    rng = np.random.default_rng(seed)
    v, y = rng.standard_normal(m.D), rng.standard_normal(m.D)
    H = inst.H(inst.tpc)
    hv = np.linalg.norm(H @ v)
    comm = max(np.linalg.norm(sg.apply(H @ v, k) - H @ sg.apply(v, k)) / hv for k in range(sg.G))
    Pv = sg.project(v)
    idem = np.linalg.norm(sg.project(Pv) - Pv) / np.linalg.norm(Pv)
    hc = np.linalg.norm(sg.project(H @ v) - H @ Pv) / hv
    log(f"check [H(t_y={inst.ty}),U_g] max {comm:.1e}; projector idempotent {idem:.1e}; [P,H] {hc:.1e}")
    ok &= comm < 1e-12 and idem < 1e-12 and hc < 1e-12
    inst.set_T(inst.tpc)
    Pu, Pd = uhf_pattern(m, "neel", seed)
    a = P.Ansatz(m, Pu, Pd, "nnbf", width=4, jastrow=False, seed=seed)
    a.p["W2"] *= 100
    x0 = a.flat().copy()
    E, f, _, _, _ = PC.proj_energy_grad(a, H, sg, 65536)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    Ef = lambda x: (a.set_flat(x), P.energy(H, sg.project(a.amplitudes(65536)))[0])[1]
    fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
    a.set_flat(x0)
    log(f"check projected gradient (anisotropic H): analytic {2 * f @ d:.10e} FD {fd:.10e}")
    ok &= abs(2 * f @ d - fd) < 1e-6 * max(1.0, abs(fd))
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
    log(f"P3.2-r3 CHECKS {'PASSED' if ok else 'FAILED'}")
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
    res = dict(args=vars(args), tpc=inst.tpc, gapmin=inst.gapmin, targets=[], snaps=[])
    dump = lambda: json.dump(res, open(args.out, "w"), indent=1)
    rngmc = np.random.default_rng(args.seed + 999)
    # snapshots
    snap_vecs = {"A": [], "B": []}
    oracle_vecs = {"A": [], "B": []}
    for d in args.snap_list:
        tp = inst.tpc + d
        e, p0 = inst.ed(tp, check_global=False)
        oracle_vecs["A" if e["c0"] < 0 else "B"].append(p0)
        br = {lab: train_branch(inst, tp, lab, args.width, args.seed, args) for lab in "AB"}
        info = {lab: state_info(inst, br[lab], tp, e["E0"]) for lab in "AB"}
        win = min("AB", key=lambda L: info[L]["E"])
        lab = "A" if info[win]["c"] < 0 else "B"
        cert = (info["A"]["c"] > 0) == (info["B"]["c"] > 0)
        snap_vecs[lab].append(br[win])
        res["snaps"].append(dict(d=d, ed=e, branches=info, winner_seed=win, label=lab,
                                 certified=bool(cert), p=inst.pvec(br[win]).tolist()))
        log(f"snap d={d:+.4f}: gap={e['gap']:.4f} c0={e['c0']:+.3f} | A dE={info['A']['dE']:.2e} "
            f"c={info['A']['c']:+.3f} | B dE={info['B']['dE']:.2e} c={info['B']['c']:+.3f} -> {lab} cert={cert}")
        dump()
    # sampled snapshot energy vectors (AT) and EC block variances: per replica
    pv_s = {X: [[mc_pvec(inst, v, args.n_samples, rngmc) for v in snap_vecs[X]]
                for _ in range(args.n_rep)] for X in "AB"}
    for d in args.target_list:
        tp = inst.tpc + d
        e, _ = inst.ed(tp)
        truth = order_of(e["c0"], args.tau)
        br = {lab: train_branch(inst, tp, lab, args.width, args.seed, args) for lab in "AB"}
        info = {lab: state_info(inst, br[lab], tp, e["E0"]) for lab in "AB"}
        win = min("AB", key=lambda L: info[L]["E"])
        dec = {"R0P": order_of(info[win]["c"], args.tau)}
        th = {}
        for nm, vv in (("EC", snap_vecs), ("ECor", oracle_vecs)):
            th[nm] = {X: (ec_block(inst, vv[X], tp) if vv[X] else np.inf) for X in "AB"}
            dec[nm] = decide_pair(th[nm]["A"], th[nm]["B"])
        for nm, vv in (("AT", snap_vecs), ("ATor", oracle_vecs)):
            th[nm] = {X: (at_block(inst, [inst.pvec(v) for v in vv[X]], tp) if vv[X] else np.inf)
                      for X in "AB"}
            dec[nm] = decide_pair(th[nm]["A"], th[nm]["B"])
        # sampled arms
        samp = {"R0P": [], "EC": [], "AT": []}
        relS = []
        cx = np.array([inst.coup(tp)[k] for k in PARTS])
        for r in range(args.n_rep):
            eA, sA = mc_energy(inst, br["A"], tp, args.n_samples, rngmc)
            eB, sB = mc_energy(inst, br["B"], tp, args.n_samples, rngmc)
            dd = decide_pair(eA, eB, sA, sB)
            samp["R0P"].append("U" if dd == "U" else order_of(info[dd]["c"], args.tau))
            thx, sex = {}, {}
            for X in "AB":
                if not snap_vecs[X]:
                    thx[X], sex[X] = np.inf, 0.0
                    continue
                vals = [(cx @ pv, np.sqrt(np.sum((cx * se) ** 2))) for pv, se in pv_s[X][r]]
                k = int(np.argmin([v[0] for v in vals]))
                thx[X], sex[X] = vals[k]
            samp["AT"].append(decide_pair(thx["A"], thx["B"], sex["A"], sex["B"]))
            thE = {}
            for X in "AB":
                if not snap_vecs[X]:
                    thE[X] = np.inf
                    continue
                mats, rel = mc_block_mats(inst, snap_vecs[X], args.n_samples, rngmc)
                relS += rel
                thE[X] = ec_block(inst, snap_vecs[X], tp, mats)
            samp["EC"].append((thE["A"], thE["B"]))
        # EC sampled: replica spread gives the standard error of each block root
        ecA = np.array([t[0] for t in samp["EC"]])
        ecB = np.array([t[1] for t in samp["EC"]])
        seA = float(np.std(ecA)) if np.all(np.isfinite(ecA)) else 0.0
        seB = float(np.std(ecB)) if np.all(np.isfinite(ecB)) else 0.0
        samp["EC"] = [decide_pair(a_, b_, seA, seB) for a_, b_ in zip(ecA, ecB)]
        row = dict(d=d, ed=e, truth=truth, branches=info, winner_seed=win, dec=dec,
                   theta={k: {X: float(v[X]) for X in "AB"} for k, v in th.items()},
                   sampled=samp, ec_offdiag_rel_se_per_sample=(float(np.median(relS)) if relS else None),
                   ec_root_se=dict(A=seA, B=seB))
        res["targets"].append(row)
        log(f"target d={d:+.4f} gap={e['gap']:.4f} truth={truth} | eps_A={info['A']['dE']:.2e} "
            f"eps_B={info['B']['dE']:.2e} | R0P={dec['R0P']} EC={dec['EC']} AT={dec['AT']} "
            f"ECor={dec['ECor']} ATor={dec['ATor']} | sampled R0P/EC/AT "
            + "/".join("".join(samp[k]) for k in ("R0P", "EC", "AT")))
        dump()
    res["wall_s"] = time.time() - t0
    dump()
    log(f"done ({res['wall_s']:.0f}s) -> {args.out}")


# ============================================================================ aggregate
def aggregate(d):
    files = sorted(glob.glob(os.path.join(d, "w*_s*.json")))
    by_w = {}
    for f in files:
        r = json.load(open(f))
        by_w.setdefault(r["args"]["width"], []).append(r)
    verdict = {"EC": [], "AT": []}
    for W, runs in sorted(by_w.items()):
        print(f"=== width {W}: seeds {[r['args']['seed'] for r in runs]}")
        if len(runs) < 3:
            print("  INCOMPLETE (< 3 seeds)")
            continue
        ds = [t["d"] for t in runs[0]["targets"]]
        hard = []
        for i, dd in enumerate(ds):
            eps = np.median([min(r["targets"][i]["branches"][X]["dE"] for X in "AB") for r in runs])
            gap = runs[0]["targets"][i]["ed"]["gap"]
            h = eps >= gap / 2
            hard.append(h)
            print(f"  target d={dd:+.4f} gap={gap:.4f} eps_med={eps:.2e} hard={h} truth={runs[0]['targets'][i]['truth']} "
                  + " ".join(f"{k}:{''.join(r['targets'][i]['dec'][k] for r in runs)}" for k in ("R0P", "EC", "AT", "ECor", "ATor")))
        ncert = int(np.median([sum(s["certified"] for s in r["snaps"]) for r in runs]))
        blocks = all(any(s["label"] == X for s in r["snaps"]) for r in runs for X in "AB")
        nh = int(sum(hard))
        print(f"  hard targets {nh}/{len(ds)}; certified snapshots (median) {ncert}/{len(runs[0]['snaps'])}; both blocks {blocks}")
        if nh < 3 or ncert < 4 or not blocks:
            print("  PREMISE-FAIL at this width")
            for X in verdict:
                verdict[X].append("PREMISE-FAIL")
            continue

        def rates(arm, sampled=False):
            cw = dec = n = 0
            for r in runs:
                for i, h in enumerate(hard):
                    if not h:
                        continue
                    t = r["targets"][i]
                    ds_ = t["sampled"][arm] if sampled else [t["dec"][arm]]
                    for x in ds_:
                        n += 1
                        dec += x != "U"
                        cw += x not in ("U", t["truth"])
            return cw / n, dec / n

        cw0, dc0 = rates("R0P")
        cw0s, _ = rates("R0P", True)
        print(f"  R0P CW={cw0:.3f} DEC={dc0:.3f} | sampled CW={cw0s:.3f}")
        for X in ("EC", "AT"):
            cw, dc = rates(X)
            cws, dcs = rates(X, True)
            flips = []
            for r in runs:
                for i, h in enumerate(hard):
                    if h:
                        t = r["targets"][i]
                        flips += [x != t["dec"][X] for x in t["sampled"][X]]
            fl = float(np.mean(flips))
            if cw0 == 0 or cw >= cw0 or cw > 0.30:
                v = "FAIL"
            elif cw <= 0.15 and dc >= 0.6 and cw0 - cw >= 0.15 and fl <= 0.10 and cws <= cw0s - 0.10:
                v = "PASS"
            else:
                v = "AMBIGUOUS"
            verdict[X].append(v)
            print(f"  {X}: CW={cw:.3f} DEC={dc:.3f} | sampled CW={cws:.3f} DEC={dcs:.3f} flip={fl:.3f} -> {v}")
    for X, vs in verdict.items():
        if vs and all(v == "PREMISE-FAIL" for v in vs):
            g = "PREMISE-FAIL (projected NQS resolves the instance; route untested)"
        elif "PASS" in vs and "FAIL" not in vs:
            g = "PASS"
        elif "FAIL" in vs:
            g = "FAIL"
        else:
            g = "AMBIGUOUS" if vs else "NO DATA"
        print(f"GLOBAL VERDICT {X}: {g}  (per width: {vs})")


# ============================================================================ cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--ty", type=float, default=1.11394)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--targets", default="-0.02,-0.005,-0.002,-0.0005,0.0005,0.002,0.005,0.02")
    ap.add_argument("--snaps", default="-0.08,-0.05,-0.03,0.03,0.05,0.08")
    ap.add_argument("--sr-iters", type=int, default=1000)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
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
        a.width = (8, 32)[a.task_id // 3]
        a.seed = (1234, 2345, 3456)[a.task_id % 3]
    sub = "results"
    if a.smoke:
        a.targets, a.snaps, a.sr_iters, a.width = "0.002", "-0.05,0.05", 3, 4
        a.n_samples, a.n_rep, a.sr_samples = 2000, 3, 200
        sub = "results_smoke"
    a.target_list = [float(s) for s in a.targets.split(",")]
    a.snap_list = [float(s) for s in a.snaps.split(",")]
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
