#!/usr/bin/env python3
"""
P3.2 round-2 decisive test: deciding a SAME-SECTOR competing order with trained NNBF branches
==============================================================================================

Question
--------
Near a same-sector avoided crossing of two competing orders, does a variational family whose
two order references SHARE one neural backflow network decide the ground-state order better
than current practice (two independently trained branches, the lower-energy one wins, relabelled
by its measured order character = "R0+")?  Arms, all trained by exact-sum SR (no Monte Carlo
noise; every expectation value is an exact sum over the full Hilbert space):

  R0p  control. Two independent NNBF (Luo & Clark) branches, branch A seeded by a Neel-UHF
       reference, branch B by a period-4 spin-stripe UHF reference, each trained --sr-iters SR
       steps. Decision = sign of the measured order character c of the lower-energy branch.
  SS   Route 2.1 (state-specific two-reference shared backflow, "2R-SS"). One wave function
       psi = c_A det[Phi_A + F_theta(n)] + c_B det[Phi_B + F_theta(n)] (per spin species),
       with ONE backflow network F_theta shared by both references; (c_A, c_B) are the exact
       lowest Ritz vector in span{a_A, a_B}; the backflow and both reference orbital sets are
       trained by SR on the lowest Ritz root theta_0. Decision = sign of c(psi).
  SA   Route 2.2 (state-averaged two-reference shared backflow, "2R-SA"). Same family, trained
       on the Gross-Oliveira-Kohn ensemble energy tr(S^{-1} H) = theta_0 + theta_1 over the
       2-dim span (equal weights), as in state-averaged QMC (shared Jastrow/orbitals, only CI
       coefficients state-specific). Decision = sign of c of the lowest Ritz root, after
       property-based (order-operator) diabatisation; the diabatic gap eps_B - eps_A and the
       coupling W are recorded.

Instance (the Hard Core): 4x3 PBC Hubbard, N_up = N_dn = 4 (doping 1/3), U = 8, t' swept across
the same-sector avoided crossing found by sector-projected ED (round-2 pre-check): in the sector
K = 0, M_x = M_y = +1, S = 0, the ground state switches from a d-wave-paired state with short-range
AF correlations peaked at q_A = (pi, 2pi/3) (t' >= -0.30, P_d ~ 0.02) to a spin-stripe state peaked
at q_B = (pi/2, 0) (t' <= -0.35, P_d ~ 0.01); the minimum in-sector gap is ~0.28 t near t' = -0.33.
Both competitors share ALL quantum numbers, so H mixes them (avoided crossing, mixed eigenstates).

Order character (diagonal operator, exact):
  c(psi) = [S_s(q_B) + S_s(-q_B)]/2 - [S_s(q_A) + S_s(q_A')]/2,  S_s(q) = <|sum_i e^{i q r_i} S^z_i|^2>/V,
  with q_A, q_A' = (pi, +-2pi/3) on 4x3 (on other clusters: the allowed q closest to (pi, pi) and its
  partner). c > 0 means stripe (B), c < 0 means d-wave/AF (A).
Truth: ED ground state of the sector (projected Lanczos with an S^2 penalty), checked against the
global fixed-(N_up, N_dn) ground state. Truth label = sign c(E0); "mixed" if |c(E0)| < --tau.

Metrics (per arm R, over all (t', width, seed) instances with a non-mixed truth):
  CW(R)  = fraction of instances where R commits (|c| >= tau) to the wrong sign,
  DEC(R) = fraction of instances where R commits,
  MAE(R) = mean |c_R - c(E0)| over ALL instances (continuous error of the deciding observable).
Pre-committed verdict for route X in {SS, SA} (printed by --aggregate):
  PASS      if CW(X) <= 0.10 and DEC(X) >= 0.6 and
            [CW(R0p) - CW(X) >= 0.10  or  MAE(X) <= 0.7 MAE(R0p)],
  FAIL      if CW(X) > 0.25, or [CW(X) >= CW(R0p) and MAE(X) >= 0.9 MAE(R0p)],
  AMBIGUOUS otherwise. Pre-committed action for AMBIGUOUS: rerun the 3 grid points nearest the
            ED gap minimum with --seeds 4 --widths 8,32,64; still AMBIGUOUS -> route KILLED.
  INVALID   (whole test) if the ED sweep shows no sign change of c(E0) across the grid, or the
            sector ground state is not the global ground state at some grid point.

Usage
-----
  python3 micro_test.py [options]

Options
-------
  --smoke {0,1}        1 = login-node smoke test (default 0): 4x2 PBC, N_up=N_dn=3, U=8,
                       t' in {-0.25}, widths 4, 1 seed, 15 SR steps; runs the correctness checks
                       (sector projector idempotent and commuting with H, sector ED equals the
                       lowest full-ED state with matching quantum numbers, finite-difference check
                       of the SS and SA gradients), then all three arms, and prints the per-instance
                       line and the aggregate. Overrides the lattice/size options below.
  --time-real {0,1}    1 = additionally time ONE SR iteration of each arm on the real 4x3, N=4+4
                       instance (D = 245025) for the cost estimate (default 0; ~1 min).
  --Lx INT, --Ly INT   Cluster (default 4, 3). Periodic in both directions (the sector projector
                       assumes PBC, translations and the two mirrors).
  --nup INT            N_up = N_dn (default 4).
  --U FLOAT            On-site repulsion (default 8.0).
  --t FLOAT            Nearest-neighbour hopping (default 1.0).
  --tp-list LIST       Comma-separated t' values (default "-0.33"); each value is one grid point,
                       one JSON. The Slurm array passes one value per task.
  --widths LIST        Comma-separated NNBF hidden widths (default "8,32").
  --seeds INT          Number of random seeds per width (default 2); seed s uses --seed + s for the
                       network initialisation, the UHF noise and the SR metric samples.
  --seed INT           Base seed (default 1234).
  --sr-iters INT       SR steps of the SS and SA two-reference objects (default 500; each step
                       evaluates two determinants and a 2x2 Ritz problem).
  --r0p-factor FLOAT   Each R0p branch gets round(r0p_factor * sr_iters) steps (default 1.5).
                       Measured on the real instance, one SS/SA step costs about 1.5x a pair of R0p
                       branch steps, so 1.5 matches wall-clock budgets. Wall times of every arm are
                       stored; --aggregate prints the ratio and downgrades a PASS to AMBIGUOUS if
                       mean wall(SS or SA) exceeds 1.2 x mean wall(R0p) (budget not matched).
  --sr-lr FLOAT        SR step size (default 0.05).
  --sr-shift FLOAT     SR diagonal shift (default 1e-3).
  --sr-samples INT     Configurations drawn exactly from |psi|^2 for the SR metric (default 1000).
  --chunk INT          Configurations per vectorised chunk (default 65536).
  --tau FLOAT          Abstention / mixed-truth threshold on |c| (default 0.01).
  --arms LIST          Comma-separated subset of R0p,SS,SA (default all three).
  --out-dir PATH       Directory for per-grid-point JSONs (default results/ next to this script).
                       Existing files are skipped unless --overwrite 1.
  --overwrite {0,1}    Default 0.
  --aggregate DIR      Do no computation: read all *.json in DIR, print the per-instance table,
                       CW/DEC/MAE per arm, the instance-validity check and the verdicts.

Outputs
-------
  One JSON per t' with keys "params", "ed" {E0, E1, E0_global, c0, c1, Pd0, Pd1, gap, valid_sector},
  "instances" [ {width, seed, arm: {E, c, Pd, fid (|<psi|E0>|^2), wall, decision, ...}} ].
"""
import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p11_core as core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- symmetry
def site_perm_species(states, lookup, V, perm):
    D = len(states)
    idx = np.empty(D, np.int64)
    sg = np.empty(D)
    for k, b in enumerate(states):
        occ = [i for i in range(V) if (b >> i) & 1]
        img = [perm[i] for i in occ]
        inv = sum(1 for a in range(len(img)) for c in range(a + 1, len(img)) if img[a] > img[c])
        nb = 0
        for i in img:
            nb |= 1 << i
        idx[k] = lookup[nb]
        sg[k] = (-1) ** inv
    return idx, sg


def apply_perm(model, e, psi):
    Psi = psi.reshape(model.Du, model.Dd)
    out = np.zeros_like(Psi)
    out[np.ix_(e[0], e[0] if model.nup == model.ndn else e[2])] = Psi * e[1][:, None] * (
        e[1] if model.nup == model.ndn else e[3])[None, :]
    return out.ravel()


def group(model):
    Lx, Ly, V = model.Lx, model.Ly, model.V
    su, lu, _, _ = core.species_basis(V, model.nup)
    els = []
    for a in range(Lx):
        for b in range(Ly):
            for sx in (1, -1):
                for sy in (1, -1):
                    p = [((sx * x + a) % Lx) + Lx * ((sy * y + b) % Ly)
                         for y in range(Ly) for x in range(Lx)]
                    els.append(site_perm_species(su, lu, V, p))
    return els


class Spin:
    def __init__(self, model):
        V = model.V
        self.m = model
        self.ops = [(core.species_annihilate(V, model.nup + 1, i).T.tocsr(),
                     core.species_annihilate(V, model.ndn, i).tocsr()) for i in range(V)]

    def splus(self, x):
        X = x.reshape(self.m.Du, self.m.Dd)
        Y = None
        for cu, ad in self.ops:
            t = cu @ (ad @ X.T).T
            Y = t if Y is None else Y + t
        return np.asarray(Y)

    def s2(self, x):
        Y = self.splus(x)
        Z = None
        for cu, ad in self.ops:
            t = cu.T @ (ad.T @ Y.T).T
            Z = t if Z is None else Z + t
        return np.asarray(Z).ravel()

    def s2_exp(self, x):
        return float(np.sum(self.splus(x) ** 2) / (x @ x))


def sector_ed(model, k=2, lam=1.0, shift=50.0, seed=1):
    """Lowest k levels of the K=0, M_x=M_y=+1, S=0 sector."""
    els = group(model)
    spin = Spin(model)

    def proj(x):
        acc = np.zeros_like(x)
        for e in els:
            acc += apply_perm(model, e, x)
        return acc / len(els)

    def mv(x):
        x = np.asarray(x).ravel()
        px = proj(x)
        return proj(model.H @ px + lam * spin.s2(px)) + shift * (x - px)

    A = LinearOperator((model.D, model.D), matvec=mv, dtype=float)
    v0 = proj(np.random.default_rng(seed).standard_normal(model.D))
    w, v = eigsh(A, k=k, which="SA", tol=1e-10, v0=v0)
    out = []
    for j in np.argsort(w):
        psi = proj(v[:, j])
        psi /= np.linalg.norm(psi)
        out.append((float(psi @ (model.H @ psi)), psi, spin.s2_exp(psi)))
    return out, proj, spin


# ----------------------------------------------------------------------------- order character
class Character:
    """c(n) per configuration such that <c> = sum p(n) c(n) (diagonal operator)."""

    def __init__(self, model):
        Lx, Ly = model.Lx, model.Ly
        kyA = min(range(Ly), key=lambda k: abs(2 * np.pi * k / Ly - np.pi))
        qs_B = [(2 * np.pi * 1 / Lx, 0.0), (-2 * np.pi * 1 / Lx, 0.0)]
        qs_A = [(np.pi, 2 * np.pi * kyA / Ly), (np.pi, -2 * np.pi * kyA / Ly)]
        iu, idd = model.iu_id()
        sz = (model.occ_u[iu].astype(float) - model.occ_d[idd].astype(float)) / 2
        def sf(q):
            ph = np.exp(1j * model.coords @ np.array(q))
            return np.abs(sz @ ph) ** 2 / model.V
        self.cn = (sf(qs_B[0]) + sf(qs_B[1])) / 2 - (sf(qs_A[0]) + sf(qs_A[1])) / 2
        self.qA, self.qB = qs_A[0], qs_B[0]

    def __call__(self, psi):
        return float(np.sum(psi ** 2 * self.cn) / (psi @ psi))


# ----------------------------------------------------------------------------- references
def uhf_pattern(model, pattern, seed, iters=3000, mix=0.3):
    """UHF Slater orbitals from an initial spin pattern: 'neel' (staggered) or 'stripe'
    (period-4 along x, (+,-,-,+), uniform along y)."""
    rng = np.random.default_rng(seed)
    V, T, U = model.V, model.T, model.U
    if pattern == "neel":
        m = np.array([(-1) ** int(x + y) for x, y in model.coords], float)
    else:
        m = np.array([[1, -1, -1, 1][int(x) % 4] for x, y in model.coords], float)
    nu = (model.nup / V) * (1 + 0.4 * m) + 0.01 * rng.standard_normal(V)
    nd = (model.ndn / V) * (1 - 0.4 * m) + 0.01 * rng.standard_normal(V)
    for _ in range(iters):
        _, vu = np.linalg.eigh(-T + np.diag(U * nd))
        _, vd = np.linalg.eigh(-T + np.diag(U * nu))
        Pu, Pd = vu[:, :model.nup], vd[:, :model.ndn]
        nu_new, nd_new = np.sum(Pu ** 2, 1), np.sum(Pd ** 2, 1)
        if max(np.abs(nu_new - nu).max(), np.abs(nd_new - nd).max()) < 1e-10:
            break
        nu = (1 - mix) * nu + mix * nu_new
        nd = (1 - mix) * nd + mix * nd_new
    return Pu.copy(), Pd.copy()


# ----------------------------------------------------------------------------- two-reference ansatz
SHARED = ("W1", "b1", "W2", "b2")


class TwoRef:
    """Two Slater references A, B with ONE shared NNBF backflow F_theta(n)."""

    def __init__(self, model, refA, refB, width, seed):
        self.a = core.Ansatz(model, refA[0], refA[1], "nnbf", width, False, seed)
        self.b = core.Ansatz(model, refB[0], refB[1], "nnbf", width, False, seed)
        for k in SHARED:
            self.b.p[k] = self.a.p[k]
        self.m = model

    def n_params(self):
        return int(self.a.n_params() + self.b.p["Pu"].size + self.b.p["Pd"].size)

    def flat(self):
        return np.concatenate([self.a.p["Pu"].ravel(), self.a.p["Pd"].ravel(),
                               self.b.p["Pu"].ravel(), self.b.p["Pd"].ravel()]
                              + [self.a.p[k].ravel() for k in SHARED])

    def set_flat(self, x):
        o = 0
        for obj, k in ((self.a, "Pu"), (self.a, "Pd"), (self.b, "Pu"), (self.b, "Pd")):
            n = obj.p[k].size
            obj.p[k] = x[o:o + n].reshape(obj.p[k].shape)
            o += n
        for k in SHARED:
            n = self.a.p[k].size
            arr = x[o:o + n].reshape(self.a.p[k].shape)
            self.a.p[k] = arr
            self.b.p[k] = arr
            o += n

    def vectors(self, chunk):
        return self.a.amplitudes(chunk), self.b.amplitudes(chunk)

    def _split(self, g, which):
        """Map a single-Ansatz flat vector/matrix (keys Pu,Pd,W1,b1,W2,b2) into TwoRef layout."""
        nP = self.a.p["Pu"].size + self.a.p["Pd"].size
        own, sh = g[..., :nP], g[..., nP:]
        z = np.zeros_like(own)
        return np.concatenate([own, z, sh] if which == "a" else [z, own, sh], axis=-1)

    def grad(self, wA, wB, chunk):
        return (self._split(self.a.grad_weighted(wA, chunk), "a")
                + self._split(self.b.grad_weighted(wB, chunk), "b"))

    def O_samples(self, idx, cA, cB, aA, aB):
        """Per-sample d log psi / d theta for psi = cA aA + cB aB at configurations idx."""
        OA = self._split(self.a.per_sample_O_fast(idx), "a")
        OB = self._split(self.b.per_sample_O_fast(idx), "b")
        psi = cA * aA[idx] + cB * aB[idx]
        return ((cA * aA[idx])[:, None] * OA + (cB * aB[idx])[:, None] * OB) / psi[:, None]


def ritz(H, aA, aB):
    HA, HB = H @ aA, H @ aB
    S = np.array([[aA @ aA, aA @ aB], [aB @ aA, aB @ aB]])
    Hm = np.array([[aA @ HA, aA @ HB], [aB @ HA, aB @ HB]])
    sc = np.sqrt(np.diag(S))
    Sn, Hn = S / np.outer(sc, sc), Hm / np.outer(sc, sc)
    Sn = Sn + 1e-12 * np.eye(2)
    th, C = sla.eigh(Hn, Sn)
    C = C / sc[:, None]
    return th, C, (HA, HB)


def ss_sa_objective(two, H, mode, chunk):
    aA, aB = two.vectors(chunk)
    th, C, (HA, HB) = ritz(H, aA, aB)
    roots = [0] if mode == "SS" else [0, 1]
    wA = np.zeros_like(aA)
    wB = np.zeros_like(aB)
    for k in roots:
        cA, cB = C[0, k], C[1, k]
        psi = cA * aA + cB * aB
        r = cA * HA + cB * HB - th[k] * psi  # (H - theta) psi, with ||psi|| = 1
        wA += cA * r * aA
        wB += cB * r * aB
    f = two.grad(wA, wB, chunk)  # = (1/2) d(objective)/d theta
    return float(sum(th[k] for k in roots)), f, th, C, aA, aB


def train_two(two, H, mode, args, rng, label):
    best = (np.inf, two.flat().copy())
    for it in range(args.sr_iters):
        obj, f, th, C, aA, aB = ss_sa_objective(two, H, mode, args.chunk)
        if not np.isfinite(obj):
            break
        if obj < best[0]:
            best = (obj, two.flat().copy())
        roots = [0] if mode == "SS" else [0, 1]
        blocks = []
        ns = args.sr_samples // len(roots)
        for k in roots:
            psi = C[0, k] * aA + C[1, k] * aB
            p = psi ** 2 / (psi @ psi)
            idx = rng.choice(len(psi), size=ns, p=p)
            O = two.O_samples(idx, C[0, k], C[1, k], aA, aB)
            blocks.append(O - O.mean(0))
        Oc = np.concatenate(blocks, 0) / np.sqrt(sum(len(b) for b in blocks))
        lam = args.sr_shift
        K = Oc @ Oc.T + lam * np.eye(Oc.shape[0])
        dx = (f - Oc.T @ np.linalg.solve(K, Oc @ f)) / lam
        step = args.sr_lr * dx
        nrm = np.linalg.norm(step)
        if nrm > 1.0:
            step /= nrm
        two.set_flat(two.flat() - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            core.log(f"{label}: it {it:5d} obj = {obj:.8f} theta = {th[0]:.6f},{th[1]:.6f}")
    obj, *_ = ss_sa_objective(two, H, mode, args.chunk)
    if obj > best[0]:
        two.set_flat(best[1])
    return two


# ----------------------------------------------------------------------------- arms
def measure(model, psi, E0vec, char):
    psi = psi / np.linalg.norm(psi)
    return dict(E=float(psi @ (model.H @ psi)), c=char(psi), Pd=model.dwave(psi),
                fid=float((psi @ E0vec) ** 2))


def decide(c, tau):
    return "U" if abs(c) < tau else ("B" if c > 0 else "A")


def run_R0p(model, refs, width, seed, args, E0vec, char):
    t0 = time.time()
    out = {}
    args = SimpleNamespace(**vars(args))
    args.sr_iters = max(1, int(round(args.r0p_factor * args.sr_iters)))
    for lab in ("A", "B"):
        rng = np.random.default_rng(seed + (0 if lab == "A" else 7))
        ans = core.Ansatz(model, refs[lab][0], refs[lab][1], "nnbf", width, False, seed)
        core.optimize_sr(ans, model.H, args, rng, f"R0p-{lab} H={width} s={seed}")
        out[lab] = measure(model, ans.amplitudes(args.chunk), E0vec, char)
        out["n_params"] = ans.n_params()
    win = "A" if out["A"]["E"] <= out["B"]["E"] else "B"
    res = dict(out[win])
    res.update(branchA=out["A"], branchB=out["B"], winner_seed=win, n_params=out["n_params"],
               wall=time.time() - t0, decision=decide(res["c"], args.tau))
    return res


def run_two(model, refs, width, seed, args, E0vec, char, mode):
    t0 = time.time()
    rng = np.random.default_rng(seed + 13)
    two = TwoRef(model, refs["A"], refs["B"], width, seed)
    train_two(two, model.H, mode, args, rng, f"{mode} H={width} s={seed}")
    aA, aB = two.vectors(args.chunk)
    th, C, _ = ritz(model.H, aA, aB)
    psi0 = C[0, 0] * aA + C[1, 0] * aB
    res = measure(model, psi0, E0vec, char)
    psi1 = C[0, 1] * aA + C[1, 1] * aB
    res["root1"] = measure(model, psi1, E0vec, char)
    # property-based diabatisation: eigenvectors of the character matrix in the span
    Cm = np.array([[np.sum(x * y * char.cn) for y in (psi0, psi1)] for x in (psi0, psi1)])
    Hm = np.diag(th)
    _, U = np.linalg.eigh(Cm)  # columns: low-c (A-like), high-c (B-like) diabats
    Hd = U.T @ Hm @ U
    res.update(theta=th.tolist(), eps_A=float(Hd[0, 0]), eps_B=float(Hd[1, 1]), W=float(Hd[0, 1]),
               weights=[float(C[0, 0] ** 2 * (aA @ aA)), float(C[1, 0] ** 2 * (aB @ aB))],
               n_params=two.n_params(), wall=time.time() - t0, decision=decide(res["c"], args.tau))
    return res


# ----------------------------------------------------------------------------- checks
def checks(model, proj, spin, levels, args):
    ok = True
    rng = np.random.default_rng(0)
    x = rng.standard_normal(model.D)
    px = proj(x)
    e1 = np.linalg.norm(proj(px) - px) / np.linalg.norm(px)
    e2 = np.linalg.norm(proj(model.H @ px) - model.H @ px) / np.linalg.norm(model.H @ px)
    core.log(f"check projector idempotent {e1:.1e}  [P,H] {e2:.1e}")
    ok &= e1 < 1e-10 and e2 < 1e-10
    w, v = np.linalg.eigh(model.H.toarray()) if model.D <= 4000 else (None, None)
    if w is not None:
        match = None
        for j in range(min(60, len(w))):
            pv = proj(v[:, j])
            if np.linalg.norm(pv) > 0.999 and spin.s2_exp(v[:, j]) < 1e-6:
                match = w[j]
                break
        core.log(f"check sector E0 {levels[0][0]:.10f} vs dense lowest sector-singlet {match:.10f}")
        ok &= match is not None and abs(match - levels[0][0]) < 1e-8
    # gradient checks for SS and SA objectives
    PuA, PdA = uhf_pattern(model, "neel", 1)
    PuB, PdB = uhf_pattern(model, "stripe", 2)
    for mode in ("SS", "SA"):
        two = TwoRef(model, (PuA, PdA), (PuB, PdB), 4, 3)
        x0 = two.flat().copy()
        x0 += 0.05 * rng.standard_normal(x0.size)
        two.set_flat(x0.copy())
        _, f, *_ = ss_sa_objective(two, model.H, mode, args.chunk)
        d = rng.standard_normal(x0.size)
        h = 1e-6
        two.set_flat(x0 + h * d)
        op = ss_sa_objective(two, model.H, mode, args.chunk)[0]
        two.set_flat(x0 - h * d)
        om = ss_sa_objective(two, model.H, mode, args.chunk)[0]
        fd, an = (op - om) / (2 * h), 2 * f @ d
        core.log(f"check {mode} gradient analytic {an:.8e} finite-diff {fd:.8e}")
        ok &= abs(fd - an) < 1e-5 * max(1.0, abs(fd))
    return ok


# ----------------------------------------------------------------------------- driver
def run_point(tp, args):
    model = core.Hubbard(args.Lx, args.Ly, "pbc", "pbc", args.nup, args.nup, args.t, tp, args.U)
    model.pair_setup(1.5)
    char = Character(model)
    t0 = time.time()
    levels, proj, spin = sector_ed(model, k=2)
    k_glob = 1 if model.D > 2 else 1
    wg, _ = (eigsh(model.H, k=k_glob, which="SA", tol=1e-10,
                   v0=np.random.default_rng(5).standard_normal(model.D)) if model.D > 400
             else (np.linalg.eigvalsh(model.H.toarray())[:1], None))
    (E0, v0, s20), (E1, v1, s21) = levels
    ed = dict(E0=E0, E1=E1, E0_global=float(np.min(wg)), gap=E1 - E0, c0=char(v0), c1=char(v1),
              Pd0=model.dwave(v0), Pd1=model.dwave(v1), S2_0=s20, S2_1=s21,
              valid_sector=bool(abs(E0 - float(np.min(wg))) < 1e-7), wall=time.time() - t0)
    core.log(f"t'={tp:+.3f} ED sector: E0={E0:.6f} E1={E1:.6f} gap={E1 - E0:.4f} "
             f"c0={ed['c0']:+.4f} c1={ed['c1']:+.4f} Pd0={ed['Pd0']:+.2e} Pd1={ed['Pd1']:+.2e} "
             f"global E0={ed['E0_global']:.6f} valid={ed['valid_sector']}")
    return model, char, levels, proj, spin, ed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P3.2 round-2 decisive test (see module docstring)")
    p.add_argument("--smoke", type=int, default=0, choices=[0, 1])
    p.add_argument("--time-real", type=int, default=0, choices=[0, 1])
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--nup", type=int, default=4)
    p.add_argument("--U", type=float, default=8.0)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--tp-list", type=str, default="-0.33")
    p.add_argument("--widths", type=str, default="8,32")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sr-iters", type=int, default=500)
    p.add_argument("--r0p-factor", type=float, default=1.5)
    p.add_argument("--sr-lr", type=float, default=0.05)
    p.add_argument("--sr-shift", type=float, default=1e-3)
    p.add_argument("--sr-samples", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--arms", type=str, default="R0p,SS,SA")
    p.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    p.add_argument("--aggregate", type=str, default="")
    a = p.parse_args(argv)
    if a.smoke:
        a.Lx, a.Ly, a.nup, a.U, a.tp_list = 4, 2, 3, 8.0, "-0.25"
        a.widths, a.seeds, a.sr_samples = "4", 1, 200
        if a.sr_iters == 500:  # only the default is replaced; an explicit value is kept
            a.sr_iters = 15
        a.out_dir = os.path.join(HERE, "results_smoke")
        a.overwrite = 1
    a.tps = [float(s) for s in a.tp_list.split(",") if s.strip()]
    a.width_list = [int(s) for s in a.widths.split(",") if s.strip()]
    a.arm_list = [s.strip() for s in a.arms.split(",") if s.strip()]
    for arm in a.arm_list:
        if arm not in ("R0p", "SS", "SA"):
            p.error(f"unknown arm {arm}")
    return a


def verdicts(rows):
    arms = sorted({k for r in rows for k in r["arms"]})
    stats = {}
    for arm in arms:
        rr = [r for r in rows if arm in r["arms"]]
        dec = [r for r in rr if r["truth"] != "U"]
        n = max(len(dec), 1)
        cw = sum(1 for r in dec if r["arms"][arm]["decision"] not in ("U", r["truth"])) / n
        dr = sum(1 for r in dec if r["arms"][arm]["decision"] != "U") / n
        mae = float(np.mean([abs(r["arms"][arm]["c"] - r["c0"]) for r in rr])) if rr else np.nan
        wall = float(np.mean([r["arms"][arm]["wall"] for r in rr])) if rr else np.nan
        stats[arm] = dict(CW=cw, DEC=dr, MAE=mae, n=len(rr), n_dec=len(dec), wall=wall)
    out = {}
    if "R0p" in stats:
        c = stats["R0p"]
        for x in ("SS", "SA"):
            if x not in stats:
                continue
            s = stats[x]
            if s["CW"] <= 0.10 and s["DEC"] >= 0.6 and (c["CW"] - s["CW"] >= 0.10
                                                         or s["MAE"] <= 0.7 * c["MAE"]):
                out[x] = "PASS" if s["wall"] <= 1.2 * c["wall"] else "AMBIGUOUS (budget)"
            elif s["CW"] > 0.25 or (s["CW"] >= c["CW"] and s["MAE"] >= 0.9 * c["MAE"]):
                out[x] = "FAIL"
            else:
                out[x] = "AMBIGUOUS"
    return stats, out


def aggregate(d, tau):
    rows, eds = [], []
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        J = json.load(open(fn))
        ed = J["ed"]
        eds.append((J["params"]["tp"], ed))
        truth = decide(ed["c0"], tau)
        for inst in J["instances"]:
            rows.append(dict(tp=J["params"]["tp"], width=inst["width"], seed=inst["seed"],
                             truth=truth, c0=ed["c0"], arms=inst["arms"]))
    eds.sort(key=lambda x: x[0])
    print("# ED sweep: t'  E0  gap  c(E0)  c(E1)  Pd(E0)  Pd(E1)  sector=global")
    for tp, ed in eds:
        print(f"  {tp:+.3f} {ed['E0']:.6f} {ed['gap']:.4f} {ed['c0']:+.4f} {ed['c1']:+.4f} "
              f"{ed['Pd0']:+.2e} {ed['Pd1']:+.2e} {ed['valid_sector']}")
    signs = {np.sign(ed["c0"]) for _, ed in eds if abs(ed["c0"]) >= tau}
    valid = len(signs) == 2 and all(ed["valid_sector"] for _, ed in eds)
    print("# instance: t' width seed truth | arm: decision c E-E0 fid")
    for r in rows:
        e0 = dict(eds)[r["tp"]]["E0"]
        print(f"  {r['tp']:+.3f} H={r['width']:3d} s={r['seed']} truth={r['truth']} | " + " | ".join(
            f"{a}: {v['decision']} c={v['c']:+.4f} dE={v['E'] - e0:.4f} fid={v['fid']:.3f}"
            f" wall={v['wall']:.0f}s" for a, v in sorted(r["arms"].items())))
    stats, verd = verdicts(rows)
    for a, s in stats.items():
        print(f"# {a:4s} CW={s['CW']:.3f} DEC={s['DEC']:.3f} MAE_c={s['MAE']:.4f} "
              f"mean_wall={s['wall']:.1f}s n={s['n']} n_nonmixed={s['n_dec']}")
    print(f"# instance validity (c(E0) changes sign across grid, sector = global): "
          f"{'VALID' if valid else 'INVALID'}")
    for x, v in verd.items():
        print(f"# VERDICT {x}: {v if valid else 'INVALID-INSTANCE'}")
    return stats, verd, valid


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate, args.tau)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:  # the smoke aggregate must only see this run's JSON
        for old in glob.glob(os.path.join(args.out_dir, "*.json")):
            os.remove(old)
    t_start = time.time()
    for tp in args.tps:
        tag = (f"L{args.Lx}x{args.Ly}_n{args.nup}_U{args.U:g}_tp{tp:+.3f}_w{args.widths.replace(',', '-')}"
               f"_s{args.seeds}_it{args.sr_iters}.json")
        fn = os.path.join(args.out_dir, tag)
        if os.path.exists(fn) and not args.overwrite:
            core.log(f"{fn} exists, skipping")
            continue
        model, char, levels, proj, spin, ed = run_point(tp, args)
        if args.smoke:
            if not checks(model, proj, spin, levels, args):
                core.log("CHECKS FAILED")
                sys.exit(2)
            core.log("ALL CHECKS PASSED")
        E0vec = levels[0][1]
        refs = {"A": uhf_pattern(model, "neel", args.seed), "B": uhf_pattern(model, "stripe", args.seed)}
        for lab in ("A", "B"):
            sl = core.Ansatz(model, refs[lab][0], refs[lab][1], "slater")
            ms = measure(model, sl.amplitudes(args.chunk), E0vec, char)
            ed[f"ref{lab}"] = ms
            core.log(f"reference {lab}: E={ms['E']:.5f} c={ms['c']:+.4f} Pd={ms['Pd']:+.2e}")
        J = dict(params=dict(vars(args), tp=tp), ed=ed, instances=[])
        for width in args.width_list:
            for s in range(args.seeds):
                seed = args.seed + s
                np.random.seed(seed)
                sargs = SimpleNamespace(**vars(args))
                inst = dict(width=width, seed=seed, arms={})
                for arm in args.arm_list:
                    if arm == "R0p":
                        inst["arms"][arm] = run_R0p(model, refs, width, seed, sargs, E0vec, char)
                    else:
                        inst["arms"][arm] = run_two(model, refs, width, seed, sargs, E0vec, char, arm)
                    v = inst["arms"][arm]
                    core.log(f"t'={tp:+.3f} H={width} s={seed} {arm}: E-E0={v['E'] - ed['E0']:.5f} "
                             f"c={v['c']:+.4f} (truth c0={ed['c0']:+.4f}) fid={v['fid']:.4f} "
                             f"decision={v['decision']} wall={v['wall']:.1f}s")
                J["instances"].append(inst)
                with open(fn, "w") as fh:
                    json.dump(J, fh, indent=1, default=float)
    if args.time_real:
        time_real(args)
    if args.smoke:
        aggregate(args.out_dir, args.tau)
    core.log(f"total wall {time.time() - t_start:.1f}s")


def time_real(args):
    """One SR iteration of each arm on the real 4x3, N=4+4, U=8, t'=-0.33 instance."""
    model = core.Hubbard(4, 3, "pbc", "pbc", 4, 4, 1.0, -0.33, 8.0)
    refs = {"A": uhf_pattern(model, "neel", 1), "B": uhf_pattern(model, "stripe", 1)}
    sargs = SimpleNamespace(**vars(args))
    sargs.sr_iters, sargs.sr_samples = 1, 1000
    rng = np.random.default_rng(0)
    for w in (8, 32):
        t0 = time.time()
        ans = core.Ansatz(model, refs["A"][0], refs["A"][1], "nnbf", w, False, 1)
        core.optimize_sr(ans, model.H, sargs, rng, f"time R0p H={w}")
        t1 = time.time()
        for mode in ("SS", "SA"):
            two = TwoRef(model, refs["A"], refs["B"], w, 1)
            train_two(two, model.H, mode, sargs, rng, f"time {mode} H={w}")
        t2 = time.time()
        core.log(f"TIMING real D={model.D} H={w}: one R0p branch step {t1 - t0:.1f}s; "
                 f"SS+SA step {t2 - t1:.1f}s")


if __name__ == "__main__":
    main()
