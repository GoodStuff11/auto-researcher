#!/usr/bin/env python3
"""
P1.2 round-2 decisive test: factorized-ratio dressing (FRD) versus neural-network backflow
==========================================================================================

Question
--------
P1.1 (Slurm job 329057) showed that on doped 4x3 Hubbard clusters the SIGN carries 61-77% of
the energy error of neural-network backflow (NNBF) and that widening NNBF barely improves it.
This test asks whether a different nodal mechanism, learned variationally by SR from the
energy (never from oracle labels), moves the sign error where NNBF does not, at a MATCHED
parameter count and the same start, optimizer and step budget.

Ansatz under test (FRD)
-----------------------
    psi(n) = Phi(n) * prod_e [ 1 + alpha_e(n) * g(R_e(n)) ],
    R_e(n) = c_e(n) Phi(m_e(n)) / Phi(n),      g(R) = R / sqrt(1 + R^2)   (|g| < 1),
where
  * Phi = Slater determinant (trainable orbitals, initialised at UHF) times the two-body Jastrow
    of P1.1 (the "sj" foil of P1.1),
  * e runs over a fixed set of "edges" = elementary off-diagonal moves of a Hamiltonian;
    m_e(n) is the configuration reached from n by move e (if the move is impossible,
    the factor is 1), c_e(n) is the signed matrix element (Jordan-Wigner signs included),
  * alpha_e(n) = [tanh(x W1 + b1) W2 + b2]_e is ONE MLP with the same hidden width H and the
    same input x (spin-resolved occupations, +-1) as NNBF; only its read-out differs:
    NNBF writes orbital shifts (V*(Nup+Ndn) outputs), FRD writes one coefficient per edge.
Two edge frames (two routes):
  * "frdK" (Route 2.2, kinetic frame): e = (bond, spin) for every nearest- and next-nearest-
    neighbour bond; the move is the single-electron hop of the Hubbard kinetic term,
    c_e = <n| -t_ij c^dag c |m>.
  * "frdX" (Route 2.5, Schrieffer-Wolff frame): e = (bond, orientation) with the move the
    spin exchange S^+_i S^-_j (orientation 0) or S^-_i S^+_j (orientation 1), i.e. the
    leading off-diagonal term of the large-U effective Hamiltonian, which connects
    configurations with the same doublon number; c_e = <n| S^+_i S^-_j |m> = -s_up s_dn.
A factor turns negative (flips the sign of psi(n)) only when alpha_e g_e < -1, i.e. when the
Hamiltonian neighbour m_e(n) "votes" against the sign that Phi assigns to n. Every factor is
local (one edge), so the product is size-consistent, unlike a linear Lanczos-type dressing.
Edge count is 2 * (#bonds) in both frames, so frdK and frdX have identical parameter counts:
P = P_sj + 2V*H + H + H*E + E (E = #edges). On 4x3 PBC with t' != 0: E = 96, and for H = 8, 32
P = 1165, 4069, versus NNBF+Jastrow with the same H: P = 1165, 4069 (matched).

Control arm
-----------
"nnbfj_H": NNBF (Luo & Clark) times the same Jastrow, same UHF start, same SR settings, same
H. Code reused from P1.1 (p11_core.py is a verbatim copy of
tests/P1.1_sign_vs_amplitude/micro_test.py).

Metrics (exact sums over the full Hilbert space, P1.1 budget() function)
-------------------------------------------------------------------------
  dE_tot  = E_theta - E_0,
  dE_sign = E_sign_up - E_0 with E_sign_up the best feasible value of
            min_{a >= 0} a^T S H S a / a^T a  (S = diag sgn psi_theta) from four L-BFGS-B starts
            and the ten Haaf fixed-node state (rigorous upper bound on the sign-only error),
  w_sign  = sum over wrong-sign configurations of |psi_0|^2,
  eC_tot  = max long-range |P_theta(b0,b) - P_0(b0,b)| of the singlet bond-pair correlation.
For route X in {frdK, frdX} and width H in --widths:
  rho_sign(X,H) = dE_sign(X_H) / dE_sign(nnbfj_H),  rho_tot(X,H) = dE_tot(X_H) / dE_tot(nnbfj_H),
  rho_C(X,H)   = eC_tot(X_H) / eC_tot(nnbfj_H).

Pre-committed decision rule (applied by --aggregate, separately for frdK and frdX; Hw = the
largest width, Hn = the smallest width):
  FAIL      if on ANY instance rho_sign(X,Hw) >= 0.9, or rho_tot(X,Hw) > 1.1;
  PASS      if on EVERY instance rho_sign(X,Hw) <= 0.7, rho_tot(X,Hw) <= 1.0,
            rho_sign(X,Hn) <= 1.0, and, on instances flagged with a same-sector
            near-degenerate competitor (Delta_1/V < 0.02 t), rho_C(X,Hw) <= 1.1;
  AMBIGUOUS otherwise. Pre-committed action: rerun the AMBIGUOUS route with --seed 2 and 3;
            PASS if the per-instance medians over the three seeds satisfy PASS, else FAIL.
  Reported, not used in the verdict: whether dE_sign(X_Hw) < dE_sign(NNBF_128) stored by P1.1
  (--p11-results), i.e. whether the dressing beats a 4x wider backflow on the sign.

Usage
-----
    python3 micro_test.py [options]

Options
-------
  --instance K      Integer 0, 1, 2, 3 or -1 (default 1). Same pre-committed clusters as P1.1:
                      0 : 4x3 PBC, N_up=N_dn=5, U=8, t'=-0.25
                      1 : 4x3 PBC, N_up=N_dn=4, U=8, t'=-0.25 (near-degenerate competitor)
                      2 : 4x3 PBC, N_up=N_dn=5, U=4, t'=-0.25
                      3 : 2x2 OBC, N_up=N_dn=1, U=4, t'=0 (code check only)
                     -1 : use --Lx --Ly --bcx --bcy --nup --ndn --U --tp below.
  --Lx, --Ly INT    Lattice size (defaults 4, 3); used only with --instance -1.
  --bcx, --bcy      "pbc" or "obc" (default pbc, pbc); only with --instance -1.
  --nup, --ndn INT  Electrons per spin (default 4, 4); only with --instance -1.
  --t FLOAT         Nearest-neighbour hopping (default 1.0).
  --tp FLOAT        Next-nearest hopping t' (default -0.25); only with --instance -1.
  --U FLOAT         On-site repulsion (default 8.0); only with --instance -1.
  --arms LIST       Comma-separated arms to train (default "nnbfj,frdK,frdX"). Valid entries:
                    nnbfj (NNBF x Jastrow control), frdK (kinetic frame), frdX (exchange frame).
  --widths LIST     Comma-separated hidden widths H (default "8,32"); every arm is run at
                    every width. Positive integers.
  --task INT        If >= 0, run only the single (arm, width, shift) triple with this index
                    in the row-major list arms x widths x shifts (default -1 = run all).
                    The Slurm array passes task = array_index mod n_triples and the
                    instance = array_index div n_triples.
  --sr-iters INT    SR iterations per arm (default 1500, as P1.1).
  --sr-lr FLOAT     SR step size (default 0.05).
  --sr-shifts LIST  Comma-separated SR diagonal shifts (default "1e-3,1e-2"). Every
                    (arm, width) is trained once per shift, and the aggregate keeps, per
                    (arm, width, instance), the run with the LOWEST variational energy E_theta
                    (selection by energy only, no reference to psi_0). This gives every arm
                    the same small optimizer-hyperparameter search: P1.1 used 1e-3 for NNBF;
                    the login-node probe showed FRD is unstable at 1e-3 on a 3x2 cluster.
  --sr-samples INT  Configurations drawn exactly from |psi|^2 for the SR metric (default 2000;
                    the force is always exact).
  --chunk INT       Configurations per evaluation chunk (default 65536).
  --lbfgs-iters INT L-BFGS-B iterations per start in the sign budget (default 4000).
  --fn-iters INT    Self-consistent ten Haaf iterations (default 10).
  --fn-floor FLOAT  Guide floor of the fixed-node solve (default 1e-3).
  --r-min FLOAT     Minimal bond distance for the pairing error (default 1.5).
  --seed INT        RNG seed (default 1234). Same seed => same UHF start for every arm.
  --out-dir PATH    Directory for result JSONs (default results/ next to this script).
                    One JSON per (instance, arm, width, seed); existing files are skipped.
  --mode STR        "run" (default): train and budget the selected arms.
                    "smoke": correctness checks plus a tiny 3x2 run (about 20 s), no files.
                    "checks": correctness checks only.
                    "timing": build instance, time 3 SR iterations of each arm at the
                    largest width, print seconds/iteration, no files.
  --aggregate DIR   If non-empty, do no computation: read all JSONs in DIR, print the
                    rho table and the pre-committed verdict per route (default "").
  --p11-results DIR P1.1 results directory used for the NNBF_128 reference line in
                    --aggregate (default ../P1.1_sign_vs_amplitude/results).

Outputs
-------
  JSON per (instance, arm, width, seed): params, ED levels, hard_core (gap per site,
  competitor flag), n_params, and the full P1.1 budget dict (E, dE_tot, dE_sign, dE_amp,
  f_E_up, E_FN, w_sign, eC_tot, eC_sign, f_C, ...), plus "alpha_stats" for FRD arms
  (fraction of |psi|^2 weight on configurations where at least one factor is negative,
  i.e. where the dressing actually flipped the sign of Phi).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p11_core as C  # noqa: E402  (verbatim copy of the P1.1 test module)

log = C.log
INSTANCES = C.INSTANCES
RHO_PASS, RHO_FAIL, TOT_PASS, TOT_FAIL, C_PASS, RHO_N = 0.7, 0.9, 1.0, 1.1, 1.1, 1.0


# ----------------------------------------------------------------------------- edge tables
def directed_hops(V, N, T):
    """For one species: for every ordered pair (a->b) with T[a,b] != 0 and a < b or a > b,
    arrays nb[k] (target species index or -1) and sg[k] (sign of <m| c^dag_b c_a |k>)."""
    states, lookup, _, _ = C.species_basis(V, N)
    out = {}
    for a in range(V):
        for b in range(V):
            if a == b or T[a, b] == 0.0:
                continue
            nb = -np.ones(len(states), dtype=np.int64)
            sg = np.zeros(len(states))
            for k, s in enumerate(states):
                s = int(s)
                if not (s >> a) & 1 or (s >> b) & 1:
                    continue
                s1 = s ^ (1 << a)
                p = C.popcount_below(s, a) + C.popcount_below(s1, b)
                nb[k] = lookup[s1 | (1 << b)]
                sg[k] = (-1) ** p
            out[(a, b)] = (nb, sg)
    return out


def bond_list(T):
    V = T.shape[0]
    return [(i, j) for i in range(V) for j in range(i + 1, V) if T[i, j] != 0.0]


class Edges:
    """Edge tables for frame 'K' (kinetic) or 'X' (exchange) on a Hubbard model m."""

    def __init__(self, m, frame):
        self.m, self.frame = m, frame
        T = m.T
        self.bonds = bond_list(T)
        hu = directed_hops(m.V, m.nup, T)
        hd = directed_hops(m.V, m.ndn, T)
        self.spec = []  # list of callables (iu, id) -> (m_full or -1, coeff)
        Dd = m.Dd
        for (i, j) in self.bonds:
            if frame == "K":
                tij = T[i, j]
                for sp_, h in (("u", hu), ("d", hd)):
                    f1, f2 = h[(i, j)], h[(j, i)]
                    self.spec.append(("K", sp_, f1, f2, tij))
            else:
                # orientation 0: S+_i S-_j : up j->i, dn i->j ; orientation 1: S-_i S+_j
                self.spec.append(("X", hu[(j, i)], hd[(i, j)]))
                self.spec.append(("X", hu[(i, j)], hd[(j, i)]))
        self.E = len(self.spec)
        self.Dd = Dd

    def neighbours(self, iu, idd):
        """Return (Mi, Cc): arrays (B, E) of neighbour full index (or -1) and coefficient."""
        B = len(iu)
        Mi = -np.ones((B, self.E), dtype=np.int64)
        Cc = np.zeros((B, self.E))
        Dd = self.Dd
        for e, s in enumerate(self.spec):
            if s[0] == "K":
                _, sp_, (n1, g1), (n2, g2), tij = s
                k = iu if sp_ == "u" else idd
                tgt = np.where(n1[k] >= 0, n1[k], n2[k])
                sg = np.where(n1[k] >= 0, g1[k], g2[k])
                ok = tgt >= 0
                if sp_ == "u":
                    full = tgt * Dd + idd
                else:
                    full = iu * Dd + tgt
                Mi[ok, e] = full[ok]
                Cc[ok, e] = -tij * sg[ok]
            else:
                _, (nu_, gu), (nd_, gd) = s
                tu, td = nu_[iu], nd_[idd]
                ok = (tu >= 0) & (td >= 0)
                Mi[ok, e] = (tu * Dd + td)[ok]
                Cc[ok, e] = -(gu[iu] * gd[idd])[ok]
        return Mi, Cc


# ----------------------------------------------------------------------------- FRD ansatz
class FRD:
    """Factorized-ratio dressing on a Slater-Jastrow base (see module docstring)."""

    def __init__(self, model, Pu, Pd, frame, width, seed):
        self.m = model
        self.base = C.Ansatz(model, Pu, Pd, "sj", jastrow=True, seed=seed)
        self.edges = Edges(model, frame)
        rng = np.random.default_rng(seed + 7919)
        nin, E = 2 * model.V, self.edges.E
        self.q = {"W1": rng.standard_normal((nin, width)) / np.sqrt(nin),
                  "b1": np.zeros(width),
                  "W2": 1e-3 * rng.standard_normal((width, E)),
                  "b2": np.zeros(E)}
        self.qkeys = list(self.q.keys())
        self.iu, self.idd = model.iu_id()
        self._phi = None

    # parameters ---------------------------------------------------------------------
    def n_params(self):
        return self.base.n_params() + int(sum(v.size for v in self.q.values()))

    def flat(self):
        return np.concatenate([self.base.flat()] + [self.q[k].ravel() for k in self.qkeys])

    def set_flat(self, x):
        nb = self.base.n_params()
        self.base.set_flat(x[:nb])
        o = nb
        for k in self.qkeys:
            n = self.q[k].size
            self.q[k] = x[o:o + n].reshape(self.q[k].shape); o += n
        self._phi = None

    # pieces -------------------------------------------------------------------------
    def _x(self, idx):
        m = self.m
        return np.concatenate([m.occ_u[self.iu[idx]], m.occ_d[self.idd[idx]]], 1).astype(float) * 2 - 1

    def phi(self, chunk):
        if self._phi is None:
            self._phi = self.base.amplitudes(chunk)
        return self._phi

    def _factors(self, idx, phi):
        """For configs idx: g (B,E), F (B,E), alpha (B,E), hidden h, x, Mi, Cc, and the
        quantity q_e = g_e v^2/(v^2+u^2) = R (1+R^2)^(-3/2) used in the chain rule."""
        Mi, Cc = self.edges.neighbours(self.iu[idx], self.idd[idx])
        v = phi[idx][:, None]
        u = Cc * np.where(Mi >= 0, phi[np.maximum(Mi, 0)], 0.0)
        den = np.sqrt(v * v + u * u)
        den = np.where(den > 0, den, 1.0)
        g = u * np.sign(v) / den
        g = np.where(Mi >= 0, g, 0.0)
        qd = g * (v * v) / (den * den)
        x = self._x(idx)
        h = np.tanh(x @ self.q["W1"] + self.q["b1"])
        al = h @ self.q["W2"] + self.q["b2"]
        F = 1.0 + al * g
        return dict(g=g, F=F, al=al, h=h, x=x, Mi=Mi, qd=qd)

    def amplitudes(self, chunk):
        phi = self.phi(chunk)
        D = self.m.D
        out = np.empty(D)
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            f = self._factors(idx, phi)
            out[idx] = phi[idx] * np.prod(f["F"], 1)
        return out

    def _safeF(self, F):
        return np.where(np.abs(F) > 1e-300, F, 1e-300)

    def grad_weighted(self, w, chunk):
        """sum_n w(n) dlog psi(n)/dtheta (exact, full basis)."""
        phi = self.phi(chunk)
        D = self.m.D
        wphi = np.zeros(D)  # weights multiplying dlog Phi(m)
        gq = {k: np.zeros_like(v) for k, v in self.q.items()}
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            f = self._factors(idx, phi)
            Fs = self._safeF(f["F"])
            gam = f["al"] * f["qd"] / Fs            # (B,E)
            wb = w[idx]
            wphi[idx] += wb * (1.0 - gam.sum(1))
            ok = f["Mi"] >= 0
            np.add.at(wphi, f["Mi"][ok], (wb[:, None] * gam)[ok])
            dal = wb[:, None] * f["g"] / Fs          # d/d alpha
            gq["W2"] += f["h"].T @ dal
            gq["b2"] += dal.sum(0)
            dpre = (dal @ self.q["W2"].T) * (1 - f["h"] ** 2)
            gq["W1"] += f["x"].T @ dpre
            gq["b1"] += dpre.sum(0)
        gb = self.base.grad_weighted(wphi, chunk)
        return np.concatenate([gb] + [gq[k].ravel() for k in self.qkeys])

    def per_sample_O_fast(self, idx):
        phi = self.phi(65536)
        f = self._factors(idx, phi)
        Fs = self._safeF(f["F"])
        gam = f["al"] * f["qd"] / Fs
        B, E = gam.shape
        Ob_self = self.base.per_sample_O_fast(idx)
        Ob = Ob_self * (1.0 - gam.sum(1))[:, None]
        ok = f["Mi"] >= 0
        rows, cols = np.nonzero(ok)
        if len(rows):
            On = self.base.per_sample_O_fast(f["Mi"][rows, cols])
            np.add.at(Ob, rows, On * gam[rows, cols][:, None])
        dal = f["g"] / Fs
        h, x = f["h"], f["x"]
        Hh = h.shape[1]
        OW2 = (h[:, :, None] * dal[:, None, :]).reshape(B, -1)
        dpre = (dal @ self.q["W2"].T) * (1 - h ** 2)
        OW1 = (x[:, :, None] * dpre[:, None, :]).reshape(B, -1)
        parts = {"W1": OW1, "b1": dpre, "W2": OW2, "b2": dal}
        return np.concatenate([Ob] + [parts[k] for k in self.qkeys], 1)

    def flip_weight(self, psi):
        """|psi|^2 weight on configs where an odd number of factors is negative."""
        phi = self.phi(65536)
        neg = np.zeros(self.m.D, dtype=bool)
        for s in range(0, self.m.D, 65536):
            idx = np.arange(s, min(s + 65536, self.m.D))
            f = self._factors(idx, phi)
            neg[idx] = (np.sum(f["F"] < 0, 1) % 2) == 1
        p = psi ** 2 / (psi @ psi)
        return float(p[neg].sum())


# --------------------------------------------------------------------------- wrappers
def make_arm(m, Pu, Pd, arm, width, seed):
    if arm == "nnbfj":
        return C.Ansatz(m, Pu, Pd, "nnbf", width=width, jastrow=True, seed=seed + width)
    if arm == "frdK":
        return FRD(m, Pu, Pd, "K", width, seed + width)
    if arm == "frdX":
        return FRD(m, Pu, Pd, "X", width, seed + width)
    raise ValueError(arm)


# ----------------------------------------------------------------------------- checks
def run_checks(seed):
    ok = C.run_checks(seed)
    # (1) kinetic edges reproduce the off-diagonal part of H acting on any vector
    for (Lx, Ly, bcx, bcy, nu, nd, tp) in [(3, 2, "pbc", "obc", 2, 2, -0.3), (2, 3, "obc", "pbc", 2, 1, 0.25)]:
        m = C.Hubbard(Lx, Ly, bcx, bcy, nu, nd, 1.0, tp, 4.0)
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(m.D)
        ed = Edges(m, "K")
        Mi, Cc = ed.neighbours(*m.iu_id())
        Ky = np.sum(Cc * np.where(Mi >= 0, y[np.maximum(Mi, 0)], 0.0), 1)
        Hoff = m.H - C.sp.diags(m.H.diagonal())
        err = float(np.abs(Ky - Hoff @ y).max())
        log(f"check kinetic edges {Lx}x{Ly} N=({nu},{nd}): max|K y - H_off y| = {err:.2e}")
        ok &= err < 1e-12
    # (2) exchange edges: on the 2-site U=200 ground state (a singlet up to O(t/U)),
    # sum_e c_e psi(m_e)/psi(n) must be -1 on every singly-occupied configuration
    m = C.Hubbard(2, 1, "obc", "obc", 1, 1, 1.0, 0.0, 200.0)
    w, U = np.linalg.eigh(m.H.toarray()); p0 = U[:, 0]
    ed = Edges(m, "X")
    Mi, Cc = ed.neighbours(*m.iu_id())
    single = m.dbl == 0
    R = np.sum(Cc * np.where(Mi >= 0, p0[np.maximum(Mi, 0)], 0.0), 1)[single] / p0[single]
    log(f"check exchange edges 2-site singlet: R = {R.tolist()} (expect -1 each)")
    ok &= bool(np.all(np.abs(R + 1) < 1e-3))
    # (2b) same on a 4-site Heisenberg ring at U=400: <sum_b S+S- + h.c.> via edges equals
    # the expectation computed from <S.S> - <SzSz> using ED
    m = C.Hubbard(4, 1, "pbc", "obc", 2, 2, 1.0, 0.0, 400.0)
    w, U = np.linalg.eigh(m.H.toarray()); p0 = U[:, 0]
    ed = Edges(m, "X")
    Mi, Cc = ed.neighbours(*m.iu_id())
    exch = float(p0 @ np.sum(Cc * np.where(Mi >= 0, p0[np.maximum(Mi, 0)], 0.0), 1))
    iu, idd = m.iu_id()
    sz = 0.5 * (m.occ_u[iu] - m.occ_d[idd]).astype(float)
    zz = float(np.sum(p0 ** 2 * sum(sz[:, i] * sz[:, (i + 1) % 4] for i in range(4))))
    J = 4.0 / 400.0
    ss = (w[0] + J) / J  # H_eff = J sum_b (S.S - 1/4) => sum_b <S.S> = E0/J + 1 (leading order)
    log(f"check exchange edges 4-ring: sum<S+S-+h.c.>/2 = {exch / 2:.4f}  "
        f"from ED sum<S.S>-<SzSz> = {ss - zz:.4f} (leading order in t/U)")
    ok &= abs(exch / 2 - (ss - zz)) < 0.05
    # (3) FRD gradients (frames K and X) against finite differences of the energy
    m = C.Hubbard(3, 2, "pbc", "obc", 2, 2, 1.0, -0.3, 6.0)
    Pu, Pd = C.uhf_orbitals(m, seed)
    for fr in ("K", "X"):
        a = FRD(m, Pu, Pd, fr, 4, seed)
        a.q["W2"] = 0.5 * np.random.default_rng(1).standard_normal(a.q["W2"].shape)
        a.q["b2"] += 0.3
        a.base.p["vj"] = 0.1 * np.random.default_rng(2).standard_normal(a.base.p["vj"].shape)
        a._phi = None
        x0 = a.flat().copy()
        psi = a.amplitudes(4096); E, Hp, nn = C.energy(m.H, psi)
        g = 2 * a.grad_weighted(psi * (Hp - E * psi) / nn, 4096)
        d = np.random.default_rng(seed).standard_normal(len(x0)); d /= np.linalg.norm(d)
        hh = 1e-6
        a.set_flat(x0 + hh * d); Ep = C.energy(m.H, a.amplitudes(4096))[0]
        a.set_flat(x0 - hh * d); Em = C.energy(m.H, a.amplitudes(4096))[0]
        a.set_flat(x0)
        fd = (Ep - Em) / (2 * hh)
        # per-sample O against the full-basis gradient: sum_n p(n) O(n) == grad of log-norm
        idx = np.arange(m.D)
        O = a.per_sample_O_fast(idx)
        w1 = psi * psi / nn
        lhs = w1 @ O
        rhs = a.grad_weighted(w1, 4096)
        e2 = float(np.abs(lhs - rhs).max())
        log(f"check FRD-{fr} gradient: analytic {g @ d:.10e}  finite-diff {fd:.10e}  "
            f"|O-vs-grad| {e2:.1e}  P={a.n_params()}")
        ok &= abs(g @ d - fd) < 1e-6 * max(1.0, abs(fd)) and e2 < 1e-8
    log(f"ALL R2 CHECKS {'PASSED' if ok else 'FAILED'}")
    return ok


# ----------------------------------------------------------------------------- main run
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 round-2 FRD vs NNBF (see module docstring)")
    ap.add_argument("--instance", type=int, default=1, choices=[-1, 0, 1, 2, 3])
    ap.add_argument("--Lx", type=int, default=4)
    ap.add_argument("--Ly", type=int, default=3)
    ap.add_argument("--bcx", choices=["pbc", "obc"], default="pbc")
    ap.add_argument("--bcy", choices=["pbc", "obc"], default="pbc")
    ap.add_argument("--nup", type=int, default=4)
    ap.add_argument("--ndn", type=int, default=4)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--tp", type=float, default=-0.25)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--arms", type=str, default="nnbfj,frdK,frdX")
    ap.add_argument("--widths", type=str, default="8,32")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--sr-iters", type=int, default=1500)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shifts", type=str, default="1e-3,1e-2")
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--mode", choices=["run", "smoke", "checks", "timing"], default="run")
    ap.add_argument("--aggregate", type=str, default="")
    ap.add_argument("--p11-results", type=str,
                    default=os.path.join(HERE, "..", "P1.1_sign_vs_amplitude", "results"))
    a = ap.parse_args(argv)
    if a.instance >= 0:
        for k, v in INSTANCES[a.instance].items():
            setattr(a, k, v)
    a.arm_list = [s.strip() for s in a.arms.split(",") if s.strip()]
    for s in a.arm_list:
        if s not in ("nnbfj", "frdK", "frdX"):
            ap.error(f"unknown arm {s}")
    a.width_list = [int(w) for w in a.widths.split(",") if w.strip()]
    a.shift_list = [float(x) for x in a.sr_shifts.split(",") if x.strip()]
    a.sr_shift = a.shift_list[0]
    return a


def setup(args):
    m = C.Hubbard(args.Lx, args.Ly, args.bcx, args.bcy, args.nup, args.ndn, args.t, args.tp, args.U)
    m.pair_setup(args.r_min)
    W, Vv, deg = C.solve_ed(m, 3, 1e-8, args.seed)
    gap = float(W[1] - W[0])
    hc = dict(levels3=W[:3].tolist(), gap_per_site=gap / (m.V * args.t),
              competitor_flag=bool(gap / (m.V * args.t) < C.COMP_TOL))
    Pu, Pd = C.uhf_orbitals(m, args.seed)
    return m, W, Vv, deg, hc, Pu, Pd


def tag(args, arm, w, shift):
    return (f"L{args.Lx}x{args.Ly}_{args.bcx}{args.bcy}_n{args.nup}-{args.ndn}_U{args.U:g}"
            f"_tp{args.tp:g}_{arm}_H{w}_sh{shift:g}_s{args.seed}")


def run_pairs(args, pairs, write=True):
    m, W, Vv, deg, hc, Pu, Pd = setup(args)
    log(f"model V={m.V} N=({m.nup},{m.ndn}) D={m.D}  E0..2={hc['levels3']}  "
        f"Delta1/V={hc['gap_per_site']:.4f}{'  competitor' if hc['competitor_flag'] else ''}")
    rng = np.random.default_rng(args.seed)
    out = []
    for arm, w, shift in pairs:
        args.sr_shift = shift
        fn = os.path.join(args.out_dir, tag(args, arm, w, shift) + ".json")
        if write and os.path.exists(fn):
            log(f"{fn} exists; skipping"); continue
        a = make_arm(m, Pu, Pd, arm, w, args.seed)
        t0 = time.time()
        C.optimize_sr(a, m.H, args, rng, f"{arm}_{w}")
        psi = a.amplitudes(args.chunk)
        b = C.budget(m, psi, W, Vv, deg, args, rng)
        b["n_params"] = a.n_params()
        b["train_seconds"] = time.time() - t0
        if isinstance(a, FRD):
            b["flip_weight"] = a.flip_weight(psi)
        log(f"{arm}_{w}: P={b['n_params']} dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} "
            f"f_E_up={b['f_E_up']:.3f} w_sign={b['w_sign']:.4f} eC_tot={b['eC_tot']:.3e}"
            + (f" flip_weight={b['flip_weight']:.4f}" if "flip_weight" in b else ""))
        res = dict(params={k: v for k, v in vars(args).items()}, instance=args.instance, arm=arm,
                   width=w, shift=shift, ed_levels=W.tolist(), hard_core=hc, budget=b)
        out.append(res)
        if write:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(fn, "w") as fh:
                json.dump(res, fh, indent=1, default=float)
            log(f"wrote {fn}")
    return out


def p11_ref(d, inst):
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        if r.get("params", {}).get("instance") == inst and "nnbf_128" in r.get("states", {}):
            return r["states"]["nnbf_128"]
    return None


def aggregate(d, p11dir):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        key = (r["instance"], r["params"]["seed"])
        k2 = (r["arm"], r["width"])
        old = rows.setdefault(key, {}).get(k2)
        if old is None or r["budget"]["E"] < old["budget"]["E"]:
            rows[key][k2] = r  # best of the SR shifts, chosen by variational energy
    if not rows:
        print("no results"); return 1
    verdict = {}
    for route in ("frdK", "frdX"):
        per_inst = []
        for (inst, seed), R in sorted(rows.items()):
            ws = sorted({w for (_, w) in R})
            if not ws:
                continue
            Hn, Hw = ws[0], ws[-1]
            need = [("nnbfj", Hn), ("nnbfj", Hw), (route, Hn), (route, Hw)]
            if not all(k in R for k in need):
                print(f"[{route}] instance {inst} seed {seed}: incomplete ({sorted(R)})"); per_inst.append("INCOMPLETE"); continue
            b = lambda k: R[k]["budget"]
            rs_w = b((route, Hw))["dE_sign"] / b(("nnbfj", Hw))["dE_sign"]
            rt_w = b((route, Hw))["dE_tot"] / b(("nnbfj", Hw))["dE_tot"]
            rs_n = b((route, Hn))["dE_sign"] / b(("nnbfj", Hn))["dE_sign"]
            rc_w = b((route, Hw))["eC_tot"] / b(("nnbfj", Hw))["eC_tot"]
            comp = R[(route, Hw)]["hard_core"]["competitor_flag"]
            ref = p11_ref(p11dir, inst)
            beat128 = (b((route, Hw))["dE_sign"] < ref["dE_sign"]) if ref else None
            if rs_w >= RHO_FAIL or rt_w > TOT_FAIL:
                v = "FAIL"
            elif rs_w <= RHO_PASS and rt_w <= TOT_PASS and rs_n <= RHO_N and (not comp or rc_w <= C_PASS):
                v = "PASS"
            else:
                v = "AMBIGUOUS"
            per_inst.append(v)
            print(f"[{route}] inst {inst} seed {seed}: rho_sign(H{Hw})={rs_w:.3f} rho_tot(H{Hw})={rt_w:.3f} "
                  f"rho_sign(H{Hn})={rs_n:.3f} rho_C(H{Hw})={rc_w:.3f} competitor={comp} "
                  f"dE_sign: {route}={b((route, Hw))['dE_sign']:.5f} nnbfj={b(('nnbfj', Hw))['dE_sign']:.5f} "
                  f"P1.1 NNBF_128={ref['dE_sign'] if ref else float('nan'):.5f} beats128={beat128} "
                  f"flip_weight={b((route, Hw)).get('flip_weight', float('nan')):.4f} -> {v}")
        if any(v == "FAIL" for v in per_inst):
            g = "FAIL"
        elif per_inst and all(v == "PASS" for v in per_inst):
            g = "PASS"
        else:
            g = "AMBIGUOUS (action: rerun with --seed 2 and 3; median decides)"
        verdict[route] = g
        print(f"ROUTE {route}: {g}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        return aggregate(args.aggregate, args.p11_results)
    if args.mode in ("checks", "smoke"):
        if not run_checks(args.seed):
            log("checks failed"); return 2
        if args.mode == "checks":
            return 0
        # tiny 3x2 doped t' != 0 run: exercises training, budget and aggregation logic
        sa = parse_args(["--instance", "-1", "--Lx", "3", "--Ly", "2", "--bcx", "pbc", "--bcy", "obc",
                         "--nup", "2", "--ndn", "2", "--U", "8", "--tp", "-0.3", "--widths", "4",
                         "--sr-iters", "60", "--sr-samples", "150", "--lbfgs-iters", "500",
                         "--fn-iters", "3"])
        res = run_pairs(sa, [(a, 4, 1e-2) for a in ("nnbfj", "frdK", "frdX")], write=False)
        for r in res:
            b = r["budget"]
            print(f"SMOKE {r['arm']}_H4: P={b['n_params']} dE_tot={b['dE_tot']:.4e} "
                  f"dE_sign={b['dE_sign']:.4e} w_sign={b['w_sign']:.4f}")
        print("SMOKE: OK")
        return 0
    pairs = [(a, w, s) for a in args.arm_list for w in args.width_list for s in args.shift_list]
    if args.mode == "timing":
        m = C.Hubbard(args.Lx, args.Ly, args.bcx, args.bcy, args.nup, args.ndn, args.t, args.tp, args.U)
        Pu, Pd = C.uhf_orbitals(m, args.seed)
        log(f"timing on D={m.D} (no ED)")
        rng = np.random.default_rng(args.seed)
        Hw = max(args.width_list)
        for arm in args.arm_list:
            a = make_arm(m, Pu, Pd, arm, Hw, args.seed)
            args2 = argparse.Namespace(**vars(args)); args2.sr_iters = 3; args2.sr_shift = 1e-2
            t0 = time.time()
            C.optimize_sr(a, m.H, args2, rng, f"timing_{arm}_{Hw}")
            log(f"TIMING {arm}_H{Hw}: {(time.time() - t0) / 3:.2f} s/iteration (P={a.n_params()})")
        return 0
    if args.task >= 0:
        pairs = [pairs[args.task]]
    run_pairs(args, pairs, write=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
