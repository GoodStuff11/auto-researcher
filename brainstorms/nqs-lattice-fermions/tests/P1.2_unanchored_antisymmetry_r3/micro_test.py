#!/usr/bin/env python3
"""
P1.2 round-3 decisive test: symmetry-projected FRD versus symmetry-projected NNBF x Jastrow
===========================================================================================

Question
--------
Does the factorized-ratio dressing (FRD) repair NODAL (sign) error that a symmetry-projected
neural-network backflow cannot, at a matched parameter count, when both are trained WITH the
projection onto psi_0's space-group irrep and total spin? Round 2 compared unprojected arms;
the round-2 critic showed that most unprojected NNBF sign error on the 4x3 PBC 4+4 cluster is
symmetry contamination, so a gain could be symmetry restoration rather than nodal repair.
Round 3 therefore decides on PROJECTED arms only, reports every unprojected arm after post-hoc
projection, and tests the SIGN HALF of the Hard Core only (no same-sector competitor at the
1e-3 t per site scale is claimed; the in-sector gap of each instance is computed and printed).

Why new instances: interim logs of P1.1-P (job 407525, still running) show projected-SR NNBF at
width 8 reaching dE_tot ~ 3e-3 (5+5, U=8), 8e-4 (4+4, U=8) and 3e-4 (5+5, U=4) on the three
4x3 PBC P1.1 clusters, i.e. those clusters leave almost nothing to repair once |G| = 48
projection is used. Round 3 uses lower-symmetry clusters (open or cylinder boundaries, |G| = 4,
16, 24) where projection averages over far fewer images; whether they still leave sign error is
itself checked by a pre-committed premise test (below).

Ansatz under test (FRD; Edges/FRD code copied verbatim from tests/P1.2_unanchored_antisymmetry_r2)
-----------------------------------------------------------------------------------------------
    psi_theta(n) = Phi(n) prod_e [1 + alpha_e(n) g(R_e(n))],  R_e = c_e Phi(m_e(n)) / Phi(n),
    g(R) = R / sqrt(1 + R^2),  alpha = MLP(x) (tanh, hidden width H, one output per edge),
Phi = Slater (trainable orbitals, UHF start) x two-body Jastrow. Frames: frdK (edges = single
hops of the Hubbard kinetic term on NN and NNN bonds) and frdX (edges = spin exchanges S+S-).
Control: ctrl = NNBF x the same Jastrow (Luo & Clark), width --ctrl-width (32). FRD width is the
largest H with n_params(FRD) <= n_params(ctrl) on that instance (frd_width); the residual gap is
< 2V + 1 + E parameters (< 1%), always in the control's favour. checks print the counts.

Arms (one Slurm array task = one (instance, seed, arm, SR shift))
-----------------------------------------------------------------
  ctrl_PS, frdK_PS, frdX_PS : trained WITH projection, psi = P_S P_chi psi_theta (decisive).
      P_chi = (1/|G|) sum_g chi(g) U_g over the cluster's space group (translations along periodic
      directions x mirrors), chi = psi_0's real 1-D irrep; P_S = exact Loewdin projector onto
      psi_0's total spin. Exact force sum_n r(n) d psi_theta(n), r = (H - E) psi / ||psi||^2;
      SR metric from --sr-samples exact draws of |P_chi psi_theta|^2 with the projected
      log-derivatives O_chi (P_S is not local, so for P_S the metric is a positive-definite
      preconditioner; the force and every fixed point are exact). Same rule for all arms.
  ctrl_U, frdK_U, frdX_U    : trained without projection (round-2 protocol), then budgeted
      as U (psi_theta), UG (P_chi psi_theta, post hoc) and UGS (P_S P_chi psi_theta, post hoc).
  Each route is trained at SR shifts 1e-3 and 1e-2; the aggregate keeps, per (instance, arm,
  seed), the shift with the lowest variational energy (energy only, no reference to psi_0).

Budgets per state (P1.1 budget(), exact sums): dE_tot, dE_sign = E_sign_up - E_0 (best of four
L-BFGS-B starts and the ten Haaf fixed node), w_sign, spread, f_E; plus <S^2>, overlap with
psi_0, flip_weight (FRD: |psi|^2 weight where an odd number of factors is negative).
Vote ablation (FRD_PS only): "abl" = P_S P_chi Phi with Phi the trained base of the same state,
i.e. the same state with every alpha_e = 0. rho_vote = dE_sign(X_PS) / dE_sign(abl).
Sampled-estimator arm (every task; P_chi-projected state for PS arms): genuine Monte Carlo from
|psi_c|^2 with the estimators a sampled VMC run would use (sampled_diag): exact Var(E_loc) and
its kurtosis, the tail shares share_El and share_O (gated), max-to-mean tails, and the
sampled-force noise eta (reported only: it depends on the parameterisation). FRD's amplitude ratios and 1/F log-derivatives are the heavy-tail risk measured here.

Pre-committed decision rule (implemented in aggregate(); constants below)
--------------------------------------------------------------------------
Per instance, seeds with all three PS arms (>= 3 needed), medians over seeds:
  premise: the instance is INFORMATIVE iff med dE_sign(ctrl_PS) >= TAU_INF = 0.01 t.
  For route X in {frdK, frdX} on an informative instance:
    rho_sign = med dE_sign(X_PS)/med dE_sign(ctrl_PS),  rho_tot = same for dE_tot,
    wins = #seeds with dE_sign(X_PS) < dE_sign(ctrl_PS) (paired by seed = same UHF start),
    rho_vote = med over seeds of dE_sign(X_PS)/dE_sign(abl),
    share_El = exact fraction of Var(E_loc) carried by the most extreme 1e-3 of |psi_c|^2 weight,
    share_O  = fraction of sum ||O_c - <O_c>||^2 carried by the top 1e-3 of the 8000 samples
               (medians over seeds; "heavy" means > 0.5 AND > 2x the control's value),
    guards: calibration error <= 1e-3 min(dE_tot) and multi-start spread < 0.05 for both arms.
    FAIL      if rho_sign >= 0.9 or rho_tot > 1.1 or share_El or share_O of X is heavy;
    PASS      if guards, rho_sign <= 0.7, rho_tot <= 1.0, wins >= ceil(2 n_seeds / 3),
              rho_vote <= 0.8 and neither share is heavy;
    AMBIGUOUS otherwise.
  Route verdict: NO-TARGET if fewer than 2 instances are informative; else FAIL if any
  informative instance FAILs, PASS if all PASS, else AMBIGUOUS with the pre-committed action:
  run seeds 4567 and 5678 (same arms) and re-aggregate with --final 1, which turns AMBIGUOUS
  into FAIL. Unprojected-arm ratios (U, post-hoc UG, UGS) are printed as the symmetry-versus-
  nodal diagnostic and do not enter the verdict.

Usage
-----
    python3 micro_test.py [options]

Options
-------
  --task-id INT     If >= 0, OVERRIDES --instance/--seed/--arm/--sr-shift from a Slurm array
                    index t in 0..107: instance = t // 36, r = t % 36, seed = SEEDS[r // 12]
                    (1234, 2345, 3456), arm = ARMS[(r % 12) // 2], shift = SHIFTS[r % 2]
                    (1e-3, 1e-2). Default -1 (off).
  --instance K      0, 1, 2 (production) or 3 (smoke). All t = 1, t' = -0.25 unless noted:
                      0 : 4x3 open x open,      N_up = N_dn = 4, U = 8 (D = 245025, |G| = 4)
                      1 : 4x3 periodic x open,  N_up = N_dn = 4, U = 8 (D = 245025, |G| = 16)
                      2 : 6x2 periodic x open,  N_up = N_dn = 5, U = 8 (D = 627264, |G| = 24)
                      3 : 3x2 periodic x open,  N_up = N_dn = 2, U = 8, t' = -0.3 (D = 225; smoke)
                    Default 0.
  --seed INT        UHF noise seed, network init seed (+ width), SR sampling stream. Default 1234.
  --arm STR         One of ctrl_PS, frdK_PS, frdX_PS, ctrl_U, frdK_U, frdX_U (see Arms).
                    Default ctrl_PS.
  --sr-shift FLOAT  SR diagonal shift (default 1e-2; the array uses 1e-3 and 1e-2).
  --ctrl-width INT  Hidden width of the NNBF control (default 32). FRD width is derived from it.
  --sr-iters INT    SR iterations (default 1500, as P1.1 / P1.1-P).
  --sr-lr FLOAT     SR step size (default 0.05).
  --sr-samples INT  Exact draws for the SR metric (default 2000).
  --chunk INT       Configurations per full-basis evaluation chunk (default 65536).
  --o-chunk INT     Samples per chunk of projected log-derivatives (each has |G| images;
                    default 64).
  --lbfgs-iters INT L-BFGS-B iterations per start in the budget (default 4000).
  --fn-iters INT    Self-consistent ten Haaf iterations (default 10).
  --fn-floor FLOAT  Fixed-node guide floor (default 1e-3).
  --nb-diag INT     Sampled-arm batches (default 8).
  --ns-diag INT     Samples per sampled-arm batch (default 1000).
  --out-dir PATH    Result directory (default results/ next to this script); one JSON per task,
                    named inst<K>_<arm>_s<seed>_sh<shift>.json.
  --overwrite {0,1} Recompute an existing JSON (default 0: skip).
  --mode STR        run (default): checks, then the task; checks: checks only;
                    smoke: checks, all six arms x 3 seeds on instance 3 for 40 SR steps into
                    results_smoke/, then the aggregate (about a minute);
                    probe: ED, irrep, spin and in-sector gap of instances 0-2 (no training);
                    timing: two SR steps of every arm on --instance (no budgets).
  --aggregate DIR   No computation: apply the decision rule to every JSON in DIR (relative to
                    this script if not absolute) and print the table and route verdicts.
  --final {0,1}     With --aggregate: 1 applies the AMBIGUOUS -> FAIL rule (after extra seeds).

Outputs
-------
  JSON per task: params, instance, arm, route, pmode, seed, shift, width, n_params, ed_levels,
  hard_core (levels, fixed-N gap, irrep, S_psi0, |G|, in-sector levels and gap), calib_err,
  train_seconds, budgets {label: P1.1 budget dict + S2, ov0, flip_weight}, E_select (variational
  energy used to choose the SR shift), sampled (sigma2, kurt, share_El, share_O, tail_El, tail_O,
  eta, n_samples).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p11_core as C  # noqa: E402  (verbatim copy of tests/P1.1_sign_vs_amplitude/micro_test.py)

log = C.log
R3_INSTANCES = {
    0: dict(Lx=4, Ly=3, bcx="obc", bcy="obc", nup=4, ndn=4, U=8.0, tp=-0.25),
    1: dict(Lx=4, Ly=3, bcx="pbc", bcy="obc", nup=4, ndn=4, U=8.0, tp=-0.25),
    2: dict(Lx=6, Ly=2, bcx="pbc", bcy="obc", nup=5, ndn=5, U=8.0, tp=-0.25),
    3: dict(Lx=3, Ly=2, bcx="pbc", bcy="obc", nup=2, ndn=2, U=8.0, tp=-0.3),
}
SEEDS = (1234, 2345, 3456)
ARMS = ("ctrl_PS", "frdK_PS", "frdX_PS", "ctrl_U", "frdK_U", "frdX_U")
SHIFTS = (1e-3, 1e-2)
# pre-committed constants
TAU_INF = 0.01
RHO_PASS, RHO_FAIL, TOT_PASS, TOT_FAIL = 0.7, 0.9, 1.0, 1.1
VOTE_MAX = 0.8
TAIL_MASS, SHARE_MAX, SHARE_FACTOR = 1e-3, 0.5, 2.0
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05
MIN_SEEDS, MIN_INFORMATIVE = 3, 2

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

# ----------------------------------------------------------------------------- checks
def r2_checks(seed):
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



# ============================================================================ symmetry
# Copied from tests/P1.1P_projected/micro_test.py (group, species_perm, SpaceGroup, Spin) and
# generalised to open boundaries: translations exist only along periodic directions, and the
# mirror is x -> -x (mod Lx) on a periodic direction, x -> Lx - 1 - x on an open one.
def group(model):
    """All space-group elements g = T^(a,b) M^(c,d) as (exponents, site permutation)."""
    Lx, Ly, V = model.Lx, model.Ly, model.V
    ax = range(Lx) if model.bcx == "pbc" else range(1)
    ay = range(Ly) if model.bcy == "pbc" else range(1)
    out = []
    for c in (0, 1):
        for d in (0, 1):
            for a in ax:
                for b in ay:
                    p = np.empty(V, dtype=np.int64)
                    for y in range(Ly):
                        for x in range(Lx):
                            if c:
                                x2 = (-x) % Lx if model.bcx == "pbc" else Lx - 1 - x
                            else:
                                x2 = x
                            if d:
                                y2 = (-y) % Ly if model.bcy == "pbc" else Ly - 1 - y
                            else:
                                y2 = y
                            x2, y2 = (x2 + a) % Lx, (y2 + b) % Ly
                            p[x + Lx * y] = x2 + Lx * y2
                    out.append(((a, b, c, d), p))
    return out


def species_perm(states, rows, lookup_V, p):
    """Index map and fermionic sign for one species under site permutation p."""
    newidx = np.empty(len(states), dtype=np.int64)
    sgn = np.empty(len(states))
    for k, occ in enumerate(rows):
        img = p[occ]
        perm = np.argsort(img, kind="stable")
        seen = np.zeros(len(perm), bool); par = 0
        for i in range(len(perm)):
            if not seen[i]:
                j, L = i, 0
                while not seen[j]:
                    seen[j] = True; j = perm[j]; L += 1
                par += L - 1
        b = 0
        for i in img:
            b |= 1 << int(i)
        newidx[k] = lookup_V[b]
        sgn[k] = -1.0 if par % 2 else 1.0
    return newidx, sgn


class SpaceGroup:
    """U_g on the fixed-(N_up, N_dn) basis and the 1-D irrep projector
    P_chi = (1/|G|) sum_g chi(g) U_g."""

    GEN = {(1, 0, 0, 0): "Tx", (0, 1, 0, 0): "Ty", (0, 0, 1, 0): "Mx", (0, 0, 0, 1): "My"}

    def __init__(self, model):
        self.m = model
        st_u, lk_u, _, rows_u = C.species_basis(model.V, model.nup)
        st_d, lk_d, _, rows_d = C.species_basis(model.V, model.ndn)
        self.elems, inv_u, isu, inv_d, isd = [], [], [], [], []
        for e, p in group(model):
            mu, su = species_perm(st_u, rows_u, lk_u, p)
            md, sd = species_perm(st_d, rows_d, lk_d, p)
            iu = np.empty_like(mu); iu[mu] = np.arange(len(mu))
            idd = np.empty_like(md); idd[md] = np.arange(len(md))
            self.elems.append(e)
            inv_u.append(iu); isu.append(su[iu]); inv_d.append(idd); isd.append(sd[idd])
        self.inv_u, self.isu = np.array(inv_u), np.array(isu)
        self.inv_d, self.isd = np.array(inv_d), np.array(isd)
        self.G = len(self.elems)
        self.chi = np.ones(self.G)
        self.lam = {}

    def apply(self, psi, k):
        m = self.m
        Psi = psi.reshape(m.Du, m.Dd)
        out = Psi[np.ix_(self.inv_u[k], self.inv_d[k])] * self.isu[k][:, None] * self.isd[k][None, :]
        return out.ravel()

    def gen_expect(self, psi):
        psi = psi / np.linalg.norm(psi)
        return {self.GEN[e]: float(psi @ self.apply(psi, k))
                for k, e in enumerate(self.elems) if e in self.GEN}

    def set_irrep(self, lam):
        self.lam = dict(lam)
        g = lambda k: lam.get(k, 1)
        self.chi = np.array([g("Tx") ** a * g("Ty") ** b * g("Mx") ** c * g("My") ** d
                             for (a, b, c, d) in self.elems], float)

    def irrep_of(self, psi0):
        ev = self.gen_expect(psi0)
        if any(abs(abs(v) - 1) > 1e-6 for v in ev.values()):
            raise RuntimeError(f"psi_0 is not in a real 1-D irrep: {ev}")
        return {k: int(round(v)) for k, v in ev.items()}

    def project(self, psi):
        out = np.zeros_like(psi)
        for k in range(self.G):
            out += self.chi[k] * self.apply(psi, k)
        return out / self.G


class Spin:
    """S^2 = S^- S^+ + S_z^2 + S_z via a sparse S^+, and the Loewdin projector onto S."""

    def __init__(self, model):
        m = self.m = model
        V, nu, nd = m.V, m.nup, m.ndn
        self.Sz = 0.5 * (nu - nd)
        Ntot = nu + nd
        self.Smax = 0.5 * min(Ntot, 2 * V - Ntot)
        self.S0 = abs(self.Sz)
        if nd == 0 or nu == V:
            self.Sp = sp.csr_matrix((1, m.D))
            return
        su, sd = m.su, m.sd
        st_u2, lk_u2, _, _ = C.species_basis(V, nu + 1)
        st_d2, lk_d2, _, _ = C.species_basis(V, nd - 1)
        Dd2 = len(st_d2)
        pc = lambda s, i: np.array([bin(int(x) & ((1 << i) - 1)).count("1") for x in s])
        r, c, v = [], [], []
        for i in range(V):
            au = np.nonzero(((su >> i) & 1) == 0)[0]
            ad = np.nonzero(((sd >> i) & 1) == 1)[0]
            if len(au) == 0 or len(ad) == 0:
                continue
            newu = lk_u2[su[au] | (1 << i)]
            newd = lk_d2[sd[ad] ^ (1 << i)]
            sg_u = (-1.0) ** pc(su[au], i)
            sg_d = (-1.0) ** pc(sd[ad], i)
            r.append((newu[:, None] * Dd2 + newd[None, :]).ravel())
            c.append((au[:, None] * m.Dd + ad[None, :]).ravel())
            v.append((sg_u[:, None] * sg_d[None, :]).ravel())
        r, c, v = np.concatenate(r), np.concatenate(c), np.concatenate(v)
        self.Sp = sp.csr_matrix((v, (r, c)), shape=(len(st_u2) * Dd2, m.D))

    def S2(self, psi):
        return self.Sp.T @ (self.Sp @ psi) + (self.Sz ** 2 + self.Sz) * psi

    def expect(self, psi):
        return float(psi @ self.S2(psi) / (psi @ psi))

    def project(self, psi, S):
        out = psi.copy()
        tgt = S * (S + 1)
        s = self.S0
        while s <= self.Smax + 1e-9:
            if abs(s - S) > 1e-9:
                ss = s * (s + 1)
                out = (self.S2(out) - ss * out) / (tgt - ss)
            s += 1
        return out


class Projector:
    """mode 'none': identity; 'G': P_chi; 'GS': P_S P_chi (S = S_psi0)."""

    def __init__(self, mode, sg, spn, S):
        self.mode, self.sg, self.spn, self.S = mode, sg, spn, S

    def chi_part(self, psi_t):
        return psi_t if self.mode == "none" else self.sg.project(psi_t)

    def full(self, psi_c):
        return self.spn.project(psi_c, self.S) if self.mode == "GS" else psi_c


# ============================================================================ projected SR
def per_sample_O_chi(ans, sg, idx, psi_t, psi_c, o_chunk):
    """d log psi_c(n)/d theta for sampled basis indices idx, psi_c = P_chi psi_theta:
    O_c(n) = sum_g c_g(n) O_theta(n_g), c_g = chi(g) sigma_g psi_theta(n_g) / (|G| psi_c(n)).
    Works for any ansatz with per_sample_O_fast (NNBF x J and FRD alike)."""
    if sg is None:
        return ans.per_sample_O_fast(idx)
    Dd, G = ans.m.Dd, sg.G
    out = np.empty((len(idx), ans.n_params()))
    for s in range(0, len(idx), o_chunk):
        ii = idx[s:s + o_chunk]
        B = len(ii)
        a, b = ii // Dd, ii % Dd
        img = (sg.inv_u[:, a] * Dd + sg.inv_d[:, b]).T
        fac = (sg.chi[:, None] * sg.isu[:, a] * sg.isd[:, b]).T / G
        coef = fac * psi_t[img] / psi_c[ii][:, None]
        Oi = ans.per_sample_O_fast(img.ravel())
        out[s:s + B] = np.einsum("bgp,bg->bp", Oi.reshape(B, G, -1), coef)
    return out


def optimize_proj(ans, H, proj, args, rng, label):
    """P1.1 SR (exact force, sampled metric, trust region, best iterate kept) for the
    variational state psi = P psi_theta, P in {P_chi, P_S P_chi}.
    Force: E[P psi_theta] has gradient 2 sum_n r(n) d psi_theta(n), r = (H - E) psi / ||psi||^2,
    because P is symmetric, idempotent and commutes with H, so P r = r.
    Metric: S-matrix of psi_c = P_chi psi_theta from --sr-samples draws of |psi_c|^2 (exact for
    mode G; for mode GS it is a positive-definite preconditioner, since P_S is not local; the
    force, hence every fixed point, is exact). Identical rule for every arm."""
    best_E, best_x = np.inf, ans.flat().copy()
    sg = proj.sg
    t0 = time.time()
    for it in range(args.sr_iters):
        x_before = ans.flat().copy()
        psi_t = ans.amplitudes(args.chunk)
        psi_c = proj.chi_part(psi_t)
        psi = proj.full(psi_c)
        E, Hp, nn = C.energy(H, psi)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x); break
        if E < best_E:
            best_E, best_x = E, x_before
        r = (Hp - E * psi) / nn
        f = ans.grad_weighted(psi_t * r, args.chunk)
        pc = psi_c ** 2 / (psi_c @ psi_c)
        idx = rng.choice(len(psi_c), size=args.sr_samples, p=pc)
        O = per_sample_O_chi(ans, sg, idx, psi_t, psi_c, args.o_chunk)
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        lam = args.sr_shift
        Kmat = Oc @ Oc.T + lam * np.eye(len(idx))
        dx = (f - Oc.T @ np.linalg.solve(Kmat, Oc @ f)) / lam
        nrm = np.linalg.norm(dx)
        step = args.sr_lr * dx
        if nrm * args.sr_lr > 1.0:
            step *= 1.0 / (nrm * args.sr_lr)
        ans.set_flat(ans.flat() - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}  "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    E = C.energy(H, proj.full(proj.chi_part(ans.amplitudes(args.chunk))))[0]
    if E > best_E:
        ans.set_flat(best_x)
    return ans


# ============================================================================ sampled arm
def sampled_diag(ans, sg, H, args, rng):
    """Genuine Monte Carlo from |psi_c|^2, psi_c = P_chi psi_theta (psi_theta if sg is None):
    the estimators a sampled VMC run of this ansatz would use. Returns
      sigma2  : exact Var_{|psi_c|^2}(E_loc),
      kurt    : exact E[(E_loc-E)^4]/sigma2^2 (heavy tails of the energy estimator),
      tail_El : max over the n_b*n_s samples of (E_loc-E)^2 / sigma2,
      tail_O  : max over samples of ||O_c - <O_c>||^2 / mean of the same (heavy tails of the
                log-derivatives, e.g. 1/F terms of FRD near factor zeros, 1/det of NNBF),
      eta     : rms over batches of ||f_hat - f_exact|| * sqrt(n_s) / sqrt(sigma2), where
                f_hat = (1/n_s) sum (E_loc - <E_loc>)(O_c - <O_c>) is the sampled SR force.
    """
    psi_t = ans.amplitudes(args.chunk)
    psi_c = sg.project(psi_t) if sg is not None else psi_t
    E, Hp, nn = C.energy(H, psi_c)
    p = psi_c ** 2 / nn
    nz = p > 0
    El = np.zeros_like(psi_c); El[nz] = Hp[nz] / psi_c[nz]
    dev = El - E
    sigma2 = float(np.sum(p * dev ** 2))
    kurt = float(np.sum(p * dev ** 4) / sigma2 ** 2) if sigma2 > 0 else float("nan")
    f_ex = ans.grad_weighted(psi_t * (Hp - E * psi_c) / nn, args.chunk)
    errs, qO, qE = [], [], []
    for _ in range(args.nb_diag):
        idx = rng.choice(len(psi_c), size=args.ns_diag, p=p)
        O = per_sample_O_chi(ans, sg, idx, psi_t, psi_c, args.o_chunk)
        Oc = O - O.mean(0)
        e = El[idx] - El[idx].mean()
        fh = (e[:, None] * Oc).mean(0)
        errs.append(float(np.linalg.norm(fh - f_ex)))
        qO.append(np.sum(Oc ** 2, 1)); qE.append(dev[idx] ** 2)
    qO, qE = np.concatenate(qO), np.concatenate(qE)
    # exact tail share of Var(E_loc): fraction carried by the most extreme TAIL_MASS of |psi_c|^2
    o = np.argsort(-np.abs(dev))
    cm = np.cumsum(p[o])
    sel = o[:int(np.searchsorted(cm, TAIL_MASS)) + 1]
    share_El = float(np.sum(p[sel] * dev[sel] ** 2) / sigma2) if sigma2 > 0 else float("nan")
    # sampled tail share of sum ||O_c||^2: fraction carried by the top TAIL_MASS of the samples
    qs = np.sort(qO)[::-1]
    share_O = float(qs[:max(1, int(round(TAIL_MASS * len(qs))))].sum() / qs.sum())
    return dict(sigma2=sigma2, kurt=kurt, share_El=share_El, share_O=share_O,
                tail_El=float(qE.max() / sigma2) if sigma2 > 0 else float("nan"),
                tail_O=float(qO.max() / qO.mean()),
                eta=float(np.sqrt(np.mean(np.square(errs))) * np.sqrt(args.ns_diag) / np.sqrt(sigma2))
                if sigma2 > 0 else float("nan"),
                n_samples=int(args.nb_diag * args.ns_diag))


# ============================================================================ arms
def frd_width(m, frame, Hc):
    """Largest FRD hidden width whose parameter count does not exceed NNBF x J at width Hc."""
    E = 2 * len(bond_list(m.T))
    nout = m.V * (m.nup + m.ndn)
    target = Hc * (2 * m.V + 1 + nout) + nout
    return max(1, (target - E) // (2 * m.V + 1 + E))


def make_arm(m, Pu, Pd, route, Hc, seed):
    if route == "ctrl":
        return C.Ansatz(m, Pu, Pd, "nnbf", width=Hc, jastrow=True, seed=seed + Hc)
    fr = {"frdK": "K", "frdX": "X"}[route]
    H = frd_width(m, fr, Hc)
    return FRD(m, Pu, Pd, fr, H, seed + H)


def width_of(a):
    return a.H if isinstance(a, C.Ansatz) else a.q["W1"].shape[1]


def r3_checks(seed):
    """Symmetry, spin and projected-SR checks on every boundary type used by R3_INSTANCES."""
    ok = True
    for (Lx, Ly, bcx, bcy, nu, nd) in [(3, 2, "obc", "obc", 2, 2), (3, 2, "pbc", "obc", 2, 2),
                                       (4, 2, "pbc", "obc", 2, 2)]:
        m = C.Hubbard(Lx, Ly, bcx, bcy, nu, nd, 1.0, -0.25, 8.0)
        W, Vv, deg = C.solve_ed(m, 3, 1e-8, seed)
        psi0 = Vv[:, 0]
        sg = SpaceGroup(m)
        rng = np.random.default_rng(seed)
        v, y = rng.standard_normal(m.D), rng.standard_normal(m.D)
        hv = np.linalg.norm(m.H @ v)
        comm = max(np.linalg.norm(sg.apply(m.H @ v, k) - m.H @ sg.apply(v, k)) / hv for k in range(sg.G))
        if deg != 1:
            log(f"check {Lx}x{Ly} {bcx}/{bcy}: [H,U_g] {comm:.1e} |G|={sg.G}; ground state degenerate "
                f"({deg}), projector checks skipped on this cluster")
            ok &= comm < 1e-12
            continue
        lam = sg.irrep_of(psi0)
        sg.set_irrep(lam)
        Pv = sg.project(v)
        e_idem = np.linalg.norm(sg.project(Pv) - Pv) / np.linalg.norm(Pv)
        e_sym = abs(v @ sg.project(y) - sg.project(v) @ y) / (np.linalg.norm(v) * np.linalg.norm(y))
        e_hc = np.linalg.norm(sg.project(m.H @ v) - m.H @ Pv) / hv
        ov = abs(psi0 @ sg.project(psi0))
        spn = Spin(m)
        S0 = float(round(0.5 * (np.sqrt(1 + 4 * spn.expect(psi0)) - 1)))
        Ps = spn.project(Pv, S0)
        e_sp = np.linalg.norm(spn.project(Ps, S0) - Ps) / np.linalg.norm(Ps)
        e_sg = np.linalg.norm(sg.project(Ps) - Ps) / np.linalg.norm(Ps)
        e_p0 = np.linalg.norm(spn.project(sg.project(psi0), S0) - psi0)
        log(f"check {Lx}x{Ly} {bcx}/{bcy} |G|={sg.G} chi={lam} S={S0:g}: [H,U_g] {comm:.1e} idem {e_idem:.1e} "
            f"sym {e_sym:.1e} [P,H] {e_hc:.1e} <psi0|P psi0> {ov:.12f} P_S idem {e_sp:.1e} "
            f"[P_S,P_chi] {e_sg:.1e} ||P_S P_chi psi0 - psi0|| {e_p0:.1e}")
        ok &= max(comm, e_idem, e_sym, e_hc, e_sp, e_sg, e_p0) < 1e-10 and abs(ov - 1) < 1e-10
    # projected gradients (modes G and GS) and projected O against finite differences
    m = C.Hubbard(3, 2, "pbc", "obc", 2, 2, 1.0, -0.3, 8.0)
    W, Vv, deg = C.solve_ed(m, 3, 1e-8, seed)
    sg = SpaceGroup(m); sg.set_irrep(sg.irrep_of(Vv[:, 0]))
    spn = Spin(m)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * spn.expect(Vv[:, 0])) - 1)))
    Pu, Pd = C.uhf_orbitals(m, seed)
    rng = np.random.default_rng(seed)
    for route in ("ctrl", "frdK", "frdX"):
        a = make_arm(m, Pu, Pd, route, 4, seed)
        if route == "ctrl":
            a.p["W2"] *= 100
        else:
            a.q["W2"] = 0.5 * np.random.default_rng(1).standard_normal(a.q["W2"].shape)
            a.q["b2"] += 0.3
            a._phi = None
        x0 = a.flat().copy()
        d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
        h = 1e-6
        for mode in ("G", "GS"):
            pr = Projector(mode, sg, spn, S0)
            Ef = lambda x: (a.set_flat(x), C.energy(m.H, pr.full(pr.chi_part(a.amplitudes(4096))))[0])[1]
            a.set_flat(x0)
            psi_t = a.amplitudes(4096); psi = pr.full(pr.chi_part(psi_t))
            E, Hp, nn = C.energy(m.H, psi)
            f = a.grad_weighted(psi_t * (Hp - E * psi) / nn, 4096)
            fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
            a.set_flat(x0)
            log(f"check projected gradient {route} mode {mode}: analytic {2 * f @ d:.10e}  finite-diff {fd:.10e}")
            ok &= abs(2 * f @ d - fd) < 1e-6 * max(1.0, abs(fd))
        psi_t = a.amplitudes(4096); psi_c = sg.project(psi_t)
        idx = rng.choice(m.D, size=6, p=psi_c ** 2 / (psi_c @ psi_c))
        O = per_sample_O_chi(a, sg, idx, psi_t, psi_c, 4)
        lp = lambda x: (a.set_flat(x), sg.project(a.amplitudes(4096))[idx])[1]
        fdO = (lp(x0 + h * d) - lp(x0 - h * d)) / (2 * h) / psi_c[idx]
        a.set_flat(x0)
        eO = np.max(np.abs(O @ d - fdO)) / max(1.0, np.max(np.abs(fdO)))
        log(f"check projected O_chi {route}: max rel err vs finite-diff {eO:.1e}")
        ok &= eO < 1e-5
    # parameter matching on the production instances (no ED needed)
    for k in (0, 1, 2):
        c = R3_INSTANCES[k]
        mm = C.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])
        Pu, Pd = np.zeros((mm.V, mm.nup)), np.zeros((mm.V, mm.ndn))
        pc = make_arm(mm, Pu, Pd, "ctrl", 32, 0).n_params()
        msg = []
        for route in ("frdK", "frdX"):
            a = make_arm(mm, Pu, Pd, route, 32, 0)
            msg.append(f"{route} H={width_of(a)} P={a.n_params()}")
            ok &= a.n_params() <= pc and pc - a.n_params() <= 2 * mm.V + 1 + a.edges.E
        log(f"check parameter match instance {k}: ctrl H=32 P={pc}; " + "; ".join(msg))
    log(f"ALL R3 CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ instance setup
def in_sector_levels(m, proj, k, seed):
    """Lowest k levels of H inside range(P_S P_chi) (Lanczos on P H P + c (1 - P))."""
    from scipy.sparse.linalg import LinearOperator
    c = 50.0 * (np.abs(m.H.diagonal()).max() + 10.0)
    P = lambda v: proj.full(proj.chi_part(v))

    def mv(v):
        v = np.ravel(v)
        pv = P(v)
        return P(m.H @ pv) + c * (v - pv)
    A = LinearOperator((m.D, m.D), matvec=mv, dtype=float)
    v0 = P(np.random.default_rng(seed).standard_normal(m.D))
    w = C.eigsh(A, k=k, which="SA", v0=v0, tol=1e-9, return_eigenvectors=False)
    return np.sort(w)


def setup(inst, seed, sector_k=2):
    c = R3_INSTANCES[inst]
    m = C.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])
    m.pair_setup(1.5)
    W, Vv, deg = C.solve_ed(m, 3, 1e-8, seed)
    if deg != 1:
        raise RuntimeError(f"instance {inst}: degenerate ground state ({deg}); projection undefined")
    sg = SpaceGroup(m)
    lam = sg.irrep_of(Vv[:, 0]); sg.set_irrep(lam)
    spn = Spin(m)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * spn.expect(Vv[:, 0])) - 1)))
    proj = Projector("GS", sg, spn, S0)
    lv = in_sector_levels(m, proj, sector_k, seed)
    hc = dict(levels3=W[:3].tolist(), gap_fixedN=float(W[1] - W[0]), irrep=lam, S_psi0=S0, G=sg.G,
              in_sector_levels=lv.tolist(), in_sector_gap=float(lv[1] - lv[0]),
              in_sector_gap_per_site=float((lv[1] - lv[0]) / m.V),
              note="sign half only: no same-sector competitor at the Hard-Core scale is claimed")
    return m, W, Vv, deg, sg, spn, S0, hc


# ============================================================================ one task
def task_spec(t):
    """Array index t in 0..107 -> (instance, seed, route, projected?, shift)."""
    inst, r = divmod(t, 36)
    seed = SEEDS[r // 12]
    arm = ARMS[(r % 12) // 2]
    shift = SHIFTS[r % 2]
    return inst, seed, arm, shift


def run_task(args):
    inst, seed, arm, shift = args.instance, args.seed, args.arm, args.sr_shift
    route, pmode = arm.split("_")
    tag = f"inst{inst}_{arm}_s{seed}_sh{shift:g}"
    out = os.path.join(args.out_dir, tag + ".json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; skipping (use --overwrite 1)"); return 0
    os.makedirs(args.out_dir, exist_ok=True)
    m, W, Vv, deg, sg, spn, S0, hc = setup(inst, seed)
    log(f"instance {inst} {R3_INSTANCES[inst]} D={m.D} |G|={sg.G} chi={hc['irrep']} S={S0:g} "
        f"E0={W[0]:.8f} gap(fixed N)={hc['gap_fixedN']:.4f} in-sector gap={hc['in_sector_gap']:.4f}")
    rng = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1000003)
    rng_d = np.random.default_rng(seed + 2000003)
    S0m = sp.diags(np.where(Vv[:, 0] >= 0, 1.0, -1.0))
    Ec, _, _ = C.sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                      [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))],
                                      args.lbfgs_iters)
    calib = float(Ec - W[0])
    log(f"calibration E_sign(s_0) - E0 = {calib:.2e}")
    Pu, Pd = C.uhf_orbitals(m, seed)
    a = make_arm(m, Pu, Pd, route, args.ctrl_width, seed)
    log(f"arm {arm}: width {width_of(a)}, n_params {a.n_params()}, SR shift {shift:g}")
    proj = Projector("GS" if pmode == "PS" else "none", sg, spn, S0)
    t0 = time.time()
    if pmode == "PS":
        optimize_proj(a, m.H, proj, args, rng, arm)
    else:
        C.optimize_sr(a, m.H, args, rng, arm)
    ttrain = time.time() - t0
    psi_t = a.amplitudes(args.chunk)
    if pmode == "PS":
        states = [("PS", proj.full(sg.project(psi_t)))]
        if route != "ctrl":
            states.append(("abl", proj.full(sg.project(a.base.amplitudes(args.chunk)))))
    else:
        pc = sg.project(psi_t)
        states = [("U", psi_t), ("UG", pc), ("UGS", Projector("GS", sg, spn, S0).full(pc))]
    res = dict(params={k: v for k, v in vars(args).items()}, instance=inst, arm=arm, route=route,
               pmode=pmode, seed=seed, shift=shift, width=width_of(a), n_params=a.n_params(),
               ed_levels=W.tolist(), hard_core=hc, calib_err=calib, train_seconds=ttrain, budgets={})
    for lab, psi in states:
        nrm = np.linalg.norm(psi)
        if not np.isfinite(nrm) or nrm < 1e-10 * np.linalg.norm(psi_t):
            res["budgets"][lab] = None; log(f"{lab}: projection vanishes"); continue
        b = C.budget(m, psi, W, Vv, deg, args, rng_b)
        b["S2"] = spn.expect(psi)
        b["ov0"] = float((Vv[:, 0] @ psi) ** 2 / (psi @ psi))
        if route != "ctrl" and lab != "abl":
            b["flip_weight"] = a.flip_weight(psi)
        for k in ("P0", "Ptheta", "Psign"):
            b.pop(k, None)
        res["budgets"][lab] = b
        log(f"{arm} {lab:>4s}: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} f_E={b['f_E_up']:.3f} "
            f"w_sign={b['w_sign']:.3e} S2={b['S2']:.3f} spread={b['spread']:.1e}"
            + (f" flip_weight={b['flip_weight']:.4f}" if "flip_weight" in b else ""))
    res["E_select"] = res["budgets"]["PS" if pmode == "PS" else "U"]["E"]
    res["sampled"] = sampled_diag(a, sg if pmode == "PS" else None, m.H, args, rng_d)
    log(f"{arm} sampled arm: " + " ".join(f"{k}={v:.3e}" for k, v in res["sampled"].items()))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    log(f"wrote {out}")
    return 0


# ============================================================================ aggregate
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def aggregate(d, final):
    d = d if os.path.isabs(d) else os.path.join(HERE, d)
    best = {}
    for f in sorted(glob.glob(os.path.join(d, "inst*_*_s*_sh*.json"))):
        r = json.load(open(f))
        key = (r["instance"], r["arm"], r["seed"])
        if key not in best or r["E_select"] < best[key]["E_select"]:
            best[key] = r  # SR shift chosen by variational energy only
    if not best:
        print("no results"); return 1
    print("P1.2 r3 aggregate: SIGN HALF ONLY (no same-sector competitor at the Hard-Core scale is claimed)")
    print(f"pre-committed: informative iff med dE_sign(ctrl_PS) >= {TAU_INF}; per route X on informative "
          f"instances: FAIL if rho_sign >= {RHO_FAIL} or rho_tot > {TOT_FAIL} or a sampled tail share "
          f"(E_loc or O) of X is > {SHARE_MAX} and > {SHARE_FACTOR}x the control's; PASS if guards & rho_sign <= {RHO_PASS} & rho_tot <= {TOT_PASS} "
          f"& paired sign wins >= 2/3 of seeds & rho_vote <= {VOTE_MAX} & sampling ok; else AMBIGUOUS"
          + ("  [FINAL mode: AMBIGUOUS -> FAIL]" if final else ""))
    verdict = {}
    informative = {}
    per = {}
    for inst in sorted({k[0] for k in best}):
        R = {(a, s): best[(i, a, s)] for (i, a, s) in best if i == inst}
        seeds = sorted({s for (_, s) in R})
        full = [s for s in seeds if all((a, s) in R for a in ("ctrl_PS", "frdK_PS", "frdX_PS"))]
        hc = next(iter(R.values()))["hard_core"]
        print(f"\ninstance {inst} {R3_INSTANCES.get(inst)}  |G|={hc['G']} chi={hc['irrep']} S={hc['S_psi0']:g} "
              f"in-sector gap={hc['in_sector_gap']:.4f} ({hc['in_sector_gap_per_site']:.4f}/site)  seeds={full}")
        g = lambda arm, lab, key: [R[(arm, s)]["budgets"][lab].get(key) if (arm, s) in R and
                                   R[(arm, s)]["budgets"].get(lab) else None for s in full]
        for arm in ARMS:
            for lab in (("PS", "abl") if arm.endswith("PS") else ("U", "UG", "UGS")):
                v = g(arm, lab, "dE_sign")
                if all(x is None for x in v):
                    continue
                vv = [x for x in v if x is not None]
                print(f"  {arm:>8s} {lab:>4s}: dE_sign med {_med(v):.5f} [min {min(vv):.5f}, max {max(vv):.5f}]  "
                      f"dE_tot med {_med(g(arm, lab, 'dE_tot')):.5f}  w_sign {_med(g(arm, lab, 'w_sign')):.3e}  "
                      f"flip_weight {_med(g(arm, lab, 'flip_weight')):.4f}  "
                      f"P={R[(arm, full[0])]['n_params'] if full and (arm, full[0]) in R else '-'}")
        if len(full) < MIN_SEEDS:
            print(f"  INCOMPLETE ({len(full)} seeds with all projected arms) -> not counted")
            informative[inst] = None
            continue
        cs = _med(g("ctrl_PS", "PS", "dE_sign"))
        informative[inst] = cs >= TAU_INF
        print(f"  premise: med dE_sign(ctrl_PS) = {cs:.5f} -> {'INFORMATIVE' if informative[inst] else 'NOT informative (projected control leaves no sign error to repair)'}")
        for route in ("frdK", "frdX"):
            X = f"{route}_PS"
            xs, xt = g(X, "PS", "dE_sign"), g(X, "PS", "dE_tot")
            cs_, ct_ = g("ctrl_PS", "PS", "dE_sign"), g("ctrl_PS", "PS", "dE_tot")
            rs = _med(xs) / _med(cs_); rt = _med(xt) / _med(ct_)
            wins = sum(1 for a_, b_ in zip(xs, cs_) if a_ is not None and b_ is not None and a_ < b_)
            rv = _med([a_ / b_ for a_, b_ in zip(xs, g(X, "abl", "dE_sign")) if a_ is not None and b_])
            sk = lambda arm, k: _med([R[(arm, s)]["sampled"][k] for s in full])
            kr = sk(X, "kurt") / sk("ctrl_PS", "kurt")
            er = sk(X, "eta") / sk("ctrl_PS", "eta")
            shE, shEc = sk(X, "share_El"), sk("ctrl_PS", "share_El")
            shO, shOc = sk(X, "share_O"), sk("ctrl_PS", "share_O")
            heavy = lambda x, c: x > SHARE_MAX and x > SHARE_FACTOR * c
            guard = []
            for s in full:
                bx, bc = R[(X, s)]["budgets"]["PS"], R[("ctrl_PS", s)]["budgets"]["PS"]
                cal = max(R[(X, s)]["calib_err"], R[("ctrl_PS", s)]["calib_err"])
                if cal > CALIB_TOL * min(bx["dE_tot"], bc["dE_tot"]):
                    guard.append(f"seed {s} calib {cal:.1e}")
                if not (bx["spread"] < SPREAD_TOL and bc["spread"] < SPREAD_TOL):
                    guard.append(f"seed {s} spread {bx['spread']:.2e}/{bc['spread']:.2e}")
            samp_ok = not heavy(shE, shEc) and not heavy(shO, shOc)
            need_w = int(np.ceil(2 * len(full) / 3))
            if rs >= RHO_FAIL or rt > TOT_FAIL or not samp_ok:
                v = "FAIL"
            elif not guard and rs <= RHO_PASS and rt <= TOT_PASS and wins >= need_w and rv <= VOTE_MAX:
                v = "PASS"
            else:
                v = "AMBIGUOUS"
            # diagnostics (not in the verdict): symmetry restoration vs nodal repair
            u = {lab: _med(g(f"{route}_U", lab, "dE_sign")) / _med(g("ctrl_U", lab, "dE_sign"))
                 for lab in ("U", "UG", "UGS")}
            print(f"  [{route}] rho_sign={rs:.3f} rho_tot={rt:.3f} paired wins {wins}/{len(full)} (need {need_w}) "
                  f"rho_vote={rv:.3f} tail share E_loc {shE:.3f} (ctrl {shEc:.3f}) O {shO:.3f} (ctrl {shOc:.3f}) "
                  f"[kurt ratio {kr:.2f}, eta ratio {er:.2f}] "
                  f"guards={'OK' if not guard else '; '.join(guard)} -> {v}"
                  + ("" if informative[inst] else " (instance not informative: not counted)"))
            print(f"           diag unprojected arms: rho_sign U={u['U']:.3f}  post-hoc P_chi={u['UG']:.3f}  "
                  f"post-hoc P_S P_chi={u['UGS']:.3f}")
            per.setdefault(route, {})[inst] = v
    for route in ("frdK", "frdX"):
        vs = [per.get(route, {}).get(i) for i in informative if informative[i]]
        if len(vs) < MIN_INFORMATIVE:
            gv = (f"NO-TARGET ({len(vs)} informative instance(s) < {MIN_INFORMATIVE}: the projected control "
                  f"leaves < {TAU_INF} t sign error, so there is nothing for a sign-repair layer to fix at ED size)")
        elif any(v == "FAIL" for v in vs):
            gv = "FAIL"
        elif all(v == "PASS" for v in vs):
            gv = "PASS"
        else:
            gv = "FAIL" if final else "AMBIGUOUS (action: add seeds 4567 and 5678, rerun --aggregate with --final 1)"
        verdict[route] = gv
        print(f"ROUTE {route}: {gv}")
    return 0


# ============================================================================ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 round-3 projected FRD vs projected NNBF x J (see docstring)")
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--instance", type=int, default=0, choices=sorted(R3_INSTANCES))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arm", choices=list(ARMS), default="ctrl_PS")
    ap.add_argument("--sr-shift", type=float, default=1e-2)
    ap.add_argument("--ctrl-width", type=int, default=32)
    ap.add_argument("--sr-iters", type=int, default=1500)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--o-chunk", type=int, default=64)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--nb-diag", type=int, default=8)
    ap.add_argument("--ns-diag", type=int, default=1000)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--mode", choices=["run", "checks", "smoke", "probe", "timing"], default="run")
    ap.add_argument("--aggregate", type=str, default="")
    ap.add_argument("--final", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task_id >= 0:
        if a.task_id >= 3 * 36:
            ap.error("--task-id must be in 0..107")
        a.instance, a.seed, a.arm, a.sr_shift = task_spec(a.task_id)
    return a


def smoke(args):
    """Checks, then every arm on the 3x2 smoke cluster (instance 3) for 40 SR steps, written to
    results_smoke/, then the aggregate over three seeds."""
    d = os.path.join(HERE, "results_smoke")
    for seed in SEEDS:
        for arm in ARMS:
            sa = parse_args(["--instance", "3", "--seed", str(seed), "--arm", arm, "--ctrl-width", "4",
                             "--sr-iters", "40", "--sr-samples", "150", "--lbfgs-iters", "400",
                             "--fn-iters", "3", "--nb-diag", "2", "--ns-diag", "200", "--o-chunk", "16",
                             "--out-dir", d, "--overwrite", "1"])
            run_task(sa)
    return aggregate(d, False)


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        return aggregate(args.aggregate, bool(args.final))
    if args.mode == "probe":
        for k in (0, 1, 2):
            m, W, Vv, deg, sg, spn, S0, hc = setup(k, args.seed, sector_k=2)
            print(f"PROBE instance {k} {R3_INSTANCES[k]} D={m.D}: {json.dumps(hc)}", flush=True)
        return 0
    if args.mode == "timing":
        m, W, Vv, deg, sg, spn, S0, hc = setup(args.instance, args.seed)
        Pu, Pd = C.uhf_orbitals(m, args.seed)
        rng = np.random.default_rng(args.seed)
        for arm in ARMS:
            route, pm = arm.split("_")
            a = make_arm(m, Pu, Pd, route, args.ctrl_width, args.seed)
            a2 = argparse.Namespace(**vars(args)); a2.sr_iters = 2
            t0 = time.time()
            if pm == "PS":
                optimize_proj(a, m.H, Projector("GS", sg, spn, S0), a2, rng, "timing_" + arm)
            else:
                C.optimize_sr(a, m.H, a2, rng, "timing_" + arm)
            log(f"TIMING instance {args.instance} {arm}: {(time.time() - t0) / 2:.2f} s/step "
                f"(width {width_of(a)}, P={a.n_params()})")
        return 0
    ok = r2_checks(args.seed) and r3_checks(args.seed)
    if not ok:
        log("checks failed; aborting"); return 2
    if args.mode == "checks":
        return 0
    if args.mode == "smoke":
        return smoke(args)
    return run_task(args)


if __name__ == "__main__":
    sys.exit(main())
