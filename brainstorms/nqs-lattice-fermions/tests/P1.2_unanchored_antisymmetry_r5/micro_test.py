#!/usr/bin/env python3
"""
P1.2 round-5 decisive test: NNBF x FRD versus NNBF, both symmetry-projected, at matched parameters,
on the shared 4x4 OBC t-t'-J instance (6+6 electrons, t' = -0.25, J = 0.4)
=====================================================================================================

Question
--------
Does the factorized-ratio dressing (FRD), placed ON TOP OF a neural-network backflow (NNBF) base,
remove nodal (sign) error that a symmetry-projected NNBF with the same total parameter count keeps?
The round-3 pilot (job 410216) found no target on 4x3 Hubbard clusters (projected-control
dE_sign <= 1e-5 t). The 4x4 OBC t-J instance below has projected sector dimension 30220 against
about 1.7k parameters, and there projected NNBF width 6 keeps dE_sign = 0.0125 t (P2.1 pilot,
seed 11). THIS TEST COVERS THE SIGN HALF OF THE HARD CORE ONLY (no same-sector competitor).

Model and instance (copied code: tj_core.py, p11_core.py, p21_harness.py are verbatim copies of
tests/P2.1_subcubic_amplitudes_r3/{tj_core.py, p11_core.py, micro_test.py})
  H = -sum_{ij,s} T_ij P c^dag_is c_js P + J sum_<ij> (S_i.S_j - n_i n_j / 4), t = 1, t' = -0.25 on
  diagonals, J = 0.4 on NN bonds, no double occupancy.
  instance 0 : 4x4 open x open, N_up = N_dn = 6, D = 1681680, |G| = 8 (C4v point group),
               psi_0 non-degenerate singlet (Mx = My = +1, D = -1), dim range(P_S P_chi) = 30220.
  instance 8 : 4x2 open x open, N_up = N_dn = 3, D = 560 (smoke pipeline only).
  instance 9 : 3x2 periodic x open, N_up = N_dn = 2, D = 90 (correctness checks only).

Ansatz (FRD on an NNBF base; class FRD below)
    psi(n) = Phi(n) * prod_e [1 + alpha_e(n) g(R_e(n))],   R_e(n) = c_e(n) Phi(m_e(n)) / Phi(n),
    g(R) = R / sqrt(1 + R^2),   alpha(n) = tanh(x W1a + b1a) W2a + b2a,   x = 2 n - 1 (2V inputs).
  Phi  = NNBF (tj_core.LRAnsatz mode "full", Luo & Clark) of hidden width 4, UHF start (seed 1234).
  edges e, frame K (frdK): every hopping bond (NN and NNN, T_ij != 0) x spin; in t-J a spin-s
         electron hops across the bond into the empty site (a hole hop), at most one direction
         is possible; c_e = <n|H|m_e> = -T_ij * (Jordan-Wigner sign).
  edges e, frame X (frdX): every NN bond (a, b) x orientation: up at b, down at a -> swapped;
         c_e = <n|H|m_e> / (J/2) = -(JW sign of up hop)(JW sign of down hop).
  inactive edge -> factor 1. Phi(m_e) is the NNBF amplitude of the neighbour (its own backflow).
  Parameters: NNBF-4 base (1284) + alpha net of width w, w = largest with total <= n_params(NNBF-6)
  = 1734 (frdK: w = 3, 1719; frdX: w = 4, 1656; printed by the checks). alpha starts at ~1e-3, so
  every FRD run starts from its NNBF-4 base with the same seed.

Arms (all trained WITH space-group projection, psi_P = P_chi psi_theta, exact-sum SR; P2.1 code)
  nnbf6  NNBF width 6 (1734 params)    DECISIVE CONTROL (identical to P2.1's projected "nnbf" arm)
  nnbf4  NNBF width 4 (1284 params)    separately trained base (honest ablation, critic item d)
  frdK   NNBF-4 x FRD-K (<= 1734)      route 5.1
  frdX   NNBF-4 x FRD-X (<= 1734)      route 5.2
  each at seeds 11, 22, 33 and SR shifts 1e-3 and 1e-2; per (arm, seed) the shift with the lower
  final variational energy is kept (energy only, no reference to psi_0).
  Sampled-force arms (critic item b; seed 11, shift 1e-2): nnbf6_mc, frdK_mc, frdX_mc. SR whose
  force AND metric come only from genuine Metropolis samples of |psi_P|^2 (100 persistent chains,
  symmetric bond-swap proposals, N_e proposals between measurements, --sr-samples per step):
  f = <(E_loc - <E_loc>)(O_P - <O_P>)>. Amplitudes, E_loc = (H psi_P)/psi_P and O_P are exact
  functions of the sampled configurations (read from the current vector; per-amplitude agreement
  is checked). The exact energy is recorded every step for diagnostics only; the final state is
  the LAST iterate (no exact-energy selection). Recorded: non-finite flag, divergence (final exact
  E above the step-0 energy), jump rate (fraction of steps whose exact energy rises by more than
  JUMP = 0.01 t), and the stall ratio dE_tot(arm_mc) / dE_tot(same arm, exact force, seed 11,
  shift 1e-2).
SR settings (P2.1): --sr-iters 1200, step 0.05, --sr-samples 2000, trust region |step| <= 1.

Per-task outputs: budgets (p11_core.budget, P1.1 definitions) of "P" (= psi_P) and "P_S"
(post-hoc spin projection, reported only): dE_tot, dE_sign = E_sign_up - E_0, w_sign, spread,
f_E, S2, ov0; sampled diagnostics (p21_harness.sampled_arm: exact Var(E_loc), exact kurtosis
kurt_ex of E_loc under |psi_P|^2, tail share, Metropolis mean/stderr, per-amplitude check);
FRD only: flip_weight (|psi_P|^2 weight where an odd number of factors is negative) and E_alpha0
(the co-adapted alpha = 0 energy; diagnostic only, NOT the ablation).

Pre-committed decision rule (aggregate(); constants below), per route X in {frdK, frdX}
  PREMISE: the test counts only if med_s dE_sign(nnbf6) >= PREMISE = 0.005 t (3 seeds). Else the
           route verdict is NO-TARGET (neither PASS nor FAIL).
  rho_sign = med dE_sign(X) / med dE_sign(nnbf6); rho_tot likewise for dE_tot;
  rho_abl  = med dE_sign(X) / med dE_sign(nnbf4);
  spread-aware margin: DISJOINT iff max_seed dE_sign(X) < min_seed dE_sign(nnbf6);
  kurt ratio kappa = med kurt_ex(X) / med kurt_ex(nnbf6);
  sampled-force arm: stall_X, stall_c (= nnbf6), jump rates jr_X, jr_c;
     MC_OK   iff X_mc finite, not diverged, stall_X <= 2 stall_c and jr_X <= max(2 jr_c, 0.02);
     MC_FAIL iff X_mc non-finite or diverged, or stall_X > 4 stall_c, or jr_X > max(5 jr_c, 0.05).
  guards: calib_err <= 1e-3 min dE_tot and budget spread < 0.05 for X, nnbf6, nnbf4; 3 seeds each.
  FAIL      if rho_sign >= 0.9, or rho_tot > 1.1, or kappa > 10, or MC_FAIL.
  PASS      if guards, rho_sign <= 0.7, DISJOINT, rho_tot <= 1.0, rho_abl <= 0.8, kappa <= 10, MC_OK.
  AMBIGUOUS otherwise. Pre-committed action: seeds 44 and 55 for nnbf6 and X (both shifts), then
           --final 1: PASS iff rho_sign <= 0.7 on 5-seed medians, >= 4 of 5 seeds with
           dE_sign(X) < med dE_sign(nnbf6), rho_tot <= 1.0, kappa <= 10 and MC_OK; else FAIL.
  Reported only: post-hoc P_S ratios, w_sign, flip_weight, E_alpha0, ov0, the cost table.

Cost measurement (critic item e; --mode cost; single thread)
  Per-sample local energy computed the honest way: every amplitude is a full NNBF forward pass on
  its occupation (no carried tables, for both families), and FRD's amplitude at a configuration
  needs the base amplitude at the configuration and at every active edge neighbour. L x L periodic
  t-J lattices (L = 4..7, doping ~1/8), random weights. Reported per size: N_e, number of base
  forward evaluations per E_loc for NNBF and for FRD-K / FRD-X, wall time per E_loc, the ratio
  T_FRD / T_NNBF, and log-log slopes in N_e. Reported, not gated (this test decides accuracy).

Usage
-----
  python3 micro_test.py --mode train --task ID [options]  one Slurm array task, ID in 0..26
  python3 micro_test.py --mode aggregate [--results DIR] [--final 0|1]
  python3 micro_test.py --mode cost [--sizes 4,5,6,7] [--nconf 3]
  python3 micro_test.py --mode checks                     correctness checks only (seconds)
  python3 micro_test.py --mode smoke                      checks + tiny pipeline + aggregate + cost
  python3 micro_test.py --mode time --instance 0 [--arm A] one SR step per arm (cost estimate)

Options
-------
  --mode {train,aggregate,cost,checks,smoke,time}   what to run (default smoke).
  --task INT       0..26. 0..23: arm = [nnbf6, nnbf4, frdK, frdX][ID // 6],
                   seed = [11, 22, 33][(ID % 6) // 2], shift = [1e-3, 1e-2][ID % 2] (exact force);
                   24, 25, 26: arm = nnbf6_mc, frdK_mc, frdX_mc, seed 11, shift 1e-2 (sampled
                   force). Explicit --arm/--seed/--sr-shift override the decoded values.
  --instance INT   0 (production), 8 (smoke pipeline), 9 (checks). Default 0.
  --arm NAME       nnbf6, nnbf4, frdK, frdX, nnbf6_mc, frdK_mc, frdX_mc.
  --seed INT       network-initialisation / SR-sampling seed (default 11).
  --sr-shift FLOAT SR diagonal shift (default 1e-3).
  --sr-iters INT (1200)  --sr-lr FLOAT (0.05)  --sr-samples INT (2000)
  --base-width INT width of the FRD base and of nnbf4 (default 4).
  --ctrl-width INT width of the decisive control nnbf6 (default 6); FRD's alpha width is the
                   largest with n_params <= n_params(NNBF ctrl-width).
  --chunk INT      configurations per full-basis chunk (default 32768).
  --o-chunk INT    samples per chunk of projected log-derivatives (default 4; x |G| images x
                   (1 + active edges) base rows for FRD).
  --lbfgs-iters INT (4000)  --fn-iters INT (10)  --fn-floor FLOAT (1e-3)  --r-min FLOAT (1.5):
                   p11_core.budget settings, as P1.1 / P2.1.
  --mc-samples INT measurements of the sampled diagnostics (default 20000; 0 disables).
  --mc-burn INT    burn-in measurements of the sampled diagnostics (default 2000).
  --chains INT     persistent Metropolis chains of the sampled-force arms (default 100).
  --sizes LIST     --mode cost: comma-separated L (default 4,5,6,7; V = L^2 <= 62).
  --nconf INT      --mode cost: random configurations per size (default 3).
  --results DIR    results directory (default results/ next to this script).
  --final {0,1}    --mode aggregate: 1 applies the 5-seed final rule (after the AMBIGUOUS action).
  --overwrite {0,1} recompute an existing result JSON (default 0).

Outputs
-------
  train: results/inst{I}_{arm}_s{seed}_sh{shift}.json (+ results/params/*.npz)
  cost:  results/cost.json        aggregate: printed table + results/aggregate.json
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
import p21_harness as H21  # noqa: E402  (verbatim copy of the P2.1 round-3 harness)
from p11_core import budget, build_lattice, energy, sign_constrained_min, solve_ed  # noqa: E402
from tj_core import TJ, LRAnsatz, between_mask, connections, eloc_brute, popcount  # noqa: E402

H21.INSTANCES[8] = dict(Lx=4, Ly=2, bcx="obc", bcy="obc", nup=3, ndn=3, tp=-0.25, J=0.4)
ARMS = ["nnbf6", "nnbf4", "frdK", "frdX"]
MC_ARMS = ["nnbf6_mc", "frdK_mc", "frdX_mc"]
SEEDS = [11, 22, 33]
SHIFTS = [1e-3, 1e-2]
MC_SEED, MC_SHIFT = 11, 1e-2
PREMISE = 0.005
RHO_PASS, RHO_FAIL, TOT_PASS, TOT_FAIL, ABL_PASS, KAPPA_MAX = 0.7, 0.9, 1.0, 1.1, 0.8, 10.0
STALL_OK, STALL_BAD, JR_OK_F, JR_OK_A, JR_BAD_F, JR_BAD_A = 2.0, 4.0, 2.0, 0.02, 5.0, 0.05
JUMP = 0.01
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05
log = H21.log


# ============================================================================ edges
def edge_list(T, nn, frame):
    V = T.shape[0]
    if frame == "K":
        return [("K", i, j, s) for i in range(V) for j in range(i + 1, V) if T[i, j] != 0.0 for s in (0, 1)]
    return [("X", a, b) for (i, j) in nn for (a, b) in ((i, j), (j, i))]


def edge_moves_bits(bu, bd, edges, T, J):
    """For bit patterns bu, bd (B,): neighbour patterns nbu, nbd (B, E), active mask and the exact
    Hamiltonian element hcoef = <n|H|m_e> (kinetic: -T_ij * JW sign; exchange: -J/2 * JW signs)."""
    one = np.int64(1)
    B, E = len(bu), len(edges)
    nbu = np.repeat(bu[:, None], E, 1)
    nbd = np.repeat(bd[:, None], E, 1)
    act = np.zeros((B, E), bool)
    hc = np.zeros((B, E))
    emp = ~(bu | bd)
    for e, ed in enumerate(edges):
        if ed[0] == "K":
            _, i, j, s = ed
            b = bu if s == 0 else bd
            a = ((((b >> j) & 1) & ((emp >> i) & 1)) | (((b >> i) & 1) & ((emp >> j) & 1))).astype(bool)
            nb = b ^ (one << i) ^ (one << j)
            sg = 1.0 - 2.0 * (popcount(b & between_mask(i, j)) & 1)
            if s == 0:
                nbu[:, e] = np.where(a, nb, bu)
            else:
                nbd[:, e] = np.where(a, nb, bd)
            hc[:, e] = np.where(a, -T[i, j] * sg, 0.0)
        else:
            _, a_, b_ = ed
            a = (((bu >> b_) & 1) & ((bd >> a_) & 1)).astype(bool)
            bm = between_mask(a_, b_)
            sg = (1.0 - 2.0 * (popcount(bu & bm) & 1)) * (1.0 - 2.0 * (popcount(bd & bm) & 1))
            nbu[:, e] = np.where(a, bu ^ (one << b_) ^ (one << a_), bu)
            nbd[:, e] = np.where(a, bd ^ (one << a_) ^ (one << b_), bd)
            hc[:, e] = np.where(a, -0.5 * J * sg, 0.0)
        act[:, e] = a
    return nbu, nbd, act, hc


def bits_to_occ(bits, V, N):
    occ = ((bits[:, None] >> np.arange(V, dtype=np.int64)) & 1).astype(np.int64)
    rows = np.nonzero(occ)[1].reshape(len(bits), N) if N > 0 else np.zeros((len(bits), 0), np.int64)
    return occ, rows


def occ_to_bits(o):
    return (np.asarray(o, dtype=np.int64) << np.arange(o.shape[-1], dtype=np.int64)).sum(-1)


# ============================================================================ FRD on an NNBF base
class FRD:
    """psi = Phi prod_e [1 + alpha_e g(R_e)] with Phi = base (LRAnsatz); see module docstring."""

    def __init__(self, m, base, frame, width, seed):
        self.m, self.base, self.frame = m, base, frame
        self.edges = edge_list(m.T, m.nn, frame)
        self.E = len(self.edges)
        self.scale = 1.0 if frame == "K" else 0.5 * m.J
        V = m.V
        rng = np.random.default_rng(seed + 7919)
        self.q = {"W1a": rng.standard_normal((2 * V, width)) / np.sqrt(2 * V), "b1a": np.zeros(width),
                  "W2a": 1e-3 * rng.standard_normal((width, self.E)), "b2a": np.zeros(self.E)}
        self.qkeys = list(self.q)
        self._phi = None
        self._tab = None

    # ---------------------------------------------------------------- parameters
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
            self.q[k] = x[o:o + n].reshape(self.q[k].shape)
            o += n
        self._phi = None

    # ---------------------------------------------------------------- pieces
    def tables(self, chunk=131072):
        """Full-basis edge tables: Mi (D, E) int32 neighbour index (-1 inactive), C (D, E) float32."""
        if self._tab is None:
            m = self.m
            Mi = -np.ones((m.D, self.E), np.int32)
            C = np.zeros((m.D, self.E), np.float32)
            for s in range(0, m.D, chunk):
                sl = slice(s, min(s + chunk, m.D))
                nbu, nbd, act, hc = edge_moves_bits(m.bu[sl], m.bd[sl], self.edges, m.T, m.J)
                k = m.index(nbu[act], nbd[act])
                assert (k >= 0).all()
                blk = Mi[sl]; blk[act] = k; Mi[sl] = blk
                C[sl] = hc / self.scale
            self._tab = (Mi, C)
        return self._tab

    def phi(self, chunk):
        if self._phi is None:
            self._phi = self.base.amplitudes(chunk)
        return self._phi

    def _alpha(self, x):
        h = np.tanh(x @ self.q["W1a"] + self.q["b1a"])
        return h, h @ self.q["W2a"] + self.q["b2a"]

    @staticmethod
    def _gfun(v, u):
        s = np.where(v >= 0, 1.0, -1.0)
        den2 = v * v + u * u
        den2 = np.where(den2 > 0, den2, 1.0)
        den = np.sqrt(den2)
        g = u * s / den
        dg_du = s * v * v / (den2 * den)
        q = g * v * v / den2
        return g, dg_du, q

    def _x_idx(self, idx):
        iu, idd = self.m.iu_id()
        return np.concatenate([self.m.occ_u[iu[idx]], self.m.occ_d[idd[idx]]], 1).astype(float) * 2 - 1

    def _fac_idx(self, idx, phi):
        Mi, C = self.tables()
        Mi, C = Mi[idx], C[idx].astype(float)
        act = Mi >= 0
        v = phi[idx][:, None]
        u = np.where(act, C * phi[np.maximum(Mi, 0)], 0.0)
        g, dg_du, q = self._gfun(v, u)
        g = np.where(act, g, 0.0)
        x = self._x_idx(idx)
        h, al = self._alpha(x)
        F = 1.0 + al * g
        return dict(Mi=Mi, C=C, act=act, g=g, dg_du=dg_du, q=np.where(act, q, 0.0), x=x, h=h, al=al, F=F,
                    v=phi[idx])

    @staticmethod
    def _safe(F):
        return np.where(np.abs(F) > 1e-300, F, 1e-300)

    # ---------------------------------------------------------------- full-basis evaluation
    def amplitudes(self, chunk):
        phi = self.phi(chunk)
        out = np.empty(self.m.D)
        for s in range(0, self.m.D, chunk):
            idx = np.arange(s, min(s + chunk, self.m.D))
            f = self._fac_idx(idx, phi)
            out[idx] = f["v"] * np.prod(f["F"], 1)
        return out

    def _alpha_backprop(self, dal, f, per_sample):
        h, x = f["h"], f["x"]
        dpre = (dal @ self.q["W2a"].T) * (1 - h ** 2)
        if per_sample:
            B = len(dal)
            g = {"W1a": (x[:, :, None] * dpre[:, None, :]).reshape(B, -1), "b1a": dpre,
                 "W2a": (h[:, :, None] * dal[:, None, :]).reshape(B, -1), "b2a": dal}
            return np.concatenate([g[k] for k in self.qkeys], 1)
        g = {"W1a": x.T @ dpre, "b1a": dpre.sum(0), "W2a": h.T @ dal, "b2a": dal.sum(0)}
        return np.concatenate([g[k].ravel() for k in self.qkeys])

    def _coefs(self, f, w):
        """Weights of dPhi(n) (cn, (B,)) and of dPhi(m_e) (cm, (B, E)) and d/dalpha (dal) in
        sum_n w(n) dpsi(n)."""
        Fs = self._safe(f["F"])
        P = np.prod(f["F"], 1)
        psi = f["v"] * P
        al = f["al"]
        cn = w * P * (1.0 - np.sum(al * f["q"] / Fs, 1))
        cm = np.where(f["act"], (w * psi)[:, None] * al / Fs * f["C"] * f["dg_du"], 0.0)
        dal = (w * psi)[:, None] * f["g"] / Fs
        return cn, cm, dal

    def force(self, rcoef, chunk):
        """sum_n rcoef(n) dpsi(n)/dtheta, exact over the full basis."""
        phi = self.phi(chunk)
        D = self.m.D
        cphi = np.zeros(D)
        ga = None
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            f = self._fac_idx(idx, phi)
            cn, cm, dal = self._coefs(f, rcoef[idx])
            cphi[idx] += cn
            a = f["act"]
            cphi += np.bincount(f["Mi"][a], weights=cm[a], minlength=D)
            gg = self._alpha_backprop(dal, f, False)
            ga = gg if ga is None else ga + gg
        return np.concatenate([self.base.force(cphi, chunk), ga])

    def grad_rows(self, idx, coef):
        """Rows coef(b) dpsi(idx_b)/dtheta (per sample), base rows evaluated per amplitude at idx_b
        and at every active edge neighbour."""
        phi = self.phi(32768)
        f = self._fac_idx(idx, phi)
        cn, cm, dal = self._coefs(f, coef)
        a = f["act"]
        B = len(idx)
        owner = np.concatenate([np.arange(B), np.nonzero(a)[0]])
        cfg = np.concatenate([idx, f["Mi"][a].astype(np.int64)])
        wts = np.concatenate([cn, cm[a]])
        cb = self.base.forward(*self.base._occ(cfg))
        G = self.base.grad(cb, wts, True)
        out = np.zeros((B, G.shape[1]))
        np.add.at(out, owner, G)
        return np.concatenate([out, self._alpha_backprop(dal, f, True)], 1)

    def per_sample_O(self, idx):
        psi = self.amplitudes(32768)[idx]
        return self.grad_rows(idx, 1.0 / np.where(np.abs(psi) > 1e-300, psi, 1e-300))

    def per_sample_O_proj(self, sg, idx, psiP, o_chunk=4):
        G = sg.G
        out = np.empty((len(idx), self.n_params()))
        for s in range(0, len(idx), o_chunk):
            ii = idx[s:s + o_chunk]
            B = len(ii)
            img = sg.inv[:, ii].T.astype(np.int64)
            coef = sg.chi[None, :] * sg.sinv[:, ii].T / (G * psiP[ii][:, None])
            out[s:s + B] = self.grad_rows(img.ravel(), coef.ravel()).reshape(B, G, -1).sum(1)
        return out

    # ---------------------------------------------------------------- per-amplitude (no tables)
    def psi_bits(self, bu, bd, T=None, J=None):
        """psi at bit patterns (B,) from per-amplitude base forward passes at each configuration
        and at its active edge neighbours. Returns (psi, number of base evaluations)."""
        m = self.m
        T = m.T if T is None else T
        J = m.J if J is None else J
        V, nu, nd = self.base.V, self.base.nu, self.base.nd
        nbu, nbd, act, hc = edge_moves_bits(bu, bd, self.edges, T, J)
        au = np.concatenate([bu, nbu[act]])
        ad = np.concatenate([bd, nbd[act]])
        ou, ru = bits_to_occ(au, V, nu)
        od, rd = bits_to_occ(ad, V, nd)
        ph = self.base.forward(ou, od, ru, rd)["psi"]
        B = len(bu)
        v = ph[:B][:, None]
        u = np.zeros(act.shape)
        u[act] = (hc[act] / self.scale) * ph[B:]
        g = np.where(act, self._gfun(v, u)[0], 0.0)
        x = np.concatenate([ou[:B], od[:B]], 1).astype(float) * 2 - 1
        _, al = self._alpha(x)
        return ph[:B] * np.prod(1.0 + al * g, 1), len(au)

    def psiP_per_amplitude(self, sg, idx):
        img = sg.inv[:, idx].T.astype(np.int64).ravel()
        ps = self.psi_bits(self.m.bu[img], self.m.bd[img])[0].reshape(len(idx), sg.G)
        return (ps * sg.chi[None, :] * sg.sinv[:, idx].T).sum(1) / sg.G

    def flip_weight(self, psi, chunk=65536):
        phi = self.phi(chunk)
        neg = np.zeros(self.m.D, bool)
        for s in range(0, self.m.D, chunk):
            idx = np.arange(s, min(s + chunk, self.m.D))
            neg[idx] = (np.sum(self._fac_idx(idx, phi)["F"] < 0, 1) % 2) == 1
        p = psi ** 2 / (psi @ psi)
        return float(p[neg].sum())


# ============================================================================ per-sample local energy
def eloc_bits(amp_fn, T, nn, J, ou, od):
    """E_loc(n) = diag + sum_m H_nm psi(m)/psi(n), every psi from amp_fn(bu, bd) -> (psi, n_eval)."""
    diag = connections(T, nn, J, ou, od)[0]
    bu, bd = occ_to_bits(ou[None]), occ_to_bits(od[None])
    edges = edge_list(T, nn, "K") + edge_list(T, nn, "X")
    nbu, nbd, act, hc = edge_moves_bits(bu, bd, edges, T, J)
    a = act[0]
    ps, nev = amp_fn(np.concatenate([bu, nbu[0, a]]), np.concatenate([bd, nbd[0, a]]))
    return float(diag + np.sum(hc[0, a] * ps[1:]) / ps[0]), nev


def base_amp_fn(base):
    def f(bu, bd):
        ou, ru = bits_to_occ(bu, base.V, base.nu)
        od, rd = bits_to_occ(bd, base.V, base.nd)
        return base.forward(ou, od, ru, rd)["psi"], len(bu)
    return f


# ============================================================================ arms
def match_alpha_width(m, base_kw, target, frame, refs):
    best = None
    for w in range(1, 200):
        a = FRD(m, LRAnsatz(m, refs[0], seed=0, **base_kw), frame, w, 0)
        if a.n_params() <= target:
            best = w
        else:
            break
    if best is None:
        raise RuntimeError(f"cannot fit FRD-{frame} into {target} parameters")
    return best


def make_arm(arm, m, args, refs, seed):
    kind = arm.replace("_mc", "")
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=args.ctrl_width).n_params()
    bkw = dict(mode="full", K=1, Hs=args.base_width)
    if kind == "nnbf6":
        a = LRAnsatz(m, refs[0], "full", K=1, Hs=args.ctrl_width, seed=seed)
        return a, dict(kind="nnbf", Hs=args.ctrl_width), target
    if kind == "nnbf4":
        a = LRAnsatz(m, refs[0], seed=seed, **bkw)
        return a, dict(kind="nnbf", Hs=args.base_width), target
    frame = kind[-1]
    w = match_alpha_width(m, bkw, target, frame, refs)
    a = FRD(m, LRAnsatz(m, refs[0], seed=seed, **bkw), frame, w, seed)
    return a, dict(kind="frd", frame=frame, base_Hs=args.base_width, alpha_width=w, E=a.E), target


# ============================================================================ sampled-force SR
class Chains:
    """Persistent Metropolis chains on p (full-basis vector), symmetric bond-swap proposals."""

    def __init__(self, m, n, rng, p):
        self.m, self.rng = m, rng
        self.bonds = np.array([(i, j) for i in range(m.V) for j in range(i + 1, m.V) if m.T[i, j] != 0.0])
        self.cur = rng.choice(m.D, size=n, p=p / p.sum())
        self.acc = self.prop = 0

    def step(self, p, nprop):
        m, one = self.m, np.int64(1)
        for _ in range(nprop):
            bu, bd = m.bu[self.cur], m.bd[self.cur]
            bb = self.bonds[self.rng.integers(len(self.bonds), size=len(self.cur))]
            i, j = bb[:, 0].astype(np.int64), bb[:, 1].astype(np.int64)
            si = ((bu >> i) & 1) - ((bd >> i) & 1)
            sj = ((bu >> j) & 1) - ((bd >> j) & 1)
            ok = si != sj
            mi, mj = (one << i), (one << j)
            nbu = bu & ~mi & ~mj
            nbd = bd & ~mi & ~mj
            nbu |= np.where(si == 1, mj, 0) | np.where(sj == 1, mi, 0)
            nbd |= np.where(si == -1, mj, 0) | np.where(sj == -1, mi, 0)
            k = m.index(nbu, nbd)
            k = np.where(ok & (k >= 0), k, self.cur)
            r = self.rng.random(len(self.cur))
            pc = p[self.cur]
            accept = ok & (r * pc < p[k])
            self.cur = np.where(accept, k, self.cur)
            self.acc += int(accept.sum()); self.prop += len(self.cur)

    def draw(self, p, nsamp):
        Ne = self.m.nup + self.m.ndn
        out = []
        while sum(len(o) for o in out) < nsamp:
            self.step(p, Ne)
            out.append(self.cur.copy())
        return np.concatenate(out)[:nsamp]


def optimize_mc(ans, m, sg, args, rng, label):
    """SR with force and metric from Metropolis samples of |P_chi psi|^2 (no exact force)."""
    hist, chains, nonfinite = [], None, False
    t0 = time.time()
    for it in range(args.sr_iters):
        psi_t, psi_c, _ = H21.state_of(ans, m, sg, "chi", 0.0, args.chunk)
        E_ex, Hp, nn = energy(m.H, psi_c)
        if not np.isfinite(E_ex):
            nonfinite = True
            log(f"{label}: non-finite state at it {it}"); break
        hist.append(E_ex)
        p = psi_c ** 2
        if chains is None:
            chains = Chains(m, args.chains, rng, p)
            chains.step(p, 20 * (m.nup + m.ndn))
        idx = chains.draw(p, args.sr_samples)
        El = Hp[idx] / psi_c[idx]
        O = ans.per_sample_O_proj(sg, idx, psi_c, args.o_chunk)
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        f = Oc.T @ ((El - El.mean()) / np.sqrt(len(idx)))
        lam = args.sr_shift
        Km = Oc @ Oc.T + lam * np.eye(len(idx))
        dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
        step = args.sr_lr * dx
        nrm = np.linalg.norm(dx)
        if not np.isfinite(nrm):
            nonfinite = True
            log(f"{label}: non-finite SR step at it {it}"); break
        if nrm * args.sr_lr > 1.0:
            step *= 1.0 / (nrm * args.sr_lr)
        ans.set_flat(ans.flat() - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E_exact = {E_ex:.8f}  E_mc = {El.mean():.6f}  |f| = {np.linalg.norm(f):.2e}"
                f"  acc = {chains.acc / max(chains.prop, 1):.3f}  ({(time.time() - t0) / (it + 1):.2f} s/step)")
    psi_c = H21.state_of(ans, m, sg, "chi", 0.0, args.chunk)[1]
    E_fin = energy(m.H, psi_c)[0]
    nonfinite |= not np.isfinite(E_fin)
    h = np.array(hist + [E_fin])
    jumps = int(np.sum(np.diff(h) > JUMP)) if len(h) > 1 else 0
    return dict(E_hist=h[::10].tolist(), E0_step=float(h[0]), E_final=float(E_fin), nonfinite=bool(nonfinite),
                diverged=bool(nonfinite or not (E_fin < h[0])), jumps=jumps,
                jump_rate=jumps / max(len(h) - 1, 1), steps=len(h) - 1,
                accept=chains.acc / max(chains.prop, 1) if chains else float("nan"),
                s_per_step=(time.time() - t0) / max(len(h) - 1, 1))


# ============================================================================ train task
def setup(inst):
    m = H21.make_model(inst)
    m.spin_setup()
    return m


def train_task(args):
    m = setup(args.instance)
    m.pair_setup(args.r_min)
    log(f"instance {args.instance} {H21.INSTANCES[args.instance]} D={m.D} arm={args.arm} seed={args.seed} "
        f"shift={args.sr_shift}")
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 4, 1e-8, H21.UHF_SEED)
    psi0 = H21.resolve_ground(m, Vv, deg)
    sg, gen = H21.build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    dim_chi, dim_S_chi = H21.sector_dims(m, sg, S0)
    log(f"ED {W.tolist()} deg {deg} |G|={sg.G} chi={gen} S={S0:g}; dim P_chi {dim_chi:.0f}, "
        f"dim P_S P_chi {dim_S_chi:.0f} ({time.time() - t0:.0f} s)")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                              [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)[0]
    calib = float(Ec - W[0])
    refs = H21.references(m)
    ans, kw, target = make_arm(args.arm, m, args, refs, args.seed)
    if isinstance(ans, FRD):
        ans.tables(); ans.set_flat(ans.flat())
    log(f"arm {args.arm}: {kw} n_params={ans.n_params()} (control {target}) calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * (ARMS + MC_ARMS).index(args.arm))
    res = dict(params={k: v for k, v in vars(args).items()}, instance=args.instance, arm=args.arm,
               seed=args.seed, shift=args.sr_shift, arm_kwargs=kw, n_params=ans.n_params(), P_ctrl=target,
               ed_levels=W.tolist(), degeneracy=deg, irrep=gen, G=sg.G, S_psi0=S0, calib_err=calib,
               dim_chi=dim_chi, dim_S_chi=dim_S_chi, hard_core="sign half only")
    t1 = time.time()
    if args.arm.endswith("_mc"):
        res["mc"] = optimize_mc(ans, m, sg, args, rng, args.arm)
    else:
        res["s_per_step"] = H21.optimize(ans, m, sg, "chi", S0, args, rng, args.arm)
    res["t_sr"] = time.time() - t1
    psi_t, psi_c, _ = H21.state_of(ans, m, sg, "chi", S0, args.chunk)
    states = [("P", psi_c)] if args.arm.endswith("_mc") else [("P", psi_c), ("P_S", m.spin_project(psi_c, S0))]
    budgets = {}
    for lab, v in states:
        nrm = np.linalg.norm(v)
        if not np.isfinite(nrm) or nrm < 1e-10 * np.linalg.norm(psi_t):
            budgets[lab] = None
            continue
        t2 = time.time()
        b = budget(m, v, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, args, rng_b)
        vn = v / nrm
        b.update(S2=m.S2_expect(vn), ov0=float((psi0 @ vn) ** 2))
        for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
            b.pop(k_, None)
        budgets[lab] = b
        log(f"{args.arm} [{lab}]: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} f_E={b['f_E_up']:.3f} "
            f"w_sign={b['w_sign']:.2e} spread={b['spread']:.1e} ov0={b['ov0']:.3f} ({time.time() - t2:.0f} s)")
    res["budgets"] = budgets
    if isinstance(ans, FRD):
        res["flip_weight"] = ans.flip_weight(psi_c)
        x0 = ans.flat().copy()
        ans.q["W2a"] = np.zeros_like(ans.q["W2a"]); ans.q["b2a"] = np.zeros_like(ans.q["b2a"])
        res["E_alpha0"] = energy(m.H, sg.project(ans.amplitudes(args.chunk)))[0]
        ans.set_flat(x0)
        log(f"{args.arm}: flip_weight={res['flip_weight']:.3e} E(alpha=0, co-adapted; diagnostic)={res['E_alpha0']:.6f}")
    if args.mc_samples > 0 and not args.arm.endswith("_mc"):
        t3 = time.time()
        res["sampled"] = H21.sampled_arm(ans, m, sg, psi_c, args.mc_samples, args.mc_burn,
                                         np.random.default_rng(args.seed + 77))
        log(f"{args.arm} sampled: " + " ".join(f"{k}={v:.4g}" for k, v in res["sampled"].items())
            + f" ({time.time() - t3:.0f} s)")
    pdir = os.path.join(args.results, "params")
    os.makedirs(pdir, exist_ok=True)
    np.savez(os.path.join(pdir, f"inst{args.instance}_{args.arm}_s{args.seed}_sh{args.sr_shift:g}.npz"),
             flat=ans.flat())
    return res


# ============================================================================ checks
def run_checks(seed):
    ok = True
    rng = np.random.default_rng(seed)
    for inst in (9, 8):
        m = setup(inst)
        mJ0 = TJ(m.Lx, m.Ly, m.bcx, m.bcy, m.nup, m.ndn, 1.0, m.tp, 0.0)
        Hd = sp.diags(m.H.diagonal())
        Hkin, Hex = mJ0.H, m.H - mJ0.H - Hd
        y = rng.standard_normal(m.D)
        refs = H21.references(m)
        for frame, Href, sc in (("K", Hkin, 1.0), ("X", Hex, 0.5 * m.J)):
            a = FRD(m, LRAnsatz(m, refs[0], "full", K=1, Hs=3, seed=seed), frame, 2, seed)
            Mi, C = a.tables()
            act = Mi >= 0
            Ky = np.sum(np.where(act, C.astype(float) * sc * y[np.maximum(Mi, 0)], 0.0), 1)
            err = np.max(np.abs(Ky - Href @ y))
            log(f"check inst {inst} D={m.D} frame {frame}: E={a.E} edges, max|sum_e c_e y(m_e) - H_{frame} y| = {err:.1e}")
            ok &= err < 1e-12
    # gradients, projected log-derivatives, per-amplitude psi, per-sample E_loc on instance 9
    m = setup(9)
    W, U = np.linalg.eigh(m.H.toarray())
    sg, gen = H21.build_symmetry(m, U[:, 0])
    refs = H21.references(m)
    for frame in ("K", "X"):
        base = LRAnsatz(m, refs[0], "full", K=1, Hs=3, seed=seed, w2_scale=0.3)
        a = FRD(m, base, frame, 3, seed)
        a.q["W2a"] = 1.5 * rng.standard_normal(a.q["W2a"].shape)
        a.q["b2a"] = 0.5 * rng.standard_normal(a.E)
        a.set_flat(a.flat())
        psi = a.amplitudes(64)
        nflip = a.flip_weight(psi)
        x0 = a.flat().copy()
        d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
        h = 1e-6
        for proj in ("none", "chi"):
            st = lambda: H21.state_of(a, m, sg, proj, 0.0, 64)
            psi_t, psi_c, ps = st()
            E, Hp, nn_ = energy(m.H, ps)
            g = 2 * a.force((Hp - E * ps) / nn_, 64)
            Ef = lambda x: (a.set_flat(x), energy(m.H, st()[2])[0])[1]
            fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
            a.set_flat(x0)
            e1 = abs(g @ d - fd) / max(1.0, abs(fd))
            msg = f"check FRD-{frame} grad proj={proj}: analytic {g @ d:.10e} fd {fd:.10e} (rel {e1:.1e})"
            ok &= e1 < 1e-6
            idx = rng.choice(m.D, size=6, p=ps ** 2 / (ps @ ps))
            if proj == "chi":
                O = a.per_sample_O_proj(sg, idx, psi_c, 2)
                lp = lambda x: (a.set_flat(x), sg.project(a.amplitudes(64))[idx])[1]
                pa = a.psiP_per_amplitude(sg, idx)
                epa = np.max(np.abs(pa - psi_c[idx])) / np.max(np.abs(psi_c[idx]))
                msg += f"; per-amplitude psi_P {epa:.1e}"
                ok &= epa < 1e-12
            else:
                O = a.per_sample_O(idx)
                lp = lambda x: (a.set_flat(x), a.amplitudes(64)[idx])[1]
            fdO = (lp(x0 + h * d) - lp(x0 - h * d)) / (2 * h) / (psi_c if proj == "chi" else psi_t)[idx]
            a.set_flat(x0)
            eO = np.max(np.abs(O @ d - fdO)) / max(1.0, np.max(np.abs(fdO)))
            msg += f"; O vs fd {eO:.1e}"
            ok &= eO < 1e-5
            log(msg)
        # per-sample local energy (honest per-amplitude path) == (H psi)/psi on every config
        Hpsi = m.H @ psi
        iu, idd = m.iu_id()
        err = errn = 0.0
        nev = []
        for j in range(m.D):
            ou = m.occ_u[iu[j]].astype(np.int64); od = m.occ_d[idd[j]].astype(np.int64)
            el, ne = eloc_bits(a.psi_bits, m.T, m.nn, m.J, ou, od)
            err = max(err, abs(el - Hpsi[j] / psi[j]) / (1 + abs(Hpsi[j] / psi[j])))
            nev.append(ne)
            phi = a.base.amplitudes(64)
            eb = eloc_brute(a.base, m.T, m.nn, m.J, ou, od)
            en = eloc_bits(base_amp_fn(a.base), m.T, m.nn, m.J, ou, od)[0]
            errn = max(errn, abs(en - eb) / (1 + abs(eb)), abs(en - (m.H @ phi)[j] / phi[j]) / (1 + abs(eb)))
        log(f"check FRD-{frame} per-sample E_loc == (H psi)/psi on all {m.D} configs: {err:.1e} "
            f"(NNBF per-sample path {errn:.1e}); flip_weight {nflip:.3f}; base evals per E_loc "
            f"mean {np.mean(nev):.1f}")
        ok &= err < 1e-9 and errn < 1e-9 and nflip > 0
    # sampled force (Metropolis) converges to the exact projected force
    m8 = setup(8)
    W8, U8 = np.linalg.eigh(m8.H.toarray())
    sg8, _ = H21.build_symmetry(m8, U8[:, 0])
    refs8 = H21.references(m8)
    a = FRD(m8, LRAnsatz(m8, refs8[0], "full", K=1, Hs=3, seed=seed, w2_scale=0.3), "K", 3, seed)
    a.q["W2a"] = 0.3 * rng.standard_normal(a.q["W2a"].shape); a.set_flat(a.flat())
    _, psi_c, _ = H21.state_of(a, m8, sg8, "chi", 0.0, 4096)
    E, Hp, nn_ = energy(m8.H, psi_c)
    f_ex = a.force((Hp - E * psi_c) / nn_, 4096)
    ch = Chains(m8, 200, np.random.default_rng(seed), psi_c ** 2)
    ch.step(psi_c ** 2, 200)
    idx = ch.draw(psi_c ** 2, 40000)
    El = Hp[idx] / psi_c[idx]
    O = a.per_sample_O_proj(sg8, idx, psi_c, 64)
    f_mc = ((El - El.mean())[:, None] * (O - O.mean(0))).mean(0)
    rel = np.linalg.norm(f_mc - f_ex) / np.linalg.norm(f_ex)
    # the sampled estimator is unbiased: its exact |psi|^2 expectation equals the exact force, and the
    # Metropolis error must be of the size predicted for independent draws (heavy tails make it large)
    p = psi_c ** 2 / nn_
    sup = np.nonzero(p > 0)[0]
    pm, Els = p[sup], Hp[sup] / psi_c[sup]
    Oa = a.per_sample_O_proj(sg8, sup, psi_c, 64)
    Om = pm @ Oa
    terms = (Els - E)[:, None] * (Oa - Om)
    e_id = np.linalg.norm(pm @ terms - f_ex) / np.linalg.norm(f_ex)
    pred = np.sqrt((pm @ (terms - pm @ terms) ** 2).sum() / len(idx)) / np.linalg.norm(f_ex)
    log(f"check sampled force (inst 8 D={m8.D}): exact expectation of the sampled estimator vs force "
        f"{e_id:.1e}; 40000 Metropolis samples rel err {rel:.3f} vs iid-predicted {pred:.3f} "
        f"(acceptance {ch.acc / ch.prop:.3f})")
    ok &= e_id < 1e-10 and rel < 3 * pred
    # parameter matching on the production instance (no Hamiltonian needed)
    c = H21.INSTANCES[0]
    m0 = TJ(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["J"], build_H=False)
    V0, n0 = m0.V, c["nup"]
    eye = (np.eye(V0)[:, :n0], np.eye(V0)[:, :n0])
    tgt = LRAnsatz(m0, eye, "full", K=1, Hs=6).n_params()
    b4 = LRAnsatz(m0, eye, "full", K=1, Hs=4).n_params()
    for fr in ("K", "X"):
        w = match_alpha_width(m0, dict(mode="full", K=1, Hs=4), tgt, fr, (eye,))
        a = FRD(m0, LRAnsatz(m0, eye, "full", K=1, Hs=4), fr, w, 0)
        log(f"check parameter match inst 0: nnbf6 {tgt}, nnbf4 {b4}, frd{fr} (base 4, alpha width {w}, "
            f"E={a.E}) {a.n_params()}")
        ok &= a.n_params() <= tgt
    log(f"ALL R5 CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ cost
def cost_measure(args):
    rows = []
    for L in args.size_list:
        V = L * L
        nh = max(2, int(round(V / 8)))
        if (V - nh) % 2:
            nh += 1
        n = (V - nh) // 2
        T, coords, bonds = build_lattice(L, L, "pbc", "pbc", 1.0, -0.25)
        nn = sorted({(min(i, j), max(i, j)) for (i, j, _) in bonds})

        class M:
            pass
        mm = M(); mm.V, mm.nup, mm.ndn, mm.T, mm.nn, mm.J = V, n, n, T, nn, 0.4
        rng = np.random.default_rng(args.seed + L)
        e, v = np.linalg.eigh(-T)
        ref = (v[:, :n] + 0.1 * rng.standard_normal((V, n)), v[:, :n] + 0.1 * rng.standard_normal((V, n)))
        base = LRAnsatz(mm, ref, "full", K=1, Hs=6, seed=args.seed, w2_scale=0.3)
        row = dict(L=L, V=V, Ne=2 * n)
        tn, nn_ev = [], []
        cfgs = [H21.random_tj_config(V, nh, rng) for _ in range(args.nconf)]
        for ou, od in cfgs:
            t0 = time.perf_counter(); _, ne = eloc_bits(base_amp_fn(base), T, nn, 0.4, ou, od)
            tn.append(time.perf_counter() - t0); nn_ev.append(ne)
        row.update(T_nnbf=float(np.median(tn)), ev_nnbf=float(np.mean(nn_ev)))
        for fr in ("K", "X"):
            a = FRD(mm, base, fr, 6, args.seed)
            a.q["W2a"] = rng.standard_normal(a.q["W2a"].shape)
            tf, ev = [], []
            for ou, od in cfgs:
                t0 = time.perf_counter(); _, ne = eloc_bits(lambda bu, bd: a.psi_bits(bu, bd, T, 0.4), T, nn, 0.4, ou, od)
                tf.append(time.perf_counter() - t0); ev.append(ne)
            row.update({f"T_frd{fr}": float(np.median(tf)), f"ev_frd{fr}": float(np.mean(ev)), f"E_{fr}": a.E,
                        f"ratio_{fr}": float(np.median(tf) / np.median(tn))})
        rows.append(row)
        log(f"cost L={L} V={V} Ne={2 * n}: NNBF {row['ev_nnbf']:.0f} evals {row['T_nnbf']:.3e} s | "
            f"FRD-K {row['ev_frdK']:.0f} evals {row['T_frdK']:.3e} s (x{row['ratio_K']:.1f}) | "
            f"FRD-X {row['ev_frdX']:.0f} evals {row['T_frdX']:.3e} s (x{row['ratio_X']:.1f})")
    sl = {}
    if len(rows) >= 2:
        ln = np.log([r["Ne"] for r in rows])
        for k in ("T_nnbf", "T_frdK", "T_frdX", "ev_nnbf", "ev_frdK", "ev_frdX"):
            sl[k] = float(np.polyfit(ln, np.log([r[k] for r in rows]), 1)[0])
        log("cost log-log slopes in N_e: " + " ".join(f"{k}={v:.2f}" for k, v in sl.items()))
    return dict(rows=rows, slopes=sl)


# ============================================================================ aggregate
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def aggregate(d, final=False):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "inst*_*_s*_sh*.json"))):
        r = json.load(open(f))
        R.setdefault(r["instance"], []).append(r)
    out = {}
    print("P1.2 r5 aggregate: SIGN HALF ONLY (no same-sector competitor is claimed)")
    print(f"pre-committed: premise med dE_sign(nnbf6) >= {PREMISE}; FAIL if rho_sign >= {RHO_FAIL} or rho_tot > "
          f"{TOT_FAIL} or kurt ratio > {KAPPA_MAX} or MC_FAIL; PASS if guards & rho_sign <= {RHO_PASS} & "
          f"seed ranges disjoint & rho_tot <= {TOT_PASS} & rho_abl <= {ABL_PASS} & kurt ratio <= {KAPPA_MAX} & MC_OK"
          + (" [FINAL 5-seed rule]" if final else ""))
    for inst, rs in sorted(R.items()):
        sel = {}   # (arm, seed) -> result with lower final variational energy over shifts
        mc = {}
        for r in rs:
            if r["arm"].endswith("_mc"):
                mc[r["arm"]] = r
                continue
            b = (r["budgets"] or {}).get("P")
            if b is None:
                continue
            k = (r["arm"], r["seed"])
            if k not in sel or b["E"] < sel[k]["budgets"]["P"]["E"]:
                sel[k] = r
        exact_s11 = {r["arm"]: r for r in rs if not r["arm"].endswith("_mc") and r["seed"] == MC_SEED
                     and abs(r["shift"] - MC_SHIFT) < 1e-12 and (r["budgets"] or {}).get("P")}
        col = lambda arm, key, lab="P": [sel[k]["budgets"][lab][key] for k in sorted(sel) if k[0] == arm
                                         and sel[k]["budgets"].get(lab)]
        kurt = lambda arm: [sel[k]["sampled"]["kurt_ex"] for k in sorted(sel) if k[0] == arm and sel[k].get("sampled")]
        print(f"\ninstance {inst} {H21.INSTANCES.get(inst)}")
        for arm in ARMS:
            ds, dt = col(arm, "dE_sign"), col(arm, "dE_tot")
            if not ds:
                continue
            seeds = [k[1] for k in sorted(sel) if k[0] == arm]
            extra = ""
            if arm.startswith("frd"):
                extra = f" flip_w {_med([sel[k].get('flip_weight') for k in sel if k[0] == arm]):.2e}"
            print(f"  {arm:6s} seeds {seeds} dE_sign med {_med(ds):.5f} [{min(ds):.5f},{max(ds):.5f}] dE_tot med "
                  f"{_med(dt):.5f} w_sign {_med(col(arm, 'w_sign')):.2e} ov0 {_med(col(arm, 'ov0')):.3f} kurt "
                  f"{_med(kurt(arm)):.3g} | P_S: dE_sign {_med(col(arm, 'dE_sign', 'P_S')):.5f}{extra}")
        c_ds = col("nnbf6", "dE_sign")
        prem = _med(c_ds)
        informative = len(c_ds) >= 3 and prem >= PREMISE
        print(f"  premise: med dE_sign(nnbf6) = {prem:.5f} over {len(c_ds)} seeds -> "
              f"{'INFORMATIVE' if informative else 'NOT informative'}")
        mc_c = mc.get("nnbf6_mc")
        for X in ("frdK", "frdX"):
            x_ds, x_dt = col(X, "dE_sign"), col(X, "dE_tot")
            if not x_ds or not c_ds:
                continue
            rho = _med(x_ds) / prem
            rho_t = _med(x_dt) / _med(col("nnbf6", "dE_tot"))
            a_ds = col("nnbf4", "dE_sign")
            rho_abl = _med(x_ds) / _med(a_ds) if a_ds else float("nan")
            disjoint = max(x_ds) < min(c_ds)
            wins = sum(v < prem for v in x_ds)
            kap = _med(kurt(X)) / _med(kurt("nnbf6"))
            guards = True
            for arm in (X, "nnbf6", "nnbf4"):
                ks = [k for k in sel if k[0] == arm]
                guards &= len(ks) >= 3
                for k in ks:
                    b = sel[k]["budgets"]["P"]
                    guards &= abs(sel[k]["calib_err"]) <= CALIB_TOL * b["dE_tot"] and b["spread"] < SPREAD_TOL
            mc_x = mc.get(X + "_mc")
            mc_ok = mc_fail = False
            mcs = "MC missing"
            if mc_x and mc_c and X in exact_s11 and "nnbf6" in exact_s11:
                def stall(rr, arm):
                    bm = (rr["budgets"] or {}).get("P")
                    dmc = bm["dE_tot"] if bm else float("inf")
                    return dmc / exact_s11[arm]["budgets"]["P"]["dE_tot"]
                sx, sc = stall(mc_x, X), stall(mc_c, "nnbf6")
                jx, jc = mc_x["mc"]["jump_rate"], mc_c["mc"]["jump_rate"]
                bad = mc_x["mc"]["nonfinite"] or mc_x["mc"]["diverged"]
                mc_ok = (not bad) and sx <= STALL_OK * sc and jx <= max(JR_OK_F * jc, JR_OK_A)
                mc_fail = bad or sx > STALL_BAD * sc or jx > max(JR_BAD_F * jc, JR_BAD_A)
                mcs = (f"MC stall {sx:.2f} (ctrl {sc:.2f}) jumps {jx:.3f} (ctrl {jc:.3f}) diverged {bad} -> "
                       f"{'OK' if mc_ok else ('FAIL' if mc_fail else 'neither')}")
            if final:
                wins_f = sum(v < prem for v in x_ds)
                ok = (rho <= RHO_PASS and wins_f >= 4 and len(x_ds) >= 5 and rho_t <= TOT_PASS
                      and kap <= KAPPA_MAX and mc_ok)
                v = "PASS" if ok else "FAIL"
            elif rho >= RHO_FAIL or rho_t > TOT_FAIL or kap > KAPPA_MAX or mc_fail:
                v = "FAIL"
            elif (guards and rho <= RHO_PASS and disjoint and rho_t <= TOT_PASS and rho_abl <= ABL_PASS
                  and kap <= KAPPA_MAX and mc_ok):
                v = "PASS"
            else:
                v = "AMBIGUOUS"
            if not informative:
                route = "NO-TARGET"
            else:
                route = v
            print(f"  [{X}] rho_sign={rho:.3f} rho_tot={rho_t:.3f} rho_abl={rho_abl:.3f} disjoint={disjoint} "
                  f"wins {wins}/{len(x_ds)} kurt ratio {kap:.2f} guards={'OK' if guards else 'FAIL'} {mcs} -> {v}"
                  + ("" if informative else " (not counted)"))
            out[f"{inst}:{X}"] = dict(rho_sign=rho, rho_tot=rho_t, rho_abl=rho_abl, disjoint=disjoint, wins=wins,
                                      kappa=kap, guards=bool(guards), mc_ok=mc_ok, mc_fail=mc_fail, verdict=v,
                                      route=route, premise=prem)
            print(f"ROUTE {X} (instance {inst}): {route}"
                  + ("  [action: seeds 44, 55 for nnbf6 and " + X + ", then --final 1]" if route == "AMBIGUOUS" else ""))
    json.dump(out, open(os.path.join(d, "aggregate.json"), "w"), indent=1, default=float)
    return out


# ============================================================================ timing
def time_step(args):
    m = setup(args.instance)
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 4, 1e-8, H21.UHF_SEED)
    psi0 = H21.resolve_ground(m, Vv, deg)
    sg, gen = H21.build_symmetry(m, psi0)
    log(f"instance {args.instance} D={m.D} ED+group {time.time() - t0:.0f} s |G|={sg.G}")
    refs = H21.references(m)
    for arm in (args.arm.split(",") if args.arm else ARMS + ["frdK_mc"]):
        ans, kw, target = make_arm(arm, m, args, refs, 11)
        if isinstance(ans, FRD):
            t0 = time.time(); ans.tables(); log(f"{arm}: edge tables {time.time() - t0:.1f} s")
        rng = np.random.default_rng(0)
        a2 = argparse.Namespace(**vars(args)); a2.sr_iters = 2
        t0 = time.time()
        if arm.endswith("_mc"):
            optimize_mc(ans, m, sg, a2, rng, arm)
        else:
            for _ in range(2):
                E, f, step = H21.sr_step(ans, m, sg, "chi", 0.0, a2, rng)
                ans.set_flat(ans.flat() - step)
        log(f"{arm}: n_params={ans.n_params()} (ctrl {target}) {kw}: {(time.time() - t0) / 2:.2f} s per SR step")


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 round-5 test (see module docstring)")
    ap.add_argument("--mode", choices=["train", "aggregate", "cost", "checks", "smoke", "time"], default="smoke")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--instance", type=int, default=0, choices=[0, 8, 9])
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--sr-shift", type=float, default=-1.0)
    ap.add_argument("--sr-iters", type=int, default=1200)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--base-width", type=int, default=4)
    ap.add_argument("--ctrl-width", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=4)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--mc-samples", type=int, default=20000)
    ap.add_argument("--mc-burn", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=100)
    ap.add_argument("--sizes", default="4,5,6,7")
    ap.add_argument("--nconf", type=int, default=3)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--final", type=int, choices=[0, 1], default=0)
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0:
        if a.task > 26:
            ap.error("--task must be in 0..26")
        if a.task < 24:
            arm, seed, sh = ARMS[a.task // 6], SEEDS[(a.task % 6) // 2], SHIFTS[a.task % 2]
        else:
            arm, seed, sh = MC_ARMS[a.task - 24], MC_SEED, MC_SHIFT
        a.arm = a.arm or arm
        a.seed = seed if a.seed < 0 else a.seed
        a.sr_shift = sh if a.sr_shift < 0 else a.sr_shift
    if a.seed < 0:
        a.seed = SEEDS[0]
    if a.sr_shift < 0:
        a.sr_shift = SHIFTS[0]
    if a.mode == "train" and a.arm not in ARMS + MC_ARMS:
        ap.error("--mode train needs --task or a valid --arm")
    a.size_list = [int(s) for s in a.sizes.split(",") if s.strip()]
    if any(L * L > 62 for L in a.size_list):
        ap.error("--sizes: L*L must be <= 62 (64-bit occupation patterns)")
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "aggregate":
        aggregate(args.results, bool(args.final)); return 0
    if args.mode == "checks":
        return 0 if run_checks(H21.UHF_SEED) else 2
    if args.mode == "time":
        time_step(args); return 0
    if args.mode == "cost":
        json.dump(cost_measure(args), open(os.path.join(args.results, "cost.json"), "w"), indent=1); return 0
    if args.mode == "smoke":
        t0 = time.time()
        if not run_checks(H21.UHF_SEED):
            return 2
        sdir = os.path.join(HERE, "results_smoke")
        os.makedirs(sdir, exist_ok=True)
        for t in range(27):
            sm = parse_args(["--mode", "train", "--task", str(t), "--instance", "8", "--sr-iters", "30",
                             "--sr-samples", "100", "--lbfgs-iters", "300", "--fn-iters", "2", "--mc-samples",
                             "400", "--mc-burn", "50", "--r-min", "1.0", "--chains", "20", "--chunk", "4096",
                             "--base-width", "2", "--ctrl-width", "4", "--results", sdir])
            res = train_task(sm)
            json.dump(res, open(os.path.join(sdir, f"inst8_{sm.arm}_s{sm.seed}_sh{sm.sr_shift:g}.json"), "w"),
                      indent=1, default=float)
        aggregate(sdir)
        sc = parse_args(["--mode", "cost", "--sizes", "4,5", "--nconf", "1"])
        cost_measure(sc)
        log(f"smoke total {time.time() - t0:.1f} s")
        return 0
    out = os.path.join(args.results, f"inst{args.instance}_{args.arm}_s{args.seed}_sh{args.sr_shift:g}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    if not run_checks(H21.UHF_SEED):
        return 2
    res = train_task(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
