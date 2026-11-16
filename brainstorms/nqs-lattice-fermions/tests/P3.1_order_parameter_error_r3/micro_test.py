#!/usr/bin/env python3
"""
P3.1 round-3 decisive test: order-parameter error of SYMMETRY-PROJECTED variational states
=========================================================================================

Question
--------
Given a symmetry-projected, SR-trained variational state psi (NNBF or the Slater-Jastrow
foil) of a doped Hubbard cluster (4x3 open, 4x3 cylinder, 6x2 ladder, 4x3 tori), can a cheap estimator predict the residual error
delta = a_0 - a_psi of the long-range d-wave pair correlation a = <A>, better than current
practice (seed spread, zero-variance extrapolation, one Lanczos step)?

Routes under test (P3.1 artifact, round 3)
------------------------------------------
  ECC  Route 3.1, ensemble-calibrated covariance ("zero-covariance extrapolation").
       Regressor c = Cov_psi(E_loc, A_loc) = <(H-E)psi|(A-a)psi>, which is signed, first
       order in the error vector and zero for every eigenstate. Over ALL projected states
       of one instance (all seeds), fit a_k = a0_hat + beta_f c_k (one intercept, one slope
       per ansatz family f in {nnbf, foil}); per-state estimate delta_hat_k = -beta_f c_k.
       A certificate is issued iff the seed-bootstrap relative SE of beta_f is <= 0.5 and
       |c_k| >= 2 SE_MC(c_k). a, c come from Monte Carlo (see "Sampling"); c uses the
       ONE-SIDED local estimator X(n)/psi(n) with X = (H-E)(A-a) psi symmetrised over HA and
       AH (single 1/psi tail, cost N_H N_A amplitude evaluations per sample). The two-sided
       product (E_loc - E)(A_loc - a) (1/psi^2 tail, cost N_H + N_A) is reported only.
  ZR   Route 3.2, Hylleraas-trained response network. A second projected NNBF phi (width
       --zr-width) maximises R[phi] = <phi|w>^2 / <Q phi|(H-E)|Q phi>, with
       w = Q (A - a) psi and Q = 1 - |psi><psi|; the optimum of R is the first-order
       response y* = [Q(H-E)Q]^-1 w. With y = (N/Dn) Q phi (N = <phi|w>, Dn the
       denominator) the estimate is delta_hat = -2 <(H-E)psi|y>, the zero-bias correction
       with a response OUTSIDE the tangent space of psi. The response network is trained
       with exact-sum gradients (as every training here); the contraction <(H-E)psi|Q phi>
       is sampled from |psi|^2 with the one-sided local estimator ((H - E) phi)(n)/psi(n)
       (cost N_H amplitude evaluations of phi per sample; the two-sided (E_loc - E) phi/psi
       is reported only). A certificate is issued iff the bootstrap SE of the sampled
       estimate is <= 0.5 of its magnitude. The capture R/R* uses the oracle R* and is a
       diagnostic only, never part of the trust rule.

Controls (estimators; same states, same samples)
------------------------------------------------
  SD   seed spread (current NQS practice, e.g. Rende et al. arXiv:2606.02794): for each
       state type (instance, population label), |delta_hat| = sample standard deviation of
       a over seeds. Unsigned; scored on |delta_hat|/|delta|.
  ZV   zero-variance extrapolation: same regression as ECC with sigma^2/V in place of c.
  L1   one Lanczos step, two-point first-order extrapolation (round-2 rule),
       delta_hat = (a_phi - a_psi)/(1 - sigma_phi/sigma_psi), exact matrix elements.
  SMA  single-state covariance with the single-mode scale omega = <w|(H-E)|w>/<w|w>,
       delta_hat = -2 c / omega (= Krylov-Galerkin response in span{w}); ablation of ECC's
       ensemble calibration. Exact.
  KR2  Krylov-Galerkin response in span{w, (H-E) w} (the killed A-Ritz family, linearised);
       ablation of ZR's trained response. Exact.
  ZRO  oracle: delta_hat = -2 <(H-E)psi | [Q(H-E)Q]^-1 w> by MINRES on the full space.
       Premise check of the first-order formula that ZR approximates; not a method.

State populations (one Slurm task = one instance x one seed)
------------------------------------------------------------
All from the UHF start of that seed (P1.1 protocol), exact-force SR (P1.1 settings,
--sr-iters steps), space group G = translations along periodic directions x {1, Mx, My,
MxMy} (|G| = 4 open 4x3, 16 cylinder 4x3, 24 ladder 6x2, 48 torus 4x3),
P_chi = projector on psi_0's 1-D irrep, P_S = Loewdin projector on psi_0's total spin.
  P_w2@<it>, P_w2        projected SR (psi = P_chi psi_theta) NNBF width 2, snapshots at the
                         iterations in --snaps plus the final (best-energy) state
  P_w8@<it>, P_w8        same, width 8
  PS_w8@<it>, PS_w8      projected SR with psi = P_S P_chi psi_theta (energy and force of the
                         spin- and space-projected state; SR metric = the P-arm metric from
                         samples of |P_chi psi_theta|^2, a preconditioner only)
  F_P, F_PS              Slater-Jastrow foil, projected SR with P_chi, resp. P_S P_chi
  UP_w8 (reported)       plain NNBF width 8 trained unprojected (P1.1), projected post hoc
  U_w8  (reported)       the unprojected state itself
States labelled "reported" never enter the verdict.

Sampling
--------
For every state, --mc-samples configurations are drawn i.i.d. and exactly from |psi|^2
(numpy choice on the full basis; stronger than Metropolis, no autocorrelation). Local
estimators E_loc(n) = (H psi)(n)/psi(n), A_loc(n) = (A psi)(n)/psi(n), phi(n)/psi(n) are
read from full-basis vectors; this is value-identical to per-sample evaluation, which the
checks verify explicitly on sampled configurations (per-sample row of H and column of A).
Per-state standard errors: --n-boot bootstrap resamples of the samples. Heavy tails are
reported as the largest single-sample share of sum |summand| and the excess kurtosis.

Metric and pre-committed decision rule (implemented in --aggregate)
-------------------------------------------------------------------
  scored state: verdict population (not "reported"), |delta| >= SCORE_REL * |a_0|.
  tracked (signed arms): trusted and 1/2 <= delta_hat/delta <= 2;
  tracked (SD):          1/2 <= |delta_hat|/|delta| <= 2 (always trusted);
  false certificate:     trusted and delta_hat/delta < 1/2 (includes the wrong sign).
  Statistical unit u = (instance, seed). T_X(u) = tracked/scored over the scored states of
  that unit (abstentions are misses); units with no scored state are dropped.
  M_X     = median_u T_X(u);
  B       = the control in {SD, ZV, L1} with the largest M (ties -> SD, ZV, L1 order);
  Delta_X = median_u [T_X(u) - T_B(u)] (paired), with a seed-unit bootstrap 90% interval;
  G_X     = pooled false-certificate fraction over scored states;
  Cov_X   = trusted fraction of scored projected-NNBF states.
  Residual-size premise gate (NO-TARGET): an instance has a target iff it has
  >= MIN_SCORED (6) scored states over >= 3 seeds, of which >= MIN_FINAL (3) are final
  (best-energy, not snapshot) states. Instances without a target are printed with the median
  |delta|/|a0| of their final states and excluded; the verdict metrics are recomputed on the
  target instances only. If fewer than MIN_TARGET_INST (2) instances have a target, both
  routes are NO-TARGET (states already exact: no pass is possible).
  ZR additionally needs the oracle ZRO to track >= ZRO_MIN of scored states (first-order
  formula valid); otherwise ZR is "FAIL (premise)".
  PASS    : M_X >= 0.5 and Delta_X >= 0.2 and G_X <= 0.10 and Cov_X >= 0.5 and the
            per-instance median of T_X(u) - T_B(u) is > 0 on EVERY instance.
  FAIL    : M_X < 0.3 or Delta_X < 0.05 or G_X > 0.20.
  AMBIGUOUS otherwise; pre-committed action: rerun with --mc-samples x10 and 6 new seeds;
            a route that is still not PASS is killed.
  Reported only: the same table in exact mode (no MC), the Spearman correlation of
  |delta_hat| with |delta|, the reported populations (U_w8, UP_w8), ECC's premise
  diagnostic omega_k = -2 c_k / delta_k (spread across seeds), ZR capture R/R*, overlaps
  |<psi_0|psi>|^2, sign-mismatch weight w_sign, <S^2>.

Usage
-----
  python3 micro_test.py [options]
  --instance K      0..6 (default 1). All t=1, t'=-0.25:
                      0 : 4x3 open x open,  N_up=N_dn=4, U=8 (D=245025, |G|=4,  S=0)
                      1 : 4x3 periodic x open (cylinder), 4+4, U=8 (D=245025, |G|=16, S=0)
                      2 : 6x2 periodic x open (ladder), 5+5, U=8 (D=627264, |G|=24, S=1)
                      3 : 4x3 torus, 5+5, U=8 (D=627264, |G|=48)
                      4 : 4x3 torus, 4+4, U=8 (D=245025, |G|=48)
                      5 : 4x3 torus, 5+5, U=4 (D=627264, |G|=48)
                      6 : 3x3 torus, 3+3, U=8 (D=7056; smoke only, never in a verdict)
                    Irreps and in-sector gaps of 0-2 are from P1.2 round 3's probe; every run
                    re-checks [H,U_g], [A,U_g] and P_S P_chi psi_0 = psi_0 and aborts otherwise.
  --seed INT        Trajectory seed (default 101). UHF noise uses seed, NNBF init seed+width
                    (+1000 for the PS arm, +2000 for foils), SR sampling default_rng(seed),
                    MC default_rng(seed+7).
  --task-id INT     If >= 0, OVERRIDES --instance and --seed: instance = t // 6,
                    seed = SEEDS[t % 6] with SEEDS = (101, 102, 103, 104, 105, 106); t in 0..35.
  --sr-iters INT    SR steps per trained state (default 400). Short on purpose: projected
                    SR is nearly exact on these 12-site clusters after ~1500 steps (P1.1-P),
                    and the population must cover the error range that projected NQS have
                    at realistic sizes. Snapshots of one trajectory are not independent; the
                    statistical unit of the verdict is the trajectory seed.
  --snaps LIST      Snapshot iterations of the NNBF populations (default "30,100"; "" off).
  --sr-lr, --sr-shift, --sr-samples, --chunk, --o-chunk   SR settings (defaults 0.05, 1e-3,
                    2000, 65536, 256; P1.1 / P1.1-P values).
  --r-min FLOAT     Minimal bond-centre distance in A (default 1.5).
  --populations S   Comma list of populations to train (default
                    "P_w2,P_w8,PS_w8,F_P,F_PS,U_w8").
  --mc-samples INT  Samples per state (default 100000).
  --n-boot INT      Bootstrap resamples per state (default 200).
  --zr-width INT    Width of the response NNBF (default 8).
  --zr-steps INT    Adam steps of the response training (default 300).
  --zr-lr FLOAT     Adam step size (default 0.02).
  --minres-tol FLOAT  Tolerance of the oracle solve (default 1e-8).
  --out PATH        JSON output (default results/inst<K>_s<seed>.json next to this script).
  --overwrite {0,1} Default 0.
  --checks-only {0,1}  Run correctness checks only.
  --aggregate DIR   No computation: apply the decision rule to every inst*_s*.json in DIR.
  --include-smoke {0,1}  Aggregate also instance 6 (smoke only; default 0).
  --min-target-inst INT  Minimum number of target instances for a verdict (default 2, the
                    pre-committed value; 1 is used only to exercise the rule on the smoke run).

Outputs: JSON with per-state records (exact and MC quantities for every arm) and timing.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, minres

import p11_core as P

HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCES = {
    # lower-symmetry clusters (P1.2 round 3; irreps and in-sector gaps from its probe)
    0: dict(Lx=4, Ly=3, bcx="obc", bcy="obc", nup=4, ndn=4, U=8.0, tp=-0.25),
    1: dict(Lx=4, Ly=3, bcx="pbc", bcy="obc", nup=4, ndn=4, U=8.0, tp=-0.25),
    2: dict(Lx=6, Ly=2, bcx="pbc", bcy="obc", nup=5, ndn=5, U=8.0, tp=-0.25),
    # the three P1.1 tori
    3: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=8.0, tp=-0.25),
    4: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=4, ndn=4, U=8.0, tp=-0.25),
    5: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=4.0, tp=-0.25),
    # smoke only
    6: dict(Lx=3, Ly=3, bcx="pbc", bcy="pbc", nup=3, ndn=3, U=8.0, tp=-0.25),
}
N_PROD = 6
SMOKE = 6
SEEDS = (101, 102, 103, 104, 105, 106)
REPORTED = ("U_w8", "UP_w8")
# pre-committed constants
SCORE_REL = 0.02
TRUST_SE = 0.5
C_SIG = 2.0
MIN_SCORED = 6
MIN_FINAL = 3
MIN_TARGET_INST = 2
ZRO_MIN = 0.7
PASS_M, PASS_D, PASS_G, PASS_COV = 0.5, 0.2, 0.10, 0.5
FAIL_M, FAIL_D, FAIL_G = 0.3, 0.05, 0.20
log = P.log


# ============================================================================ symmetry (copied from P1.2 r3 / P1.1-P)
def group(model):
    """Space-group elements T^(a,b) M^(c,d); translations only along periodic directions,
    mirrors x -> -x (periodic) or x -> L-1-x (open). Copied from P1.2 round 3."""
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
    GEN = {(1, 0, 0, 0): "Tx", (0, 1, 0, 0): "Ty", (0, 0, 1, 0): "Mx", (0, 0, 0, 1): "My"}

    def __init__(self, model):
        self.m = model
        st_u, lk_u, _, rows_u = P.species_basis(model.V, model.nup)
        st_d, lk_d, _, rows_d = P.species_basis(model.V, model.ndn)
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
        self.chi = np.array([lam.get("Tx", 1) ** a * lam.get("Ty", 1) ** b * lam["Mx"] ** c * lam["My"] ** d
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
    """S^2 via sparse S^+ and the Loewdin projector (copied from P1.1-P)."""

    def __init__(self, model):
        m = self.m = model
        V, nu, nd = m.V, m.nup, m.ndn
        self.Sz = 0.5 * (nu - nd)
        Ntot = nu + nd
        self.Smax = 0.5 * min(Ntot, 2 * V - Ntot)
        self.S0 = abs(self.Sz)
        su, sd = m.su, m.sd
        st_u2, lk_u2, _, _ = P.species_basis(V, nu + 1)
        st_d2, lk_d2, _, _ = P.species_basis(V, nd - 1)
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
        self.SpT = self.Sp.T.tocsr()

    def S2(self, psi):
        return self.SpT @ (self.Sp @ psi) + (self.Sz ** 2 + self.Sz) * psi

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


# ============================================================================ projected SR (from P1.1-P, generalised)
def _local_parts(ans, flat_idx):
    iu0, id0 = ans.iu, ans.idd
    ans.iu, ans.idd = iu0[flat_idx], id0[flat_idx]
    try:
        n = len(flat_idx)
        sl = slice(0, n)
        Ou, Od, x, h = ans._orbitals(sl)
        Au, Ad, ru, rd = ans._mats(Ou, Od, sl)
        Gu, Gd = ans._inv_T(Au), ans._inv_T(Ad)
        dOu = np.zeros((n, ans.V, ans.nu)); dOd = np.zeros((n, ans.V, ans.nd))
        bi = np.arange(n)[:, None]
        dOu[bi, ru] = Gu
        dOd[bi, rd] = Gd
        jf = ans._jfeat(sl) if ans.jas else None
        return dOu, dOd, x, h, jf
    finally:
        ans.iu, ans.idd = iu0, id0


def per_sample_O_proj(ans, sg, idx, psi_t, psi_P, o_chunk):
    m = ans.m
    Dd, G = m.Dd, sg.G
    out = np.empty((len(idx), ans.n_params()))
    for s in range(0, len(idx), o_chunk):
        ii = idx[s:s + o_chunk]
        B = len(ii)
        a, b = ii // Dd, ii % Dd
        img = (sg.inv_u[:, a] * Dd + sg.inv_d[:, b]).T
        fac = (sg.chi[:, None] * sg.isu[:, a] * sg.isd[:, b]).T / G
        coef = fac * psi_t[img] / psi_P[ii][:, None]
        dOu, dOd, x, h, jf = _local_parts(ans, img.ravel())
        c = coef.ravel()
        parts = {"Pu": (dOu.reshape(B * G, -1) * c[:, None]).reshape(B, G, -1).sum(1),
                 "Pd": (dOd.reshape(B * G, -1) * c[:, None]).reshape(B, G, -1).sum(1)}
        if ans.jas:
            parts["vj"] = (jf * c[:, None]).reshape(B, G, -1).sum(1)
        if ans.kind == "nnbf":
            dout = np.concatenate([dOu.reshape(B * G, -1), dOd.reshape(B * G, -1)], 1)
            nout, H, nin = dout.shape[1], h.shape[1], x.shape[1]
            parts["b2"] = (dout * c[:, None]).reshape(B, G, nout).sum(1)
            hc = (h * c[:, None]).reshape(B, G, H)
            parts["W2"] = np.matmul(hc.transpose(0, 2, 1), dout.reshape(B, G, nout)).reshape(B, -1)
            dpre = (dout @ ans.p["W2"].T) * (1 - h ** 2)
            parts["b1"] = (dpre * c[:, None]).reshape(B, G, H).sum(1)
            xc = (x * c[:, None]).reshape(B, G, nin)
            parts["W1"] = np.matmul(xc.transpose(0, 2, 1), dpre.reshape(B, G, H)).reshape(B, -1)
        out[s:s + B] = np.concatenate([parts[k] for k in ans.keys], 1)
    return out


class Projector:
    """Pi = P_chi (mode 'P') or P_S P_chi (mode 'PS'); symmetric, idempotent, [Pi, H] = 0."""

    def __init__(self, sg, spn, S, mode):
        self.sg, self.spn, self.S, self.mode = sg, spn, S, mode

    def __call__(self, v):
        v = self.sg.project(v)
        return self.spn.project(v, self.S) if self.mode == "PS" else v


def optimize_sr_proj(ans, H, sg, proj, args, rng, label, snaps_at=()):
    """Projected SR: state psi_Pi = Pi psi_theta; exact energy and force of psi_Pi; SR metric
    from the space-group-projected log-derivatives at samples of |P_chi psi_theta|^2 (for
    Pi = P_chi this is exactly the P1.1-P projected SR). Returns {snap: flat} snapshots."""
    snaps = {}
    best_E, best_x = np.inf, ans.flat().copy()
    for it in range(args.sr_iters):
        if it in snaps_at:
            snaps[it] = ans.flat().copy()
        x_before = ans.flat().copy()
        psi_t = ans.amplitudes(args.chunk)
        psi_g = sg.project(psi_t)
        psi = proj(psi_t) if proj.mode == "PS" else psi_g
        E, Hp, nn = P.energy(H, psi)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x); break
        if E < best_E:
            best_E, best_x = E, x_before
        r = (Hp - E * psi) / nn
        f = ans.grad_weighted(psi_t * r, args.chunk)
        pg = psi_g ** 2 / (psi_g @ psi_g)
        idx = rng.choice(len(psi_g), size=args.sr_samples, p=pg)
        O = per_sample_O_proj(ans, sg, idx, psi_t, psi_g, args.o_chunk)
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        lam = args.sr_shift
        Kmat = Oc @ Oc.T + lam * np.eye(len(idx))
        dx = (f - Oc.T @ np.linalg.solve(Kmat, Oc @ f)) / lam
        nrm = np.linalg.norm(dx)
        step = args.sr_lr * dx
        if nrm * args.sr_lr > 1.0:
            step *= 1.0 / (nrm * args.sr_lr)
        ans.set_flat(ans.flat() - step)
        if it % 100 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}")
    E = P.energy(H, proj(ans.amplitudes(args.chunk)))[0]
    if E > best_E:
        ans.set_flat(best_x)
    return snaps


# ============================================================================ order operator (from P3.1 r2)
class PairOp:
    """A = (1/N_pairs) sum_{(b0,b): d >= r_min} eta_b0 eta_b Delta_b0^dag Delta_b, matrix-free."""

    def __init__(self, model, r_min):
        self.m = model
        bonds = model.pair_bonds
        nb = len(bonds)
        eta = np.array([1.0 if b[2] == "x" else -1.0 for b in bonds])
        cen = [model.bond_center(b) for b in bonds]
        W = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                if model.dist(cen[i], cen[j]) >= r_min:
                    W[i, j] = eta[i] * eta[j]
        self.n_pairs = int(np.count_nonzero(W))
        if self.n_pairs == 0:
            raise ValueError("no bond pairs with d >= r_min; lower --r-min")
        self.W = W / self.n_pairs
        self.ops = model.pair_ops

    def deltas(self, psi):
        Psi = psi.reshape(self.m.Du, self.m.Dd)
        return np.stack([(aui @ (adj @ Psi.T).T + auj @ (adi @ Psi.T).T) / np.sqrt(2.0)
                         for (aui, adj, auj, adi) in self.ops])

    def apply(self, psi):
        X = self.deltas(psi)
        Y = np.tensordot(self.W, X, axes=(1, 0))
        out = np.zeros((self.m.Du, self.m.Dd))
        for (aui, adj, auj, adi), Yb in zip(self.ops, Y):
            out += (aui.T @ (adj.T @ Yb.T).T + auj.T @ (adi.T @ Yb.T).T) / np.sqrt(2.0)
        return out.ravel()

    def brute(self, psi):
        X = self.deltas(psi / np.linalg.norm(psi)).reshape(len(self.ops), -1)
        return float(np.sum(self.W * (X @ X.T)))


# ============================================================================ estimators
def first_order_L1(H, psi, Hp, Ap_fn):
    """One Lanczos step (2x2 Ritz in {psi, H psi}), round-2 first-order rule."""
    E = psi @ Hp
    v = Hp - E * psi
    nv = np.linalg.norm(v)
    if nv < 1e-14:
        return 0.0, 0.0
    v = v / nv
    Hv = H @ v
    Hm = np.array([[E, psi @ Hv], [v @ Hp, v @ Hv]])
    Hm = 0.5 * (Hm + Hm.T)
    w, U = np.linalg.eigh(Hm)
    phi = U[0, 0] * psi + U[1, 0] * v
    phi /= np.linalg.norm(phi)
    Hphi = H @ phi
    Ephi = phi @ Hphi
    sig_psi = np.sqrt(max(Hp @ Hp - E * E, 0.0))
    sig_phi = np.sqrt(max(Hphi @ Hphi - Ephi ** 2, 0.0))
    a_psi = psi @ Ap_fn(psi)
    a_phi = phi @ Ap_fn(phi)
    rhat = 1.0 - sig_phi / sig_psi
    return float((a_phi - a_psi) / rhat) if abs(rhat) > 1e-12 else 0.0, float(rhat)


def galerkin(Kt, r, basis):
    """delta_hat = -2 r_S^T M^-1 b in span(basis), M_ij = <v_i|K|v_j>, b_i = <v_i|w> (w = basis[0])."""
    Vb = np.array(basis)
    KV = np.array([Kt(v) for v in basis])
    M = Vb @ KV.T
    M = 0.5 * (M + M.T)
    b = Vb @ basis[0]
    rs = Vb @ r
    try:
        x = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(-2.0 * rs @ x)


def boot_se(x, n_boot, rng):
    """Bootstrap SE of the mean of x."""
    n = len(x)
    ms = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(ms.std(ddof=1))


def tail_diag(x):
    ax = np.abs(x)
    s = ax.sum()
    k = float(((x - x.mean()) ** 4).mean() / max(x.var() ** 2, 1e-300) - 3.0)
    return dict(max_share=float(ax.max() / s) if s > 0 else 0.0, ex_kurt=k)


class ResponseNet:
    """Route 3.2: projected NNBF phi = Pi phi_theta maximising R = <phi|w>^2/<Q phi|K|Q phi>."""

    def __init__(self, m, ans_psi, proj, args, seed):
        self.m, self.proj, self.args = m, proj, args
        self.net = P.Ansatz(m, ans_psi.p["Pu"].copy(), ans_psi.p["Pd"].copy(), "nnbf",
                            width=args.zr_width, jastrow=False, seed=seed)
        self.net.p["W2"] = 0.1 * np.random.default_rng(seed).standard_normal(self.net.p["W2"].shape)

    def train(self, psi, w, Kt):
        a = self.args
        net = self.net
        x = net.flat().copy()
        mom, vel = np.zeros_like(x), np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = (-np.inf, x.copy())
        hist = []
        for it in range(a.zr_steps):
            net.set_flat(x)
            pt = net.amplitudes(a.chunk)
            ph = self.proj(pt)
            qph = ph - (psi @ ph) * psi
            N = ph @ w
            Kq = Kt(qph)
            Dn = qph @ Kq
            if not (np.isfinite(N) and np.isfinite(Dn)) or Dn <= 0:
                break
            R = N * N / Dn
            if R > best[0]:
                best = (R, x.copy())
            if it % 50 == 0:
                hist.append(float(R))
            # gradient of log R w.r.t. the vector phi (in the irrep, so Pi g = g)
            g = 2.0 * w / N - 2.0 * Kq / Dn
            gp = net.grad_weighted(pt * g, a.chunk)
            # Adam ascent
            mom = b1 * mom + (1 - b1) * gp
            vel = b2 * vel + (1 - b2) * gp * gp
            mh = mom / (1 - b1 ** (it + 1)); vh = vel / (1 - b2 ** (it + 1))
            x = x + a.zr_lr * mh / (np.sqrt(vh) + eps)
        net.set_flat(best[1])
        pt = net.amplitudes(a.chunk)
        ph = self.proj(pt)
        qph = ph - (psi @ ph) * psi
        N = ph @ w
        Dn = qph @ Kt(qph)
        return ph, qph, float(N), float(Dn), float(N * N / Dn), hist


def analyze_state(m, Aop, psi_in, ans_psi, proj, psi0, a0, args, rng_mc, label, sg, spn, extra):
    t0 = time.time()
    psi = psi_in / np.linalg.norm(psi_in)
    H = m.H
    Hp = H @ psi
    E = float(psi @ Hp)
    Ap = Aop.apply(psi)
    a = float(psi @ Ap)
    r = Hp - E * psi
    sig2 = float(r @ r)
    w = Ap - a * psi                      # (A - a) psi, orthogonal to psi
    c = float(r @ w)                      # Cov(E_loc, A_loc)
    Kt = lambda v: (lambda q: (H @ q) - E * q - (psi @ ((H @ q) - E * q)) * psi)(v - (psi @ v) * psi)
    delta = a0 - a
    e = psi0 * np.sign(psi0 @ psi) - psi
    # e = psi0 - psi: delta = 2<e|w> - <e|(A - a0)|e> exactly (norm-1 states); first-order part
    delta_lin = float(2.0 * (e @ w))
    rec = dict(label=label, E=E, a=a, a0=a0, delta=delta, delta_lin=delta_lin, sig2=sig2,
               sig2_per_site=sig2 / m.V, c=c, ov0=float((psi0 @ psi) ** 2),
               w_sign=float(np.sum(psi0[np.sign(psi0) != np.sign(psi * np.sign(psi0 @ psi))] ** 2)),
               S2=spn.expect(psi), gen=sg.gen_expect(psi), **extra)
    # exact single-state estimators
    rec["L1"], rec["L1_r"] = first_order_L1(H, psi, Hp, Aop.apply)
    rec["SMA"] = galerkin(Kt, r, [w])
    rec["omega_SMA"] = float((w @ Kt(w)) / (w @ w))
    rec["KR2"] = galerkin(Kt, r, [w, Kt(w)])
    # oracle response (premise)
    op = LinearOperator((m.D, m.D), matvec=Kt, dtype=float)
    ystar, info = minres(op, w, rtol=args.minres_tol, maxiter=4000) if _minres_has_rtol() else \
        minres(op, w, tol=args.minres_tol, maxiter=4000)
    rec["ZRO"] = float(-2.0 * (r @ ystar))
    rec["ZRO_info"] = int(info)
    Rstar = float(w @ ystar)
    rec["Rstar"] = Rstar
    # Route 3.2 response network
    zr = ResponseNet(m, ans_psi, proj, args, seed=args.seed + 9001)
    ph, qph, N, Dn, R, hist = zr.train(psi, w, Kt)
    rec["ZR_capture"] = R / Rstar if Rstar > 0 else float("nan")
    rec["ZR_hist"] = hist
    cstar = N / Dn
    rec["ZR"] = float(-2.0 * cstar * (r @ qph))
    # Monte Carlo (i.i.d. exact draws from |psi|^2)
    p = psi ** 2
    p /= p.sum()
    idx = rng_mc.choice(m.D, size=args.mc_samples, p=p)
    # local estimators (value-identical to per-sample evaluation; see checks)
    HAp, AHp, Hph = H @ Ap, Aop.apply(Hp), H @ ph
    ps = psi[idx]
    El, Al, Fl = Hp[idx] / ps, Ap[idx] / ps, ph[idx] / ps
    HAl, AHl, HFl = HAp[idx] / ps, AHp[idx] / ps, Hph[idx] / ps
    Em, Am = El.mean(), Al.mean()
    # Route 3.1 regressor c = <psi|(H-E)(A-a)|psi>, ONE-SIDED local estimator (single 1/psi
    # tail, cost N_H N_A per sample), symmetrised over HA and AH:
    #   X(n) = [(HA + AH)/2 psi](n)/psi(n) - a E_loc(n) - E A_loc(n) + E a
    c_terms = 0.5 * (HAl + AHl) - Am * El - Em * Al + Em * Am
    # reported: TWO-SIDED product estimator (E_loc - E)(A_loc - a) (1/psi^2 tail, cost N_H + N_A)
    c2_terms = (El - Em) * (Al - Am)
    # Route 3.2 contraction <psi|(H-E)|phi>, one-sided: ((H phi)(n) - E phi(n))/psi(n)
    zr_terms = HFl - Em * Fl
    zr2_terms = (El - Em) * Fl
    rb = np.random.default_rng(args.seed + 31)
    mc = dict(E=float(Em), a=float(Am), sig2=float(El.var()), c=float(c_terms.mean()),
              a_se=boot_se(Al, args.n_boot, rb), c_se=boot_se(c_terms, args.n_boot, rb),
              c2=float(c2_terms.mean()), c2_se=boot_se(c2_terms, args.n_boot, rb),
              zr_contr=float(zr_terms.mean()), zr_se=boot_se(zr_terms, args.n_boot, rb) * abs(2 * cstar),
              ZR2=float(-2.0 * cstar * zr2_terms.mean()),
              c_tail=tail_diag(c_terms), c2_tail=tail_diag(c2_terms), zr_tail=tail_diag(zr_terms),
              zr2_tail=tail_diag(zr2_terms), n=int(args.mc_samples))
    mc["delta"] = a0 - mc["a"]
    mc["sig2_per_site"] = mc["sig2"] / m.V
    mc["ZR"] = float(-2.0 * cstar * mc["zr_contr"])
    rec["mc"] = mc
    rec["t_analyze"] = time.time() - t0
    log(f"{label:>10s}: a={a:+.5e} delta={delta:+.3e} (lin {delta_lin:+.3e}) ov0={rec['ov0']:.5f} "
        f"S2={rec['S2']:.2e} c={c:+.3e} [mc {mc['c']:+.3e}+-{mc['c_se']:.1e}; 2-sided {mc['c2']:+.3e}+-{mc['c2_se']:.1e}] | L1 {rec['L1']:+.2e} "
        f"SMA {rec['SMA']:+.2e} KR2 {rec['KR2']:+.2e} ZRO {rec['ZRO']:+.2e} ZR {rec['ZR']:+.2e} "
        f"[mc {mc['ZR']:+.2e}+-{mc['zr_se']:.1e}; 2-sided {mc['ZR2']:+.2e}] capt {rec['ZR_capture']:.2f} ({rec['t_analyze']:.0f}s)")
    return rec


def _minres_has_rtol():
    import inspect
    return "rtol" in inspect.signature(minres).parameters


# ============================================================================ checks
def run_checks(seed):
    ok = P.run_checks(seed)
    c = INSTANCES[SMOKE]
    m = P.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])
    m.pair_setup(1.0)
    # open and cylinder group operators commute with H (lower-symmetry production clusters)
    for (Lx, Ly, bcx, bcy) in [(3, 2, "obc", "obc"), (3, 2, "pbc", "obc")]:
        mm = P.Hubbard(Lx, Ly, bcx, bcy, 2, 2, 1.0, -0.25, 8.0)
        sgo = SpaceGroup(mm)
        vv = np.random.default_rng(seed).standard_normal(mm.D)
        e_c = max(np.linalg.norm(sgo.apply(mm.H @ vv, k) - mm.H @ sgo.apply(vv, k)) for k in range(sgo.G))
        log(f"check group {Lx}x{Ly} {bcx}/{bcy}: |G|={sgo.G} max ||[H,U_g]v|| {e_c:.1e}")
        ok &= e_c < 1e-10
    W, Vv, deg = P.solve_ed(m, 3, 1e-8, seed)
    psi0 = Vv[:, 0]
    sg = SpaceGroup(m)
    sg.set_irrep(sg.irrep_of(psi0))
    spn = Spin(m)
    S0 = round(0.5 * (np.sqrt(1 + 4 * spn.expect(psi0)) - 1))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(m.D)
    for mode in ("P", "PS"):
        pr = Projector(sg, spn, S0, mode)
        Pv = pr(v)
        e_idem = np.linalg.norm(pr(Pv) - Pv) / np.linalg.norm(Pv)
        e_hc = np.linalg.norm(pr(m.H @ v) - m.H @ Pv) / np.linalg.norm(m.H @ v)
        e_0 = np.linalg.norm(pr(psi0) - psi0)
        log(f"check projector {mode}: idempotent {e_idem:.1e} [Pi,H] {e_hc:.1e} ||Pi psi0 - psi0|| {e_0:.1e}")
        ok &= e_idem < 1e-8 and e_hc < 1e-8 and e_0 < 1e-8
    Aop = PairOp(m, 1.0)
    y = rng.standard_normal(m.D)
    e_herm = abs(y @ Aop.apply(v) - v @ Aop.apply(y)) / (np.linalg.norm(v) * np.linalg.norm(y))
    e_br = abs(v @ Aop.apply(v) / (v @ v) - Aop.brute(v))
    e_comm = max(np.linalg.norm(sg.apply(Aop.apply(v), k) - Aop.apply(sg.apply(v, k))) for k in range(sg.G))
    e_spin = np.linalg.norm(spn.S2(Aop.apply(v)) - Aop.apply(spn.S2(v))) / np.linalg.norm(v)
    log(f"check A: hermitian {e_herm:.1e} vs bond sum {e_br:.1e} [A,U_g] {e_comm:.1e} [A,S2] {e_spin:.1e}")
    ok &= e_herm < 1e-12 and e_br < 1e-12 and e_comm < 1e-10 and e_spin < 1e-10
    # per-sample local estimators equal the indexed full-vector values
    psi = psi0 + 0.05 * sg.project(rng.standard_normal(m.D))
    Hp, Ap = m.H @ psi, Aop.apply(psi)
    idx = rng.choice(m.D, size=4, p=psi ** 2 / (psi @ psi))
    err = 0.0
    for n in idx:
        row = m.H.getrow(n)
        El = float(row.data @ psi[row.indices]) / psi[n]
        en = np.zeros(m.D); en[n] = 1.0
        Al = float(Aop.apply(en) @ psi) / psi[n]
        err = max(err, abs(El - Hp[n] / psi[n]) / max(1, abs(El)), abs(Al - Ap[n] / psi[n]) / max(1, abs(Al)))
    # one-sided Route-3.1 local estimator (H A psi)(n)/psi(n) by explicit per-sample enumeration
    n = idx[0]
    row = m.H.getrow(n)
    HA_n = 0.0
    for j, hv in zip(row.indices, row.data):
        ej = np.zeros(m.D); ej[j] = 1.0
        HA_n += hv * float(Aop.apply(ej) @ psi)
    HAp = m.H @ Ap
    err = max(err, abs(HA_n - HAp[n]) / max(1.0, abs(HAp[n])))
    log(f"check per-sample E_loc, A_loc, (HA psi)_loc vs indexed full vectors: max rel err {err:.1e}")
    ok &= err < 1e-10
    # covariance identity and first-order formula on a perturbed ground state
    psi = psi / np.linalg.norm(psi)
    E = psi @ (m.H @ psi); a = psi @ Aop.apply(psi)
    rr = m.H @ psi - E * psi; ww = Aop.apply(psi) - a * psi
    p = psi ** 2
    El = (m.H @ psi) / psi; Al = Aop.apply(psi) / psi
    cov_ex = float(np.sum(p * (El - E) * (Al - a)))
    log(f"check Cov(E_loc,A_loc) = <(H-E)psi|(A-a)psi>: {cov_ex:.12e} vs {rr @ ww:.12e}")
    ok &= abs(cov_ex - rr @ ww) < 1e-10 * max(1, abs(cov_ex))
    # structured in-irrep error direction (A psi0 plus 30% random), normalised; a purely
    # random direction has a first-order term suppressed by 1/sqrt(D) and tests nothing
    u = Aop.apply(psi0); u -= (psi0 @ u) * psi0; u /= np.linalg.norm(u)
    z = sg.project(rng.standard_normal(m.D)); z -= (psi0 @ z) * psi0; z /= np.linalg.norm(z)
    u = u + 0.3 * z; u /= np.linalg.norm(u)
    for eps in (1e-2, 1e-3):
        ph = psi0 + eps * u; ph /= np.linalg.norm(ph)
        Hp = m.H @ ph; E = ph @ Hp; Ap = Aop.apply(ph); a = ph @ Ap
        rr, ww = Hp - E * ph, Ap - a * ph
        Kt = lambda q, ph=ph, E=E: (lambda z: m.H @ z - E * z - (ph @ (m.H @ z - E * z)) * ph)(q - (ph @ q) * ph)
        op = LinearOperator((m.D, m.D), matvec=Kt, dtype=float)
        ys = minres(op, ww, rtol=1e-12, maxiter=5000)[0] if _minres_has_rtol() else minres(op, ww, tol=1e-12, maxiter=5000)[0]
        d_true = psi0 @ Aop.apply(psi0) - a
        d_hat = -2 * rr @ ys
        log(f"check first-order response formula eps={eps:g}: delta {d_true:+.6e} oracle {d_hat:+.6e}")
        ok &= abs(d_hat - d_true) < 0.05 * abs(d_true) + 1e-12 if eps < 5e-3 else True
    # Adam response training: gradient of log R vs finite differences
    Pu, Pd = P.uhf_orbitals(m, seed)
    ans = P.Ansatz(m, Pu, Pd, "nnbf", width=2, seed=seed)
    pr = Projector(sg, spn, S0, "P")
    args = argparse.Namespace(zr_width=2, chunk=4096)
    zr = ResponseNet(m, ans, pr, args, seed)
    E = psi @ (m.H @ psi); a = psi @ Aop.apply(psi)
    ww = Aop.apply(psi) - a * psi
    Kt = lambda q, E=E: (lambda z: m.H @ z - E * z - (psi @ (m.H @ z - E * z)) * psi)(q - (psi @ q) * psi)

    def logR(x):
        zr.net.set_flat(x)
        ph = pr(zr.net.amplitudes(4096)); q = ph - (psi @ ph) * psi
        return np.log((ph @ ww) ** 2 / (q @ Kt(q)))
    x0 = zr.net.flat().copy()
    zr.net.set_flat(x0)
    pt = zr.net.amplitudes(4096); ph = pr(pt); q = ph - (psi @ ph) * psi
    N, Kq = ph @ ww, Kt(q)
    g = zr.net.grad_weighted(pt * (2 * ww / N - 2 * Kq / (q @ Kq)), 4096)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    fd = (logR(x0 + h * d) - logR(x0 - h * d)) / (2 * h)
    zr.net.set_flat(x0)
    log(f"check response gradient: analytic {g @ d:.10e} finite-diff {fd:.10e}")
    ok &= abs(g @ d - fd) < 1e-5 * max(1.0, abs(fd))
    log(f"P3.1-r3 CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ run
def make_model(k):
    c = INSTANCES[k]
    return P.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])


def run(args):
    t_start = time.time()
    m = make_model(args.instance)
    m.pair_setup(args.r_min)
    Aop = PairOp(m, args.r_min)
    log(f"instance {args.instance} V={m.V} N=({m.nup},{m.ndn}) D={m.D}; A has {Aop.n_pairs} bond pairs")
    W, Vv, deg = P.solve_ed(m, 3, 1e-8, args.seed)
    if deg != 1:
        raise RuntimeError("degenerate ground state")
    psi0 = Vv[:, 0]
    sg = SpaceGroup(m)
    lam = sg.irrep_of(psi0)
    sg.set_irrep(lam)
    spn = Spin(m)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * spn.expect(psi0)) - 1)))
    a0 = float(psi0 @ Aop.apply(psi0))
    # instance checks: H and A commute with every group element; psi0 is in the irrep
    vv = np.random.default_rng(args.seed).standard_normal(m.D)
    hv, av = np.linalg.norm(m.H @ vv), np.linalg.norm(Aop.apply(vv))
    e_h = max(np.linalg.norm(sg.apply(m.H @ vv, k) - m.H @ sg.apply(vv, k)) for k in range(sg.G)) / hv
    e_a = max(np.linalg.norm(sg.apply(Aop.apply(vv), k) - Aop.apply(sg.apply(vv, k))) for k in range(sg.G)) / av
    e_p = float(np.linalg.norm(Projector(sg, spn, S0, "PS")(psi0) - psi0))
    log(f"ED {W.tolist()}; |G|={sg.G}; irrep {lam}; S(psi0)={S0:g}; a0={a0:+.6e}; "
        f"[H,U_g] {e_h:.1e} [A,U_g] {e_a:.1e} ||P_S P_chi psi0 - psi0|| {e_p:.1e}")
    if max(e_h, e_a, e_p) > 1e-8:
        raise RuntimeError("symmetry checks failed on this instance")
    res = dict(params={k: v for k, v in vars(args).items()}, instance=args.instance, seed=args.seed,
               ed=W.tolist(), irrep=lam, G=sg.G, S0=S0, a0=a0, states=[], timing={})

    def dump():
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)

    rng = np.random.default_rng(args.seed)
    rng_mc = np.random.default_rng(args.seed + 7)
    Pu, Pd = P.uhf_orbitals(m, args.seed)
    prP, prPS = Projector(sg, spn, S0, "P"), Projector(sg, spn, S0, "PS")
    for pop in args.pop_list:
        t0 = time.time()
        if pop.startswith("P_w") or pop.startswith("PS_w"):
            wd = int(pop.split("_w")[1])
            ps = pop.startswith("PS")
            ans = P.Ansatz(m, Pu, Pd, "nnbf", width=wd, jastrow=False, seed=args.seed + wd + (1000 if ps else 0))
            pr = prPS if ps else prP
            snaps = optimize_sr_proj(ans, m.H, sg, pr, args, rng, pop, snaps_at=args.snap_list)
            fam = "nnbf"
        elif pop in ("F_P", "F_PS"):
            ans = P.Ansatz(m, Pu, Pd, "sj", jastrow=True, seed=args.seed + 2000)
            pr = prPS if pop == "F_PS" else prP
            snaps = optimize_sr_proj(ans, m.H, sg, pr, args, rng, pop, snaps_at=())
            fam = "foil"
        elif pop == "U_w8":
            ans = P.Ansatz(m, Pu, Pd, "nnbf", width=8, jastrow=False, seed=args.seed + 8)
            P.optimize_sr(ans, m.H, args, rng, pop)
            snaps, pr, fam = {}, prP, "nnbf"
        else:
            raise ValueError(pop)
        res["timing"][pop] = (time.time() - t0) / max(args.sr_iters, 1)
        todo = []
        x_final = ans.flat().copy()
        for it, x in sorted(snaps.items()):
            todo.append((f"{pop}@{it}", x, pr, fam))
        if pop == "U_w8":
            todo.append(("U_w8", x_final, None, fam))
            todo.append(("UP_w8", x_final, prP, fam))
        else:
            todo.append((pop, x_final, pr, fam))
        for lab, x, prj, fam_ in todo:
            ans.set_flat(x)
            psi_t = ans.amplitudes(args.chunk)
            psi = prj(psi_t) if prj is not None else psi_t
            # the unprojected U state is not in the irrep: its response network is unprojected
            rec = analyze_state(m, Aop, psi, ans, prj if prj is not None else _Identity(),
                                psi0, a0, args, rng_mc, lab, sg, spn,
                                dict(family=fam_, population=pop, reported=lab in REPORTED,
                                     n_params=ans.n_params()))
            res["states"].append(rec)
            dump()
        ans.set_flat(x_final)
    res["wall"] = time.time() - t_start
    dump()
    log(f"wrote {args.out} ({res['wall']:.0f}s)")
    return 0


class _Identity:
    mode = "none"

    def __call__(self, v):
        return v


# ============================================================================ aggregate
def _ratio(dh, d):
    return dh / d if d != 0 else float("nan")


def build_estimates(states, mode, rng):
    """Per instance: add ECC, ZV and SD estimates (needs the whole ensemble of seeds)."""
    get = (lambda s, k: s["mc"][k]) if mode == "mc" else (lambda s, k: s[k])
    ver = [s for s in states if not s["reported"]]
    seeds = sorted({s["seed"] for s in ver})
    fams = sorted({s["family"] for s in ver})
    out = {}
    for xkey, name in (("c", "ECC"), ("sig2_per_site", "ZV")):
        def fit(sel):
            X = np.array([[1.0] + [get(s, xkey) if s["family"] == f else 0.0 for f in fams] for s in sel])
            y = np.array([get(s, "a") for s in sel])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            return coef
        coef = fit(ver)
        boots = []
        for _ in range(200):
            bs = rng.choice(seeds, size=len(seeds), replace=True)
            sel = [s for b in bs for s in ver if s["seed"] == b]
            try:
                boots.append(fit(sel))
            except np.linalg.LinAlgError:
                pass
        boots = np.array(boots)
        se = boots.std(0, ddof=1) if len(boots) > 2 else np.full(len(coef), np.inf)
        for s in states:
            if s["family"] not in fams:
                continue
            j = 1 + fams.index(s["family"])
            beta = coef[j]
            dh = -beta * get(s, xkey)
            trusted = bool(abs(beta) > 0 and se[j] / abs(beta) <= TRUST_SE)
            if name == "ECC" and mode == "mc":
                trusted &= abs(s["mc"]["c"]) >= C_SIG * s["mc"]["c_se"]
            out.setdefault(id(s), {})[name] = (float(dh), trusted)
        out[f"{name}_a0hat"] = float(coef[0])
    # SD: seed spread per population label
    labels = sorted({s["label"] for s in states})
    for lab in labels:
        grp = [s for s in states if s["label"] == lab]
        sd = float(np.std([get(s, "a") for s in grp], ddof=1)) if len(grp) > 1 else float("nan")
        for s in grp:
            out.setdefault(id(s), {})["SD"] = (sd, len(grp) > 1)
    return out


def score_states(states, mode, rng):
    est = build_estimates(states, mode, rng)
    rows = []
    for s in states:
        d = s["delta"]  # truth is exact always
        e = dict(est.get(id(s), {}))
        if mode == "mc":
            zr_tr = s["mc"]["zr_se"] <= TRUST_SE * abs(s["mc"]["ZR"]) if s["mc"]["ZR"] != 0 else False
            e["ZR"] = (s["mc"]["ZR"], bool(zr_tr))
        else:
            e["ZR"] = (s["ZR"], True)
        for k in ("L1", "SMA", "KR2", "ZRO"):
            e[k] = (s[k], np.isfinite(s[k]))
        rows.append(dict(state=s, est=e, scored=(not s["reported"]) and abs(d) >= SCORE_REL * abs(s["a0"])))
    return rows, est


def classify(arm, row):
    d = row["state"]["delta"]
    dh, tr = row["est"].get(arm, (float("nan"), False))
    if not tr or not np.isfinite(dh):
        return "abstain"
    q = abs(dh) / abs(d) if arm == "SD" else dh / d
    if 0.5 <= q <= 2.0:
        return "tracked"
    if arm != "SD" and q < 0.5:
        return "false"
    return "over"


ARMS = ("ECC", "ZR", "SD", "ZV", "L1", "SMA", "KR2", "ZRO")
ROUTES = ("ECC", "ZR")
CONTROLS = ("SD", "ZV", "L1")


def metrics(rows_all):
    units = sorted({(r["state"]["instance"], r["state"]["seed"]) for r in rows_all if r["scored"]})
    T = {a: {} for a in ARMS}
    for u in units:
        sel = [r for r in rows_all if r["scored"] and (r["state"]["instance"], r["state"]["seed"]) == u]
        for a in ARMS:
            T[a][u] = np.mean([classify(a, r) == "tracked" for r in sel])
    scored = [r for r in rows_all if r["scored"]]
    met = {}
    for a in ARMS:
        cl = [classify(a, r) for r in scored]
        nn = [r for r in scored if r["state"]["family"] == "nnbf"]
        met[a] = dict(M=float(np.median(list(T[a].values()))) if units else float("nan"),
                      G=float(np.mean([c == "false" for c in cl])) if cl else float("nan"),
                      Cov=float(np.mean([classify(a, r) != "abstain" for r in nn])) if nn else float("nan"),
                      n=len(cl))
        dh = [abs(r["est"].get(a, (np.nan,))[0]) for r in scored]
        dd = [abs(r["state"]["delta"]) for r in scored]
        ok = [i for i in range(len(dh)) if np.isfinite(dh[i])]
        met[a]["spearman"] = _spearman([dh[i] for i in ok], [dd[i] for i in ok]) if len(ok) > 2 else float("nan")
    return T, met, units


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def aggregate(d, include_smoke=False):
    d = d if os.path.isabs(d) else os.path.join(HERE, d)
    files = sorted(glob.glob(os.path.join(d, "inst*_s*.json")))
    by_inst = {}
    for f in files:
        r = json.load(open(f))
        if r["instance"] == SMOKE and not include_smoke:
            continue
        for s in r["states"]:
            s["instance"], s["seed"] = r["instance"], r["seed"]
        by_inst.setdefault(r["instance"], []).extend(r["states"])
    if not by_inst:
        print("no results"); return 1
    rng = np.random.default_rng(12345)
    verdicts = {}
    for mode in ("exact", "mc"):
        rows_all = []
        print(f"===== mode {mode} =====")
        for K, states in sorted(by_inst.items()):
            rows, est = score_states(states, mode, rng)
            rows_all += rows
            nsc = sum(r["scored"] for r in rows)
            nseed = len({r["state"]["seed"] for r in rows if r["scored"]})
            print(f"instance {K}: states {len(states)}, scored {nsc} over {nseed} seeds; "
                  f"a0={states[0]['a0']:+.5e}; ECC a0_hat={est['ECC_a0hat']:+.5e}; ZV a0_hat={est['ZV_a0hat']:+.5e}")
            for lab in sorted({s['label'] for s in states}):
                g = [r for r in rows if r["state"]["label"] == lab]
                dl = np.median([r["state"]["delta"] for r in g])
                q = {a: np.median([_ratio(r["est"].get(a, (np.nan,))[0], r["state"]["delta"]) for r in g]) for a in ("ECC", "ZR", "ZRO", "L1", "SMA")}
                om = np.median([-2 * r["state"]["c"] / r["state"]["delta"] if r["state"]["delta"] else np.nan for r in g])
                print(f"   {lab:>9s} n={len(g)} med delta={dl:+.2e} ov0={np.median([r['state']['ov0'] for r in g]):.5f} "
                      f"med ratio dhat/delta: " + " ".join(f"{a}={v:+.2f}" for a, v in q.items())
                      + f"  omega_k={om:.3f}  ZR capture={np.median([r['state']['ZR_capture'] for r in g]):.2f}"
                      + ("  [reported]" if g[0]["state"]["reported"] else ""))
        T, met, units = metrics(rows_all)
        print(f"arm   M      G      Cov(NNBF)  spearman  n_scored   (units={len(units)})")
        for a in ARMS:
            x = met[a]
            print(f"{a:4s} {x['M']:.3f}  {x['G']:.3f}  {x['Cov']:.3f}      {x['spearman']:+.3f}    {x['n']}")
        if mode != "mc":
            continue
        # verdict (mc mode)
        # premise gate on the size of the residual (NO-TARGET): an instance has a target iff it
        # has >= MIN_SCORED scored states over >= 3 seeds, of which >= MIN_FINAL are final
        # (best-energy, not snapshot) states. Verdicts use target instances only.
        insts = []
        for K in sorted(by_inst):
            sc = [r for r in rows_all if r["scored"] and r["state"]["instance"] == K]
            nfin = sum("@" not in r["state"]["label"] for r in sc)
            fin_err = [abs(r["state"]["delta"]) / abs(r["state"]["a0"]) for r in rows_all
                       if r["state"]["instance"] == K and not r["state"]["reported"] and "@" not in r["state"]["label"]]
            ok_t = len(sc) >= MIN_SCORED and len({r["state"]["seed"] for r in sc}) >= 3 and nfin >= MIN_FINAL
            print(f"target gate: instance {K}: scored {len(sc)} (final {nfin}); median |delta|/|a0| of final "
                  f"states {np.median(fin_err) if fin_err else float('nan'):.2e} -> {'TARGET' if ok_t else 'NO-TARGET'}")
            if ok_t:
                insts.append(K)
        guard_ok = len(insts) >= MIN_TARGET_INST
        rows_t = [r for r in rows_all if r["state"]["instance"] in insts]
        if rows_t:
            T, met, units = metrics(rows_t)
            print(f"verdict metrics on target instances {insts} (units={len(units)}):")
            for a in ARMS:
                print(f"{a:4s} M={met[a]['M']:.3f} G={met[a]['G']:.3f} Cov={met[a]['Cov']:.3f}")
        else:
            units = []
        B = max(CONTROLS, key=lambda a: (met[a]["M"], -CONTROLS.index(a)))
        for X in ROUTES:
            diffs = np.array([T[X][u] - T[B][u] for u in units])
            Dm = float(np.median(diffs)) if len(diffs) else float("nan")
            bs = [np.median(rng.choice(diffs, len(diffs))) for _ in range(1000)] if len(diffs) else [np.nan]
            lo, hi = np.percentile(bs, [5, 95])
            per_inst = {K: float(np.median([T[X][u] - T[B][u] for u in units if u[0] == K])) if any(u[0] == K for u in units) else float("nan") for K in insts}
            x = met[X]
            if not guard_ok:
                v = f"NO-TARGET (fewer than {MIN_TARGET_INST} instances with a non-trivial residual)"
            elif X == "ZR" and met["ZRO"]["M"] < ZRO_MIN:
                v = "FAIL (premise: oracle first-order response does not track)"
            elif x["M"] >= PASS_M and Dm >= PASS_D and x["G"] <= PASS_G and x["Cov"] >= PASS_COV and all(v_ > 0 for v_ in per_inst.values()):
                v = "PASS"
            elif x["M"] < FAIL_M or Dm < FAIL_D or x["G"] > FAIL_G:
                v = "FAIL"
            else:
                v = "AMBIGUOUS"
            verdicts[X] = v
            wins = int(np.sum(diffs > 0)); losses = int(np.sum(diffs < 0))
            print(f"{X}: M={x['M']:.3f} vs best control {B} M={met[B]['M']:.3f}; paired median Delta={Dm:+.3f} "
                  f"[90% {lo:+.3f},{hi:+.3f}] wins/losses {wins}/{losses}; per-instance "
                  + " ".join(f"{K}:{v_:+.2f}" for K, v_ in per_inst.items()) + f"; G={x['G']:.3f} Cov={x['Cov']:.3f} -> {v}")
    print("VERDICT (mc mode): " + ";  ".join(f"{k} (route 3.{1 + ROUTES.index(k)}) = {v}" for k, v in verdicts.items()))
    return 0


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P3.1 round-3 decisive test (see module docstring)")
    ap.add_argument("--instance", type=int, default=1, choices=sorted(INSTANCES))
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--sr-iters", type=int, default=400)
    ap.add_argument("--snaps", type=str, default="30,100")
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--o-chunk", type=int, default=256)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--populations", type=str, default="P_w2,P_w8,PS_w8,F_P,F_PS,U_w8")
    ap.add_argument("--mc-samples", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--zr-width", type=int, default=8)
    ap.add_argument("--zr-steps", type=int, default=300)
    ap.add_argument("--zr-lr", type=float, default=0.02)
    ap.add_argument("--minres-tol", type=float, default=1e-8)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--checks-only", type=int, choices=[0, 1], default=0)
    ap.add_argument("--aggregate", type=str, default="")
    ap.add_argument("--include-smoke", type=int, choices=[0, 1], default=0)
    ap.add_argument("--min-target-inst", type=int, default=2)
    a = ap.parse_args(argv)
    if a.task_id >= 0:
        if a.task_id > 6 * N_PROD - 1:
            ap.error(f"--task-id must be in 0..{6 * N_PROD - 1}")
        a.instance = a.task_id // 6
        a.seed = SEEDS[a.task_id % 6]
    a.pop_list = [p.strip() for p in a.populations.split(",") if p.strip()]
    a.snap_list = tuple(int(x) for x in a.snaps.split(",") if x.strip() and int(x) > 0)
    if not a.out:
        a.out = os.path.join(HERE, "results", f"inst{a.instance}_s{a.seed}.json")
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        global MIN_TARGET_INST
        MIN_TARGET_INST = args.min_target_inst
        return aggregate(args.aggregate, bool(args.include_smoke))
    log("args: " + json.dumps({k: v for k, v in vars(args).items()}))
    if not run_checks(args.seed):
        log("correctness checks failed; aborting")
        return 2
    if args.checks_only:
        return 0
    if os.path.exists(args.out) and not args.overwrite:
        log(f"output {args.out} exists; use --overwrite 1. Exiting.")
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
