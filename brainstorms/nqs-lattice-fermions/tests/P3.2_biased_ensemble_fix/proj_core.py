#!/usr/bin/env python3
"""
P1.1-P decisive test: symmetry-projected sign budget of NNBF on the P1.1 clusters
==================================================================================

Question
--------
P1.1 found that most of the NNBF energy error on three 4x3 PBC Hubbard clusters
(t'=-0.25) is sign error (f_E = dE_sign/dE_tot = 0.61-0.77). Its NNBF states are not
symmetry-projected and start from a broken-symmetry UHF determinant, and a P1.3
pre-diagnostic showed that projecting one width-32 state onto psi_0's space-group
irrep removes 77% of its sign error. This test asks: once the NNBF state is projected
onto psi_0's irrep (post hoc, or trained projected), how much of its energy and
pair-correlation error is still sign error?

Physics core (Hubbard model, ED, UHF, NNBF ansatz, exact-force SR, E_sign/E_FN/E_stoq
budget, f_C) is p11_core.py in this folder, a verbatim copy of
tests/P1.1_sign_vs_amplitude/micro_test.py. Symmetry maps are copied from
tests/P1.3_wrong_sign_basins_r2 (group, species_perm, apply_op). Nothing is imported
across folders.

What one run does (one instance, one seed, one arm; one Slurm array task)
------------------------------------------------------------------------
1. Correctness checks (seconds, on a 3x3 PBC 3+3 cluster and the P1.1 checks); the run
   aborts with exit code 2 if any fails:
     * P1.1 checks (two-site analytic energy, U=0 Slater = ED, NNBF gradient vs FD);
     * space group: max ||[H, U_g] v|| over all g; projector P_chi idempotent
       (||P P v - P v||), symmetric (|x.P y - P x.y|), commuting with H; the irrep
       projection of psi_0 has overlap 1 with psi_0;
     * spin: <S^2> of the two-site singlet = 0 and of psi_0 = S(S+1) with integer S;
       [H, S^2] = 0; [S^2, U_g] = 0; the Loewdin projector P_S is idempotent and gives
       <S^2> = S(S+1) for S in {S0, S_psi0}; P_S psi_0 = psi_0;
     * projected SR: the exact gradient of E[P_chi psi_theta] matches central finite
       differences; the sampled projected log-derivatives O_P(n) match finite
       differences of log psi_P(n).
2. Builds the instance (full fixed-(N_up, N_dn) basis), runs Lanczos ED (3 levels),
   reads psi_0's real 1-D space-group irrep chi (eigenvalues of Tx, Ty, Mx, My; the run
   stops if psi_0 is not in a real 1-D irrep), and builds:
     * P_chi = (1/|G|) sum_g chi(g) U_g over the full space group G = translations x
       {1, Mx, My, MxMy} (|G| = 48 on 4x3; Mx: x -> -x, My: y -> -y). This is the whole
       point group a rectangular torus admits (C2v); no rotations for Lx != Ly;
     * the exact total-spin Loewdin projector P_S onto psi_0's total spin S_psi0 (read
       from <S^2>(psi_0); it is S = 0 on the three production instances, recorded as
       "S_psi0" in the JSON), P_S = prod_{s = S0..S_max, s != S_psi0} (S^2 - s(s+1)) /
       (S_psi0(S_psi0+1) - s(s+1)), S0 = |N_up - N_dn|/2, with S^2 = S^- S^+ + S_z^2 +
       S_z and S^+ a sparse matrix (never a dense D x D one). This full-vector projector
       is an oracle diagnostic (ED-sized test), not a method claim.
3. Optimizer calibration (P1.1 rule): E_sign with the EXACT signs s_0 from a uniform start
   must equal E_0; calib_err = E_sign(s_0) - E_0.
4. RNG replay (--rng-replay 1): P1.1 trained the Slater-Jastrow foil before NNBF with the
   same RNG; its 1500 SR metric draws are replayed (rng.random(sr_samples) per step,
   which is exactly what Generator.choice with p consumes), so arm U with seed 1234
   follows P1.1's sampling stream. Floating-point differences from a different thread
   count can still make trajectories diverge; the cross-check in --aggregate is therefore
   a tolerance check, not a bitwise one.
5. Trains NNBF for every width in --widths from the UHF start (UHF seed = --seed, NNBF
   weights seed = --seed + width, exactly as P1.1), with P1.1's SR (exact force, metric
   from --sr-samples configurations drawn exactly from |psi|^2, step --sr-lr, shift
   --sr-shift, trust region |step| <= 1, best-energy iterate kept):
     arm U  plain NNBF psi_theta, P1.1 protocol unchanged (p11_core.optimize_sr).
     arm P  projected SR: the variational state is psi_P = P_chi psi_theta, i.e.
            psi_P(n) = (1/|G|) sum_g chi(g) sigma_g(n) psi_theta(g^-1 n) (sigma_g the
            fermionic sign); energy, force and metric are those of psi_P. Because every
            amplitude is computed exactly on the full basis, psi_P costs one evaluation
            of psi_theta on the basis plus |G| permutations of a D-vector; the force is
            sum_n r(n) d psi_theta(n) with r = (H - E) psi_P / ||psi_P||^2 (P_chi is
            symmetric, idempotent and commutes with H). Only the sampled SR metric needs
            the |G| images of each sample: O_P(n) = sum_g c_g(n) O_theta(g^-1 n),
            c_g = chi(g) sigma_g psi_theta(g^-1 n) / (|G| psi_P(n)). This is the
            exact-sum version of standard symmetry-projected NQS training (e.g. Romero,
            Nys & Carleo, arXiv:2406.09077).
   Trained parameters are saved to results/params/<tag>_w<W>.npz.
6. Budgets (p11_core.budget, P1.1 definitions: E_sign upper bound from 4 L-BFGS-B starts
   plus ten Haaf fixed node, E_FN, E_stoq, f_E_up, f_E_lo, spread, w_sign, f_C) for
   these states of every width:
     arm U: "U"     psi_theta                (reproduces P1.1),
            "UP"    P_chi psi_theta          (post-hoc projection, arm U->P),
            "U_S"  P_S psi_theta,
            "UP_S" P_S P_chi psi_theta;
     arm P: "P"     psi_P,
            "P_S"  P_S psi_P.
   Each budget also stores <S^2>, |<psi_0|psi>|^2, <Tx>, <Ty>, <Mx>, <My>, and (for UP)
   the irrep weight ||P_chi psi||^2/||psi||^2. A projection with norm < 1e-10 is
   recorded as null.
7. Writes JSON results/<tag>.json (rewritten after every budget), tag =
   inst<K>_s<seed>_arm<A>.

Metric
------
  f_E = dE_sign / dE_tot (P1.1's f_E_up, an upper bound on the sign fraction; dE_tot and
  dE_sign are those of the state being budgeted), f_C = eC_sign / eC_tot (P1.1 d-wave
  bond-pair correlation definition), and the absolute dE_sign = E_sign - E_0, since
  projection can lower both numerator and denominator.

Pre-committed decision rule (applied by --aggregate; fixed before any results exist)
------------------------------------------------------------------------------------
  Per instance K in {0, 1, 2}, using the WIDEST width W present in both arms and seeds
  S_K = seeds present for both arms (needs >= 3, else the instance is INCOMPLETE):
    fP     = median over S_K of f_E(P, W)               (arm P, no spin projection),
    ratio  = median_S dE_sign(P, W) / median_S dE_sign(U, W),
    guards = for every seed in S_K: calib_err <= 1e-3 * min(dE_tot(U, W), dE_tot(P, W))
             AND spread(U, W) < 0.05 AND spread(P, W) < 0.05.
    robust(K) = guards AND fP >= 0.5 AND ratio >= 0.5,
    symdom(K) = guards AND (ratio <= 0.3 OR fP <= 0.3).
  Global verdict:
    SIGN-ROBUST          if robust(0) AND robust(2) (both 5+5 instances). The P1.1
                         conclusion stands for near-node errors.
    SYMMETRY-DOMINATED   if symdom(0) AND symdom(1) AND symdom(2). The sign branch reduces
                         mostly to symmetry restoration.
    AMBIGUOUS            otherwise; pre-committed action: report per instance and treat
                         the sign branch as live only on instances with robust(K).
  (A failed guard or missing data makes that instance AMBIGUOUS: it can be neither robust
  nor symdom. The two global conditions are mutually exclusive.)
  Reported, not part of the verdict: the same quantities for U->P, for the spin-projected
  states (P_S vs U), <S^2> per arm, f_C medians, and the arm-U seed-1234 cross-check
  against P1.1 (p11_reference/*.json): relative differences of dE_tot and dE_sign per
  width, "consistent" if both are within 15%.

Usage
-----
    python3 micro_test.py [options]

Options (all optional)
----------------------
  --instance K       0, 1, 2 or 3 (default 1). Pre-committed clusters (all t=1):
                       0 : 4x3 PBC, N_up=N_dn=5, U=8, t'=-0.25  (D = 627264)
                       1 : 4x3 PBC, N_up=N_dn=4, U=8, t'=-0.25  (D = 245025)
                       2 : 4x3 PBC, N_up=N_dn=5, U=4, t'=-0.25  (D = 627264)
                       3 : 3x3 PBC, N_up=N_dn=3, U=8, t'=-0.25  (D = 7056; smoke only,
                           group = translations + mirrors, |G| = 36, no C4 rotations)
  --arm {U,P}        U: plain NNBF (P1.1 protocol) plus post-hoc projections U->P and
                     spin projections; P: projected-SR NNBF plus its spin projection.
                     Default U.
  --task-id INT      If >= 0, OVERRIDES --instance, --seed and --arm from a Slurm array
                     index t in 0..17: instance = t // 6, seed = (1234, 2345, 3456)[(t // 2)
                     % 3], arm = ("U", "P")[t % 2]. Default -1 (off).
  --widths LIST      Comma-separated NNBF hidden widths (default "8,32,128"). The largest
                     is the one used by the decision rule.
  --sr-iters INT     SR iterations per width (default 1500, P1.1 setting).
  --sr-lr FLOAT      SR step size (default 0.05, P1.1).
  --sr-shift FLOAT   SR diagonal shift (default 1e-3, P1.1).
  --sr-samples INT   Configurations drawn exactly from |psi|^2 for the SR metric
                     (default 2000, P1.1). The force is always exact.
  --chunk INT        Configurations per forward/backward chunk (default 65536).
  --o-chunk INT      Samples per chunk when building projected log-derivatives O_P
                     (arm P only; each sample has |G| images; default 256).
  --lbfgs-iters INT  L-BFGS-B iterations per start in the budget (default 4000, P1.1).
  --fn-iters INT     Self-consistent ten Haaf iterations (default 10, P1.1).
  --fn-floor FLOAT   Fixed-node guide floor (default 1e-3, P1.1).
  --r-min FLOAT      Minimal bond distance for the pairing error (default 1.5, P1.1).
  --rng-replay {0,1} 1 (default) advances the SR RNG by P1.1's foil draws (sr_iters x
                     rng.random(sr_samples)) before NNBF, so seed 1234 follows P1.1's
                     stream; 0 skips the replay.
  --seed INT         Seed (default 1234). UHF noise uses seed, NNBF init seed + width,
                     SR sampling default_rng(seed), budgets default_rng(seed + 1000003).
  --out PATH         JSON path (default results/inst<K>_s<seed>_arm<A>.json next to this
                     script). Existing files are NOT overwritten unless --overwrite 1.
  --overwrite {0,1}  Default 0.
  --checks-only {0,1}  1 runs only the correctness checks of step 1 and exits.
  --time-step {0,1}  1 times ONE SR step (no ED, no budget; trivial irrep chi = 1, which
                     costs the same as any irrep) for --arm on --instance for every width
                     in --widths, prints a cost breakdown, and exits. Login-node safe for
                     a single step.
  --smoke-dir PATH   Output directory override for the results (default "results");
                     relative paths are taken next to this script. Used for smoke runs.
  --aggregate DIR    No computation: read every inst*_s*_arm*.json in DIR (relative to
                     this script if not absolute), apply the decision rule above, print
                     the per-instance table, the diagnostics, the P1.1 cross-check and the
                     GLOBAL VERDICT. Default "" (off).

Outputs
-------
  JSON: "params", "ed" (levels), "irrep" (chi of psi_0), "G" (|G|), "S_psi0", "calib_err",
  "budgets" -> {"<label>_w<W>": P1.1 budget dict + S2, ov0, gen, irrep_weight,
  n_params, train_time_s}, "timing" (seconds per SR step per width).
  NPZ: results/params/<tag>_w<W>.npz with the flat parameter vector and its key layout.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import p11_core as P

HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCES = {
    0: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=8.0, tp=-0.25),
    1: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=4, ndn=4, U=8.0, tp=-0.25),
    2: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, U=4.0, tp=-0.25),
    3: dict(Lx=3, Ly=3, bcx="pbc", bcy="pbc", nup=3, ndn=3, U=8.0, tp=-0.25),
}
SEEDS = (1234, 2345, 3456)
ARMS = ("U", "P")
# pre-committed constants
ROBUST_F, ROBUST_R = 0.5, 0.5
SYMDOM_F, SYMDOM_R = 0.3, 0.3
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05
MIN_SEEDS = 3
XCHECK_TOL = 0.15
P11_REF = {0: "L4x3_pbcpbc_n5-5_U8_tp-0.25_w8-32-128_j0_s1234.json",
           1: "L4x3_pbcpbc_n4-4_U8_tp-0.25_w8-32-128_j0_s1234.json",
           2: "L4x3_pbcpbc_n5-5_U4_tp-0.25_w8-32-128_j0_s1234.json"}
log = P.log


# ============================================================================ symmetry
def group(model):
    """All space-group elements g = T^(a,b) M^(c,d) as (exponents, site perm); copied
    from P1.3 r2 (reflect first, then translate)."""
    Lx, Ly, V = model.Lx, model.Ly, model.V
    out = []
    for c in (0, 1):
        for d in (0, 1):
            for a in range(Lx):
                for b in range(Ly):
                    p = np.empty(V, dtype=np.int64)
                    for y in range(Ly):
                        for x in range(Lx):
                            x2 = (-x) % Lx if c else x
                            y2 = (-y) % Ly if d else y
                            x2, y2 = (x2 + a) % Lx, (y2 + b) % Ly
                            p[x + Lx * y] = x2 + Lx * y2
                    out.append(((a, b, c, d), p))
    return out


def species_perm(states, rows, lookup_V, p):
    """Index map and fermionic sign for one species under site permutation p (copied
    from P1.3 r2 diag_sectors.py)."""
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
    """Unitary representation U_g of the space group on the fixed-(N_up, N_dn) basis and
    the 1-D irrep projector P_chi = (1/|G|) sum_g chi(g) U_g.

    U_g psi (a, b) = isu_g[a] isd_g[b] psi(inv_u_g[a], inv_d_g[b]), a/b up/down indices."""

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
        self.lam = dict(Tx=1, Ty=1, Mx=1, My=1)

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
        self.chi = np.array([lam["Tx"] ** a * lam["Ty"] ** b * lam["Mx"] ** c * lam["My"] ** d
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


# ============================================================================ spin
class Spin:
    """S^2 = S^- S^+ + S_z^2 + S_z on the fixed-(N_up, N_dn) basis via a sparse S^+, and
    the Loewdin projector onto S0 = |S_z|."""

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
            # c^dag_{i up} c_{i dn}: sign (-1)^(N_up + #dn below i + #up below i); the
            # global (-1)^N_up cancels in S^- S^+ and is dropped
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
        """Loewdin projector onto total spin S (S0 <= S <= Smax, S - S0 integer)."""
        out = psi.copy()
        tgt = S * (S + 1)
        s = self.S0
        while s <= self.Smax + 1e-9:
            if abs(s - S) > 1e-9:
                ss = s * (s + 1)
                out = (self.S2(out) - ss * out) / (tgt - ss)
            s += 1
        return out


# ============================================================================ projected SR
def _local_parts(ans, flat_idx):
    """Per-config pieces of dlog psi_theta for basis indices flat_idx (cf. per_sample_O_fast)."""
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
        dOu[bi, ru] = Gu  # rows of one config are distinct, so assignment == add.at
        dOd[bi, rd] = Gd
        jf = ans._jfeat(sl) if ans.jas else None
        return dOu, dOd, x, h, jf
    finally:
        ans.iu, ans.idd = iu0, id0


def per_sample_O_proj(ans, sg, idx, psi_t, psi_P, o_chunk):
    """O_P(n)_k = d log psi_P(n) / d theta_k for sampled basis indices idx, with
    psi_P = P_chi psi_theta: O_P(n) = sum_g c_g(n) O_theta(n_g)."""
    m = ans.m
    Dd, G = m.Dd, sg.G
    out = np.empty((len(idx), ans.n_params()))
    for s in range(0, len(idx), o_chunk):
        ii = idx[s:s + o_chunk]
        B = len(ii)
        a, b = ii // Dd, ii % Dd
        img = (sg.inv_u[:, a] * Dd + sg.inv_d[:, b]).T                       # (B, G)
        fac = (sg.chi[:, None] * sg.isu[:, a] * sg.isd[:, b]).T / G          # (B, G)
        coef = fac * psi_t[img] / psi_P[ii][:, None]                          # (B, G)
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


def proj_energy_grad(ans, H, sg, chunk):
    """E[psi_P] and the P1.1-normalised force f = sum_n r(n) d psi_theta(n),
    r = (H - E) psi_P / ||psi_P||^2 (the true gradient is 2 f)."""
    psi_t = ans.amplitudes(chunk)
    psi = sg.project(psi_t)
    E, Hp, nn = P.energy(H, psi)
    r = (Hp - E * psi) / nn
    f = ans.grad_weighted(psi_t * r, chunk)
    return E, f, psi_t, psi, nn


def sr_step_proj(ans, H, sg, args, rng, tm=None):
    """One projected SR step (P1.1 update rule with psi -> psi_P). Returns E before the step."""
    t0 = time.time()
    E, f, psi_t, psi, nn = proj_energy_grad(ans, H, sg, args.chunk)
    t1 = time.time()
    p = psi ** 2 / nn
    idx = rng.choice(len(psi), size=args.sr_samples, p=p)
    O = per_sample_O_proj(ans, sg, idx, psi_t, psi, args.o_chunk)
    t2 = time.time()
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Kmat = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Kmat, Oc @ f)) / lam
    nrm = np.linalg.norm(dx)
    step = args.sr_lr * dx
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    ans.set_flat(ans.flat() - step)
    if tm is not None:
        tm.update(force=t1 - t0, metric_O=t2 - t1, solve=time.time() - t2)
    return E, f


def optimize_sr_proj(ans, H, sg, args, rng, label):
    """P1.1 optimize_sr with the variational state psi_P = P_chi psi_theta."""
    best_E, best_x = np.inf, ans.flat().copy()
    t0 = time.time()
    for it in range(args.sr_iters):
        x_before = ans.flat().copy()
        E, f = sr_step_proj(ans, H, sg, args, rng)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x); break
        if E < best_E:
            best_E, best_x = E, x_before
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}  "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    E = P.energy(H, sg.project(ans.amplitudes(args.chunk)))[0]
    if E > best_E:
        ans.set_flat(best_x)
    return ans


# ============================================================================ checks
def run_checks(seed):
    ok = P.run_checks(seed)
    c = INSTANCES[3]
    m = P.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])
    W, Vv, deg = P.solve_ed(m, 3, 1e-8, seed)
    psi0 = Vv[:, 0]
    sg = SpaceGroup(m)
    rng = np.random.default_rng(seed)
    v, y = rng.standard_normal(m.D), rng.standard_normal(m.D)
    hv = np.linalg.norm(m.H @ v)
    comm = max(np.linalg.norm(sg.apply(m.H @ v, k) - m.H @ sg.apply(v, k)) / hv for k in range(sg.G))
    log(f"check max ||[H,U_g]v||/||Hv|| over |G|={sg.G}: {comm:.1e}  (ground degeneracy {deg})")
    ok &= comm < 1e-12 and deg == 1
    lam = sg.irrep_of(psi0)
    sg.set_irrep(lam)
    Pv = sg.project(v)
    e_idem = np.linalg.norm(sg.project(Pv) - Pv) / np.linalg.norm(Pv)
    e_sym = abs(v @ sg.project(y) - sg.project(v) @ y) / (np.linalg.norm(v) * np.linalg.norm(y))
    e_hc = np.linalg.norm(sg.project(m.H @ v) - m.H @ Pv) / hv
    p0 = sg.project(psi0)
    ov = abs(psi0 @ p0) / np.linalg.norm(p0)
    log(f"check projector chi={lam}: idempotent {e_idem:.1e}  symmetric {e_sym:.1e}  "
        f"[P,H] {e_hc:.1e}  ||P psi0|| = {np.linalg.norm(p0):.12f}  overlap {ov:.12f}")
    ok &= e_idem < 1e-12 and e_sym < 1e-12 and e_hc < 1e-12 and abs(ov - 1) < 1e-10
    ok &= abs(np.linalg.norm(p0) - 1) < 1e-10
    # spin
    m2 = P.Hubbard(2, 1, "obc", "obc", 1, 1, 1.0, 0.0, 4.0)
    w2, v2 = np.linalg.eigh(m2.H.toarray())
    s2a = Spin(m2).expect(v2[:, 0])
    spn = Spin(m)
    s2 = spn.expect(psi0)
    Sg = 0.5 * (np.sqrt(1 + 4 * s2) - 1)
    e_hs = np.linalg.norm(spn.S2(m.H @ v) - m.H @ spn.S2(v)) / hv
    e_gs = max(np.linalg.norm(spn.S2(sg.apply(v, k)) - sg.apply(spn.S2(v), k)) for k in range(sg.G))
    e_gs /= np.linalg.norm(spn.S2(v))
    ok &= abs(s2a) < 1e-10 and abs(Sg - round(Sg)) < 1e-8 and e_hs < 1e-12 and e_gs < 1e-12
    msg = []
    for St in sorted({float(spn.S0), float(round(Sg))}):
        Ps = spn.project(v, St)
        e_pi = np.linalg.norm(spn.project(Ps, St) - Ps) / np.linalg.norm(Ps)
        s2p = spn.expect(Ps)
        msg.append(f"S={St:g}: P_S idempotent {e_pi:.1e} <S2>(P_S v)={s2p:.6f}")
        ok &= e_pi < 1e-8 and abs(s2p - St * (St + 1)) < 1e-8
    e_p0 = np.linalg.norm(spn.project(psi0, round(Sg)) - psi0)
    log(f"check spin: two-site singlet <S2> = {s2a:.2e}  psi0 <S2> = {s2:.10f} (S = {Sg:.6f})  "
        f"[H,S2] {e_hs:.1e}  [S2,U_g] {e_gs:.1e}  {'; '.join(msg)}  ||P_S psi0 - psi0|| {e_p0:.1e}")
    ok &= e_p0 < 1e-8
    # projected gradient vs FD
    Pu, Pd = P.uhf_orbitals(m, seed)
    a = P.Ansatz(m, Pu, Pd, "nnbf", width=4, jastrow=False, seed=seed)
    a.p["W2"] *= 100
    x0 = a.flat().copy()
    E, f, psi_t, psi, _ = proj_energy_grad(a, m.H, sg, 4096)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    Ef = lambda x: (a.set_flat(x), P.energy(m.H, sg.project(a.amplitudes(4096)))[0])[1]
    fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
    a.set_flat(x0)
    log(f"check projected gradient: analytic {2 * f @ d:.10e}  finite-diff {fd:.10e}")
    ok &= abs(2 * f @ d - fd) < 1e-6 * max(1.0, abs(fd))
    # projected log-derivatives vs FD
    idx = rng.choice(m.D, size=6, p=psi ** 2 / (psi @ psi))
    O = per_sample_O_proj(a, sg, idx, psi_t, psi, 4)
    lp = lambda x: (a.set_flat(x), sg.project(a.amplitudes(4096))[idx])[1]
    fdO = (lp(x0 + h * d) - lp(x0 - h * d)) / (2 * h) / psi[idx]
    a.set_flat(x0)
    eO = np.max(np.abs(O @ d - fdO)) / max(1.0, np.max(np.abs(fdO)))
    log(f"check projected O_P vs finite-diff: max rel err {eO:.1e}")
    ok &= eO < 1e-5
    log(f"P1.1-P CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ timing
def time_step(args):
    m = make_model(args.instance)
    log(f"timing: instance {args.instance} D={m.D} arm {args.arm}")
    t0 = time.time()
    sg = SpaceGroup(m) if args.arm == "P" else None
    log(f"group maps built in {time.time() - t0:.1f}s (|G| = {sg.G if sg else '-'})")
    rng = np.random.default_rng(args.seed)
    Pu, Pd = P.uhf_orbitals(m, args.seed)
    res = {}
    for w in args.width_list:
        a = P.Ansatz(m, Pu, Pd, "nnbf", width=w, seed=args.seed + w)
        tm = {}
        t0 = time.time()
        if sg is None:
            a1 = time.time(); psi = a.amplitudes(args.chunk); a2 = time.time()
            E, Hp, nn = P.energy(m.H, psi)
            f = a.grad_weighted(psi * (Hp - E * psi) / nn, args.chunk); a3 = time.time()
            idx = rng.choice(len(psi), size=args.sr_samples, p=psi ** 2 / nn)
            O = a.per_sample_O_fast(idx); a4 = time.time()
            Oc = (O - O.mean(0)) / np.sqrt(len(idx))
            Kmat = Oc @ Oc.T + args.sr_shift * np.eye(len(idx))
            np.linalg.solve(Kmat, Oc @ f)
            tm = dict(amplitudes=a2 - a1, force=a3 - a1, metric_O=a4 - a3, solve=time.time() - a4)
        else:
            sr_step_proj(a, m.H, sg, args, rng, tm)
        tot = time.time() - t0
        res[w] = dict(total=tot, **tm)
        log(f"width {w:4d} (n_params {a.n_params()}): one SR step {tot:.2f}s  "
            + "  ".join(f"{k}={v:.2f}s" for k, v in tm.items()))
    return res


# ============================================================================ run
def make_model(k):
    c = INSTANCES[k]
    return P.Hubbard(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["U"])


def extra_info(m, sg, spn, psi0, psi):
    psi = psi / np.linalg.norm(psi)
    gen = sg.gen_expect(psi)
    return dict(S2=spn.expect(psi), ov0=float((psi0 @ psi) ** 2), gen=gen)


def run(args):
    out_dir = os.path.dirname(args.out)
    os.makedirs(os.path.join(out_dir, "params"), exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.out))[0]
    rng = np.random.default_rng(args.seed)
    rng_b = np.random.default_rng(args.seed + 1000003)
    m = make_model(args.instance)
    m.pair_setup(args.r_min)
    log(f"model instance {args.instance} V={m.V} N=({m.nup},{m.ndn}) D={m.D} nnz(H)={m.H.nnz}")
    W, Vv, deg = P.solve_ed(m, 3, 1e-8, args.seed)
    log(f"ED levels {W.tolist()}  ground degeneracy {deg}")
    if deg != 1:
        raise RuntimeError("degenerate ground state; the irrep projection is not defined")
    psi0 = Vv[:, 0]
    t0 = time.time()
    sg = SpaceGroup(m)
    lam = sg.irrep_of(psi0)
    sg.set_irrep(lam)
    spn = Spin(m)
    s2_0 = spn.expect(psi0)
    S_psi0 = float(round(0.5 * (np.sqrt(1 + 4 * s2_0) - 1)))
    log(f"space group |G|={sg.G} built in {time.time() - t0:.1f}s; psi_0 irrep {lam}; "
        f"psi_0 <S2> = {s2_0:.3e} (S = {S_psi0:g}); S_max = {spn.Smax}")
    res = dict(params=vars(args), ed=dict(levels=W.tolist(), degeneracy=deg), irrep=lam, G=sg.G,
               S_psi0=S_psi0, budgets={}, timing={})

    def dump():
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)

    # calibration (P1.1 rule; consumes rng.random(D) exactly as P1.1 did)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = P.sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                      [("uniform", np.ones(m.D) + 0.01 * rng.random(m.D))],
                                      args.lbfgs_iters)
    res["calib_err"] = float(Ec - W[0])
    log(f"calibration: E_sign(s_0) - E0 = {res['calib_err']:.3e}")
    dump()
    if args.rng_replay:
        for _ in range(args.sr_iters):
            rng.random(args.sr_samples)
        log(f"RNG replay: advanced by {args.sr_iters} x {args.sr_samples} foil draws")
    Pu, Pd = P.uhf_orbitals(m, args.seed)
    for w in args.width_list:
        a = P.Ansatz(m, Pu, Pd, "nnbf", width=w, jastrow=False, seed=args.seed + w)
        t0 = time.time()
        if args.arm == "U":
            P.optimize_sr(a, m.H, args, rng, f"U nnbf_{w}")
        else:
            optimize_sr_proj(a, m.H, sg, args, rng, f"P nnbf_{w}")
        ttrain = time.time() - t0
        res["timing"][str(w)] = ttrain / max(args.sr_iters, 1)
        np.savez(os.path.join(out_dir, "params", f"{tag}_w{w}.npz"), flat=a.flat(),
                 layout=json.dumps([[k, list(a.p[k].shape)] for k in a.keys]),
                 Pu0=Pu, Pd0=Pd, width=w, seed=args.seed, instance=args.instance, arm=args.arm)
        psi_t = a.amplitudes(args.chunk)
        if args.arm == "U":
            psi_up = sg.project(psi_t)
            wt = float(psi_up @ psi_up / (psi_t @ psi_t))
            states = [("U", psi_t), ("UP", psi_up), ("U_S", spn.project(psi_t, S_psi0)),
                      ("UP_S", spn.project(psi_up, S_psi0))]
        else:
            psi_p = sg.project(psi_t)
            wt = float("nan")
            states = [("P", psi_p), ("P_S", spn.project(psi_p, S_psi0))]
        for lab, psi in states:
            key = f"{lab}_w{w}"
            nrm = np.linalg.norm(psi)
            if not np.isfinite(nrm) or nrm < 1e-10 * max(np.linalg.norm(psi_t), 1e-300):
                res["budgets"][key] = None
                log(f"{key}: projection vanishes; recorded as null")
                dump(); continue
            t1 = time.time()
            b = P.budget(m, psi, W, Vv, deg, args, rng_b)
            b.update(extra_info(m, sg, spn, psi0, psi))
            b["irrep_weight"] = wt if lab == "UP" else None
            b["n_params"] = a.n_params()
            b["train_time_s"] = ttrain
            res["budgets"][key] = b
            log(f"{key:>10s}: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} "
                f"f_E={b['f_E_up']:.3f} f_C={b['f_C']:.3f} spread={b['spread']:.1e} "
                f"S2={b['S2']:.3f} ov0={b['ov0']:.3f} w_sign={b['w_sign']:.3e} "
                f"({time.time() - t1:.0f}s)")
            dump()
    dump()
    log(f"wrote {args.out}")
    return 0


# ============================================================================ aggregate
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def aggregate(d):
    d = d if os.path.isabs(d) else os.path.join(HERE, d)
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "inst*_s*_arm*.json"))):
        r = json.load(open(f))
        p = r["params"]
        R.setdefault(p["instance"], {}).setdefault(p["arm"], {})[p["seed"]] = r
    if not R:
        print("no results"); return 1
    robust, symdom = {}, {}
    print("Pre-committed rule: robust = guards & fP>=0.5 & ratio>=0.5; "
          "symdom = guards & (ratio<=0.3 | fP<=0.3); ratio = med dE_sign(P,W)/med dE_sign(U,W)")
    for K in sorted(set(R) | {0, 1, 2}):  # instance 3 (smoke) is printed, never in the verdict
        arms = R.get(K, {})
        U, Pp = arms.get("U", {}), arms.get("P", {})
        seeds = sorted(set(U) & set(Pp))

        def widths(r):
            return {int(k.split("_w")[1]) for k, v in r["budgets"].items() if v is not None}
        ws = set.intersection(*[widths(U[s]) & widths(Pp[s]) for s in seeds]) if seeds else set()
        if len(seeds) < MIN_SEEDS or not ws:
            print(f"instance {K}: INCOMPLETE (seeds with both arms: {seeds}, common widths "
                  f"{sorted(ws)}) -> AMBIGUOUS")
            robust[K] = symdom[K] = False
            continue
        Wd = max(ws)
        g = lambda arm_r, lab, key: [arm_r[s]["budgets"].get(f"{lab}_w{Wd}", {}) and
                                     arm_r[s]["budgets"][f"{lab}_w{Wd}"][key] for s in seeds]
        guard_msgs = []
        for s in seeds:
            bu, bp = U[s]["budgets"][f"U_w{Wd}"], Pp[s]["budgets"][f"P_w{Wd}"]
            cal = max(U[s]["calib_err"], Pp[s]["calib_err"])
            if cal > CALIB_TOL * min(bu["dE_tot"], bp["dE_tot"]):
                guard_msgs.append(f"seed {s}: calib {cal:.1e}")
            if not (bu["spread"] < SPREAD_TOL and bp["spread"] < SPREAD_TOL):
                guard_msgs.append(f"seed {s}: spread U {bu['spread']:.2e} P {bp['spread']:.2e}")
        guards = not guard_msgs
        fP = _med(g(Pp, "P", "f_E_up"))
        dsU, dsP = _med(g(U, "U", "dE_sign")), _med(g(Pp, "P", "dE_sign"))
        ratio = dsP / dsU if dsU > 0 else float("nan")
        robust[K] = guards and fP >= ROBUST_F and ratio >= ROBUST_R
        symdom[K] = guards and (ratio <= SYMDOM_R or fP <= SYMDOM_F)
        print(f"instance {K} {INSTANCES[K]}  W={Wd}  seeds={seeds}")
        print(f"  guards {'OK' if guards else 'FAILED: ' + '; '.join(guard_msgs)}")
        for lab, arm_r in (("U", U), ("UP", U), ("P", Pp), ("U_S", U), ("UP_S", U), ("P_S", Pp)):
            vals = {k: _med(g(arm_r, lab, k)) for k in ("dE_tot", "dE_sign", "f_E_up", "f_C", "S2", "ov0", "w_sign")}
            print(f"  {lab:>6s}: dE_tot={vals['dE_tot']:.5f} dE_sign={vals['dE_sign']:.5f} "
                  f"f_E={vals['f_E_up']:.3f} f_C={vals['f_C']:.3f} <S2>={vals['S2']:.3f} "
                  f"ov0={vals['ov0']:.3f} w_sign={vals['w_sign']:.3e}  (medians)")
        dsUP, dsPS = _med(g(U, "UP", "dE_sign")), _med(g(Pp, "P_S", "dE_sign"))
        print(f"  fP={fP:.3f}  ratio P/U={ratio:.3f}  [diag: U->P/U={dsUP / dsU:.3f}  "
              f"P_S/U={dsPS / dsU:.3f}]  robust={robust[K]} symdom={symdom[K]}")
        # cross-check of arm U seed 1234 against P1.1
        ref = os.path.join(HERE, "p11_reference", P11_REF.get(K, "none"))
        if 1234 in U and os.path.exists(ref):
            rr = json.load(open(ref))["states"]
            for w in sorted(widths(U[1234])):
                if f"nnbf_{w}" not in rr:
                    continue
                a, b = U[1234]["budgets"][f"U_w{w}"], rr[f"nnbf_{w}"]
                rt = abs(a["dE_tot"] - b["dE_tot"]) / b["dE_tot"]
                rs = abs(a["dE_sign"] - b["dE_sign"]) / b["dE_sign"]
                print(f"  P1.1 cross-check w={w}: dE_tot {a['dE_tot']:.5f} vs {b['dE_tot']:.5f} "
                      f"(rel {rt:.3f}), dE_sign {a['dE_sign']:.5f} vs {b['dE_sign']:.5f} "
                      f"(rel {rs:.3f}) -> {'consistent' if max(rt, rs) <= XCHECK_TOL else 'DEVIATES'}")
    if robust.get(0) and robust.get(2):
        gv = "SIGN-ROBUST"
    elif all(symdom.get(K) for K in (0, 1, 2)):
        gv = "SYMMETRY-DOMINATED"
    else:
        gv = "AMBIGUOUS"
    print(f"GLOBAL VERDICT: {gv}")
    if gv == "AMBIGUOUS":
        live = [K for K in (0, 1, 2) if robust.get(K)]
        print(f"  sign branch live on instances: {live if live else 'none'}")
    return 0


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.1-P symmetry-projected sign budget (see module docstring)")
    ap.add_argument("--instance", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--arm", choices=list(ARMS), default="U")
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--widths", type=str, default="8,32,128")
    ap.add_argument("--sr-iters", type=int, default=1500)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--o-chunk", type=int, default=256)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--rng-replay", type=int, choices=[0, 1], default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--checks-only", type=int, choices=[0, 1], default=0)
    ap.add_argument("--time-step", type=int, choices=[0, 1], default=0)
    ap.add_argument("--smoke-dir", type=str, default="results")
    ap.add_argument("--aggregate", type=str, default="")
    a = ap.parse_args(argv)
    if a.task_id >= 0:
        if a.task_id > 17:
            ap.error("--task-id must be in 0..17")
        a.instance = a.task_id // 6
        a.seed = SEEDS[(a.task_id // 2) % 3]
        a.arm = ARMS[a.task_id % 2]
    a.width_list = [int(w) for w in a.widths.split(",") if w.strip()]
    if not a.out:
        d = a.smoke_dir if os.path.isabs(a.smoke_dir) else os.path.join(HERE, a.smoke_dir)
        a.out = os.path.join(d, f"inst{a.instance}_s{a.seed}_arm{a.arm}.json")
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        return aggregate(args.aggregate)
    if args.time_step:
        time_step(args)
        return 0
    log("args: " + json.dumps(vars(args)))
    if not run_checks(args.seed):
        log("correctness checks failed; aborting")
        return 2
    if args.checks_only:
        return 0
    if os.path.exists(args.out) and not args.overwrite:
        log(f"output {args.out} exists; use --overwrite 1 to recompute. Exiting.")
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
