#!/usr/bin/env python3
r"""
P4.2 decisive micro-test: zero-shot size transfer of a residual-geminal fermionic NQS
=====================================================================================

Question tested
---------------
Can a translation-equivariant fermionic NQS trained on a 2 x L_src t-J ladder be
evaluated *without retraining* on a 2 x L_tgt ladder (L_tgt = 2 L_src by default) with an
energy that is much closer to that ansatz's own converged energy than a size-transferred
Minimal Contrast Foil (same transferred correlator + a fresh mean-field BCS geminal)?

Everything is computed with exact sums over the full projected Hilbert space (no Monte
Carlo noise), so every number printed is deterministic up to the seed of the initial
network weights.

Model (t-J ladder, 2 legs x L rungs)
-----------------------------------
    H = - sum_{<ij>,s} t_ij (c^+_{is} c_{js} + h.c.)  +  sum_{<ij>} J_ij (S_i.S_j - n_i n_j / 4)
on the no-double-occupancy space.  Bonds: legs (amplitude t, exchange J), rungs (t_perp, J),
both plaquette diagonals (t', exchange J_d).  Boundary condition along the legs is periodic
or antiperiodic (a fermion hopping across x = L-1 -> 0 picks up +1 or -1).  The fermionic
sign of every matrix element (including the boundary Jordan-Wigner string) is computed
exactly from the operator algebra with mode order (all up modes by site, then all down modes).
Sites are labelled i = 2 x + y, x in [0, L), y in {0, 1}.

Ansatz (route "residual geminal", the top surviving route of P4.2)
------------------------------------------------------------------
    psi(n) = exp(C_W(n)) * det_{a,b} [ F_ij(n) ]_{i = up_a, j = down_b}
    F_ij(n) = F^MF_ij(mu, D_leg, D_rung, t'_eff ; lattice)                       (size-regenerated)
            + sum_images phi^|m| df(|dx + m L|, dy) * exp( g(|dx + m L|, dy) * (h_i + h_j) )   (transferred)
  * C_W: translation-equivariant one-hidden-layer network, sum_x sum_k a_k log cosh(W_k . v_x + b_k),
    v_x = (hole, S^z) one-hot features on the (2w+1) rungs centred on rung x.  Size independent.
  * F^MF: Fourier transform of v_k/u_k of a d-wave-like BCS state on the *current* lattice
    k-grid (so filling, Fermi surface and boundary-condition shell structure are regenerated
    at every size).  E_k is regularised as sqrt(xi^2 + Delta^2 + eta^2).
  * df, g: short-range learned pair correction and its local backflow, support |dx| <= R.
    At L_src, R = --R-corr (default 1; aliased images are summed if R = L_src/2); at L_tgt they
    are copied and zero-padded to R_tgt = L_tgt // 2 for the converged optimisation.  h_i = number of holes on the 3 neighbours of site i.
  * phi = +1 (periodic) or -1 (antiperiodic).
The Minimal Contrast Foil is the same ansatz with df = g = 0 (pure projected BCS x network).

Protocol
--------
 1. ED on the source and target ladders (scipy eigsh).
 2. Source: optimise the full ansatz and, separately, the foil (df = g = 0 frozen) by exact-sum
    stochastic reconfiguration (SR) with backtracking.
 3. Target, zero shot:
      E_tr   : route parameters transferred (C_W, MF params, df, g zero-padded); E_tr is the
               lower of (i) pure zero shot and (ii) the same with only the 4 MF parameters
               refitted at L_tgt for --steps-foil SR steps (residual mode only; started from the
               better of route-MF and foil-MF params).  (ii) has exactly the cost of foil variant C,
               so the comparison stays fair; and (iii) the best foil (below) with the transferred
               learned correction df, g added on top (no extra cost).  All three are stored;
               E_tr is their minimum, which is the route's most favourable zero-shot reading.
      E_foil : min over three foil variants, all at L_tgt with a fresh MF table:
               (A) route C_W + foil MF params; (B) foil C_W + foil MF params;
               (C) best of A/B with the 4 MF params re-optimised at L_tgt (C_W frozen,
               --steps-foil SR steps; this is the strongest "fresh mean-field" foil).
 4. Target, converged: optimise the full route ansatz (R = R_tgt) for --steps-tgt SR steps
    from (i) the transferred init and (ii) the best-foil init.  E_opt = min of the two.
 5. Rung-singlet pair correlation P(d) = <Delta_0^+ Delta_d>,
    Delta_x = (c_{x0 dn} c_{x1 up} - c_{x0 up} c_{x1 dn}) / sqrt(2), for ED, transferred and
    foil states at d = L_tgt // 2 (max distance).

Scalar metric and decision (pre-committed)
------------------------------------------
    e_tr   = (E_tr   - E_opt) / V_tgt ,     e_foil = (E_foil - E_opt) / V_tgt   (energy per site, units of t)
    M      = e_tr / e_foil                  (residual fraction of the foil's zero-shot excess)
    guard  = |P_tr(D) - P_ED(D)| <= |P_foil(D) - P_ED(D)| + 1e-5
    VOID      if (E_opt - E_0)/|E_0| > --void-tol (ansatz too weak at L_tgt for the test to mean anything)
    PASS      if M < --pass-m (0.5) and e_tr < --pass-e (0.01) and guard
    FAIL      if M > --fail-m (0.9) or e_foil <= 0 or not guard
    AMBIGUOUS otherwise, and also when a PASS occurs on an instance whose ED sign structure
              agrees with the projected-BCS foil sign to more than --hc-sign (Hard Core not exercised).
Changing the three thresholds after seeing results voids the pre-commitment.
Also reported: SR steps needed from each target init to reach (E - E_opt)/V < --conv-tol,
fidelity with ED, sign agreement s_X = sum_n psi0(n)^2 sgn(psi0(n) psi_X(n)) (global sign fixed),
and the sign agreement of ED with the projected BCS foil (a Hard-Core check: it must be < 1).

Command-line options (all optional)
-----------------------------------
  --L-src INT          Source ladder length (rungs). Must be >= 3. Default 4.
  --L-tgt INT          Target ladder length. Must be >= 3 and > --L-src is not required
                       but is the intended use. Default 8.
  --doping FLOAT       Hole doping delta; holes = delta * 2L must be an integer and the electron
                       number 2L - holes must be even (S^z = 0 sector). Default 0.25
                       (2x4: 2 holes, 560 states; 2x8: 4 holes, 1,681,680 states).
  --holes-src INT      Override the number of holes on the source (ignores --doping there).
  --holes-tgt INT      Override the number of holes on the target.
  --t FLOAT            Leg hopping t (energy unit). Default 1.0.
  --tperp FLOAT        Rung hopping. Default 1.0.
  --tp FLOAT           Diagonal (next-nearest) hopping t'. Default -0.25.
  --J FLOAT            Leg and rung exchange. Default 0.5.
  --Jd FLOAT           Diagonal exchange. Default 0.0.
  --bc {periodic,antiperiodic}
                       Boundary condition along the legs for BOTH sizes.
                       periodic: c_{L} = c_0. antiperiodic: c_{L} = -c_0. Default periodic.
  --geminal {residual,pure}
                       residual: F = F^MF (regenerated at each size) + transferred short-range df
                                 (the route under test; foil is a strict sub-family).
                       pure:     F = df only (no MF part), the plain truncated-pair-function
                                 transfer; df is initialised by least-squares fit to F^MF.
                       Default residual.
  --backflow / --no-backflow
                       Enable or disable the local hole-count backflow g on df. Default enabled.
  --R-corr INT         (residual only) Range R of the learned correction df, g at the SOURCE,
                       in rungs (|dx| <= R).  Capped at L_src // 2.  Default 1.  It must be
                       below L_src // 2 for the MF / correction split to be identifiable: with
                       R = L_src // 2 the correction alone spans every pair function on the
                       source cluster, so what gets transferred is arbitrary.  For --geminal pure
                       the source range is always L_src // 2.  The converged target ansatz always
                       uses R_tgt = L_tgt // 2 (transferred df, g are zero-padded).
  --hidden INT         Hidden units of the correlator network. Default 4.
  --window INT         Half-width w of the network window in rungs; needs 2w+1 <= L_src. Default 1.
  --steps-src INT      SR steps for the source route and for the source foil. Default 600.
  --steps-tgt INT      SR steps for each of the two target optimisations. Default 250.
  --steps-foil INT     SR steps for the target MF-only foil re-optimisation (variant C). Default 60.
  --lr FLOAT           Initial SR step size (adapted by backtracking, capped at 4x). Default 0.05.
  --diag-shift FLOAT   Relative diagonal shift of the SR matrix. Default 0.01.
  --eta FLOAT          Regulariser of the BCS quasiparticle energy. Default 0.02.
  --chunk INT          Configurations per vectorised chunk. Default 100000.
  --seed INT           RNG seed for the network initialisation. Default 0.
  --void-tol FLOAT     VOID threshold on the relative error of E_opt vs ED. Default 0.05.
  --pass-m FLOAT       PASS threshold on M. Default 0.5.
  --fail-m FLOAT       FAIL threshold on M. Default 0.9.
  --pass-e FLOAT       PASS threshold on e_tr (per site, units of t). Default 0.01.
  --hc-sign FLOAT      Hard-Core check: if the sign agreement of ED with the best foil (whose sign
                       is that of a projected BCS geminal) exceeds this value, the instance's sign
                       structure is mean-field-like; a PASS is then downgraded to AMBIGUOUS and the
                       flag hardcore_weak is set. Default 0.99.
  --conv-tol FLOAT     Per-site tolerance for the steps-to-converge diagnostic. Default 1e-3.
  --out PATH           JSON results file. Default results_<geminal>_<bc>_L<src>to<tgt>.json
                       next to this script.
  --selftest           Run the fermion-sign self-test (spinless free fermions on the ladder,
                       both BCs, compared with the analytic band sum) and the Hermiticity check,
                       then exit.

Examples
--------
  python micro_test.py --selftest
  python micro_test.py --L-tgt 6 --holes-tgt 4 --steps-src 30 --steps-tgt 5 --steps-foil 3   # smoke
  python micro_test.py                                                                       # decisive run
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

U64 = np.uint64
_M1, _M2, _M4, _H01 = (U64(0x5555555555555555), U64(0x3333333333333333),
                       U64(0x0F0F0F0F0F0F0F0F), U64(0x0101010101010101))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def popcount(a):
    a = a - ((a >> U64(1)) & _M1)
    a = (a & _M2) + ((a >> U64(2)) & _M2)
    a = (a + (a >> U64(4))) & _M4
    return ((a * _H01) >> U64(56)).astype(np.int64)


# ----------------------------------------------------------------------------- lattice
class Ladder:
    def __init__(self, L, t, tperp, tp, J, Jd, bc):
        assert L >= 3, "L must be >= 3"
        self.L, self.N = L, 2 * L
        self.t, self.tperp, self.tp = t, tperp, tp
        self.phi = 1.0 if bc == "periodic" else -1.0
        self.bc = bc
        s = self.site
        self.hop, self.jb = [], []
        for x in range(L):
            ph = self.phi if x == L - 1 else 1.0
            for y in (0, 1):
                self.hop.append((s(x, y), s(x + 1, y), t * ph))
                self.jb.append((s(x, y), s(x + 1, y), J))
            self.hop.append((s(x, 0), s(x, 1), tperp))
            self.jb.append((s(x, 0), s(x, 1), J))
            for (y0, y1) in ((0, 1), (1, 0)):
                if tp != 0.0:
                    self.hop.append((s(x, y0), s(x + 1, y1), tp * ph))
                if Jd != 0.0:
                    self.jb.append((s(x, y0), s(x + 1, y1), Jd))
        self.xs = np.array([i // 2 for i in range(self.N)])
        self.ys = np.array([i % 2 for i in range(self.N)])
        self.nbrs = np.array([[s(i // 2 - 1, i % 2), s(i // 2 + 1, i % 2), s(i // 2, 1 - i % 2)]
                              for i in range(self.N)])

    def site(self, x, y):
        return 2 * (x % self.L) + y


def make_basis(N, nup, ndn):
    def strings(n):
        return np.array([sum(1 << k for k in c) for c in combinations(range(N), n)], dtype=U64)
    ups, dns = strings(nup), strings(ndn)
    out = [u | (dns[(dns & u) == 0] << U64(N)) for u in ups]
    return np.sort(np.concatenate(out))


def lookup(basis, states):
    idx = np.searchsorted(basis, states)
    idx = np.minimum(idx, len(basis) - 1)
    return idx, basis[idx] == states


def apply_string(states, ops):
    """ops: list of (mode, is_creation) in written operator order; applied right to left."""
    st = states.copy()
    sg = np.ones(len(st), dtype=np.int64)
    ok = np.ones(len(st), dtype=bool)
    for mode, dag in reversed(ops):
        bit = U64(1) << U64(mode)
        occ = (st & bit) != 0
        ok &= (~occ) if dag else occ
        sg *= 1 - 2 * (popcount(st & (bit - U64(1))) & 1)
        st = st ^ bit
    return st, sg, ok


def bitarr(states, mode):
    return ((states >> U64(mode)) & U64(1)).astype(np.int64)


def build_H(lat, basis, spinful=True):
    N, n = lat.N, len(basis)
    R, C, V = [], [], []
    diag = np.zeros(n)
    spins = (0, 1) if spinful else (0,)
    for (i, j, a) in lat.hop:
        for s in spins:
            for (p, q, site_p) in ((i + s * N, j + s * N, i), (j + s * N, i + s * N, j)):
                st, sg, ok = apply_string(basis, [(p, True), (q, False)])
                if spinful:
                    ok &= bitarr(basis, site_p + (1 - s) * N) == 0
                src = np.nonzero(ok)[0]
                idx, f = lookup(basis, st[ok])
                assert f.all(), "hopping left the basis"
                R.append(idx); C.append(src); V.append(-a * sg[ok])
    if spinful:
        for (i, j, Jb) in lat.jb:
            ui, di, uj, dj = bitarr(basis, i), bitarr(basis, i + N), bitarr(basis, j), bitarr(basis, j + N)
            diag += Jb * (0.25 * (ui - di) * (uj - dj) - 0.25 * (ui + di) * (uj + dj))
            for ops in ([(i, True), (i + N, False), (j + N, True), (j, False)],
                        [(i + N, True), (i, False), (j, True), (j + N, False)]):
                st, sg, ok = apply_string(basis, ops)
                src = np.nonzero(ok)[0]
                idx, f = lookup(basis, st[ok])
                assert f.all()
                R.append(idx); C.append(src); V.append(0.5 * Jb * sg[ok])
    H = sp.coo_matrix((np.concatenate(V).astype(float), (np.concatenate(R), np.concatenate(C))),
                      shape=(n, n)).tocsr()
    return H + sp.diags(diag)


def ed(H, k=3):
    k = min(k, H.shape[0] - 2)
    if H.shape[0] < 3000:
        w, v = np.linalg.eigh(H.toarray())
    else:
        w, v = spla.eigsh(H, k=k, which="SA", tol=1e-10)
        o = np.argsort(w); w, v = w[o], v[:, o]
    deg = int(np.sum(w[:k] < w[0] + 1e-8))
    return w[0], v[:, :deg], (w[1] - w[0]) if len(w) > 1 else np.nan


def kgrid(lat):
    th = 0.0 if lat.phi > 0 else 0.5
    kx = 2 * np.pi * (np.arange(lat.L) + th) / lat.L
    return kx


# ----------------------------------------------------------------------------- selftest
def selftest(args):
    ok_all = True
    for bc in ("periodic", "antiperiodic"):
        lat = Ladder(4, 1.0, 1.0, -0.25, 0.0, 0.0, bc)
        for nf in (2, 3):
            basis = make_basis(lat.N, nf, 0)
            H = build_H(lat, basis, spinful=False)
            herm = abs(H - H.T).max()
            e = np.linalg.eigvalsh(H.toarray())[0]
            kx = kgrid(lat)
            eps = np.sort(np.concatenate([-2 * np.cos(kx) - 1.0 * np.cos(ky) - 2 * (-0.25) * np.cos(kx) * np.cos(ky)
                                          for ky in (0.0, np.pi)]))
            ref = eps[:nf].sum()
            ok = abs(e - ref) < 1e-10 and herm < 1e-12
            ok_all &= ok
            log(f"selftest bc={bc} N_f={nf}: E_ED={e:.10f} band-sum={ref:.10f} herm={herm:.1e} {'OK' if ok else 'FAIL'}")
    lat = Ladder(4, 1.0, 1.0, -0.25, 0.5, 0.0, "periodic")
    basis = make_basis(lat.N, 3, 3)
    H = build_H(lat, basis)
    herm = abs(H - H.T).max()
    log(f"selftest t-J 2x4 (2 holes) dim={len(basis)} herm={herm:.1e} E0={np.linalg.eigvalsh(H.toarray())[0]:.8f}")
    ok_all &= herm < 1e-12
    log("SELFTEST", "PASSED" if ok_all else "FAILED")
    return ok_all


# ----------------------------------------------------------------------------- ansatz
class Ansatz:
    """Parameter vector layout: [W (H*F), b (H), a (H), mf (4), df (nc), g (nc)]."""

    def __init__(self, lat, nup, R, hidden, window, geminal, backflow, eta, tp0):
        self.lat, self.nup, self.R = lat, nup, R
        self.H, self.w = hidden, window
        self.geminal, self.backflow, self.eta = geminal, backflow, eta
        L, N = lat.L, lat.N
        assert 2 * window + 1 <= L, "network window wraps onto itself"
        assert R <= L // 2
        self.win = np.array([[lat.site(x + d, y) for d in range(-window, window + 1) for y in (0, 1)]
                             for x in range(L)])  # (L, 2(2w+1))
        self.F = 2 * self.win.shape[1]
        # classes (d, dy) with d in 0..R, excluding (0,0)
        self.classes = [(d, dy) for d in range(R + 1) for dy in (0, 1) if (d, dy) != (0, 0)]
        self.nc = len(self.classes)
        cid = {c: k for k, c in enumerate(self.classes)}
        K = -np.ones((2, N, N), dtype=np.int64)
        P = np.zeros((2, N, N))
        for i in range(N):
            for j in range(N):
                dxr, dy = lat.xs[i] - lat.xs[j], lat.ys[i] ^ lat.ys[j]
                c = 0
                for m in (-1, 0, 1):
                    d = abs(dxr + m * L)
                    if d <= R and (d, dy) in cid:
                        assert c < 2
                        K[c, i, j] = cid[(d, dy)]
                        P[c, i, j] = lat.phi ** abs(m)
                        c += 1
        self.K, self.P = K, P
        self.Kc = np.where(K < 0, 0, K)
        # parameter slices
        sizes = [("W", hidden * self.F), ("b", hidden), ("a", hidden), ("mf", 4), ("df", self.nc), ("g", self.nc)]
        self.sl, o = {}, 0
        for nm, s_ in sizes:
            self.sl[nm] = slice(o, o + s_); o += s_
        self.npar = o
        self.tp0 = tp0

    # -- MF table and its derivatives (central finite differences)
    def mf_table(self, mf):
        lat = self.lat
        mu, dl, dr, tpe = mf
        kx = kgrid(lat)
        dx = lat.xs[:, None] - lat.xs[None, :]
        dy = lat.ys[:, None] - lat.ys[None, :]
        F = np.zeros((lat.N, lat.N))
        for ky in (0.0, np.pi):
            xi = -2 * lat.t * np.cos(kx) - lat.tperp * np.cos(ky) - 2 * tpe * np.cos(kx) * np.cos(ky) - mu
            De = 2 * dl * np.cos(kx) + dr * np.cos(ky)
            E = np.sqrt(xi ** 2 + De ** 2 + self.eta ** 2)
            Fk = De / (xi + E)
            F += np.cos(ky * dy) * (np.cos(np.multiply.outer(dx, kx)) @ Fk)
        return F / lat.N

    def mf_tables(self, mf, need_grad):
        F0 = self.mf_table(mf)
        if not need_grad:
            return F0, None
        dF = []
        for m in range(4):
            h = 1e-5 * max(1.0, abs(mf[m]))
            p, q = mf.copy(), mf.copy(); p[m] += h; q[m] -= h
            dF.append((self.mf_table(p) - self.mf_table(q)) / (2 * h))
        return F0, np.array(dF)

    def basis_tables(self):
        """Linear basis B_k(i,j) of the df part at g = 0 (for least-squares fits)."""
        Bk = np.zeros((self.nc, self.lat.N, self.lat.N))
        for c in range(2):
            for k in range(self.nc):
                Bk[k] += self.P[c] * (self.K[c] == k)
        return Bk

    def init_params(self, rng, mf0):
        th = np.zeros(self.npar)
        th[self.sl["W"]] = 0.1 * rng.standard_normal(self.H * self.F)
        th[self.sl["a"]] = 0.1 * rng.standard_normal(self.H)
        th[self.sl["mf"]] = mf0
        if self.geminal == "pure":
            th[self.sl["df"]] = self.fit_df(self.mf_table(mf0))
        return th

    def fit_df(self, Ftab):
        Bk = self.basis_tables().reshape(self.nc, -1).T
        mask = np.ones(self.lat.N * self.lat.N, bool)  # all pairs
        return np.linalg.lstsq(Bk[mask], Ftab.reshape(-1)[mask], rcond=None)[0]

    def evaluate(self, th, states, need_grad):
        lat, N, n = self.lat, self.lat.N, self.nup
        B = len(states)
        nu = ((states[:, None] >> np.arange(N, dtype=U64)) & U64(1)).astype(np.int64)
        nd = ((states[:, None] >> np.arange(N, 2 * N, dtype=U64)) & U64(1)).astype(np.int64)
        hole, sz = 1 - nu - nd, nu - nd
        # ---- correlator network
        W = th[self.sl["W"]].reshape(self.H, self.F); b = th[self.sl["b"]]; a = th[self.sl["a"]]
        v = np.concatenate([hole[:, self.win], sz[:, self.win]], axis=2).astype(float)  # (B, L, F)
        z = v @ W.T + b                                                           # (B, L, H)
        lc = np.logaddexp(z, -z) - np.log(2.0)
        logJ = lc @ a
        logJ = logJ.sum(1)
        # ---- geminal
        ups = np.nonzero(nu)[1].reshape(B, n)
        dns = np.nonzero(nd)[1].reshape(B, n)
        mf = th[self.sl["mf"]]; df = th[self.sl["df"]]; g = th[self.sl["g"]]
        A = np.zeros((B, n, n))
        if self.geminal == "residual":
            Fmf, dFmf = self.mf_tables(mf, need_grad)
            A += Fmf[ups[:, :, None], dns[:, None, :]]
        hn = hole[np.arange(B)[:, None, None], lat.nbrs[None, :, :]].sum(2)        # (B, N)
        S = (hn[np.arange(B)[:, None], ups][:, :, None] + hn[np.arange(B)[:, None], dns][:, None, :]).astype(float)
        Kc_, Pc_, Ec_ = [], [], []
        for c in range(2):
            Kc = self.Kc[c][ups[:, :, None], dns[:, None, :]]
            Pc = self.P[c][ups[:, :, None], dns[:, None, :]]
            Ec = np.exp(g[Kc] * S) if self.backflow else np.ones_like(S)
            A += Pc * df[Kc] * Ec
            Kc_.append(Kc); Pc_.append(Pc); Ec_.append(Ec)
        sgn, lad = np.linalg.slogdet(A)
        good = (sgn != 0) & np.isfinite(lad)
        logabs = np.where(good, logJ + lad, -np.inf)
        if not need_grad:
            return logabs, sgn, None
        O = np.zeros((B, self.npar))
        th_ = np.tanh(z) * a                                                     # (B, L, H)
        O[:, self.sl["a"]] = lc.sum(1)
        O[:, self.sl["b"]] = th_.sum(1)
        O[:, self.sl["W"]] = np.einsum("blh,blf->bhf", th_, v).reshape(B, -1)
        Wt = np.zeros_like(A)
        if good.any():
            Wt[good] = np.transpose(np.linalg.inv(A[good]), (0, 2, 1))
        if self.geminal == "residual":
            for m in range(4):
                O[:, self.sl["mf"].start + m] = np.einsum("bij,bij->b", Wt, dFmf[m][ups[:, :, None], dns[:, None, :]])
        of, og = self.sl["df"].start, self.sl["g"].start
        for c in range(2):
            Q = Wt * Pc_[c] * Ec_[c]
            Qs = Q * S
            Kc = Kc_[c]
            valid = self.K[c][ups[:, :, None], dns[:, None, :]] >= 0
            for k in range(self.nc):
                mk = valid & (Kc == k)
                O[:, of + k] += (Q * mk).sum((1, 2))
                if self.backflow:
                    O[:, og + k] += df[k] * (Qs * mk).sum((1, 2))
        O[~good] = 0.0
        return logabs, sgn, O


# ----------------------------------------------------------------------------- exact-sum VMC
def state_vector(ans, th, basis, chunk):
    la, sg = [], []
    for s in range(0, len(basis), chunk):
        l_, g_, _ = ans.evaluate(th, basis[s:s + chunk], False)
        la.append(l_); sg.append(g_)
    la, sg = np.concatenate(la), np.concatenate(sg)
    psi = np.where(np.isfinite(la), sg * np.exp(la - la[np.isfinite(la)].max()), 0.0)
    return psi / np.linalg.norm(psi)


def energy_grad(ans, th, basis, H, chunk):
    la, sg, Os = [], [], []
    for s in range(0, len(basis), chunk):
        l_, g_, O_ = ans.evaluate(th, basis[s:s + chunk], True)
        la.append(l_); sg.append(g_); Os.append(O_)
    la, sg, O = np.concatenate(la), np.concatenate(sg), np.concatenate(Os)
    psi = np.where(np.isfinite(la), sg * np.exp(la - la[np.isfinite(la)].max()), 0.0)
    psi /= np.linalg.norm(psi)
    Hpsi = H @ psi
    E = psi @ Hpsi
    w = psi ** 2
    Ob = w @ O
    grad = 2 * (O.T @ (psi * Hpsi) - E * Ob)
    S = (O * w[:, None]).T @ O - np.outer(Ob, Ob)
    return E, grad, S


def optimise(ans, th, free, basis, H, steps, lr, shift, chunk, tag, Eref=None):
    th = th.copy()
    E, g, S = energy_grad(ans, th, basis, H, chunk)
    hist = [E]
    lr0 = lr
    for it in range(steps):
        Sf = S[np.ix_(free, free)]
        Sf = Sf + shift * np.diag(np.diag(Sf)) + 1e-6 * np.eye(len(free))
        d = np.linalg.solve(Sf, g[free])
        tried = 0
        while True:
            tn = th.copy(); tn[free] -= lr * d
            En, gn, Sn = energy_grad(ans, tn, basis, H, chunk)
            if En <= E + 1e-12 or tried >= 6:
                break
            lr *= 0.5; tried += 1
        th, E, g, S = tn, En, gn, Sn
        lr = min(lr * 1.1, 4 * lr0)
        hist.append(E)
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            extra = f" relerr={(E - Eref) / abs(Eref):.3e}" if Eref is not None else ""
            log(f"  [{tag}] step {it + 1}/{steps} E={E:.8f} lr={lr:.3g}{extra}")
    return th, E, np.array(hist)


def free_idx(ans, names):
    return np.concatenate([np.arange(ans.npar)[ans.sl[nm]] for nm in names])


# ----------------------------------------------------------------------------- observables
def pair_corr(lat, basis, low_basis, vecs, D):
    """P(D) = <Delta_0^+ Delta_D>, averaged over the columns of vecs (degenerate manifold)."""
    N = lat.N
    def delta(x, vec):
        out = np.zeros(len(low_basis))
        i0, i1 = lat.site(x, 0), lat.site(x, 1)
        for coef, ops in ((1.0, [(i0 + N, False), (i1, False)]), (-1.0, [(i0, False), (i1 + N, False)])):
            st, sg, ok = apply_string(basis, ops)
            idx, f = lookup(low_basis, st[ok])
            assert f.all()
            np.add.at(out, idx, coef * sg[ok] * vec[ok] / np.sqrt(2))
        return out
    vals = [delta(0, vecs[:, c]) @ delta(D, vecs[:, c]) for c in range(vecs.shape[1])]
    return float(np.mean(vals))


def sign_agreement(psi0, psi):
    s = np.sign(psi0 @ psi) or 1.0
    return float(np.sum(psi0 ** 2 * (np.sign(psi0 * psi * s) > 0)) - np.sum(psi0 ** 2 * (np.sign(psi0 * psi * s) < 0)))


# ----------------------------------------------------------------------------- main protocol
def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L-src", type=int, default=4)
    p.add_argument("--L-tgt", type=int, default=8)
    p.add_argument("--doping", type=float, default=0.25)
    p.add_argument("--holes-src", type=int, default=None)
    p.add_argument("--holes-tgt", type=int, default=None)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--tperp", type=float, default=1.0)
    p.add_argument("--tp", type=float, default=-0.25)
    p.add_argument("--J", type=float, default=0.5)
    p.add_argument("--Jd", type=float, default=0.0)
    p.add_argument("--bc", choices=["periodic", "antiperiodic"], default="periodic")
    p.add_argument("--geminal", choices=["residual", "pure"], default="residual")
    p.add_argument("--backflow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--R-corr", type=int, default=1)
    p.add_argument("--hidden", type=int, default=4)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--steps-src", type=int, default=600)
    p.add_argument("--steps-tgt", type=int, default=250)
    p.add_argument("--steps-foil", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--diag-shift", type=float, default=0.01)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--chunk", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--void-tol", type=float, default=0.05)
    p.add_argument("--pass-m", type=float, default=0.5)
    p.add_argument("--fail-m", type=float, default=0.9)
    p.add_argument("--pass-e", type=float, default=0.01)
    p.add_argument("--conv-tol", type=float, default=1e-3)
    p.add_argument("--hc-sign", type=float, default=0.99)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    for which, L, h in (("src", a.L_src, a.holes_src), ("tgt", a.L_tgt, a.holes_tgt)):
        if h is None:
            hf = a.doping * 2 * L
            if abs(hf - round(hf)) > 1e-9:
                p.error(f"doping*2L not an integer for L_{which}={L}")
            h = int(round(hf))
        ne = 2 * L - h
        if ne % 2 or ne <= 0:
            p.error(f"electron number {ne} on L_{which}={L} must be even and positive")
        setattr(a, f"holes_{which}", h)
    if a.out is None:
        a.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"results_{a.geminal}_{a.bc}_L{a.L_src}to{a.L_tgt}.json")
    return a


def mf_init(lat, ne, tp):
    kx = kgrid(lat)
    eps = np.sort(np.concatenate([-2 * lat.t * np.cos(kx) - lat.tperp * np.cos(ky) - 2 * tp * np.cos(kx) * np.cos(ky)
                                  for ky in (0.0, np.pi)]))
    nocc = ne // 2
    mu = 0.5 * (eps[nocc - 1] + eps[min(nocc, len(eps) - 1)])
    return np.array([mu, 0.1, -0.1, tp])


def transfer(ans_s, th_s, ans_t, zero_df=False):
    th = np.zeros(ans_t.npar)
    for nm in ("W", "b", "a", "mf"):
        th[ans_t.sl[nm]] = th_s[ans_s.sl[nm]]
    if not zero_df:
        for k, c in enumerate(ans_s.classes):
            kt = ans_t.classes.index(c)
            th[ans_t.sl["df"].start + kt] = th_s[ans_s.sl["df"].start + k]
            th[ans_t.sl["g"].start + kt] = th_s[ans_s.sl["g"].start + k]
    return th


def steps_to(hist, Eopt, V, tol):
    hit = np.nonzero((hist - Eopt) / V < tol)[0]
    return int(hit[0]) if len(hit) else None


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.selftest:
        sys.exit(0 if selftest(a) else 1)
    log("args:", vars(a))
    rng = np.random.default_rng(a.seed)
    res = {"args": vars(a)}
    mk = lambda L: Ladder(L, a.t, a.tperp, a.tp, a.J, a.Jd, a.bc)
    Ls, Lt = mk(a.L_src), mk(a.L_tgt)
    nes, net = Ls.N - a.holes_src, Lt.N - a.holes_tgt
    Rs = a.L_src // 2 if a.geminal == "pure" else min(a.R_corr, a.L_src // 2)
    Rt = a.L_tgt // 2

    # ---------------- source
    bs = make_basis(Ls.N, nes // 2, nes // 2)
    Hs = build_H(Ls, bs)
    E0s, v0s, gaps = ed(Hs)
    log(f"source 2x{a.L_src}, N_e={nes}, dim={len(bs)}: E0={E0s:.8f} gap={gaps:.4e} deg={v0s.shape[1]}")
    mk_ans = lambda lat, ne, R: Ansatz(lat, ne // 2, R, a.hidden, a.window, a.geminal, a.backflow, a.eta, a.tp)
    As = mk_ans(Ls, nes, Rs)
    th0 = As.init_params(rng, mf_init(Ls, nes, a.tp))
    free_route = free_idx(As, ["W", "b", "a"] + (["mf"] if a.geminal == "residual" else []) + ["df"] + (["g"] if a.backflow else []))
    log("optimising source route")
    th_rs, E_rs, _ = optimise(As, th0, free_route, bs, Hs, a.steps_src, a.lr, a.diag_shift, a.chunk, "src-route", E0s)
    # foil at source: residual ansatz with df = g = 0 frozen (projected BCS x network)
    Fs = Ansatz(Ls, nes // 2, Rs, a.hidden, a.window, "residual", False, a.eta, a.tp)
    thf0 = th0.copy(); thf0[Fs.sl["df"]] = 0.0; thf0[Fs.sl["g"]] = 0.0
    log("optimising source foil")
    th_fs, E_fs, _ = optimise(Fs, thf0, free_idx(Fs, ["W", "b", "a", "mf"]), bs, Hs, a.steps_src, a.lr, a.diag_shift, a.chunk, "src-foil", E0s)
    res["source"] = dict(E0=E0s, E_route=E_rs, E_foil=E_fs, dim=len(bs), N_e=nes,
                         relerr_route=(E_rs - E0s) / abs(E0s), relerr_foil=(E_fs - E0s) / abs(E0s),
                         theta_route=th_rs.tolist(), theta_foil=th_fs.tolist())
    log(f"source: E0={E0s:.6f} route={E_rs:.6f} foil={E_fs:.6f}")

    # ---------------- target ED
    bt = make_basis(Lt.N, net // 2, net // 2)
    log(f"target basis dim={len(bt)}")
    Ht = build_H(Lt, bt)
    log(f"target H nnz={Ht.nnz}")
    E0t, v0t, gapt = ed(Ht)
    V = Lt.N
    log(f"target 2x{a.L_tgt}, N_e={net}: E0={E0t:.8f} gap={gapt:.4e} deg={v0t.shape[1]}")
    blow = make_basis(Lt.N, net // 2 - 1, net // 2 - 1)
    D = a.L_tgt // 2
    P_ed = pair_corr(Lt, bt, blow, v0t, D)
    psi0 = v0t[:, 0]

    # ---------------- zero-shot transfers
    At = mk_ans(Lt, net, Rt)
    th_tr = transfer(As, th_rs, At)
    E_tr = energy_grad(At, th_tr, bt, Ht, a.chunk)[0]
    log(f"zero-shot route: E_tr={E_tr:.8f}")
    Ft = Ansatz(Lt, net // 2, Rt, a.hidden, a.window, "residual", False, a.eta, a.tp)
    thA = transfer(As, th_rs, Ft, zero_df=True); thA[Ft.sl["mf"]] = th_fs[Fs.sl["mf"]]
    thB = transfer(Fs, th_fs, Ft, zero_df=True)
    EA = energy_grad(Ft, thA, bt, Ht, a.chunk)[0]
    EB = energy_grad(Ft, thB, bt, Ht, a.chunk)[0]
    thAB = thA if EA <= EB else thB
    log(f"foil A={EA:.8f} B={EB:.8f}; re-optimising MF params (variant C)")
    thC, EC, _ = optimise(Ft, thAB, free_idx(Ft, ["mf"]), bt, Ht, a.steps_foil, a.lr, a.diag_shift, a.chunk, "tgt-foilC", E0t)
    foils = {"A": (EA, thA), "B": (EB, thB), "C": (EC, thC)}
    best = min(foils, key=lambda k: foils[k][0])
    E_foil, th_foil = foils[best]
    # route with the same cheap 4-parameter MF refit at L_tgt (identical cost to foil variant C)
    if a.geminal == "residual":
        th_alt = th_tr.copy(); th_alt[At.sl["mf"]] = th_fs[Fs.sl["mf"]]
        E_alt = energy_grad(At, th_alt, bt, Ht, a.chunk)[0]
        th_st = th_tr if E_tr <= E_alt else th_alt
        th_trC, E_trC, _ = optimise(At, th_st, free_idx(At, ["mf"]), bt, Ht, a.steps_foil, a.lr, a.diag_shift, a.chunk, "tgt-routeC", E0t)
    else:
        th_trC, E_trC = th_tr, E_tr
    # variant D: best foil (C_W and MF params) + the transferred learned correction df, g only
    th_trD = th_foil.copy()
    th_trD[At.sl["df"]] = th_tr[At.sl["df"]]; th_trD[At.sl["g"]] = th_tr[At.sl["g"]]
    if a.geminal == "pure":
        th_trD = th_tr.copy(); th_trD[At.sl["W"]] = th_foil[Ft.sl["W"]]
        th_trD[At.sl["b"]] = th_foil[Ft.sl["b"]]; th_trD[At.sl["a"]] = th_foil[Ft.sl["a"]]
    E_trD = energy_grad(At, th_trD, bt, Ht, a.chunk)[0]
    cands = {"zeroshot": (E_tr, th_tr), "mfrefit": (E_trC, th_trC), "foil+df": (E_trD, th_trD)}
    tr_best = min(cands, key=lambda k: cands[k][0])
    E_trX, th_trX = cands[tr_best]
    log(f"route zero-shot={E_tr:.8f}  route+MF-refit={E_trC:.8f}  best-foil+transferred-df={E_trD:.8f}")

    # ---------------- converged route at target
    th_foilinit = th_foil.copy()  # same layout as At (residual); df = g = 0
    if a.geminal == "pure":
        th_foilinit[At.sl["mf"]] = th_foil[Ft.sl["mf"]]
        th_foilinit[At.sl["df"]] = At.fit_df(At.mf_table(th_foil[Ft.sl["mf"]]))
    free_t = free_idx(At, ["W", "b", "a"] + (["mf"] if a.geminal == "residual" else []) + ["df"] + (["g"] if a.backflow else []))
    log("optimising target route from transferred init")
    th_o1, E_o1, h1 = optimise(At, th_tr, free_t, bt, Ht, a.steps_tgt, a.lr, a.diag_shift, a.chunk, "tgt-from-tr", E0t)
    log("optimising target route from foil init")
    th_o2, E_o2, h2 = optimise(At, th_foilinit, free_t, bt, Ht, a.steps_tgt, a.lr, a.diag_shift, a.chunk, "tgt-from-foil", E0t)
    E_opt = min(E_o1, E_o2)

    # ---------------- observables
    psi_tr = state_vector(At, th_trX, bt, a.chunk)
    psi_fo = state_vector(Ft, th_foil, bt, a.chunk)
    psi_op = state_vector(At, th_o1 if E_o1 <= E_o2 else th_o2, bt, a.chunk)
    P_tr = pair_corr(Lt, bt, blow, psi_tr[:, None], D)
    P_fo = pair_corr(Lt, bt, blow, psi_fo[:, None], D)
    P_op = pair_corr(Lt, bt, blow, psi_op[:, None], D)

    e_tr, e_foil = (E_trX - E_opt) / V, (E_foil - E_opt) / V
    M = e_tr / e_foil if e_foil > 0 else float("inf")
    guard = abs(P_tr - P_ed) <= abs(P_fo - P_ed) + 1e-5
    void = (E_opt - E0t) / abs(E0t) > a.void_tol
    if void:
        verdict = "VOID"
    elif (not guard) or e_foil <= 0 or M > a.fail_m:
        verdict = "FAIL"
    elif M < a.pass_m and e_tr < a.pass_e and guard:
        verdict = "PASS"
    else:
        verdict = "AMBIGUOUS"
    sa_foil = sign_agreement(psi0, psi_fo)
    hardcore_weak = sa_foil > a.hc_sign
    if verdict == "PASS" and hardcore_weak:
        verdict = "AMBIGUOUS"  # passed only on an instance whose sign structure is mean-field-like
    res["target"] = dict(
        E0=E0t, gap=gapt, degeneracy=int(v0t.shape[1]), dim=len(bt), N_e=net,
        E_tr_zeroshot=E_tr, E_tr_mfrefit=E_trC, E_tr_foil_plus_df=E_trD, tr_best=tr_best, E_tr=E_trX, E_foil_A=EA, E_foil_B=EB, E_foil_C=EC, foil_best=best, E_foil=E_foil,
        E_opt_from_tr=E_o1, E_opt_from_foil=E_o2, E_opt=E_opt,
        relerr_tr=(E_trX - E0t) / abs(E0t), relerr_foil=(E_foil - E0t) / abs(E0t), relerr_opt=(E_opt - E0t) / abs(E0t),
        e_tr=e_tr, e_foil=e_foil, M=M,
        steps_to_conv_from_tr=steps_to(h1, E_opt, V, a.conv_tol),
        steps_to_conv_from_foil=steps_to(h2, E_opt, V, a.conv_tol),
        P_D=dict(D=D, ED=P_ed, tr=P_tr, foil=P_fo, opt=P_op), guard=bool(guard),
        fidelity=dict(tr=float((psi0 @ psi_tr) ** 2), foil=float((psi0 @ psi_fo) ** 2), opt=float((psi0 @ psi_op) ** 2)),
        sign_agreement=dict(tr=sign_agreement(psi0, psi_tr), foil=sign_agreement(psi0, psi_fo), opt=sign_agreement(psi0, psi_op)),
        hardcore_weak=bool(hardcore_weak), verdict=verdict)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    t = res["target"]
    log("================ RESULT ================")
    log(f"E0={E0t:.6f}  E_tr={E_trX:.6f}  E_foil={E_foil:.6f} ({best})  E_opt={E_opt:.6f}")
    log(f"e_tr={e_tr:.5f}  e_foil={e_foil:.5f}  M={M:.4f}  relerr_opt={t['relerr_opt']:.3e}")
    log(f"P(D={D}): ED={P_ed:.5f} tr={P_tr:.5f} foil={P_fo:.5f} opt={P_op:.5f} guard={guard}")
    log(f"sign agreement with ED: tr={t['sign_agreement']['tr']:.4f} foil={t['sign_agreement']['foil']:.4f} opt={t['sign_agreement']['opt']:.4f}")
    log(f"steps to conv (tol {a.conv_tol}/site): from tr={t['steps_to_conv_from_tr']} from foil={t['steps_to_conv_from_foil']}")
    log(f"VERDICT: {verdict}   (results in {a.out})")


if __name__ == "__main__":
    main()
