#!/usr/bin/env python3
r"""
P4.2 round-3 decisive test: twist-complete, minimum-norm, irrep-projected size transfer of a
momentum-field neural geminal (MFNG) on the doped t-t'-J torus, judged against the
target-optimised Jastrow-BCS foil, a projected-SR NQS trained from scratch at matched budget,
and the converged target energy. Decided on medians over 3 seeds.
=========================================================================================

Routes tested (P4.2 round 3, see .subproblems/P4.2_size_transfer.md)
--------------------------------------------------------------------
Route 3.1 "TCMN transfer" (twist-complete, min-norm). The source is not one cluster but the set
of twisted source clusters whose momentum grids together are exactly the target grid
(for 4x2 -> 4x4: the 4x2 torus with y-boundary phase 0 and pi; their ky grids {0, pi} and
{pi/2, 3pi/2} together are the 4x4 ky grid). MFNG is trained JOINTLY on all of them with an
L2 (minimum-norm) penalty lam*|theta|^2, so (i) every target momentum of every pair kernel is
pinned by source data (no interpolation), and (ii) the source optimum is selected by norm
rather than by the seed. At the target the copied state is projected onto the ground-state
irrep (translations x mirrors x diagonal mirror); the irrep can be picked by the transferred
state itself ("zero-shot sector rule": the 1-D K=0 irrep with the lowest projected energy),
which the test checks against ED.
Route 3.2 "FST" (frozen-sign transfer). The transferred, projected determinant part (which
carries every sign) is frozen; only the charge Jastrow (positive, invariant under the whole
group, so it cannot change any sign of the projected state) is refit at the target.

Model (t-t'-J on an L_x x L_y torus with boundary phases phi_x, phi_y in {0, pi})
------------------------------------------------------------------------------
    H = - sum_{<ij>,s} t_ij (c~^+_{is} c~_{js} + h.c.) + J sum_{<ij> nn} (S_i.S_j - n_i n_j / 4)
Each site (x,y) bonds to (x+1,y), (x,y+1) [t, J] and (x+1,y+1), (x+1,y-1) [t']; a hop across
a boundary with phase pi gets a factor -1 (exchange terms are gauge invariant). On L = 2
directions this doubles bonds, which is exactly the folding of the 2L torus; with phase pi the
doubled hops cancel, as the band eps_k = -2t(cos kx + cos ky) - 4t' cos kx cos ky on the
shifted grid k = (2 pi m + phi)/L requires (checked in --selftest).

Ansatz (MFNG; the Jastrow-BCS foil is the same with C = 0)
-----------------------------------------------------------
    psi(n) = exp( sum_s sum_h alpha_h log cosh(U_h . v_s + beta_h) ) * det_{a,b} A_ab(n)
    A_ab   = kappa_0(r_{u_a} - r_{d_b}) + sum_{m=1}^{C} phi_m(u_a) kappa_m(r_{u_a} - r_{d_b}) phi_m(d_b)
    kappa_m(r) = (1/V) sum_{k in grid} cos(k.r) f_m(k)    (r = UNWRAPPED site difference, so a
                                                          boundary phase pi is handled exactly)
    f_m(k) = o_{2m}(k) + gamma_k o_{2m+1}(k),  o = W2 tanh(W1 z_k + b1) + b2   (neural BZ field)
    z_k = ((cos kx + cos ky)/2, cos kx cos ky), gamma_k = cos kx - cos ky
    phi_m(s) = sum_h a_{mh} tanh(Wb_h . v_s + bb_h),  v_s = (h_s, sum_nn h / 4, sum_nnn h / 4)
Exact S = 0 singlet; parameter vector size independent (transfer = copy). Default sizes give
122 parameters (route) and 62 (foil); the two 4x2 source clusters have 2 x 560 states.

Symmetry projection (round-3 rule 1)
------------------------------------
Group G = translations x {1, M_x} x {1, M_y} x {1, D} (D: (x,y) -> (y,x), square tori only),
|G| = 128 on 4x4: the P1.1-P group (translations x mirrors) plus the diagonal mirror, re-built
for the no-double-occupancy basis. For a real 1-D irrep with characters lam (generators T_x,
T_y, M_x, M_y, D), the projector Pi = |G|^-1 sum_g chi(g) P_g is represented by an isometry U
(D x n_irrep, columns = normalised Pi|rep>), built by a streaming pass over all g. Every target
arm is TRAINED and evaluated as the projected state b = U^T psi: E = b^T H_G b / b^T b with
H_G = U^T H U, exact-sum SR with dB/dtheta = U^T (psi * O) (this is the projected-SR of
tests/P1.1P_projected, written in the reduced basis). Spin needs no projector: every arm is an
exact singlet (checked: <S^2> of tr0 is printed). The unprojected ("raw") energy of every arm
is reported too. The ground-state irrep is chosen from ED (K = 0 required, 1-D, singlet,
non-degenerate). Instance rule (pre-committed): the first (t', J) of --instances whose target
ground state satisfies these conditions; if none does, the run is VOID-SECTOR.

Arms
----
role transfer (array tasks 0-2, seed = task):
  tr0       TCMN-transferred MFNG, projected, 0 target steps (zero shot)
  tr0_r2    ablation: round-2-style source (periodic cluster only, lam = 0), zero shot
  foil_tr   Jastrow-BCS foil transferred by the same TCMN rule, zero shot (diagnostic)
  trB       tr0 + B projected-SR steps (all parameters)
  fstB      tr0 + B projected-SR steps on the Jastrow only (Route 3.2)
  scrB      MFNG from MF init at the target + B projected-SR steps (matched-budget control)
  trL       trB continued to --steps-long total steps (converged-energy candidate)
role foil (tasks 3-5): foil_opt = Jastrow-BCS foil (C = 0) from MF init at the target,
           --steps-long projected-SR steps (the Minimal Contrast Foil optimised AT the target)
role scratch (tasks 6-8): scrL = MFNG from MF init, --steps-long projected-SR steps
E_opt = min over all trL and scrL (converged ansatz energy); E0 = ED (exact) in the GS irrep.

Metric and decision (pre-committed; implemented in --aggregate)
---------------------------------------------------------------
eps_X = (E_X - E0)/V per site (projected energies), m(.) = median over the 3 seeds,
A_X = E_sign_up(sgn b_X) - E0 (P1.1 fixed-sign excess in the irrep basis), spread = max - min.
  VOID         if (E_opt - E0)/|E0| > --void-tol (ansatz too weak at the target).
  premises     (I) identifiability: spread(eps_tr0) <= 0.5 m(eps_tr0)
               (S) zero-shot sector rule picks the ED irrep in >= 2 of 3 seeds
Route 3.1 PASS if all of
  (a) m(eps_tr0) <= m(eps_foil_opt)                     zero shot beats the target-optimised foil
  (b) m(eps_trB) <= 0.5 m(eps_scrB)                      beats scratch at matched budget
  (c) m(eps_trB) - eps_opt <= 0.5 (m(eps_foil_opt) - eps_opt)   close to converged
  (d) m |Pd_trB - Pd_0| <= m |Pd_foil_opt - Pd_0| + 1e-4         pairing guard
  (e) m(eps_tr0) < m(eps_tr0_r2)                         the new source design is what helps
  (I) and (S)
Route 3.1 FAIL if m(eps_tr0) >= 2 m(eps_foil_opt) or m(eps_trB) >= 0.9 m(eps_scrB)
  or spread(eps_tr0) > m(eps_tr0) or (d) fails or (S) fails; otherwise AMBIGUOUS (not promoted).
Route 3.2 PASS if m(eps_fstB) <= m(eps_foil_opt) and m(A_tr0) <= 0.5 m(A_foil_opt)
  and spread(A_tr0) <= 0.5 m(A_tr0) and (S) and calibration OK;
  FAIL if m(A_tr0) >= 0.9 m(A_foil_opt) or m(eps_fstB) >= 1.5 m(eps_foil_opt) or (S) fails;
  otherwise AMBIGUOUS (not promoted).
Calibration: L-BFGS-B with the EXACT signs from a uniform start must return E0 to 1e-3 min A.
Hard-Core report: in-irrep gap Delta_1/V (competitor flag if < 0.005 t; otherwise sign half only).

Command-line options
--------------------
  --task-id INT        Slurm array index 0-8: 0-2 role transfer, 3-5 role foil, 6-8 role
                       scratch; seed = task % 3. Overrides --role/--seed. Default -1 (off).
  --role STR           transfer | foil | scratch (default transfer).
  --seed INT           Seed (default 0).
  --src LxXLy          Source torus (default 4x2). --tgt LxXLy target (default 4x4); each
                       target length must be 1x or 2x the source length.
  --holes-src INT      Holes at the source (default 2). --holes-tgt INT (default 4).
  --t FLOAT            Nearest-neighbour hopping (default 1.0).
  --instances STR      Ordered candidate list "tp:J,tp:J,..." (default
                       "-0.25:0.4,-0.25:0.6,0.25:0.4,-0.1:0.4").
  --channels INT       Backflow channels C (default 2). --field-hidden (8), --bf-hidden (4),
                       --jas-hidden (4): hidden widths.
  --delta0, --eta      MF-init d-wave amplitude (0.3) and E_k regulariser (0.1).
  --lam FLOAT          Min-norm penalty at the source, per parameter squared (default 1e-5).
  --steps-src INT      Source SR steps (default 800).
  --budget INT         Matched small target budget B (default 20).
  --steps-long INT     Long target runs (default 200).
  --lr, --diag-shift   SR step (0.05, backtracking) and relative diagonal shift (0.01).
  --fn-iters, --fn-floor, --lbfgs-iters   fixed-node / L-BFGS-B settings (6, 1e-3, 3000).
  --r-min FLOAT        Minimal bond-centre distance in Pd (default 2.0).
  --void-tol FLOAT     VOID threshold (default 0.05).
  --chunk INT          Configurations per vectorised chunk (default 50000).
  --out PATH           JSON output (default results/<role>_s<seed>.json next to this file).
  --aggregate DIR      Read every JSON in DIR, print the global decision (no computation).
  --selftest           Self-tests only (exit 2 on failure): (a) band sums incl. twisted
                       grids, (b) Hermiticity, (c) MFNG eigenstate of (twisted) translations
                       and [H, T~] = 0, (d) log-derivatives vs finite differences,
                       (e) symmetry sectors: [H, P_g] = 0, U^T U = I, H_G spectra inside the
                       full spectrum, ED ground state inside its irrep, (f) <S^2> on a
                       two-electron cluster, (g) projected-energy gradient vs finite
                       differences, (h) twist-covering bookkeeping (union of source grids
                       equals the target grid).
  --timing             Time one projected forward+gradient pass on the target, then exit.

Examples
--------
  python micro_test.py --selftest
  python micro_test.py --timing
  python micro_test.py --src 2x2 --tgt 4x2 --holes-src 2 --holes-tgt 4 --steps-src 150 \
      --budget 5 --steps-long 30 --role transfer       # login smoke (seconds)
  python micro_test.py --task-id 0                     # one Slurm task
  python micro_test.py --aggregate results
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
from scipy.optimize import minimize

U64 = np.uint64
_M1, _M2, _M4, _H01 = (U64(0x5555555555555555), U64(0x3333333333333333),
                       U64(0x0F0F0F0F0F0F0F0F), U64(0x0101010101010101))
T0 = time.time()
GUARD_TOL, CALIB_TOL, COMP_TOL = 1e-4, 1e-3, 0.005
ROLES = ("transfer", "foil", "scratch")


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def popcount(a):
    a = a - ((a >> U64(1)) & _M1)
    a = (a & _M2) + ((a >> U64(2)) & _M2)
    a = (a + (a >> U64(4))) & _M4
    return ((a * _H01) >> U64(56)).astype(np.int64)


# ============================================================================ lattice
class Torus:
    def __init__(self, Lx, Ly, t, tp, J, phx=0.0, phy=0.0):
        assert Lx >= 2 and Ly >= 2
        assert phx in (0.0, np.pi) and phy in (0.0, np.pi), "only real twists (0, pi)"
        self.Lx, self.Ly, self.N = Lx, Ly, Lx * Ly
        self.phx, self.phy = phx, phy
        twx, twy = int(phx != 0.0), int(phy != 0.0)
        s = self.site
        self.hop, self.jb, self.xb = [], [], []
        for y in range(Ly):
            for x in range(Lx):
                for (dx, dy) in ((1, 0), (0, 1)):
                    w = (abs((x + dx) // Lx) * twx + abs((y + dy) // Ly) * twy) % 2
                    self.hop.append((s(x, y), s(x + dx, y + dy), t * (1 - 2 * w)))
                    self.jb.append((s(x, y), s(x + dx, y + dy), J))
                    self.xb.append((s(x, y), s(x + dx, y + dy), "x" if dx else "y", x, y))
                if tp != 0.0:
                    for dy in (1, -1):
                        w = (abs((x + 1) // Lx) * twx + abs((y + dy) // Ly) * twy) % 2
                        self.hop.append((s(x, y), s(x + 1, y + dy), tp * (1 - 2 * w)))
        self.xs = np.array([i % Lx for i in range(self.N)])
        self.ys = np.array([i // Lx for i in range(self.N)])
        self.nn = np.array([[s(x + 1, y), s(x - 1, y), s(x, y + 1), s(x, y - 1)]
                            for y in range(Ly) for x in range(Lx)])
        self.nnn = np.array([[s(x + 1, y + 1), s(x - 1, y + 1), s(x + 1, y - 1), s(x - 1, y - 1)]
                             for y in range(Ly) for x in range(Lx)])
        ddx = self.xs[:, None] - self.xs[None, :]
        ddy = self.ys[:, None] - self.ys[None, :]
        self.disp = (ddx + Lx - 1) + (2 * Lx - 1) * (ddy + Ly - 1)        # unwrapped difference
        m, n = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="xy")
        self.kx = ((2 * np.pi * m + phx) / Lx).ravel()
        self.ky = ((2 * np.pi * n + phy) / Ly).ravel()
        gx, gy = np.meshgrid(np.arange(-(Lx - 1), Lx), np.arange(-(Ly - 1), Ly), indexing="xy")
        gx, gy = gx.ravel(), gy.ravel()
        self.Cd = np.cos(np.outer(gx, self.kx) + np.outer(gy, self.ky)) / self.N   # (disp, k)

    def site(self, x, y):
        return (x % self.Lx) + self.Lx * (y % self.Ly)

    def name(self):
        tw = "" if (self.phx == 0 and self.phy == 0) else f"[tw {self.phx / np.pi:.0f}pi,{self.phy / np.pi:.0f}pi]"
        return f"{self.Lx}x{self.Ly}{tw}"

    def square(self):
        return self.Lx == self.Ly


class Cluster2D:
    """Periodic square-lattice cluster with general lattice vectors a1, a2 (e.g. the 8-site
    tilted cluster a1 = (2, 2), a2 = (2, -2)). Same interface as Torus (no twists). Bonds follow
    the same translation rule, so the band on the cluster's momentum grid is exact."""

    def __init__(self, a1, a2, t, tp, J, label):
        self.label = label
        A = np.array([[a1[0], a2[0]], [a1[1], a2[1]]], float)
        self.Ainv = np.linalg.inv(A)
        self.A = A
        N = int(round(abs(np.linalg.det(A))))
        pts = {}
        R = int(np.abs(A).sum()) + 1
        for x in range(-R, R + 1):
            for y in range(-R, R + 1):
                key = self._key(x, y)
                if key not in pts:
                    pts[key] = (x, y)
        assert len(pts) == N
        self.keys = sorted(pts, key=lambda k: (pts[k][1], pts[k][0]))
        self.pos = np.array([pts[k] for k in self.keys])
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.N = N
        self.Lx = self.Ly = 1                     # only used with zero twist
        self.phx = self.phy = 0.0
        self.xs, self.ys = self.pos[:, 0], self.pos[:, 1]
        s = self.site
        self.hop, self.jb, self.xb = [], [], []
        for i in range(N):
            x, y = self.pos[i]
            for (dx, dy) in ((1, 0), (0, 1)):
                self.hop.append((i, s(x + dx, y + dy), t))
                self.jb.append((i, s(x + dx, y + dy), J))
            if tp != 0.0:
                for dy in (1, -1):
                    self.hop.append((i, s(x + 1, y + dy), tp))
        self.nn = np.array([[s(x + 1, y), s(x - 1, y), s(x, y + 1), s(x, y - 1)] for x, y in self.pos])
        self.nnn = np.array([[s(x + 1, y + 1), s(x - 1, y + 1), s(x + 1, y - 1), s(x - 1, y - 1)] for x, y in self.pos])
        # momentum grid: k . a1, k . a2 in 2 pi Z, reduced to the square-lattice BZ [0, 2pi)^2
        B = 2 * np.pi * self.Ainv.T
        ks = set()
        for m in range(-N, N + 1):
            for n in range(-N, N + 1):
                k = B @ np.array([m, n])
                ks.add((round(k[0] % (2 * np.pi), 10) % round(2 * np.pi, 10), round(k[1] % (2 * np.pi), 10) % round(2 * np.pi, 10)))
        ks = sorted(ks)
        assert len(ks) == N, ks
        self.kx = np.array([k[0] for k in ks]); self.ky = np.array([k[1] for k in ks])
        dx = self.xs[:, None] - self.xs[None, :]
        dy = self.ys[:, None] - self.ys[None, :]
        pairs = sorted(set(zip(dx.ravel().tolist(), dy.ravel().tolist())))
        pidx = {p: i for i, p in enumerate(pairs)}
        self.disp = np.vectorize(lambda a, b: pidx[(a, b)])(dx, dy)
        g = np.array(pairs, float)
        self.Cd = np.cos(np.outer(g[:, 0], self.kx) + np.outer(g[:, 1], self.ky)) / N

    def _key(self, x, y):
        f = self.Ainv @ np.array([x, y], float)
        f = np.round(f - np.floor(f + 1e-9), 9) % 1.0
        return (round(f[0], 6), round(f[1], 6))

    def site(self, x, y):
        return self.index[self._key(x, y)]

    def name(self):
        return self.label

    def square(self):
        return False


def make_lattice(spec, t, tp, J):
    """'LxXLy' (periodic torus) or 't8' (8-site tilted cluster)."""
    if spec.lower() == "t8":
        return Cluster2D((2, 2), (2, -2), t, tp, J, "t8")
    return Torus(*parse_lat(spec), t, tp, J)


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
    return (H + sp.diags(diag)).tocsr()


def S2_apply(lat, basis, psi):
    """S^2 psi for an S^z = 0 state: (N_e/2) psi + sum_{i<j} (S+_i S-_j + S-_i S+_j) psi."""
    N = lat.N
    ne = int(popcount(basis[:1])[0])
    out = 0.5 * ne * psi.astype(float)
    for i in range(N):
        for j in range(i + 1, N):
            for ops in ([(i, True), (i + N, False), (j + N, True), (j, False)],
                        [(i + N, True), (i, False), (j, True), (j + N, False)]):
                st, sg, ok = apply_string(basis, ops)
                src = np.nonzero(ok)[0]
                idx, f = lookup(basis, st[ok])
                np.add.at(out, idx, sg[ok] * psi[src])
    return out


def S2_expect(lat, basis, psi):
    return float(psi @ S2_apply(lat, basis, psi)) / float(psi @ psi)


# ============================================================================ symmetry
def site_perm_map(lat, basis, perm, tw=None):
    """(idx, sgn): (P psi)[idx[n]] = sgn[n] psi[n] for the site permutation perm, with the
    fermion reordering sign of each species and, if tw is given, a factor tw[i] = +-1 for every
    electron on site i (twisted-boundary gauge factor)."""
    N = lat.N
    new = np.zeros(len(basis), dtype=U64)
    sgn = np.ones(len(basis), dtype=np.int64)
    for spin in (0, 1):
        occ = ((basis[:, None] >> (np.arange(N, dtype=U64) + U64(spin * N))) & U64(1)).astype(bool)
        n = int(occ[0].sum())
        if n == 0:
            continue
        pos = np.nonzero(occ)[1].reshape(len(basis), n)
        img = perm[pos]
        inv = np.zeros(len(basis), dtype=np.int64)
        for a in range(n):
            for b in range(a + 1, n):
                inv += img[:, a] > img[:, b]
        sgn *= 1 - 2 * (inv & 1)
        if tw is not None:
            sgn *= np.prod(tw[pos], axis=1)
        new |= np.bitwise_or.reduce((U64(1) << (img.astype(U64) + U64(spin * N))), axis=1)
    idx, f = lookup(basis, new)
    assert f.all()
    return idx, sgn


def translation(lat, basis, dx, dy):
    """Translation by (dx, dy); for a twisted torus this is the gauge-covariant translation
    (factor -1 per electron that crosses a boundary with phase pi)."""
    perm = np.array([lat.site(lat.xs[i] + dx, lat.ys[i] + dy) for i in range(lat.N)])
    wx = (lat.xs + dx) // lat.Lx
    wy = (lat.ys + dy) // lat.Ly
    tw = 1 - 2 * ((np.abs(wx) * int(lat.phx != 0) + np.abs(wy) * int(lat.phy != 0)) % 2)
    return site_perm_map(lat, basis, perm, tw)


def apply_map(m, v):
    out = np.empty_like(v)
    out[m[0]] = m[1] * v
    return out


def compose(first, second):
    """Map of 'second after first'."""
    return second[0][first[0]], first[1] * second[1][first[0]]


class Sector:
    """Translations x {1,Mx} x {1,My} x {1,D} on a periodic torus (D only if square).
    Real 1-D irreps lam = {Tx, Ty, Mx, My[, D]} in {+1,-1}; isometry U onto Pi_lam."""

    def __init__(self, lat, basis):
        assert lat.phx == 0 and lat.phy == 0
        self.lat, self.D = lat, len(basis)
        Lx, Ly, N = lat.Lx, lat.Ly, lat.N
        xs, ys = lat.xs, lat.ys
        self.Tx = translation(lat, basis, 1, 0)
        self.Ty = translation(lat, basis, 0, 1)
        self.gens = ["Tx", "Ty", "Mx", "My"] + (["D"] if lat.square() else [])
        self.pg = []                         # (exponents (c, d, e), map)
        es = (0, 1) if lat.square() else (0,)
        for e in es:
            for c in (0, 1):
                for d in (0, 1):
                    perm = np.empty(N, dtype=np.int64)
                    for i in range(N):
                        x, y = xs[i], ys[i]
                        if e:
                            x, y = y, x
                        if c:
                            x = -x
                        if d:
                            y = -y
                        perm[i] = lat.site(x, y)
                    self.pg.append(((c, d, e), site_perm_map(lat, basis, perm)))
        self.genmap = {"Tx": self.Tx, "Ty": self.Ty,
                       "Mx": [m for k, m in self.pg if k == (1, 0, 0)][0],
                       "My": [m for k, m in self.pg if k == (0, 1, 0)][0]}
        if lat.square():
            self.genmap["D"] = [m for k, m in self.pg if k == (0, 0, 1)][0]
        self.cache = {}

    def gen_expect(self, psi):
        psi = psi / np.linalg.norm(psi)
        return {g: float(psi @ apply_map(self.genmap[g], psi)) for g in self.gens}

    def irrep_of(self, psi):
        ev = self.gen_expect(psi)
        ok = all(abs(abs(v) - 1) < 1e-6 for v in ev.values())
        return ok, {k: int(np.sign(v)) for k, v in ev.items()}, ev

    def candidates(self):
        """Real 1-D K=0 irreps (Mx = My forced when D is present: D Mx D = My)."""
        out = []
        if self.lat.square():
            for m in (1, -1):
                for d in (1, -1):
                    out.append(dict(Tx=1, Ty=1, Mx=m, My=m, D=d))
        else:
            for mx in (1, -1):
                for my in (1, -1):
                    out.append(dict(Tx=1, Ty=1, Mx=mx, My=my))
        return out

    def isometry(self, lam):
        key = tuple(sorted(lam.items()))
        if key in self.cache:
            return self.cache[key]
        D = self.D
        rep = np.full(D, D, dtype=np.int64)
        c = np.zeros(D, dtype=np.int64)
        gT = (np.arange(D), np.ones(D, dtype=np.int64))
        for b in range(self.lat.Ly):
            g = gT
            for a in range(self.lat.Lx):
                chT = lam["Tx"] ** a * lam["Ty"] ** b
                for (cc, dd, ee), P in self.pg:
                    ch = chT * lam["Mx"] ** cc * lam["My"] ** dd * (lam.get("D", 1) ** ee)
                    idx, s = compose(g, P)
                    lt = idx < rep
                    eq = idx == rep
                    c = np.where(eq, c + ch * s, c)
                    c[lt] = ch * s[lt]
                    rep[lt] = idx[lt]
                g = compose(g, self.Tx)
            gT = compose(gT, self.Ty)
        allowed = c != 0
        reps, inv = np.unique(rep[allowed], return_inverse=True)
        norm = np.bincount(inv, weights=c[allowed].astype(float) ** 2)
        rows = np.nonzero(allowed)[0]
        U = sp.csr_matrix((c[allowed] / np.sqrt(norm[inv]), (rows, inv)), shape=(D, len(reps)))
        self.cache[key] = U
        return U


# ============================================================================ ansatz
def mf_mu(density, t, tp, n=256):
    k = 2 * np.pi * np.arange(n) / n
    kx, ky = np.meshgrid(k, k)
    eps = np.sort((-2 * t * (np.cos(kx) + np.cos(ky)) - 4 * tp * np.cos(kx) * np.cos(ky)).ravel())
    m = int(round(density / 2 * eps.size))
    return 0.5 * (eps[m - 1] + eps[m])


class MFNG:
    """Momentum-field neural geminal (round 2), with unwrapped-displacement kernels so that
    twisted source clusters are exact. Parameter layout is size independent."""

    def __init__(self, lat, nup, C, Hf, Hb, Hj):
        self.lat, self.n, self.C = lat, nup, C
        self.Hf, self.Hb, self.Hj = Hf, (Hb if C > 0 else 0), Hj
        ck, cy = np.cos(lat.kx), np.cos(lat.ky)
        self.z = np.stack([(ck + cy) / 2, ck * cy], 1)
        self.gam = ck - cy
        nout = 2 * (C + 1)
        sizes = [("fW1", Hf * 2), ("fb1", Hf), ("fW2", nout * Hf), ("fb2", nout),
                 ("bW", self.Hb * 3), ("bb", self.Hb), ("ba", C * self.Hb),
                 ("jW", Hj * 3), ("jb", Hj), ("ja", Hj)]
        self.sl, o = {}, 0
        for nm, s_ in sizes:
            self.sl[nm] = slice(o, o + s_); o += s_
        self.npar = o
        self.nf = self.sl["fb2"].stop
        self.jas_mask = np.zeros(self.npar, bool)
        for nm in ("jW", "jb", "ja"):
            self.jas_mask[self.sl[nm]] = True
        self.nd = lat.Cd.shape[0]

    def P(self, th, nm, shape):
        return th[self.sl[nm]].reshape(shape)

    def init(self, rng, density, t, tp, delta0, eta):
        th = np.zeros(self.npar)
        nout = 2 * (self.C + 1)
        th[self.sl["fW1"]] = rng.standard_normal(self.Hf * 2) * 1.5
        th[self.sl["fb1"]] = rng.standard_normal(self.Hf) * 0.5
        th[self.sl["fW2"]] = 0.02 * rng.standard_normal(nout * self.Hf)
        th[self.sl["bW"]] = 0.5 * rng.standard_normal(self.Hb * 3)
        th[self.sl["ba"]] = 0.05 * rng.standard_normal(self.C * self.Hb)
        th[self.sl["jW"]] = 0.1 * rng.standard_normal(self.Hj * 3)
        th[self.sl["ja"]] = 0.1 * rng.standard_normal(self.Hj)
        return self.fit_channel0(th, density, t, tp, delta0, eta)

    def fit_channel0(self, th, density, t, tp, delta0, eta):
        th = th.copy()
        nout = 2 * (self.C + 1)
        lat = self.lat
        mu = mf_mu(density, t, tp)
        eps = -2 * t * (np.cos(lat.kx) + np.cos(lat.ky)) - 4 * tp * np.cos(lat.kx) * np.cos(lat.ky)
        xi = eps - mu
        Dk = delta0 * self.gam
        E = np.sqrt(xi ** 2 + Dk ** 2 + eta ** 2)
        Fk = Dk / (xi + E)
        Fk = Fk / max(1e-12, np.abs(Fk).max())
        W1 = self.P(th, "fW1", (self.Hf, 2)); b1 = th[self.sl["fb1"]]
        h = np.tanh(self.z @ W1.T + b1)
        X = np.concatenate([h, np.ones((len(h), 1)), self.gam[:, None] * h, self.gam[:, None]], 1)
        c = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ Fk)
        W2 = self.P(th, "fW2", (nout, self.Hf)).copy(); b2 = th[self.sl["fb2"]].copy()
        W2[0], b2[0] = c[:self.Hf], c[self.Hf]
        W2[1], b2[1] = c[self.Hf + 1:2 * self.Hf + 1], c[2 * self.Hf + 1]
        th[self.sl["fW2"]] = W2.ravel(); th[self.sl["fb2"]] = b2
        return th

    def fields(self, th, need_grad):
        nout = 2 * (self.C + 1)
        W1 = self.P(th, "fW1", (self.Hf, 2)); b1 = th[self.sl["fb1"]]
        W2 = self.P(th, "fW2", (nout, self.Hf)); b2 = th[self.sl["fb2"]]
        h = np.tanh(self.z @ W1.T + b1)
        out = h @ W2.T + b2
        f = np.stack([out[:, 2 * m] + self.gam * out[:, 2 * m + 1] for m in range(self.C + 1)])
        if not need_grad:
            return f, None
        Vk = len(self.z)
        dout = np.zeros((Vk, nout, self.nf))
        dh = 1 - h ** 2
        for o in range(nout):
            g1 = (W2[o][None, :] * dh)
            dout[:, o, self.sl["fW1"]] = (g1[:, :, None] * self.z[:, None, :]).reshape(Vk, -1)
            dout[:, o, self.sl["fb1"]] = g1
            blk = np.zeros((Vk, nout, self.Hf)); blk[:, o, :] = h
            dout[:, o, self.sl["fW2"]] = blk.reshape(Vk, -1)
            dout[:, o, self.sl["fb2"].start + o] = 1.0
        df = np.stack([dout[:, 2 * m] + self.gam[:, None] * dout[:, 2 * m + 1] for m in range(self.C + 1)])
        return f, df

    def precompute(self, th, need_grad):
        f, df = self.fields(th, need_grad)
        kap = f @ self.lat.Cd.T
        CJ = None if df is None else [self.lat.Cd @ df[m] for m in range(self.C + 1)]
        return kap, CJ

    def evaluate(self, th, states, need_grad, cache=None):
        lat, N, n, C = self.lat, self.lat.N, self.n, self.C
        B = len(states)
        if cache is None:
            cache = self.precompute(th, need_grad)
        kap, CJ = cache
        nu = ((states[:, None] >> np.arange(N, dtype=U64)) & U64(1)).astype(np.int64)
        nd = ((states[:, None] >> np.arange(N, 2 * N, dtype=U64)) & U64(1)).astype(np.int64)
        hole = (1 - nu - nd).astype(float)
        v = np.stack([hole, hole[:, lat.nn].sum(2) / 4, hole[:, lat.nnn].sum(2) / 4], 2)
        jW = self.P(th, "jW", (self.Hj, 3)); jb = th[self.sl["jb"]]; ja = th[self.sl["ja"]]
        zj = v @ jW.T + jb
        lc = np.logaddexp(zj, -zj) - np.log(2.0)
        logJ = (lc @ ja).sum(1)
        ups = np.nonzero(nu)[1].reshape(B, n)
        dns = np.nonzero(nd)[1].reshape(B, n)
        D = lat.disp[ups[:, :, None], dns[:, None, :]]
        bi = np.arange(B)[:, None]
        A = kap[0][D].copy()
        if C > 0:
            bW = self.P(th, "bW", (self.Hb, 3)); bb = th[self.sl["bb"]]; ba = self.P(th, "ba", (C, self.Hb))
            tb = np.tanh(v @ bW.T + bb)
            phi = tb @ ba.T
            pu = phi[bi, ups]
            pd = phi[bi, dns]
            for m in range(1, C + 1):
                A += pu[:, :, None, m - 1] * kap[m][D] * pd[:, None, :, m - 1]
        sgn, lad = np.linalg.slogdet(A)
        good = (sgn != 0) & np.isfinite(lad)
        logabs = np.where(good, logJ + lad, -np.inf)
        if not need_grad:
            return logabs, sgn, None
        O = np.zeros((B, self.npar))
        Wt = np.zeros_like(A)
        if good.any():
            Wt[good] = np.transpose(np.linalg.inv(A[good]), (0, 2, 1))
        nd_ = self.nd
        flat = (np.arange(B)[:, None, None] * nd_ + D).ravel()
        for m in range(C + 1):
            Q = Wt if m == 0 else Wt * pu[:, :, None, m - 1] * pd[:, None, :, m - 1]
            Hm = np.bincount(flat, weights=Q.ravel(), minlength=B * nd_).reshape(B, nd_)
            O[:, :self.nf] += Hm @ CJ[m]
        if C > 0:
            delta = np.zeros_like(tb)
            for m in range(1, C + 1):
                G = Wt * kap[m][D]
                Rr = (G * pd[:, None, :, m - 1]).sum(2)
                Cc = (G * pu[:, :, None, m - 1]).sum(1)
                dphi = np.zeros((B, N))
                dphi[bi, ups] = Rr
                dphi[bi, dns] = Cc
                O[:, self.sl["ba"].start + (m - 1) * self.Hb:self.sl["ba"].start + m * self.Hb] = \
                    np.einsum("bs,bsh->bh", dphi, tb)
                delta += dphi[:, :, None] * ba[m - 1][None, None, :]
            delta *= 1 - tb ** 2
            O[:, self.sl["bW"]] = np.einsum("bsh,bsf->bhf", delta, v).reshape(B, -1)
            O[:, self.sl["bb"]] = delta.sum(1)
        tj = np.tanh(zj) * ja
        O[:, self.sl["ja"]] = lc.sum(1)
        O[:, self.sl["jb"]] = tj.sum(1)
        O[:, self.sl["jW"]] = np.einsum("bsh,bsf->bhf", tj, v).reshape(B, -1)
        O[~good] = 0.0
        return logabs, sgn, O


# ============================================================================ exact-sum projected VMC
def state_vector(ans, th, basis, chunk):
    cache = ans.precompute(th, False)
    la, sg = [], []
    for s in range(0, len(basis), chunk):
        l_, g_, _ = ans.evaluate(th, basis[s:s + chunk], False, cache)
        la.append(l_); sg.append(g_)
    la, sg = np.concatenate(la), np.concatenate(sg)
    fin = np.isfinite(la)
    psi = np.where(fin, sg * np.exp(la - la[fin].max()), 0.0)
    return psi / np.linalg.norm(psi)


class Cluster:
    """One system on which an energy is evaluated: net, basis, isometry U (D x n), H_G = U^T H U."""

    def __init__(self, net, basis, U, HG, H=None):
        self.net, self.basis, self.U, self.HG, self.H = net, basis, U, HG, H
        self.V = net.lat.N


def reduced(cl, th, chunk, need_grad):
    """Projected energy of psi_theta in cluster cl: b = U^T psi, E = b^T H_G b / b^T b; with
    need_grad also the gradient g = 2 B^T (H_G b - E b)/|b|^2 and the metric
    S = B^T B/|b|^2 - (B^T b)(B^T b)^T/|b|^4, where B = U^T (psi * O) = db/dtheta."""
    net = cl.net
    psi = state_vector(net, th, cl.basis, chunk)
    b = cl.U.T @ psi
    n2 = float(b @ b)
    if n2 < 1e-14:
        return (np.inf, n2, None, None) if need_grad else (np.inf, n2)
    Hb = cl.HG @ b
    E = float(b @ Hb) / n2
    if not need_grad:
        return E, n2
    cache = net.precompute(th, True)
    B = np.zeros((cl.U.shape[1], net.npar))
    for s in range(0, len(cl.basis), chunk):
        sl = slice(s, s + chunk)
        _, _, O = net.evaluate(th, cl.basis[sl], True, cache)
        B += cl.U[sl].T @ (psi[sl, None] * O)
    g = 2 * (B.T @ (Hb - E * b)) / n2
    Bb = B.T @ b
    S = B.T @ B / n2 - np.outer(Bb, Bb) / n2 ** 2
    return E, n2, g, S


def optimise(clusters, th, steps, lr, shift, chunk, tag, lam=0.0, mask=None, Eref=None):
    """Exact-sum (projected) SR on L = sum_c E_c/V_c + lam |theta|^2 over the clusters, with
    backtracking. mask: boolean vector of trainable parameters (None = all)."""
    th = th.copy()
    P = len(th)
    idx = np.arange(P) if mask is None else np.nonzero(mask)[0]

    def loss_grad(t_, need):
        Es, L, g, S = [], lam * float(t_ @ t_), 2 * lam * t_, np.zeros((P, P))
        for cl in clusters:
            r = reduced(cl, t_, chunk, need)
            Es.append(r[0]); L += r[0] / cl.V
            if need and r[2] is not None:
                g = g + r[2] / cl.V; S += r[3] / cl.V
        return L, Es, g, S

    L, Es, g, S = loss_grad(th, True)
    hist = [Es]
    if steps <= 0:
        return th, Es, hist
    lr0 = lr
    for it in range(steps):
        Ss = S[np.ix_(idx, idx)]
        Sr = Ss + shift * np.diag(np.diag(Ss)) + 1e-6 * max(1e-12, np.diag(Ss).max()) * np.eye(len(idx))
        d = np.zeros(P)
        d[idx] = np.linalg.solve(Sr, g[idx])
        tried = 0
        while True:
            tn = th - lr * d
            Ln, Esn, _, _ = loss_grad(tn, False)
            if (np.isfinite(Ln) and Ln <= L + 1e-13) or tried >= 6:
                break
            lr *= 0.5; tried += 1
        if np.isfinite(Ln) and Ln <= L + 1e-13:
            th = tn
            L, Es, g, S = loss_grad(th, True)
            lr = min(lr * 1.1, 4 * lr0)
        hist.append(Es)
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            extra = ""
            if Eref is not None:
                extra = " relerr=" + ",".join(f"{(e - r) / abs(r):.3e}" for e, r in zip(Es, Eref))
            log(f"  [{tag}] step {it + 1}/{steps} E={','.join(f'{e:.8f}' for e in Es)} lr={lr:.3g}{extra}")
    return th, Es, hist


# ============================================================================ sign budget (from P1.1 / r2)
def sign_constrained_min(K, starts, maxiter):
    best = (np.inf, None)
    res_all = []

    def fg(a):
        Ka = K @ a
        nn = a @ a
        R = a @ Ka / nn
        return R, 2.0 * (Ka - R * a) / nn

    for name, a0 in starts:
        a0 = np.abs(a0) / np.linalg.norm(a0)
        r = minimize(fg, a0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(a0),
                     options=dict(maxiter=maxiter, ftol=1e-15, gtol=1e-12, maxcor=20))
        res_all.append((name, float(r.fun), int(r.nit), bool(r.success)))
        if r.fun < best[0]:
            best = (float(r.fun), r.x / np.linalg.norm(r.x))
    return best[0], best[1], res_all


def _lowest(A, v0=None):
    if A.shape[0] <= 2500:
        w, U = np.linalg.eigh(A.toarray())
        return w[:1], U[:, :1]
    try:
        return spla.eigsh(A, k=1, which="SA", tol=1e-9, v0=v0, maxiter=50000)
    except spla.ArpackNoConvergence as e:
        if len(e.eigenvalues):
            return e.eigenvalues, e.eigenvectors
        return spla.eigsh(A, k=2, which="SA", tol=1e-8, maxiter=100000)


def fixed_node(K, iters, floor):
    Kc = K.tocoo()
    off = Kc.row != Kc.col
    r, c, v = Kc.row[off], Kc.col[off], Kc.data[off]
    diag = K.diagonal()
    bad = v > 0
    Hk = sp.csr_matrix((v[~bad], (r[~bad], c[~bad])), shape=K.shape)
    rb, cb, vb = r[bad], c[bad], v[bad]
    D = K.shape[0]
    g = np.ones(D)
    v0 = None
    best = (np.inf, None)
    hist = []
    for _ in range(iters + 1):
        gg = np.maximum(g, floor * g.max())
        d = diag + np.bincount(rb, weights=vb * gg[cb] / gg[rb], minlength=D)
        w, U = _lowest((Hk + sp.diags(d)).tocsr(), v0)
        g = np.abs(U[:, 0]); g /= np.linalg.norm(g)
        v0 = g
        hist.append(float(w[0]))
        if w[0] < best[0]:
            best = (float(w[0]), g.copy())
    phi = best[1]
    return best[0], float(phi @ (K @ phi)), phi, hist


def sign_budget(HG, b, E0, args):
    s = np.where(b >= 0, 1.0, -1.0)
    S = sp.diags(s)
    K = (S @ HG @ S).tocsr()
    E_FN, E_FN_var, phi, hist = fixed_node(K, args.fn_iters, args.fn_floor)
    E_lb, _, starts = sign_constrained_min(K, [("abs_theta", b), ("abs_fixed_node", phi)], args.lbfgs_iters)
    E_up = min(E_lb, E_FN, E_FN_var)
    return dict(E_sign_up=float(E_up), E_FN=float(E_FN), E_lbfgsb=float(E_lb), A=float(E_up - E0))


# ============================================================================ observables
class Pairing:
    """d-wave bond-pair correlation Pd (copied from round 2)."""

    def __init__(self, lat, basis, nup, ndn, r_min):
        self.lat, self.basis = lat, basis
        self.low = make_basis(lat.N, nup - 1, ndn - 1)
        self.ops, self.eta, dist = [], [], []
        c0 = self.center(lat.xb[0])
        for b in lat.xb:
            self.ops.append(b)
            self.eta.append(1.0 if b[2] == "x" else -1.0)
            dist.append(self.dmin(c0, self.center(b)))
        self.mask = np.array(dist) >= r_min

    def center(self, b):
        return np.array([b[3] + (0.5 if b[2] == "x" else 0.0), b[4] + (0.5 if b[2] == "y" else 0.0)])

    def dmin(self, a, b):
        d = b - a
        d[0] -= self.lat.Lx * np.round(d[0] / self.lat.Lx)
        d[1] -= self.lat.Ly * np.round(d[1] / self.lat.Ly)
        return float(np.hypot(*d))

    def delta(self, b, vec):
        N = self.lat.N
        i, j = b[0], b[1]
        out = np.zeros(len(self.low))
        for coef, ops in ((1.0, [(j + N, False), (i, False)]), (-1.0, [(j, False), (i + N, False)])):
            st, sg, ok = apply_string(self.basis, ops)
            idx, f = lookup(self.low, st[ok])
            assert f.all()
            np.add.at(out, idx, coef * sg[ok] * vec[ok] / np.sqrt(2))
        return out

    def Pd(self, vec):
        if not self.mask.any():
            return float("nan")
        v0 = self.delta(self.ops[0], vec)
        return float(np.mean([self.eta[k] * (v0 @ self.delta(self.ops[k], vec)) for k in np.nonzero(self.mask)[0]]))


def sign_agreement(psi0, psi):
    psi = psi if psi0 @ psi >= 0 else -psi
    return float(np.sum(psi0 ** 2 * (np.sign(psi0 * psi) > 0)) - np.sum(psi0 ** 2 * (np.sign(psi0 * psi) < 0)))


# ============================================================================ setup helpers
def parse_lat(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def parse_instances(s):
    out = []
    for item in s.split(","):
        tp, J = item.split(":")
        out.append((float(tp), float(J)))
    return out


def new_net(lat, nup, args, C=None):
    return MFNG(lat, nup, args.channels if C is None else C, args.field_hidden, args.bf_hidden, args.jas_hidden)


def c4v_key(kx, ky):
    """C4v-invariant label of a momentum (the field inputs z_k and |gamma_k|)."""
    return (round((np.cos(kx) + np.cos(ky)) / 2, 9), round(np.cos(kx) * np.cos(ky), 9),
            round(abs(np.cos(kx) - np.cos(ky)), 9))


def coverage(src_specs, tgt):
    """(# target momenta on a C4v orbit present in some source grid, # target momenta)."""
    ks = set()
    for spec in src_specs:
        L_ = make_lattice(spec, 1, 0, 0)
        ks |= {c4v_key(a, b) for a, b in zip(L_.kx, L_.ky)}
    Lt = Torus(*parse_lat(tgt), 1, 0, 0)
    kt = [c4v_key(a, b) for a, b in zip(Lt.kx, Lt.ky)]
    return sum(k in ks for k in kt), len(kt)


def ed_lowest(H, k=3):
    if H.shape[0] <= 2500:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = spla.eigsh(H, k=k, which="SA", tol=1e-11)
    o = np.argsort(w)
    return w[o], v[:, o]


def select_instance(args, R):
    """Pre-committed rule: first (t', J) whose target ground state is a non-degenerate K = 0
    singlet in a real 1-D irrep of the group."""
    Lx, Ly = parse_lat(args.tgt)
    nup = (Lx * Ly - args.holes_tgt) // 2
    basis = make_basis(Lx * Ly, nup, nup)
    scan = []
    for tp, J in parse_instances(args.instances):
        lat = Torus(Lx, Ly, args.t, tp, J)
        H = build_H(lat, basis)
        sec = Sector(lat, basis)
        w, v = ed_lowest(H, k=3)
        ok1, lam, ev = sec.irrep_of(v[:, 0])
        s2 = S2_expect(lat, basis, v[:, 0])
        k0 = ok1 and lam["Tx"] == 1 and lam["Ty"] == 1
        nondeg = (w[1] - w[0]) > 1e-8 * abs(w[0])
        good = bool(ok1 and k0 and nondeg and abs(s2) < 1e-6)
        rec = dict(tp=tp, J=J, E0=float(w[0]), E1=float(w[1]), gen_expect=ev, S2=float(s2),
                   one_d_irrep=bool(ok1), K0=bool(k0), nondegenerate=bool(nondeg), accepted=good)
        scan.append(rec)
        log(f"instance t'={tp} J={J}: E0={w[0]:.8f} E1={w[1]:.8f} <S2>={s2:.2e} gens="
            f"{ {k: round(x, 6) for k, x in ev.items()} } -> {'ACCEPT' if good else 'reject'}")
        if good:
            R["instance_scan"] = scan
            return dict(tp=tp, J=J, lat=lat, basis=basis, H=H, sec=sec, lam=lam, nup=nup,
                        E0_full=float(w[0]), psi0_full=v[:, 0])
    R["instance_scan"] = scan
    return None


def build_source(args, tp, J, specs):
    """Source clusters at the target's hole doping (holes = doping x sites must be an integer
    with an even electron number)."""
    Lx, Ly = parse_lat(args.tgt)
    dop = args.holes_tgt / (Lx * Ly)
    out = []
    for spec in specs:
        lat = make_lattice(spec, args.t, tp, J)
        holes = dop * lat.N
        assert abs(holes - round(holes)) < 1e-9, f"doping {dop} not commensurate with {spec}"
        ne = lat.N - int(round(holes))
        assert ne % 2 == 0 and ne > 0, f"source {spec}: electron number must be even"
        basis = make_basis(lat.N, ne // 2, ne // 2)
        H = build_H(lat, basis)
        w, _ = ed_lowest(H, k=2)
        out.append(dict(lat=lat, H=H, E0=float(w[0]), basis=basis, nup=ne // 2))
    return out


def train_source(args, src, C, lam, seed, tag):
    dens = 2 * src[0]["nup"] / src[0]["lat"].N
    clusters = []
    for s in src:
        net = new_net(s["lat"], s["nup"], args, C)
        I = sp.identity(len(s["basis"]), format="csr")
        clusters.append(Cluster(net, s["basis"], I, s["H"]))
    th0 = clusters[0].net.init(np.random.default_rng(1000 + seed), dens, args.t, args.tp_sel, args.delta0, args.eta)
    th, Es, _ = optimise(clusters, th0, args.steps_src, args.lr, args.diag_shift, args.chunk, tag,
                         lam=lam, Eref=[s["E0"] for s in src])
    # identifiability: joint SR metric at the optimum (per-site weighted)
    S = np.zeros((len(th), len(th)))
    for cl in clusters:
        S += reduced(cl, th, args.chunk, True)[3] / cl.V
    ev = np.linalg.eigvalsh(S)
    n_null = int(np.sum(ev < 1e-8 * ev.max()))
    rel = [float((e - s["E0"]) / abs(s["E0"])) for e, s in zip(Es, src)]
    log(f"source {tag}: E={['%.8f' % e for e in Es]} relerr={['%.2e' % r for r in rel]} "
        f"npar={len(th)} null-dim(S)={n_null} |theta|={np.linalg.norm(th):.3f}")
    return th, dict(E=[float(e) for e in Es], relerr=rel, n_null=n_null, npar=len(th),
                    norm=float(np.linalg.norm(th)))


# ============================================================================ main runs
def target_setup(args, R):
    inst = select_instance(args, R)
    if inst is None:
        return None
    args.tp_sel, args.J_sel = inst["tp"], inst["J"]
    lat, basis, H, sec, lam = inst["lat"], inst["basis"], inst["H"], inst["sec"], inst["lam"]
    U = sec.isometry(lam)
    HG = (U.T @ H @ U).tocsr()
    w, v = ed_lowest(HG, k=2)
    E0 = float(w[0]); V = lat.N
    b0 = v[:, 0]
    psi0 = U @ b0
    pair = Pairing(lat, basis, inst["nup"], inst["nup"], args.r_min)
    Pd0 = pair.Pd(psi0)
    gap = float(w[1] - w[0])
    s0 = np.where(b0 >= 0, 1.0, -1.0)
    K0 = (sp.diags(s0) @ HG @ sp.diags(s0)).tocsr()
    Ecal, _, _ = sign_constrained_min(K0, [("uniform", np.ones(len(b0)))], args.lbfgs_iters)
    R["target"] = dict(tp=inst["tp"], J=inst["J"], irrep=lam, n_irrep=int(U.shape[1]), dim=len(basis),
                       E0=E0, E0_full=inst["E0_full"], E1_irrep=float(w[1]), gap_per_site=gap / V,
                       competitor=bool(gap / V < COMP_TOL), Pd_E0=Pd0, calib_err=float(Ecal - E0))
    log(f"target {lat.name()} t'={inst['tp']} J={inst['J']} irrep={lam} n_irrep={U.shape[1]} "
        f"E0={E0:.8f} (full {inst['E0_full']:.8f}) gap/V={gap / V:.4e} "
        f"{'COMPETITOR' if gap / V < COMP_TOL else 'no same-irrep competitor below 0.005 t/site: sign half only'}")
    log(f"calibration: E_sign(exact signs) - E0 = {Ecal - E0:.3e}; Pd(E0)={Pd0:.4e}")
    return dict(lat=lat, basis=basis, H=H, sec=sec, lam=lam, U=U, HG=HG, E0=E0, psi0=psi0,
                pair=pair, nup=inst["nup"], V=V)


def make_recorder(args, R, T, dump):
    def record(name, net, th, E_known=None, sign=True):
        psi = state_vector(net, th, T["basis"], args.chunk)
        E_raw = float(psi @ (T["H"] @ psi))
        b = T["U"].T @ psi
        w = float(b @ b)
        if w < 1e-14:
            R["arms"][name] = dict(E=float("inf"), eps=float("inf"), weight=w, E_raw=E_raw)
            log(f"arm {name:9s}: zero weight in the ground-state irrep")
            dump(); return
        E = float(b @ (T["HG"] @ b)) / w
        psiP = T["U"] @ (b / np.sqrt(w))
        rec = dict(E=E, eps=(E - T["E0"]) / T["V"], relerr=(E - T["E0"]) / abs(T["E0"]), weight=w,
                   E_raw=E_raw, eps_raw=(E_raw - T["E0"]) / T["V"],
                   sign_agreement=sign_agreement(T["psi0"], psiP), Pd=T["pair"].Pd(psiP),
                   fidelity=float((T["psi0"] @ psiP) ** 2), theta=th.tolist())
        if sign:
            sb = sign_budget(T["HG"], b, T["E0"], args)
            rec["sign"] = sb
            rec["A"] = sb["A"]
        R["arms"][name] = rec
        log(f"arm {name:9s}: E={E:.8f} eps={rec['eps']:.4e} (raw {rec['eps_raw']:.4e}, irrep weight {w:.3f})"
            + (f" A/V={rec['A'] / T['V']:.4e}" if sign else "")
            + f" sign-agree={rec['sign_agreement']:.4f} Pd={rec['Pd']:.4e} fid={rec['fidelity']:.4f}")
        dump()
    return record


def run(args):
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "results", f"{args.role}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R = dict(params=vars(args).copy(), arms={}, role=args.role, seed=args.seed)

    def dump():
        with open(out, "w") as fh:
            json.dump(R, fh, indent=1, default=float)

    T = target_setup(args, R)
    if T is None:
        R["verdict_task"] = "VOID-SECTOR (no candidate instance has a K=0, 1-D, singlet, non-degenerate ground state)"
        log(R["verdict_task"]); dump(); return
    R["params"]["tp_sel"], R["params"]["J_sel"] = args.tp_sel, args.J_sel
    dump()
    record = make_recorder(args, R, T, dump)
    lat, nup = T["lat"], T["nup"]
    dens = 2 * nup / lat.N
    clT = lambda net: [Cluster(net, T["basis"], T["U"], T["HG"])]
    rng = np.random.default_rng(2000 + args.seed)

    if args.role == "foil":
        net = new_net(lat, nup, args, 0)
        th0 = net.init(rng, dens, args.t, args.tp_sel, args.delta0, args.eta)
        th, Es, h = optimise(clT(net), th0, args.steps_long, args.lr, args.diag_shift, args.chunk, "foil_opt",
                             Eref=[T["E0"]])
        R["hist_foil_opt"] = [e[0] for e in h]
        record("foil_opt", net, th)
    elif args.role == "scratch":
        net = new_net(lat, nup, args)
        th0 = net.init(rng, dens, args.t, args.tp_sel, args.delta0, args.eta)
        th, Es, h = optimise(clT(net), th0, args.steps_long, args.lr, args.diag_shift, args.chunk, "scrL",
                             Eref=[T["E0"]])
        R["hist_scrL"] = [e[0] for e in h]
        record("scrL", net, th)
    else:
        specs = args.sources.split(",")
        src = build_source(args, args.tp_sel, args.J_sel, specs)
        log("source clusters: " + ", ".join(f"{s['lat'].name()} E0={s['E0']:.8f}" for s in src))
        th_r, info_r = train_source(args, src, args.channels, args.lam, args.seed, "route TCMN")
        th_f, info_f = train_source(args, src, 0, args.lam, args.seed, "foil TCMN")
        th_2, info_2 = train_source(args, src[:1], args.channels, 0.0, args.seed, "route r2-style")
        R["source"] = dict(clusters=specs, coverage=coverage(specs, args.tgt),
                           E0=[s["E0"] for s in src], route=info_r, foil=info_f, r2style=info_2)
        dump()
        net = new_net(lat, nup, args)
        net_f = new_net(lat, nup, args, 0)
        # zero-shot sector rule
        sec_E = {}
        psi_tr = state_vector(net, th_r, T["basis"], args.chunk)
        for cand in T["sec"].candidates():
            Uc = T["sec"].isometry(cand)
            bc = Uc.T @ psi_tr
            wc = float(bc @ bc)
            Ec = float(bc @ (Uc.T @ (T["H"] @ (Uc @ bc)))) / wc if wc > 1e-10 else float("inf")
            sec_E[json.dumps(cand, sort_keys=True)] = dict(E=Ec, weight=wc)
        pick = min(sec_E, key=lambda k: sec_E[k]["E"])
        hit = json.loads(pick) == {k: int(v) for k, v in T["lam"].items()}
        R["sector_rule"] = dict(energies=sec_E, picked=pick, correct=bool(hit))
        log(f"zero-shot sector rule: picked {pick} (ED irrep {T['lam']}) -> {'correct' if hit else 'WRONG'}")
        R["S2_tr0"] = S2_expect(lat, T["basis"], psi_tr)
        log(f"<S^2> of the transferred state = {R['S2_tr0']:.2e}")
        record("tr0", net, th_r)
        record("tr0_r2", net, th_2, sign=False)
        record("foil_tr", net_f, th_f)
        th_tB, _, _ = optimise(clT(net), th_r, args.budget, args.lr, args.diag_shift, args.chunk, "trB", Eref=[T["E0"]])
        record("trB", net, th_tB)
        th_fs, _, _ = optimise(clT(net), th_r, args.budget, args.lr, args.diag_shift, args.chunk, "fstB",
                               mask=net.jas_mask, Eref=[T["E0"]])
        record("fstB", net, th_fs)
        th0 = net.init(rng, dens, args.t, args.tp_sel, args.delta0, args.eta)
        th_sB, _, _ = optimise(clT(net), th0, args.budget, args.lr, args.diag_shift, args.chunk, "scrB", Eref=[T["E0"]])
        record("scrB", net, th_sB)
        th_L, _, h = optimise(clT(net), th_tB, max(0, args.steps_long - args.budget), args.lr, args.diag_shift,
                              args.chunk, "trL", Eref=[T["E0"]])
        R["hist_trL"] = [e[0] for e in h]
        record("trL", net, th_L, sign=False)
    R["done"] = True
    dump()
    log(f"done ({out})")


# ============================================================================ aggregate
def _med(x):
    return float(np.median(x)) if len(x) else float("nan")


def decide(res, void_tol):
    """Pre-committed global decision. res: role -> list of result dicts (3 seeds each)."""
    tr, fo, sc = res.get("transfer", []), res.get("foil", []), res.get("scratch", [])
    out = dict()
    if len(tr) < 3 or len(fo) < 3 or len(sc) < 3:
        out["verdict_31"] = out["verdict_32"] = f"INCOMPLETE (transfer {len(tr)}, foil {len(fo)}, scratch {len(sc)} of 3)"
        return out
    T = tr[0]["target"]
    E0 = T["E0"]
    arm = lambda rs, k, f: [r["arms"][k][f] for r in rs]
    eps = {k: arm(tr, k, "eps") for k in ("tr0", "tr0_r2", "foil_tr", "trB", "fstB", "scrB", "trL")}
    eps["foil_opt"] = arm(fo, "foil_opt", "eps")
    eps["scrL"] = arm(sc, "scrL", "eps")
    A = {k: arm(tr, k, "A") for k in ("tr0", "foil_tr", "trB", "fstB", "scrB")}
    A["foil_opt"] = arm(fo, "foil_opt", "A")
    eps_opt = min(eps["trL"] + eps["scrL"])
    lat = parse_lat(tr[0]["params"]["tgt"]); Vs = lat[0] * lat[1]
    relerr_opt = eps_opt * Vs / abs(E0)
    m = {k: _med(v) for k, v in eps.items()}
    mA = {k: _med(v) for k, v in A.items()}
    spread = max(eps["tr0"]) - min(eps["tr0"])
    spreadA = max(A["tr0"]) - min(A["tr0"])
    P0 = T["Pd_E0"]
    ePd_trB = _med([abs(p - P0) for p in arm(tr, "trB", "Pd")])
    ePd_fo = _med([abs(p - P0) for p in arm(fo, "foil_opt", "Pd")])
    sec_ok = sum(bool(r["sector_rule"]["correct"]) for r in tr) >= 2
    calib = max(r["target"]["calib_err"] for r in tr + fo)
    calib_ok = calib <= CALIB_TOL * max(1e-12, min(min(v) for v in A.values()))
    c = dict(a=m["tr0"] <= m["foil_opt"], b=m["trB"] <= 0.5 * m["scrB"],
             c=(m["trB"] - eps_opt) <= 0.5 * (m["foil_opt"] - eps_opt), d=ePd_trB <= ePd_fo + GUARD_TOL,
             e=m["tr0"] < m["tr0_r2"], I=spread <= 0.5 * m["tr0"], S=sec_ok)
    out.update(m=m, mA=mA, eps_opt=eps_opt, relerr_opt=relerr_opt, spread_eps_tr0=spread, spread_A_tr0=spreadA,
               ePd_trB=ePd_trB, ePd_foil_opt=ePd_fo, criteria_31=c, calib_ok=bool(calib_ok), calib=calib,
               eps_seeds=eps, A_seeds=A)
    if relerr_opt > void_tol:
        out["verdict_31"] = out["verdict_32"] = "VOID (converged ansatz too far from E0)"
        return out
    if all(c.values()):
        v31 = "PASS"
    elif (m["tr0"] >= 2 * m["foil_opt"] or m["trB"] >= 0.9 * m["scrB"] or spread > m["tr0"]
          or not c["d"] or not c["S"]):
        v31 = "FAIL"
    else:
        v31 = "AMBIGUOUS (not promoted)"
    out["verdict_31"] = v31
    if (m["fstB"] <= m["foil_opt"] and mA["tr0"] <= 0.5 * mA["foil_opt"] and spreadA <= 0.5 * mA["tr0"]
            and sec_ok and calib_ok):
        v32 = "PASS"
    elif mA["tr0"] >= 0.9 * mA["foil_opt"] or m["fstB"] >= 1.5 * m["foil_opt"] or not sec_ok:
        v32 = "FAIL"
    else:
        v32 = "AMBIGUOUS (not promoted)"
    out["verdict_32"] = v32
    return out


def aggregate(d, void_tol):
    res = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, fn)))
        if r.get("verdict_task", "").startswith("VOID-SECTOR"):
            print(f"{fn}: {r['verdict_task']}")
            print("GLOBAL: VOID-SECTOR"); return
        if r.get("done"):
            res.setdefault(r["role"], []).append(r)
    out = decide(res, void_tol)
    if "m" in out:
        T = res["transfer"][0]["target"]
        print(f"target t'={T['tp']} J={T['J']} irrep={T['irrep']} E0={T['E0']:.8f} gap/V={T['gap_per_site']:.4e} "
              f"competitor={T['competitor']} calib={out['calib']:.2e}")
        for k, v in out["eps_seeds"].items():
            a = out["A_seeds"].get(k)
            print(f"  {k:9s} eps median={out['m'][k]:.4e} seeds={['%.4e' % x for x in v]}"
                  + (f"  A median={out['mA'][k]:.4e}" if a is not None else ""))
        print(f"  eps_opt={out['eps_opt']:.4e} relerr_opt={out['relerr_opt']:.3e} spread(eps_tr0)={out['spread_eps_tr0']:.3e} "
              f"spread(A_tr0)={out['spread_A_tr0']:.3e} |dPd| trB={out['ePd_trB']:.3e} foil_opt={out['ePd_foil_opt']:.3e}")
        print(f"  Route 3.1 criteria: {out['criteria_31']}")
    print(f"GLOBAL Route 3.1 (TCMN transfer): {out['verdict_31']}")
    print(f"GLOBAL Route 3.2 (frozen-sign transfer): {out['verdict_32']}")


# ============================================================================ self-tests
def selftest(args):
    ok_all = True

    def rep(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        log(f"selftest({name}) {msg} {'OK' if ok else 'FAIL'}")

    # (a) band sums, including twisted grids
    for (Lx, Ly, px, py) in ((4, 4, 0, 0), (4, 2, 0, 0), (4, 2, 0, np.pi), (2, 2, np.pi, 0), (2, 2, np.pi, np.pi)):
        lat = Torus(Lx, Ly, 1.0, -0.25, 0.0, px, py)
        eps = np.sort(-2 * (np.cos(lat.kx) + np.cos(lat.ky)) - 4 * (-0.25) * np.cos(lat.kx) * np.cos(lat.ky))
        for nf in (1, 3):
            basis = make_basis(lat.N, nf, 0)
            Hs = build_H(lat, basis, spinful=False)
            e = np.linalg.eigvalsh(Hs.toarray())[0]
            herm = abs(Hs - Hs.T).max()
            rep("a", abs(e - eps[:nf].sum()) < 1e-10 and herm < 1e-12,
                f"spinless {lat.name()} N_f={nf}: E_ED={e:.10f} band={eps[:nf].sum():.10f}")
    lat = make_lattice("t8", 1.0, -0.25, 0.0)
    eps = np.sort(-2 * (np.cos(lat.kx) + np.cos(lat.ky)) - 4 * (-0.25) * np.cos(lat.kx) * np.cos(lat.ky))
    for nf in (1, 3):
        basis = make_basis(lat.N, nf, 0)
        e = np.linalg.eigvalsh(build_H(lat, basis, spinful=False).toarray())[0]
        rep("a", abs(e - eps[:nf].sum()) < 1e-10, f"spinless t8 N_f={nf}: E_ED={e:.10f} band={eps[:nf].sum():.10f} "
            f"grid (units of pi) {sorted(zip(np.round(lat.kx / np.pi, 3), np.round(lat.ky / np.pi, 3)))}")
    # (b)-(d) on the two default source clusters (4x2 torus, 8-site tilted), 2 holes
    rng = np.random.default_rng(1)
    for lat in (Torus(4, 2, args.t, -0.25, 0.4), make_lattice("t8", args.t, -0.25, 0.4)):
        basis = make_basis(lat.N, 3, 3)
        H = build_H(lat, basis)
        rep("b", abs(H - H.T).max() < 1e-12, f"t-J {lat.name()} dim={len(basis)} hermitian")
        net = new_net(lat, 3, args)
        th = net.init(rng, 6 / 8, args.t, -0.25, args.delta0, args.eta) + 0.05 * rng.standard_normal(net.npar)
        psi = state_vector(net, th, basis, args.chunk)
        Tx, Ty = translation(lat, basis, 1, 0), translation(lat, basis, 0, 1)
        ox, oy = psi @ apply_map(Tx, psi), psi @ apply_map(Ty, psi)
        comm = max(np.abs(H @ apply_map(T_, v) - apply_map(T_, H @ v)).max()
                   for T_ in (Tx, Ty) for v in [rng.standard_normal(len(basis))])
        rep("c", abs(ox - 1) < 1e-10 and abs(oy - 1) < 1e-10 and comm < 1e-12,
            f"{lat.name()} <psi|T~x psi>={ox:.12f} <psi|T~y psi>={oy:.12f} max|[H,T~]v|={comm:.1e}")
        sub = basis[::5]
        la, sg, O = net.evaluate(th, sub, True)
        err = 0.0
        for p in range(net.npar):
            e = np.zeros(net.npar); e[p] = 1e-6
            lp, sp_, _ = net.evaluate(th + e, sub, False)
            lm, sm, _ = net.evaluate(th - e, sub, False)
            fin = np.isfinite(lp) & np.isfinite(lm) & np.isfinite(la) & (sp_ == sm)
            fin &= la > la[np.isfinite(la)].max() - 15     # skip symmetry zeros (round-off amplitudes)
            with np.errstate(invalid="ignore"):
                fd = (lp - lm)[fin] / 2e-6
            err = max(err, float(np.max(np.abs(fd - O[fin, p])) / (1 + np.max(np.abs(fd)))))
        rep("d", err < 1e-5, f"{lat.name()} log-derivative max rel err over {net.npar} params = {err:.2e}")
    # (e) symmetry sectors on a square (4x4, N_e = 2) and a rectangle (4x2, N_e = 4)
    for (Lx, Ly, nup) in ((4, 4, 1), (4, 2, 2)):
        lat = Torus(Lx, Ly, 1.0, -0.25, 0.4)
        basis = make_basis(lat.N, nup, nup)
        H = build_H(lat, basis)
        sec = Sector(lat, basis)
        v = rng.standard_normal(len(basis))
        comm = max(np.abs(H @ apply_map(m_, v) - apply_map(m_, H @ v)).max() for _, m_ in sec.pg)
        wf = np.linalg.eigvalsh(H.toarray())
        okU, inside, tot = True, True, 0
        for cand in sec.candidates():
            U = sec.isometry(cand)
            tot += U.shape[1]
            if U.shape[1] == 0:
                continue
            okU &= abs((U.T @ U - sp.identity(U.shape[1])).toarray()).max() < 1e-12
            wk = np.linalg.eigvalsh((U.T @ H @ U).toarray())
            inside &= all(np.min(np.abs(wf - e)) < 1e-9 for e in wk)
        w, vv = np.linalg.eigh(H.toarray())
        ok1, lam, ev = sec.irrep_of(vv[:, 0])
        inirrep = float("nan")
        if ok1 and lam["Tx"] == 1 and lam["Ty"] == 1:
            U = sec.isometry(lam)
            inirrep = float(np.linalg.norm(U.T @ vv[:, 0]) ** 2)
            lowest = np.linalg.eigvalsh((U.T @ H @ U).toarray())[0]
            inside &= abs(lowest - w[0]) < 1e-10
        rep("e", comm < 1e-12 and okU and inside and (not ok1 or not np.isfinite(inirrep) or abs(inirrep - 1) < 1e-10),
            f"{lat.name()} N_e={2 * nup}: max|[H,P_g]v|={comm:.1e}, U^T U = I, irrep spectra inside full "
            f"(K=0 1-D irrep dims sum {tot} of {len(basis)}), GS gens={ {k: round(x, 4) for k, x in ev.items()} }, "
            f"GS weight in its irrep={inirrep:.12f}")
    # (f) S^2 operator on 2x2 with N_e = 2 and N_e = 4 (spectrum must be S(S+1))
    for nup in (1, 2):
        lat = Torus(2, 2, 1.0, 0.0, 1.0)
        basis = make_basis(4, nup, nup)
        M = np.stack([S2_apply(lat, basis, e) for e in np.eye(len(basis))], 1)
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))
        okS = abs(M - M.T).max() < 1e-12 and all(min(abs(x - S * (S + 1)) for S in range(4)) < 1e-10 for x in ev)
        rep("f", okS, f"S^2 on 2x2 N_e={2 * nup}: symmetric, spectrum {sorted(set(np.round(ev, 8)))}")
    # (g) projected gradient vs finite differences (4x2, N_e = 4, ground-state irrep)
    lat = Torus(4, 2, 1.0, -0.25, 0.4)
    basis = make_basis(8, 2, 2)
    H = build_H(lat, basis)
    sec = Sector(lat, basis)
    U = sec.isometry(dict(Tx=1, Ty=1, Mx=1, My=1))
    cl = Cluster(new_net(lat, 2, args), basis, U, (U.T @ H @ U).tocsr())
    th = cl.net.init(rng, 0.5, 1.0, -0.25, args.delta0, args.eta) + 0.1 * rng.standard_normal(cl.net.npar)
    E, n2, g, S = reduced(cl, th, args.chunk, True)
    err = 0.0
    for p in rng.choice(cl.net.npar, 12, replace=False):
        e = np.zeros(cl.net.npar); e[p] = 1e-5
        fd = (reduced(cl, th + e, args.chunk, False)[0] - reduced(cl, th - e, args.chunk, False)[0]) / 2e-5
        err = max(err, abs(fd - g[p]) / (1 + abs(fd)))
    rep("g", err < 1e-6 and np.linalg.eigvalsh(S).min() > -1e-10,
        f"projected-energy gradient vs finite differences max rel err {err:.2e}; metric PSD")
    # (h) momentum coverage by the source clusters (C4v orbits)
    specs = args.sources.split(",")
    nc, nt = coverage(specs, args.tgt)
    single = coverage(specs[:1], args.tgt)[0]
    log(f"selftest(h) [report] {'complete' if nc == nt else 'INCOMPLETE'} coverage: sources {specs} cover {nc} of {nt} {args.tgt} momenta on C4v orbits "
        f"(first source alone: {single} of {nt})")
    log("SELFTEST", "PASSED" if ok_all else "FAILED")
    return ok_all


# ============================================================================ CLI
def parse_args(argv):
    p = argparse.ArgumentParser(description="P4.2 round-3 decisive test (see module docstring)")
    p.add_argument("--task-id", type=int, default=-1)
    p.add_argument("--role", default="transfer", choices=ROLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", default="4x2,t8")
    p.add_argument("--tgt", default="4x4")
    p.add_argument("--holes-tgt", type=int, default=4)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--instances", default="-0.25:0.4,-0.25:0.6,0.25:0.4,-0.1:0.4,0.0:0.4")
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--field-hidden", type=int, default=8)
    p.add_argument("--bf-hidden", type=int, default=4)
    p.add_argument("--jas-hidden", type=int, default=4)
    p.add_argument("--delta0", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=1e-5)
    p.add_argument("--steps-src", type=int, default=800)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--steps-long", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--diag-shift", type=float, default=0.01)
    p.add_argument("--fn-iters", type=int, default=6)
    p.add_argument("--fn-floor", type=float, default=1e-3)
    p.add_argument("--lbfgs-iters", type=int, default=3000)
    p.add_argument("--r-min", type=float, default=2.0)
    p.add_argument("--void-tol", type=float, default=0.05)
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default="")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--timing", action="store_true")
    a = p.parse_args(argv)
    if a.task_id >= 0:
        if a.task_id > 8:
            p.error("--task-id must be 0..8")
        a.role, a.seed = ROLES[a.task_id // 3], a.task_id % 3
    a.tp_sel = a.J_sel = None
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate, args.void_tol)
        return
    if args.selftest:
        sys.exit(0 if selftest(args) else 2)
    if args.timing:
        Lx, Ly = parse_lat(args.tgt)
        nup = (Lx * Ly - args.holes_tgt) // 2
        tp, J = parse_instances(args.instances)[0]
        lat = Torus(Lx, Ly, args.t, tp, J)
        basis = make_basis(lat.N, nup, nup)
        t0 = time.time(); H = build_H(lat, basis); t1 = time.time()
        log(f"target {lat.name()} dim={len(basis)}; H build {t1 - t0:.1f}s nnz={H.nnz}")
        net = new_net(lat, nup, args)
        th = net.init(np.random.default_rng(0), 2 * nup / lat.N, args.t, tp, args.delta0, args.eta)
        t1 = time.time(); net.evaluate(th, basis[:args.chunk], True); t2 = time.time()
        net.evaluate(th, basis[:args.chunk], False); t3 = time.time()
        nch = int(np.ceil(len(basis) / args.chunk))
        log(f"chunk={args.chunk}: grad pass {t2 - t1:.2f}s, value pass {t3 - t2:.2f}s; full grad+value "
            f"~{nch * (t2 - t1 + t3 - t2):.0f}s, value-only ~{nch * (t3 - t2):.0f}s per pass (npar={net.npar}, "
            f"foil {new_net(lat, nup, args, 0).npar})")
        return
    if not selftest(args):
        sys.exit(2)
    run(args)


if __name__ == "__main__":
    main()
