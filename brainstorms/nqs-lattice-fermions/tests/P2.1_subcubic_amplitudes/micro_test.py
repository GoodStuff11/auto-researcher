"""
P2.1 decisive micro-test: exact O(N_e)-per-ratio amplitude ratios for a
Local + Global-Bottleneck backflow (LGB-BF) determinant on the doped t-t'-U
Hubbard model, compared against brute-force determinants.

What is tested
--------------
Wave function (spinful, one N_e x N_e determinant over 2V spin-orbitals
p = sigma*V + x, sigma in {0 (up), 1 (down)}, x a lattice site):
    psi(n) = det A(n),   A_{j,:}(n) = Phi0[p_j] + h_{x_j}(n) @ Theta[sigma_j]
                                       + (u_{x_j}(n) * c(n)) @ Omega[sigma_j]
  * h_x(n)  = tanh(W1 @ patch_x(n) + b1) in R^F : LOCAL features, patch_x is
    the up/down occupations in the (2r+1)x(2r+1) Chebyshev ball around x
    (a one-layer translation-equivariant convolution of range r).
  * c(n)    = tanh(sum_x Pool[:, x, :] @ h_x(n)) in R^K : GLOBAL channels
    (position-dependent linear pooling of all local features, then tanh).
  * u_x(n)  = Wu @ h_x(n) in R^K : local gates of the global channels.
  * Phi0    = free-fermion (t-t') orbitals, Theta (2,F,N_e), Omega (2,K,N_e)
    random. Parameters are random (this is an algorithmic test, not a training
    run). A Jastrow factor is omitted: it multiplies every ratio by an O(1)-cost
    (per hop) factor and does not change any exponent.
A hop n -> n' changes the rows of all particles sitting within range r of the
two touched sites (|S| = O(r^2) rows, all columns) plus a rank-<=2K change of
every other row through c. Hence A(n') = A(n) + E_S D + Ug W, a rank
m = |S| + 2K update, and with the projected Green's function
G = [Phi0; Theta_up; Theta_dn; Omega_up; Omega_dn] @ A^{-1}
(carried across Markov steps by Woodbury, O(V N_e m) per ACCEPTED move) every
ratio costs O(N_e (|S|(F+K) + K^2)), i.e. O(N_e) at fixed (r, F, K), and a full
local energy (O(z N_e) connected configurations, z = 8 with t') costs O(N_e^2).
Brute force rebuilds A(n') and takes an LU determinant: O(N_e^3) per ratio,
O(N_e^4) per local energy.

Measured quantities per lattice size L x L (PBC, t=1, t'=--tp, U=--U,
N_e = 2*round(V(1-delta)/2), N_up = N_dn = N_e/2):
  T_E_fast / T_E_brute     : wall time of one full local energy (all hops).
  T_sw_fast / T_sw_brute   : wall time of one Metropolis sweep (N_e proposals).
  eps_ratio                : max over all hops of --nconf configurations of
                             |r_fast - r_brute| / (1 + |r_brute|).
  eps_eloc                 : max |E_loc,fast - E_loc,brute| / (1 + |E_loc,brute|).
  drift_G                  : max|G_carried - G_recomputed| / max|G_recomputed|
                             after one fast sweep (Woodbury stability).
Scaling exponents are least-squares slopes of log(time) vs log(N_e) over the
largest --fit-last sizes.

Verdict (scalar metric M = slope of log T_E_fast vs log N_e):
  PASS      : M <= 2.3 and eps_max <= 1e-9 and slope(T_E_brute) >= 3.3
              and slope(T_sw_fast) <= 3.3
  FAIL      : M >= 2.8 or eps_max > 1e-6
  AMBIGUOUS : anything else (pre-committed action: rerun with
              --sizes 24,28,32,36,40 --nconf 2; if still AMBIGUOUS, the
              asymptotic claim is declared FAILED).
eps_max = max(eps_ratio, eps_eloc) over all sizes.

Enumeration check (--check-enum, default on): on a 3x2 PBC t-t'-U cluster with
N_up = N_dn = 2 (225 configurations) it builds the exact Jordan-Wigner
Hamiltonian matrix H and the full vector psi, and checks that
  sum_n psi(n)^2 E_loc(n) / sum_n psi(n)^2  ==  psi^T H psi / psi^T psi
with E_loc computed BOTH by brute-force row-replacement determinants and by the
fast algorithm. This proves that the fast ratios implement the same variational
family and the correct fermionic sign convention.

Usage
-----
  python micro_test.py [options]

Options (all optional):
  --sizes STR      Comma-separated linear sizes L of L x L lattices.
                   Default "8,12,16,20,24,28,32" (N_e = 56 ... 896). Each L must be >= 4.
  --delta FLOAT    Hole doping delta = 1 - N_e/V. Default 0.125. 0 < delta < 1.
  --tp FLOAT       Next-nearest-neighbour hopping t' (diagonal bonds).
                   Default -0.2. Must be nonzero to keep the Hard Core
                   (non-stoquastic, no special point); 0 is allowed but warned.
  --U FLOAT        On-site repulsion. Default 8.0.
  --r INT          Range of the local backflow (Chebyshev radius). Default 1.
                   Valid: r >= 1 and 2r+1 <= L for every L in --sizes.
  --F INT          Local feature width F. Default 16. F >= 1.
  --K INT          Number of global bottleneck channels K. Default 4. K >= 0
                   (K = 0 reduces to a purely local, SCALE-type backflow).
  --bf-scale FLOAT Scale of the random backflow weights Theta, Omega relative
                   to Phi0. Default 0.3.
  --nconf INT      Number of configurations per size used for timing and for
                   the exactness check. Default 3.
  --fit-last INT   Number of largest sizes used for the slope fits. Default 4.
                   Must be <= number of sizes and >= 2.
  --seed INT       RNG seed. Default 1234.
  --check-enum / --no-check-enum
                   Run (default) or skip the 3x2 exhaustive-enumeration
                   correctness check described above.
  --smoke          Shortcut for a few-second run: --sizes 4,6 --nconf 1
                   --fit-last 2 (verdict printed but not meaningful).
  --out PATH       JSON results file. Default results_P2.1.json next to this
                   script.

Recommended: run with OMP_NUM_THREADS=1 (the sbatch script sets it) so that
BLAS threading does not distort the exponents.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------
# Lattice and model
# ----------------------------------------------------------------------------
def build_hopping(Lx, Ly, t, tp):
    """Hopping matrix T (V x V), H_kin = sum_{ij,sigma} T_ij c^dag_i c_j."""
    V = Lx * Ly
    T = np.zeros((V, V))

    def idx(x, y):
        return (x % Lx) + Lx * (y % Ly)

    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            for dx, dy, a in ((1, 0, -t), (0, 1, -t), (1, 1, -tp), (1, -1, -tp)):
                j = idx(x + dx, y + dy)
                if j == i:
                    continue
                T[i, j] += a
                T[j, i] += a
    nbrs = [np.nonzero(T[:, j])[0] for j in range(V)]
    return T, nbrs


class LGBModel:
    def __init__(self, Lx, Ly, N_up, N_dn, r, F, K, T, bf_scale, rng):
        self.Lx, self.Ly, self.V = Lx, Ly, Lx * Ly
        V = self.V
        self.N_up, self.N_dn, self.Ne = N_up, N_dn, N_up + N_dn
        Ne = self.Ne
        self.r, self.F, self.K = r, F, K
        offs = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        self.noff = len(offs)
        xs = np.arange(V) % Lx
        ys = np.arange(V) // Lx
        # patch_idx[x, o] = site x + offset o
        self.patch_idx = np.array([[((xs[s] + dx) % Lx) + Lx * ((ys[s] + dy) % Ly)
                                    for (dx, dy) in offs] for s in range(V)])
        # sites whose patch contains site s: s - o
        self.rev_idx = np.array([[((xs[s] - dx) % Lx) + Lx * ((ys[s] - dy) % Ly)
                                  for (dx, dy) in offs] for s in range(V)])
        Pin = 2 * self.noff
        self.W1 = rng.normal(size=(F, Pin)) / np.sqrt(Pin)
        self.b1 = rng.normal(size=F) * 0.1
        self.Pool = rng.normal(size=(K, V, F)) / np.sqrt(V * F) * 2.0
        self.Wu = rng.normal(size=(K, F)) / np.sqrt(F)
        # Phi0: free-fermion orbitals (tiny random symmetric perturbation lifts
        # shell degeneracies), block structure up / down columns.
        P = rng.normal(size=(V, V)) * 1e-3
        e, vec = np.linalg.eigh(T + P + P.T)
        Phi0 = np.zeros((2 * V, Ne))
        Phi0[:V, :N_up] = vec[:, :N_up]
        Phi0[V:, N_up:] = vec[:, :N_dn]
        Phi0 *= np.sqrt(V)  # O(1) entries
        self.Phi0 = Phi0
        self.Theta = rng.normal(size=(2, F, Ne)) * bf_scale / np.sqrt(F)
        self.Omega = rng.normal(size=(2, K, Ne)) * bf_scale / np.sqrt(max(K, 1))
        # stacked basis B (D x Ne); block offsets
        self.oT = 2 * V
        self.oO = 2 * V + 2 * F
        self.B = np.concatenate([Phi0, self.Theta[0], self.Theta[1],
                                 self.Omega[0], self.Omega[1]], axis=0)

    # -- network pieces ----------------------------------------------------
    def features(self, n):
        """n: (2V,) occupations -> h (V,F), pre_c (K,), c (K,), u (V,K)."""
        V = self.V
        nu, nd = n[:V], n[V:]
        patch = np.concatenate([nu[self.patch_idx], nd[self.patch_idx]], axis=1)
        h = np.tanh(patch @ self.W1.T + self.b1)
        pre_c = np.einsum('kvf,vf->k', self.Pool, h)
        c = np.tanh(pre_c)
        u = h @ self.Wu.T
        return h, pre_c, c, u

    def local_h(self, n, sites):
        V = self.V
        pi = self.patch_idx[sites]
        patch = np.concatenate([n[:V][pi], n[V:][pi]], axis=1)
        return np.tanh(patch @ self.W1.T + self.b1)

    def matrix(self, n, occ):
        """Full A(n) with rows in the particle order given by occ (spin-orbitals)."""
        V = self.V
        h, pre_c, c, u = self.features(n)
        sig = occ // V
        x = occ % V
        A = self.Phi0[occ].copy()
        for s in (0, 1):
            m = sig == s
            A[m] += h[x[m]] @ self.Theta[s] + (u[x[m]] * c) @ self.Omega[s]
        return A


# ----------------------------------------------------------------------------
# Brute force and fast ratio engines
# ----------------------------------------------------------------------------
class BruteState:
    def __init__(self, model, occ):
        self.m = model
        self.occ = occ.copy()
        self.n = np.zeros(2 * model.V)
        self.n[occ] = 1.0
        self.sgn, self.logdet = np.linalg.slogdet(model.matrix(self.n, self.occ))

    def ratio(self, i, q):
        occ2 = self.occ.copy()
        n2 = self.n.copy()
        n2[occ2[i]] = 0.0
        n2[q] = 1.0
        occ2[i] = q
        s2, l2 = np.linalg.slogdet(self.m.matrix(n2, occ2))
        return s2 * self.sgn * np.exp(l2 - self.logdet), (occ2, n2, s2, l2)

    def accept(self, info):
        self.occ, self.n, self.sgn, self.logdet = info


class FastState:
    def __init__(self, model, occ):
        self.m = model
        self.occ = occ.copy()
        self.n = np.zeros(2 * model.V)
        self.n[occ] = 1.0
        self.recompute()

    def recompute(self):
        m = self.m
        self.h, self.pre_c, self.c, self.u = m.features(self.n)
        A = m.matrix(self.n, self.occ)
        self.G = m.B @ np.linalg.inv(A)  # (D, Ne)
        # site -> particle label map, per spin
        self.lab = -np.ones((2, m.V), dtype=int)
        self.lab[self.occ // m.V, self.occ % m.V] = np.arange(m.Ne)

    def ratio(self, i, q):
        m = self.m
        V, K, Ne = m.V, m.K, m.Ne
        p = self.occ[i]
        x, y = p % V, q % V
        sig = p // V
        n2 = self.n.copy()
        n2[p] = 0.0
        n2[q] = 1.0
        C = np.unique(np.concatenate([m.rev_idx[x], m.rev_idx[y]]))
        hC = m.local_h(n2, C)
        dpre = np.einsum('kcf,cf->k', m.Pool[:, C, :], hC - self.h[C]) if K else np.zeros(0)
        pre_c2 = self.pre_c + dpre
        c2 = np.tanh(pre_c2)
        dc = c2 - self.c
        uC = hC @ m.Wu.T
        # rows S: particles sitting (in n') at sites in C
        labC = self.lab[:, C]  # old labels at sites in C
        S = labC[labC >= 0]
        if i not in S:
            S = np.append(S, i)
        S = np.unique(S)
        occ2S = self.occ[S].copy()
        occ2S[S == i] = q
        sS = occ2S // V
        xS = occ2S % V
        pos = np.searchsorted(C, xS)  # all xS are in C
        hS, uS = hC[pos], uC[pos]
        # new rows times A^{-1}:  R = G[p'] + h' G_Theta[s] + (u' c') G_Omega[s]
        G = self.G
        R = G[occ2S].copy()
        for s in (0, 1):
            ms = sS == s
            if ms.any():
                R[ms] += hS[ms] @ G[m.oT + s * m.F: m.oT + (s + 1) * m.F]
                if K:
                    R[ms] += (uS[ms] * c2) @ G[m.oO + s * K: m.oO + (s + 1) * K]
        DA = R
        DA[np.arange(len(S)), S] -= 1.0
        if K:
            # global rank-2K part on rows not in S
            sig_all = self.occ // V
            Uall = self.u[self.occ % V] * dc  # (Ne, K)
            Ug = np.zeros((Ne, 2 * K))
            Ug[sig_all == 0, :K] = Uall[sig_all == 0]
            Ug[sig_all == 1, K:] = Uall[sig_all == 1]
            Ug[S] = 0.0
            WA = G[m.oO: m.oO + 2 * K]
            top = np.concatenate([DA[:, S], DA @ Ug], axis=1)
            bot = np.concatenate([WA[:, S], WA @ Ug], axis=1)
            X = np.eye(len(S) + 2 * K) + np.concatenate([top, bot], axis=0)
            VtAinv = np.concatenate([DA, WA], axis=0)
        else:
            Ug = None
            X = np.eye(len(S)) + DA[:, S]
            VtAinv = DA
        r = np.linalg.det(X)
        info = (i, q, n2, C, hC, uC, pre_c2, c2, S, Ug, X, VtAinv)
        return r, info

    def accept(self, info):
        m = self.m
        V = m.V
        i, q, n2, C, hC, uC, pre_c2, c2, S, Ug, X, VtAinv = info
        G = self.G
        GU = G[:, S] if Ug is None else np.concatenate([G[:, S], G @ Ug], axis=1)
        self.G = G - GU @ np.linalg.solve(X, VtAinv)
        p = self.occ[i]
        self.lab[p // V, p % V] = -1
        self.lab[q // V, q % V] = i
        self.occ[i] = q
        self.n = n2
        self.h[C] = hC
        self.u[C] = uC
        self.pre_c, self.c = pre_c2, c2


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------
def hops(state, T, nbrs):
    V = state.m.V
    out = []
    for i, p in enumerate(state.occ):
        s, x = p // V, p % V
        for y in nbrs[x]:
            q = s * V + y
            if state.n[q] == 0:
                out.append((i, q, T[y, x]))
    return out


def local_energy(state, T, nbrs, U):
    V = state.m.V
    e = U * float(np.dot(state.n[:V], state.n[V:]))
    rs = []
    for (i, q, a) in hops(state, T, nbrs):
        r, _ = state.ratio(i, q)
        rs.append(r)
        e += a * r
    return e, np.array(rs)


def sweep(state, nbrs, rng, nprop):
    V = state.m.V
    acc = 0
    for _ in range(nprop):
        i = rng.integers(state.m.Ne)
        p = state.occ[i]
        s, x = p // V, p % V
        y = nbrs[x][rng.integers(len(nbrs[x]))]
        q = s * V + y
        if state.n[q] == 1:
            continue
        r, info = state.ratio(i, q)
        if rng.random() < min(1.0, r * r):
            state.accept(info)
            acc += 1
    return acc


def random_occ(V, N_up, N_dn, rng):
    up = rng.choice(V, N_up, replace=False)
    dn = rng.choice(V, N_dn, replace=False) + V
    return np.sort(np.concatenate([up, dn]))


def enum_check(args, rng):
    """Exhaustive check on a 3x2 cluster: <H> from E_loc (brute and fast) vs psi^T H psi."""
    from itertools import combinations
    Lx, Ly, Nu, Nd = 3, 2, 2, 2
    T, nbrs = build_hopping(Lx, Ly, 1.0, args.tp)
    V = Lx * Ly
    model = LGBModel(Lx, Ly, Nu, Nd, 1, args.F, args.K, T, args.bf_scale, rng)
    confs = [np.array(sorted(list(a) + [V + b for b in bb]))
             for a in combinations(range(V), Nu) for bb in combinations(range(V), Nd)]
    index = {tuple(c): k for k, c in enumerate(confs)}
    psi = np.array([np.linalg.det(model.matrix(_occ_n(c, V), c)) for c in confs])
    D = len(confs)
    H = np.zeros((D, D))
    for k, c in enumerate(confs):
        n = _occ_n(c, V)
        H[k, k] += args.U * float(np.dot(n[:V], n[V:]))
        for p in c:
            s, x = p // V, p % V
            for y in nbrs[x]:
                q = s * V + y
                if n[q] == 1:
                    continue
                lo, hi = min(p, q), max(p, q)
                sign = (-1) ** int(n[lo + 1:hi].sum())
                c2 = tuple(sorted([o for o in c if o != p] + [q]))
                H[index[c2], k] += T[y, x] * sign
    E_exact = psi @ H @ psi / (psi @ psi)
    Eb = Ef = 0.0
    for k, c in enumerate(confs):
        w = psi[k] ** 2
        Eb += w * local_energy(BruteState(model, c), T, nbrs, args.U)[0]
        Ef += w * local_energy(FastState(model, c), T, nbrs, args.U)[0]
    Z = psi @ psi
    return dict(E_exact=E_exact, E_brute=Eb / Z, E_fast=Ef / Z,
                err_brute=abs(Eb / Z - E_exact), err_fast=abs(Ef / Z - E_exact),
                hermitian=bool(np.allclose(H, H.T)))


def _occ_n(c, V):
    n = np.zeros(2 * V)
    n[np.asarray(c)] = 1.0
    return n


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sizes', default='8,12,16,20,24,28,32')
    ap.add_argument('--delta', type=float, default=0.125)
    ap.add_argument('--tp', type=float, default=-0.2)
    ap.add_argument('--U', type=float, default=8.0)
    ap.add_argument('--r', type=int, default=1)
    ap.add_argument('--F', type=int, default=16)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--bf-scale', dest='bf_scale', type=float, default=0.3)
    ap.add_argument('--nconf', type=int, default=3)
    ap.add_argument('--fit-last', dest='fit_last', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--check-enum', dest='check_enum', action='store_true', default=True)
    ap.add_argument('--no-check-enum', dest='check_enum', action='store_false')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'results_P2.1.json'))
    a = ap.parse_args(argv)
    if a.smoke:
        a.sizes, a.nconf, a.fit_last = '4,6', 1, 2
    a.sizes = [int(s) for s in a.sizes.split(',')]
    if any(L < 4 for L in a.sizes):
        ap.error('every L must be >= 4')
    if any(2 * a.r + 1 > L for L in a.sizes) or a.r < 1:
        ap.error('need r >= 1 and 2r+1 <= L')
    if not (0 < a.delta < 1):
        ap.error('0 < delta < 1 required')
    if a.F < 1 or a.K < 0 or a.nconf < 1:
        ap.error('F >= 1, K >= 0, nconf >= 1 required')
    if not (2 <= a.fit_last <= len(a.sizes)):
        ap.error('2 <= fit-last <= len(sizes) required')
    if a.tp == 0:
        print('WARNING: tp = 0 removes the next-nearest hopping (Hard Core needs tp != 0)')
    return a


def main(argv):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print(f'P2.1 micro-test  args={vars(args)}  OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")}')
    res = dict(args=vars(args), sizes=[])
    if args.check_enum:
        ec = enum_check(args, rng)
        res['enum_check'] = ec
        print(f"[enum 3x2, Nup=Ndn=2] E_exact={ec['E_exact']:.12f} E_brute={ec['E_brute']:.12f} "
              f"E_fast={ec['E_fast']:.12f} |err_brute|={ec['err_brute']:.2e} "
              f"|err_fast|={ec['err_fast']:.2e} H_hermitian={ec['hermitian']}")
    hdr = ('   L   V  Ne  T_E_fast  T_E_brute  T_sw_fast  T_sw_brute  eps_ratio  eps_eloc  drift_G  acc')
    print(hdr)
    for L in args.sizes:
        V = L * L
        Ne = 2 * int(round(V * (1 - args.delta) / 2))
        Nu = Nd = Ne // 2
        T, nbrs = build_hopping(L, L, 1.0, args.tp)
        model = LGBModel(L, L, Nu, Nd, args.r, args.F, args.K, T, args.bf_scale, rng)
        tEf, tEb, tSf, tSb, er, ee, dr, acc = [], [], [], [], 0.0, 0.0, 0.0, 0
        for _ in range(args.nconf):
            occ = random_occ(V, Nu, Nd, rng)
            fs, bs = FastState(model, occ), BruteState(model, occ)
            t0 = time.perf_counter(); Ef, rf = local_energy(fs, T, nbrs, args.U); tEf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); Eb, rb = local_energy(bs, T, nbrs, args.U); tEb.append(time.perf_counter() - t0)
            er = max(er, float(np.max(np.abs(rf - rb) / (1 + np.abs(rb)))))
            ee = max(ee, abs(Ef - Eb) / (1 + abs(Eb)))
            # timed sweeps with identical proposal streams
            seed = int(rng.integers(1 << 30))
            t0 = time.perf_counter(); acc += sweep(fs, nbrs, np.random.default_rng(seed), Ne); tSf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); sweep(bs, nbrs, np.random.default_rng(seed), Ne); tSb.append(time.perf_counter() - t0)
            Gc = fs.G.copy()
            fs.recompute()
            dr = max(dr, float(np.max(np.abs(Gc - fs.G)) / np.max(np.abs(fs.G))))
            # after the sweep both chains must sit on the same configuration
            same = np.array_equal(np.sort(fs.occ), np.sort(bs.occ))
            if not same:
                print('  note: fast and brute chains diverged (round-off at an accept boundary)')
        row = dict(L=L, V=V, Ne=Ne, T_E_fast=float(np.median(tEf)), T_E_brute=float(np.median(tEb)),
                   T_sw_fast=float(np.median(tSf)), T_sw_brute=float(np.median(tSb)),
                   eps_ratio=er, eps_eloc=ee, drift_G=dr, accepted=acc)
        res['sizes'].append(row)
        print(f"{L:4d}{V:4d}{Ne:4d}  {row['T_E_fast']:.3e}  {row['T_E_brute']:.3e}  {row['T_sw_fast']:.3e}  "
              f"{row['T_sw_brute']:.3e}  {er:.2e}  {ee:.2e}  {dr:.1e}  {acc}")
        sys.stdout.flush()
    rows = res['sizes'][-args.fit_last:]
    Nes = [r['Ne'] for r in rows]
    sl = {k: slope(Nes, [r[k] for r in rows]) for k in ('T_E_fast', 'T_E_brute', 'T_sw_fast', 'T_sw_brute')}
    eps_max = max(max(r['eps_ratio'], r['eps_eloc']) for r in res['sizes'])
    if args.check_enum:
        eps_max = max(eps_max, res['enum_check']['err_fast'] / (1 + abs(res['enum_check']['E_exact'])))
    M = sl['T_E_fast']
    if M <= 2.3 and eps_max <= 1e-9 and sl['T_E_brute'] >= 3.3 and sl['T_sw_fast'] <= 3.3:
        verdict = 'PASS'
    elif M >= 2.8 or eps_max > 1e-6:
        verdict = 'FAIL'
    else:
        verdict = 'AMBIGUOUS'
    speed = rows[-1]['T_E_brute'] / rows[-1]['T_E_fast']
    res.update(slopes=sl, M=M, eps_max=eps_max, speedup_E_largest=speed, verdict=verdict)
    print(f"slopes over Ne={Nes}: " + ', '.join(f'{k}={v:.2f}' for k, v in sl.items()))
    print(f"M = {M:.3f}   eps_max = {eps_max:.2e}   speedup(T_E) at largest size = {speed:.1f}x")
    print(f"VERDICT: {verdict}")
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'results written to {args.out}')


if __name__ == '__main__':
    main(sys.argv[1:])
