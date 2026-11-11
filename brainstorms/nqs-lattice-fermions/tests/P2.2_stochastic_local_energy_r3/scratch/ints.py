"""McMurchie-Davidson integrals for Cartesian s and p Gaussians (numpy only)."""
import numpy as np
from scipy.special import gammainc, gamma

# 6-31G** hydrogen: contracted s (3 prim), uncontracted s, uncontracted p.
H631GSS = [(0, [18.7311370, 2.8253937, 0.6401217], [0.03349460, 0.23472695, 0.81375733]),
           (0, [0.1612778], [1.0]),
           (1, [1.1], [1.0])]
STO3G_H = [(0, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454])]
CART = {0: [(0, 0, 0)], 1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]}


def boys(n, T):
    T = np.asarray(T, float)
    out = np.empty_like(T)
    small = T < 1e-7
    out[small] = 1.0 / (2 * n + 1) - T[small] / (2 * n + 3)
    Ts = T[~small]
    out[~small] = 0.5 * Ts ** (-(n + 0.5)) * gamma(n + 0.5) * gammainc(n + 0.5, Ts)
    return out


def hermite_E(i, j, t, Qx, a, b):
    """E^{ij}_t (Helgaker), vectorised over arrays Qx = A_x - B_x, a, b."""
    p = a + b; q = a * b / p
    if t < 0 or t > i + j:
        return np.zeros_like(Qx)
    if i == j == t == 0:
        return np.exp(-q * Qx * Qx)
    if j == 0:
        return (1 / (2 * p)) * hermite_E(i - 1, j, t - 1, Qx, a, b) - (q * Qx / a) * hermite_E(i - 1, j, t, Qx, a, b) \
            + (t + 1) * hermite_E(i - 1, j, t + 1, Qx, a, b)
    return (1 / (2 * p)) * hermite_E(i, j - 1, t - 1, Qx, a, b) + (q * Qx / b) * hermite_E(i, j - 1, t, Qx, a, b) \
        + (t + 1) * hermite_E(i, j - 1, t + 1, Qx, a, b)


def hermite_R(L, p, PC):
    """R_{tuv}(p, PC) for t+u+v <= L; p: (n,), PC: (n,3). Returns dict."""
    RPC2 = (PC ** 2).sum(-1)
    T = p * RPC2
    Fn = [boys(n, T) for n in range(L + 1)]
    memo = {}

    def R(t, u, v, n):
        key = (t, u, v, n)
        if key in memo:
            return memo[key]
        if t < 0 or u < 0 or v < 0:
            val = 0.0
        elif t == u == v == 0:
            val = (-2 * p) ** n * Fn[n]
        elif t > 0:
            val = (t - 1) * R(t - 2, u, v, n + 1) + PC[:, 0] * R(t - 1, u, v, n + 1)
        elif u > 0:
            val = (u - 1) * R(t, u - 2, v, n + 1) + PC[:, 1] * R(t, u - 1, v, n + 1)
        else:
            val = (v - 1) * R(t, u, v - 2, n + 1) + PC[:, 2] * R(t, u, v - 1, n + 1)
        memo[key] = val
        return val
    out = {}
    for t in range(L + 1):
        for u in range(L + 1 - t):
            for v in range(L + 1 - t - u):
                out[(t, u, v)] = np.broadcast_to(R(t, u, v, 0), p.shape).astype(float)
    return out


def build_basis(X, basis):
    """Primitive list and contraction matrix T (prims x AOs)."""
    prims, cols, aolab = [], [], []
    k = 0
    for ia, A in enumerate(X):
        for l, exps, coefs in basis:
            for lmn in CART[l]:
                for e, c in zip(exps, coefs):
                    nrm = (2 * e / np.pi) ** 0.75 * (4 * e) ** (l / 2)
                    prims.append((e, A, lmn)); cols.append((k, c * nrm))
                k += 1
                aolab.append((ia, l))
    T = np.zeros((len(prims), k))
    for ip, (kk, c) in enumerate(cols):
        T[ip, kk] = c
    return prims, T, aolab


def _pairs(prims):
    n = len(prims)
    a = np.array([p[0] for p in prims]); A = np.array([p[1] for p in prims]); L = np.array([p[2] for p in prims])
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); I = I.ravel(); J = J.ravel()
    ai, aj = a[I], a[J]; p = ai + aj
    P = (ai[:, None] * A[I] + aj[:, None] * A[J]) / p[:, None]
    Q = A[I] - A[J]
    # Hermite expansion coefficients E_{tuv} for t+u+v <= 2 (l <= 1)
    Ex = [[hermite_E(li, lj, t, Q[:, 0], ai, aj) for t in range(3)] for li, lj in [(0, 0)]]  # placeholder
    htuv = [(t, u, v) for t in range(3) for u in range(3) for v in range(3) if t + u + v <= 2]
    Eh = np.zeros((len(I), len(htuv)))
    li, lj = L[I], L[J]
    for combo in set(map(tuple, np.concatenate([li, lj], 1))):
        m = np.all(np.concatenate([li, lj], 1) == combo, 1)
        idx = np.nonzero(m)[0]
        (ix, iy, iz, jx, jy, jz) = combo
        for h, (t, u, v) in enumerate(htuv):
            Eh[idx, h] = hermite_E(ix, jx, t, Q[idx, 0], ai[idx], aj[idx]) * \
                hermite_E(iy, jy, u, Q[idx, 1], ai[idx], aj[idx]) * hermite_E(iz, jz, v, Q[idx, 2], ai[idx], aj[idx])
    return dict(I=I, J=J, a=ai, b=aj, p=p, P=P, Q=Q, Eh=Eh, htuv=htuv, li=li, lj=lj, n=n)


def one_electron(prims, X, Z):
    pr = _pairs(prims)
    n = pr["n"]; p = pr["p"]; Q = pr["Q"]; ai, aj = pr["a"], pr["b"]
    li, lj = pr["li"], pr["lj"]

    def s1d(i, j, d):
        return hermite_E(i, j, 0, Q[:, d], ai, aj) * np.sqrt(np.pi / p)
    S = np.ones(len(p)); Tk = np.zeros(len(p))
    s1 = [np.array([s1d_i for s1d_i in [None]])] if False else None
    S1 = []
    for d in range(3):
        S1.append(np.array([0.0]))
    Sd = np.zeros((3, len(p))); Td = np.zeros((3, len(p)))
    for d in range(3):
        for i in (0, 1):
            for j in (0, 1):
                m = (li[:, d] == i) & (lj[:, d] == j)
                if not m.any():
                    continue
                ii = np.nonzero(m)[0]
                pa, qa, A_, B_ = p[ii], Q[ii, d], ai[ii], aj[ii]
                def s(i_, j_):
                    if j_ < 0:
                        return 0.0
                    return hermite_E(i_, j_, 0, qa, A_, B_) * np.sqrt(np.pi / pa)
                Sd[d, ii] = s(i, j)
                Td[d, ii] = -0.5 * (j * (j - 1) * s(i, j - 2) - 2 * B_ * (2 * j + 1) * s(i, j) + 4 * B_ ** 2 * s(i, j + 2))
    S = Sd[0] * Sd[1] * Sd[2]
    Tk = Td[0] * Sd[1] * Sd[2] + Sd[0] * Td[1] * Sd[2] + Sd[0] * Sd[1] * Td[2]
    V = np.zeros(len(p))
    for C, z in zip(X, Z):
        R = hermite_R(2, p, pr["P"] - C)
        acc = np.zeros(len(p))
        for h, tuv in enumerate(pr["htuv"]):
            acc += pr["Eh"][:, h] * R[tuv]
        V += -z * 2 * np.pi / p * acc
    return S.reshape(n, n), Tk.reshape(n, n), V.reshape(n, n), pr


def eri_prims(pr, chunk=64):
    n = pr["n"]; npair = n * n
    p, P, Eh, htuv = pr["p"], pr["P"], pr["Eh"], pr["htuv"]
    sign = np.array([(-1) ** (t + u + v) for (t, u, v) in htuv])
    Ek = Eh * sign[None]
    nh = len(htuv)
    comb = {}
    G = np.zeros((npair, npair))
    for s0 in range(0, npair, chunk):
        s1 = min(npair, s0 + chunk)
        pb = p[s0:s1]
        alpha = pb[:, None] * p[None, :] / (pb[:, None] + p[None, :])
        PQ = P[s0:s1, None, :] - P[None, :, :]
        R = hermite_R(4, alpha.ravel(), PQ.reshape(-1, 3))
        acc = np.zeros((s1 - s0, npair))
        for h1, (t, u, v) in enumerate(htuv):
            inner = np.zeros((s1 - s0) * npair)
            for h2, (t2, u2, v2) in enumerate(htuv):
                inner += np.tile(Ek[:, h2], s1 - s0) * R[(t + t2, u + u2, v + v2)]
            acc += Eh[s0:s1, h1][:, None] * inner.reshape(s1 - s0, npair)
        pref = 2 * np.pi ** 2.5 / (pb[:, None] * p[None, :] * np.sqrt(pb[:, None] + p[None, :]))
        G[s0:s1] = pref * acc
    return G.reshape(n, n, n, n)


def integrals(X, basis, Z=None):
    if Z is None:
        Z = np.ones(len(X))
    prims, T, aolab = build_basis(X, basis)
    Sp, Tp, Vp, pr = one_electron(prims, X, Z)
    S = T.T @ Sp @ T
    nrm = 1 / np.sqrt(np.diag(S)); T = T * nrm[None]
    S = T.T @ Sp @ T
    h = T.T @ (Tp + Vp) @ T
    Gp = eri_prims(pr)
    G = np.einsum("ijkl,ia->ajkl", Gp, T, optimize=True)
    G = np.einsum("ajkl,jb->abkl", G, T, optimize=True)
    G = np.einsum("abkl,kc->abcl", G, T, optimize=True)
    G = np.einsum("abcl,ld->abcd", G, T, optimize=True)
    enuc = sum(Z[i] * Z[j] / np.linalg.norm(X[i] - X[j]) for i in range(len(X)) for j in range(i))
    return S, h, G, enuc, aolab
