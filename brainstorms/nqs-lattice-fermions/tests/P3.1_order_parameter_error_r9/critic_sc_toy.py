#!/usr/bin/env python3
"""
P3.1 Round 9 per-leaf critic (2026-11-16): size-consistency check of linear-span Ritz corrections.

M decoupled copies of one random d=4 subsystem: H = sum_i h_i, intensive A = (1/M) sum_i a_i,
psi0 = prod p0, trial psi = prod unit(p0 + eps e) (the same per-copy error, so the per-site quality is
fixed and the total infidelity 1 - F grows with M, as for a trained NQS at fixed width).
Arms (ratio = (a(Ritz) - a(psi)) / delta, as in t1_r10.py):
  RWR*K  span{psi, r, y(psi0), K y(psi0)}      oracle response of psi0 (route class ceiling)
  RWRK0  span{psi, r, y(psi), K y(psi)}        exact response of psi (9.1 noise-free)
  SCR    two self-consistent passes, base {psi, r} kept (10.1 noise-free, as coded in t1_r10.py)
  PROD   psi' = prod unit(p0_i + c * z_i) with z_i the per-copy first-order correction and one global c
         chosen by energy minimisation (a size-consistent, multiplicative correction; reference arm)
  MULT   psi'(n) = psi(n) exp(c y(n)/psi(n)), y = exact response of psi (as RWRK0), one global c by energy
         minimisation (a realisable multiplicative form of the response; checked and found ill-defined)
  SCRu-T SCR with the base updated, span{phi_t, r(phi_t), y(phi_t), K y(phi_t)}, T passes (T = 2, 4, 8)
  SCRd-T depth-free update: pass 1 as SCR, later passes span{phi_t, y(phi_t)} only (no further H
         applications, so a sampled version needs no H^t; T = 2, 4, 8)
Usage: python3 critic_sc_toy.py [eps] [d] [Mmax]   (seconds, login-node safe; d^M <= 4096)
"""
import functools
import sys

import numpy as np

rng = np.random.default_rng(3)
d = int(sys.argv[2]) if len(sys.argv) > 2 else 4
h = rng.standard_normal((d, d)); h = (h + h.T) / 2
a = rng.standard_normal((d, d)); a = (a + a.T) / 2
w, U = np.linalg.eigh(h)
p0 = U[:, 0]
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
e = U[:, 1:] @ rng.standard_normal(d - 1); e /= np.linalg.norm(e)
phi = p0 + eps * e; phi /= np.linalg.norm(phi)


def kron_all(vs):
    return functools.reduce(np.kron, vs)


def op_sum(o, M):
    eye = np.eye(d)
    return sum(kron_all([o if j == i else eye for j in range(M)]) for i in range(M))


def unit(v):
    return v / np.linalg.norm(v)


def ritz(H, vecs):
    B = np.array([unit(v) for v in vecs]); S = B @ B.T; Hm = B @ H @ B.T
    lam, V = np.linalg.eigh(S); k = lam > 1e-11 * lam.max(); X = V[:, k] / np.sqrt(lam[k])
    ev, Y = np.linalg.eigh(X.T @ Hm @ X)
    return unit((X @ Y[:, 0]) @ B)


def resp(H, A, ref, shift):
    D = len(ref); Q = np.eye(D) - np.outer(ref, ref); av = ref @ A @ ref
    rhs = Q @ (A @ ref - av * ref)
    return Q @ np.linalg.solve(Q @ (H - shift * np.eye(D)) @ Q + np.outer(ref, ref), rhs)


def K(H, v):
    v = unit(v); x = H @ v
    return x - (v @ x) * v


# per-copy first-order correction of phi toward p0 (exact single-copy direction), for the PROD arm
zc = p0 - (phi @ p0) * phi
for M in range(1, int(sys.argv[3]) + 1 if len(sys.argv) > 3 else 7):
    H = op_sum(h, M); A = op_sum(a, M) / M
    psi0 = kron_all([p0] * M); psi = kron_all([phi] * M); E0 = psi0 @ H @ psi0
    ap = psi @ A @ psi; dl = psi0 @ A @ psi0 - ap
    E = psi @ H @ psi; r = H @ psi - E * psi

    def f(v):
        return (v @ A @ v - ap) / dl
    ys = resp(H, A, psi0, E0); yp = resp(H, A, psi, E - 0.5)
    rstar = f(ritz(H, [psi, r, ys, K(H, ys)]))
    ph = ritz(H, [psi, r, yp, K(H, yp)]); rk0 = f(ph)
    for _ in range(2):
        y1 = resp(H, A, ph, ph @ H @ ph - 0.5)
        ph = ritz(H, [psi, r, y1, K(H, y1)])
    scr = f(ph)
    cs = np.linspace(0, 2, 201)
    es = [(lambda v: v @ H @ v)(kron_all([unit(phi + c * zc)] * M)) for c in cs]
    prod = f(kron_all([unit(phi + cs[int(np.argmin(es))] * zc)] * M))
    ratio = yp / psi
    ratio /= np.sqrt(np.sum(psi ** 2 * ratio ** 2))       # unit local-ratio scale under |psi|^2
    cm = np.linspace(-1.5, 1.5, 601)
    em = [(lambda v: v @ H @ v)(unit(psi * np.exp(c * ratio))) for c in cm]
    mult = f(unit(psi * np.exp(cm[int(np.argmin(em))] * ratio)))
    sct = {}
    ph = psi
    for t in range(1, 9):
        y1 = resp(H, A, ph, ph @ H @ ph - 0.5)
        ph = ritz(H, [ph, H @ ph - (ph @ H @ ph) * ph, y1, K(H, y1)])
        if t in (2, 4, 8):
            sct[t] = f(ph)
    scd = {}
    ph = ritz(H, [psi, r, yp, K(H, yp)])
    for t in range(2, 9):
        y1 = resp(H, A, ph, ph @ H @ ph - 0.5)
        ph = ritz(H, [ph, y1])
        if t in (2, 4, 8):
            scd[t] = f(ph)
    print(f"eps={eps} M={M} D={d ** M:5d} 1-F={1 - (psi @ psi0) ** 2:.3f} | RWR*K {rstar:.3f}  "
          f"RWRK0 {rk0:.3f}  SCR {scr:.3f}  PROD {prod:.3f}  MULT {mult:.3f}  SCRu-2/4/8 "
          + " ".join(f"{sct[t]:.3f}" for t in (2, 4, 8))
          + "  SCRd-2/4/8 " + " ".join(f"{scd[t]:.3f}" for t in (2, 4, 8)))
