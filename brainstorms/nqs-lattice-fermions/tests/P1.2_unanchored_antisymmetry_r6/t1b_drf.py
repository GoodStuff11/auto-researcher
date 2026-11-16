#!/usr/bin/env python3
"""
P1.2 round 7, tier T1b for the DRF family (routes 7.2 DRF-K, 7.3 DRF-X, 7.4 DRF-H): does a TRAINED
two-pass determinant-ratio-feature backflow beat the same parameter count spent on single-pass backflow
WIDTH (the foil that killed AO-F), on three toy sizes?
=====================================================================================================

Ansatz (class DRF). For a configuration n with occupations x = 2n - 1 (2V entries):
  pass 1 (NNBF, width H1):  h1 = tanh(x W1 + b1),  O1_s = P0_s + reshape_s(h1 W2 + b2)   (V x N_s per spin)
  Green's function:         G_s = O1_s A_s^{-1},  A_s = O1_s[rows_s(n)]        (frozen-frame hop ratios)
  edge ratio features:      R_e = (hc_e / scale_e) * prod_{s moved by e} G_s[j_s, k_s]
                            (hop i -> j of spin s: k_s = position of i in rows_s(n); equals
                            hc_e Phi1(m_e) / (scale_e Phi1(n)) for the FROZEN pass-1 orbitals; exchange
                            edges multiply the up and down entries), g_e = R_e / sqrt(1 + R_e^2)
  per-site gather:          f[c, v] = sum of g_e over active edges e of channel c incident on site v;
                            channels: K frame -> (up hops, down hops); X -> (exchange); H -> all three
  pass 2 (width H2):        h2 = tanh([x, f] W3 + b3),  O2_s = O1_s + reshape_s(h2 W4)   (residual)
  amplitude:                psi(n) = det O2_up[rows_up] * det O2_dn[rows_dn]
  feat = "O" (attribution control, one change): G_s is replaced by O1_s (no inverse); everything else equal.
With W4 = 0 the ansatz equals the pass-1 NNBF exactly. The force sum_n r(n) dpsi(n)/dtheta and the
per-sample log-derivatives are hand-written backprop (checked against finite differences, CHECK 2).

Gate 4 (T0) cost: one amplitude = 2 NNBF passes + one N_s x N_s inverse + G = O1 A^{-1} (O(V N^2)) +
O(E) lookups; a local energy needs the same (1 + n_conn) amplitudes as NNBF, so the overhead is a constant
factor (no O(V) extra passes, unlike FRD-K). --mode cost measures it on L x L OBC clusters.

Toys (t = 1, t' = -0.25, J = 0.4, open 2-leg ladders, 2 holes; t1_vote.SIZES), unprojected, UHF start,
exact-sum SR of the round-5 harness (p21_harness.optimize, proj = "none", lr 0.05, shift 1e-3,
600 metric samples, 300 iterations), seeds 11 and 22:
  size 0 : 4x2 3+3 D = 560 (V = 8);  size 1 : 5x2 4+4 D = 3150 (V = 10);  size 2 : 6x2 5+5 D = 16632 (V = 12)
Arms:
  nnbfW      LRAnsatz full width WC = 6 (the matched-WIDTH foil)    -> parameter target T
  drfK/X/H   DRF, H1 = 3, frame K / X / H, H2 = largest with n_params <= T
  drfO<F>    attribution control for frame F (feat "O"), same H1, H2 as drf<F>; run only for a frame
             that passes the kill rule (stage 2), so one change (the inverse) is isolated.
Metric: dEs(X) = min_{a>=0} <a|S H S|a>/<a|a> - E_0, S = diag(sgn psi_X) (t1_vote.dE_sign); median over
the two seeds. r_W(F) = dEs(drfF) / dEs(nnbfW); r_O(F) = dEs(drfF) / dEs(drfOF).

PRE-REGISTERED RULES (written 2026-11-16 before the first training run; the critic's T1b rule):
  premise per size: counted iff med dEs(nnbfW) >= 0.005 (else NO-TARGET at that size).
  KILL drf<F>: r_W >= 0.9 at >= 2 counted sizes including the largest counted size.
  PASS-W drf<F>: r_W <= 0.7 at every counted size AND r_W(largest) <= r_W(smallest) + 0.1.
  UNDECIDED otherwise; NO-TARGET if fewer than 2 sizes are counted.
  Attribution (stage 2, only after PASS-W or UNDECIDED): credited to the inverse (PASS) iff
  r_O <= 0.9 at the largest size; else "DEPTH-ONLY" (the gain is the second pass, not the ratios).
  EXTENSION (written 2026-11-16 after the 2-seed run, BEFORE seed 33 was run): DRF-K came out UNDECIDED
  with 2 counted sizes (r_W 0.76 at V = 10, 1.62 at V = 12). Seed 33 is added to nnbfW and drfK at all three
  sizes; the 3-seed medians replace the 2-seed ones and the same rule is applied once more. Whatever it
  returns is final at T1 (no further seeds); a second UNDECIDED blocks T2 promotion (PASS-W is required).
  Validity (else INVALID): CHECK 1 R_e at zero backflow equals hc Phi(m_e)/(scale Phi(n)) from the
  explicit determinants (rel < 1e-10); CHECK 2 analytic force vs central finite differences (rel < 1e-5)
  for feat G and O on frame H; CHECK 3 W4 = 0 reproduces the pass-1 NNBF amplitudes (< 1e-12);
  CHECK 4 per_sample_O * psi equals the force rows (rel < 1e-10).
Budget (enforced): every invocation sets RLIMIT_CPU to --cpu_cap seconds (default 600) and, before the
heavy part, times 3 SR steps per arm and aborts with BUDGET-ABORT if the projected CPU time exceeds 90%
of the cap. Thread count is capped at 4 (--threads, default 1).

Usage
-----
  python3 t1b_drf.py --mode checks
  python3 t1b_drf.py --mode train --sizes 0 --arms nnbfW,drfK,drfX,drfH
  python3 t1b_drf.py --mode verdict
  python3 t1b_drf.py --mode cost
Options
-------
  --mode {checks,train,verdict,cost}
        checks : CHECKs 1-4 on size 0 (and CHECK 1 on size 1); prints PASS/FAIL per check.
        train  : trains every (size, arm, seed) listed; writes results/t1b/<size>_<arm>_<seed>.json;
                 skips any combination whose JSON already exists (no recompute).
        verdict: reads results/t1b/*.json and prints the rule verdict per frame.
        cost   : per-amplitude wall time of NNBF (width 6) vs DRF-H (H1 = 3, H2 = 3) on L x L OBC,
                 L = 4, 5, 6, 7 at about 1/8 hole doping, batch 256, best of 20 interleaved repeats
                 (Gate 4 constant-factor check).
  --sizes LIST    comma-separated subset of {0,1,2} (default 0,1,2).
  --arms LIST     comma-separated subset of {nnbfW, drfK, drfX, drfH, drfOK, drfOX, drfOH} (default
                  nnbfW,drfK,drfX,drfH).
  --seeds LIST    comma-separated seeds (default 11,22).
  --iters INT     SR iterations (default 300).
  --threads INT   BLAS/OpenMP threads, 1..4 (default 1).
  --cpu_cap FLOAT CPU-second cap for this invocation (default 600 = 10 CPU-minutes).
"""
import os
import sys


def _pre_threads():
    n = 1
    for a in sys.argv:
        if a.startswith("--threads="):
            n = int(a.split("=")[1])
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    n = max(1, min(4, n))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)
    return n


NTHREADS = _pre_threads()

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r5")
sys.path.insert(0, R5)
sys.path.insert(0, HERE)
from p11_core import solve_ed, uhf_orbitals, energy  # noqa: E402
from tj_core import TJ, LRAnsatz, popcount, below_mask, between_mask  # noqa: E402
import p21_harness as H21  # noqa: E402
import micro_test as R5T  # noqa: E402
from t1_vote import dE_sign, SIZES  # noqa: E402

WC, H1 = 6, 3
OUT = os.path.join(HERE, "results", "t1b")
FRAMES = {"K": ("K",), "X": ("X",), "H": ("K", "X")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ edge descriptors
def build_edges(m, frame):
    """Edge list, per-edge scale, and the constant (E, C*V) gather matrix of frame F."""
    edges, scale, chan = [], [], []
    nch = {"K": 2, "X": 1, "H": 3}[frame]
    for sub in FRAMES[frame]:
        el = R5T.edge_list(m.T, m.nn, sub)
        edges += el
        scale += [1.0 if sub == "K" else 0.5 * m.J] * len(el)
        for ed in el:
            if ed[0] == "K":
                chan.append((ed[3], ed[1], ed[2]))
            else:
                chan.append((nch - 1, ed[1], ed[2]))
    Gm = np.zeros((len(edges), nch * m.V))
    for e, (c, a, b) in enumerate(chan):
        Gm[e, c * m.V + a] += 1.0
        Gm[e, c * m.V + b] += 1.0
    return edges, np.array(scale), Gm, nch


def edge_desc(bu, bd, edges, scale, T, J):
    """Per (config, edge): act, which spins move (pu, pd), target row j and source column k per spin,
    and pref = (hc / scale) * perm_u * perm_d, where perm_s = (-1)^(# spin-s electrons strictly between
    i_s and j_s) is the row-sorting sign: Phi(m_e) / Phi(n) = perm_s * G_s[j_s, k_s] per moved spin."""
    nbu, nbd, act, hc = R5T.edge_moves_bits(bu, bd, edges, T, J)
    pref = np.where(act, hc / scale[None], 0.0)
    out = dict(act=act)
    for s, b, nb in (("u", bu, nbu), ("d", bd, nbd)):
        rem = b[:, None] & ~nb
        add = nb & ~b[:, None]
        p = act & (rem != 0)
        i = np.where(p, np.log2(np.maximum(rem, 1).astype(float)).round().astype(np.int64), 0)
        j = np.where(p, np.log2(np.maximum(add, 1).astype(float)).round().astype(np.int64), 0)
        bb = np.broadcast_to(b[:, None], i.shape)
        k = np.where(p, popcount(bb & below_mask(i)), 0)
        perm = 1.0 - 2.0 * (popcount(bb & between_mask(i, j)) & 1)
        pref = np.where(p, pref * perm, pref)
        out["p" + s], out["j" + s], out["k" + s] = p, j, k
    out["pref"] = pref
    return out


# ============================================================================ DRF ansatz
class DRF:
    def __init__(self, m, ref, h1, h2, frame, seed, feat="G", w2_scale=1e-3):
        assert feat in ("G", "O")
        rng = np.random.default_rng(seed)
        self.m, self.feat, self.frame = m, feat, frame
        V, nu, nd = m.V, m.nup, m.ndn
        self.V, self.nu, self.nd = V, nu, nd
        self.edges, self.scale, self.Gm, self.nch = build_edges(m, frame)
        self.E = len(self.edges)
        Pu, Pd = ref
        no = V * (nu + nd)
        nin2 = 2 * V + self.nch * V
        p = {"P0u": Pu.copy(), "P0d": Pd.copy(),
             "W1": rng.standard_normal((2 * V, h1)) / np.sqrt(2 * V), "b1": np.zeros(h1),
             "W2": w2_scale * rng.standard_normal((h1, no)), "b2": np.zeros(no),
             "W3": rng.standard_normal((nin2, h2)) / np.sqrt(nin2), "b3": np.zeros(h2),
             "W4": w2_scale * rng.standard_normal((h2, no))}
        self.p, self.keys = p, list(p)
        self._desc = None

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def flat(self):
        return np.concatenate([self.p[k].ravel() for k in self.keys])

    def set_flat(self, x):
        o = 0
        for k in self.keys:
            n = self.p[k].size
            self.p[k] = x[o:o + n].reshape(self.p[k].shape)
            o += n

    def desc(self):
        if self._desc is None:
            m = self.m
            self._desc = edge_desc(m.bu, m.bd, self.edges, self.scale, m.T, m.J)
        return self._desc

    def _occ(self, idx):
        m = self.m
        iu, idd = m.iu_id()
        return m.occ_u[iu[idx]], m.occ_d[idd[idx]], m.rows_u[iu[idx]], m.rows_d[idd[idx]]

    @staticmethod
    def _inv(A):
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    def forward(self, ou, od, ru, rd, ds):
        p, V, nu, nd = self.p, self.V, self.nu, self.nd
        B = ou.shape[0]
        bi = np.arange(B)
        x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        o1 = h1 @ p["W2"] + p["b2"]
        c = dict(x=x, h1=h1, B=B, ds=ds)
        gsel = {}
        for s, rows, N, P0, o0 in (("u", ru, nu, p["P0u"], 0), ("d", rd, nd, p["P0d"], V * nu)):
            O1 = P0[None] + o1[:, o0:o0 + V * N].reshape(B, V, N)
            c["O1" + s] = O1
            if self.feat == "G":
                # only the needed Green's-function entries: G[j_e, k_e] = O1[j_e, :] . A^{-1}[:, k_e]
                # (O(E N) per amplitude; the inverse is the same O(N^3) as the determinant)
                A = O1[bi[:, None], rows]
                Ai = self._inv(A)
                c["Ai" + s] = Ai
                rows_e = O1[bi[:, None], ds["j" + s]]                                # (B, E, N)
                cols_e = np.take_along_axis(Ai, ds["k" + s][:, None, :], axis=2)     # (B, N, E)
                ge = np.einsum("ben,bne->be", rows_e, cols_e)
            else:
                ge = O1[bi[:, None], ds["j" + s], ds["k" + s]]
            gsel[s] = np.where(ds["p" + s], ge, 1.0)
        R = np.where(ds["act"], ds["pref"] * gsel["u"] * gsel["d"], 0.0)
        g = R / np.sqrt(1 + R * R)
        f = g @ self.Gm
        z = np.concatenate([x, f], 1)
        h2 = np.tanh(z @ p["W3"] + p["b3"])
        o2 = h2 @ p["W4"]
        dets = {}
        for s, rows, N, o0 in (("u", ru, nu, 0), ("d", rd, nd, V * nu)):
            O2 = c["O1" + s] + o2[:, o0:o0 + V * N].reshape(B, V, N)
            A2 = O2[bi[:, None], rows]
            c["A2" + s] = A2
            dets[s] = np.linalg.det(A2)
        c.update(gsel=gsel, R=R, z=z, h2=h2, psi=dets["u"] * dets["d"])
        return c

    def grad(self, c, coef, per_sample, rows_u, rows_d):
        p, V, B = self.p, self.V, c["B"]
        bi = np.arange(B)
        ds = c["ds"]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        cp = coef * c["psi"]
        dO2 = {}
        for s, rows, N in (("u", rows_u, self.nu), ("d", rows_d, self.nd)):
            dA2 = cp[:, None, None] * np.swapaxes(self._inv(c["A2" + s]), -1, -2)
            Z = np.zeros((B, V, N))
            Z[bi[:, None], rows] = dA2
            dO2[s] = Z
        dout = np.concatenate([dO2["u"].reshape(B, -1), dO2["d"].reshape(B, -1)], 1)
        g = {}
        h2, z = c["h2"], c["z"]
        if per_sample:
            g["W4"] = (h2[:, :, None] * dout[:, None, :]).reshape(B, -1)
        else:
            g["W4"] = h2.T @ dout
        dpre2 = (dout @ p["W4"].T) * (1 - h2 ** 2)
        if per_sample:
            g["W3"] = (z[:, :, None] * dpre2[:, None, :]).reshape(B, -1)
        else:
            g["W3"] = z.T @ dpre2
        g["b3"] = red(dpre2)
        df = (dpre2 @ p["W3"].T)[:, 2 * V:]
        R = c["R"]
        dR = np.where(ds["act"], (df @ self.Gm.T) * (1 + R * R) ** -1.5, 0.0)
        gs = c["gsel"]
        dO1 = {}
        for s, o, rows, N in (("u", "d", rows_u, self.nu), ("d", "u", rows_d, self.nd)):
            msk = ds["p" + s]
            dgs = np.where(msk, dR * ds["pref"] * gs[o], 0.0)
            dG = np.zeros((B, V, N))
            bb = np.broadcast_to(bi[:, None], msk.shape)
            np.add.at(dG, (bb[msk], ds["j" + s][msk], ds["k" + s][msk]), dgs[msk])
            if self.feat == "G":
                Ai, O1 = c["Ai" + s], c["O1" + s]
                AiT = np.swapaxes(Ai, -1, -2)
                d1 = dG @ AiT
                dAi = np.swapaxes(O1, -1, -2) @ dG
                dA = -AiT @ dAi @ AiT
                np.add.at(d1, (np.broadcast_to(bi[:, None], rows.shape), rows), dA)
            else:
                d1 = dG
            dO1[s] = d1 + dO2[s]
        g["P0u"], g["P0d"] = red(dO1["u"]), red(dO1["d"])
        if per_sample:
            g["P0u"], g["P0d"] = g["P0u"].reshape(B, -1), g["P0d"].reshape(B, -1)
        dout1 = np.concatenate([dO1["u"].reshape(B, -1), dO1["d"].reshape(B, -1)], 1)
        h1, x = c["h1"], c["x"]
        if per_sample:
            g["W2"] = (h1[:, :, None] * dout1[:, None, :]).reshape(B, -1)
        else:
            g["W2"] = h1.T @ dout1
        g["b2"] = red(dout1)
        dpre1 = (dout1 @ p["W2"].T) * (1 - h1 ** 2)
        g["W1"] = (x[:, :, None] * dpre1[:, None, :]).reshape(B, -1) if per_sample else x.T @ dpre1
        g["b1"] = red(dpre1)
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])

    def _ds(self, idx):
        d = self.desc()
        return {k: v[idx] for k, v in d.items()}

    def _fw(self, idx):
        ou, od, ru, rd = self._occ(idx)
        return self.forward(ou, od, ru, rd, self._ds(idx)), ru, rd

    def amplitudes(self, chunk):
        D = self.m.D
        out = np.empty(D)
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            out[idx] = self._fw(idx)[0]["psi"]
        return out

    def force(self, rcoef, chunk):
        D, tot = self.m.D, None
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            c, ru, rd = self._fw(idx)
            gg = self.grad(c, rcoef[idx], False, ru, rd)
            tot = gg if tot is None else tot + gg
        return tot

    def per_sample_O(self, idx):
        c, ru, rd = self._fw(idx)
        psi = c["psi"]
        return self.grad(c, 1.0 / np.where(np.abs(psi) > 1e-300, psi, 1e-300), True, ru, rd)


# ============================================================================ arms
def model(k):
    Lx, Ly, nu, nd = SIZES[k]
    m = TJ(Lx, Ly, "obc", "obc", nu, nd, 1.0, -0.25, 0.4)
    return m


def make(arm, m, ref, seed):
    T = LRAnsatz(m, ref, "full", K=1, Hs=WC, seed=seed).n_params()
    if arm == "nnbfW":
        return LRAnsatz(m, ref, "full", K=1, Hs=WC, seed=seed), T
    feat = "O" if arm.startswith("drfO") else "G"
    frame = arm[-1]
    h2 = 0
    while DRF(m, ref, H1, h2 + 1, frame, seed, feat).n_params() <= T:
        h2 += 1
    assert h2 >= 1, f"{arm}: no H2 fits under T = {T}"
    return DRF(m, ref, H1, h2, frame, seed, feat), T


# ============================================================================ checks
def run_checks():
    ok = {}
    for k in (0, 1):
        m = model(k)
        rng = np.random.default_rng(5)
        ref = (rng.standard_normal((m.V, m.nup)), rng.standard_normal((m.V, m.ndn)))
        a = DRF(m, ref, H1, 2, "H", 1)
        a.p["W2"][:] = 0.0
        a.p["b2"][:] = 0.0
        c = a._fw(np.arange(m.D))[0]
        du = np.linalg.det(ref[0][m.rows_u])
        dd = np.linalg.det(ref[1][m.rows_d])
        iu, idd = m.iu_id()
        phi = du[iu] * dd[idd]
        nbu, nbd, act, hc = R5T.edge_moves_bits(m.bu, m.bd, a.edges, m.T, m.J)
        Mi = -np.ones(act.shape, np.int64)
        Mi[act] = m.index(nbu[act], nbd[act])
        Rex = np.where(act, hc / a.scale[None] * phi[np.maximum(Mi, 0)], 0.0) / phi[:, None]
        rel = float(np.max(np.abs(c["R"] - Rex)) / np.max(np.abs(Rex)))
        ok[f"check1_size{k}"] = rel
        log(f"CHECK 1 size {k}: max |R_G - R_exact| / max|R| = {rel:.2e} (edges {a.E}, active frac "
            f"{act.mean():.3f})")
    m = model(0)
    ref = uhf_orbitals(m, H21.UHF_SEED)
    for feat in ("G", "O"):
        a = DRF(m, ref, H1, 2, "H", 3, feat)
        x = a.flat().copy()
        x += 0.05 * np.random.default_rng(1).standard_normal(len(x))
        a.set_flat(x)
        r = np.random.default_rng(2).standard_normal(m.D)
        f = a.force(r, 4096)
        worst = 0.0
        for kk in np.random.default_rng(4).choice(len(x), 12, replace=False):
            e = np.zeros(len(x)); e[kk] = 1e-6
            a.set_flat(x + e); p1 = a.amplitudes(4096)
            a.set_flat(x - e); p2 = a.amplitudes(4096)
            fd = r @ (p1 - p2) / 2e-6
            worst = max(worst, abs(fd - f[kk]) / max(abs(fd), 1e-8))
        a.set_flat(x)
        ok[f"check2_{feat}"] = worst
        log(f"CHECK 2 feat {feat}: analytic force vs finite differences, worst rel {worst:.2e} (12 params)")
        idx = np.random.default_rng(6).choice(m.D, 7, replace=False)
        O = a.per_sample_O(idx)
        psi = a.amplitudes(4096)
        rows = []
        for ii in idx:
            rr = np.zeros(m.D); rr[ii] = 1.0
            rows.append(a.force(rr, 4096) / psi[ii])
        rel4 = float(np.max(np.abs(O - np.array(rows))) / np.max(np.abs(O)))
        ok[f"check4_{feat}"] = rel4
        log(f"CHECK 4 feat {feat}: per_sample_O vs force rows rel {rel4:.2e}")
    a = DRF(m, ref, H1, 2, "H", 3)
    a.p["W4"][:] = 0.0
    nb = LRAnsatz(m, ref, "full", K=1, Hs=H1, seed=0)
    nb.p["W1"], nb.p["b1"], nb.p["W2"], nb.p["b2"] = a.p["W1"], a.p["b1"], a.p["W2"], a.p["b2"]
    d3 = float(np.max(np.abs(a.amplitudes(4096) - nb.amplitudes(4096))))
    ok["check3"] = d3
    log(f"CHECK 3: W4 = 0 vs pass-1 NNBF max abs diff {d3:.2e}")
    valid = (ok["check1_size0"] < 1e-10 and ok["check1_size1"] < 1e-10 and ok["check2_G"] < 1e-5
             and ok["check2_O"] < 1e-5 and ok["check3"] < 1e-12 and ok["check4_G"] < 1e-10
             and ok["check4_O"] < 1e-10)
    ok["valid"] = valid
    log(f"CHECKS valid: {valid}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(ok, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=float)
    return valid


# ============================================================================ training
def train(args):
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 30))
    os.makedirs(OUT, exist_ok=True)
    sra = SimpleNamespace(sr_iters=args.iters, sr_lr=0.05, sr_shift=1e-3, sr_samples=600, chunk=8192,
                          o_chunk=4)
    jobs = []
    for k in args.size_list:
        for arm in args.arm_list:
            for seed in args.seed_list:
                fn = os.path.join(OUT, f"{k}_{arm}_{seed}.json")
                if not os.path.exists(fn):
                    jobs.append((k, arm, seed, fn))
    if not jobs:
        log("nothing to do (all JSONs exist)")
        return 0
    cache = {}
    for k in sorted({j[0] for j in jobs}):
        m = model(k)
        W, Vv, deg = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
        cache[k] = (m, Vv[:, 0], float(W[0]), uhf_orbitals(m, H21.UHF_SEED))
    # ---- budget projection before the heavy part
    proj = 0.0
    per = {}
    for k, arm, seed, fn in jobs:
        key = (k, arm)
        if key not in per:
            m, psi0, E0, ref = cache[k]
            a, T = make(arm, m, ref, seed)
            rng = np.random.default_rng(0)
            H21.sr_step(a, m, None, "none", 0, sra, rng)  # warm-up (tables)
            t0 = time.process_time()
            for _ in range(3):
                H21.sr_step(a, m, None, "none", 0, sra, rng)
            per[key] = (time.process_time() - t0) / 3
        proj += per[key] * args.iters * 1.05 + 5.0
    used = time.process_time()
    log(f"budget: projected {proj:.0f} CPU-s for {len(jobs)} runs + {used:.0f} used; cap {args.cpu_cap:.0f}"
        f" (per-step CPU-s: {', '.join(f'{a}@{b}={v:.3f}' for (b, a), v in per.items())})")
    if proj + used > 0.9 * args.cpu_cap:
        log("BUDGET-ABORT: projected CPU exceeds 90% of the cap; split the invocation")
        return 3
    for k, arm, seed, fn in jobs:
        m, psi0, E0, ref = cache[k]
        t0, c0 = time.time(), time.process_time()
        a, T = make(arm, m, ref, seed)
        H21.optimize(a, m, None, "none", 0, sra, np.random.default_rng(seed + 5), f"s{k} {arm} {seed}")
        psi = a.amplitudes(8192)
        Et = energy(m.H, psi)[0]
        s = np.where(psi >= 0, 1.0, -1.0)
        if (psi0 @ psi) < 0:
            s = -s
        des = dE_sign(m.H, s, psi0, E0, 3000)
        row = dict(size=k, V=m.V, D=m.D, arm=arm, seed=seed, n_params=a.n_params(), T=T,
                   H2=getattr(a, "p", {}).get("W4", np.zeros((0,))).shape[0] if arm != "nnbfW" else 0,
                   dEt=Et - E0, dEs=des, sec=time.time() - t0, cpu=time.process_time() - c0)
        json.dump(row, open(fn, "w"), indent=1, default=float)
        log(f"size {k} {arm} seed {seed}: n_params {a.n_params()} (T {T}) dE_tot {Et - E0:.5f} "
            f"dE_sign {des:.5f} ({row['sec']:.0f} s, {row['cpu']:.0f} CPU-s)")
    log(f"invocation CPU total {time.process_time():.0f} s")
    return 0


# ============================================================================ verdict
def verdict(args):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*_*_*.json")))]
    chk = json.load(open(os.path.join(OUT, "checks.json"))) if os.path.exists(os.path.join(OUT, "checks.json")) \
        else {"valid": False}
    med = {}
    for r in R:
        med.setdefault((r["size"], r["arm"]), []).append(r["dEs"])
    med = {k: float(np.median(v)) for k, v in med.items()}
    sizes = sorted({k for k, _ in med})
    counted = [k for k in sizes if med.get((k, "nnbfW"), 0) >= 0.005]
    out = {}
    for F in ("K", "X", "H"):
        rows = []
        for k in sizes:
            if (k, f"drf{F}") in med and (k, "nnbfW") in med:
                rw = med[(k, f"drf{F}")] / med[(k, "nnbfW")]
                ro = med[(k, f"drf{F}")] / med[(k, f"drfO{F}")] if (k, f"drfO{F}") in med else None
                rows.append(dict(size=k, dEs_W=med[(k, "nnbfW")], dEs_drf=med[(k, f"drf{F}")], r_W=rw, r_O=ro,
                                 counted=k in counted))
        cr = [r for r in rows if r["counted"]]
        if not chk.get("valid"):
            v = "INVALID"
        elif len(cr) < 2 or len(rows) < 3:
            v = "NO-TARGET" if len(cr) < 2 and len(rows) == 3 else "INCOMPLETE"
        else:
            big = cr[-1]
            if sum(r["r_W"] >= 0.9 for r in cr) >= 2 and big["r_W"] >= 0.9:
                v = "KILL"
            elif all(r["r_W"] <= 0.7 for r in cr) and big["r_W"] <= cr[0]["r_W"] + 0.1:
                v = "PASS-W"
            else:
                v = "UNDECIDED"
            if v != "KILL" and big["r_O"] is not None:
                v += "; attribution " + ("PASS (inverse)" if big["r_O"] <= 0.9 else "DEPTH-ONLY")
        for r in rows:
            print(f"  DRF-{F} size {r['size']}: dEs_W {r['dEs_W']:.5f} dEs_drf {r['dEs_drf']:.5f} r_W {r['r_W']:.3f}"
                  + (f" r_O {r['r_O']:.3f}" if r["r_O"] is not None else "") + f" counted {r['counted']}")
        print(f"VERDICT DRF-{F}: {v}")
        out[F] = dict(rows=rows, verdict=v)
    json.dump(out, open(os.path.join(OUT, "verdict.json"), "w"), indent=1, default=float)
    return out


def synthetic_verdict_test():
    """Rule branches on synthetic r_W triples (for the dry-run record)."""
    cases = {"KILL": [1.0, 0.95, 1.1], "PASS-W": [0.5, 0.6, 0.55], "UNDECIDED": [0.8, 0.85, 0.95]}
    for want, rw in cases.items():
        cr = [dict(r_W=x) for x in rw]
        big = cr[-1]
        if sum(r["r_W"] >= 0.9 for r in cr) >= 2 and big["r_W"] >= 0.9:
            v = "KILL"
        elif all(r["r_W"] <= 0.7 for r in cr) and big["r_W"] <= cr[0]["r_W"] + 0.1:
            v = "PASS-W"
        else:
            v = "UNDECIDED"
        print(f"synthetic {rw}: {v} (want {want}) {'OK' if v == want else 'MISMATCH'}")


# ============================================================================ Gate 4 cost
def cost(args):
    from p11_core import build_lattice  # noqa: F401
    rows = []
    for L in (4, 5, 6, 7):
        V = L * L
        nh = max(2, V // 8)
        nu = nd = (V - nh) // 2
        m = SimpleNamespace(V=V, nup=nu, ndn=nd, J=0.4)
        # nearest-neighbour + diagonal hoppings on an L x L OBC cluster
        T = np.zeros((V, V))
        nn = []
        for xx in range(L):
            for yy in range(L):
                i = xx * L + yy
                for dx, dy, t in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, -0.25), (1, -1, -0.25)):
                    X, Y = xx + dx, yy + dy
                    if 0 <= X < L and 0 <= Y < L:
                        j = X * L + Y
                        T[i, j] = T[j, i] = t
                        if t == 1.0:
                            nn.append((i, j))
        m.T, m.nn = T, nn
        rng = np.random.default_rng(L)
        B = 256
        bu = np.zeros(B, np.int64); bd = np.zeros(B, np.int64)
        for b in range(B):
            perm = rng.permutation(V)
            for s in perm[:nu]:
                bu[b] |= np.int64(1) << np.int64(s)
            for s in perm[nu:nu + nd]:
                bd[b] |= np.int64(1) << np.int64(s)
        ou, ru = R5T.bits_to_occ(bu, V, nu)
        od, rd = R5T.bits_to_occ(bd, V, nd)
        ref = (rng.standard_normal((V, nu)), rng.standard_normal((V, nd)))
        a = DRF.__new__(DRF)
        a.m, a.feat, a.frame = m, "G", "H"
        a.V, a.nu, a.nd = V, nu, nd
        a.edges, a.scale, a.Gm, a.nch = build_edges(m, "H")
        a.E = len(a.edges)
        no, nin2 = V * (nu + nd), 2 * V + a.nch * V
        a.p = {"P0u": ref[0], "P0d": ref[1], "W1": rng.standard_normal((2 * V, H1)) / 9, "b1": np.zeros(H1),
               "W2": 1e-3 * rng.standard_normal((H1, no)), "b2": np.zeros(no),
               "W3": rng.standard_normal((nin2, 3)) / 9, "b3": np.zeros(3), "W4": 1e-3 * rng.standard_normal((3, no))}
        a.keys = list(a.p)
        ds = edge_desc(bu, bd, a.edges, a.scale, T, 0.4)
        mm = SimpleNamespace(V=V, nup=nu, ndn=nd)
        nb = LRAnsatz(mm, ref, "full", K=1, Hs=WC, seed=0)
        reps = 20
        nb.forward(ou, od, ru, rd); a.forward(ou, od, ru, rd, ds)   # warm-up
        tn, td = [], []
        for _ in range(reps):   # interleaved, best-of to suppress noise
            t0 = time.perf_counter(); nb.forward(ou, od, ru, rd); tn.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); a.forward(ou, od, ru, rd, ds); td.append(time.perf_counter() - t0)
        tn, td = min(tn) / B, min(td) / B
        rows.append((L, V, nu + nd, a.E, tn, td))
        log(f"cost L={L} V={V} Ne={nu + nd} E_H={a.E}: NNBF-6 {tn * 1e6:.1f} us/amp, DRF-H {td * 1e6:.1f} us/amp, "
            f"ratio {td / tn:.2f}")
    Ne = np.log([r[2] for r in rows])
    sl = np.polyfit(Ne, np.log([r[5] / r[4] for r in rows]), 1)[0]
    log(f"cost: log-log slope of the DRF/NNBF per-amplitude ratio in N_e = {sl:.2f}; "
        f"local-energy passes per sample are equal (1 + n_conn) for both by construction")
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r7 T1b: trained DRF vs matched width (see docstring)")
    ap.add_argument("--mode", choices=["checks", "train", "verdict", "cost"], required=True)
    ap.add_argument("--sizes", default="0,1,2")
    ap.add_argument("--arms", default="nnbfW,drfK,drfX,drfH")
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu_cap", type=float, default=600.0)
    a = ap.parse_args(argv)
    a.size_list = [int(s) for s in a.sizes.split(",")]
    a.arm_list = a.arms.split(",")
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    assert all(s in SIZES for s in a.size_list)
    assert all(x in ("nnbfW", "drfK", "drfX", "drfH", "drfOK", "drfOX", "drfOH") for x in a.arm_list)
    return a


def main(argv=None):
    args = parse_args(argv)
    log(f"threads {NTHREADS}")
    if args.mode == "checks":
        return 0 if run_checks() else 1
    if args.mode == "train":
        return train(args)
    if args.mode == "verdict":
        synthetic_verdict_test()
        verdict(args)
        return 0
    cost(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
