#!/usr/bin/env python3
"""
P1.2 closed-leaf falsifiers (2026-11-16, idle-time queue, SKILL.md Stage 3.5.3 step 3). Two T1 tests of the
CLOSED leaf, each with its rule written here BEFORE any run. A falsification reopens the leaf with that route
only. Reuses the t1c_warm.py machinery (sizes, gap asserts, saved foils, SR recipe, dE_sign metric).
==============================================================================================================

PART A  The Saturation Diagnostic's own falsifier (route "LRK": low-rank ratio-fed kernel carrier)
  The diagnostic (Round 7 critic, adopted at closure) names the minimal missing ingredient: "a sign carrier that
  is continuous in its parameters, takes the ratio information as input, and adds new orbital directions at a
  parameter cost below one width unit ... e.g. a low-rank update O2 = O1 + u(n) v(n)^T whose factors come from
  local ratio features through a weight-shared kernel", and its falsifying experiment: "any gap-asserted
  three-size T1 in which a ratio-fed carrier reaches r <= 0.7 against the matched-width foil from the same warm
  start, with the attribution control showing that the gain needs the inverse."
  Ansatz LRK (class LRK). Pass 1 is the trained foil (NNBF width 6): O1_s = P0_s + reshape_s(tanh(xW1+b1)W2+b2).
  Frame-H ratio features f (B, 3V) exactly as t1b_drf.DRF (frozen-frame Green's-function ratios G = O1 A^-1,
  g_e = R_e / sqrt(1 + R_e^2), gathered per site and channel). Per-site input phi_v = (x_up_v, x_dn_v, f_0v,
  f_1v, f_2v) (5 numbers). Weight-shared kernel: h_v = tanh(phi_v Wk + bk) (K = 8 hidden), U_v = h_v Ua + cu,
  T_v = h_v Ta + ct (2R outputs each, R = 2 ranks per spin). Per spin s: O2_s = O1_s + U_s (T_s^T O1_s), i.e.
  every orbital column gains the configuration-dependent site directions U_s(n) (rank R). Amplitude
  det O2_up[rows] det O2_dn[rows]. Parameter count of the carrier: 5K+K+2(2R K+2R) = 120, independent of V
  (one width unit is 145 parameters at V = 12, 65 at V = 8). Ua = cu = 0 at the start, so the warm start equals
  the foil exactly. feat "O" (attribution control) replaces G by O1 (no inverse), one change.
  Arms (each 300 more exact-sum SR steps from the saved t1c foil, recipe of t1c: lr 0.05, shift 1e-3,
  600 metric samples, unprojected, seeds 11 and 22, sizes 0, 1, 2 of t1c):
    lrk   the carrier above;    lrkO  its attribution control (run only after a PASS-W of lrk);
    wid   the foil widened to the smallest width with n_params >= n_params(lrk), zero new output rows.
  RULE A (the diagnostic's falsifier, t1c's rule with r = dEs[lrk] / dEs[wid]):
    counted size iff the foil's median dEs >= 0.005 (as t1c: sizes 1 and 2 are counted, size 0 is not).
    KILL if r >= 0.9 at >= 2 counted sizes including the largest; PASS-W if r <= 0.7 at every counted size and
    r(largest) <= r(smallest) + 0.1; UNDECIDED otherwise.
    DIAGNOSTIC FALSIFIED (reopen P1.2 with route LRK only) iff PASS-W AND dEs[lrk] <= 0.9 dEs[lrkO] at the
    largest size. PASS-W with the attribution failing: "DEPTH-ONLY", the diagnostic stands, recorded.
    KILL or UNDECIDED: the diagnostic stands (its falsifier was run at T1 and did not fire).

PART B  Projected-control re-check of the key kill (route 8.3 DRF-WS vs matched width), longer training
  Cross-node critic Major 4: the closing T1 used unprojected toys. Re-run the 8.3 comparison with the
  space-group projection P_chi (|G| = 4: identity, Mx, My, MxMy on the open 4x3 or 6x2 cluster; chi read from
  the ED ground state; SR on psi_P = P_chi psi_theta, exact force, projected per-sample log-derivatives), on
  ONE gapped toy, for twice the closing T1's steps.
  Toy: size 2 (4x3 5+5, gap 0.085) first. Stage foilP: NNBF width 6 from the UHF start, 300 projected SR steps
  (seeds 11, 22; t1c recipe otherwise). Counted iff the median dEs of psi_P(foilP) >= 0.005. If size 2 is not
  counted, size 1 (6x2 5+5, gap 0.118) is tried once under the same rule; if neither is counted the re-check is
  NO-TARGET-P (projection removes the target on these toys) and the closure is scoped to unprojected toys.
  Arms from the saved foilP, projected SR, 600 steps, scored after 300 and 600 steps:
    wsP   8.3 DRF-WS (pass 2 with W4 = 0, H2 = 6 as in t1c);  wideP  foilP widened to width 14 (as t1c wide).
  RULE B (r_P(n) = median dEs[wsP] / median dEs[wideP] after n steps; decided at n = 600):
    KILL-CONFIRMED if r_P(600) >= 0.9;  REVERSAL if r_P(600) <= 0.7 (reopens P1.2 with route 8.3 only, whose
    first step is the feature-O attribution control);  UNDECIDED-P otherwise (no reopen; the closure is scoped).
    Reported, not gated: r_P(300) and the trend r_P(600) - r_P(300) (the "300-step learning-speed" objection).

Validity (else INVALID): t1c gap asserts; CHECK A every warm start reproduces the foil amplitudes (< 1e-10 rel);
  CHECK B LRK force vs central finite differences (rel < 1e-5, feat G and O, size 0, perturbed parameters);
  CHECK C per-sample log-derivatives times psi equal force rows (< 1e-10); CHECK P the projected per-sample
  log-derivative equals force(P_chi e_n) / psi_P(n) (< 1e-10) for LRK, DRF and LRAnsatz on size 2.
Budget (enforced before every heavy part): RLIMIT_CPU per invocation (--cpu_cap, default 580 s); 3 timed SR
  steps per (size, arm) give a projection; abort (BUDGET-ABORT) if it exceeds 90% of the invocation cap or if
  the leaf ledger (results/t1e/ledger.json) plus the projection exceeds --task_cap (default 1800 CPU-s = the
  30 CPU-minute leaf cap). One thread by default, at most 4.

Usage
-----
  python3 t1e_closed.py --mode checks
  python3 t1e_closed.py --mode trainA --sizes 0,1,2 --arms lrk,wid
  python3 t1e_closed.py --mode foilP --sizes 2
  python3 t1e_closed.py --mode trainB --sizes 2 --arms wsP,wideP
  python3 t1e_closed.py --mode verdict
Options: --mode {checks,trainA,foilP,trainB,verdict}
           checks : CHECKs A, B, C, P (writes results/t1e/checks.json)
           trainA : Part A arms (lrk, lrkO, wid) from the saved t1c foils; JSON per (size, arm, seed)
           foilP  : Part B projected foil (results/t1e/foilP_<size>_<seed>.npz and JSON)
           trainB : Part B arms (wsP, wideP) from foilP, 2 x 300 projected steps, JSON with both snapshots
           verdict: applies RULE A and RULE B to the JSONs present
         --sizes LIST (subset of 0,1,2); --arms LIST; --seeds LIST (default 11,22); --threads INT (1..4);
         --cpu_cap FLOAT (default 580); --task_cap FLOAT (default 1800).
"""
import os
import sys


def _pre_threads():
    n = 1
    for i, a in enumerate(sys.argv):
        if a.startswith("--threads="):
            n = int(a.split("=")[1])
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    n = max(1, min(4, n))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)
    return n


NTHREADS = _pre_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1c_warm as C  # noqa: E402
import t1b_drf as B  # noqa: E402
from tj_core import LRAnsatz, SpaceGroup  # noqa: E402
from p11_core import energy  # noqa: E402
import p21_harness as H21  # noqa: E402
from t1_vote import dE_sign  # noqa: E402

OUT = os.path.join(HERE, "results", "t1e")
LEDGER = os.path.join(OUT, "ledger.json")
log = B.log
KH, RR = 8, 2


# ============================================================================ projected per-sample O for DRF-type
def per_sample_O_proj_drf(self, sg, idx, psiP, o_chunk=8):
    G = sg.G
    out = np.empty((len(idx), self.n_params()))
    for s in range(0, len(idx), o_chunk):
        ii = idx[s:s + o_chunk]
        nb = len(ii)
        img = sg.inv[:, ii].T.astype(np.int64)
        coef = sg.chi[None, :] * sg.sinv[:, ii].T / (G * psiP[ii][:, None])
        c, ru, rd = self._fw(img.ravel())
        gr = self.grad(c, coef.ravel(), True, ru, rd)
        out[s:s + nb] = gr.reshape(nb, G, -1).sum(1)
    return out


B.DRF.per_sample_O_proj = per_sample_O_proj_drf


# ============================================================================ LRK carrier (Part A)
class LRK(C.FastDRF):
    def __init__(self, m, ref, h1, seed, feat="G"):
        super().__init__(m, ref, h1, 1, "H", seed, feat)
        for k in ("W3", "b3", "W4"):
            del self.p[k]
        rng = np.random.default_rng(seed + 77)
        self.p["Wk"] = rng.standard_normal((5, KH)) / np.sqrt(5)
        self.p["bk"] = np.zeros(KH)
        self.p["Ua"] = np.zeros((KH, 2 * RR))
        self.p["cu"] = np.zeros(2 * RR)
        self.p["Ta"] = rng.standard_normal((KH, 2 * RR)) / np.sqrt(KH)
        self.p["ct"] = np.zeros(2 * RR)
        self.keys = list(self.p)

    def forward(self, ou, od, ru, rd, ds):
        p, V, nu, nd = self.p, self.V, self.nu, self.nd
        Bn = ou.shape[0]
        bi = np.arange(Bn)
        x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        o1 = h1 @ p["W2"] + p["b2"]
        c = dict(x=x, h1=h1, B=Bn, ds=ds)
        gsel = {}
        for s, rows, N, P0, o0 in (("u", ru, nu, p["P0u"], 0), ("d", rd, nd, p["P0d"], V * nu)):
            O1 = P0[None] + o1[:, o0:o0 + V * N].reshape(Bn, V, N)
            c["O1" + s] = O1
            if self.feat == "G":
                Ai = self._inv(O1[bi[:, None], rows])
                c["Ai" + s] = Ai
                G = O1 @ Ai                       # full frozen-frame Green's function (FastDRF trick)
            else:
                G = O1
            ge = G[bi[:, None], ds["j" + s], ds["k" + s]]
            gsel[s] = np.where(ds["p" + s], ge, 1.0)
        R = np.where(ds["act"], ds["pref"] * gsel["u"] * gsel["d"], 0.0)
        g = R / np.sqrt(1 + R * R)
        f = g @ self.Gm
        phi = np.stack([x[:, :V], x[:, V:], f[:, :V], f[:, V:2 * V], f[:, 2 * V:]], 2)    # (B, V, 5)
        hk = np.tanh(phi @ p["Wk"] + p["bk"])                                             # (B, V, K)
        U = hk @ p["Ua"] + p["cu"]
        T = hk @ p["Ta"] + p["ct"]
        dets = {}
        for si, (s, rows) in enumerate((("u", ru), ("d", rd))):
            Us, Ts = U[:, :, si * RR:(si + 1) * RR], T[:, :, si * RR:(si + 1) * RR]
            Ws = np.swapaxes(Ts, 1, 2) @ c["O1" + s]                                      # (B, R, N)
            A2 = (c["O1" + s] + Us @ Ws)[bi[:, None], rows]
            c["A2" + s], c["W" + s] = A2, Ws
            dets[s] = np.linalg.det(A2)
        c.update(gsel=gsel, R=R, phi=phi, hk=hk, U=U, T=T, psi=dets["u"] * dets["d"])
        return c

    def grad(self, c, coef, per_sample, rows_u, rows_d):
        p, V, Bn = self.p, self.V, c["B"]
        bi = np.arange(Bn)
        ds = c["ds"]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        cp = coef * c["psi"]
        U, T, hk, phi = c["U"], c["T"], c["hk"], c["phi"]
        dU, dT, dO1dir = np.zeros_like(U), np.zeros_like(T), {}
        for si, (s, rows, N) in enumerate((("u", rows_u, self.nu), ("d", rows_d, self.nd))):
            dA2 = cp[:, None, None] * np.swapaxes(self._inv(c["A2" + s]), -1, -2)
            dO2 = np.zeros((Bn, V, N))
            dO2[bi[:, None], rows] = dA2
            Us, Ts, Ws, O1 = U[:, :, si * RR:(si + 1) * RR], T[:, :, si * RR:(si + 1) * RR], c["W" + s], c["O1" + s]
            dU[:, :, si * RR:(si + 1) * RR] = dO2 @ np.swapaxes(Ws, 1, 2)
            dWs = np.swapaxes(Us, 1, 2) @ dO2
            dT[:, :, si * RR:(si + 1) * RR] = O1 @ np.swapaxes(dWs, 1, 2)
            dO1dir[s] = dO2 + Ts @ dWs
        g = {}
        if per_sample:
            g["Ua"] = np.einsum("bvk,bvo->bko", hk, dU).reshape(Bn, -1)
            g["Ta"] = np.einsum("bvk,bvo->bko", hk, dT).reshape(Bn, -1)
        else:
            g["Ua"] = np.einsum("bvk,bvo->ko", hk, dU)
            g["Ta"] = np.einsum("bvk,bvo->ko", hk, dT)
        g["cu"], g["ct"] = red(dU.sum(1)), red(dT.sum(1))
        dpre = (dU @ p["Ua"].T + dT @ p["Ta"].T) * (1 - hk ** 2)
        if per_sample:
            g["Wk"] = np.einsum("bvi,bvk->bik", phi, dpre).reshape(Bn, -1)
        else:
            g["Wk"] = np.einsum("bvi,bvk->ik", phi, dpre)
        g["bk"] = red(dpre.sum(1))
        dphi = dpre @ p["Wk"].T
        df = np.concatenate([dphi[:, :, 2], dphi[:, :, 3], dphi[:, :, 4]], 1)
        R = c["R"]
        dR = np.where(ds["act"], (df @ self.Gm.T) * (1 + R * R) ** -1.5, 0.0)
        gs = c["gsel"]
        dO1 = {}
        for s, o, rows, N in (("u", "d", rows_u, self.nu), ("d", "u", rows_d, self.nd)):
            msk = ds["p" + s]
            dgs = np.where(msk, dR * ds["pref"] * gs[o], 0.0)
            dG = np.zeros((Bn, V, N))
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
            dO1[s] = d1 + dO1dir[s]
        g["P0u"], g["P0d"] = red(dO1["u"]), red(dO1["d"])
        if per_sample:
            g["P0u"], g["P0d"] = g["P0u"].reshape(Bn, -1), g["P0d"].reshape(Bn, -1)
        dout1 = np.concatenate([dO1["u"].reshape(Bn, -1), dO1["d"].reshape(Bn, -1)], 1)
        h1, x = c["h1"], c["x"]
        g["W2"] = (h1[:, :, None] * dout1[:, None, :]).reshape(Bn, -1) if per_sample else h1.T @ dout1
        g["b2"] = red(dout1)
        dpre1 = (dout1 @ p["W2"].T) * (1 - h1 ** 2)
        g["W1"] = (x[:, :, None] * dpre1[:, None, :]).reshape(Bn, -1) if per_sample else x.T @ dpre1
        g["b1"] = red(dpre1)
        if per_sample:
            return np.concatenate([g[k].reshape(Bn, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


# ============================================================================ helpers
_SG = {}


def group(k):
    if k not in _SG:
        m, psi0, E0, ref, gap = C.model(k)
        sg = SpaceGroup(m, diag=False)
        gen = sg.set_from(psi0)
        _SG[k] = (sg, gen)
    return _SG[k]


def foilP_fn(k, seed):
    return os.path.join(OUT, f"foilP_{k}_{seed}.npz")


def load_nb(fn, k, seed):
    m, psi0, E0, ref, gap = C.model(k)
    nb = LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=seed)
    z = np.load(fn)
    for key in nb.keys:
        nb.p[key] = z[key].copy()
    return nb


def wide_to(nb, m, ref, target, seed):
    width = B.WC
    while LRAnsatz(m, ref, "full", K=1, Hs=width, seed=0).n_params() < target:
        width += 1
    return C.widen(nb, m, ref, width, seed)


def make_arm(arm, k, seed):
    m, psi0, E0, ref, gap = C.model(k)
    if arm in ("wsP", "wideP"):
        nb = load_nb(foilP_fn(k, seed), k, seed)
    else:
        nb = C.load_foil(k, seed)
    T = nb.n_params()
    if arm in ("lrk", "lrkO"):
        a = LRK(m, ref, B.WC, seed + 3000, "O" if arm == "lrkO" else "G")
        C.copy_pass1(a, nb)
    elif arm == "wid":
        a = wide_to(nb, m, ref, make_arm("lrk", k, seed)[0].n_params(), seed)
    elif arm == "wsP":
        a = C.FastDRF(m, ref, B.WC, C.ws_h2(m, ref, T), "H", seed + 2000, "G")
        C.copy_pass1(a, nb)
        a.p["W4"][:] = 0.0
    elif arm == "wideP":
        a = C.widen(nb, m, ref, 14, seed)
    else:
        raise ValueError(arm)
    return a, nb, T


def score(a, m, psi0, E0, sg=None):
    psi = a.amplitudes(8192)
    if sg is not None:
        psi = sg.project(psi)
    Et = energy(m.H, psi)[0]
    s = np.where(psi >= 0, 1.0, -1.0)
    if psi0 @ psi < 0:
        s = -s
    return float(Et - E0), float(dE_sign(m.H, s, psi0, E0, 3000))


def ledger_used():
    return float(sum(r["cpu"] for r in json.load(open(LEDGER)))) if os.path.exists(LEDGER) else 0.0


def ledger_add(mode, cpu, note):
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    rows.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, cpu=cpu, note=note))
    json.dump(rows, open(LEDGER, "w"), indent=1)


def time_steps(a, m, sg, proj, n=3):
    rng = np.random.default_rng(0)
    x0 = a.flat().copy()
    H21.sr_step(a, m, sg, proj, 0, C.sra(1), rng)
    t0 = time.process_time()
    for _ in range(n):
        H21.sr_step(a, m, sg, proj, 0, C.sra(1), rng)
    a.set_flat(x0)
    return (time.process_time() - t0) / n


def gate(per_runs_steps, args, label):
    proj = sum(p * n * 1.05 + 8.0 for p, n in per_runs_steps)
    used, prior = time.process_time(), ledger_used()
    log(f"budget {label}: projected {proj:.0f} CPU-s + {used:.0f} used now; invocation cap {args.cpu_cap:.0f}; "
        f"leaf ledger {prior:.0f} of {args.task_cap:.0f}")
    if proj + used > 0.9 * args.cpu_cap:
        log("BUDGET-ABORT: projection exceeds 90% of the invocation cap; split the invocation")
        return False
    if prior + proj + used > args.task_cap:
        log("BUDGET-ABORT: projection would take the leaf over its CPU cap")
        return False
    return True


# ============================================================================ stages
def run_checks(args):
    os.makedirs(OUT, exist_ok=True)
    ok = {}
    for k in (0, 1, 2):
        m, psi0, E0, ref, gap = C.model(k)
        for seed in args.seed_list:
            nb = C.load_foil(k, seed)
            p0 = nb.amplitudes(8192)
            for arm in ("lrk", "lrkO", "wid"):
                a = make_arm(arm, k, seed)[0]
                d = float(np.max(np.abs(a.amplitudes(8192) - p0)) / np.max(np.abs(p0)))
                ok[f"A_{k}_{seed}_{arm}"] = d
                log(f"CHECK A size {k} seed {seed} {arm}: n_params {a.n_params()} (foil {nb.n_params()}) rel {d:.1e}")
    m, psi0, E0, ref, gap = C.model(0)
    for feat in ("G", "O"):
        a = LRK(m, ref, B.WC, 3, feat)
        x = a.flat() + 0.05 * np.random.default_rng(1).standard_normal(a.n_params())
        a.set_flat(x)
        r = np.random.default_rng(2).standard_normal(m.D)
        f = a.force(r, 4096)
        worst = 0.0
        sel = list(np.random.default_rng(4).choice(len(x), 16, replace=False))
        sel += [a.n_params() - 1 - j for j in range(0, 120, 9)]      # always probe the carrier block
        for kk in sel:
            e = np.zeros(len(x))
            e[kk] = 1e-6
            a.set_flat(x + e)
            p1 = a.amplitudes(4096)
            a.set_flat(x - e)
            p2 = a.amplitudes(4096)
            fd = r @ (p1 - p2) / 2e-6
            worst = max(worst, abs(fd - f[kk]) / max(abs(fd), 1e-8))
        a.set_flat(x)
        idx = np.random.default_rng(6).choice(m.D, 5, replace=False)
        O = a.per_sample_O(idx)
        rows = [a.force(np.eye(1, m.D, ii)[0], 4096) / a.amplitudes(4096)[ii] for ii in idx]
        rel = float(np.max(np.abs(O - np.array(rows))) / np.max(np.abs(O)))
        ok[f"B_{feat}"], ok[f"C_{feat}"] = worst, rel
        log(f"CHECK B lrk feat {feat}: force vs FD worst rel {worst:.2e}; CHECK C per-sample O vs force rows {rel:.1e}")
    # CHECK P: projected per-sample O, size 2
    m, psi0, E0, ref, gap = C.model(2)
    sg, gen = group(2)
    log(f"size 2 group: |G| {sg.G} characters {gen}")
    for name, a in (("lrk", LRK(m, ref, B.WC, 3, "G")), ("drf", C.FastDRF(m, ref, B.WC, 3, "H", 3, "G")),
                    ("nnbf", LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=3))):
        x = a.flat() + 0.05 * np.random.default_rng(1).standard_normal(a.n_params())
        a.set_flat(x)
        psiP = sg.project(a.amplitudes(8192))
        idx = np.random.default_rng(6).choice(np.flatnonzero(np.abs(psiP) > 1e-3 * np.abs(psiP).max()), 4, replace=False)
        O = a.per_sample_O_proj(sg, idx, psiP, 4)
        rows = [a.force(sg.project(np.eye(1, m.D, ii)[0]), 8192) / psiP[ii] for ii in idx]
        rel = float(np.max(np.abs(O - np.array(rows))) / np.max(np.abs(O)))
        ok[f"P_{name}"] = rel
        log(f"CHECK P {name}: projected per-sample O vs force(P e_n)/psi_P rel {rel:.1e}")
    valid = all(v < 1e-10 for kk, v in ok.items() if kk[:2] in ("A_", "C_", "P_")) and ok["B_G"] < 1e-5 \
        and ok["B_O"] < 1e-5
    ok["valid"] = bool(valid)
    log(f"CHECKS valid: {valid}")
    json.dump(ok, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=float)
    ledger_add("checks", time.process_time(), "")
    return 0 if valid else 1


def run_trainA(args):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, a, s) for k in args.size_list for a in args.arm_list for s in args.seed_list
            if not os.path.exists(os.path.join(OUT, f"A_{k}_{a}_{s}.json"))]
    per = {}
    for k, arm, s in jobs:
        if (k, arm) not in per:
            per[(k, arm)] = time_steps(make_arm(arm, k, s)[0], C.model(k)[0], None, "none")
    log("per-step CPU-s: " + ", ".join(f"{a}@{k}={v:.3f}" for (k, a), v in per.items()))
    if not gate([(per[(k, a)], 300) for k, a, _ in jobs], args, "trainA"):
        ledger_add("trainA-abort", time.process_time(), f"{args.sizes} {args.arms}")
        return 3
    for k, arm, seed in jobs:
        m, psi0, E0, ref, gap = C.model(k)
        c0 = time.process_time()
        a, nb, T = make_arm(arm, k, seed)
        H21.optimize(a, m, None, "none", 0, C.sra(300), np.random.default_rng(seed + 7), f"A s{k} {arm} {seed}")
        dEt, dEs = score(a, m, psi0, E0)
        row = dict(size=k, arm=arm, seed=seed, n_params=a.n_params(), T=T, dEt=dEt, dEs=dEs,
                   cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"A_{k}_{arm}_{seed}.json"), "w"), indent=1)
        log(f"A size {k} {arm} seed {seed}: n_params {row['n_params']} (foil {T}) dE_tot {dEt:.5f} dE_sign {dEs:.5f}"
            f" ({row['cpu']:.0f} CPU-s)")
    ledger_add("trainA", time.process_time(), f"sizes {args.sizes} arms {args.arms} seeds {args.seeds}")
    log(f"leaf ledger now {ledger_used():.0f} s")
    return 0


def run_foilP(args):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, s) for k in args.size_list for s in args.seed_list if not os.path.exists(foilP_fn(k, s))]
    per = {}
    for k in sorted({j[0] for j in jobs}):
        m, psi0, E0, ref, gap = C.model(k)
        per[k] = time_steps(LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=11), m, group(k)[0], "chi")
    if not gate([(per[k], 300) for k, _ in jobs], args, "foilP"):
        ledger_add("foilP-abort", time.process_time(), f"{args.sizes}")
        return 3
    for k, seed in jobs:
        m, psi0, E0, ref, gap = C.model(k)
        sg, gen = group(k)
        c0 = time.process_time()
        nb = LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=seed)
        H21.optimize(nb, m, sg, "chi", 0, C.sra(300), np.random.default_rng(seed + 5), f"s{k} foilP {seed}")
        np.savez(foilP_fn(k, seed), **{key: nb.p[key] for key in nb.keys})
        dEt, dEs = score(nb, m, psi0, E0, sg)
        row = dict(size=k, arm="foilP", seed=seed, chi=gen, n_params=nb.n_params(), dEt=dEt, dEs=dEs,
                   cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"B_{k}_foilP_{seed}.json"), "w"), indent=1)
        log(f"size {k} foilP seed {seed} chi {gen}: dE_tot {dEt:.5f} dE_sign {dEs:.5f} ({row['cpu']:.0f} CPU-s)")
    ledger_add("foilP", time.process_time(), f"sizes {args.sizes} seeds {args.seeds}")
    log(f"leaf ledger now {ledger_used():.0f} s")
    return 0


def run_trainB(args):
    jobs = [(k, a, s) for k in args.size_list for a in args.arm_list for s in args.seed_list
            if not os.path.exists(os.path.join(OUT, f"B_{k}_{a}_{s}.json"))]
    per = {}
    for k, arm, s in jobs:
        if (k, arm) not in per:
            per[(k, arm)] = time_steps(make_arm(arm, k, s)[0], C.model(k)[0], group(k)[0], "chi")
    log("per-step CPU-s: " + ", ".join(f"{a}@{k}={v:.3f}" for (k, a), v in per.items()))
    if not gate([(per[(k, a)], 600) for k, a, _ in jobs], args, "trainB"):
        ledger_add("trainB-abort", time.process_time(), f"{args.sizes} {args.arms}")
        return 3
    for k, arm, seed in jobs:
        m, psi0, E0, ref, gap = C.model(k)
        sg, gen = group(k)
        c0 = time.process_time()
        a, nb, T = make_arm(arm, k, seed)
        snaps = {}
        for half in (1, 2):
            H21.optimize(a, m, sg, "chi", 0, C.sra(300), np.random.default_rng(seed + 7 + 100 * half),
                         f"B s{k} {arm} {seed} half {half}")
            snaps[300 * half] = score(a, m, psi0, E0, sg)
        row = dict(size=k, arm=arm, seed=seed, n_params=a.n_params(), T=T,
                   dEt300=snaps[300][0], dEs300=snaps[300][1], dEt600=snaps[600][0], dEs600=snaps[600][1],
                   cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"B_{k}_{arm}_{seed}.json"), "w"), indent=1)
        log(f"B size {k} {arm} seed {seed}: n_params {row['n_params']} dE_sign 300 {snaps[300][1]:.5f} "
            f"600 {snaps[600][1]:.5f} dE_tot 600 {snaps[600][0]:.5f} ({row['cpu']:.0f} CPU-s)")
    ledger_add("trainB", time.process_time(), f"sizes {args.sizes} arms {args.arms} seeds {args.seeds}")
    log(f"leaf ledger now {ledger_used():.0f} s")
    return 0


def medv(prefix, k, arm, key, seeds):
    v = []
    for s in seeds:
        fn = os.path.join(OUT, f"{prefix}_{k}_{arm}_{s}.json")
        if not os.path.exists(fn):
            return None
        v.append(json.load(open(fn))[key])
    return float(np.median(v))


def run_verdict(args):
    for want, rs in (("KILL", [1.0, 0.95]), ("PASS-W", [0.5, 0.55]), ("UNDECIDED", [0.8, 0.95])):
        print(f"synthetic A {rs}: {C.decide(rs)} (want {want})")
    for want, r in (("KILL-CONFIRMED", 0.95), ("REVERSAL", 0.6), ("UNDECIDED-P", 0.8)):
        print(f"synthetic B {r}: {rule_b(r)} (want {want})")
    chk = json.load(open(os.path.join(OUT, "checks.json"))) if os.path.exists(os.path.join(OUT, "checks.json")) else {}
    print(f"checks valid: {chk.get('valid')}")
    # Part A
    rows = []
    for k in (0, 1, 2):
        foil = float(np.median([json.load(open(os.path.join(C.OUT, f"{k}_foil_{s}.json")))["dEs"]
                                for s in args.seed_list]))
        lr, wd, lrO = (medv("A", k, a, "dEs", args.seed_list) for a in ("lrk", "wid", "lrkO"))
        if lr is None or wd is None:
            continue
        rows.append(dict(size=k, foil=foil, lrk=lr, wid=wd, lrkO=lrO, r=lr / wd, counted=foil >= 0.005))
        print(f"  A size {k}: dEs foil {foil:.5f} wid {wd:.5f} lrk {lr:.5f} r {lr / wd:.3f}"
              + (f" lrkO {lrO:.5f}" if lrO is not None else "") + f" counted {foil >= 0.005}")
    cr = [r for r in rows if r["counted"]]
    if not chk.get("valid"):
        va = "INVALID"
    elif len(cr) < 2:
        va = "INCOMPLETE" if len(rows) < 3 else "NO-TARGET"
    else:
        va = C.decide([r["r"] for r in cr])
        if va == "PASS-W":
            if cr[-1]["lrkO"] is None:
                va += "; attribution pending (run lrkO)"
            else:
                va += "; DIAGNOSTIC FALSIFIED" if cr[-1]["lrk"] <= 0.9 * cr[-1]["lrkO"] else "; DEPTH-ONLY"
    print(f"VERDICT A (diagnostic falsifier, route LRK): {va}")
    # Part B
    vb = "INCOMPLETE"
    for k in (2, 1):
        fp = medv("B", k, "foilP", "dEs", args.seed_list)
        if fp is None:
            continue
        print(f"  B size {k}: projected foil median dEs {fp:.5f} counted {fp >= 0.005}")
        if fp < 0.005:
            vb = "NO-TARGET-P at size %d" % k
            continue
        w3, w6 = medv("B", k, "wsP", "dEs300", args.seed_list), medv("B", k, "wsP", "dEs600", args.seed_list)
        d3, d6 = medv("B", k, "wideP", "dEs300", args.seed_list), medv("B", k, "wideP", "dEs600", args.seed_list)
        if None in (w3, w6, d3, d6):
            vb = "INCOMPLETE"
            break
        print(f"  B size {k}: wsP {w3:.5f} -> {w6:.5f}; wideP {d3:.5f} -> {d6:.5f}; r_P(300) {w3 / d3:.3f} "
              f"r_P(600) {w6 / d6:.3f} trend {w6 / d6 - w3 / d3:+.3f}")
        vb = rule_b(w6 / d6) if chk.get("valid") else "INVALID"
        break
    print(f"VERDICT B (projected re-check of 8.3): {vb}")
    print(f"leaf CPU ledger: {ledger_used():.0f} s")
    return 0


def rule_b(r):
    return "KILL-CONFIRMED" if r >= 0.9 else ("REVERSAL" if r <= 0.7 else "UNDECIDED-P")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 closed-leaf falsifiers (see docstring)")
    ap.add_argument("--mode", choices=["checks", "trainA", "foilP", "trainB", "verdict"], required=True)
    ap.add_argument("--sizes", default="2")
    ap.add_argument("--arms", default="lrk,wid")
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu_cap", type=float, default=580.0)
    ap.add_argument("--task_cap", type=float, default=1800.0)
    a = ap.parse_args(argv)
    a.size_list = [int(s) for s in a.sizes.split(",")]
    a.arm_list = a.arms.split(",")
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    assert all(s in (0, 1, 2) for s in a.size_list)
    assert all(x in ("lrk", "lrkO", "wid", "wsP", "wideP") for x in a.arm_list)
    return a


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 30))
    log(f"threads {NTHREADS}")
    return {"checks": run_checks, "trainA": run_trainA, "foilP": run_foilP, "trainB": run_trainB,
            "verdict": run_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
