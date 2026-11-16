#!/usr/bin/env python3
"""
P1.2 round 8, tier T1 for the warm-started ratio-feature routes 8.3 (DRF-WS) and 8.4 (DRF-T), run exactly as
the round-7 per-leaf critic pre-registered them (leaf section "Round 7 per-leaf critic", "Pre-registered T1
for 8.3 (and 8.4)"), against the full-width NNBF foil, on gap-asserted toy sizes.
=============================================================================================================

Sizes (critic's list; validity assert E_1 - E_0 >= 0.02 t at every size, else the script stops):
  size 0 : 4x2 3+3 (V = 8)    size 1 : 6x2 5+5 (V = 12)    size 2 : 4x3 5+5 (V = 12)
  Sizes 1 and 2 have the same V and D; the order is the critic's list order, so "the largest" in the rule
  means size 2 (the 4x3 cluster, the only one with a 3-leg direction). Written before any run.
Model: t-J, t = 1, t' = -0.25, J = 0.4, open boundaries, unprojected (projected toys are NO-TARGET).
Start (stage "foil"): the foil nnbfW = LRAnsatz full, width 6, UHF start, 300 exact-sum SR steps (the T1b
  recipe of t1b_drf.py: lr 0.05, shift 1e-3, 600 metric samples), seeds 11 and 22. Its parameters are saved
  (results/t1c/foil_<size>_<seed>.npz) and every arm starts from them.
Arms (stage "train"), each trained for 300 more SR steps from that saved state, same recipe:
  ws    (8.3) DRF of t1b_drf.py, frame H (hole hops and exchanges), pass 1 = the trained foil (H1 = 6), pass 2
        width H2 with W4 = 0 (an exact identity start; W3 random, seed-fixed). "The largest H2 that fits":
        the largest H2 with n_params(ws) <= 2 T, T = n_params(foil) (pass 2 at most doubles the foil). This
        gives H2 = 6 = H1 at V = 12 (critic Major 2: the T1b pass 2 had H2 = 2). Written before any run.
  wsO   (8.3 attribution control) as ws with feat "O" (inverse removed); run only after a PASS-W of ws.
  wide  the foil widened to the smallest width with n_params >= n_params(ws); the new hidden units have random
        W1 rows (seed-fixed) and ZERO W2 rows, so it starts from exactly the foil's state.
  dt    (8.4, DRF-T, tied re-entry) h = tanh(x W1 + b1) + tanh([x, f] W3 + b3), O2 = P0 + reshape(h W2 + b2),
        f the frame-H ratio features of pass 1 (O1 = P0 + reshape(tanh(x W1 + b1) W2 + b2)). This is DRF with
        H2 = H1 and W4 tied to W2. W3 = 0, b3 = 0 at the start (exact identity).
  dtO   (8.4 attribution control) dt with feat "O"; run only after a PASS-W of dt.
  widet the foil widened to the smallest width with n_params >= n_params(dt), as for wide.
Metric: dEs(X) = min_{a>=0} <a|S H S|a>/<a|a> - E_0 with S = diag(sgn psi_X) (t1_vote.dE_sign); medians over
  the two seeds.
PRE-REGISTERED RULE (the critic's, copied verbatim in meaning; written 2026-11-16 before any run):
  counted size iff the foil's median dEs >= 0.005.
  r = dEs[route] / dEs[its wide foil] (ws vs wide; dt vs widet).
  KILL if r >= 0.9 at >= 2 counted sizes including the largest counted size.
  PASS-W if r <= 0.7 at every counted size AND r(largest) <= r(smallest) + 0.1.
  UNDECIDED otherwise; NO-TARGET if fewer than 2 sizes are counted.
  After a PASS-W: credited to the inverse iff dEs[ws] <= 0.9 dEs[wsO] at the largest size (dt/dtO likewise).
AMENDMENT (written 2026-11-16 after the 8.4 sizes-0/1 run, BEFORE any dts run): at V = 12 the dt arm never
  left its warm start. The first SR step of the critic's recipe (shift 1e-3, lr 0.05) is clipped to norm 1 and
  raises E by 0.68; the best-energy guard of p21_harness.optimize then returns the start, so dt at size 1
  equals the foil to all printed digits. Probes of 8 steps from the size-1 seed-11 start: lr 0.01 and 0.003
  also diverge, shift 1e-2 diverges, shift 1e-1 descends monotonically. Arm "dts" is dt trained with SR
  shift 0.1 (the only change); widet keeps the recipe (it trained normally). This change can only favour the
  route, so a KILL of dts is final; a PASS-W of dts additionally requires widet retrained at shift 0.1 to
  keep r <= 0.7 before promotion. "dt" rows are reported as 8.4-as-run and decide nothing.
Validity (else INVALID): gap assert; CHECK A every warm start reproduces the foil's amplitudes (ws, wsO with
  W4 = 0; dt, dtO with W3 = b3 = 0; wide, widet with zero new W2 rows) to < 1e-10 relative; CHECK B for dt
  the analytic force matches central finite differences (rel < 1e-5) on size 0 (the ws force is t1b CHECK 2);
  CHECK C the retrained foil's dEs at size 0 seed 11 is reproducible (two foil runs agree to 1e-9 is not
  re-run to save budget; determinism is inherited from t1b, which reproduced to 1e-9 in critic_r7_premise).
Budget (enforced, set before the heavy part): every invocation sets RLIMIT_CPU to --cpu_cap (default 540 s)
  and times 3 SR steps per (size, arm) before training; it aborts with BUDGET-ABORT if the projection exceeds
  90% of the per-invocation cap, or if the task ledger (results/t1c/ledger.json: actual CPU-s of every earlier
  invocation) plus the projection exceeds --task_cap (default 1800 s = 30 CPU-minutes). One thread by default;
  --threads is capped at 4. Actual CPU-s of each run and invocation are logged and added to the ledger.
Usage
-----
  python3 t1c_warm.py --mode gaps
  python3 t1c_warm.py --mode foil  --sizes 0,1,2
  python3 t1c_warm.py --mode checks
  python3 t1c_warm.py --mode train --sizes 1 --arms ws,wide
  python3 t1c_warm.py --mode verdict
Options: --mode {gaps,foil,checks,train,verdict}; --sizes LIST (subset of 0,1,2); --arms LIST (subset of
  ws,wsO,wide,dt,dtO,widet); --seeds LIST (default 11,22); --iters INT (default 300); --threads INT (1..4);
  --cpu_cap FLOAT (default 540); --task_cap FLOAT (default 1800).
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
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1b_drf as B  # noqa: E402
from tj_core import TJ, LRAnsatz  # noqa: E402
from p11_core import solve_ed, uhf_orbitals, energy  # noqa: E402
import p21_harness as H21  # noqa: E402
from t1_vote import dE_sign  # noqa: E402

SIZES = {0: (4, 2, 3, 3), 1: (6, 2, 5, 5), 2: (4, 3, 5, 5)}
GAP_MIN = 0.02
OUT = os.path.join(HERE, "results", "t1c")
LEDGER = os.path.join(OUT, "ledger.json")
ROUTE_ARMS = {"8.3": ("ws", "wsO", "wide"), "8.4-as-run": ("dt", "dtO", "widet"), "8.4": ("dts", "dtsO", "widet"),
              "8.4-tiebreak": ("dts", "dtsO", "widets")}  # tie-break pre-registered in the leaf (Round 8)
SHIFT_ARMS = {"dts": 0.1, "dtsO": 0.1, "widets": 0.1}  # widets: widet retrained at shift 0.1 (tie-break)
log = B.log


# ============================================================================ faster forward (same maths)
class FastDRF(B.DRF):
    """t1b_drf.DRF with the frozen-ratio gather done as G_s = O1_s A_s^{-1} in full (B V N^2 flops) followed by
    a (B, E) lookup, instead of two (B, E, N) gathers. Identical in exact arithmetic; CHECK D compares it with
    t1b_drf.DRF.forward. grad() is inherited unchanged (it uses only Ai, O1, gsel, R)."""

    def forward(self, ou, od, ru, rd, ds):
        p, V, nu, nd = self.p, self.V, self.nu, self.nd
        B_ = ou.shape[0]
        bi = np.arange(B_)
        x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        o1 = h1 @ p["W2"] + p["b2"]
        c = dict(x=x, h1=h1, B=B_, ds=ds)
        gsel = {}
        for s, rows, N, P0, o0 in (("u", ru, nu, p["P0u"], 0), ("d", rd, nd, p["P0d"], V * nu)):
            O1 = P0[None] + o1[:, o0:o0 + V * N].reshape(B_, V, N)
            c["O1" + s] = O1
            if self.feat == "G":
                Ai = self._inv(O1[bi[:, None], rows])
                c["Ai" + s] = Ai
                G = O1 @ Ai
            else:
                G = O1
            ge = G[bi[:, None], ds["j" + s], ds["k" + s]]
            gsel[s] = np.where(ds["p" + s], ge, 1.0)
        R = np.where(ds["act"], ds["pref"] * gsel["u"] * gsel["d"], 0.0)
        g = R / np.sqrt(1 + R * R)
        z = np.concatenate([x, g @ self.Gm], 1)
        h2 = np.tanh(z @ p["W3"] + p["b3"])
        o2 = h2 @ p["W4"]
        dets = {}
        for s, rows, N, o0 in (("u", ru, nu, 0), ("d", rd, nd, V * nu)):
            A2 = (c["O1" + s] + o2[:, o0:o0 + V * N].reshape(B_, V, N))[bi[:, None], rows]
            c["A2" + s] = A2
            dets[s] = np.linalg.det(A2)
        c.update(gsel=gsel, R=R, z=z, h2=h2, psi=dets["u"] * dets["d"])
        return c

    # The SR step calls amplitudes() and then force() at the same parameters; the forward chunks of the
    # amplitudes() call are cached and reused by force() (bit-identical; saves one full forward per step).
    def amplitudes(self, chunk):
        D = self.m.D
        out = np.empty(D)
        cache = []
        for s in range(0, D, chunk):
            idx = np.arange(s, min(s + chunk, D))
            c, ru, rd = self._fw(idx)
            out[idx] = c["psi"]
            cache.append((idx, c, ru, rd))
        self._fcache = (self.flat().copy(), chunk, cache)
        return out

    def force(self, rcoef, chunk):
        fc = getattr(self, "_fcache", None)
        if fc is not None and fc[1] == chunk and np.array_equal(fc[0], self.flat()):
            tot = None
            for idx, c, ru, rd in fc[2]:
                gg = self.grad(c, rcoef[idx], False, ru, rd)
                tot = gg if tot is None else tot + gg
            return tot
        return super().force(rcoef, chunk)


# ============================================================================ tied re-entry (8.4)
class DRFT(FastDRF):
    """DRF with H2 = H1 and W4 tied to W2 (see docstring). Implemented by aliasing W4 to W2 in the forward
    pass and folding the W4 gradient block into the W2 block."""

    def __init__(self, m, ref, h1, seed, feat="G"):
        super().__init__(m, ref, h1, h1, "H", seed, feat)
        self.p["W3"][:] = 0.0
        self.p["b3"][:] = 0.0
        self.keys = [k for k in self.p if k != "W4"]
        self._full = list(self.p)

    def forward(self, ou, od, ru, rd, ds):
        self.p["W4"] = self.p["W2"]
        return super().forward(ou, od, ru, rd, ds)

    def grad(self, c, coef, per_sample, rows_u, rows_d):
        self.p["W4"] = self.p["W2"]
        keys = self.keys
        self.keys = self._full
        try:
            g = super().grad(c, coef, per_sample, rows_u, rows_d)
        finally:
            self.keys = keys
        sz = {k: self.p[k].size for k in self._full}
        off, o = {}, 0
        for k in self._full:
            off[k] = o
            o += sz[k]
        sl = lambda k: slice(off[k], off[k] + sz[k])  # noqa: E731
        if per_sample:
            g[:, sl("W2")] += g[:, sl("W4")]
            return np.concatenate([g[:, sl(k)] for k in self.keys], 1)
        g[sl("W2")] += g[sl("W4")]
        return np.concatenate([g[sl(k)] for k in self.keys])

    def n_params(self):
        return int(sum(self.p[k].size for k in self.keys))

    def flat(self):
        return np.concatenate([self.p[k].ravel() for k in self.keys])

    def set_flat(self, x):
        o = 0
        for k in self.keys:
            n = self.p[k].size
            self.p[k] = x[o:o + n].reshape(self.p[k].shape)
            o += n
        self.p["W4"] = self.p["W2"]


# ============================================================================ models and arms
_MCACHE = {}


def model(k):
    if k not in _MCACHE:
        Lx, Ly, nu, nd = SIZES[k]
        m = TJ(Lx, Ly, "obc", "obc", nu, nd, 1.0, -0.25, 0.4)
        W, Vv, _ = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
        gap = float(W[1] - W[0])
        assert gap >= GAP_MIN, f"size {k} {SIZES[k]}: gap {gap:.4f} < {GAP_MIN} (counted toy sizes need a gap)"
        _MCACHE[k] = (m, Vv[:, 0], float(W[0]), uhf_orbitals(m, H21.UHF_SEED), gap)
    return _MCACHE[k]


def sra(iters):
    return SimpleNamespace(sr_iters=iters, sr_lr=0.05, sr_shift=1e-3, sr_samples=600, chunk=8192, o_chunk=4)


def foil_fn(k, seed):
    return os.path.join(OUT, f"foil_{k}_{seed}.npz")


def load_foil(k, seed):
    m, psi0, E0, ref, gap = model(k)
    nb = LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=seed)
    z = np.load(foil_fn(k, seed))
    for key in nb.keys:
        nb.p[key] = z[key].copy()
    return nb


def widen(nb, m, ref, width, seed):
    w = LRAnsatz(m, ref, "full", K=1, Hs=width, seed=seed + 1000)
    h = nb.p["W1"].shape[1]
    w.p["P0u"], w.p["P0d"] = nb.p["P0u"].copy(), nb.p["P0d"].copy()
    w.p["W1"][:, :h], w.p["b1"][:h] = nb.p["W1"], nb.p["b1"]
    w.p["b1"][h:] = 0.0
    w.p["W2"][:h], w.p["W2"][h:] = nb.p["W2"], 0.0
    w.p["b2"] = nb.p["b2"].copy()
    return w


def copy_pass1(dr, nb):
    for key in ("W1", "b1", "W2", "b2"):
        dr.p[key] = nb.p[key].copy()
    dr.p["P0u"], dr.p["P0d"] = nb.p["P0u"][0].copy(), nb.p["P0d"][0].copy()


def ws_h2(m, ref, T):
    h2 = 0
    while B.DRF(m, ref, B.WC, h2 + 1, "H", 0).n_params() <= 2 * T:
        h2 += 1
    return h2


def make(arm, k, seed):
    m, psi0, E0, ref, gap = model(k)
    nb = load_foil(k, seed)
    T = nb.n_params()
    if arm in ("ws", "wsO"):
        a = FastDRF(m, ref, B.WC, ws_h2(m, ref, T), "H", seed + 2000, "O" if arm == "wsO" else "G")
        copy_pass1(a, nb)
        a.p["W4"][:] = 0.0
    elif arm in ("dt", "dtO", "dts", "dtsO"):
        a = DRFT(m, ref, B.WC, seed + 2000, "O" if arm.endswith("O") else "G")
        copy_pass1(a, nb)
        a.p["W4"] = a.p["W2"]
    else:
        target = make("ws" if arm == "wide" else "dt", k, seed)[0].n_params()
        width = B.WC
        while LRAnsatz(m, ref, "full", K=1, Hs=width, seed=0).n_params() < target:
            width += 1
        a = widen(nb, m, ref, width, seed)
    return a, nb, T


# ============================================================================ budget ledger
def ledger_used():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(sum(r["cpu"] for r in json.load(open(LEDGER))))


def ledger_add(mode, cpu, note):
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    rows.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, cpu=cpu, note=note))
    json.dump(rows, open(LEDGER, "w"), indent=1)


def budget_gate(per_runs, args, label):
    """per_runs: list of per-step CPU-s, one per run. Returns True if the invocation may proceed."""
    proj = sum(p * args.iters * 1.05 + 5.0 for p in per_runs)
    used = time.process_time()
    prior = ledger_used()
    log(f"budget {label}: projected {proj:.0f} CPU-s for {len(per_runs)} runs + {used:.0f} used now; "
        f"invocation cap {args.cpu_cap:.0f}; task ledger {prior:.0f} of {args.task_cap:.0f}")
    if proj + used > 0.9 * args.cpu_cap:
        log("BUDGET-ABORT: projection exceeds 90% of the invocation cap; split the invocation")
        return False
    if prior + proj + used > args.task_cap:
        log("BUDGET-ABORT: projection would take the task over its CPU cap")
        return False
    return True


def time_steps(a, m, n=3):
    rng = np.random.default_rng(0)
    x0 = a.flat().copy()
    H21.sr_step(a, m, None, "none", 0, sra(1), rng)
    t0 = time.process_time()
    for _ in range(n):
        H21.sr_step(a, m, None, "none", 0, sra(1), rng)
    a.set_flat(x0)
    return (time.process_time() - t0) / n


def score(a, m, psi0, E0):
    psi = a.amplitudes(8192)
    Et = energy(m.H, psi)[0]
    s = np.where(psi >= 0, 1.0, -1.0)
    if psi0 @ psi < 0:
        s = -s
    return float(Et - E0), float(dE_sign(m.H, s, psi0, E0, 3000))


# ============================================================================ stages
def run_gaps(args):
    for k in SIZES:
        m, psi0, E0, ref, gap = model(k)
        log(f"size {k} {SIZES[k]}: V {m.V} D {m.D} E0 {E0:.6f} gap {gap:.4f} (assert >= {GAP_MIN}) OK")
    return 0


def run_foil(args):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, s) for k in args.size_list for s in args.seed_list if not os.path.exists(foil_fn(k, s))]
    if not jobs:
        log("nothing to do (all foils exist)")
        return 0
    per = {}
    for k in sorted({j[0] for j in jobs}):
        m, psi0, E0, ref, gap = model(k)
        per[k] = time_steps(LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=11), m)
    if not budget_gate([per[k] for k, _ in jobs], args, "foil"):
        return 3
    rows = []
    for k, seed in jobs:
        m, psi0, E0, ref, gap = model(k)
        c0 = time.process_time()
        nb = LRAnsatz(m, ref, "full", K=1, Hs=B.WC, seed=seed)
        H21.optimize(nb, m, None, "none", 0, sra(args.iters), np.random.default_rng(seed + 5), f"s{k} foil {seed}")
        np.savez(foil_fn(k, seed), **{key: nb.p[key] for key in nb.keys})
        dEt, dEs = score(nb, m, psi0, E0)
        row = dict(size=k, V=m.V, D=m.D, gap=gap, arm="foil", seed=seed, n_params=nb.n_params(), dEt=dEt, dEs=dEs,
                   cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"{k}_foil_{seed}.json"), "w"), indent=1)
        rows.append(row)
        log(f"size {k} foil seed {seed}: n_params {row['n_params']} dE_tot {dEt:.5f} dE_sign {dEs:.5f} "
            f"({row['cpu']:.0f} CPU-s)")
    tot = time.process_time()
    ledger_add("foil", tot, f"sizes {args.sizes} seeds {args.seeds}")
    log(f"invocation CPU total {tot:.0f} s (ledger now {ledger_used():.0f} s)")
    return 0


def run_checks(args):
    os.makedirs(OUT, exist_ok=True)
    ok = {}
    for k in args.size_list:
        m, psi0, E0, ref, gap = model(k)
        for seed in args.seed_list:
            if not os.path.exists(foil_fn(k, seed)):
                continue
            nb = load_foil(k, seed)
            p0 = nb.amplitudes(8192)
            for arm in ("ws", "wsO", "wide", "dt", "dtO", "widet"):
                a = make(arm, k, seed)[0]
                d = float(np.max(np.abs(a.amplitudes(8192) - p0)) / np.max(np.abs(p0)))
                ok[f"A_{k}_{seed}_{arm}"] = d
                log(f"CHECK A size {k} seed {seed} {arm}: n_params {a.n_params()} (foil {nb.n_params()}) "
                    f"warm-start amplitude rel diff {d:.1e}")
    # CHECK D: FastDRF equals t1b_drf.DRF (amplitudes and force) on a perturbed state, sizes 0 and 1
    for kd in (0, 1):
        m, psi0, E0, ref, gap = model(kd)
        for feat in ("G", "O"):
            a1 = B.DRF(m, ref, B.WC, 3, "H", 3, feat)
            x = a1.flat() + 0.05 * np.random.default_rng(1).standard_normal(a1.n_params())
            a1.set_flat(x)
            a2 = FastDRF(m, ref, B.WC, 3, "H", 3, feat)
            a2.set_flat(x.copy())
            r = np.random.default_rng(2).standard_normal(m.D)
            p1, p2 = a1.amplitudes(8192), a2.amplitudes(8192)
            f1, f2 = a1.force(r, 8192), a2.force(r, 8192)
            d = max(float(np.max(np.abs(p1 - p2)) / np.max(np.abs(p1))),
                    float(np.max(np.abs(f1 - f2)) / np.max(np.abs(f1))))
            ok[f"A_D_{kd}_{feat}"] = d
            log(f"CHECK D size {kd} feat {feat}: FastDRF vs DRF amplitudes and force rel diff {d:.1e}")
    # CHECK B: tied force vs finite differences on size 0
    m, psi0, E0, ref, gap = model(0)
    for feat in ("G", "O"):
        a = DRFT(m, ref, B.WC, 3, feat)
        x = a.flat().copy()
        x += 0.05 * np.random.default_rng(1).standard_normal(len(x))
        a.set_flat(x)
        r = np.random.default_rng(2).standard_normal(m.D)
        f = a.force(r, 4096)
        worst = 0.0
        for kk in np.random.default_rng(4).choice(len(x), 16, replace=False):
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
        psi = a.amplitudes(4096)
        rows = [a.force(np.eye(1, m.D, ii)[0], 4096) / psi[ii] for ii in idx]
        rel4 = float(np.max(np.abs(O - np.array(rows))) / np.max(np.abs(O)))
        ok[f"B_{feat}"], ok[f"B4_{feat}"] = worst, rel4
        log(f"CHECK B dt feat {feat}: force vs FD worst rel {worst:.2e}; per_sample_O vs force rows {rel4:.1e}")
    valid = all(v < 1e-10 for kk, v in ok.items() if kk.startswith("A_")) and ok["B_G"] < 1e-5 \
        and ok["B_O"] < 1e-5 and ok["B4_G"] < 1e-10 and ok["B4_O"] < 1e-10
    ok["valid"] = bool(valid)
    log(f"CHECKS valid: {valid}")
    json.dump(ok, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=float)
    ledger_add("checks", time.process_time(), "")
    return 0 if valid else 1


def run_train(args):
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 30))
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, arm, s) for k in args.size_list for arm in args.arm_list for s in args.seed_list
            if not os.path.exists(os.path.join(OUT, f"{k}_{arm}_{s}.json"))]
    if not jobs:
        log("nothing to do (all JSONs exist)")
        return 0
    per = {}
    for k, arm, s in jobs:
        if (k, arm) not in per:
            a = make(arm, k, s)[0]
            per[(k, arm)] = time_steps(a, model(k)[0])
    log("per-step CPU-s: " + ", ".join(f"{a}@{k}={v:.3f}" for (k, a), v in per.items()))
    if not budget_gate([per[(k, a)] for k, a, _ in jobs], args, "train"):
        ledger_add("train-abort", time.process_time(), f"sizes {args.sizes} arms {args.arms}")
        return 3
    for k, arm, seed in jobs:
        m, psi0, E0, ref, gap = model(k)
        c0 = time.process_time()
        a, nb, T = make(arm, k, seed)
        rec = sra(args.iters)
        rec.sr_shift = SHIFT_ARMS.get(arm, rec.sr_shift)
        H21.optimize(a, m, None, "none", 0, rec, np.random.default_rng(seed + 7), f"s{k} {arm} {seed}")
        dEt, dEs = score(a, m, psi0, E0)
        row = dict(size=k, V=m.V, D=m.D, gap=gap, arm=arm, seed=seed, n_params=a.n_params(), T=T,
                   H2=int(a.p["W4"].shape[0]) if arm in ("ws", "wsO", "dt", "dtO", "dts", "dtsO") else 0,
                   shift=rec.sr_shift,
                   width=int(a.p["W1"].shape[1]), dEt=dEt, dEs=dEs, cpu=time.process_time() - c0)
        json.dump(row, open(os.path.join(OUT, f"{k}_{arm}_{seed}.json"), "w"), indent=1)
        log(f"size {k} {arm} seed {seed}: n_params {row['n_params']} (foil {T}) width {row['width']} H2 {row['H2']} "
            f"dE_tot {dEt:.5f} dE_sign {dEs:.5f} ({row['cpu']:.0f} CPU-s)")
    tot = time.process_time()
    ledger_add("train", tot, f"sizes {args.sizes} arms {args.arms} seeds {args.seeds}")
    log(f"invocation CPU total {tot:.0f} s (ledger now {ledger_used():.0f} s)")
    return 0


def decide(rs):
    """rs: list of r at counted sizes in size order. Returns the rule verdict."""
    if len(rs) < 2:
        return "NO-TARGET"
    if sum(r >= 0.9 for r in rs) >= 2 and rs[-1] >= 0.9:
        return "KILL"
    if all(r <= 0.7 for r in rs) and rs[-1] <= rs[0] + 0.1:
        return "PASS-W"
    return "UNDECIDED"


def run_verdict(args):
    for want, rs in (("KILL", [1.0, 0.95]), ("PASS-W", [0.5, 0.55]), ("UNDECIDED", [0.8, 0.95]),
                     ("PASS-W", [0.6, 0.7]), ("UNDECIDED", [0.5, 0.65])):
        got = decide(rs)
        print(f"synthetic {rs}: {got} (want {want}) {'OK' if got == want else 'MISMATCH'}")
    chk = json.load(open(os.path.join(OUT, "checks.json"))) if os.path.exists(os.path.join(OUT, "checks.json")) \
        else {"valid": False}
    med = {}
    for k in SIZES:
        for arm in ("foil",) + sum(ROUTE_ARMS.values(), ()):
            v = [json.load(open(os.path.join(OUT, f"{k}_{arm}_{s}.json")))["dEs"] for s in args.seed_list
                 if os.path.exists(os.path.join(OUT, f"{k}_{arm}_{s}.json"))]
            if len(v) == len(args.seed_list):
                med[(k, arm)] = float(np.median(v))
    counted = [k for k in SIZES if med.get((k, "foil"), 0.0) >= 0.005]
    out = {}
    for route, (arm, armO, wide) in ROUTE_ARMS.items():
        rows = []
        for k in SIZES:
            if (k, arm) in med and (k, wide) in med:
                rows.append(dict(size=k, foil=med.get((k, "foil")), wide=med[(k, wide)], route=med[(k, arm)],
                                 r=med[(k, arm)] / med[(k, wide)], counted=k in counted,
                                 rO=(med[(k, arm)] / med[(k, armO)]) if (k, armO) in med else None))
        if not rows:
            continue
        for r in rows:
            print(f"  {route} size {r['size']}: dEs foil {r['foil']:.5f} {wide} {r['wide']:.5f} {arm} {r['route']:.5f}"
                  f" r {r['r']:.3f}" + (f" r_O {r['rO']:.3f}" if r["rO"] is not None else "")
                  + f" counted {r['counted']}")
        cr = [r for r in rows if r["counted"]]
        if not chk.get("valid"):
            v = "INVALID"
        elif {r["size"] for r in cr} != set(counted):   # an uncounted size decides nothing and may be absent
            v = "INCOMPLETE"
        else:
            v = decide([r["r"] for r in cr])
            if v == "PASS-W" and cr[-1]["rO"] is not None:
                v += "; attribution " + ("PASS (inverse)" if cr[-1]["rO"] <= 0.9 else "DEPTH-ONLY")
        print(f"VERDICT route {route}: {v}")
        out[route] = dict(rows=rows, verdict=v)
    print(f"task CPU ledger: {ledger_used():.0f} s")
    json.dump(out, open(os.path.join(OUT, "verdict.json"), "w"), indent=1, default=float)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r8 T1 of routes 8.3 and 8.4 (see docstring)")
    ap.add_argument("--mode", choices=["gaps", "foil", "checks", "train", "verdict"], required=True)
    ap.add_argument("--sizes", default="0,1,2")
    ap.add_argument("--arms", default="ws,wide")
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu_cap", type=float, default=540.0)
    ap.add_argument("--task_cap", type=float, default=1800.0)
    a = ap.parse_args(argv)
    a.size_list = [int(s) for s in a.sizes.split(",")]
    a.arm_list = a.arms.split(",")
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    assert all(s in SIZES for s in a.size_list)
    assert all(x in ("ws", "wsO", "wide", "dt", "dtO", "dts", "dtsO", "widet", "widets") for x in a.arm_list)
    return a


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 30))
    log(f"threads {NTHREADS}")
    return {"gaps": run_gaps, "foil": run_foil, "checks": run_checks, "train": run_train,
            "verdict": run_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
