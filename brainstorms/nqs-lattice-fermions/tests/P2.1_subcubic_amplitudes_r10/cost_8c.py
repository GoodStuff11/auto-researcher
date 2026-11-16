#!/usr/bin/env python3
"""
P2.1 Round 8c: cost per decorrelated sample R(L, H) of Route 7.1 (LH-HFPS) against NNBF at widths
H = 6, 32, 64, 128 AND against the cubic pairing-frame baseline (dense HFPS, the T3 arm hfps_dense).
The rule is pre-registered in the leaf artifact ("Round 8c ... pre-registered" and its extension in
"Round 10"). The round-8b harness (../P2.1_subcubic_amplitudes_r8b/cost_per_sample.py: proposal, sweep,
tau_int, NNBF engine, route engine) is imported read-only and used unchanged.
=====================================================================================================

Metric: c = t_sw * tau_int (wall time per sweep of 4V attempts incl. one local energy, times the
integrated autocorrelation time of E_loc in sweeps). R_N(L, H) = c_NNBF(H) / c_route,
R_D(L) = c_dense / c_route.

Arms (one per process; Jastrow factors omitted in every arm, as in round 8b):
  hfps    route 7.1 engine (accept_update.HFPSAcc, slaved form), plus patch, F = 4
  dense   dense HFPS (pairing-frame NNBF), F = 4, best implementation of the cubic baseline:
          A(n) = F0[ru, rd] + sum_f g_f(n) (K Omega_f)[ru, rd],  g = tanh(W x + b) of the whole
          configuration; K Omega_f (V x V, f = 1..F) precomputed once, so a ratio costs F N^2 (gather)
          plus one N x N determinant (batched slogdet, 64 at a time, like the NNBF engine). Same F0, K,
          Omega scales as the route engine (t1_r7.LHHFPS), W ~ N(0, 1 / 2V).
  nnbf    NNBF engine of round 8b at width --H.

Modes
  --mode check   dense engine: fast slot-ordered batched E_loc == t1_r7.eloc_brute (sorted rows, explicit
                 Jordan-Wigner signs) on 4x4 and 6x6 PBC after random accepted moves; exit 2 on failure.
  --mode run     --arm {hfps,dense,nnbf} [--H] : write results_8c/<arm>[_H<H>].json
  --mode aggregate  table, R_N(L, H), R_D(L) and the pre-registered verdicts.
Options: --sizes (8,12,16) --budgets (CPU seconds per size; default 30,50,110) --H (6) --F (4) --seed (11).
Threads: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, one process per arm, `ulimit -t 200`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r8b"))
sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r7"))
import cost_per_sample as CP  # noqa: E402  (read-only)
import t1_r7 as T1  # noqa: E402
from lh_geom import geometry  # noqa: E402

OUT = os.path.join(HERE, "results_8c")


class DenseHFPS(CP.NNBF):
    def __init__(self, T, n, F, rng):
        V = T.shape[0]
        self.V, self.n, self.H = V, n, F
        e, v = np.linalg.eigh(-T)
        self.F0 = v[:, :n] @ v[:, :n].T + 0.1 * rng.standard_normal((V, V)) / np.sqrt(V)
        K = 0.5 * rng.standard_normal((V, V)) / np.sqrt(V)
        Om = 0.5 * rng.standard_normal((V, F, V)) / np.sqrt(F * V)
        self.KO = np.einsum("mu,ufv->fmv", K, Om)          # (F, V, V): (K Omega_f)
        self.W1 = rng.standard_normal((2 * V, F)) / np.sqrt(2 * V)
        self.b1 = np.zeros(F)
        self.ops = 0

    def logdet_pair(self, g, ru, rd):
        A = self.F0[ru[:, :, None], rd[:, None, :]] + np.einsum(
            "cf,fcij->cij", g, self.KO[:, ru[:, :, None], rd[:, None, :]])
        N = ru.shape[1]
        self.ops += ru.shape[0] * (self.H * N * N + 2 * N ** 3 // 3)
        return np.linalg.slogdet(A)

    def amp_sorted(self, ou, od):
        g = np.tanh(self._x(ou, od) @ self.W1 + self.b1)[None]
        sg, la = self.logdet_pair(g, np.nonzero(ou)[0][None], np.nonzero(od)[0][None])
        return sg[0] * np.exp(la[0])

    def setup(self, ou, od):
        self.ou, self.od = ou.copy(), od.copy()
        self.a = self._x(ou, od) @ self.W1 + self.b1
        self.rows = [np.nonzero(ou)[0].copy(), np.nonzero(od)[0].copy()]
        self.slot = [{int(x): i for i, x in enumerate(r)} for r in self.rows]
        sg, la = self.logdet_pair(np.tanh(self.a)[None], self.rows[0][None], self.rows[1][None])
        self.sg, self.la = sg[0], la[0]

    def eval_batch(self, mlist):
        C = len(mlist)
        g = np.tanh(self.a[None] + np.array([self._da(m) for m in mlist]))
        rows = [np.empty((C, self.n), dtype=np.int64) for _ in (0, 1)]
        for c, m in enumerate(mlist):
            for s in (0, 1):
                r = self.rows[s].copy()
                for (ss, x, y) in m:
                    if ss == s:
                        r[self.slot[s][x]] = y
                rows[s][c] = r
        sgn = np.empty(C); logr = np.empty(C)
        for i0 in range(0, C, 64):
            sg, la = self.logdet_pair(g[i0:i0 + 64], rows[0][i0:i0 + 64], rows[1][i0:i0 + 64])
            sgn[i0:i0 + 64] = sg * self.sg
            logr[i0:i0 + 64] = la - self.la
            if C == 1:
                self._new = (sg[0], la[0])
        return sgn * np.exp(logr), g

    def ratio(self, moves):
        r, _ = self.eval_batch([moves])
        self._last = moves
        return r[0]

    def accept(self, moves):
        assert self._last is moves
        self.a = self.a + self._da(moves)
        rows = self._new_rows(moves)
        for (s, x, y) in moves:
            i = self.slot[s].pop(x)
            self.slot[s][y] = i
        self.rows = rows
        self.sg, self.la = self._new
        self.ou, self.od = T1.apply_moves(self.ou, self.od, moves)
        return 0


def make(arm, T, L, n, args, rng):
    if arm == "dense":
        return DenseHFPS(T, n, args.F, rng)
    if arm == "nnbf":
        return CP.NNBF(T, n, args.H, rng)
    import accept_update as AU
    return AU.HFPSAcc(T, L, L, "pbc", n, 4, "plus", rng)


def name(args):
    return {"hfps": "7.1 LH-HFPS", "dense": f"dense HFPS F={args.F}", "nnbf": f"NNBF H={args.H}"}[args.arm]


def fname(arm, H):
    return f"{arm}_H{H}.json" if arm == "nnbf" else f"{arm}.json"


def run(args):
    out = {"arm": args.arm, "name": name(args), "H": args.H, "sizes": {}}
    for L, bud in zip(args.size_list, args.budget_list):
        rng = np.random.default_rng(args.seed + L)
        T, nn = geometry(L, L, "pbc", T1.TP)
        n = T1.filling(L * L)
        nbr = CP.neighbour_table(T); nnset = set(nn)
        obj = make(args.arm, T, L, n, args, rng)
        ou, od = T1.rand_cfg(L * L, n, rng)
        obj.setup(ou, od)
        c0 = time.process_time()
        Es, tsw, npr_t, nac_t = [], [], 0, 0
        while time.process_time() - c0 < bud:
            w0 = time.perf_counter()
            E, npr, nac = CP.sweep(obj, T, nn, nbr, nnset, rng, args.arm != "hfps")
            tsw.append(time.perf_counter() - w0)
            Es.append(E); npr_t += npr; nac_t += nac
        ns = len(Es); burn = max(10, ns // 10)
        Ek = np.array(Es[burn:])
        tau, terr, W = CP.tau_int(Ek) if len(Ek) > 20 else (np.nan, np.nan, 0)
        t = float(np.mean(tsw[burn:])) if ns > burn else float(np.mean(tsw))
        out["sizes"][str(L)] = dict(L=L, Ne=2 * n, sweeps=ns, t_sw=t, tau=tau, tau_err=terr, cost=t * tau,
                                    acceptance=nac_t / max(npr_t, 1), cpu_s=time.process_time() - c0)
        print(f"[{name(args)}] L={L:2d} Ne={2 * n:3d} sweeps={ns:4d} t_sw={t:.3e}s tau={tau:.2f}+-{terr:.2f} "
              f"cost={t * tau:.3e}s acc={nac_t / max(npr_t, 1):.2f} <E>/Ne={Ek.mean() / (2 * n):+.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, fname(args.arm, args.H)), "w"), indent=1)


def check(args):
    ok = True
    for L in (4, 6):
        rng = np.random.default_rng(3 + L)
        T, nn = geometry(L, L, "pbc", T1.TP)
        n = T1.filling(L * L)
        obj = DenseHFPS(T, n, args.F, rng)
        nbr = CP.neighbour_table(T); nnset = set(nn)
        obj.setup(*T1.rand_cfg(L * L, n, rng))
        for k in range(5):
            for _ in range(5 * L * L):
                mv = CP.propose(obj.ou, obj.od, nbr, nnset, rng)
                if mv is not None and rng.random() < min(1.0, obj.ratio(mv) ** 2):
                    obj.accept(mv)
            Ef = CP.eloc_nnbf(obj, T, nn)
            Eb = T1.eloc_brute(obj.amp_sorted, T, nn, obj.ou, obj.od)
            err = abs(Ef - Eb) / (1 + abs(Eb))
            ok &= err < 1e-9
            print(f"check dense L={L} cfg {k}: E_loc fast={Ef:+.10f} brute={Eb:+.10f} rel.err={err:.1e}")
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def verdict(rs):
    R16, e16 = rs[-1][1], rs[-1][2]
    grows = all(rs[i + 1][1] > rs[i][1] for i in range(len(rs) - 1))
    v = "KEEP-FOR-T3" if (R16 >= 3 and grows) else ("DOWNGRADE" if R16 >= 1 else "KILL-COST")
    unc = any(R16 - e16 < th < R16 + e16 for th in (1.0, 3.0))
    return v + (" (uncertain)" if unc else "")


def aggregate(args):
    R = {f[:-5]: json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")}
    if "hfps" not in R:
        print("INCOMPLETE: no route result"); return
    sizes = sorted(int(s) for s in R["hfps"]["sizes"])
    for k, r in R.items():
        print(f"{r['name']:16s} " + "  ".join(
            f"L={L}: t_sw={r['sizes'][str(L)]['t_sw']:.3e} tau={r['sizes'][str(L)]['tau']:.2f}"
            f"+-{r['sizes'][str(L)]['tau_err']:.2f} sweeps={r['sizes'][str(L)]['sweeps']}" for L in sizes))
    ro = R["hfps"]["sizes"]
    for k, r in R.items():
        if k == "hfps":
            continue
        rs = []
        for L in sizes:
            a, b = r["sizes"][str(L)], ro[str(L)]
            Rv = a["cost"] / b["cost"]
            rel = np.hypot(a["tau_err"] / a["tau"], b["tau_err"] / b["tau"])
            rs.append((L, Rv, Rv * rel, a["t_sw"] / b["t_sw"]))
        txt = "; ".join(f"L={L}: R={Rv:.2f}+-{e:.2f} (t_sw ratio {rt:.2f})" for (L, Rv, e, rt) in rs)
        print(f"R[{r['name']} / route]: {txt} -> {verdict(rs)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run", "check", "aggregate"], default="run")
    ap.add_argument("--arm", choices=["hfps", "dense", "nnbf"], default="dense")
    ap.add_argument("--sizes", default="8,12,16")
    ap.add_argument("--budgets", default="30,50,110")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--F", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.budget_list = [float(s) for s in args.budgets.split(",")]
    if args.mode == "check":
        sys.exit(check(args))
    if args.mode == "aggregate":
        aggregate(args); return
    run(args)


if __name__ == "__main__":
    main()
