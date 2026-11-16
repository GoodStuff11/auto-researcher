#!/usr/bin/env python3
"""
P2.1 Round 11b: sublattice-hidden-row falsifier of the Saturation Diagnostic (cost only).
=========================================================================================

Rule pre-registered in the leaf artifact ("Round 11b ... Pre-registration"), before any run.

Route variant: local-feature HFPS (Route 7.1, pairing frame, plus patch, slaved accepted-move update of
accept_update.HFPSAcc) with hidden rows only on a sublattice S_h of V_h < V sites:
    M = [[ F0[up, dn],     K[up, :]  ],        K: V x V_h,  C[m, :] = h_m . Omega_m  (Omega_m: F x V),
         [ -C(n)[:, dn],   I_{V_h}   ]]        h_m = tanh(W x_patch(m) + b),  m in S_h,
    psi = det M = det(F0 + K C(n))[up, dn]    (Schur complement; used as the explicit reference).
The augmented inverse has n = N + V_h; a move changes only the hidden rows of S_h whose patch contains a
touched site. Implementation: subclass of accept_update.HFPSAcc (imported read-only) in which the inherited
attribute V means the hidden-row count V_h; the physical site count is self.Vphys. Every inherited use of
self.V (augmented size, identity block, zero K-change block) is a hidden-row dimension, so the inherited
ratio / accept / fresh code is reused unchanged.

Sublattices (--sub): full (all sites, v_h = 1; reproduces t1_r7.LHHFPS at the same seed), half
(checkerboard, x + y even, v_h = 1/2), quarter (x and y both even, v_h = 1/4; with the plus patch the
(odd, odd) sites lie in no patch).

Modes
  --mode check      T0 gates on 5x3 OBC, 6x6 and 8x8 PBC for half and quarter:
                    (i) augmented-det amplitude == explicit det(F0 + K C)[up, dn] (site-by-site loop),
                        and full == t1_r7.LHHFPS at equal seed; rel. err <= 1e-12
                    (ii) fast E_loc == brute E_loc (sorted rows, explicit JW signs, explicit amplitude),
                        rel. err <= 1e-12, after random accepted moves
                    (iii) drift <= 1e-8 after --accepts accepted moves without refresh (checkpoints
                        every --check), metric of accept_update.compare.
                    Prints PASS or BUG; exit 2 on BUG.
  --mode run        --arm {half,quarter,full,dense,nnbf} [--H]: c = t_sw * tau_int at --sizes with CPU
                    --budgets per size (8b sweep / proposal / tau_int imported read-only); dense =
                    cost_8c.DenseHFPS (F = 4), nnbf = cost_per_sample.NNBF (width --H).
                    Writes results/<arm>[_H<H>].json.
  --mode aggregate  table, R_D(L) = c_dense / c_variant with 1-sigma error, R_N (reported only), and the
                    pre-registered verdict (REOPEN / not resolved / Diagnostic stands / INCOMPLETE).
Options: --sizes (8,12,16)  --budgets (15,25,120)  --H (6)  --F (4)  --seed (11)
         --accepts (10000)  --check (1000)
Threads: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, one process per arm, `ulimit -t 200`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("P2.1_subcubic_amplitudes_r10", "P2.1_subcubic_amplitudes_r8b", "P2.1_subcubic_amplitudes_r7"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import t1_r7 as T1  # noqa: E402  (read-only)
import accept_update as AU  # noqa: E402  (read-only)
import cost_per_sample as CP  # noqa: E402  (read-only)
import cost_8c as C8  # noqa: E402  (read-only)
from lh_geom import geometry, patch_index  # noqa: E402

OUT = os.path.join(HERE, "results")
TOL_EXACT, TOL_DRIFT = 1e-12, 1e-8


def hidden_sites(Lx, Ly, sub):
    xs, ys = np.arange(Lx * Ly) % Lx, np.arange(Lx * Ly) // Lx
    if sub == "full":
        m = np.ones(Lx * Ly, bool)
    elif sub == "half":
        m = (xs + ys) % 2 == 0
    elif sub == "quarter":
        m = (xs % 2 == 0) & (ys % 2 == 0)
    else:
        raise ValueError(sub)
    return np.nonzero(m)[0]


class SubHFPS(AU.HFPSAcc):
    def __init__(self, T, Lx, Ly, bc, n, F, patch, sub, rng):
        V = T.shape[0]
        hs = hidden_sites(Lx, Ly, sub)
        Vh = len(hs)
        self.n, self.F, self.hs, self.Vphys = n, F, hs, V
        e, v = np.linalg.eigh(-T)
        # same draw order and scales as t1_r7.LHHFPS (so sub = full reproduces it at equal seed)
        self.F0 = v[:, :n] @ v[:, :n].T + 0.1 * rng.standard_normal((V, V)) / np.sqrt(V)
        P = patch_index(Lx, Ly, bc, patch)[hs]
        self.loc = T1.Local(P, F, rng)
        self.loc.inv = [np.nonzero((P == s).any(1))[0] for s in range(V)]   # physical site -> hidden rows
        self.K = 0.5 * rng.standard_normal((V, Vh)) / np.sqrt(V)
        self.Om = 0.5 * rng.standard_normal((Vh, F, V)) / np.sqrt(F * V)
        self.V = Vh            # inherited code uses self.V as the hidden-row dimension
        self.ops = 0

    def amp_explicit(self, ou, od):
        """Reference amplitude: det(F0 + K C)[sorted up, sorted dn], features by an explicit site loop."""
        P, W, b = self.loc.P, self.loc.W, self.loc.b
        p = P.shape[1]
        C = np.zeros((len(self.hs), self.Vphys))
        for m in range(len(self.hs)):
            x = np.zeros(2 * p)
            for a in range(p):
                s = P[m, a]
                if s >= 0:
                    x[a] = 2.0 * ou[s] - 1.0
                    x[p + a] = 2.0 * od[s] - 1.0
            h = np.tanh(x @ W + b)
            for f in range(self.F):
                C[m] += h[f] * self.Om[m, f]
        A = self.F0 + self.K @ C
        return np.linalg.det(A[np.ix_(np.nonzero(ou)[0], np.nonzero(od)[0])])


# ------------------------------------------------------------------ T0 gates
def parse_geom(g):
    if g == "5x3obc":
        return 5, 3, "obc", 6
    L = int(g)
    return L, L, "pbc", T1.filling(L * L)


def eloc_brute_explicit(obj, T, nn):
    return T1.eloc_brute(obj.amp_explicit, T, nn, obj.ou, obj.od)


def check(args):
    ok = True
    print("== (i) amplitude: augmented det vs explicit det(F0 + K C); full vs t1_r7.LHHFPS", flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = parse_geom(g)
        T, nn = geometry(Lx, Ly, bc, T1.TP)
        a = SubHFPS(T, Lx, Ly, bc, n, args.F, "plus", "full", np.random.default_rng(5))
        ref = T1.LHHFPS(T, Lx, Ly, bc, n, args.F, "plus", np.random.default_rng(5))
        rng = np.random.default_rng(9)
        e_full = max(abs(a.amp(*c) - ref.amp(*c)) / abs(ref.amp(*c))
                     for c in [T1.rand_cfg(Lx * Ly, n, rng) for _ in range(3)])
        ok &= e_full <= TOL_EXACT
        print(f"   {g}: full vs LHHFPS max rel err {e_full:.1e}")
        for sub in ("half", "quarter"):
            o = SubHFPS(T, Lx, Ly, bc, n, args.F, "plus", sub, np.random.default_rng(6))
            cov = len(np.unique(o.loc.P[o.loc.P >= 0])) / (Lx * Ly)
            em = 0.0
            for _ in range(4):
                c = T1.rand_cfg(Lx * Ly, n, rng)
                ex = o.amp_explicit(*c)
                em = max(em, abs(o.amp(*c) - ex) / abs(ex))
            ok &= em <= TOL_EXACT
            print(f"   {g} {sub:7s}: V_h={len(o.hs)} n_aug={n + len(o.hs)} patch coverage={cov:.2f} "
                  f"max rel err {em:.1e}")
    print("== (ii) fast E_loc vs brute (explicit amplitude, sorted rows, JW signs), after accepted moves",
          flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = parse_geom(g)
        T, nn = geometry(Lx, Ly, bc, T1.TP)
        for sub in ("half", "quarter"):
            rng = np.random.default_rng(13)
            o = SubHFPS(T, Lx, Ly, bc, n, args.F, "plus", sub, rng)
            o.setup(*T1.rand_cfg(Lx * Ly, n, rng))
            em = 0.0
            for k in range(3):
                acc = 0
                while acc < 20:
                    a, _ = AU.mc_step(o, T, nn, rng)
                    acc += a
                Ef, _ = AU.eloc(o, T, nn)
                Eb = eloc_brute_explicit(o, T, nn)
                em = max(em, abs(Ef - Eb) / (1 + abs(Eb)))
            ok &= em <= TOL_EXACT
            print(f"   {g} {sub:7s}: E_fast={Ef:+.12f} E_brute={Eb:+.12f} max rel err {em:.1e}", flush=True)
    print(f"== (iii) drift over {args.accepts} accepted moves, no refresh (checkpoint every {args.check})",
          flush=True)
    for g in ("5x3obc", "6", "8"):
        Lx, Ly, bc, n = parse_geom(g)
        T, nn = geometry(Lx, Ly, bc, T1.TP)
        for sub in ("half", "quarter"):
            rng = np.random.default_rng(17)
            o = SubHFPS(T, Lx, Ly, bc, n, args.F, "plus", sub, rng)
            o.setup(*T1.rand_cfg(Lx * Ly, n, rng))
            acc = prop = 0
            dmax = 0.0
            t0 = time.perf_counter()
            while acc < args.accepts:
                a, _ = AU.mc_step(o, T, nn, rng)
                prop += 1
                acc += a
                if a and acc % args.check == 0:
                    et, eE, _ = AU.compare(o, T, nn)
                    dmax = max(dmax, et, eE)
            ok &= dmax <= TOL_DRIFT
            print(f"   {g} {sub:7s}: accepts={acc} acceptance={acc / prop:.2f} max drift={dmax:.1e} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print("CHECK", "PASS" if ok else "BUG")
    return 0 if ok else 2


# ------------------------------------------------------------------ cost
def make(arm, T, L, n, args, rng):
    if arm == "dense":
        return C8.DenseHFPS(T, n, args.F, rng)
    if arm == "nnbf":
        return CP.NNBF(T, n, args.H, rng)
    return SubHFPS(T, L, L, "pbc", n, args.F, "plus", arm, rng)


def fname(arm, H):
    return f"{arm}_H{H}" if arm == "nnbf" else arm


def run(args):
    out = {"arm": args.arm, "H": args.H, "sizes": {}}
    for L, bud in zip(args.size_list, args.budget_list):
        rng = np.random.default_rng(args.seed + L)
        T, nn = geometry(L, L, "pbc", T1.TP)
        n = T1.filling(L * L)
        nbr = CP.neighbour_table(T); nnset = set(nn)
        obj = make(args.arm, T, L, n, args, rng)
        obj.setup(*T1.rand_cfg(L * L, n, rng))
        batched = args.arm in ("dense", "nnbf")
        c0 = time.process_time()
        Es, tsw, npr_t, nac_t = [], [], 0, 0
        while time.process_time() - c0 < bud:
            w0 = time.perf_counter()
            E, npr, nac = CP.sweep(obj, T, nn, nbr, nnset, rng, batched)
            tsw.append(time.perf_counter() - w0)
            Es.append(E); npr_t += npr; nac_t += nac
        ns = len(Es); burn = max(10, ns // 10)
        Ek = np.array(Es[burn:])
        tau, terr, W = CP.tau_int(Ek) if len(Ek) > 20 else (float("nan"), float("nan"), 0)
        t = float(np.mean(tsw[burn:])) if ns > burn else float(np.mean(tsw))
        Vh = len(obj.hs) if hasattr(obj, "hs") else None
        out["sizes"][str(L)] = dict(L=L, Ne=2 * n, V_h=Vh, sweeps=ns, t_sw=t, tau=tau, tau_err=terr,
                                    cost=t * tau, acceptance=nac_t / max(npr_t, 1),
                                    cpu_s=time.process_time() - c0)
        print(f"[{fname(args.arm, args.H)}] L={L:2d} Ne={2 * n:3d} V_h={Vh} sweeps={ns:4d} t_sw={t:.3e}s "
              f"tau={tau:.2f}+-{terr:.2f} cost={t * tau:.3e}s acc={nac_t / max(npr_t, 1):.2f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, fname(args.arm, args.H) + ".json"), "w"), indent=1)


def ratio(a, b):
    R = a["cost"] / b["cost"]
    return R, R * np.hypot(a["tau_err"] / a["tau"], b["tau_err"] / b["tau"]), a["t_sw"] / b["t_sw"]


def aggregate(args):
    R = {f[:-5]: json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")}
    for k, r in R.items():
        print(f"{k:9s} " + "  ".join(f"L={L}: t_sw={s['t_sw']:.3e} tau={s['tau']:.2f}+-{s['tau_err']:.2f} "
                                     f"sweeps={s['sweeps']}" for L, s in r["sizes"].items()))
    if "dense" not in R:
        print("INCOMPLETE: no dense arm"); return
    reopen = []
    for v in ("half", "quarter"):
        if v not in R:
            print(f"{v}: INCOMPLETE"); continue
        rs = {L: ratio(R["dense"]["sizes"][L], R[v]["sizes"][L]) for L in R[v]["sizes"]}
        print(f"R_D[{v}] = c_dense / c_{v}: " + "; ".join(
            f"L={L}: {Rv:.2f}+-{e:.2f} (t_sw ratio {rt:.2f})" for L, (Rv, e, rt) in rs.items()))
        R16, e16, _ = rs["16"]
        if not np.isfinite(R16) or not np.isfinite(e16):
            verdict = "INCOMPLETE"
        elif R16 - e16 >= 3:
            verdict = "REOPEN"; reopen.append(v)
        elif R16 >= 3:
            verdict = "not resolved (central >= 3, 1-sigma interval reaches below 3) -> Diagnostic stands"
        else:
            verdict = "R_D(16) < 3 -> Diagnostic stands"
        print(f"   {v}: {verdict}")
        for k in R:
            if k.startswith("nnbf"):
                rn = {L: ratio(R[k]["sizes"][L], R[v]["sizes"][L]) for L in R[v]["sizes"]}
                print(f"   R_N[{k} / {v}] (reported only): " + "; ".join(
                    f"L={L}: {Rv:.2f}+-{e:.2f} (t_sw {rt:.2f})" for L, (Rv, e, rt) in rn.items()))
    print("VERDICT:", f"REOPEN P2.1 with {', '.join(reopen)}" if reopen else "Saturation Diagnostic stands")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "run", "aggregate"], default="check")
    ap.add_argument("--arm", choices=["half", "quarter", "full", "dense", "nnbf"], default="half")
    ap.add_argument("--sizes", default="8,12,16")
    ap.add_argument("--budgets", default="15,25,120")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--F", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--accepts", type=int, default=10000)
    ap.add_argument("--check", type=int, default=1000)
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
