#!/usr/bin/env python3
"""
P2.1 Round 12c: cost gate for the `half` sublattice variant at larger hidden width F (cost only).
==================================================================================================
Rule pre-registered in the leaf artifact ("Round 12c ... Pre-registration"), before any timed run.

Engine and metric: the Round 11c carried-inverse compiled engine and its metric, imported read-only
(../P2.1_subcubic_amplitudes_r11c/delayed.py). Nothing in r11c, r12 or r12b is edited. The only change
is the hidden width F of the `half` route arm (delayed.Route already takes F; 11c used F = 4).
Arms:
  half_F<F>  `half` (checkerboard hidden rows, v_h = 1/2, plus patch) at hidden width F, rmax = 16
             (the 11c choice for half, not rescanned), parameters from sub_hfps.SubHFPS at seed 11 + L
             (half_F4 is the 11c `half` arm, same parameters)
  dense2     dense HFPS at F = 4 (the 11c scored baseline, same seed and wave function), re-measured
Modes
  --mode params   parameter counts of half_F (L = 12, 16 engine objects; toy SubHFPSAnsatz on the ladder
                  clusters V = 8, 12, 14, 15) and of the F = 4 references (full, dense). No timing.
  --mode gate     T0 gates for half_F8 and half_F16 on 5x3 OBC and 6x6 PBC: per-configuration normalised
                  ratio error <= 1e-12 (11c amendment A1), compiled E_loc vs t1_r7.eloc_brute <= 1e-12 rel,
                  drift <= 1e-8 after --accepts accepted moves without refresh. Exit 2 on BUG.
  --mode ops      op-count prediction: move statistics (b, k, down moves, pending rank r, connections)
                  sampled by running the compiled half engine move by move from Python (--ops-sweeps
                  sweeps after --ops-burn compiled sweeps), then multiply-add counts per sweep for each F
                  and for dense2 (F = 4). Prints R_ops(L, F) and the predicted R_D(16, F) =
                  R_D^{11c}(16) * ops_half(4) / ops_half(F). No timing is scored.
  --mode run      --arm {half_F4, half_F8, half_F12, half_F16, dense2} --wave {A,B}: delayed.run unchanged
                  (c = t_sw * tau_int, Sokal tau, 4V attempts per sweep) at --sizes with CPU --budgets;
                  writes results/<wave>_<arm>.json.
  --mode aggregate  R_D(L) = c_dense2 / c_half_F against the dense2 run of the same wave, 1 sigma from the two
                  tau errors, and the pre-registered verdict (GATE-PASS at the largest F > 4 with
                  R_D(16) - sigma >= 3; GATE-FAIL if none; INCOMPLETE on a NaN tau at L = 16).
Options: --sizes (12,16) --budgets (20,90) --rmax (16) --seed (11) --accepts (2000) --ops-sweeps (3)
         --ops-burn (5) --Fs (4,8,12,16)
Threads: OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1, one process per arm, at most 4 at a time, `ulimit -t`
per process (run_all.sh). Use NUMBA_CACHE_DIR outside the repository.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r11c"))
import delayed as D  # noqa: E402  (read-only; its imports set up r11b, r10, r8b, r7)

OUT = os.path.join(HERE, "results")
RD11C_16 = (5.05, 1.27)   # 11c half R_D(16) vs dense2 (aggregate_output.txt)
P_PLUS = 5                # plus patch size

_orig_make = D.make


def make_f(arm, T, Lx, Ly, bc, n, rmax, seed):
    if arm.startswith("half_F"):
        return D.Route("half", T, Lx, Ly, bc, n, rmax, seed, F=int(arm[6:]))
    return _orig_make(arm, T, Lx, Ly, bc, n, rmax, seed)


D.make = make_f   # in-memory only, so delayed.run / warm build the F-variants


# ------------------------------------------------------------------ params
def params(args):
    print("== engine objects (sub_hfps.SubHFPS / cost_8c.DenseHFPS) at L x L PBC")
    for L in (12, 16):
        lat = D.lattice(L, L, "pbc")
        n = D.T1.filling(L * L)
        for F in args.F_list:
            o = make_f(f"half_F{F}", lat[0], L, L, "pbc", n, 16, 1).py
            cnt = o.F0.size + o.K.size + o.Om.size + o.loc.W.size + o.loc.b.size
            print(f"L={L} half_F{F:<2d} V_h={len(o.hs)} params={cnt}  (F0 {o.F0.size}, K {o.K.size}, "
                  f"Om {o.Om.size}, W+b {o.loc.W.size + o.loc.b.size})")
        o = D.Route("full", lat[0], L, L, "pbc", n, 16, 1, F=4).py
        print(f"L={L} full_F4   params={o.F0.size + o.K.size + o.Om.size + o.loc.W.size + o.loc.b.size}")
        d = D.Dense2(lat[0], n, 1).py
        cnt = sum(np.asarray(v).size for k, v in vars(d).items()
                  if isinstance(v, np.ndarray) and v.dtype.kind == "f")
        print(f"L={L} dense2_F4 float arrays held by DenseHFPS: {cnt}")
    print("== toy SubHFPSAnsatz (t1_sub, read-only) n_eff on the ladder clusters (masked entries excluded)")
    sys.path.insert(0, os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12"))
    import t1_sub as TS  # noqa: E402  (read-only)
    import micro_test as mt  # noqa: E402
    mt.INSTANCES.update({24: dict(Lx=7, Ly=2, bcx="obc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4),
                         25: dict(Lx=5, Ly=3, bcx="obc", bcy="obc", nup=7, ndn=7, tp=-0.25, J=0.4)})
    for inst in (20, 23, 24, 25):
        m = mt.make_model(inst)
        F0 = np.zeros((m.V, m.V))
        row = []
        for F in args.F_list:
            row.append(f"half_F{F}={TS.SubHFPSAnsatz(m, F0, F, 0, 'half', seed=1).n_eff()}")
        row.append(f"full_F4={TS.SubHFPSAnsatz(m, F0, 4, 0, 'full', seed=1).n_eff()}")
        print(f"inst {inst} V={m.V}: " + "  ".join(row))


# ------------------------------------------------------------------ gate
def gate(args):
    ok = True
    for g in ("5x3obc", "6"):
        Lx, Ly, bc, n = D.parse_geom(g)
        lat = D.lattice(Lx, Ly, bc)
        for F in (8, 16):
            obj = make_f(f"half_F{F}", lat[0], Lx, Ly, bc, n, args.rmax, 21)
            obj.setup(*D.T1.rand_cfg(Lx * Ly, n, obj.rng))
            D.seed_nb(5)
            en = eE = 0.0
            for rep in range(3):
                D.do_sweep(obj, lat, 4 * Lx * Ly, False)
                mv, ref = D.brute_ratios(obj, lat)
                fast = np.array([D.ratio_c(obj, tuple(int(x) for x in m)) for m in mv])
                en = max(en, float(np.max(np.abs(fast - ref)) / np.max(np.abs(ref))))
                Eb = D.T1.eloc_brute(obj.py.amp_explicit, lat[0], lat[1], obj.ou, obj.od)
                Ec = D.eloc_c(obj, lat)
                eE = max(eE, abs(Ec - Eb) / abs(Eb))
            # drift
            nacc = 0
            while nacc < args.accepts:
                _, _, a = D.do_sweep(obj, lat, 4 * Lx * Ly, False)
                nacc += a
            Mi = obj.Mi()
            E1 = D.eloc_c(obj, lat)
            C1, H1 = obj.C.copy(), obj.H.copy()
            M, C, H = obj.build(obj.ou, obj.od, obj.ru, obj.rd)
            dMi = D.nerr(Mi, np.linalg.inv(M))
            dC = max(D.nerr(C1, C), D.nerr(H1, H))
            obj.setup(obj.ou, obj.od, obj.ru, obj.rd)
            dE = abs(D.eloc_c(obj, lat) - E1) / abs(E1)
            g_ok = en <= D.TOL_EXACT and eE <= D.TOL_EXACT and max(dMi, dC, dE) <= D.TOL_DRIFT
            ok &= g_ok
            print(f"[gate] {g:7s} half_F{F:<2d}: ratio nerr={en:.2e}  E_loc rel={eE:.2e}  drift after {nacc} "
                  f"accepts: Mi {dMi:.2e} C,H {dC:.2e} E {dE:.2e} -> {'PASS' if g_ok else 'BUG'}", flush=True)
    print("GATES:", "PASS" if ok else "BUG")
    return 0 if ok else 2


# ------------------------------------------------------------------ ops
def route_ops(st, F, N, Vh, V):
    """Multiply-adds per sweep of the carried-inverse half engine from sampled move statistics."""
    n = N + Vh
    tot = 0.0
    for (b, k, dn, r) in st["ratios"]:
        o = b * F * 2 * P_PLUS            # new features of the b changed hidden rows
        o += b * F * N                    # D rows of the changed hidden rows (the bFN term)
        o += r * n * k                    # low-rank correction of Mi[:, R]
        o += k * n * k                    # S = D Mi[:, R]
        if dn:
            o += n * n + 2 * r * n + k * n   # y = Mi ct (Mi0 ct and low rank), D y
        tot += o
    for (b, k, dn, r) in st["accepts"]:
        kk = k + (1 if dn else 0)
        o = k * n * n + 2 * k * r * n     # D Mi0 and its low-rank correction
        o += (r * n if dn else 0) + kk * kk * n   # Mi row j correction, solve
        o += kk * n * n                   # amortised flush (n^2 r per flush every ~r / kk accepts)
        o += b * F * V                    # C rows update
        tot += o
    return tot / st["sweeps"]


def dense_ops(st, F, N):
    per_ratio = (F + 1) * N * N + N ** 3 / 3.0
    return (len(st["ratios"]) * per_ratio + len(st["accepts"]) * 2 * N * F) / st["sweeps"]


def sample_stats(L, args):
    lat = D.lattice(L, L, "pbc")
    T, nn, Tc, nbr, nnb, isnn, nnl = lat
    n = D.T1.filling(L * L)
    obj = make_f("half_F4", T, L, L, "pbc", n, args.rmax, args.seed + L)
    obj.setup(*D.T1.rand_cfg(L * L, n, obj.rng))
    D.seed_nb(args.seed + L)
    for _ in range(args.ops_burn):
        D.do_sweep(obj, lat, 4 * L * L, False)
    rng = np.random.default_rng(3)
    st = dict(ratios=[], accepts=[], sweeps=args.ops_sweeps)
    for _ in range(args.ops_sweeps):
        for _ in range(4 * L * L):
            mv = D.propose(obj.ou, obj.od, nbr, nnb, isnn)
            if mv[0] == -2:
                continue
            r0 = int(obj.rr[0])
            rho = D.r_ratio(*mv, *obj.args_ratio())
            k, b = int(obj.sc["k"][0]), int(obj.sc["k"][1])
            rec = (b, k, mv[2] >= 0, r0)
            st["ratios"].append(rec)
            if rng.random() < min(1.0, rho * rho):
                D.r_accept(*mv, *obj.args_accept())
                st["accepts"].append(rec)
        diag, cm, cf = D.connections(obj.ou, obj.od, Tc, nbr, nnb, nnl, D.JX)
        for c in range(cm.shape[0]):
            mv = tuple(int(x) for x in cm[c])
            r0 = int(obj.rr[0])
            D.r_ratio(*mv, *obj.args_ratio())
            st["ratios"].append((int(obj.sc["k"][1]), int(obj.sc["k"][0]), mv[2] >= 0, r0))
    return st, n, obj.Vh, L * L


def ops(args):
    for L in args.size_list:
        st, n, Vh, V = sample_stats(L, args)
        N = n  # electrons per spin = slots
        nr, na = len(st["ratios"]), len(st["accepts"])
        bm = np.mean([r[0] for r in st["ratios"]])
        km = np.mean([r[1] for r in st["ratios"]])
        print(f"L={L} N={N} V_h={Vh}: ratios/sweep={nr / st['sweeps']:.0f} accepts/sweep={na / st['sweeps']:.0f} "
              f"<b>={bm:.2f} <k>={km:.2f} down-move fraction={np.mean([r[2] for r in st['ratios']]):.2f} "
              f"<r>={np.mean([r[3] for r in st['ratios']]):.1f}")
        dn = dense_ops(st, 4, N)
        base = route_ops(st, 4, N, Vh, V)
        for F in args.F_list:
            ro = route_ops(st, F, N, Vh, V)
            pred = RD11C_16[0] * base / ro if L == 16 else float("nan")
            print(f"   half_F{F:<2d}: ops/sweep={ro:.3e}  F-dependent share={(ro - route_ops(st, 0, N, Vh, V)) / ro:.2f}"
                  f"  R_ops=dense2/half={dn / ro:.2f}  ops(F)/ops(4)={ro / base:.2f}"
                  + (f"  predicted R_D(16)={pred:.2f}" if L == 16 else ""), flush=True)
        print(f"   dense2_F4: ops/sweep={dn:.3e}")


# ------------------------------------------------------------------ run / aggregate
def run(args):
    D.OUT = OUT
    ns = SimpleNamespace(arm=f"{args.wave}_{args.arm}", rmax=args.rmax, seed=args.seed, tsw_only="",
                         min_sweeps=3, size_list=args.size_list, budget_list=args.budget_list)
    # delayed.run names the object by args.arm; strip the wave prefix for make_f
    D.make = lambda arm, *a: make_f(arm[2:] if arm[:2] in ("A_", "B_") else arm, *a)
    D.run(ns)


def aggregate(args):
    R = {f[:-5]: json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")}
    for k, r in R.items():
        print(f"{k:12s} " + "  ".join(f"L={L}: t_sw={s['t_sw']:.3e} tau={s['tau']:.2f}+-{s['tau_err']:.2f} "
                                      f"sweeps={s['sweeps']}" for L, s in r["sizes"].items()))
    passed, inc = [], []
    for F in args.F_list:
        key = [k for k in R if k.endswith(f"_half_F{F}")]
        if not key:
            print(f"half_F{F}: INCOMPLETE (no run)"); inc.append(F); continue
        k = key[0]
        dref = k.split("_")[0] + "_dense2"
        if dref not in R:
            print(f"half_F{F}: INCOMPLETE (no dense2 in wave)"); inc.append(F); continue
        txt = []
        for L in R[k]["sizes"]:
            Rv, e, rt = D.ratio(R[dref]["sizes"][L], R[k]["sizes"][L])
            txt.append(f"L={L}: {Rv:.2f}+-{e:.2f} (t_sw ratio {rt:.2f})")
        R16, e16, _ = D.ratio(R[dref]["sizes"]["16"], R[k]["sizes"]["16"])
        if not (np.isfinite(R16) and np.isfinite(e16)):
            print(f"half_F{F} vs {dref}: " + "; ".join(txt) + " -> INCOMPLETE (tau at L=16)"); inc.append(F); continue
        hit = R16 - e16 >= 3
        if hit:
            passed.append(F)
        print(f"half_F{F} vs {dref}: " + "; ".join(txt) + f" -> R_D(16)-sigma={R16 - e16:.2f} "
              f"({'>= 3' if hit else '< 3'})")
    big = [F for F in passed if F > 4]
    if big:
        v = f"GATE-PASS: largest F with R_D(16)-sigma >= 3 is F={max(big)}"
    elif any(F > 4 for F in inc):
        v = "INCOMPLETE (a larger-F arm has no L=16 tau)"
    else:
        v = "GATE-FAIL: no F > 4 keeps R_D(16)-sigma >= 3"
    print("VERDICT:", v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["params", "gate", "ops", "run", "aggregate"], default="gate")
    ap.add_argument("--arm", choices=["half_F4", "half_F8", "half_F12", "half_F16", "dense2"], default="half_F8")
    ap.add_argument("--wave", choices=["A", "B"], default="A")
    ap.add_argument("--sizes", default="12,16")
    ap.add_argument("--budgets", default="20,90")
    ap.add_argument("--rmax", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--accepts", type=int, default=2000)
    ap.add_argument("--ops-sweeps", type=int, default=3)
    ap.add_argument("--ops-burn", type=int, default=5)
    ap.add_argument("--Fs", default="4,8,12,16")
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.budget_list = [float(s) for s in args.budgets.split(",")]
    args.F_list = [int(s) for s in args.Fs.split(",")]
    if args.mode == "params":
        params(args)
    elif args.mode == "gate":
        sys.exit(gate(args))
    elif args.mode == "ops":
        ops(args)
    elif args.mode == "run":
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
