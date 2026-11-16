#!/usr/bin/env python3
"""
P2.1 Round 12d: `half` (checkerboard hidden rows) at larger hidden width F on the Round 12b exact-sum ladder.
==============================================================================================================

Leaf section "Round 12d: half at larger F (2026-11-16)". Round 12b (job 704752) killed `half` at F = 4 with
rho = 2.44, 1.59, 2.07, 1.80; all are below the Round 12c margin bar of 5, so the margin rule sends `half` to
an accuracy test at the GATE-PASS width F = 16 (Round 12c), with F = 8 as a second point.

Nothing is edited upstream. ../P2.1_subcubic_amplitudes_r12b/ladder.py (and through it r12/t1_sub.py) is
imported read-only: training is ladder.train (Round 12 plateau test, same seed and RNG streams, checkpoints every
250 steps, resume), the ansatz is t1_sub.SubHFPSAnsatz, the verdict is t1_sub.decide, all unchanged. Only the
hidden width F of the trained arm and the results directory differ.

Ladder (same instances as Round 12b): 20 = 4x2 3+3 (V = 8), 23 = 4x3 5+5 (V = 12), 24 = 7x2 6+6 (V = 14),
25 = 5x3 7+7 (V = 15); t-t'-J, t' = -0.25, J = 0.4, OBC. Reference `full` at F = 4 is read from
../P2.1_subcubic_amplitudes_r12/results (size 20) and ../P2.1_subcubic_amplitudes_r12b/results (23, 24, 25).

Arms trained here (results/<arm>_F<F>/t1_<arm>_inst<I>_s11.json, checkpoints in results/<arm>_F<F>/ckpt/):
  half_F16  primary arm (Round 12c candidate width)
  half_F8   second point; parameter-matched to full_F4 (n_eff within 4 percent on every size)
  full_F8   parameter-matched reference for half_F16 (n_eff within 7 percent on every size); reported comparison
Task table (Slurm array index -> arm, instance):  0-3 half_F16 on 20, 23, 24, 25;  4-7 half_F8;  8-11 full_F8.

Comparisons printed by --mode aggregate, each with t1_sub.decide (rule unchanged: KILL if rho > 1.25 at >= 2
counted sizes, or rho strictly rising in V over >= 3 counted sizes with rho(V_max) > 1.15; PASS-TO-T2 (= PASS-T1)
if no kill clause on >= 3 counted sizes; else INCONCLUSIVE):
  A  half_F16 vs full_F4   PRIMARY, decides the route (pre-registered in the leaf)
  B  half_F8  vs full_F4   matched parameters (second point)
  C  half_F16 vs full_F8   matched parameters (reported, interpreted per the leaf)
  D  full_F8  vs full_F4   reported only: does F alone help the all-sites route

Modes
  --mode params     n_eff of every arm on every ladder size (no ED; seconds).
  --mode check      Pre-flight correctness at F = 16 and F = 8: (a) half amplitudes from the ansatz forward pass
                    equal t1_sub.explicit_amp (independent site-by-site determinant build) to rel. err <= 1e-12 on 40
                    random configurations of instances 20 and 23, Ha = 0 (the trained setting) and Ha = 2, with
                    perturbed parameters; (b) masked entries stay exactly 0 after set_flat. Exit 2 on failure.
  --mode tasks      Print the task table.
  --mode train      --task K, or --arm-key {half_F16, half_F8, full_F8} --instance I. Calls ladder.train.
                    Skips if the result file exists unless --overwrite 1.
  --mode probe      --arm-key A --instance I --probe-steps N: CPU-s per SR step and peak RSS. No output file.
  --mode progress   Checkpoint report for every arm (information only; no early stop).
  --mode aggregate  Prints the per-size table and the four comparisons A-D.
  --mode synthetic  Checks that aggregate maps arms into decide() correctly on 4 synthetic cases.
Options: --max-steps (6000) --max-cpu (10800 CPU-s, SR loop) --ckpt-every (250) --results-root DIR (./results)
  --overwrite {0,1}. All other training options are ladder.py's pre-registered defaults (seed 11, Ha 0, proj none,
  SR lr 0.05, shift 1e-3, 1000 samples, plateau from step 300, window 200, tol 2e-4).
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1 (set in the job file).
"""
import argparse
import json
import os
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R12B = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12b")
R12 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12")
sys.path.insert(0, R12B)
import ladder as L  # noqa: E402  (read-only)

T, mt, log = L.T, L.mt, L.log
SIZES = (20, 23, 24, 25)
ARMKEYS = {"half_F16": ("half", 16), "half_F8": ("half", 8), "full_F8": ("full", 8)}
TASKS = [(k, i) for k in ARMKEYS for i in SIZES]
COMPARISONS = [("A", "half_F16", "full_F4", "PRIMARY"), ("B", "half_F8", "full_F4", "matched parameters"),
               ("C", "half_F16", "full_F8", "matched parameters, reported"),
               ("D", "full_F8", "full_F4", "reported only")]


def arm_dir(root, key):
    return os.path.join(root, key)


def ladder_args(key, inst, a):
    arm, F = ARMKEYS[key]
    la = L.parse_args(["--mode", "train", "--instance", str(inst), "--arm", arm, "--F", str(F),
                       "--max-steps", str(a.max_steps), "--max-cpu", str(a.max_cpu),
                       "--ckpt-every", str(a.ckpt_every), "--results", arm_dir(a.results_root, key)])
    la.probe_steps = a.probe_steps
    return la


def params(a):
    rows = [("full_F4", "full", 4), ("half_F4", "half", 4), ("half_F8", "half", 8), ("full_F8", "full", 8),
            ("half_F16", "half", 16)]
    print(f"{'inst':>4} {'V':>3} " + " ".join(f"{r[0]:>9}" for r in rows))
    for inst in SIZES:
        m = mt.make_model(inst); refs = mt.references(m)
        ne = [T.make(arm, m, refs[1], F, 0, 11).n_eff() for _, arm, F in rows]
        print(f"{inst:>4} {m.V:>3} " + " ".join(f"{n:>9}" for n in ne))


def check(a):
    rng = np.random.default_rng(12)
    ok = True
    for inst in (20, 23):
        m = mt.make_model(inst); refs = mt.references(m)
        for F in (16, 8):
            for Ha in (0, 2):
                s = T.SubHFPSAnsatz(m, refs[1], F, Ha, "half", seed=3)
                x = s.flat()
                s.set_flat(x + 0.3 * rng.standard_normal(len(x)))   # perturb everything; set_flat re-zeroes masked
                z = np.abs(s.flat()[~s.mask_flat]).max()
                sel = rng.choice(m.D, 40, replace=False)
                ou, od, ru, rd = s._occ(sel)
                fast = s.forward(ou, od, ru, rd)["psi"]
                ref = np.array([T.explicit_amp(s, ou[k], od[k]) for k in range(len(sel))])
                err = np.max(np.abs(fast - ref)) / np.max(np.abs(ref))
                good = err <= 1e-12 and z == 0.0
                ok &= good
                log(f"check inst {inst} half F={F} Ha={Ha}: n_eff={s.n_eff()} V_h={len(s.hs)} explicit amplitude "
                    f"rel err {err:.1e}; max |masked| after set_flat {z:.1e} -> {'OK' if good else 'BUG'}")
    log(f"CHECK {'PASS' if ok else 'BUG'}")
    return ok


def load(root):
    """(armkey, inst) -> result. full_F4 from Round 12 (inst 20) and Round 12b (23-25)."""
    R = {}
    for d, insts in ((os.path.join(R12, "results"), (20,)), (os.path.join(R12B, "results"), (23, 24, 25))):
        for i in insts:
            fn = os.path.join(d, f"t1_full_inst{i}_s11.json")
            if os.path.exists(fn):
                r = json.load(open(fn)); assert r["F"] == 4 and r["arm"] == "full"
                R[("full_F4", i)] = r
    for key, (arm, F) in ARMKEYS.items():
        for i in SIZES:
            fn = os.path.join(arm_dir(root, key), f"t1_{arm}_inst{i}_s11.json")
            if os.path.exists(fn):
                r = json.load(open(fn)); assert r["F"] == F and r["arm"] == arm
                R[(key, i)] = r
    return R


def compare(R, var, ref):
    """t1_sub.decide unchanged, with `var` in the 'half' slot and `ref` in the 'full' slot."""
    RR = {}
    for i in SIZES:
        if (ref, i) in R:
            RR[("full", i)] = R[(ref, i)]
        if (var, i) in R:
            RR[("half", i)] = R[(var, i)]
    saved = T.TOYS
    T.TOYS = {i: L.LADDER[i] for i in SIZES}
    try:
        return T.decide(RR)["half"]
    finally:
        T.TOYS = saved


def aggregate(a):
    R = load(a.results_root)
    print(f"{'inst':>4} {'V':>3} {'arm':>9} {'n_eff':>6} {'conv':>5} {'steps':>6} {'dE_tot':>9} {'dE_sign':>9} "
          f"{'spread':>8} {'cpu_s':>7}")
    for i in SIZES:
        for key in ("full_F4", "full_F8", "half_F8", "half_F16"):
            r = R.get((key, i))
            if r is None:
                print(f"{i:>4} {'':>3} {key:>9}  missing"); continue
            b = r["budgets"]["P"]
            print(f"{i:>4} {r['V']:>3} {key:>9} {r['n_eff']:>6} {str(r['converged']):>5} {r['steps_done']:>6} "
                  f"{b['dE_tot']:9.2e} {b['dE_sign']:9.2e} {b['spread']:8.1e} {r['total_cpu']:7.0f}")
    print(f"total CPU-s (Round 12d runs only): "
          f"{sum(r['total_cpu'] for (k, _), r in R.items() if k in ARMKEYS):.0f}")
    for lab, var, ref, what in COMPARISONS:
        v, why, rows = compare(R, var, ref)
        print(f"{lab} {var} vs {ref} ({what}): {v}  ({'; '.join(why)})")
        for inst, V, r in rows:
            print(f"    inst {inst} V={V}: " + (f"rho = {r:.3f}" if isinstance(r, float) else r))


def synthetic(a):
    def res(key, inst, ds, conv=True):
        arm, F = (ARMKEYS.get(key) or ("full", 4))
        return dict(arm=arm, F=F, instance=inst, V=L.LADDER[inst]["Lx"] * L.LADDER[inst]["Ly"], gap=0.1,
                    converged=conv, calib_err=0.0, budgets=dict(P=dict(dE_sign=ds, dE_tot=0.05, spread=0.0)))
    f4 = [0.00147, 0.00795, 0.00376, 0.00148]
    cases = [
        ("F16 recovers (rho ~1.1 everywhere)", [1.1 * x for x in f4], {}, "PASS-TO-T2"),
        ("F16 still loses (rho 1.4 at 2 sizes)", [1.4 * f4[0], 1.4 * f4[1], f4[2], f4[3]], {}, "KILL"),
        ("F16 unconverged at 2 sizes", [x for x in f4], {23: False, 24: False}, "INCONCLUSIVE"),
        ("rising trend 1.0, 1.1, 1.2, 1.24", [r * x for r, x in zip((1.0, 1.1, 1.2, 1.24), f4)], {}, "KILL"),
    ]
    ok = True
    for name, h16, unconv, want in cases:
        R = {("full_F4", i): res("full_F4", i, f) for i, f in zip(SIZES, f4)}
        R.update({("half_F16", i): res("half_F16", i, h, conv=not (i in unconv)) for i, h in zip(SIZES, h16)})
        got = compare(R, "half_F16", "full_F4")[0]
        ok &= got == want
        print(f"{'OK ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def progress(a):
    for key in ARMKEYS:
        d = os.path.join(arm_dir(a.results_root, key), "ckpt")
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if fn.endswith(".json"):
                st = json.load(open(os.path.join(d, fn)))
                print(f"{key} {fn}: step {st['it']}, E-E0 = {st['traj'][-1] - st['E0']:.6f}, "
                      f"best = {st['best_E'] - st['E0']:.6f}, SR CPU {st['sr_cpu']:.0f} s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["params", "check", "tasks", "train", "probe", "progress", "aggregate", "synthetic"])
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--arm-key", choices=list(ARMKEYS))
    p.add_argument("--instance", type=int, choices=SIZES)
    p.add_argument("--probe-steps", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--max-cpu", type=float, default=10800.0)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--results-root", default=os.path.join(HERE, "results"))
    p.add_argument("--overwrite", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "params":
        params(a); return
    if a.mode == "check":
        sys.exit(0 if check(a) else 2)
    if a.mode == "synthetic":
        sys.exit(0 if synthetic(a) else 2)
    if a.mode == "tasks":
        for k, (key, i) in enumerate(TASKS):
            print(f"task {k}: {key} instance {i} ({L.LADDER[i]['Lx']}x{L.LADDER[i]['Ly']} "
                  f"{L.LADDER[i]['nup']}+{L.LADDER[i]['ndn']})")
        return
    if a.mode == "aggregate":
        aggregate(a); return
    if a.mode == "progress":
        progress(a); return
    if a.task is not None:
        a.arm_key, a.instance = TASKS[a.task]
    assert a.arm_key and a.instance is not None, "need --task or --arm-key and --instance"
    if a.mode == "probe":
        assert a.probe_steps > 0
        L.train(ladder_args(a.arm_key, a.instance, a))
        log(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2:.2f} GB")
        return
    a.probe_steps = 0
    la = ladder_args(a.arm_key, a.instance, a)
    os.makedirs(la.results, exist_ok=True)
    fn = os.path.join(la.results, f"t1_{la.arm}_inst{la.instance}_s{la.seed}.json")
    if os.path.exists(fn) and not a.overwrite:
        log(f"{fn} exists; refusing to overwrite"); return
    r = L.train(la)
    r["arm_key"] = a.arm_key
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
