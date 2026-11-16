#!/usr/bin/env python3
"""
P2.1 Round 12b: exact-sum ladder for the sublattice sign-capacity test (pre-registered fallback of Round 12).
============================================================================================================

Same question, arms, protocol and rule as Round 12 (leaf "Round 12 ... T1 rule"; decide() of
../P2.1_subcubic_amplitudes_r12/t1_sub.py is imported and used unchanged). What changes: larger toys, and
convergence budgets large enough that the reference arm `full` can reach its plateau. Round 12 is imported
read-only; its training loop is reproduced here step for step, with checkpoints added.

Ladder (t-t'-J, t' = -0.25, J = 0.4, OBC, N_up = N_dn, singlet non-degenerate ground state, gap >= 0.02 t
asserted in --mode check and again at the start of every training run):
  23: 4x3 5+5 (V = 12, D = 16632,  gap 0.085)  same instance as Round 12 (reused key)
  24: 7x2 6+6 (V = 14, D = 84084,  gap 0.037)
  25: 5x3 7+7 (V = 15, D = 51480,  gap 0.196)
  V = 13 has no rectangular OBC cluster with a 2D t' (13 x 1 is a chain), so the ladder skips it.
  Scored together with Round 12's counted size 20 (4x2 3+3, V = 8), read from ../P2.1_subcubic_amplitudes_r12/results.

Arms: full, half, quarter (t1_sub.SubHFPSAnsatz, F = 4, Ha = 0, seed 11, unprojected SR). dense is not run.

Convergence criterion (stated before any ladder run): Round 12's plateau test unchanged. From step 300 on,
every 50 steps, stop when the best energy improved by less than 2e-4 t over the last 200 steps. A run that
reaches --max-steps (6000) or --max-cpu (10800 CPU-s) before the plateau is converged = false, and its size does not count.

Modes
  --mode check      For every ladder instance (20, 23, 24, 25): ED, then asserts gap >= 0.02 t, non-degenerate
                    singlet ground state; prints D, dim_S_chi, V_h and n_eff per arm, and the dense memory
                    of the largest per-step arrays. Exit 2 on a failed assert.
  --mode tasks      Print the Slurm task table (task id -> instance, arm).
  --mode train      --task K (array index), or --instance I --arm A. Trains to the plateau with checkpoints
                    every --ckpt-every steps in <results>/ckpt (a restarted task resumes from its checkpoint),
                    then runs one budget (p11_core.budget) and writes <results>/t1_<arm>_inst<I>_s<seed>.json in
                    the Round 12 format. Skips if the result exists (unless --overwrite 1).
  --mode probe      --instance I --arm A --probe-steps N: N SR steps, prints CPU-s per step and peak RSS. No output file.
  --mode progress   Anytime report: prints each checkpoint's step, E - E0, best E - E0 and SR CPU-s. It never
                    promotes; no early stop is defined because the scored quantity (dE_sign) exists only after
                    the budget.
  --mode aggregate  Loads Round 12 size 20 results plus the ladder results and applies t1_sub.decide (unchanged rule)
                    over sizes 20, 23, 24, 25. Prints KILL / PASS-TO-T2 / INCONCLUSIVE per variant.
  --mode synthetic  t1_sub.decide on 12 synthetic cases over the ladder sizes (every rule branch, plus 3 ladder-shaped).
Options (defaults = the pre-registered values): --seed 11 --F 4 --Ha 0 --proj none --sr-lr 0.05 --sr-shift 1e-3
  --sr-samples 1000 --min-steps 300 --max-steps 6000 --plateau-window 200 --plateau-tol 2e-4
  --max-cpu 10800 (CPU-s for the SR loop, summed over threads; the budget runs after it) --ckpt-every 250
  --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --chunk 32768 --o-chunk 8 --r-min 1.5
  --results DIR (./results) --r12-results DIR (../P2.1_subcubic_amplitudes_r12/results) --overwrite {0,1}
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1 in the Slurm job (probe: 4 threads cost 3.7x the CPU per step for no wall gain on a loaded node).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
R12 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12")
sys.path.insert(0, R12)
import t1_sub as T  # noqa: E402  (read-only)

mt, LT = T.mt, T.LT
from p11_core import budget, sign_constrained_min  # noqa: E402

log = LT.log
NEW = {24: dict(Lx=7, Ly=2, bcx="obc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4),
       25: dict(Lx=5, Ly=3, bcx="obc", bcy="obc", nup=7, ndn=7, tp=-0.25, J=0.4)}
mt.INSTANCES.update(NEW)
LADDER = {20: T.TOYS[20], 23: T.TOYS[23], **NEW}
T.TOYS = LADDER            # decide() and synthetic() iterate over t1_sub.TOYS
RUN_SIZES = (23, 24, 25)
ARMS = ("full", "half", "quarter")
TASKS = [(i, a) for i in RUN_SIZES for a in ARMS]


def _ckpt_path(args):
    return os.path.join(args.results, "ckpt", f"{args.arm}_inst{args.instance}_s{args.seed}.json")


def _save_ckpt(args, st):
    fn = _ckpt_path(args)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    tmp = fn + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh)
    os.replace(tmp, fn)


def train(args):
    """t1_sub.train, step for step (same RNG streams, same plateau test), plus checkpoint and resume."""
    t_start = time.process_time()
    m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = T.setup(args.instance, args.r_min)
    assert gap >= T.GAP_MIN, f"gap {gap} < {T.GAP_MIN}"
    log(f"instance {args.instance} {LADDER[args.instance]} D={m.D} E0={W[0]:.10f} gap={gap:.4f} |G|={sg.G} "
        f"chi={gen} dim_S_chi={dS:.0f} arm={args.arm} seed={args.seed} threads={os.environ.get('OMP_NUM_THREADS')}")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a = T.make(args.arm, m, refs[1], args.F, args.Ha, args.seed)
    log(f"arm {args.arm}: F={args.F} Ha={args.Ha} n_eff={a.n_eff()} P/dim_S_chi={a.n_eff() / dS:.2f} "
        f"calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * T.RNG_OFF[args.arm])
    best_E, best_x, traj, best_hist = np.inf, a.flat().copy(), [], []
    stop, converged, it0, cpu_prev = "max-steps", False, 0, 0.0
    ck = _ckpt_path(args)
    if os.path.exists(ck) and args.probe_steps == 0:
        st = json.load(open(ck))
        a.set_flat(np.array(st["x"]))
        best_x, best_E = np.array(st["best_x"]), st["best_E"]
        traj, best_hist, it0, cpu_prev = st["traj"], st["best_hist"], st["it"], st["sr_cpu"]
        rng.bit_generator.state = st["rng"]
        log(f"resumed from {ck} at step {it0} (SR CPU so far {cpu_prev:.0f} s)")
    t0 = time.process_time() - cpu_prev
    n_max = args.probe_steps if args.probe_steps else args.max_steps
    for it in range(it0, n_max):
        x0 = a.flat().copy()
        E, f, step = mt.sr_step(a, m, sg, args.proj, 0, args, rng)
        traj.append(E)
        if not np.isfinite(E):
            stop = f"non-finite energy at it {it}"; break
        if E < best_E:
            best_E, best_x = E, x0
        best_hist.append(best_E)
        a.set_flat(x0 - step)
        if it % 50 == 0:
            log(f"{args.arm}: it {it:5d} E-E0 = {E - W[0]:.6f} best = {best_E - W[0]:.6f} "
                f"({(time.process_time() - t0) / (it + 1):.3f} CPU-s/step)")
        if it + 1 >= args.min_steps and (it + 1) % 50 == 0:
            if best_hist[-1 - args.plateau_window] - best_E < args.plateau_tol:
                stop, converged = f"plateau at it {it + 1}", True
                break
        if time.process_time() - t0 > args.max_cpu:
            stop = f"CPU cap at it {it + 1}"; break
        if not args.probe_steps and (it + 1) % args.ckpt_every == 0:
            _save_ckpt(args, dict(x=a.flat().tolist(), best_x=np.asarray(best_x).tolist(), best_E=float(best_E),
                                  traj=[float(e) for e in traj], best_hist=[float(e) for e in best_hist],
                                  it=it + 1, sr_cpu=time.process_time() - t0, rng=rng.bit_generator.state,
                                  E0=float(W[0])))
    sr_cpu = time.process_time() - t0
    if args.probe_steps:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        log(f"PROBE inst {args.instance} {args.arm}: D={m.D} n_eff={a.n_eff()} {len(traj)} steps, "
            f"{sr_cpu / len(traj):.3f} CPU-s/step, peak RSS {rss:.2f} GB")
        return None
    a.set_flat(best_x)
    log(f"{args.arm}: stop = {stop}; best E - E0 = {best_E - W[0]:.6f}; SR CPU {sr_cpu:.0f} s")
    psi_t, psi_c, psi = mt.state_of(a, m, sg, args.proj, 0, args.chunk)
    b = budget(m, psi_c, W, Vv, deg, args, rng_b)
    for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
        b.pop(k_, None)
    masked = getattr(a, "max_masked_seen", 0.0)
    log(f"{args.arm} [P]: dE_tot={b['dE_tot']:.6f} dE_sign={b['dE_sign']:.6f} f_E={b['f_E_up']:.3f} "
        f"spread={b['spread']:.1e}; max |masked entry| proposed by SR = {masked:.1e}")
    return dict(instance=args.instance, V=m.V, D=m.D, gap=gap, arm=args.arm, seed=args.seed, F=args.F,
                Ha=args.Ha, n_eff=a.n_eff(), dim_S_chi=dS, G=sg.G, E0=float(W[0]), calib_err=calib,
                converged=converged, stop=stop, steps_done=len(traj), traj=traj, sr_cpu=sr_cpu,
                total_cpu=time.process_time() - t_start, max_masked=masked, budgets=dict(P=b),
                params={k: v for k, v in vars(args).items()})


def run_check(args):
    ok = True
    for inst in LADDER:
        m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = T.setup(inst)   # asserts non-degenerate singlet
        g = gap >= T.GAP_MIN
        ok &= g
        refs = mt.references(m)
        ne = {arm: T.make(arm, m, refs[1], args.F, args.Ha, args.seed).n_eff() for arm in ARMS}
        hs = {s: len(T.hidden_sites(m.Lx, m.Ly, s)) for s in ("half", "quarter")}
        # largest dense per-step arrays: O (sr_samples x P) and K (sr_samples^2), amplitudes over D (complex chunks)
        mem = (args.sr_samples * ne["full"] + args.sr_samples ** 2 + 8 * m.D + 4 * args.chunk * ne["full"]) * 8 / 1e9
        log(f"check inst {inst} {LADDER[inst]}: V={m.V} D={m.D} E0={W[0]:.8f} gap={gap:.4f} (>= {T.GAP_MIN}: {g}) "
            f"singlet non-degenerate |G|={sg.G} dim_S_chi={dS:.0f} V_h half={hs['half']} quarter={hs['quarter']} "
            f"n_eff={ne} dense-array estimate {mem:.2f} GB")
    log(f"CHECK {'PASS' if ok else 'BUG'}")
    return ok


def load_ladder(args):
    R = {}
    for d, insts in ((args.r12_results, (20,)), (args.results, RUN_SIZES)):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.startswith("t1_") and fn.endswith(f"_s{args.seed}.json"):
                r = json.load(open(os.path.join(d, fn)))
                if r["instance"] in insts and r["arm"] in ARMS:
                    R[(r["arm"], r["instance"])] = r
    return R


def aggregate(args):
    R = load_ladder(args)
    print(f"{'inst':>4} {'V':>3} {'D':>6} {'gap':>6} {'arm':>8} {'n_eff':>6} {'conv':>5} {'steps':>6} "
          f"{'dE_tot':>9} {'dE_sign':>9} {'f_E':>6} {'spread':>8} {'cpu_s':>7}")
    for inst in sorted(LADDER):
        for arm in ARMS:
            r = R.get((arm, inst))
            if r is None:
                print(f"{inst:>4} {'':>3} {'':>6} {'':>6} {arm:>8}  missing"); continue
            b = r["budgets"]["P"]
            print(f"{inst:>4} {r['V']:>3} {r['D']:>6} {r['gap']:6.3f} {arm:>8} {r['n_eff']:>6} {str(r['converged']):>5} "
                  f"{r['steps_done']:>6} {b['dE_tot']:9.2e} {b['dE_sign']:9.2e} {b['f_E_up']:6.3f} "
                  f"{b['spread']:8.1e} {r['total_cpu']:7.0f}")
    print(f"total CPU-s (ladder runs only): {sum(r['total_cpu'] for (a_, i), r in R.items() if i in RUN_SIZES):.0f}")
    for var, (v, why, rows) in T.decide(R).items():
        print(f"{var}: {v}  ({'; '.join(why)})")
        for inst, V, r in rows:
            print(f"    inst {inst} V={V}: " + (f"rho = {r:.3f}" if isinstance(r, float) else r))


def progress(args):
    d = os.path.join(args.results, "ckpt")
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".json"):
            st = json.load(open(os.path.join(d, fn)))
            print(f"{fn}: step {st['it']}, E-E0 = {st['traj'][-1] - st['E0']:.6f}, "
                  f"best = {st['best_E'] - st['E0']:.6f}, SR CPU {st['sr_cpu']:.0f} s")


def synthetic(args):
    """t1_sub's nine synthetic cases with the per-size overrides re-keyed to the ladder sizes (20, 23, 24, 25),
    plus three cases shaped like the expected ladder outcomes. decide() itself is t1_sub's, unchanged."""
    S = sorted(LADDER)

    def res(arm, inst, ds, conv=True, gap=0.1, calib=0.0, spread=0.0):
        return dict(arm=arm, instance=inst, V=LADDER[inst]["Lx"] * LADDER[inst]["Ly"], gap=gap, converged=conv,
                    calib_err=calib, budgets=dict(P=dict(dE_sign=ds, dE_tot=0.05, spread=spread)))

    def case(full, half, quarter, kw=None, vkw=None):
        kw, vkw, R = kw or {}, vkw or {}, {}
        for inst, f, h, q in zip(S, full, half, quarter):
            R[("full", inst)] = res("full", inst, f, **kw.get(inst, {}))
            R[("half", inst)] = res("half", inst, h, **vkw.get(inst, {}))
            R[("quarter", inst)] = res("quarter", inst, q, **vkw.get(inst, {}))
        return T.decide(R)
    a, b, c, d = S
    cases = [
        ("both pass", case([.01] * 4, [.011] * 4, [.012] * 4), "PASS-TO-T2", "PASS-TO-T2"),
        ("half KILL ratio at 2", case([.01] * 4, [.013, .013, .01, .01], [.01] * 4), "KILL", "PASS-TO-T2"),
        ("ratio > 1.25 at 1 only", case([.01] * 4, [.013, .01, .01, .01], [.01] * 4), "PASS-TO-T2", "PASS-TO-T2"),
        ("quarter KILL trend", case([.01] * 4, [.01] * 4, [.010, .0105, .011, .012]), "PASS-TO-T2", "KILL"),
        ("trend but small", case([.01] * 4, [.01] * 4, [.010, .0103, .0106, .011]), "PASS-TO-T2", "PASS-TO-T2"),
        ("premise fails at 2 -> INCONCLUSIVE", case([.0005, .0005, .01, .01], [.01] * 4, [.01] * 4),
         "INCONCLUSIVE", "INCONCLUSIVE"),
        ("unconverged full at 2", case([.01] * 4, [.01] * 4, [.01] * 4, kw={a: dict(conv=False), b: dict(conv=False)}),
         "INCONCLUSIVE", "INCONCLUSIVE"),
        ("gap fail + premise fail", case([.01, .0001, .01, .01], [.01] * 4, [.01] * 4, kw={a: dict(gap=0.01)}),
         "INCONCLUSIVE", "INCONCLUSIVE"),
        ("calibration fail at 1, kill at 2 others", case([.01] * 4, [.02] * 4, [.01] * 4, kw={a: dict(calib=1.0)}),
         "KILL", "PASS-TO-T2"),
        ("ladder-shaped: size 20 rho 2.4/9.9, one new size above 1.25",
         case([.00147, .008, .0005, .0005], [.00359, .0126, .01, .01], [.0146, .0148, .01, .01]), "KILL", "KILL"),
        ("ladder-shaped: new sizes all rho ~1.0", case([.00147, .008, .005, .004], [.00359, .008, .005, .004],
                                                      [.0146, .008, .005, .004]), "PASS-TO-T2", "PASS-TO-T2"),
        ("ladder-shaped: variant unconverged at every new size",
         case([.00147, .008, .005, .004], [.00359, .02, .02, .02], [.0146, .02, .02, .02],
              vkw={b: dict(conv=False), c: dict(conv=False), d: dict(conv=False)}), "INCONCLUSIVE", "INCONCLUSIVE"),
    ]
    ok = True
    for name, out, eh, eq in cases:
        good = out["half"][0] == eh and out["quarter"][0] == eq
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}: half {out['half'][0]} (want {eh}), quarter {out['quarter'][0]} (want {eq})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases, sizes {S})")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["check", "tasks", "train", "probe", "progress", "aggregate", "synthetic"],
                   required=True)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--instance", type=int, choices=sorted(LADDER))
    p.add_argument("--probe-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--F", type=int, default=4)
    p.add_argument("--Ha", type=int, default=0)
    p.add_argument("--proj", choices=["none", "chi"], default="none")
    p.add_argument("--sr-lr", type=float, default=0.05)
    p.add_argument("--sr-shift", type=float, default=1e-3)
    p.add_argument("--sr-samples", type=int, default=1000)
    p.add_argument("--min-steps", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--plateau-window", type=int, default=200)
    p.add_argument("--plateau-tol", type=float, default=2e-4)
    p.add_argument("--max-cpu", type=float, default=10800.0)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--lbfgs-iters", type=int, default=4000)
    p.add_argument("--fn-iters", type=int, default=10)
    p.add_argument("--fn-floor", type=float, default=1e-3)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--o-chunk", type=int, default=8)
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--r12-results", default=os.path.join(R12, "results"))
    p.add_argument("--overwrite", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "check":
        sys.exit(0 if run_check(args) else 2)
    if args.mode == "synthetic":
        sys.exit(0 if synthetic(args) else 2)
    if args.mode == "tasks":
        for k, (i, a_) in enumerate(TASKS):
            print(f"task {k}: instance {i} ({LADDER[i]['Lx']}x{LADDER[i]['Ly']} {LADDER[i]['nup']}+{LADDER[i]['ndn']}) arm {a_}")
        return
    if args.mode == "aggregate":
        aggregate(args); return
    if args.mode == "progress":
        progress(args); return
    if args.task is not None:
        args.instance, args.arm = TASKS[args.task]
    assert args.instance is not None and args.arm is not None, "need --task or --instance and --arm"
    if args.mode == "probe":
        assert args.probe_steps > 0, "--mode probe needs --probe-steps N > 0"
        train(args); return
    args.probe_steps = 0
    os.makedirs(args.results, exist_ok=True)
    fn = os.path.join(args.results, f"t1_{args.arm}_inst{args.instance}_s{args.seed}.json")
    if os.path.exists(fn) and not args.overwrite:
        log(f"{fn} exists; refusing to overwrite"); return
    r = train(args)
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
