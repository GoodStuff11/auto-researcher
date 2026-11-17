#!/usr/bin/env python3
"""
P2.1 Round 15, route R15.1, T1: polished re-score of `half_F16` against `full_F4` at V = 15 (instance 25).
==========================================================================================================

Pre-registered rule: RULE.md in this directory. Per arm: recover the scored state of the original SR run from its
checkpoint (exact resume if the checkpoint predates the scored best), re-score it with the unchanged budget (replay
check), polish it by exact-gradient L-BFGS (<= 300 iterations or <= 480 CPU-s), score the polished state with the
unchanged budget. rho_pol = dE_sign_pol(half_F16) / dE_sign_pol(full_F4); KILL if > 1.25, PASS (licenses the
V = 12, 14 polish as a new test) if <= 1.15, AMBIGUOUS between; guards in RULE.md make it INCONCLUSIVE.

Imported read-only (nothing upstream is edited): ../P2.1_subcubic_amplitudes_r12b/ladder.py (instance table,
pre-registered option defaults) and through it ../P2.1_subcubic_amplitudes_r12/t1_sub.py (setup, make, RNG_OFF,
CALIB_TOL, SPREAD_TOL, PREMISE), micro_test.sr_step and p11_core (budget, energy, sign_constrained_min).

Arms (fixed; no other values are accepted):
  full_F4   arm full, F = 4.  ckpt ../P2.1_subcubic_amplitudes_r12b/results/ckpt/full_inst25_s11.json,
            stored result ../P2.1_subcubic_amplitudes_r12b/results/t1_full_inst25_s11.json (job 704752)
  half_F16  arm half, F = 16. ckpt ../P2.1_subcubic_amplitudes_r12d/results/half_F16/ckpt/half_inst25_s11.json,
            stored result ../P2.1_subcubic_amplitudes_r12d/results/half_F16/t1_half_inst25_s11.json (job 731409)

Modes
  --mode probe      --arm A: setup, start-state recovery, then --probe-iters L-BFGS iterations; prints CPU-s per
                    function+gradient evaluation and per iteration and the setup/replay CPU. No output file.
  --mode polish     --arm A: the full per-arm protocol; writes results/polish_<arm>_inst25.json. Skips if that file
                    exists (unless --overwrite 1).
  --mode aggregate  Reads both result files and prints the per-arm table and the pre-registered verdict.
  --mode synthetic  The verdict function on synthetic inputs covering every branch. Exit 2 on failure.
Options: --max-iter 300 (L-BFGS iterations) --max-cpu 480 (CPU-s of the L-BFGS loop) --probe-iters 5
  --results DIR (./results) --overwrite {0,1}.
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1 (set in the job file).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(TESTS, "P2.1_subcubic_amplitudes_r12b"))
import ladder as L  # noqa: E402  (read-only)

T, mt, log = L.T, L.mt, L.log
from p11_core import budget, energy, sign_constrained_min  # noqa: E402

INST = 25
ARMS = {
    "full_F4": dict(arm="full", F=4,
                    ckpt=os.path.join(TESTS, "P2.1_subcubic_amplitudes_r12b/results/ckpt/full_inst25_s11.json"),
                    res=os.path.join(TESTS, "P2.1_subcubic_amplitudes_r12b/results/t1_full_inst25_s11.json")),
    "half_F16": dict(arm="half", F=16,
                     ckpt=os.path.join(TESTS, "P2.1_subcubic_amplitudes_r12d/results/half_F16/ckpt/half_inst25_s11.json"),
                     res=os.path.join(TESTS, "P2.1_subcubic_amplitudes_r12d/results/half_F16/t1_half_inst25_s11.json")),
}
KILL, PASS = 1.25, 1.15
FALL_TOL, FALL_WIN = 2e-4, 50
REPLAY_E_TOL, REPLAY_DS_TOL = 1e-9, 1e-6


def fresh_rng_b(seed, D):
    """The budget RNG stream of ladder.train: rng(seed + 1000003) after the calibration draw of D numbers."""
    r = np.random.default_rng(seed + 1000003)
    r.random(D)
    return r


def score(a, m, W, Vv, deg, la):
    psi = a.amplitudes(la.chunk)
    b = budget(m, psi, W, Vv, deg, la, fresh_rng_b(la.seed, m.D))
    for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
        b.pop(k_, None)
    return b


def recover_start(a, m, sg, W, la, spec, stored):
    """Scored state of the stored run: ckpt best_x if it is the scored best, else exact SR resume to the argmin."""
    st = json.load(open(spec["ckpt"]))
    traj = np.array(stored["traj"])
    k = int(np.argmin(traj))
    info = dict(ckpt_it=st["it"], scored_step=k, scored_E_minus_E0=float(traj[k] - W[0]), replay_steps=0,
                replay_max_dE=0.0)
    if abs(st["best_E"] - traj[k]) <= REPLAY_E_TOL:
        a.set_flat(np.array(st["best_x"]))
        return info
    assert k >= st["it"], "scored best predates the checkpoint but differs from its best_E"
    a.set_flat(np.array(st["x"]))
    rng = np.random.default_rng(la.seed + 1000 * T.RNG_OFF[la.arm])
    rng.bit_generator.state = st["rng"]
    dmax = 0.0
    for it in range(st["it"], k + 1):
        x0 = a.flat().copy()
        E, f, step = mt.sr_step(a, m, sg, la.proj, 0, la, rng)
        dmax = max(dmax, abs(E - traj[it]))
        if it == k:
            a.set_flat(x0)
            break
        a.set_flat(x0 - step)
    info.update(replay_steps=k + 1 - st["it"], replay_max_dE=float(dmax))
    log(f"replayed SR steps {st['it']}..{k}: max |E - stored E| = {dmax:.1e}")
    return info


def polish_lbfgs(a, m, W, la, max_iter, max_cpu):
    mask = a.mask_flat if hasattr(a, "mask_flat") else np.ones(len(a.flat()), bool)
    x_full = a.flat().copy()
    best = dict(E=np.inf, x=None)
    nfev = [0]

    def fg(y):
        x = x_full.copy(); x[mask] = y
        a.set_flat(x)
        psi = a.amplitudes(la.chunk)
        E, Hp, nn = energy(m.H, psi)
        g = 2.0 * a.force((Hp - E * psi) / nn, la.chunk)
        nfev[0] += 1
        if E < best["E"]:
            best.update(E=E, x=a.flat().copy())
        return E, g[mask]

    hist, t0 = [], time.process_time()
    stop = ["converged (L-BFGS tolerance)"]

    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun))
        if time.process_time() - t0 > max_cpu:
            stop[0] = f"CPU cap at iteration {len(hist)}"
            raise StopIteration

    y0 = x_full[mask].copy()
    E_start = fg(y0)[0]
    res = minimize(fg, y0, jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=max_iter, maxcor=10, ftol=1e-14, gtol=1e-10, maxfun=10 * max_iter))
    if res.nit >= max_iter and not stop[0].startswith("CPU"):
        stop[0] = f"iteration cap {max_iter}"
    elif not stop[0].startswith("CPU"):
        stop[0] = f"converged ({res.message})"
    a.set_flat(best["x"])
    at_cap = "cap" in stop[0]
    h = [E_start] + hist
    drop_last = float(h[-1 - FALL_WIN] - h[-1]) if len(h) > FALL_WIN else float(h[0] - h[-1])
    return dict(E_start_minus_E0=float(E_start - W[0]), E_pol_minus_E0=float(best["E"] - W[0]), nit=len(hist),
                nfev=nfev[0], stop=stop[0], at_cap=at_cap, drop_last50=drop_last,
                still_falling=bool(at_cap and drop_last > FALL_TOL), lbfgs_cpu=time.process_time() - t0,
                hist=[float(e - W[0]) for e in h])


def run_arm(args, probe=False):
    t_start = time.process_time()
    spec = ARMS[args.arm]
    la = L.parse_args(["--mode", "train", "--instance", str(INST), "--arm", spec["arm"], "--F", str(spec["F"])])
    stored = json.load(open(spec["res"]))
    assert stored["arm"] == spec["arm"] and stored["F"] == spec["F"] and stored["instance"] == INST
    m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = T.setup(INST, la.r_min)
    rng_c = np.random.default_rng(la.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_c.random(m.D))], la.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a = T.make(spec["arm"], m, refs[1], spec["F"], la.Ha, la.seed)
    info = recover_start(a, m, sg, W, la, spec, stored)
    E_rec = energy(m.H, a.amplitudes(la.chunk))[0] - W[0]
    info["recovered_E_minus_E0"] = float(E_rec)
    log(f"{args.arm}: D={m.D} gap={gap:.4f} n_eff={a.n_eff()} calib={calib:.1e}; start E-E0 = {E_rec:.9f} "
        f"(stored scored {info['scored_E_minus_E0']:.9f}); setup+recovery CPU {time.process_time() - t_start:.0f} s")
    if probe:
        t1 = time.process_time()
        pr = polish_lbfgs(a, m, W, la, args.probe_iters, 1e9)
        log(f"PROBE {args.arm}: {pr['nit']} iterations, {pr['nfev']} evaluations, "
            f"{pr['lbfgs_cpu'] / pr['nfev']:.2f} CPU-s/evaluation, {pr['lbfgs_cpu'] / max(pr['nit'], 1):.2f} "
            f"CPU-s/iteration; E-E0 {pr['E_start_minus_E0']:.6f} -> {pr['E_pol_minus_E0']:.6f}")
        t2 = time.process_time()
        a.set_flat(a.flat())
        b = score(a, m, W, Vv, deg, la)
        log(f"PROBE budget CPU {time.process_time() - t2:.0f} s (dE_sign {b['dE_sign']:.4e}); total {time.process_time() - t_start:.0f} s")
        return None
    b0 = score(a, m, W, Vv, deg, la)
    ds_stored = stored["budgets"]["P"]["dE_sign"]
    replay_rel = abs(b0["dE_sign"] - ds_stored) / ds_stored
    log(f"{args.arm}: replay budget dE_sign = {b0['dE_sign']:.6e} (stored {ds_stored:.6e}, rel {replay_rel:.1e})")
    pol = polish_lbfgs(a, m, W, la, args.max_iter, args.max_cpu)
    log(f"{args.arm}: polish {pol['stop']}; E-E0 {pol['E_start_minus_E0']:.6f} -> {pol['E_pol_minus_E0']:.6f} "
        f"({pol['nit']} it, {pol['nfev']} ev, {pol['lbfgs_cpu']:.0f} CPU-s, drop over last 50 it "
        f"{pol['drop_last50']:.1e})")
    b1 = score(a, m, W, Vv, deg, la)
    log(f"{args.arm}: polished dE_tot = {b1['dE_tot']:.6e} dE_sign = {b1['dE_sign']:.6e} spread = {b1['spread']:.1e}")
    replay_ok = info["replay_max_dE"] <= REPLAY_E_TOL and replay_rel <= REPLAY_DS_TOL
    return dict(arm_key=args.arm, arm=spec["arm"], F=spec["F"], instance=INST, V=m.V, D=m.D, gap=gap,
                n_eff=a.n_eff(), E0=float(W[0]), calib_err=calib, start=info, replay_dE_sign=b0["dE_sign"],
                stored_dE_sign=ds_stored, replay_rel=replay_rel, replay_ok=bool(replay_ok), polish=pol,
                budgets=dict(start=b0, P=b1), total_cpu=time.process_time() - t_start,
                x_pol=a.flat().tolist(), params=dict(max_iter=args.max_iter, max_cpu=args.max_cpu))


def verdict(R):
    """R: {'full_F4': result, 'half_F16': result}. Returns (verdict, rho_pol, reasons)."""
    if any(k not in R for k in ARMS):
        return "MISSING", None, [f"missing {[k for k in ARMS if k not in R]}"]
    f, h = R["full_F4"], R["half_F16"]
    bf, bh = f["budgets"]["P"], h["budgets"]["P"]
    rho = bh["dE_sign"] / bf["dE_sign"]
    why = []
    for k, r in R.items():
        p, b = r["polish"], r["budgets"]["P"]
        if not p["E_pol_minus_E0"] < p["E_start_minus_E0"]:
            why.append(f"{k}: polish did not lower the energy")
        if p["still_falling"]:
            why.append(f"{k}: still falling at the cap ({p['drop_last50']:.1e} > {FALL_TOL} over {FALL_WIN} it)")
        if not r["replay_ok"]:
            why.append(f"{k}: replay mismatch")
        if r["calib_err"] > T.CALIB_TOL * b["dE_tot"] or not (b["spread"] < T.SPREAD_TOL):
            why.append(f"{k}: calibration/spread guard")
    if bf["dE_sign"] < T.PREMISE:
        why.append(f"premise: polished full_F4 dE_sign {bf['dE_sign']:.2e} < {T.PREMISE}")
    if why:
        return "INCONCLUSIVE", rho, why
    if rho > KILL:
        return "KILL", rho, [f"rho_pol(15) = {rho:.3f} > {KILL}"]
    if rho <= PASS:
        return "PASS", rho, [f"rho_pol(15) = {rho:.3f} <= {PASS}: polish V = 12, 14 and re-apply the Round 12d rule "
                             f"(new pre-registered test)"]
    return "AMBIGUOUS", rho, [f"{PASS} < rho_pol(15) = {rho:.3f} <= {KILL}: NOT RESOLVED, to the per-leaf critic"]


def synthetic(_):
    def res(ds, e0=0.02, e1=0.019, falling=False, replay=True, spread=0.0):
        return dict(calib_err=0.0, replay_ok=replay,
                    polish=dict(E_start_minus_E0=e0, E_pol_minus_E0=e1, still_falling=falling, drop_last50=3e-4),
                    budgets=dict(P=dict(dE_sign=ds, dE_tot=0.02, spread=spread)))
    cases = [("rho 1.9", dict(full_F4=res(1.4e-3), half_F16=res(2.66e-3)), "KILL"),
             ("rho 1.0", dict(full_F4=res(1.4e-3), half_F16=res(1.4e-3)), "PASS"),
             ("rho 1.20", dict(full_F4=res(1.4e-3), half_F16=res(1.68e-3)), "AMBIGUOUS"),
             ("rho exactly 1.15", dict(full_F4=res(2e-3), half_F16=res(2.3e-3)), "PASS"),
             ("half still falling", dict(full_F4=res(1.4e-3), half_F16=res(2.66e-3, falling=True)), "INCONCLUSIVE"),
             ("polish raised E", dict(full_F4=res(1.4e-3, e1=0.021), half_F16=res(1.4e-3)), "INCONCLUSIVE"),
             ("replay mismatch", dict(full_F4=res(1.4e-3, replay=False), half_F16=res(1.4e-3)), "INCONCLUSIVE"),
             ("premise fails", dict(full_F4=res(0.9e-3), half_F16=res(2.0e-3)), "INCONCLUSIVE"),
             ("spread fails", dict(full_F4=res(1.4e-3), half_F16=res(2.0e-3, spread=0.1)), "INCONCLUSIVE"),
             ("missing arm", dict(full_F4=res(1.4e-3)), "MISSING")]
    ok = True
    for name, R, want in cases:
        got = verdict(R)[0]
        ok &= got == want
        print(f"{'OK ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def aggregate(args):
    R = {}
    for k in ARMS:
        fn = os.path.join(args.results, f"polish_{k}_inst{INST}.json")
        if os.path.exists(fn):
            R[k] = json.load(open(fn))
    for k, r in R.items():
        p, b0, b1 = r["polish"], r["budgets"]["start"], r["budgets"]["P"]
        print(f"{k:9s} start E-E0 {p['E_start_minus_E0']:.6f} -> polished {p['E_pol_minus_E0']:.6f}; "
              f"dE_sign {b0['dE_sign']:.4e} -> {b1['dE_sign']:.4e}; dE_tot {b1['dE_tot']:.4e}; {p['stop']}, "
              f"{p['nit']} it, last-50 drop {p['drop_last50']:.1e}; replay ok {r['replay_ok']}; CPU {r['total_cpu']:.0f} s")
    if len(R) == 2:
        print(f"dE_tot ratio (reported only): {R['half_F16']['budgets']['P']['dE_tot'] / R['full_F4']['budgets']['P']['dE_tot']:.3f}")
    v, rho, why = verdict(R)
    print(f"VERDICT R15.1 (V=15 polished): {v}" + (f"; rho_pol = {rho:.3f}" if rho is not None else "")
          + f"  ({'; '.join(why)})")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["probe", "polish", "aggregate", "synthetic"])
    p.add_argument("--arm", choices=list(ARMS))
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-cpu", type=float, default=480.0)
    p.add_argument("--probe-iters", type=int, default=5)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--overwrite", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "synthetic":
        sys.exit(0 if synthetic(args) else 2)
    if args.mode == "aggregate":
        aggregate(args); return
    assert args.arm, "--arm is required"
    if args.mode == "probe":
        run_arm(args, probe=True); return
    os.makedirs(args.results, exist_ok=True)
    fn = os.path.join(args.results, f"polish_{args.arm}_inst{INST}.json")
    if os.path.exists(fn) and not args.overwrite:
        log(f"{fn} exists; refusing to overwrite"); return
    r = run_arm(args)
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
