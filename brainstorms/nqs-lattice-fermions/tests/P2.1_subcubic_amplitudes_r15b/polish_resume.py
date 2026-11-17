#!/usr/bin/env python3
"""
P2.1 Round 15, route R15.1, T1 resume: continue the `half_F16` L-BFGS polish of job 742029 to the convergence guard.
====================================================================================================================

Pre-registered rule: RULE_resume.md in this directory (thresholds of RULE.md unchanged). Job 742029 (RULE.md) ended
INCONCLUSIVE only because `half_F16` hit its 480 CPU-s cap still falling (3.7e-4 t over its last 50 iterations > the
2e-4 guard); `full_F4` met the guard (2.7e-5). This script resumes `half_F16` ONLY, from its polished state
(`x_pol` of results/polish_half_F16_inst25.json, the full flat parameter vector, masked entries already zero), and
re-uses the `full_F4` polished result of job 742029 unchanged.

Per run (mode resume):
  1. Setup exactly as polish.py (instance 25, ED, calibration, ansatz `half`, F = 16, seed 11 defaults of ladder.py).
  2. Load x_pol. Replay checks: exact energy E - E0 must equal the stored polished value to 1e-9 t, and the unchanged
     budget (fresh RNG stream, as polish.py `score`) must reproduce the stored polished dE_sign to relative 1e-6;
     a mismatch makes the result INCONCLUSIVE (which RULE_resume.md records as AMBIGUOUS).
  3. L-BFGS-B (same options as polish.py: maxcor 10, ftol 1e-14, gtol 1e-10, unmasked coordinates only) from x_pol.
     It stops at the FIRST of: (a) the convergence guard, i.e. the energy fell by <= 2e-4 t over the last 50
     iterations of this resume (checked from iteration 50 of the resume on; this is the RULE.md definition that
     `full_F4` already satisfies at its saved state, so both arms are stopped by the same criterion), (b) the
     L-BFGS tolerances, (c) --max-iter iterations, (d) --max-cpu CPU-s of the L-BFGS loop. Stopped by (c) or (d)
     with a last-50 drop > 2e-4 t -> still_falling = True. The L-BFGS curvature memory of job 742029 was not saved,
     so the resume starts with an empty memory (first few iterations are gradient-like; this only costs iterations).
  4. Score the lowest-energy evaluated point with the unchanged budget -> dE_sign_pol(half_F16).
  Writes results/resume_half_F16_inst25.json (refuses to overwrite unless --overwrite 1). The energy history is saved
  (hist, E - E0 per iteration, starting with the resume start point) every iteration count, plus x_pol.

Modes (--mode, required; these are the only values):
  resume     The protocol above. Writes the result file.
  probe      Steps 1-2 then --probe-iters L-BFGS iterations; prints CPU-s per iteration and setup CPU. No file.
  aggregate  Reads results/polish_full_F4_inst25.json (job 742029) and results/resume_half_F16_inst25.json, applies
             polish.verdict (RULE.md thresholds and guards, unchanged), and maps INCONCLUSIVE -> AMBIGUOUS (no more
             compute) as RULE_resume.md requires. Prints the table and the verdict.
  synthetic  The resume verdict mapping on synthetic inputs (every branch). Exit 2 on failure.

Options:
  --max-iter N      L-BFGS iteration cap of the resume (default 3000).
  --max-cpu S       CPU-s cap of the L-BFGS loop (default 7200).
  --probe-iters N   iterations in probe mode (default 5).
  --results DIR     results directory (default ./results).
  --overwrite {0,1} overwrite an existing resume result file (default 0).
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1 (set in the job file). Imports polish.py and through it ladder.py,
t1_sub.py, micro_test.py, p11_core.py read-only.
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
sys.path.insert(0, HERE)
import polish as P  # noqa: E402  (read-only; carries ladder/t1_sub/p11_core imports)

L, T, mt, log = P.L, P.T, P.mt, P.log
energy, sign_constrained_min = P.energy, P.sign_constrained_min
ARM = "half_F16"
FALL_TOL, FALL_WIN = P.FALL_TOL, P.FALL_WIN


def lbfgs_to_guard(a, m, W, la, max_iter, max_cpu):
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

    y0 = x_full[mask].copy()
    E_start = fg(y0)[0]
    h = [float(E_start)]
    t0 = time.process_time()
    stop = [None]

    def cb(intermediate_result):
        h.append(float(intermediate_result.fun))
        n = len(h) - 1
        if n >= FALL_WIN and h[-1 - FALL_WIN] - h[-1] <= FALL_TOL:
            stop[0] = f"convergence guard at iteration {n}"
            raise StopIteration
        if time.process_time() - t0 > max_cpu:
            stop[0] = f"CPU cap at iteration {n}"
            raise StopIteration
        if n % 100 == 0:
            log(f"  it {n}: E-E0 {h[-1] - W[0]:.6f}, last-50 drop {h[-1 - FALL_WIN] - h[-1] if n >= FALL_WIN else float('nan'):.1e}, "
                f"CPU {time.process_time() - t0:.0f} s")

    res = minimize(fg, y0, jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=max_iter, maxcor=10, ftol=1e-14, gtol=1e-10, maxfun=10 * max_iter))
    nit = len(h) - 1
    if stop[0] is None:
        stop[0] = f"iteration cap {max_iter}" if nit >= max_iter else f"converged ({res.message})"
    a.set_flat(best["x"])
    at_cap = "cap" in stop[0]
    drop_last = float(h[-1 - FALL_WIN] - h[-1]) if len(h) > FALL_WIN else float(h[0] - h[-1])
    return dict(E_start_minus_E0=float(E_start - W[0]), E_pol_minus_E0=float(best["E"] - W[0]), nit=nit,
                nfev=nfev[0], stop=stop[0], at_cap=at_cap, drop_last50=drop_last,
                still_falling=bool(at_cap and drop_last > FALL_TOL), lbfgs_cpu=time.process_time() - t0,
                hist=[float(e - W[0]) for e in h])


def run(args, probe=False):
    t_start = time.process_time()
    spec = P.ARMS[ARM]
    prev = json.load(open(os.path.join(args.results, f"polish_{ARM}_inst{P.INST}.json")))
    la = L.parse_args(["--mode", "train", "--instance", str(P.INST), "--arm", spec["arm"], "--F", str(spec["F"])])
    m, W, Vv, deg, gap, psi0, sg, gen, dchi, dS = T.setup(P.INST, la.r_min)
    rng_c = np.random.default_rng(la.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_c.random(m.D))], la.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a = T.make(spec["arm"], m, refs[1], spec["F"], la.Ha, la.seed)
    a.set_flat(np.array(prev["x_pol"]))
    E_rec = float(energy(m.H, a.amplitudes(la.chunk))[0] - W[0])
    dE_load = abs(E_rec - prev["polish"]["E_pol_minus_E0"])
    log(f"{ARM}: D={m.D} n_eff={a.n_eff()} calib={calib:.1e}; loaded x_pol E-E0 = {E_rec:.9f} "
        f"(stored {prev['polish']['E_pol_minus_E0']:.9f}, |diff| {dE_load:.1e}); setup CPU {time.process_time() - t_start:.0f} s")
    if probe:
        pr = lbfgs_to_guard(a, m, W, la, args.probe_iters, 1e9)
        log(f"PROBE {ARM}: {pr['nit']} it, {pr['nfev']} ev, {pr['lbfgs_cpu'] / max(pr['nit'], 1):.2f} CPU-s/it, "
            f"{pr['lbfgs_cpu'] / pr['nfev']:.2f} CPU-s/ev; E-E0 {pr['E_start_minus_E0']:.6f} -> {pr['E_pol_minus_E0']:.6f}; "
            f"total CPU {time.process_time() - t_start:.0f} s")
        return None
    b0 = P.score(a, m, W, Vv, deg, la)
    ds_stored = prev["budgets"]["P"]["dE_sign"]
    replay_rel = abs(b0["dE_sign"] - ds_stored) / ds_stored
    log(f"{ARM}: replay budget dE_sign = {b0['dE_sign']:.6e} (stored {ds_stored:.6e}, rel {replay_rel:.1e})")
    pol = lbfgs_to_guard(a, m, W, la, args.max_iter, args.max_cpu)
    log(f"{ARM}: resume {pol['stop']}; E-E0 {pol['E_start_minus_E0']:.6f} -> {pol['E_pol_minus_E0']:.6f} "
        f"({pol['nit']} it, {pol['nfev']} ev, {pol['lbfgs_cpu']:.0f} CPU-s, last-50 drop {pol['drop_last50']:.1e})")
    b1 = P.score(a, m, W, Vv, deg, la)
    log(f"{ARM}: resumed dE_tot = {b1['dE_tot']:.6e} dE_sign = {b1['dE_sign']:.6e} spread = {b1['spread']:.1e}")
    replay_ok = dE_load <= P.REPLAY_E_TOL and replay_rel <= P.REPLAY_DS_TOL
    return dict(arm_key=ARM, arm=spec["arm"], F=spec["F"], instance=P.INST, V=m.V, D=m.D, gap=gap,
                n_eff=a.n_eff(), E0=float(W[0]), calib_err=calib,
                start=dict(source="x_pol of results/polish_half_F16_inst25.json (job 742029)",
                           loaded_E_minus_E0=E_rec, load_abs_diff=dE_load),
                replay_dE_sign=b0["dE_sign"], stored_dE_sign=ds_stored, replay_rel=replay_rel,
                replay_ok=bool(replay_ok), polish=pol, budgets=dict(start=b0, P=b1),
                total_cpu=time.process_time() - t_start, x_pol=a.flat().tolist(),
                params=dict(max_iter=args.max_iter, max_cpu=args.max_cpu, stop_guard=[FALL_TOL, FALL_WIN]))


def resume_verdict(R):
    v, rho, why = P.verdict(R)
    if v == "INCONCLUSIVE":
        return "AMBIGUOUS", rho, ["INCONCLUSIVE again -> AMBIGUOUS (RULE_resume.md), no more compute"] + why
    return v, rho, why


def aggregate(args):
    R = {}
    for k, fn in (("full_F4", f"polish_full_F4_inst{P.INST}.json"), (ARM, f"resume_{ARM}_inst{P.INST}.json")):
        fp = os.path.join(args.results, fn)
        if os.path.exists(fp):
            R[k] = json.load(open(fp))
    for k, r in R.items():
        p, b1 = r["polish"], r["budgets"]["P"]
        print(f"{k:9s} E-E0 {p['E_start_minus_E0']:.6f} -> {p['E_pol_minus_E0']:.6f}; dE_sign {b1['dE_sign']:.4e}; "
              f"dE_tot {b1['dE_tot']:.4e}; {p['stop']}, {p['nit']} it, last-50 drop {p['drop_last50']:.1e}; "
              f"replay ok {r['replay_ok']}")
    v, rho, why = resume_verdict(R)
    print(f"VERDICT R15.1 resume (V=15, half_F16 resumed to the guard): {v}"
          + (f"; rho_pol = {rho:.3f}" if rho is not None else "") + f"  ({'; '.join(why)})")


def synthetic(_):
    def res(ds, e0=0.02, e1=0.019, falling=False, replay=True):
        return dict(calib_err=0.0, replay_ok=replay,
                    polish=dict(E_start_minus_E0=e0, E_pol_minus_E0=e1, still_falling=falling, drop_last50=3e-4),
                    budgets=dict(P=dict(dE_sign=ds, dE_tot=0.02, spread=0.0)))
    cases = [("rho 1.69", dict(full_F4=res(1.495e-3), half_F16=res(2.53e-3)), "KILL"),
             ("rho 1.10", dict(full_F4=res(1.495e-3), half_F16=res(1.64e-3)), "PASS"),
             ("rho 1.20", dict(full_F4=res(1.495e-3), half_F16=res(1.79e-3)), "AMBIGUOUS"),
             ("still falling, rho 1.6", dict(full_F4=res(1.495e-3), half_F16=res(2.4e-3, falling=True)), "AMBIGUOUS"),
             ("replay mismatch", dict(full_F4=res(1.495e-3), half_F16=res(2.4e-3, replay=False)), "AMBIGUOUS"),
             ("missing", dict(full_F4=res(1.495e-3)), "MISSING")]
    ok = True
    for name, R, want in cases:
        got = resume_verdict(R)[0]
        ok &= got == want
        print(f"{'OK ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SYNTHETIC {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["resume", "probe", "aggregate", "synthetic"])
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--max-cpu", type=float, default=7200.0)
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
    if args.mode == "probe":
        run(args, probe=True); return
    fn = os.path.join(args.results, f"resume_{ARM}_inst{P.INST}.json")
    if os.path.exists(fn) and not args.overwrite:
        log(f"{fn} exists; refusing to overwrite"); return
    r = run(args)
    json.dump(r, open(fn, "w"))
    log(f"wrote {fn}")


if __name__ == "__main__":
    main()
