#!/usr/bin/env python3
r"""
P4.2 round 5, part 1: CLOSING RUN for Route 3.1 (TCMN size transfer) under the critic-fixed rule.
================================================================================================

Why this exists
---------------
The round-3 zero-shot pilot (job 410960, ../P4.2_size_transfer_r3/results_zeroshot/task{0,1,2}.json;
4x4 torus, 4 holes, t' = -0.25, J = 0.4, irrep Tx=Ty=Mx=My=+1, D=-1, E0 = -13.93410177) gave
eps_tr0 = 0.663 / 0.759 / 0.615 t/site and picked the ED irrep in 2 of 3 seeds, so its pre-committed
kill rule did not fire. But that pilot ran with --budget 0, so its "scrB" arm is the UNTRAINED
mean-field (MF) initialisation (eps 0.318 / 0.130 / 0.285): the zero-shot transferred state is
2.1-5.8x WORSE than doing nothing. This script evaluates the critic-fixed decision rule with the
fewest extra target steps, reusing the pilot's transferred parameters (no source retraining).

Critic fixes applied (round-3 per-leaf critic, master doc)
----------------------------------------------------------
* Score as EXCESS over eps_opt. eps_opt (best ansatz energy) is not measured here; it is
  BRACKETED: 0 <= eps_opt <= eps_best, eps_best = the lowest eps seen by any arm, step or seed.
  Every ratio criterion is linear in eps_opt, so it is checked at both ends of the bracket; a
  criterion is "robust" only if it holds at both ends.
* VOID tightened: --void-tol 0.01 (was 0.05). Non-VOID is certified only if
  eps_best * V / |E0| <= 0.01; otherwise "VOID not excluded" and PROMOTE is impossible.
* Gate on steps-to-eps at B in {10, 20, 50} with a plateau check. For each B: level
  l_B = median over seeds of eps_scr(B); n_tr = median over seeds of the first step at which
  eps_tr <= l_B (None if never within the run); speed-up S(B) = B / n_tr (0 if never). B is
  COUNTED only if (P1) m eps_scr(B) - eps_best >= 1e-3 t/site (scratch not converged) and
  (plateau) m [eps_scr(B-5) - eps_scr(B)] >= 0.02 m [eps_scr(B) - eps_best] (scratch still moving).
* Route 3.2 (frozen-sign transfer) is merged into 3.1 as a DIAGNOSTIC: fixed-sign excess A of the
  zero-shot transferred state vs the MF init (pilot), and A of tr / scr after 20 and 50 steps.
* COMP_TOL = 1e-3 t/site (was 0.005) for the same-irrep competitor flag.

Arms (per seed s = 0, 1, 2; all trained and scored in the ED irrep by projected exact-sum SR)
------------------------------------------------------------------------------------------
  tr    pilot tr0 parameters (task s) + --steps projected-SR steps (one continuous trajectory)
  scr   MFNG from the MF init (rng 2000+s, identical to the pilot's scrB init) + --steps steps
  foil  Jastrow-BCS foil (C = 0) from its MF init (rng 2000+s, as r3 role foil) + --foil-steps
        steps. SR here never raises the energy, so eps_foil(K) is an UPPER bound on the r3
        eps_foil_opt; it certifies the r3 FAIL clause m(eps_tr0) >= 2 m(eps_foil_opt).
  nfp   DIAGNOSTIC for the round-5 mutation R5.1 (not in the 3.1 verdict): the pilot source
        optimum with its source-null directions reset to the source run's own MF init
        (r5_common.null_filter, tol --null-tol), zero shot + --nfp-steps steps.
Consistency checks (else the task is INVALID): re-evaluated eps of tr at step 0 and of scr at
step 0 equal the pilot's tr0 / scrB eps to 1e-8 (relative); the null count at --null-tol 1e-8
equals the pilot's source n_null.

Decision (pre-committed; --aggregate). m(.) = median over the 3 seeds.
--------------------------------------------------------------------
  CLOSED-FAIL if any of
    F1 (criterion b fails robustly, B = 20 counted): m eps_tr(20) - x >= 0.9 (m eps_scr(20) - x)
       for x = 0 and x = eps_best;
    F2 (r3 FAIL clause, certified): m eps_tr0 >= 2 m eps_foil(K) >= 2 m eps_foil_opt;
    F3 transfer slower than scratch at every counted B (S(B) < 1, at least one B counted);
    F4 spread(eps_tr0) > m eps_tr0;   F5 sector rule (S) correct in fewer than 2 seeds.
  PROMOTE (run the full r3 array with these fixes) only if all of
    (I) spread <= m/2, (S), (e) m eps_tr0 < m eps_tr0_r2 (pilot), robust b-PASS at B = 20
    (m eps_tr(20) - x <= 0.5 (m eps_scr(20) - x) at both ends), S(B) >= 2 at >= 2 counted B,
    and non-VOID certified.
  CLOSED-NOT-PROMOTED otherwise. Note: (e) already FAILS on the pilot data (0.663 vs 0.454
  medians; worse in every seed), so PROMOTE is excluded before this run; the run decides between
  CLOSED-FAIL and CLOSED-NOT-PROMOTED and measures how much a warm start recovers.

Command-line options
--------------------
  --seed INT          0, 1 or 2: which pilot task (task{seed}.json) to continue. Default 0.
  --pilot-dir PATH    Folder with the pilot JSONs (default ../P4.2_size_transfer_r3/results_zeroshot).
                      Model, lattice, instance and network sizes are all read from the pilot JSON.
  --steps INT         Target steps of tr and scr (default 50).
  --ckpts STR         Comma list of checkpoint steps, the B values of the rule (default "10,20,50").
                      Full records (E, eps, fidelity, Pd) at 0 and every checkpoint; the
                      fixed-sign budget A at the last two checkpoints and at 0.
  --foil-steps INT    Steps of the foil arm (default 10).
  --nfp-steps INT     Steps of the nfp diagnostic arm (default 20).
  --null-tol FLOAT    Relative eigenvalue threshold of the source-null space (default 1e-8, the
                      r3 n_null convention).
  --wall-hours FLOAT  Stop stepping (and record) after this many hours (default 3.6). Arms run in
                      the order tr, scr, foil, nfp.
  --out PATH          Output JSON (default results_close31/close31_s{seed}.json next to this file).
  --aggregate DIR     Read every close31_s*.json in DIR (plus the pilot JSONs) and print the decision.
  --plateau-frac, --void-tol, --comp-tol   Rule constants (0.02, 0.01, 1e-3); listed for
                      transparency, change only for a declared re-analysis.

Examples
--------
  python3 close31.py --seed 0                                   # one Slurm task (~3 h, 8 threads)
  python3 close31.py --aggregate results_close31
  python3 close31.py --seed 0 --pilot-dir ../P4.2_size_transfer_r3/results_smoke --steps 6 \
      --ckpts 2,4,6 --foil-steps 2 --nfp-steps 2 --out results_smoke_close31/close31_s0.json  # smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_core as rc  # noqa: E402
from r5_common import med, null_filter, optimise_ckpt, steps_to  # noqa: E402

PREM_EPS = 1e-3


def r3_args(pilot_params):
    p = dict(pilot_params)
    a = rc.parse_args([f"--instances={p['tp_sel']}:{p['J_sel']}"])
    for k, v in p.items():
        if k in ("instances", "out", "aggregate", "selftest", "timing", "task_id", "role", "seed"):
            continue
        setattr(a, k, v)
    a.tp_sel = a.J_sel = None
    return a


def run(cli):
    t_start = time.time()
    deadline = t_start + cli.wall_hours * 3600
    pil = json.load(open(os.path.join(cli.pilot_dir, f"task{cli.seed}.json")))
    assert pil["role"] == "transfer" and pil["seed"] == cli.seed
    args = r3_args(pil["params"])
    rc.COMP_TOL = cli.comp_tol
    out = cli.out or os.path.join(HERE, "results_close31", f"close31_s{cli.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R = dict(cli=vars(cli), seed=cli.seed, pilot=os.path.abspath(cli.pilot_dir), arms={}, hist={}, checks={})

    def dump():
        with open(out, "w") as fh:
            json.dump(R, fh, indent=1, default=float)

    T = rc.target_setup(args, R)
    assert T is not None
    R["target"]["competitor_tol"] = cli.comp_tol
    record = rc.make_recorder(args, R, T, dump)
    lat, nup, V, E0 = T["lat"], T["nup"], T["V"], T["E0"]
    dens = 2 * nup / lat.N
    cl = rc.Cluster(rc.new_net(lat, nup, args), T["basis"], T["U"], T["HG"])
    ck = sorted({int(x) for x in cli.ckpts.split(",")})
    sign_at = {0} | set(ck[-2:])

    def ckpt_cb(name, net):
        def f(n, th):
            record(f"{name}@{n}", net, th, sign=(n in sign_at))
        return f

    # ---- tr: continue the pilot's transferred state
    th_tr0 = np.array(pil["arms"]["tr0"]["theta"])
    _, h = optimise_ckpt(cl, th_tr0, cli.steps, args.lr, args.diag_shift, args.chunk, "tr", E0, V,
                         ckpts=[0] + ck, on_ckpt=ckpt_cb("tr", cl.net), deadline=deadline)
    R["hist"]["tr"] = h
    R["checks"]["tr0_reproduced"] = abs(h[0] - pil["arms"]["tr0"]["eps"]) <= 1e-8 * max(1.0, abs(h[0]))
    dump()
    # ---- scr: the pilot's own MF init
    th0 = cl.net.init(np.random.default_rng(2000 + cli.seed), dens, args.t, args.tp_sel, args.delta0, args.eta)
    _, h = optimise_ckpt(cl, th0, cli.steps, args.lr, args.diag_shift, args.chunk, "scr", E0, V,
                         ckpts=[0] + ck, on_ckpt=ckpt_cb("scr", cl.net), deadline=deadline)
    R["hist"]["scr"] = h
    R["checks"]["scr0_reproduced"] = abs(h[0] - pil["arms"]["scrB"]["eps"]) <= 1e-8 * max(1.0, abs(h[0])) \
        if pil["params"]["budget"] == 0 else None
    dump()
    # ---- foil: upper bound on eps_foil_opt
    netf = rc.new_net(lat, nup, args, 0)
    clf = rc.Cluster(netf, T["basis"], T["U"], T["HG"])
    thf = netf.init(np.random.default_rng(2000 + cli.seed), dens, args.t, args.tp_sel, args.delta0, args.eta)
    thf, h = optimise_ckpt(clf, thf, cli.foil_steps, args.lr, args.diag_shift, args.chunk, "foil", E0, V,
                           deadline=deadline)
    R["hist"]["foil"] = h
    record("foil@end", netf, thf, sign=False)
    # ---- nfp diagnostic (R5.1 premise on this instance)
    src = rc.build_source(args, args.tp_sel, args.J_sel, pil["params"]["sources"].split(","))
    cls = [rc.Cluster(rc.new_net(s["lat"], s["nup"], args, args.channels), s["basis"],
                      rc.sp.identity(len(s["basis"]), format="csr"), s["H"]) for s in src]
    rstar = int(np.argmin(pil["source"]["route"]["restart_losses"]))
    th_init = cls[0].net.init(np.random.default_rng(1000 + 100 * cli.seed + rstar), 2 * src[0]["nup"] / src[0]["lat"].N,
                              args.t, args.tp_sel, args.delta0, args.eta)
    _, info8 = null_filter(cls, th_tr0, th_init, args.chunk, 1e-8)
    R["checks"]["n_null_reproduced"] = info8["n_null"] == pil["source"]["route"]["n_null"]
    th_nfp, info = null_filter(cls, th_tr0, th_init, args.chunk, cli.null_tol)
    info["E_src_nfp"] = [rc.reduced(c, th_nfp, args.chunk, False)[0] for c in cls]
    info["E_src_tr0"] = [rc.reduced(c, th_tr0, args.chunk, False)[0] for c in cls]
    info["E_src_exact"] = [s["E0"] for s in src]
    R["nfp_info"] = info
    rc.log(f"nfp: n_null={info['n_null']} removed shift fraction {info['frac_shift_removed']:.3f}; "
           f"source E tr0 {info['E_src_tr0']} -> nfp {info['E_src_nfp']} (exact {info['E_src_exact']})")
    _, h = optimise_ckpt(cl, th_nfp, cli.nfp_steps, args.lr, args.diag_shift, args.chunk, "nfp", E0, V,
                         ckpts=[0], on_ckpt=ckpt_cb("nfp", cl.net), deadline=deadline)
    R["hist"]["nfp"] = h
    R["valid"] = bool(R["checks"]["tr0_reproduced"] and R["checks"]["scr0_reproduced"] is not False
                      and R["checks"]["n_null_reproduced"])
    R["done"] = True
    R["wall_s"] = time.time() - t_start
    dump()
    rc.log(f"done ({out}); valid={R['valid']}")


# ============================================================================ aggregate
def decide(rs, pils, cli):
    T = rs[0]["target"]
    Lx, Ly = rc.parse_lat(pils[0]["params"]["tgt"])
    V = Lx * Ly
    E0 = T["E0"]
    H = {k: [r["hist"][k] for r in rs] for k in ("tr", "scr", "foil", "nfp")}
    eps_best = min(min(h) for k in H for h in H[k])
    relerr_best = eps_best * V / abs(E0)
    e_tr0 = [p["arms"]["tr0"]["eps"] for p in pils]
    e_r2 = [p["arms"]["tr0_r2"]["eps"] for p in pils]
    spread = max(e_tr0) - min(e_tr0)
    sec = sum(bool(p["sector_rule"]["correct"]) for p in pils)
    at = lambda hs, n: [h[n] if n < len(h) else float("nan") for h in hs]
    ck = sorted({int(x) for x in rs[0]["cli"]["ckpts"].split(",")})
    rows, counted = {}, []
    for B in ck:
        ms = med(at(H["scr"], B))
        mt = med(at(H["tr"], B))
        mov = med([h[B - 5] - h[B] if B < len(h) and B >= 5 else float("nan") for h in H["scr"]])
        P1 = ms - eps_best >= PREM_EPS
        plat_ok = mov >= cli.plateau_frac * (ms - eps_best)
        n_tr = [steps_to(h, ms) for h in H["tr"]]
        n_med = med([n if n is not None else float("inf") for n in n_tr]) if any(n is not None for n in n_tr) \
            else float("inf")
        if sum(n is None for n in n_tr) >= 2:
            n_med = float("inf")
        S = B / n_med if np.isfinite(n_med) and n_med > 0 else (float("inf") if n_med == 0 else 0.0)
        rows[B] = dict(m_scr=ms, m_tr=mt, P1=bool(P1), moving=bool(plat_ok), counted=bool(P1 and plat_ok),
                       n_tr=n_tr, speedup=S,
                       b_pass_robust=bool(all(mt - x <= 0.5 * (ms - x) for x in (0.0, eps_best))),
                       b_fail_robust=bool(all(mt - x >= 0.9 * (ms - x) for x in (0.0, eps_best))))
        if P1 and plat_ok:
            counted.append(B)
    foilK = [h[-1] for h in H["foil"]]
    F = dict(F1=bool(20 in rows and rows[20]["counted"] and rows[20]["b_fail_robust"]),
             F2=bool(med(e_tr0) >= 2 * med(foilK)),
             F3=bool(len(counted) >= 1 and all(rows[B]["speedup"] < 1 for B in counted)),
             F4=bool(spread > med(e_tr0)), F5=bool(sec < 2))
    P = dict(I=bool(spread <= 0.5 * med(e_tr0)), S=bool(sec >= 2), e=bool(med(e_tr0) < med(e_r2)),
             b20=bool(20 in rows and rows[20]["counted"] and rows[20]["b_pass_robust"]),
             speed=bool(sum(rows[B]["speedup"] >= 2 for B in counted) >= 2),
             nonvoid=bool(relerr_best <= cli.void_tol))
    if any(F.values()):
        v = "CLOSED-FAIL (" + ", ".join(k for k, x in F.items() if x) + ")"
    elif all(P.values()):
        v = "PROMOTE"
    else:
        v = "CLOSED-NOT-PROMOTED (" + ", ".join(k for k, x in P.items() if not x) + " not met)"
    diag = dict(
        A_tr0_pilot=[p["arms"]["tr0"]["A"] for p in pils], A_mfinit_pilot=[p["arms"]["scrB"]["A"] for p in pils],
        A_tr={B: [r["arms"].get(f"tr@{B}", {}).get("A") for r in rs] for B in ck[-2:]},
        A_scr={B: [r["arms"].get(f"scr@{B}", {}).get("A") for r in rs] for B in ck[-2:]},
        eps_nfp0=at(H["nfp"], 0), eps_nfp_end=[h[-1] for h in H["nfp"]], eps_scr0=at(H["scr"], 0),
        n_null_nfp=[r["nfp_info"]["n_null"] for r in rs])
    return dict(verdict=v, F=F, P=P, rows=rows, counted=counted, eps_best=eps_best, relerr_best=relerr_best,
                eps_tr0=e_tr0, eps_tr0_r2=e_r2, spread=spread, sector_correct=sec, eps_foilK=foilK, diag=diag,
                valid=[r.get("valid") for r in rs])


def aggregate(cli):
    rs = []
    for fn in sorted(os.listdir(cli.aggregate)):
        if fn.startswith("close31_s") and fn.endswith(".json"):
            r = json.load(open(os.path.join(cli.aggregate, fn)))
            if r.get("done"):
                rs.append(r)
    if len(rs) < 3:
        print(f"INCOMPLETE: {len(rs)} of 3 seeds done")
        return
    rs.sort(key=lambda r: r["seed"])
    pils = [json.load(open(os.path.join(r["pilot"], f"task{r['seed']}.json"))) for r in rs]
    if not all(r.get("valid") for r in rs):
        print(f"INVALID: consistency checks {[r['checks'] for r in rs]}")
        return
    d = decide(rs, pils, cli)
    T = rs[0]["target"]
    print(f"target t'={T['tp']} J={T['J']} irrep={T['irrep']} E0={T['E0']:.8f} gap/V={T['gap_per_site']:.4e} "
          f"competitor(<{cli.comp_tol})={T['gap_per_site'] < cli.comp_tol}  (sign half only unless flagged)")
    print(f"  pilot: eps_tr0={['%.4f' % x for x in d['eps_tr0']]} eps_tr0_r2={['%.4f' % x for x in d['eps_tr0_r2']]} "
          f"spread={d['spread']:.3f} sector-rule correct {d['sector_correct']}/3")
    print(f"  eps_best={d['eps_best']:.4e} (relerr {d['relerr_best']:.3e}; VOID-tight {cli.void_tol}); "
          f"eps_foil(K)={['%.4f' % x for x in d['eps_foilK']]} (upper bound on eps_foil_opt)")
    for B, r in d["rows"].items():
        print(f"  B={B:3d}: m eps_scr={r['m_scr']:.4e} m eps_tr={r['m_tr']:.4e} P1={r['P1']} moving={r['moving']} "
              f"counted={r['counted']} n_tr={r['n_tr']} S={r['speedup']:.2f} b-pass={r['b_pass_robust']} "
              f"b-fail={r['b_fail_robust']}")
    print(f"  FAIL clauses {d['F']}")
    print(f"  PROMOTE conditions {d['P']}")
    g = d["diag"]
    print(f"  [3.2 diagnostic] A zero-shot tr0={['%.3f' % x for x in g['A_tr0_pilot']]} vs MF init "
          f"{['%.3f' % x for x in g['A_mfinit_pilot']]}; A tr@B={g['A_tr']} scr@B={g['A_scr']}")
    print(f"  [R5.1 diagnostic] nfp eps at 0 {['%.4f' % x for x in g['eps_nfp0']]} (scr0 "
          f"{['%.4f' % x for x in g['eps_scr0']]}), at end {['%.4f' % x for x in g['eps_nfp_end']]}; n_null {g['n_null_nfp']}")
    print(f"GLOBAL Route 3.1 (TCMN transfer, critic-fixed rule): {d['verdict']}")


def parse_args(argv):
    p = argparse.ArgumentParser(description="P4.2 r5 closing run for Route 3.1 (see module docstring)")
    p.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    p.add_argument("--pilot-dir", default=os.path.join(HERE, "..", "P4.2_size_transfer_r3", "results_zeroshot"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ckpts", default="10,20,50")
    p.add_argument("--foil-steps", type=int, default=10)
    p.add_argument("--nfp-steps", type=int, default=20)
    p.add_argument("--null-tol", type=float, default=1e-8)
    p.add_argument("--wall-hours", type=float, default=3.6)
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default="")
    p.add_argument("--plateau-frac", type=float, default=0.02)
    p.add_argument("--void-tol", type=float, default=0.01)
    p.add_argument("--comp-tol", type=float, default=1e-3)
    return p.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    if cli.aggregate:
        aggregate(cli)
    else:
        run(cli)


if __name__ == "__main__":
    main()
