#!/usr/bin/env python3
"""
P1.2 round 6 critic, tier T2 (one seed, no training): three further read-outs of the trained round-5
FRD-K state (seed 11, shift 1e-3) on the 4x4 OBC t-t'-J instance, which decide the premises the
critic found open after job 697584. Same state, instance, budget and RNG as t2_readout.py (imported).
=====================================================================================================

Read-outs (all projected by P_chi, budgets by p11_core.budget as in round 5):
  base     Phi alone (every factor set to 1; the co-adapted NNBF-4 base)        -> task 0
  absfroz  Phi * prod_e |1 + alpha_e g(R^F_e)| (frozen-frame ratios, every flip removed): the AO-F
           drop-in with the trained alpha                                       -> task 1
  gvote    psi_G = P_chi psi_ident with its sign flipped wherever the hole-hop vote of psi_G fires,
           e_K(n) = sum_{e in K} H_{n m_e} psi_G(m_e) / psi_G(n) > 0 (t1_vote.vote; precision 0.80 and
           recall 0.48 measured on P_chi Phi in job 697584)                     -> task 2
Constants as in t2_readout.py: dEs_K = 0.009751 (stored FRD-K), dEs_C = 0.012045 (round-5 3-seed nnbf6
median), gap = dEs_C - dEs_K.

PRE-REGISTERED RULES (written 2026-11-16, before submission):
  base   : b = (dEs_base - dEs_K)/gap.  BASE-CARRIES if b < 0.25 (the sign gain sits in the co-adapted
           base's own nodes: AO mechanism 2, which any positive factor offers, so nnbfJ is the decisive
           foil); REFRAME-CARRIES if b >= 1 (the positive amplitudes |F| reshape the projected nodes:
           AO mechanism 1, a point-group projection effect, so AO needs a three-size Gate 4 trend before
           any T3); MIXED otherwise.
  absfroz: u' = (dEs_absfroz - dEs_K)/gap. DROP-IN-OK if u' <= 0.25, DROP-IN-LOST if u' >= 1, else
           PARTIAL. A read-out is not a retrain: DROP-IN-LOST does not kill AO-F, it orders the arms of
           test A-C (aoK before aoF); DROP-IN-OK lets A-C start aoF from the read-out parameters.
  gvote  : z = (dEs_gvote - dEs_K)/gap. VOTE-USABLE if z <= -0.5 (the projected vote applied without
           any gate removes at least half a gap of sign error that SR left in place: supports the
           worker's "barrier (b) blocks it" reading and reopens a trainable-vote route);
           VOTE-HARMFUL if z >= 0 (the reading is rejected: even at precision 0.8 the false alarms
           outweigh the repairs); UNRESOLVED otherwise. Diagnostic only, it is not a route (Route 6.8).
  Validity (every task): E_0 equals the stored E_0 to 1e-9 and the identity energy equals the stored
           energy to 1e-8 (as in t2_readout.py); else INVALID.
Pre-flight dry run (--mode dryrun): the toy round-5 smoke state (instance 8), all three read-outs, with
the synthetic gap of t2_readout.py's dry run, plus controls: base with alpha set to 0 must equal ident (to 1e-12);
the number of configurations where the hole-hop vote fires on psi_0 is reported (the K frame alone
may fire on an eigenstate; only the full frame H never does).

Usage:
  python3 t2c_readout.py --mode task --task {0,1,2}    production (Slurm only)
  python3 t2c_readout.py --mode dryrun                  toy pre-flight (seconds)
  python3 t2c_readout.py --mode aggregate [--results DIR]
Options: --task {0,1,2} (0 base, 1 absfroz, 2 gvote); --results DIR (default results/); --chunk INT.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_readout as T2  # noqa: E402  (sets sys.path for the r5 modules)
from p11_core import budget, energy, solve_ed  # noqa: E402
import p21_harness as H21  # noqa: E402
import micro_test as R5T  # noqa: E402
from t1_frozen import frozen_factors  # noqa: E402
from t1_vote import vote  # noqa: E402

KINDS = {0: "base", 1: "absfroz", 2: "gvote"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r6 critic T2c read-outs (see docstring)")
    ap.add_argument("--mode", choices=["task", "dryrun", "aggregate"], required=True)
    ap.add_argument("--task", type=int, choices=[0, 1, 2], default=0)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--chunk", type=int, default=65536)
    return ap.parse_args(argv)


def vector(frd, sg, kind, chunk):
    m = frd.m
    if kind == "base":
        return frd.phi(chunk).copy()
    if kind == "absfroz":
        phi = frd.phi(chunk)
        out = np.empty(m.D)
        for s in range(0, m.D, chunk):
            idx = np.arange(s, min(s + chunk, m.D))
            out[idx] = phi[idx] * np.abs(frozen_factors(frd, m.bu[idx], m.bd[idx], "sm")[0])
        return out
    psiG = sg.project(T2.readout(frd, "ident", chunk))
    f = R5T.FRD(m, frd.base, "K", 1, 0)
    Mi, C = f.tables()
    eK, _ = vote(psiG, Mi.astype(np.int64), C.astype(float) * f.scale)
    return np.where(eK > 0, -psiG, psiG), float((psiG[eK > 0] ** 2).sum() / (psiG @ psiG))


def run(cfg, kinds, args, tag):
    os.makedirs(args.results, exist_ok=True)
    path = os.path.join(args.results, f"t2c_{tag}.json")
    t0 = time.time()
    m, st, frd = T2.load_state(cfg)
    bargs = SimpleNamespace(**st["params"])
    W, Vv, deg = solve_ed(m, 4, 1e-8, H21.UHF_SEED)
    psi0 = H21.resolve_ground(m, Vv, deg)
    sg, gen = H21.build_symmetry(m, psi0)
    E_ident = energy(m.H, sg.project(T2.readout(frd, "ident", args.chunk)))[0]
    res = dict(tag=tag, E0=float(W[0]), E0_stored=st["ed_levels"][0], E_ident=float(E_ident),
               E_stored=st["budgets"]["P"]["E"], dEs_K=st["budgets"]["P"]["dE_sign"], dEs_C=cfg["dEs_C"])
    T2.log(f"loaded D={m.D} E0 {W[0]:.10f} (stored {st['ed_levels'][0]:.10f}); E_ident {E_ident:.10f} "
           f"(stored {res['E_stored']:.10f})")
    if args.mode == "dryrun":
        f = R5T.FRD(m, frd.base, "K", 1, 0)
        Mi, C = f.tables()
        e0, _ = vote(psi0, Mi.astype(np.int64), C.astype(float) * f.scale)
        res["ctrl_vote_psi0_flags"] = int((e0 > 0).sum())
        q = {k: v.copy() for k, v in frd.q.items()}
        frd.q["W2a"][:] = 0; frd.q["b2a"][:] = 0
        res["ctrl_base_eq_ident_alpha0"] = float(np.max(np.abs(T2.readout(frd, "ident", args.chunk)
                                                              - frd.phi(args.chunk))))
        frd.q = q
        T2.log(f"controls: vote(psi0) flags {res['ctrl_vote_psi0_flags']} (reported); base==ident at alpha=0: {res['ctrl_base_eq_ident_alpha0']:.1e}")
    for kind in kinds:
        t1 = time.time()
        out = vector(frd, sg, kind, args.chunk)
        if kind == "gvote":
            v, res["gvote_flag_weight"] = out
            vc = v
        else:
            vc = sg.project(out)
        rng_b = np.random.default_rng(st["seed"] + 1000003)
        rng_b.random(m.D)
        b = budget(m, vc, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, bargs, rng_b)
        res[f"E_{kind}"] = float(energy(m.H, vc)[0])
        res[f"dEs_{kind}"] = float(b["dE_sign"]); res[f"dEt_{kind}"] = float(b["dE_tot"])
        res[f"wsign_{kind}"] = float(b["w_sign"]); res[f"spread_{kind}"] = float(b["spread"])
        T2.log(f"{kind}: E={res[f'E_{kind}']:.8f} dE_sign={b['dE_sign']:.6f} dE_tot={b['dE_tot']:.6f} "
               f"w_sign={b['w_sign']:.2e} ({time.time() - t1:.0f} s)")
        json.dump(res, open(path, "w"), indent=1, default=float)
    res["seconds"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1, default=float)
    return res


def verdicts(r):
    ok = abs(r["E0"] - r["E0_stored"]) < 1e-9 and abs(r["E_ident"] - r["E_stored"]) < 1e-8
    gap = r["dEs_C"] - r["dEs_K"]
    out = dict(valid=bool(ok), gap=gap)
    if "dEs_base" in r:
        b = (r["dEs_base"] - r["dEs_K"]) / gap
        out.update(b=b, base="BASE-CARRIES" if b < 0.25 else ("REFRAME-CARRIES" if b >= 1 else "MIXED"))
    if "dEs_absfroz" in r:
        u = (r["dEs_absfroz"] - r["dEs_K"]) / gap
        out.update(u_abs=u, absfroz="DROP-IN-OK" if u <= 0.25 else ("DROP-IN-LOST" if u >= 1 else "PARTIAL"))
    if "dEs_gvote" in r:
        z = (r["dEs_gvote"] - r["dEs_K"]) / gap
        out.update(z=z, gvote="VOTE-USABLE" if z <= -0.5 else ("VOTE-HARMFUL" if z >= 0 else "UNRESOLVED"))
    if not ok:
        out["note"] = "INVALID"
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "aggregate":
        for tag in ("task0", "task1", "task2", "dryrun"):
            p = os.path.join(args.results, f"t2c_{tag}.json")
            if os.path.exists(p):
                print(f"{tag}: " + json.dumps(verdicts(json.load(open(p))), default=float))
        return 0
    if args.mode == "dryrun":
        cfg = dict(T2.DRY)
        cfg["dEs_C"] = json.load(open(cfg["json"]))["budgets"]["P"]["dE_sign"] + 0.02
        r = run(cfg, ["base", "absfroz", "gvote"], args, "dryrun")
        v = verdicts(r)
        print("DRYRUN verdicts: " + json.dumps(v, default=float))
        ok = v["valid"] and r["ctrl_base_eq_ident_alpha0"] < 1e-12 and all(k in v for k in ("base", "absfroz", "gvote"))
        print(f"DRYRUN pipeline and controls OK: {bool(ok)}")
        return 0 if ok else 2
    run(dict(T2.PROD), [KINDS[args.task]], args, f"task{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
