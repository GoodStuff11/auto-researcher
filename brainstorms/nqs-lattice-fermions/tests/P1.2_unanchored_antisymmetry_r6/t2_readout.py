#!/usr/bin/env python3
"""
P1.2 round 6, tier T2 pilot (one seed): mechanism read-outs of the trained round-5 FRD-K state on the
decisive 4x4 OBC t-t'-J instance. No training: every quantity is an exact sum over the full basis.
=====================================================================================================

Input: the round-5 FRD-K parameters of seed 11, SR shift 1e-3 (the shift the round-5 aggregate selects
for seed 11), r5 results/params/inst0_frdK_s11_sh0.001.npz, and its result JSON (stored E, dE_sign,
E_0). The instance, budget settings and P_chi projection are exactly round 5's (r5 micro_test.setup,
p11_core.budget with lbfgs_iters, fn_iters, fn_floor, r_min read from the stored JSON; the budget RNG
is re-created as in r5 train_task, so the identity read-out reproduces the stored budget).

Read-outs (psi_X = Phi * prod_e F_e with the trained Phi and alpha; all projected by P_chi):
  ident   F_e as trained (FRD-K itself)                              -> validity / positive control
  abs     |F_e| instead of F_e  (every vote flip removed, amplitudes kept)       -> task 0
  frozen  F^F_e = 1 + alpha_e g(R^F_e) with frame-frozen ratios (FRD-F, t1_frozen.py) -> task 1
  rflip   (dry run only) ident with the sign of a random 2% of |psi|^2 weight flipped -> negative control
Task 0 also reports the vote lift at V = 16 for the trained base Phi (frames K, X, H = K u X; same
definitions as t1_vote.py; U = unprojected Phi, G = P_chi Phi), which completes the Gate 4 trend of
t1_vote.py (lift_K at V = 8, 10, 12 for the UHF anchor was 2.90, 1.27, 2.05).

PRE-REGISTERED RULES (written before submission, 2026-11-16). Constants: dEs_K = stored dE_sign of the
FRD-K state (0.00975), dEs_C = round-5 3-seed median dE_sign of the control nnbf6 (0.012045), gap =
dEs_C - dEs_K (0.0023).
  T2-a (task 0, FRD-K mechanism): vote share v = (dEs_abs - dEs_K) / gap.
       VOTE-INERT if v < 0.25: the flips carry under a quarter of FRD-K's sign gain over the control;
         FRD-K's novelty claim (a sign-capable vote) fails whatever job 553724 returns, and the gain is
         credited to amplitude off-loading (route AO-F opens, FRD-F loses its rationale).
       VOTE-CARRIES if v >= 0.5: the vote mechanism is confirmed; AO-F is killed.
       MIXED otherwise: both stay live; the 5-seed read-out (all seeds of 553724) decides.
  T2-b (task 1, FRD-F): u = (dEs_frozen - dEs_K) / gap.
       FRD-F DROP-IN-PASS if u <= 0.25 (frozen ratios keep at least 75% of the gain, at NNBF cost);
       FRD-F KILLED if u >= 1 (the frozen read-out loses the whole gain);
       AMBIGUOUS otherwise: FRD-F needs its own training (a T3 arm frdF beside frdK, pre-planned).
  Gate 4 lift (task 0, reported with a rule): lift_K(V = 16, U) < 1.2 -> FRD-K's vote is FINITE-SIZE
       (Gate 4 FAIL); >= 1.5 -> EXTENSIVE-OK; else UNDECIDED. (The base differs from the toy anchor:
       the trained NNBF-4 base, not UHF. Recorded as a caveat.)
  Validity (both tasks): the identity read-out must reproduce the stored projected energy to 1e-8 and
       (dry run only, where it is cheap) the stored dE_sign to 1e-6; the ED E_0 must equal the stored E_0
       to 1e-9, and the parameter count must be 1719. Any failure -> INVALID.
  Early stop: no training, so no futility checkpoints. Each stage writes its partial JSON (anytime
       reporting); a timeout leaves INCOMPLETE, not a verdict.

Pre-flight dry run (--mode dryrun, 4x2 OBC 3+3, instance 8, round-5 smoke FRD-K parameters seed 11
shift 1e-3): runs ident, abs, frozen and rflip with the synthetic dry-run constant dEs_C = dEs_K + 0.02
(the toy FRD-K has no gain over its control, so the real median would leave the ratios undefined);
the controls must give their intended verdicts: ident -> VOTE-INERT (v ~ 0), rflip -> VOTE-CARRIES
(v >= 0.5); then the aggregate is run on synthetic results covering every branch.

Usage
-----
  python3 t2_readout.py --mode task --task {0,1}       production (4x4, 1681680 states; Slurm only)
  python3 t2_readout.py --mode dryrun                    toy pre-flight (seconds)
  python3 t2_readout.py --mode aggregate [--results DIR] prints the verdicts from task JSONs
  python3 t2_readout.py --mode synthetic                 aggregate on synthetic numbers (every branch)
Options
-------
  --task {0,1}   0: ident + abs + V = 16 lift; 1: ident (energy only) + frozen.
  --results DIR  output folder (default results/ next to this script).
  --chunk INT    configurations per chunk (default 65536).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r5")
sys.path.insert(0, R5)
sys.path.insert(0, HERE)
from p11_core import budget, energy, solve_ed  # noqa: E402
from tj_core import LRAnsatz  # noqa: E402
import p21_harness as H21  # noqa: E402
import micro_test as R5T  # noqa: E402
from t1_frozen import frozen_factors  # noqa: E402
from t1_vote import vote  # noqa: E402

PROD = dict(inst=0, params=os.path.join(R5, "results", "params", "inst0_frdK_s11_sh0.001.npz"),
            json=os.path.join(R5, "results", "inst0_frdK_s11_sh0.001.json"), dEs_C=0.012045398459125778,
            n_params=1719, base_Hs=4)
DRY = dict(inst=8, params=os.path.join(R5, "results_smoke", "params", "inst8_frdK_s11_sh0.001.npz"),
           json=os.path.join(R5, "results_smoke", "inst8_frdK_s11_sh0.001.json"), dEs_C=None, n_params=356,
           base_Hs=2)
V_INERT, V_CARRY, U_PASS, U_KILL, LIFT_FAIL, LIFT_OK = 0.25, 0.5, 0.25, 1.0, 1.2, 1.5


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r6 T2 read-outs (see docstring)")
    ap.add_argument("--mode", choices=["task", "dryrun", "aggregate", "synthetic"], required=True)
    ap.add_argument("--task", type=int, choices=[0, 1], default=0)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--chunk", type=int, default=65536)
    return ap.parse_args(argv)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(cfg):
    m = R5T.setup(cfg["inst"])
    st = json.load(open(cfg["json"]))
    m.pair_setup(st["params"]["r_min"])
    V = m.V
    base = LRAnsatz(m, (np.zeros((V, m.nup)), np.zeros((V, m.ndn))), "full", K=1, Hs=cfg["base_Hs"], seed=0)
    flat = np.load(cfg["params"])["flat"]
    E = len(R5T.edge_list(m.T, m.nn, "K"))
    nb = base.n_params()
    w = (len(flat) - nb - E) // (2 * V + 1 + E)
    frd = R5T.FRD(m, base, "K", w, 0)
    assert frd.n_params() == len(flat) == cfg["n_params"], (frd.n_params(), len(flat))
    frd.set_flat(flat)
    frd.tables()
    return m, st, frd


def readout(frd, kind, chunk, rng=None):
    m = frd.m
    phi = frd.phi(chunk)
    out = np.empty(m.D)
    for s in range(0, m.D, chunk):
        idx = np.arange(s, min(s + chunk, m.D))
        if kind == "frozen":
            P = frozen_factors(frd, m.bu[idx], m.bd[idx], "sm")[0]
        else:
            P = np.prod(frd._fac_idx(idx, phi)["F"], 1)
            if kind == "abs":
                P = np.abs(P)
        out[idx] = phi[idx] * P
    return out


def run(cfg, which, args, tag):
    os.makedirs(args.results, exist_ok=True)
    path = os.path.join(args.results, f"t2_{tag}.json")
    t0 = time.time()
    m, st, frd = load_state(cfg)
    bargs = SimpleNamespace(**st["params"])
    W, Vv, deg = solve_ed(m, 4, 1e-8, H21.UHF_SEED)
    psi0 = H21.resolve_ground(m, Vv, deg)
    sg, gen = H21.build_symmetry(m, psi0)
    res = dict(tag=tag, inst=cfg["inst"], E0=float(W[0]), E0_stored=st["ed_levels"][0], irrep=gen,
               n_params=frd.n_params(), dEs_K_stored=st["budgets"]["P"]["dE_sign"], dEs_C=cfg["dEs_C"],
               E_stored=st["budgets"]["P"]["E"])
    log(f"loaded inst {cfg['inst']} D={m.D} |G|={sg.G} ED {W[0]:.10f} (stored {st['ed_levels'][0]:.10f}) "
        f"({time.time() - t0:.0f} s)")
    json.dump(res, open(path, "w"), indent=1, default=float)
    states = {}
    for kind in [k for k in which if k != "lift"]:
        t1 = time.time()
        if kind == "rflip":
            v = readout(frd, "ident", args.chunk)
            p = v ** 2 / (v @ v)
            r = np.random.default_rng(9)
            pick = np.zeros(m.D, bool); o = r.permutation(m.D); c = np.cumsum(p[o])
            pick[o[:np.searchsorted(c, 0.02) + 1]] = True
            v = np.where(pick, -v, v)
        else:
            v = readout(frd, kind, args.chunk)
        vc = sg.project(v)
        Ev = energy(m.H, vc)[0]
        res[f"E_{kind}"] = float(Ev)
        states[kind] = (v, vc)
        need_budget = kind != "ident" or args.mode == "dryrun"   # production: ident checks the energy only
        if need_budget:
            rng_b = np.random.default_rng(st["seed"] + 1000003)
            rng_b.random(m.D)   # consumed by the r5 calibration, so the budget start matches round 5
            b = budget(m, vc, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, bargs, rng_b)
            res[f"dEs_{kind}"] = float(b["dE_sign"]); res[f"dEt_{kind}"] = float(b["dE_tot"])
            res[f"spread_{kind}"] = float(b["spread"]); res[f"wsign_{kind}"] = float(b["w_sign"])
        log(f"{kind}: E={Ev:.8f}" + (f" dE_sign={res[f'dEs_{kind}']:.6f} dE_tot={res[f'dEt_{kind}']:.6f}"
                                    if f"dEs_{kind}" in res else "") + f" ({time.time() - t1:.0f} s)")
        json.dump(res, open(path, "w"), indent=1, default=float)
    if "lift" in which or args.mode == "dryrun":
        t1 = time.time()
        phi = frd.phi(args.chunk)
        pz = psi0 ** 2 / (psi0 @ psi0)
        s0 = np.where(psi0 >= 0, 1.0, -1.0)
        tabs = {}
        for F in ("K", "X"):
            f = R5T.FRD(m, frd.base, F, 1, 0)
            Mi, C = f.tables()
            tabs[F] = (Mi.astype(np.int64), C.astype(float) * f.scale)
        tabs["H"] = (np.concatenate([tabs["K"][0], tabs["X"][0]], 1), np.concatenate([tabs["K"][1], tabs["X"][1]], 1))
        for lab, ref in (("U", phi), ("G", sg.project(phi))):
            s_b = np.where(ref >= 0, 1.0, -1.0)
            if pz[s_b == s0].sum() < 0.5:
                s_b, ref = -s_b, -ref
            wr = s_b != s0
            for F in ("K", "X", "H"):
                eF, _ = vote(ref, *tabs[F])
                D = eF > 0
                wb, pD = pz[wr].sum(), pz[D].sum()
                res[f"lift_{F}_{lab}"] = float((pz[D & wr].sum() / pD) / wb) if pD > 0 and wb > 0 else float("nan")
                res[f"recall_{F}_{lab}"] = float(pz[D & wr].sum() / wb) if wb > 0 else float("nan")
            res[f"w_b_{lab}"] = float(pz[wr].sum())
        log("lift: " + " ".join(f"{k}={v:.3f}" for k, v in res.items() if k.startswith(("lift", "recall", "w_b")))
            + f" ({time.time() - t1:.0f} s)")
    res["seconds"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1, default=float)
    return res


def verdicts(r):
    out = {}
    ok = (abs(r["E0"] - r["E0_stored"]) < 1e-9 and r["n_params"] in (1719, 356)
          and "E_ident" in r and abs(r["E_ident"] - r.get("E_stored", r["E_ident"])) < 1e-8)
    if "dEs_ident" in r:
        ok &= abs(r["dEs_ident"] - r["dEs_K_stored"]) < 1e-6
    out["valid"] = bool(ok)
    dK = r.get("dEs_ident", r["dEs_K_stored"])
    gap = r["dEs_C"] - dK
    out["gap"] = gap
    if gap <= 0:
        out["note"] = "no gain over the control: read-outs undefined"
        return out
    for kind in ("abs", "rflip", "ident"):
        if f"dEs_{kind}" in r:
            v = (r[f"dEs_{kind}"] - dK) / gap
            out[f"v_{kind}"] = v
            out[f"T2a_{kind}"] = "VOTE-INERT" if v < V_INERT else ("VOTE-CARRIES" if v >= V_CARRY else "MIXED")
    if "dEs_frozen" in r:
        u = (r["dEs_frozen"] - dK) / gap
        out["u_frozen"] = u
        out["T2b"] = "DROP-IN-PASS" if u <= U_PASS else ("KILLED" if u >= U_KILL else "AMBIGUOUS")
    if "lift_K_U" in r:
        lk = r["lift_K_U"]
        out["gate4_lift"] = "FINITE-SIZE" if lk < LIFT_FAIL else ("EXTENSIVE-OK" if lk >= LIFT_OK else "UNDECIDED")
    if not ok:
        out = {"valid": False, "note": "INVALID: validity assert failed", **{k: v for k, v in out.items() if k != "valid"}}
    return out


def aggregate(d):
    allv = {}
    for tag in ("task0", "task1", "dryrun"):
        p = os.path.join(d, f"t2_{tag}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            v = verdicts(r)
            allv[tag] = v
            print(f"{tag}: " + json.dumps(v, default=float))
    return allv


def synthetic():
    base = dict(E0=-1.0, E0_stored=-1.0, n_params=1719, E_ident=-0.9, dEs_K_stored=0.010, dEs_ident=0.010,
                dEs_C=0.012)
    cases = {"abs inert": dict(dEs_abs=0.0102), "abs mixed": dict(dEs_abs=0.0108),
             "abs carries": dict(dEs_abs=0.0115), "frozen pass": dict(dEs_frozen=0.0104),
             "frozen ambiguous": dict(dEs_frozen=0.0111), "frozen killed": dict(dEs_frozen=0.0125),
             "lift finite": dict(lift_K_U=1.1), "lift undecided": dict(lift_K_U=1.3), "lift ok": dict(lift_K_U=1.7),
             "invalid E0": dict(E0=-0.99), "invalid budget": dict(dEs_ident=0.0101, dEs_abs=0.0102),
             "no gain": dict(dEs_C=0.009, dEs_abs=0.0102)}
    for name, c in cases.items():
        r = dict(base); r.update(c)
        print(f"synthetic {name:18s}: " + json.dumps(verdicts(r), default=lambda x: round(float(x), 4)))


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "synthetic":
        synthetic(); return 0
    if args.mode == "aggregate":
        aggregate(args.results); return 0
    if args.mode == "dryrun":
        cfg = dict(DRY)
        # the toy has no FRD gain over its control (smoke frdK dE_sign 0.036 against nnbf6 0.001), so the
        # dry run uses a synthetic control level dEs_C = stored dEs_K + 0.02 to give the read-out ratios a
        # positive denominator; it tests the pipeline and the controls, not the physics.
        cfg["dEs_C"] = json.load(open(cfg["json"]))["budgets"]["P"]["dE_sign"] + 0.02
        res = run(cfg, ["ident", "abs", "frozen", "rflip"], args, "dryrun")
        v = verdicts(res)
        print("DRYRUN verdicts: " + json.dumps(v, default=float))
        intended = (v.get("valid") and v.get("T2a_ident") == "VOTE-INERT" and v.get("T2a_rflip") == "VOTE-CARRIES")
        print(f"DRYRUN controls give intended verdicts: {bool(intended)}")
        synthetic()
        return 0 if intended else 2
    cfg = dict(PROD)
    which = ["ident", "abs", "lift"] if args.task == 0 else ["ident", "frozen"]
    run(cfg, [w for w in which if w != "lift"] + (["lift"] if "lift" in which else []), args, f"task{args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
