#!/usr/bin/env python3
"""
P3.1 Round 10: T2-v2 premise run for Route 10.1u (self-consistent response Ritz, base updated each pass)
========================================================================================================

Replaces the Round 9 T2 (submit_t2_resp.sbatch, not submitted) as specified by the Round 9 per-leaf critic: one
task under the 2 CPU-hour cap, exact (noise-free) responses on the real converged width-6 projected NNBF states of
the 4x4 OBC 6+6 instance (D = 1681680), seeds 11, 22, 33 (P2.1 r3 parameters; the targets of job 698163).
Machinery (instance, symmetry projector P_chi, target loading, minres solves, Ritz) is imported unchanged from
../P3.1_order_parameter_error_r9/t2_resp_premise.py (class Ctx).

Arms (lowest Ritz vector of H; ratio = (a(phi) - a(psi)) / delta on the tails pair / spin / charge):
  L1R        span{psi, r}                                                           blind control
  RWRstarK   span{psi, r, y*_A, K y*_A}, y*_A = (H - E_0)^{-1} Q_0 (A - a_0) psi_0 oracle ceiling (class premise)
  SCRu0      phi_0 = psi; phi_{t+1} = Ritz of span{phi_t, (H-E)phi_t, y_A(phi_t), K y_A(phi_t)}; T = 2 and 4
  RWRK0      = SCRu0 at T = 1 (attribution; free)
  SCRd0      pass 1 as SCRu0; later passes span{phi_t, y_A(phi_t)}; T = 4                (reported)
  Response y_A(phi) = (Q(H - s)Q)^{-1} Q P_chi (A - a_phi) phi, Q = P_chi - |phi><phi|, at two shifts:
    "decide"   s = E_L1R(phi) - 0.1 t   (deciding; a trained response can use it, critic objection 4)
    "default"  s = 2 E_L1R(phi) - E_phi (rounds 8-10 convention; reported only)
  For every response the run prints (s - E_0)/omega_1 (omega_1 = 0.2089 t from job 698163) and
  |<psi_0|unit(y)>|^2.

Phases (cheapest decision first; rows are written to --out after every arm):
  1  ED; load the three targets; validity: delta must reproduce job 698163 within 1e-3 relative for every item,
     else INVALID and stop before any minres solve (critic minor objection 3).
  2  L1R for all seeds; y*_A (3 solves, seed-independent); RWRstarK for all seeds.
     EARLY STOP (class KILL): if RWRstarK tracks fewer than 6 of the 8 scored items, stop.
  3  SCRu0 at the deciding shift, seeds 11, 22, 33 in that order (12 solves per seed). Seed 11's CPU time is
     printed; if it exceeds 40 CPU-minutes, stop and re-plan (critic). FUTILITY: after each seed, if the tracked
     count so far plus the number of remaining scored items is below 4, stop with KILL.
  4  Reported arms, if CPU remains (soft guard: an arm starts only if its projected cost fits the --cpu-budget):
     SCRd0 (decide), SCRu0 (default shift).
Scored items: |delta| >= 0.10 |a_0| (job 698163: pair x3, spin seeds 11 and 22, charge x3 = 8). Tracked: [1/2, 2].

Pre-registered rule (critic, Round 9 per-leaf critic, "T2 decision"; copied into the leaf before submission):
  KILL the response class  if RWRstarK tracks fewer than 6 of 8 items.
  PASS 10.1u               if SCRu0 (T = 4, decide shift) tracks >= 6 of 8, including >= 1 spin item.
  KILL 10.1u               if it tracks fewer than 4 of 8.
  AMBIGUOUS                otherwise.
  The effective sample is about 3 states (critic minor objection 2).

Usage
  python3 t2v2_scru_premise.py --instance main --out results/t2v2_main.json        (the Slurm task)
  python3 t2v2_scru_premise.py --instance small --out results/t2v2_dry.json        (pre-flight dry run)
  python3 t2v2_scru_premise.py --selftest-verdict                                  (every verdict branch)
  python3 t2v2_scru_premise.py --aggregate results/t2v2_main.json                  (verdict from the JSON)
Options
  --instance {main, small}  main: 4x4 OBC 6+6, seeds 11, 22, 33 (real states).
                            small: 4x4 OBC 2+2 (D = 10920), machinery only: "seed" 0 = positive control
                            psi_0 + 0.05 e (first-order error; the oracle and SCRu0 must track it), "seed" 1 =
                            negative control unit(P_chi psi_0 exp(0.8 xi)) (non-first-order; reported); no
                            validity check (no job-698163 reference on this instance).
  --chunk INT               amplitude-evaluation chunk size (default 32768).
  --cpu-budget FLOAT        CPU-seconds available to this process (default 6900 = 2 CPU-h minus margin).
  --wall-budget FLOAT       wall seconds available (default 3300 = the 1 h Slurm limit minus margin); the reported
                            arms of phase 4 start only if both budgets allow (deciding phases are never skipped).
  --out PATH                JSON output (rewritten after every arm).
  Note: (s - E_0)/omega_1 uses omega_1 = 0.2089 t of the main instance; on --instance small it is only a label.
  --aggregate PATH          print the verdict from an output JSON and exit.
  --selftest-verdict        synthetic records covering every verdict branch.
"""
import argparse
import importlib.util
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R9 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r9")
_spec = importlib.util.spec_from_file_location("t2r", os.path.join(R9, "t2_resp_premise.py"))
t2r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t2r)
r6, rc, log, P21, REF = t2r.r6, t2r.rc, t2r.log, t2r.P21, t2r.REF
REAL = ("pair", "spin", "charge")
OMEGA1 = 0.2089
SEEDS = {"main": (11, 22, 33), "small": (0, 1)}


def unit(v):
    return v / np.linalg.norm(v)


def cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def tracked(x):
    return x is not None and 0.5 <= x <= 2.0


def verdict(rec):
    """rec: {seed: {obs: {scored, ratios{arm: x}}}}. Returns (verdict, details)."""
    items = [(s, k, rec[s][k]) for s in rec for k in REAL if rec[s][k]["scored"]]
    n = len(items)
    cnt = {a: sum(1 for _, _, d in items if tracked(d["ratios"].get(a))) for a in ("RWRstarK", "SCRu0_T4")}
    spin = sum(1 for _, k, d in items if k == "spin" and tracked(d["ratios"].get("SCRu0_T4")))
    need6, need4 = 0.75 * n, 0.5 * n          # 6 and 4 of 8; scaled if the item count differs
    if cnt["RWRstarK"] < need6:
        v = "KILL (response class: oracle ceiling)"
    elif cnt["SCRu0_T4"] >= need6 and spin >= 1:
        v = "PASS"
    elif cnt["SCRu0_T4"] < need4:
        v = "KILL (10.1u)"
    else:
        v = "AMBIGUOUS"
    return v, dict(n_items=n, tracked=cnt, spin_tracked_SCRu0_T4=spin)


def selftest():
    def rec(vals):   # vals: list of (seed, obs, scored, oracle, scru)
        out = {}
        for s, k, sc, o, u in vals:
            out.setdefault(s, {kk: dict(scored=False, ratios={}) for kk in REAL})
            out[s][k] = dict(scored=sc, ratios={"RWRstarK": o, "SCRu0_T4": u})
        return out
    base = [(11, "pair", True), (11, "spin", True), (11, "charge", True), (22, "pair", True), (22, "spin", True),
            (22, "charge", True), (33, "pair", True), (33, "charge", True)]
    cases = {
        "PASS": ([1.0] * 8, [1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1], "PASS"),
        "AMBIGUOUS (no spin)": ([1.0] * 8, [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0], "AMBIGUOUS"),
        "AMBIGUOUS (5 of 8)": ([1.0] * 8, [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1], "AMBIGUOUS"),
        "KILL 10.1u": ([1.0] * 8, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1], "KILL (10.1u)"),
        "KILL class": ([1.0] * 5 + [0.1] * 3, [1.0] * 8, "KILL (response class: oracle ceiling)"),
    }
    ok = True
    for lab, (o, u, want) in cases.items():
        v, det = verdict(rec([(s, k, sc, oo, uu) for (s, k, sc), oo, uu in zip(base, o, u)]))
        ok &= v == want
        print(f"selftest {lab:22s} expected {want:40s} got {v:40s} {det}")
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


class Run:
    def __init__(self, args):
        self.a = args
        self.t0 = time.time()
        m, E0, psi0, sg, S0, ob = r6.setup(args.instance)
        psi0 = psi0 if psi0.sum() > 0 else -psi0
        if args.instance == "main":
            assert m.D == 1681680 and S0 == 0, (m.D, S0)
        self.m, self.E0, self.psi0, self.sg = m, E0, psi0, sg
        self.C = t2r.Ctx(m, sg, ob)
        self.a0 = {k: float(psi0 @ self.C.apply(k, psi0)) for k in REAL}
        self.out = dict(args=vars(args), E0=E0, a0=self.a0, omega1=OMEGA1, seeds={}, shifts=[], cpu={})
        self.psi = {}

    def dump(self):
        self.out["cpu"]["total"] = cpu()
        json.dump(self.out, open(self.a.out, "w"), indent=1, default=float)

    def target(self, seed):
        C, m, psi0 = self.C, self.m, self.psi0
        if self.a.instance == "main":
            f = os.path.join(P21, f"inst0_nnbf_s{seed}.npz")
            assert os.path.exists(f), f
            ans = rc.load_params(rc.nnbf(m, 6, seed), f)
            psi = unit(self.sg.project(ans.amplitudes(self.a.chunk)))
        elif seed == 0:
            w = self.sg.project(C.apply("spin", psi0) + C.apply("charge", psi0))
            w -= psi0 * (psi0 @ w)
            psi = unit(psi0 + 0.05 * unit(-C.solve(w, self.E0, psi0)))
        else:
            rng = np.random.default_rng(5)
            psi = unit(self.sg.project(psi0 * np.exp(0.8 * rng.standard_normal(m.D))))
        return psi if psi @ psi0 > 0 else -psi

    def responses(self, phi, mode, tag):
        C, m = self.C, self.m
        phi = unit(phi)
        E = float(phi @ (m.H @ phi))
        _, EL1 = C.ritz_vec([phi, m.H @ phi - E * phi])
        s = EL1 - 0.1 if mode == "decide" else 2.0 * EL1 - E
        ys = []
        for k in REAL:
            a = float(phi @ C.apply(k, phi))
            ys.append(unit(C.solve(C.apply(k, phi) - a * phi, s, phi)))
        ov = [float((self.psi0 @ y) ** 2) for y in ys]
        rec = dict(tag=tag, mode=mode, s_minus_E0_over_omega1=(s - self.E0) / OMEGA1, psi0_weight=ov)
        self.out["shifts"].append(rec)
        log(f"    response {tag} [{mode}] (s-E0)/omega1 {rec['s_minus_E0_over_omega1']:+.3f} "
            f"|<psi0|y>|^2 " + " ".join(f"{x:.3f}" for x in ov))
        return ys

    def score(self, seed, arm, phi):
        d = self.out["seeds"][str(seed)]
        for k in REAL:
            d[k]["ratios"][arm] = float(phi @ self.C.apply(k, phi) - d[k]["a_psi"]) / d[k]["delta"]
        log(f"  seed {seed} {arm:12s} " + " ".join(f"{k} {d[k]['ratios'][arm]:+.3f}" for k in REAL)
            + f"  (wall {time.time() - self.t0:.0f} s, cpu {cpu():.0f} s)")
        self.dump()

    def rec(self):
        return {s: d for s, d in self.out["seeds"].items()}

    def run(self):
        a, C, m = self.a, self.C, self.m
        seeds = SEEDS[a.instance]
        # ---------------- phase 1: targets and validity
        ref = json.load(open(REF))["seeds"] if a.instance == "main" else {}
        for s in seeds:
            psi = self.target(s)
            self.psi[s] = psi
            ap = {k: float(psi @ C.apply(k, psi)) for k in REAL}
            dl = {k: self.a0[k] - ap[k] for k in REAL}
            self.out["seeds"][str(s)] = {k: dict(a_psi=ap[k], delta=dl[k], ratios={},
                                                 scored=bool(abs(dl[k]) >= 0.10 * abs(self.a0[k]))) for k in REAL}
            self.out["seeds"][str(s)]["overlap2"] = float((psi @ self.psi0) ** 2)
            log(f"seed {s}: overlap^2 {(psi @ self.psi0) ** 2:.4f} | " + " ".join(
                f"{k} delta {dl[k]:+.3e} {'scored' if abs(dl[k]) >= 0.1 * abs(self.a0[k]) else 'ns'}" for k in REAL))
            if a.instance == "main":
                dev = {k: abs(dl[k] - ref[str(s)][k]["delta"]) / abs(ref[str(s)][k]["delta"]) for k in REAL}
                ok = all(x <= 1e-3 for x in dev.values())
                log(f"VALIDITY seed {s} vs job 698163: rel dev {dev} -> {'OK' if ok else 'INVALID'}")
                if not ok:
                    self.out["verdict"] = "INVALID"
                    self.dump()
                    return 1
        n_sc = sum(1 for s in seeds for k in REAL if self.out["seeds"][str(s)][k]["scored"])
        log(f"scored items: {n_sc}")
        self.dump()
        # ---------------- phase 2: L1R, oracle ceiling
        ystar = [C.solve(C.apply(k, self.psi0) - self.a0[k] * self.psi0, self.E0, self.psi0) for k in REAL]
        for s in seeds:
            psi = self.psi[s]
            base = [psi, m.H @ psi - float(psi @ (m.H @ psi)) * psi]
            self.score(s, "L1R", C.ritz_vec(base)[0])
            self.score(s, "RWRstarK", C.ritz_vec(base + ystar + [C.K(v) for v in ystar])[0])
        n_or = sum(1 for s in seeds for k in REAL
                   if self.out["seeds"][str(s)][k]["scored"] and tracked(self.out["seeds"][str(s)][k]["ratios"]["RWRstarK"]))
        log(f"oracle ceiling RWRstarK tracks {n_or} of {n_sc}")
        if n_or < 0.75 * n_sc:
            log("EARLY STOP: oracle ceiling below 6 of 8 -> KILL (response class)")
            self.finish()
            return 0
        # ---------------- phase 3: SCRu0 at the deciding shift
        phi1 = {}
        done_sc, tr = 0, 0
        for i, s in enumerate(seeds):
            c0, w0 = cpu(), time.time()
            phi = self.psi[s]
            for t in range(1, 5):
                ys = self.responses(phi, "decide", f"s{s} SCRu0 pass {t}")
                phi = C.ritz_vec([phi, m.H @ phi - float(phi @ (m.H @ phi)) * phi] + ys + [C.K(v) for v in ys])[0]
                if t == 1:
                    phi1[s] = phi
                    self.score(s, "RWRK0", phi)
                if t in (2, 4):
                    self.score(s, f"SCRu0_T{t}", phi)
            self.out["cpu"][f"SCRu0_seed{s}"] = cpu() - c0
            self.out["cpu"][f"SCRu0_wall_seed{s}"] = time.time() - w0
            log(f"seed {s}: SCRu0 CPU {cpu() - c0:.0f} s")
            if i == 0 and a.instance == "main" and cpu() - c0 > 2400:
                log("STOP AND RE-PLAN: seed 11 SCRu0 took more than 40 CPU-minutes (critic)")
                self.out["verdict"] = "INCOMPLETE (seed-11 cost > 40 CPU-min)"
                self.dump()
                return 0
            d = self.out["seeds"][str(s)]
            done_sc += sum(1 for k in REAL if d[k]["scored"])
            tr += sum(1 for k in REAL if d[k]["scored"] and tracked(d[k]["ratios"]["SCRu0_T4"]))
            if a.instance == "main" and tr + (n_sc - done_sc) < 0.5 * n_sc:
                log(f"FUTILITY STOP: SCRu0 tracked {tr}, at most {tr + n_sc - done_sc} reachable < 4 -> KILL (10.1u)")
                self.finish()
                return 0
        # ---------------- phase 4: reported arms under the soft budget
        per_seed = max(v for k, v in self.out["cpu"].items() if k.startswith("SCRu0_seed"))
        per_wall = max(v for k, v in self.out["cpu"].items() if k.startswith("SCRu0_wall"))

        def over(f):
            return cpu() + f * per_seed > a.cpu_budget or time.time() - self.t0 + f * per_wall > a.wall_budget
        for s in seeds:
            if over(0.8):
                log(f"soft budget: SCRd0 skipped from seed {s}"); break
            ph = phi1[s]
            for t in range(2, 5):
                ys = self.responses(ph, "decide", f"s{s} SCRd0 pass {t}")
                ph = C.ritz_vec([ph] + ys)[0]
            self.score(s, "SCRd0_T4", ph)
        for s in seeds:
            if over(1.1):
                log(f"soft budget: default-shift SCRu0 skipped from seed {s}"); break
            phi = self.psi[s]
            for t in range(1, 5):
                ys = self.responses(phi, "default", f"s{s} SCRu0def pass {t}")
                phi = C.ritz_vec([phi, m.H @ phi - float(phi @ (m.H @ phi)) * phi] + ys + [C.K(v) for v in ys])[0]
            self.score(s, "SCRu0def_T4", phi)
        self.finish()
        return 0

    def finish(self):
        v, det = verdict(self.rec())
        self.out["verdict"], self.out["verdict_details"] = v, det
        log(f"T2-v2 VERDICT: {v} | {det} | wall {time.time() - self.t0:.0f} s, cpu {cpu():.0f} s")
        self.dump()


def aggregate(path):
    d = json.load(open(path))
    if d.get("verdict") == "INVALID":
        log("INVALID")
        return 1
    for s, r in d["seeds"].items():
        log(f"seed {s}: " + " | ".join(f"{k}{'' if r[k]['scored'] else '(ns)'} " + " ".join(
            f"{a} {x:+.2f}" for a, x in r[k]["ratios"].items()) for k in REAL))
    v, det = verdict(d["seeds"])
    log(f"T2-v2 VERDICT: {v} | {det}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", choices=("main", "small"), default="main")
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--cpu-budget", type=float, default=6900.0)
    p.add_argument("--wall-budget", type=float, default=3300.0)
    p.add_argument("--out", default=None)
    p.add_argument("--aggregate", default=None)
    p.add_argument("--selftest-verdict", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.selftest_verdict:
        return selftest()
    if a.aggregate:
        return aggregate(a.aggregate)
    a.out = a.out or os.path.join(HERE, "results", f"t2v2_{a.instance}.json")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    return Run(a).run()


if __name__ == "__main__":
    sys.exit(main())
