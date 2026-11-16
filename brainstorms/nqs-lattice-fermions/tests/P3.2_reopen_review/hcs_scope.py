#!/usr/bin/env python3
r"""
P3.2 reopening review (2026-11-16): T1 scope check of the closed-leaf Saturation Diagnostic.
============================================================================================

Question. The closed-leaf falsifier (../P3.2_closed_falsifier) found that a sampled Lanczos step (L1)
at 2e4 samples per branch abstains almost everywhere on targets |d| <= 0.0005, whose in-irrep gaps
(1.4e-3 t to 2.7e-3 t, i.e. 1e-4 to 2e-4 t per site) lie BELOW the Invariants' Hard-Core scale
(|delta|/V of order 1e-3 t). Does "the sampled step loses" also hold at the Hard-Core per-site scale?
It also measures, as a diagnostic only, the variance ratio of a mixture-paired energy-difference
estimator against independent sampling (candidate C2 of the review) and the Bhattacharyya overlap.

States: the W0 states of the falsifier (round-4 width-4 snapshot NQS nearest each target, evaluated at
the target t', projected into the irrep K=0, Mx=My=+1), built with the unchanged helpers of
../P3.2_closed_falsifier/sampled_bl.py and ../P3.2_biased_ensemble_fix/micro_test.py (read-only imports).

Rules per instance (U = abstain, tau = 0.02 on the order character c):
  NAIVE-x  lower exact branch energy wins.
  NAIVE-s  sampled: E_X = mean local energy over n_s i.i.d. draws of |psi_X|^2; abstain if
           |E_A - E_B| <= 2 hypot(SE_A, SE_B), SE from the same draws.
  L1-x     exact one Lanczos step per branch; the lower Ritz energy wins.
  L1-s     sampled L1, identical to sampled_bl.sampled_L1.

Pre-registered rule (written in the leaf before the run):
  Every instance asserts Delta_2 = E_2 - E_0 >= 0.02 t (isolated doublet).
  Counted instances: targets with in-irrep gap Delta_1 >= 0.02 t, all three seeds.
  r_V, w_V = right-commit and wrong-commit fractions over counted instances x replicates.
  SCOPED   if r(L1-s) >= 0.5 and w(L1-s) <= 0.05 (the "sampled L1 loses" sentence holds only below
           the Hard-Core per-site scale);
  EXTENDS  if r(L1-s) < 0.25 or w(L1-s) > 0.10 (it holds at the Hard-Core scale too);
  PARTIAL  otherwise.
  No verdict reopens P3.2: L1 fails Gates 0 and 4 as a route. Uncounted targets are printed for context.

Usage
  python3 hcs_scope.py --seed 1234            one seed, writes results/s<seed>.json
  python3 hcs_scope.py --aggregate results    applies the rule to results/*.json

Options
  --seed INT       1234, 2345 or 3456 (width-4 snapshot files exist only for these).
  --targets LIST   comma list of d = t' - t'_c (default "-0.005,-0.002,0.002,0.005"; write --targets=...).
  --n-s INT        samples per branch (default 20000, the falsifier's budget).
  --reps INT       sampling replicates (default 40).
  --tau FLOAT      |c| below which the order label is U (default 0.02).
  --threads INT    BLAS threads, 1 to 4 (default 2).
  --aggregate DIR  aggregate DIR/*.json, print the table and the verdict, and exit.
Budget guard: before any heavy work the run asserts that the estimated CPU time (25 s per target,
the falsifier's measured rate, times threads) is at most 600 s, and a SIGALRM watchdog stops it at 600 s.
"""
import argparse
import glob
import json
import os
import signal
import sys


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P3.2 Hard-Core-scale scope check (see module docstring)")
    ap.add_argument("--seed", type=int, default=1234, choices=[1234, 2345, 3456])
    ap.add_argument("--targets", default="-0.005,-0.002,0.002,0.005")
    ap.add_argument("--n-s", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--aggregate", default="")
    a = ap.parse_args(argv)
    a.target_list = [float(s) for s in a.targets.split(",")]
    return a


ARGS = parse_args()
assert 1 <= ARGS.threads <= 4, "thread cap is 4"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(ARGS.threads)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P3.2_closed_falsifier"))
import numpy as np  # noqa: E402

cnt = lambda L: "/".join(str(L.count(k)) for k in "ABU")  # noqa: E731


def run(a):
    est = 25.0 * len(a.target_list) * a.threads
    assert est <= 600, f"estimated CPU {est:.0f}s exceeds the 600 s per-run cap"
    signal.alarm(600)
    import sampled_bl as SB
    MT = SB.MT
    inst = MT.Instance(1.11394, 8.0)
    snaps = np.load(os.path.join(SB.FIX, "results", f"w4_s{a.seed}_snap_theta.npz"))
    pool = {X: [(float(k[1:8]), snaps[k]) for k in snaps.files if k.endswith(X)] for X in "AB"}
    out = dict(seed=a.seed, rows=[])
    for d in a.target_list:
        tp = inst.tpc + d
        w, v = inst.cl.sector_eigs(inst.coup(tp), 3)
        gap1, gap2 = float(w[1] - w[0]), float(w[2] - w[0])
        assert gap2 >= 0.02, f"Delta_2={gap2} < 0.02 t"
        v0 = inst.cl.full_vec(v[:, 0])
        v0 /= np.linalg.norm(v0)
        truth = MT.order_of(inst.char(v0), a.tau)
        H = inst.H(tp)
        inst.set_T(tp)
        br = {}
        for X in "AB":
            ds, theta = min(pool[X], key=lambda q: abs(q[0] - d))
            ans = MT.new_ansatz(inst, X, 4, a.seed)
            ans.set_flat(np.array(theta, float).copy())
            psi = inst.sg.project(ans.amplitudes(65536))
            psi /= np.linalg.norm(psi)
            Hp = H @ psi
            H2p = H @ Hp
            E = float(psi @ Hp)
            EL, chi = SB.ritz_vecs(H, np.stack([psi, Hp], 1))
            br[X] = dict(psi=psi, Hp=Hp, H2p=H2p, E=E, var=float(Hp @ Hp - E * E), c=inst.char(psi),
                         EL=EL, cL=inst.char(chi))
        low = "A" if br["A"]["E"] < br["B"]["E"] else "B"
        naive_x = MT.order_of(br[low]["c"], a.tau)
        lowL = "A" if br["A"]["EL"] < br["B"]["EL"] else "B"
        L1_x = MT.order_of(br[lowL]["cL"], a.tau)
        # C2 diagnostic: mixture-paired estimator of E_A - E_B (2n draws from p) vs independent (n each)
        pA, pB = br["A"]["psi"] ** 2, br["B"]["psi"] ** 2
        p = 0.5 * (pA + pB)
        g = (br["A"]["psi"] * br["A"]["Hp"] - br["A"]["E"] * pA
             - br["B"]["psi"] * br["B"]["Hp"] + br["B"]["E"] * pB)
        m = p > 0
        R_pair = float(np.sum(g[m] ** 2 / p[m])) / 2 / (br["A"]["var"] + br["B"]["var"])
        bhatt = float(np.sum(np.abs(br["A"]["psi"] * br["B"]["psi"])))
        rng = np.random.default_rng(7000 * a.seed + int(1e5 * (d + 1)))
        dN, dL = [], []
        for _ in range(a.reps):
            r = {}
            for X, pX in (("A", pA), ("B", pB)):
                idx = rng.choice(len(pX), size=a.n_s, p=pX / pX.sum())
                e1 = br[X]["Hp"][idx] / br[X]["psi"][idx]
                r[X] = (e1.mean(), e1.std() / np.sqrt(a.n_s))
            wc = br["A"]["c"] if r["A"][0] < r["B"][0] else br["B"]["c"]
            dN.append("U" if abs(r["A"][0] - r["B"][0]) <= 2 * np.hypot(r["A"][1], r["B"][1])
                      else MT.order_of(wc, a.tau))
            EA, seA, cA = SB.sampled_L1(br["A"]["psi"], br["A"]["Hp"], br["A"]["H2p"], inst.cn, a.n_s, rng)
            EB, seB, cB = SB.sampled_L1(br["B"]["psi"], br["B"]["Hp"], br["B"]["H2p"], inst.cn, a.n_s, rng)
            dL.append("U" if abs(EA - EB) <= 2 * np.hypot(seA, seB)
                      else MT.order_of(cA if EA < EB else cB, a.tau))
        row = dict(d=d, gap1=gap1, gap2=gap2, truth=truth, naive_x=naive_x, L1_x=L1_x,
                   b={X: br[X]["E"] - float(w[0]) for X in "AB"}, bL={X: br[X]["EL"] - float(w[0]) for X in "AB"},
                   var={X: br[X]["var"] for X in "AB"}, R_pair=R_pair, bhatt=bhatt, NAIVE_s=dN, L1_s=dL)
        MT.P.log(f"s={a.seed} d={d:+.4f} gap1={gap1:.4f} gap2={gap2:.3f} truth={truth} "
                 f"b_A={row['b']['A']:.2e} b_B={row['b']['B']:.2e} bL_A={row['bL']['A']:.2e} "
                 f"bL_B={row['bL']['B']:.2e} | NAIVE-x {naive_x} L1-x {L1_x} | NAIVE-s {cnt(dN)} "
                 f"L1-s {cnt(dL)} | R_pair {R_pair:.3f} Bhatt {bhatt:.3f}")
        out["rows"].append(row)
    json.dump(out, open(os.path.join(HERE, "results", f"s{a.seed}.json"), "w"), indent=1, default=float)


def aggregate(dr):
    rows = []
    for f in sorted(glob.glob(os.path.join(dr, "s*.json"))):
        r = json.load(open(f))
        rows += [dict(x, seed=r["seed"]) for x in r["rows"]]
    print(" seed       d   gap1 counted truth NAIVE-x L1-x | NAIVE-s(A/B/U) L1-s(A/B/U) | "
          "Db_naive  Db_L1     R_pair Bhatt")
    for x in rows:
        print(f"{x['seed']:5d} {x['d']:+.4f} {x['gap1']:.4f} {'Y' if x['gap1'] >= 0.02 else 'n':>7}   "
              f"{x['truth']}      {x['naive_x']}      {x['L1_x']}  | {cnt(x['NAIVE_s']):>13} {cnt(x['L1_s']):>11} | "
              f"{x['b']['B'] - x['b']['A']:+.2e} {x['bL']['B'] - x['bL']['A']:+.2e} {x['R_pair']:.3f} {x['bhatt']:.3f}")
    C = [x for x in rows if x["gap1"] >= 0.02]
    assert C, "no counted instance"
    for V in ("NAIVE_s", "L1_s"):
        r = float(np.mean([d == x["truth"] for x in C for d in x[V]]))
        w = float(np.mean([d not in ("U", x["truth"]) for x in C for d in x[V]]))
        print(f"{V}: {len(C)} counted instances: right {r:.3f} wrong {w:.3f}")
        if V == "L1_s":
            v = "SCOPED" if (r >= 0.5 and w <= 0.05) else ("EXTENDS" if (r < 0.25 or w > 0.10) else "PARTIAL")
            print(f"VERDICT scope check: {v}")


if __name__ == "__main__":
    if ARGS.aggregate:
        aggregate(ARGS.aggregate if os.path.isabs(ARGS.aggregate) else os.path.join(HERE, ARGS.aggregate))
    else:
        run(ARGS)
