#!/usr/bin/env python3
"""
P3.2 round 5, certificate-power PRECHECK (exact ED quantities + the existing job-451852 states' numbers)
=====================================================================================================

Question. Route 5.1 (ECD, Eckart-certified direct decision) decides the order of the irrep ground state
psi_0 from the sign of c(psi) = <psi|O|psi> of ONE projected NQS psi, and commits only when a fidelity
certificate excludes the other sign. Before any training is spent, this script asks: with the energy
errors eps that projected NNBF actually reaches on this instance (job 451852, widths 4 and 8), how strong
can such a certificate possibly be? It uses EXACT eps and gaps (an upper bound on certificate power: the
sampled estimates of round 5 can only weaken it).

Certificates (eta = eps / Delta_1 is the Eckart bound on 1 - F, F = |<psi_0|psi>|^2, valid when E < E_1):
  RIG   |c(psi) - c_0| <= DeltaO * sqrt(eta)                     (trace distance; DeltaO = spread of O
                                                                  restricted to the projected sector)
  SIG   |c(psi) - c_0| <= eta * DeltaO + 2 sqrt(eta (1 - eta)) * sigma_0(O)   (decomposition
                                                                  psi = a psi_0 + b phi; sigma_0(O) is the
                                                                  standard deviation of O in psi_0)
  TWO   two-level refinement: psi = a psi_0 + b psi_1 + r chi with |b|^2 Delta_1 + |r|^2 Delta_2 <= eps;
        worst case over (b, r) of |c(psi) - c_0| using exact c_1 and O_01 = <psi_0|O|psi_1> and the
        RIG-type bound for the chi part (DeltaO); reported as the best one could hope for.
  A decision commits when |c(psi)| > bound; then sign(c(psi)) = sign(c_0) is guaranteed (exact eps).

Usage
  python3 precheck.py [--fix-results DIR] [--out FILE]
Options
  --fix-results DIR   folder with the job-451852 JSONs w{4,8}_s*.json
                      (default ../P3.2_biased_ensemble_fix/results)
  --out FILE          JSON output (default results_precheck/precheck.json)
Output: per target the exact quantities and, per width, the commit rate of each certificate on the
lowest-energy state among the four control branches of every seed (cold A/B, warm A/B), plus the
largest eta at which each certificate can still commit given |c| ~ |c_0|.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "P3.2_biased_ensemble_fix")
sys.path.insert(0, FIX)
import micro_test as MT  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-results", default=os.path.join(FIX, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_precheck", "precheck.json"))
    return ap.parse_args(argv)


def bounds(eta, dO, sig0, c1=None, c0=None, O01=None, D1=None, D2=None, eps=None):
    out = {"RIG": dO * np.sqrt(eta) if eta < 1 else np.inf,
           "SIG": (eta * dO + 2 * np.sqrt(eta * (1 - eta)) * sig0) if eta < 1 else np.inf}
    if c1 is not None:
        worst = 0.0
        bmax = min(1.0, eps / D1)
        for b2 in np.linspace(0, bmax, 201):
            r2 = max(0.0, min(1 - b2, (eps - b2 * D1) / D2))
            a2 = 1 - b2 - r2
            dev = (abs(b2 * (c1 - c0)) + 2 * np.sqrt(a2 * b2) * abs(O01)
                   + dO * np.sqrt(r2) + 2 * np.sqrt(b2 * r2) * dO)
            worst = max(worst, dev)
        out["TWO"] = worst
    return out


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    inst = MT.Instance(1.11394, 8.0)
    cl = inst.cl
    import scipy.sparse as sp
    Ob = (cl.B.T @ (sp.diags(inst.cn) @ cl.B)).toarray()     # O in the space-group irrep basis
    Osec = cl.Q.T @ Ob @ cl.Q                                 # restricted to the S = 0 projected sector
    wO = np.linalg.eigvalsh(Osec)
    dO = float(wO[-1] - wO[0])
    print(f"t'_c={inst.tpc:+.6f} min gap {inst.gapmin:.3e}; O spread in projected sector "
          f"[{wO[0]:+.3f}, {wO[-1]:+.3f}] -> DeltaO = {dO:.3f} (full-space spread "
          f"{inst.cn.max() - inst.cn.min():.3f})")
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.fix_results, "w*_s*.json"))):
        r = json.load(open(f))
        runs.setdefault(r["args"]["width"], []).append(r)
    targets = [t["d"] for t in next(iter(runs.values()))[0]["targets"]]
    exact = {}
    for d in targets:
        tp = inst.tpc + d
        w, v = cl.sector_eigs(inst.coup(tp), 3)
        p = [cl.full_vec(v[:, k]) for k in range(3)]
        c = [inst.char(x) for x in p]
        O01 = float(np.sum(p[0] * inst.cn * p[1]))
        sig0 = float(np.sqrt(np.sum(p[0] ** 2 * inst.cn ** 2) - c[0] ** 2))
        D1, D2 = float(w[1] - w[0]), float(w[2] - w[0])
        exact[d] = dict(E0=float(w[0]), D1=D1, D2=D2, c0=c[0], c1=c[1], c2=c[2], O01=O01, sig0=sig0)
        eta_star = {}
        for nm in ("RIG", "SIG"):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if bounds(mid, dO, sig0)[nm] < abs(c[0]):
                    lo = mid
                else:
                    hi = mid
            eta_star[nm] = lo
        exact[d]["eta_star"] = eta_star
        print(f"d={d:+.4f} D1={D1:.4f} D2={D2:.4f} c0={c[0]:+.3f} c1={c[1]:+.3f} c2={c[2]:+.3f} "
              f"O01={O01:+.3f} sigma0(O)={sig0:.3f} | max eta to commit at |c|=|c0|: "
              f"RIG {eta_star['RIG']:.4f} SIG {eta_star['SIG']:.4f} "
              f"(eps <= {eta_star['RIG'] * D1:.1e} / {eta_star['SIG'] * D1:.1e})")
    summary = {}
    for W in sorted(runs):
        tally = {nm: dict(commit=0, wrong=0, n=0) for nm in ("RIG", "SIG", "TWO")}
        inner = {nm: dict(commit=0, n=0) for nm in ("RIG", "SIG", "TWO")}
        for r in runs[W]:
            for t in r["targets"]:
                d, ex = t["d"], exact[t["d"]]
                cands = []
                for arm in ("R0P", "R0P-warm"):
                    for X in "AB":
                        b = t["ctrl"][arm]["branches"][X]
                        cands.append((b["dE"], b["c"], arm + X))
                eps, cpsi, nm_b = min(cands)
                eta = eps / ex["D1"]
                bd = bounds(eta, dO, ex["sig0"], ex["c1"], ex["c0"], ex["O01"], ex["D1"], ex["D2"], eps)
                line = []
                for nm in tally:
                    com = abs(cpsi) > bd[nm]
                    wrong = com and np.sign(cpsi) != np.sign(ex["c0"])
                    tally[nm]["n"] += 1
                    tally[nm]["commit"] += int(com)
                    tally[nm]["wrong"] += int(wrong)
                    if abs(d) < 0.001:
                        inner[nm]["n"] += 1
                        inner[nm]["commit"] += int(com)
                    line.append(f"{nm}:{bd[nm]:.3f}{'*' if com else ''}")
                print(f"  W={W} s={r['args']['seed']} d={d:+.4f} best={nm_b} eps={eps:.2e} eta={eta:.3f} "
                      f"c={cpsi:+.3f} (c0={ex['c0']:+.3f}) bounds " + " ".join(line))
        summary[W] = dict(all=tally, inner=inner)
        print(f"W={W}: " + "; ".join(f"{nm} DEC {v['commit']}/{v['n']} wrong {v['wrong']} | inner DEC "
                                     f"{inner[nm]['commit']}/{inner[nm]['n']}" for nm, v in tally.items()))
    json.dump(dict(DeltaO=dO, exact={str(k): v for k, v in exact.items()},
                   summary={str(k): v for k, v in summary.items()}), open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
