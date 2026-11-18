#!/usr/bin/env python3
"""
P5.1 R51.11 `qopt`, Gate 4 third-volume weight test (rule: RULE_R51.11_vol3.md, pre-registered)
================================================================================================
Exact E_q[w^2] for q ~ |psi| (and for the guided mixture 0.8 p + 0.2 |psi|) of the R51.10 `kappa` start states on the
open two-leg ladder 5x2 (N_e 8), 6x2 (N_e 10) and 7x2 (N_e 12), plus e^H of p, over the full (+,+) sector.
Amplitude-only: the orbit reduction of ns_run.GCtx is rebuilt without the Hamiltonian (no Hr), so the 7x2 pass fits in
a few GB.  `stats` is copied unchanged from t0_weightvar_R51.11.py (round 7), so values at 5x2 / 6x2 must reproduce
2.244 / 3.212 (a built-in check, printed as CHECK lines).

Usage (lit_env, one thread):
    export OMP_NUM_THREADS=1; ~/.conda/envs/lit_env/bin/python t0_weightvar_vol3_R51.11.py [CASE ...]

Arguments
  CASE   any of S5x2, S6x2, S7x2 (default: all three, in that order).  Each case uses
         ../P5.1_size_scaling_r04/results/kappa_<CASE>_start.json's "theta" path (relative to P5.1_size_scaling_r04).
Output: one line per case, `RESULT <CASE> {json}` with keys V, D (sector size), expH, Ew2_q1, Ew2_guided,
Ew2_q1_kqconv, cpu_s; then `FIT` with the least-squares slopes of ln Ew2_q1 and ln expH against V (when all three
cases ran), and the rule verdict.  Nothing is written to disk; the caller tees stdout to logs/.
"""
import json
import os
import sys
import time

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

WD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSD = os.path.join(WD, "P5.1_size_scaling_r04")
sys.path.insert(0, NSD)
os.chdir(NSD)
import ns_run as NS  # noqa: E402
import torch  # noqa: E402

CASES = {"S5x2": (5, 2, 8), "S6x2": (6, 2, 10), "S7x2": (7, 2, 12)}
ROUND7 = {"S5x2": 2.244, "S6x2": 3.212}


def parse_args(argv):
    only = argv or list(CASES)
    for k in only:
        if k not in CASES:
            raise SystemExit(f"unknown case {k}; valid: {list(CASES)}")
    return only


def stats(size, pr):  # verbatim from t0_weightvar_R51.11.py
    a = np.abs(pr); m = a > 1e-14 * a.max()
    size, a = size[m], a[m]
    Z1, Z2, Z3 = (size * a).sum(), (size * a**2).sum(), (size * a**3).sum()
    pc = a**2 / Z2; q1 = a / Z1; qg = 0.8 * pc + 0.2 * q1
    H = -(size * pc * np.log(pc)).sum()
    pu = size * pc; qk = np.sqrt(size) * a; qk /= qk.sum()  # kq.py convention
    return dict(D=float(size.sum()), expH=float(np.exp(H)), Ew2_q1=float(Z1 * Z3 / Z2**2),
                Ew2_guided=float((size * pc**2 / qg).sum()), Ew2_q1_kqconv=float((pu**2 / qk).sum()))


def rep_amplitudes(Lx, Ly, Ne, start):
    """ns_run.GCtx.__init__ + energy() without Hr: orbit sizes and normalised representative amplitudes."""
    m = NS.V.Model("che", Lx, Ly, Ne, 16, -0.25, 8.0, (1, 1))
    sec = NS.V.exact_sector(m)
    nU, nD = NS.V.exact_configs(sec)
    maps = NS.TG.pg_maps(m, sec)
    idx = np.stack([mp[0] for mp in maps]); sg = np.stack([mp[1] for mp in maps])
    dim = sec.dim
    rep_of = idx[np.argmin(idx, 0), np.arange(dim)]
    reps = np.unique(rep_of)
    srt = np.sort(idx[:, reps], 0)
    size = 1.0 + (np.diff(srt, axis=0) != 0).sum(0)
    th = torch.tensor(np.load(start))
    assert th.numel() == m.nparam
    ls, ss = [], []
    with torch.no_grad():
        for s0 in range(0, dim, 8192):
            l, s = m.logpsi0(th, nU[s0:s0 + 8192], nD[s0:s0 + 8192]); ls.append(l); ss.append(s)
    l, s = torch.cat(ls), torch.cat(ss)
    phi0 = (s * torch.exp(l - l.max())).numpy()
    pr = 0.25 * (sg[:, reps] * phi0[idx[:, reps]]).sum(0)
    pr = pr / np.sqrt(np.sum(size * pr * pr))
    return size, pr, dim


def main(argv):
    only = parse_args(argv)
    out = {}
    for k in only:
        Lx, Ly, Ne = CASES[k]
        t = time.process_time()
        st = json.load(open(f"results/kappa_{k}_start.json"))["theta"]
        size, pr, dim = rep_amplitudes(Lx, Ly, Ne, st)
        o = stats(size, pr); o["V"] = Lx * Ly; o["dim"] = dim; o["start"] = st; o["cpu_s"] = time.process_time() - t
        out[k] = o
        print("RESULT", k, json.dumps(o), flush=True)
        if k in ROUND7:
            print(f"CHECK {k}: Ew2_q1 {o['Ew2_q1']:.3f} vs round-7 {ROUND7[k]:.3f}", flush=True)
    if all(k in out for k in CASES):
        Vs = np.array([out[k]["V"] for k in CASES], float)
        s = np.polyfit(Vs, np.log([out[k]["Ew2_q1"] for k in CASES]), 1)[0]
        sH = np.polyfit(Vs, np.log([out[k]["expH"] for k in CASES]), 1)[0]
        E14 = out["S7x2"]["Ew2_q1"]
        verdict = "FAIL" if (s > 0.1 or np.exp(s) > np.exp(sH)) else "PASS"
        shape = "EXP-OR-FASTER" if E14 >= 4.60 else ("POLY" if E14 <= 4.37 else "SHAPE-UNRESOLVED")
        s1214 = np.log(E14 / out["S6x2"]["Ew2_q1"]) / 2
        a1214 = np.log(E14 / out["S6x2"]["Ew2_q1"]) / np.log(14 / 12)
        print("FIT", json.dumps(dict(slope_lnEw2=s, factor=np.exp(s), slope_lnexpH=sH, factor_expH=np.exp(sH),
                                     slope_12_14=s1214, loglog_exp_12_14=a1214, E14=E14,
                                     verdict=verdict, shape=shape)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
