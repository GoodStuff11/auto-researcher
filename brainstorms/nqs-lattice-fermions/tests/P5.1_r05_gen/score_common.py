#!/usr/bin/env python3
"""Shared first-passage statistic for RULE_cmp.md (R51.16) and RULE_qopt_T2.md (R51.11), revision 1 (auditor):
sustained crossing.  T(arm) = N x t*, t* = first step t >= 20 at which the mean of R over steps t-19..t is >= R*;
infinite (censored) if no such step.  R = (E_start - E)/(E_start - E0) against R51.1's 4x3 start."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
NF = os.path.join(H, "..", "P5.1_noise_floor_r01", "results")
E0 = -7.828637164695285  # lowest (+,+) level, ../P5.1_size_scaling_r04/results/info_S4x3_N10.json
ES = -7.805526273788163  # R51.1's 4x3 start energy
RP = 0.7475
WIN = 20


def R_of(E):
    return (ES - np.asarray(E)) / (ES - E0)


def rseries(path):
    """R per step from a ns_run.py run.json (absolute energies E)."""
    return R_of([s["E"] for s in json.load(open(os.path.join(H, path, "run.json")))["steps"]])


def rseries_nf(name):
    """R per step from an R51.1 nf_run.py run.json (stores R against the same start and gap)."""
    return np.array([s["R"] for s in json.load(open(os.path.join(NF, name, "run.json")))["steps"]])


def tstar(R, rstar):
    if len(R) < WIN:
        return None
    m = np.convolve(R, np.ones(WIN) / WIN, mode="valid")  # m[j] = mean of steps j+1 .. j+WIN
    i = np.nonzero(m >= rstar)[0]
    return None if len(i) == 0 else int(i[0] + WIN)


def T(R, N, rstar):
    t = tstar(R, rstar)
    return float("inf") if t is None else float(N * t)


def guided_T(rstar):
    """T_guided = min(R51.1 arm B, R51.1 arm A followed by its continuation), both by the sustained rule."""
    tb = T(rseries_nf("N65536_tau0.005_s9000"), 65536, rstar)
    ra = rseries_nf("N16384_tau0.005_s9000")
    cont = os.path.join(H, "results", "guided_N16384_tau0.005_cont", "run.json")
    if os.path.exists(cont):
        ra = np.concatenate([ra, rseries("results/guided_N16384_tau0.005_cont")])
    return min(tb, T(ra, 16384, rstar)), tb


if __name__ == "__main__":  # re-score R51.1's stored guided arms with the sustained rule (no new data)
    out = {}
    for frac in (0.80, 0.90, 0.95):
        rs = frac * RP
        out[frac] = {k: T(rseries_nf(d), n, rs) for k, (d, n) in
                     {"A": ("N16384_tau0.005_s9000", 16384), "B": ("N65536_tau0.005_s9000", 65536),
                      "C": ("N262144_tau0.005_s9000", 262144), "D": ("N16384_tau0.0025_s9000", 16384)}.items()}
        print(frac, out[frac])
    json.dump(out, open(os.path.join(H, "results", "rescore_R51.1_sustained.json"), "w"), indent=1)
