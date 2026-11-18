#!/usr/bin/env python3
"""Round 9 scale audit for RULE_cmp.md rev 1 (R51.16) and RULE_qopt_T2.md rev 1 (R51.11).
score_common.py mixes two gaps: R51.1's guided arms A/B use their stored R (gap 0.022373, nf_run's E0 -7.827899,
the scale on which R_pol = 0.7475 and T_B = 7.41M were stated), while ns_run arms (continuation, born, q1) are put on
gap 0.023111 (E0_pg -7.828637).  This script recomputes every arm from absolute energies on ONE gap and applies the
rules' numeric R* = frac x 0.7475.  g1 = 0.022373 is the pre-registered scale (R_pol, R*, T_B all quoted on it)."""
import json
import os

import numpy as np

import score_common as C

G = {"g1_0.022373_prereg": 0.022372760483589715, "g2_0.023111": C.ES - C.E0}


def E_nf(name):
    return np.array([s["E"] for s in json.load(open(os.path.join(C.NF, name, "run.json")))["steps"]])


def E_ns(p):
    return np.array([s["E"] for s in json.load(open(os.path.join(C.H, p, "run.json")))["steps"]])


def R(E, g):
    return (C.ES - E) / g


out = {}
for gk, g in G.items():
    out[gk] = {}
    for frac in (0.90, 0.80):
        rs = frac * C.RP
        tb = C.T(R(E_nf("N65536_tau0.005_s9000"), g), 65536, rs)
        ra = np.concatenate([R(E_nf("N16384_tau0.005_s9000"), g), R(E_ns("results/guided_N16384_tau0.005_cont"), g)])
        ta = C.T(ra, 16384, rs)
        tg = min(tb, ta)
        arms = {}
        for k, (p, n) in {"born16k": ("results/born_N16384_tau0.005", 16384),
                          "born65k_tau0.01": ("results/born_N65536_tau0.01", 65536),
                          "q1_16k": ("results/q1_N16384_tau0.005", 16384),
                          "q1_65k": ("results/q1_N65536_tau0.005", 65536)}.items():
            r = R(E_ns(p), g)
            arms[k] = dict(T=C.T(r, n, rs), steps=len(r), R_end=float(r[-1]),
                           R_max_trail20=float(np.convolve(r, np.ones(20) / 20, "valid").max()))
        tborn = min(arms["born16k"]["T"], arms["born65k_tau0.01"]["T"])
        cr = min(tborn, 14.8e6) / tg
        tq = min(arms["q1_16k"]["T"], arms["q1_65k"]["T"])
        qr = tq / tg
        out[gk][frac] = dict(Rstar=rs, T_B=tb, T_Acont=ta, T_guided=tg, A_cont_Rend=float(ra[-1]),
                             A_cont_tstar=C.tstar(ra, rs), arms=arms,
                             cmp=dict(T_born=tborn, ratio_lb=cr, verdict="GUIDED-FEWER" if cr >= 2 else (
                                 "NO-GAIN" if cr <= 1.25 else "MIXED")),
                             qopt=dict(T_q=tq, ratio=qr, verdict="Q-FEWER" if qr <= 0.5 else (
                                 "NO-GAIN" if qr >= 0.8 else "MIXED")))
        print(gk, frac, json.dumps(out[gk][frac]))
json.dump(out, open(os.path.join(C.H, "results", "score_scales.json"), "w"), indent=1)
