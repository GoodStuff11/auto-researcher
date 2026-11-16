#!/usr/bin/env python3
"""
P1.2 round 7, tier T1 for route 6.7F (frozen-frame vote initialization, the Gate-4-cost mutation of the
reopened route 6.7): does the hole-hop vote, computed from the MATCHED-WIDTH foil's own trained state,
lower that foil's sign error on three toy sizes? (Premise of every trainable-vote route, at a fixed base,
with zero added parameters, so the foil comparison is exact.)
=====================================================================================================

Why this test: job 698178 (gvote) found the projected vote of the trained 4x4 NNBF-4 state removes
1.33 gaps of sign error (VOTE-USABLE), while T1-a (t1_vote.py) found the raw vote of the UHF anchor
RAISES dEs at every toy size. The two differ in base (anchor vs trained) and in projection. Projected
toys are NO-TARGET, so the toy version of the question is the trained, unprojected base. A route that
initializes its factors at the vote can only help if the vote helps the base it is applied to.

Toys: t1_vote.SIZES (V = 8, 10, 12, open 2-leg ladders, t' = -0.25, J = 0.4). Base: nnbfW of t1b_drf.py
(LRAnsatz full width 6, UHF start, 300 SR steps, seeds 11 and 22), retrained here with the identical
deterministic recipe; validity requires dEs(base) to match results/t1b/<size>_nnbfW_<seed>.json to 1e-9.
Read-outs per (size, seed), sign flipped wherever the vote fires (e > 0), no gate, no training:
  exact  : e(n) = sum_{e in K} H_{n m_e} psi(m_e) / psi(n)  (t1_vote.vote; cost O(V) extra passes)
  frozen : e(n) = sum_{e in K} R^F_e(n), R^F from the trained orbitals' Green's function (t1b_drf DRF
           pass 1 with the trained parameters; one inverse per amplitude, route 6.7F's cost)
Statistic: q = med over seeds of dEs(read-out) / dEs(base).
PRE-REGISTERED RULES (written 2026-11-16 before the first run):
  counted size iff med dEs(base) >= 0.005.
  6.7F premise PASS iff q_frozen <= 0.9 at >= 2 counted sizes including the largest counted size.
  KILL iff q_frozen >= 1.0 at >= 2 counted sizes including the largest. UNDECIDED otherwise.
  The exact-vote q is reported for attribution (frozen vs exact) and decides nothing.
  Validity: base dEs reproduces t1b, and the frozen ratios at the trained state with zero backflow
  output equal the exact ratios (inherited CHECK 1 of t1b_drf.py).
Budget: RLIMIT_CPU 600 s, 1 thread.
Usage: python3 t1_vote67.py [--sizes 0,1,2] [--seeds 11,22]
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1b_drf as B  # noqa: E402
from t1_vote import vote, frame_tables, dE_sign  # noqa: E402
from p11_core import solve_ed, uhf_orbitals  # noqa: E402
import p21_harness as H21  # noqa: E402


def flip_dEs(m, psi, e, psi0, E0):
    s = np.where(psi >= 0, 1.0, -1.0)
    s = np.where(e > 0, -s, s)
    if (psi0 @ (s * np.abs(psi))) < 0:
        s = -s
    return dE_sign(m.H, s, psi0, E0, 3000), float(((e > 0) * psi ** 2).sum() / (psi @ psi))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0,1,2")
    ap.add_argument("--seeds", default="11,22")
    a = ap.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (600, 630))
    sra = SimpleNamespace(sr_iters=300, sr_lr=0.05, sr_shift=1e-3, sr_samples=600, chunk=8192, o_chunk=4)
    rows, valid = [], True
    for k in [int(x) for x in a.sizes.split(",")]:
        m = B.model(k)
        W, Vv, _ = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
        psi0, E0 = Vv[:, 0], float(W[0])
        ref = uhf_orbitals(m, H21.UHF_SEED)
        Mi, hc = frame_tables(m, "K")
        for seed in [int(x) for x in a.seeds.split(",")]:
            nb, T = B.make("nnbfW", m, ref, seed)
            H21.optimize(nb, m, None, "none", 0, sra, np.random.default_rng(seed + 5), f"s{k} base {seed}")
            psi = nb.amplitudes(8192)
            s = np.where(psi >= 0, 1.0, -1.0)
            if psi0 @ psi < 0:
                s = -s
            d_base = dE_sign(m.H, s, psi0, E0, 3000)
            ref_js = json.load(open(os.path.join(B.OUT, f"{k}_nnbfW_{seed}.json")))
            ok = abs(d_base - ref_js["dEs"]) < 1e-9
            valid &= ok
            e_ex = vote(psi, Mi, hc)[0]
            d_ex, w_ex = flip_dEs(m, psi, e_ex, psi0, E0)
            dr = B.DRF(m, ref, B.WC, 1, "K", 0)
            for key in ("W1", "b1", "W2", "b2"):
                dr.p[key] = nb.p[key].copy()
            dr.p["P0u"], dr.p["P0d"] = nb.p["P0u"][0].copy(), nb.p["P0d"][0].copy()
            c = dr._fw(np.arange(m.D))[0]
            e_fr = c["R"].sum(1)
            d_fr, w_fr = flip_dEs(m, psi, e_fr, psi0, E0)
            r = dict(size=k, V=m.V, seed=seed, dEs_base=d_base, base_matches_t1b=ok, dEs_exact=d_ex, w_flip_exact=w_ex,
                     dEs_frozen=d_fr, w_flip_frozen=w_fr)
            rows.append(r)
            print(f"[{time.strftime('%H:%M:%S')}] size {k} seed {seed}: dEs base {d_base:.5f} (t1b match {ok}) | exact vote "
                  f"{d_ex:.5f} (flip w {w_ex:.3f}) | frozen vote {d_fr:.5f} (flip w {w_fr:.3f})", flush=True)
    sizes = sorted({r["size"] for r in rows})
    summ = []
    for k in sizes:
        rr = [r for r in rows if r["size"] == k]
        mb = float(np.median([r["dEs_base"] for r in rr]))
        qf = float(np.median([r["dEs_frozen"] / r["dEs_base"] for r in rr]))
        qe = float(np.median([r["dEs_exact"] / r["dEs_base"] for r in rr]))
        summ.append(dict(size=k, med_base=mb, q_frozen=qf, q_exact=qe, counted=mb >= 0.005))
        print(f"  size {k}: med dEs base {mb:.5f} q_frozen {qf:.3f} q_exact {qe:.3f} counted {mb >= 0.005}")
    cr = [s for s in summ if s["counted"]]
    if not valid:
        v = "INVALID"
    elif len(cr) < 2:
        v = "NO-TARGET"
    elif sum(s["q_frozen"] <= 0.9 for s in cr) >= 2 and cr[-1]["q_frozen"] <= 0.9:
        v = "PASS"
    elif sum(s["q_frozen"] >= 1.0 for s in cr) >= 2 and cr[-1]["q_frozen"] >= 1.0:
        v = "KILL"
    else:
        v = "UNDECIDED"
    print(f"VERDICT route 6.7F vote premise on the matched-width foil: {v} (valid {valid}); "
          f"CPU {time.process_time():.0f} s")
    json.dump(dict(rows=rows, summary=summ, verdict=v), open(os.path.join(HERE, "results", "t1_vote67.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
