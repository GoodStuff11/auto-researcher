#!/usr/bin/env python3
"""
P3.1 round 9: tier-1 toy computations for Routes 9.1 (RWR-K), 9.2 (EDR), 9.3 (TMW-K). Exact linear algebra.
==========================================================================================================

Reuses ../P3.1_order_parameter_error_r8/t1_portfolio.py unchanged: instances (3x2, 3x3, 6x2, 4x3 OBC t-t'-J at
n = 2/3), observables (pair / spin / charge long-range tails; pairSF / spinSF / chargeSF), the eight trial
states per size (first-order ground states of wrong Hamiltonians plus high-omega junk, overlap 0.94), the
Ritz routine, ORC(3) and L1R.

Arms (Ritz of H; ratio = (a(lowest Ritz vector) - a(psi)) / delta, delta = a_0 - a_psi):
  L1R      span{psi, (H-E)psi}                                                (blind control)
  ORC3     span{psi, (H-E)psi, |1>,|2>,|3>}  exact levels                      (comparison arm)
  RWR_R    span{psi, r, y~_pair, y~_spin, y~_charge}                           (route 8.1 grade R, reproduced)
  RWRK_g   RWR plus one Krylov partner (H - <H>_y~) y~ per response, grade g in {R, L, J}
           grade R = (eta_low 0.3, eta_high 0.5); L = (0.3, 0); J = (0, 0.5)
           y~ = unit(unit(y) + eta_low g + eta_high j), y = (Q(H-s)Q)^{-1} Q (A - a_psi) psi (as round 8)
  EDR_R    span{psi, r, phi_1..3, y~ (grade R), Krylov partners of y~}
           phi_k = unit(|k> + 0.2 g_k + 0.2 j_k)  (trained-quality excited states)
  ESR3n    span{psi, r, phi_1..3}  (the noisy phi alone, reported)
  TMWall   span{psi, r} + {v_m} for density, spin, d-pair standing waves of ALL OBC modes m != (0,0)
  TMWK     TMWall plus one Krylov partner per v_m
Scored item: real-space tail with |delta| >= 0.10 |a_0|; tracked if ratio in [1/2, 2].

Pre-registered rules (leaf, Round 9, written before --mode main was run). Trend sizes 3x3 -> 4x3
(3x2 is printed only: its 12-level sector is spanned trivially by >= 12 vectors).
  9.1 RWRK_R: PASS if tracked >= 0.75 overall (3x3, 6x2, 4x3) and f(4x3) >= f(3x3) - 0.1 and
      f(4x3) >= ORC3(4x3) + 0.2; KILL if overall < 0.5 or f(4x3) < f(3x3) - 0.25; else UNCLEAR.
  9.2 EDR_R: same; PASS additionally needs f(4x3) >= RWRK_R(4x3) + 0.1 (otherwise at most UNCLEAR).
  9.3 TMWK: same thresholds as 9.1.
  Reported: the tracked fraction on the "real-like" subset (partner structure factor |rel. error| < 0.05).

Usage
  python3 t1_r9.py --mode smoke                 # 3x2 only, all arms, no verdict (seconds)
  python3 t1_r9.py --mode main --out results/t1_r9.json
Options
  --mode {smoke, main}  smoke: 3x2 only, prints rows; main: sizes 3x2,3x3,6x2,4x3, verdicts, JSON.
  --n-low INT           cap on exact in-sector levels (default 60, as round 8).
  --cpu-limit INT       hard RLIMIT_CPU in CPU-seconds, set before any ED (default 600 = 10 CPU-min).
  --out PATH            JSON output for main ("none" = no file).
Threads: OMP/OPENBLAS/MKL_NUM_THREADS = 4.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R8 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r8")
_spec = importlib.util.spec_from_file_location("t1", os.path.join(R8, "t1_portfolio.py"))
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)
REAL, SF, ALL, log, unit = t1.REAL, t1.SF, t1.ALL, t1.log, t1.unit
PARTNER = dict(pair="pairSF", spin="spinSF", charge="chargeSF")
GR = {"R": (0.3, 0.5), "L": (0.3, 0.0), "J": (0.0, 0.5)}
ARMS = ("L1R", "ORC3", "RWR_R", "RWRK_R", "RWRK_L", "RWRK_J", "EDR_R", "ESR3n", "TMWall", "TMWK")


def krylov(I, v):
    v = unit(v)
    w = I.m.H @ v
    w = w - (v @ w) * v
    return I.proj(w)


def two_mode_all(I):
    """v -> O_m^dag O_m v for rho, S, Delta_d standing waves, all modes (mx < Lx, my < Ly), m != (0,0)."""
    m, ob = I.m, I.ob
    modes = [(a, b) for a in range(m.Lx) for b in range(m.Ly) if (a, b) != (0, 0)]

    def f(mode, xy):
        return (math.cos(math.pi * mode[0] * (xy[0] + 0.5) / m.Lx)
                * math.cos(math.pi * mode[1] * (xy[1] + 0.5) / m.Ly))
    nu = np.array([(m.bu >> i) & 1 for i in range(m.V)], dtype=float)
    nd = np.array([(m.bd >> i) & 1 for i in range(m.V)], dtype=float)
    szs = 0.5 * (nu - nd)
    eta = np.array([1.0 if b[2] == "x" else -1.0 for b in m.pair_bonds])
    cen = [m.bond_center(b) for b in m.pair_bonds]
    ex1 = {(i, j): ob._exchange([(i, j)]) for i in range(m.V) for j in range(i + 1, m.V)}
    ops = []
    for mode in modes:
        fs = np.array([f(mode, m.coords[i]) for i in range(m.V)])
        rho = fs @ (ob.n_site_occ - ob.nbar)
        ops.append(lambda v, d=rho ** 2: d * v)
        dg = (fs @ szs) ** 2
        Ssum = sp.diags(dg) + sum(2 * fs[i] * fs[j] * E for (i, j), E in ex1.items()) \
            + sp.diags(sum(fs[i] ** 2 * (0.75 * (ob.n_site_occ[i] % 2) - szs[i] ** 2) for i in range(m.V)))
        ops.append(lambda v, S=Ssum.tocsr(): S @ v)
        fb = eta * np.array([f(mode, c) for c in cen])
        Mb = np.outer(fb, fb)
        ops.append(lambda v, Mb=Mb: ob.apply_pieces_all("pair", v, [Mb])[0])
    return ops, len(modes)


def run_instance(I, rng):
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in ALL}
    tmops, nmodes = two_mode_all(I)
    rows = []
    for lab, Vop in t1.perturbations(I).items():
        psi = t1.trial_state(I, Vop, rng)          # same stream use as round 8 for the trial state itself
        if psi @ I.psi0 < 0:
            psi = -psi
        Epsi = float(psi @ (I.m.H @ psi))
        r = I.m.H @ psi - Epsi * psi
        _, E_L1 = t1.ritz(I, [psi, r], [])
        s = Epsi - 2.0 * (Epsi - E_L1)
        a_psi = {k: float(psi @ I.apply(k, psi)) for k in ALL}
        delta = {k: a0[k] - a_psi[k] for k in ALL}
        row = dict(size=I.name, state=lab, delta=delta, a0=a0,
                   scored={k: bool(abs(delta[k]) >= t1.SCORE_REL * abs(a0[k])) for k in ALL},
                   sf_rel={k: delta[PARTNER[k]] / a0[PARTNER[k]] for k in REAL})

        def score(arm, vecs):
            out, _ = t1.ritz(I, vecs, REAL)
            row[arm] = {k: out[k] / delta[k] for k in REAL}
            row.setdefault("span", {})[arm] = len(vecs)
        score("L1R", [psi, r])
        score("ORC3", [psi, r] + [I.U[:, n] for n in range(min(3, I.n_low))])
        y = {k: unit(I.solve(I.apply(k, psi) - a_psi[k] * psi, s, orth=psi)) for k in REAL}
        rng9 = np.random.default_rng(900 + len(rows) + 31 * I.m.V)
        gnoise = {k: unit(I.U @ rng9.standard_normal(I.n_low)) for k in REAL}
        jnoise = {k: I.junk(rng9, unit(psi)) for k in REAL}
        yt = {g: [unit(y[k] + el * gnoise[k] + eh * jnoise[k]) for k in REAL] for g, (el, eh) in GR.items()}
        score("RWR_R", [psi, r] + yt["R"])
        for g in GR:
            score(f"RWRK_{g}", [psi, r] + yt[g] + [krylov(I, v) for v in yt[g]])
        phis = []
        for n in range(min(3, I.n_low)):
            gk = unit(I.U @ rng9.standard_normal(I.n_low))
            gk -= I.U[:, n] * (I.U[:, n] @ gk)
            phis.append(unit(I.U[:, n] + 0.2 * unit(gk) + 0.2 * I.junk(rng9, I.U[:, n])))
        score("ESR3n", [psi, r] + phis)
        score("EDR_R", [psi, r] + phis + yt["R"] + [krylov(I, v) for v in yt["R"]])
        tm = []
        for op in tmops:
            v = op(psi)
            v = I.proj(v - (psi @ v) * psi)
            if np.linalg.norm(v) > 1e-10:
                tm.append(v)
        score("TMWall", [psi, r] + tm)
        score("TMWK", [psi, r] + tm + [krylov(I, v) for v in tm])
        rows.append(row)
        log(f"{I.name} {lab:7s} " + " | ".join(
            f"{k}{'' if row['scored'][k] else '(ns)'} d/a0 {delta[k] / a0[k]:+.2f} sf {row['sf_rel'][k]:+.3f}: "
            + " ".join(f"{a} {row[a][k]:+.2f}" for a in ARMS) for k in REAL))
    log(f"{I.name}: modes {nmodes}, TMW vectors {len(tm)}, sector levels found {I.n_found}")
    return rows


def frac(rows, arm, sizes, reallike=False):
    it = [(r, k) for r in rows for k in REAL if r["scored"][k] and r["size"] in sizes
          and (not reallike or abs(r["sf_rel"][k]) < 0.05)]
    if not it:
        return float("nan"), 0
    return sum(1 for r, k in it if 0.5 <= r[arm][k] <= 2.0) / len(it), len(it)


def verdicts(rows):
    trend = ["3x3", "6x2", "4x3"]
    allsz = ["3x2"] + trend
    for arm in ARMS:
        f, n = frac(rows, arm, trend)
        fr, nr = frac(rows, arm, trend, True)
        print(f"  tracked {arm:7s} trend-sizes {f:.2f} (n={n}) | " + "  ".join(
            f"{s} {frac(rows, arm, [s])[0]:.2f}" for s in allsz) + f" | real-like {fr:.2f} (n={nr})")
    o3 = frac(rows, "ORC3", ["4x3"])[0]
    out = {}
    for route, arm in (("9.1", "RWRK_R"), ("9.2", "EDR_R"), ("9.3", "TMWK")):
        f = frac(rows, arm, trend)[0]
        lo = frac(rows, arm, ["3x3"])[0]
        hi = frac(rows, arm, ["4x3"])[0]
        if f < 0.5 or hi < lo - 0.25:
            v = "KILL"
        elif f >= 0.75 and hi >= lo - 0.1 and hi >= o3 + 0.2:
            v = "PASS"
            if route == "9.2" and hi < frac(rows, "RWRK_R", ["4x3"])[0] + 0.1:
                v = "UNCLEAR"
        else:
            v = "UNCLEAR"
        out[route] = dict(arm=arm, overall=f, f3x3=lo, f4x3=hi, ORC3_4x3=o3, verdict=v)
        print(f"ROUTE {route} ({arm}): overall {f:.2f}  3x3 {lo:.2f}  4x3 {hi:.2f}  ORC3(4x3) {o3:.2f} -> {v}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("smoke", "main"), required=True)
    p.add_argument("--n-low", type=int, default=60)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--out", default=os.path.join(HERE, "results", "t1_r9.json"))
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    log(f"RLIMIT_CPU set to {a.cpu_limit} s before any ED")
    t0 = time.time()
    sizes = ["3x2"] if a.mode == "smoke" else ["3x2", "3x3", "6x2", "4x3"]
    rows = []
    for s in sizes:
        I = t1.Inst(s, a.n_low)
        rows += run_instance(I, np.random.default_rng(7))
    if a.mode == "main":
        out = verdicts(rows)
        if a.out != "none":
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as fh:
                json.dump(dict(rows=rows, verdicts=out), fh, indent=1, default=str)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    log(f"total wall {time.time() - t0:.0f}s, CPU {ru.ru_utime + ru.ru_stime:.0f}s")


if __name__ == "__main__":
    main()
