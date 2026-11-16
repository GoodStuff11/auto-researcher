#!/usr/bin/env python3
"""
P3.1 round 9, Generation Batch 10: tier-1 toy runs for Routes 10.1 (SCR), 10.2 (TMW-SCR), 10.3 (TMW-K2).
=======================================================================================================

Same instances, trial states (identical to t1_r9.py: the shared stream np.random.default_rng(7) is consumed
only by trial_state), scoring and size comparison as t1_r9.py.

Arms (Ritz of H; ratio = (a(Ritz) - a(psi)) / delta on the real-space tails pair / spin / charge):
  ORC3, L1R     as round 8/9 (comparison and blind control)
  RWRK_R        t1_r9 arm 9.1 (recomputed with the same noise stream, for the 10.2 comparison)
  RWRK_0        exact response of psi, with Krylov partner (attribution)
  RWRstarK      exact response of psi_0, with Krylov partner (attribution; not available in practice)
  SCR2          route 10.1: phi_0 = psi; for t = 0, 1: y~_A(phi_t) (grade R, fresh noise), Krylov partners;
                phi_{t+1} = lowest Ritz vector of span{psi, r, y~(phi_t), K y~(phi_t)}; score phi_2
  TMWK          t1_r9 arm 9.3 (recomputed)
  TMWSCR        route 10.2: phi_1 = TMWK Ritz vector; Ritz in TMWK span + y~_A(phi_1) (grade R) + partners
  TMWK2         route 10.3: TMW span with two orthogonalised Krylov partners per vector
Response of a state phi: y = (Q(H - s)Q)^{-1} Q (A - a_phi) phi, Q = 1 - |phi><phi|,
s = E_phi - 2 (E_phi - E_L1R(phi)) (round 8 convention).

Rules (leaf, Round 9, Batch 10 pre-registration, written before --mode main): as t1_r9 rule 9.1 per arm;
TMWSCR must also beat TMWK and RWRK_R at 4x3 by >= 0.1 to PASS.

Usage
  python3 t1_r10.py --mode smoke              # 3x2 only
  python3 t1_r10.py --mode main --out results/t1_r10.json
  python3 t1_r10.py --mode main --overlap2 0.92 --out results/t1b_r10.json    # T1b (calibrated error weight)
Options: --mode {smoke, main}; --n-low INT (60); --cpu-limit INT (600 CPU-s, set before any ED); --out PATH;
  --overlap2 FLOAT: T1b; error scaled so that the final |<psi_0|psi>|^2 equals it (junk keeps 2x the energy
  error of the first-order part). Default: the round-8 model. Routes 9.1 (RWRK_R) and 9.3 (TMWK) are also
  given a verdict line under the same rule (the T1b re-opening rule).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("t1_r9", os.path.join(HERE, "t1_r9.py"))
r9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r9)
t1, REAL, ALL, log, unit, krylov = r9.t1, r9.REAL, r9.ALL, r9.log, r9.unit, r9.krylov
ARMS = ("L1R", "ORC3", "RWRK_R", "RWRK_0", "RWRstarK", "SCR2", "TMWK", "TMWSCR", "TMWK2")
ETA = (0.3, 0.5)


def ritz_vec(I, vecs):
    B = np.array([unit(v) for v in vecs])
    HB = np.array([I.m.H @ b for b in B])
    S = 0.5 * (B @ B.T + (B @ B.T).T); H = B @ HB.T; H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ H @ X)
    return unit((X @ Y[:, 0]) @ B)


def responses(I, phi, rng=None):
    phi = unit(phi)
    E = float(phi @ (I.m.H @ phi))
    r = I.m.H @ phi - E * phi
    _, EL1 = t1.ritz(I, [phi, r], [])
    s = E - 2.0 * (E - EL1)
    ys = []
    for k in REAL:
        a = float(phi @ I.apply(k, phi))
        y = unit(I.solve(I.apply(k, phi) - a * phi, s, orth=phi))
        if rng is not None:
            y = unit(y + ETA[0] * unit(I.U @ rng.standard_normal(I.n_low)) + ETA[1] * I.junk(rng, phi))
        ys.append(y)
    return ys


def krylov2(I, v):
    k1 = unit(krylov(I, v))
    w = I.m.H @ k1
    w = w - (unit(v) @ w) * unit(v) - (k1 @ w) * k1
    return I.proj(w)


OV2 = None   # set by --overlap2: target |<psi_0|psi>|^2 of the final trial state (junk included)


def trial_state(I, Vop, rng):
    """t1.trial_state, or (with --overlap2) the same construction scaled to |<psi_0|psi>|^2 = OV2."""
    if OV2 is None:
        return t1.trial_state(I, Vop, rng)
    psi0, E0 = I.psi0, I.E0
    w = I.proj(Vop(psi0))
    w -= psi0 * (psi0 @ w)
    e = unit(-I.solve(w, E0, orth=psi0))
    Ee = e @ (I.m.H @ e) - E0
    j = I.junk(rng, psi0)
    Ej = j @ (I.m.H @ j) - E0
    lam2 = (1.0 / OV2 - 1.0) / (1.0 + t1.JUNK_RATIO * Ee / Ej)
    mu = np.sqrt(t1.JUNK_RATIO * lam2 * Ee / Ej)
    v = np.sqrt(lam2) * e + mu * j          # e, j are both orthogonal to psi0; rescale for any e.j overlap
    v *= np.sqrt(1.0 / OV2 - 1.0) / np.linalg.norm(v)
    psi = unit(psi0 + v)
    assert abs((psi @ psi0) ** 2 - OV2) < 1e-8
    return psi


def run_instance(I, rng):
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in ALL}
    ystar = []
    for k in REAL:
        ystar.append(I.solve(I.apply(k, I.psi0) - a0[k] * I.psi0, I.E0, orth=I.psi0))
    tmops, nmodes = r9.two_mode_all(I)
    rows = []
    for lab, Vop in t1.perturbations(I).items():
        psi = trial_state(I, Vop, rng)
        if psi @ I.psi0 < 0:
            psi = -psi
        Epsi = float(psi @ (I.m.H @ psi))
        r = I.m.H @ psi - Epsi * psi
        a_psi = {k: float(psi @ I.apply(k, psi)) for k in ALL}
        delta = {k: a0[k] - a_psi[k] for k in ALL}
        row = dict(size=I.name, state=lab, delta=delta, a0=a0,
                   scored={k: bool(abs(delta[k]) >= t1.SCORE_REL * abs(a0[k])) for k in ALL})

        def score(arm, phi):
            row[arm] = {k: float(phi @ I.apply(k, phi) - a_psi[k]) / delta[k] for k in REAL}
        base = [psi, r]
        score("L1R", ritz_vec(I, base))
        score("ORC3", ritz_vec(I, base + [I.U[:, n] for n in range(min(3, I.n_low))]))
        # RWRK_R with exactly the t1_r9 noise stream
        y = responses(I, psi)
        rng9 = np.random.default_rng(900 + len(rows) + 31 * I.m.V)
        g = {k: unit(I.U @ rng9.standard_normal(I.n_low)) for k in REAL}
        j = {k: I.junk(rng9, unit(psi)) for k in REAL}
        yR = [unit(y[i] + ETA[0] * g[k] + ETA[1] * j[k]) for i, k in enumerate(REAL)]
        score("RWRK_R", ritz_vec(I, base + yR + [krylov(I, v) for v in yR]))
        score("RWRK_0", ritz_vec(I, base + y + [krylov(I, v) for v in y]))
        score("RWRstarK", ritz_vec(I, base + ystar + [krylov(I, v) for v in ystar]))
        rng10 = np.random.default_rng(1000 + len(rows) + 31 * I.m.V)
        phi = psi
        for _ in range(2):
            yt = responses(I, phi, rng10)
            phi = ritz_vec(I, base + yt + [krylov(I, v) for v in yt])
        score("SCR2", phi)
        tm = []
        for op in tmops:
            v = op(psi)
            v = I.proj(v - (psi @ v) * psi)
            if np.linalg.norm(v) > 1e-10:
                tm.append(v)
        tmk = base + tm + [krylov(I, v) for v in tm]
        phiT = ritz_vec(I, tmk)
        score("TMWK", phiT)
        yt = responses(I, phiT, rng10)
        score("TMWSCR", ritz_vec(I, tmk + yt + [krylov(I, v) for v in yt]))
        score("TMWK2", ritz_vec(I, tmk + [krylov2(I, v) for v in tm]))
        rows.append(row)
        log(f"{I.name} {lab:7s} " + " | ".join(
            f"{k}{'' if row['scored'][k] else '(ns)'}: " + " ".join(f"{a} {row[a][k]:+.2f}" for a in ARMS)
            for k in REAL))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("smoke", "main"), required=True)
    p.add_argument("--n-low", type=int, default=60)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--out", default=os.path.join(HERE, "results", "t1_r10.json"))
    p.add_argument("--overlap2", type=float, default=None,
                   help="T1b: scale the error so that |<psi_0|psi>|^2 = OVERLAP2 (default: round-8 model)")
    a = p.parse_args()
    global OV2
    OV2 = a.overlap2
    log(f"error model: {'round-8 (overlap amplitude 0.94 + junk)' if OV2 is None else f'|<psi0|psi>|^2 = {OV2}'}")
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    log(f"RLIMIT_CPU set to {a.cpu_limit} s before any ED")
    t0 = time.time()
    sizes = ["3x2"] if a.mode == "smoke" else ["3x2", "3x3", "6x2", "4x3"]
    rows = []
    for s in sizes:
        I = t1.Inst(s, a.n_low)
        rows += run_instance(I, np.random.default_rng(7))
    if a.mode == "main":
        trend = ["3x3", "6x2", "4x3"]
        for arm in ARMS:
            f, n = r9.frac(rows, arm, trend)
            print(f"  tracked {arm:8s} trend-sizes {f:.2f} (n={n}) | " + "  ".join(
                f"{s} {r9.frac(rows, arm, [s])[0]:.2f}" for s in ["3x2"] + trend))
        o3 = r9.frac(rows, "ORC3", ["4x3"])[0]
        out = {}
        for route, arm in (("10.1", "SCR2"), ("10.2", "TMWSCR"), ("10.3", "TMWK2"), ("9.1", "RWRK_R"), ("9.3", "TMWK")):
            f = r9.frac(rows, arm, trend)[0]
            lo, hi = r9.frac(rows, arm, ["3x3"])[0], r9.frac(rows, arm, ["4x3"])[0]
            if f < 0.5 or hi < lo - 0.25:
                v = "KILL"
            elif f >= 0.75 and hi >= lo - 0.1 and hi >= o3 + 0.2:
                v = "PASS"
                if route == "10.2" and hi < max(r9.frac(rows, "TMWK", ["4x3"])[0],
                                                r9.frac(rows, "RWRK_R", ["4x3"])[0]) + 0.1:
                    v = "UNCLEAR"
            else:
                v = "UNCLEAR"
            out[route] = dict(arm=arm, overall=f, f3x3=lo, f4x3=hi, ORC3_4x3=o3, verdict=v)
            print(f"ROUTE {route} ({arm}): overall {f:.2f}  3x3 {lo:.2f}  4x3 {hi:.2f}  ORC3(4x3) {o3:.2f} -> {v}")
        if a.out != "none":
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as fh:
                json.dump(dict(rows=rows, verdicts=out), fh, indent=1, default=str)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    log(f"total wall {time.time() - t0:.0f}s, CPU {ru.ru_utime + ru.ru_stime:.0f}s")


if __name__ == "__main__":
    main()
