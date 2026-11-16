#!/usr/bin/env python3
"""
P3.1 T3 design critic (2026-11-16): exact check of a missing Gate 2 foil, on the 4x3 T1 trial states.

Question. Does the observable information in the 10.1d sources matter, or does any shift-inverted correction of
the same depth do as well? Compares, in exact arithmetic (minres responses, no training noise), on the eight 4x3
T1 trial states of t1_scru.py (fixed per-site infidelity, same rng stream):
  SCRd0   10.1d with the observable sources b_A = unit(Q (A - a) phi), A = pair, spin, charge (the T3 arm).
  SCRdE0  the same pipeline with sources that carry no observable information: b_p = unit(Q (H - E)^p phi),
          p = 1, 2, 3 (generalised Davidson with three Krylov-sourced shift-invert vectors).
  SD4     four restarted span{phi, (H-E) phi} steps (the T1 foil), for reference.
Shift s = E_L1R(phi) - 0.1 t; T = 4; pass 1 span {phi, (H-E)phi, y, K y}, passes 2-4 span {phi, y}.
Scored: |delta| >= 0.10 |a_0|; tracked: ratio in [1/2, 2].
Usage: python3 critic_t3_foil_check.py --threads 4
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("t1s", os.path.join(HERE, "t1_scru.py"))
t1s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1s)
np, t1, unit, log, REAL = t1s.np, t1s.t1, t1s.unit, t1s.log, t1s.REAL


def chain(I, psi, sources):
    phi = psi
    for t in range(1, 5):
        phi = unit(phi)
        _, EL1 = t1s.ritz_vec(I, [phi, t1s.resid(I, phi)])
        s = EL1 - 0.1
        assert s < I.E0
        ys = [unit(I.solve(b, s, orth=phi)) for b in sources(phi)]
        if t == 1:
            phi = t1s.scru_pass(I, phi, ys)
        else:
            phi = t1s.ritz_vec(I, [phi] + ys)[0]
    return phi


def obs_sources(I):
    def f(phi):
        out = []
        for k in REAL:
            a = float(phi @ I.apply(k, phi))
            out.append(I.apply(k, phi) - a * phi)
        return out
    return f


def krylov_sources(I):
    def f(phi):
        out, v = [], phi
        E = float(phi @ (I.m.H @ phi))
        for _ in range(3):
            v = I.proj(I.m.H @ v - E * v)
            out.append(v - phi * (phi @ v))
        return out
    return f


def main():
    I, info = t1s.build("4x3", 15)
    perts = t1.perturbations(I)
    rng = np.random.default_rng(7)
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
    tally = {"SCRd0": 0, "SCRdE0": 0, "SD4": 0, "L1R": 0}
    n = 0
    for lab in perts:
        psi = t1s.trial_state(I, perts[lab], rng)
        ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
        dl = {k: a0[k] - ap[k] for k in REAL}
        sc = {k: abs(dl[k]) >= t1.SCORE_REL * abs(a0[k]) for k in REAL}
        phis = {"SCRd0": chain(I, psi, obs_sources(I)), "SCRdE0": chain(I, psi, krylov_sources(I)),
                "L1R": t1s.ritz_vec(I, [psi, t1s.resid(I, psi)])[0]}
        ph = psi
        for _ in range(4):
            ph = t1s.ritz_vec(I, [ph, t1s.resid(I, ph)])[0]
        phis["SD4"] = ph
        line = []
        for k in REAL:
            rr = {a: (float(p @ I.apply(k, p)) - ap[k]) / dl[k] for a, p in phis.items()}
            if sc[k]:
                n += 1
                for a, x in rr.items():
                    tally[a] += int(0.5 <= x <= 2.0)
            line.append(f"{k}{'' if sc[k] else '(ns)'} " + " ".join(f"{a} {x:+.2f}" for a, x in rr.items()))
        log(f"4x3 {lab:8s} | " + " | ".join(line))
    log(f"CRITIC FOIL CHECK 4x3 (exact): scored items {n}; tracked " + ", ".join(f"{a} {c}" for a, c in tally.items()))


if __name__ == "__main__":
    sys.exit(main())
