#!/usr/bin/env python3
"""
P3.1 Round 11, task A: tier-1 test of Route 10.4 (observable-free generalised-Davidson error bars).
===================================================================================================

Route 10.4. From a trial state psi, build phi_T by the 10.1d chain with OBSERVABLE-FREE sources
b_p = unit(Q (H - E)^p phi), p = 1, 2, 3 (the T3 foil F3 / critic arm SCRdE0), and report
delta-hat_A = a(phi_T) - a(psi) for EVERY observable A at once. Here every generated vector carries
realistic (grade-R) noise instead of being an exact minres solve.

Chain (identical to critic_t3_foil_check.chain): shift s = E_L1R(phi) - 0.1 t (assert s < E_0);
pass 1: phi_1 = lowest Ritz vector of span{phi, (H-E)phi, y_j, K y_j}; passes 2..4: span{phi, y_j};
y_j = noisy unit((Q(H-s)Q)^{-1} b_j). Noise (grade R of t1_scru.py, fresh draw per vector):
y~ = unit(unit(y) + 0.3 g + 0.5 j_y), g a random unit vector on the in-sector levels with omega <= 1 t,
j_y high-energy junk.

Arms (ratio = (a(phi) - a(psi)) / delta on the tails pair / spin / charge; tracked if in [1/2, 2]):
  DAV-R   10.4, sources b_p (deciding)
  OBS-R   10.1d, observable sources b_A = unit(Q (A - a) phi) (reported reference, same noise)
  RND-R   Gate 2 foil: sources = unit(Q r_j), r_j random in-sector vectors (shift-invert, no residual info)
  SD4     Gate 2 foil: four restarted span{phi, (H-E)phi} steps (Krylov without shift-invert)
  L1R     negative control span{psi, (H-E)psi}

Instances (OBC t-t'-J, t' = -0.25, J = 0.4): 4x2 (3+3, V = 8), 5x2 (3+3, V = 10), 4x3 (4+4, V = 12).
Validity asserts (t1_scru.build): non-degenerate psi_0, omega_1 >= 0.02 t, window omega <= 1 t covered,
>= 3 window levels.
Trial states: t1_scru.trial_state (8 perturbations, fixed per-site infidelity 0.92^(V/16), rng(7)).

Gate 3 stress ("gap surgery", --delta D): H is replaced by H_D = H - (omega_1 - D)|1><1| (the lowest
in-sector excited level moved down to omega = D; every other level, every eigenvector and E_0 unchanged).
Trial states are rebuilt with H_D, so their first-order part carries the larger |1> admixture that a
near-degenerate competitor produces. This isolates the T0 prediction: the in-plane part of a
shift-inverted residual is proportional to D / (E_0 - s), the in-plane part of an observable source is not.

Pre-registered rule: leaf, "Round 11 thinking work (2026-11-16)", written before any --mode main run.

Usage
  python3 t1_davidson.py --mode census --size 4x2
  python3 t1_davidson.py --mode main --size 4x3 [--delta 0.02]   # JSON results/dav_<size>[_d0.02].json
  python3 t1_davidson.py --mode verdict
Options
  --mode {census, main, verdict}
  --size {4x2, 5x2, 4x3}
  --delta FLOAT   gap surgery: move level 1 to omega = delta (must be >= 0.02 and < omega_2); default none
  --states INT    number of trial states (default 8)
  --n-low INT     eigsh levels (default 15)
  --cpu-limit INT hard RLIMIT_CPU seconds (default 600)
  --threads {1,2,3,4}  BLAS threads (default 1)
  --exact-arms    also run DAV-0 and OBS-0 (the same chains without noise; reported attribution only).
                  Output then goes to results/dav_<tag>_exact.json so the deciding files are not overwritten.
"""
import os
import sys
_nt = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _nt
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R10 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r10")
_spec = importlib.util.spec_from_file_location("t1s", os.path.join(R10, "t1_scru.py"))
t1s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1s)
np, t1, unit, log, REAL = t1s.np, t1s.t1, t1s.unit, t1s.log, t1s.REAL
t1.SIZES.update({"4x2": dict(Lx=4, Ly=2, nup=3, ndn=3)})
SIZES = ("4x2", "5x2", "4x3")
STRESS = 0.02
ARMS = ("L1R", "SD4", "DAV-R", "OBS-R", "RND-R")
EXACT_ARMS = "--exact-arms" in __import__("sys").argv
RES = os.path.join(HERE, "results")
ETA = (0.3, 0.5)


class HSurg:
    """H - c |u><u| as a matmul-able operator."""

    def __init__(self, H, u, c):
        self.H, self.u, self.c, self.shape = H, u, c, H.shape

    def __matmul__(self, v):
        return self.H @ v - self.c * self.u * (self.u @ v)

    def __rsub__(self, other):
        # t1_portfolio.perturbations forms (H' - H) for the t' and J perturbations; the perturbation operators
        # stay the native ones (V is a perturbation, not the Hamiltonian)
        return other - self.H


def cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def noisy(I, y, rng, base):
    if rng is None:
        return unit(y)
    g = unit(I.Uw @ rng.standard_normal(I.nwin))
    return unit(unit(y) + ETA[0] * g + ETA[1] * I.junk(rng, base))


def sources(I, kind, phi, rng):
    if kind == "OBS":
        return [I.apply(k, phi) - float(phi @ I.apply(k, phi)) * phi for k in REAL]
    if kind == "DAV":
        out, v = [], phi
        E = float(phi @ (I.m.H @ phi))
        for _ in range(3):
            v = I.proj(I.m.H @ v - E * v)
            out.append(v - phi * (phi @ v))
        return out
    if kind == "RND":
        out = []
        for _ in range(3):
            r = I.proj(rng.standard_normal(I.m.D))
            out.append(r - phi * (phi @ r))
        return out
    raise KeyError(kind)


def chain(I, psi, kind, rng):
    phi = psi
    for t in range(1, 5):
        phi = unit(phi)
        _, EL1 = t1s.ritz_vec(I, [phi, t1s.resid(I, phi)])
        s = EL1 - 0.1
        assert s < I.E0, ("VALIDITY: shift not below E0", s, I.E0)
        ys = [noisy(I, I.solve(b, s, orth=phi), rng, phi) for b in sources(I, kind, phi, rng)]
        if t == 1:
            phi = t1s.scru_pass(I, phi, ys)
        else:
            phi = t1s.ritz_vec(I, [phi] + ys)[0]
    return phi


def build(size, n_low, delta):
    I, info = t1s.build(size, n_low)
    if delta is not None:
        assert delta >= 0.02 and delta < I.om[1], ("surgery gap out of range", delta, I.om[:2])
        I.m.H = HSurg(I.m.H, I.U[:, 0], float(I.om[0]) - delta)
        I.om = I.om.copy()
        I.om[0] = delta
        u = I.U[:, 0]
        assert abs(float(u @ (I.m.H @ u)) - I.E0 - delta) < 1e-8
        info["surgery"] = dict(omega1_native=info["omega1"], delta=delta)
        log(f"gap surgery: level 1 moved from omega {info['omega1']:.4f} to {delta:.4f}")
    return I, info


def run(I, nstates, cap, dump):
    perts = t1.perturbations(I)
    rng = np.random.default_rng(7)
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
    rows, worst = [], 0.0
    for si, lab in enumerate(list(perts)[:nstates]):
        c0 = cpu()
        if cpu() + 1.2 * worst > cap - 15:
            log(f"soft CPU guard: stop before state {si}")
            break
        psi = t1s.trial_state(I, perts[lab], rng)
        ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
        dl = {k: a0[k] - ap[k] for k in REAL}
        w1 = float(I.U[:, 0] @ psi) ** 2
        row = dict(size=I.name, state=lab, delta=dl, w1=w1,
                   scored={k: bool(abs(dl[k]) >= t1.SCORE_REL * abs(a0[k])) for k in REAL})
        phis = {"L1R": t1s.ritz_vec(I, [psi, t1s.resid(I, psi)])[0]}
        ph = psi
        for _ in range(4):
            ph = t1s.ritz_vec(I, [ph, t1s.resid(I, ph)])[0]
        phis["SD4"] = ph
        rs = np.random.default_rng(20_000 + 97 * I.m.V + si)
        for kind in ("DAV", "OBS", "RND"):
            phis[f"{kind}-R"] = chain(I, psi, kind, rs)
        if EXACT_ARMS:   # reported attribution arms, noise-free (no rng use, so the deciding arms are unchanged)
            for kind in ("DAV", "OBS"):
                phis[f"{kind}-0"] = chain(I, psi, kind, None)
        for arm, p in phis.items():
            row[arm] = {k: float(p @ I.apply(k, p) - ap[k]) / dl[k] for k in REAL}
        rows.append(row)
        worst = max(worst, cpu() - c0)
        json.dump(dict(rows=rows), open(dump, "w"), indent=1)
        log(f"{I.name} {lab:7s} |<1|psi>|^2 {w1:.4f} cpu {cpu() - c0:.0f}s | " + " | ".join(
            f"{k}{'' if row['scored'][k] else '(ns)'} " + " ".join(f"{a} {row[a][k]:+.2f}" for a in phis)
            for k in REAL))
    return rows


def frac(rows, arm):
    it = [(r, k) for r in rows for k in REAL if r["scored"][k]]
    if not it:
        return float("nan"), 0
    return sum(1 for r, k in it if 0.5 <= r[arm][k] <= 2.0) / len(it), len(it)


def verdict():
    F, n = {}, {}
    for s in SIZES:
        for tag in (s, f"{s}_d{STRESS}"):
            f = os.path.join(RES, f"dav_{tag}.json")
            if os.path.exists(f):
                rows = json.load(open(f))["rows"]
                for arm in ARMS:
                    F[(tag, arm)], n[tag] = frac(rows, arm)
                print(f"  {tag:10s} n={n[tag]:2d} " + " ".join(f"{a} {F[(tag, a)]:.2f}" for a in ARMS))
    need = list(SIZES) + [f"{s}_d{STRESS}" for s in SIZES]
    if any(t not in n or n[t] < 6 for t in need):
        print("VERDICT 10.4 T1: INCOMPLETE (a size or stress run missing or < 6 scored items)")
        return
    d = {s: F[(s, "DAV-R")] for s in SIZES}
    foil = {s: max(F[(s, "SD4")], F[(s, "RND-R")]) for s in SIZES}
    st = {s: F[(f"{s}_d{STRESS}", "DAV-R")] for s in SIZES}
    big = ("5x2", "4x3")
    if any(d[s] < 0.5 for s in big):
        v = "KILL (tracking)"
    elif all(foil[s] >= d[s] for s in big):
        v = "KILL (Gate 2)"
    elif all(st[s] < 0.5 for s in SIZES):
        v = "KILL (Gate 3 stress)"
    elif (all(d[s] >= 0.75 for s in SIZES) and all(d[s] >= foil[s] + 0.1 for s in big)
          and all(st[s] >= 0.5 for s in SIZES)):
        v = "UNCLEAR (Gate 4: drop > 0.15 from V=8 to V=12)" if d["4x2"] - d["4x3"] > 0.15 else "PASS"
    else:
        v = "UNCLEAR"
    print("VERDICT 10.4 T1: " + v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("census", "main", "verdict"), required=True)
    p.add_argument("--size", choices=SIZES, default="4x2")
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--n-low", type=int, default=15)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--exact-arms", action="store_true", help="add noise-free DAV-0 and OBS-0 (reported only)")
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    os.makedirs(RES, exist_ok=True)
    if a.mode == "verdict":
        return verdict()
    I, info = build(a.size, a.n_low, a.delta)
    log(f"build cpu {cpu():.0f}s")
    if a.mode == "main":
        tag = a.size if a.delta is None else f"{a.size}_d{a.delta}"
        out = os.path.join(RES, f"dav_{tag}{'_exact' if a.exact_arms else ''}.json")
        rows = run(I, a.states, a.cpu_limit, out)
        json.dump(dict(info=info, rows=rows), open(out, "w"), indent=1, default=str)
        for arm in [a for a in rows[0] if a in ARMS or a.endswith("-0")]:
            fr, n = frac(rows, arm)
            print(f"  tracked {arm:6s} {tag} {fr:.2f} (n={n})")
    log(f"total CPU {cpu():.0f}s")


if __name__ == "__main__":
    main()
