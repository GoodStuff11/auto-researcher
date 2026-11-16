#!/usr/bin/env python3
"""
P3.1 Round 12: tier-1 test of Routes 12.1 (observable-sourced gap witness, OGW), 12.3 (cluster-block bracket, CBR)
and 12.4 (multi-source response, MSR). Exact linear algebra on ED toys, grade-R noise on every generated vector.
====================================================================================================================

Pre-registered rules: leaf .subproblems/P3.1_order_parameter_error.md, "Round 12: refill independent of the T3",
"T1 pre-registration for Batch 12", written before any run of this script.

Instances (OBC t-t'-J, t' = -0.25, J = 0.4; built by ../P3.1_order_parameter_error_r11/t1_davidson.build, which
asserts a non-degenerate psi_0, omega_1 >= 0.02 t, window omega <= 1 t covered and >= 3 window levels):
  4x2 (3+3, V = 8), 5x2 (3+3, V = 10), 4x3 (4+4, V = 12); native, or with gap surgery --delta 0.02
  (H -> H - (omega_1 - delta)|1><1|; trial states rebuilt with the surgered H).
Trial states: t1_scru.trial_state, the 8 perturbations of t1_portfolio, fixed per-site infidelity 0.92^(V/16).
Noise (grade R, t1_davidson.noisy): y~ = unit(unit(y) + 0.3 g + 0.5 j), fresh draw per vector.
Shift: s = E_L1R(phi) - 0.1 t (assert s < E_0).

Chains (lowest Ritz vector of H):
  pass 1      span{phi, (H-E)phi, y~_i, K y~_i}       K v = P(H - <H>_v) v
  passes 2..T span{phi, y~_i}                          (depth-free, as 10.1d)
  witness     span{phi_T, (H-E)phi_T, y~_i(phi_T), K y~_i(phi_T)}   -> Ritz values theta_0 < theta_1,
              g-hat = theta_1 - theta_0; block = the two lowest Ritz vectors
Sources:
  OBS   unit-free Q(A - a)phi for the three tails pair / spin / charge          (10.1d; T = 4; + witness pass)
  DAV   Q(H - E)^p phi, p = 1, 2, 3                                              (10.4 sources; T = 4; + witness)
  MSR   the 4 longest distance-class pieces of each correlator (r5_core pieces), m <= 12  (T = 2; Route 12.4)
  RND   12 random in-sector vectors                                             (T = 2; stress runs only; foil)
  L1R   witness of span{psi, (H-E)psi} (no chain)

Scoring per scored item (|delta| >= 0.1 |a_0|): tracked if the ratio (a(phi) - a(psi))/delta lies in [1/2, 2];
certified if g-hat >= 0.1; covered if a_0 in [lo - 0.1|delta|, hi + 0.1|delta|], [lo, hi] the eigenvalue range of
the 2x2 compression of A on the block; width ratio (hi - lo)/w_ex, w_ex the eigenvalue range of A on the exact
plane span{psi_0, psi_1}. Reported: sum_i alpha_i^2 in pass 1 (alpha_i^2 = |<psi_1|y~_i>|^2) for OBS and MSR.

Usage
  python3 t1_r12.py --mode main --size 4x2 [--delta 0.02] [--states 8]   # JSON results/r12_<size>[_d0.02].json
  python3 t1_r12.py --mode verdict                                        # print tables and the three verdicts
Options
  --mode {main, verdict}
  --size {4x2, 5x2, 4x3}
  --delta FLOAT   gap surgery target (>= 0.02, < omega_2); default none (native)
  --states INT    number of trial states, taken in the fixed order (default 8)
  --n-low INT     eigsh levels (default 15)
  --cpu-limit INT hard RLIMIT_CPU seconds (default 600); a soft guard starts a state only if it projects to fit
  --threads {1,2,3,4}  BLAS threads (default 1)
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
_spec = importlib.util.spec_from_file_location(
    "dav", os.path.join(HERE, "..", "P3.1_order_parameter_error_r11", "t1_davidson.py"))
dav = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]          # t1_davidson reads sys.argv at import time (--exact-arms); keep it clean
_spec.loader.exec_module(dav)
sys.argv = _argv
t1s, t1, np, unit, log, REAL = dav.t1s, dav.t1, dav.np, dav.unit, dav.log, dav.REAL
SIZES = ("4x2", "5x2", "4x3")
STRESS = 0.02
GSTAR = 0.1
RES = os.path.join(HERE, "results")


def cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def ritz_all(I, vecs):
    B = np.array([unit(v) for v in vecs])
    HB = np.array([I.m.H @ b for b in B])
    S = B @ B.T; S = 0.5 * (S + S.T)
    H = B @ HB.T; H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ H @ X)
    C = X @ Y
    return ev, [unit(C[:, k] @ B) for k in range(min(2, C.shape[1]))]


def make_sources(I, kind, rng_src):
    if kind == "OBS":
        return lambda phi: [I.apply(k, phi) - float(phi @ I.apply(k, phi)) * phi for k in REAL]
    if kind == "DAV":
        def f(phi):
            out, v = [], phi
            E = float(phi @ (I.m.H @ phi))
            for _ in range(3):
                v = I.proj(I.m.H @ v - E * v)
                out.append(v - phi * (phi @ v))
            return out
        return f
    if kind == "MSR":
        ob = I.ob
        plist = [(k, P) for k in REAL for P in ob.pieces[k][-4:]]

        def f(phi):
            out = []
            for k, P in plist:
                w = ob.apply_piece(k, P, phi)
                out.append(w - float(phi @ w) * phi)
            return out
        return f
    if kind == "RND":
        def f(phi):
            out = []
            for _ in range(12):
                r = I.proj(rng_src.standard_normal(I.m.D))
                out.append(r - phi * (phi @ r))
            return out
        return f
    raise KeyError(kind)


def noisy_resp(I, phi, src, rng):
    _, EL1 = t1s.ritz_vec(I, [phi, t1s.resid(I, phi)])
    s = EL1 - 0.1
    assert s < I.E0, ("VALIDITY: shift not below E0", s, I.E0)
    return [dav.noisy(I, I.solve(b, s, orth=phi), rng, phi) for b in src(phi)]


def chain(I, psi, src, rng, T, witness):
    phi, info = unit(psi), {}
    for t in range(1, T + 1):
        ys = noisy_resp(I, phi, src, rng)
        if t == 1:
            info["alpha2"] = float(sum((I.U[:, 0] @ y) ** 2 for y in ys))
            phi = t1s.scru_pass(I, phi, ys)
        else:
            phi = t1s.ritz_vec(I, [phi] + ys)[0]
        phi = unit(phi)
    if witness:
        ys = noisy_resp(I, phi, src, rng)
        ev, blk = ritz_all(I, [phi, t1s.resid(I, phi)] + ys + [t1s.krylov(I, v) for v in ys])
        info["ghat"] = float(ev[1] - ev[0])
        info["block"] = blk
    return phi, info


def bracket(I, k, vecs):
    AV = [I.apply(k, v) for v in vecs]
    M = np.array([[float(a @ b) for b in AV] for a in vecs])
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    return float(ev[0]), float(ev[-1])


def run(I, nstates, cap, dump, stress):
    perts = t1.perturbations(I)
    rng = np.random.default_rng(7)
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
    plane = [I.psi0, I.U[:, 0]]
    wex = {k: (lambda b: b[1] - b[0])(bracket(I, k, plane)) for k in REAL}
    rows, worst = [], 0.0
    for si, lab in enumerate(list(perts)[:nstates]):
        c0 = cpu()
        if cpu() + 1.2 * worst > cap - 15:
            log(f"soft CPU guard: stop before state {si}")
            break
        psi = t1s.trial_state(I, perts[lab], rng)
        ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
        dl = {k: a0[k] - ap[k] for k in REAL}
        row = dict(size=I.name, state=lab, a0=a0, delta=dl, wex=wex,
                   w1=float(I.U[:, 0] @ psi) ** 2,
                   scored={k: bool(abs(dl[k]) >= t1.SCORE_REL * abs(a0[k])) for k in REAL})
        ev, _ = ritz_all(I, [psi, t1s.resid(I, psi)])
        row["ghat_L1R"] = float(ev[1] - ev[0])
        rs = np.random.default_rng(30_000 + 97 * I.m.V + si)
        arms = [("OBS", 4, True), ("DAV", 4, True), ("MSR", 2, False)] + ([("RND", 2, False)] if stress else [])
        for kind, T, wit in arms:
            phi, info = chain(I, psi, make_sources(I, kind, rs), rs, T, wit)
            row[kind] = {k: float(phi @ I.apply(k, phi) - ap[k]) / dl[k] for k in REAL}
            row[f"alpha2_{kind}"] = info["alpha2"]
            if wit:
                row[f"ghat_{kind}"] = info["ghat"]
                row[f"br_{kind}"] = {k: bracket(I, k, info["block"]) for k in REAL}
        rows.append(row)
        worst = max(worst, cpu() - c0)
        json.dump(dict(rows=rows), open(dump, "w"), indent=1)
        log(f"{I.name} {lab:7s} w1 {row['w1']:.3f} g(OBS) {row['ghat_OBS']:.3f} g(DAV) {row['ghat_DAV']:.3f} "
            f"g(L1R) {row['ghat_L1R']:.3f} a2 OBS {row['alpha2_OBS']:.2f} MSR {row['alpha2_MSR']:.2f} cpu "
            f"{cpu() - c0:.0f}s | " + " | ".join(
                f"{k}{'' if row['scored'][k] else '(ns)'} " + " ".join(
                    f"{a} {row[a][k]:+.2f}" for a in ("OBS", "DAV", "MSR", "RND") if a in row) for k in REAL))
    return rows


# ============================================================================ scoring and verdicts
def items(rows):
    return [(r, k) for r in rows for k in REAL if r["scored"][k]]


def tracked(r, k, arm):
    return 0.5 <= r[arm][k] <= 2.0


def covered(r, k, arm):
    lo, hi = r[f"br_{arm}"][k]
    tol = 0.1 * abs(r["delta"][k])
    return lo - tol <= r["a0"][k] <= hi + tol


def stats(rows, arm):
    it = items(rows)
    n = len(it)
    if n == 0:
        return None
    cert = [(r, k) for r, k in it if r[f"ghat_{arm}"] >= GSTAR]
    abst = [(r, k) for r, k in it if r[f"ghat_{arm}"] < GSTAR]
    FC = sum(1 for r, k in cert if not tracked(r, k, arm)) / n
    CT = sum(1 for r, k in cert if tracked(r, k, arm)) / n
    cov = (sum(1 for r, k in abst if covered(r, k, arm)) / len(abst)) if abst else float("nan")
    wr = [(r[f"br_{arm}"][k][1] - r[f"br_{arm}"][k][0]) / r["wex"][k] for r, k in abst]
    honest = (sum(1 for r, k in cert if tracked(r, k, arm)) + sum(1 for r, k in abst if covered(r, k, arm))) / n
    return dict(n=n, FC=FC, CT=CT, n_abst=len(abst), cov=cov,
                wr=float(np.median(wr)) if wr else float("nan"), honest=honest)


def frac(rows, arm):
    it = items(rows)
    return (sum(1 for r, k in it if tracked(r, k, arm)) / len(it)) if it else float("nan")


def load(tag):
    f = os.path.join(RES, f"r12_{tag}.json")
    return json.load(open(f))["rows"] if os.path.exists(f) else None


def verdict():
    nat, st = {}, {}
    for s in SIZES:
        nat[s], st[s] = load(s), load(f"{s}_d{STRESS}")
    print("tracked fraction (scored items):")
    for lab, D in (("native", nat), ("stress", st)):
        for s in SIZES:
            rows = D[s]
            if rows is None:
                print(f"  {lab} {s}: missing"); continue
            arms = [a for a in ("OBS", "DAV", "MSR", "RND") if a in rows[0]]
            a2 = {a: float(np.median([r[f"alpha2_{a}"] for r in rows])) for a in ("OBS", "MSR")}
            print(f"  {lab:6s} {s} n={len(items(rows)):2d} states={len(rows)} " + " ".join(
                f"{a} {frac(rows, a):.2f}" for a in arms)
                + f" | median sum alpha^2 OBS {a2['OBS']:.2f} MSR {a2['MSR']:.2f}")
    print("witness and block (FC = certified & untracked, CT = certified & tracked; over scored items):")
    S = {}
    for lab, D in (("native", nat), ("stress", st)):
        for s in SIZES:
            if D[s] is None:
                continue
            for arm in ("OBS", "DAV"):
                S[(lab, s, arm)] = x = stats(D[s], arm)
                print(f"  {lab:6s} {s} {arm}: n={x['n']:2d} FC {x['FC']:.2f} CT {x['CT']:.2f} abstained "
                      f"{x['n_abst']:2d} coverage {x['cov']:.2f} median width ratio {x['wr']:.2f} "
                      f"honest {x['honest']:.2f} | median g-hat "
                      f"{np.median([r['ghat_' + arm] for r in D[s]]):.3f}")
            print(f"  {lab:6s} {s} L1R witness: median g-hat {np.median([r['ghat_L1R'] for r in D[s]]):.3f}")
    need = all(nat[s] is not None and st[s] is not None for s in SIZES)
    if not need or any(len(items(D[s])) < 6 for D in (nat, st) for s in SIZES):
        print("VERDICTS: INCOMPLETE (a run is missing or has < 6 scored items)")
        return

    def ogw(arm):
        if any(S[("stress", s, arm)]["FC"] > 0.25 for s in SIZES) or \
                any(S[("native", s, arm)]["CT"] < 0.5 for s in SIZES):
            return "KILL"
        if all(S[("stress", s, arm)]["FC"] <= 0.1 for s in SIZES) and \
                all(S[("native", s, arm)]["CT"] >= 0.6 for s in SIZES):
            return "PASS"
        return "UNCLEAR"

    def cbr(arm):
        X = [S[("stress", s, arm)] for s in SIZES]
        if any(x["n_abst"] < 4 for x in X):
            return "INCOMPLETE"
        if any(x["cov"] < 0.5 for x in X) or any(x["wr"] > 3 for x in X):
            return "KILL"
        if all(x["cov"] >= 0.8 and x["wr"] <= 1.5 and x["honest"] >= 0.8 for x in X):
            return "PASS"
        return "UNCLEAR"
    v1, v1d = ogw("OBS"), ogw("DAV")
    print(f"VERDICT 12.1 OGW: {v1}  (DAV witness under the same rule: {v1d}; W2 label: "
          f"{'adds nothing for detection' if v1d == 'PASS' else 'observable sourcing adds'})")
    v3, v3d = cbr("OBS"), cbr("DAV")
    print(f"VERDICT 12.3 CBR: {v3}  (DAV block under the same rule: {v3d}; W2 label: "
          f"{'adds nothing' if v3d == 'PASS' else 'observable sourcing adds'})")
    fm = {s: frac(st[s], "MSR") for s in SIZES}
    fo = {s: frac(st[s], "OBS") for s in SIZES}
    fr = {s: frac(st[s], "RND") for s in SIZES}
    fn = {s: frac(nat[s], "MSR") for s in SIZES}
    if all(fm[s] < 0.5 for s in SIZES):
        v4 = "KILL (tracking)"
    elif all(fr[s] >= fm[s] - 0.1 for s in SIZES):
        v4 = "KILL (Gate 2)"
    elif all(fm[s] >= 0.5 and fm[s] >= fo[s] + 0.25 for s in SIZES) and all(fn[s] >= 0.75 for s in SIZES):
        v4 = "UNCLEAR (Gate 4: native drop > 0.15)" if fn["4x2"] - fn["4x3"] > 0.15 else "PASS"
    else:
        v4 = "UNCLEAR"
    print(f"VERDICT 12.4 MSR: {v4}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("main", "verdict"), required=True)
    p.add_argument("--size", choices=SIZES, default="4x2")
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--n-low", type=int, default=15)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=1)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    os.makedirs(RES, exist_ok=True)
    if a.mode == "verdict":
        return verdict()
    I, info = dav.build(a.size, a.n_low, a.delta)
    log(f"build cpu {cpu():.0f}s; MSR sources: " + ", ".join(
        f"{k} {len(I.ob.pieces[k][-4:])} (dists {[round(d, 2) for d in I.ob.piece_dists[k][-4:]]})" for k in REAL))
    tag = a.size if a.delta is None else f"{a.size}_d{a.delta}"
    out = os.path.join(RES, f"r12_{tag}.json")
    rows = run(I, a.states, a.cpu_limit, out, a.delta is not None)
    json.dump(dict(info=info, rows=rows), open(out, "w"), indent=1, default=str)
    log(f"total CPU {cpu():.0f}s")


if __name__ == "__main__":
    main()
