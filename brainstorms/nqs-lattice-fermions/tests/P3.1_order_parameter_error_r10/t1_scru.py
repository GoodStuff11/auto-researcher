#!/usr/bin/env python3
"""
P3.1 Round 10: tier-1 test of Route 10.1u (self-consistent response Ritz with the base updated each pass).
=========================================================================================================

Implements the T1 rule pre-registered by the Round 9 per-leaf critic (leaf, "Route 10.1u"), on a genuine size
trend of distinct V (the 6x2 / 4x3 pair of Rounds 8-10 shares V = 12 and D = 34650, and the 3x3 sector has only
60 levels, so neither gives a size trend). Exact linear algebra, no sampling, no training.

Instances (OBC t-t'-J, t = 1, t' = -0.25, J = 0.4, via ../P3.1_order_parameter_error_r8/t1_portfolio.py):
  5x2 (3+3, n = 0.60, V = 10), 4x3 (4+4, n = 0.67, V = 12), 7x2 (5+5, n = 0.71, V = 14)   trend sizes
  6x2 (4+4, V = 12)                                                                        reported only
  3x3 (3+3, V = 9; 60-level sector)                                                        reported only
  The density cannot be held at 2/3 for three distinct V below D ~ 3e5; it varies 0.60-0.71 (stated in the leaf).
  Validity asserts per instance: non-degenerate psi_0; in-sector gap omega_1 >= 0.02 t; all in-sector levels with
  omega <= 1 t found (the highest level found has omega > 1 t); at least 3 window levels; the sector holds more
  than 4x the largest Ritz span (checked when the sector dimension is known, else by the eigsh level count).

Error model (critic rule, items i-ii):
  psi = N[psi_0 + lam e + mu j] (t1_r10.trial_state construction; e = first-order ground state of H + V for the
  8 perturbations of t1_portfolio.perturbations; j = high-omega junk carrying 2x the energy error of lam e),
  scaled to fixed per-site infidelity |<psi_0|psi>|^2 = 0.92^(V/16).
  Response training error, grade R, fresh draw every pass: y~ = unit(unit(y) + 0.3 g + 0.5 j_y), g a random unit
  vector over the in-sector levels with omega <= 1 t (not the 60 lowest), j_y the junk of t1_portfolio (unchanged).
  The noise amplitude is relative to a unit-norm response at every size; a fixed-width network's response error
  is unlikely to be size-independent, so this model is optimistic (critic minor objection 1).

Response of a state phi: y_A = (Q (H - s) Q)^{-1} Q (A - a_phi) phi in the sector, Q = 1 - |phi><phi|.
  Deciding shift (critic item iii): s = E_L1R(phi) - 0.1 t.  Default shift s = 2 E_L1R - E_phi (reported only).

Arms (lowest Ritz vector of H; ratio = (a(phi_T) - a(psi)) / delta on the real-space tails pair / spin / charge):
  L1R      span{psi, (H-E)psi}                                          blind control
  ORC3     span{psi, (H-E)psi, |1>, |2>, |3>}                           comparison arm (critic rule)
  SCRu-R   10.1u: phi_0 = psi; phi_{t+1} = Ritz of span{phi_t, (H-E)phi_t, y~_A(phi_t), K y~_A(phi_t)},
           K v = (H - <H>_v) v; grade-R noise, fresh each pass; scored at T = 2 and T = 4 (one chain)
  SCRd-R   10.1d: pass 1 as SCRu-R; later passes span{phi_t, y~_A(phi_t)}; scored at T = 4
  SCR2-R   fixed base (10.1 as coded): two passes of span{psi, (H-E)psi, y~(phi_t), K y~(phi_t)} (comparison)
  SD4      Gate 2 foil added by the worker before the run: span{phi_t, (H-E)phi_t} for 4 passes (no responses;
           restarted steepest descent from psi alone, the information S1/S2 says is blind on the real states)
  SCRu-0   SCRu at T = 4 with exact responses (attribution: how much the grade-R noise costs)
  SCRu-Rdef  SCRu-R T = 4 at the default shift (reported only, critic item iii)
Scored item: |delta| >= 0.10 |a_0|; tracked if ratio in [1/2, 2].

Decoupled-copy check (--mode copytoy; critic "extra check"): the critic_sc_toy.py model (M copies of one random
d = 6 subsystem, the same per-copy error eps, so 1 - F grows with M), eps in {0.3, 0.5}, M = 1..4, SCRu-R T = 4
with grade-R noise on every response (window = levels with omega <= 2.9 omega_1(h), i.e. omega <= 2.95 in the
units of h: single excitations at 1.02 and 2.70 and double excitations at 2.04; the first run used omega <= 1,
which is EMPTY because omega_1(h) = 1.018, so it had no in-window noise and is void; output
copytoy_void_window_output.txt; junk = (H - E)x for a
random diagonal reweighting x of the base, with the window and psi_0 projected out, as t1_portfolio.junk),
shift E_L1R - 0.1. Deciding number per (eps, M): the median ratio over 5 noise draws (seeds 0-4); min and max
are printed.

Pre-registered rule: in the leaf, Round 10, "T1 pre-registration for 10.1u", written before --mode main ran.

Usage
  python3 t1_scru.py --mode census --size 7x2          # build one instance, print validity data, no arms
  python3 t1_scru.py --mode smoke  --size 5x2 --states 2   # first 2 trial states only, all arms
  python3 t1_scru.py --mode main   --size 4x3          # all 8 trial states, JSON to results/t1_scru_<size>.json
  python3 t1_scru.py --mode copytoy                    # decoupled-copy check, JSON to results/copytoy.json
  python3 t1_scru.py --mode verdict                    # read the JSONs, print fractions and the verdict
Options
  --mode {census, smoke, main, copytoy, verdict}
  --size {3x3, 5x2, 6x2, 4x3, 7x2}   instance for census / smoke / main
  --n-low INT     number of in-sector levels requested from eigsh (default 40; must cover omega <= 1 t)
  --states INT    number of trial states (default 8; smoke uses the first --states of the fixed list)
  --cpu-limit INT hard RLIMIT_CPU in CPU-seconds, set before any ED (default 600 = 10 CPU-min)
  --threads INT   BLAS threads, 1-4 (default 4; read before numpy is imported). 1 thread costs less CPU time
                  because the sparse matvecs are single-threaded anyway.
  --part {0, 1}   main only: run trial states 0-3 or 4-7 (one CPU-capped run each; output t1_scru_<size>_p<part>.json;
                  the verdict mode merges the parts)
  --no-extra      skip the reported-only arms SCRu-0 and SCRu-Rdef (cost control; the deciding arms stay)
Threads: OMP/OPENBLAS/MKL_NUM_THREADS = --threads (default 4), set before numpy is imported.
"""
import os
import sys
_nt = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "4"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _nt
import argparse  # noqa: E402
import functools  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R8 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r8")
_spec = importlib.util.spec_from_file_location("t1", os.path.join(R8, "t1_portfolio.py"))
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)
REAL, log, unit = t1.REAL, t1.log, t1.unit
t1.SIZES.update({"5x2": dict(Lx=5, Ly=2, nup=3, ndn=3), "7x2": dict(Lx=7, Ly=2, nup=5, ndn=5)})
TREND = ("5x2", "4x3", "7x2")
REPORTED = ("3x3", "6x2")
ETA = (0.3, 0.5)
WINDOW = 1.0
SHIFT_OFF = 0.1
COPY_WIN = 2.9   # copy-toy window in units of the single-copy gap (see docstring; the omega <= 1 window was empty)
ARMS = ("L1R", "ORC3", "SCRu-R2", "SCRu-R4", "SCRd-R4", "SCR2-R", "SD4", "SCRu-04", "SCRu-Rdef4")
RES = os.path.join(HERE, "results")


def cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


# ============================================================================ instance helpers
def build(size, n_low):
    I = t1.Inst(size, n_low)
    om = I.om
    nwin = int(np.sum(om <= WINDOW))
    I.Uw = I.U[:, :nwin]
    I.nwin = nwin
    V = I.m.V
    I.ov2 = 0.92 ** (V / 16)
    info = dict(size=size, V=V, D=I.m.D, nup=I.m.nup, n=2 * I.m.nup / V, omega1=float(om[0]),
                n_found=I.n_found, n_low=I.n_low, omega_top=float(om[-1]), nwin=nwin,
                dim_sector=I.dim_sector, ov2=I.ov2, E0=I.E0, chi=str(I.gen), S0=I.S0,
                n_pair=I.ob.n_pair, n_site=I.ob.n_site)
    log(f"census {size}: " + json.dumps(info))
    assert om[0] >= 0.02, f"gap omega_1 = {om[0]:.4f} < 0.02 t"
    assert om[-1] > WINDOW, f"window not covered: highest found omega {om[-1]:.3f} <= {WINDOW}; raise --n-low"
    assert nwin >= 3, f"only {nwin} window levels"
    span = 2 + 6  # SCRu span size
    if I.dim_sector is not None:
        assert I.dim_sector > 4 * span, f"sector dim {I.dim_sector} too small"
    return I, info


def trial_state(I, Vop, rng):
    """t1_r10.trial_state with target |<psi_0|psi>|^2 = I.ov2."""
    psi0, E0 = I.psi0, I.E0
    w = I.proj(Vop(psi0))
    w -= psi0 * (psi0 @ w)
    e = unit(-I.solve(w, E0, orth=psi0))
    Ee = e @ (I.m.H @ e) - E0
    j = I.junk(rng, psi0)
    Ej = j @ (I.m.H @ j) - E0
    lam2 = (1.0 / I.ov2 - 1.0) / (1.0 + t1.JUNK_RATIO * Ee / Ej)
    mu = np.sqrt(t1.JUNK_RATIO * lam2 * Ee / Ej)
    v = np.sqrt(lam2) * e + mu * j
    v *= np.sqrt(1.0 / I.ov2 - 1.0) / np.linalg.norm(v)
    psi = unit(psi0 + v)
    assert abs((psi @ psi0) ** 2 - I.ov2) < 1e-8
    return psi if psi @ psi0 > 0 else -psi


def krylov(I, v):
    v = unit(v)
    w = I.m.H @ v
    return I.proj(w - (v @ w) * v)


def resid(I, phi):
    return I.m.H @ phi - float(phi @ (I.m.H @ phi)) * phi


def ritz_vec(I, vecs):
    B = np.array([unit(v) for v in vecs])
    HB = np.array([I.m.H @ b for b in B])
    S = 0.5 * (B @ B.T + (B @ B.T).T); H = B @ HB.T; H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(S)
    keep = lam > 1e-11 * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ H @ X)
    return unit((X @ Y[:, 0]) @ B), float(ev[0])


def responses(I, phi, rng, shift="decide"):
    phi = unit(phi)
    E = float(phi @ (I.m.H @ phi))
    _, EL1 = ritz_vec(I, [phi, resid(I, phi)])
    s = EL1 - SHIFT_OFF if shift == "decide" else 2.0 * EL1 - E
    ys = []
    for k in REAL:
        a = float(phi @ I.apply(k, phi))
        y = unit(I.solve(I.apply(k, phi) - a * phi, s, orth=phi))
        if rng is not None:
            g = unit(I.Uw @ rng.standard_normal(I.nwin))
            y = unit(y + ETA[0] * g + ETA[1] * I.junk(rng, phi))
        ys.append(y)
    return ys, s


def scru_pass(I, phi, ys):
    return ritz_vec(I, [phi, resid(I, phi)] + ys + [krylov(I, v) for v in ys])[0]


# ============================================================================ main arms
def run_instance(I, states, extra, cap, dump=None):
    """states: indices into the fixed list of 8 perturbations. Trial states of skipped indices below max(states)
    are still built, so the shared stream rng(7) is consumed exactly as in a full run. Soft CPU guard: a state is
    started only if cpu() + 1.2 x (largest per-state cost so far) < cap - 15; rows are written after every state."""
    perts = t1.perturbations(I)
    names = list(perts.keys())
    rng = np.random.default_rng(7)
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
    rows = []
    worst = 0.0
    for si, lab in enumerate(names[:max(states) + 1]):
        c0 = cpu()
        psi = trial_state(I, perts[lab], rng)
        if si not in states:
            continue
        if cpu() + 1.2 * worst > cap - 15:
            log(f"soft CPU guard: stop before state {si} ({lab}); cpu {cpu():.0f}s, worst state {worst:.0f}s")
            break
        ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
        dl = {k: a0[k] - ap[k] for k in REAL}
        row = dict(size=I.name, state=lab, a0=a0, delta=dl,
                   scored={k: bool(abs(dl[k]) >= t1.SCORE_REL * abs(a0[k])) for k in REAL})

        def score(arm, phi):
            row[arm] = {k: float(phi @ I.apply(k, phi) - ap[k]) / dl[k] for k in REAL}
        r = resid(I, psi)
        score("L1R", ritz_vec(I, [psi, r])[0])
        score("ORC3", ritz_vec(I, [psi, r] + [I.U[:, n] for n in range(3)])[0])
        rs = np.random.default_rng(10_000 + 97 * I.m.V + si)
        shifts = []
        # SCRu-R chain (T = 1..4); phi_1 is shared with SCRd-R and SCR2-R (identical pass-1 span)
        phi = psi
        for t in range(1, 5):
            ys, s = responses(I, phi, rs); shifts.append((s - I.E0) / I.om[0])
            phi = scru_pass(I, phi, ys)
            if t == 1:
                phi1 = phi
            if t in (2, 4):
                score(f"SCRu-R{t}", phi)
        # SCRd-R: later passes span{phi_t, y~(phi_t)}
        ph = phi1
        for t in range(2, 5):
            ys, _ = responses(I, ph, rs)
            ph = ritz_vec(I, [ph] + ys)[0]
        score("SCRd-R4", ph)
        # SCR2-R: fixed base {psi, r}
        ys, _ = responses(I, phi1, rs)
        score("SCR2-R", ritz_vec(I, [psi, r] + ys + [krylov(I, v) for v in ys])[0])
        # SD4 foil
        ph = psi
        for _ in range(4):
            ph = ritz_vec(I, [ph, resid(I, ph)])[0]
        score("SD4", ph)
        if extra:
            ph = psi
            for _ in range(4):
                ys, _ = responses(I, ph, None)
                ph = scru_pass(I, ph, ys)
            score("SCRu-04", ph)
            ph = psi
            for _ in range(4):
                ys, sdef = responses(I, ph, rs, shift="default")
                ph = scru_pass(I, ph, ys)
            score("SCRu-Rdef4", ph)
            row["default_shift_last"] = (sdef - I.E0) / I.om[0]
        row["shift_over_omega1"] = shifts
        rows.append(row)
        worst = max(worst, cpu() - c0)
        if dump:
            json.dump(dict(rows=rows), open(dump, "w"), indent=1, default=str)
        log(f"{I.name} {lab:7s} (s-E0)/w1 {shifts[0]:+.2f} cpu {cpu() - c0:.0f}s | " + " | ".join(
            f"{k}{'' if row['scored'][k] else '(ns)'}: " + " ".join(
                f"{a} {row[a][k]:+.2f}" for a in ARMS if a in row) for k in REAL))
    return rows


def frac(rows, arm, sizes):
    it = [(r, k) for r in rows for k in REAL if r["scored"][k] and r["size"] in sizes and arm in r]
    if not it:
        return float("nan"), 0
    return sum(1 for r, k in it if 0.5 <= r[arm][k] <= 2.0) / len(it), len(it)


# ============================================================================ decoupled-copy check
def copytoy():
    rng0 = np.random.default_rng(3)
    d = 6
    h = rng0.standard_normal((d, d)); h = (h + h.T) / 2
    a = rng0.standard_normal((d, d)); a = (a + a.T) / 2
    w, U = np.linalg.eigh(h)
    p0 = U[:, 0]
    e = U[:, 1:] @ rng0.standard_normal(d - 1); e /= np.linalg.norm(e)

    def kron_all(vs):
        return functools.reduce(np.kron, vs)

    def op_sum(o, M):
        eye = np.eye(d)
        return sum(kron_all([o if j == i else eye for j in range(M)]) for i in range(M))
    out = {}
    for eps in (0.3, 0.5):
        phi1 = unit(p0 + eps * e)
        for M in range(1, 5):
            H = op_sum(h, M); A = op_sum(a, M) / M
            ev, EV = np.linalg.eigh(H)
            E0 = ev[0]; psi0 = EV[:, 0]
            wcut = COPY_WIN * (w[1] - w[0])
            win = EV[:, 1:][:, (ev[1:] - E0) <= wcut]
            high = EV[:, 1:][:, (ev[1:] - E0) > wcut]
            D = d ** M
            psi = kron_all([phi1] * M); psi = psi if psi @ psi0 > 0 else -psi
            ap = psi @ A @ psi; dl = psi0 @ A @ psi0 - ap

            def ritz(vecs):
                B = np.array([unit(v) for v in vecs]); S = B @ B.T; Hm = B @ H @ B.T
                lam, V = np.linalg.eigh(S); k = lam > 1e-11 * lam.max(); X = V[:, k] / np.sqrt(lam[k])
                _, Y = np.linalg.eigh(X.T @ Hm @ X)
                return unit((X @ Y[:, 0]) @ B)

            def resp(ref, rng):
                E = ref @ H @ ref; r = H @ ref - E * ref
                s = ritz([ref, r]) @ H @ ritz([ref, r]) - SHIFT_OFF
                Q = np.eye(D) - np.outer(ref, ref); av = ref @ A @ ref
                y = unit(Q @ np.linalg.solve(Q @ (H - s * np.eye(D)) @ Q + np.outer(ref, ref),
                                             Q @ (A @ ref - av * ref)))
                g = unit(win @ rng.standard_normal(win.shape[1])) if win.shape[1] else np.zeros(D)
                x = ref * rng.standard_normal(D)
                j = H @ x - E * x
                j -= win @ (win.T @ j); j -= psi0 * (psi0 @ j)
                return unit(y + ETA[0] * g + ETA[1] * unit(j))

            def K(v):
                v = unit(v); x = H @ v
                return x - (v @ x) * v
            ratios = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                ph = psi
                for _ in range(4):
                    ys = resp(ph, rng)
                    ph = ritz([ph, H @ ph - (ph @ H @ ph) * ph, ys, K(ys)])
                ratios.append(float((ph @ A @ ph - ap) / dl))
            med = float(np.median(ratios))
            out[f"eps{eps}_M{M}"] = dict(eps=eps, M=M, D=D, infid=float(1 - (psi @ psi0) ** 2), nwin=win.shape[1],
                                         nhigh=high.shape[1], ratios=ratios, median=med)
            log(f"copytoy eps={eps} M={M} D={D:5d} 1-F={1 - (psi @ psi0) ** 2:.3f} nwin={win.shape[1]:4d} "
                f"SCRu-R4 median {med:+.3f} (min {min(ratios):+.3f}, max {max(ratios):+.3f})")
    return out


# ============================================================================ verdict
def verdict():
    rows = []
    for s in TREND + REPORTED:
        for tag in (s, f"{s}_p0", f"{s}_p1"):
            f = os.path.join(RES, f"t1_scru_{tag}.json")
            if os.path.exists(f):
                rows += json.load(open(f))["rows"]
    have = sorted({r["size"] for r in rows})
    print("sizes present:", have)
    for arm in ARMS:
        print(f"  tracked {arm:10s} " + "  ".join(
            f"{s} {frac(rows, arm, [s])[0]:.2f} (n={frac(rows, arm, [s])[1]})" for s in TREND + REPORTED))
    ct = json.load(open(os.path.join(RES, "copytoy.json")))
    copy_ok = all(0.5 <= v["median"] <= 2.0 for v in ct.values())
    print(f"  copytoy medians: " + " ".join(f"{k} {v['median']:+.2f}" for k, v in ct.items()) + f" -> in range: {copy_ok}")
    if not all(s in have for s in TREND):
        print("VERDICT: INCOMPLETE (missing trend sizes)")
        return
    f = {s: frac(rows, "SCRu-R4", [s])[0] for s in TREND}
    o3 = {s: frac(rows, "ORC3", [s])[0] for s in TREND}
    sd = {s: frac(rows, "SD4", [s])[0] for s in TREND}
    big = ("4x3", "7x2")
    if any(f[s] < 0.5 for s in big) or not copy_ok:
        v = "KILL"
    elif all(f[s] >= 0.75 for s in TREND) and all(f[s] >= o3[s] + 0.1 for s in big):
        v = "PASS"
        if all(sd[s] >= 0.75 for s in TREND) and all(sd[s] >= o3[s] + 0.1 for s in big):
            v = "UNCLEAR (Gate 2 foil SD4 also passes)"
        elif f["5x2"] - f["7x2"] > 0.15:
            v = "UNCLEAR (Gate 4: drop > 0.15 from V=10 to V=14)"
    else:
        v = "UNCLEAR"
    print(f"ROUTE 10.1u SCRu-R4: " + " ".join(f"{s} {f[s]:.2f}" for s in TREND)
          + " | ORC3 " + " ".join(f"{s} {o3[s]:.2f}" for s in TREND)
          + " | SD4 " + " ".join(f"{s} {sd[s]:.2f}" for s in TREND) + f" -> {v}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("census", "smoke", "main", "copytoy", "verdict"), required=True)
    p.add_argument("--size", choices=TREND + REPORTED, default="5x2")
    p.add_argument("--n-low", type=int, default=40)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--no-extra", action="store_true")
    p.add_argument("--part", type=int, choices=(0, 1), default=None)
    p.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    return p.parse_args()


def main():
    a = parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    log(f"RLIMIT_CPU {a.cpu_limit} s set before any ED; mode {a.mode}")
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    if a.mode == "verdict":
        verdict()
    elif a.mode == "copytoy":
        json.dump(copytoy(), open(os.path.join(RES, "copytoy.json"), "w"), indent=1)
    else:
        I, info = build(a.size, a.n_low)
        log(f"build cpu {cpu():.0f}s")
        if a.mode != "census":
            st = list(range(a.states)) if a.mode == "smoke" else \
                (list(range(8)) if a.part is None else list(range(4 * a.part, 4 * a.part + 4)))
            tag = a.size if a.part is None else f"{a.size}_p{a.part}"
            out = os.path.join(RES, f"t1_scru_{tag}.json") if a.mode == "main" else None
            rows = run_instance(I, st, not a.no_extra, a.cpu_limit, out)
            for arm in ARMS:
                fr, n = frac(rows, arm, [a.size])
                print(f"  tracked {arm:10s} {a.size} {fr:.2f} (n={n})")
            if out:
                json.dump(dict(info=info, rows=rows), open(out, "w"), indent=1, default=str)
    log(f"total wall {time.time() - t0:.0f}s, CPU {cpu():.0f}s")


if __name__ == "__main__":
    main()
