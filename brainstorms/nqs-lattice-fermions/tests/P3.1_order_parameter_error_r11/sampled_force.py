#!/usr/bin/env python3
"""
P3.1 Round 11, task B: does the SAMPLED Hylleraas training force change the tracked fraction of 10.1d?
=====================================================================================================

The T3 (t3_scrd.py) trains each response network u = Q P y_theta on R(u) = <u|H-s|u> / <u|b>^2 with the
EXACT-SUM force f = sum_n g(n) dy(n)/dtheta, g = Q[(H-s)u - R <u|b> b] / <u|b>^2, and only the SR metric
sampled. Here the same chain (10.1d, T = 4; pass 1 span{phi, (H-E)phi, u_A, K u_A}, later span{phi, u_A};
exact-sum Ritz and readout as in the T3) is run twice per trial state, identical except for the force:

  exact    f from the exact sums (the T3 force); best iterate kept by the exact R (as t3_scrd.train)
  sampled  every global scalar estimated from samples, VMC-style:
             x_i ~ |y_P|^2 / Z  (Ns draws, the SAME draws feed the SR metric), z_j ~ phi^2 (Ns draws)
             local values y_P(x), (H y_P)(x), phi(x), (H phi)(x), b(x) exact (VMC local estimators)
             m_phi = mean phi/y, m_b = mean b/y, m_E = mean (Hy)/y - s, m_Hphi = mean ((H-s)phi)/y
             Z^ = mean_z y/phi / m_phi  (two-sample ratio estimator of the norm of y)
             c = Z^ m_phi (= <phi|y>), ub = Z^ m_b (= <u|b>, b orthogonal to phi), Phy = Z^ m_Hphi,
             N = Z^ m_E - 2 c Phy + c^2 (E_phi - s), R = N / ub^2, q = Phy - c (E_phi - s)
             g(x) = [(Hy - s y)(x) - c ((H-s)phi)(x) - R ub b(x) - q phi(x)] / ub^2
             f = Z^ mean_x g(x) O_P(x) / y_P(x)
           E_phi is exact (a property of the base state, measured once per pass with many samples in VMC).
           Best iterate kept by the SAMPLED R (exact R is not available in VMC).
Both arms use the same SR solve (primal form (Oc^T Oc + lambda)^-1 f, lambda = 1e-3, step 0.05, trust region
1; mathematically identical to t3_scrd.sr_solve), the same initial networks and the same rng seeds.

Instances: 3x3 3+3 (V = 9, D = 1680), 4x2 3+3 (V = 8, D = 560). Trial states: t1_scru.trial_state
(8 perturbations, fixed per-site infidelity 0.92^(V/16), rng(7)). Response networks: perturbed copies (0.05 rms)
of an NNBF of width --width trained --gs-iters SR steps on the ground state (the T3 recipe, scaled down).
Scored item |delta| >= 0.10 |a_0|; tracked = ratio in [1/2, 2].

Pre-registered rule: leaf, "Round 11 thinking work (2026-11-16)", Task B, written before any --mode main run.

Usage
  python3 sampled_force.py --mode timing --size 3x3          # CPU per SR step, both forces
  python3 sampled_force.py --mode main --size 3x3            # JSON results/sf_<size>.json
  python3 sampled_force.py --mode verdict
Options
  --mode {timing, main, verdict}
  --size {3x3, 4x2}
  --states INT      trial states to attempt (default 8; a soft CPU guard stops earlier, rows are kept)
  --width INT       NNBF width (default 2)
  --gs-iters INT    ground-state SR steps of the seed network (default 60)
  --cold INT        pass-1 SR steps per response network (default 60)
  --warm INT        pass 2..4 SR steps per response network (default 20)
  --ns INT          samples per step for metric and force (default 2000)
  --cpu-limit INT   hard RLIMIT_CPU seconds (default 600)
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
R10 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r10")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t3 = _load("t3", os.path.join(R10, "t3_scrd.py"))
t1s = _load("t1s", os.path.join(R10, "t1_scru.py"))
np, rc, unit, log, REAL = t3.np, t3.rc, t3.unit, t3.log, t3.REAL
t3.t1.SIZES.update({"4x2": dict(Lx=4, Ly=2, nup=3, ndn=3)})
SIZES = ("3x3", "4x2")
RES = os.path.join(HERE, "results")


def cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def sr_step(ans, sg, yP, f, idx, a):
    O = ans.per_sample_O_proj(sg, idx, yP, a.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    S = Oc.T @ Oc + a.sr_shift * np.eye(O.shape[1])
    dx = np.linalg.solve(S, f)
    nrm = np.linalg.norm(dx)
    step = a.sr_lr * dx
    if nrm * a.sr_lr > 1.0:
        step *= 1.0 / (nrm * a.sr_lr)
    return step, O


def sampled_force(C, ans, yP, phi, b, s, Hphi, Ephi, idx, jdx, O):
    y = yP
    Hy = C.m.H @ y
    yx = y[idx]
    m_phi = np.mean(phi[idx] / yx)
    m_b = np.mean(b[idx] / yx)
    m_E = np.mean(Hy[idx] / yx) - s
    m_Hphi = np.mean((Hphi[idx] - s * phi[idx]) / yx)
    Z = np.mean(y[jdx] / phi[jdx]) / m_phi
    c, ub, Phy = Z * m_phi, Z * m_b, Z * m_Hphi
    N = Z * m_E - 2 * c * Phy + c * c * (Ephi - s)
    R = N / ub ** 2
    q = Phy - c * (Ephi - s)
    g = ((Hy[idx] - s * yx) - c * (Hphi[idx] - s * phi[idx]) - R * ub * b[idx] - q * phi[idx]) / ub ** 2
    f = Z * np.mean((g / yx)[:, None] * O, axis=0)
    return float(R), f


def train(C, ans, phi, b, s, iters, kind, a, seed):
    rng = np.random.default_rng(seed)
    Hphi = C.m.H @ phi
    Ephi = float(phi @ Hphi)
    best, best_x = np.inf, ans.flat().copy()
    for it in range(iters):
        x0 = ans.flat().copy()
        yP = C.sg.project(ans.amplitudes(a.chunk))
        pc = yP ** 2
        idx = rng.choice(len(pc), size=a.ns, p=pc / pc.sum())
        jdx = rng.choice(len(phi), size=a.ns, p=phi ** 2 / (phi @ phi))
        O = ans.per_sample_O_proj(C.sg, idx, yP, a.o_chunk)
        if kind == "exact":
            R, _, _, g = t3.resp_eval(ans, C, phi, b, s, a)
            f = ans.force(g, a.chunk)
        else:
            R, f = sampled_force(C, ans, yP, phi, b, s, Hphi, Ephi, idx, jdx, O)
        if not np.isfinite(R) or not np.all(np.isfinite(f)):
            raise AssertionError(f"VALIDITY: non-finite objective/force ({kind}, step {it})")
        if R < best:
            best, best_x = R, x0
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        dx = np.linalg.solve(Oc.T @ Oc + a.sr_shift * np.eye(O.shape[1]), f)
        step = a.sr_lr * dx
        nrm = np.linalg.norm(dx)
        if nrm * a.sr_lr > 1.0:
            step *= 1.0 / (nrm * a.sr_lr)
        ans.set_flat(x0 - step)
    if kind == "exact":   # final iterate competes on the exact R (as t3_scrd.train)
        Rf = t3.resp_eval(ans, C, phi, b, s, a)[0]
        if not Rf <= best:
            ans.set_flat(best_x)
    else:                 # sampled arm: final iterate is not re-scored exactly; the best sampled-R iterate is kept
        ans.set_flat(best_x)


def chain(C, psi, ans_gs, kind, a, si):
    phi, nets = psi, []
    diag = []
    for t in range(1, 5):
        s = C.shift(phi)
        us = []
        for j, k in enumerate(REAL):
            b = C.source(k, phi)
            if t == 1:
                nets.append(t3.perturbed_copy(C, ans_gs, a.width, 1000 * si + 100 * (j + 1), 0.05))
            train(C, nets[j], phi, b, s, a.cold if t == 1 else a.warm, kind, a, 7000 * si + 100 * t + j)
            R, _, u, _ = t3.resp_eval(nets[j], C, phi, b, s, a)
            y = C.solve(b, s, phi)
            diag.append(dict(pass_=t, obs=k, R_over_Rstar=R / C.R_of(y, b, s),
                             eta=1.0 - float(unit(u) @ unit(y)) ** 2))
            us.append(u)
        if t == 1:
            phi = C.ritz_vec([phi, C.m.H @ phi - C.E(phi) * phi] + us + [C.Kv(v) for v in us])[0]
        else:
            phi = C.ritz_vec([phi] + us)[0]
    return phi, diag


def exact_chain(C, psi):
    phi = psi
    for t in range(1, 5):
        s = C.shift(phi)
        ys = [C.solve(C.source(k, phi), s, phi) for k in REAL]
        if t == 1:
            phi = C.ritz_vec([phi, C.m.H @ phi - C.E(phi) * phi] + ys + [C.Kv(y) for y in ys])[0]
        else:
            phi = C.ritz_vec([phi] + ys)[0]
    return phi


def setup(a):
    C = t3.Ctx(a.size, 15)
    C.I.ov2 = 0.92 ** (C.m.V / 16)
    ans_gs = rc.nnbf(C.m, a.width, 1)
    a.gs_width = a.width
    t3.train(ans_gs, lambda x: t3.gs_eval(x, C, a), C, a, np.random.default_rng(1), a.gs_iters, "seed net")
    return C, ans_gs


def main_run(a):
    C, ans_gs = setup(a)
    perts = t1s.t1.perturbations(C.I)
    rng = np.random.default_rng(7)
    rows, worst = [], 0.0
    out = os.path.join(RES, f"sf_{a.size}.json")
    for si, lab in enumerate(list(perts)[:a.states]):
        psi = t1s.trial_state(C.I, perts[lab], rng)
        psi = unit(C.sg.project(psi))
        psi = psi if psi @ C.psi0 > 0 else -psi
        if cpu() + 1.15 * worst > a.cpu_limit - 20:
            log(f"soft CPU guard: stop before state {si}")
            break
        c0 = cpu()
        ap = {k: C.a(k, psi) for k in REAL}
        dl = {k: C.a0[k] - ap[k] for k in REAL}
        row = dict(state=lab, delta=dl, scored={k: bool(abs(dl[k]) >= 0.10 * abs(C.a0[k])) for k in REAL})
        phis = {"L1R": C.ritz_vec([psi, C.m.H @ psi - C.E(psi) * psi])[0], "SCRd0": exact_chain(C, psi)}
        for kind in ("exact", "sampled"):
            phis[f"NN-{kind}"], row[f"diag_{kind}"] = chain(C, psi, ans_gs, kind, a, si)
        for arm, p in phis.items():
            row[arm] = {k: (C.a(k, p) - ap[k]) / dl[k] for k in REAL}
        rows.append(row)
        worst = max(worst, cpu() - c0)
        json.dump(dict(size=a.size, rows=rows), open(out, "w"), indent=1)
        med = {kind: (float(np.median([d["R_over_Rstar"] for d in row[f"diag_{kind}"]])),
                      float(np.median([d["eta"] for d in row[f"diag_{kind}"]]))) for kind in ("exact", "sampled")}
        log(f"{a.size} {lab:7s} cpu {cpu() - c0:.0f}s | " + " | ".join(
            f"{k}{'' if row['scored'][k] else '(ns)'} " + " ".join(f"{arm} {row[arm][k]:+.2f}" for arm in phis)
            for k in REAL) + " | median R/R*, eta: " + " ".join(
            f"{kd} {m[0]:.3f} {m[1]:.3f}" for kd, m in med.items()))
    report(rows, a.size)


def frac(rows, arm):
    it = [(r, k) for r in rows for k in REAL if r["scored"][k]]
    return (sum(1 for r, k in it if 0.5 <= r[arm][k] <= 2.0) / len(it) if it else float("nan")), len(it)


def report(rows, size):
    for arm in ("L1R", "SCRd0", "NN-exact", "NN-sampled"):
        f, n = frac(rows, arm)
        print(f"  tracked {arm:10s} {size} {f:.2f} (n={n}, states={len(rows)})")
    for kind in ("exact", "sampled"):
        d = [x for r in rows for x in r[f"diag_{kind}"]]
        print(f"  {kind:8s} median R/R* {np.median([x['R_over_Rstar'] for x in d]):.3f} "
              f"median eta {np.median([x['eta'] for x in d]):.3f} (pass 4 only: eta "
              f"{np.median([x['eta'] for x in d if x['pass_'] == 4]):.3f})")


def verdict():
    res = {}
    for s in SIZES:
        f = os.path.join(RES, f"sf_{s}.json")
        if not os.path.exists(f):
            print(f"{s}: missing")
            continue
        rows = json.load(open(f))["rows"]
        report(rows, s)
        fe, n = frac(rows, "NN-exact")
        fs, _ = frac(rows, "NN-sampled")
        res[s] = (fe, fs, n)
    counted = {s: v for s, v in res.items() if v[0] >= 0.5 and v[2] >= 6}
    for s, v in res.items():
        if s not in counted:
            print(f"  {s}: UNINFORMATIVE (f_exact {v[0]:.2f}, n {v[2]}); not counted")
    if not counted:
        print("VERDICT B: NO VERDICT (no counted size)")
        return
    ok = all(abs(fs - fe) <= 0.1 for fe, fs, _ in counted.values())
    print("VERDICT B: " + ("GAP-CLOSED" if ok else "GAP-BINDS") + " | " + " ".join(
        f"{s}: exact {fe:.2f} sampled {fs:.2f} (n={n})" for s, (fe, fs, n) in counted.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("timing", "main", "verdict"), required=True)
    p.add_argument("--size", choices=SIZES, default="3x3")
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--gs-iters", type=int, default=60)
    p.add_argument("--cold", type=int, default=60)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--ns", type=int, default=2000)
    p.add_argument("--cpu-limit", type=int, default=600)
    p.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=1)
    a = p.parse_args()
    a.sr_lr, a.sr_shift, a.sr_samples, a.o_chunk, a.chunk = 0.05, 1e-3, a.ns, 8, 32768
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    os.makedirs(RES, exist_ok=True)
    if a.mode == "verdict":
        return verdict()
    if a.mode == "timing":
        C, ans_gs = setup(a)
        psi = unit(C.psi0 + 0.2 * C.U[:, 0])
        s = C.shift(psi)
        b = C.source("pair", psi)
        for kind in ("exact", "sampled"):
            ans = t3.perturbed_copy(C, ans_gs, a.width, 5, 0.05)
            c0 = cpu()
            train(C, ans, psi, b, s, 10, kind, a, 3)
            R = t3.resp_eval(ans, C, psi, b, s, a)[0]
            log(f"timing {kind}: {(cpu() - c0) / 10:.3f} CPU-s/step, R/R* after 10 steps "
                f"{R / C.R_of(C.solve(b, s, psi), b, s):.3f}")
        return
    main_run(a)
    log(f"total CPU {cpu():.0f}s")


if __name__ == "__main__":
    main()
