#!/usr/bin/env python3
"""
P3.1 round 5, step 1: does the shared 4x4 OBC t-t'-J instance give P3.1 a target?
==================================================================================

Measures, for every converged projected NNBF state available from the P2.1 round-3 jobs
(parameter files tests/P2.1_subcubic_amplitudes_r3/results/params/inst0_nnbf_s{seed}.npz,
width 6, P_chi-projected SR, 1200 steps, best-energy iterate), the error of the long-range
d-wave pair, spin and charge correlations (r5_core.Observables) relative to ED:
    rel = |a_psi - a_0| / |a_0|,
for the trained state "P" = P_chi psi_theta and its post-hoc spin projection "P_S".
Also reports the in-irrep low-lying spectrum (Lanczos on P_S P_chi H P_chi P_S), the overlap
with psi_0, the energy error, and the one- and two-step Lanczos values of every observable.

Pre-committed rule (round-5 brief, step 1): the instance has a P3.1 target iff, on the
"P" states, the MEDIAN over available seeds of rel is >= 0.02 for at least one observable
AND the projected control's median dE_sign >= 0.005 t (read from the P2.1 result JSONs).
Otherwise NO-TARGET: record and stop route generation for this instance.

Usage
-----
  python3 target_probe.py --checks                 correctness checks on a 3x2 cluster (seconds)
  python3 target_probe.py [--seeds 11,22,33] [--params-dir DIR] [--out FILE] [--spectrum 1]
Options
  --checks          Run only the correctness checks: S_tot^2 rebuilt from the S_i.S_j operators
                    (3x2 PBC-x, 2+2 electrons); A Hermitian, [A, U_g] = 0 for every g of the point
                    group, [A, S^2] = 0, <A_pair> vs a direct bond-pair sum (4x4 OBC, 2+2 electrons,
                    production thresholds).
  --seeds LIST      Seeds whose parameter files are read; missing files are skipped (default 11,22,33).
  --params-dir DIR  Directory of P2.1 parameter files (default ../P2.1_subcubic_amplitudes_r3/results/params).
  --results-dir DIR Directory of P2.1 result JSONs (for dE_sign; default ../P2.1_subcubic_amplitudes_r3/results).
  --spectrum {0,1}  Compute the lowest 5 in-irrep (P_S P_chi) levels (default 1, a few minutes).
  --out FILE        JSON output (default results/target_probe.json next to this script).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r5_core as rc  # noqa: E402
from tj_core import TJ, SpaceGroup  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def checks():
    m = TJ(3, 2, "pbc", "obc", 2, 2, 1.0, -0.25, 0.4)
    m.spin_setup()
    W, Vv, _ = rc.solve_ed(m, 3, 1e-8, 1)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(m.D)
    ob = rc.Observables.__new__(rc.Observables)
    ob.m = m
    allp = [(i, j) for i in range(m.V) for j in range(i + 1, m.V)]
    import scipy.sparse as sp
    nu = np.array([(m.bu >> i) & 1 for i in range(m.V)], float)
    nd = np.array([(m.bd >> i) & 1 for i in range(m.V)], float)
    sz = 0.5 * (nu - nd)
    SS = sp.diags(sum(sz[i] * sz[j] for i, j in allp)) + ob._exchange(allp)
    Si2 = sp.diags(0.75 * ((nu + nd) == 1).sum(0))
    S2op = 2 * SS + Si2
    err = np.linalg.norm(S2op @ v - m.S2(v)) / np.linalg.norm(m.S2(v))
    log(f"check S_tot^2 = sum_i S_i^2 + 2 sum_(i<j) S_i.S_j (exchange signs): rel err {err:.1e}")
    assert err < 1e-12
    # full observable object, production thresholds, on the 4x4 OBC geometry with 2+2 electrons
    m = TJ(4, 4, "obc", "obc", 2, 2, 1.0, -0.25, 0.4)
    m.spin_setup()
    W, Vv, _ = rc.solve_ed(m, 3, 1e-8, 1)
    v = rng.standard_normal(m.D)
    ob = rc.Observables(m)
    sg, _ = rc.build_symmetry(m, Vv[:, 0])
    for k in rc.OBS:
        Av = ob.apply(k, v)
        u = rng.standard_normal(m.D)
        herm = abs(u @ Av - v @ ob.apply(k, u)) / (np.linalg.norm(u) * np.linalg.norm(Av))
        comm = max(np.linalg.norm(ob.apply(k, sg.apply(v, g)) - sg.apply(Av, g)) for g in range(sg.G)) \
            / np.linalg.norm(Av)
        s2c = np.linalg.norm(ob.apply(k, m.S2(v)) - m.S2(Av)) / np.linalg.norm(Av)
        log(f"check A_{k}: n_pairs {ob.n_pair if k == 'pair' else ob.n_site} hermitian {herm:.1e} "
            f"[A,U_g] {comm:.1e} [A,S^2] {s2c:.1e}")
        assert herm < 1e-12 and comm < 1e-12 and s2c < 1e-12
    # pair expectation equals the bond-pair average of tj_core.pairing-style overlaps
    p = Vv[:, 0] / np.linalg.norm(Vv[:, 0])
    vs = [Dop @ p for Dop in ob.D_ops]
    direct = sum(ob.M[a, b] * (vs[a] @ vs[b]) for a in range(len(vs)) for b in range(len(vs)))
    log(f"check <A_pair> operator vs direct bond-pair sum: {ob.expect('pair', p):.12e} vs {direct:.12e}")
    assert abs(ob.expect("pair", p) - direct) < 1e-13
    log("TARGET-PROBE CHECKS PASSED")


def lanczos_states(H, psi, steps):
    """Ground Ritz vectors in span{psi, H psi, ..., H^p psi} for p = 1..steps (exact)."""
    basis = [psi / np.linalg.norm(psi)]
    out = []
    for p in range(1, steps + 1):
        w = H @ basis[-1]
        for b in basis:
            w -= (b @ w) * b
        for b in basis:
            w -= (b @ w) * b
        basis.append(w / np.linalg.norm(w))
        B = np.column_stack(basis)
        Hs = B.T @ (H @ B)
        e, U = np.linalg.eigh(Hs)
        out.append(B @ U[:, 0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--params-dir", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results"))
    ap.add_argument("--spectrum", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "target_probe.json"))
    a = ap.parse_args(argv)
    if a.checks:
        checks(); return
    checks()
    m, W, psi0, sg, S0, ob = rc.setup(4, log)
    a0 = ob.all(psi0)
    a0["charge_conn"] = ob.connected_charge(psi0)
    log("a0: " + " ".join(f"{k}={v:+.6e}" for k, v in a0.items()))
    res = dict(ed_levels=W.tolist(), a0=a0, n_pair=ob.n_pair, n_site=ob.n_site, R_PAIR=rc.R_PAIR,
               R_SITE=rc.R_SITE, states={})
    if a.spectrum:
        t0 = time.time()
        def PH(v):
            x = m.spin_project(sg.project(v), S0)
            y = m.H @ x
            return m.spin_project(sg.project(y), S0)
        Lop = sla.LinearOperator((m.D, m.D), matvec=PH, dtype=float)
        v0 = m.spin_project(sg.project(np.random.default_rng(3).standard_normal(m.D)), S0)
        # shift so that the out-of-sector null space (eigenvalue 0) lies above the sector levels
        e, U = sla.eigsh(Lop, k=6, which="SA", v0=v0, tol=1e-10)
        o = np.argsort(e); e, U = e[o], U[:, o]
        ins = [float(x) for x in e if x < -1.0]
        A01 = {k: [float(abs(U[:, n] @ ob.apply(k, psi0)) / np.linalg.norm(U[:, n])) for n in range(len(e))]
               for k in rc.OBS}
        log(f"in-irrep (chi, S={S0:g}) levels {ins} gaps {[x - ins[0] for x in ins]} ({time.time() - t0:.0f} s)")
        log("|<n|A|0>| per level: " + "; ".join(f"{k} " + " ".join(f"{x:.2e}" for x in v) for k, v in A01.items()))
        res.update(irrep_levels=ins, A_n0=A01)
    for seed in [int(s) for s in a.seeds.split(",")]:
        f = os.path.join(a.params_dir, f"inst0_nnbf_s{seed}.npz")
        if not os.path.exists(f):
            log(f"seed {seed}: no parameter file ({f}); skipped")
            continue
        jf = os.path.join(a.results_dir, f"inst0_nnbf_s{seed}.json")
        dsign = None
        if os.path.exists(jf):
            dsign = json.load(open(jf))["budgets"]["P"]["dE_sign"]
        ans = rc.load_params(rc.nnbf(m, 6, seed), f)
        psi_t = ans.amplitudes(32768)
        psiP = sg.project(psi_t); psiP /= np.linalg.norm(psiP)
        psiPS = m.spin_project(psiP, S0); psiPS /= np.linalg.norm(psiPS)
        for lab, v in (("P", psiP), ("P_S", psiPS)):
            E = float(v @ (m.H @ v))
            row = dict(seed=seed, label=lab, dE=E - W[0], ov0=float((psi0 @ v) ** 2), dE_sign_P=dsign,
                       a=ob.all(v), a_conn_charge=ob.connected_charge(v))
            row["rel"] = {k: abs(row["a"][k] - a0[k]) / abs(a0[k]) for k in rc.OBS}
            row["rel_conn_charge"] = abs(row["a_conn_charge"] - a0["charge_conn"]) / abs(a0["charge_conn"])
            Ls = lanczos_states(m.H, v, 2)
            row["lanczos"] = [dict(dE=float(x @ (m.H @ x)) - W[0], a=ob.all(x)) for x in Ls]
            res["states"][f"s{seed}_{lab}"] = row
            log(f"seed {seed} [{lab}] dE={row['dE']:.5f} ov0={row['ov0']:.4f} dE_sign(P)={dsign} | "
                + " ".join(f"{k}: a={row['a'][k]:+.5e} d={a0[k] - row['a'][k]:+.3e} rel={row['rel'][k]:.3f}" for k in rc.OBS)
                + f" | conn-charge rel={row['rel_conn_charge']:.3f}")
            for p, L in enumerate(row["lanczos"], 1):
                log(f"    L{p}: dE={L['dE']:.5f} " + " ".join(
                    f"{k}: d={a0[k] - L['a'][k]:+.3e}" for k in rc.OBS))
    P = [r for r in res["states"].values() if r["label"] == "P"]
    verdict = "NO-DATA"
    if P:
        med = {k: float(np.median([r["rel"][k] for r in P])) for k in rc.OBS}
        ds = [r["dE_sign_P"] for r in P if r["dE_sign_P"] is not None]
        med_sign = float(np.median(ds)) if ds else float("nan")
        hit = [k for k in rc.OBS if med[k] >= 0.02]
        premise = med_sign >= 0.005
        verdict = ("TARGET " + ",".join(hit)) if (hit and premise) else "NO-TARGET"
        log(f"median rel error over {len(P)} P states: " + " ".join(f"{k}={v:.4f}" for k, v in med.items())
            + f"; median dE_sign(P)={med_sign:.4f} (premise >= 0.005: {premise})")
        res.update(median_rel=med, median_dE_sign=med_sign, n_seeds=len(P))
    res["verdict"] = verdict
    log(f"STEP-1 VERDICT: {verdict}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
