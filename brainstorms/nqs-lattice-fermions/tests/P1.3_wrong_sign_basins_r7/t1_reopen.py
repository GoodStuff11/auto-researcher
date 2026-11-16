#!/usr/bin/env python3
"""
P1.3 reopening review (2026-11-16): T1 tests of the routes generated from the CAPACITY closure.
================================================================================================
Rules below were written before any run. Exact sums only; no Slurm job.

Instance: the r6 probe toy (P1.2 t1c size 2): t-J, t = 1, t' = -0.25, J = 0.4, open 4x3, 5+5, D = 16632,
E1 - E0 = 0.0847 (gap assert >= 0.02 in C.model). Group {1, Mx, My, MxMy}, chi from psi_0. Ansatz NNBF
(LRAnsatz "full", K = 1), width 3 (675 parameters), trained as psi_P = P_chi psi_theta. Cold starts S =
results/S_2_w3_<seed>.npz of tests/P1.3_wrong_sign_basins_r6 (seeds 11, 22, 33); C300 from R_2_w3_<seed>.json.
dEs = E_sign - E_0 (P1.2 t1_vote.dE_sign), dEt = E(psi_P) - E_0.

MODE spectrum (R7.3 pre-step, also a free diagnostic for R7.1)
  In-sector levels: lowest 4 eigenvalues of P_chi H P_chi (+100 (1 - P_chi)), with <S^2>, at t' in
  {-0.45,-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35} (J = 0.4). Overlaps |<E_k|psi_P(S)>|^2, k = 0..3, of the
  three r6 cold starts at t' = -0.25.
  R7.3 gate (pre-registered): R7.3 runs its trap test only if some t' has an in-sector gap Delta_in with
  0.02 <= Delta_in <= 0.15 (0.0017t to 0.0125t per site). Otherwise R7.3 is KILLED at T1 as NO-INSTANCE
  (no same-sector competitor near the Hard-Core scale on a counted toy). It also reports, at t' = -0.25,
  whether any cold start is "trapped": dEt >= 0.5 Delta_in and ov_1 > ov_0.

MODE lmfit (R7.1: is r6's ORACLE-MISS an artefact of first-order fitting?)
  From S, maximise F = <t|psi_P>^2/(<t|t><psi_P|psi_P>) towards the SAME sign-oracle target as r6,
  t = sgn(psi_0)|psi_P(S)|, with Levenberg-Marquardt on the exact Jacobian J = d psi_P/d theta over the whole
  basis (residual u - v, u = psi_P/|psi_P|, v = t/|t|; tangent A = (1 - u u^T) J/|psi_P|; step
  -(A^T A + mu I)^-1 A^T (u - v); accept if F rises, else mu *= 4, up to 6 retries; mu /= 3 on acceptance).
  At most --lm_iters (default 30) accepted iterations; dEs every 5; keep the lowest-dEs checkpoint L0.
  Then release L0 for 100 + 200 projected energy SR steps (r6 recipe and rng seeds seed+21, seed+22).
  PRE-REGISTERED RULE (medians over seeds 11, 22, 33; c0 = med dEs(S)):
    LM-INVALID  LM's best F is not above r6's best SR F (from the r6 trace) on at least 2 of 3 seeds: the fit is
                not stronger than SR, the test says nothing; leaf stays closed, scope note only.
    LM-MISS     med dEs(L0) > 0.5 c0: a second-order fit to the exact signs also cannot halve the sign error
                at this width; ORACLE-MISS is not a first-order artefact; R7.1 KILLED, leaf stays closed.
    otherwise   release, then r6's decide() (BASIN / SIGN-DEGENERATE / TRADE / RELAX / UNCLEAR) against r6's C300.
                BASIN reopens P1.3 with the LM oracle as its direct probe; every other label keeps it closed.

MODE capture (R7.3, only if the spectrum gate opened; at the t' with the smallest admissible Delta_in)
  Cold projected SR, width 3, 300 steps, from three references: UHF seed 11, UHF seed 22, Fermi sea of -T.
  Oracle arm: LM fidelity fit to psi_0 itself (15 iterations) from the UHF-11 start, then 300 SR steps.
  PRE-REGISTERED RULE: NO-BASIN if 0 of 3 cold runs are trapped (KILL R7.3); BASIN-CANDIDATE if at least one
  cold run is trapped and the oracle arm ends untrapped with dEt at least 0.25 Delta_in below every trapped run
  (reopen with SNO-PE re-audit); TRADE/UNCLEAR otherwise (stays closed).

Budget (enforced in code before the heavy part): RLIMIT_CPU per invocation (--cpu_cap, default 580 s); a timed
Jacobian and SR step project each invocation's cost; abort if over 90% of --cpu_cap or if the review ledger
(results/ledger.json) plus the projection exceeds --task_cap (default 1800 CPU-s). Threads 1..4 (default 1).

Usage
-----
  python3 t1_reopen.py --mode spectrum
  python3 t1_reopen.py --mode lmfit --seeds 11
  python3 t1_reopen.py --mode verdict
  python3 t1_reopen.py --mode capture         (only if the spectrum gate opened)
Options: --mode {spectrum,lmfit,capture,verdict}; --seeds LIST (default 11,22,33); --lm_iters INT (default 30);
         --threads INT (1..4); --cpu_cap FLOAT (default 580); --task_cap FLOAT (default 1800).
"""
import os
import sys


def _pre_threads():
    n = 1
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    n = max(1, min(4, n))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)
    return n


NTHREADS = _pre_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.join(os.path.dirname(HERE), "P1.3_wrong_sign_basins_r6")
sys.path.insert(0, R6)
import t1_basin_probe as P6  # noqa: E402  (imports P1.2 r6 helpers: C = t1c_warm, E = t1e_closed)
C, E, H21 = P6.C, P6.E, P6.H21
from tj_core import TJ, SpaceGroup, LRAnsatz  # noqa: E402
from p11_core import solve_ed, uhf_orbitals  # noqa: E402

OUT = os.path.join(HERE, "results")
LEDGER = os.path.join(OUT, "ledger.json")
log = C.log
K, W = 2, 3


def ledger_used():
    return float(sum(r["cpu"] for r in json.load(open(LEDGER)))) if os.path.exists(LEDGER) else 0.0


def ledger_add(mode, cpu, note):
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    rows.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, cpu=cpu, note=note))
    json.dump(rows, open(LEDGER, "w"), indent=1)


def gate(proj, args, label):
    used, prior = time.process_time(), ledger_used()
    log(f"{label}: projected {proj:.0f} CPU-s + {used:.0f} used; invocation cap {args.cpu_cap:.0f}; "
        f"review ledger {prior:.0f} of {args.task_cap:.0f}")
    if proj + used > 0.9 * args.cpu_cap or prior + proj + used > args.task_cap:
        log("BUDGET-ABORT")
        ledger_add(label + "-abort", time.process_time(), "")
        sys.exit(3)


def in_sector_levels(m, sg, k=4):
    P = sg.project
    op = LinearOperator((m.D, m.D), matvec=lambda v: P(m.H @ P(v)) + 100.0 * (v - P(v)), dtype=float)
    v0 = P(np.random.default_rng(5).standard_normal(m.D))
    w, v = eigsh(op, k=k, which="SA", v0=v0, tol=1e-10)
    o = np.argsort(w)
    w, v = w[o], v[:, o]
    if not hasattr(m, "Sp"):
        m.spin_setup()
    s2 = [m.S2_expect(v[:, i]) for i in range(k)]
    return w, v, s2


def run_spectrum(args):
    os.makedirs(OUT, exist_ok=True)
    gate(9 * 25.0, args, "spectrum")
    rows, adm = [], []
    for tp in (-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35):
        m = TJ(4, 3, "obc", "obc", 5, 5, 1.0, tp, 0.4)
        Wd, Vv, deg = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
        sg = SpaceGroup(m, diag=False)
        if deg != 1 or float(Wd[1] - Wd[0]) < 0.02:
            log(f"t' {tp:+.2f}: gap {Wd[1]-Wd[0]:.4f} deg {deg}: not a counted toy (gap assert), skipped")
            continue
        gen = sg.set_from(Vv[:, 0])
        w, v, s2 = in_sector_levels(m, sg)
        din = float(w[1] - w[0])
        ok = 0.02 <= din <= 0.15
        if ok:
            adm.append(tp)
        rows.append(dict(tp=tp, chi=gen, E=list(map(float, w)), S2=s2, gap=float(Wd[1] - Wd[0]), gap_in=din,
                         admissible=ok))
        log(f"t' {tp:+.2f}: chi {gen} gap {Wd[1]-Wd[0]:.4f} in-sector levels-E0 "
            f"{[round(x - w[0], 4) for x in w]} <S^2> {[round(x, 3) for x in s2]} Delta_in {din:.4f} "
            f"admissible {ok}")
        if abs(tp + 0.25) < 1e-9:
            for seed in (11, 22, 33):
                a = P6.load(os.path.join(R6, "results", f"S_2_w3_{seed}.npz"), K, W, seed)
                psi = sg.project(a.amplitudes(8192))
                psi /= np.linalg.norm(psi)
                ov = [float((v[:, i] @ psi) ** 2) for i in range(4)]
                dEt = float(psi @ (m.H @ psi) - w[0])
                trapped = dEt >= 0.5 * din and ov[1] > ov[0]
                log(f"  r6 cold start S seed {seed}: dEt {dEt:.5f} ov_k {[round(x, 5) for x in ov]} trapped {trapped}")
                rows[-1].setdefault("S_overlaps", {})[seed] = dict(ov=ov, dEt=dEt, trapped=trapped)
    verdict = "OPEN (trap test runs at t' = %s)" % min(adm, key=lambda t: [r["gap_in"] for r in rows
                                                                              if r["tp"] == t][0]) if adm \
        else "KILLED at T1: NO-INSTANCE (no admissible in-sector gap on the counted toy)"
    log(f"R7.3 spectrum gate: {verdict}")
    json.dump(dict(rows=rows, verdict=verdict), open(os.path.join(OUT, "spectrum.json"), "w"), indent=1)
    ledger_add("spectrum", time.process_time(), verdict)
    return 0


def jacobian(a, sg, chunk=128):
    """J[n, p] = d psi_P(n)/d theta_p over the whole basis (psiP = 1 turns O_P into J)."""
    D = a.m.D
    out = np.empty((D, a.n_params()))
    ones = np.ones(D)
    for s in range(0, D, 2048):
        idx = np.arange(s, min(s + 2048, D))
        out[idx] = a.per_sample_O_proj(sg, idx, ones, chunk)
    return out


def lm_fit(a, sg, tgt, iters, m, psi0, E0, label, eval_every=5):
    v = tgt / np.linalg.norm(tgt)
    mu = None
    best = None
    trace = []

    def state():
        p = sg.project(a.amplitudes(8192))
        n = np.linalg.norm(p)
        u = p / n
        if u @ v < 0:
            u = -u
        return u, n, float((u @ v) ** 2)
    u, n, F = state()
    dEt, dEs = E.score(a, m, psi0, E0, sg)
    best = (dEs, a.flat().copy(), 0, F)
    trace.append((0, F, dEt, dEs))
    acc = 0
    while acc < iters:
        Jm = jacobian(a, sg)
        sgn = 1.0 if (sg.project(a.amplitudes(8192)) @ v) >= 0 else -1.0
        A = sgn * (Jm - np.outer(u, u @ Jm)) / n
        G = A.T @ A
        g = A.T @ (u - v)
        if mu is None:
            mu = 1e-3 * np.trace(G) / len(G)
        x0 = a.flat().copy()
        ok = False
        for _ in range(7):
            d = -np.linalg.solve(G + mu * np.eye(len(G)), g)
            a.set_flat(x0 + d)
            u2, n2, F2 = state()
            if np.isfinite(F2) and F2 > F:
                ok = True
                mu /= 3.0
                break
            mu *= 4.0
        if not ok:
            a.set_flat(x0)
            log(f"{label}: LM stalled at accepted it {acc} (mu {mu:.2e}), F {F:.6f}")
            break
        u, n, F = u2, n2, F2
        acc += 1
        if acc % eval_every == 0 or acc == iters:
            dEt, dEs = E.score(a, m, psi0, E0, sg)
            trace.append((acc, F, dEt, dEs))
            log(f"{label}: LM it {acc}: F {F:.6f} dEt {dEt:.5f} dEs {dEs:.5f} mu {mu:.2e}")
            if dEs < best[0]:
                best = (dEs, a.flat().copy(), acc, F)
    a.set_flat(best[1])
    return best, trace, F


def run_lmfit(args):
    os.makedirs(OUT, exist_ok=True)
    m, psi0, E0, ref, gap = C.model(K)
    sg, gen = E.group(K)
    jobs = [s for s in args.seed_list if not os.path.exists(os.path.join(OUT, f"L_{s}.json"))]
    if not jobs:
        log("nothing to do")
        return 0
    a = P6.load(os.path.join(R6, "results", f"S_2_w3_{jobs[0]}.npz"), K, W, jobs[0])
    t0 = time.process_time()
    jacobian(a, sg)
    tj = time.process_time() - t0
    per = E.time_steps(a, m, sg, "chi")
    proj = len(jobs) * (args.lm_iters * (tj + 4 * 0.3) * 1.15 + (args.lm_iters // 5 + 4) * 3.0 + 300 * per * 1.05)
    log(f"Jacobian {tj:.1f} CPU-s; SR step {per:.3f} CPU-s; params {a.n_params()}")
    gate(proj, args, "lmfit")
    for seed in jobs:
        c0 = time.process_time()
        fn = os.path.join(R6, "results", f"S_2_w3_{seed}.npz")
        r6 = json.load(open(os.path.join(R6, "results", f"R_2_w3_{seed}.json")))
        a = P6.load(fn, K, W, seed)
        tgt = np.where(psi0 >= 0, 1.0, -1.0) * np.abs(sg.project(a.amplitudes(8192)))
        best, trace, Ffin = lm_fit(a, sg, tgt, args.lm_iters, m, psi0, E0, f"L {seed}")
        row = dict(seed=seed, S=r6["S"], C300=r6["C300"], C100=r6["C100"],
                   F_SR_best=max(x[1] for x in r6["O0"]["trace"]), O0_SR=r6["O0"]["dEs"],
                   L0=dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)), it=best[2], F=best[3]),
                   F_LM_final=Ffin, trace=trace)
        H21.optimize(a, m, sg, "chi", 0, C.sra(100), np.random.default_rng(seed + 21), f"L {seed} a")
        row["O100"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)))
        H21.optimize(a, m, sg, "chi", 0, C.sra(200), np.random.default_rng(seed + 22), f"L {seed} b")
        row["O300"] = dict(zip(("dEt", "dEs"), E.score(a, m, psi0, E0, sg)))
        row["cpu"] = time.process_time() - c0
        json.dump(row, open(os.path.join(OUT, f"L_{seed}.json"), "w"), indent=1)
        log(f"seed {seed}: S {r6['S']['dEs']:.5f} | SR oracle O0 {r6['O0']['dEs']:.5f} (F {row['F_SR_best']:.5f}) | "
            f"LM L0 {row['L0']['dEs']:.5f} at it {best[2]} (F {best[3]:.5f}, final F {Ffin:.5f}) | "
            f"release O100 {row['O100']['dEs']:.5f} O300 {row['O300']['dEs']:.5f} (dEt {row['O300']['dEt']:.5f}) "
            f"vs C300 {r6['C300']['dEs']:.5f} (dEt {r6['C300']['dEt']:.5f}) ({row['cpu']:.0f} CPU-s)")
    ledger_add("lmfit", time.process_time(), f"seeds {jobs}")
    return 0


def decide_lm(rows):
    c0 = float(np.median([r["S"]["dEs"] for r in rows]))
    L0 = float(np.median([r["L0"]["dEs"] for r in rows]))
    stronger = sum(r["L0"]["F"] > r["F_SR_best"] or r["F_LM_final"] > r["F_SR_best"] for r in rows)
    st = dict(c0=c0, L0=L0, LM_F_above_SR=stronger)
    if stronger < 2:
        return "LM-INVALID", st
    if L0 > 0.5 * c0:
        return "LM-MISS", st
    rr = [dict(S=r["S"], O0=r["L0"], O100=r["O100"], O300=r["O300"], C300=r["C300"]) for r in rows]
    v, st2 = P6.decide(rr)
    st.update(st2)
    return v, st


def run_verdict(args):
    def syn(S, L0, F, O3, C3, eO, eC):
        return dict(S=dict(dEs=S), L0=dict(dEs=L0, F=F), F_LM_final=F, F_SR_best=0.99, O100=dict(dEs=O3),
                    O300=dict(dEs=O3, dEt=eO), C300=dict(dEs=C3, dEt=eC))
    for want, r in (("LM-INVALID", syn(0.02, 0.005, 0.98, 0.006, 0.02, 0.09, 0.1)),
                    ("LM-MISS", syn(0.02, 0.015, 0.995, 0.02, 0.02, 0.1, 0.1)),
                    ("BASIN", syn(0.02, 0.005, 0.995, 0.006, 0.02, 0.09, 0.1)),
                    ("RELAX (K2, capacity)", syn(0.02, 0.005, 0.995, 0.019, 0.02, 0.1, 0.1))):
        got = decide_lm([r, r, r])[0]
        print(f"synthetic {want}: {got} {'OK' if got == want else 'MISMATCH'}")
    rows = [json.load(open(os.path.join(OUT, f"L_{s}.json"))) for s in args.seed_list
            if os.path.exists(os.path.join(OUT, f"L_{s}.json"))]
    for r in rows:
        print(f"  seed {r['seed']}: S {r['S']['dEs']:.5f} | SR-oracle O0 {r['O0_SR']:.5f} best F {r['F_SR_best']:.5f}"
              f" | LM L0 {r['L0']['dEs']:.5f} (it {r['L0']['it']}, F {r['L0']['F']:.5f}; final F "
              f"{r['F_LM_final']:.5f}) | O300 {r['O300']['dEs']:.5f} dEt {r['O300']['dEt']:.5f} | C300 "
              f"{r['C300']['dEs']:.5f} dEt {r['C300']['dEt']:.5f}")
    if len(rows) < len(args.seed_list):
        print("R7.1 VERDICT: INCOMPLETE")
    else:
        v, st = decide_lm(rows)
        print("  " + " ".join(f"{a} {b:.5f}" if isinstance(b, float) else f"{a} {b}" for a, b in st.items()))
        print(f"R7.1 VERDICT: {v}")
    print(f"review CPU ledger: {ledger_used():.0f} s")
    return 0


def run_trap(args, sp):
    """R7.3 competitor-capture test, rule in the module docstring (written before the spectrum was computed)."""
    if os.path.exists(os.path.join(OUT, "capture.json")):
        log("capture.json exists; nothing to do")
        return 0
    tp = [r for r in sp["rows"] if r["admissible"]]
    tp = min(tp, key=lambda r: r["gap_in"])["tp"]
    m = TJ(4, 3, "obc", "obc", 5, 5, 1.0, tp, 0.4)
    Wd, Vv, deg = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
    assert deg == 1 and Wd[1] - Wd[0] >= 0.02
    psi0, E0 = Vv[:, 0], float(Wd[0])
    sg = SpaceGroup(m, diag=False)
    gen = sg.set_from(psi0)
    w, v, s2 = in_sector_levels(m, sg)
    din = float(w[1] - w[0])
    assert abs(w[0] - E0) < 1e-7
    rng = np.random.default_rng(3)
    eT, U = np.linalg.eigh(-m.T + np.diag(1e-3 * rng.standard_normal(m.V)))
    refs = {"uhf11": uhf_orbitals(m, 11), "uhf22": uhf_orbitals(m, 22),
            "fermi": (U[:, :m.nup].copy(), U[:, :m.ndn].copy())}
    a = LRAnsatz(m, refs["uhf11"], "full", K=1, Hs=W, seed=11)
    per = E.time_steps(a, m, sg, "chi")
    t0 = time.process_time()
    jacobian(a, sg)
    tj = time.process_time() - t0
    gate(4 * 300 * per * 1.05 + 15 * (tj + 2.0) + 12 * 3.0, args, "capture")

    def ev(a):
        psi = sg.project(a.amplitudes(8192))
        psi /= np.linalg.norm(psi)
        ov = [float((v[:, i] @ psi) ** 2) for i in range(4)]
        dEt, dEs = E.score(a, m, psi0, E0, sg)
        return dict(dEt=dEt, dEs=dEs, ov=ov, trapped=bool(dEt >= 0.5 * din and ov[1] > ov[0]))
    res = dict(tp=tp, chi=gen, gap_in=din, levels=list(map(float, w - w[0])), S2=s2, arms={})
    for name, ref in refs.items():
        a = LRAnsatz(m, ref, "full", K=1, Hs=W, seed=11)
        H21.optimize(a, m, sg, "chi", 0, C.sra(300), np.random.default_rng(16), f"cold {name}")
        res["arms"][name] = ev(a)
        log(f"cold {name}: {res['arms'][name]}")
    a = LRAnsatz(m, refs["uhf11"], "full", K=1, Hs=W, seed=11)
    best, trace, Ff = lm_fit(a, sg, psi0.copy(), 15, m, psi0, E0, "oracle psi0")
    res["arms"]["oracle_W0"] = dict(ev(a), F=Ff)
    H21.optimize(a, m, sg, "chi", 0, C.sra(300), np.random.default_rng(16), "oracle release")
    res["arms"]["oracle"] = ev(a)
    log(f"oracle W0: {res['arms']['oracle_W0']}; released: {res['arms']['oracle']}")
    cold = [res["arms"][k] for k in refs]
    caught = [c for c in cold if c["trapped"]]
    o = res["arms"]["oracle"]
    if not caught:
        verdict = "NO-BASIN (0 of 3 cold runs captured by the competitor): R7.3 KILLED"
    elif (not o["trapped"]) and all(o["dEt"] <= c["dEt"] - 0.25 * din for c in caught):
        verdict = "BASIN-CANDIDATE: reopen with SNO-PE re-audit"
    else:
        verdict = "TRADE/UNCLEAR: stays closed"
    res["verdict"] = verdict
    log(f"R7.3 VERDICT (t' {tp}, Delta_in {din:.4f}): {len(caught)} of 3 cold runs captured -> {verdict}")
    json.dump(res, open(os.path.join(OUT, "capture.json"), "w"), indent=1)
    ledger_add("capture", time.process_time(), verdict)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.3 reopening review T1 (see docstring)")
    ap.add_argument("--mode", choices=["spectrum", "lmfit", "capture", "verdict"], required=True)
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--lm_iters", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpu_cap", type=float, default=580.0)
    ap.add_argument("--task_cap", type=float, default=1800.0)
    a = ap.parse_args(argv)
    a.seed_list = [int(s) for s in a.seeds.split(",")]
    return a


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (int(args.cpu_cap), int(args.cpu_cap) + 30))
    log(f"threads {NTHREADS}")
    if args.mode == "capture":
        sp = json.load(open(os.path.join(OUT, "spectrum.json")))
        if not sp["verdict"].startswith("OPEN"):
            log(f"trap test not run: {sp['verdict']}")
            return 0
        return run_trap(args, sp)
    return {"spectrum": run_spectrum, "lmfit": run_lmfit, "verdict": run_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
