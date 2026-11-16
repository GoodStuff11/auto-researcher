#!/usr/bin/env python3
"""
P3.2 round 6: kill check of "ESR at a crossing" (trained in-irrep excited state + subspace Ritz)
================================================================================================

Question. Round 5 found that at NNBF width 4 the naive rule "lower final branch energy wins" is
wrong at d = -0.0005 and d = -0.0002 on 2 of 3 seeds, with a differential branch bias
Delta b = b_B - b_A of about +3.5e-3 to +5.5e-3 t, above the in-irrep gaps (1.4e-3 to 2.7e-3 t).
Round-6 idea (transfer of P3.1 ESR0): train the ground state psi and a penalty-orthogonal in-irrep
excited state phi_1 (sequential penalty VMC, Choo et al. PRL 121, 167204 (2018)), Ritz-diagonalise
H in span{psi, phi_1} (or span{psi, (H - E)psi, phi_1}), and decide A/B from the order character
of the lowest Ritz vector.  The 2x2 argument (leaf doc, round 6) predicts that this CANNOT beat
the naive rule: the lowest eigenvector of a real symmetric 2x2 matrix has more than half its
weight on the basis vector with the lower diagonal, so the Ritz decision is the decision of the
lower-energy trained state, i.e. the energy rule with boundary delta + Delta b' = 0.  This script
tests that prediction on the real instance with real trained width-4 states (exact sums).

Instance. Unchanged round-4/5 methods instance (imported from ../P3.2_biased_ensemble_fix):
4x3 PBC Hubbard, N_up = N_dn = 4, U = 8, t_x = 1, t_y = 1.11394, irrep K = 0, M_x = M_y = +1,
target t' = t'_c + d.  Order character c > 0: stripe (B); c < 0: near-Neel (A).

Per (seed, target d):
  1. ED: E0, E1, c(E0), c(E1), exact in-irrep vectors v0, v1 (truth = sign of c(E0)).
  2. Warm branches psi_A, psi_B: projected NNBF started from the nearest round-4 certified snapshot
     parameters of that order (../P3.2_biased_ensemble_fix/results/w<W>_s<seed>_snap_theta.npz),
     --sr-iters projected-SR steps with exact force (the round-4 R0P-warm arm, which was wrong at
     d = -0.0005, -0.0002 for seeds 1234 and 3456).
  3. Naive baseline (round-5 critic): lower final energy wins; abstain if
     |E_A - E_B| <= 2 hypot(SE_A, SE_B), SE_X = sqrt(var_X / --n-mc) (final-state sampling noise).
  4. psi = the lower-energy branch (the best ground state width W finds).  phi_1 = penalty state:
     minimise <phi|H + beta |psi><psi| |phi>/<phi|phi> (H' commutes with the irrep projector because
     psi lies in the irrep, so the projected exact force of proj_core applies unchanged), started
     from the OTHER branch's parameters (its most favourable start), --phi-iters SR steps.
  5. Exact Ritz of H in the spans
       K1    span{psi, phi_1}                         (the round-6 idea, K = 1)
       ESR0  span{psi, (H-E_psi)psi, phi_1}            (P3.1 ESR0 with K = 1)
       PAIR  span{psi_A, psi_B}                        (order-seeded pair; no penalty)
       BL    span{psi, (H-E)psi, phi_1, (H-E_1)phi_1}  (one block-Lanczos step; the proposed
                                                        alternative, reported in-sample only)
       POS   span{psi, v_other}  positive control: v_other = exact in-irrep eigenvector whose c has
                                 the sign opposite to psi (a perfect partner; the Ritz must reach truth)
       NEG   span{psi, r}        negative control: r = random in-irrep vector (Ritz must return psi)
     For each span: Ritz energy error theta_0 - E0, c of the Ritz ground vector, decision
     (sign of c; U if |c| < --tau), weight |<psi|chi_0>|^2 on psi, and for K1 and PAIR the
     "lower-diagonal" lemma prediction (order of the basis vector with the lower energy).

Pre-registered kill-check rule (also in the leaf doc, written before the run; printed at the end):
  Let N = set of (seed, target) where the naive baseline COMMITS WRONG.
  INCONCLUSIVE   if |N| < 2, or POS fails to reach truth on any instance, or NEG changes the
                 decision of psi on any instance, or |<psi|phi_1>|^2 > 0.05 on any instance.
  KILL-CONFIRMED if on every instance in N both K1 and ESR0 are wrong or abstain, AND the K1
                 decision equals the lower-diagonal prediction on every instance.
  IDEA-SURVIVES  if K1 or ESR0 decides correctly on at least half of N and makes no wrong commit
                 on any instance where the naive baseline is right.
  MIXED          otherwise (treated as KILLED for K1/ESR0: no gain without a new mechanism).

Usage
  python3 esr_killcheck.py --smoke 1
      Checks only plus a 3-step pipeline run at one target (plumbing; seconds to minutes):
      penalty force vs central finite differences; Ritz machinery vs exact ED (span{v0, v1, r}
      must give E0 and E1 to 1e-9); penalty state orthogonality; writes results_smoke/.
  python3 esr_killcheck.py --seed 1234 --targets=-0.0005,-0.0002,0.0005
      Full kill check for one seed (writes results/w<W>_s<seed>.json).
  python3 esr_killcheck.py --aggregate results
      Applies the pre-registered rule to every results/*.json and prints the verdict.

Options
  --width INT       NNBF hidden width (default 4; snapshot file for this width/seed must exist).
  --seed INT        seed of the round-4 run whose snapshot parameters warm-start the branches
                    (default 1234; 1234/2345/3456 exist for widths 4 and 8).
  --targets LIST    comma-separated d = t' - t'_c (default "-0.0005,-0.0002,0.0005").
                    Use "--targets=-0.0005,..." (with "=") because values start with "-".
  --sr-iters INT    projected-SR steps per warm branch (default 800, as round 4).
  --phi-iters INT   projected-SR steps for the penalty state phi_1 (default 800).
  --beta FLOAT      penalty strength in t (default 0.5; must exceed the in-irrep gap, 1e-3 to 1e-2).
  --sr-lr, --sr-shift, --sr-samples, --chunk, --o-chunk   SR settings (defaults 0.05, 1e-3,
                    2000, 65536, 256 as round 4).
  --n-mc INT        final-state sample count defining SE for the naive abstention (default 20000).
  --tau FLOAT       |c| below which a state is "U" (default 0.02).
  --window INT      SR noise window passed to the round-4 optimizer (default 100; logged only).
  --smoke INT       1 = smoke mode (above).
  --out PATH        output JSON (default results/w<W>_s<seed>.json or results_smoke/...).
  --aggregate DIR   apply the rule to DIR/*.json and exit.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "P3.2_biased_ensemble_fix")
sys.path.insert(0, FIX)
import micro_test as MT  # noqa: E402  (round-4 instance, warm training, state_info)
import p11_core as P  # noqa: E402
import proj_core as PC  # noqa: E402

log = P.log


class PenaltyH:
    """H' = H + beta u u^T with u a unit vector in the irrep (only `@` on vectors is needed)."""

    def __init__(self, H, u, beta):
        self.H, self.u, self.beta = H, u / np.linalg.norm(u), beta

    def __matmul__(self, x):
        return self.H @ x + self.beta * self.u * (self.u @ x)


def ritz(H, vecs):
    """Exact Ritz pairs of H in span(vecs) (canonical orthogonalisation, drops |s| < 1e-10)."""
    V = np.array(vecs).T
    S = V.T @ V
    s, U = np.linalg.eigh(S)
    keep = s > 1e-10 * s.max()
    X = U[:, keep] / np.sqrt(s[keep])
    B = V @ X                      # orthonormal basis
    h = B.T @ (H @ B)
    w, c = np.linalg.eigh(h)
    return w, B @ c


def ritz_report(inst, H, vecs, E0, psi, tau):
    w, Y = ritz(H, vecs)
    y0 = Y[:, 0] / np.linalg.norm(Y[:, 0])
    c = inst.char(y0)
    return dict(theta0=float(w[0]), err=float(w[0] - E0), c=c, dec=MT.order_of(c, tau),
                w_psi=float((y0 @ psi) ** 2), dim=int(len(w)))


def train_penalty(inst, tp, lab, width, seed, args, theta0, psi_ref):
    """phi_1 = P_chi psi_theta minimising <phi|H + beta psi psi^T|phi>/<phi|phi>, warm from theta0."""
    inst.set_T(tp)
    ans = MT.new_ansatz(inst, lab, width, seed)
    ans.set_flat(np.array(theta0, float).copy())
    rng = np.random.default_rng(seed + 4242)
    Hp = PenaltyH(inst.H(tp), psi_ref, args.beta)
    it0, args.sr_iters = args.sr_iters, args.phi_iters
    try:
        ans, st = MT.optimize(ans, Hp, inst.sg, args, rng, f"phi1({lab}) t'={tp:+.5f} W={width} s={seed}")
    finally:
        args.sr_iters = it0
    phi = inst.sg.project(ans.amplitudes(args.chunk))
    return phi / np.linalg.norm(phi), st


def ed_full(inst, tp):
    w, v = inst.cl.sector_eigs(inst.coup(tp), 2)
    v0, v1 = inst.cl.full_vec(v[:, 0]), inst.cl.full_vec(v[:, 1])
    v0, v1 = v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)
    return dict(E0=float(w[0]), E1=float(w[1]), gap=float(w[1] - w[0]), c0=inst.char(v0),
                c1=inst.char(v1)), v0, v1


def checks(inst, seed):
    ok = True
    rng = np.random.default_rng(seed)
    tp = inst.tpc + 0.002
    inst.set_T(tp)
    H = inst.H(tp)
    e, v0, v1 = ed_full(inst, tp)
    r = inst.sg.project(rng.standard_normal(inst.m.D))
    w, _ = ritz(H, [v0 + 0.3 * r, v1 - 0.2 * r, r])
    # span{v0+..., v1+..., r} = span{v0, v1, r}: lowest two Ritz values must be E0, E1
    err = max(abs(w[0] - e["E0"]), abs(w[1] - e["E1"]))
    log(f"check Ritz vs ED: span{{v0,v1,r}} theta0-E0={w[0] - e['E0']:.1e} theta1-E1={w[1] - e['E1']:.1e}")
    ok &= err < 1e-9
    # penalty force vs central finite differences (real instance, width 4)
    a = MT.new_ansatz(inst, "B", 4, seed)
    x0 = a.flat().copy()
    Hp = PenaltyH(H, v0, 0.5)
    E, f, *_ = PC.proj_energy_grad(a, Hp, inst.sg, 65536)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    Ef = lambda x: (a.set_flat(x), P.energy(Hp, inst.sg.project(a.amplitudes(65536)))[0])[1]
    fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
    a.set_flat(x0)
    log(f"check penalty force: analytic {2 * f @ d:.10e} FD {fd:.10e}")
    ok &= abs(2 * f @ d - fd) < 1e-6 * max(1.0, abs(fd))
    # penalty energy identity: <phi|H'|phi> = <H> + beta |<v0|phi>|^2
    psi = inst.sg.project(a.amplitudes(65536)); psi /= np.linalg.norm(psi)
    lhs = psi @ (Hp @ psi); rhs = psi @ (H @ psi) + 0.5 * (v0 @ psi) ** 2
    log(f"check penalty identity: {lhs:.12f} vs {rhs:.12f}")
    ok &= abs(lhs - rhs) < 1e-10
    log("ESR-KC CHECKS " + ("PASSED" if ok else "FAILED"))
    return ok


def run(args):
    t0 = time.time()
    inst = MT.Instance(args.ty, args.U)
    log(f"instance 4x3 PBC 4+4 U={args.U} t_y={args.ty}: t'_c={inst.tpc:+.6f} min gap {inst.gapmin:.3e}")
    if not checks(inst, args.seed):
        sys.exit(2)
    snapf = os.path.join(FIX, "results", f"w{args.width}_s{args.seed}_snap_theta.npz")
    snaps = np.load(snapf)
    pool = {X: [(float(k[1:8]), snaps[k]) for k in snaps.files if k.endswith(X)] for X in "AB"}
    res = dict(args={k: v for k, v in vars(args).items()}, tpc=inst.tpc, rows=[])
    for d in args.target_list:
        tp = inst.tpc + d
        e, v0, v1 = ed_full(inst, tp)
        truth = MT.order_of(e["c0"], args.tau)
        H = inst.H(tp)
        br, th = {}, {}
        for X in "AB":
            ds, theta = min(pool[X], key=lambda p: abs(p[0] - d))
            psiX, thX, st = MT.train_branch(inst, tp, X, args.width, args.seed, args, theta0=theta, tag="warm ")
            info = MT.state_info(inst, psiX, tp, e["E0"])
            info["SE"] = float(np.sqrt(max(info["var"], 0.0) / args.n_mc))
            info["src_d"] = ds
            br[X], th[X] = (psiX, info), thX
        EA, EB = br["A"][1]["E"], br["B"][1]["E"]
        low = "A" if EA < EB else "B"
        oth = "B" if low == "A" else "A"
        noise = 2 * np.hypot(br["A"][1]["SE"], br["B"][1]["SE"])
        naive = MT.order_of(br[low][1]["c"], args.tau) if abs(EA - EB) > noise else "U"
        psi = br[low][0]
        phi, stp = train_penalty(inst, tp, oth, args.width, args.seed, args, th[oth], psi)
        phi_info = MT.state_info(inst, phi, tp, e["E0"])
        ov = float((psi @ phi) ** 2)
        Epsi, Ephi = float(psi @ (H @ psi)), float(phi @ (H @ phi))
        rpsi = H @ psi - Epsi * psi
        rphi = H @ phi - Ephi * phi
        vo = v0 if np.sign(inst.char(v0)) != np.sign(br[low][1]["c"]) else v1
        rnd = inst.sg.project(np.random.default_rng(args.seed + 7).standard_normal(inst.m.D))
        spans = dict(K1=[psi, phi], ESR0=[psi, rpsi, phi], PAIR=[br["A"][0], br["B"][0]],
                     BL=[psi, rpsi, phi, rphi], POS=[psi, vo], NEG=[psi, rnd])
        rit = {k: ritz_report(inst, H, v, e["E0"], psi, args.tau) for k, v in spans.items()}
        lowdiag_K1 = MT.order_of(br[low][1]["c"] if Epsi < Ephi else phi_info["c"], args.tau)
        row = dict(d=d, ed=e, truth=truth, naive=naive, low=low, dE_AB=EA - EB, noise=noise,
                   branches={X: br[X][1] for X in "AB"}, phi=phi_info, phi_overlap_psi=ov,
                   phi_stats={k: v for k, v in stp.items() if k != "hist_ex_tail"},
                   ritz=rit, lowdiag_K1=lowdiag_K1)
        res["rows"].append(row)
        log(f"d={d:+.4f} gap={e['gap']:.2e} truth={truth} | b_A={EA - (e['E0'] if e['c0'] < 0 else e['E1']):.2e} "
            f"b_B={EB - (e['E0'] if e['c0'] > 0 else e['E1']):.2e} naive={naive} | phi: dE={phi_info['dE']:.2e} "
            f"c={phi_info['c']:+.3f} |<psi|phi>|^2={ov:.1e} | "
            + " ".join(f"{k}:{r['dec']}(c={r['c']:+.3f},w_psi={r['w_psi']:.3f},err={r['err']:.1e})"
                       for k, r in rit.items()) + f" | lowdiag(K1)={lowdiag_K1}")
        json.dump(res, open(args.out, "w"), indent=1)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(args.out, "w"), indent=1)
    log(f"done, wall {res['wall_s']:.0f}s -> {args.out}")


def aggregate(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        for x in r["rows"]:
            x["seed"] = r["args"]["seed"]
            rows.append(x)
    print(f"{'seed':>5} {'d':>8} truth naive  K1 ESR0 PAIR  BL POS NEG lowdiag |<psi|phi>|^2 w_psi(K1)")
    for x in rows:
        R = x["ritz"]
        print(f"{x['seed']:>5} {x['d']:+8.4f}   {x['truth']}     {x['naive']}    {R['K1']['dec']}   "
              f"{R['ESR0']['dec']}    {R['PAIR']['dec']}    {R['BL']['dec']}   {R['POS']['dec']}   "
              f"{R['NEG']['dec']}      {x['lowdiag_K1']}     {x['phi_overlap_psi']:.1e}       {R['K1']['w_psi']:.3f}")
    N = [x for x in rows if x["naive"] not in ("U", x["truth"])]
    R_ok = [x for x in rows if x["naive"] == x["truth"]]
    incon = (len(N) < 2 or any(x["ritz"]["POS"]["dec"] != x["truth"] for x in rows)
             or any(x["ritz"]["NEG"]["dec"] != MT.order_of(x["branches"][x["low"]]["c"], 0.02) for x in rows)
             or any(x["phi_overlap_psi"] > 0.05 for x in rows))
    bad = lambda x, k: x["ritz"][k]["dec"] != x["truth"]
    kill = (all(bad(x, "K1") and bad(x, "ESR0") for x in N)
            and all(x["ritz"]["K1"]["dec"] == x["lowdiag_K1"] for x in rows))
    fixes = max(sum(not bad(x, k) for x in N) for k in ("K1", "ESR0")) if N else 0
    newwrong = any(x["ritz"][k]["dec"] not in ("U", x["truth"]) for x in R_ok for k in ("K1", "ESR0"))
    surv = N and fixes >= len(N) / 2 and not newwrong
    v = ("INCONCLUSIVE" if incon else "KILL-CONFIRMED" if kill else "IDEA-SURVIVES" if surv else "MIXED")
    bl = sum(not bad(x, "BL") for x in N)
    print(f"|N| (naive committed wrong) = {len(N)}; K1/ESR0 best fixes {fixes}/{len(N)}; "
          f"new wrong on naive-right instances: {newwrong}; BL (alternative, in-sample) fixes {bl}/{len(N)}")
    print(f"VERDICT ESR-K1 kill check: {v}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ty", type=float, default=1.11394)
    ap.add_argument("--U", type=float, default=8.0)
    ap.add_argument("--targets", default="-0.0005,-0.0002,0.0005")
    ap.add_argument("--sr-iters", type=int, default=800)
    ap.add_argument("--phi-iters", type=int, default=800)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--o-chunk", type=int, default=256)
    ap.add_argument("--n-mc", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--aggregate", default="")
    a = ap.parse_args(argv)
    sub = "results"
    if a.smoke:
        a.targets, a.sr_iters, a.phi_iters, a.sr_samples = "-0.0005", 3, 3, 200
        sub = "results_smoke"
    a.target_list = [float(s) for s in a.targets.split(",")]
    if not a.out:
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
        a.out = os.path.join(HERE, sub, f"w{a.width}_s{a.seed}.json")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.aggregate:
        aggregate(a.aggregate if os.path.isabs(a.aggregate) else os.path.join(HERE, a.aggregate))
        return
    run(a)


if __name__ == "__main__":
    main()
