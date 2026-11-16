#!/usr/bin/env python3
r"""
P3.2 closed-leaf falsifier (2026-11-16): does a SAMPLED Lanczos step keep the exact-arithmetic gain?
====================================================================================================

Round 6 (job 619080) found that one symmetric block-Lanczos step on exact width-4 states decides
9 of 9 instances (in-sample, exact matrices). The leaf closed with "compare branch energies with
final-state sigma_MC abstention after one symmetric Lanczos step on both branches" as recommended
practice. This script tests whether that step keeps its accuracy with finite samples.

States. The trained target-point branch states of 619080 were not saved (only their energies). The
saved width-4 states are the round-4 certified snapshot parameters
(../P3.2_biased_ensemble_fix/results/w4_s<seed>_snap_theta.npz, at |d| >= 0.03). Each branch psi_X
(X = A, B) is the snapshot of order X nearest to the target, evaluated at the target t' with NO
further training ("W0 states", the warm start of the 619080 branches). Projected into the irrep.
Instance: 4x3 PBC Hubbard 4+4, U = 8, t_y = 1.11394, irrep K = 0, Mx = My = +1, targets
d = t' - t'_c in --targets, seeds 1234 2345 3456 (the 619080 grid).

Decision rules (order A or B; U = abstain; tau = 0.02 on the order character c):
  NAIVE-x   lower exact energy wins; order of that branch (exact, no abstention).
  NAIVE-s   NAIVE-x with abstention |E_A - E_B| <= 2 hypot(SE_A, SE_B), SE = sqrt(var / n_mc).
  L1        one Lanczos step per branch (Sorella 2001): Ritz in span{psi_X, H psi_X}; the branch
            with the lower Ritz energy wins and the order is c of its Ritz vector.
  PBL       one symmetric block-Lanczos step on the pair: Ritz in span{psi_A, H psi_A, psi_B, H psi_B};
            order of the lowest Ritz vector.
  Exact versions use the vectors. Sampled versions use i.i.d. draws (no Markov-chain
  autocorrelation, which is optimistic) and exact local values, as a VMC Lanczos step computes them:
  L1 samples n ~ |psi_X|^2 (--n-s per branch) and uses e1 = (H psi)(n)/psi(n),
  e2 = (H^2 psi)(n)/psi(n): <H> = mean e1, <H^2> = mean e1^2, <H^3> = mean e1 e2, and
  c(chi) = mean[(x0 + x1 e1)^2 cn] / mean[(x0 + x1 e1)^2]; abstain if the Ritz-energy difference is
  within 2 jackknife SE (20 blocks). PBL samples 2 --n-s draws from p = (psi_A^2 + psi_B^2)/2 and
  estimates the 4x4 overlap, Hamiltonian and c matrices by importance weights 1/p; abstain if
  |c| <= max(tau, 2 SE_jackknife(c)).

Pre-registered rule (also in the leaf file, written before the run)
  N0 = instances where NAIVE-x is wrong. For V in {L1, PBL}: premise_V holds if |N0| >= 2, exact V
  is right on at least half of N0, and exact V is never wrong where NAIVE-x is right.
  No premise holds -> INCONCLUSIVE-PREMISE (the W0 states do not reproduce the trained-state
  situation; a T2 job that retrains, saves and samples is prepared instead).
  For each V whose premise holds, at --n-s = 20000 over --reps replicates:
    f = fraction (over N0 x replicates) where sampled V commits RIGHT;
    w = fraction (over all instances x replicates) where sampled V commits WRONG.
    KEEPS if f >= 0.5 and w <= 0.05; LOSES if f < 0.25 or w > 0.10; PARTIAL otherwise.
  Leaf consequence: neither outcome reopens P3.2 (BL failed Gate 0 as a route; it is practice).
  KEEPS confirms the recommended practice under sampling; LOSES scopes it to exact arithmetic.

Usage
  python3 sampled_bl.py --seed 1234            one seed, all targets (writes results/s<seed>.json)
  python3 sampled_bl.py --aggregate results    applies the rule to results/*.json

Options
  --seed INT        snapshot seed: 1234, 2345 or 3456 (width 4 files exist for these).
  --targets LIST    comma list of d (default "-0.0005,-0.0002,0.0005"; write "--targets=-0.0005,...").
  --n-s LIST        comma list of per-branch sample counts (default "2000,20000"; the rule reads 20000).
  --reps INT        sampling replicates per n_s (default 40).
  --n-mc INT        sample count for the NAIVE-s standard error (default 20000, as 619080).
  --tau FLOAT       |c| below which the order is U (default 0.02).
  --chunk INT       amplitude chunk (default 65536).
  --out PATH        output JSON (default results/s<seed>.json).
  --aggregate DIR   apply the rule to DIR/*.json and exit.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "P3.2_biased_ensemble_fix")
sys.path.insert(0, FIX)
import micro_test as MT  # noqa: E402

log = MT.P.log


def ritz_vecs(H, V):
    """Lowest Ritz pair of H in span of the columns of V (canonical orthogonalisation)."""
    S = V.T @ V
    s, U = np.linalg.eigh(S)
    keep = s > 1e-10 * s.max()
    X = U[:, keep] / np.sqrt(s[keep])
    B = V @ X
    w, c = np.linalg.eigh(B.T @ (H @ B))
    return float(w[0]), B @ c[:, 0]


def ritz_mats(S, Hm, cut=1e-8):
    S, Hm = 0.5 * (S + S.T), 0.5 * (Hm + Hm.T)
    s, U = np.linalg.eigh(S)
    keep = s > cut * s.max()
    X = U[:, keep] / np.sqrt(s[keep])
    w, c = np.linalg.eigh(X.T @ Hm @ X)
    return float(w[0]), X @ c[:, 0]


def jk(vals_fn, idx_blocks):
    """Jackknife SE of a scalar statistic given a function of the kept-block index list."""
    nb = len(idx_blocks)
    th = np.array([vals_fn(np.concatenate([b for j, b in enumerate(idx_blocks) if j != i])) for i in range(nb)])
    return float(np.sqrt((nb - 1) / nb * np.sum((th - th.mean()) ** 2)))


def sampled_L1(psi, Hp, H2p, cn, n, rng, nblk=20):
    p = psi ** 2
    idx = rng.choice(len(p), size=n, p=p / p.sum())
    e1 = Hp[idx] / psi[idx]
    e2 = H2p[idx] / psi[idx]
    cc = cn[idx]

    def stat(sel):
        m1, m2, m3 = e1[sel].mean(), (e1[sel] ** 2).mean(), (e1[sel] * e2[sel]).mean()
        w, v = eigh(np.array([[m1, m2], [m2, m3]]), np.array([[1.0, m1], [m1, m2]]))
        return float(w[0]), v[:, 0]

    allsel = np.arange(n)
    E, x = stat(allsel)
    amp = x[0] + x[1] * e1
    c = float(np.mean(amp ** 2 * cc) / np.mean(amp ** 2))
    blocks = np.array_split(allsel, nblk)
    se = jk(lambda s: stat(s)[0], blocks)
    return E, se, c


def sampled_PBL(V, HV, cn, n, rng, nblk=20, tau=0.02):
    p = 0.5 * (V[:, 0] ** 2 + V[:, 2] ** 2)
    p = p / p.sum()
    idx = rng.choice(len(p), size=n, p=p)
    Xs, HXs, w, cc = V[idx], HV[idx], 1.0 / p[idx], cn[idx]

    def stat(sel):
        Xw = Xs[sel] * w[sel, None]
        S = Xw.T @ Xs[sel] / len(sel)
        Hm = Xw.T @ HXs[sel] / len(sel)
        _, y = ritz_mats(S, Hm)
        chi = Xs[sel] @ y
        return float(np.sum(chi ** 2 * w[sel] * cc[sel]) / np.sum(chi ** 2 * w[sel]))

    allsel = np.arange(n)
    c = stat(allsel)
    se = jk(stat, np.array_split(allsel, nblk))
    dec = "U" if abs(c) <= max(tau, 2 * se) else ("B" if c > 0 else "A")
    return c, se, dec


def run(a):
    t0 = time.time()
    inst = MT.Instance(1.11394, 8.0)
    log(f"instance: t'_c={inst.tpc:+.6f} min gap {inst.gapmin:.3e} D={inst.m.D}")
    snaps = np.load(os.path.join(FIX, "results", f"w4_s{a.seed}_snap_theta.npz"))
    pool = {X: [(float(k[1:8]), snaps[k]) for k in snaps.files if k.endswith(X)] for X in "AB"}
    cn = inst.cn
    res = dict(args=vars(a), tpc=inst.tpc, rows=[])
    for d in a.target_list:
        tp = inst.tpc + d
        w, v = inst.cl.sector_eigs(inst.coup(tp), 2)
        v0 = inst.cl.full_vec(v[:, 0]); v0 /= np.linalg.norm(v0)
        E0, c0 = float(w[0]), inst.char(v0)
        truth = MT.order_of(c0, a.tau)
        H = inst.H(tp)
        inst.set_T(tp)
        br = {}
        for X in "AB":
            ds, theta = min(pool[X], key=lambda q: abs(q[0] - d))
            ans = MT.new_ansatz(inst, X, 4, a.seed)
            ans.set_flat(np.array(theta, float).copy())
            psi = inst.sg.project(ans.amplitudes(a.chunk))
            psi /= np.linalg.norm(psi)
            Hp = H @ psi
            H2p = H @ Hp
            E = float(psi @ Hp)
            var = float(Hp @ Hp - E ** 2)
            EL, chi = ritz_vecs(H, np.stack([psi, Hp], 1))
            br[X] = dict(psi=psi, Hp=Hp, H2p=H2p, E=E, var=var, c=inst.char(psi), src_d=ds,
                         EL=EL, cL=inst.char(chi))
        low = "A" if br["A"]["E"] < br["B"]["E"] else "B"
        naive_x = MT.order_of(br[low]["c"], a.tau)
        se = {X: np.sqrt(max(br[X]["var"], 0) / a.n_mc) for X in "AB"}
        naive_s = naive_x if abs(br["A"]["E"] - br["B"]["E"]) > 2 * np.hypot(se["A"], se["B"]) else "U"
        lowL = "A" if br["A"]["EL"] < br["B"]["EL"] else "B"
        L1_x = MT.order_of(br[lowL]["cL"], a.tau)
        V = np.stack([br["A"]["psi"], br["A"]["Hp"], br["B"]["psi"], br["B"]["Hp"]], 1)
        HV = np.stack([br["A"]["Hp"], br["A"]["H2p"], br["B"]["Hp"], br["B"]["H2p"]], 1)
        EP, chiP = ritz_vecs(H, V)
        PBL_x = MT.order_of(inst.char(chiP), a.tau)
        row = dict(d=d, E0=E0, c0=c0, truth=truth, naive_x=naive_x, naive_s=naive_s, L1_x=L1_x, PBL_x=PBL_x,
                   branches={X: {k: br[X][k] for k in ("E", "var", "c", "src_d", "EL", "cL")} for X in "AB"},
                   b={X: br[X]["E"] - E0 for X in "AB"}, bL={X: br[X]["EL"] - E0 for X in "AB"},
                   PBL_err=EP - E0, sampled={})
        log(f"s={a.seed} d={d:+.4f} truth={truth} | E-E0: A {row['b']['A']:.2e} B {row['b']['B']:.2e} "
            f"(L1: A {row['bL']['A']:.2e} B {row['bL']['B']:.2e}; PBL {row['PBL_err']:.2e}) | c_A={br['A']['c']:+.3f} "
            f"c_B={br['B']['c']:+.3f} | exact: NAIVE {naive_x} NAIVE-s {naive_s} L1 {L1_x} PBL {PBL_x}")
        for ns in a.ns_list:
            rng = np.random.default_rng(1000 * a.seed + int(1e5 * (d + 1)) + ns)
            decL, decP = [], []
            for _ in range(a.reps):
                EA, seA, cA = sampled_L1(br["A"]["psi"], br["A"]["Hp"], br["A"]["H2p"], cn, ns, rng)
                EB, seB, cB = sampled_L1(br["B"]["psi"], br["B"]["Hp"], br["B"]["H2p"], cn, ns, rng)
                win_c = cA if EA < EB else cB
                decL.append("U" if abs(EA - EB) <= 2 * np.hypot(seA, seB) else MT.order_of(win_c, a.tau))
                decP.append(sampled_PBL(V, HV, cn, 2 * ns, rng, tau=a.tau)[2])
            row["sampled"][str(ns)] = dict(L1=decL, PBL=decP)
            cnt = lambda L: {k: L.count(k) for k in "ABU"}  # noqa: E731
            log(f"    n_s={ns}: sampled L1 {cnt(decL)}  PBL {cnt(decP)}")
        res["rows"].append(row)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    log(f"done, wall {res['wall_s']:.0f}s -> {a.out}")


def aggregate(dr, ns="20000"):
    rows = []
    for f in sorted(glob.glob(os.path.join(dr, "s*.json"))):
        r = json.load(open(f))
        for x in r["rows"]:
            x["seed"] = r["args"]["seed"]
            rows.append(x)
    print(f"{'seed':>5} {'d':>8} truth NAIVE-x NAIVE-s L1-x PBL-x | sampled n_s={ns}: L1 (A/B/U)  PBL (A/B/U)")
    for x in rows:
        s = x["sampled"].get(ns, {"L1": [], "PBL": []})
        c = lambda L: "/".join(str(L.count(k)) for k in "ABU")  # noqa: E731
        print(f"{x['seed']:>5} {x['d']:+8.4f}   {x['truth']}      {x['naive_x']}       {x['naive_s']}      "
              f"{x['L1_x']}     {x['PBL_x']}   |  {c(s['L1']):>9}  {c(s['PBL']):>9}")
    N0 = [x for x in rows if x["naive_x"] != x["truth"]]
    Rk = [x for x in rows if x["naive_x"] == x["truth"]]
    print(f"|N0| (exact naive wrong) = {len(N0)} of {len(rows)}")
    any_prem = False
    for V in ("L1", "PBL"):
        k = f"{V}_x"
        fixes = sum(x[k] == x["truth"] for x in N0)
        breaks = sum(x[k] not in ("U", x["truth"]) for x in Rk)
        prem = len(N0) >= 2 and fixes >= len(N0) / 2 and breaks == 0
        any_prem |= prem
        line = f"{V}: exact fixes {fixes}/{len(N0)}, exact new wrong {breaks} -> premise {prem}"
        if prem:
            R = len(rows[0]["sampled"][ns][V])
            f = np.mean([d == x["truth"] for x in N0 for d in x["sampled"][ns][V]])
            w = np.mean([d not in ("U", x["truth"]) for x in rows for d in x["sampled"][ns][V]])
            v = "KEEPS" if (f >= 0.5 and w <= 0.05) else ("LOSES" if (f < 0.25 or w > 0.10) else "PARTIAL")
            line += f"; sampled (R={R}): f={f:.3f} w={w:.3f} -> {v}"
        print(line)
    if not any_prem:
        print("VERDICT sampled BL: INCONCLUSIVE-PREMISE")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P3.2 sampled block-Lanczos falsifier (see docstring)")
    ap.add_argument("--seed", type=int, default=1234, choices=[1234, 2345, 3456])
    ap.add_argument("--targets", default="-0.0005,-0.0002,0.0005")
    ap.add_argument("--n-s", default="2000,20000")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-mc", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--out", default="")
    ap.add_argument("--aggregate", default="")
    a = ap.parse_args(argv)
    a.target_list = [float(s) for s in a.targets.split(",")]
    a.ns_list = [int(s) for s in a.n_s.split(",")]
    if not a.out:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        a.out = os.path.join(HERE, "results", f"s{a.seed}.json")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.aggregate:
        aggregate(a.aggregate if os.path.isabs(a.aggregate) else os.path.join(HERE, a.aggregate))
        return
    run(a)


if __name__ == "__main__":
    main()
