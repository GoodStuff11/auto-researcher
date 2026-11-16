#!/usr/bin/env python3
"""
P1.2 round 6, tier T1: is the one-hop Hamiltonian vote informative about a reference's wrong signs,
and does that information survive as the cluster grows? (Gate 4 three-size trend for FRD-K; T1 for
the FRD-H mutation of the failed FRD-X; shared premise of every vote route.)
=====================================================================================================

Toy instances (exact, Hilbert dimension <= ~1.7e4, all with 2 holes, t = 1, t' = -0.25, J = 0.4,
open x open 2-leg ladders, the same t-t'-J code as round 5: r5/tj_core.py, r5/p11_core.py):
  size 0 : 4x2, N_up = N_dn = 3, D = 560     (V = 8)
  size 1 : 5x2, N_up = N_dn = 4, D = 3150    (V = 10)
  size 2 : 6x2, N_up = N_dn = 5, D = 16632   (V = 12)

Reference Phi: the UHF Slater determinant of p11_core.uhf_orbitals (seed 1234, the same start every
round-5 arm uses), Gutzwiller-projected by the t-J basis. Phi is the foil anchor; it is what FRD's
factors see before any training. Its sign is s_b = sgn Phi, globally aligned with psi_0.

Vote of a frame F (FRD's information): for every edge e in F active at n, the neighbour m_e and the
exact Hamiltonian element H_{n m_e} (r5 micro_test.edge_moves_bits); the frame off-diagonal local energy
    e_F(n) = sum_{e in F} H_{n m_e} Phi(m_e) / Phi(n).
For the exact ground state, e_H(n) = E_0 - H_nn < 0 at every n (eigen-equation), so e_F(n) > 0 votes
"the sign at n is wrong". Frames: K (hole hops, NN + NNN; FRD-K), X (spin exchange; FRD-X), H = K u X
(the full off-diagonal Hamiltonian; FRD-H, the round-6 mutation of FRD-X).
Detector D_F(n) = [e_F(n) > 0]. Wrong set Wr = {n : s_b(n) != sgn psi_0(n)}, weight p = psi_0^2.
    w_b      = p(Wr)                        (wrong-sign weight of the reference)
    recall_F = p(D_F and Wr) / p(Wr)
    lift_F   = p(Wr | D_F) / p(Wr)          (1 = uninformative)
    dEs(s)   = min_{a >= 0} <a|S H S|a>/<a|a> - E_0 (p11_core.sign_constrained_min, starts |psi_0|
               and uniform), for s_b and for the vote-repaired sign s_F = s_b * (-1)^{D_F}.
Reported only: the max-edge detector [max_e H_{n m_e} Phi(m_e)/Phi(n) > 1] (a single factor of FRD
can flip n only through one large edge term), and the same quantities for the projected reference
P_chi Phi ("G" rows).

Controls (run first, every size):
  positive: Phi := psi_0. Must give D_H empty (no false flags) and e_H(n) = E_0 - H_nn to 1e-8 on every
            n with |psi_0(n)| > 1e-6 max|psi_0| (symmetry zeros of psi_0 have no defined ratio; fix of
            2026-11-16 after the first run divided by them. A reference amplitude below 1e-10 of its
            maximum gets no vote. Rules unchanged).
  negative: s_b replaced by random signs times |psi_0| (seed 5). lift_H must lie in [0.8, 1.25]
            ("a vote cast by random neighbour signs carries no information").
  checks  : sum_e H_e y(m_e) equals (H - diag H) y for a random y to 1e-12 (frame H tables), and the
            UHF determinant amplitudes equal r5 tj_core.LRAnsatz (full, W2 = 0) amplitudes.

PRE-REGISTERED KILL RULES (written before the first run, 2026-11-16):
  R1 (Gate 4 for FRD-K's vote, U rows): FINITE-SIZE if lift_K is strictly decreasing over V = 8, 10,
     12 AND lift_K(12) < 1.2  ->  FRD-K's sign mechanism is a finite-size bonus, Gate 4 FAIL.
     EXTENSIVE-OK if lift_K >= 1.5 at all three sizes and lift_K(12) >= 0.5 lift_K(8).
     Otherwise UNDECIDED (the V = 16 lift of the T2 job decides).
  R2 (FRD-H, mutation of FRD-X): KILLED if lift_H <= lift_K at >= 2 of 3 sizes OR recall_H <= recall_K
     at >= 2 of 3 sizes (adding the exchange edges adds no wrong-sign information). Else T1-PASS.
  R3 (validity): any control or check failing makes the run INVALID (no verdict).
  Prediction recorded (not a rule): lift_X < lift_K at every size (explains the FRD-X FAIL).

Usage
-----
  OMP_NUM_THREADS=4 python3 t1_vote.py [--sizes 0,1,2] [--out results/t1_vote.json]
Options
-------
  --sizes LIST   comma-separated subset of {0, 1, 2} (default all three; the rules need all three).
  --out PATH     JSON output (default results/t1_vote.json next to this script).
  --maxiter INT  L-BFGS-B iterations of sign_constrained_min (default 3000).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r5")
sys.path.insert(0, R5)
from p11_core import sign_constrained_min, solve_ed, uhf_orbitals  # noqa: E402
from tj_core import TJ, LRAnsatz  # noqa: E402
import p21_harness as H21  # noqa: E402
import micro_test as R5T  # noqa: E402

SIZES = {0: (4, 2, 3, 3), 1: (5, 2, 4, 4), 2: (6, 2, 5, 5)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r6 T1 vote informativeness (see docstring)")
    ap.add_argument("--sizes", default="0,1,2")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t1_vote.json"))
    ap.add_argument("--maxiter", type=int, default=3000)
    a = ap.parse_args(argv)
    a.size_list = [int(s) for s in a.sizes.split(",")]
    assert all(s in SIZES for s in a.size_list)
    return a


def uhf_amplitudes(m, Pu, Pd):
    du = np.linalg.det(Pu[m.rows_u])
    dd = np.linalg.det(Pd[m.rows_d])
    iu, idd = m.iu_id()
    return du[iu] * dd[idd]


def frame_tables(m, frame):
    edges = R5T.edge_list(m.T, m.nn, "K") + R5T.edge_list(m.T, m.nn, "X") if frame == "H" else \
        R5T.edge_list(m.T, m.nn, frame)
    nbu, nbd, act, hc = R5T.edge_moves_bits(m.bu, m.bd, edges, m.T, m.J)
    Mi = -np.ones(act.shape, np.int64)
    Mi[act] = m.index(nbu[act], nbd[act])
    assert (Mi[act] >= 0).all()
    return Mi, hc


def vote(phi, Mi, hc):
    act = Mi >= 0
    good = np.abs(phi) > 1e-10 * np.max(np.abs(phi))   # configurations where the ratio is defined
    den = np.where(good, phi, 1.0)[:, None]
    terms = np.where(act & good[:, None], hc * phi[np.maximum(Mi, 0)], 0.0) / den
    return terms.sum(1), terms.max(1, initial=-np.inf, where=act & good[:, None])


def stats(s_b, s0, p, eF, emax):
    wr = s_b != s0
    D = eF > 0
    Dm = emax > 1.0
    wb = p[wr].sum()
    pD = p[D].sum()
    out = dict(w_b=float(wb), flag_weight=float(pD),
               recall=float(p[D & wr].sum() / wb) if wb > 0 else float("nan"),
               lift=float((p[D & wr].sum() / pD) / wb) if (pD > 0 and wb > 0) else float("nan"),
               recall_max=float(p[Dm & wr].sum() / wb) if wb > 0 else float("nan"),
               lift_max=float((p[Dm & wr].sum() / p[Dm].sum()) / wb) if (p[Dm].sum() > 0 and wb > 0)
               else float("nan"))
    return out, D


def dE_sign(H, s, psi0, E0, maxiter):
    S = sp.diags(s)
    K = (S @ H @ S).tocsr()
    uni = np.ones(len(s)) + 0.01 * np.random.default_rng(3).random(len(s))
    return float(sign_constrained_min(K, [("abs_exact", psi0), ("uniform", uni)], maxiter)[0] - E0)


def run_size(k, args):
    t0 = time.time()
    Lx, Ly, nu, nd = SIZES[k]
    m = TJ(Lx, Ly, "obc", "obc", nu, nd, 1.0, -0.25, 0.4)
    W, Vv, deg = solve_ed(m, 2, 1e-8, H21.UHF_SEED)
    psi0 = Vv[:, 0]
    E0 = float(W[0])
    p = psi0 ** 2 / (psi0 @ psi0)
    s0 = np.where(psi0 >= 0, 1.0, -1.0)
    res = dict(size=k, Lx=Lx, Ly=Ly, nup=nu, ndn=nd, D=m.D, V=m.V, E0=E0, gap=float(W[1] - W[0]), deg=deg)
    ok = deg == 1
    Pu, Pd = uhf_orbitals(m, H21.UHF_SEED)
    phi = uhf_amplitudes(m, Pu, Pd)
    # check: equals LRAnsatz full with zero backflow output
    la = LRAnsatz(m, (Pu, Pd), "full", K=1, Hs=2, seed=0)
    la.p["W2"][:] = 0.0
    chk_amp = float(np.max(np.abs(la.amplitudes(65536) - phi)) / np.max(np.abs(phi)))
    tabs = {F: frame_tables(m, F) for F in ("K", "X", "H")}
    y = np.random.default_rng(1).standard_normal(m.D)
    Hoff = m.H - sp.diags(m.H.diagonal())
    MiH, hcH = tabs["H"]
    chk_H = float(np.max(np.abs(np.where(MiH >= 0, hcH * y[np.maximum(MiH, 0)], 0).sum(1) - Hoff @ y)))
    # positive control: Phi := psi_0
    eH0, _ = vote(psi0, *tabs["H"])
    pc_flags = int(np.sum(eH0 > 0))
    nz = np.abs(psi0) > 1e-6 * np.max(np.abs(psi0))   # symmetry zeros of psi_0 excluded (ratio undefined)
    pc_err = float(np.max(np.abs(eH0 - (E0 - m.H.diagonal()))[nz]))
    # negative control: random signs times |psi_0|
    rnd = np.where(np.random.default_rng(5).random(m.D) < 0.5, -1.0, 1.0) * np.abs(psi0)
    eHr, emr = vote(rnd, *tabs["H"])
    nc, _ = stats(np.where(rnd >= 0, 1.0, -1.0), s0, p, eHr, emr)
    res["checks"] = dict(amp_vs_LRAnsatz=chk_amp, frame_H_vs_Hoff=chk_H, pos_ctrl_flags=pc_flags,
                         pos_ctrl_err=pc_err, neg_ctrl_lift_H=nc["lift"])
    valid = (chk_amp < 1e-10 and chk_H < 1e-12 and pc_flags == 0 and pc_err < 1e-8
             and 0.8 <= nc["lift"] <= 1.25 and ok)
    res["valid"] = bool(valid)
    # sign premise of the eigen-equation vote
    res["frac_E0_below_diag"] = float(np.mean(E0 - m.H.diagonal() < 0))
    sg, _ = H21.build_symmetry(m, psi0)
    for lab, ref in (("U", phi), ("G", sg.project(phi))):
        s_b = np.where(ref >= 0, 1.0, -1.0)
        if p[s_b == s0].sum() < 0.5:
            s_b, ref = -s_b, -ref
        row = dict(dEs_base=dE_sign(m.H, s_b, psi0, E0, args.maxiter))
        for F in ("K", "X", "H"):
            eF, em = vote(ref, *tabs[F])
            st, D = stats(s_b, s0, p, eF, em)
            st["dEs_repaired"] = dE_sign(m.H, s_b * np.where(D, -1.0, 1.0), psi0, E0, args.maxiter)
            row[F] = st
        res[lab] = row
    res["seconds"] = time.time() - t0
    return res


def verdict(R):
    U = [r["U"] for r in R]
    lk = [u["K"]["lift"] for u in U]
    lh = [u["H"]["lift"] for u in U]
    rk = [u["K"]["recall"] for u in U]
    rh = [u["H"]["recall"] for u in U]
    lx = [u["X"]["lift"] for u in U]
    dec = all(lk[i + 1] < lk[i] for i in range(len(lk) - 1))
    if dec and lk[-1] < 1.2:
        r1 = "FINITE-SIZE (Gate 4 FAIL for FRD-K's vote)"
    elif all(v >= 1.5 for v in lk) and lk[-1] >= 0.5 * lk[0]:
        r1 = "EXTENSIVE-OK"
    else:
        r1 = "UNDECIDED (V = 16 lift of the T2 job decides)"
    worse = sum(lh[i] <= lk[i] for i in range(len(lk)))
    worse_r = sum(rh[i] <= rk[i] for i in range(len(lk)))
    r2 = "KILLED" if (worse >= 2 or worse_r >= 2) else "T1-PASS"
    pred = all(lx[i] < lk[i] for i in range(len(lk)))
    return dict(R1_frdK_gate4=r1, R2_frdH=r2, lift_K=lk, lift_H=lh, lift_X=lx, recall_K=rk, recall_H=rh,
                prediction_liftX_below_liftK=pred)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    R = []
    for k in args.size_list:
        r = run_size(k, args)
        R.append(r)
        print(f"size {k} {r['Lx']}x{r['Ly']} {r['nup']}+{r['ndn']} D={r['D']} E0={r['E0']:.6f} gap={r['gap']:.4f} "
              f"valid={r['valid']} checks={r['checks']} E0<H_nn frac={r['frac_E0_below_diag']:.3f} "
              f"({r['seconds']:.0f} s)")
        for lab in ("U", "G"):
            row = r[lab]
            print(f"  [{lab}] w_b={row['K']['w_b']:.4f} dEs_base={row['dEs_base']:.5f}")
            for F in ("K", "X", "H"):
                s = row[F]
                print(f"     {F}: flag_w={s['flag_weight']:.4f} recall={s['recall']:.3f} lift={s['lift']:.3f} "
                      f"| max-edge recall={s['recall_max']:.3f} lift={s['lift_max']:.3f} "
                      f"| dEs repaired={s['dEs_repaired']:.5f}")
    out = dict(results=R)
    if len(R) == 3 and all(r["valid"] for r in R):
        out["verdict"] = verdict(R)
        print("VERDICT:", json.dumps(out["verdict"]))
    else:
        print("VERDICT: INVALID or incomplete (rules need three valid sizes)")
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
