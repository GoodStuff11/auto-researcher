#!/usr/bin/env python3
"""
P1.2 round 6, tier T1 for route FRD-F (frozen-frame FRD, round-5 Route 5.4): correctness, Gate 4 cost
trend on three sizes, and fidelity of the frozen-frame substitution in the trained 4x4 FRD-K states.
=====================================================================================================

FRD-K (round 5):  psi(n) = Phi(n) prod_e [1 + alpha_e(n) g(R_e)],  R_e = c_e Phi(m_e) / Phi(n),
  Phi = NNBF (r5 tj_core.LRAnsatz, mode "full"), Phi(m_e) = the NNBF amplitude of the neighbour with ITS
  OWN backflow (one extra NNBF forward pass per active edge).
FRD-F (this file): R^F_e = c_e det(O_n[rows(m_e)]) / det(O_n[rows(n)]), where O_n = P0 + backflow(n) is
  the V x N orbital matrix of the base frozen at n. For a hole hop of spin s from site j to site i this
  is one Sherman-Morrison ratio, (O_n[i] . B_n^{-1}[:, k]) times the reordering sign (-1)^{# occupied
  spin-s sites strictly between i and j}, k = position of j in the sorted rows, B_n = O_n[rows(n)].
  An amplitude is then one NNBF pass plus O(E N) work, and a local energy needs as many NNBF passes as
  NNBF's own local energy.

Parts (all run by default)
  A  checks: (1) Sherman-Morrison frozen ratio == direct determinant of O_n[rows(m_e)] (relative 1e-10)
     on 4x2 OBC 3+3 with a random NNBF (w2_scale 0.3); (2) FRD-F == FRD-K exactly when the backflow
     output is zero (W2 = b2 = 0), where frozen and true frames coincide (1e-12).
  B  cost: per-sample local energy E_loc = diag + sum_m H_nm psi(m)/psi(n), every amplitude computed per
     configuration (no tables), L x L periodic t-J, L = 4, 5, 6, doping ~1/8 (r5 cost_measure set-up,
     base width 6, alpha width 6, random weights, 3 configurations), single thread. Timed: NNBF, FRD-K
     (r5 psi_bits, honest), FRD-F (Sherman-Morrison). Ratios T_X / T_NNBF and log-log slopes in N_e.
  C  fidelity: the trained round-5 FRD-K state (instance 0, 4x4 OBC 6+6, seed 11, shift 1e-3, the
     aggregate-selected shift) is re-read with FRD-F at the SAME parameters. Configurations are drawn
     from |Phi|^2 of the trained base (full-basis base vector, 1681680 states) and re-weighted by
     |prod F|^2, so the weights are |psi|^2 of the unprojected FRD-K state. Reported: w_flip (weight where
     prod F < 0, FRD-K's own flips; round 5 measured 2.7e-3 under |P_chi psi|^2), w_dis (weight where
     sgn prod F^F != sgn prod F), the weighted median of |log|prod F^F / prod F||, and the fraction of
     active edges with |R| > 0.5 whose frozen ratio has the other sign.

PRE-REGISTERED KILL RULES (written before the first run, 2026-11-16):
  F1 validity: check (1) above 1e-10 or check (2) above 1e-12 -> INVALID (no verdict).
  F2 Gate 4 cost: FRD-F KILLED if T_FRDF/T_NNBF at L = 6 exceeds 3, or if the log-log slope of the ratio
     T_FRDF/T_NNBF against N_e over L = 4, 5, 6 exceeds 0.5.
  F3 fidelity: FRD-F KILLED as the cost fix of FRD-K if w_dis >= w_flip (freezing changes more sign
     weight than the flips the trained state uses) or the weighted median |log ratio| exceeds 0.1.
  PASS (F2 and F3 not triggered) -> T2-b: exact budget of the FRD-F re-read at 4x4 (job t2_readout).

Usage
-----
  OMP_NUM_THREADS=4 python3 t1_frozen.py [--parts A,B,C] [--nsamp 20000] [--sizes 4,5,6]
Options
-------
  --parts LIST   subset of {A, B, C} (default all).
  --nsamp INT    part C: configurations drawn from |Phi|^2 (default 20000).
  --sizes LIST   part B: lattice sizes L (default 4,5,6; L*L <= 62).
  --nconf INT    part B: configurations per size (default 3).
  --params PATH  part C: trained FRD-K parameters (default r5 results/params/inst0_frdK_s11_sh0.001.npz).
  --out PATH     JSON output (default results/t1_frozen.json).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(os.path.dirname(HERE), "P1.2_unanchored_antisymmetry_r5")
sys.path.insert(0, R5)
from p11_core import build_lattice  # noqa: E402
from tj_core import TJ, LRAnsatz, popcount, between_mask  # noqa: E402
import p21_harness as H21  # noqa: E402
import micro_test as R5T  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.2 r6 T1 for FRD-F (see docstring)")
    ap.add_argument("--parts", default="A,B,C")
    ap.add_argument("--nsamp", type=int, default=20000)
    ap.add_argument("--sizes", default="4,5,6")
    ap.add_argument("--nconf", type=int, default=3)
    ap.add_argument("--params", default=os.path.join(R5, "results", "params", "inst0_frdK_s11_sh0.001.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "t1_frozen.json"))
    a = ap.parse_args(argv)
    a.part_list = a.parts.split(",")
    a.size_list = [int(s) for s in a.sizes.split(",")]
    return a


# ---------------------------------------------------------------------------- frozen-frame amplitudes
def orbitals(base, x):
    """Frozen orbital matrices O_u (B, V, nu), O_d (B, V, nd) of an LRAnsatz 'full', K = 1."""
    p, V, nu, nd = base.p, base.V, base.nu, base.nd
    out = np.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    Ou = out[:, :V * nu].reshape(-1, V, nu) + p["P0u"][0][None]
    Od = out[:, V * nu:V * (nu + nd)].reshape(-1, V, nd) + p["P0d"][0][None]
    return Ou, Od


def frozen_factors(frd, bu, bd, method="sm"):
    """prod_e F^F_e and the per-edge (R, active) at bit patterns bu, bd under the frozen frame."""
    base, m = frd.base, frd.m
    V, nu, nd = base.V, base.nu, base.nd
    B = len(bu)
    ou, ru = R5T.bits_to_occ(bu, V, nu)
    od, rd = R5T.bits_to_occ(bd, V, nd)
    x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
    Ou, Od = orbitals(base, x)
    bi = np.arange(B)
    Bu = Ou[bi[:, None], ru]; Bd = Od[bi[:, None], rd]
    nbu, nbd, act, hc = R5T.edge_moves_bits(bu, bd, frd.edges, m.T, m.J)
    ratio = np.zeros(act.shape)
    if method == "direct":
        du, dd = np.linalg.det(Bu), np.linalg.det(Bd)
        for e in range(frd.E):
            a = act[:, e]
            if not a.any():
                continue
            _, ru2 = R5T.bits_to_occ(nbu[a, e], V, nu)
            _, rd2 = R5T.bits_to_occ(nbd[a, e], V, nd)
            ia = np.nonzero(a)[0]
            ratio[a, e] = (np.linalg.det(Ou[ia[:, None], ru2]) * np.linalg.det(Od[ia[:, None], rd2])
                           / (du[a] * dd[a]))
    else:
        Iu = np.linalg.inv(Bu); Id = np.linalg.inv(Bd)
        one = np.int64(1)
        for e, (_, i, j, s) in enumerate(frd.edges):
            a = act[:, e]
            if not a.any():
                continue
            b = (bu if s == 0 else bd)[a]
            src = np.where((b >> j) & 1, j, i)
            dst = np.where((b >> j) & 1, i, j)
            k = popcount(b & ((one << src) - 1))
            O, I = (Ou, Iu) if s == 0 else (Od, Id)
            ia = np.nonzero(a)[0]
            perm = 1.0 - 2.0 * (popcount(b & between_mask(i, j)) & 1)
            ratio[a, e] = perm * np.einsum("bn,bn->b", O[ia, dst], I[ia, :, k])
    R = np.where(act, (hc / frd.scale) * ratio, 0.0)
    g = np.where(act, R / np.sqrt(1.0 + R * R), 0.0)   # g(R) = R / sqrt(1 + R^2), as FRD._gfun
    _, al = frd._alpha(x)
    return np.prod(1.0 + al * g, 1), R, act


def true_factors(frd, bu, bd):
    """prod_e F_e and per-edge R of FRD-K at bit patterns (neighbour amplitudes by their own NNBF pass)."""
    base, m = frd.base, frd.m
    V, nu, nd = base.V, base.nu, base.nd
    nbu, nbd, act, hc = R5T.edge_moves_bits(bu, bd, frd.edges, m.T, m.J)
    au = np.concatenate([bu, nbu[act]]); ad = np.concatenate([bd, nbd[act]])
    ou, ru = R5T.bits_to_occ(au, V, nu)
    od, rd = R5T.bits_to_occ(ad, V, nd)
    ph = base.forward(ou, od, ru, rd)["psi"]
    B = len(bu)
    R = np.zeros(act.shape)
    R[act] = (hc[act] / frd.scale) * ph[B:] / np.repeat(ph[:B], act.sum(1))
    g = np.where(act, R / np.sqrt(1.0 + R * R), 0.0)
    x = np.concatenate([ou[:B], od[:B]], 1).astype(float) * 2 - 1
    _, al = frd._alpha(x)
    return np.prod(1.0 + al * g, 1), R, act, ph[:B]


def psi_bits_frozen(frd, bu, bd):
    ou, ru = R5T.bits_to_occ(bu, frd.base.V, frd.base.nu)
    od, rd = R5T.bits_to_occ(bd, frd.base.V, frd.base.nd)
    ph = frd.base.forward(ou, od, ru, rd)["psi"]
    return ph * frozen_factors(frd, bu, bd)[0], len(bu)


# ---------------------------------------------------------------------------- parts
def part_A():
    m = TJ(4, 2, "obc", "obc", 3, 3, 1.0, -0.25, 0.4)
    refs = H21.references(m)
    base = LRAnsatz(m, refs[0], "full", K=1, Hs=3, seed=5, w2_scale=0.3)
    frd = R5T.FRD(m, base, "K", 2, 5)
    frd.q["W2a"] = np.random.default_rng(6).standard_normal(frd.q["W2a"].shape)
    idx = np.arange(m.D)
    Fs, Rs, act = frozen_factors(frd, m.bu[idx], m.bd[idx], "sm")
    Fd, Rd, _ = frozen_factors(frd, m.bu[idx], m.bd[idx], "direct")
    c1 = float(np.max(np.abs(Rs - Rd)) / np.max(np.abs(Rd)))
    base.p["W2"][:] = 0.0; base.p["b2"][:] = 0.0
    frd._phi = None
    Ff, _, _ = frozen_factors(frd, m.bu[idx], m.bd[idx], "sm")
    Ft, _, _, _ = true_factors(frd, m.bu[idx], m.bd[idx])
    c2 = float(np.max(np.abs(Ff - Ft)) / np.max(np.abs(Ft)))
    psi_tab = frd.amplitudes(4096)
    phi = frd.phi(4096)
    c3 = float(np.max(np.abs(psi_tab - phi * Ft)) / np.max(np.abs(psi_tab)))
    print(f"A: SM vs direct frozen ratio rel {c1:.2e}; FRD-F vs FRD-K at zero backflow rel {c2:.2e}; "
          f"true_factors vs r5 FRD.amplitudes rel {c3:.2e}")
    return dict(sm_vs_direct=c1, frozen_eq_true_no_backflow=c2, true_vs_r5=c3,
                valid=bool(c1 < 1e-10 and c2 < 1e-12 and c3 < 1e-10))


def part_B(args):
    rows = []
    for L in args.size_list:
        V = L * L
        nh = max(2, int(round(V / 8)))
        if (V - nh) % 2:
            nh += 1
        n = (V - nh) // 2
        T, coords, bonds = build_lattice(L, L, "pbc", "pbc", 1.0, -0.25)
        nn = sorted({(min(i, j), max(i, j)) for (i, j, _) in bonds})

        class M:
            pass
        mm = M(); mm.V, mm.nup, mm.ndn, mm.T, mm.nn, mm.J = V, n, n, T, nn, 0.4
        rng = np.random.default_rng(11 + L)
        e, v = np.linalg.eigh(-T)
        ref = (v[:, :n] + 0.1 * rng.standard_normal((V, n)), v[:, :n] + 0.1 * rng.standard_normal((V, n)))
        base = LRAnsatz(mm, ref, "full", K=1, Hs=6, seed=11, w2_scale=0.3)
        frd = R5T.FRD(mm, base, "K", 6, 11)
        frd.q["W2a"] = rng.standard_normal(frd.q["W2a"].shape)
        cfgs = [H21.random_tj_config(V, nh, rng) for _ in range(args.nconf)]
        row = dict(L=L, V=V, Ne=2 * n)
        for lab, fn in (("nnbf", R5T.base_amp_fn(base)),
                        ("frdK", lambda bu, bd: frd.psi_bits(bu, bd, T, 0.4)),
                        ("frdF", lambda bu, bd: psi_bits_frozen(frd, bu, bd))):
            ts, ev = [], []
            for ou, od in cfgs:
                fn(*[R5T.occ_to_bits(o[None]) for o in (ou, od)])   # warm-up
                t0 = time.perf_counter(); _, ne = R5T.eloc_bits(fn, T, nn, 0.4, ou, od)
                ts.append(time.perf_counter() - t0); ev.append(ne)
            row[f"T_{lab}"] = float(np.median(ts)); row[f"ev_{lab}"] = float(np.mean(ev))
        # consistency: frozen E_loc finite, base passes equal NNBF's
        row["ratio_frdK"] = row["T_frdK"] / row["T_nnbf"]
        row["ratio_frdF"] = row["T_frdF"] / row["T_nnbf"]
        rows.append(row)
        print(f"B: L={L} Ne={2 * n}: NNBF {row['ev_nnbf']:.0f} passes {row['T_nnbf']:.3e} s | FRD-K "
              f"{row['ev_frdK']:.0f} passes x{row['ratio_frdK']:.2f} | FRD-F {row['ev_frdF']:.0f} passes "
              f"x{row['ratio_frdF']:.2f}")
    ln = np.log([r["Ne"] for r in rows])
    sl = {k: float(np.polyfit(ln, np.log([r[k] for r in rows]), 1)[0])
          for k in ("ratio_frdK", "ratio_frdF", "T_nnbf", "T_frdK", "T_frdF")}
    print("B: log-log slopes in N_e: " + " ".join(f"{k}={v:.2f}" for k, v in sl.items()))
    return dict(rows=rows, slopes=sl)


def part_C(args):
    t0 = time.time()
    m = TJ(4, 4, "obc", "obc", 6, 6, 1.0, -0.25, 0.4, build_H=False)
    V = m.V
    dummy = (np.zeros((V, 6)), np.zeros((V, 6)))
    base = LRAnsatz(m, dummy, "full", K=1, Hs=4, seed=0)
    frd = R5T.FRD(m, base, "K", 3, 0)
    flat = np.load(args.params)["flat"]
    assert len(flat) == frd.n_params() == 1719, (len(flat), frd.n_params())
    frd.set_flat(flat)
    phi = base.amplitudes(65536)
    print(f"C: base vector over D={m.D} ({time.time() - t0:.0f} s)")
    rng = np.random.default_rng(2026)
    pr = phi ** 2 / (phi @ phi)
    idx = rng.choice(m.D, size=args.nsamp, p=pr)
    PT, PF, Rt, Rf, A = [], [], [], [], []
    for s in range(0, len(idx), 1000):
        ii = idx[s:s + 1000]
        ft, rt, act, ph = true_factors(frd, m.bu[ii], m.bd[ii])
        assert np.allclose(ph, phi[ii], rtol=1e-10, atol=0)
        ff, rf, _ = frozen_factors(frd, m.bu[ii], m.bd[ii], "sm")
        PT.append(ft); PF.append(ff); Rt.append(rt[act]); Rf.append(rf[act])
    PT, PF = np.concatenate(PT), np.concatenate(PF)
    Rt, Rf = np.concatenate(Rt), np.concatenate(Rf)
    w = PT ** 2 / np.sum(PT ** 2)
    w_flip = float(w[PT < 0].sum())
    w_dis = float(w[np.sign(PF) != np.sign(PT)].sum())
    lr = np.abs(np.log(np.abs(PF) / np.abs(PT)))
    o = np.argsort(lr)
    med = float(lr[o][np.searchsorted(np.cumsum(w[o]), 0.5)])
    big = np.abs(Rt) > 0.5
    edge_flip = float(np.mean(np.sign(Rf[big]) != np.sign(Rt[big]))) if big.any() else float("nan")
    rel_R = float(np.median(np.abs(Rf - Rt) / np.maximum(np.abs(Rt), 1e-12)))
    n_flip = int(np.sum(PT < 0))
    print(f"C: nsamp={args.nsamp} flipped samples {n_flip}; w_flip={w_flip:.3e} w_dis={w_dis:.3e} "
          f"weighted median |log ratio|={med:.3e}; edges |R|>0.5: {big.sum()} sign-changed frac {edge_flip:.3e}; "
          f"median rel |R^F - R|={rel_R:.3e} ({time.time() - t0:.0f} s)")
    return dict(nsamp=args.nsamp, n_flip=n_flip, w_flip=w_flip, w_dis=w_dis, med_abs_logratio=med,
                n_big_edges=int(big.sum()), big_edge_sign_change=edge_flip, median_rel_R=rel_R)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {}
    if "A" in args.part_list:
        out["A"] = part_A()
    if "B" in args.part_list:
        out["B"] = part_B(args)
    if "C" in args.part_list:
        out["C"] = part_C(args)
    if all(k in out for k in "ABC"):
        if not out["A"]["valid"]:
            v = "INVALID"
        else:
            r6 = out["B"]["rows"][-1]["ratio_frdF"]
            sl = out["B"]["slopes"]["ratio_frdF"]
            f2 = r6 > 3 or sl > 0.5
            f3 = out["C"]["w_dis"] >= out["C"]["w_flip"] or out["C"]["med_abs_logratio"] > 0.1
            v = ("KILLED (F2 cost)" if f2 else "") + (" KILLED (F3 fidelity)" if f3 else "")
            v = v.strip() or "PASS -> T2-b"
        out["verdict"] = v
        print("VERDICT FRD-F:", v)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
