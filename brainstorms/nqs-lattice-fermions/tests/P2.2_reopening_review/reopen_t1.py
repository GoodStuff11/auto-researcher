#!/usr/bin/env python3
r"""
P2.2 reopening review (2026-11-16), tier-1 test of routes C1 (VPS, canonical basis) and C3'
(VPS in a localized Loewdin basis), with plain magnitude screening (SCREEN) as the foil.
===============================================================================================

Variational product-weighted screening (VPS).  For each dropped off-diagonal pair {n, n'} with
element h = H_{nn'} and weight w = g(n')/g(n), g(n) = exp(alpha . n_spatial), the 2x2 block
[[|h| w, -h], [-h, |h|/w]] is PSD.  So H' = H_eps + diag(D) >= H (operator order), with
D(n) = sum_dropped |h| w_{n->n'}, and <psi|H'|psi> >= <psi|H|psi> >= E_0 for every psi.
The certified excess on psi0 is b_V = sum_dropped [ |h| (psi_n^2 w + psi_n'^2 / w) - 2 h psi_n psi_n' ]
(convex in alpha; alpha is minimised at every eps).  D here uses the exact dropped set, which is
optimistic relative to the practical O(N^2) lookup bound (valid for a kill, not sufficient for a pass).

Everything is exact (ED of the full S_z = 0 sector; no Monte Carlo).  Integrals, RHF and the
Hamiltonian builder are imported unchanged from ../P2.2_stochastic_local_energy/micro_test.py
(STO-3G linear hydrogen chains).

Pre-registered rule (also in the leaf, "Reopening review (2026-11-16)", written before any run)
  Instance: H_N chains, N = 6, 8, 10; R = largest of --R-list with gap E_1 - E_0 >= 0.02 Ha at all
  sizes (asserted).  INVALID if canonical HF weight >= 0.7 at any size or frustrated fraction < 0.1.
  f = n_kept / n_EX at the largest eps on the 41-point log grid (1e-7 .. 1e-1 Ha) whose error <= tau
  (tau = 0.2 mHa).  n_kept = E_{psi0^2}[# kept off-diagonal terms in the row]; n_EX = the same count
  for all terms in the CAN basis.  Error: SCREEN |b_S| (uncertified); VPS b_V (certified).
  Per basis, better VPS arm (VPS1 or VPSg):
    KILL if f_VPS(10) >= 0.5 or f_VPS not strictly falling over N = 6, 8, 10.
    PASS if f_VPS(10) <= 0.5, strictly falling, and f_VPS(10) <= 3 f_S*(10)
         (f_S* = better SCREEN over both bases).
    AMBIGUOUS otherwise.

Resource caps (enforced before the heavy part): 4 BLAS threads; RLIMIT_CPU = min(600 s, remaining
leaf budget); leaf budget 1800 CPU-s recorded in results/cpu_ledger.json across runs.

Command-line options
--------------------
--sizes INT...      chain lengths (default 6 8 10).  Each must be even.
--R-list FLOAT...   candidate bond lengths in bohr, tried in the order given (default
                    3.0 2.8 2.6 2.4 2.2 2.0).  The first R whose gap at the largest size is
                    >= --gap-min is used; the gap is then asserted at every size.
--gap-min FLOAT     minimum E_1 - E_0 in Ha (default 0.02).
--tau FLOAT         error tolerance in Ha (default 2e-4).
--bases LIST        any of CAN (canonical RHF MOs) and LOW (Loewdin S^-1/2 site orbitals)
                    (default CAN LOW).
--no-floor          skip the E_0(H') floor diagonalisation (faster).
--R FLOAT           use this bond length and skip the R scan (the gap is still asserted per size).
--arms LIST         subset of SCREEN VPS1 VPSg to evaluate (default all three).  The verdict
                    needs SCREEN and at least one VPS arm.
--smoke             tiny run: sizes 4 6, no ledger charge beyond actual use; checks the PSD
                    bound numerically (lambda_min(H' - H) >= -1e-12) and that b_V >= 0.
--out PATH          output JSON (default results/t1_<sizes>.json next to this script).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
import argparse
import json
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results", "cpu_ledger.json")
BUDGET = 1800.0
RUN_CAP = 600.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--R-list", type=float, nargs="+", default=[3.0, 2.8, 2.6, 2.4, 2.2, 2.0])
    ap.add_argument("--gap-min", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=2e-4)
    ap.add_argument("--bases", nargs="+", default=["CAN", "LOW"], choices=["CAN", "LOW"])
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--arms", nargs="+", default=["SCREEN", "VPS1", "VPSg"],
                    choices=["SCREEN", "VPS1", "VPSg"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.smoke:
        a.sizes = [4, 6]
    assert all(n % 2 == 0 for n in a.sizes), "sizes must be even"
    return a


def cpu_used():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def ledger_load():
    try:
        with open(LEDGER) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"runs": [], "total_cpu_s": 0.0}


def ledger_add(entry):
    L = ledger_load()
    L["runs"].append(entry)
    L["total_cpu_s"] = sum(r["cpu_s"] for r in L["runs"])
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "w") as f:
        json.dump(L, f, indent=1)
    return L["total_cpu_s"]


def enforce_caps():
    used = ledger_load()["total_cpu_s"]
    cap = min(RUN_CAP, BUDGET - used)
    if cap < 60:
        sys.exit(f"ABORT: leaf CPU budget exhausted ({used:.0f} of {BUDGET:.0f} CPU-s used)")
    resource.setrlimit(resource.RLIMIT_CPU, (int(cap), int(cap) + 5))
    print(f"caps: 4 threads; this run <= {cap:.0f} CPU-s; ledger so far {used:.0f} CPU-s", flush=True)


# ---------------------------------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(HERE, "..", "P2.2_stochastic_local_energy"))
import micro_test as mt  # noqa: E402  (unchanged round-1 integrals and Hamiltonian builder)


def lowest(H, k=1):
    if H.shape[0] <= 2500:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = eigsh(H, k=k, which="SA", tol=1e-11)
    o = np.argsort(w)
    return w[o], v[:, o]


def build(n_atoms, R, basis):
    X = mt.geometry(n_atoms, R, "chain")
    S, hao, Gao, enuc = mt.ao_integrals(X)
    nocc = n_atoms // 2
    Cm, _ = mt.rhf(S, hao, Gao, nocc)
    if basis == "LOW":
        s, U = np.linalg.eigh(S)
        C = U @ np.diag(s ** -0.5) @ U.T
    else:
        C = Cm
    h = C.T @ hao @ C
    g = np.einsum("pqrs,pa,qb,rc,sd->abcd", Gao, C, C, C, C, optimize=True)
    K = n_atoms
    codes, occ, _ = mt.build_space(K, nocc, nocc)
    H = mt.build_hamiltonian(h, g, enuc, codes, occ, K)
    return H, codes, occ, K, nocc


def pair_data(H, psi, occ, K):
    U = sp.triu(H, k=1).tocoo()
    r, c, hv = U.row, U.col, U.data
    keep = np.abs(hv) > 1e-12  # symmetry zeros are not terms
    r, c, hv = r[keep], c[keep], hv[keep]
    occs = occ[:, :K].astype(np.int8) + occ[:, K:].astype(np.int8)  # spatial occupations 0..2
    d = occs[c] - occs[r]  # in -2..2
    code = ((d + 2).astype(np.int64) * (5 ** np.arange(K))).sum(1)
    uc, inv = np.unique(code, return_inverse=True)
    dmat = np.stack([(uc // 5 ** p) % 5 - 2 for p in range(K)], 1).astype(float)
    return dict(r=r, c=c, h=hv, pr=psi[r] ** 2, pc=psi[c] ** 2, x2=2 * hv * psi[r] * psi[c],
                inv=inv, dmat=dmat, occs=occs)


def vps_excess(P, mask):
    """min over alpha of sum_mask |h|(pr e^{a.d} + pc e^{-a.d}) - x2.  Returns (b_V, alpha)."""
    a_h = np.abs(P["h"][mask])
    nG = len(P["dmat"])
    C1 = np.bincount(P["inv"][mask], a_h * P["pr"][mask], minlength=nG)
    C2 = np.bincount(P["inv"][mask], a_h * P["pc"][mask], minlength=nG)
    lin = P["x2"][mask].sum()
    nz = (C1 + C2) > 0
    C1, C2, Dm = C1[nz], C2[nz], P["dmat"][nz]
    if len(C1) == 0:
        return 0.0, np.zeros(P["dmat"].shape[1])
    sc = max(C1.max(), C2.max())

    def f(a):
        z = np.clip(Dm @ a, -50, 50)
        e1, e2 = C1 / sc * np.exp(z), C2 / sc * np.exp(-z)
        return (e1 + e2).sum(), Dm.T @ (e1 - e2)

    res = minimize(f, np.zeros(Dm.shape[1]), jac=True, method="L-BFGS-B")
    return float(res.fun * sc - lin), res.x


def run_basis(n_atoms, R, basis, args, nEX_can, smoke_checks):
    t0 = time.time()
    H, codes, occ, K, nocc = build(n_atoms, R, basis)
    w, v = lowest(H, 2)
    E0, psi = w[0], v[:, 0]
    gap = w[1] - w[0]
    P = pair_data(H, psi, occ, K)
    ntot = float((P["pr"] + P["pc"]).sum())
    out = {"basis": basis, "E0": float(E0), "gap": float(gap), "n_all": ntot,
           "n_pairs": int(len(P["h"]))}
    if basis == "CAN":
        hf = sum(1 << i for i in range(nocc)) | (sum(1 << i for i in range(nocc)) << K)
        out["hf_weight"] = float(psi[np.searchsorted(codes, hf)] ** 2)
        Hc = H.tocoo(); m = Hc.row != Hc.col
        prod = Hc.data[m] * psi[Hc.row[m]] * psi[Hc.col[m]]
        out["frustrated"] = float(np.abs(prod[prod > 0]).sum() / np.abs(prod).sum())
        nEX_can = ntot
    # g0 from one-body occupations of psi0 (spatial, per spin)
    rho = np.clip((psi ** 2) @ P["occs"] / 2.0, 1e-6, 1 - 1e-6)
    a0 = 0.5 * np.log(rho / (1 - rho))
    lg = P["occs"] @ a0
    lg -= lg.max()
    keys = {"SCREEN": np.abs(P["h"]), "VPS1": np.abs(P["h"]),
            "VPSg": np.abs(P["h"]) * np.exp(lg[P["r"]] + lg[P["c"]])}
    arms = {}
    for arm, key in keys.items():
        if arm not in args.arms:
            continue
        # VPSg keys are |h| times a normalised product weight (<= 1), so they reach far below
        # 1e-7; its grid is extended to 1e-15 at the same density (post-smoke fix, logged).
        grid = np.logspace(-15, -1, 113) if arm == "VPSg" else np.logspace(-7, -1, 41)
        rows = []
        for eps in grid:
            mask = key <= eps
            nk = ntot - float((P["pr"][mask] + P["pc"][mask]).sum())
            if arm == "SCREEN":
                err, alpha = abs(float(-P["x2"][mask].sum())), None
            else:
                err, alpha = vps_excess(P, mask)
                if smoke_checks:
                    assert err >= -1e-12, f"b_V negative: {err}"
            rows.append({"eps": float(eps), "err": float(err), "f": nk / nEX_can,
                         "f_own": nk / ntot})
        ok = [i for i, rw in enumerate(rows) if rw["err"] <= args.tau]
        best = rows[max(ok)] if ok else None
        rec = {"curve": rows, "best": best}
        if arm != "SCREEN" and best is not None and (not args.no_floor or smoke_checks):
            mask = key <= best["eps"]
            _, alpha = vps_excess(P, mask)
            dvec = P["dmat"][P["inv"][mask]] @ alpha
            wgt = np.exp(np.clip(dvec, -50, 50))
            ah = np.abs(P["h"][mask])
            Dg = np.bincount(P["r"][mask], ah * wgt, minlength=H.shape[0]) + \
                np.bincount(P["c"][mask], ah / wgt, minlength=H.shape[0])
            rr, cc = P["r"][mask], P["c"][mask]
            Hdrop = sp.coo_matrix((P["h"][mask], (rr, cc)), shape=H.shape)
            Hp = (H - Hdrop - Hdrop.T + sp.diags(Dg)).tocsr()
            if smoke_checks and H.shape[0] <= 2500:
                lam = np.linalg.eigvalsh((Hp - H).toarray()).min()
                assert lam >= -1e-10, f"H' - H not PSD: {lam}"
                rec["psd_min_eig"] = float(lam)
            if not args.no_floor:
                rec["floor"] = float(lowest(Hp, 1)[0][0] - E0)
        arms[arm] = rec
        b = rec["best"]
        print(f"  H{n_atoms} {basis} {arm:6s} f={b['f'] if b else float('nan'):.4f} "
              f"(own {b['f_own'] if b else float('nan'):.4f}) eps*={b['eps'] if b else float('nan'):.2e} "
              f"err={b['err'] if b else float('nan'):.2e}"
              + (f" floor={rec['floor']:.2e}" if 'floor' in rec else ""), flush=True)
    out["arms"] = arms
    out["time_s"] = time.time() - t0
    return out, nEX_can


def verdict(res, sizes, tau):
    big = max(sizes)
    fS = min(res[str(big)][b]["arms"]["SCREEN"]["best"]["f"] for b in res[str(big)]
             if b in ("CAN", "LOW"))
    v = {}
    for b in ("CAN", "LOW"):
        if b not in res[str(big)]:
            continue
        best = None
        for arm in [a for a in ("VPS1", "VPSg") if a in res[str(big)][b]["arms"]]:
            fs = []
            for n in sizes:
                bb = res[str(n)][b]["arms"][arm]["best"]
                fs.append(bb["f"] if bb else 1.0)
            if best is None or fs[-1] < best[1][-1]:
                best = (arm, fs)
        arm, fs = best
        falling = all(fs[i + 1] < fs[i] for i in range(len(fs) - 1))
        if fs[-1] >= 0.5 or not falling:
            vv = "KILL"
        elif fs[-1] <= 3 * fS:
            vv = "PASS"
        else:
            vv = "AMBIGUOUS"
        v[b] = {"arm": arm, "f_by_size": fs, "falling": falling, "fS_star": fS, "verdict": vv}
    return v


def main(argv=None):
    args = parse_args(argv)
    enforce_caps()
    t0 = time.time()
    big = max(args.sizes)
    # choose R by the gap at the largest size (CAN basis; the spectrum is basis independent)
    R_use = args.R
    for R in ([] if R_use is not None else args.R_list):
        H, *_ = build(big, R, "CAN")
        w, _ = lowest(H, 2)
        print(f"R scan: H{big} R={R} gap={w[1]-w[0]:.4f} Ha", flush=True)
        if w[1] - w[0] >= args.gap_min:
            R_use = R
            break
    assert R_use is not None, "no R in --R-list has the required gap"
    res = {}
    for n in args.sizes:
        res[str(n)] = {}
        nEX = None
        for b in ["CAN"] + [x for x in args.bases if x != "CAN"]:
            o, nEX = run_basis(n, R_use, b, args, nEX, args.smoke)
            if b not in args.bases:
                continue
            res[str(n)][b] = o
        c = res[str(n)]["CAN"]
        print(f"H{n} R={R_use}: gap={c['gap']:.4f} HFw={c['hf_weight']:.3f} frustr={c['frustrated']:.3f}",
              flush=True)
        assert c["gap"] >= args.gap_min, f"gap assertion failed at N={n}"
    can = [res[str(n)]["CAN"] for n in args.sizes]
    invalid = any(c["hf_weight"] >= 0.7 for c in can) or any(c["frustrated"] < 0.1 for c in can)
    v = {"INVALID": True} if invalid else verdict(res, args.sizes, args.tau)
    print("VERDICT:", json.dumps(v), flush=True)
    out = args.out or os.path.join(HERE, "results", f"t1_{'_'.join(map(str, args.sizes))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "R": R_use, "results": res, "verdict": v}, f, indent=1)
    used = cpu_used()
    tot = ledger_add({"sizes": args.sizes, "smoke": args.smoke, "cpu_s": used,
                      "wall_s": time.time() - t0})
    print(f"cpu this run {used:.0f} s; ledger total {tot:.0f} of {BUDGET:.0f} CPU-s", flush=True)


if __name__ == "__main__":
    main()
