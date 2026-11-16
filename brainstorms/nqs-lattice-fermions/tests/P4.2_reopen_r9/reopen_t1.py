#!/usr/bin/env python3
r"""
P4.2 reopening review (2026-11-16), tier-1 toy runs.
=====================================================

Question: the Saturation Diagnostic of P4.2 names "site-resolved pairing" as the only large lever
(free pair matrix F_ij oracle: 1.51x over base on 4x3 OBC 5+5 at 600 SR steps). That oracle also
BROKE the mirror symmetry of the raw state (irrep weight 0.44). This script separates the two
readings and tests one new size-independent carrier of boundary-resolved pairing.

Variants (each adds ONE zero-initialised linear channel to the geminal of the base MFNG-O,
A_ij += (M theta_x)[r_up_i * N + r_dn_j], via critic_capacity.MFNGX's linear-map path, so every
variant starts from exactly the base state of the critic runs, init rng 2000):

  --variant fsym   G-SYMMETRIC site-resolved pair matrix (diagnostic, NOT size independent):
                   one parameter per orbit of ordered site pairs (i, j) under G x {swap i<->j},
                   G = {1, Mx, My, MxMy} of the open cluster. The free-F oracle restricted to
                   F_{gi,gj} = F_ij, i.e. site resolution WITHOUT symmetry breaking. On an OBC
                   cluster the only thing that breaks translation is the boundary, so any gain of
                   this channel is a boundary effect.
  --variant gf     GRAPH-FILTER pair kernel (route R9.2, SIZE INDEPENDENT): the pair matrix is a
                   polynomial matrix function of the cluster's own one-body operators,
                   F = sum_{p=0}^{P} c_p T_p(h~) + sum_{p=0}^{P} d_p (T_p(h~) D + D T_p(h~)) / 2,
                   with h~ = h / ||h||_2 the open-cluster hopping matrix (t on nn bonds, t' on
                   diagonals) scaled into [-1, 1], T_p the Chebyshev polynomials and D the d-wave
                   bond matrix (+1 on x bonds, -1 on y bonds). 2 (P+1) parameters, independent of
                   cluster size; boundary resolution comes from the missing bonds of h, not from
                   class labels. Commutes with G, so the raw state stays in the irrep.

Caps (enforced in code BEFORE the heavy part): BLAS/OpenMP threads forced to --threads (at most 4);
per-run CPU cap --cpu-cap-s (at most 600 s) turned into an SR wall-clock deadline after the setup
cost is measured; a leaf ledger (ledger.json next to this file) of CPU seconds actually used by
earlier runs; a run is refused if ledger total + per-run cap exceeds 1800 CPU-s.
Instance assert: in-irrep gap E1 - E0 >= 0.02 t (Hard-Core anti-pattern of the master doc).

Pre-registered rule (written in the leaf before the runs): see "Reopening review (2026-11-16)",
T1 kill rules K1-K3 in .subproblems/P4.2_size_transfer.md. The rule compares relerr at step 300
with the critic's base (1.909%) and free-F oracle (1.170%) trajectories at step 300
(../P4.2_size_transfer_r5/results_critic/train_base.json, train_freeF.json).

Command-line options
--------------------
  --variant STR     fsym | gf (required unless --selftest).
                      fsym : G-symmetric free pair matrix (diagnostic).
                      gf   : Chebyshev graph-filter pair kernel (route R9.2).
  --order INT       Chebyshev order P of the gf channel (default 6; ignored for fsym).
  --steps INT       SR steps (default 300).
  --threads INT     BLAS/OpenMP threads, 1 to 4 (default 1).
  --cpu-cap-s FLOAT per-run CPU cap in seconds, at most 600 (default 600).
  --out PATH        JSON output (required unless --selftest).
  --selftest        log-derivative check of both channels by central finite differences on 4x2
                    OBC 3+3 (critic_capacity's selftest protocol) and a zero-init identity check;
                    also checks that the gf matrices commute with every G permutation; exit.
  All other physics and SR settings are critic_capacity.py defaults (4x3 OBC 5+5, t'=-0.25,
  J=0.4, lr 0.05, relative shift 0.01, MFNG-O sizes 4/16/16/16), so the runs are step-matched with
  the critic's ablation.

Examples
--------
  python3 reopen_t1.py --selftest
  python3 reopen_t1.py --variant fsym --out results/fsym.json
  python3 reopen_t1.py --variant gf --out results/gf.json
"""
import argparse
import json
import os
import sys
import time


def parse_args(argv):
    p = argparse.ArgumentParser(description="P4.2 reopening review T1 (see docstring)")
    p.add_argument("--variant", choices=("fsym", "gf"), default=None)
    p.add_argument("--order", type=int, default=6)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cpu-cap-s", type=float, default=600.0)
    p.add_argument("--out", default="")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if not a.selftest and (a.variant is None or not a.out):
        p.error("--variant and --out are required unless --selftest")
    if not 1 <= a.threads <= 4:
        p.error("--threads must be 1..4")
    if a.cpu_cap_s > 600:
        p.error("--cpu-cap-s must be at most 600")
    return a


A = parse_args(sys.argv[1:])
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = str(A.threads)

import resource  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "..", "P4.2_size_transfer_r5")
sys.path.insert(0, R5)
import critic_capacity as cc  # noqa: E402  (read only; not modified)
import r3_core as rc  # noqa: E402
import obc_transfer as ot  # noqa: E402
from r5_common import optimise_ckpt  # noqa: E402

LEDGER = os.path.join(HERE, "ledger.json")
TOTAL_CAP = 1800.0


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def group_perms(lat):
    Lx, Ly = lat.Lx, lat.Ly
    s = lambda x, y: x + Lx * y
    out = []
    for fx in (0, 1):
        for fy in (0, 1):
            out.append(np.array([s(Lx - 1 - x if fx else x, Ly - 1 - y if fy else y)
                                 for y in range(Ly) for x in range(Lx)]))
    return out


def fsym_map(lat):
    N = lat.N
    G = group_perms(lat)
    key, M = {}, np.zeros((N * N, 0))
    cols = []
    for i in range(N):
        for j in range(N):
            orb = min(min((g[i], g[j]), (g[j], g[i])) for g in G)
            key.setdefault(orb, len(key))
            cols.append((i * N + j, key[orb]))
    M = np.zeros((N * N, len(key)))
    for r, c in cols:
        M[r, c] = 1.0
    return M


def gf_mats(lat, order):
    N = lat.N
    h = np.zeros((N, N))
    for i, j, tt in lat.hop:
        h[i, j] = h[j, i] = -tt
    D = np.zeros((N, N))
    for i, j, d, _, _ in lat.xb:
        D[i, j] = D[j, i] = 1.0 if d == "x" else -1.0
    hs = h / np.linalg.norm(h, 2)
    T = [np.eye(N), hs.copy()]
    while len(T) < order + 1:
        T.append(2 * hs @ T[-1] - T[-2])
    mats = T[:order + 1] + [(Tp @ D + D @ Tp) / 2 for Tp in T[:order + 1]]
    return [m / max(1e-12, np.abs(m).max()) for m in mats]


def gf_map(lat, order):
    return np.stack([m.ravel() for m in gf_mats(lat, order)], 1)


class MFNGL(cc.MFNGX):
    """Base MFNG-O plus one zero-initialised linear pair channel A_ij += (M th_Fb)[i*N+j]."""

    def __init__(self, lat, nup, M, C=4, Hf=16, Hb=16, Hj=16):
        super().__init__(lat, nup, C, Hf, Hb, Hj)
        self.bnd_f = 1
        self.Mb = M
        self.sl["Fb"] = slice(self.npar, self.npar + M.shape[1])
        self.npar += M.shape[1]


def make(lat, nup, variant, order, small=False):
    M = fsym_map(lat) if variant == "fsym" else gf_map(lat, order)
    if small:
        return MFNGL(lat, nup, M, 2, 6, 5, 4)
    return MFNGL(lat, nup, M)


def selftest(a):
    ca = cc.parse_args(["--selftest"])
    lat = ot.OBC(4, 2, ca.t, ca.tp, ca.J, ca.kq)
    basis = rc.make_basis(lat.N, 3, 3)
    ok = True
    for var in ("fsym", "gf"):
        net = make(lat, 3, var, 4, small=True)
        rng = np.random.default_rng(3)
        th = net.init(rng, 0.75, ca.t, ca.tp, ca.delta0, ca.eta)
        t0 = th.copy()
        th = th + 0.05 * rng.standard_normal(net.npar)
        st = basis[:300]
        la, _, O = net.evaluate(th, st, True)
        u = rng.standard_normal(net.npar)
        h = 1e-6
        lp, _, _ = net.evaluate(th + h * u, st, False)
        lm, _, _ = net.evaluate(th - h * u, st, False)
        an = O @ u
        fin = np.isfinite(la) & (np.abs(an) < 1e2)
        fd = (lp - lm)[fin] / (2 * h)
        err = np.max(np.abs(fd - an[fin])) / max(1e-12, np.max(np.abs(fd)))
        tb = net.base.init(np.random.default_rng(3), 0.75, ca.t, ca.tp, ca.delta0, ca.eta)
        l1, _, _ = net.base.evaluate(tb, st, False)
        l2, _, _ = net.evaluate(t0, st, False)
        same = np.allclose(l1, l2)
        good = err < 1e-4 and same
        ok &= good
        rc.log(f"selftest {var}: npar={net.npar} extra={net.Mb.shape[1]} rel grad err {err:.2e} "
               f"zero-init equals base: {same} -> {'OK' if good else 'FAIL'}")
    lat3 = ot.OBC(4, 3, ca.t, ca.tp, ca.J, ca.kq)
    G = group_perms(lat3)
    comm = max(np.abs(m[np.ix_(g, g)] - m).max() for m in gf_mats(lat3, 6) for g in G)
    fs = fsym_map(lat3)
    good = comm < 1e-12
    ok &= good
    rc.log(f"selftest 4x3: gf matrices G-invariant max dev {comm:.1e}; fsym orbits {fs.shape[1]} "
           f"(free-F oracle has {lat3.N * (lat3.N + 1) // 2}) -> {'OK' if good else 'FAIL'}")
    rc.log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


def ledger_total():
    try:
        return sum(r["cpu_s"] for r in json.load(open(LEDGER)))
    except FileNotFoundError:
        return 0.0


def run(a):
    used = ledger_total()
    if used + a.cpu_cap_s > TOTAL_CAP:
        rc.log(f"REFUSED: ledger {used:.0f} CPU-s + per-run cap {a.cpu_cap_s:.0f} > {TOTAL_CAP:.0f}")
        sys.exit(3)
    c0 = cpu_s()
    ca = cc.parse_args(["--mode", "train", "--out", "x", "--steps", str(a.steps)])
    T, tgt = cc.setup_target(ca)
    gap = tgt["gap_per_site"] * T["V"]
    assert gap >= 0.02, f"in-irrep gap {gap:.4f} t < 0.02 t"
    lat, nup = T["lat"], T["nup"]
    net = make(lat, nup, a.variant, a.order)
    th = net.init(np.random.default_rng(2000), 2 * nup / lat.N, ca.t, ca.tp, ca.delta0, ca.eta)
    cl = rc.Cluster(net, T["basis"], T["U"], T["HG"])
    setup = cpu_s() - c0
    budget = a.cpu_cap_s - setup - 30.0        # 30 s reserved for the final evaluation
    deadline = time.time() + budget / a.threads
    rc.log(f"variant={a.variant} npar={net.npar} extra={net.Mb.shape[1]} gap={gap:.4f}t setup {setup:.0f} CPU-s; "
           f"SR budget {budget:.0f} CPU-s (ledger before run {used:.0f} CPU-s)")
    th, h = optimise_ckpt(cl, th, a.steps, ca.lr, ca.diag_shift, ca.chunk, a.variant, T["E0"], T["V"],
                          deadline=deadline)
    psi = rc.state_vector(net, th, T["basis"], ca.chunk)
    b = T["U"].T @ psi
    w = float(b @ b)
    E = float(b @ (T["HG"] @ b)) / w
    E0, V = T["E0"], T["V"]
    rel = lambda e: e * V / abs(E0)
    res = dict(variant=a.variant, order=a.order, npar=net.npar, extra=int(net.Mb.shape[1]), target=tgt,
               gap=gap, steps_done=len(h) - 1, hist=h, E=E, relerr=(E - E0) / abs(E0), irrep_weight=w,
               relerr_at={k: rel(h[k]) for k in (50, 100, 150, 200, 300) if k < len(h)})
    cpu = cpu_s() - c0
    res["cpu_s"] = cpu
    try:
        L = json.load(open(LEDGER))
    except FileNotFoundError:
        L = []
    L.append(dict(variant=a.variant, out=a.out, cpu_s=cpu, threads=a.threads))
    json.dump(L, open(LEDGER, "w"), indent=1)
    rc.log(f"RESULT {a.variant}: steps={len(h) - 1} relerr={(E - E0) / abs(E0):.4%} irrep weight={w:.3f} "
           f"relerr_at={ {k: round(100 * v, 3) for k, v in res['relerr_at'].items()} } (percent) "
           f"CPU {cpu:.0f} s; ledger total {ledger_total():.0f} CPU-s")
    return res


def main():
    a = A
    if a.selftest:
        sys.exit(0 if selftest(a) else 2)
    res = run(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    rc.log(f"done ({a.out})")


if __name__ == "__main__":
    main()
