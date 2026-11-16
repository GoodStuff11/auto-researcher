"""Round 11 per-leaf critic (P2.1): operation-count crossover model for the VMC sweep cost.

Purpose: back the Saturation Diagnostic with numbers. For Route 7.1 (local-feature HFPS, plus patch, F=4)
and its cubic same-frame baseline (dense HFPS, F=4), and for NNBF of width H, count the floating-point
operations of one VMC sweep with the formulas the *implemented* engines use:
  route ratio  (t1_r7.LHHFPS.ratio):  b F N + n b k + k^3   (+ b n + 2 b F + k^2 if a down particle moves)
  route accept (accept_update, slaved pairing frame): 2 n^2 k + 2 V n k + 2 k n V + V V_h k + k^3
                                                      (+ 3 n^2 + 3 V n + 2 n V + 2 V V_h if a down particle moves)
  dense ratio  (cost_8c.DenseHFPS):   F N^2 + 2 N^3 / 3
  NNBF ratio   (cost_per_sample.NNBF): 2 (H N^2 + 2 N^3 / 3)
with N = N_e / 2, n = N + V_h (V_h hidden rows; V_h = V in the implemented route), k = |R| (b changed hidden
rows, plus the up-particle row if an up particle moves). Proposal statistics (valid proposals per sweep of 4V
attempts, move types, b per move, connections per local energy) are sampled on random configurations at
doping 1/8 with the 8b proposal (cost_per_sample.propose) and t1_r7.conn_list. Acceptance a = 0.31 (8c logs).
A sweep = proposals (one ratio each) + a * proposals accepted-move updates + one local energy.

The op ratio R_ops(L) = ops_baseline / ops_route is compared with the measured sweep-time ratio of Round 8c
(results_8c/*.json): eta(L) = (t_sw ratio) / R_ops is the implementation efficiency of the route relative to the
baseline. Crossover sizes are printed for R = 3 (the 8b/8c KEEP threshold) in pure operations, and with the L=16
eta held fixed. Hypothetical reduced hidden-row densities v_h = V_h / V in {1/2, 1/4} are also costed (b scaled
by v_h, rounded up), as the 'minimal extra structure' of the diagnostic. Op counts only; no training.
Runs in under a minute, single thread. Usage: python3 crossover_model.py > crossover_model_output.txt
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r7")
R8B = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r8b")
R10 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r10")
sys.path.insert(0, R7)
sys.path.insert(0, R8B)
import t1_r7 as T1  # noqa: E402  (read-only)
import cost_per_sample as CP  # noqa: E402  (read-only)
from lh_geom import geometry, patch_index  # noqa: E402

F_ROUTE, A_ACC = 4, 0.31
LS = [8, 12, 16, 20, 24, 28, 32, 40, 48]


def stats(L, patch, rng, ncfg=20):
    T, nn = geometry(L, L, "pbc", T1.TP)
    V = L * L
    n = T1.filling(V)
    P = patch_index(L, L, "pbc", patch)
    inv = [np.nonzero((P == s).any(1))[0] for s in range(V)]
    nbr = CP.neighbour_table(T)
    nnset = set(nn)
    props, conns = [], []
    att = CP.ATT_PER_SITE * V
    for c in range(ncfg):
        ou, od = T1.rand_cfg(V, n, rng)
        for _ in range(att // ncfg + (1 if c < att % ncfg else 0)):
            mv = CP.propose(ou, od, nbr, nnset, rng)
            if mv is not None:
                props.append(mv)
        _, cl = T1.conn_list(T, nn, ou, od)
        conns.extend(m for (_, m) in cl)
    nprop = float(len(props))          # valid proposals in one sweep of 4V attempts
    nconn = len(conns) / ncfg          # connections per local energy

    def b_of(m):
        touched = [x for (_, x, _) in m] + [y for (_, _, y) in m]
        return len(np.unique(np.concatenate([inv[s] for s in touched])))
    return V, n, props, nprop, conns, nconn, b_of


def route_ops(m, b, N, V, Vh):
    n = N + Vh
    up = any(s == 0 for (s, _, _) in m)
    dn = any(s == 1 for (s, _, _) in m)
    k = b + (1 if up else 0)
    r = b * F_ROUTE * N + n * b * k + k ** 3 + ((b * n + 2 * b * F_ROUTE + k * k) if dn else 0)
    a = 2 * n * n * k + 2 * V * n * k + 2 * k * n * V + V * Vh * k + k ** 3
    if dn:
        a += 3 * n * n + 3 * V * n + 2 * n * V + 2 * V * Vh
    return r, a


def sweep_ops(L, patch, vh, rng):
    V, N, props, nprop, conns, nconn, b_of = stats(L, patch, rng)
    Vh = int(round(vh * V))
    rp = ap = 0.0
    for m in props:
        r, a = route_ops(m, int(np.ceil(b_of(m) * vh)), N, V, Vh)
        rp += r
        ap += a
    rc = sum(route_ops(m, int(np.ceil(b_of(m) * vh)), N, V, Vh)[0] for m in conns) / (len(conns) / nconn)
    route = rp + A_ACC * ap + rc
    dense = (nprop + nconn) * (F_ROUTE * N * N + 2 * N ** 3 / 3)
    nnbf = {H: (nprop + nconn) * 2 * (H * N * N + 2 * N ** 3 / 3) for H in (6, 32, 64, 128)}
    return dict(V=V, Ne=2 * N, nprop=nprop, nconn=nconn, route=route, dense=dense, nnbf=nnbf,
                parts=dict(ratio_prop=rp, accept=A_ACC * ap, eloc=rc))


def measured():
    out = {}
    for f in ("hfps", "dense", "nnbf_H6", "nnbf_H32", "nnbf_H64", "nnbf_H128"):
        with open(os.path.join(R10, "results_8c", f + ".json")) as fh:
            d = json.load(fh)
        out[f] = {int(k): v["t_sw"] for k, v in d["sizes"].items()}
    return out


def rops(s, lab):
    return (s["dense"] if lab == "dense" else s["nnbf"][lab]) / s["route"]


def main():
    rng = np.random.default_rng(11)
    meas = measured()
    print("== op counts per VMC sweep: Route 7.1 (plus patch, F=4, V_h=V) against dense HFPS (F=4) and NNBF(H)")
    res = {}
    for L in LS:
        s = sweep_ops(L, "plus", 1.0, rng)
        res[L] = s
        p = s["parts"]
        print(f"L={L:2d} Ne={s['Ne']:4d} props/sweep={s['nprop']:6.0f} conns/Eloc={s['nconn']:6.0f} "
              f"route={s['route']:.2e} [accept {p['accept'] / s['route']:.2f}, proposals "
              f"{p['ratio_prop'] / s['route']:.2f}, Eloc {p['eloc'] / s['route']:.2f}] dense={s['dense']:.2e} "
              f"R_ops(dense)={rops(s, 'dense'):.2f} "
              + " ".join(f"R_ops(nnbf{H})={rops(s, H):.2f}" for H in (6, 32, 64, 128)))
    print("== implementation efficiency eta = measured t_sw ratio (Round 8c) / R_ops")
    eta = {}
    keys = (("dense", "dense"), ("nnbf_H6", 6), ("nnbf_H32", 32), ("nnbf_H64", 64), ("nnbf_H128", 128))
    for key, lab in keys:
        parts = []
        for L in (8, 12, 16):
            rm = meas[key][L] / meas["hfps"][L]
            eta[(lab, L)] = rm / rops(res[L], lab)
            parts.append(f"L={L}: t_sw ratio={rm:.2f} R_ops={rops(res[L], lab):.2f} eta={eta[(lab, L)]:.2f}")
        print(f"{str(lab):>6}: " + "  ".join(parts))

    def cross(lab, scale):
        return next((L for L in LS if rops(res[L], lab) * scale >= 3.0), None)
    print("== smallest L in the grid with R >= 3 (pure ops; and with eta(L=16) held fixed)")
    for _, lab in keys:
        print(f"{str(lab):>6}: ops -> L={cross(lab, 1.0)}; measured constants (eta16={eta[(lab, 16)]:.2f}) -> "
              f"L={cross(lab, eta[(lab, 16)])}")
    print("== hypothetical hidden-row density v_h = V_h / V (b scaled by v_h): R_ops(dense) by L")
    for vh in (1.0, 0.5, 0.25):
        rr = [rops(sweep_ops(L, "plus", vh, rng), "dense") for L in LS]
        cr = next((L for L, r in zip(LS, rr) if r >= 3), None)
        print(f"v_h={vh:4.2f}: " + " ".join(f"L{L}:{r:.2f}" for L, r in zip(LS, rr)) + f"  -> ops crossover L={cr}")
    print("== 3x3 patch (b up to 14), same pairing-frame op model, v_h=1: R_ops(dense) by L")
    rr = [rops(sweep_ops(L, "3x3", 1.0, rng), "dense") for L in LS]
    print(" ".join(f"L{L}:{r:.2f}" for L, r in zip(LS, rr)))


if __name__ == "__main__":
    main()
