#!/usr/bin/env python3
"""
P2.1 round 8b: cost per decorrelated sample of the three live routes against NNBF (critic Major 4).
===================================================================================================

Metric (pre-registered in the leaf, section "Round 8b"):
    c = t_sw * tau_int,   t_sw = wall time per Metropolis sweep (proposal ratios + accepted-move
    table updates + one local energy), tau_int = integrated autocorrelation time of E_loc in sweeps
    (Sokal automatic window, c_w = 6; error tau * sqrt(2 (2W + 1) / n)).
    R(L) = c_NNBF / c_route.  KEEP-FOR-T3 if R(16) >= 3 and R(16) > R(12) > R(8); DOWNGRADE if
    R(16) >= 1 but not growing; KILL-COST if R(16) < 1. "uncertain" if the 1-sigma interval of
    R(16) straddles 3 or 1.

Arms (T2 widths; Jastrow factors omitted in every arm):
    lh    LH-HFDS (Route 6.1'), accept_update.SHAcc with lam = 0, plus patch, F = 5
    sh    SH-HFDS (Route 7.3),  accept_update.SHAcc with lam scale 0.3, plus patch, F = 5
    hfps  LH-HFPS (Route 7.1),  accept_update.HFPSAcc (slaved stage-2 form), plus patch, F = 4
    nnbf  NNBF (Luo & Clark form of tj_core.LRAnsatz "full", K = 1) with backflow width H = 6:
          B_s(n) = Phi0_s[occ_s] + out_s(n)[occ_s],  out_s = tanh(W1^T x + b1) W2_s  (x = 2n - 1),
          psi = det B_up det B_dn. Carried: hidden pre-activation (O(H) update per move), slot-ordered
          rows, log|det| and sign. A proposal costs one forward of the occupied rows (H N^2 per spin)
          and two N x N determinants; the local energy evaluates all connections in batches.
The route engines (t1_r7.py, accept_update.py in ../P2.1_subcubic_amplitudes_r7) are imported
read-only and are not modified (jobs 698229 / 699118 import lh_train.py and t2_routes.py).

Random parameters ("realistic scale"): the route arms use the round-7 initialisation (Phi0 = lowest
orbitals of -T plus 0.1 / sqrt(V) noise; K ~ 0.5 / sqrt(V); Omega ~ 0.5 / sqrt(F n)), which puts the
backflow correction K C at about 20% of the reference entries and |E_loc| / N_e = O(1). NNBF uses the
same Phi0, W1 ~ N(0, 1 / 2V), b1 = 0, and W2 scaled so the RMS of out is 0.2 / sqrt(V), i.e. the same
20% relative backflow strength. Caveat: trained states can have different autocorrelation times.

Model: t-t'-J, t = 1, t' = -0.25, J = 0.4, L x L PBC, doping 1/8 (t1_r7.filling).
Proposal (identical for every arm, symmetric): pick a site x and one of its 8 hopping neighbours y
uniformly; hop if x is occupied and y empty; spin exchange if (x, y) is a nearest-neighbour up-down
pair; otherwise a null attempt (no amplitude evaluated). Acceptance min(1, |rho|^2). One sweep =
4 V attempts; E_loc is evaluated once per sweep. Burn-in: the first max(10, 10%) sweeps are dropped.

Options
  --arm {lh,sh,hfps,nnbf}  arm to run (one per process; run the four in parallel, single-threaded)
  --sizes LIST     PBC side lengths (default 8,12,16)
  --budgets LIST   CPU seconds per size, same length as --sizes (default 35,75,160); the sweep loop
                   stops when process CPU time spent on that size exceeds the budget
  --H INT          NNBF backflow width (default 6)
  --seed INT       RNG seed (default 11)
  --mode {run,check,aggregate}
                   run: measure t_sw and tau_int for --arm, write results/<arm>.json
                   check: NNBF fast (slot-ordered, batched) E_loc against t1_r7.eloc_brute
                          (sorted rows + explicit Jordan-Wigner signs) on 4x4 and 6x6; exit 2 on failure
                   aggregate: read results/*.json, print the table, R(L) and the verdict per route
Threads: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cap with `ulimit -t 290`.
Output: printed log; results/<arm>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r7")
sys.path.insert(0, R7)
import t1_r7 as T1  # noqa: E402
import accept_update as AU  # noqa: E402
from lh_geom import geometry  # noqa: E402

ATT_PER_SITE = 4
NAMES = {"lh": "6.1' LH-HFDS", "sh": "7.3 SH-HFDS", "hfps": "7.1 LH-HFPS", "nnbf": "NNBF (H=6)"}


# ------------------------------------------------------------------ NNBF with carried state
class NNBF:
    def __init__(self, T, n, H, rng):
        V = T.shape[0]
        self.V, self.n, self.H = V, n, H
        e, v = np.linalg.eigh(-T)
        self.P0 = [v[:, :n] + 0.1 * rng.standard_normal((V, n)) / np.sqrt(V) for _ in range(2)]
        self.W1 = rng.standard_normal((2 * V, H)) / np.sqrt(2 * V)
        self.b1 = np.zeros(H)
        # RMS of tanh(a) with a ~ N(0, 1) is about 0.63; target RMS(out) = 0.2 / sqrt(V)
        s = 0.2 / np.sqrt(V) / (0.63 * np.sqrt(H))
        self.W2 = [s * rng.standard_normal((H, V, n)) for _ in range(2)]
        self.ops = 0

    def _x(self, ou, od):
        return np.concatenate([ou, od]).astype(float) * 2 - 1

    def logdet_rows(self, h, rows_s, s):
        """h: (C, H); rows_s: (C, N) -> sign, logabs of B_s for each of the C configurations."""
        Bm = self.P0[s][rows_s] + np.einsum("ch,hcnm->cnm", h, self.W2[s][:, rows_s])
        N = rows_s.shape[1]
        self.ops += rows_s.shape[0] * (self.H * N * N + 2 * N ** 3 // 3)
        return np.linalg.slogdet(Bm)

    def amp_sorted(self, ou, od):
        """Amplitude with sorted rows (for the brute-force check)."""
        h = np.tanh(self._x(ou, od) @ self.W1 + self.b1)[None]
        out = 1.0
        for s, o in ((0, ou), (1, od)):
            sg, la = self.logdet_rows(h, np.nonzero(o)[0][None], s)
            out *= sg[0] * np.exp(la[0])
        return out

    def setup(self, ou, od):
        self.ou, self.od = ou.copy(), od.copy()
        self.a = self._x(ou, od) @ self.W1 + self.b1
        self.rows = [np.nonzero(ou)[0].copy(), np.nonzero(od)[0].copy()]
        self.slot = [{int(x): i for i, x in enumerate(r)} for r in self.rows]
        h = np.tanh(self.a)[None]
        self.sg, self.la = [], []
        for s in (0, 1):
            sg, la = self.logdet_rows(h, self.rows[s][None], s)
            self.sg.append(sg[0]); self.la.append(la[0])

    def _da(self, moves):
        """Change of the pre-activation for a move list [(s, x, y)] (site x -> y, spin s)."""
        V = self.V
        d = np.zeros(self.H)
        for (s, x, y) in moves:
            d += -2.0 * self.W1[s * V + x] + 2.0 * self.W1[s * V + y]
        return d

    def _new_rows(self, moves):
        rows = [self.rows[0].copy(), self.rows[1].copy()]
        for (s, x, y) in moves:
            rows[s][self.slot[s][x]] = y
        return rows

    def eval_batch(self, mlist):
        """In-place-row amplitude ratios for a list of move lists (batched)."""
        C = len(mlist)
        h = np.tanh(self.a[None] + np.array([self._da(m) for m in mlist]))
        logr = np.zeros(C); sgn = np.ones(C)
        self._spin = []
        for s in (0, 1):
            rows = np.empty((C, self.n), dtype=np.int64)
            for c, m in enumerate(mlist):
                r = self.rows[s].copy()
                for (ss, x, y) in m:
                    if ss == s:
                        r[self.slot[s][x]] = y
                rows[c] = r
            for i0 in range(0, C, 64):
                sg, la = self.logdet_rows(h[i0:i0 + 64], rows[i0:i0 + 64], s)
                sgn[i0:i0 + 64] *= sg * self.sg[s]
                logr[i0:i0 + 64] += la - self.la[s]
                if C == 1:
                    self._spin.append((sg[0], la[0]))
        return sgn * np.exp(logr), h

    def ratio(self, moves):
        r, h = self.eval_batch([moves])
        self._last = (moves, self._spin)
        return r[0]

    def accept(self, moves):
        """Accept the move last passed to ratio() (its per-spin determinants are reused)."""
        assert self._last[0] is moves
        self.a = self.a + self._da(moves)
        rows = self._new_rows(moves)
        for (s, x, y) in moves:
            i = self.slot[s].pop(x)
            self.slot[s][y] = i
        self.rows = rows
        for s in (0, 1):
            self.sg[s], self.la[s] = self._last[1][s]
        self.ou, self.od = T1.apply_moves(self.ou, self.od, moves)
        return 0


def eloc_nnbf(obj, T, nn):
    diag, cl = T1.conn_list(T, nn, obj.ou, obj.od)
    if not cl:
        return diag
    r, _ = obj.eval_batch([m for (_, m) in cl])
    return diag + float(np.dot([c for (c, _) in cl], r))


def eloc_route(obj, T, nn):
    diag, cl = T1.conn_list(T, nn, obj.ou, obj.od)
    return diag + sum(c * obj.ratio(m) for (c, m) in cl)


# ------------------------------------------------------------------ proposal and sweep
def neighbour_table(T):
    return [np.nonzero(T[x])[0] for x in range(T.shape[0])]


def propose(ou, od, nbr, nnset, rng):
    V = len(ou)
    x = int(rng.integers(V))
    y = int(nbr[x][rng.integers(len(nbr[x]))])
    ex, ey = ou[x] + od[x], ou[y] + od[y]
    if ex and not ey:
        return [(0 if ou[x] else 1, x, y)]
    if ex and ey and (min(x, y), max(x, y)) in nnset and ou[x] != ou[y]:
        if ou[x]:                       # up x -> y, down y -> x
            return [(0, x, y), (1, y, x)]
        return [(0, y, x), (1, x, y)]   # up y -> x, down x -> y
    return None


def sweep(obj, T, nn, nbr, nnset, rng, is_nnbf):
    V = T.shape[0]
    nprop = nacc = 0
    for _ in range(ATT_PER_SITE * V):
        mv = propose(obj.ou, obj.od, nbr, nnset, rng)
        if mv is None:
            continue
        nprop += 1
        rho = obj.ratio(mv)
        if rng.random() < min(1.0, rho * rho):
            obj.accept(mv)
            nacc += 1
    E = eloc_nnbf(obj, T, nn) if is_nnbf else eloc_route(obj, T, nn)
    return E, nprop, nacc


# ------------------------------------------------------------------ autocorrelation
def tau_int(x, cw=6.0):
    x = np.asarray(x, float)
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    W = 1
    for M in range(1, n):
        tau = 1.0 + 2.0 * acf[1:M + 1].sum()
        W = M
        if M >= cw * tau:
            break
    tau = max(tau, 0.5)
    err = tau * np.sqrt(2.0 * (2 * W + 1) / n)
    return float(tau), float(err), int(W)


# ------------------------------------------------------------------ driver
def make(arm, T, L, n, H, rng):
    if arm == "nnbf":
        return NNBF(T, n, H, rng)
    if arm == "hfps":
        return AU.HFPSAcc(T, L, L, "pbc", n, 4, "plus", rng)
    return AU.SHAcc(T, L, L, "pbc", n, 5, "plus", 0.0 if arm == "lh" else 0.3, rng)


def run(args):
    out = {"arm": args.arm, "name": NAMES[args.arm], "sizes": {}}
    for L, bud in zip(args.size_list, args.budget_list):
        rng = np.random.default_rng(args.seed + L)
        T, nn = geometry(L, L, "pbc", T1.TP)
        n = T1.filling(L * L)
        nbr = neighbour_table(T)
        nnset = set(nn)
        obj = make(args.arm, T, L, n, args.H, rng)
        ou, od = T1.rand_cfg(L * L, n, rng)
        obj.setup(ou, od)
        c0 = time.process_time()
        Es, tsw, nprops, naccs = [], [], 0, 0
        while time.process_time() - c0 < bud:
            w0 = time.perf_counter()
            E, npr, nac = sweep(obj, T, nn, nbr, nnset, rng, args.arm == "nnbf")
            tsw.append(time.perf_counter() - w0)
            Es.append(E); nprops += npr; naccs += nac
        ns = len(Es)
        burn = max(10, ns // 10)
        Ek = np.array(Es[burn:])
        tau, terr, W = tau_int(Ek) if len(Ek) > 20 else (np.nan, np.nan, 0)
        t = float(np.mean(tsw[burn:])) if ns > burn else float(np.mean(tsw))
        res = dict(L=L, Ne=2 * n, V=L * L, sweeps=ns, burn=burn, t_sw=t, tau=tau, tau_err=terr, W=W,
                   cost=t * tau, prop_per_sweep=nprops / ns, acceptance=naccs / max(nprops, 1),
                   E_mean_per_Ne=float(Ek.mean() / (2 * n)) if len(Ek) else np.nan,
                   cpu_s=time.process_time() - c0)
        out["sizes"][str(L)] = res
        print(f"[{NAMES[args.arm]}] L={L:2d} Ne={2 * n:3d} sweeps={ns:4d} (burn {burn}) t_sw={t:.3e}s "
              f"tau_int={tau:.2f}+-{terr:.2f} (W={W}) cost=t_sw*tau={t * tau:.3e}s "
              f"prop/sweep={nprops / ns:.0f} acc={naccs / max(nprops, 1):.2f} "
              f"<E_loc>/Ne={res['E_mean_per_Ne']:+.3f} cpu={res['cpu_s']:.0f}s", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{args.arm}.json"), "w") as f:
        json.dump(out, f, indent=1)


def check(args):
    ok = True
    for L in (4, 6):
        rng = np.random.default_rng(3 + L)
        T, nn = geometry(L, L, "pbc", T1.TP)
        n = T1.filling(L * L)
        obj = NNBF(T, n, args.H, rng)
        nbr = neighbour_table(T); nnset = set(nn)
        ou, od = T1.rand_cfg(L * L, n, rng)
        obj.setup(ou, od)
        for k in range(6):
            for _ in range(5 * L * L):        # move the carried state so slot order != sorted order
                mv = propose(obj.ou, obj.od, nbr, nnset, rng)
                if mv is not None and rng.random() < min(1.0, obj.ratio(mv) ** 2):
                    obj.accept(mv)
            Ef = eloc_nnbf(obj, T, nn)
            Eb = T1.eloc_brute(obj.amp_sorted, T, nn, obj.ou, obj.od)
            err = abs(Ef - Eb) / (1 + abs(Eb))
            ok &= err < 1e-9
            # carried log|det| against a fresh setup
            f = NNBF.__new__(NNBF); f.__dict__.update(obj.__dict__); f.setup(obj.ou, obj.od)
            print(f"check L={L} cfg {k}: E_loc fast={Ef:+.10f} brute={Eb:+.10f} rel.err={err:.1e} "
                  f"|carried log|psi| - fresh|={abs(sum(obj.la) - sum(f.la)):.1e}")
    # symmetric proposal: reverse move exists with the same probability
    T, nn = geometry(8, 8, "pbc", T1.TP)
    nbr = neighbour_table(T)
    ok &= all(len(v) == 8 for v in nbr) and all(x in nbr[y] for x in range(64) for y in nbr[x])
    print("proposal symmetric (8 neighbours each, symmetric adjacency):", ok)
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def aggregate(args):
    R = {}
    for a in NAMES:
        p = os.path.join(HERE, "results", f"{a}.json")
        if os.path.exists(p):
            R[a] = json.load(open(p))
    if "nnbf" not in R:
        print("INCOMPLETE: no NNBF result"); return
    sizes = sorted(int(s) for s in R["nnbf"]["sizes"])
    print("arm             L   Ne  sweeps   t_sw[s]   tau_int        cost[s]    acc")
    for a in R:
        for L in sizes:
            r = R[a]["sizes"].get(str(L))
            if r:
                print(f"{R[a]['name']:14s} {L:2d} {r['Ne']:4d} {r['sweeps']:6d} {r['t_sw']:.3e} "
                      f"{r['tau']:5.2f}+-{r['tau_err']:4.2f} {r['cost']:.3e} {r['acceptance']:.2f}")
    for a in ("lh", "sh", "hfps"):
        if a not in R:
            print(f"{NAMES[a]}: INCOMPLETE"); continue
        rs = []
        for L in sizes:
            nb, ro = R["nnbf"]["sizes"][str(L)], R[a]["sizes"][str(L)]
            Rv = nb["cost"] / ro["cost"]
            rel = np.hypot(nb["tau_err"] / nb["tau"], ro["tau_err"] / ro["tau"])
            rt = nb["t_sw"] / ro["t_sw"]
            rs.append((L, Rv, Rv * rel, rt))
        txt = "; ".join(f"L={L}: R={Rv:.2f}+-{e:.2f} (t_sw ratio {rt:.2f})" for (L, Rv, e, rt) in rs)
        R16, e16 = rs[-1][1], rs[-1][2]
        grows = all(rs[i + 1][1] > rs[i][1] for i in range(len(rs) - 1))
        if R16 >= 3 and grows:
            v = "KEEP-FOR-T3"
        elif R16 >= 1:
            v = "DOWNGRADE"
        else:
            v = "KILL-COST"
        unc = any(R16 - e16 < th <= R16 + e16 or R16 - e16 <= th < R16 + e16 for th in (1.0, 3.0))
        print(f"{NAMES[a]}: {txt} -> {v}{' (uncertain: 1-sigma interval at L=16 straddles a threshold)' if unc else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=list(NAMES), default="nnbf")
    ap.add_argument("--sizes", default="8,12,16")
    ap.add_argument("--budgets", default="35,75,160")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--mode", choices=["run", "check", "aggregate"], default="run")
    args = ap.parse_args(argv)
    args.size_list = [int(s) for s in args.sizes.split(",")]
    args.budget_list = [float(s) for s in args.budgets.split(",")]
    assert len(args.size_list) == len(args.budget_list)
    if args.mode == "check":
        sys.exit(check(args))
    if args.mode == "aggregate":
        aggregate(args)
        return
    run(args)


if __name__ == "__main__":
    main()
