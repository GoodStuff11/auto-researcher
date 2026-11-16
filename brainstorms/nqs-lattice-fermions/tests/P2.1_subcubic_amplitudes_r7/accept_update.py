#!/usr/bin/env python3
"""
P2.1 round-8 tier-1 (T1): accepted-move (Woodbury) update of the carried tables and its numerical
drift, for the three live routes (round-6 per-leaf critic, Major 4).
==================================================================================================

The per-connection O(N_e) local energy of t1_r7.py needs carried tables. After an ACCEPTED
Metropolis move they must be updated in O(n^2 b) (n = N + V augmented size, b = changed hidden rows),
not rebuilt in O(n^3), or the sweep cost is O(N_e^4). This script implements those updates on top
of the round-7 engines (t1_r7.SHHFDS / t1_r7.LHHFPS, imported, not modified) and measures drift
against a from-scratch rebuild in the SAME row/column slot order.

Updates (k = |R| = number of changed rows, R = [electron row i] + hidden rows N + B):
  Slater frame (SH-HFDS 7.3; LH-HFDS 6.1' = lam 0), per spin, rows R change by D (k x n):
      S = I + D Mi[:, R],  Z = S^-1 D Mi,  Mi <- Mi - Mi[:, R] Z,  Gr <- Gr - Gr[:, R] Z.
  Pairing frame (LH-HFPS 7.1): stage 1 (rows R, incl. the up row; W = change of the column
  table rows): Mi1 = Mi - Mi[:,R] Z, Gr1 = Gr - Gr[:,R] Z, Gc1 = Gc - Mi[:,R] (Z cols) + Mi1[:,R] W,
  X1 = X - Gr[:,R] (Z cols) + Gr1[:,R] W.  Stage 2 (column j: down x_d -> y_d), u = Gc1[:,y_d] - e_j,
  beta = Gc1[j,y_d], p = F0[:,y_d] - F0[:,x_d], q = X1[:,y_d] - X1[:,x_d]:
      Mi2 = Mi1 - u Mi1[j]/beta,  Gr2 = Gr1 + (p - q) Mi1[j]/beta,
      Gc2 = Gc1 - u Gc1[j]/beta,  X2 = X1 + (p - q) Gc1[j]/beta.
  Operation count per accepted move O(n^2 k + V n k) (+ k n V for the pairing frame): quadratic in
  N_e at fixed (F, b), so a sweep (N_e accepted moves) is O(N_e^3), the F2 floor.

Pre-registered rule (leaf artifact, Round 8, written before the run):
  * algebra: after ONE accepted move the carried tables equal a fresh rebuild to <= 1e-11
    (normalised max error); otherwise BUG (not a verdict).
  * drift metric delta = max over checkpoints (every --check accepted moves) of
    max(normalised max error of Mi, Gr[, Gc, X]; |E_loc(carried) - E_loc(fresh)| / (1 + |E_loc|)).
  * PASS if delta <= 1e-8 after >= 1e4 accepted moves with NO refresh, on every geometry.
  * Otherwise K* = accepted moves until delta first exceeds 1e-8; the refresh cost n^3 / K* stays
    below the update cost n^2 b iff K* >= n / b. KILL if K* < n_t / b at the target size
    (16x16, doping 1/8: n_t = 368, b = 10, so K* < 37); else PASS-with-refresh (every K*/2 moves).
  * cost: KILL if the slope of the operation count per accepted move in N_e exceeds 2.15.
    A timing slope below the op-count slope is overhead-bound (round-6 Minor 2), never a kill.

Options
  --route {sh,lh,hfps,all}   sh = SH-HFDS (7.3, lam 0.3), lh = LH-HFDS (6.1', lam 0), hfps = LH-HFPS (7.1)
  --geoms LIST     drift geometries: "5x3obc" and/or PBC side lengths (default 5x3obc,6,8)
  --accepts INT    accepted moves per drift chain, no refresh (default 10000)
  --check INT      checkpoint interval in accepted moves (default 1000)
  --sizes LIST     PBC L for the cost-per-accepted-move timing (default 8,16,24)
  --taccepts INT   accepted moves timed per size (default 200)
  --patch KIND     site | plus | 3x3 (default plus); --F INT (default 8); --seed INT (default 7)
  --hfps-form {slaved,feedback}  LH-HFPS stage-2 form (default slaved). "feedback" uses the
                   identities Gc[:, x_d] = e_j and X columns in place of Mi dc and Gr dc; it is
                   algebraically identical but numerically unstable (kept to reproduce the record).
Threads: OMP_NUM_THREADS / OPENBLAS_NUM_THREADS <= 4; run under `ulimit -t 600`.
Output: printed log (saved as accept_update_output.txt).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1_r7 as T1  # noqa: E402
from lh_geom import geometry  # noqa: E402

TOL_DRIFT, TOL_ALG, N_TARGET, B_TARGET, SLOPE_MAX = 1e-8, 1e-11, 368, 10, 2.15


def nerr(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-300, np.max(np.abs(b))))


# ------------------------------------------------------------------ Slater frame (6.1', 7.3)
class SHAcc(T1.SHHFDS):
    """SH-HFDS with carried slot order and accepted-move updates."""

    def build_rows(self, s, rows, H):
        V, N = self.V, len(rows)
        M = np.zeros((N + V, N + V))
        M[:N, :N] = self.P0[s][rows]
        M[:N, N:] = self.K[s][rows]
        M[N:, :N] = -np.einsum("mf,mfn->mn", H, self.Om[s])
        M[N:, N:] = np.eye(V)
        cp = np.einsum("mf,mdf->md", H, self.lam[s])
        for d in range(4):
            ok = self.nb[:, d] >= 0
            M[N + np.nonzero(ok)[0], N + self.nb[ok, d]] += cp[ok, d]
        return M

    def setup(self, ou, od, rows=None):
        self.ou, self.od = ou.copy(), od.copy()
        self.H = self.loc.feats(ou, od)
        if rows is None:
            rows = [np.nonzero(ou)[0], np.nonzero(od)[0]]
        self.rows = [np.array(r, int).copy() for r in rows]
        self.c = []
        for s in (0, 1):
            Mi = np.linalg.inv(self.build_rows(s, self.rows[s], self.H))
            allrows = np.hstack([self.P0[s], self.K[s]])
            self.c.append(dict(Mi=Mi, Gr=allrows @ Mi, N=len(self.rows[s]), allrows=allrows,
                               slot={int(x): i for i, x in enumerate(self.rows[s])}))

    def accept(self, moves):
        ou, od = T1.apply_moves(self.ou, self.od, moves)
        B = self.loc.changed([x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        Hn = self.loc.feats(ou, od, B)
        dH = Hn - self.H[B]
        b = len(B)
        mv = {s: (x, y) for (s, x, y) in moves}
        ops = 0
        for s in (0, 1):
            c = self.c[s]
            N, V = c["N"], self.V
            n = N + V
            Dh = np.zeros((b, n))
            Dh[:, :N] = -np.einsum("bf,bfn->bn", dH, self.Om[s][B])
            dcp = np.einsum("bf,bdf->bd", dH, self.lam[s][B])
            for d in range(4):
                nbB = self.nb[B, d]
                ok = nbB >= 0
                Dh[np.nonzero(ok)[0], N + nbB[ok]] += dcp[ok, d]
            if s in mv:
                x, y = mv[s]
                i = c["slot"].pop(x)
                R = np.concatenate([[i], N + B])
                D = np.vstack([c["allrows"][y] - c["allrows"][x], Dh])
                c["slot"][y] = i
                self.rows[s][i] = y
            else:
                R = N + B
                D = Dh
            k = len(R)
            MiR = c["Mi"][:, R]
            Z = np.linalg.solve(np.eye(k) + D @ MiR, D @ c["Mi"])
            c["Gr"] -= c["Gr"][:, R] @ Z
            c["Mi"] -= MiR @ Z
            ops += 2 * n * n * k + V * n * k + k ** 3
        self.H[B] = Hn
        self.ou, self.od = ou, od
        return ops

    def tables(self):
        return [(f"Mi[{s}]", self.c[s]["Mi"]) for s in (0, 1)] + [(f"Gr[{s}]", self.c[s]["Gr"]) for s in (0, 1)]

    def fresh(self):
        f = SHAcc.__new__(SHAcc)
        f.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("c", "H", "rows", "ou", "od")})
        f.setup(self.ou, self.od, self.rows)
        return f


# ------------------------------------------------------------------ pairing frame (7.1)
class HFPSAcc(T1.LHHFPS):
    """LH-HFPS with carried slot order and two-stage accepted-move updates.
    slaved = True (default): Gc and X are slaves; the stage-2 vectors u = Mi1 dc, q = Gr1 dc and
    gj = Mi1[j] allcols1 come from the primary tables. slaved = False uses the identities
    Gc1[:, x_d] = e_j and X columns instead; it is algebraically identical but feeds Gc and X errors
    back into Mi and Gr, and the first T1 run showed it is numerically unstable."""
    slaved = True

    def build_rc(self, ru, rd, H):
        V, N = self.V, len(ru)
        C = np.einsum("mf,mfv->mv", H, self.Om)
        M = np.zeros((N + V, N + V))
        M[:N, :N] = self.F0[np.ix_(ru, rd)]
        M[:N, N:] = self.K[ru]
        M[N:, :N] = -C[:, rd]
        M[N:, N:] = np.eye(V)
        return M, C

    def setup(self, ou, od, ru=None, rd=None):
        self.ou, self.od = ou.copy(), od.copy()
        self.H = self.loc.feats(ou, od)
        ru = np.nonzero(ou)[0] if ru is None else ru
        rd = np.nonzero(od)[0] if rd is None else rd
        self.ru, self.rd = np.array(ru, int).copy(), np.array(rd, int).copy()
        M, self.C = self.build_rc(self.ru, self.rd, self.H)
        self.N = len(self.ru)
        self.Mi = np.linalg.inv(M)
        allrows = np.hstack([self.F0[:, self.rd], self.K])
        allcols = np.vstack([self.F0[self.ru, :], -self.C])
        self.Gr = allrows @ self.Mi
        self.Gc = self.Mi @ allcols
        self.X = self.Gr @ allcols
        self.su = {int(x): i for i, x in enumerate(self.ru)}
        self.sd = {int(x): j for j, x in enumerate(self.rd)}

    def accept(self, moves):
        ou, od = T1.apply_moves(self.ou, self.od, moves)
        B = self.loc.changed([x for (_, x, _) in moves] + [y for (_, _, y) in moves])
        Hn = self.loc.feats(ou, od, B)
        dCB = np.einsum("bf,bfv->bv", Hn - self.H[B], self.Om[B])          # new C[B] - old C[B]
        N, V = self.N, self.V
        n = N + V
        mv = {s: (x, y) for (s, x, y) in moves}
        Rh = N + B
        allcols = np.vstack([self.F0[self.ru, :], -self.C])
        if 0 in mv:
            xu, yu = mv[0]
            i = self.su.pop(xu)
            R = np.concatenate([[i], Rh])
            W = np.vstack([self.F0[yu] - self.F0[xu], -dCB])               # k x V
            Kp = np.vstack([self.K[yu] - self.K[xu], np.zeros((len(B), V))])
        else:
            R = Rh
            W = -dCB
            Kp = np.zeros((len(B), V))
        k = len(R)
        D = np.hstack([W[:, self.rd], Kp])                                # k x n
        MiR = self.Mi[:, R]
        Z = np.linalg.solve(np.eye(k) + D @ MiR, D @ self.Mi)
        Zc = Z @ allcols
        GrR = self.Gr[:, R].copy()
        self.Mi -= MiR @ Z
        self.Gr -= GrR @ Z
        self.Gc += -MiR @ Zc + self.Mi[:, R] @ W
        self.X += -GrR @ Zc + self.Gr[:, R] @ W
        ops = 2 * n * n * k + 2 * V * n * k + 2 * k * n * V + V * V * k + k ** 3
        if 0 in mv:
            self.ru[i] = yu
            self.su[yu] = i
        self.C[B] += dCB
        if 1 in mv:
            xd, yd = mv[1]
            j = self.sd.pop(xd)
            if self.slaved:      # stable form: u, q, gj from the primary tables Mi, Gr only
                ac1 = np.vstack([self.F0[self.ru, :], -self.C])
                dc = ac1[:, yd] - ac1[:, xd]
                u = self.Mi @ dc
                q = self.Gr @ dc
                gj = self.Mi[j] @ ac1
            else:                # unstable form (Gc, X feed back into Mi, Gr): kept for the record
                u = self.Gc[:, yd].copy(); u[j] -= 1.0
                q = self.X[:, yd] - self.X[:, xd]
                gj = self.Gc[j].copy()
            beta = 1.0 + u[j]
            r = self.Mi[j].copy()
            pq = (self.F0[:, yd] - self.F0[:, xd]) - q
            self.Mi -= np.outer(u, r) / beta
            self.Gr += np.outer(pq, r) / beta
            self.Gc -= np.outer(u, gj) / beta
            self.X += np.outer(pq, gj) / beta
            self.rd[j] = yd
            self.sd[yd] = j
            ops += 3 * n * n + 3 * V * n + 2 * n * V + 2 * V * V
        self.H[B] = Hn
        self.ou, self.od = ou, od
        return ops

    def tables(self):
        return [("Mi", self.Mi), ("Gr", self.Gr), ("Gc", self.Gc), ("X", self.X)]

    def fresh(self):
        f = HFPSAcc.__new__(HFPSAcc)
        skip = ("H", "ou", "od", "ru", "rd", "C", "Mi", "Gr", "Gc", "X", "su", "sd", "N")
        f.__dict__.update({k: v for k, v in self.__dict__.items() if k not in skip})
        f.setup(self.ou, self.od, self.ru, self.rd)
        return f


# ------------------------------------------------------------------ driver
def make(route, T, Lx, Ly, bc, n, args, rng):
    if route == "hfps":
        return HFPSAcc(T, Lx, Ly, bc, n, args.F, args.patch, rng)
    return SHAcc(T, Lx, Ly, bc, n, args.F, args.patch, 0.0 if route == "lh" else 0.3, rng)


def eloc(obj, T, nn):
    diag, cl = T1.conn_list(T, nn, obj.ou, obj.od)
    return diag + sum(c * obj.ratio(m) for (c, m) in cl), cl


def compare(obj, T, nn):
    f = obj.fresh()
    e_tab = max(nerr(a, b) for (_, a), (_, b) in zip(obj.tables(), f.tables()))
    Ec, _ = eloc(obj, T, nn)
    Ef, _ = eloc(f, T, nn)
    return e_tab, abs(Ec - Ef) / (1 + abs(Ef)), Ef


def mc_step(obj, T, nn, rng):
    """One Metropolis proposal from the connection list; returns (accepted, ops)."""
    diag, cl = T1.conn_list(T, nn, obj.ou, obj.od)
    c, mv = cl[rng.integers(len(cl))]
    rho = obj.ratio(mv)
    if rng.random() < min(1.0, rho * rho):
        return True, obj.accept(mv)
    return False, 0


def parse_geom(g):
    if g == "5x3obc":
        return 5, 3, "obc", 6
    L = int(g)
    return L, L, "pbc", T1.filling(L * L)


def drift_route(route, args, rng):
    name = {"sh": "SH-HFDS (7.3, lam 0.3)", "lh": "LH-HFDS (6.1', lam 0)", "hfps": "LH-HFPS (7.1)"}[route]
    worst = 0.0
    kstar_min = np.inf
    for g in args.geom_list:
        Lx, Ly, bc, n = parse_geom(g)
        T, nn = geometry(Lx, Ly, bc, T1.TP)
        obj = make(route, T, Lx, Ly, bc, n, args, rng)
        ou, od = T1.rand_cfg(Lx * Ly, n, rng)
        obj.setup(ou, od)
        # algebra check after one accepted move
        while not mc_step(obj, T, nn, rng)[0]:
            pass
        e1, eE1, _ = compare(obj, T, nn)
        alg = "OK" if max(e1, eE1) <= TOL_ALG else "BUG"
        print(f"   [{name}] {g}: Ne={2 * n} n={n + Lx * Ly} one accept: table err {e1:.1e}, "
              f"E_loc err {eE1:.1e} -> algebra {alg}", flush=True)
        if alg == "BUG":
            return "BUG", np.nan, np.nan
        acc, prop, kstar, dmax = 1, 0, np.inf, 0.0
        t0 = time.perf_counter()
        while acc < args.accepts:
            a, _ = mc_step(obj, T, nn, rng)
            prop += 1
            acc += a
            if a and acc % args.check == 0:
                et, eE, Ef = compare(obj, T, nn)
                d = max(et, eE)
                dmax = max(dmax, d)
                if d > TOL_DRIFT and not np.isfinite(kstar):
                    kstar = acc
                print(f"      accepts={acc:6d} proposals={prop:6d} table err={et:.1e} E_loc err={eE:.1e} "
                      f"E_loc/Ne={Ef / (2 * n):+.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        print(f"   [{name}] {g}: n_aug={n + Lx * Ly} accepts={acc} "
              f"acceptance={acc / max(prop, 1):.2f} max drift={dmax:.1e} K*={kstar}", flush=True)
        worst = max(worst, dmax)
        kstar_min = min(kstar_min, kstar)
    return "OK", worst, kstar_min


def cost_route(route, args, rng):
    Ns, ts, os_ = [], [], []
    for L in args.size_list:
        T, nn = geometry(L, L, "pbc", T1.TP)
        n = T1.filling(L * L)
        obj = make(route, T, L, L, "pbc", n, args, rng)
        ou, od = T1.rand_cfg(L * L, n, rng)
        obj.setup(ou, od)
        tacc, oacc, acc = 0.0, 0, 0
        while acc < args.taccepts:
            diag, cl = T1.conn_list(T, nn, obj.ou, obj.od)
            c, mv = cl[rng.integers(len(cl))]
            rho = obj.ratio(mv)
            if rng.random() < min(1.0, rho * rho):
                t0 = time.perf_counter()
                oacc += obj.accept(mv)
                tacc += time.perf_counter() - t0
                acc += 1
        t1 = time.perf_counter(); obj.fresh(); tf = time.perf_counter() - t1
        Ns.append(2 * n); ts.append(tacc / acc); os_.append(oacc / acc)
        print(f"   L={L:2d} Ne={2 * n:4d} n={n + L * L:4d} t_accept={ts[-1]:.3e}s ops/accept={os_[-1]:.3e} "
              f"t_rebuild={tf:.3e}s (rebuild/accept = {tf / ts[-1]:.0f})", flush=True)
    st = float(np.polyfit(np.log(Ns), np.log(ts), 1)[0])
    so = float(np.polyfit(np.log(Ns), np.log(os_), 1)[0])
    return st, so


def run(route, args):
    name = {"sh": "SH-HFDS (Route 7.3)", "lh": "LH-HFDS (Route 6.1')", "hfps": "LH-HFPS (Route 7.1)"}[route]
    rng = np.random.default_rng(args.seed)
    print(f"== {name}: drift over >= {args.accepts} accepted moves, no refresh, patch={args.patch} F={args.F}",
          flush=True)
    st, worst, kstar = drift_route(route, args, rng)
    if st == "BUG":
        print(f"== {name}: BUG in the accepted-move algebra (not a verdict)")
        return "BUG"
    print(f"== {name}: cost per accepted move", flush=True)
    sl_t, sl_o = cost_route(route, args, rng)
    kmin = N_TARGET / B_TARGET
    if sl_o > SLOPE_MAX:
        v = f"KILL (op-count slope {sl_o:.2f} > {SLOPE_MAX})"
    elif worst <= TOL_DRIFT:
        v = "PASS (no refresh needed)"
    elif kstar >= kmin:
        v = f"PASS-with-refresh (K* = {kstar} >= n_t/b = {kmin:.0f}; refresh every {kstar // 2})"
    else:
        v = f"KILL (K* = {kstar} < n_t/b = {kmin:.0f})"
    note = " [timing overhead-bound]" if sl_t < sl_o - 0.2 else ""
    print(f"== {name}: max drift={worst:.1e} K*={kstar} slope(t_accept)={sl_t:.2f} "
          f"slope(ops/accept)={sl_o:.2f}{note} -> {v}", flush=True)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", choices=["sh", "lh", "hfps", "all"], default="all")
    ap.add_argument("--geoms", default="5x3obc,6,8")
    ap.add_argument("--accepts", type=int, default=10000)
    ap.add_argument("--check", type=int, default=1000)
    ap.add_argument("--sizes", default="8,16,24")
    ap.add_argument("--taccepts", type=int, default=200)
    ap.add_argument("--patch", choices=["site", "plus", "3x3"], default="plus")
    ap.add_argument("--F", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hfps-form", choices=["slaved", "feedback"], default="slaved")
    args = ap.parse_args(argv)
    HFPSAcc.slaved = args.hfps_form == "slaved"
    args.geom_list = args.geoms.split(",")
    args.size_list = [int(s) for s in args.sizes.split(",")]
    t0 = time.time()
    for r in (["hfps", "sh", "lh"] if args.route == "all" else [args.route]):
        run(r, args)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
