#!/usr/bin/env python3
"""
P4.1 Stage-3.5 tier-1 (T1) three-size Gate-4 trend for the route portfolio (2026-11-16).
========================================================================================

Toy ladder: open Lx x 2 strips (open along x and y), Lx = 3, 4, 5 (N = 6, 8, 10 sites), two holes
(N_e = N - 2: 2+2, 3+3, 4+4), U = 8, t' = -0.25. Hilbert dimensions 225, 3136, 44100. Every Fermi sea
is closed-shell (gaps 0.232, 0.486, 0.750), every ground state is a non-degenerate singlet in the
(Mx, My) = (+1, +1) irrep (ED via r3_micro_test.ground_state). Exact sums everywhere: every arm is
trained with the exact P_G (4-element point group) and the exact full-vector P_S, by exact-gradient
L-BFGS (r2_micro_test.optimise), exactly the protocol of job 508440 at toy size.

Arms (all exact S = 0 for every theta)
--------------------------------------
  che      R1 base (K = 1): closed-form-Che family, global charge MLP backflow, hidden width H = 4
           (fixed at every size, like the fixed-width 12-48-site ladder of job 686990).
  che_k2   R1: rank-2 mixture, fixed offsets on the shared backflow (r5_test.CheK), H = 4.
  che_k8   R1: rank-8 mixture, H = 4.
  pcg      R2: P_S of a charge-backflowed broken-symmetry PAIRING Gaussian (r2_micro_test.DetArm
           "pcg": det f_c[U, D], f_c = B1(c) f0 B2(c)^T, f0 general = singlet + triplet / AF pairing),
           MLP hidden width 4.
  che_loc  R3: UHF reference with a LOCAL, translation-equivariant charge backflow:
           Phi_s(c) = (1 + eta_s(c)) Phi_s0, eta_s(c)_ij = MLP_class(ij)(f_ij(c)) on the 3x3 mask,
           f_ij = (d_i, h_i, d_j, h_j, #holon nbrs of i, #doublon nbrs of i, same for j); one shared
           MLP (8 -> 6 -> 6 outputs = 3 bond classes x 2 spins). Its backflow parameter count is
           independent of N. Exact P_S of each c-conditioned UHF is one closed-form Che evaluation.

Pre-registered T1 rules (written into the leaf artifact before any `run` output existed)
-----------------------------------------------------------------------------------------
eps_X(N) = (E_X - E0)/|E0| at the final iteration (or at the largest common logged iteration if a run
hit its CPU cap). Premise P_T1(N): eps_che(N) >= 1e-4, otherwise size N is NO-TARGET (che near exact).
  R1 (indicator only, cannot kill an in-flight route): R(K,N) = eps_kK/eps_che.
      NON-EXTENSIVE-INDICATED if R(8,N) is non-decreasing over the valid sizes and
      R(8,10) - R(8,6) >= 0.15;  FLAT if |R(8,10) - R(8,6)| < 0.15;  otherwise MIXED.
  R2 (pcg) and R3 (che_loc), rho_X(N) = eps_X/eps_che:
      KILL  if rho_X(N) >= 1 at every valid size, or rho_X(10) - rho_X(6) >= 0.15 with rho_X(10) >= 0.9;
      PASS  if rho_X(10) <= 0.8 and rho_X(10) - rho_X(6) <= 0.0 (gain not shrinking with N);
      AMBIGUOUS otherwise.

Usage
-----
  python t1_scaling.py --mode smoke            # <S^2>, gradient FD check, single-amplitude Che check
  python t1_scaling.py --mode run --Lx 4 --arm che_loc [--seed 0] [--maxiter 300] [--threads 2]
           [--cpu-min 10]                       # one run, writes results/L{Lx}_{arm}_s{seed}.json
  python t1_scaling.py --mode report           # applies the rules above to results/
--Lx {3, 4, 5}; --arm {che, che_k2, che_k8, pcg, che_loc}; --cpu-min caps CPU minutes
(wall cap = cpu-min / threads).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "P4.1_cheap_su2_r5"))
# Thread cap BEFORE numpy loads its BLAS (the first T1 runs set it after the numpy import, so BLAS
# ignored it and the N = 10 che / che_k2 runs used about 22-24 CPU-minutes; see the leaf tier log).
_thr = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--threads"), "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _thr

import numpy as np  # noqa: E402

ARMS = ["che", "che_k2", "che_k8", "pcg", "che_loc"]
LXS = [3, 4, 5]
U_, TP = 8.0, -0.25
H_FIXED = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["smoke", "run", "report"])
    p.add_argument("--Lx", type=int, default=3, choices=LXS)
    p.add_argument("--arm", default="che", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cpu-min", type=float, default=10.0, dest="cpu_min")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    return p.parse_args(argv)


a0 = parse_args(sys.argv[1:]) if __name__ == "__main__" else None
if a0 is not None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(a0.threads)

import torch  # noqa: E402
import r5_test as R5  # noqa: E402

F, M3, R = R5.F, R5.M3, R5.R
log = R.log
torch.set_default_dtype(torch.float64)


# ============================================================================ R3 arm: local equivariant backflow
class CheLoc(R.Base):
    """phi = e^{J(c)} det[(1+eta_up(c)) Phi_up0][U] det[(1+eta_dn(c)) Phi_dn0][D]; eta local, shared MLP."""

    def __init__(self, sec, hidden=6, seed=0):
        self.sec = sec
        lat = sec.lat
        L, Nu, Nd = lat.L, sec.Nu, sec.Nd
        self.L, self.Nu, self.Nd = L, Nu, Nd
        self.Fj, feat = R.charge_features(sec)
        d, h = feat[:, :L], feat[:, L:]
        A = np.zeros((L, L))
        for i in range(L):
            A[i, lat.nn[i]] = 1.0
        hn, dn = h @ A.T, d @ A.T
        ii, jj = np.nonzero(lat.mask)
        self.ii, self.jj = ii, jj
        cls = np.where(ii == jj, 0, np.where(A[ii, jj] > 0, 1, 2))
        self.cls = torch.tensor(cls)
        f = np.stack([d[:, ii], h[:, ii], d[:, jj], h[:, jj], hn[:, ii], dn[:, ii], hn[:, jj], dn[:, jj]], -1)
        self.f = torch.tensor(f)  # (nC, npair, 8)
        self.Hl = hidden
        self.setup([("P0u", (L, Nu)), ("P0d", (L, Nd)), ("jv", (self.Fj.shape[1],)),
                    ("W1", (8, hidden)), ("b1", (hidden,)), ("W2", (hidden, 6)), ("b2", (6,))])
        Phi, _ = F.fermi_sea(sec)
        rng = np.random.default_rng(10_000 + seed)
        p = dict(P0u=Phi.copy(), P0d=Phi.copy(), jv=F.gutz(self, L),
                 W1=rng.standard_normal((8, hidden)) / np.sqrt(8), b1=np.zeros(hidden),
                 W2=1e-3 * rng.standard_normal((hidden, 6)), b2=np.zeros(6))
        self.theta0 = self.pack(p)
        self.Fj_t = torch.tensor(self.Fj)
        self.cidx = torch.tensor(sec.cidx)
        self.ru = torch.tensor(sec.up.occ[sec.iu])
        self.rd = torch.tensor(sec.dn.occ[sec.id])

    def _phi_t(self, th):
        sec, L = self.sec, self.L
        tt = torch.tensor(th, requires_grad=True)
        p, k = {}, 0
        for (name, shp), sz in zip(self.shapes, self.sizes):
            p[name] = tt[k:k + sz].reshape(shp)
            k += sz
        J = self.Fj_t @ p["jv"]
        expJ = torch.exp(J - J.max().detach())
        out = torch.tanh(self.f @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]  # (nC, npair, 6)
        nC = out.shape[0]
        orbs = []
        for s, P0 in ((0, p["P0u"]), (1, p["P0d"])):
            sel = torch.gather(out, 2, (2 * self.cls + s).view(1, -1, 1).expand(nC, -1, 1))[..., 0]
            eta = torch.zeros(nC, L, L)
            eta[:, self.ii, self.jj] = sel
            orbs.append(P0[None] + eta @ P0[None])
        Ou, Od = orbs
        c = self.cidx
        Au = Ou[c[:, None], self.ru]  # (dim, Nu, Nu)
        Ad = Od[c[:, None], self.rd]
        phi = torch.linalg.det(Au) * torch.linalg.det(Ad) * expJ[c]
        return tt, phi, Ou, Od

    def forward(self, th, chunk):
        tt, phi, _, _ = self._phi_t(th)
        return phi.detach().numpy().reshape(self.sec.shape), dict(tt=tt, phi=phi)

    def vjp(self, cc, vbar, chunk):
        (g,) = torch.autograd.grad((cc["phi"] * torch.tensor(vbar.ravel())).sum(), cc["tt"])
        return g.numpy().copy()

    def orbitals_at(self, th, cid):
        _, _, Ou, Od = self._phi_t(th)
        return Ou[cid].detach().numpy(), Od[cid].detach().numpy()


# ============================================================================ builders
def sector(Lx):
    sec = M3.make_sector(Lx, 2, 2 * Lx - 2, U_, TP, "obc", "obc")
    sg = M3.SG(sec)
    P0, E0, s2, lam, nxt = M3.ground_state(sec, sg)
    return sec, sg, P0, E0, lam


def build(sec, arm, seed):
    if arm == "che":
        return F.build_arm_fix(sec, "che", H_FIXED, seed, chunk=1 << 16)
    if arm in ("che_k2", "che_k8"):
        return R5.build_r5(sec, arm, seed, chunk=1 << 16, H=H_FIXED, width=H_FIXED)
    if arm == "pcg":
        return R.DetArm(sec, "pcg", H_FIXED, seed)
    if arm == "che_loc":
        return CheLoc(sec, 6, seed)
    raise ValueError(arm)


# ============================================================================ modes
def smoke(a):
    ok = True
    t0 = time.time()
    sec, sg, P0, E0, lam = sector(3)
    log(f"3x2 Ne=4 dim={sec.dim} E0={E0:.8f} irrep={lam}")
    for arm in ARMS:
        st = M3.State(sec, build(sec, arm, 0), True, sg, 1 << 16)
        th = st.arm.theta0 + 0.1 * np.random.default_rng(3).standard_normal(st.arm.nparam)
        E, g, psi = st.energy_grad(th)
        s2 = float(np.sum(psi * sec.S2(psi)) / np.sum(psi * psi))
        rel = M3.fd_check(st, th)
        good = abs(s2) < 1e-10 and rel < 1e-5
        ok &= good
        log(f"  {arm:8s} P={st.arm.nparam:4d} E={E:.6f} <S^2>={s2:+.1e} grad FD rel.err={rel:.1e} "
            f"[{'PASS' if good else 'FAIL'}]")
    # che_loc per-amplitude polynomial algorithm: closed-form Che on the c-conditioned orbitals
    arm = CheLoc(sec, 6, 1)
    th = arm.theta0 + 0.2 * np.random.default_rng(5).standard_normal(arm.nparam)
    p = arm.unpack(th)
    p["jv"][:] = 0
    th = arm.pack(p)
    phi, _ = arm.forward(th, 0)
    ref = sec.projS(0)(phi).ravel()
    err = 0.0
    for i in range(0, sec.dim, 7):
        U = sec.up.occ[sec.iu[i]]
        D = sec.dn.occ[sec.id[i]]
        Ou, Od = arm.orbitals_at(th, sec.cidx[i])
        nodes = M3.gl_nodes(len(U) + len(D))
        v = M3.che_single(Ou, Od, U, D, nodes)
        err = max(err, abs(v - ref[i]))
    good = err < 1e-10 * max(1.0, np.abs(ref).max())
    ok &= good
    log(f"  che_loc: |closed-form Che single amplitude - full-vector P_S| = {err:.1e} "
        f"(max|amp| {np.abs(ref).max():.2e}) [{'PASS' if good else 'FAIL'}]")
    log(f"SMOKE {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f}s)")
    return ok


def run(a):
    os.makedirs(a.outdir, exist_ok=True)
    fn = os.path.join(a.outdir, f"L{a.Lx}_{a.arm}_s{a.seed}.json")
    if os.path.exists(fn):
        log(f"exists: {fn}")
        return
    sec, sg, P0, E0, lam = sector(a.Lx)
    arm = build(sec, a.arm, a.seed)
    st = M3.State(sec, arm, True, sg, 1 << 16)
    traj = []

    def logf(msg):
        log(msg)
        mt = re.search(r"iter\s+(\d+)\s+E_best = (\S+)", msg)
        if mt:
            traj.append([int(mt.group(1)), float(mt.group(2))])
    c0 = time.process_time()
    th, E, it, sec_ = R.optimise(st, arm.theta0, a.maxiter, a.cpu_min / 60.0 / a.threads, logf=logf)
    cpu = (time.process_time() - c0) / 60.0
    _, _, psi = st.energy_grad(th)
    s2 = float(np.sum(psi * sec.S2(psi)) / np.sum(psi * psi))
    out = dict(Lx=a.Lx, N=2 * a.Lx, arm=a.arm, seed=a.seed, P=arm.nparam, E=E, E0=E0,
               eps=(E - E0) / abs(E0), iters=it, wall_s=sec_, cpu_min=cpu, S2=s2,
               capped=bool(it < a.maxiter and sec_ >= 3600 * a.cpu_min / 60.0 / a.threads * 0.999), traj=traj)
    log(json.dumps({k: v for k, v in out.items() if k != "traj"}))
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)


def eps_at(r, n):
    """eps at iteration n from the logged trajectory (n = None: final)."""
    if n is None:
        return r["eps"]
    pts = [e for (i, e) in r["traj"] if i <= n]
    return (pts[-1] - r["E0"]) / abs(r["E0"]) if pts else np.nan


def report(a):
    rs = {}
    for fn in glob.glob(os.path.join(a.outdir, "L*_s0.json")):
        r = json.load(open(fn))
        rs[(r["Lx"], r["arm"])] = r
    common = None
    if any(r["capped"] for r in rs.values()):
        common = min(r["iters"] for r in rs.values()) // 50 * 50
        print(f"some run capped: comparing at common iteration {common}")
    print(f"{'N':>3} " + " ".join(f"{k:>10s}" for k in ARMS) + "   (eps; P in brackets)")
    eps = {}
    for Lx in LXS:
        row = []
        for k in ARMS:
            r = rs.get((Lx, k))
            e = eps_at(r, common) if r else np.nan
            eps[(Lx, k)] = e
            row.append(f"{e:10.3e}" if r else f"{'-':>10s}")
        print(f"{2 * Lx:3d} " + " ".join(row) + "   [" + ", ".join(
            str(rs[(Lx, k)]["P"]) if (Lx, k) in rs else "-" for k in ARMS) + "]")
    for Lx in LXS:
        bad = [k for k in ARMS if (Lx, k) in rs and abs(rs[(Lx, k)]["S2"]) > 1e-10]
        if bad:
            print(f"N={2 * Lx}: exactness violated for {bad}")
    valid = [Lx for Lx in LXS if np.isfinite(eps[(Lx, "che")]) and eps[(Lx, "che")] >= 1e-4]
    print("premise P_T1 (eps_che >= 1e-4): valid sizes N =", [2 * Lx for Lx in valid])
    print("ratios eps_X/eps_che:")
    ratio = {}
    for k in ARMS[1:]:
        ratio[k] = {Lx: eps[(Lx, k)] / eps[(Lx, "che")] for Lx in valid}
        print(f"  {k:8s} " + "  ".join(f"N={2 * Lx}: {v:.3f}" for Lx, v in ratio[k].items()))
    if len(valid) < 2 or 3 not in valid or 5 not in valid:
        print("VERDICTS: INCOMPLETE / NO-TARGET (N = 6 and N = 10 must both be valid)")
        return
    r8 = [ratio["che_k8"][Lx] for Lx in valid]
    d8 = ratio["che_k8"][5] - ratio["che_k8"][3]
    if all(np.diff(r8) >= 0) and d8 >= 0.15:
        v1 = "NON-EXTENSIVE-INDICATED"
    elif abs(d8) < 0.15:
        v1 = "FLAT"
    else:
        v1 = "MIXED"
    print(f"R1 che_kK indicator: R(8,10) - R(8,6) = {d8:+.3f} -> {v1}")
    for k, name in (("pcg", "R2 pcg"), ("che_loc", "R3 che_loc")):
        rr = ratio[k]
        dk = rr[5] - rr[3]
        if all(v >= 1 for v in rr.values()) or (dk >= 0.15 and rr[5] >= 0.9):
            v = "KILL"
        elif rr[5] <= 0.8 and dk <= 0.0:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        print(f"{name}: rho(10) = {rr[5]:.3f}, rho(10) - rho(6) = {dk:+.3f} -> {v}")


def main():
    a = a0
    torch.set_num_threads(a.threads)
    {"smoke": smoke, "run": run, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
