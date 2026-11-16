#!/usr/bin/env python3
"""
P2.1 round-8 T2 pilots for Route 7.1 (LH-HFPS, pairing frame + density Jastrow) and Route 7.3
(SH-HFDS, sparse hidden-hidden block), on instance 4 (5x3 OBC 6+6 t-t'-J), seed 11, 75 P_chi-projected
SR steps: the protocol, instance, thresholds and early stop of the Route 6.1' T2 (job 698229,
lh_train.py). The rules are in the leaf artifact, Round 8. lh_train.py and its imports are imported
read-only and are NOT modified (job 698229 uses them).
================================================================================================

Ansaetze (both have the tj_core.LRAnsatz interface, so micro_test.sr_step, the P_chi projection
and p11_core.budget() are reused unchanged):
  HFPSAnsatz (Route 7.1), N_up = N_dn = N:
      psi(n) = exp(J_d(n) + J_t(n)) det( F0[occ_up, occ_dn] + K[occ_up, :] C(n)[:, occ_dn] ),
      C[m, v] = h_m(n) . Omega[m, :, v]   (Omega: V x F x V),  h_m = tanh(W x_patch(m) + b),
      J_d = sum_{i <= j} v_ij nt_i nt_j   (two-body density Jastrow, nt = n_up + n_dn; Gate 4 of 7.1),
      J_t = w2a . tanh(W1a x + b1a)       (x = 2n - 1; width Ha fills the parameter budget).
    F0 = d-wave BCS pairing function (micro_test.references, the round-5 Foil reference),
    K ~ N(0, 1/V), Omega ~ 1e-3 N(0, 1), v = 0, w2a = 0: the start is the projected BCS state.
    Equal to the augmented determinant det[[F0[ou, od], K[ou]], [-C[:, od], I_V]] (Schur).
  SHAnsatz (Route 7.3): LH-HFDS (lh_train.LHAnsatz) plus a sparse hidden-hidden block, per spin s:
      psi = exp(J_t) prod_s det [[ Phi0_s[occ_s], K_s[occ_s, :] ], [ -C_s(n), I + L_s(n) ]],
      L_s[m, nb_d(m)] = h_m(n) . lam_{s,d}   (d = +x, -x, +y, -y; lam_{s,d} in R^F shared over
      sites, so the arm stays at the matched parameter count; lam ~ 1e-2 N(0, 1)).

Arms (P_target = parameter count of NNBF width 6 = 1626 on instance 4; matched within 2V + 2):
  route hfps:  task 0 hfps_plus (route, plus patch, F = 4)   task 1 hfps_site (foil: one-site
               features = Tocchio-Becca pairing backflow with long-range K)   task 2 agpj (reported
               only: projected BCS x tanh Jastrow, the round-5 Foil arm, micro_test ARM_DEF)
               reference: nnbf from job 698229 (--ref-results, t2_nnbf_inst4_s11.json)
  route sh:    task 0 sh_plus (route, plus patch, F = 5, Ha = 0)
               foil: the better (smaller dE_sign) of lh_plus and lh_site from job 698229;
               reference: nnbf from job 698229.
  Toy-only arms (pre-flight): the same names on instance 7; nnbf, sj, lh_plus, lh_site are read from
  lh_train's pre-flight directory (results_preflight).

Pre-registered rule (both routes; lh_train.verdict, unchanged): INVALID (calibration, multi-start
spread, reference not reproducing the pilot E(it 0), reference stopped early); KILL if the route
early-stopped at SR step 50 (E(50) - E0 > 2 (E_pilot(50) - E0)); UNINFORMATIVE if dE_sign(nnbf) <
0.005; KILL if rho_N >= 1.5 or rho_F > 0.85; PROMOTE if rho_N <= 1.15 and rho_F <= 0.85; AMBIGUOUS
otherwise. rho_N = dE_sign(route)/dE_sign(nnbf), rho_F = dE_sign(route)/dE_sign(foil).
Route sh only: INCOMPLETE if either lh foil is missing.

Modes
  --mode check       --route R: geometry asserts; amplitude == independent augmented-matrix build
                     (t1_r7.LHHFPS / t1_r7.SHHFDS) on random configurations; analytic energy
                     gradient vs central finite differences (no projection, P_chi) on toy
                     instances 9 and 8; projected per-sample O vs finite differences (instance 8);
                     fast E_loc (round-7 engine ratio x Jastrow ratio) == (H psi)/psi on EVERY
                     configuration of instance 8 (full psi, Jastrow included). Exit code 2 on failure.
  --mode train       one task: --task {0,1,2} (hfps) / {0} (sh), or --arm NAME; --instance (default 4).
                     Writes <results>/t2_<arm>_inst<I>_s<seed>.json and params/*.npy; refuses to
                     overwrite without --overwrite 1. Early stop at SR step 50 for hfps_* and sh_*
                     arms on instance 4.
  --mode aggregate   --route R: verdict from <results> and <ref-results> (instance --instance).
  --mode synthetic   --route R: verdict on synthetic results covering every branch.
  --mode preflight   --route R --part {check,synthetic,controls,inst4}:
                       check     = --mode check;  synthetic = --mode synthetic;
                       controls  = toy-instance-7 pipeline verdicts from trained toy results
                                   (train them first with --mode train --instance 7 --toy 1);
                       inst4     = instance-4 validity asserts and per-arm step timing
                                   (--preflight-steps SR steps per arm).
Options: --route {hfps,sh} (required for check/aggregate/synthetic/preflight; inferred from --arm
for train); --seed (11); --F (default 4 for hfps, 5 for sh); --results DIR (default
./results_<route>); --ref-results DIR (default ./results, job 698229); --toy-results DIR (default
./results_preflight, lh_train's toy runs); --toy {0,1} (1 = the toy settings of lh_train's
pre-flight: 400 samples, 1500 L-BFGS iterations, 4 fixed-node iterations; use with --instance 7);
--sr-iters (75) --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10
--fn-floor 1e-3 (the pilot's values); --overwrite {0,1}.
Threads: OMP_NUM_THREADS etc. (set by the sbatch file; <= 4 for local runs, under ulimit -t 600).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lh_train as LT  # noqa: E402  (read-only import; registers toy instances 7 and 8)
from lh_geom import check_geometry, neighbours4, patch_index, patch_input  # noqa: E402
import micro_test as mt  # noqa: E402
from p11_core import budget, energy, sign_constrained_min, solve_ed  # noqa: E402
from tj_core import LRAnsatz  # noqa: E402

log = LT.log
ARMS = {"hfps": ["hfps_plus", "hfps_site", "agpj"], "sh": ["sh_plus"]}
PATCH = {"hfps_plus": "plus", "hfps_site": "site", "sh_plus": "plus"}
RNG_OFFSET = {"hfps_plus": 201, "hfps_site": 202, "agpj": 203, "sh_plus": 301}
F_DEFAULT = {"hfps": 4, "sh": 5}


def route_of(arm):
    return "hfps" if arm.startswith("hfps") or arm == "agpj" else "sh"


# ============================================================================ ansaetze
class HFPSAnsatz(LRAnsatz):
    """LH-HFPS (Route 7.1); see module docstring."""

    def __init__(self, model, F0, patch, F, Ha, seed=0, om_scale=1e-3):
        rng = np.random.default_rng(seed)
        m = model
        assert m.nup == m.ndn, "the pairing frame needs N_up = N_dn"
        self.m, self.V, self.nu, self.nd = m, m.V, m.nup, m.ndn
        self.mode, self.K, self.r, self.Kref, self.trunk = "hfps", 1, 0, 1, False
        self.F, self.Ha, self.Hs = F, Ha, 0
        self.P = patch_index(m.Lx, m.Ly, (m.bcx, m.bcy), patch)
        p_sz, V = self.P.shape[1], m.V
        self.iu = np.triu_indices(V)
        p = {"F0": F0.copy(), "K": rng.standard_normal((V, V)) / np.sqrt(V),
             "Om": om_scale * rng.standard_normal((V, F, V)),
             "Wl": rng.standard_normal((2 * p_sz, F)) / np.sqrt(2 * p_sz), "bl": np.zeros(F),
             "vJ": np.zeros(len(self.iu[0]))}
        if Ha > 0:
            p["W1a"] = rng.standard_normal((2 * V, Ha)) / np.sqrt(2 * V)
            p["b1a"] = np.zeros(Ha)
            p["w2a"] = np.zeros(Ha)
        self.p = p
        self.keys = list(p.keys())

    def jastrow(self, ou, od):
        p = self.p
        nt = (ou + od).astype(float)
        Q = nt[:, self.iu[0]] * nt[:, self.iu[1]]
        J = Q @ p["vJ"]
        x = np.concatenate([ou, od], 1).astype(float) * 2 - 1
        ha = np.tanh(x @ p["W1a"] + p["b1a"]) if self.Ha > 0 else None
        if ha is not None:
            J = J + ha @ p["w2a"]
        return J, Q, x, ha

    def forward(self, ou, od, ru, rd):
        p, V = self.p, self.V
        B, N = ou.shape[0], ru.shape[1]
        xp = patch_input(ou, od, self.P)
        h = np.tanh(xp @ p["Wl"] + p["bl"])                                   # (B, V, F)
        C = np.swapaxes(np.matmul(np.swapaxes(h, 0, 1), p["Om"]), 0, 1)       # (B, V, V)
        Crd = np.take_along_axis(C, np.broadcast_to(rd[:, None, :], (B, V, N)), 2)   # (B, V, N)
        Kr = p["K"][ru]                                                       # (B, N, V)
        A = p["F0"][ru[:, :, None], rd[:, None, :]] + Kr @ Crd
        det = np.linalg.det(A)
        J, Q, x, ha = self.jastrow(ou, od)
        return dict(xp=xp, h=h, ru=ru, rd=rd, B=B, Crd=Crd, Kr=Kr, A=A, Q=Q, x=x, ha=ha, J=J, S=det,
                    psi=np.exp(J) * det)

    def grad(self, c, coef, per_sample):
        p, B, V = self.p, c["B"], self.V
        ru, rd = c["ru"], c["rd"]
        N = ru.shape[1]
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        tk = coef * c["psi"]
        G = tk[:, None, None] * self._invT(c["A"])                             # (B, N, N)
        gK = G @ np.swapaxes(c["Crd"], 1, 2)                                   # (B, N, V)
        dCrd = np.swapaxes(c["Kr"], 1, 2) @ G                                  # (B, V, N)
        dC = np.zeros((B, V, V))
        dC[bi, :, rd] = np.swapaxes(dCrd, 1, 2)
        h = c["h"]
        g = {}
        if per_sample:
            Z = np.zeros((B, V, V)); Z[bi[:, :, None], ru[:, :, None], rd[:, None, :]] = G
            ZK = np.zeros((B, V, V)); ZK[bi, ru] = gK
            g["F0"], g["K"] = Z, ZK
            g["Om"] = np.einsum("bmf,bmv->bmfv", h, dC)
        else:
            fi = (ru[:, :, None] * V + rd[:, None, :]).ravel()
            g["F0"] = np.bincount(fi, weights=G.ravel(), minlength=V * V).reshape(V, V)
            Sc = sp.csr_matrix((np.ones(ru.size), (ru.ravel(), np.arange(ru.size))), shape=(V, ru.size))
            g["K"] = Sc @ gK.reshape(-1, V)
            g["Om"] = np.matmul(np.swapaxes(np.swapaxes(h, 0, 1), 1, 2), np.swapaxes(dC, 0, 1))   # (V, F, V)
        dh = np.swapaxes(np.matmul(np.swapaxes(dC, 0, 1), np.swapaxes(p["Om"], 1, 2)), 0, 1)    # (B, V, F)
        dpre = dh * (1 - h ** 2)
        if per_sample:
            g["Wl"] = np.einsum("bvi,bvf->bif", c["xp"], dpre)
            g["bl"] = dpre.sum(1)
        else:
            g["Wl"] = np.einsum("bvi,bvf->if", c["xp"], dpre)
            g["bl"] = dpre.sum((0, 1))
        g["vJ"] = tk[:, None] * c["Q"] if per_sample else c["Q"].T @ tk
        if self.Ha > 0:
            x, ha = c["x"], c["ha"]
            dprea = tk[:, None] * p["w2a"][None] * (1 - ha ** 2)
            g["W1a"] = (x[:, :, None] * dprea[:, None, :]) if per_sample else x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(tk[:, None] * ha)
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


class SHAnsatz(LT.LHAnsatz):
    """SH-HFDS (Route 7.3); see module docstring."""

    def __init__(self, model, ref, patch, F, Ha, seed=0, om_scale=1e-3, lam_scale=1e-2):
        super().__init__(model, ref, patch, F, Ha, seed=seed, om_scale=om_scale)
        rng = np.random.default_rng(seed + 7)
        m = model
        self.mode = "sh"
        self.nb = neighbours4(m.Lx, m.Ly, (m.bcx, m.bcy))
        self.nbm = [np.nonzero(self.nb[:, d] >= 0)[0] for d in range(4)]
        self.p["lamu"] = lam_scale * rng.standard_normal((4, F))
        self.p["lamd"] = lam_scale * rng.standard_normal((4, F))
        self.keys = list(self.p.keys())

    def forward(self, ou, od, ru, rd):
        p, V = self.p, self.V
        B = ou.shape[0]
        xp = patch_input(ou, od, self.P)
        h = np.tanh(xp @ p["Wl"] + p["bl"])
        c = dict(xp=xp, h=h, ru=ru, rd=rd, B=B, x=np.concatenate([ou, od], 1).astype(float) * 2 - 1)
        det = np.ones(B)
        for s, rows in (("u", ru), ("d", rd)):
            N = rows.shape[1]
            n = N + V
            C = np.swapaxes(np.matmul(np.swapaxes(h, 0, 1), p["Om" + s]), 0, 1)   # (B, V, N)
            M = np.zeros((B, n, n))
            M[:, :N, :N] = p["P0" + s][rows]
            M[:, :N, N:] = p["K" + s][rows]
            M[:, N:, :N] = -C
            M[:, N + np.arange(V), N + np.arange(V)] = 1.0
            Lv = h @ p["lam" + s].T                                               # (B, V, 4)
            for d in range(4):
                ms = self.nbm[d]
                M[:, N + ms, N + self.nb[ms, d]] += Lv[:, ms, d]
            c["M" + s] = M
            det = det * np.linalg.det(M)
        if self.Ha > 0:
            ha = np.tanh(c["x"] @ p["W1a"] + p["b1a"])
            J = ha @ p["w2a"]
            c["ha"] = ha
        else:
            J = np.zeros(B)
        c["J"], c["S"] = J, det
        c["psi"] = np.exp(J) * det
        return c

    def grad(self, c, coef, per_sample):
        p, B, V, F = self.p, c["B"], self.V, self.F
        bi = np.arange(B)[:, None]
        red = (lambda a: a) if per_sample else (lambda a: a.sum(0))
        tk = coef * c["psi"]
        h = c["h"]
        g = {}
        dh = np.zeros((B, V, F))
        for s, rows in (("u", c["ru"]), ("d", c["rd"])):
            N = rows.shape[1]
            G = tk[:, None, None] * self._invT(c["M" + s])                       # (B, n, n)
            gP, gK = G[:, :N, :N], G[:, :N, N:]
            dC = -G[:, N:, :N]                                                   # (B, V, N)
            if per_sample:
                Z = np.zeros((B, V, N)); Z[bi, rows] = gP
                ZK = np.zeros((B, V, V)); ZK[bi, rows] = gK
                g["P0" + s], g["K" + s] = Z, ZK
                g["Om" + s] = np.einsum("bvf,bvn->bvfn", h, dC)
            else:
                Sc = sp.csr_matrix((np.ones(rows.size), (rows.ravel(), np.arange(rows.size))), shape=(V, rows.size))
                g["P0" + s] = Sc @ gP.reshape(-1, N)
                g["K" + s] = Sc @ gK.reshape(-1, V)
                g["Om" + s] = np.matmul(np.swapaxes(np.swapaxes(h, 0, 1), 1, 2), np.swapaxes(dC, 0, 1))
            dh += np.swapaxes(np.matmul(np.swapaxes(dC, 0, 1), np.swapaxes(p["Om" + s], 1, 2)), 0, 1)
            gl = np.zeros((B, 4, F))
            lam = p["lam" + s]
            for d in range(4):
                ms = self.nbm[d]
                dL = G[:, N + ms, N + self.nb[ms, d]]                               # (B, |ms|)
                gl[:, d] = np.einsum("bm,bmf->bf", dL, h[:, ms])
                dh[:, ms] += dL[:, :, None] * lam[d][None, None, :]
            g["lam" + s] = red(gl)
        dpre = dh * (1 - h ** 2)
        if per_sample:
            g["Wl"] = np.einsum("bvi,bvf->bif", c["xp"], dpre)
            g["bl"] = dpre.sum(1)
        else:
            g["Wl"] = np.einsum("bvi,bvf->if", c["xp"], dpre)
            g["bl"] = dpre.sum((0, 1))
        if self.Ha > 0:
            x, ha = c["x"], c["ha"]
            dprea = tk[:, None] * p["w2a"][None] * (1 - ha ** 2)
            g["W1a"] = (x[:, :, None] * dprea[:, None, :]) if per_sample else x.T @ dprea
            g["b1a"] = red(dprea)
            g["w2a"] = red(tk[:, None] * ha)
        if per_sample:
            return np.concatenate([g[k].reshape(B, -1) for k in self.keys], 1)
        return np.concatenate([g[k].ravel() for k in self.keys])


def hfps_count(m, patch, F, Ha):
    V, p_sz = m.V, patch_index(m.Lx, m.Ly, "obc", patch).shape[1]
    return 2 * V * V + V * F * V + 2 * p_sz * F + F + V * (V + 1) // 2 + (2 * V + 2) * Ha


def sh_count(m, patch, F, Ha):
    return LT.lh_count(m, patch, F, Ha) + 8 * F


def make_arm(arm, m, refs, seed, F, match_width=6):
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=match_width).n_params()
    if arm in LT.ARMS or arm == "sj":
        return LT.make_arm(arm, m, refs, seed, 5)
    if arm == "agpj":
        ns = SimpleNamespace(match_width=match_width, hs_sign=6, nnbfj_width=4, K=4)
        kw, _ = mt.arm_kwargs("agpj", m, ns, refs)
        return LRAnsatz(m, refs[1], seed=seed, **kw), kw, target
    patch = PATCH[arm]
    if arm.startswith("hfps"):
        Ha = max(0, int(round((target - hfps_count(m, patch, F, 0)) / (2 * m.V + 2))))
        a = HFPSAnsatz(m, refs[1], patch, F, Ha, seed=seed)
        assert a.n_params() == hfps_count(m, patch, F, Ha)
        return a, dict(mode="hfps", patch=patch, F=F, Ha=Ha), target
    Ha = max(0, int(round((target - sh_count(m, patch, F, 0)) / (2 * m.V + 2))))
    a = SHAnsatz(m, refs[0], patch, F, Ha, seed=seed)
    assert a.n_params() == sh_count(m, patch, F, Ha)
    return a, dict(mode="sh", patch=patch, F=F, Ha=Ha), target


# ============================================================================ checks
def engine_for(a, m, rng):
    """Independent round-7 augmented-matrix engine carrying the ansatz's weights."""
    import t1_r7
    if isinstance(a, HFPSAnsatz):
        o = t1_r7.LHHFPS.__new__(t1_r7.LHHFPS)
        o.V, o.n, o.F, o.ops = m.V, m.nup, a.F, 0
        o.F0, o.K, o.Om = a.p["F0"], a.p["K"], a.p["Om"]
    else:
        o = t1_r7.SHHFDS.__new__(t1_r7.SHHFDS)
        o.V, o.n, o.F, o.ops = m.V, m.nup, a.F, 0
        o.nb = neighbours4(m.Lx, m.Ly, (m.bcx, m.bcy))
        o.P0, o.K, o.Om = [a.p["P0u"], a.p["P0d"]], [a.p["Ku"], a.p["Kd"]], [a.p["Omu"], a.p["Omd"]]
        o.lam = [np.repeat(a.p["lam" + s][None], m.V, 0) for s in ("u", "d")]
    o.loc = t1_r7.Local(a.P, a.F, rng)
    o.loc.W, o.loc.b = a.p["Wl"], a.p["bl"]
    return o


def jastrow_single(a, ou, od):
    if isinstance(a, HFPSAnsatz):
        return float(a.jastrow(ou[None], od[None])[0][0])
    if a.Ha > 0:
        x = np.concatenate([ou, od]).astype(float) * 2 - 1
        return float(np.tanh(x @ a.p["W1a"] + a.p["b1a"]) @ a.p["w2a"])
    return 0.0


def run_check(route, seed=5):
    import t1_r7
    ok = check_geometry()
    log(f"check geometry (OBC patch indexing): {ok}")
    rng = np.random.default_rng(seed)
    arms = ["hfps_plus", "hfps_site"] if route == "hfps" else ["sh_plus"]
    for inst in (9, 8):
        m = mt.make_model(inst)
        refs = mt.references(m)
        W, Vv, deg = solve_ed(m, 2, 1e-8, 1234)
        sg, gen = mt.build_symmetry(m, mt.resolve_ground(m, Vv, deg))
        for arm in arms:
            a, kw, tgt = make_arm(arm, m, refs, seed, 3)
            x = a.flat(); a.set_flat(x + 0.05 * rng.standard_normal(len(x)))   # generic point (J, lam != 0)
            o = engine_for(a, m, rng)
            idx = rng.choice(m.D, size=min(40, m.D), replace=False)
            ou, od, ru, rd = a._occ(idx)
            ps = a.forward(ou, od, ru, rd)["S"]
            pb = np.array([o.amp(ou[k], od[k]) for k in range(len(idx))])
            e_amp = float(np.max(np.abs(ps - pb)) / np.max(np.abs(pb)))
            log(f"check amplitude [{arm} inst {inst} {kw}] ansatz vs augmented det: {e_amp:.1e}")
            ok &= e_amp < 1e-12
            for proj in ("none", "chi"):
                ok &= LT.fd_check(a, m, sg, proj, rng, f"{arm} inst {inst} Ha={a.Ha}")
            if inst != 8:
                continue
            psi_c = sg.project(a.amplitudes(4096))
            ii = rng.choice(m.D, size=6, replace=False)
            O = a.per_sample_O_proj(sg, ii, psi_c)
            d = rng.standard_normal(a.n_params()); x0 = a.flat().copy(); eps = 1e-6
            a.set_flat(x0 + eps * d); pp = sg.project(a.amplitudes(4096))[ii]
            a.set_flat(x0 - eps * d); pm = sg.project(a.amplitudes(4096))[ii]
            a.set_flat(x0)
            e_O = float(np.max(np.abs(O @ d - (pp - pm) / (2 * eps) / psi_c[ii])))
            log(f"check O_P [{arm}] vs fd: {e_O:.1e}")
            ok &= e_O < 1e-5
            # fast E_loc (engine ratio x Jastrow ratio) == (H psi)/psi, full psi, every configuration
            psi = a.amplitudes(4096)
            Hp = m.H @ psi
            emax, nskip = 0.0, 0
            for k in range(m.D):
                if abs(psi[k]) < 1e-8 * np.max(np.abs(psi)):
                    nskip += 1
                    continue
                oc = a._occ(np.array([k]))
                u, d_ = oc[0][0].astype(int), oc[1][0].astype(int)
                o.setup(u, d_)
                J0 = jastrow_single(a, u, d_)
                diag, cl = t1_r7.conn_list(m.T, m.nn, u, d_)
                Ef = diag
                for (cc, mm) in cl:
                    u2, d2 = t1_r7.apply_moves(u, d_, mm)
                    Ef += cc * o.ratio(mm) * np.exp(jastrow_single(a, u2, d2) - J0)
                emax = max(emax, abs(Ef - Hp[k] / psi[k]) / (1 + abs(Hp[k] / psi[k])))
            log(f"check fast E_loc == (H psi)/psi on all {m.D} configs ({nskip} near-node skipped) "
                f"[{arm}, OBC 3x3, full psi incl. Jastrow]: {emax:.1e}")
            ok &= emax < 1e-9
    log("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    return bool(ok)


# ============================================================================ train
def train(args):
    m = mt.make_model(args.instance)
    m.spin_setup(); m.pair_setup(args.r_min)
    log(f"instance {args.instance} {mt.INSTANCES[args.instance]} D={m.D} arm={args.arm} seed={args.seed}")
    W, Vv, deg = solve_ed(m, 4, 1e-8, mt.UHF_SEED)
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    dim_chi, dim_S_chi = mt.sector_dims(m, sg, S0)
    assert S0 == 0.0, "the projected SR here assumes a singlet ground state"
    pil = LT.PILOT.get(args.instance)
    if pil is not None:
        assert abs(W[0] - pil["E0"]) < 1e-8, (W[0], pil["E0"])
        assert dim_S_chi == pil["dim_S_chi"] and sg.G == pil["G"], (dim_S_chi, sg.G)
    log(f"E0={W[0]:.10f} deg={deg} |G|={sg.G} chi={gen} dim_S_chi={dim_S_chi:.0f}")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = mt.references(m)
    a, kw, target = make_arm(args.arm, m, refs, args.seed, args.F)
    if pil is not None:
        assert abs(a.n_params() - target) <= 2 * m.V + 2, (a.n_params(), target)
    log(f"arm {args.arm}: {kw} n_params={a.n_params()} target={target} calib={calib:.2e}")
    rng = np.random.default_rng(args.seed + 1000 * RNG_OFFSET.get(args.arm, LT.RNG_OFFSET.get(args.arm, 0)))
    early = None
    if args.arm in PATCH and pil is not None:
        early = dict(E0=pil["E0"], E_it50=pil["E_it50"])
    traj, stop, sps = LT.optimize_traj(a, m, sg, args, rng, args.arm, early)
    psi_t, psi_c, psi = mt.state_of(a, m, sg, "chi", S0, args.chunk)
    budgets = {}
    for lab, v in (("P", psi_c), ("P_S", m.spin_project(psi_c, S0))):
        t2 = time.time()
        b = budget(m, v, W, psi0[:, None] if deg > 1 else Vv, 1 if deg > 1 else deg, args, rng_b)
        vn = v / np.linalg.norm(v)
        b.update(S2=m.S2_expect(vn), ov0=float((psi0 @ vn) ** 2))
        for k_ in ("P0", "Ptheta", "Psign", "fn_history", "sign_starts"):
            b.pop(k_, None)
        budgets[lab] = b
        log(f"{args.arm} [{lab}]: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} f_E={b['f_E_up']:.3f} "
            f"spread={b['spread']:.1e} ov0={b['ov0']:.3f} ({time.time() - t2:.0f} s)")
    pdir = os.path.join(args.results, "params"); os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, f"t2_{args.arm}_inst{args.instance}_s{args.seed}.npy"), a.flat())
    return dict(instance=args.instance, arm=args.arm, seed=args.seed, arm_kwargs=kw, n_params=a.n_params(),
                P_target=target, E0=float(W[0]), calib_err=calib, G=sg.G, dim_S_chi=dim_S_chi, irrep=gen,
                traj=traj, steps_done=len(traj), early_stop=stop, s_per_step=sps,
                threads=os.environ.get("OMP_NUM_THREADS", "?"), budgets=budgets,
                params={k: v for k, v in vars(args).items()})


# ============================================================================ verdict
def decide(route, R, pilot_check=True):
    """Pre-registered rule (Round 8). R: dict arm -> result dict or None. Returns (verdict, why, nums)."""
    if route == "hfps":
        return LT.verdict(R.get("hfps_plus"), R.get("nnbf"), R.get("hfps_site"), pilot_check)
    lp, ls = R.get("lh_plus"), R.get("lh_site")
    if lp is None or ls is None:
        return "INCOMPLETE", ["an lh foil (job 698229) is missing"], {}
    ds = {k: R[k]["budgets"].get("P", {}).get("dE_sign", np.nan) for k in ("lh_plus", "lh_site")}
    if not all(np.isfinite(v) for v in ds.values()):
        return "INVALID", ["an lh foil has no P budget"], {}
    foil = min(ds, key=ds.get)
    v, why, nums = LT.verdict(R.get("sh_plus"), R.get("nnbf"), R[foil], pilot_check)
    return v, why + [f"foil = {foil} (dE_sign {ds[foil]:.5f})"], nums


def load(d, arm, inst, seed):
    f = os.path.join(d, f"t2_{arm}_inst{inst}_s{seed}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def collect(args):
    R = {arm: load(args.results, arm, args.instance, args.seed) for arm in ARMS[args.route]}
    for arm in ("nnbf", "lh_plus", "lh_site"):
        R[arm] = load(args.ref_results, arm, args.instance, args.seed)
    return R


def aggregate(args):
    R = collect(args)
    for arm, r in R.items():
        if r is None:
            log(f"{arm:9s} missing"); continue
        b = r["budgets"]["P"]
        log(f"{arm:9s} steps={r['steps_done']:3d} stop={r['early_stop']} P={r['n_params']} "
            f"dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} spread={b['spread']:.1e} "
            f"E(50)={r['traj'][50] if len(r['traj']) > 50 else float('nan'):.6f} s/step={r['s_per_step']:.2f}")
    v, why, nums = decide(args.route, R, pilot_check=args.instance in LT.PILOT)
    name = "7.1 (LH-HFPS)" if args.route == "hfps" else "7.3 (SH-HFDS)"
    log(f"ROUTE {name} T2 VERDICT: {v}  " + "; ".join(why) + "  "
        + " ".join(f"{k}={val:.4g}" for k, val in nums.items()))
    if args.route == "hfps" and R.get("agpj") is not None and R.get("nnbf") is not None:
        log(f"reported only: agpj (projected BCS x J) rho_N = "
            f"{R['agpj']['budgets']['P']['dE_sign'] / R['nnbf']['budgets']['P']['dE_sign']:.3f}")
    return v


def synthetic(route):
    def res(ds, dt=0.2, stop=None, inst=4, traj0=-8.06278092, spread=0.0, calib=0.0, steps=75):
        return dict(instance=inst, calib_err=calib, early_stop=stop, steps_done=steps, params=dict(sr_iters=75),
                    traj=[traj0] + [0.0] * 74, budgets=dict(P=dict(dE_sign=ds, dE_tot=dt, spread=spread)))
    rt = "hfps_plus" if route == "hfps" else "sh_plus"

    def case(r, n, f, f2=None):
        R = {rt: r, "nnbf": n}
        if route == "hfps":
            R["hfps_site"] = f
        else:
            R["lh_plus"], R["lh_site"] = f, (f2 if f2 is not None else res(1.0))
        return R
    cases = [("PROMOTE", case(res(0.020), res(0.019), res(0.030))),
             ("KILL", case(res(0.030), res(0.019), res(0.050))),
             ("KILL", case(res(0.020), res(0.019), res(0.022))),
             ("AMBIGUOUS", case(res(0.025), res(0.019), res(0.040))),
             ("KILL", case(res(0.020, stop="EARLY KILL at it 50: ..."), res(0.019), res(0.030))),
             ("UNINFORMATIVE", case(res(0.003), res(0.004), res(0.005))),
             ("INVALID", case(res(0.020), res(0.019, traj0=-8.0), res(0.030))),
             ("INVALID", case(res(0.020, spread=0.2), res(0.019), res(0.030))),
             ("INVALID", case(res(0.020), res(0.019, steps=60), res(0.030))),
             ("INCOMPLETE", case(res(0.020), None, res(0.030))),
             ("INCOMPLETE", case(None, res(0.019), res(0.030)))]
    if route == "sh":
        # foil selection: the better lh arm is the foil, in either order
        cases += [("KILL", case(res(0.020), res(0.019), res(0.030), res(0.022))),       # lh_site is foil
                  ("PROMOTE", case(res(0.020), res(0.019), res(0.050), res(0.030))),    # lh_site is foil
                  ("KILL", case(res(0.020), res(0.019), res(0.022), res(0.030))),       # lh_plus is foil
                  ("INCOMPLETE", {rt: res(0.020), "nnbf": res(0.019), "lh_plus": res(0.03), "lh_site": None})]
    ok = True
    for want, R in cases:
        got, why, _ = decide(route, R)
        log(f"synthetic [{route}]: expected {want:13s} got {got:13s} ({'; '.join(why)})")
        ok &= got == want
    log("SYNTHETIC: every branch returns its intended verdict" if ok else "SYNTHETIC: FAILED")
    return ok


def controls(args):
    """Toy-instance-7 pipeline verdicts (results trained beforehand)."""
    T, S = args.toy_results, args.results
    R = {a: load(T, a, 7, args.seed) for a in ("nnbf", "sj", "lh_plus", "lh_site")}
    R.update({a: load(S, a, 7, args.seed) for a in ARMS[args.route]})
    for a, r in R.items():
        log(f"toy {a:9s} " + ("missing" if r is None else f"P={r['n_params']} dE_sign={r['budgets']['P']['dE_sign']:.5f} "
                              f"dE_tot={r['budgets']['P']['dE_tot']:.5f} s/step={r['s_per_step']:.2f}"))
    ok = True
    if args.route == "hfps":
        pos = LT.verdict(R["nnbf"], R["nnbf"], R["sj"], False)   # foil must be strictly worse: Slater x J
        neg = LT.verdict(R["agpj"], R["nnbf"], R["hfps_site"], False)
        real = decide("hfps", R, False)
        log(f"toy positive control (route := reference itself, foil := Slater x J): {pos[0]} {pos[2]}")
        log(f"toy negative control (route := projected BCS x J, no backflow): {neg[0]} {neg[2]}")
    else:
        pos = LT.verdict(R["nnbf"], R["nnbf"], R["sj"], False)
        Rn = dict(R); Rn["sh_plus"] = R["sj"]
        neg = decide("sh", Rn, False)
        real = decide("sh", R, False)
        log(f"toy positive control (route := reference itself, foil := Slater x J): {pos[0]} {pos[2]}")
        log(f"toy negative control (route := Slater x J, foil := better lh arm): {neg[0]} {neg[2]} {neg[1][-1]}")
    log(f"toy pipeline (route arm; verdict meaningless at toy size): {real[0]} {real[2]}")
    ok &= pos[0] == "PROMOTE" and neg[0] == "KILL"
    log("CONTROLS: intended verdicts" if ok else "CONTROLS: FAILED")
    return ok


def preflight_inst4(args):
    ok = True
    m = mt.make_model(4)
    W, Vv, deg = solve_ed(m, 2, 1e-8, mt.UHF_SEED)
    psi0 = mt.resolve_ground(m, Vv, deg)
    sg, gen = mt.build_symmetry(m, psi0)
    m.spin_setup()
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    d_chi, d_S = mt.sector_dims(m, sg, S0)
    pil = LT.PILOT[4]
    log(f"instance 4: E0={W[0]:.10f} (pilot {pil['E0']:.10f}) |G|={sg.G} S={S0:g} dim_S_chi={d_S:.0f} deg={deg} "
        f"N_up=N_dn={m.nup}")
    ok &= abs(W[0] - pil["E0"]) < 1e-8 and sg.G == 4 and S0 == 0 and d_S == pil["dim_S_chi"] and m.nup == m.ndn
    refs = mt.references(m)
    ta = parse_args(["--mode", "train", "--instance", "4", "--arm", ARMS[args.route][0]])
    times = {}
    for arm in ARMS[args.route]:
        a, kw, tgt = make_arm(arm, m, refs, 11, args.F)
        rng = np.random.default_rng(11 + 1000 * RNG_OFFSET[arm])
        t0 = time.time()
        Es = []
        for _ in range(args.preflight_steps):
            E, f, step = mt.sr_step(a, m, sg, "chi", 0, ta, rng)
            a.set_flat(a.flat() - step); Es.append(E)
        times[arm] = (time.time() - t0) / args.preflight_steps
        ok_p = abs(a.n_params() - tgt) <= 2 * m.V + 2
        ok &= ok_p and np.isfinite(Es[0])
        log(f"instance 4 {arm:9s} {kw} P={a.n_params()} (target {tgt}, matched {ok_p}) E(it0)={Es[0]:.8f} "
            f"{times[arm]:.2f} s/step at {os.environ.get('OMP_NUM_THREADS', '?')} threads")
    return ok


# ============================================================================ cli
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["check", "train", "aggregate", "synthetic", "preflight"], default="check")
    ap.add_argument("--route", choices=["", "hfps", "sh"], default="")
    ap.add_argument("--task", type=int, default=-1, choices=[-1, 0, 1, 2])
    ap.add_argument("--arm", default="", choices=["", "hfps_plus", "hfps_site", "agpj", "sh_plus",
                                                  "nnbf", "sj", "lh_plus", "lh_site"])
    ap.add_argument("--part", choices=["check", "synthetic", "controls", "inst4"], default="check")
    ap.add_argument("--instance", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--F", type=int, default=0)
    ap.add_argument("--toy", type=int, choices=[0, 1], default=0)
    ap.add_argument("--sr-iters", type=int, default=75)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--preflight-steps", type=int, default=1)
    ap.add_argument("--results", default="")
    ap.add_argument("--ref-results", default=os.path.join(HERE, "results"))
    ap.add_argument("--toy-results", default=os.path.join(HERE, "results_preflight"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.arm and not a.route:
        a.route = route_of(a.arm)
    if a.task >= 0:
        if not a.route:
            ap.error("--task needs --route")
        a.arm = ARMS[a.route][a.task]
    if not a.route:
        ap.error("--route is required")
    if a.mode == "train" and not a.arm:
        ap.error("--mode train needs --task or --arm")
    if a.F == 0:
        a.F = F_DEFAULT[a.route]
    if a.toy:
        a.sr_samples, a.lbfgs_iters, a.fn_iters = 400, 1500, 4
    if not a.results:
        a.results = os.path.join(HERE, f"results_{a.route}" + ("_preflight" if a.instance == 7 else ""))
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "check" or (args.mode == "preflight" and args.part == "check"):
        return 0 if run_check(args.route) else 2
    if args.mode == "synthetic" or (args.mode == "preflight" and args.part == "synthetic"):
        return 0 if synthetic(args.route) else 2
    if args.mode == "preflight":
        return 0 if (controls(args) if args.part == "controls" else preflight_inst4(args)) else 2
    if args.mode == "aggregate":
        aggregate(args); return 0
    out = os.path.join(args.results, f"t2_{args.arm}_inst{args.instance}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    res = train(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
