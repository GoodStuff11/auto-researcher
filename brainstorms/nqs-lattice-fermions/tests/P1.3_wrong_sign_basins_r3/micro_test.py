#!/usr/bin/env python3
"""
P1.3 round-3 decisive test: is there still a wrong-sign basin once NNBF is trained WITH
symmetry projection, and if so, do two escape routes beat projected SR and a fair null?
=======================================================================================

Background
----------
P1.1-P (job 407525, partial results) shows that projected SR removes essentially all of
the NNBF sign error on the P1.1 instances: on 4x3 PBC 4+4, U=8, t'=-0.25, projected-SR
NNBF of width 8 ends at dE_tot = 8e-4 and dE_sign = 8e-6 (three seeds), against
dE_sign = 0.09 for unprojected SR. On those instances there is no basin left to escape.
The only place a basin can survive projection is a SAME-SECTOR competitor: a state in
psi_0's space-group irrep and total spin that the projected optimizer can converge to
instead of psi_0. This test therefore runs on 4x3 PBC 4+4 clusters at the same-sector
avoided crossing, selected by sector_scan.py (ED, competitors confirmed in one sector in
code), and asks first whether such a basin exists (premise), and only then whether the
routes escape it.

Physics core (Hubbard model, ED, UHF, NNBF, exact-force SR, E_sign / ten Haaf E_FN) is
p11_core.py, a verbatim copy of the P1.1 test. Symmetry code (space-group irrep projector
P_chi, total-spin Loewdin projector P_S, projected log-derivatives) is symm.py, copied
verbatim from tests/P1.1P_projected/micro_test.py. Nothing is imported across folders.

Variational state of every projected arm
----------------------------------------
psi_P = P_S P_chi psi_theta (psi_0's irrep chi of the space group G = translations x
{1, Mx, My, MxMy}, |G| = 48, and psi_0's total spin S). Energy and force are exact sums:
E = <psi_P|H|psi_P>/<psi_P|psi_P>, half-gradient f = sum_n r(n) d psi_theta(n) with
r = (H - E) psi_P / ||psi_P||^2 (valid because P_S P_chi is a real symmetric idempotent
commuting with H). The SR metric is the Fubini-Study metric of P_chi psi_theta, from
--sr-samples configurations drawn exactly from |P_chi psi_theta|^2 with the projected
log-derivatives of P1.1-P (exact for the P_chi factor; for the P_S factor it is a positive
definite preconditioner, not the exact metric; the fixed points are exact stationary
points of E[psi_P] because the force is exact). SR step, shift and trust region are P1.1's.

Arms (one Slurm task = one instance x one seed x one arm; budget T = --budget steps)
-------------------------------------------------------------------------------------
Every arm uses exactly T SR-step equivalents, except the two oracle witnesses, which get
--pretrain extra fidelity steps first. A sampled acceptance decision (below) costs one
step-equivalent per scored state (it draws --acc-samples configurations, the same number
as one SR metric). The final state of every arm is the lowest-energy iterate of its last
--final-window steps.
  U      unprojected NNBF, plain SR, AF-UHF start (the P1.1 protocol). Evaluated as is AND
         after post-hoc projection (U->P). Context only; never a control.
  PSaf   PROJECTED SR from the AF-UHF reference (P1.1 start). The projected control
         required by round-3 rule 1. Also a "cold run" for the premise.
  PSfs   projected SR from a Fermi-sea reference (lowest N_sigma orbitals of the hopping
         matrix, shell ties broken by a 1e-3 seed-dependent random on-site field). Cold run.
  PScdw  projected SR from a charge-modulated UHF reference (UHF started from a density
         with a hole-rich column and AF order). Cold run.
  PW0    ORACLE witness: --pretrain projected natural-gradient steps on -log F(psi_0, psi_P)
         from the AF-UHF start, then T projected SR steps. Sets the floor reachable in the
         family once SR starts in psi_0's basin.
  PW1    ORACLE trap test: the same, but pre-trained toward the in-sector competitor psi_1
         (the second level of psi_0's sector). If projected SR leaves psi_1, the competitor
         is not an attractor of projected SR.
  N      NULL, matched restarts from symmetry-adapted references (critic objection 1):
         --null-k projected-SR runs from Fermi-sea references with independent shell
         tie-breaks and NNBF seeds, each L = (warm-up + proposal steps of arm A) // null-k
         steps, then the sampled acceptance picks one, which continues for the rest of the
         budget. No 1-RDM feedback, no penalty.
  A      ROUTE 3.1, SNO-PE (symmetric-natural-orbital re-anchoring, projected, sampled
         energy acceptance): --warmup projected SR steps from AF-UHF, then --cycles cycles.
         Each cycle estimates the 1-RDM of psi_P from --acc-samples samples of |psi_P|^2
         (local estimator (c^dag_i c_j psi)(n)/psi(n), genuinely sampled), averages it over
         G, diagonalises it, and builds four candidates by replacing NNBF's reference
         orbitals (backflow weights kept): top-N_sigma natural orbitals, and the same with
         the last occupied / first empty orbital exchanged for up, down, both. Incumbent and
         candidates are relaxed --relax projected SR steps and compared by the sampled
         acceptance. Rest of the budget: projected SR.
  O      ROUTE 3.2, OCP (in-sector orthogonal-complement probe): --probe-at projected SR
         steps from AF-UHF give psi_a. A probe network psi_b (fresh NNBF on a Fermi-sea
         reference) is trained --probe projected SR steps on E[psi_b] + --lam * F(psi_a,
         psi_b), F the fidelity. Both are then relaxed --relax plain projected steps and
         compared by the sampled acceptance; the winner continues for the rest of the budget.
         Because both states live in psi_0's sector, the probe can only find a lower state
         of the SAME sector (escape) or a same-sector competitor (certificate).
         A sampled-estimator side channel (rule 4) records, every 25 probe steps, the
         two-sample Monte Carlo estimate F_hat = <psi_a/psi_b>_{|psi_b|^2} <psi_b/psi_a>_{|psi_a|^2}
         from --acc-samples samples per state, its standard error, and the exact F.
  O0     ABLATION of O: identical schedule with --lam 0 (the probe is a plain restart).
Sampled acceptance (shared by N, A, O, O0)
------------------------------------------
For each state, draw --acc-samples configurations from |psi_P|^2, compute the local energy
E_loc(n) = (H psi_P)(n)/psi_P(n) at each sample (the VMC estimator; here read from H psi_P,
which gives the same per-sample value a per-amplitude implementation computes from the
connected configurations), and form the mean and standard error. The candidate with the
lowest mean replaces the incumbent only if mean_c + 2 sqrt(se_c^2 + se_inc^2) < mean_inc.
Each decision is logged together with the exact-energy decision (argmin of exact E), so
the aggregate can measure how often sampling changed the decision.

Evaluation of every final state
-------------------------------
E_var, e_var = E_var - E_0, E_sign (P1.1: min over L-BFGS-B starts and ten Haaf fixed node,
an upper bound on min_{a>=0} a^T S H S a / a^T a) and e_sign = E_sign - E_0, E_FN, the
sign-mismatch weight w, overlaps ov_k = |<psi_k^sec|psi>|^2 with the three lowest
in-sector states, <S^2>, <Tx>, <Ty>, <Mx>, <My>. "trapped" = (e_var >= 0.5 gap_in AND
ov_1 > ov_0), gap_in = E_1^sec - E_0.

Pre-committed decision rule (implemented in --aggregate; per instance, n = seeds that
have every required arm, n >= --min-seeds = 5, else the instance is INCOMPLETE)
------------------------------------------------------------------------------------
Premise (gates the routes):
  TRAP     the trap test is valid (median fidelity of PW1 to psi_1 after pre-training
           >= 0.5) and PW1 is trapped on >= ceil(2n/3) seeds.
  capture  number of trapped final states among the 3n cold runs {PSaf, PSfs, PScdw}.
  depth    d(i) = e_var(PSaf, i) - e_var(PW0, i); D = median_i d(i).
  BASIN    TRAP and capture >= max(2, ceil(3n/6)).
  NO-BASIN capture == 0 and D <= 0.05 gap_in.
  UNCLEAR  otherwise.
Routes R in {A, O} (only evaluated under BASIN):
  b(i) = min(e_sign(PSaf, i), e_sign(N, i)), rho_R(i) = (b(i) - e_sign(R, i)) / b(i),
  dEv_R = median_i [e_var(R, i) - min(e_var(PSaf, i), e_var(N, i))],
  trap_R = number of trapped seeds of R; trap_ctrl = min(trapped PSaf, trapped N).
  Gates (each is a FAIL if violated, with its reason printed):
    G-budget   every non-oracle arm used exactly T step-equivalents.
    G-ctrl     trap_R < trap_ctrl (strict): R must beat the projected control AND the null.
    G-ablation A: covered by N (same references minus the 1-RDM feedback is N's design;
               b already contains N). O: median(rho_O - rho_O0) >= 0.10 and
               rho_O > rho_O0 on >= n - 1 seeds.
    G-sampled  A: the sampled acceptance agrees with the exact decision in >= 80% of its
               cycles. O: in >= 90% of the probe checkpoints |F_hat - F| <= max(0.02, 3 se)
               and se <= 0.05.
  PASS(R)  all gates hold, median rho_R >= 0.25, rho_R > 0 on >= n - 1 seeds, dEv_R <= 0.01.
  FAIL(R)  any gate violated, or median rho_R < 0.10, or rho_R > 0 on <= n/2 seeds.
  AMBIGUOUS(R) otherwise; action: 6 more seeds per instance, kill R unless the 12-seed set
           passes.
  Under NO-BASIN: R is "N/A (premise falsified)"; under UNCLEAR: "UNTESTED (premise unclear)".
Global (all instances in the results folder, smoke excluded):
  premise  FALSIFIED if NO-BASIN on every instance; CONFIRMED if BASIN on every instance;
           otherwise MIXED (reported per instance; no universal claim is made).
  route R  PASS only if PASS on every instance; FAIL if FAIL on any BASIN instance.

Usage
-----
    python3 micro_test.py [options]
    python3 micro_test.py --aggregate results

Options
-------
  --instance {c1,c2,smoke}  c1: 4x3 PBC, N_up=N_dn=4, U=6, t'=-0.37 (D=245025), the
                    smallest same-sector gap found by sector_scan.py on 4x3 at 4+4
                    (gap_in = 0.234t = 0.0195t per site; sector_scan_n4.txt).
                    c2: 4x3 PBC, 4+4, U=8, t'=-0.34 (gap_in = 0.273t, the avoided crossing
                    used by P3.2 round 2).
                    smoke: 4x3 PBC, N_up=N_dn=2, U=8, t'=-0.3 (D=4356), code checks only.
                    Default c1.
  --arm {U,PSaf,PSfs,PScdw,PW0,PW1,N,A,O,O0}  arm to run (default PSaf). "all" runs every
                    arm in sequence (smoke use).
  --task-id INT     if >= 0, OVERRIDES --instance, --seed, --arm from a Slurm array index t
                    in 0..119: instance = ("c1","c2")[t // 60], seed = 1234 + (t % 60) // 10,
                    arm = ARMS[t % 10]. Default -1 (off).
  --seed INT        seed s (default 1234): UHF noise, reference tie-breaks, NNBF init
                    (seed + width), SR sampling default_rng(s), acceptance default_rng(s+17).
  --width INT       NNBF hidden width (default 32).
  --budget INT      T, SR-step equivalents per arm (default 1500, P1.1).
  --warmup INT      arm A warm-up steps (default 300).
  --cycles INT      arm A cycles (default 2).
  --relax INT       relaxation steps per scored state in A, O, O0 (default 40).
  --null-k INT      starts of the null arm N (default 3).
  --probe-at INT    steps before the probe in O, O0 (default 600).
  --probe INT       probe steps in O, O0 (default 300).
  --lam FLOAT       OCP penalty strength in units of t (default 2.0; O0 forces 0).
  --pretrain INT    oracle pre-training steps of PW0, PW1 (default 300).
  --acc-samples INT samples per sampled estimate (default 2000).
  --final-window INT final state = lowest-energy iterate of the last this-many steps
                    (default 100).
  --sr-lr FLOAT     SR step (default 0.05).  --sr-shift FLOAT  diagonal shift (default 1e-3).
  --sr-samples INT  SR metric samples (default 2000).
  --chunk INT       configurations per forward chunk (default 65536).
  --o-chunk INT     samples per chunk for projected log-derivatives (default 256).
  --fn-iters INT    ten Haaf iterations in the evaluation (default 4).
  --fn-floor FLOAT  fixed-node guide floor (default 1e-3).
  --lbfgs-iters INT L-BFGS-B iterations per start (default 3000).
  --min-seeds INT   seeds needed per instance in --aggregate (default 5).
  --out PATH        JSON path (default results/<instance>_s<seed>_<arm>.json next to this
                    script); not overwritten unless --overwrite 1.
  --overwrite {0,1} default 0.
  --checks-only {0,1}  1: run the correctness checks and exit.
  --aggregate DIR   no computation: apply the decision rule to every *.json in DIR
                    (relative to this script if not absolute) and print the verdicts.
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import p11_core as P
import symm

HERE = os.path.dirname(os.path.abspath(__file__))
INST = {"c1": dict(Lx=4, Ly=3, nup=4, ndn=4, U=6.0, tp=-0.37),
        "c2": dict(Lx=4, Ly=3, nup=4, ndn=4, U=8.0, tp=-0.34),
        "smoke": dict(Lx=4, Ly=3, nup=2, ndn=2, U=8.0, tp=-0.30)}
ARMS = ("U", "PSaf", "PSfs", "PScdw", "PW0", "PW1", "N", "A", "O", "O0")
COLD = ("PSaf", "PSfs", "PScdw")
ORACLE = ("PW0", "PW1")
log = P.log


# ============================================================================ setup
def make_model(name):
    c = INST[name]
    return P.Hubbard(c["Lx"], c["Ly"], "pbc", "pbc", c["nup"], c["ndn"], 1.0, c["tp"], c["U"])


def make_ctx(m, a):
    W, Vv, deg = P.solve_ed(m, 2, 1e-8, a.seed)
    if deg != 1:
        raise RuntimeError("degenerate ground state")
    sg, spn = symm.SpaceGroup(m), symm.Spin(m)
    lam = sg.irrep_of(Vv[:, 0])
    s2 = spn.expect(Vv[:, 0])
    S = float(round(0.5 * (np.sqrt(1 + 4 * s2) - 1)))
    proj = symm.SectorProjector(m, sg, spn, lam, S)
    w, Us = symm.sector_lowest(m, proj, 3, a.seed)
    if abs(w[0] - W[0]) > 1e-7:
        raise RuntimeError(f"sector ground state {w[0]} != global {W[0]}")
    same = []
    for k in range(3):
        ge = sg.gen_expect(Us[:, k])
        same.append(all(abs(ge[g] - lam[g]) < 1e-6 for g in lam)
                    and abs(spn.expect(Us[:, k]) - S * (S + 1)) < 1e-6)
    if Us[:, 0] @ Vv[:, 0] < 0:
        Us[:, 0] *= -1
    ops_u, ops_d = one_body_ops(m.V, m.nup), one_body_ops(m.V, m.ndn)
    return dict(m=m, E0=float(W[0]), Esec=w.tolist(), gap=float(w[1] - w[0]), psis=Us,
                sg=sg, spn=spn, proj=proj, lam=lam, S=S, same_sector=same,
                perms=symm.group(m), ops_u=ops_u, ops_d=ops_d, steps=0)


# ============================================================================ references
def uhf_from(m, nu, nd, iters=2000, mix=0.3):
    """P1.1 UHF loop (p11_core.uhf_orbitals) from given initial densities."""
    T, U = m.T, m.U
    for _ in range(iters):
        eu, vu = np.linalg.eigh(-T + np.diag(U * nd))
        ed, vd = np.linalg.eigh(-T + np.diag(U * nu))
        Pu, Pd = vu[:, :m.nup], vd[:, :m.ndn]
        nu_new, nd_new = np.sum(Pu ** 2, 1), np.sum(Pd ** 2, 1)
        if max(np.abs(nu_new - nu).max(), np.abs(nd_new - nd).max()) < 1e-10:
            break
        nu, nd = (1 - mix) * nu + mix * nu_new, (1 - mix) * nd + mix * nd_new
    return Pu.copy(), Pd.copy()


def reference(m, kind, seed):
    rng = np.random.default_rng(seed + 31)
    if kind == "af":
        return P.uhf_orbitals(m, seed)
    if kind == "fs":
        out = []
        for n in (m.nup, m.ndn):
            _, v = np.linalg.eigh(-m.T + np.diag(1e-3 * rng.standard_normal(m.V)))
            out.append(v[:, :n].copy())
        return tuple(out)
    if kind == "cdw":
        x = np.array([c[0] for c in m.coords]); y = np.array([c[1] for c in m.coords])
        stag = (-1.0) ** (x + y)
        hole = np.where(x == 0, -0.4, 0.4 / 3)
        nu = (m.nup / m.V) * (1 + hole + 0.3 * stag) + 0.01 * rng.standard_normal(m.V)
        nd = (m.ndn / m.V) * (1 + hole - 0.3 * stag) + 0.01 * rng.standard_normal(m.V)
        return uhf_from(m, nu, nd)
    raise ValueError(kind)


def new_ans(m, kind, seed, width):
    Pu, Pd = reference(m, kind, seed)
    return P.Ansatz(m, Pu, Pd, "nnbf", width=width, jastrow=False, seed=seed + width)


def clone(ans):
    new = copy.copy(ans)
    new.p = {k: v.copy() for k, v in ans.p.items()}
    return new


# ============================================================================ projected SR
def pstate(ans, ctx, a):
    psi_t = ans.amplitudes(a.chunk)
    psi_c = ctx["sg"].project(psi_t)
    return psi_t, psi_c, ctx["spn"].project(psi_c, ctx["S"])


def penergy(ans, ctx, a):
    return P.energy(ctx["m"].H, pstate(ans, ctx, a)[2])[0]


def fid_half_grad(ans, psi_t, psi, nn, ref, a):
    """Half-gradient of F = <ref|psi_P>^2/(<ref|ref><psi_P|psi_P>) for ref in the sector:
    d<ref|psi_P> = <ref|d psi_theta>, d<psi_P|psi_P> = 2 <psi_P|d psi_theta>."""
    ov = ref @ psi
    F = ov * ov / (nn * (ref @ ref))
    g = ans.grad_weighted(psi_t * (ref / ov - psi / nn), a.chunk)
    return F, F * g


def sr_step(ans, ctx, a, rng, force=None):
    """One projected SR step. force(ans, psi_t, psi, nn, E, Hp) -> half-gradient, or None
    for the energy force. Returns E before the step."""
    H = ctx["m"].H
    psi_t, psi_c, psi = pstate(ans, ctx, a)
    E, Hp, nn = P.energy(H, psi)
    if force is None:
        f = ans.grad_weighted(psi_t * (Hp - E * psi) / nn, a.chunk)
    else:
        f = force(ans, psi_t, psi, nn, E, Hp)
    pc = psi_c ** 2
    idx = rng.choice(len(psi), size=a.sr_samples, p=pc / pc.sum())
    O = symm.per_sample_O_proj(ans, ctx["sg"], idx, psi_t, psi_c, a.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = a.sr_shift
    Km = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
    step = a.sr_lr * dx
    nrm = np.linalg.norm(step)
    if nrm > 1.0:
        step /= nrm
    ans.set_flat(ans.flat() - step)
    ctx["steps"] += 1
    return E


def sr_run(ans, ctx, n, rng, a, label, force=None, final_window=0, hook=None):
    best = (np.inf, None)
    for it in range(n):
        x = ans.flat().copy()
        E = sr_step(ans, ctx, a, rng, force)
        if not np.isfinite(E):
            ans.set_flat(x); break
        if final_window and it >= n - final_window and E < best[0]:
            best = (E, x)
        if hook is not None:
            hook(it, ans)
        if it % 100 == 0:
            log(f"  {label}: it {it} E={E:.6f}")
    if final_window:
        Ef = penergy(ans, ctx, a)
        if best[1] is not None and best[0] < Ef:
            ans.set_flat(best[1])
    return ans


def sr_run_unproj(ans, ctx, n, rng, a, label, final_window=0):
    """Plain SR on psi_theta (P1.1 step), used only by arm U."""
    H = ctx["m"].H
    best = (np.inf, None)
    for it in range(n):
        x = ans.flat().copy()
        psi = ans.amplitudes(a.chunk)
        E, Hp, nn = P.energy(H, psi)
        f = ans.grad_weighted(psi * (Hp - E * psi) / nn, a.chunk)
        idx = rng.choice(len(psi), size=a.sr_samples, p=psi ** 2 / nn)
        O = ans.per_sample_O_fast(idx)
        Oc = (O - O.mean(0)) / np.sqrt(len(idx))
        Km = Oc @ Oc.T + a.sr_shift * np.eye(len(idx))
        dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / a.sr_shift
        step = a.sr_lr * dx
        nrm = np.linalg.norm(step)
        if nrm > 1.0:
            step /= nrm
        ans.set_flat(ans.flat() - step)
        ctx["steps"] += 1
        if final_window and it >= n - final_window and E < best[0]:
            best = (E, x)
        if it % 100 == 0:
            log(f"  {label}: it {it} E={E:.6f}")
    if final_window:
        Ef = P.energy(H, ans.amplitudes(a.chunk))[0]
        if best[1] is not None and best[0] < Ef:
            ans.set_flat(best[1])
    return ans


# ============================================================================ sampled estimators
def sampled_energy(H, psi, N, rng):
    p = psi ** 2
    idx = rng.choice(len(psi), size=N, p=p / p.sum())
    el = (H @ psi)[idx] / psi[idx]
    m, s = float(el.mean()), float(el.std(ddof=1) / np.sqrt(N))
    z = (el - el.mean()) / max(el.std(), 1e-300)
    return dict(mean=m, se=s, kurt=float(np.mean(z ** 4)))


def accept(ctx, a, rng, states):
    """states: list of (name, ans), states[0] = incumbent. Sampled decision with a 2-sigma
    margin; returns (index chosen, record). Costs one step-equivalent per state."""
    H = ctx["m"].H
    rows = []
    for name, ans in states:
        psi = pstate(ans, ctx, a)[2]
        se = sampled_energy(H, psi, a.acc_samples, rng)
        rows.append(dict(name=name, exact=P.energy(H, psi)[0], **se))
        ctx["steps"] += 1
    inc = rows[0]
    j = min(range(len(rows)), key=lambda k: rows[k]["mean"])
    if j != 0 and not (rows[j]["mean"] + 2 * math.hypot(rows[j]["se"], inc["se"]) < inc["mean"]):
        j = 0
    jex = min(range(len(rows)), key=lambda k: rows[k]["exact"])
    rec = dict(rows=rows, chosen=rows[j]["name"], exact_choice=rows[jex]["name"],
               agree=(j == jex))
    log("  accept: " + " ".join(f"{r['name']}={r['mean']:.5f}+-{r['se']:.5f}(ex {r['exact']:.5f})"
                                for r in rows) + f" -> {rows[j]['name']} (exact: {rows[jex]['name']})")
    return j, rec


def sampled_fidelity(pa, pb, N, rng):
    """Two-sample MC estimate F_hat = <pa/pb>_{|pb|^2} <pb/pa>_{|pa|^2} and its standard
    error (delta method)."""
    ib = rng.choice(len(pb), size=N, p=pb ** 2 / (pb @ pb))
    ia = rng.choice(len(pa), size=N, p=pa ** 2 / (pa @ pa))
    x, y = pa[ib] / pb[ib], pb[ia] / pa[ia]
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1) / N, y.var(ddof=1) / N
    F = float(mx * my)
    se = float(math.sqrt(my * my * vx + mx * mx * vy))
    return F, se, float(np.max(np.abs(x)) / max(abs(mx), 1e-300))


# ============================================================================ 1-RDM / NOs (copied from P1.3 r2)
def one_body_ops(V, N):
    ops = [[None] * V for _ in range(V)]
    st, lk, _, _ = P.species_basis(V, N)
    for i in range(V):
        for j in range(V):
            r, c, v = [], [], []
            for k, b in enumerate(st):
                if not (b >> j) & 1:
                    continue
                b1 = b ^ (1 << j); s1 = P.popcount_below(b, j)
                if (b1 >> i) & 1:
                    continue
                b2 = b1 | (1 << i); s2 = P.popcount_below(b1, i)
                r.append(lk[b2]); c.append(k); v.append((-1.0) ** (s1 + s2))
            ops[i][j] = sp.csr_matrix((v, (r, c)), shape=(len(st), len(st)))
    return ops


def rdm1_sampled(m, psi, ops_u, ops_d, idx):
    """rho_ij = < (c^dag_i c_j psi)(n) / psi(n) > over sampled n (local estimator)."""
    Psi = psi.reshape(m.Du, m.Dd)
    a, b = idx // m.Dd, idx % m.Dd
    den = Psi[a, b]
    V = m.V
    ru, rd = np.zeros((V, V)), np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            ru[i, j] = np.mean((ops_u[i][j] @ Psi)[a, b] / den)
            rd[i, j] = np.mean((ops_d[i][j] @ Psi.T).T[a, b] / den)
    return 0.5 * (ru + ru.T), 0.5 * (rd + rd.T)


def sym_nos(rho, perms):
    V = rho.shape[0]
    R = np.zeros_like(rho)
    Pm = {}
    for e, p in perms:
        Pg = np.zeros((V, V)); Pg[p, np.arange(V)] = 1.0
        R += Pg @ rho @ Pg.T
        Pm[e] = Pg
    R /= len(perms)
    split = 1e-7 * (Pm[(0, 0, 0, 1)] + 0.5 * Pm[(0, 0, 1, 0)])
    w, U = np.linalg.eigh(R + 0.5 * (split + split.T))
    o = np.argsort(-w)
    return w[o], U[:, o]


# ============================================================================ arms
def arm_U(ctx, a, rng, rng_acc):
    ans = new_ans(ctx["m"], "af", a.seed, a.width)
    return sr_run_unproj(ans, ctx, a.budget, rng, a, "U", a.final_window), {}


def arm_PS(kind):
    def run(ctx, a, rng, rng_acc):
        ans = new_ans(ctx["m"], kind, a.seed, a.width)
        return sr_run(ans, ctx, a.budget, rng, a, f"PS{kind}", final_window=a.final_window), {}
    return run


def arm_PW(k):
    def run(ctx, a, rng, rng_acc):
        ans = new_ans(ctx["m"], "af", a.seed, a.width)
        ref = ctx["psis"][:, k]

        def force(an, psi_t, psi, nn, E, Hp):
            F, hg = fid_half_grad(an, psi_t, psi, nn, ref, a)
            return -hg / max(F, 1e-12)
        sr_run(ans, ctx, a.pretrain, rng, a, f"PW{k} pre", force=force)
        F0 = float((pstate(ans, ctx, a)[2] @ ref) ** 2 / (pstate(ans, ctx, a)[2] @ pstate(ans, ctx, a)[2]))
        ctx["steps"] = 0  # oracle pre-training is outside the matched budget
        sr_run(ans, ctx, a.budget, rng, a, f"PW{k}", final_window=a.final_window)
        return ans, dict(F_after_pretrain=F0, pretrain=a.pretrain)
    return run


def proposal_steps(a, n_cand=4):
    return a.warmup + a.cycles * ((n_cand + 1) * a.relax + 1 + (n_cand + 1))


def arm_N(ctx, a, rng, rng_acc):
    L = proposal_steps(a) // a.null_k
    starts = []
    for j in range(a.null_k):
        ans = new_ans(ctx["m"], "fs", a.seed + 7919 * j, a.width)
        sr_run(ans, ctx, L, rng, a, f"N{j}")
        starts.append((f"start{j}", ans))
    j, rec = accept(ctx, a, rng_acc, starts)
    ans = starts[j][1]
    rest = a.budget - ctx["steps"]
    sr_run(ans, ctx, rest, rng, a, "N", final_window=a.final_window)
    return ans, dict(L=L, decision=rec)


def arm_A(ctx, a, rng, rng_acc):
    m = ctx["m"]
    ans = new_ans(m, "af", a.seed, a.width)
    sr_run(ans, ctx, a.warmup, rng, a, "A")
    hist = []
    for cyc in range(a.cycles):
        psi = pstate(ans, ctx, a)[2]
        idx = rng_acc.choice(m.D, size=a.acc_samples, p=psi ** 2 / (psi @ psi))
        ctx["steps"] += 1  # one sampling pass for the 1-RDM
        ru, rd = rdm1_sampled(m, psi, ctx["ops_u"], ctx["ops_d"], idx)
        wu, Uu = sym_nos(ru, ctx["perms"])
        wd, Ud = sym_nos(rd, ctx["perms"])

        def occ(Umat, n, swap):
            cols = list(range(n))
            if swap:
                cols[-1] = n
            return Umat[:, cols].copy()
        states = [("stay", clone(ans))]
        for name, su, sd in (("sno", 0, 0), ("swap_u", 1, 0), ("swap_d", 0, 1), ("swap_ud", 1, 1)):
            c = clone(ans)
            c.p["Pu"] = occ(Uu, m.nup, su); c.p["Pd"] = occ(Ud, m.ndn, sd)
            states.append((name, c))
        for name, c in states:
            sr_run(c, ctx, a.relax, rng, a, f"A {name}")
        j, rec = accept(ctx, a, rng_acc, states)
        ans = states[j][1]
        rec.update(cycle=cyc, no_occ_u=wu.tolist(), no_occ_d=wd.tolist())
        hist.append(rec)
    rest = a.budget - ctx["steps"]
    sr_run(ans, ctx, rest, rng, a, "A", final_window=a.final_window)
    return ans, dict(history=hist)


def arm_O(lam_override=None):
    def run(ctx, a, rng, rng_acc):
        m = ctx["m"]
        lam = a.lam if lam_override is None else lam_override
        ans = new_ans(m, "af", a.seed, a.width)
        sr_run(ans, ctx, a.probe_at, rng, a, "O a")
        ref = pstate(ans, ctx, a)[2].copy()
        probe = new_ans(m, "fs", a.seed + 104729, a.width)
        checks = []

        def force(an, psi_t, psi, nn, E, Hp):
            f = an.grad_weighted(psi_t * (Hp - E * psi) / nn, a.chunk)
            if lam == 0:
                return f
            F, hg = fid_half_grad(an, psi_t, psi, nn, ref, a)
            return f + lam * hg

        def hook(it, an):
            if it % 25 == 0 or it == a.probe - 1:
                pb = pstate(an, ctx, a)[2]
                Fex = float((ref @ pb) ** 2 / ((ref @ ref) * (pb @ pb)))
                Fh, se, mx = sampled_fidelity(ref, pb, a.acc_samples, rng_acc)
                checks.append(dict(it=it, F=Fex, F_hat=Fh, se=se, max_ratio=mx))
        sr_run(probe, ctx, a.probe, rng, a, "O probe", force=force, hook=hook)
        sr_run(ans, ctx, a.relax, rng, a, "O a relax")
        sr_run(probe, ctx, a.relax, rng, a, "O b relax")
        j, rec = accept(ctx, a, rng_acc, [("a", ans), ("b", probe)])
        ans = (ans, probe)[j]
        rest = a.budget - ctx["steps"]
        sr_run(ans, ctx, rest, rng, a, "O", final_window=a.final_window)
        return ans, dict(decision=rec, fid_checks=checks, lam=lam)
    return run


RUNNERS = {"U": arm_U, "PSaf": arm_PS("af"), "PSfs": arm_PS("fs"), "PScdw": arm_PS("cdw"),
           "PW0": arm_PW(0), "PW1": arm_PW(1), "N": arm_N, "A": arm_A, "O": arm_O(),
           "O0": arm_O(0.0)}


# ============================================================================ evaluation
def esign(m, a, psi, p0):
    s = np.where(psi >= 0, 1.0, -1.0)
    S = sp.diags(s)
    K = (S @ m.H @ S).tocsr()
    E_FN, E_FN_var, phi, _ = P.fixed_node(K, a.fn_iters, a.fn_floor)
    E_lb, _, starts = P.sign_constrained_min(
        K, [("abs_theta", psi), ("abs_fixed_node", phi), ("abs_exact", p0)], a.lbfgs_iters)
    return dict(E_sign=float(min(E_lb, E_FN, E_FN_var)), E_FN=float(E_FN),
                sign_starts=[(n, float(e)) for n, e, _, _ in starts])


def evaluate(ctx, a, psi):
    m = ctx["m"]
    nrm = np.linalg.norm(psi)
    if not np.isfinite(nrm) or nrm < 1e-12:
        return None
    psi = psi / nrm
    p0 = ctx["psis"][:, 0]
    if p0 @ psi < 0:
        p0 = -p0
    E = P.energy(m.H, psi)[0]
    out = dict(E=E, e_var=E - ctx["E0"], **esign(m, a, psi, p0))
    out["e_sign"] = out["E_sign"] - ctx["E0"]
    out["w_sign"] = float(np.sum(p0[np.sign(psi) * np.sign(p0) < 0] ** 2))
    out["ov"] = [float((ctx["psis"][:, k] @ psi) ** 2) for k in range(ctx["psis"].shape[1])]
    out["S2"] = ctx["spn"].expect(psi)
    out["gen"] = ctx["sg"].gen_expect(psi)
    out["trapped"] = bool(out["e_var"] >= 0.5 * ctx["gap"] and out["ov"][1] > out["ov"][0])
    return out


# ============================================================================ checks
def run_checks(seed):
    ok = P.run_checks(seed)

    class _A:
        chunk, o_chunk, sr_samples, sr_lr, sr_shift = 4096, 64, 200, 0.05, 1e-3
        seed, acc_samples = 1234, 4000
    a = _A()
    a.seed = seed
    m = make_model("smoke")
    ctx = make_ctx(m, a)
    log(f"check smoke instance: E0={ctx['E0']:.10f} sector levels {np.round(ctx['Esec'], 6).tolist()} "
        f"chi={ctx['lam']} S={ctx['S']:g} same_sector={ctx['same_sector']}")
    ok &= all(ctx["same_sector"])
    proj = ctx["proj"]
    rng = np.random.default_rng(seed)
    v, y = rng.standard_normal(m.D), rng.standard_normal(m.D)
    Pv = proj(v)
    e_idem = np.linalg.norm(proj(Pv) - Pv) / np.linalg.norm(Pv)
    e_sym = abs(v @ proj(y) - Pv @ y) / (np.linalg.norm(v) * np.linalg.norm(y))
    e_hc = np.linalg.norm(proj(m.H @ v) - m.H @ Pv) / np.linalg.norm(m.H @ v)
    log(f"check P_S P_chi: idempotent {e_idem:.1e} symmetric {e_sym:.1e} [P,H] {e_hc:.1e}")
    ok &= e_idem < 1e-10 and e_sym < 1e-10 and e_hc < 1e-10
    # projected gradient (full P) vs FD
    an = new_ans(m, "af", seed, 4)
    an.p["W2"] *= 100
    x0 = an.flat().copy()
    psi_t, psi_c, psi = pstate(an, ctx, a)
    E, Hp, nn = P.energy(m.H, psi)
    f = an.grad_weighted(psi_t * (Hp - E * psi) / nn, a.chunk)
    d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
    h = 1e-6
    Ef = lambda x: (an.set_flat(x), penergy(an, ctx, a))[1]
    fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h); an.set_flat(x0)
    log(f"check P_S P_chi energy gradient: analytic {2 * f @ d:.10e} fd {fd:.10e}")
    ok &= abs(2 * f @ d - fd) < 1e-6 * max(1, abs(fd))
    # fidelity half-gradient vs FD (ref = in-sector psi_1)
    ref = ctx["psis"][:, 1]
    F, hg = fid_half_grad(an, psi_t, psi, nn, ref, a)

    def Fof(x):
        an.set_flat(x); q = pstate(an, ctx, a)[2]
        return (q @ ref) ** 2 / ((q @ q) * (ref @ ref))
    fd = (Fof(x0 + h * d) - Fof(x0 - h * d)) / (2 * h); an.set_flat(x0)
    log(f"check projected fidelity gradient: analytic {2 * hg @ d:.10e} fd {fd:.10e}")
    ok &= abs(2 * hg @ d - fd) < 1e-6 * max(1, abs(fd))
    # sampled energy unbiased, sampled fidelity consistent
    se = sampled_energy(m.H, psi, a.acc_samples, rng)
    log(f"check sampled energy {se['mean']:.6f} +- {se['se']:.6f} vs exact {E:.6f}")
    ok &= abs(se["mean"] - E) < 5 * se["se"]
    q = proj(rng.standard_normal(m.D)); q += 3 * psi / np.linalg.norm(psi)
    Fex = (q @ psi) ** 2 / ((q @ q) * (psi @ psi))
    Fh, sef, _ = sampled_fidelity(psi, q, a.acc_samples, rng)
    log(f"check sampled fidelity {Fh:.5f} +- {sef:.5f} vs exact {Fex:.5f}")
    ok &= abs(Fh - Fex) < 5 * sef + 1e-3
    # sampled 1-RDM vs exact (trace and max deviation within 6 standard errors scale)
    idx = rng.choice(m.D, size=20000, p=psi ** 2 / (psi @ psi))
    ru, rd = rdm1_sampled(m, psi, ctx["ops_u"], ctx["ops_d"], idx)
    Pf = (psi / np.linalg.norm(psi)).reshape(m.Du, m.Dd)
    ru_ex = np.array([[np.sum(Pf * (ctx["ops_u"][i][j] @ Pf)) for j in range(m.V)] for i in range(m.V)])
    dev = np.max(np.abs(ru - 0.5 * (ru_ex + ru_ex.T)))
    log(f"check sampled 1-RDM: tr up {np.trace(ru):.6f} (N_up={m.nup}) max dev from exact {dev:.3e}")
    ok &= abs(np.trace(ru) - m.nup) < 1e-8 and dev < 0.05
    # references are valid orbitals
    for kind in ("af", "fs", "cdw"):
        Pu, Pd = reference(m, kind, seed)
        ok &= np.allclose(Pu.T @ Pu, np.eye(m.nup), atol=1e-10) and Pu.shape == (m.V, m.nup)
    log(f"P1.3 r3 CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ run
def run(a):
    m = make_model(a.instance)
    t0 = time.time()
    ctx = make_ctx(m, a)
    log(f"instance {a.instance} D={m.D} E0={ctx['E0']:.8f} sector levels {ctx['Esec']} "
        f"gap_in={ctx['gap']:.5f} chi={ctx['lam']} S={ctx['S']:g} same_sector={ctx['same_sector']} "
        f"({time.time() - t0:.0f}s)")
    arms = ARMS if a.arm == "all" else (a.arm,)
    res = dict(params={k: v for k, v in vars(a).items()}, instance=a.instance, seed=a.seed,
               E0=ctx["E0"], Esec=ctx["Esec"], gap=ctx["gap"], chi=ctx["lam"], S=ctx["S"],
               same_sector=ctx["same_sector"], arms={})

    def dump():
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    for arm in arms:
        rng = np.random.default_rng(a.seed)
        rng_acc = np.random.default_rng(a.seed + 17)
        ctx["steps"] = 0
        t1 = time.time()
        ans, info = RUNNERS[arm](ctx, a, rng, rng_acc)
        steps = ctx["steps"]
        if arm == "U":
            psi_t = ans.amplitudes(a.chunk)
            ev = dict(final=evaluate(ctx, a, psi_t), projected=evaluate(ctx, a, ctx["proj"](psi_t)))
        else:
            ev = dict(final=evaluate(ctx, a, pstate(ans, ctx, a)[2]))
        res["arms"][arm] = dict(steps=steps, info=info, time_s=time.time() - t1, **ev)
        f = ev["final"]
        log(f"ARM {arm}: steps={steps} e_var={f['e_var']:.5f} e_sign={f['e_sign']:.5f} "
            f"ov={np.round(f['ov'], 4).tolist()} trapped={f['trapped']} S2={f['S2']:.2e} "
            f"({time.time() - t1:.0f}s)")
        if arm == "U" and ev["projected"]:
            g = ev["projected"]
            log(f"ARM U->P: e_var={g['e_var']:.5f} e_sign={g['e_sign']:.5f} ov={np.round(g['ov'], 4).tolist()}")
        dump()
    log(f"wrote {a.out}")
    return 0


# ============================================================================ aggregate
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def aggregate_instance(inst, files, a):
    """Returns (premise, {route: verdict})."""
    by_seed = {}
    for r in files:
        for arm, v in r["arms"].items():
            by_seed.setdefault(r["seed"], {})[arm] = v
    gap = files[0]["gap"]
    print(f"===== instance {inst}: E0={files[0]['E0']:.6f} gap_in={gap:.5f} ({gap / 12:.5f} per site) "
          f"chi={files[0]['chi']} S={files[0]['S']} same_sector={files[0]['same_sector']}")
    req = ("PSaf", "PSfs", "PScdw", "PW0", "PW1", "N", "A", "O", "O0")
    seeds = sorted(s for s, d in by_seed.items() if all(k in d for k in req))
    n = len(seeds)
    for s in sorted(by_seed):
        d = by_seed[s]
        row = " ".join(f"{k}:{d[k]['final']['e_sign']:.4f}/{d[k]['final']['e_var']:.4f}"
                       f"{'T' if d[k]['final']['trapped'] else ''}" for k in ARMS if k in d)
        if "U" in d and d["U"].get("projected"):
            row += f" U->P:{d['U']['projected']['e_sign']:.4f}/{d['U']['projected']['e_var']:.4f}"
        print(f"  seed {s} (e_sign/e_var, T=trapped): {row}")
    if n < a.min_seeds or not all(files[0]["same_sector"][:2]):
        print(f"  INCOMPLETE: {n} complete seeds (need {a.min_seeds}) or competitor not same-sector")
        return "INCOMPLETE", {"A": "INCOMPLETE", "O": "INCOMPLETE"}
    D = {s: by_seed[s] for s in seeds}
    T = files[0]["params"]["budget"]
    fin = lambda s, k: D[s][k]["final"]
    trap_w1 = sum(fin(s, "PW1")["trapped"] for s in seeds)
    capture = sum(fin(s, k)["trapped"] for s in seeds for k in COLD)
    depth = _med([fin(s, "PSaf")["e_var"] - fin(s, "PW0")["e_var"] for s in seeds])
    F1 = _med([D[s]["PW1"]["info"]["F_after_pretrain"] for s in seeds])
    valid = F1 >= 0.5
    print(f"  trap test valid (median fidelity of PW1 to psi_1 after pre-training {F1:.3f} >= 0.5): {valid}")
    TRAP = valid and trap_w1 >= math.ceil(2 * n / 3)
    if TRAP and capture >= max(2, math.ceil(3 * n / 6)):
        prem = "BASIN"
    elif capture == 0 and depth <= 0.05 * gap:
        prem = "NO-BASIN"
    else:
        prem = "UNCLEAR"
    sp_ = (np.percentile([fin(s, "PSaf")["e_sign"] for s in seeds], [0, 100]))
    print(f"  premise: PW1 trapped {trap_w1}/{n} (TRAP={TRAP}); cold runs trapped {capture}/{3 * n}; "
          f"median depth e_var(PSaf)-e_var(PW0) = {depth:.5f} (0.05 gap = {0.05 * gap:.5f}) -> {prem}")
    print(f"  spread: PSaf e_sign range over seeds [{sp_[0]:.2e}, {sp_[1]:.2e}]; "
          f"median e_sign PSaf {_med([fin(s, 'PSaf')['e_sign'] for s in seeds]):.2e}, "
          f"PW0 {_med([fin(s, 'PW0')['e_sign'] for s in seeds]):.2e}, "
          f"U {_med([fin(s, 'U')['e_sign'] for s in seeds if 'U' in D[s]]):.2e}, "
          f"U->P {_med([D[s]['U']['projected']['e_sign'] for s in seeds if 'U' in D[s]]):.2e}")
    budget_bad = [f"{s}:{k}={D[s][k]['steps']}" for s in seeds for k in req
                  if k not in ORACLE and D[s][k]["steps"] != T]
    out = {}
    for R in ("A", "O"):
        b = {s: min(fin(s, "PSaf")["e_sign"], fin(s, "N")["e_sign"]) for s in seeds}
        rho = {s: (b[s] - fin(s, R)["e_sign"]) / b[s] if b[s] > 0 else float("nan") for s in seeds}
        pos = sum(rho[s] > 0 for s in seeds)
        mrho = _med(rho.values())
        dEv = _med([fin(s, R)["e_var"] - min(fin(s, "PSaf")["e_var"], fin(s, "N")["e_var"]) for s in seeds])
        trapR = sum(fin(s, R)["trapped"] for s in seeds)
        trap_ctrl = min(sum(fin(s, "PSaf")["trapped"] for s in seeds), sum(fin(s, "N")["trapped"] for s in seeds))
        gates = []
        if budget_bad:
            gates.append("G-budget " + ",".join(budget_bad))
        if not trapR < trap_ctrl:
            gates.append(f"G-ctrl trapped {trapR} not < control {trap_ctrl}")
        if R == "A":
            dec = [h["agree"] for s in seeds for h in D[s]["A"]["info"]["history"]]
            fr = sum(dec) / max(len(dec), 1)
            print(f"  A sampled-vs-exact acceptance agreement {fr:.2f} over {len(dec)} cycles")
            if fr < 0.8:
                gates.append(f"G-sampled agreement {fr:.2f} < 0.8")
        else:
            rho0 = {s: (b[s] - fin(s, "O0")["e_sign"]) / b[s] for s in seeds}
            dlt = _med([rho[s] - rho0[s] for s in seeds])
            better = sum(rho[s] > rho0[s] for s in seeds)
            print(f"  O ablation: median rho_O - rho_O0 = {dlt:.3f}, better on {better}/{n}")
            if not (dlt >= 0.10 and better >= n - 1):
                gates.append(f"G-ablation penalty adds nothing (median {dlt:.3f}, {better}/{n})")
            ch = [c for s in seeds for c in D[s]["O"]["info"]["fid_checks"]]
            good = sum(abs(c["F_hat"] - c["F"]) <= max(0.02, 3 * c["se"]) and c["se"] <= 0.05 for c in ch)
            fr = good / max(len(ch), 1)
            print(f"  O sampled fidelity estimator OK at {good}/{len(ch)} checkpoints "
                  f"(max se {max([c['se'] for c in ch] + [0]):.3f})")
            if fr < 0.9:
                gates.append(f"G-sampled fidelity estimator OK at {fr:.2f} < 0.9")
        print(f"  route {R}: median rho={mrho:.3f} rho>0 on {pos}/{n} dEvar={dEv:+.4f} "
              f"trapped {trapR} vs control {trap_ctrl}  gates: {'OK' if not gates else '; '.join(gates)}")
        if prem == "NO-BASIN":
            v = "N/A (premise falsified: no basin after projection)"
        elif prem == "UNCLEAR":
            v = "UNTESTED (premise unclear)"
        elif gates or mrho < 0.10 or pos <= n / 2:
            v = "FAIL"
        elif mrho >= 0.25 and pos >= n - 1 and dEv <= 0.01:
            v = "PASS"
        else:
            v = "AMBIGUOUS (add 6 seeds; kill unless 12-seed set passes)"
        print(f"  -> route {R}: {v}")
        out[R] = v
    return prem, out


def aggregate(d, a):
    d = d if os.path.isabs(d) else os.path.join(HERE, d)
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    if not rows:
        print("no results"); return 1
    prem, routes = {}, {}
    for inst in sorted({r["instance"] for r in rows}):
        p, rv = aggregate_instance(inst, [r for r in rows if r["instance"] == inst], a)
        if inst != "smoke":
            prem[inst], routes[inst] = p, rv
    if not prem:
        print("only smoke results; no global verdict"); return 0
    ps = set(prem.values())
    gp = "FALSIFIED" if ps == {"NO-BASIN"} else "CONFIRMED" if ps == {"BASIN"} else "MIXED/INCOMPLETE"
    print(f"GLOBAL PREMISE (basin after projection): {gp}  {prem}")
    for R in ("A", "O"):
        vs = [routes[i][R] for i in routes]
        if all(v == "PASS" for v in vs):
            g = "PASS"
        elif any(v == "FAIL" for v in vs):
            g = "FAIL"
        else:
            g = "NOT PASSED: " + "; ".join(f"{i}={routes[i][R]}" for i in routes)
        print(f"GLOBAL ROUTE {R}: {g}")
    return 0


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P1.3 round-3 test (see module docstring)")
    ap.add_argument("--instance", choices=list(INST), default="c1")
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], default="PSaf")
    ap.add_argument("--task-id", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--relax", type=int, default=40)
    ap.add_argument("--null-k", type=int, default=3)
    ap.add_argument("--probe-at", type=int, default=600)
    ap.add_argument("--probe", type=int, default=300)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--pretrain", type=int, default=300)
    ap.add_argument("--acc-samples", type=int, default=2000)
    ap.add_argument("--final-window", type=int, default=100)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--o-chunk", type=int, default=256)
    ap.add_argument("--fn-iters", type=int, default=4)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--lbfgs-iters", type=int, default=3000)
    ap.add_argument("--min-seeds", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--checks-only", type=int, choices=[0, 1], default=0)
    ap.add_argument("--aggregate", default="")
    a = ap.parse_args(argv)
    if a.task_id >= 0:
        if a.task_id > 119:
            ap.error("--task-id must be in 0..119")
        a.instance = ("c1", "c2")[a.task_id // 60]
        a.seed = 1234 + (a.task_id % 60) // 10
        a.arm = ARMS[a.task_id % 10]
    if not a.out:
        a.out = os.path.join(HERE, "results", f"{a.instance}_s{a.seed}_{a.arm}.json")
    # budget consistency: every non-oracle arm must fit its schedule into T
    if a.probe_at + a.probe + 2 * a.relax + 2 > a.budget or proposal_steps(a) > a.budget:
        ap.error("schedule exceeds --budget")
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.aggregate:
        return aggregate(a.aggregate, a)
    log("args: " + json.dumps(vars(a)))
    if not run_checks(a.seed):
        log("correctness checks failed; aborting")
        return 2
    if a.checks_only:
        return 0
    if os.path.exists(a.out) and not a.overwrite:
        log(f"{a.out} exists; use --overwrite 1"); return 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
