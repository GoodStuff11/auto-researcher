#!/usr/bin/env python3
"""
P2.1 round-5 decisive test: at a fixed, narrow parameter budget, does spending the sign capacity
as a SUM of K configuration-dependent rank-1 terms over one carried reference (SK-LR "sk1", and
its pairing-frame twin SK-AGP "skagp") keep the sign quality of symmetry-projected NNBF (with or
without a Jastrow factor), at O(N_e) cost per connection?
================================================================================================

What changed from round 3 (round-3 critic, items 3 and 5-9)
------------------------------------------------------------
* Item 3: "the restriction bites" was false. det(A0 + phi c^T) = det A0 (1 + c^T A0^-1 phi), so a
  rank-1 term with an unrestricted c(n) already represents an arbitrary scalar function times the
  reference. Rank alone is NOT binding. The hypothesis is now about how a fixed parameter budget P
  is split: a narrow sign trunk (width --hs-sign = 6) feeding K low-rank terms, plus a positive
  amplitude net taking the rest of P, versus NNBF (all of P in the backflow) and NNBF x Jastrow.
  The premise arm hf1 (one rank-1 term, same split) is kept as a CAPACITY premise: if hf1 already
  matches the reference, the K-term sum is unnecessary (REDUNDANT).
* Item 6: new control "nnbfj" = NNBF of width --nnbfj-width (default 4) x a tanh Jastrow whose
  width fills the same P. Ref_X = the better (smaller median) of nnbf and nnbfj, per metric X.
* Item 7: seeds are PAIRED (same seed index across arms); rho_s and rho_t are medians of per-seed
  ratios against the Ref arm. The rank ratio sigma (sk1 versus the single rank-4 term hf4) and its
  paired dominance are REPORTED ONLY (not in the verdict), since rank is not binding.
* Item 8: the cost check gates on the time PER CONNECTION measured at large N_e (up to L = 48,
  N_e = 2016) and on the measured fast/brute per-connection speedup at L = 32 (N_e = 896), not on
  a fitted total-time exponent at sizes where interpreter overhead dominates.
* Item 9: every sampled arm must reproduce the trained projected vector per amplitude to
  per_amp_check <= 1e-10, else the instance is AMBIGUOUS (guard).
* Item 5: the verdict needs >= 2 counted instances; one counted instance gives UNINFORMATIVE.

Model (tj_core.TJ): H = -sum_{ij,s} T_ij P c^dag_is c_js P + J sum_<ij> (S_i.S_j - n_i n_j / 4),
t = 1, t' on diagonals, J on NN bonds, no double occupancy. Instances (INSTANCES below; sector
dimensions in ../P2.1_subcubic_amplitudes_r3/sector_dims_output.txt and screen_output.txt):
  0 : 4x4 OBC 6+6, t' = -0.25, J = 0.4, D = 1681680, |G| = 8, dim range(P_S P_chi) = 30220.
      Projected NNBF width 6 (P = 1734): median dE_sign = 0.0125 t over seeds 11/22/33. QUALIFIES.
  1 : 4x4 cylinder 6+6 (sector 15268): 0.0038 t. 2 : 4x4 OBC 7+7 (sector 6540): 0.0028 t (seed
      11). Neither qualifies; kept for the record only.
  3 : 4x4 OBC 5+5 (6 holes), D = 2018016, |G| = 8, sector 42302, P = 1478 (ratio 28.6), ED gap 0.013.
  4 : 5x3 OBC 6+6 (3 holes), D = 420420, |G| = 4, sector 15080, P = 1626 (ratio 9.3), ED gap 0.025.
      3 and 4 are the round-5 candidates for the second qualifying instance (screen_output.txt).
      They are screened by a one-seed pilot (submit_pilot.sbatch).
  9 : 3x2 smoke cluster (projected sector dimension 5, every arm exact: smoke verdicts are
      meaningless by design).
The decisive array runs on PRODUCTION = (0, --new-instance). Hard Core: doped, t' != 0, 2D,
non-stoquastic; no verified same-sector competitor. SIGN HALF OF THE HARD CORE ONLY.

Families (tj_core.LRAnsatz; psi = exp(J(n)) sum_k w_k det B_k(n), J a separate tanh net):
  full : NNBF, B_{k,s} = (Phi0_{k,s} + out_{k,s}(n))[occ_s]                 (Luo & Clark)
  lr   : B_{k,s} = Phi0_s[occ_s] + Phih_{k,s}[occ_s] C_{k,s}(n), one shared reference Phi0
         (SK-LR; K = 1 is HFDS / Liu-Clark with a Jastrow and a frozen hidden block)
  agp  : B_k = F0[occ_up, occ_dn] + Phih_k[occ_up] C_k(n)[:, occ_dn], shared pairing reference
         F0 (SK-AGP; r = 0 is projected BCS x neural Jastrow, the Foil)
References: Phi0 = UHF orbitals of the U = 8 Hubbard model on the same lattice (seed 1234);
F0 = d-wave BCS pairing function (tj_core.dwave_bcs_F).
Arms (all at P = parameter count of NNBF width --match-width = 6, within one hidden unit):
  nnbf     full K=1, width 6                 P_chi     control (Ref candidate)
  nnbfj    full K=1, width 4, + Jastrow      P_chi     control (Ref candidate; critic item 6)
  hf1      lr K=1 r=1                        P_chi     capacity premise
  hf4      lr K=1 r=4                        P_chi     equal-rank comparison (reported only)
  sk1      lr K=4 r=1                        P_chi     ROUTE 3.1 (SK-LR)
  sk2      lr K=4 r=2                        P_chi     rank trend (reported only)
  sjnet    lr r=0 (Slater x neural J)        P_chi     Gate-2 control for 3.1
  agpj     agp r=0 (BCS x neural J, Foil)    P_chi     Gate-2 control and premise for 3.2
  agh4     agp K=1 r=4                       P_chi     equal-rank comparison for 3.2 (reported)
  skagp    agp K=4 r=1                       P_chi     ROUTE 3.2 (SK-AGP)
  nnbf_PS, sk1_PS   trained with P_S P_chi (oracle spin projection)
  nnbf_U,  sk1_U    unprojected training, budgets also after post-hoc projection
Training: SR with the EXACT force and a sampled metric (--sr-iters, step 0.05, shift 1e-3,
2000 samples, trust region |step| <= 1, best-energy iterate kept); projected arms train
psi_P = P_chi psi_theta with projected log-derivatives in the metric; "_PS" arms use the metric of
P_chi psi_theta (a preconditioner approximation; energy and force exact).
Budgets: p11_core.budget (P1.1 definitions; E_sign upper estimate = min of 4 L-BFGS-B starts and
the ten Haaf fixed node) for "P" (trained P_chi psi_theta) and "P_S" (post-hoc P_S P_chi
psi_theta); "PS" for _PS arms; "U", "UP", "UP_S" for unprojected arms.
Sampled arm (SAMPLED arms, state "P"): Metropolis from |psi_P|^2; reports var_ex, kurtosis, tail
share, binned stderr, z = |mean - E| / stderr and per_amp_check (max relative deviation of 200
visited amplitudes recomputed per amplitude on the |G| images from the trained vector).

Pre-committed decision rule (--mode aggregate)
----------------------------------------------
For instance I and metric X in {dE_sign, dE_tot} (state "P"): Ref arm A_X(I) = the one of
{nnbf, nnbfj} with the smaller median X; Ref_X = that median.
PREMISE GATE (instance I counts): median dE_sign(nnbf) >= 0.01 t (the qualification bar used to
  select the instance) AND Ref_sign >= 0.005 t (the better control still keeps a sign error).
  Fewer than 2 counted instances -> both routes UNINFORMATIVE (critic item 5).
Paired ratios: rho_X = median over seeds s of X(route, s) / X(A_X, s).
Guards (else AMBIGUOUS): 3 seeds of route, premise arm and both controls; calib_err <= 1e-3
  dE_tot; multi-start spread < 0.05; per_amp_check <= 1e-10 for every sampled seed of the route
  and of the controls.
ROUTE 3.1 (sk1; premise hf1; Gate-2 control sjnet), on every counted instance:
  REDUNDANT (-> FAIL) if beta = med dE_sign(hf1) / Ref_sign < 1.3
  PASS if rho_s <= 1.10 and rho_t <= 1.15 and gamma = med dE_sign(sk1)/med dE_sign(sjnet) <= 0.6
       and post-hoc-spin rho_s(P_S) <= 1.15 and oracle med dE_sign(sk1_PS)/med dE_sign(nnbf_PS)
       <= 1.15 and v = med var_ex(sk1)/med var_ex(Ref_sign arm) <= 2 and median z <= 4,
       on every counted instance, and the lr cost check PASSes
  FAIL if on any counted instance rho_s >= 1.30, rho_t >= 1.40, gamma >= 0.85, oracle >= 1.30,
       v >= 4 or median z > 6; or the lr cost check FAILs
  AMBIGUOUS otherwise. Action: 3 more seeds (44, 55, 66) for sk1, hf1, nnbf and nnbfj on the
       counted instances; if still AMBIGUOUS, FAIL.
ROUTE 3.2 (skagp; premise and Gate-2 control agpj = the Foil): the same rule with no oracle arm
  and the agp cost check.
Reported only: sigma = paired median dE_sign(route)/dE_sign(equal-rank arm) with paired
dominance (of 3), rho of every arm, rank trend (hf1, hf4, sk1, sk2), U versus U->P, <S^2>,
overlaps, f_E, eC_tot, seed spreads, sector dims.

Cost check (--mode cost; one thread; folded into the verdict; critic item 8)
---------------------------------------------------------------------------
L x L PBC t-t'-J, t' = -0.25, J = 0.4, delta = 1/8, random configurations, random weights; SK-LR
(lr, K=4, r=1, Hs=64, Ha=128) and SK-AGP (agp, same sizes). Per size: n_conn (hops + exchanges),
t_c = T_E / n_conn (carried-table time per connection), t_sw = T_sweep / n_prop (SK-LR only),
t_cb = T_brute / n_conn for L <= --brute-max, eps (carried-table E_loc after the sweep versus
brute force), drift (carried tables versus a fresh setup).
Per family: PASS if eps <= 1e-8 and drift <= 1e-8 at every size, the slope of t_c in N_e over the
three largest sizes is in [0.5, 1.3] (O(N_e) per connection, i.e. C_E = O(N_e^2); a slope below
0.5 means interpreter overhead still dominates, which is AMBIGUOUS), (lr) the slope of
t_sw is <= 2.3, and the speedup t_cb / t_c at the largest brute size is >= 20. FAIL if eps or
drift > 1e-6, slope(t_c) >= 1.8, (lr) slope(t_sw) >= 2.8, or speedup < 5. AMBIGUOUS otherwise;
action: rerun with --sizes 32,48,56,64 --brute-max 32; if still AMBIGUOUS, FAIL.
Total-time slopes and the brute per-connection slope are reported, not gated.

Usage
-----
  python3 micro_test.py --mode train --task ID [--new-instance I]   one Slurm array task
  python3 micro_test.py --mode train --instance I --arm A --seed S  one explicit task
  python3 micro_test.py --mode cost [--sizes 16,24,32,40,48] [--brute-max 32]
  python3 micro_test.py --mode aggregate [--results DIR] [--new-instance I]
  python3 micro_test.py --mode smoke                                 checks + tiny pipeline
  python3 micro_test.py --mode time --instance I --arm A             one SR step (cost estimate)

Options
-------
  --mode {train,cost,aggregate,smoke,time}  What to run (default smoke).
  --task INT        Array index 0..(2*3*NARM - 1) for --mode train: instance =
                    PRODUCTION[task // (3*NARM)] with PRODUCTION = (0, --new-instance),
                    arm = ARM_LIST[(task % (3*NARM)) // 3], seed = SEEDS[task % 3]
                    (NARM = 14; ARM_LIST below; SEEDS = 11, 22, 33). Explicit --instance /
                    --arm / --seed override the decoded values.
  --new-instance INT  The second production instance (default 3).
  --instance INT    0-4 (see above) or 9 (smoke cluster).
  --arm NAME        One of ARM_LIST.
  --seed INT        Network-initialisation and SR-sampling seed.
  --K INT           Terms of sk1, sk2, skagp (default 4).
  --hs-sign INT     Sign-trunk width of every lr/agp arm with r > 0 (default 6).
  --match-width INT NNBF width whose parameter count every arm matches (default 6).
  --nnbfj-width INT Backflow width of the nnbfj control (default 4; its Jastrow fills P).
  --sr-iters INT (1200)  --sr-lr FLOAT (0.05)  --sr-shift FLOAT (1e-3)  --sr-samples INT (2000)
  --chunk INT       Configurations per forward chunk (default 32768).
  --o-chunk INT     Samples per chunk of projected log-derivatives (default 8; x |G| images).
  --lbfgs-iters INT (4000)  --fn-iters INT (10)  --fn-floor FLOAT (1e-3)  --r-min FLOAT (1.5):
                    budget settings, as P1.1.
  --mc-samples INT  Measurements in the sampled arm (default 20000; 0 disables it).
  --mc-burn INT     Burn-in measurements (default 2000).
  --sizes LIST      --mode cost: comma-separated L (default 16,24,32,40,48).
  --brute-max INT   --mode cost: largest L with brute force (default 32).
  --nconf INT       --mode cost: configurations per size (default 2).
  --results DIR     Results directory (default results/ next to this script).
  --overwrite {0,1} Recompute an existing result JSON (default 0).

Outputs
-------
  train: results/inst{I}_{arm}_s{seed}.json (params, arm kwargs, n_params, ED levels, irrep, |G|,
         sector dims, calib_err, budgets, sampled stats, timing) and results/params/*.npz.
  cost:  results/cost_check.json (rows, per-connection slopes, speedup, per-family verdicts).
  aggregate: printed table and verdicts; results/aggregate.json.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p11_core import budget, energy, sign_constrained_min, solve_ed, uhf_orbitals  # noqa: E402
from tj_core import (TJ, LRAnsatz, LRCarried, SpaceGroup, connections, dwave_bcs_F,  # noqa: E402
                     eloc_brute, eloc_fast_agp)
import scipy.sparse as sp  # noqa: E402

INSTANCES = {0: dict(Lx=4, Ly=4, bcx="obc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4),
             1: dict(Lx=4, Ly=4, bcx="pbc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4),
             2: dict(Lx=4, Ly=4, bcx="obc", bcy="obc", nup=7, ndn=7, tp=-0.25, J=0.4),
             3: dict(Lx=4, Ly=4, bcx="obc", bcy="obc", nup=5, ndn=5, tp=-0.25, J=0.4),   # screen h6
             4: dict(Lx=5, Ly=3, bcx="obc", bcy="obc", nup=6, ndn=6, tp=-0.25, J=0.4),   # screen r53n6
             9: dict(Lx=3, Ly=2, bcx="pbc", bcy="obc", nup=2, ndn=2, tp=-0.25, J=0.4)}
ARM_LIST = ["nnbf", "nnbfj", "hf1", "hf4", "sk1", "sk2", "sjnet", "agpj", "agh4", "skagp",
            "nnbf_PS", "sk1_PS", "nnbf_U", "sk1_U"]
NARM = len(ARM_LIST)
SEEDS = [11, 22, 33]
PRODUCTION = (0, "new")   # the decisive array: instance 0 and --new-instance
UHF_SEED, F0_SEED = 1234, 1234
ARM_DEF = {
    "nnbf": dict(mode="full", K=1, free=None, proj="chi"),
    "nnbfj": dict(mode="full", K=1, Hs="nnbfj", free="Ha", proj="chi"),
    "hf1": dict(mode="lr", K=1, r=1, free="Ha", proj="chi"),
    "hf4": dict(mode="lr", K=1, r=4, free="Ha", proj="chi"),
    "sk1": dict(mode="lr", K="K", r=1, free="Ha", proj="chi"),
    "sk2": dict(mode="lr", K="K", r=2, free="Ha", proj="chi"),
    "sjnet": dict(mode="lr", K=1, r=0, Hs=0, free="Ha", proj="chi"),
    "agpj": dict(mode="agp", K=1, r=0, Hs=0, free="Ha", proj="chi"),
    "agh4": dict(mode="agp", K=1, r=4, free="Ha", proj="chi"),
    "skagp": dict(mode="agp", K="K", r=1, free="Ha", proj="chi"),
    "nnbf_PS": dict(mode="full", K=1, free=None, proj="spin_chi"),
    "sk1_PS": dict(mode="lr", K="K", r=1, free="Ha", proj="spin_chi"),
    "nnbf_U": dict(mode="full", K=1, free=None, proj="none"),
    "sk1_U": dict(mode="lr", K="K", r=1, free="Ha", proj="none"),
}
SAMPLED = {"nnbf", "nnbfj", "hf1", "hf4", "sk1", "skagp", "agpj"}
# pre-committed thresholds
RHO_S_PASS, RHO_T_PASS, GAMMA_PASS = 1.10, 1.15, 0.6
SPIN_PASS, V_PASS, Z_PASS = 1.15, 2.0, 4.0
RHO_S_FAIL, RHO_T_FAIL, GAMMA_FAIL, SPIN_FAIL, V_FAIL, Z_FAIL = 1.30, 1.40, 0.85, 1.30, 4.0, 6.0
BITE = 1.3            # capacity premise: med dE_sign(premise arm) / Ref_sign must be >= BITE
QUALIFY = 0.01        # instance qualifies if med dE_sign(nnbf) >= 0.01 t (selection bar)
PREMISE_REF = 0.005   # ... and the better control (nnbf or nnbfj) keeps Ref_sign >= 0.005 t
MIN_COUNTED = 2       # critic item 5: fewer counted instances -> UNINFORMATIVE
CALIB_TOL, SPREAD_TOL, AMP_TOL = 1e-3, 0.05, 1e-10
# cost check (critic item 8): per-connection time at large N_e and measured speedup
EPS_C, SLOPE_C, SLOPE_SW, SPEEDUP, SLOPE_GUARD = 1e-8, 1.3, 2.3, 20.0, 0.5
EPS_CF, SLOPE_CF, SLOPE_SWF, SPEEDUP_F = 1e-6, 1.8, 2.8, 5.0
ROUTES = {"3.1": dict(arm="sk1", rank="hf4", premise="hf1", gate2="sjnet", oracle=("sk1_PS", "nnbf_PS"),
                      cost="lr"),
          "3.2": dict(arm="skagp", rank="agh4", premise="agpj", gate2="agpj", oracle=None, cost="agp")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ setup
def make_model(k):
    c = INSTANCES[k]
    return TJ(c["Lx"], c["Ly"], c["bcx"], c["bcy"], c["nup"], c["ndn"], 1.0, c["tp"], c["J"])


def resolve_ground(m, Vv, deg):
    """psi_0 with definite characters under translations x {Mx, My} inside a degenerate level."""
    if deg == 1:
        return Vv[:, 0]
    sg = SpaceGroup(m, diag=False)
    gens = [k for k, e in enumerate(sg.elems) if e in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0),
                                                        (0, 0, 1, 0, 0), (0, 0, 0, 1, 0))]
    Q = Vv[:, :deg]
    Ms = [Q.T @ np.column_stack([sg.apply(Q[:, i], k) for i in range(deg)]) for k in gens]
    rng = np.random.default_rng(7)
    A = sum(rng.standard_normal() * Mg for Mg in Ms)
    w, U = np.linalg.eig(A)
    for j in range(deg):
        q = np.real(U[:, j]); q /= np.linalg.norm(q)
        if all(np.linalg.norm(Mg @ q - np.round(q @ Mg @ q) * q) < 1e-8 for Mg in Ms):
            return Q @ q
    raise RuntimeError("could not resolve a 1-D irrep inside the degenerate ground level")


def build_symmetry(m, psi0):
    for diag in (True, False):
        sg = SpaceGroup(m, diag=diag)
        try:
            gen = sg.set_from(psi0)
            return sg, gen
        except RuntimeError:
            continue
    raise RuntimeError("psi_0 not in a 1-D irrep of the abelian subgroup")


def sector_dims(m, sg, S0):
    """Exact dimensions of range(P_chi) (character sum (1/|G|) sum_g chi_g tr U_g) and of
    range(P_S P_chi) for S = S0 = |S_z| = 0: d(S=0, chi) = d(chi, S_z=0) - d(chi, S_z=1), using
    the (N_up+1, N_dn-1) sector with the same group elements and characters (valid because S^+
    commutes with every U_g, checked in run_checks)."""
    def dchi(model, group):
        tr = [float(np.sum((group.inv[k] == np.arange(model.D)) * group.sinv[k])) for k in range(group.G)]
        return float(np.dot(sg.chi, tr) / group.G)
    d0 = dchi(m, sg)
    dS = float("nan")
    if S0 == 0 and m.ndn > 0:
        m1 = TJ(m.Lx, m.Ly, m.bcx, m.bcy, m.nup + 1, m.ndn - 1, m.t, m.tp, m.J, build_H=False)
        g1 = SpaceGroup(m1, diag=any(e[4] == 1 for e in sg.elems))
        assert [tuple(e) for e in g1.elems] == [tuple(e) for e in sg.elems]
        dS = d0 - dchi(m1, g1)
    return d0, dS


def references(m):
    Pu, Pd = uhf_orbitals(m, UHF_SEED)
    F0 = dwave_bcs_F(m, seed=F0_SEED)
    return (Pu, Pd), F0


def arm_kwargs(arm, m, args, refs):
    d = ARM_DEF[arm]
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=args.match_width).n_params()
    r = d.get("r", 0)
    hs = d.get("Hs", args.hs_sign if r > 0 else 0)
    if hs == "nnbfj":
        hs = args.nnbfj_width
    kw = dict(mode=d["mode"], K=(args.K if d["K"] == "K" else d["K"]), r=r,
              shared_ref=d.get("shared_ref", True), Hs=hs, Ha=0)
    if d["free"] is None:
        kw.update(Hs=args.match_width)
        return kw, target
    ref = refs[1] if d["mode"] == "agp" else refs[0]
    f = d["free"]

    def cnt(w):
        k2 = dict(kw); k2[f] = w
        return LRAnsatz(m, ref, **k2).n_params()
    c1, c2 = cnt(1), cnt(2)
    kw[f] = int(round((target - (c1 - (c2 - c1))) / (c2 - c1)))
    if kw[f] < 1:
        raise RuntimeError(f"arm {arm}: cannot match parameter count {target}")
    return kw, target


def make_ansatz(arm, m, args, refs, seed):
    kw, target = arm_kwargs(arm, m, args, refs)
    ref = refs[1] if kw["mode"] == "agp" else refs[0]
    return LRAnsatz(m, ref, seed=seed, **kw), kw, target


# ============================================================================ SR
def state_of(ans, m, sg, proj, S, chunk):
    psi_t = ans.amplitudes(chunk)
    if proj == "none":
        return psi_t, psi_t, psi_t
    psi_c = sg.project(psi_t)
    psi = m.spin_project(psi_c, S) if proj == "spin_chi" else psi_c
    return psi_t, psi_c, psi


def sr_step(ans, m, sg, proj, S, args, rng):
    psi_t, psi_c, psi = state_of(ans, m, sg, proj, S, args.chunk)
    E, Hp, nn = energy(m.H, psi)
    f = ans.force((Hp - E * psi) / nn, args.chunk)
    pc = psi_c ** 2
    idx = rng.choice(len(pc), size=args.sr_samples, p=pc / pc.sum())
    O = ans.per_sample_O(idx) if proj == "none" else ans.per_sample_O_proj(sg, idx, psi_c, args.o_chunk)
    Oc = (O - O.mean(0)) / np.sqrt(len(idx))
    lam = args.sr_shift
    Km = Oc @ Oc.T + lam * np.eye(len(idx))
    dx = (f - Oc.T @ np.linalg.solve(Km, Oc @ f)) / lam
    step = args.sr_lr * dx
    nrm = np.linalg.norm(dx)
    if nrm * args.sr_lr > 1.0:
        step *= 1.0 / (nrm * args.sr_lr)
    return E, f, step


def optimize(ans, m, sg, proj, S, args, rng, label):
    best_E, best_x = np.inf, ans.flat().copy()
    t0 = time.time()
    for it in range(args.sr_iters):
        x0 = ans.flat().copy()
        E, f, step = sr_step(ans, m, sg, proj, S, args, rng)
        if not np.isfinite(E):
            log(f"{label}: non-finite energy at it {it}; restoring best")
            ans.set_flat(best_x)
            break
        if E < best_E:
            best_E, best_x = E, x0
        ans.set_flat(x0 - step)
        if it % 50 == 0 or it == args.sr_iters - 1:
            log(f"{label}: it {it:5d}  E = {E:.8f}  |f| = {np.linalg.norm(f):.2e}  "
                f"({(time.time() - t0) / (it + 1):.2f} s/step)")
    E = energy(m.H, state_of(ans, m, sg, proj, S, args.chunk)[2])[0]
    if not (E <= best_E):
        ans.set_flat(best_x)
    return (time.time() - t0) / max(args.sr_iters, 1)


# ============================================================================ sampled arm
def bond_list(m):
    return [(i, j) for i in range(m.V) for j in range(i + 1, m.V) if m.T[i, j] != 0.0]


def sampled_arm(ans, m, sg, psiP, nsamp, nburn, rng):
    """Metropolis from |psi_P|^2 (P_chi psi_theta); local energy from the exact H row."""
    psiP = psiP / np.linalg.norm(psiP)
    Hp = m.H @ psiP
    E = float(psiP @ Hp)
    with np.errstate(divide="ignore", invalid="ignore"):
        Eloc = np.where(np.abs(psiP) > 0, Hp / psiP, 0.0)
    p = psiP ** 2
    var_ex = float(p @ (Eloc - E) ** 2)
    kurt = float(p @ (Eloc - E) ** 4) / var_ex ** 2
    sig = np.sqrt(var_ex)
    tail = float(p[np.abs(Eloc - E) > 5 * sig] @ ((Eloc - E)[np.abs(Eloc - E) > 5 * sig] ** 2)) / var_ex
    bonds = bond_list(m)
    one = np.int64(1)
    cur = int(rng.choice(m.D, p=p))
    bu, bd = int(m.bu[cur]), int(m.bd[cur])
    Ne = m.nup + m.ndn
    meas, visited, acc, prop = [], [], 0, 0
    for s in range(nburn + nsamp):
        for _ in range(Ne):
            i, j = bonds[rng.integers(len(bonds))]
            si = ((bu >> i) & 1) - ((bd >> i) & 1)   # +1 up, -1 down, 0 empty
            sj = ((bu >> j) & 1) - ((bd >> j) & 1)
            prop += 1
            if si == sj:
                continue
            nbu, nbd = bu, bd
            for site, val in ((i, si), (j, sj)):
                nbu &= ~(one << site); nbd &= ~(one << site)
            for site, val in ((j, si), (i, sj)):
                if val == 1:
                    nbu |= one << site
                elif val == -1:
                    nbd |= one << site
            k = int(m.index(np.array([nbu]), np.array([nbd]))[0])
            if rng.random() < min(1.0, p[k] / p[cur]):
                cur, bu, bd = k, nbu, nbd
                acc += 1
        if s >= nburn:
            meas.append(Eloc[cur])
            if len(visited) < 200:
                visited.append(cur)
    x = np.array(meas)
    # binning analysis: max stderr over block sizes with >= 32 blocks
    se = x.std() / np.sqrt(len(x))
    b = 1
    while len(x) // (2 * b) >= 32:
        b *= 2
        nb = len(x) // b
        xb = x[:nb * b].reshape(nb, b).mean(1)
        se = max(se, xb.std() / np.sqrt(nb))
    vis = np.array(visited)
    pa = ans.psiP_per_amplitude(sg, vis)
    scale = np.linalg.norm(sg.project(ans.amplitudes(32768)))
    chk = float(np.max(np.abs(pa / scale - psiP[vis])) / np.max(np.abs(psiP[vis])))
    return dict(E_exact=E, mean=float(x.mean()), stderr=float(se), z=float(abs(x.mean() - E) / se),
                var_mc=float(x.var()), var_ex=var_ex, kurt_ex=kurt, tail_share=tail,
                max_dev_sigma=float(np.max(np.abs(x - E)) / sig), accept=acc / max(prop, 1),
                n=len(x), per_amp_check=chk)


# ============================================================================ train task
def train_task(args):
    m = make_model(args.instance)
    m.spin_setup()
    m.pair_setup(args.r_min)
    log(f"instance {args.instance} {INSTANCES[args.instance]} D={m.D} arm={args.arm} seed={args.seed}")
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 4, 1e-8, UHF_SEED)
    psi0 = resolve_ground(m, Vv, deg)
    sg, gen = build_symmetry(m, psi0)
    S2_0 = m.S2_expect(psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * S2_0) - 1)))
    dim_chi, dim_S_chi = sector_dims(m, sg, S0)
    log(f"projected sector dimensions: dim range(P_chi) = {dim_chi:.0f}, dim range(P_S P_chi) = {dim_S_chi:.0f}")
    log(f"ED levels {W.tolist()} deg {deg}; |G|={sg.G} chi(psi_0)={gen}; S(psi_0)={S0:g} ({time.time() - t0:.0f} s)")
    rng_b = np.random.default_rng(args.seed + 1000003)
    S0m = sp.diags(np.where(psi0 >= 0, 1.0, -1.0))
    Ec, _, _ = sign_constrained_min((S0m @ m.H @ S0m).tocsr(),
                                    [("uniform", np.ones(m.D) + 0.01 * rng_b.random(m.D))], args.lbfgs_iters)
    calib = float(Ec - W[0])
    refs = references(m)
    ans, kw, target = make_ansatz(args.arm, m, args, refs, args.seed)
    proj = ARM_DEF[args.arm]["proj"]
    psi_t, psi_c, psi = state_of(ans, m, sg, proj, S0, args.chunk)
    init_ratio = float(np.linalg.norm(psi) / np.linalg.norm(psi_t))
    log(f"arm {args.arm}: {kw} n_params={ans.n_params()} target={target} proj={proj} "
        f"calib={calib:.2e} init |P psi|/|psi|={init_ratio:.3e}")
    rng = np.random.default_rng(args.seed + 1000 * ARM_LIST.index(args.arm))
    t1 = time.time()
    sps = optimize(ans, m, sg, proj, S0, args, rng, args.arm)
    t_sr = time.time() - t1
    psi_t, psi_c, psi = state_of(ans, m, sg, proj, S0, args.chunk)
    if proj == "chi":
        states = [("P", psi_c), ("P_S", m.spin_project(psi_c, S0))]
    elif proj == "spin_chi":
        states = [("PS", psi)]
    else:
        states = [("U", psi_t), ("UP", sg.project(psi_t)), ("UP_S", m.spin_project(sg.project(psi_t), S0))]
    budgets = {}
    for lab, v in states:
        nrm = np.linalg.norm(v)
        if not np.isfinite(nrm) or nrm < 1e-10 * np.linalg.norm(psi_t):
            budgets[lab] = None
            continue
        t2 = time.time()
        Vref = psi0[:, None] if deg > 1 else Vv
        b = budget(m, v, W, Vref, 1 if deg > 1 else deg, args, rng_b)
        vn = v / nrm
        b.update(S2=m.S2_expect(vn), ov0=float((psi0 @ vn) ** 2))
        for k_ in ("P0", "Ptheta", "Psign"):
            b.pop(k_, None)
        budgets[lab] = b
        log(f"{args.arm} [{lab}]: dE_tot={b['dE_tot']:.5f} dE_sign={b['dE_sign']:.5f} f_E={b['f_E_up']:.3f} "
            f"eC={b['eC_tot']:.2e} spread={b['spread']:.1e} S2={b['S2']:.3f} ov0={b['ov0']:.3f} "
            f"({time.time() - t2:.0f} s)")
    sampled = None
    if args.arm in SAMPLED and args.mc_samples > 0:
        t3 = time.time()
        sampled = sampled_arm(ans, m, sg, psi_c, args.mc_samples, args.mc_burn,
                              np.random.default_rng(args.seed + 77))
        log(f"{args.arm} sampled: " + " ".join(f"{k}={v:.4g}" for k, v in sampled.items())
            + f" ({time.time() - t3:.0f} s)")
    pdir = os.path.join(args.results, "params")
    os.makedirs(pdir, exist_ok=True)
    np.savez(os.path.join(pdir, f"inst{args.instance}_{args.arm}_s{args.seed}.npz"), flat=ans.flat(),
             layout=json.dumps([[k, list(ans.p[k].shape)] for k in ans.keys]))
    return dict(params={k: v for k, v in vars(args).items()}, instance=args.instance, arm=args.arm,
                seed=args.seed, proj=proj, arm_kwargs=kw, n_params=ans.n_params(), P_target=target,
                ed_levels=W.tolist(), degeneracy=deg, irrep=gen, G=sg.G, S_psi0=S0, calib_err=calib,
                init_proj_ratio=init_ratio, dim_chi=dim_chi, dim_S_chi=dim_S_chi, budgets=budgets, sampled=sampled, t_sr=t_sr, s_per_step=sps)


# ============================================================================ cost check
def random_tj_config(V, nh, rng):
    perm = rng.permutation(V)
    n = (V - nh) // 2
    ou = np.zeros(V, int); od = np.zeros(V, int)
    ou[perm[:n]] = 1; od[perm[n:2 * n]] = 1
    return ou, od


class _Dims:
    pass


def random_ansatz(mode, L, n, T, coords, seed, K=4, r=1, Hs=64, Ha=128, tp=-0.25):
    rng = np.random.default_rng(seed)
    m = _Dims(); m.V, m.nup, m.ndn = L * L, n, n
    m.Lx = m.Ly = L; m.t, m.tp, m.coords = 1.0, tp, coords
    if mode == "agp":
        ref = dwave_bcs_F(m, noise=0.05, seed=seed)
    else:
        e, v = np.linalg.eigh(-T)
        ref = (v[:, :n] + 0.1 * rng.standard_normal((m.V, n)) / np.sqrt(m.V),
               v[:, :n] + 0.1 * rng.standard_normal((m.V, n)) / np.sqrt(m.V))
    a = LRAnsatz(m, ref, mode, K=K, r=r, Hs=Hs, Ha=Ha, seed=seed, w2_scale=0.3)
    a.p["w2a"] = 0.1 * rng.standard_normal(Ha)
    a.p["wk"] = 1 + 0.3 * rng.standard_normal(K)
    return a


def _slope(rows, key):
    rr = [q for q in rows if q.get(key)]
    if len(rr) < 2:
        return float("nan")
    return float(np.polyfit(np.log([q["Ne"] for q in rr]), np.log([q[key] for q in rr]), 1)[0])


def cost_check(args):
    """Critic item 8: gate on the time per connection at large N_e and on the measured speedup
    over brute force per connection, not on total-time exponents at overhead-dominated sizes."""
    from p11_core import build_lattice
    out = dict(rows={}, slopes={}, speedup={}, verdict={})
    for fam in ("lr", "agp"):
        rows = []
        for L in args.size_list:
            V = L * L
            nh = int(round(V / 8))
            if (V - nh) % 2:
                nh += 1
            n = (V - nh) // 2
            T, coords, bonds = build_lattice(L, L, "pbc", "pbc", 1.0, -0.25)
            nn = sorted({(min(i, j), max(i, j)) for (i, j, _) in bonds})
            a = random_ansatz(fam, L, n, T, coords, args.seed + L)
            rng = np.random.default_rng(args.seed + 7 * L)
            ts, tc, tsw, tcb, eps, drift, ncs = [], [], [], [], 0.0, 0.0, []
            for _ in range(args.nconf):
                ou, od = random_tj_config(V, nh, rng)
                if fam == "lr":
                    t0 = time.perf_counter(); c = LRCarried(a, T, nn, 0.4, ou, od); ts.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); c.sweep(rng, 2 * n); tsw.append((time.perf_counter() - t0) / (2 * n))
                    t0 = time.perf_counter(); Ef = c.eloc(); te = time.perf_counter() - t0
                    ou2, od2 = c.occ["u"].copy(), c.occ["d"].copy()
                    drift = max(drift, c.drift())
                else:
                    Ef, t_set, te = eloc_fast_agp(a, T, nn, 0.4, ou, od)
                    ts.append(t_set); ou2, od2 = ou, od
                _, hops, exch = connections(T, nn, 0.4, ou2, od2)
                nc = len(hops) + len(exch)
                ncs.append(nc); tc.append(te / nc)
                if L <= args.brute_max:
                    t0 = time.perf_counter()
                    Eb = eloc_brute(a, T, nn, 0.4, ou2, od2)
                    tcb.append((time.perf_counter() - t0) / nc)
                    eps = max(eps, abs(Ef - Eb) / (1 + abs(Eb)))
            row = dict(L=L, V=V, Ne=2 * n, n_conn=float(np.median(ncs)), T_setup=float(np.median(ts)),
                       t_c=float(np.median(tc)), t_sw=float(np.median(tsw)) if tsw else None,
                       t_cb=float(np.median(tcb)) if tcb else None, eps=eps if tcb else None, drift=drift)
            rows.append(row)
            log(f"cost[{fam}] L={L:3d} Ne={2 * n:4d} n_conn={row['n_conn']:.0f} setup={row['T_setup']:.3e} "
                f"t_c={row['t_c']:.3e} t_sw={row['t_sw'] if tsw else float('nan'):.3e} "
                f"t_cb={row['t_cb'] if tcb else float('nan'):.3e} eps={row['eps'] if tcb else float('nan'):.1e} "
                f"drift={drift:.1e}")
        big = rows[-3:]
        bb = [q for q in rows if q["t_cb"]]
        sl = dict(setup=_slope(big, "T_setup"), t_c=_slope(big, "t_c"), t_sw=_slope(big, "t_sw"),
                  t_cb=_slope(bb[-3:], "t_cb"))
        spd = bb[-1]["t_cb"] / next(q["t_c"] for q in rows if q["L"] == bb[-1]["L"]) if bb else float("nan")
        e_max = max([q["eps"] for q in rows if q["eps"] is not None] or [float("nan")])
        d_max = max(q["drift"] for q in rows)
        sw_ok = fam != "lr" or sl["t_sw"] <= SLOPE_SW
        sw_bad = fam == "lr" and sl["t_sw"] >= SLOPE_SWF
        if not (e_max <= EPS_CF) or d_max > EPS_CF or sl["t_c"] >= SLOPE_CF or sw_bad or not (spd >= SPEEDUP_F):
            v = "FAIL"
        elif (e_max <= EPS_C and d_max <= EPS_C and SLOPE_GUARD <= sl["t_c"] <= SLOPE_C and sw_ok
              and spd >= SPEEDUP):
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        out["rows"][fam], out["slopes"][fam], out["speedup"][fam], out["verdict"][fam] = rows, sl, spd, v
        out.setdefault("eps", {})[fam] = e_max
        out.setdefault("drift", {})[fam] = d_max
        log(f"cost[{fam}] slopes " + " ".join(f"{k}={v_:.2f}" for k, v_ in sl.items())
            + f" speedup(L={bb[-1]['L'] if bb else 0})={spd:.1f} eps={e_max:.1e} drift={d_max:.1e} -> {v}")
    return out


# ============================================================================ checks
def run_checks(seed):
    ok = True
    m2 = TJ(2, 1, "obc", "obc", 1, 1, 1.0, 0.0, 0.4)
    e2 = np.linalg.eigvalsh(m2.H.toarray())[0]
    log(f"check two-site t-J singlet energy {e2:.12f} (exact -J = -0.4)"); ok &= abs(e2 + 0.4) < 1e-12
    m = make_model(9)
    m.spin_setup(); m.pair_setup(1.0)
    H = m.H
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(m.D)
    hv = np.linalg.norm(H @ v)
    e_herm = abs(H - H.T).max()
    e_s2 = np.linalg.norm(m.S2(H @ v) - H @ m.S2(v)) / hv
    W, U = np.linalg.eigh(H.toarray())
    psi0 = U[:, 0]
    sg, gen = build_symmetry(m, psi0)
    e_g = max(np.linalg.norm(sg.apply(H @ v, k) - H @ sg.apply(v, k)) for k in range(sg.G)) / hv
    Pv = sg.project(v)
    e_idem = np.linalg.norm(sg.project(Pv) - Pv) / np.linalg.norm(Pv)
    e_p0 = np.linalg.norm(sg.project(psi0) - psi0)
    S0 = float(round(0.5 * (np.sqrt(1 + 4 * m.S2_expect(psi0)) - 1)))
    Ps = m.spin_project(v, S0)
    e_ps = np.linalg.norm(m.spin_project(Ps, S0) - Ps) / np.linalg.norm(Ps)
    e_comm = np.linalg.norm(m.spin_project(sg.project(v), S0) - sg.project(m.spin_project(v, S0))) / np.linalg.norm(v)
    log(f"check smoke t-J: D={m.D} herm {e_herm:.1e} [H,S2] {e_s2:.1e} [H,U_g] {e_g:.1e} (|G|={sg.G}, chi={gen}) "
        f"P_chi idem {e_idem:.1e} P_chi psi0-psi0 {e_p0:.1e} P_S idem {e_ps:.1e} [P_S,P_chi] {e_comm:.1e}")
    ok &= max(e_herm, e_s2, e_g, e_idem, e_p0, e_ps, e_comm) < 1e-10
    # S^+ commutes with U_g (needed by sector_dims), and sector_dims equals the projector rank
    m1 = TJ(m.Lx, m.Ly, m.bcx, m.bcy, m.nup + 1, m.ndn - 1, m.t, m.tp, m.J, build_H=False)
    g1 = SpaceGroup(m1, diag=any(e[4] == 1 for e in sg.elems))
    e_sp = max(np.linalg.norm(m.Sp @ sg.apply(v, k) - g1.apply(m.Sp @ v, k)) for k in range(sg.G))
    d0, dS = sector_dims(m, sg, S0)
    Pd = np.column_stack([m.spin_project(sg.project(e), S0) for e in np.eye(m.D)])
    rk_c = np.linalg.matrix_rank(np.column_stack([sg.project(e) for e in np.eye(m.D)]), 1e-8)
    rk_s = np.linalg.matrix_rank(Pd, 1e-8)
    log(f"check [S+,U_g] {e_sp:.1e}; sector dims: chi {d0:.1f} (rank {rk_c}), S=0 & chi {dS:.1f} (rank {rk_s})")
    ok &= e_sp < 1e-10 and abs(d0 - rk_c) < 1e-6 and abs(dS - rk_s) < 1e-6
    refs = references(m)
    iu, idd = m.iu_id()
    nn = m.nn
    for name, kw, ref in [("full K=2", dict(mode="full", K=2, Hs=4, Ha=3, shared_ref=False), refs[0]),
                          ("lr K=3 r=2", dict(mode="lr", K=3, r=2, Hs=5, Ha=3), refs[0]),
                          ("lr r=0", dict(mode="lr", K=1, r=0, Hs=0, Ha=3), refs[0]),
                          ("agp K=3 r=2", dict(mode="agp", K=3, r=2, Hs=5, Ha=3), refs[1]),
                          ("agp r=0", dict(mode="agp", K=1, r=0, Hs=0, Ha=3), refs[1])]:
        a = LRAnsatz(m, ref, seed=seed, w2_scale=0.3, **kw)
        if a.Ha:
            a.p["w2a"] = 0.3 * rng.standard_normal(a.Ha)
        if a.K > 1:
            a.p["wk"] = 1 + 0.5 * rng.standard_normal(a.K)
        x0 = a.flat().copy()
        h = 1e-6
        d = rng.standard_normal(len(x0)); d /= np.linalg.norm(d)
        for proj in ("none", "chi", "spin_chi"):
            psi_t, psi_c, psi = state_of(a, m, sg, proj, S0, 4096)
            E, Hp, nn_ = energy(H, psi)
            g = 2 * a.force((Hp - E * psi) / nn_, 4096)
            Ef = lambda x: (a.set_flat(x), energy(H, state_of(a, m, sg, proj, S0, 4096)[2])[0])[1]
            fd = (Ef(x0 + h * d) - Ef(x0 - h * d)) / (2 * h)
            a.set_flat(x0)
            e1 = abs(g @ d - fd) / max(1.0, abs(fd))
            ok &= e1 < 1e-6
            msg = f"check grad [{name}, proj={proj}]: analytic {g @ d:.10e} fd {fd:.10e}"
            if proj == "chi":
                idx = rng.choice(m.D, size=5, p=psi ** 2 / (psi @ psi))
                O = a.per_sample_O_proj(sg, idx, psi_c, 2)
                lp = lambda x: (a.set_flat(x), sg.project(a.amplitudes(4096))[idx])[1]
                fdO = (lp(x0 + h * d) - lp(x0 - h * d)) / (2 * h) / psi_c[idx]
                a.set_flat(x0)
                eO = np.max(np.abs(O @ d - fdO)) / max(1.0, np.max(np.abs(fdO)))
                pa = a.psiP_per_amplitude(sg, idx)
                epa = np.max(np.abs(pa - psi_c[idx])) / np.max(np.abs(psi_c[idx]))
                msg += f"; O_P vs fd {eO:.1e}; per-amplitude psi_P {epa:.1e}"
                ok &= eO < 1e-5 and epa < 1e-12
            log(msg)
        if kw["mode"] in ("lr", "agp"):
            psi = a.amplitudes(4096)
            Hpsi = H @ psi
            err = errb = 0.0
            for j in range(m.D):
                ou = m.occ_u[iu[j]].astype(int); od = m.occ_d[idd[j]].astype(int)
                ref_ = Hpsi[j] / psi[j]
                ef = LRCarried(a, m.T, nn, m.J, ou, od).eloc() if kw["mode"] == "lr" else \
                    eloc_fast_agp(a, m.T, nn, m.J, ou, od)[0]
                eb = eloc_brute(a, m.T, nn, m.J, ou, od)
                err = max(err, abs(ef - ref_) / (1 + abs(ref_))); errb = max(errb, abs(eb - ref_) / (1 + abs(ref_)))
            log(f"check fast E_loc == (H psi)/psi on all {m.D} configs [{name}]: fast {err:.1e} brute {errb:.1e}")
            ok &= err < 1e-9 and errb < 1e-9
    # carried tables across a Metropolis sweep (lr), 12x12 random weights
    from p11_core import build_lattice
    L = 6
    T, coords, bonds = build_lattice(L, L, "pbc", "pbc", 1.0, -0.25)
    nn6 = sorted({(min(i, j), max(i, j)) for (i, j, _) in bonds})
    a = random_ansatz("lr", L, 16, T, coords, seed)
    ou, od = random_tj_config(L * L, 4, rng)
    c = LRCarried(a, T, nn6, 0.4, ou, od)
    acc = c.sweep(rng, 200)
    Ef = c.eloc()
    Eb = eloc_brute(a, T, nn6, 0.4, c.occ["u"], c.occ["d"])
    dr = c.drift()
    log(f"check 6x6 sweep: {acc}/200 accepted, drift {dr:.1e}, carried E_loc vs brute {abs(Ef - Eb) / (1 + abs(Eb)):.1e}")
    ok &= acc > 5 and dr < 1e-10 and abs(Ef - Eb) / (1 + abs(Eb)) < 1e-10
    log(f"ALL CHECKS {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


# ============================================================================ aggregate
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def aggregate(d, new_instance=3):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "inst*_*_s*.json"))):
        r = json.load(open(f))
        R.setdefault(r["instance"], {}).setdefault(r["arm"], {})[r["seed"]] = r
    out = dict(instances={}, routes={})
    lab_of = {a: ("P" if ARM_DEF[a]["proj"] == "chi" else ("PS" if ARM_DEF[a]["proj"] == "spin_chi" else "U"))
              for a in ARM_DEF}

    def by_seed(I, arm, key, lab=None):
        lab = lab or lab_of[arm]
        rr = R.get(I, {}).get(arm, {})
        return {s: rr[s]["budgets"][lab][key] for s in sorted(rr) if rr[s]["budgets"].get(lab)}

    def vals(I, arm, key, lab=None):
        return list(by_seed(I, arm, key, lab).values())

    def paired(I, arm, ref_arm, key, lab=None):
        """Critic item 7: median over seeds of the per-seed ratio (same seed index in both arms)."""
        x, y = by_seed(I, arm, key, lab), by_seed(I, ref_arm, key, lab)
        rat = [x[s] / y[s] for s in x if s in y and y[s] > 0]
        return _med(rat), int(sum(x[s] < y[s] for s in x if s in y)), len(rat)

    def guard(I, arm):
        rr = R.get(I, {}).get(arm, {})
        msgs = []
        for s in sorted(rr):
            b = rr[s]["budgets"].get(lab_of[arm])
            if b is None:
                msgs.append(f"{arm} s{s}: null state"); continue
            if rr[s]["calib_err"] > CALIB_TOL * b["dE_tot"]:
                msgs.append(f"{arm} s{s}: calib {rr[s]['calib_err']:.1e}")
            if not b["spread"] < SPREAD_TOL:
                msgs.append(f"{arm} s{s}: spread {b['spread']:.2e}")
            smp = rr[s].get("sampled")
            if arm in SAMPLED and smp is not None and not smp["per_amp_check"] <= AMP_TOL:
                msgs.append(f"{arm} s{s}: per_amp_check {smp['per_amp_check']:.1e} > {AMP_TOL:g}")
        if len(rr) < 3:
            msgs.append(f"{arm}: {len(rr)} seeds")
        return msgs

    insts = sorted(k for k in R if k in INSTANCES)
    ref = {}
    for I in insts:
        tab = {}
        Ns = INSTANCES[I]["nup"]
        # Ref arm per metric: the better (smaller median) of nnbf and nnbfj (critic item 6)
        refarm = {}
        for key in ("dE_sign", "dE_tot"):
            cands = [(_med(vals(I, c, key)), c) for c in ("nnbf", "nnbfj") if c in R.get(I, {})]
            cands = [q for q in cands if np.isfinite(q[0])]
            refarm[key] = min(cands)[1] if cands else "nnbf"
        rs = _med(vals(I, refarm["dE_sign"], "dE_sign"))
        rt = _med(vals(I, refarm["dE_tot"], "dE_tot"))
        rsS = _med(vals(I, refarm["dE_sign"], "dE_sign", "P_S"))
        q0 = _med(vals(I, "nnbf", "dE_sign"))
        ref[I] = dict(sign=rs, tot=rt, signPS=rsS, qualify=q0, arm_sign=refarm["dE_sign"], arm_tot=refarm["dE_tot"])
        anyr = next(iter(next(iter(R[I].values())).values()))
        print(f"instance {I} {INSTANCES[I]}: med dE_sign(nnbf)={q0:.5f} Ref_sign={rs:.5f} ({refarm['dE_sign']}) "
              f"Ref_tot={rt:.5f} ({refarm['dE_tot']}) Ref_sign(P_S)={rsS:.5f} "
              f"dim(P_S P_chi)={anyr.get('dim_S_chi', float('nan')):.0f} P_target={anyr.get('P_target')}")
        for arm in ARM_LIST:
            if arm not in R.get(I, {}):
                continue
            ds, dt = vals(I, arm, "dE_sign"), vals(I, arm, "dE_tot")
            isP = lab_of[arm] == "P"
            row = dict(n=len(ds), dE_sign=_med(ds), dE_tot=_med(dt),
                       rho_s=paired(I, arm, refarm["dE_sign"], "dE_sign")[0] if isP else _med(ds) / rs,
                       rho_t=paired(I, arm, refarm["dE_tot"], "dE_tot")[0] if isP else _med(dt) / rt,
                       spread_sign=(max(ds) - min(ds)) / _med(ds) if ds else float("nan"),
                       f_E=_med(vals(I, arm, "f_E_up")), eC=_med(vals(I, arm, "eC_tot")),
                       S2=_med(vals(I, arm, "S2")), ov0=_med(vals(I, arm, "ov0")),
                       r_over_N=ARM_DEF[arm].get("r", Ns) / Ns if ARM_DEF[arm]["mode"] != "full" else 1.0)
            if isP:
                row["rho_s_PS"] = paired(I, arm, refarm["dE_sign"], "dE_sign", "P_S")[0]
            if lab_of[arm] == "U":
                row["dE_sign_UP"] = _med(vals(I, arm, "dE_sign", "UP"))
                row["dE_tot_UP"] = _med(vals(I, arm, "dE_tot", "UP"))
            smp = [R[I][arm][s]["sampled"] for s in sorted(R[I][arm]) if R[I][arm][s].get("sampled")]
            if smp:
                row["var_ex"] = _med([q["var_ex"] for q in smp]); row["z"] = _med([q["z"] for q in smp])
                row["kurt"] = _med([q["kurt_ex"] for q in smp]); row["tail"] = _med([q["tail_share"] for q in smp])
                row["per_amp_check"] = max(q["per_amp_check"] for q in smp)
            tab[arm] = row
            print(f"  {arm:>8s} n={row['n']} dE_sign={row['dE_sign']:.5f} dE_tot={row['dE_tot']:.5f} "
                  f"rho_s={row['rho_s']:.3f} rho_t={row['rho_t']:.3f} spread={row['spread_sign']:.2f} "
                  f"f_E={row['f_E']:.3f} r/N={row['r_over_N']:.2f}"
                  + (f" rho_s(P_S)={row['rho_s_PS']:.3f}" if "rho_s_PS" in row else "")
                  + (f" UP: dE_sign={row['dE_sign_UP']:.5f}" if "dE_sign_UP" in row else "")
                  + (f" var_ex={row['var_ex']:.4g} z={row['z']:.2f} kurt={row['kurt']:.3g} tail={row['tail']:.3f}"
                     f" amp_chk={row['per_amp_check']:.1e}" if "var_ex" in row else ""))
        out["instances"][I] = tab
        out.setdefault("ref", {})[I] = ref[I]
    cost = None
    cf = os.path.join(d, "cost_check.json")
    if os.path.exists(cf):
        cost = json.load(open(cf))
        print(f"cost check: {cost['verdict']} slopes {cost['slopes']} speedup {cost.get('speedup')}")
    counted = {I: bool(ref[I]["qualify"] >= QUALIFY and ref[I]["sign"] >= PREMISE_REF) for I in insts}
    out["counted"] = counted
    print(f"premise gate (med dE_sign(nnbf) >= {QUALIFY:g} t and Ref_sign >= {PREMISE_REF:g} t): " + ", ".join(
        f"instance {I}: {ref[I]['qualify']:.4g}/{ref[I]['sign']:.4g} -> {'counts' if counted[I] else 'DOES NOT COUNT'}"
        for I in insts))
    smoke = insts == [9]
    prod = (0, new_instance)
    primary = [I for I in insts if (I in prod or smoke) and (counted[I] or smoke)]
    for name, rt_ in ROUTES.items():
        arm, rk, pm, g2 = rt_["arm"], rt_["rank"], rt_["premise"], rt_["gate2"]
        need = (arm, pm, g2, "nnbf", "nnbfj")
        if len(primary) < (1 if smoke else MIN_COUNTED):
            v = f"UNINFORMATIVE ({len(primary)} counted instance(s); {MIN_COUNTED} needed)"
            out["routes"][name] = dict(verdict=v)
            print(f"ROUTE {name} ({arm}) VERDICT: {v}")
            continue
        if any(a not in R.get(I, {}) for I in primary for a in need):
            out["routes"][name] = dict(verdict="INCOMPLETE")
            print(f"ROUTE {name} ({arm}) VERDICT: INCOMPLETE")
            continue
        cv = cost["verdict"][rt_["cost"]] if cost else "MISSING"
        tags, per = [], {}
        for I in primary:
            T1 = out["instances"][I]
            ra = ref[I]["arm_sign"]
            q = dict(beta=T1[pm]["dE_sign"] / ref[I]["sign"], rho_s=T1[arm]["rho_s"], rho_t=T1[arm]["rho_t"],
                     gamma=T1[arm]["dE_sign"] / T1[g2]["dE_sign"], rho_s_PS=T1[arm].get("rho_s_PS", float("nan")))
            if rk in R.get(I, {}):
                q["sigma_reported"], q["dominance_reported"], _ = paired(I, arm, rk, "dE_sign")
            orc = None
            if rt_["oracle"]:
                orc = (_med(vals(I, rt_["oracle"][0], "dE_sign")) / _med(vals(I, rt_["oracle"][1], "dE_sign"))
                       if all(o in R.get(I, {}) for o in rt_["oracle"]) else float("nan"))
            q["oracle"] = orc
            q["v"] = T1[arm].get("var_ex", float("nan")) / T1[ra].get("var_ex", float("nan"))
            q["z"] = T1[arm].get("z", float("nan"))
            g = guard(I, arm) + guard(I, pm) + guard(I, "nnbf") + guard(I, "nnbfj")
            q["guards"] = g
            fail = (q["rho_s"] >= RHO_S_FAIL or q["rho_t"] >= RHO_T_FAIL or q["gamma"] >= GAMMA_FAIL
                    or (orc is not None and np.isfinite(orc) and orc >= SPIN_FAIL)
                    or q["v"] >= V_FAIL or q["z"] > Z_FAIL)
            pas = (not g and q["rho_s"] <= RHO_S_PASS and q["rho_t"] <= RHO_T_PASS and q["gamma"] <= GAMMA_PASS
                   and q["rho_s_PS"] <= SPIN_PASS and (orc is None or orc <= SPIN_PASS)
                   and q["v"] <= V_PASS and q["z"] <= Z_PASS)
            t = "FAIL" if fail else ("PASS" if pas else "AMBIGUOUS")
            if q["beta"] < BITE:
                t = "REDUNDANT"
            tags.append(t)
            per[I] = dict(tag=t, **q)
            print(f"ROUTE {name} ({arm}) instance {I}: " + " ".join(
                f"{k}={v_:.3f}" if isinstance(v_, float) else f"{k}={v_}" for k, v_ in q.items()) + f" -> {t}")
        if any(t == "REDUNDANT" for t in tags):
            v = "FAIL (REDUNDANT: one rank-1 term at the same parameter split already matches the reference)"
        elif any(t == "FAIL" for t in tags) or cv == "FAIL":
            v = "FAIL"
        elif all(t == "PASS" for t in tags) and cv == "PASS":
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        out["routes"][name] = dict(verdict=v, cost=cv, per_instance=per)
        print(f"ROUTE {name} ({arm}) cost={cv} VERDICT: {v}")
    with open(os.path.join(d, "aggregate.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return 0


# ============================================================================ timing
def time_step(args):
    m = make_model(args.instance)
    m.spin_setup()
    t0 = time.time()
    W, Vv, deg = solve_ed(m, 4, 1e-8, UHF_SEED)
    psi0 = resolve_ground(m, Vv, deg)
    sg, gen = build_symmetry(m, psi0)
    log(f"instance {args.instance} D={m.D} ED+group {time.time() - t0:.0f}s |G|={sg.G} chi={gen} deg={deg}")
    refs = references(m)
    for arm in (args.arm.split(",") if args.arm else ARM_LIST):
        ans, kw, target = make_ansatz(arm, m, args, refs, 11)
        rng = np.random.default_rng(0)
        t0 = time.time()
        sr_step(ans, m, sg, ARM_DEF[arm]["proj"], 0.0, args, rng)
        log(f"{arm}: n_params={ans.n_params()} (target {target}) kw={kw}: one SR step {time.time() - t0:.2f}s")


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.1 round-5 test (see module docstring)")
    ap.add_argument("--mode", choices=["train", "cost", "aggregate", "smoke", "time"], default="smoke")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--instance", type=int, default=-1, choices=[-1] + sorted(INSTANCES))
    ap.add_argument("--new-instance", type=int, default=3, choices=[k for k in INSTANCES if k not in (0, 9)])
    ap.add_argument("--nnbfj-width", type=int, default=4)
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--hs-sign", type=int, default=6)
    ap.add_argument("--match-width", type=int, default=6)
    ap.add_argument("--sr-iters", type=int, default=1200)
    ap.add_argument("--sr-lr", type=float, default=0.05)
    ap.add_argument("--sr-shift", type=float, default=1e-3)
    ap.add_argument("--sr-samples", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--o-chunk", type=int, default=8)
    ap.add_argument("--lbfgs-iters", type=int, default=4000)
    ap.add_argument("--fn-iters", type=int, default=10)
    ap.add_argument("--fn-floor", type=float, default=1e-3)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--mc-samples", type=int, default=20000)
    ap.add_argument("--mc-burn", type=int, default=2000)
    ap.add_argument("--sizes", default="16,24,32,40,48")
    ap.add_argument("--brute-max", type=int, default=32)
    ap.add_argument("--nconf", type=int, default=2)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0:
        if a.task >= len(PRODUCTION) * 3 * NARM:
            ap.error(f"--task must be in 0..{len(PRODUCTION) * 3 * NARM - 1}")
        if a.instance < 0:
            a.instance = (0, a.new_instance)[a.task // (3 * NARM)]
        if not a.arm:
            a.arm = ARM_LIST[(a.task % (3 * NARM)) // 3]
        if a.seed < 0:
            a.seed = SEEDS[a.task % 3]
    if a.seed < 0:
        a.seed = SEEDS[0]
    if a.mode == "train" and (a.instance < 0 or a.arm not in ARM_DEF):
        ap.error("--mode train needs --task, or --instance and a valid --arm")
    a.size_list = [int(s) for s in a.sizes.split(",") if s.strip()]
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.mode == "aggregate":
        return aggregate(args.results, args.new_instance)
    if args.mode == "time":
        time_step(args)
        return 0
    if args.mode == "cost":
        res = cost_check(args)
        json.dump(res, open(os.path.join(args.results, "cost_check.json"), "w"), indent=1, default=float)
        return 0
    if args.mode == "smoke":
        t0 = time.time()
        if not run_checks(UHF_SEED):
            return 2
        sdir = args.results
        for arm in ARM_LIST:
            for seed in SEEDS:
                sm = parse_args(["--mode", "train", "--instance", "9", "--arm", arm, "--seed", str(seed),
                                 "--sr-iters", "20", "--sr-samples", "60", "--lbfgs-iters", "300",
                                 "--fn-iters", "2", "--match-width", "8", "--hs-sign", "3",
                                 "--mc-samples", "400", "--mc-burn", "50", "--r-min", "1.0",
                                 "--results", sdir])
                res = train_task(sm)
                json.dump(res, open(os.path.join(sdir, f"inst9_{arm}_s{seed}.json"), "w"), indent=1, default=float)
        sc = parse_args(["--mode", "cost", "--sizes", "6,8,10", "--brute-max", "10", "--nconf", "1"])
        json.dump(cost_check(sc), open(os.path.join(sdir, "cost_check.json"), "w"), indent=1, default=float)
        aggregate(sdir, args.new_instance)
        log(f"smoke total {time.time() - t0:.1f} s")
        return 0
    out = os.path.join(args.results, f"inst{args.instance}_{args.arm}_s{args.seed}.json")
    if os.path.exists(out) and not args.overwrite:
        log(f"{out} exists; use --overwrite 1"); return 0
    if not run_checks(UHF_SEED):
        return 2
    res = train_task(args)
    json.dump(res, open(out, "w"), indent=1, default=float)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
