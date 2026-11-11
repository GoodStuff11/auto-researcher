#!/usr/bin/env python3
"""
P2.1 round-3 decisive test: does the shared-reference low-rank sum (SK-LR, and its geminal-frame
twin SK-AGP) keep NNBF's sign quality where the low-rank restriction actually bites?
================================================================================================

Physics question
----------------
Round 2 tested SK-LR on 4x3 Hubbard clusters with N_sigma in {4, 5} at rank r = 2, i.e. at
r/N_sigma = 0.4-0.5, where a low-rank hop response is almost no restriction, with loose
thresholds and no projected control. This test
  * goes to N_sigma = 7 (4x4 t-t'-J, 2 holes, D = 411840) at rank r = 1 (r/N_sigma = 0.14) and
    keeps a smaller instance (4x3 t-t'-J, 2 holes, N_sigma = 5, D = 16632) for the size trend at
    fixed r;
  * checks that the restriction bites (a single rank-1 hidden-fermion term, "hf1", must be
    measurably worse in sign than projected NNBF), and asks whether a SUM of K = 4 rank-1 terms
    over one shared reference ("sk1") recovers NNBF's sign quality AND beats a single term of the
    same total rank K*r = 4 ("hf4", same per-ratio cost);
  * trains every compared arm WITH space-group projection (projected SR, exact sums), adds
    oracle total-spin-projected training (P_S P_chi) for NNBF and sk1, and reports every state
    after post-hoc spin projection;
  * uses 3 seeds per arm and medians, with a pairwise-dominance rule where arms are compared;
  * runs a genuine Metropolis Monte Carlo arm from |psi_P|^2 for the routes and controls and
    measures the local-energy variance and tails (rule 4 of the round-3 brief);
  * folds the cost check into the verdict: the carried-table local energy, the Metropolis sweep
    with Sherman-Morrison table updates (SK-LR), the per-configuration setup, and brute force,
    on L x L t-J lattices up to L = 32 (N_e = 896), with the brute slope as the regime guard.

Model (tj_core.TJ): H = -sum_{ij,s} T_ij P c^dag_is c_js P + J sum_<ij> (S_i.S_j - n_i n_j / 4),
t = 1, t' = -0.25 (diagonals), J = 0.4 (NN only), PBC, no double occupancy. Instances:
  0 : 4x3 PBC, N_up = N_dn = 5 (2 holes, delta = 1/6), D = 16632. psi_0: non-degenerate,
      trivial irrep of translations x {Mx, My} (|G| = 48), singlet (probe_output.txt).
  1 : 4x4 PBC, N_up = N_dn = 7 (2 holes, delta = 1/8), D = 411840. The ground level is a
      degenerate singlet pair at momenta (pi, 0) and (0, pi), exchanged by the diagonal mirror
      (probe_output.txt; scratch/deg.py). psi_0 is taken as the member with definite
      characters under translations x {Mx, My} (|G| = 64); the projector is onto that 1-D
      irrep, so the diagonal mirror, which maps psi_0 to its partner, is not used.
  9 : 3x2 (PBC x, OBC y), N_up = N_dn = 2, D = 90 (smoke only, not a Hard-Core instance).
Hard Core: both instances are doped, t' != 0, 2D and non-stoquastic. Neither has a verified
same-sector near-degenerate competitor (instance 1's degenerate partner is in another irrep).
THIS TEST COVERS THE SIGN HALF OF THE HARD CORE ONLY.

Families (tj_core.LRAnsatz; psi = exp(J(n)) sum_k w_k det B_k(n), J a separate tanh net):
  full : NNBF, B_{k,s} = (Phi0_{k,s} + out_{k,s}(n))[occ_s]                 (Luo & Clark)
  lr   : B_{k,s} = Phi0_s[occ_s] + Phih_{k,s}[occ_s] C_{k,s}(n), C_{k,s} r x N_s, one shared
         reference Phi0 (SK-LR; K = 1 is HFDS / Liu-Clark with a Jastrow)
  agp  : B_k = F0[occ_up, occ_dn] + Phih_k[occ_up] C_k(n)[:, occ_dn], C_k r x V, shared
         pairing reference F0 (SK-AGP; r = 0 is projected BCS x neural Jastrow, the Foil)
References: Phi0 = UHF orbitals of the U = 8 Hubbard model on the same lattice (seed 1234);
F0 = d-wave BCS pairing function with a small extended-s admixture and noise (tj_core.dwave_bcs_F).
Arms (all at the parameter count of NNBF width --match-width, within one hidden unit; the sign
trunk of every r > 0 low-rank arm has width --hs-sign and the amplitude net takes the rest):
  nnbf     full K=1                         projected (P_chi)          control
  mdnnbf   full K=4 separate refs           projected                  control (unrestricted sum)
  hf1      lr   K=1 r=1                     projected                  bite premise (HFDS rank 1)
  hf4      lr   K=1 r=4                     projected                  equal-rank control for sk1
  sk1      lr   K=4 r=1                     projected                  ROUTE 3.1 (SK-LR)
  sk2      lr   K=4 r=2                     projected                  rank trend
  sjnet    lr   r=0 (Slater x neural J)     projected                  Gate-2 control
  agpj     agp  r=0 (BCS x neural J, Foil)  projected                  Foil / premise for SK-AGP
  agh4     agp  K=1 r=4                     projected                  equal-rank control for skagp
  skagp    agp  K=4 r=1                     projected                  ROUTE 3.2 (SK-AGP)
  nnbf_PS  as nnbf, trained with P_S P_chi (oracle spin projection)
  sk1_PS   as sk1,  trained with P_S P_chi (oracle spin projection)
  nnbf_U   as nnbf, unprojected training (P1.1 protocol), budgets also after post-hoc projection
  sk1_U    as sk1,  unprojected training, budgets also after post-hoc projection
Training: SR with the EXACT force and a sampled metric (P1.1 settings: --sr-iters 1500,
step 0.05, shift 1e-3, 2000 samples, trust region |step| <= 1, best-energy iterate kept).
Projected arms train psi_P = P_chi psi_theta (exact-sum version of Romero, Nys & Carleo,
arXiv:2406.09077); the metric uses the projected log-derivatives O_P(n) computed per
amplitude on the |G| images of each sample. "_PS" arms train psi = P_S P_chi psi_theta (energy and
force exact); their metric is that of P_chi psi_theta (a preconditioner approximation, stated).
Budgets: p11_core.budget (P1.1 definitions: E_sign upper estimate = min of 4 L-BFGS-B starts and
the ten Haaf fixed node; dE_tot, dE_sign, dE_amp, f_E_up, spread, w_sign, pair error eC_tot) for
  projected arms: "P" (the trained P_chi psi_theta) and "P_S" (post-hoc P_S P_chi psi_theta);
  _PS arms: "PS"; unprojected arms: "U", "UP" (post-hoc P_chi) and "UP_S" (post-hoc P_S P_chi).
Sampled arm (nnbf, hf4, sk1, skagp, agpj; state "P"): Metropolis from |psi_P|^2 with symmetric
proposals (swap the contents of the two ends of a random t / t' bond), --mc-samples measurements
every N_e proposals after --mc-burn measurements of burn-in. Amplitudes are read from the trained
P_chi psi_theta vector; 200 visited configurations are re-evaluated per amplitude (forward passes
on the |G| images) and must agree to 1e-10 relative. Reports the exact local-energy variance
var_ex = sum p (E_loc - E)^2, the sampled variance, the exact kurtosis, the tail share (fraction
of var_ex from |E_loc - E| > 5 sigma), the binned standard error and z = |mean - E| / stderr.

Pre-committed decision rule (implemented in --mode aggregate; medians over seeds)
---------------------------------------------------------------------------------
Ref_X(I) = min(med X(nnbf), med X(mdnnbf)) on instance I for X in {dE_sign, dE_tot} (state "P").
rho_s = med dE_sign(route)/Ref_sign, rho_t = med dE_tot(route)/Ref_tot.
Guards (else AMBIGUOUS): every seed of route, nnbf, mdnnbf and the rank control has
calib_err <= 1e-3 dE_tot and spread < 0.05; 3 seeds present.
ROUTE 3.1 (sk1; rank control hf4; premise arm hf1; Gate-2 control sjnet)
  premise  beta = med dE_sign(hf1)/Ref_sign(1) >= 1.3 (the restriction bites); if not -> FAIL
           ("REDUNDANT": a single rank-1 term already matches NNBF, so the sum is unnecessary)
  PASS if  rho_s(0), rho_s(1) <= 1.10 and rho_t(0), rho_t(1) <= 1.15
           and sigma = med dE_sign(sk1)/med dE_sign(hf4) on instance 1 <= 0.85 with
               pairwise dominance #(sk1 seed < hf4 seed in dE_sign) >= 7 of 9
           and gamma = med dE_sign(sk1)/med dE_sign(sjnet) on instance 1 <= 0.6
           and trend rho_s(1) - rho_s(0) <= 0.10 (no growth of the penalty with N at fixed r)
           and spin: rho_s from the post-hoc "P_S" budgets <= 1.15 on instance 1, and the
               oracle ratio med dE_sign(sk1_PS)/med dE_sign(nnbf_PS) <= 1.15
           and sampled: v = med var_ex(sk1)/med var_ex(nnbf) <= 2 and median z <= 4 (inst. 1)
           and the cost check for the lr family is PASS
  FAIL if  premise fails, or on instance 1 rho_s >= 1.30 or rho_t >= 1.40 or sigma >= 1.0 or
           gamma >= 0.85 or oracle ratio >= 1.30, or trend >= 0.25, or v >= 4 or median z > 6,
           or the lr cost check is FAIL
  AMBIGUOUS otherwise. Pre-committed action: 3 more seeds (--seeds 44,55,66) for sk1, hf4,
           nnbf, mdnnbf on instance 1; if still AMBIGUOUS, FAIL.
ROUTE 3.2 (skagp; rank control agh4; premise and Gate-2 control agpj = Foil): the same rule
  with sk1 -> skagp, hf4 -> agh4, hf1 and sjnet -> agpj (premise: med dE_sign(agpj)/Ref_sign(1)
  >= 1.3, i.e. the Foil's sign is measurably worse than NNBF's), no oracle-spin arm (the post-hoc
  P_S check applies), and the agp cost check.
Reported, not in the verdict: rho_s and rho_t of every arm, the rank trend (hf1, hf4, sk1,
sk2 versus r/N_sigma at both sizes), U versus U->P, <S^2>, overlaps, f_E, eC_tot, seed spreads.

Cost check (--mode cost; single thread; folded into the verdict)
-----------------------------------------------------------------
L x L PBC t-t'-J lattices, t' = -0.25, J = 0.4, delta = 1/8, random configurations; SK-LR
(lr, K=4, r=1, Hs=64, Ha=128) and SK-AGP (agp, K=4, r=1, Hs=64, Ha=128) with random weights.
Per size: T_setup (per-configuration table construction, O(V N^2)), T_E (all connections with
carried tables), T_sweep (SK-LR only: N_e Metropolis proposals with accepted-move
Sherman-Morrison updates), drift (carried tables after the sweep versus a fresh setup), eps
(relative difference of the carried-table local energy, AFTER the sweep, from brute force),
T_brute (every connected amplitude rebuilt from scratch, O(N^4)).
Per family: PASS if eps <= 1e-8 and drift <= 1e-8 at every size, slope(T_E) <= 2.4 and (lr)
slope(T_sweep) <= 3.3 over the three largest sizes, with slope(T_brute) >= 3.3 over the three
largest brute sizes as the regime guard. FAIL if eps or drift > 1e-6, or slope(T_E) >= 2.8, or
(lr) slope(T_sweep) >= 3.8. AMBIGUOUS otherwise (also when the guard fails); action: rerun with
--sizes 16,24,32,40 --brute-max 40; if still AMBIGUOUS, the route's verdict is FAIL.
T_setup's slope is reported (expected ~3) and is not gated: the setup is paid once per Markov
chain, and its per-sample equivalent is the sweep.

Usage
-----
  python3 micro_test.py --mode train --task ID [options]   one Slurm array task (0..83)
  python3 micro_test.py --mode cost [--sizes 12,16,20,24,28,32] [--brute-max 32]
  python3 micro_test.py --mode aggregate [--results DIR]
  python3 micro_test.py --mode smoke                        correctness checks + tiny pipeline
  python3 micro_test.py --mode time --instance 1 --arm sk1  one SR step (cost estimate)

Options
-------
  --mode {train,cost,aggregate,smoke,time}  What to run (default smoke).
  --task INT        Array index 0..83 for --mode train: instance = task // 42,
                    arm = ARM_LIST[(task % 42) // 3], seed = SEEDS[task % 3] with
                    ARM_LIST = nnbf, mdnnbf, hf1, hf4, sk1, sk2, sjnet, agpj, agh4, skagp,
                    nnbf_PS, sk1_PS, nnbf_U, sk1_U and SEEDS = 11, 22, 33. Explicit
                    --instance / --arm / --seed override the decoded values.
  --instance INT    0, 1 (production) or 9 (smoke cluster).
  --arm NAME        One of the 14 arm names above.
  --seed INT        Network-initialisation and SR-sampling seed.
  --K INT           Terms of mdnnbf, sk1, sk2, skagp (default 4).
  --hs-sign INT     Sign-trunk width of every lr/agp arm with r > 0 (default 64).
  --match-width INT NNBF width whose parameter count every arm matches (default 128).
  --sr-iters INT (1500)  --sr-lr FLOAT (0.05)  --sr-shift FLOAT (1e-3)  --sr-samples INT (2000)
  --chunk INT       Configurations per forward chunk (default 32768).
  --o-chunk INT     Samples per chunk of projected log-derivatives (default 8; x |G| images).
  --lbfgs-iters INT (4000)  --fn-iters INT (10)  --fn-floor FLOAT (1e-3)  --r-min FLOAT (1.5):
                    budget settings, as P1.1.
  --mc-samples INT  Measurements in the sampled arm (default 20000; 0 disables it).
  --mc-burn INT     Burn-in measurements (default 2000).
  --sizes LIST      --mode cost: comma-separated L (default 12,16,20,24,28,32).
  --brute-max INT   --mode cost: largest L with brute force (default 32).
  --nconf INT       --mode cost: configurations per size (default 2).
  --results DIR     Results directory (default results/ next to this script).
  --overwrite {0,1} Recompute an existing result JSON (default 0).

Outputs
-------
  train: results/inst{I}_{arm}_s{seed}.json: params, arm kwargs, n_params, target, ED levels,
         degeneracy, irrep characters, |G|, S_psi0, calib_err, initial projected-norm ratio,
         budgets {label: P1.1 budget + S2, ov0}, sampled {stats}, t_sr, s_per_step;
         results/params/inst{I}_{arm}_s{seed}.npz (flat parameters + layout).
  cost:  results/cost_check.json (rows, slopes, per-family verdicts).
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

INSTANCES = {0: dict(Lx=4, Ly=3, bcx="pbc", bcy="pbc", nup=5, ndn=5, tp=-0.25, J=0.4),
             1: dict(Lx=4, Ly=4, bcx="pbc", bcy="pbc", nup=7, ndn=7, tp=-0.25, J=0.4),
             9: dict(Lx=3, Ly=2, bcx="pbc", bcy="obc", nup=2, ndn=2, tp=-0.25, J=0.4)}
ARM_LIST = ["nnbf", "mdnnbf", "hf1", "hf4", "sk1", "sk2", "sjnet", "agpj", "agh4", "skagp",
            "nnbf_PS", "sk1_PS", "nnbf_U", "sk1_U"]
SEEDS = [11, 22, 33]
UHF_SEED, F0_SEED = 1234, 1234
ARM_DEF = {
    "nnbf": dict(mode="full", K=1, free=None, proj="chi"),
    "mdnnbf": dict(mode="full", K="K", free="Hs", shared_ref=False, proj="chi"),
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
SAMPLED = {"nnbf", "hf4", "sk1", "skagp", "agpj"}
# pre-committed thresholds
RHO_S_PASS, RHO_T_PASS, RANK_PASS, DOM_PASS, GAMMA_PASS, TREND_PASS = 1.10, 1.15, 0.85, 7, 0.6, 0.10
SPIN_PASS, V_PASS, Z_PASS = 1.15, 2.0, 4.0
RHO_S_FAIL, RHO_T_FAIL, RANK_FAIL, GAMMA_FAIL, TREND_FAIL, SPIN_FAIL, V_FAIL, Z_FAIL = \
    1.30, 1.40, 1.0, 0.85, 0.25, 1.30, 4.0, 6.0
BITE = 1.3
CALIB_TOL, SPREAD_TOL = 1e-3, 0.05
EPS_C, SLOPE_E, SLOPE_SW, BRUTE_GUARD = 1e-8, 2.4, 3.3, 3.3
EPS_CF, SLOPE_EF, SLOPE_SWF = 1e-6, 2.8, 3.8
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


def references(m):
    Pu, Pd = uhf_orbitals(m, UHF_SEED)
    F0 = dwave_bcs_F(m, seed=F0_SEED)
    return (Pu, Pd), F0


def arm_kwargs(arm, m, args, refs):
    d = ARM_DEF[arm]
    target = LRAnsatz(m, refs[0], "full", K=1, Hs=args.match_width).n_params()
    r = d.get("r", 0)
    kw = dict(mode=d["mode"], K=(args.K if d["K"] == "K" else d["K"]), r=r,
              shared_ref=d.get("shared_ref", True), Hs=d.get("Hs", args.hs_sign if r > 0 else 0), Ha=0)
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
                init_proj_ratio=init_ratio, budgets=budgets, sampled=sampled, t_sr=t_sr, s_per_step=sps)


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
    from p11_core import build_lattice
    out = dict(rows={}, slopes={}, verdict={})
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
            ts, te, tsw, tb, eps, drift = [], [], [], [], 0.0, 0.0
            for _ in range(args.nconf):
                ou, od = random_tj_config(V, nh, rng)
                if fam == "lr":
                    t0 = time.perf_counter(); c = LRCarried(a, T, nn, 0.4, ou, od); ts.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); c.sweep(rng, 2 * n); tsw.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); Ef = c.eloc(); te.append(time.perf_counter() - t0)
                    ou2, od2 = c.occ["u"].copy(), c.occ["d"].copy()
                    drift = max(drift, c.drift())
                else:
                    Ef, t_set, t_e = eloc_fast_agp(a, T, nn, 0.4, ou, od)
                    ts.append(t_set); te.append(t_e); ou2, od2 = ou, od
                if L <= args.brute_max:
                    t0 = time.perf_counter()
                    Eb = eloc_brute(a, T, nn, 0.4, ou2, od2)
                    tb.append(time.perf_counter() - t0)
                    eps = max(eps, abs(Ef - Eb) / (1 + abs(Eb)))
            row = dict(L=L, V=V, Ne=2 * n, T_setup=float(np.median(ts)), T_E=float(np.median(te)),
                       T_sweep=float(np.median(tsw)) if tsw else None,
                       T_brute=float(np.median(tb)) if tb else None, eps=eps if tb else None, drift=drift)
            rows.append(row)
            log(f"cost[{fam}] L={L:3d} Ne={2 * n:4d} setup={row['T_setup']:.3e} T_E={row['T_E']:.3e} "
                f"sweep={row['T_sweep'] if tsw else float('nan'):.3e} brute={row['T_brute'] if tb else float('nan'):.3e} "
                f"eps={row['eps'] if tb else float('nan'):.1e} drift={drift:.1e}")
        big = rows[-3:]
        bb = [q for q in rows if q["T_brute"]][-3:]
        sl = dict(setup=_slope(big, "T_setup"), E=_slope(big, "T_E"), sweep=_slope(big, "T_sweep"),
                  brute=_slope(bb, "T_brute"))
        e_max = max([q["eps"] for q in rows if q["eps"] is not None] or [float("nan")])
        d_max = max(q["drift"] for q in rows)
        sw_ok = fam != "lr" or sl["sweep"] <= SLOPE_SW
        sw_bad = fam == "lr" and sl["sweep"] >= SLOPE_SWF
        if not (e_max <= EPS_CF) or d_max > EPS_CF or sl["E"] >= SLOPE_EF or sw_bad:
            v = "FAIL"
        elif e_max <= EPS_C and d_max <= EPS_C and sl["E"] <= SLOPE_E and sw_ok and sl["brute"] >= BRUTE_GUARD:
            v = "PASS"
        else:
            v = "AMBIGUOUS"
        out["rows"][fam], out["slopes"][fam], out["verdict"][fam] = rows, sl, v
        out.setdefault("eps", {})[fam] = e_max
        out.setdefault("drift", {})[fam] = d_max
        log(f"cost[{fam}] slopes " + " ".join(f"{k}={v_:.2f}" for k, v_ in sl.items())
            + f" eps={e_max:.1e} drift={d_max:.1e} -> {v}")
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


def aggregate(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "inst*_*_s*.json"))):
        r = json.load(open(f))
        R.setdefault(r["instance"], {}).setdefault(r["arm"], {})[r["seed"]] = r
    out = dict(instances={}, routes={})
    lab_of = {a: ("P" if ARM_DEF[a]["proj"] == "chi" else ("PS" if ARM_DEF[a]["proj"] == "spin_chi" else "U"))
              for a in ARM_DEF}

    def vals(I, arm, key, lab=None):
        lab = lab or lab_of[arm]
        rr = R.get(I, {}).get(arm, {})
        return [rr[s]["budgets"][lab][key] for s in sorted(rr) if rr[s]["budgets"].get(lab)]

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
        if len(rr) < 3:
            msgs.append(f"{arm}: {len(rr)} seeds")
        return msgs

    insts = sorted(k for k in R if k in (0, 1, 9))
    ref = {}
    for I in insts:
        rs = min(_med(vals(I, "nnbf", "dE_sign")), _med(vals(I, "mdnnbf", "dE_sign")))
        rt = min(_med(vals(I, "nnbf", "dE_tot")), _med(vals(I, "mdnnbf", "dE_tot")))
        rsS = min(_med(vals(I, "nnbf", "dE_sign", "P_S")), _med(vals(I, "mdnnbf", "dE_sign", "P_S")))
        ref[I] = (rs, rt, rsS)
        Ns = INSTANCES[I]["nup"]
        print(f"instance {I} {INSTANCES[I]}: Ref_sign={rs:.5f} Ref_tot={rt:.5f} Ref_sign(P_S)={rsS:.5f}")
        tab = {}
        for arm in ARM_LIST:
            if arm not in R.get(I, {}):
                continue
            ds, dt = vals(I, arm, "dE_sign"), vals(I, arm, "dE_tot")
            row = dict(n=len(ds), dE_sign=_med(ds), dE_tot=_med(dt), rho_s=_med(ds) / rs, rho_t=_med(dt) / rt,
                       spread_sign=(max(ds) - min(ds)) / _med(ds) if ds else float("nan"),
                       f_E=_med(vals(I, arm, "f_E_up")), eC=_med(vals(I, arm, "eC_tot")),
                       S2=_med(vals(I, arm, "S2")), ov0=_med(vals(I, arm, "ov0")),
                       r_over_N=ARM_DEF[arm].get("r", Ns) / Ns if ARM_DEF[arm]["mode"] != "full" else 1.0)
            if lab_of[arm] == "P":
                row["rho_s_PS"] = _med(vals(I, arm, "dE_sign", "P_S")) / rsS
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
    cost = None
    cf = os.path.join(d, "cost_check.json")
    if os.path.exists(cf):
        cost = json.load(open(cf))
        print(f"cost check: {cost['verdict']} slopes {cost['slopes']}")
    prod = [I for I in (0, 1) if I in R]
    big = 1 if 1 in R else (prod[-1] if prod else (9 if 9 in R else None))
    small = 0 if (0 in R and big == 1) else big
    for name, rt_ in ROUTES.items():
        arm, rk, pm, g2 = rt_["arm"], rt_["rank"], rt_["premise"], rt_["gate2"]
        if big is None or any(a not in R.get(big, {}) for a in (arm, rk, pm, g2, "nnbf", "mdnnbf")):
            out["routes"][name] = dict(verdict="INCOMPLETE")
            print(f"ROUTE {name} ({arm}) VERDICT: INCOMPLETE")
            continue
        T1, T0 = out["instances"][big], out["instances"][small]
        q = dict(beta=T1[pm]["rho_s"], rho_s1=T1[arm]["rho_s"], rho_t1=T1[arm]["rho_t"],
                 rho_s0=T0[arm]["rho_s"], rho_t0=T0[arm]["rho_t"],
                 sigma=T1[arm]["dE_sign"] / T1[rk]["dE_sign"],
                 gamma=T1[arm]["dE_sign"] / T1[g2]["dE_sign"],
                 trend=T1[arm]["rho_s"] - T0[arm]["rho_s"], rho_s_PS=T1[arm].get("rho_s_PS", float("nan")))
        a_s, r_s = vals(big, arm, "dE_sign"), vals(big, rk, "dE_sign")
        q["dominance"] = int(sum(x < y for x in a_s for y in r_s))
        if rt_["oracle"] and all(o in R.get(big, {}) for o in rt_["oracle"]):
            q["oracle"] = _med(vals(big, rt_["oracle"][0], "dE_sign")) / _med(vals(big, rt_["oracle"][1], "dE_sign"))
        else:
            q["oracle"] = float("nan") if rt_["oracle"] else None
        q["v"] = T1[arm].get("var_ex", float("nan")) / T1["nnbf"].get("var_ex", float("nan"))
        q["z"] = T1[arm].get("z", float("nan"))
        cv = cost["verdict"][rt_["cost"]] if cost else "MISSING"
        q["cost"] = cv
        g = sum((guard(I, a) for I in {big, small} for a in (arm, rk, "nnbf", "mdnnbf")), [])
        q["guards"] = g
        orc = q["oracle"]
        fail = (q["beta"] < BITE or q["rho_s1"] >= RHO_S_FAIL or q["rho_t1"] >= RHO_T_FAIL
                or q["sigma"] >= RANK_FAIL or q["gamma"] >= GAMMA_FAIL or q["trend"] >= TREND_FAIL
                or (orc is not None and np.isfinite(orc) and orc >= SPIN_FAIL)
                or q["v"] >= V_FAIL or q["z"] > Z_FAIL or cv == "FAIL")
        pas = (not g and q["beta"] >= BITE and max(q["rho_s1"], q["rho_s0"]) <= RHO_S_PASS
               and max(q["rho_t1"], q["rho_t0"]) <= RHO_T_PASS and q["sigma"] <= RANK_PASS
               and q["dominance"] >= DOM_PASS and q["gamma"] <= GAMMA_PASS and q["trend"] <= TREND_PASS
               and q["rho_s_PS"] <= SPIN_PASS and (orc is None or orc <= SPIN_PASS)
               and q["v"] <= V_PASS and q["z"] <= Z_PASS and cv == "PASS")
        v = "FAIL" if fail else ("PASS" if pas else "AMBIGUOUS")
        if q["beta"] < BITE:
            v = "FAIL (REDUNDANT: restriction does not bite)"
        out["routes"][name] = dict(verdict=v, **q)
        print(f"ROUTE {name} ({arm}): " + " ".join(f"{k}={v_:.3f}" if isinstance(v_, float) else f"{k}={v_}"
                                                    for k, v_ in q.items()))
        print(f"ROUTE {name} ({arm}) VERDICT: {v}")
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
    ap = argparse.ArgumentParser(description="P2.1 round-3 test (see module docstring)")
    ap.add_argument("--mode", choices=["train", "cost", "aggregate", "smoke", "time"], default="smoke")
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--instance", type=int, default=-1, choices=[-1, 0, 1, 9])
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--hs-sign", type=int, default=64)
    ap.add_argument("--match-width", type=int, default=128)
    ap.add_argument("--sr-iters", type=int, default=1500)
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
    ap.add_argument("--sizes", default="12,16,20,24,28,32")
    ap.add_argument("--brute-max", type=int, default=32)
    ap.add_argument("--nconf", type=int, default=2)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    a = ap.parse_args(argv)
    if a.task >= 0:
        if a.task > 83:
            ap.error("--task must be in 0..83")
        if a.instance < 0:
            a.instance = a.task // 42
        if not a.arm:
            a.arm = ARM_LIST[(a.task % 42) // 3]
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
        return aggregate(args.results)
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
        aggregate(sdir)
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
