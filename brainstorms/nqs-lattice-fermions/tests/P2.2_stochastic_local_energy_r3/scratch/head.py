#!/usr/bin/env python3
"""
P2.2 round-3 decisive test: exactly-cheap local energies by window-keyed
backflow, versus exact enumeration, screening and semistochastic local
energies of an unrestricted neural-network-backflow-Jastrow (NNBF-J) state,
on stretched hydrogen rings with polarization (p) functions, at N = 6 and
N = 8 electrons, all numbers measured on the same instance at the same N.

Routes under test
-----------------
R3.1 WIBJ   Window-input backflow x all-orbital Jastrow (an ansatz whose EXACT
            local energy is cheap).  psi(n) = J(n) det Phi(n_W)[occ n], where
            the backflow network sees only the occupations n_W of the window
            W (the lowest K_W spatial MOs, both spins; K_W = N by default),
            Phi is M x N over ALL orbitals, and
            J(n) = exp(sum_p u_p n_p + sum_{p<q} v_pq n_p n_q) over all
            spin-orbitals.  All connected n' that share the same window
            change share Phi(n'_W): the local energy is exact with one
            backflow evaluation per distinct window key (at most
            ~N^2 W_v^2 / 4 keys, independent of M) plus O(1) arithmetic per
            connected term (2x2 Green's-function minor x Jastrow ratio).
            Estimator: KEYX (exact).
R3.2 SHRJ   Unrestricted NNBF-J trained with an external-input shrinkage
            penalty mu * ||W1[:, ext]||^2 (lambda-continuation from WIBJ,
            mu = inf, to NNBFJ, mu = 0), with the window-keyed evaluation
            used as an additive control variate (PFCV): the surrogate
            evaluates the network with the external input frozen at n, it
            is summed exactly over all terms, the terms with |y| > eps are
            evaluated exactly, and the residual x - y on the rest gets K
            draws from p ~ 0.9 |y| + 0.1 |h|.  Unbiased.  The penalty makes
            the residual (the external-input response) small by
            construction.
Controls (same instance, same N, unrestricted NNBFJ state, same seed)
--------------------------------------------------------------------
EXACT       full enumeration with exact incremental NNBF-J amplitudes.
SCREEN      keep |H_nn'| > eps (biased; admissible only if
            |bias| + 2 SE <= --bias-tol).
SS          semistochastic: |H| > eps exact + K heat-bath draws p ~ |H|.
PFCV        (ablation) the R3.2 estimator on the unpenalised NNBFJ state.
All estimators are run per sample with the real per-term code (row
generation by Slater-Condon rules, incremental exact amplitudes for every
term actually evaluated, window-keyed backflow), on samples n drawn iid
from |psi|^2.  The iid draw uses the full state vector (an exact sampler
standing in for MCMC; it is not part of any estimator's cost).  Training
uses the exact energy on the full space (also not part of any estimator's
cost).

Metric
------
For each estimator X on state s: per-sample estimates Ehat_X(n) on B
samples; C_X = s^2_X * t_X, with s^2_X the empirical per-sample variance
and t_X the measured mean wall time per sample (row generation included,
single thread).  Tunable estimators (SCREEN eps; SS and PFCV eps, K) are
selected on the first half of the samples and evaluated on the second
half.  Control cost C_ctrl = min(C_EXACT, C_SS, C_SCREEN[admissible]) on
the NNBFJ state of the same seed.
  M_R3.1 = C_KEYX(WIBJ) / C_ctrl(NNBFJ)
  M_R3.2 = C_PFCV(SHRJ) / C_ctrl(NNBFJ)
plus dE_X = E_var - E_FCI for every arm.  Seeds: --seeds (default 3);
per-instance values are medians over seeds of the per-seed (paired) ratios.

Verdict (pre-committed; computed by --aggregate)
------------------------------------------------
Per route, per N in {6, 8}, with K_min, K_max the smallest/largest active
spaces run:
  dE_ok      median dE_route(K_max) <= max(1.6 mHa, 1.25 * median dE_NNBFJ(K_max))
  PASS       dE_ok and M(K_max) <= 0.5 and M(K_max) < M(K_min)
  FAIL       not dE_ok, or M(K_max) >= 1
  AMBIGUOUS  otherwise.  Pre-committed action: rerun N = 6 with
             --n-active 15; PASS iff M(15) <= min(M(14), 0.5), else KILL.
Global per route: PASS iff PASS at N = 6 and N = 8; FAIL if FAIL at either.
Hard Core: the instances carry non-stoquastic sign structure and strong
multireference character (reported: HF weight, frustrated fraction);
same-sector competitors are not examined: SIGN HALF ONLY.

Command-line options
--------------------
--n-atoms INT          H atoms in the ring = N electrons (even, >= 2). Default 6.
--n-active INT [...]   Active-space sizes K (lowest RHF MOs kept; K <= #AOs).
                       Default 8 10 12 14.
--bond-length FLOAT    H-H distance in bohr (> 0). Default 2.8.
--window INT           K_W, window size in spatial MOs (>= N/2). Default: N.
--arms LIST            Subset of NNBFJ WIBJ SHRJ. Default all three.
--seeds INT [...]      Network-initialisation seeds. Default 0 1 2.
--hidden INT           Backflow hidden width. Default 48.
--maxiter INT          L-BFGS iterations on the exact energy. Default 1500.
--mu FLOAT             SHRJ external-input shrinkage. Default 1e-2.
--n-samples INT        Samples B per state (even, >= 20). Default 2000.
--eps-screen FLOAT...  SCREEN thresholds. Default 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2.
--eps-grid FLOAT...    SS / PFCV thresholds. Default 0 1e-3 1e-2 3e-2.
--k-grid INT...        SS / PFCV draws per sample. Default 1 4 16 64.
--bias-tol FLOAT       Admissible SCREEN bias in Ha. Default 1.6e-4.
--out PATH             Result JSON. Default results_<tag>.json here.
--smoke                Integral, Hamiltonian, ansatz and estimator checks plus
                       a tiny H4 run (< 30 s).
--aggregate DIR        Read every results_*.json in DIR and print the verdict.
"""
import argparse
import glob
import json
import os
import time
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import gammainc, gamma

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["NNBFJ", "WIBJ", "SHRJ"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.2 round-3 local-energy test")
    ap.add_argument("--n-atoms", type=int, default=6)
    ap.add_argument("--n-active", type=int, nargs="+", default=[8, 10, 12, 14])
    ap.add_argument("--bond-length", type=float, default=2.8)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--mu", type=float, default=1e-2)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--eps-screen", type=float, nargs="+", default=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--eps-grid", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 3e-2])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--bias-tol", type=float, default=1.6e-4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", type=str, default=None)
    a = ap.parse_args(argv)
    if a.aggregate:
        return a
    if a.n_atoms < 2 or a.n_atoms % 2:
        ap.error("--n-atoms must be even and >= 2")
    if a.window is None:
        a.window = a.n_atoms
    if a.window < a.n_atoms // 2:
        ap.error("--window must be >= N/2")
    if a.bond_length <= 0 or a.hidden < 1 or a.maxiter < 1 or a.mu < 0:
        ap.error("--bond-length > 0, --hidden >= 1, --maxiter >= 1, --mu >= 0")
    if a.n_samples < 20 or a.n_samples % 2:
        ap.error("--n-samples must be even and >= 20")
    if min(a.k_grid) < 1 or min(a.eps_grid) < 0:
        ap.error("--k-grid >= 1, --eps-grid >= 0")
    if a.smoke:
        a.n_atoms, a.n_active, a.window, a.hidden, a.maxiter = 4, [6, 8], 4, 8, 200
        a.seeds, a.n_samples, a.arms = [0, 1, 2], 200, ARMS
        a.k_grid, a.eps_grid = [1, 4, 16], [0.0, 1e-2]
    if a.out is None:
        tag = "smoke" if a.smoke else f"N{a.n_atoms}_K{'-'.join(map(str, a.n_active))}_{'-'.join(a.arms)}"
        a.out = os.path.join(HERE, f"results_{tag}.json")
    return a
