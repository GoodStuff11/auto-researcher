#!/usr/bin/env python3
"""
P3.1 round 7: low-energy subspace estimation (route ESR) -- premise measurement and injected check
====================================================================================================

Instance: the shared 4x4 OBC t-t'-J cluster (6+6, t'=-0.25, J=0.4, D=1681680), built through the r6
module ../P3.1_order_parameter_error_r6/micro_test.py (loaded as "r6", which itself imports r5_core).
"--instance small" = 4x4 OBC 2+2 cluster, smoke only (no physics claim).

Notation. |n>, n = 1..N_LOW: exact in-irrep (chi, S = S_0) eigenvectors, omega_n = E_n - E_0.
psi = alpha psi_0 + e;  c_n = <n|e>;  A_n0 = <n|(A - a_0)|psi_0>;  delta_1 = -2 alpha sum_n c_n A_n0.

Part 1 (exact, on the real converged width-6 projected NNBF states present at run time; P2.1 seeds
11/22/33):
  (a) Concentration curve R(K) = [-2 alpha sum_{n<=K} c_n A_n0] / delta_1, K = 1..N_LOW, and the
      oracle-ESR curve: delta_hat(K) / delta with phi_k = |k> exactly (Ritz in span{psi, Hpsi, |1..K>}).
      K_half = smallest K with the oracle-ESR ratio in [1/2, 2] for all larger K up to N_LOW.
  (b) Oracle certificate envelope C* = alpha sqrt(2 chi_A dE) (chi_A = 2 sum_n |A_n0|^2/omega_n over the
      whole sector, by Lanczos quadrature from (A - a_0) psi_0; dE exact). Any certificate that only uses
      the energy window must be at least C*; printed as C*/|delta|.
  (c) If >= 2 seeds: cosines between the seeds' in-window error vectors P_K e_s, and the multi-seed
      Ritz (MSR) estimate: Ritz of H in span{psi_s, (H - E_s) psi_s : all seeds}; delta_hat = a_R - a_psi.

Part 2 (injected check, pre-registered; ESR must pass it before any excited-state training):
  Window K = min(#{n : omega_n <= 1 t}, K_MAX=24, N_LOW - 6).  Synthetic "excited NQS" phi_k, k = 1..K:
     phi_k = N[ |k> + eta_rot g_k + eta_leak l_k + mu_k j_k ]
       g_k : random unit vector in span{|m> : m <= K, m != k}   (rotation inside the window)
       l_k : random unit vector in span{|m> : K < m <= N_LOW}   (leak to low levels outside the window)
       j_k : high-omega junk (H - E_0) P_chi[|k> xi], psi_0 and all |m> removed, normalised
       mu_k: set so that the junk alone costs eps = kappa * dE_ref (dE_ref = real seed-11 dE, else 0.052)
     Grade R (must pass): kappa = 1, eta_rot = 0.3, eta_leak = 0.1.
     Grade H (reported):  kappa = 2, eta_rot = 0.5, eta_leak = 0.3.
     No psi_0 or e component is put into phi_k (no oracle information).
  Targets: the real states (Part 1) and set A: 8 synthetic seeds psi = N[psi_0 + 0.30 u + mu j],
     u a random unit combination of the 9 lowest |n>, mu = 0.25 (1 + 0.3 U(-1,1)) (r6 set-A construction,
     single level, magnitudes matched to the real state's overlap ~0.94). Set W (reported): same with u
     random over all N_LOW levels.
  Arms (all Ritz problems for H in the span of normalised basis vectors; (H - E_v) v replaces H v):
     ESR   span{psi, (H-E)psi, phi_1..phi_K, (H-E)phi_1..(H-E)phi_K}
     ESR0  span{psi, (H-E)psi, phi_1..phi_K}                 (no Krylov cleaning of the phi_k)
     ESRs  ESR with the first K' = ceil(2K/3) phi only        (stability arm)
     L1R   span{psi, (H-E)psi}                                (saturated control; S2 says blind)
     ORC   exact Ritz with the exact |1..K> (window ceiling; diagnostic, exact matrices only)
  delta_hat = a(Ritz ground vector) - a(psi). Sampled arm (deciding): every matrix element
  <b_i|O|b_j> (O = 1, H, A_pair, A_spin, A_charge) is a mixture-importance-sampling mean of
  b_i(n) (O b_j)(n) / p(n) with n ~ p = (|psi|^2/|psi|^2 + sum_k |phi_k|^2/|phi_k|^2)/(K+1),
  local values taken from exact vectors at the draws (r5/r6 convention). SE from a batch bootstrap
  (40 batches, --n-boot replicas). Exact-matrix arm printed alongside.
  DECIDING ARM: ESR0 (changed from ESR after the small-cluster smoke, before any main-instance run:
  the (H-E)phi_k vectors doubled the sampled basis and added more noise than they removed junk).
  ESR0s = ESR0 with the first K' phi only. Trusted iff SE <= 0.5 |delta_hat| and
  |delta_hat(ESR0) - delta_hat(ESR0s)| <= 0.5 |delta_hat(ESR0)| (same rule for ESR with ESRs).
  Item = (target, observable); scored iff |delta| >= 0.10 |a_0|. Tracked: trusted and
  1/2 <= delta_hat/delta <= 2; false: trusted and < 1/2; over: trusted and > 2. G = false/trusted
  (defined with >= 3 trusted).
  INJECT-PASS (grade R, arm ESR0, sampled) iff ALL of:
     set A: >= 9 scored, tracked fraction >= 0.7, median over seeds of the per-seed tracked fraction
            >= 2/3, G <= 0.10 (defined), and tracked(ESR0) - tracked(L1R) >= 0.3 (fractions);
     real : >= 2/3 of the scored real items tracked and no false certificate.
  Otherwise INJECT-FAIL: ESR is killed and no excited-state training is spent.

Usage
  python3 micro_test.py --instance small --n-low 14 --n-seeds 3 --mc-samples 50000 --n-boot 30   (smoke)
  python3 micro_test.py --instance main  --n-low 36 --n-seeds 8 --mc-samples 200000 --n-boot 100 (Slurm)
Outputs: <results>/esr_inject.json; verdict line "INJECT (ESR, grade R): ... -> INJECT-PASS|FAIL".
"""
import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "r6", os.path.join(HERE, "..", "P3.1_order_parameter_error_r6", "micro_test.py"))
r6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r6)
rc = r6.rc
OBS = r6.OBS
log = r6.log

K_MAX, OMEGA_WIN = 24, 1.0
GRADES = {"R": dict(kappa=1.0, rot=0.3, leak=0.1), "H": dict(kappa=2.0, rot=0.5, leak=0.3)}
SCORE_REL, TRUST = 0.10, 0.5


# ============================================================================ helpers
def unit(v):
    return v / np.linalg.norm(v)


def shifted(m, v):
    """(H - E_v) v for normalised v."""
    Hv = m.H @ v
    return Hv - (v @ Hv) * v


def ritz_shift(S, H, A, keep_rel):
    """Lowest Ritz vector of (H, S) with diagonal scaling and canonical orthogonalisation.
    Returns a(Ritz) - a(basis vector 0) for each A in the dict A."""
    S = 0.5 * (S + S.T); H = 0.5 * (H + H.T)
    d = np.sqrt(np.abs(np.diag(S)))
    Sn = S / np.outer(d, d); Hn = H / np.outer(d, d)
    lam, V = np.linalg.eigh(Sn)
    keep = lam > keep_rel * lam.max()
    X = V[:, keep] / np.sqrt(lam[keep])
    ev, Y = np.linalg.eigh(X.T @ Hn @ X)
    c = X @ Y[:, 0]
    out = {}
    for k, Ak in A.items():
        Ak = 0.5 * (Ak + Ak.T) / np.outer(d, d)
        out[k] = float(c @ Ak @ c / (c @ Sn @ c) - Ak[0, 0] / Sn[0, 0])
    return out


def classify(dh, se, d, stab_ok=True):
    if not np.isfinite(dh):
        return "abstain"
    if se > TRUST * abs(dh) or not stab_ok:
        return "untrusted"
    r = dh / d
    if r < 0.5:
        return "false"
    return "tracked" if r <= 2 else "over"


# ============================================================================ basis blocks
class Block:
    """Basis vectors b (normalised) with stored O b for O in (1, H, A_obs...)."""

    def __init__(self, m, ob):
        self.m, self.ob = m, ob
        self.b, self.Hb, self.Ab = [], [], {k: [] for k in OBS}

    def add(self, v):
        v = unit(v)
        self.b.append(v); self.Hb.append(self.m.H @ v)
        for k in OBS:
            self.Ab[k].append(self.ob.apply(k, v))

    def add_state(self, v):
        """Adds v and its Krylov-cleaning vector (H - E_v) v; returns their indices."""
        v = unit(v)
        i = len(self.b)
        self.add(v); self.add(shifted(self.m, v))
        return i, i + 1


def exact_mats(blocks_b, blocks_Ob):
    B = np.array(blocks_b)
    return {name: B @ np.array(Ob).T for name, Ob in blocks_Ob.items()}


def gather(block, idx):
    """Values at the draws idx of every basis vector and every O b."""
    out = {"S": np.array([x[idx] for x in block.b]), "H": np.array([x[idx] for x in block.Hb])}
    for k in OBS:
        out[k] = np.array([x[idx] for x in block.Ab[k]])
    return out


# ============================================================================ Part 1 helpers
def chi_A(m, E0, psi0, ob, a0, steps):
    out = {}
    for k in OBS:
        w = ob.apply(k, psi0) - a0[k] * psi0
        nw = np.linalg.norm(w)
        Q, th, S = r6.lanczos_basis(m.H, w, steps)
        wt = nw ** 2 * S[0] ** 2
        om = th - E0
        sel = om > 1e-6
        out[k] = float(2 * np.sum(wt[sel] / om[sel]))
        del Q
    return out


def real_states(m, sg, psi0, args):
    states = {}
    if args.instance == "main":
        for seed in (11, 22, 33):
            f = os.path.join(args.p21_params, f"inst0_nnbf_s{seed}.npz")
            if not os.path.exists(f):
                log(f"seed {seed}: no parameter file; skipped"); continue
            ans = rc.load_params(rc.nnbf(m, 6, seed), f)
            v = unit(sg.project(ans.amplitudes(32768)))
            states[f"w6_s{seed}"] = v if v @ psi0 > 0 else -v
    else:
        rng0 = np.random.default_rng(5)
        for s in range(2):
            v = unit(sg.project(psi0 * np.exp(0.3 * rng0.standard_normal(m.D))))
            states[f"small_synth{s}"] = v
    return states


# ============================================================================ injection pieces
def junk_from(m, sg, base, psi0, U, rng):
    x = sg.project(base * rng.standard_normal(m.D))
    j = m.H @ x - (base @ (m.H @ base)) * x
    j -= U @ (U.T @ j)
    j -= psi0 * (psi0 @ j)
    return unit(j)


def excited_set(m, sg, psi0, E0, U, om, K, grade, dE_ref, rng):
    g = GRADES[grade]
    eps = g["kappa"] * dE_ref
    phis, eps_act = [], []
    nl = U.shape[1]
    for k in range(K):
        uk = U[:, k]
        cw = rng.standard_normal(K); cw[k] = 0.0
        gk = unit(U[:, :K] @ cw)
        lk = unit(U[:, K:] @ rng.standard_normal(nl - K))
        jk = junk_from(m, sg, uk, psi0, U, rng)
        Ej = jk @ (m.H @ jk)
        Ek = E0 + om[k]
        mu = math.sqrt(eps / max(Ej - Ek - eps, 1e-6))
        v = unit(uk + g["rot"] * gk + g["leak"] * lk + mu * jk)
        assert np.linalg.norm(sg.project(v) - v) < 1e-8
        phis.append(v)
        eps_act.append(float(v @ (m.H @ v) - Ek))
    return phis, eps_act


def esr_item_estimates(m, ob, psi, phi_block, K, Kp, args, rng):
    """Returns {arm: {obs: (d_hat_sampled, se, d_hat_exact)}} for one target psi."""
    blk = Block(m, ob)
    blk.add_state(psi)
    nb_psi = len(blk.b)
    # indices: psi block = 0,1 ; phi block: phi_k at 2+2k, (H-E)phi_k at 3+2k
    allb = blk.b + phi_block.b
    allH = blk.Hb + phi_block.Hb
    allA = {k: blk.Ab[k] + phi_block.Ab[k] for k in OBS}
    nphi = K
    sets = {
        "L1R": [0, 1],
        "ESR0": [0, 1] + [nb_psi + 2 * i for i in range(nphi)],
        "ESR": [0, 1] + [nb_psi + j for j in range(2 * nphi)],
        "ESRs": [0, 1] + [nb_psi + j for j in range(2 * Kp)],
        "ESR0s": [0, 1] + [nb_psi + 2 * i for i in range(Kp)],
    }
    # exact matrices
    B = np.array(allb)
    Sx = B @ B.T
    Hx = B @ np.array(allH).T
    Ax = {k: B @ np.array(allA[k]).T for k in OBS}
    exact = {}
    for arm, ix in sets.items():
        ii = np.ix_(ix, ix)
        exact[arm] = ritz_shift(Sx[ii], Hx[ii], {k: Ax[k][ii] for k in OBS}, 1e-11)
    # sampling: mixture of |psi|^2 and |phi_k|^2 (all unit-normalised)
    comps = [allb[0]] + [allb[nb_psi + 2 * i] for i in range(nphi)]
    p = np.zeros(m.D)
    for v in comps:
        p += v * v
    p /= len(comps)
    idx = rng.choice(m.D, size=args.mc_samples, p=p / p.sum())
    w = 1.0 / p[idx]
    Bs = B[:, idx]
    Hs = np.array([x[idx] for x in allH])
    As = {k: np.array([x[idx] for x in allA[k]]) for k in OBS}
    nbat = 40
    bat = np.array_split(np.arange(len(idx)), nbat)

    def mats(rows):
        ww = w[rows]
        Bw = Bs[:, rows] * ww
        return (Bw @ Bs[:, rows].T, Bw @ Hs[:, rows].T, {k: Bw @ As[k][:, rows].T for k in OBS})
    per = [mats(r) for r in bat]

    def est(sel):
        S = sum(per[i][0] for i in sel); H = sum(per[i][1] for i in sel)
        A = {k: sum(per[i][2][k] for i in sel) for k in OBS}
        return S, H, A
    full = est(range(nbat))
    reps = [est(rng.integers(0, nbat, nbat)) for _ in range(args.n_boot)]
    out = {}
    for arm, ix in sets.items():
        ii = np.ix_(ix, ix)
        f = ritz_shift(full[0][ii], full[1][ii], {k: full[2][k][ii] for k in OBS}, 1e-7)
        rb = [ritz_shift(S[ii], H[ii], {k: A[k][ii] for k in OBS}, 1e-7) for S, H, A in reps]
        out[arm] = {k: (f[k], float(np.std([r[k] for r in rb], ddof=1)), exact[arm][k]) for k in OBS}
    return out


def oracle_shift(m, ob, v, oracle, K):
    """Exact Ritz shift with the exact eigenvectors |1..K> (no excited-state error): window ceiling."""
    blk = Block(m, ob); blk.add_state(v)
    B = np.array(blk.b + oracle.b[:K])
    Sx, Hx = B @ B.T, B @ np.array(blk.Hb + oracle.Hb[:K]).T
    Ax = {k: B @ np.array(blk.Ab[k] + oracle.Ab[k][:K]).T for k in OBS}
    return ritz_shift(Sx, Hx, Ax, 1e-11)


def score_target(name, psi, a0, est, items, set_name, seed):
    for k in OBS:
        d = a0[k] - float(psi @ items["ob"].apply(k, psi))
        scored = abs(d) >= SCORE_REL * abs(a0[k])
        row = dict(set=set_name, seed=seed, target=name, obs=k, delta=d, scored=bool(scored), est={})
        stab = {a: abs(est[a][k][0] - est[b][k][0]) <= 0.5 * abs(est[a][k][0])
                for a, b in (("ESR", "ESRs"), ("ESR0", "ESR0s"))}
        for arm in ("ESR0", "ESR", "ESR0s", "ESRs", "L1R", "ORC"):
            dh, se, dx = est[arm][k]
            cls = classify(dh, se, d, stab.get(arm, True))
            row["est"][arm] = dict(d=dh, se=se, d_exact=dx, cls=cls)
        items["rows"].append(row)
        log(f"{set_name} {name} {k}: delta={d:+.3e} (rel {abs(d) / abs(a0[k]):.3f}, scored {scored}) | "
            + " ".join(f"{a}={e['d']:+.2e}+-{e['se']:.1e}(ex {e['d_exact']:+.2e})[{e['cls']}]"
                       for a, e in row["est"].items()))


def summarize(rows, arm, set_name):
    sc = [r for r in rows if r["set"] == set_name and r["scored"]]
    cl = [r["est"][arm]["cls"] for r in sc]
    n_tr = cl.count("tracked"); n_f = cl.count("false")
    n_trust = sum(c in ("tracked", "false", "over") for c in cl)
    per = {}
    for s in sorted({r["seed"] for r in sc}):
        c2 = [r["est"][arm]["cls"] for r in sc if r["seed"] == s]
        per[s] = c2.count("tracked") / len(c2)
    return dict(scored=len(sc), tracked=n_tr, false=n_f, trusted=n_trust,
                frac=n_tr / len(sc) if sc else float("nan"),
                median_seed=float(np.median(list(per.values()))) if per else float("nan"),
                G=n_f / n_trust if n_trust >= 3 else float("nan"))


# ============================================================================ main
def run(args):
    t0 = time.time()
    m, E0, psi0, sg, S0, ob = r6.setup(args.instance)
    a0 = ob.all(psi0)
    log("a0 " + json.dumps(a0))
    e, U = r6.sector_eigs(m, sg, S0, args.n_low + 1)
    assert abs(e[0] - E0) < 1e-6, (e[0], E0)
    U = U[:, 1:]; om = e[1:] - E0
    nl = U.shape[1]
    log(f"{nl} in-irrep levels: omega = {np.round(om, 4).tolist()} ({time.time() - t0:.0f} s)")
    out = dict(instance=args.instance, a0=a0, E0=E0, omega=om.tolist(), args=vars(args), part1={})
    W = {k: U.T @ (ob.apply(k, psi0) - a0[k] * psi0) for k in OBS}
    chi = chi_A(m, E0, psi0, ob, a0, args.lanczos_steps)
    log("chi_A (whole sector, Lanczos quadrature) " + json.dumps(chi))
    K = int(min(np.sum(om <= OMEGA_WIN), K_MAX, nl - 6))
    K = max(K, 2) if args.K is None else args.K   # --K: smoke/debug override only
    Kp = int(math.ceil(2 * K / 3))
    log(f"window: K = {K} levels (omega <= {om[K - 1]:.3f}); stability K' = {Kp}; "
        f"#levels <= 1t among computed = {int(np.sum(om <= 1.0))} of {nl}")
    out.update(K=K, Kp=Kp)
    # ---------------------------------------------------------------- Part 1
    states = real_states(m, sg, psi0, args)
    dE_ref = 0.052
    oracle = Block(m, ob)
    for n in range(nl):
        oracle.add(U[:, n])
    for key, v in states.items():
        alpha = float(psi0 @ v)
        ev = v - alpha * psi0
        c = U.T @ ev
        dE = float(v @ (m.H @ v) - E0)
        if key == "w6_s11":
            dE_ref = dE
        rec = dict(alpha=alpha, dE=dE, obs={})
        for k in OBS:
            d = a0[k] - ob.expect(k, v)
            w = ob.apply(k, psi0) - a0[k] * psi0
            d1 = float(-2 * alpha * (ev @ w))
            cum = np.cumsum(-2 * alpha * c * W[k]) / d1
            # oracle ESR (exact |k>) Ritz curve
            blk = Block(m, ob); blk.add_state(v)
            Bb = np.array(blk.b + oracle.b)
            Hb = np.array(blk.Hb + oracle.Hb)
            Ab = np.array(blk.Ab[k] + oracle.Ab[k])
            Sx, Hx, Ax = Bb @ Bb.T, Bb @ Hb.T, Bb @ Ab.T
            curve = []
            for KK in range(1, nl + 1):
                ix = [0, 1] + [2 + i for i in range(KK)]
                ii = np.ix_(ix, ix)
                curve.append(ritz_shift(Sx[ii], Hx[ii], {k: Ax[ii]}, 1e-11)[k] / d)
            ok = [0.5 <= x <= 2 for x in curve]
            khalf = next((i + 1 for i in range(nl) if all(ok[i:])), None)
            env = alpha * math.sqrt(2 * chi[k] * dE)
            rec["obs"][k] = dict(delta=d, delta1=d1, R=cum.tolist(), oracle_curve=curve, K_half=khalf,
                                 envelope=env, envelope_ratio=env / abs(d))
            log(f"P1 {key} {k}: delta={d:+.3e} delta1={d1:+.3e} R(K=9,{K},{nl})="
                f"{cum[min(8, nl - 1)]:.3f},{cum[K - 1]:.3f},{cum[-1]:.3f} | oracle-ESR ratio K=9,{K},{nl}: "
                f"{curve[min(8, nl - 1)]:.3f},{curve[K - 1]:.3f},{curve[-1]:.3f} K_half={khalf} | "
                f"oracle envelope C*/|delta| = {env / abs(d):.2f}")
        out["part1"][key] = rec
    keys = list(states)
    if len(keys) >= 2:
        P = {s: U[:, :K] @ (U[:, :K].T @ (states[s] - (psi0 @ states[s]) * psi0)) for s in keys}
        cos = {f"{a}|{b}": float(P[a] @ P[b] / (np.linalg.norm(P[a]) * np.linalg.norm(P[b])))
               for i, a in enumerate(keys) for b in keys[i + 1:]}
        log("seed in-window error cosines " + json.dumps(cos))
        blk = Block(m, ob)
        for s in keys:
            blk.add_state(states[s])
        B = np.array(blk.b); Sx = B @ B.T; Hx = B @ np.array(blk.Hb).T
        Ax = {k: B @ np.array(blk.Ab[k]).T for k in OBS}
        msr = {}
        for t, s in enumerate(keys):
            order = [2 * t, 2 * t + 1] + [j for j in range(len(blk.b)) if j not in (2 * t, 2 * t + 1)]
            ii = np.ix_(order, order)
            r = ritz_shift(Sx[ii], Hx[ii], {k: Ax[k][ii] for k in OBS}, 1e-11)
            msr[s] = {k: r[k] / out["part1"][s]["obs"][k]["delta"] for k in OBS}
        log("MSR (multi-seed Ritz, exact) delta_hat/delta " + json.dumps(msr))
        out["seed_cos"] = cos; out["MSR_ratio"] = msr
    out["dE_ref"] = dE_ref
    log(f"Part 1 done ({time.time() - t0:.0f} s); dE_ref = {dE_ref:.4f}")
    # ---------------------------------------------------------------- Part 2
    rng = np.random.default_rng(2031)
    targets = [("real", i, k, v) for i, (k, v) in enumerate(states.items())]
    for name, nlev in (("A", min(9, nl)), ("W", nl)):
        for s in range(args.n_seeds):
            u = unit(U[:, :nlev] @ rng.standard_normal(nlev))
            mu = 0.25 * (1 + 0.3 * rng.uniform(-1, 1))
            v = unit(psi0 + 0.30 * u + mu * r6.junk(m, sg, psi0, E0, U, rng))
            assert np.linalg.norm(sg.project(v) - v) < 1e-8
            targets.append((name, s, f"{name}{s}", v))
    out["grades"] = {}
    verdict = None
    for grade in ("R", "H"):
        phis, eps_act = excited_set(m, sg, psi0, E0, U, om, K, grade, dE_ref, rng)
        log(f"grade {grade}: excited-state energy errors eps_k/omega_k = "
            f"{np.round(np.array(eps_act) / om[:K], 3).tolist()}")
        pb = Block(m, ob)
        for ph in phis:
            pb.add_state(ph)
        items = dict(ob=ob, rows=[])
        for set_name, seed, tname, v in targets:
            est = esr_item_estimates(m, ob, v, pb, K, Kp, args, rng)
            orc = oracle_shift(m, ob, v, oracle, K)
            est["ORC"] = {k: (orc[k], 0.0, orc[k]) for k in OBS}
            score_target(tname, v, a0, est, items, set_name, seed)
        rows = items["rows"]
        summ = {sn: {arm: summarize(rows, arm, sn) for arm in ("ESR0", "ESR", "ESR0s", "ESRs", "L1R", "ORC")}
                for sn in ("A", "W", "real")}
        for sn, d_ in summ.items():
            log(f"grade {grade} SET {sn}: " + "; ".join(
                f"{a} {s['tracked']}/{s['scored']} med_seed {s['median_seed']:.2f} G {s['G']:.2f}"
                for a, s in d_.items()))
        out["grades"][grade] = dict(eps_over_omega=(np.array(eps_act) / om[:K]).tolist(), rows=rows, summary=summ)
        del pb
        if grade == "R":
            A_, L_, R_ = summ["A"]["ESR0"], summ["A"]["L1R"], summ["real"]["ESR0"]
            okA = (A_["scored"] >= 9 and A_["frac"] >= 0.7 and A_["median_seed"] >= 2 / 3
                   and np.isfinite(A_["G"]) and A_["G"] <= 0.10
                   and A_["frac"] - (L_["frac"] if np.isfinite(L_["frac"]) else 0.0) >= 0.3)
            okR = R_["scored"] >= 1 and R_["tracked"] >= (2 / 3) * R_["scored"] and R_["false"] == 0
            verdict = bool(okA and okR)
            out["inject_pass"] = verdict
            log(f"INJECT (ESR0, grade R): set A scored {A_['scored']} tracked {A_['tracked']} frac {A_['frac']:.2f} "
                f"median_seed {A_['median_seed']:.2f} G {A_['G']:.2f} (L1R frac {L_['frac']:.2f}); real tracked "
                f"{R_['tracked']}/{R_['scored']} false {R_['false']} -> {'INJECT-PASS' if verdict else 'INJECT-FAIL'}")
    os.makedirs(args.results, exist_ok=True)
    for g in out["grades"].values():
        for r in g["rows"]:
            r["seed"] = int(r["seed"])
    json.dump(out, open(os.path.join(args.results, "esr_inject.json"), "w"), indent=1, default=float)
    log(f"done ({time.time() - t0:.0f} s)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="main", choices=["main", "small"])
    ap.add_argument("--n-low", type=int, default=36)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--K", type=int, default=None, help="window override (smoke only; not used on main)")
    ap.add_argument("--mc-samples", type=int, default=200000)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--lanczos-steps", type=int, default=120)
    ap.add_argument("--p21-params", default=os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
