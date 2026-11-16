#!/usr/bin/env python3
r"""
P2.2 closed-leaf falsifiers (2026-11-16): the two named follow-ups of the Saturation Diagnostic.
================================================================================================

Both modes reuse the round-5 code unchanged (../P2.2_stochastic_local_energy_r5/micro_test.py:
Inst, make_net, run_sr, parse_arm, calibrate, final_dE, CASJ).  Pre-registered rules are below and
in the leaf file (section "Closed-leaf falsifier (2026-11-16)"), written before any run.

--mode ale   R5.2 (anchored local energy, regime F_B) in its untested decisive regime: COLD start
             (the network initialisation, no L-BFGS warm-up), step-limited SR.  H4 distorted chain
             R = 4.4 bohr, K = 12 (dim 4356, HF weight 0.49), NNBF-J hidden 32, hJ 16, init seed 0
             (the same initial theta for every arm and seed; the seed only drives sampling).
             Budget C_b = n_ref * B * <|C|>_pi(theta_0) * (C_GEN + amp) operations (and the
             calibrated wall-clock analogue), i.e. n_ref EXACT steps.  Every arm runs in "steps"
             mode: a cheaper local energy buys more SR steps (the regime where Statement 3 does
             not apply).  Arms are micro_test.py arm strings.
--mode fj    F_A2 accuracy premise: CASJ (window multi-determinant x pair Jastrow; orbitals do not
             depend on the configuration, so Wick/Cholesky contraction of a whole excitation class
             is possible) with the external-external pair Jastrow entries removed (the factorizable
             Jastrow that the R4.0 contraction requires).  Window = the first N spatial MOs, as in
             WIBJ and R4.0.  Trained by L-BFGS on the exact energy exactly as Part W (maxiter 6000).
             --fj 0 trains the unmasked CASJ (pipeline control: must reproduce the 452238 value
             0.390 mHa at K = 12, seed 0, to 0.01 mHa).
--aggregate  apply the pre-registered rules to DIR/ale_*.json and DIR/fj_*.json and print verdicts.

Pre-registered rule, mode ale (tau = 0.2 mHa; medians over the 3 seeds of an arm)
  nu   = max(|med EXACT - med EXACTNF|, half the range of the 6 EXACT+EXACTNF values) at C_b.
  D    = min over controls {EXACT, SCREEN:*, SS:*} of the median dE at C_b (ctrl*).
  R*   = the ALE arm with the lowest median dE at C_b/2;  Rh, Rf = its medians at C_b/2 and C_b.
  premise STEP-LIMITED: med EXACT(C_b/2) - med EXACT(C_b) > 2 nu  and  med EXACT(C_b) > 1.6 mHa.
  Per cost axis (ops, wall):  FAIL if Rf > D + nu;  PASS if Rh <= D and min_seed R*(C_b/2) <=
  min_seed ctrl*(C_b);  AMBIGUOUS otherwise (this is micro_test.verdict_L with its warm-start
  premise replaced by STEP-LIMITED).
  Overall: INCONCLUSIVE if the premise fails; FAIL if either axis FAILs; PASS if both PASS;
  AMBIGUOUS otherwise.
  Consequence: PASS reopens P2.2 with route R5.2 only (regime F_B, cold start).  FAIL kills R5.2
  in its decisive regime.  AMBIGUOUS is recorded; no further run is pre-committed at T1.

Pre-registered rule, mode fj (F_A2 accuracy premise)
  KILL-F_A2 if the masked CASJ (fj 1) at K = 12, seed 0, has dE > tau = 0.2 mHa (the leaf's
  revival condition asks for the restricted ansatz's own error <= tau at K_cost >= 40; restricted
  ansaetze measured so far rise with K, so failing at K = 12 is decisive).  Otherwise
  PREMISE-HOLDS (F_A2 goes to the K-trend test at K = 16, 20; not run here).
  INVALID if the unmasked control misses 0.390 mHa by more than 0.01 mHa.

Command-line options
--------------------
--mode {ale,fj}       required unless --aggregate.
--arms LIST           ale: arm strings (default EXACT EXACTNF SCREEN:3e-3 SCREEN:1e-2 SS:1e-2:16
                      ALE0:4:16 ALE0:8:16 ALE1:4:16).  EXACTNF uses seeds + 10 (noise floor).
--seeds INT...        ale: sampling seeds (default 0 1 2).  fj: network seeds (default 0).
--n-ref INT           ale: budget in EXACT steps (default 50).
--batch INT           ale: SR samples per step B (default 500).
--max-steps-factor INT  ale: step cap = factor * n_ref (default 8).
--n-wall INT          ale: samples for the wall-clock calibration (default 20).
--K INT               active spatial MOs (default 12).
--fj {0,1}            fj: 1 = remove external-external pair Jastrow entries (default 1).
--maxiter INT         fj: L-BFGS iterations (default 6000, as Part W).
--out PATH            output JSON (default results/ale_<arms>.json or results/fj<fj>_K<K>_s<seed>.json).
--aggregate DIR       print both verdicts from DIR and exit.
--nu {range,iqr}      aggregate: noise-floor spread term, half the range (default; first run) or half
                      the interquartile range (second run, pre-registered after the first returned
                      INCONCLUSIVE because EXACT was bimodal: one seed escaped a 220 mHa trap).
--results-dir DIR     output folder under this directory (default results; second run results_r2).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "..", "P2.2_stochastic_local_energy_r5")
sys.path.insert(0, R5)
import keyx_core as kc  # noqa: E402
import micro_test as mt  # noqa: E402

TAU_MHA = 0.2
DEFAULT_ARMS = ["EXACT", "EXACTNF", "SCREEN:3e-3", "SCREEN:1e-2", "SS:1e-2:16", "ALE0:4:16", "ALE0:8:16",
                "ALE1:4:16"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P2.2 closed-leaf falsifiers (see docstring)")
    ap.add_argument("--mode", choices=["ale", "fj"], default=None)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-ref", type=int, default=50)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--max-steps-factor", type=int, default=8)
    ap.add_argument("--n-wall", type=int, default=20)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--fj", type=int, choices=[0, 1], default=1)
    ap.add_argument("--maxiter", type=int, default=6000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--nu", choices=["range", "iqr"], default="range")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args(argv)
    if a.aggregate:
        return a
    if a.mode is None:
        ap.error("--mode ale|fj is required")
    for s in a.arms:
        mt.parse_arm(s)
    if a.seeds is None:
        a.seeds = [0, 1, 2] if a.mode == "ale" else [0]
    # micro_test.run_sr reads these attributes
    a.lr, a.shift, a.max_dtheta, a.ess_min, a.b_cap = 0.5, 1e-4, 1.0, 0.5, 4
    os.makedirs(os.path.join(HERE, a.results_dir), exist_ok=True)
    if a.out is None:
        a.out = os.path.join(HERE, a.results_dir, ("ale_" + "_".join(s.replace(":", "-") for s in a.arms) + ".json")
                             if a.mode == "ale" else f"fj{a.fj}_K{a.K}_s{a.seeds[0]}.json")
    return a


# ---------------------------------------------------------------------------- mode ale
def run_ale(a):
    inst = mt.Inst(4, 4.4, 0.25, 1, a.K)
    net = mt.make_net(a.K, 4, "NNBFJ", 32, 16, 0)
    th0 = net.theta0.copy()
    amp = kc.amp_ops(net)
    psi, _ = net.amplitudes(th0, inst.occ, inst.oi)
    wv = psi ** 2 / (psi @ psi)
    mCpi = float(wv @ np.diff(inst.Hoff.indptr))
    cal = mt.calibrate(inst, net, th0, a.n_wall, np.random.default_rng(4242), 16)
    costs = {"amp_ops": amp, "cal": cal}
    b_ops = a.n_ref * a.batch * mCpi * (kc.C_GEN + amp)
    b_cal = a.n_ref * a.batch * (mt.cal_gen(cal, mCpi) + cal["ts"] + mt.cal_amp(cal, mCpi))
    E_init = 1e3 * (float(psi @ (inst.H @ psi) / (psi @ psi)) - inst.E0)
    print(f"cold start dE={E_init:.2f} mHa P={th0.size} <|C|>_pi={mCpi:.0f} budget ops={b_ops:.3e} cal={b_cal:.2f}s",
          flush=True)
    out = {"mode": "ale", "args": {k: v for k, v in vars(a).items()}, "info": inst.info, "dE_init_mHa": E_init,
           "budget_ops": b_ops, "budget_cal": b_cal, "arms": {}}
    for arm_s in a.arms:
        arm = mt.parse_arm(arm_s)
        seeds = [s + 10 for s in a.seeds] if arm["kind"] == "EXACTNF" else a.seeds
        out["arms"][arm_s] = []
        for s in seeds:
            t1 = time.time()
            tr = mt.run_sr(inst, net, th0, arm, s, a, costs, b_ops, b_cal)
            out["arms"][arm_s].append({"seed": s, **tr})
            print(f"  {arm_s:12s} seed={s:2d} steps={len(tr['step']):4d} anchors={sum(tr['anchor']):3d} "
                  f"dE(C_b)[ops]={mt.final_dE(tr, 'ops', b_ops):8.3f} [cal]={mt.final_dE(tr, 'cal', b_cal):8.3f} "
                  f"dE(C_b/2)[ops]={mt.final_dE(tr, 'ops', b_ops / 2):8.3f} ({time.time() - t1:.0f}s)", flush=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, default=float)
    return out


def verdict_ale(runs_by_arm, b_ops, b_cal, nu_mode="range"):
    med = lambda v: float(np.median(v)) if np.all(np.isfinite(v)) else float("inf")  # noqa: E731
    res = {}
    need = {"EXACT", "EXACTNF"}
    if not need <= set(runs_by_arm) or not any(k.startswith("ALE") for k in runs_by_arm):
        return {"verdict": "INCOMPLETE", "arms": sorted(runs_by_arm)}
    ex_full = [mt.final_dE(r, "ops", b_ops) for r in runs_by_arm["EXACT"]]
    ex_half = [mt.final_dE(r, "ops", b_ops / 2) for r in runs_by_arm["EXACT"]]
    for axis, Cb in (("ops", b_ops), ("cal", b_cal)):
        full = {k: [mt.final_dE(r, axis, Cb) for r in v] for k, v in runs_by_arm.items()}
        half = {k: [mt.final_dE(r, axis, Cb / 2) for r in v] for k, v in runs_by_arm.items()}
        ctrls = [k for k in full if k == "EXACT" or k.startswith("SCREEN") or k.startswith("SS")]
        routes = [k for k in full if k.startswith("ALE")]
        cstar = min(ctrls, key=lambda k: med(full[k]))
        rstar = min(routes, key=lambda k: med(half[k]))
        D = med(full[cstar])
        pool = np.array(full["EXACT"] + full["EXACTNF"])
        spread = (0.5 * float(np.ptp(pool)) if nu_mode == "range"
                  else 0.5 * float(np.subtract(*np.percentile(pool, [75, 25]))))
        nu = max(abs(med(full["EXACT"]) - med(full["EXACTNF"])), spread)
        Rh, Rf = med(half[rstar]), med(full[rstar])
        best_r, best_c = float(np.nanmin(half[rstar])), float(np.nanmin(full[cstar]))
        v = "FAIL" if Rf > D + nu else ("PASS" if (Rh <= D and best_r <= best_c) else "AMBIGUOUS")
        res[axis] = dict(verdict=v, ctrl_star=cstar, D=D, nu=nu, route_star=rstar, Rh=Rh, Rf=Rf,
                         best_route_half=best_r, best_ctrl_full=best_c,
                         medians_full={k: med(x) for k, x in full.items()},
                         medians_half={k: med(x) for k, x in half.items()})
    nu_ops = res["ops"]["nu"]
    prem = (med(ex_half) - med(ex_full) > 2 * nu_ops) and med(ex_full) > 1.6
    vs = [res[x]["verdict"] for x in ("ops", "cal")]
    if not prem:
        v = "INCONCLUSIVE (premise STEP-LIMITED fails)"
    elif "FAIL" in vs:
        v = "FAIL"
    elif all(x == "PASS" for x in vs):
        v = "PASS"
    else:
        v = "AMBIGUOUS"
    return {"premise_step_limited": bool(prem), "EXACT_med_half": med(ex_half), "EXACT_med_full": med(ex_full),
            "axes": res, "verdict": v}


# ---------------------------------------------------------------------------- mode fj
class CASJFJ(mt.CASJ):
    """CASJ with the external-external pair Jastrow entries fixed at zero (window = first N
    spatial MOs; spin-orbital p has spatial index p mod K)."""

    def __init__(self, K, N, rng, mask_ext=True):
        super().__init__(K, N, rng)
        p, q = self.iu
        ext = (p % K >= N) & (q % K >= N)
        self.vmask = ~ext if mask_ext else np.ones(len(p), bool)
        off = sum(int(np.prod(s)) for k, s in self.shapes if k != "v")
        self.vsl = slice(off, off + len(p))
        assert self.shapes[-1][0] == "v"
        self.gmask = np.ones(self.theta0.size)
        self.gmask[self.vsl] = self.vmask.astype(float)
        self.theta0 = self.theta0 * self.gmask
        self.n_ext_pairs = int(ext.sum())

    def energy_grad(self, th, H, occ, occ_idx):
        E, g = super().energy_grad(th * self.gmask, H, occ, occ_idx)
        return E, g * self.gmask


def run_fj(a):
    N, K = 4, a.K
    h, g, enuc, Erhf, eps = kc.build_molecule(N, 4.4, 0.25, 1, kc.H631GSS, K, "valence-p")
    ham = kc.SpatialHam(h, g, enuc, K, N)
    codes, occ, oi = kc.build_space(K, N // 2, N // 2)
    H = kc.build_hamiltonian(ham, codes, oi)
    w2, _ = kc.ground_states(H, 2)
    E0 = float(w2[0])
    out = {"mode": "fj", "fj": a.fj, "K": K, "E0": E0, "runs": []}
    for seed in a.seeds:
        net = CASJFJ(K, N, np.random.default_rng(1000 * seed + 7), mask_ext=bool(a.fj))
        t1 = time.time()
        r = minimize(lambda t: net.energy_grad(t, H, occ, oi), net.theta0, jac=True, method="L-BFGS-B",
                     options={"maxiter": a.maxiter, "maxcor": 30, "gtol": 1e-12, "ftol": 1e-16})
        th = r.x * net.gmask
        psi, _ = net.amplitudes(th, occ, oi)
        Ev = float(psi @ (H @ psi) / (psi @ psi))
        assert np.all(th[net.vsl][~net.vmask] == 0.0)
        run = {"seed": seed, "dE_mHa": 1e3 * (Ev - E0), "P": int(th.size), "P_free": int(net.gmask.sum()),
               "n_ext_pairs_removed": int((~net.vmask).sum()), "nit": int(r.nit), "t_s": time.time() - t1}
        out["runs"].append(run)
        print(f"  CASJ fj={a.fj} K={K} seed={seed}: dE={run['dE_mHa']:.4f} mHa P={run['P']} free={run['P_free']} "
              f"removed ext-ext pairs={run['n_ext_pairs_removed']} nit={r.nit} ({run['t_s']:.0f}s)", flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, default=float)
    return out


# ---------------------------------------------------------------------------- aggregate
def aggregate(d, nu_mode="range"):
    runs, b = {}, None
    for f in sorted(glob.glob(os.path.join(d, "ale_*.json"))):
        x = json.load(open(f))
        # each process has its own wall-clock calibration: express the wall axis as a fraction of that
        # process's own wall budget (budget 1); the operation axis is deterministic and identical
        if b is not None and abs(x["budget_ops"] / b[0] - 1) > 1e-9:
            raise SystemExit(f"operation budgets differ between files ({f})")
        b = (x["budget_ops"], 1.0)
        for k, v in x["arms"].items():
            for r in v:
                r["cal"] = [c / x["budget_cal"] for c in r["cal"]]
            runs[k] = v
    if runs:
        va = verdict_ale(runs, *b, nu_mode=nu_mode)
        print(f"noise floor definition: {nu_mode}")
        print("=== mode ale (R5.2, cold start, step-limited)")
        if "axes" in va:
            print(f"premise STEP-LIMITED: {va['premise_step_limited']} (EXACT median dE at C_b/2 "
                  f"{va['EXACT_med_half']:.3f}, at C_b {va['EXACT_med_full']:.3f} mHa)")
            for ax, r in va["axes"].items():
                print(f"[{ax}] ctrl*={r['ctrl_star']} D={r['D']:.3f} nu={r['nu']:.3f} route*={r['route_star']} "
                      f"Rh={r['Rh']:.3f} Rf={r['Rf']:.3f} best route(C_b/2)={r['best_route_half']:.3f} "
                      f"best ctrl(C_b)={r['best_ctrl_full']:.3f} -> {r['verdict']}")
                print("      medians at C_b:   " + "  ".join(f"{k}={m:.3f}" for k, m in r["medians_full"].items()))
                print("      medians at C_b/2: " + "  ".join(f"{k}={m:.3f}" for k, m in r["medians_half"].items()))
        print(f"VERDICT R5.2 cold-start: {va['verdict']}")
    fj = {}
    for f in sorted(glob.glob(os.path.join(d, "fj*_K*_s*.json"))):
        x = json.load(open(f))
        for r in x["runs"]:
            fj[(x["fj"], x["K"], r["seed"])] = r["dE_mHa"]
    if fj:
        print("=== mode fj (F_A2 accuracy premise)")
        for k, v in sorted(fj.items()):
            print(f"fj={k[0]} K={k[1]} seed={k[2]}: dE={v:.4f} mHa")
        ctrl = fj.get((0, 12, 0))
        test = fj.get((1, 12, 0))
        if ctrl is not None and abs(ctrl - 0.390) > 0.01:
            v = f"INVALID (control {ctrl:.4f} mHa, expected 0.390)"
        elif test is None:
            v = "INCOMPLETE"
        else:
            v = "KILL-F_A2" if test > TAU_MHA else "PREMISE-HOLDS"
        print(f"VERDICT F_A2 premise: {v}")


def main(argv=None):
    a = parse_args(argv)
    if a.aggregate:
        aggregate(a.aggregate if os.path.isabs(a.aggregate) else os.path.join(HERE, a.aggregate), a.nu)
        return
    t0 = time.time()
    (run_ale if a.mode == "ale" else run_fj)(a)
    print(f"done ({time.time() - t0:.0f}s) -> {a.out}")


if __name__ == "__main__":
    main()
