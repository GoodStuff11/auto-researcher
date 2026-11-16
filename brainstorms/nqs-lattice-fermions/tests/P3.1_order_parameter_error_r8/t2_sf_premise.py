#!/usr/bin/env python3
"""
P3.1 round 8, Route 8.3 (structure-factor targeting) -- tier-2 premise pilot on the real trained states
=====================================================================================================

Question. On the shared 4x4 OBC t-t'-J instance (6+6, D = 1681680) the long-range real-space aggregates
need K_1/2 = 2 (pair), 6 (spin), 11 (charge) exact in-irrep levels (job 454376). The tier-1 toy run
(t1_portfolio.py) found the pair and charge STRUCTURE FACTORS concentrated on the lowest level (median
K_1/2 = 1 at every toy size), the spin structure factor not. Does that hold on the real converged
width-6 projected NNBF states, and does ESR0 with the three trained excited states (job 456094) recover
the structure-factor errors?

Computed (exact sums, no sampling, no training):
  * ED, the 30 lowest in-irrep (chi, S = 0) levels |1..30> (r6.sector_eigs, as excited_pilot.py).
  * For each available seed s in {11, 22, 33}: psi_s = unit(P_chi NNBF_w6_s) (P2.1 r3 parameters), and
    for the six observables pair, spin, charge (r5_core long-range aggregates) and pairSF, spinSF,
    chargeSF (t1_portfolio.sf_operators: d-wave pair SF at q = 0, spin and charge SF at (pi, pi)):
    a_0, delta = a_0 - a_psi, scored iff |delta| >= 0.10 |a_0|, L1R ratio, ORC(K) ratio for K = 1..30,
    K_1/2 (smallest K with every ORC(K' >= K) ratio in [1/2, 2]; inf if ORC(30) is outside).
  * Seed 11 only: ESR0(K) ratio with the trained phi_1..phi_K (K = 3, plus K = 4..6 if job 618646 has
    written them at run time; phi_k = P_S P_chi NNBF(seed 11 + 100 k) from
    ../P3.1_order_parameter_error_r7/results_excpilot/params).
  * Reference reproduction (pre-flight requirement, decides validity): the seed-11 ESR0(K=3) ratios of
    pair, spin, charge must reproduce job 456094 (1.173, 0.187, 0.102) within 0.005; otherwise INVALID.

Pre-registered verdict (also in the leaf, Round 8, Route 8.3, written before submission):
  NO-TARGET  if fewer than 2 of the 3 SF observables are scored on at least 2 of the available seeds.
  PASS       if, for every SF observable that is scored on >= 2 seeds, the median K_1/2 over its scored
             seeds is <= 3, AND on seed 11 the ESR0(K=3) ratio lies in [1/2, 2] for every scored SF
             observable. -> design the tier-3 test (sampled ESR0 on SF observables, seeds 11/22/33).
  KILL       if every scored SF observable has median K_1/2 > 6, OR the seed-11 ESR0(K=3) ratio is
             outside [1/2, 2] for every scored SF observable.
  AMBIGUOUS  otherwise -> rescore with K = 6 when job 618646 has finished (same rule with K = 6 in
             place of K = 3, and K_1/2 <= 6 in place of <= 3); not PASS then -> KILL.
  Early stop (checkpoint after each seed, results JSON rewritten): if after two seeds no SF observable is
  scored on any seed, stop with NO-TARGET.

Usage
  python3 t2_sf_premise.py --instance main --out results/t2_sf_premise.json       (Slurm, see sbatch)
  python3 t2_sf_premise.py --instance small --out results/dry_small.json           (pre-flight dry run)
  python3 t2_sf_premise.py --selftest-verdict                                      (verdict branches)
Options
  --instance {main, small}  main: 4x4 OBC 6+6 with the real P2.1 states and trained phi_k.
                            small: 4x4 OBC 2+2 (D = 10920), machinery only: the "seeds" are synthetic
                            states unit(P_chi psi_0 exp(0.3 xi)) and the "trained phi_k" are exact levels
                            plus 0.2 random in-window rotation (no physics claim); reference check skipped.
  --n-low INT               number of exact in-irrep levels (default 30).
  --seeds LIST              comma list of P2.1 seeds (default 11,22,33; missing files are skipped).
  --chunk INT               amplitude-evaluation chunk size (default 32768).
  --out PATH                results JSON (rewritten after every seed).
  --selftest-verdict        feed synthetic records covering every verdict branch to the verdict function
                            and print the verdicts; no physics.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R7 = os.path.join(HERE, "..", "P3.1_order_parameter_error_r7")
P21 = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r3", "results", "params")
PHI = os.path.join(R7, "results_excpilot", "params")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r7 = _load("r7", os.path.join(R7, "micro_test.py"))
r6, rc, log = r7.r6, r7.rc, r7.log
t1 = _load("t1", os.path.join(HERE, "t1_portfolio.py"))
REAL, SF, ALL = t1.REAL, t1.SF, t1.ALL
REF_456094 = {"pair": 1.173, "spin": 0.187, "charge": 0.102}


def unit(v):
    return v / np.linalg.norm(v)


def ritz_shifts(m, apply, vecs, names):
    B = np.array([unit(v) for v in vecs])
    HB = np.array([m.H @ b for b in B])
    A = {k: B @ np.array([apply(k, b) for b in B]).T for k in names}
    return r7.ritz_shift(B @ B.T, B @ HB.T, A, 1e-11)


def k_half(ratios):
    ok = [(0.5 <= r <= 2.0) for r in ratios]
    if not ok[-1]:
        return math.inf
    K = len(ok)
    while K - 1 >= 1 and ok[K - 2]:
        K -= 1
    return K


def verdict(seeds, Kesr=3):
    """seeds: {seed: {obs: dict(scored, Khalf, ESR0={K: ratio})}}. Returns (verdict, details)."""
    sc = {k: [s for s, d in seeds.items() if d[k]["scored"]] for k in SF}
    target = [k for k in SF if len(sc[k]) >= 2]
    if len(target) < 2:
        return "NO-TARGET", dict(scored_seeds=sc)
    medK = {k: float(np.median([seeds[s][k]["Khalf"] for s in sc[k]])) for k in target}
    s11 = seeds.get(11, {})
    e3 = {k: s11.get(k, {}).get("ESR0", {}).get(Kesr) for k in target if s11.get(k, {}).get("scored")}
    ok3 = {k: (r is not None and 0.5 <= r <= 2.0) for k, r in e3.items()}
    kcap = 3 if Kesr == 3 else 6
    if all(medK[k] <= kcap for k in target) and e3 and all(ok3.values()):
        v = "PASS"
    elif all(medK[k] > 6 for k in target) or (e3 and not any(ok3.values())):
        v = "KILL"
    else:
        v = "AMBIGUOUS"
    return v, dict(scored_seeds=sc, medKhalf=medK, ESR0=e3)


def selftest():
    def rec(sc, K, e=None):
        return dict(scored=sc, Khalf=K, ESR0={3: e} if e is not None else {})
    base = {k: rec(False, 1) for k in REAL}
    cases = {
        "NO-TARGET": {11: {**base, "pairSF": rec(True, 1, 1.0), "spinSF": rec(False, 9), "chargeSF": rec(False, 1)},
                      22: {**base, "pairSF": rec(True, 1), "spinSF": rec(False, 9), "chargeSF": rec(False, 1)}},
        "PASS": {11: {**base, "pairSF": rec(True, 1, 1.1), "spinSF": rec(False, 9), "chargeSF": rec(True, 2, 0.8)},
                 22: {**base, "pairSF": rec(True, 1), "spinSF": rec(False, 9), "chargeSF": rec(True, 3)}},
        "KILL": {11: {**base, "pairSF": rec(True, 9, 0.2), "spinSF": rec(True, 12, 0.1), "chargeSF": rec(True, 1, 0.3)},
                 22: {**base, "pairSF": rec(True, 9), "spinSF": rec(True, 15), "chargeSF": rec(True, 1)}},
        "AMBIGUOUS": {11: {**base, "pairSF": rec(True, 5, 1.0), "spinSF": rec(False, 9), "chargeSF": rec(True, 2, 0.3)},
                      22: {**base, "pairSF": rec(True, 4), "spinSF": rec(False, 9), "chargeSF": rec(True, 2)}},
    }
    ok = True
    for want, seeds in cases.items():
        got, det = verdict(seeds)
        ok &= got == want
        print(f"selftest: expected {want:9s} got {got:9s} {det}")
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def run(args):
    t0 = time.time()
    m, E0, psi0, sg, S0, ob = r6.setup(args.instance)
    psi0 = psi0 if psi0.sum() > 0 else -psi0
    if args.instance == "main":
        assert m.D == 1681680 and S0 == 0, (m.D, S0)
    sf = t1.sf_operators(m, ob)

    def apply(k, v):
        return t1.apply_obs(ob, sf, k, v)
    e, U = r6.sector_eigs(m, sg, S0, args.n_low + 1)
    assert abs(e[0] - E0) < 1e-6, (e[0], E0)
    om = e[1:] - E0
    U = U[:, 1:]
    nl = U.shape[1]
    log(f"{nl} in-irrep levels: omega = {np.round(om, 4).tolist()} ({time.time() - t0:.0f} s)")
    a0 = {k: float(psi0 @ apply(k, psi0)) for k in ALL}
    log("a0 = " + " ".join(f"{k} {a0[k]:+.4e}" for k in ALL))
    HU = np.array([m.H @ U[:, n] for n in range(nl)])
    AU = {k: np.array([apply(k, U[:, n]) for n in range(nl)]) for k in ALL}
    out = dict(args=vars(args), E0=E0, omega=om.tolist(), a0=a0, seeds={}, ref_check=None)
    # ------------------------------------------------------------- targets and trained phi
    targets, phis = {}, []
    if args.instance == "main":
        for s in [int(x) for x in args.seeds.split(",")]:
            f = os.path.join(P21, f"inst0_nnbf_s{s}.npz")
            if not os.path.exists(f):
                log(f"seed {s}: no parameter file; skipped"); continue
            ans = rc.load_params(rc.nnbf(m, 6, s), f)
            v = unit(sg.project(ans.amplitudes(args.chunk)))
            targets[s] = v if v @ psi0 > 0 else -v
        for k in range(1, 7):
            f = os.path.join(PHI, f"phi{k}_s11_w6_main.npz")
            if not os.path.exists(f):
                break
            ans = rc.load_params(rc.nnbf(m, 6, 11 + 100 * k), f)
            phis.append(unit(m.spin_project(sg.project(ans.amplitudes(args.chunk)), S0)))
        log(f"targets: seeds {list(targets)}; trained phi_1..{len(phis)} loaded")
    else:
        rng = np.random.default_rng(5)
        for s in (11, 22):
            targets[s] = unit(sg.project(psi0 * np.exp(0.3 * rng.standard_normal(m.D))))
        for k in range(min(3, nl)):
            g = U[:, :min(nl, 6)] @ rng.standard_normal(min(nl, 6))
            phis.append(unit(U[:, k] + 0.2 * unit(g)))
        log("small: synthetic targets and synthetic phi (machinery only)")
    # ------------------------------------------------------------- per seed
    for s, psi in targets.items():
        ts = time.time()
        Epsi = float(psi @ (m.H @ psi))
        r = m.H @ psi - Epsi * psi
        base = [psi, unit(r)]
        Bb = np.array(base)
        S_full = np.block([[Bb @ Bb.T, Bb @ U], [U.T @ Bb.T, np.eye(nl)]])
        HBb = np.array([m.H @ b for b in Bb])
        H_full = np.block([[Bb @ HBb.T, Bb @ HU.T], [U.T @ HBb.T, U.T @ HU.T]])
        rec = dict(dE=Epsi - E0, overlap2=float((psi0 @ psi) ** 2))
        for k in ALL:
            Ab = np.array([apply(k, b) for b in Bb])
            A_full = np.block([[Bb @ Ab.T, Bb @ AU[k].T], [U.T @ Ab.T, U.T @ AU[k].T]])
            A_full = 0.5 * (A_full + A_full.T)
            apsi = float(psi @ apply(k, psi))
            d = a0[k] - apsi
            rat = []
            for K in range(0, nl + 1):
                idx = np.arange(K + 2)
                sh = r7.ritz_shift(S_full[np.ix_(idx, idx)], H_full[np.ix_(idx, idx)],
                                   {k: A_full[np.ix_(idx, idx)]}, 1e-11)[k]
                rat.append(sh / d)
            esr = {}
            if s == 11 and phis:
                for K in range(1, len(phis) + 1):
                    esr[K] = ritz_shifts(m, apply, base + phis[:K], [k])[k] / d
            rec[k] = dict(a_psi=apsi, delta=d, rel=d / a0[k] if a0[k] else math.nan,
                          scored=bool(abs(d) >= 0.10 * abs(a0[k])), L1R=rat[0], ORC=rat[1:],
                          Khalf=k_half(rat[1:]), ESR0=esr)
            log(f"seed {s} {k:8s} a0 {a0[k]:+.4e} delta {d:+.3e} (rel {rec[k]['rel']:+.3f}) "
                f"{'scored' if rec[k]['scored'] else 'not scored'} | L1R {rat[0]:+.3f} ORC3 {rat[3]:+.3f} "
                f"ORC6 {rat[6]:+.3f} ORC{nl} {rat[nl]:+.3f} K_1/2 {rec[k]['Khalf']} | ESR0 "
                + " ".join(f"K={K}:{x:+.3f}" for K, x in esr.items()))
        out["seeds"][s] = rec
        log(f"seed {s} done ({time.time() - ts:.0f} s); dE {rec['dE']:.5f} overlap^2 {rec['overlap2']:.4f}")
        json.dump(out, open(args.out, "w"), indent=1, default=float)
        done = list(out["seeds"])
        if len(done) == 2 and not any(out["seeds"][x][k]["scored"] for x in done for k in SF):
            log("EARLY STOP: no SF observable scored on the first two seeds -> NO-TARGET")
            out["verdict"] = "NO-TARGET (early stop)"
            json.dump(out, open(args.out, "w"), indent=1, default=float)
            return 0
    # ------------------------------------------------------------- reference check and verdict
    if args.instance == "main" and 11 in out["seeds"] and len(phis) >= 3:
        dev = {k: abs(out["seeds"][11][k]["ESR0"][3] - REF_456094[k]) for k in REAL}
        ok = all(x <= 0.005 for x in dev.values())
        out["ref_check"] = dict(dev=dev, ok=ok)
        log(f"reference reproduction (job 456094 ESR0(K=3) pair/spin/charge): deviations "
            f"{ {k: round(x, 4) for k, x in dev.items()} } -> {'OK' if ok else 'INVALID'}")
    seeds = {s: {k: dict(scored=d[k]["scored"], Khalf=d[k]["Khalf"], ESR0=d[k]["ESR0"]) for k in ALL}
             for s, d in out["seeds"].items()}
    v3, det3 = verdict(seeds, 3)
    if out.get("ref_check") and not out["ref_check"]["ok"]:
        v3 = "INVALID"
    out["verdict_K3"], out["verdict_details_K3"] = v3, det3
    log(f"ROUTE 8.3 T2 VERDICT (K=3): {v3} | {det3}")
    if len(phis) >= 6:
        v6, det6 = verdict(seeds, 6)
        out["verdict_K6"], out["verdict_details_K6"] = v6, det6
        log(f"ROUTE 8.3 T2 VERDICT (K=6 rescore): {v6} | {det6}")
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    log(f"done ({time.time() - t0:.0f} s)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", choices=("main", "small"), default="main")
    p.add_argument("--n-low", type=int, default=30)
    p.add_argument("--seeds", default="11,22,33")
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--out", default=os.path.join(HERE, "results", "t2_sf_premise.json"))
    p.add_argument("--selftest-verdict", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest_verdict:
        return selftest()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
