#!/usr/bin/env python3
"""
P4.1 round-7 T2 aggregation and pre-registered verdicts: T2a (R2 `pcg-scal` at 12 sites) and premise W (R3).
=============================================================================================================

Usage:
    python3 t2_report.py --mode {check, report, preflight} [--resdir PATH] [--timing PATH] [--dmrgdir PATH]

Modes
-----
  check      Synthetic results covering every branch of both rules; prints each scenario, its verdict and
             whether it is the intended one, then "CHECK PASS n/n" (save to logs/check_output.txt).
  report     Reads the production results and prints the verdicts:
               T2a: <resdir>/t2a/*.json (pcg, r1, foil; seed 1 if present), the che pilot JSON
                    ../P4.1_cheap_su2_scaling/logs/pilot/pg_che/*_che_seed0.json, and the cost ratio from --timing.
               W:   <resdir>/w12/*.json (che at H = 4, 8, 32; H = 16 is the pilot), <resdir>/w24/*.json (che at
                    H = 16 and 48, 24 sites), and the 24-site (+,+) reference from scaling_report.load_refs(--dmrgdir).
  preflight  Runs both rules on the toy pipeline outputs in --resdir (vmc_t2.py --mode pipeline) with real files,
             plus the controls built from those files (positive: the pcg slot replaced by an exact state; negative:
             the pcg slot replaced by the che run itself; W positive / negative likewise), and prints each verdict.

Options
-------
  --resdir PATH   results root (default <script dir>/results)
  --timing PATH   timing JSON from vmc_t2.py --mode timing with a COMPILED Pfaffian (default none: KILL-COST not
                  evaluated; the pure-torch Parlett-Reid timing is implementation-bound, see logs/pfaffian_microbench.txt)
  --dmrgdir PATH  DMRG JSONs (default ../P4.1_cheap_su2_scaling/dmrg)

Pre-registered rules (copied verbatim into the leaf before any production output)
-------------------------------------------------------------------------------
T2a, R2 `pcg-scal`, eps = (E - E0)/|E0| exact over the (+,+) sector, rho = eps_pcg / eps_che (che = pilot, 2.858e-3):
  INVALID     any arm with |<S^2>| > 1e-10 (pcg, r1, foil are exact singlets by construction: a bug)
  INCOMPLETE  a gating arm (pcg, r1) missing, or capped by --max-hours without an early kill
  NO-TARGET   eps_che < 1e-3 (che saturated; nothing to beat)
  KILL-COST   t_step(pcg)/t_step(che) > 5 at 48 sites, measured with a compiled Pfaffian (checked first; required
              before any 24-site T2b)
  KILL        the pcg run was stopped by the pre-registered futility clause (early_kill set), or
              rho >= 1, or eps_pcg >= eps_r1
  PASS        rho <= 0.8 and eps_pcg <= 0.8 eps_r1
  AMBIGUOUS   otherwise -> seed 1 of pcg (and r1); the rule is re-applied to the best of seeds of each arm
  Futility (in-job, pcg only): exact eps of the current parameters at step 250 > 2.29e-2 (= 8 eps_che), or at step
  500 > 1.43e-2 (= 5 eps_che).
W, premise for R3 (che only; one seed):
  W-12 (descriptive output): eps(H) at H = 4, 8, 16 (pilot), 32; H*(12) = smallest H of the ladder whose doubling
       lowers eps by less than 10%; r12 = (eps16 - eps32)/eps16.
  INCOMPLETE  any of the W-12 or W-24 runs missing or capped
  W FAILS     E48 >= E16 - 2 sigma (sigma = sqrt(dE16^2 + dE48^2), sampling errors), whatever the reference
  PENDING-REFERENCE  first clause holds but no valid 24-site (+,+) reference (scaling_report.select_ref status OK)
  NO-TARGET   E16 - E_ref <= 2 (dE16 + sigma_ref) (che at H = 16 saturated at 24 sites)
  W HOLDS     E48 < E16 - 2 sigma and r24 = (E16 - E48)/(E16 - E_ref) > r12
  W FAILS     otherwise
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCAL = os.path.join(HERE, "..", "P4.1_cheap_su2_scaling")
E0_12 = -7.827899034271753
PILOT_CHE = os.path.join(SCAL, "logs", "pilot", "pg_che", "S4x3_N10_U8_tp-0.25_pg11_che_seed0.json")
COST_KILL = 5.0
FUT = {250: 8.0, 500: 5.0}  # futility multiples of eps_che (pilot)


def eps_of(r, E0):
    return (r["E"] - E0) / abs(E0)


# ============================================================================ T2a rule
def rule_t2a(che, pcg, r1, cost_ratio, E0=E0_12, foil=None):
    """che: dict (E); pcg, r1: lists of seed dicts (E, S2, capped, early_kill); cost_ratio: float or None."""
    notes = []
    for name, runs in (("pcg", pcg), ("r1", r1), ("foil", foil or [])):
        for r in runs:
            if r.get("S2") is not None and abs(r["S2"]) > 1e-10:
                return "INVALID", [f"{name} seed {r.get('seed')}: <S^2> = {r['S2']:.1e} (exact singlet expected)"]
    if cost_ratio is not None and cost_ratio > COST_KILL:
        return "KILL-COST", [f"t_step(pcg)/t_step(che) at 48 sites = {cost_ratio:.2f} > {COST_KILL}"]
    if cost_ratio is None:
        notes.append("cost ratio at 48 sites not measured")
    if not pcg or not r1 or che is None:
        return "INCOMPLETE", notes + ["missing gating arm (che pilot, pcg or r1)"]
    for r in pcg:
        if r.get("early_kill"):
            e = eps_of(r, E0)
            return "KILL", notes + [f"pcg seed {r.get('seed')} stopped by futility at step {r['early_kill']} (eps {e:.3e})"]
    if any(r.get("capped") for r in pcg + r1):
        return "INCOMPLETE", notes + ["a gating run hit --max-hours (resubmit; it resumes from its checkpoint)"]
    ec = eps_of(che, E0)
    if ec < 1e-3:
        return "NO-TARGET", notes + [f"eps_che = {ec:.2e} < 1e-3"]
    ep = min(eps_of(r, E0) for r in pcg)
    er = min(eps_of(r, E0) for r in r1)
    rho = ep / ec
    notes.append(f"eps_che {ec:.3e}, eps_pcg {ep:.3e} (best of {len(pcg)}), eps_r1 {er:.3e} (best of {len(r1)}), "
                 f"rho {rho:.3f}, eps_pcg/eps_r1 {ep / er:.3f}")
    if foil:
        notes.append(f"foil (descriptive): eps {min(eps_of(r, E0) for r in foil):.3e}")
    if rho >= 1 or ep >= er:
        return "KILL", notes
    if rho <= 0.8 and ep <= 0.8 * er:
        return "PASS", notes
    return ("AMBIGUOUS" if len(pcg) == 1 else "AMBIGUOUS-FINAL"), notes


# ============================================================================ W rule
def rule_w(w12, w24, ref24):
    """w12: {H: eps} at 12 sites (needs 16 and 32; 4 and 8 descriptive); w24: {H: (E, dE, capped)} for 16, 48;
    ref24: (E_ref, sigma_ref, status) or None."""
    notes = []
    if any(h not in w12 for h in (4, 8, 16, 32)) or any(h not in w24 for h in (16, 48)):
        return "INCOMPLETE", ["missing W run(s)"]
    if any(v[2] for v in w24.values()) or any(isinstance(v, tuple) for v in w12.values()):
        return "INCOMPLETE", ["a W run hit --max-hours"]
    Hs = sorted(w12)
    hstar = next((h for h, h2 in zip(Hs, Hs[1:]) if h2 == 2 * h and (w12[h] - w12[h2]) / w12[h] < 0.1), None)
    r12 = (w12[16] - w12[32]) / w12[16]
    notes.append("W-12 eps(H): " + ", ".join(f"H={h}: {w12[h]:.3e}" for h in Hs)
                 + f"; H*(12) = {hstar if hstar is not None else '> ' + str(Hs[-1])}; r12 = {r12:.3f}")
    (E16, d16, _), (E48, d48, _) = w24[16], w24[48]
    sig = math.sqrt(d16 ** 2 + d48 ** 2)
    notes.append(f"W-24: E16 {E16:.5f} +- {d16:.5f}, E48 {E48:.5f} +- {d48:.5f}, (E16 - E48)/sigma = {(E16 - E48) / sig:.2f}")
    if not E48 < E16 - 2 * sig:
        return "W FAILS", notes + ["H = 48 does not lower the 24-site energy beyond 2 sigma"]
    if ref24 is None or ref24[2] != "OK":
        return "PENDING-REFERENCE", notes + ["no valid 24-site (+,+) reference yet (job 687302)"]
    Er, sr, _ = ref24
    if E16 - Er <= 2 * (d16 + sr):
        return "NO-TARGET", notes + [f"E16 - E_ref = {E16 - Er:.5f} within 2 (dE16 + sigma_ref)"]
    r24 = (E16 - E48) / (E16 - Er)
    notes.append(f"r24 = {r24:.3f} vs r12 = {r12:.3f}")
    return ("W HOLDS" if r24 > r12 else "W FAILS"), notes


# ============================================================================ check
def check():
    che = dict(E=E0_12 + 2.858e-3 * abs(E0_12))

    def run(eps, seed=0, **kw):
        d = dict(E=E0_12 + eps * abs(E0_12), S2=0.0, capped=False, early_kill=None, seed=seed)
        d.update(kw)
        return d
    T = [
        ("T2a PASS", dict(che=che, pcg=[run(1.5e-3)], r1=[run(2.5e-3)], cost_ratio=3.0), "PASS"),
        ("T2a KILL rho>=1", dict(che=che, pcg=[run(3.0e-3)], r1=[run(5e-3)], cost_ratio=3.0), "KILL"),
        ("T2a KILL pcg>=r1", dict(che=che, pcg=[run(2.0e-3)], r1=[run(1.9e-3)], cost_ratio=3.0), "KILL"),
        ("T2a KILL futility", dict(che=che, pcg=[run(3e-2, early_kill=250)], r1=[run(2e-3)], cost_ratio=3.0), "KILL"),
        ("T2a KILL-COST", dict(che=che, pcg=[run(1e-3)], r1=[run(2e-3)], cost_ratio=6.0), "KILL-COST"),
        ("T2a AMBIGUOUS (rho 0.9)", dict(che=che, pcg=[run(2.57e-3)], r1=[run(4e-3)], cost_ratio=3.0), "AMBIGUOUS"),
        ("T2a AMBIGUOUS (pcg 0.9 r1)", dict(che=che, pcg=[run(1.8e-3)], r1=[run(2.0e-3)], cost_ratio=3.0), "AMBIGUOUS"),
        ("T2a best-of-seeds PASS", dict(che=che, pcg=[run(2.57e-3), run(1.5e-3, 1)], r1=[run(2.5e-3), run(2.6e-3, 1)],
                                        cost_ratio=3.0), "PASS"),
        ("T2a AMBIGUOUS after seed 1", dict(che=che, pcg=[run(2.57e-3), run(2.6e-3, 1)], r1=[run(4e-3), run(4e-3, 1)],
                                             cost_ratio=3.0), "AMBIGUOUS-FINAL"),
        ("T2a NO-TARGET", dict(che=dict(E=E0_12 + 5e-4 * abs(E0_12)), pcg=[run(1e-4)], r1=[run(2e-4)], cost_ratio=3.0),
         "NO-TARGET"),
        ("T2a INCOMPLETE missing r1", dict(che=che, pcg=[run(1e-3)], r1=[], cost_ratio=3.0), "INCOMPLETE"),
        ("T2a INCOMPLETE capped", dict(che=che, pcg=[run(1e-3, capped=True)], r1=[run(2e-3)], cost_ratio=3.0),
         "INCOMPLETE"),
        ("T2a INVALID S^2", dict(che=che, pcg=[run(1e-3, S2=1e-3)], r1=[run(2e-3)], cost_ratio=3.0), "INVALID"),
    ]
    ok = 0
    n = 0
    for name, kw, want in T:
        v, nt = rule_t2a(**kw)
        n += 1
        ok += v == want
        print(f"[{'OK ' if v == want else 'BAD'}] {name:32s} -> {v:16s} (intended {want}) | {'; '.join(nt)}")
    w12 = {4: 5e-3, 8: 3.5e-3, 16: 2.86e-3, 32: 2.6e-3}
    ref = (-16.60, 0.002, "OK")
    W = [
        ("W HOLDS", dict(w12=w12, w24={16: (-16.50, 0.003, False), 48: (-16.53, 0.003, False)}, ref24=ref), "W HOLDS"),
        ("W FAILS no 2sigma gain", dict(w12=w12, w24={16: (-16.50, 0.003, False), 48: (-16.505, 0.003, False)},
                                        ref24=ref), "W FAILS"),
        ("W FAILS r24<=r12", dict(w12=w12, w24={16: (-16.50, 0.001, False), 48: (-16.505, 0.001, False)}, ref24=ref),
         "W FAILS"),
        ("W FAILS without reference", dict(w12=w12, w24={16: (-16.50, 0.003, False), 48: (-16.50, 0.003, False)},
                                           ref24=None), "W FAILS"),
        ("W PENDING-REFERENCE", dict(w12=w12, w24={16: (-16.50, 0.003, False), 48: (-16.53, 0.003, False)},
                                     ref24=(None, None, "AMBIGUOUS-REFERENCE")), "PENDING-REFERENCE"),
        ("W NO-TARGET", dict(w12=w12, w24={16: (-16.598, 0.0005, False), 48: (-16.5995, 0.0001, False)},
                             ref24=(-16.6, 0.0006, "OK")), "NO-TARGET"),
        ("W INCOMPLETE missing", dict(w12={16: 2.8e-3, 32: 2.6e-3}, w24={16: (-16.5, 0.003, False)}, ref24=ref),
         "INCOMPLETE"),
        ("W INCOMPLETE capped", dict(w12=w12, w24={16: (-16.50, 0.003, False), 48: (-16.53, 0.003, True)}, ref24=ref),
         "INCOMPLETE"),
    ]
    for name, kw, want in W:
        v, nt = rule_w(**kw)
        n += 1
        ok += v == want
        print(f"[{'OK ' if v == want else 'BAD'}] {name:32s} -> {v:16s} (intended {want}) | {'; '.join(nt)}")
    print(f"CHECK {'PASS' if ok == n else 'FAIL'} {ok}/{n}")
    return ok == n


# ============================================================================ file readers
def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def cost_ratio_from(path):
    if path is None or not os.path.exists(path):
        return None
    t = json.load(open(path))
    rows = [r for r in t["rows"] if r["L"] == 48]
    che = [r["t_step"] for r in rows if r["arm"] == "che" and r["H"] == 16]
    pcg = [r["t_step"] for r in rows if r["arm"] == "pcg"]
    return pcg[0] / che[0] if che and pcg else None


def newest_timing():
    fs = sorted(glob.glob(os.path.join(HERE, "logs", "timing", "timing_t2_*.json")), key=os.path.getmtime)
    return fs[-1] if fs else None


def report(a):
    che = json.load(open(PILOT_CHE)) if os.path.exists(PILOT_CHE) else None
    rs = load(os.path.join(a.resdir, "t2a", "S4x3_*.json"))
    cr = cost_ratio_from(a.timing) if a.timing else None  # only a compiled-Pfaffian timing decides KILL-COST
    pcg = [r for r in rs if r["arm"] == "pcg"]
    r1 = [r for r in rs if r["arm"] == "r1"]
    foil = [r for r in rs if r["arm"] == "foil"]
    v, nt = rule_t2a(che, pcg, r1, cr, foil=foil)
    print(f"VERDICT T2a (R2 pcg-scal): {v}")
    for n_ in nt:
        print(f"  {n_}")
    for r in pcg:
        if r.get("ckpt_exact"):
            print("  pcg seed %d exact checkpoints: " % r["seed"]
                  + ", ".join(f"{k}: {c['eps']:.3e}" for k, c in sorted(r["ckpt_exact"].items(), key=lambda x: int(x[0]))))
    w12 = {16: eps_of(che, E0_12)} if che else {}
    for r in load(os.path.join(a.resdir, "w12", "S4x3_*.json")):
        w12[r["H"]] = (eps_of(r, E0_12), "capped") if r["capped"] else eps_of(r, E0_12)
    w24 = {r["H"]: (r["E"], r["dE"], r["capped"]) for r in load(os.path.join(a.resdir, "w24", "S8x3_*.json"))}
    sys.path.insert(0, SCAL)
    ref24 = None
    try:
        import scaling_report as SR
        ref24 = SR.load_refs(a.dmrgdir).get(24)
    except Exception as e:  # noqa: BLE001
        print(f"  (reference loader failed: {e})")
    v, nt = rule_w(w12, w24, ref24)
    print(f"VERDICT W (R3 width premise): {v}")
    for n_ in nt:
        print(f"  {n_}")


def preflight(a):
    """toy pipeline (4x2) on real files: the rules parse them; controls built from the same files."""
    rs = load(os.path.join(a.resdir, "S4x2_*.json"))
    by = {(r["arm"], r["H"]): r for r in rs}
    E0 = rs[0]["E0"]
    che = next(r for (arm, H), r in by.items() if arm == "che" and H == 16)
    che2 = next(r for (arm, H), r in by.items() if arm == "che" and H == 2)
    pcg = [r for r in rs if r["arm"] == "pcg"]
    r1 = [r for r in rs if r["arm"] == "r1"]
    print(f"toy 4x2: E0 {E0:.6f}; " + ", ".join(f"{r['arm']} H={r['H']} eps {eps_of(r, E0):.3e} S2 {r['S2']:.1e} "
                                               f"ckpt {list(r['ckpt_exact'])}" for r in rs))
    out = []
    v, nt = rule_t2a(che, pcg, r1, 3.0, E0=E0)
    out.append(("real toy files", v, None, nt))
    exact = dict(che, E=E0, seed=0, S2=0.0, capped=False, early_kill=None)
    v, nt = rule_t2a(che, [exact], r1, 3.0, E0=E0)
    out.append(("positive control (pcg slot = exact state)", v, "PASS", nt))
    v, nt = rule_t2a(che, [dict(che, seed=0, S2=0.0, early_kill=None)], r1, 3.0, E0=E0)
    out.append(("negative control (pcg slot = che itself)", v, "KILL", nt))
    fc = load(os.path.join(a.resdir, "futility_ctrl", "S4x2_*.json"))
    v, nt = rule_t2a(che, fc, r1, 3.0, E0=E0)
    out.append(("futility control (real early-killed pcg file)", v, "KILL", nt))
    # W on toy files: 'H = 16' slot = che H=2 run, 'H = 48' slot = che H=16 run or itself
    w12 = {4: 3e-3, 8: 2e-3, 16: 1e-3, 32: 0.9e-3}
    e2 = (che2["E"], 1e-4, False)
    v, nt = rule_w(w12, {16: e2, 48: (E0, 1e-6, False)}, (E0, 0.0, "OK"))
    out.append(("W positive control (H48 slot = exact state)", v, "W HOLDS", nt))
    v, nt = rule_w(w12, {16: e2, 48: e2}, (E0, 0.0, "OK"))
    out.append(("W negative control (both slots = same run)", v, "W FAILS", nt))
    ok = True
    for name, v, want, nt in out:
        good = want is None or v == want
        ok &= good
        print(f"[{'OK ' if good else 'BAD'}] {name:48s} -> {v}" + (f" (intended {want})" if want else "")
              + f" | {'; '.join(nt)}")
    print(f"PREFLIGHT {'PASS' if ok else 'FAIL'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["check", "report", "preflight"])
    p.add_argument("--resdir", default=os.path.join(HERE, "results"))
    p.add_argument("--timing", default=None)
    p.add_argument("--dmrgdir", default=os.path.join(SCAL, "dmrg"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "check":
        sys.exit(0 if check() else 1)
    {"report": report, "preflight": preflight}[a.mode](a)


if __name__ == "__main__":
    main()
