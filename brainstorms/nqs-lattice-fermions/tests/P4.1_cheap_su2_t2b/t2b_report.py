#!/usr/bin/env python3
"""
P4.1 R2 `pcg-scal` T2b aggregation and pre-registered verdict (24 sites against 12 sites, 2026-11-16).
=====================================================================================================

Usage:
    python3 t2b_report.py --mode {check, report, preflight} [--resdir PATH] [--dmrgdir PATH]

Modes
-----
  check      Synthetic results covering every branch of the rule; prints each scenario, its verdict and whether it is
             the intended one, then "CHECK PASS n/n" (save to logs/check_output.txt).
  report     Reads <resdir>/t2b/S8x3_*.json and S4x3_*.json (arms pcg, che, r1; seed 1 if present), the 24-site (+,+)
             reference from scaling_report.load_refs(--dmrgdir) (job 687302), and prints the verdict, the checkpoint
             trajectories, and two context lines that never gate: rho(12) at 1000 steps (T2a, 0.449) and, if job
             686990 has produced them, che and che_k8 at 24 sites from ../P4.1_cheap_su2_scaling/results.
  preflight  Runs the rule on the toy pipeline outputs in --resdir (t2b.py --mode pipeline: s24/, s12/,
             futility_ctrl/, disarmed_ctrl/) with real files, plus controls built from them, and prints each verdict.

Options
-------
  --resdir PATH   results root (default <script dir>/results)
  --dmrgdir PATH  DMRG JSONs (default ../P4.1_cheap_su2_scaling/dmrg)

Pre-registered rule (copied verbatim into the leaf section "R2 T2b design (2026-11-16)" before any production run)
------------------------------------------------------------------------------------------------------------------
Inputs. 24 sites (open 8x3, N_e = 20, (+,+)): E +- dE (262,144 samples, chain-blocked) of pcg, che, r1,
3000 SR steps each, best of seeds per arm. 12 sites (open 4x3, N_e = 10, (+,+)): exact E of pcg, che, r1, same
3000-step protocol. E_ref(24) +- sigma_ref from the (+,+) sector DMRG (687302, select_ref status OK); E_ref(12) = ED.
  rho24 = (E_pcg - E_ref)/(E_che - E_ref), sigma_rho from dE_pcg, dE_che, sigma_ref (shared reference propagated);
  rho12 = eps_pcg/eps_che at 12 sites (exact, no error bar);  d = E_r1 - E_pcg at 24 sites, sigma_d = hypot(dE).
Order of evaluation:
  INVALID     a 12-site arm with |<S^2>| > 1e-10
  KILL        pcg at 24 sites stopped by the pre-registered futility clause (early_kill set)
  INCOMPLETE  pcg, che or r1 missing or capped at 24 sites (resubmit; tasks resume)
  KILL        d <= 0: pcg does not beat r1 at 24 sites (point estimate; needs no reference)
  INCOMPLETE-REFERENCE  no valid 24-site (+,+) reference yet
  NO-TARGET   E_che - E_ref <= 2 hypot(dE_che, sigma_ref) (che saturated at 24 sites; rho undefined)
  KILL        rho24 >= 0.85 (point estimate)
  INCOMPLETE  a 12-site arm (pcg, che) missing or capped (the 12-site baseline is needed for PASS)
  PASS        rho24 + 2 sigma_rho <= 0.60  and  (rho24 - rho12) + 2 sigma_rho <= 0.15  and  d > 2 sigma_d
  AMBIGUOUS   otherwise -> seed 1 of pcg, che and r1 at 24 sites; re-apply to the best seed of each arm
  AMBIGUOUS-FINAL  still ambiguous with two seeds
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
RHO_PASS, RHO_KILL, RISE_MAX = 0.60, 0.85, 0.15
RHO12_T2A = 0.449  # 1000-step T2a value (context only)


def rho_sigma(Ep, dEp, Ec, dEc, Er, sr):
    den = Ec - Er
    rho = (Ep - Er) / den
    return rho, math.sqrt(dEp ** 2 + (rho * dEc) ** 2 + ((1.0 - rho) * sr) ** 2) / abs(den)


def best(runs):
    return min(runs, key=lambda r: r["E"]) if runs else None


def rule_t2b(r24, r12, ref24):
    """r24, r12: {arm: [run dicts (E, dE, S2, capped, early_kill, seed)]}; ref24: (E_ref, sigma_ref, status) or None."""
    notes = []
    for arm, runs in r12.items():
        for r in runs:
            if r.get("S2") is not None and abs(r["S2"]) > 1e-10:
                return "INVALID", [f"12-site {arm} seed {r.get('seed')}: <S^2> = {r['S2']:.1e}"]
    for r in r24.get("pcg", []):
        if r.get("early_kill"):
            return "KILL", [f"pcg seed {r.get('seed')} at 24 sites stopped by the futility clause at step {r['early_kill']}"]
    for arm in ("pcg", "che", "r1"):
        if not r24.get(arm):
            return "INCOMPLETE", [f"24-site {arm} missing"]
        if any(r.get("capped") for r in r24[arm]):
            return "INCOMPLETE", [f"24-site {arm} capped (resubmit; it resumes)"]
    nseed = max(len(r24[a]) for a in ("pcg", "che", "r1"))
    p, c, r = best(r24["pcg"]), best(r24["che"]), best(r24["r1"])
    d, sd = r["E"] - p["E"], math.hypot(r["dE"], p["dE"])
    notes.append(f"24 sites: E_pcg {p['E']:.5f} +- {p['dE']:.5f}, E_che {c['E']:.5f} +- {c['dE']:.5f}, "
                 f"E_r1 {r['E']:.5f} +- {r['dE']:.5f}; E_r1 - E_pcg = {d:+.5f} ({d / sd:+.1f} sigma); seeds {nseed}")
    if d <= 0:
        return "KILL", notes + ["pcg does not beat r1 at 24 sites"]
    if ref24 is None or ref24[2] != "OK":
        return "INCOMPLETE-REFERENCE", notes + ["no valid 24-site (+,+) reference (job 687302)"]
    Er, sr = ref24[0], ref24[1]
    if c["E"] - Er <= 2 * math.hypot(c["dE"], sr):
        return "NO-TARGET", notes + [f"E_che - E_ref = {c['E'] - Er:.5f} not resolved from 0"]
    rho, sg = rho_sigma(p["E"], p["dE"], c["E"], c["dE"], Er, sr)
    notes.append(f"E_ref {Er:.6f} +- {sr:.1e}; eps_che {(c['E'] - Er) / abs(Er):.3e}, eps_pcg {(p['E'] - Er) / abs(Er):.3e},"
                 f" eps_r1 {(r['E'] - Er) / abs(Er):.3e}; rho24 = {rho:.3f} +- {sg:.3f}")
    if rho >= RHO_KILL:
        return "KILL", notes + [f"rho24 >= {RHO_KILL}"]
    for arm in ("pcg", "che"):
        if not r12.get(arm) or any(x.get("capped") for x in r12[arm]):
            return "INCOMPLETE", notes + [f"12-site {arm} (3000 steps) missing or capped"]
    e12 = {arm: (best(r12[arm])["E"] - E0_12) / abs(E0_12) for arm in r12 if r12[arm]}
    rho12 = e12["pcg"] / e12["che"]
    notes.append(f"12 sites (3000 steps): eps_che {e12['che']:.3e}, eps_pcg {e12['pcg']:.3e}"
                 + (f", eps_r1 {e12['r1']:.3e}" if "r1" in e12 else "")
                 + f"; rho12 = {rho12:.3f}; rise rho24 - rho12 = {rho - rho12:+.3f} (vs T2a 1000-step rho12 {RHO12_T2A}: "
                 f"{rho - RHO12_T2A:+.3f}, context)")
    if rho + 2 * sg <= RHO_PASS and (rho - rho12) + 2 * sg <= RISE_MAX and d > 2 * sd:
        return "PASS", notes
    why = []
    if rho + 2 * sg > RHO_PASS:
        why.append(f"rho24 + 2 sigma = {rho + 2 * sg:.3f} > {RHO_PASS}")
    if (rho - rho12) + 2 * sg > RISE_MAX:
        why.append(f"rise + 2 sigma = {rho - rho12 + 2 * sg:.3f} > {RISE_MAX}")
    if d <= 2 * sd:
        why.append("pcg below r1 but not by 2 sigma")
    return ("AMBIGUOUS" if nseed == 1 else "AMBIGUOUS-FINAL"), notes + why


# ============================================================================ check
def check():
    Er = -16.60
    E12 = E0_12

    def run(E, dE=0.004, seed=0, **kw):
        return dict(dict(E=E, dE=dE, seed=seed, S2=None, capped=False, early_kill=None), **kw)

    def e24(eps):
        return Er + eps * abs(Er)

    def e12(eps):
        return E12 + eps * abs(E12)

    base12 = {"pcg": [run(e12(1.2e-3), 0.0, S2=0.0)], "che": [run(e12(2.8e-3), 0.0)], "r1": [run(e12(2.3e-3), 0.0, S2=0.0)]}
    ref = (Er, 2e-4, "OK")

    def s24(ep, ec=1.0e-2, er=8e-3, dE=0.004, **kw):
        return {"pcg": [run(e24(ep), dE, **kw)], "che": [run(e24(ec), dE)], "r1": [run(e24(er), dE)]}

    sc = [
        ("PASS: rho24 0.40, rho12 0.43, pcg << r1", s24(4.0e-3), base12, ref, "PASS"),
        ("KILL: rho24 0.90", s24(9.0e-3, er=1.2e-2), base12, ref, "KILL"),
        ("KILL: pcg above r1 (with reference)", s24(5e-3, er=4.5e-3), base12, ref, "KILL"),
        ("KILL: pcg above r1, reference-free", s24(5e-3, er=4.5e-3), base12, None, "KILL"),
        ("KILL: futility stop fired", s24(6e-3, early_kill=750), base12, ref, "KILL"),
        ("AMBIGUOUS: rho24 0.70 (between thresholds)", s24(7.0e-3, er=1.1e-2), base12, ref, "AMBIGUOUS"),
        ("AMBIGUOUS: rho24 0.55 but 2 sigma crosses 0.60", s24(5.5e-3, er=1.1e-2, dE=0.01), base12, ref, "AMBIGUOUS"),
        ("AMBIGUOUS: rise 0.25 (rho12 0.20, rho24 0.45)", s24(4.5e-3),
         dict(base12, pcg=[run(e12(0.56e-3), 0.0, S2=0.0)]), ref, "AMBIGUOUS"),
        ("AMBIGUOUS: pcg below r1 by < 2 sigma", s24(4.0e-3, er=4.3e-3), base12, ref, "AMBIGUOUS"),
        ("AMBIGUOUS-FINAL: two seeds, still rho24 0.70",
         {"pcg": [run(e24(7e-3)), run(e24(7.2e-3), seed=1)], "che": [run(e24(1e-2)), run(e24(1.01e-2), seed=1)],
          "r1": [run(e24(1.1e-2)), run(e24(1.12e-2), seed=1)]}, base12, ref, "AMBIGUOUS-FINAL"),
        ("PASS: best of seeds (seed 1 resolves)",
         {"pcg": [run(e24(6e-3)), run(e24(4e-3), seed=1)], "che": [run(e24(1e-2)), run(e24(1.02e-2), seed=1)],
          "r1": [run(e24(8e-3)), run(e24(8.1e-3), seed=1)]}, base12, ref, "PASS"),
        ("INCOMPLETE: r1 missing at 24", {"pcg": [run(e24(4e-3))], "che": [run(e24(1e-2))]}, base12, ref, "INCOMPLETE"),
        ("INCOMPLETE: pcg capped at 24", s24(4e-3, capped=True), base12, ref, "INCOMPLETE"),
        ("INCOMPLETE: 12-site baseline missing", s24(4e-3), {"che": base12["che"]}, ref, "INCOMPLETE"),
        ("INCOMPLETE-REFERENCE: pcg < r1, no reference", s24(4e-3), base12, None, "INCOMPLETE-REFERENCE"),
        ("INCOMPLETE-REFERENCE: reference not OK", s24(4e-3), base12, (None, None, "AMBIGUOUS-REFERENCE"),
         "INCOMPLETE-REFERENCE"),
        ("NO-TARGET: che at the reference", s24(-1e-4, ec=1e-4, er=5e-4), base12, ref, "NO-TARGET"),
        ("INVALID: 12-site pcg S^2 = 1e-6", s24(4e-3), dict(base12, pcg=[run(e12(1.2e-3), 0.0, S2=1e-6)]), ref, "INVALID"),
        ("KILL beats missing 12-site: rho24 0.95", s24(9.5e-3, er=1.2e-2), {}, ref, "KILL"),
    ]
    ok = 0
    for name, a24, a12, rf, want in sc:
        v, nt = rule_t2b(a24, a12, rf)
        good = v == want
        ok += good
        print(f"[{'OK ' if good else 'BAD'}] {name:52s} -> {v:22s} (intended {want}) | {'; '.join(nt)}")
    print(f"CHECK {'PASS' if ok == len(sc) else 'FAIL'} {ok}/{len(sc)}")
    return ok == len(sc)


# ============================================================================ readers
def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def by_arm(rs):
    out = {}
    for r in rs:
        out.setdefault(r["arm"], []).append(r)
    return out


def ckpt_lines(rs, Er=None):
    for r in rs:
        if r.get("ckpt"):
            items = sorted(r["ckpt"].items(), key=lambda x: int(x[0]))
            txt = ", ".join(f"{k}: {c['E']:.5f}+-{c['dE']:.5f}"
                            + (f" (eps {(c['E'] - Er) / abs(Er):.2e})" if Er is not None else "")
                            + (f" [futility {c['futility']}]" if "futility" in c else "") for k, c in items)
            print(f"  {r['arm']} L={r['L']} seed {r['seed']} checkpoints: {txt}")


def report(a):
    rs = load(os.path.join(a.resdir, "t2b", "S*x3_*.json"))
    r24 = by_arm([r for r in rs if r["L"] == 24])
    r12 = by_arm([r for r in rs if r["L"] == 12])
    sys.path.insert(0, SCAL)
    ref24 = None
    try:
        import scaling_report as SR
        ref24 = SR.load_refs(a.dmrgdir).get(24)
    except Exception as e:  # noqa: BLE001
        print(f"  (reference loader failed: {e})")
    v, nt = rule_t2b(r24, r12, ref24)
    print(f"VERDICT T2b (R2 pcg-scal, 24 vs 12 sites): {v}")
    for n_ in nt:
        print(f"  {n_}")
    ckpt_lines([r for rr in r24.values() for r in rr], ref24[0] if ref24 and ref24[2] == "OK" else None)
    ckpt_lines([r for rr in r12.values() for r in rr], E0_12)
    ctx = load(os.path.join(SCAL, "results", "S8x3_*_pg11_*.json"))
    for r in ctx:
        if r.get("arm") in ("che", "che_k8") and not r.get("capped"):
            print(f"  context (686990, not gating): {r['arm']} seed {r.get('seed')} at 24 sites E {r['E']:.5f} +- {r['dE']:.5f}")


def preflight(a):
    s24 = by_arm(load(os.path.join(a.resdir, "s24", "S4x2_*.json")))
    s12 = by_arm(load(os.path.join(a.resdir, "s12", "S4x2_*.json")))
    E0 = s12["che"][0]["E0"]
    print(f"toy 4x2 (+,+): E0 {E0:.6f}; " + "; ".join(
        f"{slot} {arm} E {r['E']:.5f}+-{r['dE']:.5f} ckpt {list(r['ckpt'])}"
        for slot, d in (("s24", s24), ("s12", s12)) for arm, rr in d.items() for r in rr))
    ref = (E0, 0.0, "OK")
    # the rule hard-codes E0_12 for the 12-site slot: map toy energies onto the same eps scale
    def to12(runs):
        return [dict(r, E=E0_12 + (r["E"] - E0) / abs(E0) * abs(E0_12)) for r in runs]
    t12 = {k: to12(v) for k, v in s12.items()}
    out = []
    v, nt = rule_t2b(s24, t12, ref)
    out.append(("real toy files", v, None, nt))
    ex24 = dict(s24["pcg"][0], E=E0, dE=1e-6, early_kill=None, capped=False)
    ex12 = dict(t12["pcg"][0], E=E0_12, S2=0.0)
    v, nt = rule_t2b(dict(s24, pcg=[ex24]), dict(t12, pcg=[ex12]), ref)
    out.append(("positive control (pcg slots = exact state)", v, "PASS", nt))
    v, nt = rule_t2b(dict(s24, pcg=[dict(s24["che"][0], arm="pcg")]), t12, ref)
    out.append(("negative control (pcg slot = che itself, rho 1)", v, "KILL", nt))
    v, nt = rule_t2b(dict(s24, pcg=[dict(s24["r1"][0], arm="pcg")]), t12, ref)
    out.append(("negative control (pcg slot = r1 itself, d = 0)", v, "KILL", nt))
    fc = load(os.path.join(a.resdir, "futility_ctrl", "S4x2_*.json"))
    v, nt = rule_t2b(dict(s24, pcg=fc), t12, ref)
    out.append(("futility control (real early-killed pcg file)", v, "KILL", nt))
    dc = load(os.path.join(a.resdir, "disarmed_ctrl", "S4x2_*.json"))
    armed = [c.get("futility") for r in dc for c in r["ckpt"].values() if "futility" in c]
    good_dis = bool(dc) and not dc[0]["early_kill"] and all(s.startswith("disarmed") for s in armed) and armed
    out.append(("disarmed control (no reference: must run to the end)", "RAN" if good_dis else "STOPPED/BAD", "RAN",
                [f"futility states {armed}"]))
    v, nt = rule_t2b(dict(s24, pcg=[ex24]), dict(t12, pcg=[ex12]), None)
    out.append(("missing reference (pcg exact, below r1)", v, "INCOMPLETE-REFERENCE", nt))
    ok = True
    for name, v, want, nt in out:
        good = want is None or v == want
        ok &= good
        print(f"[{'OK ' if good else 'BAD'}] {name:52s} -> {v}" + (f" (intended {want})" if want else "")
              + f" | {'; '.join(nt)}")
    print(f"PREFLIGHT {'PASS' if ok else 'FAIL'}")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["check", "report", "preflight"])
    p.add_argument("--resdir", default=os.path.join(HERE, "results"))
    p.add_argument("--dmrgdir", default=os.path.join(SCAL, "dmrg"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.mode == "check":
        sys.exit(0 if check() else 1)
    if a.mode == "preflight":
        sys.exit(0 if preflight(a) else 1)
    report(a)


if __name__ == "__main__":
    main()
