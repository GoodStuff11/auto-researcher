#!/usr/bin/env python3
"""
Pre-registered size-scaling verdict for P4.1 (`che_kK` vs `che` vs plain NNBF on open Lx x 3 strips).

Usage (login node, any python3 with numpy):
    python3 scaling_report.py [--resdir PATH] [--dmrgdir PATH] [--check]

Options
-------
  --resdir PATH   VMC result JSONs written by `vmc.py --mode train` (default <script dir>/results)
  --dmrgdir PATH  DMRG JSONs written by dmrg_ref.jl (default <script dir>/dmrg)
  --check         run the synthetic logic check (no data needed) and exit; prints every scenario
                  and its verdict, and fails loudly if any verdict differs from the intended one.

Definitions (written into the leaf doc before submission)
---------------------------------------------------------
  Ladder Lx in {4, 8, 12, 16}, Ly = 3: N = 12, 24, 36, 48 sites. Reference E_ref(N): exact ED at
  N = 12 (E_0 = -7.827899034271753); DMRG truncation-extrapolated energy (2S^z = 0) otherwise.
  For arm X at size N: E_X = lowest final energy over seeds (variational best-of-seeds), with its
  error bar; eps_X = (E_X - E_ref)/|E_ref|.
  R(K, N) = eps_{che_kK} / eps_{che}   (rank gain ratio; R < 1 is a gain).
  nnbf at equal compute ("nnbf_wall"): every nnbf run is trained for n_eq(N) = round(3000 r_N) SR
  steps, r_N = t_step(che_k8)/t_step(nnbf) measured in one process on one node (vmc.py --mode timing,
  logs/timing_output.txt); E_nnbf_wall = best final energy over seeds. Actual wall times are printed
  next to it. "nnbf_iter" = nnbf at step 3000 (equal iterations; reported, not gating).
  A comparison "a < b" is RESOLVED-TRUE if a + 2 sigma < b, RESOLVED-FALSE if a - 2 sigma > b, else
  UNRESOLVED (sigma combines the VMC error bars and the reference error).
  Reference (amendment 2026-11-16, second): E_ref(N) is the (+,+)-SECTOR energy E(+,+): the My = +1
  penalty DMRG run (dmrg_ref.jl --sector=1) with <Inv>, <My> >= 0.95; or the plain run only if its
  characters are clean (>= 0.95). Otherwise the size is AMBIGUOUS-REFERENCE (not scored). The sector
  gap E(My=-1) - E(+,+) is printed; if negative beyond 2 sigma the size is marked "ground state in the
  My=-1 sector" (scoring stays within (+,+)).
  Reference validity at N: sigma_ref(N) < 0.1 * min(|E_che - E_k8|, |E_k8 - E_nnbf_wall|).
  Premises: every arm has its seeds, no run capped, every size has a valid reference. Otherwise
  INCOMPLETE (missing runs) or AMBIGUOUS-REFERENCE.
  Premise gate P_R (amendment 2026-11-16, made before any production output existed):
  R(8, 12) <= 0.75 RESOLVED at 2 sigma under this protocol. If P_R fails the verdict is
  NO-BASELINE-GAIN (the protocol cannot see the 508440 rank gain at 12 sites; no scaling conclusion;
  decider: restore the projection / optimiser that produced it) and D1-D3, S1-S3 are not evaluated.

Verdict (Nmax = 48)
-------------------
  DOES-NOT-SCALE if (D1) R(8, Nmax) >= 0.9 RESOLVED, or (D2) R(8, N) rises monotonically over the
    ladder AND a linear fit R(8, N) = a + b N reaches 1 before N = 100, or (D3) E_nnbf_wall < E_k8 at
    Nmax RESOLVED.
  SCALES if (S1) R(8, Nmax) <= 0.75 RESOLVED, AND (S2) E_k8 < E_nnbf_wall at Nmax RESOLVED, AND
    (S3) R(8, Nmax) - R(8, 12) <= 0.15.
  AMBIGUOUS otherwise. Decider: if only resolution is missing, re-evaluate the saved parameters
    with 4x the evaluation samples; otherwise add Lx = 24 (72 sites) for che, che_k8 and nnbf, 2 seeds,
    and apply the same rule with Nmax = 72.
  Reported, not gating: R(K, N) for K = 2, 4, 8; gap fraction g(K, N) = (eps_che - eps_K)/(eps_che -
    eps_nnbf_wall) when eps_nnbf_wall < eps_che; per-site error (E_X - E_ref)/N; the smallest K with
    eps_K(N) <= eps_k8(12); wall time per SR step and per sample; seed means.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = [12, 24, 36, 48]
ARMS = ["che", "che_k2", "che_k4", "che_k8", "nnbf"]
SEEDS = {12: 3, 24: 2, 36: 1, 48: 2}   # 36 sites: 1 seed (amendment 2026-11-16, P_G budget)
N_IT = 3000          # SR steps of every che arm (equal iterations)
E0_12 = -7.827899034271753
S1_MAX, D1_MIN, S3_RISE, D2_HORIZON = 0.75, 0.9, 0.15, 100
REF_NOTES = {}
PR_MAX = 0.75        # premise P_R: R(8,12) <= 0.75 resolved at 2 sigma


def cmp_lt(a, b, sig):
    """a < b? -> 'T' resolved true, 'F' resolved false, 'U' unresolved (2 sigma)."""
    if a + 2 * sig < b:
        return "T"
    if a - 2 * sig > b:
        return "F"
    return "U"


def ratio(eK, ec, sK, sc):
    R = eK / ec
    return R, abs(R) * np.sqrt((sK / eK) ** 2 + (sc / ec) ** 2)


def size_stats(runs, ref, sref, N):
    """runs: list of dicts (arm, seed, E, dE, wall, capped, extras). Returns per-arm summary."""
    out = {}
    for arm in ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        b = min(rs, key=lambda r: r["E"])
        out[arm] = dict(n=len(rs), E=b["E"], dE=b["dE"], Emean=float(np.mean([r["E"] for r in rs])),
                        eps=(b["E"] - ref) / abs(ref), seps=np.hypot(b["dE"], sref) / abs(ref),
                        wall=float(np.median([r["wall"] for r in rs])), capped=any(r.get("capped") for r in rs),
                        persite=(b["E"] - ref) / N)
    nn = [r for r in runs if r["arm"] == "nnbf"]
    if nn:
        # equal compute: the nnbf final point IS the pre-registered step n_eq(N) = round(N_IT * r_N)
        b = min(nn, key=lambda r: r["E"])
        out["nnbf_wall"] = dict(E=b["E"], dE=b["dE"], wall=float(np.median([r["wall"] for r in nn])),
                                eps=(b["E"] - ref) / abs(ref), seps=np.hypot(b["dE"], sref) / abs(ref),
                                persite=(b["E"] - ref) / N, iters=b["iters_done"])
        eq = [r["extras"][str(N_IT)] for r in nn if str(N_IT) in r.get("extras", {})]
        if eq:
            e = min(eq, key=lambda v: v["E"])
            out["nnbf_iter"] = dict(E=e["E"], dE=e["dE"], wall=e["wall"], eps=(e["E"] - ref) / abs(ref),
                                    seps=np.hypot(e["dE"], sref) / abs(ref), persite=(e["E"] - ref) / N)
    return out


def verdict(table, refs):
    """table[N][arm] summaries; refs[N] = (E_ref, sigma_ref). Returns (verdict, lines)."""
    lines = []
    for N in LADDER:
        if N not in table:
            return "INCOMPLETE", [f"size N={N} missing"]
        for arm in ARMS:
            if arm not in table[N] or table[N][arm]["n"] < SEEDS[N]:
                return "INCOMPLETE", [f"N={N} {arm}: {table[N].get(arm, {}).get('n', 0)} of {SEEDS[N]} seeds"]
            if table[N][arm]["capped"]:
                return "INCOMPLETE", [f"N={N} {arm}: capped run"]
        if len(refs[N]) > 2 and refs[N][2] != "OK":
            return "AMBIGUOUS-REFERENCE", [f"N={N}: no valid (+,+) reference ({refs[N][2]})"]
        if "nnbf_wall" not in table[N]:
            return "INCOMPLETE", [f"N={N}: no nnbf run"]
        t = table[N]
        dmin = min(abs(t["che"]["E"] - t["che_k8"]["E"]), abs(t["che_k8"]["E"] - t["nnbf_wall"]["E"]))
        if refs[N][1] >= 0.1 * dmin:
            return "AMBIGUOUS-REFERENCE", [f"N={N}: sigma_ref {refs[N][1]:.2e} >= 0.1 x {dmin:.2e}"]
    R8, sR8 = {}, {}
    for N in LADDER:
        t = table[N]
        R8[N], sR8[N] = ratio(t["che_k8"]["eps"], t["che"]["eps"], t["che_k8"]["seps"], t["che"]["seps"])
    lines.append("R(8,N): " + ", ".join(f"N={N}: {R8[N]:.3f}+-{sR8[N]:.3f}" for N in LADDER))
    # premise gate P_R (amendment 2026-11-16, before any production output): the protocol must see the rank gain at 12 sites
    N0 = LADDER[0]
    if not (R8[N0] + 2 * sR8[N0] <= PR_MAX):
        lines.append(f"premise P_R fails: R(8,{N0}) = {R8[N0]:.3f} +- {sR8[N0]:.3f} is not <= {PR_MAX} at 2 sigma; "
                     "D1-D3 and S1-S3 not evaluated")
        lines.append("decider: restore the projection / optimiser that produced the 508440 rank gain, then rerun")
        return "NO-BASELINE-GAIN", lines
    Nm = LADDER[-1]
    t = table[Nm]
    sig_w = np.hypot(t["che_k8"]["dE"], t["nnbf_wall"]["dE"])
    k8_beats = cmp_lt(t["che_k8"]["E"], t["nnbf_wall"]["E"], sig_w)
    nnbf_beats = cmp_lt(t["nnbf_wall"]["E"], t["che_k8"]["E"], sig_w)
    d1 = cmp_lt(D1_MIN, R8[Nm], sR8[Nm] / 1.0) == "T" or (R8[Nm] - 2 * sR8[Nm] >= D1_MIN)
    rises = all(R8[LADDER[i + 1]] > R8[LADDER[i]] for i in range(len(LADDER) - 1))
    b, a = np.polyfit(np.array(LADDER, float), np.array([R8[N] for N in LADDER]), 1)
    Nstar = (1 - a) / b if b > 0 else np.inf
    d2 = rises and Nstar < D2_HORIZON
    d3 = nnbf_beats == "T"
    s1 = R8[Nm] + 2 * sR8[Nm] <= S1_MAX
    s2 = k8_beats == "T"
    s3 = R8[Nm] - R8[LADDER[0]] <= S3_RISE
    lines.append(f"linear fit R(8,N) = {a:.3f} + {b:.2e} N -> reaches 1 at N* = {Nstar:.0f}; monotone rise: {rises}")
    lines.append(f"N={Nm}: E_k8 {t['che_k8']['E']:.5f}+-{t['che_k8']['dE']:.5f} vs E_nnbf_wall {t['nnbf_wall']['E']:.5f}"
                 f"+-{t['nnbf_wall']['dE']:.5f} (walls: nnbf {t['nnbf_wall']['wall'] / 3600:.1f} h, k8 {t['che_k8']['wall'] / 3600:.1f} h):"
                 f" k8<nnbf {k8_beats}, nnbf<k8 {nnbf_beats}")
    lines.append(f"clauses: D1 {d1} D2 {d2} D3 {d3} | S1 {s1} S2 {s2} S3 {s3}")
    if d1 or d2 or d3:
        return "DOES-NOT-SCALE", lines
    if s1 and s2 and s3:
        return "SCALES", lines
    unresolved = (R8[Nm] + 2 * sR8[Nm] > S1_MAX >= R8[Nm]) or (k8_beats == "U")
    lines.append("decider: " + ("re-evaluate saved parameters with 4x n_eval (resolution only)" if unresolved and
                                R8[Nm] <= S1_MAX and s3 else "add Lx = 24 (72 sites) for che, che_k8, nnbf, 2 seeds"))
    return "AMBIGUOUS", lines


def select_ref(N, plain, myp, mym):
    """Reference for the (+,+)-projected arms at size N (amendment 2026-11-16, second).
    plain / myp / mym: DMRG result dicts (2S^z = 0) of the plain run and the My = +1 / My = -1 penalty runs, or None.
    Returns (E_ref, sigma_ref, status, notes); status "OK" or "AMBIGUOUS-REFERENCE"."""
    notes = []
    if myp is not None and myp.get("chi_inv", 0) >= 0.95 and myp.get("chi_My", 0) >= 0.95:
        E, sig = myp["E_extrap"], myp["E_extrap_err"]
        notes.append(f"E(+,+) = {E:.6f} +- {sig:.1e} from the My=+1 sector run")
    elif myp is not None:
        return None, None, "AMBIGUOUS-REFERENCE", [f"My=+1 sector run is not (+,+): <Inv> {myp.get('chi_inv')}, <My> {myp.get('chi_My')}"]
    elif plain is not None and plain.get("chi_inv", 0) >= 0.95 and plain.get("chi_My", 0) >= 0.95:
        E, sig = plain["E_extrap"], plain["E_extrap_err"]
        notes.append(f"E(+,+) = {E:.6f} +- {sig:.1e} from the plain run (clean characters)")
    else:
        return None, None, "AMBIGUOUS-REFERENCE", ["no sector-resolved E(+,+) and the plain run has |<My>| < 0.95 or a wrong sign"]
    if mym is not None:
        gap = mym["E_extrap"] - E
        err = mym["E_extrap_err"] + sig
        notes.append(f"sector gap E(My=-1) - E(+,+) = {gap:+.5f} +- {err:.5f}"
                     + (" -> ground state in the My=-1 sector (scoring stays within (+,+))" if gap < -2 * err else ""))
    return E, sig, "OK", notes


def load_refs(dmrgdir):
    refs = {12: (E0_12, 0.0, "OK")}
    notes = {12: ["ED, (+,+) ground state (logs/ed_irrep_4x3_output.txt)"]}
    runs = {}
    for fn in glob.glob(os.path.join(dmrgdir, "dmrg_S*_Sz20*.json")):
        d = json.load(open(fn))
        if d["Ly"] != 3 or d["N"] not in LADDER or d["N"] == 12:
            continue
        key = {1: "myp", -1: "mym"}.get(int(d.get("sector_My", 0)), "plain")
        runs.setdefault(d["N"], {})[key] = d
    for N, r in runs.items():
        E, sig, st, nt = select_ref(N, r.get("plain"), r.get("myp"), r.get("mym"))
        refs[N] = (E, sig, st)
        notes[N] = nt
    REF_NOTES.update(notes)
    return refs


def spin_check(dmrgdir):
    """E(S^z=1) - min over the S^z=0 runs (plain, My=+1, My=-1) at each N; > 2 sigma means the
    lowest S^z=0 state is not part of an S >= 1 multiplet, i.e. a singlet ground state."""
    out = []
    z0, z2 = {}, {}
    for fn in glob.glob(os.path.join(dmrgdir, "dmrg_S*.json")):
        d = json.load(open(fn))
        tgt = z0 if d["Sz2"] == 0 else z2
        if d["Sz2"] in (0, 2) and (d["N"] not in tgt or d["E_extrap"] < tgt[d["N"]]["E_extrap"]):
            tgt[d["N"]] = d
    for N in sorted(set(z0) & set(z2)):
        gap = z2[N]["E_extrap"] - z0[N]["E_extrap"]
        err = z0[N]["E_extrap_err"] + z2[N]["E_extrap_err"]
        out.append(f"N={N}: E(Sz=1) - min E(Sz=0) = {gap:+.5f} +- {err:.5f} -> "
                   f"{'singlet ground state' if gap > 2 * err else 'NOT RESOLVED / not singlet'}")
    return out


def report(a):
    refs = load_refs(a.dmrgdir)
    runs = {}
    for fn in glob.glob(os.path.join(a.resdir, "S*x3_*.json")):
        r = json.load(open(fn))
        runs.setdefault(r["L"], []).append(r)
    print("References:")
    for N in LADDER:
        print(f"  N={N}: " + (f"{refs[N][2]}" if N in refs else "missing") + "; " + "; ".join(REF_NOTES.get(N, [])))
    for s in spin_check(a.dmrgdir):
        print("  " + s)
    table = {}
    for N in LADDER:
        if N not in runs or N not in refs or refs[N][2] != "OK":
            continue
        table[N] = size_stats(runs[N], refs[N][0], refs[N][1], N)
        t = table[N]
        print(f"\nN={N} (E_ref {refs[N][0]:.6f})")
        print("  arm        seeds  E_best            eps        per-site   E_mean      wall/h  R(K)")
        for arm in ARMS + ["nnbf_iter", "nnbf_wall"]:
            if arm not in t:
                continue
            x = t[arm]
            R = (f"{x['eps'] / t['che']['eps']:.3f}" if arm.startswith("che") and "che" in t else "")
            print(f"  {arm:10s} {x.get('n', '-')!s:5s}  {x['E']:.5f}+-{x['dE']:.5f}  {x['eps']:.3e}  {x['persite']:.2e}"
                  f"  {x.get('Emean', float('nan')):.5f}  {x['wall'] / 3600:6.1f}  {R}")
        if "nnbf_wall" in t and "che" in t and t["nnbf_wall"]["eps"] < t["che"]["eps"]:
            g = {arm: (t["che"]["eps"] - t[arm]["eps"]) / (t["che"]["eps"] - t["nnbf_wall"]["eps"])
                 for arm in ["che_k2", "che_k4", "che_k8"] if arm in t}
            print("  gap fraction vs nnbf_wall: " + ", ".join(f"{k} {v:+.2f}" for k, v in g.items()))
    if 12 in table and "che_k8" in table[12]:
        target = table[12]["che_k8"]["eps"]
        for N in sorted(table):
            ks = [k for k, arm in [(1, "che"), (2, "che_k2"), (4, "che_k4"), (8, "che_k8")]
                  if arm in table[N] and table[N][arm]["eps"] <= target]
            print(f"  K needed for eps <= eps_k8(12) = {target:.2e} at N={N}: {min(ks) if ks else '>8'}")
    v, lines = verdict(table, refs)
    print("\n" + "\n".join(lines))
    print(f"\nVERDICT: {v}")


def check():
    """synthetic scenarios -> intended verdicts."""
    def mk(R8s, ek8_48, enn_48, sig=1e-4, eche=1e-2, capped=False, seeds=None, sref=0.0):
        table, refs = {}, {}
        for N, R8 in zip(LADDER, R8s):
            Eref = -0.8 * N
            refs[N] = (Eref, sref)
            ec = eche
            t = {}
            for arm, e in [("che", ec), ("che_k2", ec * (1 + R8) / 2), ("che_k4", ec * (1 + 3 * R8) / 4), ("che_k8", ec * R8)]:
                E = Eref + e * abs(Eref)
                t[arm] = dict(n=(seeds or SEEDS)[N], E=E, dE=sig, eps=e, seps=sig / abs(Eref), wall=3600.0,
                              capped=capped, persite=(E - Eref) / N, Emean=E)
            enn = ec * 0.9 if N != 48 else None
            Enn = Eref + (enn * abs(Eref) if enn is not None else 0)
            if N == 48:
                Enn = t["che_k8"]["E"] + (enn_48 - ek8_48)
            t["nnbf"] = dict(n=(seeds or SEEDS)[N], E=Enn, dE=sig, eps=(Enn - Eref) / abs(Eref), seps=sig, wall=3600.0,
                             capped=False, persite=0, Emean=Enn)
            t["nnbf_wall"] = dict(E=Enn, dE=sig, wall=3500.0, eps=(Enn - Eref) / abs(Eref), seps=sig, persite=0)
            table[N] = t
        return table, refs
    cases = [
        ("flat strong gain, k8 beats nnbf", mk([0.40, 0.42, 0.45, 0.47], 0.0, 0.05), "SCALES"),
        ("R rises to 0.95", mk([0.40, 0.65, 0.85, 0.95], 0.0, 0.05), "DOES-NOT-SCALE"),
        ("monotone rise, fit hits 1 before 100", mk([0.40, 0.55, 0.70, 0.85], 0.0, 0.05), "DOES-NOT-SCALE"),
        ("flat gain but nnbf wins at 48", mk([0.40, 0.42, 0.45, 0.47], 0.05, 0.0), "DOES-NOT-SCALE"),
        ("R(48)=0.8 non-monotone, k8 wins", mk([0.70, 0.85, 0.75, 0.80], 0.0, 0.05), "AMBIGUOUS"),
        ("flat gain, k8 vs nnbf unresolved", mk([0.40, 0.42, 0.45, 0.47], 0.0, 1e-4), "AMBIGUOUS"),
        ("rise of 0.2 but R(48)=0.7", mk([0.50, 0.55, 0.72, 0.70], 0.0, 0.05), "AMBIGUOUS"),
        ("no gain even at 12 sites (R(8,12)=0.95)", mk([0.95, 0.96, 0.97, 0.98], 0.0, 0.05), "NO-BASELINE-GAIN"),
        ("weak 12-site gain (R(8,12)=0.8), rising", mk([0.80, 0.85, 0.92, 0.97], 0.0, 0.05), "NO-BASELINE-GAIN"),
        ("12-site gain unresolved at 2 sigma", mk([0.74, 0.74, 0.74, 0.74], 0.0, 0.05, sig=0.3), "NO-BASELINE-GAIN"),
        ("missing seed", mk([0.40, 0.42, 0.45, 0.47], 0.0, 0.05, seeds={12: 3, 24: 1, 36: 1, 48: 2}), "INCOMPLETE"),
        ("capped run", mk([0.40, 0.42, 0.45, 0.47], 0.0, 0.05, capped=True), "INCOMPLETE"),
        ("reference too coarse", mk([0.40, 0.42, 0.45, 0.47], 0.0, 0.05, sref=0.5), "AMBIGUOUS-REFERENCE"),
    ]
    bad = 0
    cl = dict(E_extrap=-30.0, E_extrap_err=1e-4, chi_inv=1.0, chi_My=1.0)
    mixed = dict(cl, chi_My=0.63)
    lowm = dict(cl, E_extrap=-30.01, chi_My=-1.0, sector_My=-1)
    highm = dict(cl, E_extrap=-29.95, chi_My=-1.0, sector_My=-1)
    ref_cases = [
        ("sector run (+,+) available", (36, mixed, dict(cl, sector_My=1), highm), "OK", None),
        ("mixed plain run, no sector run", (48, mixed, None, None), "AMBIGUOUS-REFERENCE", None),
        ("clean plain run, no sector run", (24, cl, None, None), "OK", None),
        ("My=+1 run landed in (-,+)", (48, mixed, dict(cl, chi_inv=-1.0, sector_My=1), None), "AMBIGUOUS-REFERENCE", None),
        ("ground state in My=-1 sector", (48, mixed, dict(cl, sector_My=1), lowm), "OK", "ground state in the My=-1"),
    ]
    for name, args, want, flag in ref_cases:
        E, sig, st, nt = select_ref(*args)
        ok = st == want and (flag is None or any(flag in x for x in nt))
        bad += not ok
        print(f"[{'ok' if ok else 'WRONG'}] reference: {name}: {st} (intended {want}) " + "; ".join(nt))
    for name, (table, refs), want in cases:
        v, lines = verdict(table, refs)
        ok = v == want
        bad += not ok
        print(f"[{'ok' if ok else 'WRONG'}] {name}: {v} (intended {want})")
        for l_ in lines:
            print("      " + l_)
    n = len(cases) + len(ref_cases)
    print(f"CHECK {'PASS' if bad == 0 else 'FAIL'}: {n - bad}/{n} scenarios give the intended verdict")
    if bad:
        sys.exit(1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P4.1 size-scaling verdict (see module docstring)")
    p.add_argument("--resdir", default=os.path.join(HERE, "results"))
    p.add_argument("--dmrgdir", default=os.path.join(HERE, "dmrg"))
    p.add_argument("--check", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.check:
        check()
    else:
        report(a)


if __name__ == "__main__":
    main()
