#!/usr/bin/env python3
"""
P3.2 round 5: LCC (ladder-crossing certificate) decision rule, evaluated on finished runs
========================================================================================

Rule (Route 5.3 of .subproblems/P3.2_biased_ensemble.md; pre-registered, nothing here reads the ED truth
except the scoring and the oracle-eps diagnostic):
  Data per width W and seed s: at every ladder point d (the targets of the run) two projected-NNBF
  branches, A (Neel-seeded) and B (stripe-seeded), of a fixed arm (warm = common snapshot start, the
  primary arm; cold = secondary arm), each with energy E_X(d), local-energy variance var_X(d) and order
  character c_X(d).
  1. Branch identity. A ladder point is used only if sign c_A < 0 and sign c_B > 0 (no crossover).
  2. Crossing. Per seed fit Delta(d) = E_B(d) - E_A(d) = alpha_s + beta_s d by least squares over the
     usable ladder points, excluding the decided target itself (leave-one-out, LOO; --no-loo keeps it);
     dc_s = -alpha_s / beta_s. Pooled: dc = mean_s dc_s, SE = std_s(dc_s) / sqrt(S) (seeds are the
     independent replicas; the warm ladder shares its start across d, so within-seed residuals understate
     realisation noise), beta = mean_s beta_s, beta_lo = beta - t * std_s(beta_s) / sqrt(S).
  3. Bias half-width. b = eps_hat / beta_lo, where eps_hat = max over X in {A, B} and over the 2 usable
     ladder points nearest dc of [ median_s E_X(d) - E0_hat(d) - g_X(d) ], with the diabatic excess
     g_B(d) = max(0, beta_lo (d - dc)), g_A(d) = max(0, -beta_lo (d - dc)) (the excited-order branch sits
     about beta |d - dc| above E0 by construction; beta_lo makes the subtraction conservative), and
     E0_hat(d) is the zero-variance
     extrapolation at d: the smaller of two intercepts (linear fit of E against var over all branch
     states of the width at d, cold and warm, all seeds; and the same fit over the states of the
     lower-mean-energy order only), minus 2 standard errors of that intercept. Rationale: the branch
     energy difference is beta (d - d_c) + Delta_b with |Delta_b| <= max_X eps_X near the crossing.
  4. Decide. With t = Student-t 97.5% quantile at S - 1 degrees of freedom:
        B if d < dc - (t SE + b);  A if d > dc + (t SE + b);  otherwise abstain (U).
     Also abstain if fewer than 4 usable ladder points remain, or beta_lo <= 0.
  Oracle-eps diagnostic: step 3 with the exact eps_X = E_X - E0 (upper bound on what step 3 can give).
  Rule-4 (sampled) emulation: every E_X(d) is replaced by E_X(d) + N(0, var_X(d) / n_vmc) (i.i.d. VMC of
  the final state with n_vmc draws), --n-rep replicas; reports CW, DEC and the flip rate against the
  exact-energy decision.

Scoring: CW = wrong commits / commits; DEC = commits / targets; inner targets |d| < 0.001.
PASS (pre-registered in the artifact): on the FRESH runs only (width 6 seeds 1234/2345/3456 and width 8
seeds 4567/5678/6789; widths 4 and 8 of job 451852 were seen while the rule was designed and are
in-sample) the pooled LCC (warm arm, LOO, estimated eps) has CW <= 0.05 over all certified decisions
of both fresh widths, DEC >= 0.5 on the inner targets at the widest fresh width (8), and the rule-4
emulation has CW <= 0.05 and flip rate <= 0.10. FAIL if any certified decision is wrong at a fresh width
(CW > 0.05 with at most 12 decisions means one wrong), or inner DEC at width 8 < 0.5. No AMBIGUOUS band.

Usage
  python3 lcc.py --runs DIR [DIR ...] [--fresh DIR [DIR ...]] [--arm warm|cold] [--no-loo]
                 [--n-vmc N] [--n-rep R] [--out FILE]
Options
  --runs DIR...   folders with run JSONs w{W}_s{S}.json written by ../P3.2_biased_ensemble_fix/micro_test.py
                  (any width; grouped by (folder kind, width)). Default: ../P3.2_biased_ensemble_fix/results.
  --fresh DIR...  folders whose runs count for the verdict (out-of-sample). Default: results_w6,
                  results_w8fresh in this folder (skipped if absent; then no verdict is printed).
  --arm {warm, cold}  branch arm used for the ladder fit: warm = R0P-warm (common snapshot start; primary),
                  cold = R0P (independent UHF starts). Default warm. Both are always printed; --arm only
                  selects which one enters the verdict.
  --no-loo        include the decided target in its own fit (default: leave-one-out).
  --n-vmc N       VMC draws per final state in the rule-4 emulation (default 1000000).
  --n-rep R       emulation replicas (default 50).
  --out FILE      JSON summary (default results_lcc/lcc_summary.json).
Comparison arms printed per width from the same JSONs: R0P, R0P-warm, R0P-warm-abs (round-4 control,
2-sigma SR-window abstention), SO-EC (EC), oracle EC (ECor).
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "P3.2_biased_ensemble_fix")
ARMKEY = {"warm": "R0P-warm", "cold": "R0P"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[os.path.join(FIX, "results")])
    ap.add_argument("--fresh", nargs="+", default=[os.path.join(HERE, "results_w6"),
                                                    os.path.join(HERE, "results_w8fresh")])
    ap.add_argument("--arm", choices=("warm", "cold"), default="warm")
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--n-vmc", type=int, default=1000000)
    ap.add_argument("--n-rep", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(HERE, "results_lcc", "lcc_summary.json"))
    return ap.parse_args(argv)


def load(dirs):
    groups = {}
    dirs = list(dict.fromkeys(os.path.abspath(x) for x in dirs))   # de-duplicate
    for dr in dirs:
        for f in sorted(glob.glob(os.path.join(dr, "w*_s*.json"))):
            r = json.load(open(f))
            if len(r.get("targets", [])) < len(r["args"]["targets"].split(",")):
                print(f"  skip incomplete run {f}")
                continue
            key = (os.path.basename(os.path.normpath(dr)), r["args"]["width"])
            groups.setdefault(key, []).append(r)
    for key, runs in groups.items():
        seeds = [r["args"]["seed"] for r in runs]
        assert len(set(seeds)) == len(seeds), f"{key}: duplicate seeds {seeds}"
        assert len({r["args"]["targets"] for r in runs}) == 1, f"{key}: target lists differ"
    return groups


def e0_hat(runs, i):
    """Zero-variance extrapolated E0 at target index i (conservative: min of two intercepts - 2 SE)."""
    E, V, L = [], [], []
    for r in runs:
        t = r["targets"][i]
        for arm in ("R0P", "R0P-warm"):
            for X in "AB":
                b = t["ctrl"][arm]["branches"][X]
                E.append(b["E"]); V.append(b["var"]); L.append(X)
    E, V, L = map(np.array, (E, V, L))
    cands = []
    lowX = "A" if E[L == "A"].mean() < E[L == "B"].mean() else "B"
    for m in (np.ones(len(E), bool), L == lowX):
        res = stats.linregress(V[m], E[m])
        cands.append(res.intercept - 2 * res.intercept_stderr)
    return min(cands)


def decide(runs, arm, energies, eps_mode, loo, e0h):
    """Pooled LCC decision for every target. energies[s][i][X] -> E. Returns list of (dec, info)."""
    S = len(runs)
    tq = stats.t.ppf(0.975, S - 1)
    ds = np.array([t["d"] for t in runs[0]["targets"]])
    out = []
    for k, dk in enumerate(ds):
        dcs, betas, ok = [], [], True
        for s, r in enumerate(runs):
            use = []
            for i, t in enumerate(r["targets"]):
                if loo and i == k:
                    continue
                br = t["ctrl"][arm]["branches"]
                if br["A"]["c"] < 0 < br["B"]["c"]:
                    use.append(i)
            if len(use) < 4:
                ok = False
                break
            x = ds[use]
            y = np.array([energies[s][i]["B"] - energies[s][i]["A"] for i in use])
            beta, alpha = np.polyfit(x, y, 1)
            dcs.append(-alpha / beta); betas.append(beta)
        if not ok:
            out.append(("U", dict(reason="ladder")))
            continue
        dcs, betas = np.array(dcs), np.array(betas)
        dc, se = dcs.mean(), dcs.std(ddof=1) / np.sqrt(S)
        beta_lo = betas.mean() - tq * betas.std(ddof=1) / np.sqrt(S)
        if beta_lo <= 0:
            out.append(("U", dict(reason="slope")))
            continue
        cand = [i for i in range(len(ds)) if not (loo and i == k)]
        near = sorted(cand, key=lambda i: abs(ds[i] - dc))[:2]
        eps = 0.0
        for i in near:
            for X in "AB":
                Em = np.median([energies[s][i][X] for s in range(S)])
                ref = runs[0]["targets"][i]["ed"]["E0"] if eps_mode == "oracle" else e0h[i]
                excess = max(0.0, (1 if X == "B" else -1) * beta_lo * (ds[i] - dc))
                eps = max(eps, Em - ref - excess)
        b = eps / beta_lo
        half = tq * se + b
        dec = "B" if dk < dc - half else ("A" if dk > dc + half else "U")
        out.append((dec, dict(dc=dc, se=se, beta_lo=beta_lo, eps=eps, b=b, half=half)))
    return out


def exact_energies(runs, arm):
    return [[{X: t["ctrl"][arm]["branches"][X]["E"] for X in "AB"} for t in r["targets"]] for r in runs]


def score(decs, truths, ds, inner_only=False):
    sel = [j for j in range(len(ds)) if (abs(ds[j]) < 1e-3 or not inner_only)]
    com = [j for j in sel if decs[j] != "U"]
    wrong = [j for j in com if decs[j] != truths[j]]
    return dict(n=len(sel), commit=len(com), wrong=len(wrong),
                CW=(len(wrong) / len(com) if com else 0.0), DEC=len(com) / len(sel))


def evaluate(key, runs, a, rng):
    ds = [t["d"] for t in runs[0]["targets"]]
    truths = [t["truth"] for t in runs[0]["targets"]]
    e0h = [e0_hat(runs, i) for i in range(len(ds))]
    print(f"=== {key[0]} width {key[1]}: seeds {[r['args']['seed'] for r in runs]}  truths {''.join(truths)}")
    print("  E0_hat - E0 per target: " + " ".join(f"{e0h[i] - runs[0]['targets'][i]['ed']['E0']:+.1e}"
                                                for i in range(len(ds))))
    res = {}
    for armname, arm in ARMKEY.items():
        for eps_mode in ("est", "oracle"):
            for loo in (True, False):
                decs = decide(runs, arm, exact_energies(runs, arm), eps_mode, loo, e0h)
                d = [x[0] for x in decs]
                tag = f"LCC-{armname}-{eps_mode}-{'loo' if loo else 'in'}"
                res[tag] = dict(dec="".join(d), all=score(d, truths, ds), inner=score(d, truths, ds, True))
                if eps_mode == "est" and loo:
                    res[tag]["detail"] = [x[1] for x in decs]
                print(f"  {tag:24s} {''.join(d)}  CW={res[tag]['all']['CW']:.3f} DEC={res[tag]['all']['DEC']:.2f}"
                      f"  inner DEC={res[tag]['inner']['DEC']:.2f} wrong={res[tag]['all']['wrong']}")
    for j, x in enumerate(res[f"LCC-{a.arm}-est-{'in' if a.no_loo else 'loo'}"].get("detail", [])):
        if "dc" in x:
            print(f"    d={ds[j]:+.4f} dc={x['dc']:+.2e} tSE={x['half'] - x['b']:.1e} eps_hat={x['eps']:.1e} "
                  f"b={x['b']:.1e} half={x['half']:.1e}")
    # rule-4 emulation (primary configuration)
    arm = ARMKEY[a.arm]
    base = res[f"LCC-{a.arm}-est-{'in' if a.no_loo else 'loo'}"]["dec"]
    flips, cw_c, dec_c, wr = 0, 0, 0, 0
    for _ in range(a.n_rep):
        en = [[{X: t["ctrl"][arm]["branches"][X]["E"]
                + rng.normal(0, np.sqrt(t["ctrl"][arm]["branches"][X]["var"] / a.n_vmc)) for X in "AB"}
               for t in r["targets"]] for r in runs]
        d = [x[0] for x in decide(runs, arm, en, "est", not a.no_loo, e0h)]
        flips += sum(1 for p, q in zip(d, base) if p != q)
        dec_c += sum(1 for p in d if p != "U")
        wr += sum(1 for p, tr in zip(d, truths) if p != "U" and p != tr)
    res["sampled"] = dict(flip=flips / (a.n_rep * len(ds)), CW=(wr / dec_c if dec_c else 0.0),
                          DEC=dec_c / (a.n_rep * len(ds)))
    print(f"  rule-4 emulation ({a.n_vmc} draws, {a.n_rep} reps): CW={res['sampled']['CW']:.3f} "
          f"DEC={res['sampled']['DEC']:.2f} flip={res['sampled']['flip']:.3f}")
    for nm in ("R0P", "R0P-warm", "R0P-warm-abs", "EC", "ECor"):
        per = ["".join(r["targets"][j]["dec"].get(nm, "?") for r in runs) for j in range(len(ds))]
        com = sum(1 for p, tr in zip(per, truths) for ch in p if ch != "U")
        wrong = sum(1 for p, tr in zip(per, truths) for ch in p if ch not in ("U", tr))
        res[nm] = dict(per_target=per, CW=(wrong / com if com else 0.0), DEC=com / (len(ds) * len(runs)))
        print(f"  {nm:13s} {' '.join(per)}  CW={res[nm]['CW']:.3f} DEC={res[nm]['DEC']:.2f} (per seed)")
    return res


def main(argv=None):
    a = parse_args(argv)
    rng = np.random.default_rng(20261114)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fresh_names = {os.path.basename(os.path.normpath(f)) for f in a.fresh}
    groups = load(list(a.runs) + [f for f in a.fresh if os.path.isdir(f)])
    summary, fresh = {}, {}
    for key in sorted(groups):
        runs = groups[key]
        if len(runs) < 2:
            print(f"  {key}: fewer than 2 seeds, skipped")
            continue
        summary[f"{key[0]}/w{key[1]}"] = r = evaluate(key, runs, a, rng)
        if key[0] in fresh_names:
            fresh[key[1]] = r
    tag = f"LCC-{a.arm}-est-{'in' if a.no_loo else 'loo'}"
    if fresh:
        com = sum(v[tag]["all"]["commit"] for v in fresh.values())
        wrong = sum(v[tag]["all"]["wrong"] for v in fresh.values())
        cw = wrong / com if com else 0.0
        Wmax = max(fresh)
        dec_in = fresh[Wmax][tag]["inner"]["DEC"]
        s_ok = all(v["sampled"]["CW"] <= 0.05 and v["sampled"]["flip"] <= 0.10 for v in fresh.values())
        verdict = "PASS" if (cw <= 0.05 and dec_in >= 0.5 and s_ok) else "FAIL"
        print(f"VERDICT LCC ({tag}, fresh widths {sorted(fresh)}): CW={cw:.3f} ({wrong}/{com}) "
              f"inner DEC at W={Wmax} {dec_in:.2f} sampled-ok {s_ok} -> {verdict}")
        summary["verdict"] = dict(tag=tag, CW=cw, DEC_inner=dec_in, sampled_ok=s_ok, verdict=verdict)
    else:
        print("no fresh runs present: in-sample evaluation only, no verdict")
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
