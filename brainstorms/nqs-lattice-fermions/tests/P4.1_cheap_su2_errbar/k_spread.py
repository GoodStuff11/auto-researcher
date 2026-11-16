"""Run-to-run spread of finished 24-site (8x3, (+,+), P_G) `che` energies vs quoted per-run sigma.
Reads result JSONs and job logs read-only. Prints the run table, the spread estimates, k, and the
T2b status check (no 24-site che/pcg JSON in results/t2b, no results/t2b_wall, no aggregation output)."""
import glob, json, os, re, statistics as st
from scipy.stats import chi2
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = []
def add(name, f, proto):
    d = json.load(open(f)); assert d["L"] == 24 and d["arm"] == "che" and d["pg"] == "1,1" and not d["capped"]
    runs.append((name, d["E"], d["dE"], proto, d)); return d
add("W-24 H16 (698224 t0)", f"{T}/P4.1_cheap_su2_t2/results/w24/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0.json", "cold")
add("W-24 H48 (698224 t1/700568)", f"{T}/P4.1_cheap_su2_t2/results/w24/S8x3_N20_U8_tp-0.25_pg11_che_H48_seed0.json", "cold")
add("R15b A final (705667)", f"{T}/P4.1_cheap_su2_opt/results_t2/A/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0.json", "warm")
add("R15b B (704030 t1)", f"{T}/P4.1_cheap_su2_opt/results_t2/B/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0.json", "warm")
# interim A: capped at step 288; its JSON was overwritten by the resume, so read the job log
m = re.search(r"final E \(average of last (\d+) steps\) (-?[\d.]+) \+- ([\d.]+)",
              open(f"{T}/P4.1_cheap_su2_opt/logs/P41nsrich_704030_0.out").read())
runs.append(("R15b A interim (704030 t0, capped 288)", float(m[2]), float(m[3]), "warm", None))
print(f"{'run':40s} {'E':>11s} {'dE':>8s} proto steps chains lr H n_eval")
for n, E, dE, p, d in runs:
    extra = "288/300 1024 0.02 16 65536" if d is None else f"{d['iters_done']}/{d['iters']} {d['hyper']['n_chains']} {d['hyper']['lr']} {d['H']} {d['n_eval']}"
    print(f"{n:40s} {E:11.5f} {dE:8.5f} {p:5s} {extra}")
def est(label, sel):
    E = [r[1] for r in sel]; s = [r[2] for r in sel]; sd = st.stdev(E); med = st.median(s); nu = len(E) - 1
    lo, hi = sd * (nu / chi2.ppf(0.975, nu)) ** .5, sd * (nu / chi2.ppf(0.025, nu)) ** .5
    print(f"{label}: n={len(E)} SD={sd:.5f} median sigma={med:.5f} k={sd/med:.2f}  (95% CI on k: {lo/med:.2f}-{hi/med:.2f})")
    return sd / med
warm = [r for r in runs if r[3] == "warm"]; cold = [r for r in runs if r[3] == "cold"]
kw = est("warm-start family (R15b A interim, A final, B)", warm)
kc = est("cold 1000-step family (W-24 H16, H48)", cold)
a_i, a_f = runs[4], runs[2]
dd = a_f[1] - a_i[1]; sq = (a_f[2]**2 + a_i[2]**2) ** .5
print(f"A interim vs final: diff={dd:.5f} = {abs(dd)/sq:.2f} x combined sigma; implied per-run k={abs(dd)/sq:.2f}")
kp = abs(dd) / sq
import math
k_round = math.ceil(max(kw, kc, kp) * 2) / 2
print(f"k estimates: warm {kw:.2f}, cold {kc:.2f}, A pair {kp:.2f}; max rounded up to 0.5 -> k_eff = {k_round}")
# T2b status
t2b = f"{T}/P4.1_cheap_su2_t2b"
print("results/t2b 24-site JSON:", sorted(os.path.basename(f) for f in glob.glob(f"{t2b}/results/t2b/S8x3*.json")))
print("results/t2b_wall exists:", os.path.exists(f"{t2b}/results/t2b_wall"))
rep = [f for f in glob.glob(f"{t2b}/logs/*") if re.search(r"report", f) and "wall_report_now" not in f]
print("aggregation report outputs in logs/ (excluding the pre-run wall_report_now):", rep)
vd = [f for f in glob.glob(f"{t2b}/**/*", recursive=True) if os.path.isfile(f) and f.endswith((".txt", ".out"))
      and "VERDICT" in open(f, errors="ignore").read() and "preflight" not in f and "check" not in f and "pipeline" not in f]
print("files with a VERDICT line outside preflight/check/pipeline:", vd)
