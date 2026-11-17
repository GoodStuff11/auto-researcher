"""P5.1 r01 T1a (RULE_A.md): plateau-type diagnostics of R38's guided SR runs (job 752885), read only.

Usage (from this directory, thread variables exported first):
    ~/.conda/envs/lit_env/bin/python floor_diag.py [--no-energy]
Options
  --no-energy  skip A2 (the replica-average exact energy, which builds the 12-site context, about 1 CPU-min).
Inputs (read only): ../P4.1_cheap_su2_r25/results_guided_s{6000,7000,8000}/{shift.json, theta_shift.npy}.
Output: results/floor_diag.json; prints the A1-A3 verdicts of RULE_A.md.
"""
import json
import os
import sys

for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k_] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
R25 = os.path.join(TESTS, "P4.1_cheap_su2_r25")
SEEDS = (6000, 7000, 8000)
W = slice(40, 60)


def main():
    out = dict(per_replica={})
    pool = dict(real=[], oracle=[], rule2=[], rises=[], sig=[], dEsel=[])
    for s in SEEDS:
        d = json.load(open(os.path.join(R25, f"results_guided_s{s}", "shift.json")))
        st = d["steps"]
        rr = dict(real=[], oracle=[], rule2=[])
        for x in st[W]:
            Rprev = x["R"] - x["dR"]
            g = [c["R_oracle"] - Rprev for c in x["cands"]]
            b = min(x["cands"], key=lambda c: c["dE_val"])
            rr["real"].append(x["dR"])
            rr["oracle"].append(max(max(g), 0.0))
            rr["rule2"].append(b["R_oracle"] - Rprev if b["dE_val"] < -2 * b["sig_val"] else 0.0)
            pool["sig"].append(b["sig_val"])
            pool["dEsel"].append(b["dE_val"])
            if x["dR"] < 0:
                pool["rises"].append(-x["dR"])
        for k in rr:
            pool[k] += rr[k]
        out["per_replica"][s] = dict({k: float(np.mean(v)) for k, v in rr.items()}, R60=st[-1]["R"])
    m = {k: float(np.mean(pool[k])) for k in ("real", "oracle", "rule2")}
    out["window_means"] = m
    out["n_rises"] = len(pool["rises"])
    out["mean_rise"] = float(np.mean(pool["rises"])) if pool["rises"] else 0.0
    out["median_sig_val"], out["median_dE_sel"] = float(np.median(pool["sig"])), float(np.median(pool["dEsel"]))
    out["A1"] = ("SELECTION-FLOOR" if m["oracle"] >= 0.005 and m["real"] <= 0.002 else
                 "DIRECTION-FLOOR" if m["oracle"] < 0.002 else "MIXED")
    out["A3"] = ("PASS" if m["rule2"] >= 0.004 and m["rule2"] >= 2 * m["real"] else
                 "KILLED" if m["rule2"] <= m["real"] else "MARGINAL")
    print(json.dumps(out, indent=1))
    if "--no-energy" not in sys.argv:
        sys.path.insert(0, os.path.join(TESTS, "P4.1_cheap_su2_r19"))
        import polish as P  # noqa: E402  (read only)
        import torch  # noqa: E402
        ctx = P.ctx_for("che")
        E00 = ctx.energy(ctx.th0)[0]
        gap = E00 - P.E0
        ths = [np.load(os.path.join(R25, f"results_guided_s{s}", "theta_shift.npy")) for s in SEEDS]
        Rk = [(E00 - ctx.energy(torch.tensor(t))[0]) / gap for t in ths]
        Rbar = (E00 - ctx.energy(torch.tensor(np.mean(ths, 0)))[0]) / gap
        spread = float(np.mean([np.linalg.norm(t - np.mean(ths, 0)) for t in ths]))
        A2 = "NOISE-BALL" if Rbar >= max(Rk) + 0.05 else "NO-BALL" if Rbar <= max(Rk) else "WEAK"
        out.update(R_replicas=Rk, R_mean_theta=Rbar, mean_dist_to_mean=spread, A2=A2, cpu_min=P.cpu_min())
        print(f"A2: R(replicas) {np.round(Rk, 4)}; R(mean theta) {Rbar:.4f}; |theta_k - mean| {spread:.3f} -> {A2}")
    print(f"A1 {out['A1']}; A3 {out['A3']}")
    json.dump(out, open(os.path.join(HERE, "results", "floor_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
