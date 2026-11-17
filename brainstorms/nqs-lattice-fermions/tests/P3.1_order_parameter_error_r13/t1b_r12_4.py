#!/usr/bin/env python3
"""
P3.1 Route 12.4 (multi-source response, MSR) T1b. Rule: RULE_R12.4.md in this directory (copied verbatim from the
Round 12 Route Portfolio of the leaf before this script was run).

  T1b at Delta = 0.05 t on 5x2 and 4x3: KILL if MSR-12 tracks at most OBS + 0.1 on both; it must also report the
  coherent competitor weight (signal only).

Reuses ../P3.1_order_parameter_error_r12/t1_r12.py unchanged (instances, gap surgery, trial states, grade-R noise
ETA = (0.3, 0.5), chains, scoring). Differences from the T1 main run, for the login-node CPU cap (<= 10 CPU-min):
only the two arms the rule compares are run (OBS: 10.1d, T = 4; MSR-12: T = 2); no witness pass and no DAV/RND arms
(they do not enter the rule; the witness does not change the tracked state).
Coherent competitor weight: a noisy response is y~ = (unit(y) + 0.3 g + 0.5 j) / N; its signal-only competitor
amplitude is <psi_1|unit(y)> / N. Reported per state as sum_i of its square (pass 1), beside the total
alpha^2 = sum_i |<psi_1|y~_i>|^2 that T1 reported (which includes the noise's own projection on psi_1).

Usage
  python3 t1b_r12_4.py --mode main --size 5x2 --cpu-limit 110
  python3 t1b_r12_4.py --mode main --size 4x3 --cpu-limit 470
  python3 t1b_r12_4.py --mode verdict
"""
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "r12", os.path.join(HERE, "..", "P3.1_order_parameter_error_r12", "t1_r12.py"))
r12 = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(r12)
sys.argv = _argv
dav, t1s, t1, np, unit, log, REAL = r12.dav, r12.t1s, r12.t1, r12.np, r12.unit, r12.log, r12.REAL
DELTA = 0.05
SIZES = ("5x2", "4x3")
RES = os.path.join(HERE, "results")


def noisy_resp_coh(I, phi, src, rng):
    """As r12.noisy_resp, but also returns the coherent (signal-only) competitor weight."""
    _, EL1 = t1s.ritz_vec(I, [phi, t1s.resid(I, phi)])
    s = EL1 - 0.1
    assert s < I.E0, ("VALIDITY: shift not below E0", s, I.E0)
    ys, coh = [], 0.0
    for b in src(phi):
        y = I.solve(b, s, orth=phi)
        yt_raw = None
        # replicate dav.noisy exactly, keeping the normalisation N
        g = unit(I.Uw @ rng.standard_normal(I.nwin))
        yt_raw = unit(y) + dav.ETA[0] * g + dav.ETA[1] * I.junk(rng, phi)
        N = float(np.linalg.norm(yt_raw))
        ys.append(yt_raw / N)
        coh += (float(I.U[:, 0] @ unit(y)) / N) ** 2
    return ys, coh


def chain(I, psi, src, rng, T):
    phi, info = unit(psi), {}
    for t in range(1, T + 1):
        ys, coh = noisy_resp_coh(I, phi, src, rng)
        if t == 1:
            info["alpha2"] = float(sum((I.U[:, 0] @ y) ** 2 for y in ys))
            info["coh2"] = coh
            phi = t1s.scru_pass(I, phi, ys)
        else:
            phi = t1s.ritz_vec(I, [phi] + ys)[0]
        phi = unit(phi)
    return phi, info


def run(I, cap, dump):
    perts = t1.perturbations(I)
    rng = np.random.default_rng(7)
    a0 = {k: float(I.psi0 @ I.apply(k, I.psi0)) for k in REAL}
    rows, worst = [], 0.0
    for si, lab in enumerate(list(perts)):
        c0 = r12.cpu()
        if r12.cpu() + 1.2 * worst > cap - 10:
            log(f"soft CPU guard: stop before state {si}")
            break
        psi = t1s.trial_state(I, perts[lab], rng)
        ap = {k: float(psi @ I.apply(k, psi)) for k in REAL}
        dl = {k: a0[k] - ap[k] for k in REAL}
        row = dict(size=I.name, state=lab, a0=a0, delta=dl, w1=float(I.U[:, 0] @ psi) ** 2,
                   scored={k: bool(abs(dl[k]) >= t1.SCORE_REL * abs(a0[k])) for k in REAL})
        rs = np.random.default_rng(30_000 + 97 * I.m.V + si)
        for kind, T in (("OBS", 4), ("MSR", 2)):
            phi, info = chain(I, psi, r12.make_sources(I, kind, rs), rs, T)
            row[kind] = {k: float(phi @ I.apply(k, phi) - ap[k]) / dl[k] for k in REAL}
            row[f"alpha2_{kind}"] = info["alpha2"]
            row[f"coh2_{kind}"] = info["coh2"]
        rows.append(row)
        worst = max(worst, r12.cpu() - c0)
        json.dump(dict(rows=rows), open(dump, "w"), indent=1)
        log(f"{I.name} {lab:7s} w1 {row['w1']:.3f} a2 OBS {row['alpha2_OBS']:.2f} MSR {row['alpha2_MSR']:.2f} "
            f"| coh2 OBS {row['coh2_OBS']:.3f} MSR {row['coh2_MSR']:.3f} cpu {r12.cpu() - c0:.0f}s | " + " | ".join(
                f"{k}{'' if row['scored'][k] else '(ns)'} OBS {row['OBS'][k]:+.2f} MSR {row['MSR'][k]:+.2f}"
                for k in REAL))
    return rows


def verdict():
    res = {}
    for s in SIZES:
        f = os.path.join(RES, f"t1b_{s}_d{DELTA}.json")
        if not os.path.exists(f):
            print(f"{s}: missing"); print("VERDICT R12.4 T1b: INCOMPLETE"); return
        rows = json.load(open(f))["rows"]
        n = len(r12.items(rows))
        fo, fm = r12.frac(rows, "OBS"), r12.frac(rows, "MSR")
        med = {f"{q}_{a}": float(np.median([r[f"{q}_{a}"] for r in rows]))
               for q in ("alpha2", "coh2") for a in ("OBS", "MSR")}
        res[s] = dict(n=n, states=len(rows), OBS=fo, MSR=fm, **med)
        print(f"{s} Delta={DELTA}: states {len(rows)} scored {n} | tracked OBS {fo:.2f} MSR-12 {fm:.2f} "
              f"(MSR - OBS = {fm - fo:+.2f}) | median total alpha^2 OBS {med['alpha2_OBS']:.2f} "
              f"MSR {med['alpha2_MSR']:.2f} | median coherent alpha^2 OBS {med['coh2_OBS']:.3f} "
              f"MSR {med['coh2_MSR']:.3f}")
    if any(res[s]["n"] < 6 for s in SIZES):
        v = "INCOMPLETE (< 6 scored items on an instance)"
    elif all(res[s]["MSR"] <= res[s]["OBS"] + 0.1 for s in SIZES):
        v = "KILL (MSR-12 tracks at most OBS + 0.1 on both instances)"
    else:
        v = "PASS (MSR-12 tracks more than OBS + 0.1 on at least one instance)"
    print(f"VERDICT R12.4 T1b: {v}")
    json.dump(dict(res=res, verdict=v), open(os.path.join(RES, "t1b_verdict.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("main", "verdict"), required=True)
    p.add_argument("--size", choices=SIZES, default="5x2")
    p.add_argument("--cpu-limit", type=int, default=110)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit + 5))
    os.makedirs(RES, exist_ok=True)
    if a.mode == "verdict":
        return verdict()
    I, info = dav.build(a.size, 15, DELTA)
    log(f"build cpu {r12.cpu():.0f}s")
    out = os.path.join(RES, f"t1b_{a.size}_d{DELTA}.json")
    rows = run(I, a.cpu_limit, out)
    json.dump(dict(info=info, rows=rows), open(out, "w"), indent=1, default=str)
    log(f"total CPU {r12.cpu():.0f}s")


if __name__ == "__main__":
    main()
