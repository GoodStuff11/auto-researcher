#!/usr/bin/env python3
"""
P4.1 R2 T2b with the equal-wall-time amendment: verdict = t2b_report.rule_t2b, then the amended PASS clause (2026-11-16).
=====================================================================================================================

Usage:
    python3 t2b_wall_report.py --mode {check, report, preflight} [--resdir PATH] [--dmrgdir PATH] [--toydir PATH]

This script imports t2b_report.py READ-ONLY (rule_t2b, best, load, by_arm, rho_sigma, E0_12) and does not change its
rule: the original T2b verdict is computed first, exactly as t2b_report.py --mode report prints it. The amendment
(leaf section "R2 T2b equal-wall-time amendment (2026-11-16)", written before any T2b result was aggregated) only
acts on a PASS.

Modes
-----
  check      Synthetic results covering every branch of the amended rule (and that every non-PASS original verdict
             passes through unchanged); prints "CHECK PASS n/n" (save to logs/wall_check_output.txt).
  report     Reads <resdir>/t2b/S*x3_*.json (T2b, job 699824) and <resdir>/t2b_wall/S*x3_*_wall.json (che_wall,
             r1_wall), the 24-site (+,+) reference (scaling_report.load_refs(--dmrgdir), job 687302), prints the
             original verdict, the amended verdict, the realised cost ratio and the equal-wall context lines.
  preflight  Toy 4x2 dry run: T2b toy files from <toydir>/results_preflight/{s24,s12} and che_wall toy files from
             <toydir>/results_preflight_wall/{s24,s12} (preflight_wall.sh), plus the identity control
             (plain_* against id_*), real-file controls, and prints each verdict (logs/wall_preflight_output.txt).

Options
-------
  --resdir PATH   results root (default <script dir>/results)
  --dmrgdir PATH  DMRG JSONs (default ../P4.1_cheap_su2_scaling/dmrg)
  --toydir PATH   folder holding results_preflight and results_preflight_wall (default <script dir>)

Amended rule (pre-registered)
-----------------------------
  V0 = t2b_report.rule_t2b(T2b runs). If V0 is not PASS, the verdict is V0, unchanged (KILL, AMBIGUOUS, ...).
  If V0 is PASS:
    INCOMPLETE-WALL  che_wall at 24 sites missing or capped (resubmit array index 0; it resumes);
    INCOMPLETE-WALL  realised ratio R = (s/step of pcg)/(s/step of che), both from the 699824 training trajectories at
                     24 sites (intervals containing a checkpoint excluded), exceeds 1.10 x R_PRE = 3.08: che_wall was
                     given less than the wall time of pcg; rerun it at ceil(3000 R / 100) x 100 steps, tag suffix _wall2;
    PASS             d_w = E_che_wall - E_pcg > 2 sigma_w, sigma_w = hypot(dE_che_wall, dE_pcg) (best of seeds per arm,
                     24 sites; no reference needed): pcg beats che at matched parameters AND at equal wall time;
    PARAMETER-EFFICIENT-ONLY  otherwise: a scoped result (pcg is better per parameter and per step, not per CPU-hour);
                     not a scaling PASS, so R2 does not go to T3 as a cost-competitive method.
  Context, never gating: rho24_wall = (E_pcg - E_ref)/(E_che_wall - E_ref), rho12_wall = eps_pcg/eps_che_wall at 12
  sites (exact), d_r1w = E_r1_wall - E_pcg at 24 sites.
"""
import argparse
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2b_report as TR  # noqa: E402  (read only)

SCAL = TR.SCAL
E0_12 = TR.E0_12
R_PRE = {24: 2.8, 12: 1.7}  # pre-registered pcg/che cost ratios that set the che_wall step counts (8400, 5100)
R1_PRE = 5.5                # pcg/r1 at 24 sites (r1_wall, 16500 steps)
R_TOL = 1.10


def sps(runs, min_it=50):
    """seconds per SR step from training trajectories [it, E, std, elapsed], intervals with a checkpoint excluded."""
    dt = dn = 0.0
    for r in runs or []:
        tr = r.get("traj") or []
        ck = [int(k) for k in (r.get("ckpt") or {})]
        for (a, _, _, ta), (b, _, _, tb) in zip(tr, tr[1:]):
            if a >= min_it and tb > ta and not any(a < c <= b for c in ck):
                dt += tb - ta
                dn += b - a
    return dt / dn if dn else None


def rule_wall(r24, r12, w24, w12, ref24, r_pre=R_PRE[24], min_it=50):
    """r24, r12: T2b runs by arm; w24, w12: wall runs by arm (che, r1). Returns (verdict, V0, notes)."""
    v0, notes = TR.rule_t2b(r24, r12, ref24)
    notes = list(notes)
    p = TR.best(r24.get("pcg", []))
    ctx = []
    if p is not None:
        cw = TR.best(w24.get("che", []))
        if cw is not None and ref24 is not None and ref24[2] == "OK" and cw["E"] - ref24[0] > 0:
            rw, sw = TR.rho_sigma(p["E"], p["dE"], cw["E"], cw["dE"], ref24[0], ref24[1])
            ctx.append(f"context: rho24_wall = {rw:.3f} +- {sw:.3f}")
        rw1 = TR.best(w24.get("r1", []))
        if rw1 is not None:
            d1 = rw1["E"] - p["E"]
            ctx.append(f"context: E_r1_wall - E_pcg = {d1:+.5f} ({d1 / math.hypot(rw1['dE'], p['dE']):+.1f} sigma)")
    p12, c12w = TR.best(r12.get("pcg", [])), TR.best(w12.get("che", []))
    if p12 is not None and c12w is not None:
        e = [(x["E"] - E0_12) / abs(E0_12) for x in (p12, c12w)]
        ctx.append(f"context: 12 sites eps_pcg {e[0]:.3e}, eps_che_wall {e[1]:.3e}, rho12_wall = {e[0] / e[1]:.3f}")
    if v0 != "PASS":
        return v0, v0, notes + [f"original verdict {v0} stands (the amendment acts only on a PASS)"] + ctx
    cws = w24.get("che", [])
    if not cws:
        return "INCOMPLETE-WALL", v0, notes + ["che_wall at 24 sites missing (sbatch --array=0 submit_t2b_wall.sbatch)"] + ctx
    if any(r.get("capped") for r in cws):
        return "INCOMPLETE-WALL", v0, notes + ["che_wall at 24 sites capped (resubmit index 0; it resumes)"] + ctx
    sp, sc = sps(r24.get("pcg"), min_it), sps(r24.get("che"), min_it)
    if sp and sc:
        R = sp / sc
        notes.append(f"realised cost ratio pcg/che at 24 sites {R:.2f} ({sp:.2f} / {sc:.2f} s/step); pre-registered {r_pre}")
        if R > R_TOL * r_pre:
            n_new = int(math.ceil(3000 * R / 100.0) * 100)
            return "INCOMPLETE-WALL", v0, notes + [f"che_wall under-budgeted: rerun at {n_new} steps, tag suffix _wall2"] + ctx
    else:
        notes.append("realised cost ratio not available (no trajectories); pre-registered value used")
    cw = TR.best(cws)
    d, s = cw["E"] - p["E"], math.hypot(cw["dE"], p["dE"])
    notes.append(f"equal wall: E_che_wall {cw['E']:.5f} +- {cw['dE']:.5f} (seeds {len(cws)}), E_pcg {p['E']:.5f}; "
                 f"d_w = {d:+.5f} ({d / s:+.1f} sigma)")
    if d > 2 * s:
        return "PASS", v0, notes + ["pcg beats che at matched parameters and at equal wall time"] + ctx
    return "PARAMETER-EFFICIENT-ONLY", v0, notes + ["pcg does not beat che_wall by 2 sigma: scoped result"] + ctx


# ============================================================================ check
def check():
    Er = -16.60

    def traj(s_per_step, iters=3000):
        return [[it, 0.0, 1.0, it * s_per_step] for it in range(0, iters + 1, 10)]

    def run(E, dE=0.004, seed=0, sps_=None, **kw):
        r = dict(E=E, dE=dE, seed=seed, S2=None, capped=False, early_kill=None, ckpt={})
        if sps_ is not None:
            r["traj"] = traj(sps_)
        return dict(r, **kw)

    def e24(eps):
        return Er + eps * abs(Er)

    def e12(eps):
        return E0_12 + eps * abs(E0_12)

    b12 = {"pcg": [run(e12(1.2e-3), 0.0, S2=0.0)], "che": [run(e12(2.8e-3), 0.0)], "r1": [run(e12(2.3e-3), 0.0, S2=0.0)]}
    ref = (Er, 2e-4, "OK")

    def s24(ep, ec=1.0e-2, er=8e-3, sp=10.8, sc=3.9, **kw):
        return {"pcg": [run(e24(ep), sps_=sp, **kw)], "che": [run(e24(ec), sps_=sc)], "r1": [run(e24(er), sps_=2.0)]}

    def w(ecw, **kw):
        return {"che": [run(e24(ecw), **kw)]}

    w12 = {"che": [run(e12(2.0e-3), 0.0)]}
    sc = [
        ("PASS: original PASS, pcg beats che_wall by 5.9 sigma", s24(4e-3), b12, w(6e-3), w12, ref, "PASS"),
        ("PEO: che_wall 1.5 sigma above pcg", s24(4e-3), b12, w(4.5e-3), w12, ref, "PARAMETER-EFFICIENT-ONLY"),
        ("PEO: che_wall below pcg", s24(4e-3), b12, w(3e-3), w12, ref, "PARAMETER-EFFICIENT-ONLY"),
        ("PEO: best of che_wall seeds (seed 1 within 2 sigma)", s24(4e-3), b12,
         {"che": [run(e24(6e-3)), run(e24(4.5e-3), seed=1)]}, w12, ref, "PARAMETER-EFFICIENT-ONLY"),
        ("INCOMPLETE-WALL: che_wall missing", s24(4e-3), b12, {}, w12, ref, "INCOMPLETE-WALL"),
        ("INCOMPLETE-WALL: che_wall capped", s24(4e-3), b12, w(6e-3, capped=True), w12, ref, "INCOMPLETE-WALL"),
        ("INCOMPLETE-WALL: realised ratio 3.38 > 3.08", s24(4e-3, sp=13.2, sc=3.9), b12, w(6e-3), w12, ref,
         "INCOMPLETE-WALL"),
        ("PASS: realised ratio 3.05 within 10%", s24(4e-3, sp=11.9, sc=3.9), b12, w(6e-3), w12, ref, "PASS"),
        ("PASS: 12-site che_wall missing (context only)", s24(4e-3), b12, w(6e-3), {}, ref, "PASS"),
        ("PASS: r1_wall below pcg (context only)", s24(4e-3), b12, dict(w(6e-3), r1=[run(e24(3e-3))]), w12, ref, "PASS"),
        ("KILL unchanged: rho24 0.90, che_wall far above", s24(9e-3, er=1.2e-2), b12, w(2e-2), w12, ref, "KILL"),
        ("KILL unchanged: pcg above r1", s24(5e-3, er=4.5e-3), b12, w(6e-3), w12, ref, "KILL"),
        ("AMBIGUOUS unchanged: rho24 0.70", s24(7e-3, er=1.1e-2), b12, w(2e-2), w12, ref, "AMBIGUOUS"),
        ("INCOMPLETE-REFERENCE unchanged", s24(4e-3), b12, w(6e-3), w12, None, "INCOMPLETE-REFERENCE"),
        ("INCOMPLETE unchanged: pcg capped", s24(4e-3, capped=True), b12, w(6e-3), w12, ref, "INCOMPLETE"),
        ("NO-TARGET unchanged", s24(-1e-4, ec=1e-4, er=5e-4), b12, w(6e-3), w12, ref, "NO-TARGET"),
    ]
    ok = 0
    for name, a24, a12, b24, b12_, rf, want in sc:
        v, v0, nt = rule_wall(a24, a12, b24, b12_, rf)
        good = v == want
        ok += good
        print(f"[{'OK ' if good else 'BAD'}] {name:52s} -> {v:24s} (original {v0}; intended {want}) | {'; '.join(nt)}")
    print(f"CHECK {'PASS' if ok == len(sc) else 'FAIL'} {ok}/{len(sc)}")
    return ok == len(sc)


# ============================================================================ report
def report(a):
    t2b_files = glob.glob(os.path.join(a.resdir, "t2b", "S*x3_*.json"))
    assert not any("_wall" in f for f in t2b_files), "a wall file is inside results/t2b: t2b_report would misread it"
    rs = [r for r in TR.load(os.path.join(a.resdir, "t2b", "S*x3_*.json"))]
    ws = TR.load(os.path.join(a.resdir, "t2b_wall", "S*x3_*_wall.json"))
    r24, r12 = (TR.by_arm([r for r in rs if r["L"] == L]) for L in (24, 12))
    w24, w12 = (TR.by_arm([r for r in ws if r["L"] == L]) for L in (24, 12))
    sys.path.insert(0, SCAL)
    ref24 = None
    try:
        import scaling_report as SR
        ref24 = SR.load_refs(a.dmrgdir).get(24)
    except Exception as e:  # noqa: BLE001
        print(f"  (reference loader failed: {e})")
    v, v0, nt = rule_wall(r24, r12, w24, w12, ref24)
    print(f"VERDICT T2b original rule: {v0}")
    print(f"VERDICT T2b with the equal-wall-time amendment: {v}")
    for n_ in nt:
        print(f"  {n_}")
    for L, d, ratio in ((24, r24, R_PRE[24]), (12, r12, R_PRE[12])):
        sp, sc = sps(d.get("pcg")), sps(d.get("che"))
        if sp and sc:
            print(f"  realised s/step at {L} sites: pcg {sp:.3f}, che {sc:.3f}, ratio {sp / sc:.2f} (pre-registered {ratio})")
    for r in ws:
        print(f"  wall run {r['tag']}: E {r['E']:.5f} +- {r['dE']:.5f}, steps {r['iters_done']}/{r['iters']}, "
              f"capped {r['capped']}, wall {r['wall'] / 3600:.2f} h")
    TR.ckpt_lines(ws, ref24[0] if ref24 and ref24[2] == "OK" else None)


# ============================================================================ preflight
def preflight(a):
    out = []
    # identity control: the wrapper changes only the tag
    for slot in ("s12", "s24"):
        pl = TR.load(os.path.join(a.toydir, "results_preflight_wall", f"plain_{slot}", "S4x2_*.json"))
        idw = TR.load(os.path.join(a.toydir, "results_preflight_wall", f"id_{slot}", "S4x2_*_wall.json"))
        same = bool(pl) and bool(idw) and all(pl[0][k] == idw[0][k] for k in ("E", "dE", "iters_done")) and \
            [t[:3] for t in pl[0]["traj"]] == [t[:3] for t in idw[0]["traj"]] and \
            {k: (c["E"], c["dE"]) for k, c in pl[0]["ckpt"].items()} == {k: (c["E"], c["dE"]) for k, c in idw[0]["ckpt"].items()}
        out.append((f"identity control {slot} (t2b.py vs t2b_wall.py)", "IDENTICAL" if same else "DIFFERENT",
                    "IDENTICAL", [f"final E, checkpoint E and training energies equal bitwise (wall times excluded); E {pl[0]['E']:.10f}; tags {pl[0]['tag']} / {idw[0]['tag']}"]))
    s24 = TR.by_arm(TR.load(os.path.join(a.toydir, "results_preflight", "s24", "S4x2_*.json")))
    s12 = TR.by_arm(TR.load(os.path.join(a.toydir, "results_preflight", "s12", "S4x2_*.json")))
    w24 = TR.by_arm(TR.load(os.path.join(a.toydir, "results_preflight_wall", "s24", "S4x2_*_wall.json")))
    w12 = TR.by_arm(TR.load(os.path.join(a.toydir, "results_preflight_wall", "s12", "S4x2_*_wall.json")))
    E0 = s12["che"][0]["E0"]
    print(f"toy 4x2 (+,+): E0 {E0:.6f}; pcg s24 {s24['pcg'][0]['E']:.5f}+-{s24['pcg'][0]['dE']:.5f}; che s24 "
          f"{s24['che'][0]['E']:.5f}; che_wall s24 {w24['che'][0]['E']:.5f}+-{w24['che'][0]['dE']:.5f} "
          f"({w24['che'][0]['iters']} steps, ckpts {list(w24['che'][0]['ckpt'])}); che_wall s12 {w12['che'][0]['E']:.5f}")
    ref = (E0, 0.0, "OK")

    def to12(runs):
        return [dict(r, E=E0_12 + (r["E"] - E0) / abs(E0) * abs(E0_12)) for r in runs]

    t12 = {k: to12(v) for k, v in s12.items()}
    tw12 = {k: to12(v) for k, v in w12.items()}
    v0_real = TR.rule_t2b(s24, t12, ref)[0]
    v, v0, nt = rule_wall(s24, t12, w24, tw12, ref, r_pre=10.0)
    out.append(("real toy files (original verdict must pass through)", v, v0_real, nt))
    ex24 = dict(s24["pcg"][0], E=E0, dE=1e-6, early_kill=None, capped=False)
    ex12 = dict(t12["pcg"][0], E=E0_12, S2=0.0)
    P24, P12 = dict(s24, pcg=[ex24]), dict(t12, pcg=[ex12])
    v, v0, nt = rule_wall(P24, P12, w24, tw12, ref, r_pre=10.0)
    out.append(("positive control (pcg = exact state; real che_wall)", v, "PASS", nt))
    v, v0, nt = rule_wall(P24, P12, {"che": [dict(w24["che"][0], E=E0, dE=1e-6)]}, tw12, ref, r_pre=10.0)
    out.append(("negative control (che_wall slot = exact state too)", v, "PARAMETER-EFFICIENT-ONLY", nt))
    v, v0, nt = rule_wall(P24, P12, {"che": [dict(w24["che"][0], E=ex24["E"] + 1e-6)]}, tw12, ref, r_pre=10.0)
    out.append(("negative control (che_wall at pcg's energy, d_w ~ 0)", v, "PARAMETER-EFFICIENT-ONLY", nt))
    v, v0, nt = rule_wall(P24, P12, {}, tw12, ref, r_pre=10.0)
    out.append(("missing che_wall", v, "INCOMPLETE-WALL", nt))
    v, v0, nt = rule_wall(P24, P12, w24, tw12, ref, r_pre=0.5, min_it=1)
    out.append(("realised-ratio guard on real toy trajectories (R_pre 0.5)", v, "INCOMPLETE-WALL", nt))
    ok = True
    for name, v, want, nt in out:
        good = v == want
        ok &= good
        print(f"[{'OK ' if good else 'BAD'}] {name:58s} -> {v} (intended {want}) | {'; '.join(nt)}")
    print(f"PREFLIGHT {'PASS' if ok else 'FAIL'}")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["check", "report", "preflight"])
    p.add_argument("--resdir", default=os.path.join(HERE, "results"))
    p.add_argument("--dmrgdir", default=os.path.join(SCAL, "dmrg"))
    p.add_argument("--toydir", default=HERE)
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
