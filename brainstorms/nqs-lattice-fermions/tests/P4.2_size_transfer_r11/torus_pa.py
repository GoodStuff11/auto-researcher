#!/usr/bin/env python3
r"""
P4.2 round 11: R5.1 (PA, prior-anchored copy) on the 4x4 TORUS, one seed, anchored arm only.
==============================================================================================

Why this exists (R9.5 shortcut)
-------------------------------
R9.5 wanted a 47 CPU-h MFNG-O capacity pilot on the 4x4 torus before running R5.1 there, because
R5.1's pre-committed capacity rule needs the scratch plateau at <= 3% relative error. Job 451914
(R3.1 closing run, ../P4.2_size_transfer_r5/results_close31/) already trained the size-independent
periodic MFNG (122 parameters; channels 2, field 8, backflow 4, Jastrow 4) from its MF init on
exactly this target (4x4 torus, 6+6, t'=-0.25, J=0.4, irrep Tx=Ty=Mx=My=+1, D=-1, n_irrep 13070,
E0=-13.93410177) for 50 projected-SR steps: eps(50) = 0.01858 / 0.01707 / 0.01828 t/site
= 2.13 / 1.96 / 2.10 % relative error. Projected SR with backtracking never raises the energy, so
the family's plateau is <= 1.96 % (<= 3 %) for THIS family. We therefore run R5.1 with the same
MFNG family (not MFNG-O, which is an OBC-only class with a different 8x8-quadrature kernel), on
R3.1's grid-complete periodic sources (4x2 torus and 8-site tilted t8, 3+3), and reuse 451914's
`scr` arm (3 seeds) as the control. Only the anchored arm is new.

Arm `pa` (seed 0)
-----------------
Sources 4x2 + t8 fitted with the L2-SP anchor lam |theta - theta_init|^2 toward the source run's
own MF init (best of 4 restarts by source loss, 800 SR steps, rng 1000 + 100 seed + r), ladder
lam in {1e-2, 1e-3, 1e-4}: the LARGEST whose source fit has max relative error <= 1e-2 on both
sources, else the smallest with anchor_ok = False (INVALID). Every rung tried is charged (wall).
theta is copied unchanged to the target and trained 50 projected-SR steps (lr 0.05, shift 0.01,
identical to 451914). Records at 0, 10, 20, 50.

Consistency check (else INVALID): the seed-0 MF init evaluated here reproduces 451914's
scr eps(0) = 0.3181487336 to 1e-8 (same target, sector, projection and code).

Decision (pre-registered in RULE.md; --aggregate). Restates the leaf's R5.1 rule
(obc_transfer.py docstring), with the three adaptations listed in RULE.md.

Usage
-----
  python3 torus_pa.py --seed 0                    # Slurm task (~1 h, 8 threads)
  python3 torus_pa.py --timing --timing-steps 40  # login-node timing (no ED, < 5 CPU-min)
  python3 torus_pa.py --aggregate results/pa_s0.json
  python3 torus_pa.py --pilot-dir ../P4.2_size_transfer_r3/results_smoke --steps 6 --ckpts 2,6 --steps-src 100 \
      --src-restarts 1 --out results_smoke/pa_s0.json   # smoke (2+2 electrons; INVALID by design: scr0 check)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "..", "P4.2_size_transfer_r5")
sys.path.insert(0, R5)
import r3_core as rc  # noqa: E402
from close31 import r3_args  # noqa: E402
from r5_common import identified_fraction, med, optimise_ckpt  # noqa: E402

PILOT = os.path.join(HERE, "..", "P4.2_size_transfer_r3", "results_zeroshot")
CONTROL = os.path.join(R5, "results_close31")
SCR0_451914 = 0.3181487336346709      # 451914 seed-0 scr eps(0)
X_FROZEN = 0.017071481144602108       # eps_best of 451914 (all arms, steps, seeds); frozen in RULE.md
B_RULE = (20, 50)
PLATEAU_FRAC, P1_EPS = 0.02, 1e-3


class Anchored:
    """theta -> net(theta_init + theta), so r3_core.optimise's lam |theta|^2 is the L2-SP anchor."""

    def __init__(self, net, th_init):
        self.net, self.th0 = net, th_init
        self.lat, self.npar, self.jas_mask = net.lat, net.npar, net.jas_mask

    def precompute(self, th, need_grad):
        return self.net.precompute(th + self.th0, need_grad)

    def evaluate(self, th, states, need_grad, cache=None):
        return self.net.evaluate(th + self.th0, states, need_grad, cache)


def train_rung(args, src, lam, seed, steps, restarts, tag):
    t0 = time.time()
    dens = 2 * src[0]["nup"] / src[0]["lat"].N
    best, losses = None, []
    for r in range(restarts):
        net0 = rc.new_net(src[0]["lat"], src[0]["nup"], args)
        th_init = net0.init(np.random.default_rng(1000 + 100 * seed + r), dens, args.t, args.tp_sel,
                            args.delta0, args.eta)
        cls = [rc.Cluster(Anchored(rc.new_net(s["lat"], s["nup"], args), th_init), s["basis"],
                          rc.sp.identity(len(s["basis"]), format="csr"), s["H"]) for s in src]
        d, Es, _ = rc.optimise(cls, np.zeros(net0.npar), steps, args.lr, args.diag_shift, args.chunk,
                               f"{tag} restart {r}", lam=lam, Eref=[s["E0"] for s in src])
        L = sum(e / c.V for e, c in zip(Es, cls)) + lam * float(d @ d)
        losses.append(L)
        if best is None or L < best[0]:
            best = (L, th_init + d, th_init, Es, cls, d)
    _, th, th_init, Es, cls, d = best
    wall = time.time() - t0
    rel = [float((e - s["E0"]) / abs(s["E0"])) for e, s in zip(Es, src)]
    return th, th_init, cls, d, dict(anchor=lam, E=[float(e) for e in Es], relerr=rel, restart_losses=losses,
                                     dist_init=float(np.linalg.norm(d)), wall_s=wall)


def run(cli):
    t_start = time.time()
    deadline = t_start + cli.wall_hours * 3600
    pil = json.load(open(os.path.join(cli.pilot_dir, "task0.json")))
    args = r3_args(pil["params"])
    out = cli.out or os.path.join(HERE, "results", f"pa_s{cli.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R = dict(cli=vars(cli), seed=cli.seed, arms={}, hist={}, checks={}, valid=False, done=False)

    def dump():
        with open(out, "w") as fh:
            json.dump(R, fh, indent=1, default=float)

    T = rc.target_setup(args, R)
    assert T is not None
    R["setup_s"] = time.time() - t_start
    record = rc.make_recorder(args, R, T, dump)
    lat, nup, V, E0 = T["lat"], T["nup"], T["V"], T["E0"]
    cl = rc.Cluster(rc.new_net(lat, nup, args), T["basis"], T["U"], T["HG"])
    # consistency with the 451914 control (seed-0 MF init, same as 451914 scr)
    th_scr0 = cl.net.init(np.random.default_rng(2000 + 0), 2 * nup / lat.N, args.t, args.tp_sel, args.delta0, args.eta)
    e_scr0 = (rc.reduced(cl, th_scr0, args.chunk, False)[0] - E0) / V
    R["checks"]["scr0_eps"] = e_scr0
    R["checks"]["scr0_reproduced"] = bool(abs(e_scr0 - SCR0_451914) <= 1e-8)
    rc.log(f"check: seed-0 MF init eps={e_scr0:.10f} vs 451914 {SCR0_451914:.10f} -> {R['checks']['scr0_reproduced']}")
    dump()
    # ---- anchored sources (charged: source ED + every rung)
    t_src = time.time()
    src = rc.build_source(args, args.tp_sel, args.J_sel, pil["params"]["sources"].split(","))
    t_ed = time.time() - t_src
    ladder, charged, res = [], t_ed, None
    for lam in sorted((float(x) for x in cli.anchors.split(",")), reverse=True):
        res = train_rung(args, src, lam, cli.seed, cli.steps_src, cli.src_restarts, f"src anchor {lam:g}")
        charged += res[4]["wall_s"]
        ladder.append(res[4])
        rc.log(f"source anchor {lam:g}: relerr={['%.2e' % x for x in res[4]['relerr']]} "
               f"|d|={res[4]['dist_init']:.3f} wall={res[4]['wall_s']:.1f}s")
        if max(res[4]["relerr"]) <= cli.src_tol:
            break
    th_src, th_init, cls, d, info = res
    info["identified"] = identified_fraction(cls, d, args.chunk, cli.id_tol)
    R["source"] = dict(ladder=ladder, selected=info, anchor_ok=bool(max(info["relerr"]) <= cli.src_tol),
                       src_ed_s=t_ed, src_charged_s=charged)
    rc.log(f"source charged {charged:.1f}s; anchor {info['anchor']:g} ok={R['source']['anchor_ok']} "
           f"identified frac={info['identified']['frac_identified']:.3f}")
    dump()
    # ---- target fine-tune of the copied theta
    ck = sorted({int(x) for x in cli.ckpts.split(",")})

    def on_ckpt(n, th):
        record(f"pa@{n}", cl.net, th, sign=(n in (0, ck[-1])))

    clock = []
    _, h = optimise_ckpt(cl, th_src, cli.steps, args.lr, args.diag_shift, args.chunk, "pa", E0, V,
                         ckpts=[0] + ck, on_ckpt=on_ckpt, deadline=deadline, clock=clock,
                         on_step=lambda n, hh: (R["hist"].__setitem__("pa", list(hh)),
                                                R.__setitem__("clock", list(clock)),
                                                dump() if n % 10 == 0 else None))
    R["hist"]["pa"] = h
    R["clock"] = clock
    R["valid"] = bool(R["checks"]["scr0_reproduced"] and R["source"]["anchor_ok"])
    R["done"] = len(h) == cli.steps + 1
    R["wall_s"] = time.time() - t_start
    dump()
    rc.log(f"done ({out}); valid={R['valid']} steps={len(h) - 1}")


# ============================================================================ aggregate
def first_reach(series, level):
    for n, v in enumerate(series):
        if v <= level:
            return n
    return None


def aggregate(path):
    r = json.load(open(path))
    C = [json.load(open(os.path.join(CONTROL, f"close31_s{s}.json")))["hist"] for s in range(3)]
    scr = [c["scr"] for c in C]
    cp = [c["tr"] for c in C]                 # unanchored (min-norm) copy from the same sources: ablation
    pa = r["hist"]["pa"]
    T = json.load(open(os.path.join(CONTROL, "close31_s0.json")))["target"]
    V, E0 = 16, T["E0"]
    x = X_FROZEN
    step_s = r["clock"][-1] / max(1, len(r["clock"]) - 1)
    src_steps = r["source"]["src_charged_s"] / step_s
    print(f"pa seed {r['seed']}: valid={r['valid']} done={r['done']} steps={len(pa) - 1} "
          f"step={step_s:.1f}s source charged {r['source']['src_charged_s']:.1f}s = {src_steps:.2f} target steps")
    print(f"x (frozen eps_best of 451914) = {x:.6f} t/site = {100 * x * V / abs(E0):.3f}% (capacity <= 3%: {x * V / abs(E0) <= 0.03})")
    l0 = min(med([s[0] for s in scr]), pa[0])
    S, counted = {}, []
    for B in B_RULE:
        lB = med([s[B] for s in scr])
        drop = med([s[B - 5] - s[B] for s in scr])
        cnt = (lB - x >= P1_EPS) and (drop >= PLATEAU_FRAC * (lB - x))
        if cnt:
            counted.append(B)
        spans_c = []
        for s in scr:
            n0, n1 = first_reach(s, l0), first_reach(s, lB)
            spans_c.append(float("inf") if n0 is None or n1 is None else n1 - n0)
        n0, n1 = first_reach(pa, l0), first_reach(pa, lB)
        span_x = float("inf") if n0 is None or n1 is None else n1 - n0 + src_steps
        sc = float(np.median(spans_c))
        S[B] = 0.0 if not np.isfinite(span_x) else (sc / span_x if span_x > 0 else float("inf"))
        print(f"B={B}: level {lB:.5f} counted={cnt} control spans {spans_c} (median {sc}) pa span {span_x:.2f} S_step={S[B]:.3f}")
    b = B_RULE[-1]
    mC, mcp, e_pa = med([s[b] for s in scr]), med([c[b] for c in cp]), pa[b] if len(pa) > b else float("nan")
    b_pass = e_pa - x <= 0.5 * (mC - x)
    b_fail = e_pa - x >= 0.9 * (mC - x)
    abl = e_pa < mcp
    print(f"b={b}: eps_pa={e_pa:.5f} m eps_scr={mC:.5f} m eps_cp={mcp:.5f}; b-pass={b_pass} b-fail={b_fail} ablation={abl}")
    print(f"diagnostic, x=0 end: eps_pa(b)/m eps_scr(b) = {e_pa / mC:.3f}")
    if not r["valid"] or not r["done"]:
        v = "INVALID" if not r["valid"] else "INCOMPLETE"
    elif all(S[B] < 1.25 for B in counted) and counted or b_fail:
        v = "FAIL"
    elif len([B for B in counted if S[B] >= 2]) >= 2 and b_pass and abl:
        v = "PASS"
    else:
        v = "AMBIGUOUS"
    act = {"PASS": "reopen P4.2 on tori (R9.5 unshelved; release seeds 1-2 + scr+ control)",
           "FAIL": "close R9.5; P4.2 -> EXHAUSTED at this scale (if the optimiser row also holds)",
           "AMBIGUOUS": "close R9.5 at this budget (reopening requires PASS)",
           "INVALID": "fix and rerun (no decision)", "INCOMPLETE": "no decision"}[v]
    print(f"GLOBAL R5.1-torus PA (seed {r['seed']}): {v} -> {act}")


def timing(cli):
    """Login-node timing without ED/isometry: one target grad + value pass on one chunk, and one
    anchored source restart for --timing-steps steps."""
    pil = json.load(open(os.path.join(PILOT, "task0.json")))
    args = r3_args(pil["params"])
    args.tp_sel, args.J_sel = -0.25, 0.4
    t0 = time.time()
    lat = rc.make_lattice("4x4", args.t, -0.25, 0.4)
    basis = rc.make_basis(lat.N, 6, 6)
    net = rc.new_net(lat, 6, args)
    th = net.init(np.random.default_rng(2000), 0.75, args.t, -0.25, args.delta0, args.eta)
    rc.log(f"target basis {len(basis)} built in {time.time() - t0:.1f}s, npar={net.npar}")
    cache = net.precompute(th, True)
    ch = basis[:args.chunk]
    t1 = time.time(); net.evaluate(th, ch, True, cache); tg = time.time() - t1
    t1 = time.time(); net.evaluate(th, ch, False, cache); tv = time.time() - t1
    nch = len(basis) / args.chunk
    rc.log(f"chunk {args.chunk}: grad {tg:.2f}s value {tv:.2f}s -> per SR step (grad + ~1.3 value passes) "
           f"~{nch * (tg + 1.3 * tv):.0f}s at this thread count")
    t1 = time.time()
    src = rc.build_source(args, -0.25, 0.4, ["4x2", "t8"])
    ted = time.time() - t1
    res = train_rung(args, src, 1e-2, 0, cli.timing_steps, 1, "timing src")
    per = res[4]["wall_s"] / cli.timing_steps
    rc.log(f"source ED {ted:.1f}s; anchored source SR {per:.3f}s/step -> one rung (4 restarts x 800) "
           f"~{4 * 800 * per:.0f}s, full ladder (3 rungs) ~{3 * 4 * 800 * per + ted:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ckpts", default="10,20,50")
    p.add_argument("--anchors", default="1e-2,1e-3,1e-4")
    p.add_argument("--src-tol", type=float, default=1e-2)
    p.add_argument("--src-restarts", type=int, default=4)
    p.add_argument("--steps-src", type=int, default=800)
    p.add_argument("--id-tol", type=float, default=1e-6)
    p.add_argument("--wall-hours", type=float, default=1.05)
    p.add_argument("--out", default=None)
    p.add_argument("--pilot-dir", default=PILOT, help="r3 pilot JSONs (model/instance/sizes); smoke: ../P4.2_size_transfer_r3/results_smoke")
    p.add_argument("--aggregate", default="")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--timing-steps", type=int, default=40)
    cli = p.parse_args()
    if cli.aggregate:
        aggregate(cli.aggregate)
    elif cli.timing:
        timing(cli)
    else:
        run(cli)


if __name__ == "__main__":
    main()
