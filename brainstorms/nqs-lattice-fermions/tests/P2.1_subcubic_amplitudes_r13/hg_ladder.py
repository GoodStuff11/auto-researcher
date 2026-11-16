#!/usr/bin/env python3
"""
P2.1 Round 13: hole-gated (hg) ladder job (T1 completion at size 23, then the T2 sizes 24 and 25). Not submitted.
================================================================================================================

Runs ../P2.1_subcubic_amplitudes_r12b/ladder.train (imported read-only: the Round 12 training loop step for step,
with checkpoint and resume) with t1_gated.GatedAnsatz('hg') injected into t1_sub.make of this process only.
Protocol and caps are the Round 12b ones, except --max-cpu (default 6600 SR CPU-s, so a task stays under 2 CPU-h
with the budget step). Results in ./results_ladder/t1_hg_inst<I>_s11.json, the Round 12 format.

Tasks (--task K): 0 -> instance 23 (4x3 5+5, V = 12; completes the Round 13 T1, whose size-23 run hit the
480 CPU-s cap), 1 -> instance 24 (7x2 6+6, V = 14), 2 -> instance 25 (5x3 7+7, V = 15).
Modes
  --mode train      --task K: train to the plateau (or the caps), then one budget; refuses to overwrite.
  --mode probe      --task K --probe-steps N: CPU-s per step, no output.
  --mode aggregate  rho = dE_sign(hg) / dE_sign(full_F4) over sizes 20 (from ./results_acc), 23, 24, 25, and the
                    Round 12 rule (t1_sub.decide logic: KILL if rho > 1.25 at >= 2 counted sizes, or rho rising
                    strictly in V over >= 3 counted sizes with rho(V_max) > 1.15; PASS-TO-T2 if no kill on >= 3
                    counted sizes; INCONCLUSIVE otherwise).
Threads: OMP/OPENBLAS/MKL_NUM_THREADS=1.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R12B = os.path.join(HERE, "..", "P2.1_subcubic_amplitudes_r12b")
sys.path.insert(0, R12B)
sys.path.insert(0, HERE)
import ladder as LD  # noqa: E402  (read-only; imports t1_sub as LD.T)
import t1_gated as TG  # noqa: E402

TASKS = {0: 23, 1: 24, 2: 25}
OUT = os.path.join(HERE, "results_ladder")


def install():
    T = LD.T
    old = T.make

    def make(arm, m, F0, F, Ha, seed):
        if arm == "hg":
            return TG.GatedAnsatz(m, F0, F, Ha, "hg", seed=seed)
        return old(arm, m, F0, F, Ha, seed)
    T.make = make
    T.RNG_OFF["hg"] = 305


def aggregate():
    T = LD.T
    R = {}
    for d in (os.path.join(HERE, "results_acc"), OUT):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.startswith("t1_hg_") and fn.endswith(".json"):
                    r = json.load(open(os.path.join(d, fn)))
                    if r["converged"] or r["instance"] not in R:
                        R[("half", r["instance"])] = r     # scored through decide()'s 'half' slot
    for inst in (20, 23, 24, 25):
        p = os.path.join(TG.R12 if inst == 20 else R12B, "results", f"t1_full_inst{inst}_s11.json")
        R[("full", inst)] = json.load(open(p))
    v, why, rows = T.decide(R)["half"]
    print(f"VERDICT hg (Round 12 rule): {v} ({'; '.join(why)})")
    for inst, V, r in rows:
        print(f"    inst {inst} V={V}: " + (f"rho = {r:.3f}" if isinstance(r, float) else r))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "probe", "aggregate"], required=True)
    ap.add_argument("--task", type=int, choices=sorted(TASKS), default=0)
    ap.add_argument("--max-cpu", type=float, default=6600.0)
    ap.add_argument("--probe-steps", type=int, default=5)
    a = ap.parse_args(argv)
    if a.mode == "aggregate":
        aggregate(); return
    install()
    inst = TASKS[a.task]
    args = LD.parse_args(["--mode", "train", "--instance", str(inst), "--arm", "full", "--max-cpu", str(a.max_cpu),
                          "--results", OUT])
    args.arm = "hg"
    if a.mode == "probe":
        args.probe_steps = a.probe_steps
        LD.train(args); return
    args.probe_steps = 0
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"t1_hg_inst{inst}_s11.json")
    if os.path.exists(fn):
        LD.log(f"{fn} exists; refusing to overwrite"); return
    r = LD.train(args)
    json.dump(r, open(fn, "w"))
    LD.log(f"wrote {fn}")


if __name__ == "__main__":
    main()
