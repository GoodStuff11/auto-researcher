#!/usr/bin/env python3
"""
P4.1 ledger row 4, cheaper version (RULE_amend3.md): rho_abl = eps_pcg / eps_rank5 only.
========================================================================================
Drives the frozen ../P4.1_cheap_su2_r22/ladder.py (unchanged; it resumes from ../P4.1_cheap_su2_r22/results):
  phase 1  rank5 alone to STOP-RULE, STOP-NOGAIN or the 30-step hard cap (RULE_amend1), CPU cap 200 min.
  phase 2  only if rank5 is settled (RULE_amend2 guard) and pcg's current eps is above 0.60 x L5, with
           L5 = min(eps_rank5_end, eps_rank5_inf): pcg one ladder step per call until eps_pcg <= 0.60 x L5
           (STOP-DOMINANCE: ladder steps never raise E, so every later pcg endpoint and its tail-extrapolated value
           stay below the line), a pcg STOP-RULE / STOP-NOGAIN, or pcg's 30-step hard cap.
  verdict  results/verdict_row4.json. rho_pol is not read (che is not run): it stays at its provisional 742206 value.
Usage: ~/.conda/envs/lit_env/bin/python row4.py [--budget-min 400]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--budget-min", type=float, default=400.0)  # wall minutes (1 thread: wall ~= CPU)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
R22 = os.path.join(os.path.dirname(HERE), "P4.1_cheap_su2_r22")
PY = sys.executable
sys.argv = [os.path.join(R22, "ladder.py"), "--mode", "verdict"]
sys.path.insert(0, R22)
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import ladder as LD  # noqa: E402  (frozen; parse_args sees --mode verdict, __main__ block not run)

t0 = time.time()
wall = lambda: (time.time() - t0) / 60  # noqa: E731
KEEP, RETRACT, CAP = LD.KEEP, LD.RETRACT, LD.HARD_CAP
out = dict(rule="RULE_amend3.md", phases=[])


def state(arm):
    return json.load(open(os.path.join(R22, "results", f"state_{arm}.json")))


def ladder(arm, steps, cap):
    cmd = [PY, "-u", os.path.join(R22, "ladder.py"), "--mode", "arm", "--arm", arm, "--steps", str(steps),
           "--threads", "1", "--cpu-cap", str(cap)]
    r = subprocess.run(cmd, cwd=R22, capture_output=True, text=True)
    with open(os.path.join(HERE, "logs", f"ladder_{arm}.txt"), "a") as f:
        f.write(r.stdout + r.stderr)
    return r.returncode


def cls(rho):
    return "PAIRING-KEPT" if rho <= KEEP else "PAIRING-RETRACTED" if rho >= RETRACT else "ABL-CONFOUNDED"


def finish(v):
    out.update(verdict=v, wall_min=wall())
    json.dump(out, open(os.path.join(HERE, "results", "verdict_row4.json"), "w"), indent=1)
    print(f"VERDICT (rho_abl): {v}")
    sys.exit(0)


# ---------------- phase 1: rank5
ladder("rank5", CAP, 200)
s5 = state("rank5")
if s5["stop"] is None:
    finish("INCOMPLETE (rank5 cut by its CPU cap)")
e5, e5i, n5, fit5 = LD.endpoint(s5)
out["rank5"] = dict(stop=s5["stop"], steps=n5, eps=e5, eps_inf=e5i, fit=fit5)
print(f"rank5: {s5['stop']} after {n5} steps; eps {e5:.5e}, tail-extrapolated {e5i:.5e}")
if fit5 and fit5.get("unsettled"):
    finish("AMBIGUOUS (rank5 tail fit UNSETTLED at the hard cap; RULE_amend1 item 2)")
L5 = min(e5, e5i) if e5i > 0 else e5
target = KEEP * L5
out["target_eps_pcg"] = target

# ---------------- phase 2: pcg only as far as rho_abl needs
while True:
    sp = state("pcg")
    acc = [x for x in sp["steps"] if x.get("accepted")]
    ep = acc[-1]["eps_new"] if acc else (sp["E_start"] - LD.E0) / abs(LD.E0)
    out["pcg"] = dict(stop=sp["stop"], steps=len(acc), eps=ep)
    if ep <= target:
        out["pcg"]["stop_row4"] = "STOP-DOMINANCE"
        finish(f"PAIRING-KEPT by dominance (eps_pcg {ep:.4e} <= 0.60 x {L5:.4e}; rho_abl <= {ep / L5:.3f})")
    if sp["stop"] in ("STOP-RULE", "STOP-NOGAIN", "INVALID") or len(sp["steps"]) >= CAP:
        break
    if wall() + 10.5 > a.budget_min:
        finish("INCOMPLETE (row-4 budget reached in the pcg phase)")
    ladder("pcg", len(sp["steps"]) + 1, 12)
    if state("pcg")["steps"] == sp["steps"]:
        finish("INCOMPLETE (pcg call made no step)")

# pcg finished without reaching the dominance line: read rho_abl with the unchanged r22 rules
if sp["stop"] == "INVALID":
    finish("INVALID")
if len(sp["steps"]) >= CAP:
    sp["stop"] = "STOP-CAP"
ep, epi, npc, fitp = LD.endpoint(sp)
rho, rho_i = ep / e5, (epi / e5i if e5i > 0 else float("inf"))
c1, c2 = cls(rho), cls(rho_i)
bad = bool(fitp and fitp.get("unsettled"))
out.update(rho_abl=rho, rho_abl_inf=rho_i, cls_end=c1, cls_inf=c2, pcg_fit=fitp)
finish(c1 if (c1 == c2 and not bad) else f"AMBIGUOUS ({c1} at endpoints, {c2} extrapolated)")
