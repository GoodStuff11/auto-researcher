#!/usr/bin/env python3
"""Check of the --mode report decision logic on synthetic JSONs (no physics).
Usage: python check_report_logic.py   (writes into a temporary directory; prints the verdicts of
four scenarios: PASS, FAIL on accuracy, NO-TARGET, FAIL on cost)."""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import micro_test as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def scenario(name, eps, cost_ok=True):
    d = tempfile.mkdtemp()
    tag = "T"
    for arm, e in eps.items():
        for s in range(3):
            r = dict(ansatz=arm, seed=s, nparam=2000, iters=2000, capped=False, E=0.0, eps=e * (1 + 0.01 * s),
                     S2=1e-16 if arm != "nnbf_G" else 0.3, Cs={"1,0": 0.1}, Cs_exact={"1,0": 0.1},
                     Pd={"2,0": 0.0}, Pd_exact={"2,0": 0.0}, sign=dict(dE_sign=1e-5), posthoc=dict(eps=e))
            json.dump(r, open(os.path.join(d, f"{tag}_{arm}_seed{s}.json"), "w"))
    C = json.load(open(os.path.join(HERE, "results", "cost.json")))
    if not cost_ok:
        C["slopes"]["sqrvb"] = C["slopes"]["scf"] = 9.0
    json.dump(C, open(os.path.join(d, "cost.json"), "w"))
    print(f"--- scenario: {name}")
    M.report(M.parse_args(["--mode", "report", "--outdir", d, "--tag", tag]))
    shutil.rmtree(d)


base = dict(sqrvb=1e-4, sqrvb0=1e-3, scf=1e-4, nnbf_SG=1e-3, nnbf_G=2e-3, che=5e-4, foil=1e-2)
scenario("routes beat every control", base)
scenario("routes worse than projected control", dict(base, sqrvb=2e-3, scf=2e-3))
scenario("projected control near-exact", dict(base, nnbf_SG=1e-5))
scenario("cost gate violated", base, cost_ok=False)
