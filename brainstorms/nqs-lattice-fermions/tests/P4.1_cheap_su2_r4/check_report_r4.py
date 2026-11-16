"""Check r4_test.py --mode report on synthetic results (no options). Scenarios: PASS/PASS,
FAIL(accuracy)/FAIL, premise not met (NO-TARGET), cost FAIL."""
import json, os, shutil, tempfile, contextlib, io
import r4_test as T

def scen(name, eps, cost_k2=1.0, cost_che=3.0, agp=(1e-3, 1e-4)):
    d = tempfile.mkdtemp()
    a = T.parse_args(["--mode", "report", "--Ne", "10", "--outdir", d])
    tag = T.tag_of(a)
    for k, e in eps.items():
        for s in range(3):
            json.dump(dict(ansatz=k, seed=s, P=2600, eps=e * (1 + 0.01 * s), S2=1e-16, capped=False),
                      open(os.path.join(d, f"{tag}_w16_{k}_seed{s}.json"), "w"))
    json.dump(dict(rows=[dict(kagp2=cost_k2, che_closed=cost_che)]), open(os.path.join(d, "cost_r4.json"), "w"))
    json.dump(dict(tocchio1=dict(eps=agp[0]), tocchio2=dict(eps=agp[1]), puhf=dict(eps=1e-5)),
              open(os.path.join(d, f"agp_{tag}.json"), "w"))
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        T.report(a)
    print(f"--- {name}\n" + "\n".join(l for l in f.getvalue().splitlines() if "Route" in l))
    shutil.rmtree(d)

scen("expect 4.2 PASS, 4.1 PASS", dict(tocchio=1e-3, kagp2=5e-4, che=8e-4, sfb=4e-4, nnbf_SG=3e-4))
scen("expect 4.2 FAIL(accuracy), 4.1 FAIL", dict(tocchio=1e-3, kagp2=9.5e-4, che=8e-4, sfb=8e-4, nnbf_SG=3e-4))
scen("expect 4.2 NO-TARGET", dict(tocchio=1e-3, kagp2=5e-4, che=8e-4, sfb=4e-4, nnbf_SG=3e-4), agp=(5e-5, 1e-5))
scen("expect 4.2 FAIL(cost)", dict(tocchio=1e-3, kagp2=5e-4, che=8e-4, sfb=4e-4, nnbf_SG=3e-4), cost_k2=4.0)
