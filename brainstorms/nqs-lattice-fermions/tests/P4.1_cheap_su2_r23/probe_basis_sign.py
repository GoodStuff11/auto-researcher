"""P4.1 r23 pre-run probe (no tree-arm training): how much sign structure does the exact (+,+) ground state carry
in the coupling-tree basis? eps of |psi0| (all signs +), psi0-weight on negative states, and the number of states
carrying 99% of the weight. Informs the VOI prediction only; decides nothing. Writes results_probe/basis_sign.json."""
import sys, json, os
sys.argv = [sys.argv[0], "--mode", "check"]
import tree_nqs as T
import numpy as np
ctx = T.Ctx()
Eg, g0 = T.ground(ctx)
w = ctx.size * g0 * g0
if np.sum(w[g0 > 0]) < 0.5:
    g0 = -g0
o = np.argsort(-w); cw = np.cumsum(w[o])
out = dict(E_ground=Eg, vs_E0=abs(Eg - T.E0), eps_allplus=T.eps_of(ctx, np.abs(g0)),
           w_negative=float(w[g0 < 0].sum()), n99=int(np.searchsorted(cw, 0.99)) + 1, nrep=len(ctx.reps))
print(out)
json.dump(out, open(os.path.join(T.HERE, "results_probe/basis_sign.json"), "w"), indent=1)
