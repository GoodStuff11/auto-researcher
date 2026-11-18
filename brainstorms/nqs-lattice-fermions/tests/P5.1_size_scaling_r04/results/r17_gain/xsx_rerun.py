"""Round-17 Major-4 check: rerun R51.8's 5x2 exact arm (deterministic) logging per step sqrt(x^T S x) and f.x."""
import sys, json, os, numpy as np
D = '/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04'
sys.path.insert(0, D)
import ns_run as NS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xsx_log.jsonl')
orig = NS.T.d_shift
def logged(w, U, f, tau=0.005):
    d, ml = orig(w, U, f, tau)
    x = U @ ((U.T @ f) / (w + 1e-3))
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(dict(ml=ml, fx=float(f @ x), fd=float(f @ d))) + '\n')
    return d, ml
NS.T.d_shift = logged
NS.main()
