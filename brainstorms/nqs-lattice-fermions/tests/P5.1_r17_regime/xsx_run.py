"""R51.32: ns_run.py unchanged, plus a per-step log of the exact step's metric norm (round-17 xsx_rerun.py pattern).
Usage: XSX_LOG=<file.jsonl> python3 xsx_run.py <ns_run.py arguments>. Wraps trsr.d_shift: returns its output
unchanged (round 17: energies identical to ns_run.py, max diff 0) and appends ml = sqrt(x^T S x), f.x, f.d per call.
On resume ns_run.py does not repeat steps, so the log has one line per executed step."""
import json
import os
import sys
D = '/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04'
sys.path.insert(0, D)
import ns_run as NS  # noqa: E402  (sets thread variables before numpy, L63)
OUT = os.environ['XSX_LOG']
orig = NS.T.d_shift


def logged(w, U, f, tau=0.005):
    d, ml = orig(w, U, f, tau)
    x = U @ ((U.T @ f) / (w + 1e-3))
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(dict(ml=ml, fx=float(f @ x), fd=float(f @ d), tau=tau)) + '\n')
    return d, ml


NS.T.d_shift = logged
NS.main()
