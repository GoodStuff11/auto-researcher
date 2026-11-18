#!/usr/bin/env python3
"""Diagnostic after envelope.py (not the pre-registered statistic): best monotone schedule A -> B -> C (N never
decreases, at most two switches, breakpoints on a 0.01 grid), which removes the bin-by-bin noise picking."""
import contextlib
import io
import json

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import envelope as E
out = {}
for frac in (0.80, 0.90, 0.95):
    tgt = frac * E.RP
    g = np.round(np.arange(0.05, tgt, 0.01), 3)
    best = (np.inf, None)
    for i, r1 in enumerate(g):
        for r2 in g[i:]:
            c = E.first("A", r1) + (E.first("B", r2) - E.first("B", r1)) + (E.first("C", tgt) - E.first("C", r2))
            if c < best[0]:
                best = (c, (float(r1), float(r2)))
    single = min(E.first(k, tgt) for k in "ABC")
    out[frac] = dict(mono=float(best[0]), breaks=best[1], gain=float(single / best[0]))
    print(frac, out[frac])
json.dump(out, open(E.os.path.join(E.H, "envelope_mono.json"), "w"), indent=1)
