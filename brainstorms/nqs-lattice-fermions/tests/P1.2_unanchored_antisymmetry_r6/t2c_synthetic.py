#!/usr/bin/env python3
"""Synthetic aggregate check for t2c_readout.verdicts: every rule branch once. Usage: python3 t2c_synthetic.py"""
import t2c_readout as C

BASE = dict(E0=-1, E0_stored=-1, E_ident=-.9, E_stored=-.9, dEs_K=0.010, dEs_C=0.012)
CASES = {"base carries": dict(dEs_base=0.0102), "base mixed": dict(dEs_base=0.0110), "reframe": dict(dEs_base=0.013),
         "absfroz ok": dict(dEs_absfroz=0.0104), "absfroz partial": dict(dEs_absfroz=0.011),
         "absfroz lost": dict(dEs_absfroz=0.0125), "gvote usable": dict(dEs_gvote=0.0085),
         "gvote unresolved": dict(dEs_gvote=0.0095), "gvote harmful": dict(dEs_gvote=0.0105),
         "invalid": dict(E_ident=-0.8, dEs_base=0.0102)}
for name, c in CASES.items():
    r = dict(BASE); r.update(c)
    print(f"synthetic {name:17s}: {C.verdicts(r)}")
