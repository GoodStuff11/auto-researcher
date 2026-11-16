# one-off patch (amendment 2026-11-16, second): sector-resolved (+,+) references in scaling_report.py
import os
fn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scaling_report.py")
s = open(fn).read()
i = s.index("def load_refs(dmrgdir):")
j = s.index("def spin_check(dmrgdir):")
new = '''def select_ref(N, plain, myp, mym):
    """Reference for the (+,+)-projected arms at size N (amendment 2026-11-16, second).
    plain / myp / mym: DMRG result dicts (2S^z = 0) of the plain run and the My = +1 / My = -1 penalty runs, or None.
    Returns (E_ref, sigma_ref, status, notes); status "OK" or "AMBIGUOUS-REFERENCE"."""
    notes = []
    if myp is not None and myp.get("chi_inv", 0) >= 0.95 and myp.get("chi_My", 0) >= 0.95:
        E, sig = myp["E_extrap"], myp["E_extrap_err"]
        notes.append(f"E(+,+) = {E:.6f} +- {sig:.1e} from the My=+1 sector run")
    elif myp is not None:
        return None, None, "AMBIGUOUS-REFERENCE", [f"My=+1 sector run is not (+,+): <Inv> {myp.get('chi_inv')}, <My> {myp.get('chi_My')}"]
    elif plain is not None and plain.get("chi_inv", 0) >= 0.95 and plain.get("chi_My", 0) >= 0.95:
        E, sig = plain["E_extrap"], plain["E_extrap_err"]
        notes.append(f"E(+,+) = {E:.6f} +- {sig:.1e} from the plain run (clean characters)")
    else:
        return None, None, "AMBIGUOUS-REFERENCE", ["no sector-resolved E(+,+) and the plain run has |<My>| < 0.95 or a wrong sign"]
    if mym is not None:
        gap = mym["E_extrap"] - E
        err = mym["E_extrap_err"] + sig
        notes.append(f"sector gap E(My=-1) - E(+,+) = {gap:+.5f} +- {err:.5f}"
                     + (" -> ground state in the My=-1 sector (scoring stays within (+,+))" if gap < -2 * err else ""))
    return E, sig, "OK", notes


def load_refs(dmrgdir):
    refs = {12: (E0_12, 0.0, "OK")}
    notes = {12: ["ED, (+,+) ground state (logs/ed_irrep_4x3_output.txt)"]}
    runs = {}
    for fn in glob.glob(os.path.join(dmrgdir, "dmrg_S*_Sz20*.json")):
        d = json.load(open(fn))
        if d["Ly"] != 3 or d["N"] not in LADDER or d["N"] == 12:
            continue
        key = {1: "myp", -1: "mym"}.get(int(d.get("sector_My", 0)), "plain")
        runs.setdefault(d["N"], {})[key] = d
    for N, r in runs.items():
        E, sig, st, nt = select_ref(N, r.get("plain"), r.get("myp"), r.get("mym"))
        refs[N] = (E, sig, st)
        notes[N] = nt
    REF_NOTES.update(notes)
    return refs


'''
s = s[:i] + new + s[j:]
s = s.replace("IRREP_BAD = set()\n", "REF_NOTES = {}\n")
s = s.replace('''    if IRREP_BAD:
        return "INCOMPLETE", [f"irrep mismatch (DMRG character < 0) at N = {sorted(IRREP_BAD)}: rerun those sizes in the corrected irrep"]
''', "")
a = '''        if "nnbf_wall" not in table[N]:'''
assert s.count(a) == 1
s = s.replace(a, '''        if len(refs[N]) > 2 and refs[N][2] != "OK":
            return "AMBIGUOUS-REFERENCE", [f"N={N}: no valid (+,+) reference ({refs[N][2]})"]
''' + a)
s = s.replace('''    for fn in sorted(glob.glob(os.path.join(dmrgdir, "dmrg_S*_Sz20_*.json"))):''',
              '''    for fn in sorted(f for f in glob.glob(os.path.join(dmrgdir, "dmrg_S*_Sz20_*.json")) if "_My" not in f):''')
s = s.replace('''        print(f"  N={N}: " + (f"E_ref = {refs[N][0]:.6f} +- {refs[N][1]:.1e}" if N in refs else "missing"))''',
              '''        print(f"  N={N}: " + (f"{refs[N][2]}" if N in refs else "missing") + "; " + "; ".join(REF_NOTES.get(N, [])))''')
s = s.replace('''        if N not in runs or N not in refs:''', '''        if N not in runs or N not in refs or refs[N][2] != "OK":''')
# check scenarios for select_ref
a = '''    bad = 0
'''
assert s.count(a) == 1
s = s.replace(a, '''    bad = 0
    cl = dict(E_extrap=-30.0, E_extrap_err=1e-4, chi_inv=1.0, chi_My=1.0)
    mixed = dict(cl, chi_My=0.63)
    lowm = dict(cl, E_extrap=-30.01, chi_My=-1.0, sector_My=-1)
    highm = dict(cl, E_extrap=-29.95, chi_My=-1.0, sector_My=-1)
    ref_cases = [
        ("sector run (+,+) available", (36, mixed, dict(cl, sector_My=1), highm), "OK", None),
        ("mixed plain run, no sector run", (48, mixed, None, None), "AMBIGUOUS-REFERENCE", None),
        ("clean plain run, no sector run", (24, cl, None, None), "OK", None),
        ("My=+1 run landed in (-,+)", (48, mixed, dict(cl, chi_inv=-1.0, sector_My=1), None), "AMBIGUOUS-REFERENCE", None),
        ("ground state in My=-1 sector", (48, mixed, dict(cl, sector_My=1), lowm), "OK", "ground state in the My=-1"),
    ]
    for name, args, want, flag in ref_cases:
        E, sig, st, nt = select_ref(*args)
        ok = st == want and (flag is None or any(flag in x for x in nt))
        bad += not ok
        print(f"[{'ok' if ok else 'WRONG'}] reference: {name}: {st} (intended {want}) " + "; ".join(nt))
''')
s = s.replace('''    print(f"CHECK {'PASS' if bad == 0 else 'FAIL'}: {len(cases) - bad}/{len(cases)} scenarios give the intended verdict")''',
              '''    n = len(cases) + len(ref_cases)
    print(f"CHECK {'PASS' if bad == 0 else 'FAIL'}: {n - bad}/{n} scenarios give the intended verdict")''')
s = s.replace('''  Reference validity at N: sigma_ref(N)''', '''  Reference (amendment 2026-11-16, second): E_ref(N) is the (+,+)-SECTOR energy E(+,+): the My = +1
  penalty DMRG run (dmrg_ref.jl --sector=1) with <Inv>, <My> >= 0.95; or the plain run only if its
  characters are clean (>= 0.95). Otherwise the size is AMBIGUOUS-REFERENCE (not scored). The sector
  gap E(My=-1) - E(+,+) is printed; if negative beyond 2 sigma the size is marked "ground state in the
  My=-1 sector" (scoring stays within (+,+)).
  Reference validity at N: sigma_ref(N)''')
open(fn, "w").write(s)
print("patched")
