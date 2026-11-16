# one-off patch: spin check compares the 2S^z=2 run with the lowest 2S^z=0 energy over plain and sector runs
import os
fn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scaling_report.py")
s = open(fn).read()
i = s.index("def spin_check(dmrgdir):")
j = s.index("def report(a):")
new = '''def spin_check(dmrgdir):
    """E(S^z=1) - min over the S^z=0 runs (plain, My=+1, My=-1) at each N; > 2 sigma means the
    lowest S^z=0 state is not part of an S >= 1 multiplet, i.e. a singlet ground state."""
    out = []
    z0, z2 = {}, {}
    for fn in glob.glob(os.path.join(dmrgdir, "dmrg_S*.json")):
        d = json.load(open(fn))
        tgt = z0 if d["Sz2"] == 0 else z2
        if d["Sz2"] in (0, 2) and (d["N"] not in tgt or d["E_extrap"] < tgt[d["N"]]["E_extrap"]):
            tgt[d["N"]] = d
    for N in sorted(set(z0) & set(z2)):
        gap = z2[N]["E_extrap"] - z0[N]["E_extrap"]
        err = z0[N]["E_extrap_err"] + z2[N]["E_extrap_err"]
        out.append(f"N={N}: E(Sz=1) - min E(Sz=0) = {gap:+.5f} +- {err:.5f} -> "
                   f"{'singlet ground state' if gap > 2 * err else 'NOT RESOLVED / not singlet'}")
    return out


'''
s = s[:i] + new + s[j:]
open(fn, "w").write(s)
print("patched")
