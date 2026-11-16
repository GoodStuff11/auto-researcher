# one-off patch (amendment 2026-11-16): add the My-sector penalty option to dmrg_ref.jl
import os
fn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dmrg_ref.jl")
s = open(fn).read()
rep = [
('"cutoff" => 1e-12, "threads" => 16,', '"cutoff" => 1e-12, "threads" => 16, "sector" => 0, "penalty" => 10.0,'),
('''    fn = joinpath(o["outdir"], @sprintf("dmrg_S%dx%d_N%d_Sz2%d_U%g_tp%g.json", Lx, Ly, Ne, Sz2, o["U"], o["tp"]))''',
'''    sec = o["sector"]
    stag = sec == 0 ? "" : (sec > 0 ? "_Myp" : "_Mym")
    fn = joinpath(o["outdir"], @sprintf("dmrg_S%dx%d_N%d_Sz2%d%s_U%g_tp%g.json", Lx, Ly, Ne, Sz2, stag, o["U"], o["tp"]))'''),
('''    H = hubbard_mpo(sites, Lx, Ly, o["U"], o["tp"])
''', '''    H0 = hubbard_mpo(sites, Lx, Ly, o["U"], o["tp"])
    H = H0
    if sec != 0  # H + penalty * (1 - sec*My)/2: DMRG restricted (variationally) to the My = sec sector
        My = my_mpo(sites, Lx, Ly)
        Id = MPO(sites, "Id")
        H = +(H0, (0.5 * o["penalty"]) * Id, (-0.5 * o["penalty"] * sec) * My; cutoff=1e-13)
    end
'''),
('''    ci, cy, cx = reflection_characters(psi, sites, Lx, Ly, Ne)''', '''    ci, cy, cx = reflection_characters(psi, sites, Lx, Ly, Ne)
    EH = real(inner(psi', H0, psi) / inner(psi, psi))
    @printf("sector My = %+d: <H> of final MPS = %.10f (penalised energy %.10f)\\n", sec, EH, stages[end]["energy"])'''),
('''\\"chi_Mx\\": $cx, ''', '''\\"chi_Mx\\": $cx, \\"sector_My\\": $sec, \\"penalty\\": $(o["penalty"]), \\"E_H_final\\": $EH, '''),
('''function fswap_gate(s1, s2)''', '''"""
My (y -> Ly-1-y in every column) as an MPO: product over columns x of the fermionic transposition of
sites a = x*Ly+1 and c = x*Ly+Ly, for each spin s: S_s = 1 - n_as - n_cs + c+_as c_cs + c+_cs c_as
(= (-1)^{n_d}, d = (c_a - c_c)/sqrt2; maps c_a <-> c_c and fixes the vacuum). Ly = 3 only.
"""
function my_mpo(sites, Lx, Ly)
    @assert Ly == 3
    M = MPO(sites, "Id")
    for x in 0:Lx-1
        a, c = x * Ly + 1, x * Ly + Ly
        for (n, cd, cc) in (("Nup", "Cdagup", "Cup"), ("Ndn", "Cdagdn", "Cdn"))
            os = OpSum()
            os += 1.0, "Id", a
            os += -1.0, n, a
            os += -1.0, n, c
            os += 1.0, cd, a, cc, c
            os += 1.0, cd, c, cc, a
            M = apply(MPO(os, sites), M; cutoff=1e-14)
        end
    end
    return M
end

function fswap_gate(s1, s2)'''),
('''  --threads=INT     BLAS threads (default 16).''', '''  --threads=INT     BLAS threads (default 16).
  --sector=INT      0 (default): plain DMRG (lowest state of the (N_e, S^z) sector). +1 or -1: DMRG of
                    H + penalty * (1 - sector*My)/2, i.e. the lowest state with y-mirror character My = sector
                    (amendment 2026-11-16: reference for the (+,+)-projected VMC arms). Output file gets
                    _Myp / _Mym; <H> of the final MPS and its characters are recorded.
  --penalty=FLOAT   penalty weight (default 10.0, larger than any low-lying gap).'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(fn, "w").write(s)
print("patched")
