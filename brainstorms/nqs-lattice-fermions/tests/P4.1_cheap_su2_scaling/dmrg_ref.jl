"""
dmrg_ref.jl — DMRG reference energies for the P4.1 size-scaling ladder (ITensorMPS, U(1)xU(1) QNs).

Model (identical to vmc.py): Hubbard on an Lx x Ly strip, OPEN along x and y, site index
i = x*Ly + y + 1 (MPS order = this order: column by column, width Ly), hopping t = 1 on
nearest-neighbour bonds and t' on the diagonal bonds, H = -sum_{<ij>,s} t_ij (c+_is c_js + h.c.)
+ U sum_i n_iup n_idn. Electrons N_e, magnetisation 2S^z = N_up - N_dn.

Usage (from this folder):
    ~/.juliaup/bin/julia --project=dmrg_env -t 1 dmrg_ref.jl [options]
    (BLAS threads are set from --threads.)

Options (all optional; --key=value form):
  --Lx=INT          strip length (default 4). Ladder: 4, 8, 12, 16.
  --Ly=INT          strip width (default 3).
  --Ne=INT          electrons (default: 5/6 * Lx * Ly rounded to the nearest even integer).
  --Sz2=INT         2 S^z = N_up - N_dn, must have the parity of Ne (default 0). Use 0 and 2 to
                    determine the ground-state spin: if E(2S^z=2) > E(2S^z=0) beyond both error
                    bars, the ground state is a singlet (an S >= 1 multiplet has an S^z = 1 member).
  --U=FLOAT         on-site repulsion (default 8.0).
  --tp=FLOAT        diagonal hopping t' (default -0.25).
  --maxdims=LIST    comma-separated bond-dimension schedule (default "100,200,400,800,1200,1600,2000").
                    Every entry is one stage of --sweeps sweeps at that maximal bond dimension.
  --sweeps=INT      sweeps per stage (default 4); the first stage gets 2x as many.
  --noise=FLOAT     noise term for stages whose bond dimension is below the last three entries
                    (default 1e-6); the last three stages run with noise 0 so that the recorded
                    (energy, truncation error) pairs are variational and extrapolable.
  --cutoff=FLOAT    SVD cutoff (default 1e-12).
  --threads=INT     BLAS threads (default 16).
  --sector=INT      0 (default): plain DMRG (lowest state of the (N_e, S^z) sector). +1 or -1: DMRG of
                    H + penalty * (1 - sector*My)/2, i.e. the lowest state with y-mirror character My = sector
                    (amendment 2026-11-16: reference for the (+,+)-projected VMC arms). Output file gets
                    _Myp / _Mym; <H> of the final MPS and its characters are recorded.
  --penalty=FLOAT   penalty weight (default 10.0, larger than any low-lying gap).
  --write_above=INT environment tensors are written to disk (ITensorMPS write_when_maxdim_exceeds)
                    for stages with maxdim above this value (default 1000000 = never), to stay within memory.
  --write_path=PATH directory for those files (default: Julia tempdir(), i.e. node-local /tmp).
  --outdir=PATH     output folder (default <script dir>/dmrg).
  --overwrite       recompute even if the output JSON exists (default: skip if it exists).

Output: <outdir>/dmrg_S<Lx>x<Ly>_N<Ne>_Sz2<Sz2>_U<U>_tp<tp>.json with fields
  stages: list of {maxdim, energy (last sweep), truncerr (max discarded weight of last sweep),
          maxlinkdim, seconds}
  E_extrap, E_extrap_err: linear fit E(truncerr) over the last three (noise-free) stages,
          intercept at truncerr = 0; error = |E(last stage) - E_extrap| (conservative: the whole
          extrapolation distance) — see the leaf doc for how this enters the validity rule.
  chi_inv, chi_My, chi_Mx: reflection characters <psi|g|psi>/<psi|psi> of the final MPS (g = inversion
          = Mx My, y-mirror, x-mirror; see reflection_characters). A value near +-1 fixes the irrep.
  plus all model parameters.
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "logging.jl"))

function parse_cli(args)
    o = Dict{String,Any}("Lx" => 4, "Ly" => 3, "Ne" => -1, "Sz2" => 0, "U" => 8.0, "tp" => -0.25,
        "maxdims" => "100,200,400,800,1200,1600,2000", "sweeps" => 4, "noise" => 1e-6,
        "cutoff" => 1e-12, "threads" => 16, "sector" => 0, "penalty" => 10.0, "write_above" => 1000000, "write_path" => tempdir(), "outdir" => joinpath(@__DIR__, "dmrg"), "overwrite" => false)
    for a in args
        if a == "--overwrite"
            o["overwrite"] = true
            continue
        end
        startswith(a, "--") && occursin("=", a) || error("bad argument $a (use --key=value)")
        k, v = split(a[3:end], "=", limit=2)
        haskey(o, k) || error("unknown option --$k")
        o[k] = o[k] isa Int ? parse(Int, v) : o[k] isa Float64 ? parse(Float64, v) : String(v)
    end
    N = o["Lx"] * o["Ly"]
    o["Ne"] < 0 && (o["Ne"] = 2 * round(Int, 5 * N / 12))
    iseven(o["Ne"] - o["Sz2"]) || error("Sz2 must have the parity of Ne")
    o["maxdims"] = parse.(Int, split(o["maxdims"], ","))
    return o
end

function hubbard_mpo(sites, Lx, Ly, U, tp)
    os = OpSum()
    idx(x, y) = x * Ly + y + 1
    for x in 0:Lx-1, y in 0:Ly-1
        i = idx(x, y)
        for (dx, dy, t) in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, tp), (1, -1, tp))
            X, Y = x + dx, y + dy
            (0 <= X < Lx && 0 <= Y < Ly) || continue
            j = idx(X, Y)
            for (cd, c) in (("Cdagup", "Cup"), ("Cdagdn", "Cdn"))
                os += -t, cd, i, c, j
                os += -t, cd, j, c, i
            end
        end
        os += U, "Nupdn", i
    end
    return MPO(os, sites)
end

function initial_state(N, Ne, Sz2)
    nh = N - Ne
    holes = Set(round(Int, (k - 0.5) * N / nh) + 1 for k in 1:nh)
    st = String[]
    for i in 1:N
        i in holes ? push!(st, "Emp") : push!(st, isodd(count(!=("Emp"), st)) ? "Dn" : "Up")
    end
    for _ in 1:(Sz2 ÷ 2)
        st[findlast(==("Dn"), st)] = "Up"
    end
    @assert count(==("Up"), st) == (Ne + Sz2) ÷ 2 && count(==("Dn"), st) == (Ne - Sz2) ÷ 2
    return st
end

"""
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

function fswap_gate(s1, s2)
    # fermionic swap of two adjacent Electron sites: |a,b> -> (-1)^{n(a) n(b)} |b,a>
    nf = (0, 1, 1, 2)  # Emp, Up, Dn, UpDn
    G = ITensor(prime(s1), prime(s2), dag(s1), dag(s2))
    for a in 1:4, b in 1:4
        G[prime(s1) => b, prime(s2) => a, s1 => a, s2 => b] = (-1.0)^(nf[a] * nf[b])
    end
    return G
end

"""
Reflection characters <psi|g|psi>/<psi|psi> for g = Inv (= Mx My, the full reversal of the site
order i -> N+1-i) and My (y -> Ly-1-y inside each column); Mx = Inv * My. g acts as c_{i,s} -> c_{g(i),s}.
Inv: reversed MPS times the diagonal sign (-1)^{(Nf^2 - Nf)/2 - D} (reordering of the site blocks;
D = number of doubly occupied sites). My: fermionic adjacent swaps (Ly = 3: sites 1 and 3 of each column).
"""
function reflection_characters(psi, sites, Lx, Ly, Nf; cutoff=1e-10, maxdim=10000)
    N = length(psi)
    nrm = inner(psi, psi)
    T = [replaceinds(psi[N + 1 - k], [sites[N + 1 - k]], [sites[k]]) for k in 1:N]
    pr = MPS(T)
    for k in 1:N
        Dk = op("Id", sites[k]) - 2 * op("Nupdn", sites[k])
        pr[k] = noprime(pr[k] * Dk)
    end
    sgn = iseven((Nf * Nf - Nf) ÷ 2) ? 1.0 : -1.0
    chi_inv = sgn * inner(psi, pr) / nrm
    @assert Ly == 3
    pm = copy(psi)
    for x in 0:Lx-1
        a = x * Ly + 1
        for (i, j) in ((a, a + 1), (a + 1, a + 2), (a, a + 1))
            pm = apply(fswap_gate(sites[i], sites[j]), pm; cutoff=cutoff, maxdim=maxdim)
        end
    end
    chi_my = inner(psi, pm) / nrm
    return chi_inv, chi_my, chi_inv * chi_my
end

function run_dmrg(o)
    BLAS.set_num_threads(o["threads"])
    Lx, Ly, Ne, Sz2 = o["Lx"], o["Ly"], o["Ne"], o["Sz2"]
    N = Lx * Ly
    mkpath(o["outdir"])
    sec = o["sector"]
    stag = sec == 0 ? "" : (sec > 0 ? "_Myp" : "_Mym")
    fn = joinpath(o["outdir"], @sprintf("dmrg_S%dx%d_N%d_Sz2%d%s_U%g_tp%g.json", Lx, Ly, Ne, Sz2, stag, o["U"], o["tp"]))
    if isfile(fn) && !o["overwrite"]
        println("$fn exists; skipping")
        return
    end
    sites = siteinds("Electron", N; conserve_qns=true)
    H0 = hubbard_mpo(sites, Lx, Ly, o["U"], o["tp"])
    H = H0
    if sec != 0  # H + penalty * (1 - sec*My)/2: DMRG restricted (variationally) to the My = sec sector
        My = my_mpo(sites, Lx, Ly)
        Id = MPO(sites, "Id")
        H = +(H0, (0.5 * o["penalty"]) * Id, (-0.5 * o["penalty"] * sec) * My; cutoff=1e-13)
    end
    st = initial_state(N, Ne, Sz2)
    psi = random_mps(sites, st; linkdims=10)
    md = o["maxdims"]
    stages = Any[]
    nst = length(md)
    for (s, D) in enumerate(md)
        nsw = s == 1 ? 2 * o["sweeps"] : o["sweeps"]
        nz = s <= nst - 3 ? o["noise"] : 0.0
        obs = DMRGObserver()
        t0 = time()
        E, psi = dmrg(H, psi; nsweeps=nsw, maxdim=D, cutoff=o["cutoff"], noise=nz, observer=obs, outputlevel=0,
            write_when_maxdim_exceeds=o["write_above"], write_path=o["write_path"])
        tr = truncerrors(obs)[end]
        dt = time() - t0
        push!(stages, Dict("maxdim" => D, "energy" => E, "truncerr" => tr, "maxlinkdim" => maxlinkdim(psi),
            "seconds" => dt, "noise" => nz, "sweep_energies" => energies(obs)))
        @printf("[%s] maxdim %5d  E = %.10f  truncerr = %.3e  maxlinkdim %d  noise %.0e  (%.0f s)\n",
            string(Dates.now()), D, E, tr, maxlinkdim(psi), nz, dt)
        flush(stdout)
    end
    nN = flux(psi)
    last3 = stages[end-2:end]
    xs = [s["truncerr"] for s in last3]
    ys = [s["energy"] for s in last3]
    A = hcat(ones(3), xs)
    coef = A \ ys
    Eex = coef[1]
    err = abs(stages[end]["energy"] - Eex)
    @printf("E_extrap = %.10f +- %.2e  (last stage %.10f; slope %.3e); N_e check %s\n", Eex, err,
        stages[end]["energy"], coef[2], string(nN))
    ci, cy, cx = reflection_characters(psi, sites, Lx, Ly, Ne)
    EH = real(inner(psi', H0, psi) / inner(psi, psi))
    @printf("sector My = %+d: <H> of final MPS = %.10f (penalised energy %.10f)\n", sec, EH, stages[end]["energy"])
    @printf("reflection characters: <Inv> = %+.6f  <My> = %+.6f  <Mx> = <Inv><My> = %+.6f\n", ci, cy, cx)
    open(fn, "w") do f
        js = "{\"Lx\": $Lx, \"Ly\": $Ly, \"N\": $N, \"Ne\": $Ne, \"Sz2\": $Sz2, \"U\": $(o["U"]), \"tp\": $(o["tp"]), " *
             "\"E_extrap\": $Eex, \"E_extrap_err\": $err, \"E_last\": $(stages[end]["energy"]), \"chi_inv\": $ci, \"chi_My\": $cy, \"chi_Mx\": $cx, \"sector_My\": $sec, \"penalty\": $(o["penalty"]), \"E_H_final\": $EH, \"stages\": [" *
             join([@sprintf("{\"maxdim\": %d, \"energy\": %.12f, \"truncerr\": %.6e, \"maxlinkdim\": %d, \"seconds\": %.1f, \"noise\": %.1e}",
                 s["maxdim"], s["energy"], s["truncerr"], s["maxlinkdim"], s["seconds"], s["noise"]) for s in stages], ", ") * "]}"
        write(f, js)
    end
    println("wrote $fn")
end

function (@main)(ARGS)
    o = parse_cli(ARGS)
    log_path = make_log_path(@__DIR__, "dmrg_ref")
    with_logging(log_path) do
        run_dmrg(o)
    end
end
