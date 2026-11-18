"""
r47_dmrg.jl — R47 T1 teacher-pipeline rehearsal at 12 sites (RULE_T1.md, "Teacher-correction pipeline check").

Reruns the stored 4x3 DMRG (../P4.1_cheap_su2_scaling/dmrg_ref.jl settings: ITensorMPS, U(1)xU(1) QNs, open 4x3,
N_e = 10, 2S^z = 0, U = 8, t' = -0.25, plain sector, maxdims 100,200,400,600,800,1000, 4 sweeps per stage (first stage
8), noise 1e-6 except the last three stages, cutoff 1e-12), then
  1. dumps the final MPS (Julia Serialization) to <outdir>/mps.jls,
  2. computes <S^2> as an MPO (sum_ij Sz_i Sz_j + (S+_i S-_j + S-_i S+_j)/2),
  3. contracts the MPS to a dense tensor (12 sites only) and writes every entry with N_up = N_dn = 5 as three raw
     little-endian arrays: sector_ubits.bin / sector_dbits.bin (Int64; bit j-1 = occupation of site j, j = x*3+y+1)
     and sector_amp.bin (Float64, ITensor's own basis convention; r47_t1.py --mode dmrgcheck maps it to the python
     basis and fixes the fermion sign convention by the overlap with the exact ground state),
  4. draws --nsamples perfect samples (ITensorMPS sample!) and records each sample's amplitude obtained by contracting
     the MPS with one-hot states (samples.json: ubits, dbits, amp),
  5. writes dmrg_meta.json: S2_mpo, E_last, maxdims, seconds, norm, maxlinkdim.
Model and site order are identical to dmrg_ref.jl (hubbard_mpo / initial_state copied, not modified).

Usage (from this folder):
    ~/.juliaup/bin/julia --project=../P4.1_cheap_su2_scaling/dmrg_env -t 1 r47_dmrg.jl [options]

Options (all optional; --key=value form):
  --maxdims=LIST    comma-separated bond-dimension schedule (default "100,200,400,600,800,1000" = the stored run).
                    Smaller lists (e.g. "50,100,200") are for local smoke tests only.
  --sweeps=INT      sweeps per stage (default 4; the first stage gets 2x as many).
  --noise=FLOAT     noise for all but the last three stages (default 1e-6; with fewer than four stages, none).
  --cutoff=FLOAT    SVD cutoff (default 1e-12).
  --threads=INT     BLAS threads (default 1).
  --nsamples=INT    number of perfect samples (default 1024).
  --seed=INT        RNG seed for the samples (default 7).
  --outdir=PATH     output folder (default <script dir>/dmrg_out). Skips (prints and exits) if dmrg_meta.json exists
                    there, unless --overwrite.
  --overwrite       recompute even if the output exists.
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Printf
using Random
using Serialization

include(joinpath(@__DIR__, "..", "P4.1_cheap_su2_scaling", "logging.jl"))

const LX, LY, NE = 4, 3, 10

function parse_cli(args)
    o = Dict{String,Any}("maxdims" => "100,200,400,600,800,1000", "sweeps" => 4, "noise" => 1e-6, "cutoff" => 1e-12,
        "threads" => 1, "nsamples" => 1024, "seed" => 7, "outdir" => joinpath(@__DIR__, "dmrg_out"), "overwrite" => false)
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
    o["maxdims"] = parse.(Int, split(o["maxdims"], ","))
    return o
end

function hubbard_mpo(sites, Lx, Ly, U, tp)  # copy of dmrg_ref.jl
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

function initial_state(N, Ne, Sz2)  # copy of dmrg_ref.jl
    nh = N - Ne
    holes = Set(round(Int, (k - 0.5) * N / nh) + 1 for k in 1:nh)
    st = String[]
    for i in 1:N
        i in holes ? push!(st, "Emp") : push!(st, isodd(count(!=("Emp"), st)) ? "Dn" : "Up")
    end
    return st
end

function s2_mpo(sites)
    N = length(sites)
    os = OpSum()
    for i in 1:N, j in 1:N
        os += 1.0, "Sz", i, "Sz", j
        os += 0.5, "S+", i, "S-", j
        os += 0.5, "S-", i, "S+", j
    end
    return MPO(os, sites)
end

bits(v, set) = sum(Int64(1) << (j - 1) for j in eachindex(v) if v[j] in set; init=Int64(0))

function run_rehearsal(o)
    BLAS.set_num_threads(o["threads"])
    out = o["outdir"]
    mkpath(out)
    if isfile(joinpath(out, "dmrg_meta.json")) && !o["overwrite"]
        println("$(joinpath(out, "dmrg_meta.json")) exists; skipping")
        return
    end
    N = LX * LY
    sites = siteinds("Electron", N; conserve_qns=true)
    H = hubbard_mpo(sites, LX, LY, 8.0, -0.25)
    Random.seed!(1234)
    psi = random_mps(sites, initial_state(N, NE, 0); linkdims=10)
    md = o["maxdims"]
    t0 = time()
    E = 0.0
    for (s, D) in enumerate(md)
        nz = s <= length(md) - 3 ? o["noise"] : 0.0
        E, psi = dmrg(H, psi; nsweeps=(s == 1 ? 2 : 1) * o["sweeps"], maxdim=D, cutoff=o["cutoff"], noise=nz, outputlevel=0)
        @printf("maxdim %5d  E = %.10f  maxlinkdim %d  (%.0f s)\n", D, E, maxlinkdim(psi), time() - t0)
        flush(stdout)
    end
    secs = time() - t0
    normalize!(psi)
    serialize(joinpath(out, "mps.jls"), psi)
    S2 = real(inner(psi', s2_mpo(sites), psi))
    @printf("<S^2> (MPO) = %.3e   w_T <= %.3e\n", S2, S2 / 2)
    # dense tensor and sector entries
    T = psi[1]
    for j in 2:N
        T *= psi[j]
    end
    arr = Array(T, sites...)
    ub, db, am = Int64[], Int64[], Float64[]
    for I in CartesianIndices(arr)
        v = Tuple(I)
        nu = count(x -> x == 2 || x == 4, v)
        nd = count(x -> x == 3 || x == 4, v)
        (nu == NE ÷ 2 && nd == NE ÷ 2) || continue
        push!(ub, bits(v, (2, 4)))
        push!(db, bits(v, (3, 4)))
        push!(am, arr[I])
    end
    write(joinpath(out, "sector_ubits.bin"), htol.(ub))
    write(joinpath(out, "sector_dbits.bin"), htol.(db))
    write(joinpath(out, "sector_amp.bin"), htol.(am))
    @printf("sector entries %d, sector norm^2 %.12f\n", length(am), sum(abs2, am))
    # perfect samples with amplitudes
    Random.seed!(o["seed"])
    su, sd, sa = Int64[], Int64[], Float64[]
    for _ in 1:o["nsamples"]
        v = sample!(psi)
        a = ITensor(1.0)
        for j in 1:N
            a *= psi[j] * onehot(dag(sites[j]) => v[j])
        end
        push!(su, bits(v, (2, 4)))
        push!(sd, bits(v, (3, 4)))
        push!(sa, scalar(a))
    end
    open(joinpath(out, "samples.json"), "w") do f
        write(f, "{\"ubits\": [" * join(su, ",") * "], \"dbits\": [" * join(sd, ",") * "], \"amp\": [" *
                 join([@sprintf("%.17e", x) for x in sa], ",") * "]}")
    end
    open(joinpath(out, "dmrg_meta.json"), "w") do f
        write(f, @sprintf("{\"S2_mpo\": %.17e, \"E_last\": %.12f, \"maxdims\": \"%s\", \"seconds\": %.1f, \"maxlinkdim\": %d, \"sector_norm2\": %.15f, \"nsamples\": %d}",
            S2, E, join(md, ","), secs, maxlinkdim(psi), sum(abs2, am), o["nsamples"]))
    end
    println("wrote $(out)")
end

function (@main)(ARGS)
    o = parse_cli(ARGS)
    log_path = make_log_path(@__DIR__, "r47_dmrg")
    with_logging(log_path) do
        run_rehearsal(o)
    end
end
