# HVA Gate Set + Shared Trotter Coefficients Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Hermitian, `s_z`-conserving, brick-wall-ordered HVA gate set for the Hubbard model, plus optional coefficient sharing across gates in the Trotter optimizer with exact gradients.

**Architecture:** Coefficient sharing is a linear reparameterization `a = E·θ` sitting strictly *above* the circuit: the forward map is a gather, the gradient is a scatter-add. All expansion happens inside the loss functions before they call the existing circuit code, so `trotter_evolution.jl` and `trotter_gpu_kernels.jl` are never touched and keep receiving a full length-`M` step-coefficient vector.

**Tech Stack:** Julia 1.12, Zygote + ChainRulesCore (custom `rrule`s), LinearMaps, SparseArrays, `Test` stdlib.

**Spec:** `docs/superpowers/specs/2026-09-14-hva-ansatz-shared-coefficients-design.md`

## Global Constraints

- Working directory for ALL paths and commands below: `/home/jek354/research/ML-signproblem/experimenting/ed/`. All relative paths are relative to that directory.
- Run Julia as `julia --project=..` (the `Project.toml`/`Manifest.toml` live one level up, in `ML-signproblem/experimenting/`).
- Lattice convention (from `file_conventions.txt`): `Lvec = (Lx, Ly)`, site index `s = y + Ly*x`, C-order. **Axis 1 is horizontal (x); axis 2 is vertical (y).**
- Site `s` is **1-based**; it occupies bit `s-1`. Mask is `DtMb(1) << DtMb(s-1)`. `DtMb === UInt32`.
- The h.c.-deduplication convention used throughout `TamFermion.jl` is `(g.cre_up, g.cre_dn) <= (g.ann_up, g.ann_dn)`. Every gate this plan creates must satisfy it.
- `param_map === nothing` must short-circuit to today's exact behavior everywhere. This is the regression guarantee.
- Gradients for shared parameters **accumulate**: allocate with `zeros`, never `Vector{Float64}(undef, ·)`, and use `+=`.
- Do not modify `trotter_evolution.jl` or `trotter_gpu_kernels.jl`.
- Out of scope: the Python `TamFermion.py` copies, the `UCC_Hubbard/Current workflow/` Julia copy, GPU-path tests.

### Deviation from the spec, deliberate

The spec placed the three sharing helpers in `trotter_core_optimization.jl`. This plan puts them in a new focused file `trotter_shared_params.jl` instead — one clear responsibility, ~70 lines, independently reviewable. Everything else follows the spec.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `TamFermion.jl` | Modify | Add `enumerate_ferm_excitations_HVA` and `is_diagonal_gate`; export both. |
| `trotter_shared_params.jl` | **Create** | The three pure sharing helpers: `num_shared_params`, `expand_shared_coefficients`, `contract_shared_gradient`. No dependencies beyond Base. |
| `trotter_optimization.jl` | Modify | `include("trotter_shared_params.jl")` first; extend `export` list. |
| `trotter_loss.jl` | Modify | Add `param_map` kwarg to `adjoint_loss`, `energy_loss`, and both `rrule`s. Expand on the way in, contract on the way out. |
| `trotter_core_optimization.jl` | Modify | `param_map` kwarg on `optimize_unitary`; `M_full` computed from `n_params`; antihermitian-vs-diagonal guard; `grow_coefficients` gains `param_map`. |
| `trotter.jl` | Modify | Re-export the new names at `Trotter` level. |
| `testing/test_hva_gates_and_param_map.jl` | **Create** | All tests, one `@testset` block added per task. |

---

## Task 1: `enumerate_ferm_excitations_HVA`

**Files:**
- Modify: `TamFermion.jl` (add functions after `enumerate_ferm_excitations`, which ends at line ~528; add exports near line 20)
- Modify: `trotter.jl:33` (re-export list)
- Test: `testing/test_hva_gates_and_param_map.jl` (create)

**Interfaces:**
- Consumes: `FGate`, `DtMb`, `findForwardLatticeNeighbors(Lvec; use_pbc)` (returns `ndim × N` matrix of 1-based forward neighbours, `0` where an OBC neighbour is missing), `unravel_c(idx, dims)` (0-based coordinate tuple, from `TamLib`), `getReducedHilSpace`, `combineSpinInts`, `fgateToTauSector`.
- Produces:
  - `enumerate_ferm_excitations_HVA(Lvec; use_pbc::Bool=false, tie::Symbol=:full) -> (gates::Vector{FGate}, param_map::Vector{Int})`
  - `is_diagonal_gate(g::FGate) -> Bool`

---

- [ ] **Step 1: Write the failing test**

Create `testing/test_hva_gates_and_param_map.jl`:

```julia
#=
test_hva_gates_and_param_map.jl

Tests for enumerate_ferm_excitations_HVA and for shared Trotter coefficients.
Run from the `ed/` directory:

    julia --project=.. testing/test_hva_gates_and_param_map.jl
=#

using Test
using LinearAlgebra
using Random

const ED_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ED_DIR, "trotter.jl"))

using .Trotter
using .Trotter.TamFermion
using .Trotter.TrotterOptimization

# ───────────────────────────────────────────────────────────────────────
# Shared helpers
# ───────────────────────────────────────────────────────────────────────

"""Build the combined 2N-bit sector basis in the SAME ordering that
`HubbardRealSpace` uses for its matrix: up index slow, down index fast."""
function build_sector_basis(Lvec, nvec)
    N = prod(Lvec)
    bu = TamFermion.getReducedHilSpace(N, nvec[1]; returnOcc=false)
    bd = TamFermion.getReducedHilSpace(N, nvec[2]; returnOcc=false)
    return [TamFermion.combineSpinInts(bu[iu], bd[id], N)
            for iu in eachindex(bu) for id in eachindex(bd)]
end

dense_taus(gates, N, basis) =
    [Matrix(t) for t in TamFermion.fgateToTauSector(gates, N, basis; antihermitian=false)]

@testset "HVA gate set" begin
    @testset "structure on $(Lvec)" for Lvec in [(2, 2), (3, 2)]
        N = prod(Lvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)

        # param_map is a contiguous surjection onto 1:n_params
        @test length(pmap) == length(gates)
        @test sort(unique(pmap)) == collect(1:maximum(pmap))

        # The first N gates are the on-site interactions, one per site.
        @test all(is_diagonal_gate, gates[1:N])
        @test all(!is_diagonal_gate, gates[N+1:end])
        # ...and they all share one coefficient under the default tie=:full
        @test all(pmap[1:N] .== pmap[1])
        @test !any(pmap[N+1:end] .== pmap[1])

        # Every gate respects the h.c.-deduplication convention.
        @test all(g -> (g.cre_up, g.cre_dn) <= (g.ann_up, g.ann_dn), gates)

        # Hopping gates come in up/down pairs, one spin channel each.
        for g in gates[N+1:end]
            up_only = (g.cre_dn == 0 && g.ann_dn == 0)
            dn_only = (g.cre_up == 0 && g.ann_up == 0)
            @test up_only ⊻ dn_only
        end
    end

    @testset "gate counts and grouping on (2,2) OBC" begin
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA((2, 2))
        # 4 on-site; horizontal bonds (1,3),(2,4) parity 0 and none parity 1;
        # vertical bonds (1,2),(3,4) parity 0 and none parity 1.
        # 2 spin channels per bond => 4 + 4 + 4 = 12 gates, 3 parameter groups.
        @test length(gates) == 12
        @test maximum(pmap) == 3
        @test count(==(1), pmap) == 4
        @test count(==(2), pmap) == 4
        @test count(==(3), pmap) == 4
    end

    @testset "generators are Hermitian and commute within a group" begin
        Lvec, nvec = (3, 2), (2, 1)
        N = prod(Lvec)
        basis = build_sector_basis(Lvec, nvec)
        gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)
        taus = dense_taus(gates, N, basis)

        for t in taus
            @test t ≈ t'
        end
        for k in 1:N
            @test isdiag(taus[k])
        end
        for k in eachindex(taus), l in eachindex(taus)
            if k < l && pmap[k] == pmap[l]
                @test norm(taus[k] * taus[l] - taus[l] * taus[k]) < 1e-10
            end
        end
    end

    @testset "tie modes" begin
        Lvec = (3, 2)
        N = prod(Lvec)
        gates, pmap_full = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:full)
        _, pmap_none = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:none)
        _, pmap_spin = TamFermion.enumerate_ferm_excitations_HVA(Lvec; tie=:spin)

        @test pmap_none == collect(1:length(gates))

        # (3,2) OBC: horizontal parity-0 bonds (1,3),(2,4); parity-1 (3,5),(4,6);
        # vertical parity-0 (1,2),(3,4),(5,6); vertical parity-1 is EMPTY and
        # must be dropped, so 4 groups rather than 5.
        @test maximum(pmap_full) == 4

        # :spin gives one parameter per site plus one per bond.
        n_bonds = (length(gates) - N) ÷ 2
        @test maximum(pmap_spin) == N + n_bonds
        @test pmap_spin[1:N] == collect(1:N)
        for b in 1:n_bonds
            @test pmap_spin[N+2b-1] == pmap_spin[N+2b]
        end
    end

    @testset "odd periodic axis is rejected" begin
        @test_throws ArgumentError TamFermion.enumerate_ferm_excitations_HVA((3, 2); use_pbc=true)
        @test_throws ArgumentError TamFermion.enumerate_ferm_excitations_HVA((3, 2); tie=:nonsense)
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: FAIL — `UndefVarError: enumerate_ferm_excitations_HVA not defined` (the `TamFermion.` prefix means it fails at call time inside the testset).

- [ ] **Step 3: Implement**

In `TamFermion.jl`, extend the fermion-gate export line (currently line 20):

```julia
export singleSpinChannelCombos, enumerate_ferm_excitations, enumerate_ferm_excitations_HVA, sortGatesByIJ, is_diagonal_gate
```

Then insert the following immediately after the end of `enumerate_ferm_excitations` (just before the `tau_g_operator_sector` docstring):

```julia
"""
    is_diagonal_gate(g::FGate) → Bool

`true` when `C†_I C_J` has `I == J`, i.e. the generator τ_g is diagonal in the
occupation basis. Such gates are annihilated by the antihermitian convention.
"""
is_diagonal_gate(g::FGate) = (g.cre_up == g.ann_up) && (g.cre_dn == g.ann_dn)

"""
Renumber `key` so distinct values become 1, 2, 3, … in order of first
appearance. Groups that never occur simply never get an id, which is how
empty (axis, parity) blocks are dropped.
"""
function _renumber_contiguous(key::AbstractVector{Int})
    lookup = Dict{Int,Int}()
    out = Vector{Int}(undef, length(key))
    next = 1
    for (k, v) in enumerate(key)
        id = get(lookup, v, 0)
        if id == 0
            id = next
            lookup[v] = id
            next += 1
        end
        out[k] = id
    end
    return out
end

"""
    enumerate_ferm_excitations_HVA(Lvec; use_pbc=false, tie=:full) → (gates, param_map)

Hamiltonian-Variational-Ansatz gate set for the Hubbard model on a lattice of
dimensions `Lvec`. Strictly **Hermitian** (the on-site interaction gates are
diagonal), does **not** conserve momentum, **does** conserve `s_z`.

Unlike [`enumerate_ferm_excitations`](@ref) the result is deliberately **not**
passed through `sortGatesByIJ` — the emitted order *is* the ansatz:

1. on-site `n_{i↑} n_{i↓}`, sites `1:N`;
2. for each lattice axis `a = 1:ndim` (axis 1 is horizontal), the hopping bonds
   of parity 0 then the bonds of parity 1.

A bond's parity is the parity of its source site's coordinate along that axis, so
parity-0 bonds are `(0,1),(2,3),…` and parity-1 bonds are `(1,2),(3,4),…`. Within
one parity group the bonds have disjoint support and therefore commute.

Each bond emits **two** gates, spin-up then spin-down, because an `FGate` is a
single product operator and cannot represent a sum over spin channels. `param_map`
is what ties them back together.

`tie` selects the returned `param_map`:
- `:full` (default) — one coefficient for the on-site block and one per
  (axis, parity) hopping block, shared across both spin channels.
- `:none` — `collect(1:length(gates))`, every gate independent.
- `:spin` — one coefficient per site for the on-site block, and one per bond
  shared by that bond's up/down pair.

`use_pbc=true` throws for an axis of odd length > 2: a periodic ring of odd length
admits no 2-colouring into commuting halves.
"""
function enumerate_ferm_excitations_HVA(Lvec;
    use_pbc::Bool=false,
    tie::Symbol=:full)

    if !(tie in (:full, :none, :spin))
        throw(ArgumentError("tie must be one of :full, :none, :spin; got :$tie"))
    end

    dims = Tuple(collect(Int, Lvec))
    N = prod(dims)
    ndim = length(dims)

    if use_pbc
        for (a, L) in enumerate(dims)
            if L > 2 && isodd(L)
                throw(ArgumentError(
                    "use_pbc=true with odd length L=$L on axis $a: a periodic ring " *
                    "of odd length cannot be 2-coloured into commuting bond groups."))
            end
        end
    end

    sitebit(s::Integer) = DtMb(1) << DtMb(s - 1)

    gates = FGate[]
    block_key = Int[]   # on-site block = 1; (axis, parity) blocks = 2, 3, …
    fine_key = Int[]    # on-site = site index; bond gates = N + bond index

    # ── Block 1: on-site interaction ────────────────────────────────────
    for s in 1:N
        m = sitebit(s)
        push!(gates, FGate(m, m, m, m))
        push!(block_key, 1)
        push!(fine_key, s)
    end

    # ── Hopping blocks, axis-major then parity ──────────────────────────
    neighbors = findForwardLatticeNeighbors(dims; use_pbc=use_pbc)
    block = 1
    nbond = 0
    for a in 1:ndim
        for parity in 0:1
            block += 1
            for i in 1:N
                j = neighbors[a, i]
                j == 0 && continue                      # missing OBC neighbour
                coords = unravel_c(i - 1, dims)         # 0-based
                (coords[a] % 2) == parity || continue
                nbond += 1
                # Canonicalise to (cre_up, cre_dn) <= (ann_up, ann_dn). Swapping
                # cre/ann leaves τ_g = C†_I C_J + C†_J C_I unchanged.
                lo, hi = i < j ? (sitebit(i), sitebit(j)) : (sitebit(j), sitebit(i))
                push!(gates, FGate(lo, hi, DtMb(0), DtMb(0)))   # spin up
                push!(block_key, block)
                push!(fine_key, N + nbond)
                push!(gates, FGate(DtMb(0), DtMb(0), lo, hi))   # spin down
                push!(block_key, block)
                push!(fine_key, N + nbond)
            end
        end
    end

    param_map = if tie === :none
        collect(1:length(gates))
    elseif tie === :full
        _renumber_contiguous(block_key)
    else # :spin
        _renumber_contiguous(fine_key)
    end

    return gates, param_map
end
```

In `trotter.jl`, extend the TamFermion re-export line (currently line 33):

```julia
export get_basis_sector, enumerate_ferm_excitations, enumerate_ferm_excitations_HVA, fgateToTauSector, is_diagonal_gate
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: PASS — `Test Summary: | Pass  Total` for the `HVA gate set` testset, no failures.

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research
git add ML-signproblem/experimenting/ed/TamFermion.jl \
        ML-signproblem/experimenting/ed/trotter.jl \
        ML-signproblem/experimenting/ed/testing/test_hva_gates_and_param_map.jl
git commit -m "feat: add enumerate_ferm_excitations_HVA brick-wall Hubbard gate set"
```

---

## Task 2: Sharing helpers

**Files:**
- Create: `trotter_shared_params.jl`
- Modify: `trotter_optimization.jl` (include list ~line 23, export list ~line 27)
- Modify: `trotter.jl:36` (re-export list)
- Test: `testing/test_hva_gates_and_param_map.jl` (append a testset)

**Interfaces:**
- Consumes: nothing beyond Base.
- Produces:
  - `num_shared_params(param_map, num_gates::Int) -> Int`
  - `expand_shared_coefficients(theta::AbstractVector, param_map, num_gates::Int, P::Int) -> AbstractVector` (the gather `a = E·θ`)
  - `contract_shared_gradient(grad_a::AbstractVector, param_map, num_gates::Int, P::Int) -> Vector` (the scatter-add `Eᵀ·grad_a`)
  - In all three, `param_map` is `Union{Nothing,AbstractVector{Int}}` and `nothing` means identity.

---

- [ ] **Step 1: Write the failing test**

Append to `testing/test_hva_gates_and_param_map.jl`:

```julia
@testset "sharing helpers" begin
    pmap = [1, 1, 2, 3, 3, 3]
    num_gates = 6
    P = 2

    @test num_shared_params(nothing, num_gates) == 6
    @test num_shared_params(pmap, num_gates) == 3

    # Gather: layer-major, theta[(l-1)*n_params + pmap[j]] lands at a[(l-1)*num_gates+j]
    theta = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]   # 2 layers × 3 params
    a = expand_shared_coefficients(theta, pmap, num_gates, P)
    @test a == [10.0, 10.0, 20.0, 30.0, 30.0, 30.0,
                40.0, 40.0, 50.0, 60.0, 60.0, 60.0]

    # Scatter-add is the exact transpose of the gather.
    grad_a = collect(1.0:12.0)
    grad_theta = contract_shared_gradient(grad_a, pmap, num_gates, P)
    @test grad_theta == [1.0 + 2.0, 3.0, 4.0 + 5.0 + 6.0,
                         7.0 + 8.0, 9.0, 10.0 + 11.0 + 12.0]

    # <grad_a, E*theta> == <E'*grad_a, theta>  (the adjoint identity itself)
    @test dot(grad_a, a) ≈ dot(grad_theta, theta)

    # nothing short-circuits to identity
    @test expand_shared_coefficients(theta, nothing, 3, P) === theta
    @test contract_shared_gradient(grad_a, nothing, 6, P) === grad_a

    # Length validation
    @test_throws ArgumentError expand_shared_coefficients(theta, pmap, 5, P)
    @test_throws ArgumentError expand_shared_coefficients([1.0, 2.0], pmap, num_gates, P)
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: FAIL — `UndefVarError: num_shared_params not defined`.

- [ ] **Step 3: Implement**

Create `trotter_shared_params.jl`:

```julia
#=
trotter_shared_params.jl

Shared Trotter coefficients.

Letting several gates be driven by one variational parameter is a *linear
reparameterisation* of the circuit:

    a = E * theta                       (a gather)
    d loss / d theta = E' * d loss / d a   (a scatter-ADD)

where `a` is the per-step coefficient vector the circuit already consumes
(length `P * num_gates`, layer-major) and `theta` is the reduced parameter
vector (length `P * n_params`). Concretely

    a[(l-1)*num_gates + j] == theta[(l-1)*n_params + param_map[j]]

for layer `l in 1:P` and gate `j in 1:num_gates`.

Because the reparameterisation sits strictly *above* the circuit, none of the
checkpointing, rematerialisation or GPU adjoint code needs to know about it:
those routines keep receiving a full length-`P*num_gates` vector. In particular
the local name `param_idx` inside those loops remains a *gate* index used to
look up `gpu_ops`, exactly as before.

The chain rule gives the scatter-add whether or not the gates sharing a
parameter commute. Commutation matters for the ansatz being a faithful HVA
layer, not for gradient correctness.

`param_map === nothing` means "no sharing" and short-circuits every helper here
to identity, which is what guarantees byte-for-byte unchanged behaviour for all
existing callers.
=#

"""
    num_shared_params(param_map, num_gates) → Int

Number of distinct variational parameters **per Trotter layer**.
`nothing` means one parameter per gate.
"""
num_shared_params(param_map::Nothing, num_gates::Int) = num_gates
num_shared_params(param_map::AbstractVector{Int}, num_gates::Int) = maximum(param_map)

function _check_shared(param_map::AbstractVector{Int}, theta_len::Int, num_gates::Int, P::Int)
    if length(param_map) != num_gates
        throw(ArgumentError(
            "length(param_map) = $(length(param_map)) does not match num_gates = $num_gates"))
    end
    n_params = maximum(param_map)
    if theta_len != P * n_params
        throw(ArgumentError(
            "coefficient vector has length $theta_len but num_exponentials * n_params = " *
            "$P * $n_params = $(P * n_params)"))
    end
    return n_params
end

"""
    expand_shared_coefficients(theta, param_map, num_gates, P) → a

Gather the reduced parameter vector `theta` into the full per-step coefficient
vector `a` of length `P * num_gates` that the circuit consumes.
"""
expand_shared_coefficients(theta::AbstractVector, ::Nothing, num_gates::Int, P::Int) = theta

function expand_shared_coefficients(theta::AbstractVector{T}, param_map::AbstractVector{Int},
    num_gates::Int, P::Int) where {T}
    n_params = _check_shared(param_map, length(theta), num_gates, P)
    a = Vector{T}(undef, P * num_gates)
    for l in 1:P
        off_a = (l - 1) * num_gates
        off_t = (l - 1) * n_params
        for j in 1:num_gates
            a[off_a+j] = theta[off_t+param_map[j]]
        end
    end
    return a
end

"""
    contract_shared_gradient(grad_a, param_map, num_gates, P) → grad_theta

Scatter-**add** a full per-step gradient back onto the reduced parameters. This
is the exact transpose of `expand_shared_coefficients`, so a shared parameter's
gradient is the sum of the gradients of every step it drives.
"""
contract_shared_gradient(grad_a::AbstractVector, ::Nothing, num_gates::Int, P::Int) = grad_a

function contract_shared_gradient(grad_a::AbstractVector{T}, param_map::AbstractVector{Int},
    num_gates::Int, P::Int) where {T}
    if length(param_map) != num_gates
        throw(ArgumentError(
            "length(param_map) = $(length(param_map)) does not match num_gates = $num_gates"))
    end
    n_params = maximum(param_map)
    if length(grad_a) != P * num_gates
        throw(ArgumentError(
            "grad_a has length $(length(grad_a)) but num_exponentials * num_gates = " *
            "$(P * num_gates)"))
    end
    grad_theta = zeros(T, P * n_params)   # MUST be zeros: entries are accumulated
    for l in 1:P
        off_a = (l - 1) * num_gates
        off_t = (l - 1) * n_params
        for j in 1:num_gates
            grad_theta[off_t+param_map[j]] += grad_a[off_a+j]
        end
    end
    return grad_theta
end
```

In `trotter_optimization.jl`, add the include as the **first** sub-component (before `trotter_gpu_kernels.jl`):

```julia
# Sub-components
include("trotter_shared_params.jl")
include("trotter_gpu_kernels.jl")
include("trotter_evolution.jl")
include("trotter_loss.jl")
include("trotter_core_optimization.jl")
include("trotter_scan.jl")
```

and extend its `export` line with the three names:

```julia
export adjoint_loss, energy_loss, optimize_unitary, interaction_scan_map_to_state, extract_convergence_info, grow_coefficients,
       StridedCheckpoints, determine_checkpoint_stride, apply_unitary_checkpoints, backward_adjoint_propagation, apply_unitary,
       strip_global_phase, get_gpu_gate_ops, gpu_apply_gate_exp!, to_device_vector,
       num_shared_params, expand_shared_coefficients, contract_shared_gradient
```

In `trotter.jl`, extend the Trotter-level re-export (currently line 36) with the same three names appended:

```julia
export adjoint_loss, energy_loss, optimize_unitary, interaction_scan_map_to_state, extract_convergence_info, grow_coefficients,
       StridedCheckpoints, determine_checkpoint_stride, apply_unitary_checkpoints, backward_adjoint_propagation, apply_unitary,
       strip_global_phase, get_gpu_gate_ops, gpu_apply_gate_exp!, to_device_vector,
       num_shared_params, expand_shared_coefficients, contract_shared_gradient
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: PASS — both `HVA gate set` and `sharing helpers` testsets green.

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research
git add ML-signproblem/experimenting/ed/trotter_shared_params.jl \
        ML-signproblem/experimenting/ed/trotter_optimization.jl \
        ML-signproblem/experimenting/ed/trotter.jl \
        ML-signproblem/experimenting/ed/testing/test_hva_gates_and_param_map.jl
git commit -m "feat: add gather/scatter helpers for shared Trotter coefficients"
```

---

## Task 3: Wire `param_map` into the loss functions

This is the task where gradient correctness is actually established. Read the `trotter_shared_params.jl` header comment before starting.

**Files:**
- Modify: `trotter_loss.jl` (all four definitions in the file)
- Modify: `trotter_evolution.jl:243` — `apply_unitary` signature only, adding a `param_map` kwarg that expands up front. **No other change to that file.**
- Test: `testing/test_hva_gates_and_param_map.jl` (append a testset)

**Interfaces:**
- Consumes: `num_shared_params`, `expand_shared_coefficients`, `contract_shared_gradient` from Task 2; `enumerate_ferm_excitations_HVA` from Task 1; the unchanged `apply_unitary_checkpoints` and `backward_adjoint_propagation`.
- Produces: `param_map::Union{Nothing,AbstractVector{Int}}=nothing` keyword on `adjoint_loss`, `energy_loss`, `apply_unitary`, and on both `rrule`s. When it is non-`nothing`, the first positional argument `A` is the **reduced** vector of length `num_exponentials * maximum(param_map)`, and the gradient returned has that same length.

---

- [ ] **Step 1: Write the failing test**

Append to `testing/test_hva_gates_and_param_map.jl`:

```julia
using Zygote

"""Central finite differences of `f` at `x`."""
function fd_gradient(f, x::AbstractVector; h=1e-5)
    g = similar(x)
    for k in eachindex(x)
        xp = copy(x); xp[k] += h
        xm = copy(x); xm[k] -= h
        g[k] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

@testset "shared-coefficient gradients" begin
    Random.seed!(20260914)
    Lvec, nvec = (2, 2), (1, 1)
    N = prod(Lvec)
    basis = build_sector_basis(Lvec, nvec)
    d = length(basis)

    gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)
    num_gates = length(gates)
    tau_terms = TamFermion.fgateToTauSector(gates, N, basis; antihermitian=false)

    ref = normalize!(randn(ComplexF64, d))
    target = normalize!(randn(ComplexF64, d))
    H = TamFermion.HubbardRealSpace(1.0, 4.0, Lvec, nvec; use_pbc=false, returnBasis=false)
    @test size(H, 1) == d

    PP = Ref(1)   # the layer count the closures below read; set before each call
    make_loss(kind, pm) = A -> (kind === :overlap ?
        adjoint_loss(A, gates, tau_terms, ref, target, basis, N;
                     num_exponentials=PP[], antihermitian=false, param_map=pm) :
        energy_loss(A, gates, tau_terms, H, ref, basis, N;
                    num_exponentials=PP[], antihermitian=false, param_map=pm))

    for P in (1, 2), kind in (:overlap, :energy)
        PP[] = P
        n_params = maximum(pmap)
        theta = 0.3 .* randn(P * n_params)

        f_tied = make_loss(kind, pmap)
        f_free = make_loss(kind, nothing)

        # 1. Tied forward equals the expanded untied forward.
        a = expand_shared_coefficients(theta, pmap, num_gates, P)
        @test f_tied(theta) ≈ f_free(a)

        # 2. Analytic gradient equals central finite differences.
        g_analytic = Zygote.gradient(f_tied, theta)[1]
        g_fd = fd_gradient(f_tied, theta)
        @test isapprox(g_analytic, g_fd; rtol=1e-5, atol=1e-7)

        # 3. Analytic gradient equals the group-wise SUM of the untied gradient.
        g_free = Zygote.gradient(f_free, a)[1]
        g_contracted = contract_shared_gradient(g_free, pmap, num_gates, P)
        @test isapprox(g_analytic, g_contracted; rtol=1e-9, atol=1e-12)
    end

    # 4. Regression guarantee: an identity param_map reproduces param_map=nothing.
    PP[] = 2
    identity_map = collect(1:num_gates)
    a = 0.2 .* randn(2 * num_gates)
    f_id = make_loss(:overlap, identity_map)
    f_nothing = make_loss(:overlap, nothing)
    @test f_id(a) == f_nothing(a)
    @test Zygote.gradient(f_id, a)[1] == Zygote.gradient(f_nothing, a)[1]

    # 5. apply_unitary accepts a reduced vector via param_map.
    PP[] = 1
    theta = 0.3 .* randn(maximum(pmap))
    v_tied = apply_unitary(theta, gates, ref, basis, N, 1; param_map=pmap)
    v_free = apply_unitary(expand_shared_coefficients(theta, pmap, num_gates, 1),
                           gates, ref, basis, N, 1)
    @test v_tied ≈ v_free
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: FAIL — `MethodError` / `got unsupported keyword argument "param_map"` from `adjoint_loss`.

- [ ] **Step 3: Implement**

In `trotter_evolution.jl`, change **only** the `apply_unitary` signature and its first line:

```julia
function apply_unitary(A::AbstractArray, gates, ref::AbstractArray, basis, N::Int, num_exponentials::Int;
    antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    A = expand_shared_coefficients(A, param_map, length(gates), num_exponentials)
    P = num_exponentials
    num_gates = length(gates)
    M = P * num_gates
    # ... body unchanged ...
```

Replace `trotter_loss.jl` in full:

```julia
#=
trotter_loss.jl

Overlap and energy loss functions with custom ChainRules rrules for Trotter optimization.

Coefficient sharing (`param_map`) is handled entirely here: the reduced parameter
vector is expanded to the full per-step vector on the way in, and the full per-step
gradient is contracted back on the way out. See trotter_shared_params.jl. The
circuit routines below this layer never see a `param_map`.
=#

# ═══════════════════════════════════════════════════════════════════════
# OVERLAP LOSS
# ═══════════════════════════════════════════════════════════════════════

"""
    adjoint_loss(A, gates, tau_terms, ref, target, basis, N; ...) -> loss

Compute the infidelity loss: 1 - |<target|U(A)|ref>|^2.

With `param_map !== nothing`, `A` is the reduced parameter vector of length
`num_exponentials * maximum(param_map)` and gates sharing an entry of `param_map`
are driven by the same coefficient.
"""
function adjoint_loss(A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    A_steps = expand_shared_coefficients(A, param_map, length(gates), num_exponentials)
    ref_evolved = apply_unitary(A_steps, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    target_dev = to_device_vector(target, use_gpu, datatype)
    return 1.0 - abs2(dot(target_dev, ref_evolved))
end

function ChainRulesCore.rrule(::typeof(adjoint_loss), A::AbstractArray, gates, tau_terms, ref::AbstractArray, target::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    num_gates = length(gates)
    A_steps = expand_shared_coefficients(A, param_map, num_gates, num_exponentials)
    t = @elapsed begin
        target_dev = to_device_vector(target, use_gpu, datatype)
        phis = apply_unitary_checkpoints(A_steps, gates, ref, basis, N, num_exponentials;
            antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        evolved_ref = last_checkpoint(phis)
        overlap = dot(target_dev, evolved_ref)
        loss = 1.0 - abs2(overlap)
        println("loss: $loss")
    end
    println("Forward time: $t")

    function adjoint_loss_pullback(y)
        t = @elapsed begin
            init_adjoint_state = (2 * overlap * conj(y)) * target_dev
            grad_steps = backward_adjoint_propagation(A_steps, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials;
                antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
            grad_A = contract_shared_gradient(grad_steps, param_map, num_gates, num_exponentials)
        end
        println("Gradient time: $t")
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, adjoint_loss_pullback
end

# ═══════════════════════════════════════════════════════════════════════
# ENERGY LOSS
# ═══════════════════════════════════════════════════════════════════════

"""
    energy_loss(A, gates, tau_terms, H, ref, basis, N; ...) -> loss

Compute the variational energy loss: <ref|U(A)^† H U(A)|ref>.

With `param_map !== nothing`, `A` is the reduced parameter vector of length
`num_exponentials * maximum(param_map)`.
"""
function energy_loss(A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    A_steps = expand_shared_coefficients(A, param_map, length(gates), num_exponentials)
    ref_evolved = apply_unitary(A_steps, gates, ref, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    return real(dot(ref_evolved, H * ref_evolved))
end

function ChainRulesCore.rrule(::typeof(energy_loss), A::AbstractArray, gates, tau_terms, H, ref::AbstractArray, basis, N::Int;
    num_exponentials::Int=1, antihermitian::Bool=false, use_gpu::Bool=false, datatype::Type{<:Number}=ComplexF64,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    num_gates = length(gates)
    A_steps = expand_shared_coefficients(A, param_map, num_gates, num_exponentials)
    phis = apply_unitary_checkpoints(A_steps, gates, ref, basis, N, num_exponentials;
        antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
    evolved_ref = last_checkpoint(phis)
    loss = real(dot(evolved_ref, H * evolved_ref))
    println("loss: $loss")
    function energy_loss_pullback(y)
        init_adjoint_state = (-2 * conj(y)) * (H * evolved_ref)
        grad_steps = backward_adjoint_propagation(A_steps, gates, tau_terms, phis, init_adjoint_state, basis, N, num_exponentials;
            antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype)
        grad_A = contract_shared_gradient(grad_steps, param_map, num_gates, num_exponentials)
        return NoTangent(), grad_A, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return loss, energy_loss_pullback
end
```

Two things to check while transcribing, both easy to get wrong:

1. The number of `NoTangent()` values returned is unchanged (8 total: one for the function itself plus one per positional argument). `param_map` is a **keyword**, so it contributes none.
2. `backward_adjoint_propagation` is called with `A_steps`, the **expanded** vector — it needs the per-step coefficients to rebuild the inverse gates. Passing the reduced `A` there would silently produce wrong numbers for any `P > 1`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: PASS — all three testsets green. The finite-difference comparison is the load-bearing one; if it fails only for `P=2` the per-layer offset arithmetic in `expand_shared_coefficients`/`contract_shared_gradient` is wrong.

- [ ] **Step 5: Commit**

```bash
cd /home/jek354/research
git add ML-signproblem/experimenting/ed/trotter_loss.jl \
        ML-signproblem/experimenting/ed/trotter_evolution.jl \
        ML-signproblem/experimenting/ed/testing/test_hva_gates_and_param_map.jl
git commit -m "feat: support shared coefficients in Trotter loss functions and gradients"
```

---

## Task 4: `optimize_unitary`, `grow_coefficients`, and the antihermitian guard

**Files:**
- Modify: `trotter_core_optimization.jl` — `grow_coefficients` (~line 87) and `optimize_unitary` (~line 251 onwards)
- Test: `testing/test_hva_gates_and_param_map.jl` (append a testset)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `optimize_unitary(...; param_map::Union{Nothing,AbstractVector{Int}}=nothing, ...)`, returning `A_opt` of length `num_exponentials * num_shared_params(param_map, length(gates))`.
  - `grow_coefficients(old_coeffs, old_num_exponentials, new_num_exponentials, num_gates; param_map=nothing)`.

---

- [ ] **Step 1: Write the failing test**

Append to `testing/test_hva_gates_and_param_map.jl`:

```julia
@testset "optimize_unitary and grow_coefficients with sharing" begin
    Random.seed!(20260915)
    Lvec, nvec = (2, 2), (1, 1)
    N = prod(Lvec)
    basis = build_sector_basis(Lvec, nvec)
    d = length(basis)

    gates, pmap = TamFermion.enumerate_ferm_excitations_HVA(Lvec)
    num_gates = length(gates)
    n_params = maximum(pmap)
    tau_terms = TamFermion.fgateToTauSector(gates, N, basis; antihermitian=false)

    ref = normalize!(randn(ComplexF64, d))
    target = normalize!(randn(ComplexF64, d))

    P = 2
    A_opt, final_loss, metrics = optimize_unitary(gates, tau_terms, ref, target, basis, N;
        loss_type=:overlap, num_exponentials=P, param_map=pmap,
        maxiters=10, initialization_samples=0, antihermitian=false)

    # The returned vector lives in the REDUCED space.
    @test length(A_opt) == P * n_params
    @test isfinite(final_loss)
    @test 0.0 <= final_loss <= 1.0 + 1e-8
    @test length(metrics["optimization_gradients"][1][end]) == P * n_params

    # grow_coefficients grows in units of n_params, not num_gates.
    grown = grow_coefficients(A_opt, P, 3, num_gates; param_map=pmap)
    @test length(grown) == 3 * n_params
    @test grown[1:P*n_params] == A_opt
    @test all(grown[P*n_params+1:end] .== 0.0)

    # Default (no sharing) behaviour of grow_coefficients is unchanged.
    old = collect(1.0:Float64(2 * num_gates))
    @test grow_coefficients(old, 2, 3, num_gates) ==
          vcat(old, zeros(num_gates))

    # A diagonal gate under the antihermitian convention is rejected.
    @test_throws ArgumentError optimize_unitary(gates, tau_terms, ref, target, basis, N;
        loss_type=:overlap, num_exponentials=1, param_map=pmap,
        maxiters=1, initialization_samples=0, antihermitian=true)
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: FAIL — unsupported keyword argument `param_map` on `optimize_unitary`.

- [ ] **Step 3: Implement**

In `trotter_core_optimization.jl`, replace `grow_coefficients` with:

```julia
"""
    grow_coefficients(old_coeffs, old_num_exponentials, new_num_exponentials, num_gates; param_map=nothing) -> Vector{Float64}

Extend a Trotter coefficient vector that was optimized for `old_num_exponentials` layers into
one usable for a larger `new_num_exponentials`, so the newly-added (later) layers can be
optimized starting from an already-converged shorter ansatz instead of from scratch.

With `param_map !== nothing` the vector grows in units of
`num_shared_params(param_map, num_gates)` per layer rather than `num_gates`.
"""
function grow_coefficients(old_coeffs::AbstractVector, old_num_exponentials::Int, new_num_exponentials::Int, num_gates::Int;
    param_map::Union{Nothing,AbstractVector{Int}}=nothing)
    if new_num_exponentials < old_num_exponentials
        error("new_num_exponentials ($new_num_exponentials) must be >= old_num_exponentials ($old_num_exponentials)")
    end
    n_params = num_shared_params(param_map, num_gates)
    old_len = old_num_exponentials * n_params
    if length(old_coeffs) != old_len
        error("length(old_coeffs) = $(length(old_coeffs)) does not match old_num_exponentials * n_params = $old_len")
    end
    new_coeffs = zeros(Float64, new_num_exponentials * n_params)
    new_coeffs[1:old_len] .= old_coeffs
    return new_coeffs
end
```

In `optimize_unitary`, make exactly four edits.

**(a)** Add the keyword to the signature, immediately after `active_indices`:

```julia
    active_indices::Union{Nothing,AbstractVector{Int}}=nothing,
    param_map::Union{Nothing,AbstractVector{Int}}=nothing,
```

**(b)** Replace the `M_full` line:

```julia
    M_full = num_exponentials * length(gates)
```

with the guard plus the corrected count:

```julia
    if antihermitian && any(TamFermion.is_diagonal_gate, gates)
        throw(ArgumentError(
            "antihermitian=true with a gate set containing diagonal gates: " *
            "tau_g_operator_sector maps those to the zero operator, so their " *
            "coefficients would be unoptimisable. The HVA gate set from " *
            "enumerate_ferm_excitations_HVA is Hermitian by construction; use " *
            "antihermitian=false."))
    end

    n_params = num_shared_params(param_map, length(gates))
    M_full = num_exponentials * n_params
```

**(c)** Pass `param_map` to both losses inside the closure `f`:

```julia
    f = (A, p=nothing) -> begin
        A_full = isnothing(active_indices) ? A : embed_active_params(A, active_indices, M_full)
        if loss_type == :overlap
            return adjoint_loss(A_full, gates, tau_terms, ref_prep, target_prep, basis, N;
                num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype,
                param_map=param_map)
        elseif loss_type == :energy
            return energy_loss(A_full, gates, tau_terms, target_prep, ref_prep, basis, N;
                num_exponentials=num_exponentials, antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype,
                param_map=param_map)
        else
            error("Unknown loss_type: $loss_type")
        end
    end
```

**(d)** Pass `param_map` to the final diagnostic evolution near the end of the function:

```julia
    ref_evolved = apply_unitary(curr_A, gates, ref_prep, basis, N, num_exponentials; antihermitian=antihermitian, use_gpu=use_gpu, datatype=datatype, param_map=param_map)
```

Nothing else in the function changes. `M = isnothing(active_indices) ? M_full : length(active_indices)`, `embed_active_params`, the multi-start random sampling, `zeros(eltype(A_init), M)`, and `A_zero = zeros(Float64, M_full)` all pick up the reduced dimension automatically now that `M_full` is right.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: PASS — all four testsets green.

- [ ] **Step 5: Verify no regression in existing callers**

`grow_coefficients` is called by `run_trotter_scan_optimization.jl` and
`run_pruning_analysis.jl` with four positional arguments. Confirm those call sites
still type-check under the new signature (the fourth positional is still `num_gates`
and `param_map` defaults to `nothing`):

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
grep -n "grow_coefficients\|optimize_unitary(" run_trotter_scan_optimization.jl run_pruning_analysis.jl trotter_scan.jl system_scaling.jl
```

Expected: every call passes `grow_coefficients(coeffs, old_P, new_P, num_gates)` positionally with no fifth positional argument, and no `optimize_unitary` call passes a positional argument where `param_map` now sits (it is a keyword, so this is guaranteed). If any call site passes a fifth positional argument to `grow_coefficients`, convert it to `param_map=`.

- [ ] **Step 6: Commit**

```bash
cd /home/jek354/research
git add ML-signproblem/experimenting/ed/trotter_core_optimization.jl \
        ML-signproblem/experimenting/ed/testing/test_hva_gates_and_param_map.jl
git commit -m "feat: thread param_map through optimize_unitary and grow_coefficients"
```

---

## Final verification

- [ ] **Run the whole test file once more from a clean session**

```bash
cd /home/jek354/research/ML-signproblem/experimenting/ed
julia --project=.. testing/test_hva_gates_and_param_map.jl
```

Expected: four testsets, zero failures, zero errors.

- [ ] **Confirm the untouched files really are untouched**

```bash
cd /home/jek354/research
git diff --stat HEAD~4 -- ML-signproblem/experimenting/ed/trotter_gpu_kernels.jl
git diff HEAD~4 -- ML-signproblem/experimenting/ed/trotter_evolution.jl
```

Expected: `trotter_gpu_kernels.jl` has no diff at all. `trotter_evolution.jl` shows only the `apply_unitary` signature plus its one added `expand_shared_coefficients` line — if anything inside `apply_unitary_checkpoints` or `backward_adjoint_propagation` changed, that is a design violation; revert it.
