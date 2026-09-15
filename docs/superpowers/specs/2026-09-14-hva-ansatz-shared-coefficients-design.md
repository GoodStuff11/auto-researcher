# HVA gate set + shared Trotter coefficients

Date: 2026-09-14
Target tree: `ML-signproblem/experimenting/ed/` (Julia)

## Goal

Two coupled additions to the Trotter stack:

1. `enumerate_ferm_excitations_HVA` — a Hamiltonian-Variational-Ansatz gate set for
   the Hubbard model: strictly Hermitian (diagonal on-site gates included), no
   momentum conservation, `s_z` conserved, emitted in a fixed brick-wall order.
2. Optional **coefficient sharing** in the trotterization: several gates driven by a
   single variational parameter, with correct gradients.

The two are coupled because an `FGate` is a single product operator `C†_I C_J`, not a
sum. The spin-up and spin-down halves of a hopping term, `c†_{iup}c_{jup}` and
`c†_{idn}c_{jdn}`, therefore cannot be one gate. Sharing is what ties them back
together, so the feature is load-bearing for the ansatz, not a convenience.

## Part 1 — `enumerate_ferm_excitations_HVA`

### Signature

```julia
enumerate_ferm_excitations_HVA(Lvec; use_pbc::Bool=false, tie::Symbol=:full)
    -> (gates::Vector{FGate}, param_map::Vector{Int})
```

Lives in `TamFermion.jl` next to `enumerate_ferm_excitations`, and is added to the
module's export list and to the re-export list in `trotter.jl`.

### Conventions

Per `file_conventions.txt`, `Lvec = (Lx, Ly)` with `s = y + Ly*x`, so **axis 1 is
horizontal (x)** and axis 2 is vertical (y). Site `s` (1-based) occupies bit `s-1`;
the mask is `DtMb(1) << (s-1)`, matching `int2occ` and `fermionNNHopping`.

### Gate order

The ordering *is* the ansatz. `sortGatesByIJ` is deliberately **not** called — unlike
`enumerate_ferm_excitations`, which sorts before returning.

1. **On-site**, sites `1:N`, one gate each:
   `FGate(m_i, m_i, m_i, m_i)`. These hit the `s_I == s_J` diagonal branch of
   `tau_g_operator_sector`, giving the `n_{iup} n_{idn}` interaction.
2. For each axis `a in 1:ndim`, for each parity `p in 0:1`: the hopping bonds along
   axis `a` whose source-site coordinate along `a` has parity `p`.

For 2D this is exactly: on-site, horizontal-A, horizontal-B, vertical-A, vertical-B.

Bonds come from `findForwardLatticeNeighbors(Lvec; use_pbc)`, skipping `0` entries
(missing OBC neighbors). Parity is `unravel_c(i-1, dims)[a] % 2` for source site `i`,
so parity-0 bonds are `(0,1),(2,3),...` and parity-1 are `(1,2),(3,4),...`. Within one
parity group the bonds have disjoint support and therefore commute.

Each bond emits **two** gates, up then down:

```
FGate(m_i, m_j, 0, 0)      # spin up
FGate(0, 0, m_i, m_j)      # spin down
```

each canonicalized so that `(cre_up, cre_dn) <= (ann_up, ann_dn)`, matching the
h.c.-deduplication convention `enumerate_ferm_excitations` applies. Canonicalizing
swaps `(cre, ann)`, which leaves `tau_g = C†_I C_J + C†_J C_I` unchanged.

### Boundary conditions

`use_pbc` defaults to `false`. `use_pbc=true` combined with an odd-length axis throws:
a periodic ring of odd length admits no 2-coloring into commuting halves, so the
"two commuting groups per axis" invariant would silently break.

Known limitation, accepted: an OBC ansatz used against a PBC Hubbard Hamiltonian has a
fidelity floor from the missing wrap bonds.

### Tying schemes

`tie` selects the returned `param_map`:

- `:full` (default) — on-site is one group; each (axis, parity) hopping block is one
  group, shared across both spin channels. 2D gives 5 groups.
- `:none` — `param_map = collect(1:num_gates)`, every gate independent.
- `:spin` — on-site gates get one parameter per site; each bond gets one parameter
  shared by its up/down pair.

Empty groups are dropped and group ids renumbered contiguously, so
`n_params == maximum(param_map)` always holds. Empty groups occur for a 1D `Lvec`
(no axis 2) and for `L == 2` under OBC (parity 1 has no bonds).

### Validation

The HVA gate set contains diagonal gates, which `tau_g_operator_sector` maps to the
zero operator when `antihermitian=true`. The loss/optimize entry points therefore
throw when `antihermitian=true` and any supplied gate is diagonal
(`g.cre_up == g.ann_up && g.cre_dn == g.ann_dn`).

## Part 2 — Shared coefficients

### The mathematical fact this rests on

Sharing is a linear reparameterization. Let `a` be the per-step coefficient vector the
circuit already consumes, of length `M = P * num_gates`, and `theta` the reduced
parameter vector, of length `P * n_params`. Then

```
a = E * theta        (a gather)
d loss / d theta = E' * (d loss / d a)        (a scatter-ADD)
```

with

```
a[(l-1)*num_gates + j] = theta[(l-1)*n_params + param_map[j]]
```

for layer `l in 1:P` and gate `j in 1:num_gates`. The chain rule gives the scatter-add
regardless of whether the gates sharing a parameter commute; commutation matters for
the ansatz being a faithful HVA layer, not for gradient correctness.

### Consequence for the implementation

Because the reparameterization sits strictly *above* the circuit, the checkpoint,
rematerialization, and GPU adjoint code is **unchanged**. `trotter_evolution.jl` and
`trotter_gpu_kernels.jl` are not modified. They continue to receive a full length-`M`
vector.

Rejected alternative: threading `param_map` down into `apply_unitary_checkpoints` and
`backward_adjoint_propagation`. It would force the loops to carry two distinct indices
where they currently carry one, and the existing local name `param_idx` in those loops
is in fact a *gate* index used to look up `gpu_ops` — the two coincide today, and
under that alternative they would not. Keeping expansion above the circuit preserves
the coincidence and removes the class of bug entirely.

### New helpers

In `trotter_core_optimization.jl`:

```julia
num_shared_params(param_map, num_gates) -> Int
expand_shared_coefficients(theta, param_map, num_gates, P) -> a
contract_shared_gradient(grad_a, param_map, num_gates, P) -> grad_theta
```

`contract_shared_gradient` allocates with `zeros`, never `Vector{Float64}(undef, .)`,
and accumulates with `+=`. `param_map === nothing` short-circuits all three to
identity behavior.

### Wiring

Exactly four call sites, in `trotter_loss.jl`:

- `adjoint_loss` — add `param_map` kwarg; expand `theta -> a` before `apply_unitary`.
- `rrule(adjoint_loss, ...)` — expand before `apply_unitary_checkpoints`; in the
  pullback, call the unmodified `backward_adjoint_propagation` to get `grad_a`, then
  `contract_shared_gradient`. The pullback's `NoTangent()` count is unchanged, since
  `param_map` is a keyword, not a positional argument.
- `energy_loss` and its `rrule` — identically.

In `trotter_core_optimization.jl`:

- `optimize_unitary` gains a `param_map` kwarg. `M_full = num_exponentials *
  num_shared_params(param_map, length(gates))`. It is passed to both losses and used to
  expand before the final diagnostic `apply_unitary` call.
- `grow_coefficients` gains an `n_params` argument (defaulting to `num_gates`) and
  grows in units of `n_params` per layer.

Everything downstream needs no change once `M_full` is right: `active_indices` /
`embed_active_params`, the multi-start random sampling, the `zeros(M)` baseline-loss
call, and the metrics arrays all operate on the reduced `theta`.

`apply_unitary` also gains a `param_map` kwarg that expands up front, so direct callers
outside the loss functions behave consistently.

## Testing

One new script, `testing/test_hva_gates_and_param_map.jl`, run with plain `julia`,
following the existing ad-hoc convention in `testing/`. Lattices `(2,2)` and `(3,2)`,
sectors `(n_up, n_dn) = (1,1)` and `(2,1)`, CPU path.

1. **Gate-set structure** — per-group gate counts; every `tau` is Hermitian as a dense
   matrix; on-site gates are diagonal; all pairs of gates *within* a parity group
   commute (dense commutator norm ~ 0); `param_map` is a contiguous surjection onto
   `1:n_params`.
2. **Untied sharing is a no-op** — `param_map = collect(1:num_gates)` reproduces
   `param_map = nothing` for both loss and gradient to machine precision.
3. **Tied forward matches expanded untied** — `loss(theta; param_map)` equals
   `loss(expand_shared_coefficients(theta, ...); param_map=nothing)`.
4. **Gradient correctness** — analytic `grad_theta` from the rrule versus central
   finite differences of the tied loss, and versus the group-wise sum of the untied
   `grad_a`. Run for both `:overlap` and `:energy`, and for both `P=1` and `P=2`.
   `P=2` is the case that catches a per-layer indexing error.
5. **`grow_coefficients`** round-trips under a tied map.

## Out of scope

- The Python `TamFermion.py` copies and the `UCC_Hubbard/Current workflow/` Julia copy.
- GPU-path testing (the GPU code is unmodified; correctness follows from the CPU tests
  plus the fact that expansion happens above it).
