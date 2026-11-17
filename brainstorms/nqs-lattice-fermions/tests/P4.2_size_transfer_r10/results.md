# P4.2 r10: truncated-metric LM polish of the 704844 SR-3000 states — NOT RUN (blocked)

**Status: BLOCKED, no verdict.** Rule in `RULE.md` (written first) is unchanged and unapplied.

## Why
The 704844 SR-3000 states were never saved. `results_t2/sr_*.json` (written by
`P4.2_size_transfer_r5/critic_capacity.py train`) hold `params` = the CLI namespace (`vars(a)`),
plus E, relerr and the history; `train()` never writes theta. No `.npy` or checkpoint exists.
Only the LM-7000 s runs (`lm_*.json`) carry `theta`. The leaf's "saved 3000-step SR states" premise is wrong.

## Cost to unblock (over the 10 CPU-minute cap, so stopped as instructed)
Regenerate the three SR-3000 runs with theta saved (`regen_sr.py`, same code path, deterministic
init rng 2000, exact sums; it checks E against the 704844 value). Single-thread wall times from the 704844 logs:
base 2652 s, spin_jas2cs 4215 s, c8h32 6968 s, so about **3.85 CPU-h**. The polish then needs about 1 CPU-minute
(1 to 2 CPU-s per step, 8 steps, 3 arms). Tier 1-10 CPU-h; must go through compute_gate.py.
All three arms are needed: HOLDS needs every arm above 1.02%, and P4.1's polish removed 54% of a gap,
which would take even base (1.294%) below 1.02%.

## What was run (smoke tests only, about 10 CPU-s)
`polish.py` on the saved LM-7000 s base theta (`logs/smoke_polish_on_lm_base*.txt`): reload energy matches
to 0; at cutoff 1e-5 (kept rank 14) and 1e-8 (rank 34, matching the leaf's effective tangent rank) the
predicted and realised gain is 0 to the printed digits (relerr stays 1.7784%); at 1e-11 the full step raises
the energy (26%, 6.4%, 2.1% at s = 1, 1/2, 1/4). So the LM-trained base is already a truncated-LM
stationary point at 1.778%, worse than SR-3000's 1.294%. That shows the polish code works. It does not
answer the question, which is about the SR states.

## Ready to run once states exist
    python3 regen_sr.py --variant {base,spin_jas2cs,c8h32}          # via compute_gate, ~3.85 CPU-h total
    python3 polish.py --variant V --theta-json results/sr3000_V.json  # ~20 CPU-s each
    python3 polish.py --aggregate
