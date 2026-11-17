# P4.1 round 22: R38 `guided` multi-step T1 (pre-registered)

Written 2026-11-17 after the one-step screen (RULE_B: V11 Rbar 0.086 against base 0.016, PASS-T1) and after R21's
result (job 748762: `shift` arm R(15) = 0.195, 6.1 CPU-min), before any guided multi-step run.

Protocol: identical to R21's `shift` arm under RULE_amend1 (che theta0, 15 steps, N = 16,384, absolute shift 1e-3,
held-out metric-length ladder {0.005, 0.01, 0.02, 0.05} scored by correlated sampling on a fresh Born-distribution
set, step skipped if no length lowers the held-out energy), with one change: the samples that build S_T and f are
drawn from q = 0.8 p + 0.2 p_flat (p ~ size*pr^2, p_flat ~ size*|pr|), seed 5000 + step, with self-normalised
importance weights p/q. Same cost per step (one O row per sample). Code: `guided.py` (imports r24 `trsr.py` read
only and replaces its sr_stats). Interactive, about 6-7 CPU-min.

Rule (against R21's `shift` R(15) = 0.195):
* GUIDED-WINS if R_guided(15) >= 0.295 and no step raises the exact energy by more than 0.02 of the gap.
* NO-GAIN if R_guided(15) <= 0.225.
* MARGINAL otherwise.
Reported only: the step at which R first reaches 0.195 (sample-cost ratio to R21's 15 steps).
GUIDED-WINS makes guided SR the leaf's sampled optimiser candidate (next: the same comparison from the `pcg`
endpoint, then the 24-site protocol with the critic); NO-GAIN kills R38 at T1.
