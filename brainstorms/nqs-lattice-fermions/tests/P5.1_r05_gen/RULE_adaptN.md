# P5.1 round 5, R51.15 `adaptN` T1 (stored data, pre-registered)
Written 2026-11-17 before `envelope.py` ran.

Question. Would a samples-per-step schedule (small N early, larger N as the per-step gain shrinks; dynamic sample
size / critical batch size) beat the best fixed N of R51.1 in total samples to a target fraction of R_pol = 0.7475?

Statistic. From R51.1's stored arms A (16,384), B (65,536), C (262,144), D (16,384, tau/2) at 4x3, n_N(R) = samples
to first reach R (running max). Envelope = sum over 59 bins of R in [0.05, target] of the cheapest arm's samples for
that bin (Markov approximation: rate depends only on R and N). Gain g = best single arm / envelope.

The envelope is optimistic (min over noisy curves picks favourable fluctuations; it ignores that state at equal R
differs between arms), so it can kill the route but not promote it beyond T1.

* KILL if g < 1.25 at both 0.90 and 0.95 R_pol (the oracle switch saves less than a quarter).
* PASS (route goes to a prepared T2 with an on-the-fly noise-scale rule) if g >= 1.5 at 0.90 or 0.95.
* MARGINAL otherwise (kept, no compute; revisit with R51.8's 5x2 ladder data).
