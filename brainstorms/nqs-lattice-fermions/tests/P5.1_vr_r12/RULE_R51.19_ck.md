# RULE — R51.19 per-component c_k (critic round 15, Minor 9). Written 2026-11-18 before any c_k number was computed.
1. Statistic: rho_diag = kappa(g + diag(c) D)/kappa_guided at the oracle (exact-sum, guided-weighted) optimum of the metric-weighted quantity over all per-component c_k (`vr_ck.py`, same states as T1); per-component regression c_k = -cov_w(g_k,D_k)/var_w(D_k) and rho_k = 1-R_k^2 reported alongside.
2. REOPEN (only a sampled-fit T1 is owed) if rho_diag at the 6x2 endpoint meets the T1 PASS threshold eff_6 = 3 rho_6 <= 0.5, i.e. rho_6 <= 0.167; starts and 5x2 reported only.
3. KILL STANDS otherwise, and is final: exact per-component c_k is an upper bound on what any sampled fit achieves.
