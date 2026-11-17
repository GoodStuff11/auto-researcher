# R12.4 (multi-source response, MSR) — T1b pre-registered rule

Copied verbatim from the leaf `.subproblems/P3.1_order_parameter_error.md` before `t1b_r12_4.py` was run (2026-11-17).

Round 12 Route Portfolio, row 12.4:
> T1b at $\Delta=0.05t$ on 5×2 and 4×3: KILL if MSR-12 tracks at most OBS $+0.1$ on both; it must also report the coherent competitor weight (signal only)
> Next action: KILL → the noise floor is not lowered by more sources; PASS → T2 on the real states with 12 exact responses

Card R12.4 (SHELVED), "Why shelved":
> Its next step was a login-node T1b at $\Delta=0.05t$ on $5\times2$ and $4\times3$ (at most 10 CPU-minutes, capped cost about 0.2 CPU-h): KILL if the 12-source chain tracks at most the 3-source chain plus 0.1 on both. [...] Predicted outcome, recorded instead of a run: KILL with $p\approx0.6$.

Implementation notes (fixed before the run):
* Instances, gap surgery, trial states, grade-R noise (0.3, 0.5), chains, scoring (scored $|\delta|\ge0.1|a_0|$; tracked ratio in $[1/2,2]$) are those of `../P3.1_order_parameter_error_r12/t1_r12.py`, reused unchanged.
* PASS = not KILL (MSR-12 > OBS + 0.1 on at least one of the two instances). INCOMPLETE if an instance has fewer than 6 scored items (the T1 minimum).
* Only the OBS (10.1d, T = 4) and MSR-12 (T = 2) arms are run, to fit the 10 CPU-minute cap; CPU caps 110 s (5×2) and 470 s (4×3).
* Coherent competitor weight: $\sum_i (\langle\psi_1|\mathrm{unit}(y_i)\rangle/N_i)^2$ in pass 1, with $N_i$ the norm of the noisy vector before normalisation; reported beside the total $\sum_i|\langle\psi_1|\tilde y_i\rangle|^2$.
