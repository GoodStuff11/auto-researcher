# P5.1 round 5, R51.16 `cmp`: guided fixed-step SR against plain SR / MinSR at equal total samples (pre-registered)
Written 2026-11-17 before any arm below ran (only a 3-step timing pre-flight, `logs/preflight_born16k.txt`).

Why. The leaf question asks for fewer total samples than SR or MinSR need for the same accuracy. Every arm so far is
guided SR; MinSR is the same step as SR from the same samples (push-through identity, P4.1 R33), so "SR or MinSR" is
one comparator: plain SR with Born (|psi|^2) draws, at the samples per step and step length a practitioner would pick.
R51.1 showed that at equal total samples fewer samples per step win, so the comparator must include a small-N,
many-step Born arm (MinSR's usual regime is N < P; at N = 16,384 > P = 2637 the step is still MinSR's), and a
larger step length (practitioners tune the learning rate). This test uses only the 4x3 instance, where guided arms
already exist (R51.1), so it does not depend on R51.8's running 5x2 job.

Instance: open 4x3 Hubbard, 5+5, U = 8, t' = -0.25, `che` (P = 2637), R51.1's SR-trained start; R = fraction of the
start's gap to E0 removed, exact scoring each step (as R51.1); R_pol = 0.7475; target R* = 0.90 R_pol = 0.6728.
Budget per arm 13.5M draws (= 2 x 6.75M, guided B's total samples to R*).

Arms (`ns_run.py`, exact i.i.d. draws, shift 1e-3, seeds 9000 + step):
* born N 16,384, tau 0.005, 824 steps     * born N 65,536, tau 0.005, 206 steps
* born N 65,536, tau 0.010, 206 steps     * born N 262,144, tau 0.005, 51 steps
* guided N 16,384, tau 0.005, continued from R51.1 arm A's step-120 theta for 292 more steps (412 in all = 6.75M),
  seeds 9121 + step: the small-N guided arm R51.1 stopped at 120 steps.
Statistic: T(arm) = N x first step with R >= R* (censored at the arm's budget). T_born = min over born arms;
T_guided = min(6.75M (R51.1 B), T(guided 16k continued)).

Verdicts (`score_cmp.py`):
* GUIDED-FEWER if T_born >= 2 T_guided (a born arm censored at 13.5M counts as >= 13.5M);
* NO-GAIN if T_born <= 1.25 T_guided;  MIXED otherwise.
Secondary (reported, not scored): the same at 0.80 R_pol; R after the full budget per arm.
Next actions: GUIDED-FEWER → the comparator leg of "What would close it" is met at 12 sites; the same comparison is
then read from R51.8's 5x2 born/guided arms (its own rule) and repeated at 6x2. NO-GAIN → guided fixed-step SR is
not an answer to the Question; the leaf's optimiser search moves to the sampler (R51.11) and variance reduction.
MIXED → the 5x2 comparison (R51.8) decides which.
What it does not show: one seed per arm; one instance; tau not scanned below 0.005 for Born; exact i.i.d. draws.
