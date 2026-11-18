# P5.1 round 5, R51.16 `cmp`: guided fixed-step SR against plain SR at equal total samples (pre-registered)
Written 2026-11-17 before any arm ran (only a 3-step timing pre-flight, `logs/preflight_born16k.txt`).

REVISION 1 (2026-11-17, before any arm ran; compute auditor, CHANGES REQUIRED on digest da2f6097605b):
(a) scope: the verdict is about plain SR with Born draws. MinSR gives the same step from the same samples at N >= P
(push-through identity, P4.1 R33), which covers every arm here (P = 2637, all N >= 16,384). MinSR at N < P is NOT
tested: at 4x3 an N <= 2048 arm needs about 7,000 steps to spend the budget, and at 5x2 the only N < P arm (Born
1024, R51.8) ended above E_start. MinSR at N < P moves to a larger-P instance, a later unit;
(b) the Born 262,144 arm and the Born 65,536 tau 0.005 arm are dropped (262k is certain to be censored: guided 262k
needs 55+ steps, Born is worse than guided at equal N, and kappa_born / kappa_guided is 1.2e4 at 4x3 against 1.9e3 at
5x2, where Born needed about 9x guided's samples);
(c) first passage is a sustained crossing (`score_common.py`): T = N x t*, t* the first step at which the trailing
20-step mean of R is >= R*. R51.1's arm B re-scored with the same rule (`results/rescore_R51.1_sustained.json`):
T_B = 7.41M at R* = 0.90 R_pol (4.33M at 0.80); A and D never cross 0.90 within 120 steps;
(d) budgets and the guided continuation re-sized to that T_B; R51.11's arms moved to their own job and rule.
R51.11's T2 (`RULE_qopt_T2.md`) uses this rule's T_guided.

Why. The leaf question asks for fewer total samples than SR or MinSR need. Every arm so far is guided SR. R51.1
showed that at equal total samples fewer samples per step win, so the comparator includes a small-N, many-step Born
arm and a larger step length (practitioners tune the learning rate). 4x3 only, where guided arms exist (R51.1).

Instance: open 4x3 Hubbard, 5+5, U = 8, t' = -0.25, `che` (P = 2637), R51.1's SR-trained start; R = fraction of the
start's gap to E0 removed, exact scoring every step; R_pol = 0.7475; R* = 0.90 R_pol = 0.6728.

Arms (`ns_run.py`, exact i.i.d. draws, shift 1e-3):
* guided N 16,384, tau 0.005, continued from R51.1 arm A's step-120 theta for 332 steps (seeds 9121 + step); the
  sustained window runs across the join (A's 120 steps + the continuation = 452 steps = 7.41M draws).
* born N 16,384, tau 0.005, 904 steps (14.8M = 2 T_B), seeds 9000 + step.
* born N 65,536, tau 0.010, 226 steps (14.8M), seeds 9000 + step.
T_guided = min(T_B, T(A + continuation)); T_born = min over the two born arms (censored at 14.8M).

Verdicts (`score_cmp.py`):
* GUIDED-FEWER if T_born >= 2 T_guided (a born arm censored at 14.8M counts as >= 14.8M, which is >= 2 T_guided);
* NO-GAIN if T_born <= 1.25 T_guided;  MIXED otherwise.
Secondary (reported, not scored): the same at 0.80 R_pol; R at the end of each arm.
Next actions: GUIDED-FEWER → the comparator leg of "What would close it" is met at 12 sites for plain SR (MinSR at
N < P still owed); NO-GAIN → guided fixed-step SR is not an answer; the leaf moves to sampler design (R51.11);
MIXED → read R51.8's 5x2 born/guided arms by its own rule.
What it does not show: MinSR at N < P; one seed per arm; one instance; Born step length scanned at two values only;
exact i.i.d. draws.
