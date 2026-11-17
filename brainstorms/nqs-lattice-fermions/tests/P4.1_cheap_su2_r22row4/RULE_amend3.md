# P4.1 r22: pre-registered amendment 3 (ledger row 4, cheaper version: rho_abl only)

Written 2026-11-17, round 25, after the per-leaf critic (round 23, objection 1) found the r22 resume refusal was not a
G2 refusal (p 0.55 < 0.8), and **before any further polish step**. It amends `../P4.1_cheap_su2_r22/RULE.md`,
`RULE_amend1.md` and `RULE_amend2.md` as below; everything else stands.

## Disclosure

Written after seeing the 742206 trajectories and the provisional ratios (rho_pol 0.586, rho_abl 0.550), and the
RULE_amend2 fits (rank5 r 0.764, eps_inf 5.98e-4; pcg r 0.910, RMS 0.682). It drops the che arm and rho_pol from this
job (rho_pol stays provisional; che's tail, r about 0.97, makes it the expensive and predictably AMBIGUOUS half).

## Amendment

1. **Scope.** Only rho_abl = eps_pcg / eps_rank5 is read. The che arm is not run. rho_pol is reported as its 742206
   provisional value and decides nothing.
2. **rank5** runs unchanged to STOP-RULE (predicted gain < 1e-3 g_pcg), STOP-NOGAIN or the 30-step hard cap
   (RULE_amend1), with the RULE_amend2 guard. If its fit is UNSETTLED at the cap: AMBIGUOUS, no pcg steps, no further
   compute on rho_abl at 12 sites.
3. **pcg by dominance.** Ladder steps are accepted only if they lower the exact energy, so eps_pcg never rises, and
   the tail-extrapolated eps_pcg_inf <= eps_pcg_end. With L5 = min(eps_rank5_end, eps_rank5_inf), once
   eps_pcg <= 0.60 L5, both rho_abl (endpoints) and rho_abl (extrapolated) are <= 0.60 for every later pcg state, so
   PAIRING-KEPT is fixed (STOP-DOMINANCE) and pcg's own tail fit is not needed. pcg is advanced one ladder step per
   call (ladder.py `--steps n+1`; the STOP-CAP labels below 30 this writes are bookkeeping) until STOP-DOMINANCE,
   a pcg STOP-RULE / STOP-NOGAIN, or its 30-step hard cap; in the last two cases rho_abl is read with the unchanged
   thresholds and RULE_amend2 guard (UNSETTLED -> AMBIGUOUS).
4. **No dominance shortcut toward RETRACTED**: pcg can only fall, so PAIRING-RETRACTED and ABL-CONFOUNDED are read only
   at a finished pcg arm (item 3).
5. Unchanged: ladder protocol, thresholds (PAIRING-KEPT <= 0.60, PAIRING-RETRACTED >= 0.85), INVALID and INCOMPLETE.

## What each outcome does (the r22 joint table restricted to its rho_abl column)

* PAIRING-KEPT: R2's gain over its own rank-5 truncation is attributed to pairing at 12 sites, on polished arms;
  the summary's pairing credit uses this like-for-like ablation (critic objection 13). R2 stays SHELVED (no 24-site
  optimiser), its "mechanism" flag is cleared. Ledger row 4 closes (rho_abl half); the rho_pol half is closed as
  AMBIGUOUS-predicted, not run.
* ABL-CONFOUNDED: R2 flagged "mechanism unattributed" at 12 sites; row 4 closes.
* PAIRING-RETRACTED: R2's pairing claim retracted (advantage is the backflow form B1 f0 B2^T); row 4 closes.
* AMBIGUOUS / INCOMPLETE / INVALID: no change.
