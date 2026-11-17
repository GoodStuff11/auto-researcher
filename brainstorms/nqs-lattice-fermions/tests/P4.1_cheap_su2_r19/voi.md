# VOI note: P4.1 rung 1 (`submit_rung1.sbatch`)

* leaf: P4.1
* route: R2 (`pcg`) with the optimiser R18'' (truncated-metric linear method)
* tier: T1
* decision: Exact verdict at N8, with rho_pol = eps_pcg / eps_che and both arms polished (RULE.md).
  * RETRACT (rho_pol >= 0.85): R2's T2a PASS is retracted, R2 is killed at 12 sites, and P4.1 turns to an optimiser route.
  * KEEP-LIVE (rho_pol <= 0.60): R2 stays live, and every 24-site comparison must use polished arms.
  * CONFOUNDED (in between): R2 stays at T2a with an optimiser-confounded flag.

  Sampled verdict, f = R_sampled(4) / R_exact(4) per arm, with 8192 samples per step:
  * SAMPLED-KEEPS (f >= 0.5 on every required arm): rung 2 is prepared. It is the 24-site warm-started sampled polish, 16 to 32 CPU-h, and needs the critic.
  * SAMPLED-LOSES: no 24-site rung runs. The next step is a 12-site T1 on the sampled optimiser (sample count and cutoff).

  RETRACT together with SAMPLED-LOSES, plus R16 and R17 dying at the polished parent, moves the leaf to EXHAUSTED at this scale.
* prediction: exact KEEP-LIVE and SAMPLED-LOSES.
  * Exact marginal: P(KEEP-LIVE) about 0.65, P(CONFOUNDED) about 0.25, P(RETRACT) about 0.10.
    * The local probe step gave R = 0.240 for `pcg` against 0.269 for `che` at the same protocol.
    * If `pcg` follows `che`'s trajectory, eps_pcg(N8) is about 0.5e-3 against eps_che(N8) of about 1.2e-3, so rho is about 0.45.
  * Sampled marginal: P(SAMPLED-LOSES) about 0.97. This is now nearly certain.
    * The local smoke test of the job's own sampled `che` step 1 (same seed, deterministic) raised the exact energy by 4.9 gaps, R = -4.906. The in-sample reweighted energy claimed a drop of 0.13 Ha over the same step. With 8192 samples for 2637 parameters, the step overfits its own samples.
    * Three more fooled steps would have to climb from R = -4.9 to R >= 0.27.
* p: 0.62
* prediction-changes-decision: yes, through the exact part. KEEP-LIVE moves R2 from "T2a, optimiser-confounded" to "live, with polished arms required at 24 sites". RETRACT kills R2. Either outcome changes the leaf.
  * The sampled part is predictable (p about 0.97). Its predicted LOSES leaves the plan unchanged: no rung 2 without a pass.
  * It stays in the job only because it adds about 10 CPU-min to an exact part that must run anyway, and because it turns the step-1 observation into the pre-registered 4-step verdict.
  * The orchestrator may delete the sampled lines and record SAMPLED-LOSES as predicted from step 1. The exact verdict is unaffected.
  * The sampled `pcg` arm is skipped automatically in that case (RULE.md amendment 1).
* cheapest-kill-tried: No T0 argument fixes rho_pol. How much slack a trained state has is not set by its architecture, and the Round 10 constant-direction one-step test did not separate the arms: `pcg` 6.7% against `che` 5 to 8%.
  * Local T1 probe: one exact `pcg` step, 3 CPU-min, logs/probe_pcg_1step_1thread.txt, R = 0.240. This shows that `pcg` has slack comparable to `che`'s, but one step cannot fix the N8 ratio: `che` went from 0.27 to 0.54 in 4 steps.
  * For the sampled part, the T0 noise-floor argument favours LOSES. At about 3 samples per parameter, S is noise-dominated below about 1e-2 of its top eigenvalue. The local T1 smoke test, sampled `che` step 1 (2.3 CPU-min, logs/probe_sampled_che_1step_1thread.txt), all but decides it: R = -4.906. The in-sample correlated-sampling line search accepted a step that its own samples scored as a 0.13 Ha drop.
* cost-cpuh: 1.1
* cheapest-version: one seed, two arms, one instance, and sampled `pcg` only when needed; nothing smaller decides the rule.
  * Arms: only the two that the rule uses, `che` and `pcg`. The r12 draft's `r1` and `che_k8` arms are dropped.
  * Seeds: one, because the exact sums are deterministic and the sampled seeds are fixed.
  * Instance: 12 sites, the smallest instance on which the T2a ratio exists.
  * The sampled `che` arm decides rung 2 on every exact verdict. The sampled `pcg` arm runs only when the exact verdict keeps R2 alive.
  * The 8 exact steps are the leaf's "up to 8". `che` adds only steps 5 to 8, resumed from Round 11.
  * The whole run is about 64 CPU-min. That exceeds the 10 CPU-min interactive allowance, so it cannot run as a local T1.
* convergence: eight exact steps should land within about 0.02 of each arm's plateau, judged from the measured per-step gains below.
  * In Round 11, `che`'s predicted gain per step fell as 0.378, 0.218, 0.087, 0.031. Steps 5 to 8 should therefore add at most about 0.05 of its gap.
  * `pcg`'s step 1 gained R = 0.240 against a predicted 0.259, the same pattern. Eight steps should land within about 0.02 of each arm's plateau for this protocol.
  * The sampled part is fixed at 4 steps by the rule.
* wall-time-basis: measured on 1 thread. An exact step takes 3.0 min (logs/probe_pcg_1step_1thread.txt, 178 s wall, 9.1 GB peak RSS). Sampled `che` step 1, including the exact S/H check, takes 2.3 min (logs/probe_sampled_che_1step_1thread.txt).
  * The work is 8 `pcg` plus 4 `che` exact steps (36 min), 4 sampled `che` steps (9.2 min), and about 4 min of set-up and verdicts: about 49 min.
  * x1.3 is 64 min, so --time 1:05:00 on 1 CPU, with --mem=16G.
* power: the exact part is deterministic and decided by one run; the sampled part has one fixed seed per arm, with P(f within 0.1 of the bar) about 0.2.
  * The exact part is deterministic, with no run-to-run spread. P(CONFOUNDED) is about 0.25 from the prior on rho, and P(INCOMPLETE) is about 0.05 (wall time).
  * The sampled part uses one fixed seed per arm, and its seed-to-seed spread is unmeasured. P(f within 0.1 of the 0.5 bar) is about 0.2. In that case a second seed is the next release, not a re-read.
