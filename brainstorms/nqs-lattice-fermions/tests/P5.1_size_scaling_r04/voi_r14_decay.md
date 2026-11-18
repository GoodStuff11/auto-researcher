# VOI note: P5.1 R51.7 `rmavg` unshelve test at 10 and 12 sites (`submit_r14_decay.sbatch`, DRAFT; `RULE_r14_decay.md`)
* leaf: P5.1
* route: R51.7
* tier: T2
* decision: $\rho(V)=N^{\rm DEC}/N^{\rm CONT}$ at 10 and 12 sites. DECAY-HELPS with $r'_{\rm DEC}<3.52$ → the Answer's growth sentence says a decaying, averaged step grew the need less than the occupied configurations; R51.7 LIVE, next a 120-step rev-3 ladder. DECAY-HELPS with $r'_{\rm DEC}\ge3.52$ or MIXED → constant-factor saving at most, numbers in What we showed, R51.7 settled on the "none removes it" list. DECAY-NO-GAIN → R51.7 KILLED at T2 in its premise. UNRESOLVABLE/INCOMPLETE → no class.
* prediction: MIXED. Basis: at 10 sites the exact arm is a limit cycle and the 65k guided arm has nearly levelled, so the shortfall is mostly a noise ball that a 4.3× smaller final step plus averaging shrinks (R51.7's halved step removed 14% while still descending, implying a ball of about 28%): $\rho(10)$ about 0.3–0.6. At 12 sites every arm is still descending (exact z −3.7), so decay mainly slows progress: $\rho(12)$ about 0.7–1.1. MIXED 0.45, DECAY-NO-GAIN 0.25, DECAY-HELPS 0.20, UNRESOLVABLE or INCOMPLETE 0.10.
* p: 0.45
* prediction-changes-decision: yes. MIXED settles R51.7 on the "none removes it" list (today shelved with its trigger met) and puts the 10-site saving in What we showed.
* cheapest-kill-tried: R51.7's T2 (halved step, 4x3) was out of premise (all arms descending; card). No T0 argument fixes the noise-ball fraction at a levelled-off state; the continuation from existing step-120 states avoids rerunning 120 steps.
* cost-cpuh: 6.0 (1 task × 6 h × 1 CPU; expected 4–5.1), plus about 0.1 CPU-h interactive for the driver's identity test.
* cheapest-version: continuations (60 steps) from the existing step-120 states, not new 120-step runs; two rungs bracket each size's need plus one below for DEC's censoring; S10 second seeds exist already (16k, 65k); no exact arm at 12 sites (a further 5.8 CPU-h; the bounce is measured at 10 sites, where it exists).
* convergence: fixed 60-step continuation, tail window $k$ = 31–60; not a convergence criterion. The 10-site exact arm's last-40 slope is −1.7e-4 in $R$ (levelled); the 12-site arms are not levelled (in the prediction).
* wall-time-basis: in-job rates of jobs 763968, 769098, 769376 halved for 60 steps (listed in the sbatch header): 237 min × 1.3 = 308 min → 6 h.
* power: one seed at 12 sites; $\rho$ thresholds 0.5 and 0.8 are wide against the one-seed $\ln N$ noise (about 0.04–0.05).
