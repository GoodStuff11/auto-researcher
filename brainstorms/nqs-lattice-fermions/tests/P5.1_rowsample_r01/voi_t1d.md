# VOI note: P5.1 round 2, R51.3 `levrow` T1d comparison arm (`submit_t1d_cont.sbatch`; rule RULE_T1d.md)

* leaf: P5.1
* route: R51.3
* tier: T1
* decision: R51.3's leverage-row steps continued from T1c's endpoint (eps 6.111e-4) until the exact ladder's eps_L = 5.848e-4 (measured interactively: 3 ladder steps, 8.16 CPU-min, gamma_L = 3.23e-6 per CPU-min). PASS (rho >= 3) sends R51.3 to its sampled-VMC version (leverage-weighted draws) at 12 then 24 sites; KILLED (rho < 1.5) sends it to translation repair; MARGINAL leaves T1c's MARGINAL standing and R51.3 waits on R51.1 unit 1.
* prediction: PASS.
  * Probabilities: PASS 0.80, MARGINAL 0.15, KILLED 0.05.
  * Basis: the drop to reach is 2.63e-5; T1c's last 5 steps gained 7.6e-6 to 1.0e-5 each at 0.44 CPU-min, so 3-4 steps (1.4-2.0 CPU-min with the amortised 0.063 per step) are expected, rho about 3.5-5. It falls below 3 only if the gain per step drops under about 6e-6 immediately, or the 10-step-old scores (built at the 742206 endpoint) stop working (T1b: 20-step-stale scores kept phi 0.95).
* p: 0.80
* prediction-changes-decision: yes. T1c's comparator was measured at a higher eps and the like-for-like ladder rate turned out 2.3 times lower (0.44 gamma_LM), so whether R51.3 clears 3x against it is the one number between MARGINAL and a sampled-VMC build; the leaf's next spend (a 12-site sampled implementation, several CPU-h) depends on it, and an 80% prediction is not enough to commit that spend.
* cheapest-kill-tried: T1c data reused as far as possible: it ends exactly at the start point, so no stored steps cover the interval; extrapolating T1c's falling gains (the prediction above) is the T0 version and is what this run checks. The ladder baseline was run interactively within the 10 CPU-min cap, which left no room for this arm.
* cost-cpuh: 0.25 (cap 15 min, 1 task; expected use 0.05)
* cheapest-version: the run stops at the first step reaching eps_L (expected 3-4 steps, about 2 CPU-min); 20 steps is only the ceiling. No smaller instance: the comparison is defined at this state.
* convergence: not required; the rate is measured over a fixed eps interval. If eps_L is not reached in 20 steps the run reports the rate over the steps done, flagged.
* wall-time-basis: measured in T1c on 1 thread: 0.44 CPU-min per step, setup and start energy 0.1; 20 x 0.44 + 0.1 = 8.9 min, x1.3 = 11.6 -> --time 00:15:00.
* power: deterministic seeds; T1c's step-to-step gain scatter is about 20%, so rho's uncertainty from one draw sequence is about +-0.7 at rho 4; the 3x threshold is about 1.5 sd below the central prediction.
