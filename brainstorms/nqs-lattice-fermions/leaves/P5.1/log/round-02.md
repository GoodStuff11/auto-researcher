# P5.1 round 2 (2026-11-17): R51.3 `levrow` T1c (worker)

Task (orchestrator): run R51.3's T1c as the card pre-registered it: 10 exact-energy-scored polishing steps from the
742206 `che` endpoint with the metric from leverage-sampled rows, against the exact ladder's rate
gamma_LM = 7.27e-6 relative energy per CPU-min; PASS at $\ge3\gamma_{\rm LM}$, KILL below $1.5\gamma_{\rm LM}$; count the full cost of the
leverage scores. Also noted by the orchestrator: R51.1 unit 1 was submitted as a gated job this round.

## Rule file
`tests/P5.1_rowsample_r01/RULE_T1c.md`, written before `t1c.py` ran. Details the card left open, fixed there:
* Scores computed once at the start (full Jacobian, Gram, eigh, ridge leverage with lambda = 1e-7 w_max), no refresh
  during the 10 steps (T1b: 20-step-stale scores keep phi = 0.95). Their whole cost is charged to the run, i.e.
  amortised over the 10 steps.
* Per step: exact energy and gradient (`../P4.1_cheap_su2_r25/exgrad.py`), f = G/2; q = 0.5 Born(current) +
  0.5 ell/sum(ell); m = 8P = 21,096 i.i.d. rows (seed = step index), Jacobian only for distinct rows; S~ normalised
  by S~_00 and centred; one eigh; cutoff ladder {1e-5, 1e-6, 1e-7}, per rung first-accept metric length in
  {0.002, 0.001, 0.0005} (eps drop > 1e-7, the ladder's MIN_GAIN), lowest-energy rung accepted; STOP-NOGAIN if none.
* CPU = process time (user + sys) after setup (setup 0.06 excluded, as the ladder's per-step times exclude it);
  the validity check is excluded.
* Validity: |G/2 - f_full| / |f_full| < 1e-6 at the start.

## Run
`t1c.py`, one thread, `ulimit -t 600`, log `tests/P5.1_rowsample_r01/logs/t1c.txt`, record `results/t1c_state.json`
(also `t1c_theta.npy`, `lev0.npy`). Process 6.52 CPU-min (user 323 s + sys 69 s), maxrss 7.9 GB; all 10 steps fit,
no gated job needed.
* Scores: 1.896 CPU-min (sum ell 2162.9, as T1). Check: relative difference 1.34e-9 (VALID), 0.13 CPU-min not charged.

| step | accepted cut | tau | eps gain | step CPU-min | distinct rows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1e-7 | 0.001 | 1.66e-5 | 0.484 | 11,827 |
| 2 | 1e-7 | 0.001 | 1.19e-5 | 0.447 | 11,941 |
| 3 | 1e-6 | 0.002 | 1.34e-5 | 0.448 | 12,091 |
| 4 | 1e-6 | 0.002 | 1.01e-5 | 0.446 | 12,001 |
| 5 | 1e-7 | 0.001 | 1.27e-5 | 0.437 | 12,018 |
| 6 | 1e-6 | 0.002 | 1.02e-5 | 0.437 | 12,082 |
| 7 | 1e-7 | 0.001 | 9.76e-6 | 0.435 | 12,013 |
| 8 | 1e-6 | 0.002 | 9.56e-6 | 0.438 | 12,018 |
| 9 | 1e-7 | 0.001 | 7.56e-6 | 0.435 | 11,914 |
| 10 | 1e-6 | 0.002 | 8.90e-6 | 0.437 | 11,934 |

Every rung qualified at every step; rungs 1e-5 and 1e-6 accepted tau 0.002 at the first trial, rung 1e-7 always
needed the second trial (0.002 rose, 0.001 gained), so each step spent 4 exact energies (0.18 CPU-min) on the ladder.

## Verdict
eps 7.21736e-4 -> 6.11115e-4 (drop 1.106e-4) in 6.34 charged CPU-min (scores 1.90 + steps 4.44).
gamma = 1.745e-5 per CPU-min = **2.40 gamma_LM → MARGINAL** (PASS needs 3.0, KILL below 1.5).
Reported only: without the score cost 3.42 gamma_LM; steps 6–10 without score cost 2.90 gamma_LM.
Refresh schedule and amortised cost: one score build (1.90 CPU-min) for the whole run, 0.19 CPU-min per step over
10 steps (30% of the charged total).

## Reading
* The leverage-row metric works as a multi-step optimiser: every step gained, at 0.44 CPU-min against the exact
  ladder's 2.86, and 7.6% distinct rows (13% draws) sufficed throughout with scores never refreshed.
* It misses 3× because (a) the score build is a fixed 1.9 CPU-min, and (b) the per-step gain halves over 10 steps
  (1.66e-5 → 8.9e-6), so the second half alone, scores free, is under 3× (2.90).
* Comparator caveat: gamma_LM was taken from ladder steps 8–12 at eps 8.3–7.2e-4; this run sits at 7.2–6.1e-4,
  where the exact ladder's gains would also be smaller, so the bar is if anything harder than like-for-like. A
  like-for-like comparator (exact ladder from the same eps) was not run.
* Post-hoc arithmetic (not a verdict): the 1e-5 rung never won and the 1e-7 rung's 0.002 trial never gained;
  dropping both saves 2 energies per step (0.35 CPU-min per step) → 2.8× on these same 10 steps. 3× then needs the
  scores reused over about 30 steps at second-half gains ($1.9/N+0.35\le0.42$).
* Options for the orchestrator: (1) pre-register a trimmed-ladder run of at least 30 steps on one score build (about
  12–13 CPU-min, so a gated quick job, or two interactive resumes); (2) a like-for-like comparator (3 exact-ladder
  steps from 6.1e-4, about 9 CPU-min); (3) defer R51.3 until R51.1 unit 1 says whether conditioning matters at all.

## Compute
T1c: 0.11 CPU-h interactive (6.52 CPU-min); see the T1d section for the round total.

## T1d (coordinator follow-up, option 2): like-for-like comparator from the same point
The T1c verdict stays MARGINAL and is not rescored. Rule `tests/P5.1_rowsample_r01/RULE_T1d.md`, written before
either arm ran: both arms start at T1c's endpoint (eps 6.11115e-4). Baseline = the exact cutoff ladder with
`../P4.1_cheap_su2_r22/ladder.py`'s step unchanged, as many steps as fit under a 9.4 CPU-min cap. Comparison =
R51.3 continued with T1c's protocol unchanged (seeds 11, 12, ...), T1c's scores reused without refresh, score cost
charged as the amortised 30-step reuse (1.896/30 = 0.0632 CPU-min per step), until eps $\le$ eps_L (cap 20
steps). rho = gamma_R / gamma_L; PASS at $\rho\ge3$, KILL below 1.5, else MARGINAL. Reported only: rho with the
score build charged in full.

### Baseline arm (interactive, `t1d_ladder.py`, `logs/t1d_ladder.txt`, `results/t1d_ladder.json`)
| step | cut | s | eps after | step CPU-min |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1e-6 | 1.0 | 6.01188e-4 | 2.76 |
| 2 | 1e-7 | 0.5 | 5.97888e-4 | 2.73 |
| 3 | 1e-6 | 1.0 | 5.84794e-4 | 2.68 |

eps_L = 5.84794e-4 after 3 steps, 8.16 CPU-min of steps; gamma_L = 3.225e-6 per CPU-min = 0.44 gamma_LM. Process
8.22 CPU-min (user 366 s + sys 130 s), maxrss 9.3 GB. The ladder's per-step gain here (9.9e-6, 3.3e-6, 1.3e-5) is
about half its 742206 last-5 average (2.1e-5), confirming T1c's comparator caveat.

### Comparison arm: prepared as a gated quick job (does not fit: 8.2 of the 10 CPU-min cap used)
`t1d_cont.py`, `submit_t1d_cont.sbatch` (1 task, 8G, 00:15:00), `voi_t1d.md`. Gate check (`--pre-audit`):
`cap = 1 array x 0.25 h x (1 tasks x 1 cpus) = 0.2 CPU-h [quick note]`; leaf committed 2.3/300 CPU-h; GATE: PASS.
Not submitted. Prediction: PASS (p 0.80): drop to cover 2.63e-5, T1c's last steps gained 7.6e-6 to 1.0e-5 at 0.44
CPU-min, so 3–4 steps, rho about 3.5–5; with the score build charged in full, rho about 2.

Compute this round: 6.52 + 8.22 = 14.7 CPU-min interactive (0.25 CPU-h). Leaf total 0.60 CPU-h interactive plus
R51.1 unit 1 (gated job).
