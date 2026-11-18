# Round 11 — scoring of 769376 (R51.8 unit 1c, 6×2 ladder), 2026-11-18

Worker pass; no commits, no Slurm. Job 769413 (P4.1) untouched. Both array tasks finished (`unit 1c task 0/1 done (STOP=0)`);
all seven arms `status: DONE`, 120 steps each. Ledger: 10.27 CPU-h used of 14.33.
Scorer: `tests/P5.1_size_scaling_r04/score_r08.py` (new, read-only on data) → `results/score_r08.json`. Rule:
`RULE.md` revision 2 plus the rev-2 addendum (detrended sd_w) and the 2026-11-17 addendum (first-crossing comparator,
censored Born), applied as written.

## Reference (6×2, open, U = 8, t' = −0.25, 5+5, (+,+), P = 2637, 156,816 representatives)
* $E_{\rm start}=-7.605891$, $E_0$ (lowest (+,+) level, `info_S6x2_N10.json`) $=-7.614591$: gap to $E_0$ **0.0086997**.
* Exact arm: $E_{\rm win}$ (mean steps 61–120) $=-7.612902$, so $G_{\rm ref}=0.0070107$ (0.806 of the gap to $E_0$);
  last-20 mean $-7.612991$, $G_{20}=0.0071000$. First crossing of 90% $G_{\rm ref}$ at step 32.
* Exact-arm slope over steps 61–120: $-5.0\times10^{-6}$/step ± $1.35\times10^{-6}$, z = −3.71, i.e. −4.3% of
  $G_{\rm ref}$ over 60 steps → **flagged "exact still descending"** (not scored). At 5×2 it was z = −1.10.

## Primary statistic (rev. 2)
| guided $N$ | $s$ (61–120) | detrended sd_w | raw window sd | first 90% step | $s$ last-20 (5%, reported) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.4787 | 0.0043 | 0.0328 | — | 0.456 |
| 16,384 | 0.2510 | 0.0008 | 0.0315 | — | 0.230 |
| 65,536 | 0.1326 | 0.0022 | 0.0301 | 112 | 0.115 |
| 262,144 | 0.0686 | 0.0024 | 0.0221 | 66 | 0.060 |
| Born 65,536 | 0.2997 | | | — | |
| Born 262,144 | 0.1827 | | | — | |

* Bracket 65,536–262,144; $\ln s$–$\ln N$ interpolation at $s=0.10$: **$N_{\rm needed}(6\times2)=118{,}600$**.
* Conditioning: gap $0.1326-0.0686=0.0641$ against $2\,{\rm sd}_w=0.0048$ (detrended upper rung; one seed, no seed
  term): 26.5 sd_w → **RESOLVABLE**. (Raw undetrended sd_w 0.0221 would give 2.9 sd_w, still RESOLVABLE.)
* $r'=118{,}600/31{,}280=3.79$; $g_3=1.728$, $g_{H1}=e^{H}(6\times2)/e^{H}(5\times2)=2750/781=3.52$.
  $N_{\rm needed}\ge110{,}100$ → **SUPPORT-LIKE** (margin 1.08× above the threshold).
* 5%/last-20 statistic (reported only): censored above 262,144 (top rung 0.060).

## Comparator (2026-11-17 addendum: first crossing of $E\le E_{\rm start}-0.9G_{\rm ref}$)
* Guided: 65,536 × 112 = **7.34M** (262,144 × 66 = 17.3M). Born: neither rung crosses in 120 steps → censored,
  $T_{\rm born}\ge262{,}144\times120=31.46$M. Ratio **≥ 4.29 ≥ 2** → the size counts toward GUIDED-FEWER.
  With 5×2 (9.3–9.4×): **GUIDED-FEWER at both scored sizes** (one seed per arm at 6×2, one Born seed at both).
* Sustained 90% (reported only): guided 19.9M (262,144 × 76), Born censored.

## R51.10 (`RULE_kappa.md` rev. 1) against this step
Recorded prediction (VOI note, ledger): KAPPA-POLY, $\kappa$ ratio 0.70, so "about 22k if $N$ tracked $\kappa$"
(31,280 × 0.70 = 21,900). Measured 118,600, 5.4× the $\kappa$-tracking value; class SUPPORT-LIKE ≠ KAPPA-POLY at a
decided (non-AMBIGUOUS) step → **R51.10 KILLED as a predictor** (kill rule rev. 1). The one-state exact
noise-to-signal ratio fell while the multi-step need rose 3.8×.

## Caveats (stated, not scored)
* One seed per arm. The 65,536 rung's $s$ lies $\ln(0.1326/0.10)=0.28$ ln-units from 0.10, inside the 0.3 ln-unit
  second-seed clause, which is therefore triggered. The 1.08× margin over 110,100 is 0.074 in ln $N$; a 5×2-sized
  seed shift (0.0037 in $s$) on one bracketing rung moves ln $N_{\rm needed}$ by 0.03 (lower rung) to 0.05 (upper
  rung), so the margin is about 1.5–2 seed-sd: likely but not secure.
* The exact arm is still descending (z = −3.7): a lower true late-time reference would raise every guided $s$ and
  push $N_{\rm needed}$ up, i.e. away from POLY. The flag does not weaken the SUPPORT-LIKE side.
* Representativeness (L40): the 6×2 start's relative error to $E_0$ is $1.1\times10^{-3}$ against $5.5\times10^{-4}$
  at 5×2; doping 1/6 vs 1/5. Over a size factor 1.2 a fast polynomial and the support growth cannot be separated
  in principle; SUPPORT-LIKE here says the need tracked the support at this one step.

## What the verdict releases (per RULE.md; nothing prepared)
1. Rev-1 action for SUPPORT-LIKE: a force variance-reduction / resampling generation batch first; the Answer
   reports the trend.
2. Seed clause: a second seed (seed0 19000) of the 65,536 rung (and, for a matched bracket, 262,144), as its own
   gated job, to check the 1.08× margin.
3. 7×2 ladder: needed for the three-size class (SUPPORT-LIKE-3 needs both steps SUPPORT-LIKE; "What would close
   it" needs SUPPORT-LIKE at two size steps); costed after the batch in 1.
4. 4×3 exact arm: the geometry question stays open by both of RULE.md's triggers (κ(6×2)/κ(4×3) = 0.61, outside
   [1/1.5, 1.5]; and the rev-1 4×3 bound r ≤ 2.78 cannot be SUPPORT-LIKE against $g_H$ = 3.89, disagreeing with
   $r'$), so it is released, but R51.10's flag now comes from a killed predictor and the 4×3 bound was on the old
   5% statistic; rank it after 1–2.
Not released: step redesign (the size is RESOLVABLE).

## Ledger
`compute_gate.py refresh` (2 jobs); `record 769376 --verdict SUPPORT-LIKE ... --changed yes --hit no`
(prediction POLY-CONSISTENT, p 0.50, missed; comparator prediction GUIDED-FEWER hit).
