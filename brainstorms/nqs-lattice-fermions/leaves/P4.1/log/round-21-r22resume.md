# P4.1 round 21: r22 resume prepared (job 742206 INCOMPLETE)

2026-11-17. Worker note; not submitted.

* 742206 result (results/verdict.json): pcg 9.262e-4 -> 4.230e-4 (9 steps, stop null, CPU cap 74 hit); che 1.217e-3 ->
  7.217e-4 (STOP-CAP at 12, guard eps_inf 6.930e-4); rank5 5.081e-3 -> 7.693e-4 (8 steps, stop null, CPU cap hit).
  Provisional rho_pol 0.586 end / 0.610 vs che eps_inf; rho_abl 0.550. Verdict INCOMPLETE (RULE.md: changes nothing).
* Rule applied: RULE.md INCOMPLETE clause (resume from state_<arm>.json by a new gated job). RULE.md Protocol item 5
  FIXES the step cap at 12 per arm, and ladder.py counts steps across resumes. So the cap is not raised: che is final
  (STOP-CAP, verdict through the guard's tail rule) and only pcg (<= 3 steps) and rank5 (<= 4 steps) resume. RULE.md
  is not silent on this, so no RULE_resume.md was written.
* Predicted-gain decay (742206 logs): pcg 0.138 -> 0.0249 g (x0.81/step, ~15 steps to 1e-3 g), rank5 2.26 -> 0.047 g
  (x0.68 recent, ~10 steps), che 7.1e-3 -> 4.4e-3 g over steps 8-12 (x0.89, ~13 steps). No arm reaches STOP-RULE inside
  12 steps; all three will be read through the guard.
* Per-step CPU (1 thread): pcg mean 8.07 / max 9.40, rank5 8.14 / 8.84, che 2.86 CPU-min. Resume: --time 00:50:00
  (worst rank5 36.1 min x1.3), --cpu-cap 45 (cannot cut before step 12), array 0-1, cap 1.7 CPU-h.
* Files: tests/P4.1_cheap_su2_r22/submit_resume.sbatch, tests/P4.1_cheap_su2_r22/voi_resume.md.
  Gate (--pre-audit): PASS, cap 1.7 CPU-h, VOI digest 65b42d13d284.
* Prediction (voi_resume.md): rho_pol KEEP-LIVE 0.93 (needs only a 1.7% further pcg drop); rho_abl PAIRING-KEPT 0.35 /
  ABL-CONFOUNDED 0.30 / UNSETTLED 0.33 / RETRACTED 0.02; joint KEEP-LIVE/PAIRING-KEPT p 0.33.
* Open issue for the orchestrator: if a ratio is UNSETTLED after this job, the guard says "resume", but the fixed
  12-step cap leaves ladder.py nothing to resume. Further polish then needs a cap amendment pre-registered before
  that resume runs (likely for rho_abl: pcg's alternating gains can give r >= 1, an unbounded tail, eps_inf clipped to 0).
