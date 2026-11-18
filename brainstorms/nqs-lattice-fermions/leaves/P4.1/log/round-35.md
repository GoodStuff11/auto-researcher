# P4.1 round 35: R47 gate-refused (G2), ledger closed, exit 3 to EXHAUSTED at this budget (editor, 2026-11-18)

No jobs, no commits. Source: orchestrator decision on R47's T1 (`tests/P4.1_cheap_su2_r47/`: RULE_T1.md with Amendments 1–2, `voi.md` digest cd50cd7bb7ce, `compute_gate.py check --pre-audit` PASS at a 7.7 CPU-h cap).

## 1. Orchestrator decision: R47 T1 GATE-REFUSED (G2)
Recorded prediction (online-distillation pre-flight, log `round-34-R47`, `voi.md`): KILL 0.40, UNSETTLED 0.30, INVALID 0.25, PASS 0.05. Unprojected spin-reading students lose to `che` on spin contamination ($\kappa_u=0.73$ at 100 evaluations, 0.18 at 400; `nnbf_u` vs `che`), while the exactly projected arm on the same samples leads ($\kappa_p\approx1.8$). A one-sided 24-site distillation test therefore has no power to show the 12-site loss; every outcome but PASS (p 0.05) leads to the same action (close P4.1). R47 → SHELVED; card rewritten (Why shelved / What would unshelve it); kill_index row added (SHELVED, L68); L68's Counter-evidence and Sources updated (R47 SHELVED). T1 files are left as prepared (not submitted).

## 2. Reopening ledger, final (all rows closed)

| Item | Source | Route or test | Outcome |
| :--- | :--- | :--- | :--- |
| 1–3. sample reuse; exact optimiser ≥3× the ladder; force shrinkage | closure attempt | R34, R28, R39 | KILLED(G4), KILLED(T0), KILLED(T1) |
| 4. R2 vs its rank-5 ablation, converged at 12 sites | What would close it | job 759446, RULE_amend3 | DIAGNOSTIC-HOLDS (pairing gain unattributed; ABL-CONFOUNDED) |
| 5. spin-tree basis beyond seed 0 | What would close it | R22 seeds 1–2, 5.4 CPU-h | GATE-REFUSED(G2) |
| 6. charge-only extensive terms | What would close it | R16; R17 | R16 KILLED(T1); R17 GATE-REFUSED(G2) |
| 7. sampled optimiser reaching the polished 12-site state | What would close it | R21, R38 | KILLED(G2), KILLED(T1) |
| 8. R30's translation repair | critic r23, item 11 | R41 | KILLED(T1) |
| 9. converged charge-only vs projected spin-reading | critic r23, item 4 | R40 (11.0 CPU-h), recast at 24 sites as R42 | GATE-REFUSED(auditor); R42 CHE-NO-WORSE (job 759907) |
| 10. spin-flip-even nnbf baseline at 24 sites | critic r29, Major 2 | R43 (job 764949) | CHE-STILL-WINS: R42 stands |
| 11. 12-site capacity loss at 24 sites | R45 LOSS (job 769413) | R47 T1 (7.7 CPU-h), then T2 (~26) | GATE-REFUSED(G2), orchestrator 2026-11-18 |

## 3. Exit check (ref/states.md)
Exit 1 (OPEN): no falsifier FALSIFIED, no SURVIVED-T1. Exit 2 (at this scale): no, rows 5, 6 (R17), 9, 11 are GATE-REFUSED. Exit 3 applies: every row closed, at least one GATE-REFUSED. Certificate of `round-29-critic` §2 adopted with the updates below; State EXHAUSTED at this budget, marked "per-leaf critic sign-off owed". Answer unchanged (the exit needs no new wording; its 12-site clause already carries R45's rescoped reading). What it would take rewritten around the refused steps.

## 4. Certificate updates to `round-29-critic` §2
* Item 1(b): adds R45's fit-accuracy loss at equal effort, $\kappa=2.23$ (half budget 1.85; 1.64 → 2.23 from 500 to 2000 evaluations; one seed; not converged), as rescoped by the per-leaf critic (round 34): not a converged capacity claim. "Not claimed" adds 24-site expressivity.
* Item 2: adds jobs 764949 (R43) and 769413 (R45); critic round 34.
* Item 3: unchanged (residual: recipe tuned on `che`; $S=2,4$ contamination and a spin penalty untested).
* Item 4: row 10 R43 CHE-STILL-WINS; R45 LOSS; row 11 GATE-REFUSED(G2).
* Item 5 adds: R47 T1 (7.7 CPU-h) then T2 (~26 CPU-h): what it tests, whether the 12-site loss persists at 24 sites via one-sided distillation; refused G2 (prediction above); unshelve: a cheap symmetry-restoring step for the student (exact projection affordable at the target size, or an $S^2$ penalty shown to reach the projected arm's fidelity) or a larger budget. R40, R22 seeds 1–2, R17 unchanged.
* Item 6 adds the R47 unshelve terms as evidence that would reopen the leaf.
* Item 5 of 3a (missing ingredients): every ingredient row ends KILLED or GATE-REFUSED (table above).

## 5. Other edits
* `summary.md`: rewritten for EXHAUSTED at this budget (Route Portfolio: no live route; ledger moved here; certificate in brief; 9,086 characters).
* Front page: P4.1 answer-bullet prefix, P4 "What it adds up to" bullet (12-site clause now R45's fit result; 24-site test unaffordable), "Running" line, Current-leaf-state row.
* `kill_index.md` line for R51.19 (P5.1): `≈3` put in math to clear the raw-inequality warning (no content change).
* Lint: `leaves/P4.1`, `kill_index.md`, `library.md`, front page: 0 errors, 0 warnings.
* Not done: per-leaf critic sign-off on the certificate (owed); `history.md` Revision & Critic Log entry left to the orchestrator.
