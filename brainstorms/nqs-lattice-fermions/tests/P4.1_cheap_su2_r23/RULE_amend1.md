# P4.1 r23 amendment 1: seed 0 first (one-seed decision), written 2026-11-17 before any tree-arm run

Reason: the skill's cheapest-version rule (release seeds one at a time when one seed can decide). RULE.md stands
except as stated here.

## Stage 1 (job `submit_tree.sbatch`)

Seed 0 alone: stage A (7000 Adam steps, RULE.md erratum), then the r22 ladder polish of seed 0 (RULE.md stage B,
unchanged), the Gate-2 sign diagnostic, and `tree_nqs.py --mode verdict1`, all in `results/oneseed/`.

## One-seed decision

eps_ref = min(1.217e-3, eps_che,lad) = 7.217e-4 (guard: 6.930e-4), as in RULE.md.

| verdict1 | condition |
| :-- | :-- |
| KILL (G2), one seed | polished seed-0 eps_tree >= 2.5 eps_ref = 1.804e-3, AND guard-extrapolated eps_tree,inf >= 2.5 eps_ref,inf |
| RELEASE | otherwise: run seeds 1-2 (new gated job) and apply RULE.md's 3-seed rule unchanged (best stage-A seed polished in `results/`) |
| INCOMPLETE | no final stop for seed 0 (resume) |

A one-seed run can never give PASS-T1 or AMBIGUOUS; those need the 3-seed rule.

## Why the factor is 2.5 and not the suggested 2

Source: every multi-seed arm of this leaf with per-seed eps on disk (`tests/P4.1_cheap_su2_r{3,5,6}/results/*_seed*.json`,
12 arms, 2-3 seeds each, 4x3 Hubbard U = 8, t' = -0.25). Worst/best seed ratio: median 1.24, range 1.08-1.81; the
largest is r3 `che` at N = 8 (7.12e-5 / 8.44e-5 / 1.29e-4). A one-seed KILL is wrong only if the best of three seeds
would land below eps_ref while seed 0 is above the threshold, i.e. if worst/best exceeded the factor. Factor 2 sits
only 10% above the observed maximum 1.81, and those arms are mature architectures with no learned spin-sector sign;
the tree arm learns an almost maximally mixed sign (results_probe/basis_sign.json: 48% negative weight), where
wrong-sign basins can widen the seed spread. 2.5 gives a 38% margin over the worst observed spread and still kills
early in the predicted case (prediction: eps_tree well above 1e-3).
