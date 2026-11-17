#!/bin/bash
# P4.1 ledger row 4, cheaper version (RULE_amend3.md): rank5 alone to its stop, then pcg only until rho_abl is
# fixed by dominance. Resumes job 742206's saved states in ../P4.1_cheap_su2_r22/results with the frozen ladder.py.
# Per-step CPU measured in 742206 (1 thread, wall ~= CPU): rank5 8.14 mean / 8.84 max, pcg 8.07 / 9.40; set-up
# 0.6 / 0.3 (pcg is re-launched per step: ~9.9 per call worst).
#   rank5 worst (22 more steps to the 30-step cap): 22 x 8.84 + 0.7 = 195 min (ladder --cpu-cap 200)
#   pcg worst (21 calls to its 30-step cap):         21 x 9.9       = 208 min (driver budget 400 min total)
#   Worst 403 min -> --time 07:00:00, 1 CPU -> cap 7.0 CPU-h. Expected: rank5 ~11-17 steps, pcg 2-5 calls,
#   about 2-3.3 CPU-h. Memory 16G (r19/r22 peak 9.1 GB).
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi_row4.md.
#SBATCH --partition=kim
#SBATCH --job-name=p41-row4-rank5-dominance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=07:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22row4/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22row4/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22row4
cd "$DIR"
~/.conda/envs/lit_env/bin/python -u row4.py --budget-min 400 | tee logs/row4_output.txt
