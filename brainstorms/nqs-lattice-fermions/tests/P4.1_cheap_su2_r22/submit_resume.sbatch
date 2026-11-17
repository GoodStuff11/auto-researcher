#!/bin/bash
# P4.1 r22 RESUME of job 742206 (INCOMPLETE) under RULE_amend1.md: all three arms, symmetrically, run until the
# predicted-gain stop (< 1e-3 g, STOP-RULE), STOP-NOGAIN, or the 30-total-step hard cap (--steps 30; steps from
# 742206 count). che (STOP-CAP at 12) is reopened by ladder.py's RULE_amend1 path. Thresholds and guard unchanged.
# Array of 3 independent 1-CPU tasks: 0 = pcg (<= 21 more steps), 1 = rank5 (<= 22), 2 = che (<= 18).
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi_resume.md.
# Per-step CPU measured in 742206 (logs/arm_*_output.txt, 1 thread, wall ~= CPU): pcg mean 8.07 / max 9.40,
# rank5 8.14 / 8.84, che 2.86 / 2.93 CPU-min; set-up 0.3 / 0.6 / 0.1.
# Sized so the 30-step hard cap, not the CPU cap, binds in the worst case (every step at the max step cost):
#   pcg 21 x 9.40 + 0.5 = 198 min; rank5 22 x 8.84 + 0.7 = 195 min; che 18 x 2.93 + 0.1 = 53 min.
#   Worst 198 x 1.3 = 257 min -> --time 04:20:00 (array tasks share --time).
#   --cpu-cap 210: ladder.py stops before a step only if elapsed + last step > 210; worst pcg check before its 21st
#   step is 0.5 + 20 x 9.40 + 9.40 = 198 < 210.
#SBATCH --partition=kim
#SBATCH --job-name=p41-r22-resume
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:20:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22/logs/%x_%A_%a.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22
PY=~/.conda/envs/lit_env/bin/python
ARMS=(pcg rank5 che)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
cd "$DIR"
echo "r22 resume (RULE_amend1) arm $ARM on $(hostname), job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} (resume of 742206)"

# Same ladder protocol as 742206; ladder.py reloads state_<arm>.json / theta_<arm>.npy and continues to --steps 30.
$PY -u ladder.py --mode arm --arm "$ARM" --steps 30 --threads 1 --cpu-cap 210 | tee "logs/arm_${ARM}_resume_output.txt"
$PY -u ladder.py --mode verdict | tee "logs/verdict_resume_after_${ARM}.txt"
