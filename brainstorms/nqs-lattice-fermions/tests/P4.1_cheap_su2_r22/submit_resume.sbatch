#!/bin/bash
# P4.1 r22 RESUME of job 742206 (verdict INCOMPLETE). Rule: RULE.md, unchanged ("INCOMPLETE ... The script resumes
# from results/state_<arm>.json; the missing arm is resumed by a new gated job").
# RULE.md fixes the step cap at 12 steps per arm (Protocol item 5); ladder.py counts steps across resumes, so this
# job can add at most 3 steps to pcg (9 done) and 4 to rank5 (8 done). che already ended STOP-CAP at 12 steps
# (final; its verdict uses the RULE.md guard's geometric-tail extrapolation, eps_inf 6.930e-4) and is NOT resumed.
# Array of 2 independent 1-CPU tasks: 0 = pcg, 1 = rank5. Prepared, NOT submitted by the worker; submit only
# through compute_gate.py submit with voi_resume.md.
# Per-step CPU measured in 742206 (logs/arm_*_output.txt, 1 thread, wall ~= CPU): pcg mean 8.07, max 9.40;
# rank5 mean 8.14, max 8.84 CPU-min; set-up 0.3 (pcg) / 0.6 (rank5) CPU-min.
# Worst task rank5: 4 x 8.84 + 0.7 = 36.1 min, x1.3 = 47 min -> 00:50:00 (pcg: 3 x 9.40 + 0.5 = 28.7, x1.3 = 37).
# --cpu-cap 45: ladder.py stops before a step only if elapsed + last step > 45; worst rank5 check before step 4 is
# 0.7 + 3 x 8.84 + 8.84 = 36.1 < 45, so the CPU cap cannot truncate before the 12-step cap is reached.
#SBATCH --partition=kim
#SBATCH --job-name=p41-r22-resume
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22/logs/%x_%A_%a.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22
PY=~/.conda/envs/lit_env/bin/python
ARMS=(pcg rank5)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
cd "$DIR"
echo "r22 resume arm $ARM on $(hostname), job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} (resume of 742206)"

# Same protocol and same --steps 12 as 742206; ladder.py reloads state_<arm>.json / theta_<arm>.npy and continues.
$PY -u ladder.py --mode arm --arm "$ARM" --steps 12 --threads 1 --cpu-cap 45 | tee "logs/arm_${ARM}_resume_output.txt"
$PY -u ladder.py --mode verdict | tee "logs/verdict_resume_after_${ARM}.txt"
