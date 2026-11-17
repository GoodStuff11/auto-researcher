#!/bin/bash
# P4.1 r22: cutoff-ladder polish of three 12-site arms (R19 full ladder + R20 step 2). Rule: RULE.md.
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# Array of 3 independent 1-CPU tasks: 0 = pcg, 1 = rank5, 2 = che. Each task polishes its arm to STOP-RULE,
# STOP-NOGAIN or the 12-step cap, then runs the verdict (the last task to finish prints the full verdict;
# earlier ones print INCOMPLETE). Each arm resumes from results/state_<arm>.json if cut.
# Wall time (measured, 1 thread, see voi.md): pcg/rank5 step ~4.8 CPU-min (r21 probe 4.3 for 4 rungs + 1 rung;
# local rank5 1-step probe logs/probe_rank5_1step.log), che ~2.9. Worst task = 12 x 4.8 + ~1.5 set-up = 59 min,
# x1.3 = 77 min -> 01:20:00 for every task (array tasks share --time). ladder.py's --cpu-cap 74 stops before a
# step that would cross the limit, leaving the arm resumable.
#SBATCH --partition=kim
#SBATCH --job-name=p41-r22-ladder
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:20:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22/logs/%x_%A_%a.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r22
PY=~/.conda/envs/lit_env/bin/python
ARMS=(pcg rank5 che)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
cd "$DIR"
echo "r22 ladder arm $ARM on $(hostname), job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"

# Do not redo a finished arm (rerunning-jobs rule): ladder.py exits at once if state_<arm>.json has a final stop.
$PY -u ladder.py --mode arm --arm "$ARM" --steps 12 --threads 1 --cpu-cap 74 | tee "logs/arm_${ARM}_output.txt"
$PY -u ladder.py --mode verdict | tee "logs/verdict_after_${ARM}.txt"
