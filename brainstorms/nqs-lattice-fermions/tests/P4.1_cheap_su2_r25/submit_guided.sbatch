#!/bin/bash
# P4.1 round 22, route R38 `guided` T1b at 12 sites (che only). Rule: RULE_D.md.
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi_guided.md.
# Wall time (measured, 1 thread): R21's shift arm, same per-step work, 6.1 CPU-min for 15 steps = 0.41 CPU-min/step
# (../P4.1_cheap_su2_r24/results/shift.json cpu_min); set-up 0.25. 60 x 0.41 + 0.25 = 24.9 min; x1.3 = 32 -> 0:40:00.
# Memory: 2.0 GB peak RSS (R21 probe) -> --mem=4G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --array=0-2
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r25/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r25/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r25
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
SEED0=$(( 6000 + 1000 * ${SLURM_ARRAY_TASK_ID:-0} ))
echo "r25 R38 guided T1b seed0 $SEED0 on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u guided.py 60 "$SEED0" "results_guided_s$SEED0" | tee -a "logs/guided_s$SEED0.txt"
