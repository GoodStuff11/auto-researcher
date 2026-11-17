#!/bin/bash
# P5.1 round 1, route R1 `nfloor`, unit 1 (arms A and B of RULE_B.md). Prepared, NOT submitted by the worker;
# submit only through compute_gate.py submit with voi.md.
# Wall time (measured, 1 thread, logs/preflight.txt): 0.221 CPU-min/step at N=16384, 0.441 at N=65536; set-up 0.3-0.6.
# Arm B: 120 x 0.441 + 0.6 = 53.5 min; x1.3 = 69.5 -> 01:10:00 (arm A needs 35 min under the same limit).
# Memory: 4.7 GB peak for the 157k-row exact build; arm B holds ~4 copies of a 65536 x 2637 array (5.5 GB) -> 12G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:10:00
#SBATCH --array=0-1
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_noise_floor_r01
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
NS=(16384 65536)
N=${NS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "P5.1 R1 nfloor unit 1: N $N tau 0.005 on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u nf_run.py --nsamp "$N" --tau 0.005 --steps 120 --seed0 9000 --cpu-cap 66 | tee -a "logs/nf_N${N}_tau0.005_s9000.txt"
