#!/bin/bash
# P4.1 R42 `practitioner-24`, stage 1 (seeds 0-1). Rule: RULE.md. Card: leaves/P4.1/cards/R42.md.
# Prepared by the round-27 worker, NOT submitted; submit only through compute_gate.py submit with voi.md.
# Task k: arm = (che, nnbf)[k % 2], seed = k / 2. Open 8x3, N_e 20, U 8, t' -0.25, P_G (+,+), H 16, P 9711 both arms.
# Equal compute (L35): every task trains for exactly 180 CPU-min on ONE core (guided SR), then scores.
# Wall time (measured 2026-11-17, logs/probe_local.txt, 1 thread, 1024 chains): che 0.166 CPU-min per step,
# nnbf 0.090 (guided = plain); start-up 1.0 / 0.4. Scoring (estimated from the measured 128-sample E_loc + S^2 cost):
# che about 32 CPU-min (131,072-sample energy, 32,768 S^2, 32,768 half-budget energy), nnbf about 17.
# che task 1 + 180 + 32 = 213 min; x 1.25 margin = 266 -> 04:30:00. Memory: 0.87 GB peak RSS (probe) -> 3G.
# Stage 2 (seed 2), only if RULE.md's stage-2 clause fires, as a new gated job:
#   sbatch --array=4-5 submit_r42.sbatch
#SBATCH --job-name=P41r42
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:30:00
#SBATCH --array=0-3
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r42/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r42/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r42
cd "$DIR"
K=${SLURM_ARRAY_TASK_ID:-0}
ARMS=(che nnbf)
ARM=${ARMS[$((K % 2))]}
SEED=$((K / 2))
echo "r42 task $K: arm $ARM seed $SEED on $(hostname), job ${SLURM_JOB_ID:-local}"
~/.conda/envs/lit_env/bin/python -u r42.py --mode train --arm "$ARM" --seed "$SEED" --Lx 8 --guided 1 \
    --cpu-min 180 --n-chains 1024 --n-eval 131072 --n-s2 32768 --n-half 32768 --outdir "$DIR/results"
