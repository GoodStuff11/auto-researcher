#!/bin/bash
#SBATCH --job-name=P1.2r5_seeds4455
#SBATCH --partition=kim
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.2 round 5: pre-committed AMBIGUOUS action for route frdK (2026-11-13).
# Adds seeds 44 and 55 for the arms nnbf6 (control) and frdK, each at both SR shifts (1e-3, 1e-2; the
# aggregate keeps the lower-energy shift), with exactly the settings of submit_test.sbatch.
# Task t: arm = (nnbf6, frdK)[t // 4], seed = (44, 55)[(t // 2) % 2], shift = (1e-3, 1e-2)[t % 2].
# Afterwards: python3 micro_test.py --mode aggregate --results results --final 1
#   (forces PASS or FAIL on 5-seed medians).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(nnbf6 frdK)
SEEDS=(44 55)
SHIFTS=(1e-3 1e-2)
T=${SLURM_ARRAY_TASK_ID}
ARM=${ARMS[$((T / 4))]}
SEED=${SEEDS[$(((T / 2) % 2))]}
SH=${SHIFTS[$((T % 2))]}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode train --instance 0 --arm "${ARM}" --seed "${SEED}" --sr-shift "${SH}" \
    --sr-iters 1200 --sr-lr 0.05 --sr-samples 2000 --base-width 4 --ctrl-width 6 --lbfgs-iters 4000 \
    --fn-iters 10 --fn-floor 1e-3 --mc-samples 20000 --mc-burn 2000 --chains 100
