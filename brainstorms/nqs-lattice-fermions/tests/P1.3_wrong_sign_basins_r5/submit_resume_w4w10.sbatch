#!/bin/bash
#SBATCH --job-name=P1.3r5_resume_w4w10
#SBATCH --partition=kim
#SBATCH --array=12-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.3 round 5, widths 4 and 10 resume (pre-committed action for GLOBAL VERDICT UNCLEAR with
# w4 and w10 UNCLEAR-CONV, job 507179, 2026-11-14). Width 6 was already resumed (_ext JSONs).
# Task id -> width = [6, 4, 10][id // 12], arm = [Cuhf, Cfs, Ccdw, W][(id % 12) // 3],
# seed = [11, 22, 33][id % 3], as in submit_test.sbatch. Each run continues from its saved result
# JSON for 600 more energy steps (--rel 600) and writes an _ext JSON used by the aggregate.
# Cost: 600 x 12.2 s (w4) or 14.9 s (w10) ~ 2.0-2.5 h per task.
# Afterwards: python3 micro_test.py --mode aggregate --results results --instance 0
#   (if still UNCLEAR, the pre-committed width-trend tie-break decides).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
WIDTHS=(6 4 10)
ARMS=(Cuhf Cfs Ccdw W)
SEEDS=(11 22 33)
T=${SLURM_ARRAY_TASK_ID}
W=${WIDTHS[$((T / 12))]}
ARM=${ARMS[$(((T % 12) / 3))]}
SEED=${SEEDS[$((T % 3))]}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode train --instance 0 --task "${T}" --pre 600 --rel 600 \
    --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 \
    --init-from "results/inst0_${ARM}_w${W}_s${SEED}.json"
