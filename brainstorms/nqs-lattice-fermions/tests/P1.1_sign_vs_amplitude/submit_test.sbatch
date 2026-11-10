#!/bin/bash
#SBATCH --job-name=P1.1_sign_amp
#SBATCH --partition=kim
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.1 sign-versus-amplitude error budget. One array task per pre-committed instance
# (0: 4x3 n=5+5 U=8 t'=-0.25; 1: 4x3 n=4+4 U=8 t'=-0.25; 2: 4x3 n=5+5 U=4 t'=-0.25). CPU only.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --instance "${SLURM_ARRAY_TASK_ID}" \
    --widths 8,32,128 --sr-iters 1500 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --seed 1234
