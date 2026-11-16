#!/bin/bash
#SBATCH --job-name=P1.2r5_frd
#SBATCH --partition=kim
#SBATCH --array=0-26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.2 round 5: NNBF-4 x FRD (kinetic / exchange frame) versus projected NNBF-6 at matched parameters on the
# 4x4 OBC t-t'-J instance (6+6, t' = -0.25, J = 0.4; D = 1681680, |G| = 8). See micro_test.py docstring.
# Tasks 0-23: arm = [nnbf6, nnbf4, frdK, frdX][id // 6], seed = [11, 22, 33][(id % 6) // 2],
#             SR shift = [1e-3, 1e-2][id % 2]  (exact force, projected SR, 1200 steps).
# Tasks 24-26: sampled-force SR arms nnbf6_mc, frdK_mc, frdX_mc (seed 11, shift 1e-2, Metropolis force + metric).
# Cost per SR step and peak memory: see timing_output.txt (measured on a kim node at 8 threads).
# Submit from this directory:  cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --mode aggregate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --instance 0 --sr-iters 1200 --sr-lr 0.05 \
    --sr-samples 2000 --base-width 4 --ctrl-width 6 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 \
    --mc-samples 20000 --mc-burn 2000 --chains 100
