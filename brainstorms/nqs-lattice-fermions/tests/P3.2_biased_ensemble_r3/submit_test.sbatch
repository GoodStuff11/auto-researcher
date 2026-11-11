#!/bin/bash
#SBATCH --job-name=P3.2_r3
#SBATCH --partition=kim
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.2 round 3: same-irrep near-conical-intersection decision test (CPU only).
# One array task per (width, seed): width = (2, 4, 8)[t // 3], seed = (1234, 2345, 3456)[t % 3].
# Each task: ED (irrep build ~20 s, 14 points x ~15 s), then 2 projected-SR NNBF branches at each
# of 6 snapshot points and 10 targets = 32 trainings x 800 steps, then sampled arms (20 replicas
# x 20000 draws; minutes).
# Cost (measured, time_step.py, 8 threads on a shared compute node; timing_output.txt):
#   2.08 / 2.19 / 2.04 s per projected SR step at widths 2 / 4 / 8 (D = 245025, |G| = 48).
#   32 x 800 x 2.2 s = 15.6 h worst case, + ED and sampling < 0.5 h -> ~16 h; 24 h gives 1.5x
#   margin (16 threads should be faster than the measured 8). Memory: P1.1-P peaked at
#   ~2 GB on this D; the dense irrep (5214^2 doubles = 0.2 GB) is extra; 16G gives > 4x margin.
# Submit from this directory so logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --sr-iters 800 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --n-samples 20000 --n-rep 20 --tau 0.02
