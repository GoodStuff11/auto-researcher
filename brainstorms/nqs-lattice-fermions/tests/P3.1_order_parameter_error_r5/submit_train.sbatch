#!/bin/bash
#SBATCH --job-name=P3.1r5_train
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round 5 decisive test, part 1: projected-SR NNBF widths 3 and 12 (width 6 comes from P2.1
# jobs 410988 / 451706), seeds 11, 22, 33, on the shared 4x4 OBC t-t'-J instance.
# Task id: width = [3, 12][id // 3], seed = [11, 22, 33][id % 3].
# Cost: one SR step at width 6 is 12.7 s on 8 threads (P2.1 log); width 12 about 2x, so at most
# about 8.5 h for 1200 steps. Submit from this directory: sbatch submit_train.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
W=( 3 12 ); S=( 11 22 33 )
/share/apps/software/anaconda3/bin/python3 micro_test.py --mode train \
    --width ${W[$((SLURM_ARRAY_TASK_ID / 3))]} --seed ${S[$((SLURM_ARRAY_TASK_ID % 3))]} --sr-iters 1200
