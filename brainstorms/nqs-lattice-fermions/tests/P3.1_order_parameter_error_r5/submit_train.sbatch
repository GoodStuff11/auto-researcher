#!/bin/bash
#SBATCH --job-name=P3.1r5_train
#SBATCH --partition=kim
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round 5 decisive test, part 1 (ONLY after GATE-PASS of submit_gate.sbatch): projected-SR NNBF
# width 6 (critic: not widths 3/12), 8 new seeds, on the shared 4x4 OBC t-t'-J instance. Seeds
# 11/22/33 already exist from P2.1 (410988, 451706).
# Task id: seed = [44, 55, 66, 77, 88, 99, 111, 122][id].
# Cost: one SR step at width 6 is 12.7-14.3 s on 8 threads (P2.1 log), so about 4.8 h for 1200 steps.
# Submit from this directory: sbatch submit_train.sbatch   (NOT submitted)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
S=( 44 55 66 77 88 99 111 122 )
/share/apps/software/anaconda3/bin/python3 micro_test.py --mode train \
    --width 6 --seed ${S[$SLURM_ARRAY_TASK_ID]} --sr-iters 1200
