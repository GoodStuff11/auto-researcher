#!/bin/bash
#SBATCH --job-name=P3.1r5_analyze
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 5 decisive test, part 2: all arms (OSG, EHX, L1, L2, SD, ZV, SMA, OSGX, ZRO) on the 18
# converged projected states (widths 3, 6, 12 x seeds 11, 22, 33 x {P, P_S}), then the verdict.
# Needs the training array (submit_train.sbatch) AND P2.1 job 451706 (width-6 seeds 22, 33):
#   sbatch --dependency=afterok:<TRAIN_JOB>:451706 submit_analyze.sbatch
# Cost: see the per-state time in smoke/analyze_output.txt (about 80 s per state at 20k samples; budget 2 h for 18 states at 100k).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode analyze --widths 3,6,12 --seeds 11,22,33 --mc-samples 100000 --n-boot 100
${PY} micro_test.py --mode aggregate
