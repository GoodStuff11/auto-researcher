#!/bin/bash
#SBATCH --job-name=P3.1r7_esr
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 7: Part 1 (concentration curve, oracle-ESR ceiling, oracle certificate envelope, seed
# cosines and MSR on the real width-6 states present at run time) and Part 2 (pre-registered injected
# check of route ESR0, grade R deciding; grade H reported). Writes results/esr_inject.json only.
# Submit from this directory: sbatch submit_inject.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py --instance main --n-low 36 --n-seeds 8 \
    --mc-samples 200000 --n-boot 100 --lanczos-steps 120 --results results
