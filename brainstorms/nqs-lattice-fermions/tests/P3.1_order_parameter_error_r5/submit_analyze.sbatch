#!/bin/bash
#SBATCH --job-name=P3.1r5_analyze
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 5 decisive test, part 2 (ONLY after GATE-PASS): all arms on the 11 width-6 seeds, label P,
# plus the injected check, then the pre-committed FULL rule (EHX vs ZV; micro_test.py docstring).
# Missing parameter files are skipped. Every dependency is afterany:
#   sbatch --dependency=afterany:<TRAIN_JOB> submit_analyze.sbatch   (NOT submitted)
# Cost: a few minutes per state at 2 partners, growing with partners (2 hybrids each); budget 6 h.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode analyze --widths 6 --seeds 11,22,33,44,55,66,77,88,99,111,122 --labels P \
    --mc-samples 100000 --n-boot 100 --results results_full
${PY} micro_test.py --mode aggregate --rule full --results results_full
