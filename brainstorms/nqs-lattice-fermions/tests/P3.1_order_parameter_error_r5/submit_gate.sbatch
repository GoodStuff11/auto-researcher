#!/bin/bash
#SBATCH --job-name=P3.1r5_gate
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 5 CHEAP GATE (critic fix): analyze-only, width 6, seeds 11/22/33 (P2.1 parameter files;
# seeds 22/33 come from P2.1 job 451706, missing files are skipped and logged), label P only, plus the
# injected-sign-flip check; then the pre-committed gate rule (micro_test.py docstring):
#   GATE-PASS iff INJECT-PASS and M_EHX >= 0.5 and median_u [T_EHX - T_ZV] > 0 and G_EHX <= 0.10
#   (>= 3 trusted). GATE-PASS -> train >= 8 more width-6 seeds (submit_train.sbatch). Otherwise
#   EHX = "Route 2.3 restated" and the leaf is marked for mutation.
# Writes only to results_gate/. Cost: ED + a few minutes per state + the inject check; under 2 h.
# Submit from this directory: sbatch --dependency=afterany:451706 submit_gate.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode analyze --widths 6 --seeds 11,22,33 --labels P --mc-samples 100000 --n-boot 100 --results results_gate
${PY} micro_test.py --mode aggregate --rule gate --results results_gate
