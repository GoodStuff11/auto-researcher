#!/bin/bash
#SBATCH --job-name=P1.3r5_resume_w6
#SBATCH --partition=kim
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.3 round 5, width-6 pilot resume (pre-committed action for PILOT VERDICT UNCLEAR-CONV, 2026-11-13).
# Each of the 12 width-6 runs (task t: arm block t//3 in Cuhf,Cfs,Ccdw,W; seed index t%3 in 11,22,33)
# continues from its saved result JSON for 600 more energy steps (--rel 600). Per the round-5 fix,
# --init-from carries W0, traces and the step offset forward, advances the RNG (no replay) and writes an
# _ext JSON that the aggregate uses in place of the base run.
# Cost: 600 steps x 12.2-13.2 s/step (width 6, 8 threads) ~ 2.2 h per task.
# Afterwards: python3 micro_test.py --mode aggregate --instance 0 --pilot w6
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ARMS=(Cuhf Cfs Ccdw W)
SEEDS=(11 22 33)
T=${SLURM_ARRAY_TASK_ID}
ARM=${ARMS[$((T / 3))]}
SEED=${SEEDS[$((T % 3))]}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode train --instance 0 --task "${T}" --pre 600 --rel 600 \
    --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 \
    --init-from "results/inst0_${ARM}_w6_s${SEED}.json"
