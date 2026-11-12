#!/bin/bash
#SBATCH --job-name=P2.1r5_pilot
#SBATCH --partition=kim
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 5: control-only pilot for a SECOND qualifying instance (critic item 5).
# Task a -> instance 3 + a (micro_test.INSTANCES; chosen from screen_output.txt), arm nnbf
# (projected NNBF width 6, the round-3 control), seed 11, the same settings as the round-3 pilot
# (job 410988), so the result is reused by the decisive array (an existing JSON is skipped).
# Pre-committed: a candidate QUALIFIES if its dE_sign (state "P") >= 0.01 t; the 3-seed
# confirmation (seeds 22, 33) is part of the decisive array (submit_test.sbatch).
# Cost: one SR step on D ~ 1.7-2.0M is ~13 s at 16 threads (instance 0: 12.7 s); 1200 steps
# ~4.5 h, plus ED, sector dims, two budgets and the sampled arm (~20 min). 16 h leaves margin.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
INST=$((3 + SLURM_ARRAY_TASK_ID))
${PY} micro_test.py --mode train --instance "${INST}" --arm nnbf --seed 11 --K 4 --hs-sign 6 \
    --match-width 6 --sr-iters 1200 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --mc-samples 20000 --mc-burn 2000
