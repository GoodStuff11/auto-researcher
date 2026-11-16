#!/bin/bash
#SBATCH --job-name=P2.1r9_T2lh3
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 9, T2 pilot of Route 6.1'' (LH-HFDS, 3x3 patch) on instance 4 (5x3 OBC 6+6 t-t'-J), NEW seed 22,
# 75 projected SR steps (protocol of job 698229). Rule: leaf artifact, Round 9 "Pre-registered T2 rule,
# Route 6.1''" with amendment A1. Verdict: python3 t2_3x3.py --mode aggregate (after all six tasks).
#   task 0 lh_3x3 (route)   task 1 lh_plus (patch-size foil)   task 2 lh_site (reported)
#   task 3 nnbf (H = 6, matched P)   task 4 nnbf_c (H_c = 16, op-count matched; the compute foil)
#   task 5 nnbf_w (H = 48, reported only)
# H_c, H_w and every arm's step-0 energy are fixed in preflight_config.json (the run asserts them).
# Budget (pre-flight, 4 threads): s/step lh_3x3 10.3 (16.1 in job 698229), nnbf 3.1, nnbf_c ~3.2,
# nnbf_w 3.9; 75 steps + ED + two budgets <= ~1.75 CPU-h per task at 4 CPUs, ~7 CPU-h in total.
# Existing result JSONs are skipped (t2_3x3.py refuses to overwrite without --overwrite 1).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} t2_3x3.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --instance 4 --seed 22 --F 5 --Hc 16 --Hw 48 \
    --sr-iters 75 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
