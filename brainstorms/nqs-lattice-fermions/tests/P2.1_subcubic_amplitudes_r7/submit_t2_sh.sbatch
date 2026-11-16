#!/bin/bash
#SBATCH --job-name=P2.1r8_T2sh
#SBATCH --partition=kim
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 8, T2 pilot of Route 7.3 (SH-HFDS, sparse hidden-hidden block) on instance 4, seed 11,
# 75 projected SR steps; protocol, thresholds and early stop of the Route 6.1' T2 (job 698229).
# QUEUED: submit only with --dependency=afterok:698229 (its foils lh_plus / lh_site and reference
# nnbf come from 698229). Rule: leaf artifact, Round 8 "T2 rule, Route 7.3".
#   task 0 sh_plus (route, plus patch, F = 5, lam shared over sites, Ha = 0)
# Foil: the better of lh_plus and lh_site (698229); reference: nnbf (698229).
# Verdict: python3 t2_routes.py --mode aggregate --route sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} t2_routes.py --mode train --route sh --task "${SLURM_ARRAY_TASK_ID}" --instance 4 --seed 11 --F 5 \
    --sr-iters 75 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
