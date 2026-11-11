#!/bin/bash
#SBATCH --job-name=P3.1pilot
#SBATCH --partition=kim
#SBATCH --array=0,6,12,18,24,30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round-3 TARGET PILOT (per-leaf critic, 2026-11-11). It decides whether converged projected
# states on these 6 instances carry an order-parameter error large enough to score, before the
# 36-task job (submit_test.sbatch) is allowed to run.
# One task per instance (task t -> instance t//6, seed 101). Exact quantities only: ZR training is
# reduced to 1 step and Monte Carlo to 2000 draws, so the reported ZRO/L1/KR2/SMA ratios are exact.
# Populations: P_w1, P_w2, P_w4 (projected NNBF) and F_P (projected foil), 1500 SR steps, snapshots
# at steps 100 and 400.
# Go criterion for the full job, pre-committed: at least 2 instances where
#   (i)   at least 3 converged (step-1500) finals have |delta| >= 0.02 |a0|,
#   (ii)  ZRO/delta lies in [0.5, 2] on at least 70% of those finals, and
#   (iii) L1 tracks (delta_hat/delta in [1/2, 2]) fewer than 90% of them.
# Otherwise NO-GO: re-scope P3.1 to larger sampled systems or a same-irrep near-degenerate instance.
# Cost estimate: 4-7 h per task (critic's estimate from the measured 1.6-4.8 s per projected SR step).
# Read: the per-population lines in P3.1pilot_*.out, or the JSONs in results_pilot/.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p results_pilot
INST=$(( SLURM_ARRAY_TASK_ID / 6 ))
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --sr-iters 1500 --snaps 100,400 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --populations P_w1,P_w2,P_w4,F_P --r-min 1.5 \
    --mc-samples 2000 --n-boot 10 --zr-width 8 --zr-steps 1 --zr-lr 0.02 \
    --out "results_pilot/inst${INST}_s101.json"
