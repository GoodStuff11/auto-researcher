#!/bin/bash
#SBATCH --job-name=P2.1r2_lowrank_sign
#SBATCH --partition=kim
#SBATCH --array=0-36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 2: sign quality of low-rank-hop-response determinant families versus full NNBF
# at matched parameter count, exact sums on the three P1.1 4x3 PBC clusters (t'=-0.25).
# Tasks 0-35: instance = id//12, arm = [nnbf128, mdnnbf, lchfds, schfds, sklr, sjnet][(id%12)//2],
# seed = [1234, 4321][id%2]. Task 36: cost side check (fast vs brute local energy, L=8..32).
# Submit from this directory so logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Cost estimate (measured on the login node, 4 BLAS threads, instance 1, D=245025): one SR
# iteration costs 2.3 s (nnbf128) to 7 s (sklr, the most expensive arm; mdnnbf 5.5 s).
# Instance 0/2 (D=627264) is 2.6x larger: <= 18 s/iteration -> <= 7.5 h for 1500 iterations,
# plus ED (minutes) and the P1.1 budget (<= 1 h: 11 sparse eigsh + 4 L-BFGS-B starts).
# Worst task ~8.5 h on 4 threads (less on 8), so 24 h leaves > 2x margin. Peak memory
# ~10 GB (sparse H of instance 0 plus 65536-config chunks); 24 GB requested.
# Task 36 takes minutes (brute-force local energy at L=24 dominates).
# Afterwards (any node, seconds): python3 micro_test.py --mode aggregate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
if [ "${SLURM_ARRAY_TASK_ID}" -eq 36 ]; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    ${PY} micro_test.py --mode cost --sizes 8,12,16,20,24,28,32 --brute-max 24 --nconf 2 \
        --r 2 --K 4 --hs-sign 64 --tp -0.25 --U 8 --doping 0.125 --seed 1234
else
    ${PY} micro_test.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --r 2 --K 4 --hs-sign 64 \
        --match-width 128 --sr-iters 1500 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
        --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
fi
