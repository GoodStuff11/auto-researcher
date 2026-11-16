#!/bin/bash
#SBATCH --job-name=P3.1r7_excpilot
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 7: ESR0 excited-state pilot (ED-scale premise test, exact sums). Sequential penalty VMC
# (beta = 2t) for K = 3 in-irrep (chi, S=0) excited states at NNBF width 6 against the trained seed-11
# psi, then exact eps/omega, window weights, psi_0 contamination, principal angles, wall time, and the
# exact-matrix ESR0(K=3) vs ORC(K=3) ratios. Writes results_excpilot/ only (resumable per state).
# Submit from this directory: sbatch submit_excpilot.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 excited_pilot.py --mode run --instance main --K 3 --beta 2.0 \
    --seed 11 --width 6 --sr-iters 1200 --n-low 30 --results results_excpilot
