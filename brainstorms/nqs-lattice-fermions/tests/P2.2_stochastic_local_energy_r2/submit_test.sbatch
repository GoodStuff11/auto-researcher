#!/bin/bash
#SBATCH --job-name=p22r2_eloc
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r2/p22r2_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r2/p22r2_%A_%a.err
# CPU only. P2.2 round-2 decisive test: positivised (R2.1) and perturbative-tail
# (R2.2) local energies vs exact/screening/surrogate-screening/completion/HB/SS/SCV.
# H4 (N = 4) in a growing s-Gaussian basis, ns = 2..6 -> M = 16..48 spin-orbitals,
# mean |C(n)| from ~100 to ~2500.  Task 0: square ring R = 2.8 bohr; task 1: chain R = 2.8.
# Global verdict (pre-committed): a route PASSES only if it PASSES on both tasks,
# FAILS if it FAILS on either, AMBIGUOUS otherwise; read it with
#   python3 micro_test.py --aggregate <this dir>
# Cost estimate (measured on kim-compute-04, 4 threads, see artifact):
#   ns=3 (dim 4356) and ns=4 (dim 14400) with maxiter 400 took PROBE_TIME_PLACEHOLDER in total.
#   Scaling the L-BFGS step with nnz(H) (~1.9e8 at ns=6) and the analysis with
#   the number of row terms gives an expected 3-8 h per task, peak memory ~10-15 GB.
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r2
cd "$DIR"
GEOMS=(ring chain)
GEOM=${GEOMS[$SLURM_ARRAY_TASK_ID]}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --n-atoms 4 --ns 2 3 4 5 6 --geometry "$GEOM" --bond-length 2.8 \
    --hidden 64 --maxiter 2000 --snapshots 200 \
    --n-rows 3000 --exact-rows-max 15000 --top-tail 1e-3 \
    --bias-tol 1.6e-4 --de-max-mHa 10 --kappa-widths 256 \
    --out "$DIR/results_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${GEOM}.json"
