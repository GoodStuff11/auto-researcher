#!/bin/bash
#SBATCH --job-name=P4.2_r2_transfer
#SBATCH --partition=kim
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r2/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r2/%x_%A_%a.err
# P4.2 round 2: zero-shot SIGN transfer of the momentum-field neural geminal (MFNG),
# t-t'-J torus 4x2 (2 holes, 560 states) -> 4x4 (4 holes, 1,681,680 states), t'=-0.25, J=0.4.
# One array task per seed (0, 1, 2). Seed 0 decides; seeds 1 and 2 are the pre-committed
# AMBIGUOUS follow-up and run in parallel so no sequential resubmission is needed.
# Cost estimate (measured on the login node with --timing, chunk 50000): one MFNG
# forward+gradient pass over the 4x4 basis ~45 s, energy-only pass ~12 s, so ~60 s per SR step.
# Target SR steps: 3 arms x 20 (budget) + 2 x 150 (converged) = 360 steps ~6 h. Source training
# (6 x 800 steps on 560 states) ~2 min. H build + k=0 orbit basis + unrestricted eigsh ~20 min.
# E_sign brackets in the k=0 block (~1.05e5 orbits): 6 arms x (7 sparse eigsh + 2 L-BFGS-B) ~1 h.
# Pairing correlations ~5 min. Total ~7-8 h per task; 24 h leaves a 3x margin. Memory: H CSR
# ~1-2 GB, basis and vectors < 1 GB, chunked log-derivatives 50000 x 218 doubles; 32 GB is ample.
# Afterwards (any node, seconds):
#   python3 micro_test.py --aggregate results
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r2
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"
$PY micro_test.py --seed "${SLURM_ARRAY_TASK_ID}" --out "$DIR/results/r2_4x2to4x4_s${SLURM_ARRAY_TASK_ID}.json"
