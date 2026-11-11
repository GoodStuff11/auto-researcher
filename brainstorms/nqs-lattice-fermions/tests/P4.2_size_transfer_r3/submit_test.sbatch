#!/bin/bash
#SBATCH --job-name=P4.2_r3_transfer
#SBATCH --partition=kim
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r3/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r3/%x_%A_%a.err
# P4.2 round 3: twist-complete (multi-cluster), min-norm, irrep-projected size transfer of MFNG.
# Sources: 4x2 torus + 8-site tilted cluster (2 holes each, 560 states each, together every
# C4v momentum orbit of the target); target 4x4 torus, 4 holes (1,681,680 states).
# Instance (pre-committed rule): first (t', J) in --instances whose 4x4 ground state is a
# non-degenerate K=0 singlet in a real 1-D irrep; otherwise VOID-SECTOR.
# Array: tasks 0-2 role transfer (seeds 0-2), 3-5 role foil (target-optimised Jastrow-BCS),
# 6-8 role scratch (projected-SR MFNG from scratch, long). Decision on medians over 3 seeds.
# Cost (login --timing, 8 threads, chunk 50000; timing_output.txt): one MFNG gradient pass over the
# 4x4 basis ~52 s, value pass ~14 s, so ~70 s per projected SR step with backtracking.
#   transfer: 20 (trB) + 20 (fstB) + 20 (scrB) + 180 (trL) = 240 steps ~ 4.7 h; source training
#   (3 x 800 steps on 2 x 560 states) ~ 5 min; instance scan per candidate (H build, 10 symmetry
#   maps, eigsh on 1.68M, <S^2>) ~ 15-20 min, worst case 5 candidates ~ 1.5 h; zero-shot sector
#   rule (4 isometries) ~ 5 min; 7 arms x (E_sign bracket in the ~1.3e4-orbit irrep block + Pd)
#   ~ 20 min. Total <= ~7 h.
#   foil / scratch: 200 steps ~ 4 h + scan. 24 h leaves > 3x margin.
# Memory: H CSR ~1 GB (COO build peaks ~3-4 GB), symmetry maps ~0.3 GB, chunk derivatives
# 50000 x 122 doubles; 32 GB is ample.
# Afterwards (any node, seconds):
#   python3 micro_test.py --aggregate results
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r3
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"
mkdir -p results
$PY micro_test.py --task-id "${SLURM_ARRAY_TASK_ID}" --out "$DIR/results/task${SLURM_ARRAY_TASK_ID}.json"
