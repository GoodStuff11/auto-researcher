#!/bin/bash
#SBATCH --job-name=P4.2_r5_close31
#SBATCH --partition=kim
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%A_%a.err
# P4.2 round 5: closing run of Route 3.1 under the critic-fixed rule (close31.py docstring).
# Task = seed s in {0,1,2}: continues the pilot (job 410960) transferred state and its MF init on
# the 4x4 torus, 4 holes, t'=-0.25, J=0.4. Arms tr (50 steps), scr (50), foil (10), nfp (20).
# Cost: ~70 s per projected SR step (r3 timing_output.txt), 130 steps ~2.6 h, setup + source
# rebuild + 9 records ~0.3 h. A wall guard (--wall-hours 3.6) stops stepping and records in time.
# Pilot peak RSS was 2.5 GB; 16 GB is ample.
# Afterwards: python3 close31.py --aggregate results_close31
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"
mkdir -p results_close31
$PY close31.py --seed "${SLURM_ARRAY_TASK_ID}" --out "$DIR/results_close31/close31_s${SLURM_ARRAY_TASK_ID}.json"
