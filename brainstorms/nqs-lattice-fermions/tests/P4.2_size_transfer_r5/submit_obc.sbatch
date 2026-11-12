#!/bin/bash
#SBATCH --job-name=P4.2_r5_obc
#SBATCH --partition=kim
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%A_%a.err
# P4.2 round 5 decisive test (critic-fixed; obc_transfer.py docstring): 4x4 OBC t-t'-J, 6+6,
# t'=-0.25, J=0.4. Tasks 0-2 scr, 3-5 scr+ (20 charged kernel+Jastrow steps), 6-8 cp, 9-11 pa
# (R5.1); seed = task % 3. ELT was killed at Gate 0 and has no arm.
# SUBMIT ONLY IF the capacity pilot (submit_capacity.sbatch) gives CAPACITY DECISION: RUN ARRAY.
# Cost: ~110-140 s per projected SR step (timing_output.txt), 150 steps ~5-6 h, plus 6 records with
# the sign budget; cp/pa sources (2 x 560 states, up to 3 anchors x 4 restarts x 800 steps) take
# minutes and are charged to S. --wall-hours 22 guards the 24 h limit.
# Afterwards: python3 obc_transfer.py --aggregate results --capacity-dir results_capacity
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"
mkdir -p results
$PY obc_transfer.py --task-id "${SLURM_ARRAY_TASK_ID}" --out "$DIR/results/task${SLURM_ARRAY_TASK_ID}.json"
