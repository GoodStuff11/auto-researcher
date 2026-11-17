#!/bin/bash
# P5.1 round 2, R51.3 `levrow` T1d comparison arm (RULE_T1d.md). Prepared, NOT submitted by the worker;
# submit only through compute_gate.py submit with voi_t1d.md, and only after results/t1d_ladder.json exists.
# Wall time (measured in T1c, 1 thread): 0.44 CPU-min per step, setup + start energy 0.1; at most 20 steps:
# 20 x 0.44 + 0.1 = 8.9 min; x1.3 = 11.6 -> 00:15:00. Memory: no full Jacobian (T1c peak 7.9 GB was the score
# build); about 12k x 2638 rows and the sampled 21k x 2638 matrix, under 2 GB, plus the orbit context and autograd buffers; 8G for margin.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_rowsample_r01/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_rowsample_r01/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_rowsample_r01
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
test -f results/t1d_ladder.json
echo "P5.1 R51.3 T1d continuation on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u t1d_cont.py --max-steps 20 | tee -a logs/t1d_cont.txt
