#!/bin/bash
# P4.1 R40 (RULE.md): che vs nnbf_SG, both exactly projected onto S = 0, ladder-polished, 12 sites, seed 0.
# One CPU, three sequential phases (cheapest cap; the nnbf ladder needs the pretraining):
#   1. che ladder from the r5 drift4000 theta       <= 30 steps x 2.93 (r22 max) + 0.2  =  88 CPU-min  (cap 95)
#   2. nnbf L-BFGS pretraining, 4000 iterations     4000 x 1.2-1.35 evals x 0.0645 (measured) = 310-350 CPU-min (cap 360)
#   3. nnbf ladder                                   <= 30 steps x 4.7 (extrapolated)     = 141 CPU-min (cap 150)
#   Worst case 605 CPU-min = 10.1 h of caps; --time 11:00:00 -> cap 11.0 CPU-h. Memory: Y, HY 3.3 GB each +
#   expansion temporaries ~5 GB (nnbf); r19 che peak 9.1 GB -> 24G.
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
#SBATCH --partition=kim
#SBATCH --job-name=p41-r40-che-vs-nnbfSG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=11:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r40/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r40/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r40
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
$PY -u r40.py --mode check --threads 1 | tee logs/check.txt
$PY -u r40.py --mode arm --arm che --steps 30 --threads 1 --cpu-cap 95 | tee logs/arm_che.txt
$PY -u r40.py --mode pretrain --pre-iters 4000 --threads 1 --cpu-cap 360 | tee logs/pretrain_nnbf.txt
$PY -u r40.py --mode arm --arm nnbf --steps 30 --threads 1 --cpu-cap 150 | tee logs/arm_nnbf.txt
$PY -u r40.py --mode verdict | tee logs/verdict.txt
