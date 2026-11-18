#!/bin/bash
# P5.1 R51.8 `nscale` GEOMETRY CHECK (RULE.md rev. 1): the 4x3 S12 exact (N = infinity) reference arm (R51.1's
# start), tau 0.005, 120 steps, seed0 9000.  Round-6 copy of submit_unit1_ref12.sbatch (unchanged physics) with the
# signal path fixed: B:USR1 now reaches the batch shell, and python runs via exec so it receives the signal itself.
# Released by the geometry trigger (kappa(6x2)/kappa(4x3) = 0.61 < 1/1.5; leaves/P5.1/log/round-06-scoring.md).
# Prepared round 6, NOT submitted by the worker; submit only through compute_gate.py submit with voi_r06_ref12.md.
# Wall time: 2-step interactive pre-flight (logs/preflight_S4x3_exact.txt) 1.67 CPU-min/step; x 120 = 200 min
# + 1 setup; x 1.8 (in-job / interactive; 1.54-1.60 measured by job 763968) = 362; x 1.3 = 470 min -> 08:00:00.
# Memory 11 GB measured -> 20G.
# Resumability: ns_run.py saves theta + history atomically every step (state.npz); it stops cleanly before a step
# that would pass --cpu-cap 450, or after the current step on the USR1 sent 10 min before the wall limit (or on
# SIGTERM); rerunning this script resumes with the same seeds.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "P5.1 R51.8 geometry check (4x3 exact reference) on $(hostname), job ${SLURM_JOB_ID:-local}"
exec $PY -u ns_run.py --Lx 4 --Ly 3 --Ne 10 --exact --steps 120 --seed0 9000 --cpu-cap 450 >> logs/s12_exact.txt 2>&1
