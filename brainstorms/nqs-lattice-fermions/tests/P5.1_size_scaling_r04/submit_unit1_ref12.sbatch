#!/bin/bash
# P5.1 R51.8 `nscale` GEOMETRY CHECK (RULE.md revision 1; formerly unit 1b): the 4x3 S12 exact (N = infinity)
# reference arm (R51.1's start), tau 0.005, 120 steps.  DEMOTED by the compute auditor (2026-11-17): bought only if
# S10's ladder leaves the cross-geometry ratio undecided (16,850 < N_needed(S10) < 151,700) AND kappa or the
# ladders show geometry matters (RULE.md).  Its earlier gate submission was refused; voi_unit1b.md must be rewritten
# for the geometry-check decision before any submission.  Not submitted by the worker.
# Wall time: 2-step interactive pre-flight (logs/preflight_S4x3_exact.txt) 1.67 CPU-min/step; x 1.8 (in-job /
# interactive) = 3.0; x 120 = 361 min; x 1.3 = 469 min -> --time 08:00:00.  Memory 11 GB measured -> 20G.
# Resumability (audit item 4): ns_run.py saves theta + history atomically every step (state.npz), stops cleanly on
# --cpu-cap 450 or on the USR1 Slurm sends 10 min before the wall limit; rerunning the same command resumes with the
# same seeds.  Output appended with >> (no tee, which the signal would kill).
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "P5.1 R51.8 geometry check (4x3 exact reference) on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u ns_run.py --Lx 4 --Ly 3 --Ne 10 --exact --steps 120 --seed0 9000 --cpu-cap 450 >> logs/s12_exact.txt 2>&1
