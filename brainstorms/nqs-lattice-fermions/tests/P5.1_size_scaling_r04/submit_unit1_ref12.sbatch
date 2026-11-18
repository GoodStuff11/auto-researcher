#!/bin/bash
# P5.1 R51.8 `nscale` unit 1, part b (RULE.md): the S12 exact (N = infinity) reference arm (4x3, N_e 10, R51.1's
# start), tau 0.005, 120 steps.  Prepared round 4, NOT submitted by the worker; submit only through
# compute_gate.py submit with voi_unit1.md.  S12's guided rungs are R51.1's arms A, B, C (on disk).
# Wall time: 2-step interactive pre-flight (logs/preflight_S4x3_exact.txt) 1.67 CPU-min/step; x 1.8 (in-job /
# interactive) = 3.0; x 120 = 361 min; x 1.3 = 469 min -> --time 08:00:00 (cpu-cap 470 so the script stops and
# saves before the wall limit).  Memory 14.5 GB measured (kappa.py, same arrays) -> 20G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "P5.1 R51.8 nscale unit 1b (S12 exact reference) on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u ns_run.py --Lx 4 --Ly 3 --Ne 10 --exact --steps 120 --seed0 9000 --cpu-cap 470 | tee -a logs/s12_exact.txt
