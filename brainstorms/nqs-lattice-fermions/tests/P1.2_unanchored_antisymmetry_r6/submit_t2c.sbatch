#!/bin/bash
#SBATCH --job-name=P1.2r6_T2c
#SBATCH --partition=kim
#SBATCH --array=0,2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.2 round 6 critic, T2c one-seed read-outs (2026-11-16) of the trained round-5 FRD-K state (seed 11,
# shift 1e-3) on the 4x4 OBC t-t'-J instance, no training. Each task: one projected budget (job 697584
# took 11-13 min per budget at 4 CPUs), capped at 2 CPU-hours.
# task 0: base (Phi alone) -> b;  task 2: gvote (projected vote) -> z  (task 1, absfroz, dropped: AO-F killed at T1 by t1_ao.py)
# Afterwards: python3 t2c_readout.py --mode aggregate   (pre-registered rules in the t2c_readout.py docstring)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 t2c_readout.py --mode task --task "${SLURM_ARRAY_TASK_ID}"
