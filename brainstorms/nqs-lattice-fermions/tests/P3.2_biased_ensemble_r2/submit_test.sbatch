#!/bin/bash
#SBATCH --job-name=P3.2r2_2R
#SBATCH --partition=kim
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.2 round 2: same-sector avoided crossing on 4x3 PBC, N_up=N_dn=4, U=8; one array task per t'
# grid point (t' = -0.25 -0.28 -0.31 -0.33 -0.35 -0.38 -0.42; ED pre-check: c(E0) flips sign
# between -0.31 and -0.33, minimum in-sector gap 0.273 t at -0.33). Arms R0p (control), SS, SA;
# widths 8 and 32; 2 seeds -> 4 instances per task, 28 in total. CPU only.
# Cost estimate (measured on kim-compute-04, D=245025, 8 threads): one R0p branch SR step ~1.3-1.9 s,
# one SS or SA step ~4.5-5.6 s. Per instance: R0p 2 x 750 x ~1.6 s ~ 40 min, SS 500 x ~5 s ~ 42 min,
# SA ~ 45 min -> ~2.1 h; 4 instances + ED (~1 min) ~ 8.5 h per task. 24 h leaves > 2.5x margin.
# Peak memory: H (11.6M nnz) + 65536-config chunks of width-32 backflow orbitals, ~6-8 GB.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
TPS=(-0.25 -0.28 -0.31 -0.33 -0.35 -0.38 -0.42)
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --tp-list="${TPS[$SLURM_ARRAY_TASK_ID]}" --nup 4 --U 8 \
    --widths 8,32 --seeds 2 --seed 1234 --sr-iters 500 --r0p-factor 1.5 \
    --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 1000 --tau 0.01 --arms R0p,SS,SA
