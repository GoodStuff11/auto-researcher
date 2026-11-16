#!/bin/bash
#SBATCH --job-name=P3.1_r2
#SBATCH --partition=kim
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round-2 decisive test: order-parameter error estimators (A-Ritz = route 2.1,
# sign-defect Ritz = route 2.2) versus one/two Lanczos steps and cross-width variance
# extrapolation, on REAL SR-trained SJ and NNBF (widths 8, 32, 128) states of the three
# P1.1 instances (array task = P1.1 instance 0, 1, 2). CPU only.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Cost estimate (measured): training is identical to P1.1 job 329057 on 8 CPUs:
# instance 0 (D=627264) 9.2 h, instance 1 (D=245025) 3.3 h, instance 2 7.1 h. Analysis per
# state: one A application 0.05 s and one H application 0.015 s at D=245025 (login node),
# 9 A + 10 H applications plus 2e5-sample MC with 100 bootstraps, about 10 s per state,
# 12 states per task. Expected wall time <= 10 h; 24 h leaves > 2x margin. Memory as P1.1.
# Afterwards (seconds, any node): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --instance "${SLURM_ARRAY_TASK_ID}" \
    --widths 8,32,128 --sr-iters 1500 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --snap-fracs 0.2,0.5 --taus 0,0.5,1,2 --r-min 1.5 \
    --mc-samples 200000 --n-boot 100 --mc-seed 99 --r-min-trust 0.05 --max-rel-se 0.5 \
    --seed 1234
# Pre-committed AMBIGUOUS follow-up (only if the pooled verdict of a route is AMBIGUOUS):
# /share/apps/software/anaconda3/bin/python3 micro_test.py --instance "${SLURM_ARRAY_TASK_ID}" \
#     --mc-samples 2000000 --mc-seed 4321 --out results_ambig/inst${SLURM_ARRAY_TASK_ID}.json
