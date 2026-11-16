#!/bin/bash
#SBATCH --job-name=P2.1r3_sklr
#SBATCH --partition=kim
#SBATCH --array=0-117
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 3: shared-reference low-rank sums (SK-LR = sk1, SK-AGP = skagp) versus symmetry-projected
# NNBF at a narrow, matched parameter budget (NNBF width 6) on three 4x4 t-t'-J clusters whose projected
# sectors are 3-17x larger than the networks (sector_dims_output.txt). See micro_test.py docstring.
# Tasks 0-116: instance = id // 39 (0: 4x4 OBC 6+6, 1: 4x4 cylinder 6+6, 2: 4x4 OBC 7+7),
#   arm = [nnbf, hf1, hf4, sk1, sk2, sjnet, agpj, agh4, skagp, nnbf_PS, sk1_PS, nnbf_U, sk1_U][(id % 39) // 3],
#   seed = [11, 22, 33][id % 3].
# Task 117: cost check (one thread; L = 12..32, brute force to L = 32).
# Submit from this directory so logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Cost estimate (measured on a kim compute node, 8 BLAS threads, instance 0, D = 1681680, one SR step;
# timing_output.txt): nnbf 14.3 s, hf1 10.3 s, hf4 13.6 s, sk1 33.9 s, sk2 42.3 s, sjnet 8.6 s,
# agpj 5.4 s, agh4 10.3 s, skagp 20.2 s, nnbf_PS 15.9 s, sk1_PS 28.8 s, nnbf_U 13.0 s, sk1_U 29.3 s.
# With --sr-iters 1200 the worst task (sk2 on instances 0/1) is ~14 h of SR at 8 threads (less at 16),
# plus ED and sector dimensions (~3 min), two P1.1 budgets (estimated <= 1 h each on D = 1.7M) and the
# Metropolis arm (minutes). Instance 2 (D = 411840) is ~4x cheaper. 24 h leaves margin.
# Peak memory: sparse H (~1 GB), group maps (~70 MB), L-BFGS-B history (~1 GB), forward chunks of
# 32768 configurations; 32 GB requested.
# Afterwards (any node, seconds): python3 micro_test.py --mode aggregate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
if [ "${SLURM_ARRAY_TASK_ID}" -eq 117 ]; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    ${PY} micro_test.py --mode cost --sizes 12,16,20,24,28,32 --brute-max 32 --nconf 2 --seed 11
else
    ${PY} micro_test.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --K 4 --hs-sign 6 \
        --match-width 6 --sr-iters 1200 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
        --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --mc-samples 20000 --mc-burn 2000
fi
