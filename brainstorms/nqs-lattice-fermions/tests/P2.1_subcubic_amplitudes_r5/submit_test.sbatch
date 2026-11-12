#!/bin/bash
#SBATCH --job-name=P2.1r5_sklr
#SBATCH --partition=kim
#SBATCH --array=0-84
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 5 decisive array: SK-LR (sk1) and SK-AGP (skagp) versus projected NNBF and
# NNBF x Jastrow at a matched, narrow budget (NNBF width 6) on the two qualifying instances
# {0, NEWI}. See the micro_test.py docstring for arms, metrics and the pre-committed rule.
# Tasks 0-83: instance = (0, NEWI)[id // 42], arm = ARM_LIST[(id % 42) // 3], seed = [11, 22, 33][id % 3],
#   ARM_LIST = nnbf, nnbfj, hf1, hf4, sk1, sk2, sjnet, agpj, agh4, skagp, nnbf_PS, sk1_PS, nnbf_U, sk1_U.
#   Existing result JSONs are skipped (instance 0 nnbf seeds 11/22/33 from the round-3 pilot, and
#   instance NEWI nnbf seed 11 from the round-5 pilot), so those tasks exit in seconds.
# Task 84: cost check (one thread; per-connection times up to L = 48, brute force to L = 32).
# NEWI is the qualifying pilot instance (3 or 4). Submit only after the pilot qualifies it:
#   cd <this dir> && sbatch --export=ALL,NEWI=<3|4> submit_test.sbatch
# Cost (round-3 timing_output.txt, instance 0, 8 threads, one SR step): nnbf 14 s, hf1 10 s, hf4 14 s,
# sk1 34 s, sk2 42 s, sjnet 9 s, agpj 5 s, agh4 10 s, skagp 20 s; nnbfj ~ nnbf. With 1200 SR steps the
# worst task (sk2) is ~14 h at 8 threads (less at 16), plus ED (~1 min), two budgets (~20 min) and the
# sampled arm. 24 h leaves margin. Peak memory well under 32 GB.
# Afterwards (any node, seconds): python3 micro_test.py --mode aggregate --new-instance NEWI
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${NEWI:?set NEWI to the qualifying pilot instance, e.g. sbatch --export=ALL,NEWI=3 submit_test.sbatch}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
if [ "${SLURM_ARRAY_TASK_ID}" -eq 84 ]; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    ${PY} micro_test.py --mode cost --sizes 16,24,32,40,48 --brute-max 32 --nconf 2 --seed 11
else
    ${PY} micro_test.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --new-instance "${NEWI}" --K 4 \
        --hs-sign 6 --match-width 6 --nnbfj-width 4 --sr-iters 1200 --sr-lr 0.05 --sr-shift 1e-3 \
        --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --mc-samples 20000 --mc-burn 2000
fi
