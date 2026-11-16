#!/bin/bash
#SBATCH --job-name=P2.1r7_T2lh
#SBATCH --partition=kim
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 7, T2 pilot of Route 6.1' (LH-HFDS) on instance 4 (5x3 OBC 6+6 t-t'-J), seed 11,
# decision at the pre-registered 25% checkpoint (75 projected SR steps). Rule: leaf artifact,
# Round 7 "T2 rule"; verdict: python3 lh_train.py --mode aggregate (after all four tasks).
#   task 0 lh_plus (route arm, plus-shaped 5-site patch)    task 1 lh_site (foil, one-site features)
#   task 2 nnbf    (reference rerun of pilot 457482 to step 75, same RNG stream; its step-0
#                   energy must reproduce -8.06278092)      task 3 lh_3x3  (reported only)
# Budget (preflight_inst4_output.txt, 4 threads): s/step lh_plus 11.7, lh_site 6.7, nnbf 4.2, lh_3x3 18.4;
# 75 steps + ED + two budgets ~ 0.7-2.0 CPU-h per task at 4 CPUs (cheapest CPU-h; route step is
# 2.8x NNBF, so the critic's 16-CPU plan would exceed the 2 h cap).
# Existing result JSONs are skipped (lh_train.py refuses to overwrite without --overwrite 1).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} lh_train.py --mode train --task "${SLURM_ARRAY_TASK_ID}" --instance 4 --seed 11 --F 5 \
    --sr-iters 75 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
