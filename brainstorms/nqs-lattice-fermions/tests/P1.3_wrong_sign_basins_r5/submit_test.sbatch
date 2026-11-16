#!/bin/bash
#SBATCH --job-name=P1.3r5_basin
#SBATCH --partition=kim
#SBATCH --array=0-35%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.3 round 5: basin versus capacity on the shared 4x4 OBC t-t'-J 6+6 instance (see micro_test.py
# docstring for arms and the pre-committed rule).
# Task id -> width = [6, 4, 10][id // 12], arm = [Cuhf, Cfs, Ccdw, W][(id % 12) // 3],
#            seed = [11, 22, 33][id % 3]. Tasks 0-11 are the width-6 pilot (Cuhf 0-2, Cfs 3-5,
#            Ccdw 6-8, W 9-11).
# Each task: 900 energy SR steps (W: after a fidelity phase of at most 600 steps that stops at
# F >= ov0_ref + 0.02 and is not charged to the energy budget), exact sums on D = 1681680,
# plus 4 (W) or 3 (cold) P1.1 budgets and ED.
# Cost (timing_output.txt, 8 threads: 12.1-12.2 s/step at width 4, 12.2-13.2 at width 6,
# 14.6-14.9 at width 10; P2.1 job 410988: 12.7 s/step at width 6 over 1200 steps):
#   cold task <= 900 x 14.9 s = 3.7 h + 3 budgets x ~6 min + ED/calibration ~5 min ~ 4.1 h;
#   worst case W at width 10: (600 + 900) x 14.9 s = 6.2 h + 4 x 6 min + 5 min ~ 6.8 h.
# The 14 h limit leaves about 2x margin over the worst case.
# Pilot first (width 6), from this directory so logs land next to the script:
#   cd <this dir> && sbatch --array=0-11 submit_test.sbatch
#   then: python3 micro_test.py --mode aggregate --instance 0 --pilot w6   (prints the decision)
# Only if the pilot says so: sbatch --array=12-35%12 submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --mode aggregate --instance 0
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} micro_test.py --mode train --instance 0 --task "${SLURM_ARRAY_TASK_ID}" --pre 600 --rel 900 \
    --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
