#!/bin/bash
#SBATCH --job-name=P3.1r6_ladder
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-16
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round 6 DECISIVE TEST, training half (NOT SUBMITTED; only after INJECT-PASS of
# submit_inject.sbatch). Per-seed NNBF capacity ladders on the shared instance: widths 2,3,4 for
# seeds 11,22,33,44,55 and width 6 for seeds 44,55 (width 6 of 11/22/33 comes from P2.1) = 17 tasks.
# Uses the round-5 trainer (projected SR, P2.1 settings). Then:
#   sbatch --dependency=afterany:<THIS_JOB> --wrap "python3 micro_test.py --mode ladder --widths 2,3,4,6 --seeds 11,22,33,44,55"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
TASKS=(2:11 3:11 4:11 2:22 3:22 4:22 2:33 3:33 4:33 2:44 3:44 4:44 6:44 2:55 3:55 4:55 6:55)
W=${TASKS[$SLURM_ARRAY_TASK_ID]%%:*}; S=${TASKS[$SLURM_ARRAY_TASK_ID]##*:}
/share/apps/software/anaconda3/bin/python3 ../P3.1_order_parameter_error_r5/micro_test.py --mode train --width ${W} --seed ${S} \
    --params-dir results/params --results results
